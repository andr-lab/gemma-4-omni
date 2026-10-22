import pytest

import numpy as np

from duplex_kernel.listener import (
    AUDIO_FEATURES_PER_FRAME,
    AUDIO_START_AFTER_TEXT,
    DEFAULT_CONTEXT_FRAMES,
    NATIVE_LM_CUSTOM_MAX_SEQ_LEN,
    NATIVE_LM_EXACT_MAX_SEQ_LEN,
    AssistantReferenceFrame,
    GemmaAudioTowerBackend,
    GemmaListener,
    GemmaListenerState,
    _maybe_rollover_native_context,
    _native_lm_max_seq_len,
    native_frame_capacity,
)


def test_exact_backend_capacity_supports_300_seconds(monkeypatch):
    monkeypatch.setenv("GEMMA4_ATEN_SDPA_BRIDGE", "1")
    monkeypatch.setenv("GEMMA4_ATEN_RMSNORM_BRIDGE", "1")

    assert _native_lm_max_seq_len() == NATIVE_LM_EXACT_MAX_SEQ_LEN == 16384
    assert native_frame_capacity(16384) == 8175
    assert AUDIO_START_AFTER_TEXT + 3750 * AUDIO_FEATURES_PER_FRAME <= 16384
    assert native_frame_capacity(16384, prefix_len=141) == (16384 - 141) // 2


def test_custom_backend_keeps_4096_limit(monkeypatch):
    monkeypatch.delenv("GEMMA4_ATEN_SDPA_BRIDGE", raising=False)
    monkeypatch.delenv("GEMMA4_ATEN_RMSNORM_BRIDGE", raising=False)

    assert _native_lm_max_seq_len() == NATIVE_LM_CUSTOM_MAX_SEQ_LEN == 4096
    assert native_frame_capacity(4096) == 2031


def test_exact_capacity_requires_both_bridges(monkeypatch):
    monkeypatch.setenv("GEMMA4_ATEN_SDPA_BRIDGE", "1")
    monkeypatch.delenv("GEMMA4_ATEN_RMSNORM_BRIDGE", raising=False)
    assert _native_lm_max_seq_len() == NATIVE_LM_CUSTOM_MAX_SEQ_LEN


def test_custom_eager_diagnostic_uses_custom_capacity_only_for_exact_opt_in(monkeypatch):
    listener = object.__new__(GemmaAudioTowerBackend)
    listener._native_cuda_graph_requested = False
    listener.runtime_profile = "production"
    listener._native_attention_backend = 0
    listener._native_rmsnorm_backend = 2

    monkeypatch.setenv("GEMMA4_NATIVE_LISTENER_ALLOW_CUSTOM_EAGER_DIAGNOSTIC", "1")
    monkeypatch.setenv("GEMMA4_NATIVE_LISTENER_CUDA_GRAPH", "0")
    monkeypatch.setenv("GEMMA4_ATTENTION_BACKEND", "CUSTOM")
    monkeypatch.setenv("GEMMA4_RMSNORM_BACKEND", "NATIVE_EXACT")
    assert listener.native_max_seq_len == NATIVE_LM_CUSTOM_MAX_SEQ_LEN == 4096

    listener._native_attention_backend = 3
    monkeypatch.setenv("GEMMA4_NATIVE_LISTENER_ALLOW_CUSTOM_EAGER_DIAGNOSTIC", "0")
    monkeypatch.setenv("GEMMA4_ATTENTION_BACKEND", "NATIVE_SPLITK_GQA")
    assert listener.native_max_seq_len == NATIVE_LM_EXACT_MAX_SEQ_LEN == 16384


def test_native_frame_capacity_rejects_invalid_length():
    with pytest.raises(ValueError, match="max_seq_len"):
        native_frame_capacity(AUDIO_START_AFTER_TEXT)


class _SyntheticNativeCache:
    def __init__(self) -> None:
        self.current_position = AUDIO_START_AFTER_TEXT
        self.rollovers: list[tuple[int, int]] = []

    def compact_audio_history(self, *, expected_cached_frames: int, retain_frames: int) -> int:
        expected_position = AUDIO_START_AFTER_TEXT + expected_cached_frames * AUDIO_FEATURES_PER_FRAME
        assert self.current_position == expected_position
        dropped = expected_cached_frames - retain_frames
        self.current_position = AUDIO_START_AFTER_TEXT + retain_frames * AUDIO_FEATURES_PER_FRAME
        self.rollovers.append((expected_cached_frames, retain_frames))
        return dropped


class _SyntheticNativeBackend:
    def __init__(self, *, max_seq_len: int, retain_frames: int = 8) -> None:
        self.frame_capacity = native_frame_capacity(max_seq_len)
        self.retain_frames = int(retain_frames)
        self.native = _SyntheticNativeCache()
        self.reference_ids: list[int | None] = []
        self._zero_hidden = np.zeros((1, 4), dtype=np.float32)

    def encode(self, _pcm, reference, state):
        _maybe_rollover_native_context(
            state,
            self.native,
            frame_capacity=self.frame_capacity,
            retain_frames=min(state.context_frames, self.retain_frames),
        )
        expected_position = AUDIO_START_AFTER_TEXT + state.gemma_lm_audio_frames * AUDIO_FEATURES_PER_FRAME
        assert self.native.current_position == expected_position
        self.reference_ids.append(None if reference is None else reference.frame_id)
        state.gemma_lm_audio_frames += 1
        self.native.current_position += AUDIO_FEATURES_PER_FRAME
        return self._zero_hidden


def test_native_context_rolls_before_the_first_frame_that_would_overflow():
    max_seq_len = AUDIO_START_AFTER_TEXT + 7 * AUDIO_FEATURES_PER_FRAME
    backend = _SyntheticNativeBackend(max_seq_len=max_seq_len)
    listener = GemmaListener(backend=backend)
    state = GemmaListenerState(context_frames=3)
    frame = np.zeros(1280, dtype=np.float32)

    for frame_id in range(7):
        output = listener.step(frame, None, state)
        assert output.frame_id == frame_id
        if frame_id < 6:
            assert backend.native.rollovers == []

    assert backend.native.rollovers == []
    output = listener.step(frame, None, state)

    assert output.frame_id == 7
    assert backend.native.rollovers == [(7, 3)]
    assert backend.native.current_position == AUDIO_START_AFTER_TEXT + 4 * AUDIO_FEATURES_PER_FRAME
    assert state.gemma_lm_context_start_frame == 4
    assert state.gemma_lm_audio_frames == 4
    assert state.gemma_lm_context_rollover_count == 1
    assert state.gemma_lm_context_truncated_frames == 4
    assert state.next_frame_id == state.gemma_lm_context_start_frame + state.gemma_lm_audio_frames
    assert state.dropped_listener_frames == 0


def test_15000_synthetic_frames_keep_order_timing_and_bounded_context():
    max_seq_len = NATIVE_LM_EXACT_MAX_SEQ_LEN
    backend = _SyntheticNativeBackend(max_seq_len=max_seq_len, retain_frames=DEFAULT_CONTEXT_FRAMES)
    listener = GemmaListener(backend=backend)
    state = GemmaListenerState(context_frames=DEFAULT_CONTEXT_FRAMES)
    frame = np.zeros(1280, dtype=np.float32)
    reference = np.full(1280, 0.25, dtype=np.float32)

    for frame_id in range(15_000):
        assistant = None if frame_id == 0 else AssistantReferenceFrame(
            frame_id=frame_id - 1,
            pcm=reference,
            emitted_at_ns=frame_id * 80_000_000,
            sample_rate=16_000,
        )
        output = listener.step(frame, assistant, state)
        assert output.frame_id == frame_id
        assert output.audio_timestamp == (frame_id + 1) * 80_000_000

    assert len(backend.reference_ids) == 15_000
    assert backend.reference_ids[0] is None
    assert backend.reference_ids[1:] == list(range(14_999))
    assert backend.native.current_position <= max_seq_len
    assert backend.native.current_position == (
        AUDIO_START_AFTER_TEXT + state.gemma_lm_audio_frames * AUDIO_FEATURES_PER_FRAME
    )
    assert state.next_frame_id == 15_000
    assert state.listener_input_frames == 15_000
    assert state.dropped_listener_frames == 0
    assert state.future_reference_leaks == 0
    assert state.wrong_reference_frames == 0
    assert state.gemma_lm_context_rollover_count == 1
    assert state.gemma_lm_context_truncated_frames == 8175 - DEFAULT_CONTEXT_FRAMES
    assert state.next_frame_id == state.gemma_lm_context_start_frame + state.gemma_lm_audio_frames
    assert len(state.user_pcm_history) == DEFAULT_CONTEXT_FRAMES
    assert len(state.assistant_pcm_history) == DEFAULT_CONTEXT_FRAMES
