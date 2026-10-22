from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.nn import functional as F
from types import SimpleNamespace
from types import MethodType

from duplex_kernel.listener_microbatch_frontend import (
    MEL_WINDOW_ROWS,
    SOFT_TOKENS_PER_FRAME,
    ListenerMicrobatchAudioFrontend,
    mel_window_geometry,
    validate_mel_window_rows,
    _user_mel_window,
)
from duplex_kernel.microbatch_contract import LISTENER_Q_PER_FRAME
from duplex_kernel.native_control6_phase import native_control6_phase_t
from training.duplex_qlora.native_control6 import HIDDEN_WIDTH, NativeControl6LoRA
from duplex_kernel.listener import (
    AUDIO_START_AFTER_TEXT,
    GemmaAudioTowerBackend,
    GemmaListenerState,
    _select_audio_tower_rows,
)


class _TinyAudioModel:
    def get_audio_features(self, **_kwargs):
        raise AssertionError("workspace geometry test must not run the model")


class _RejectBooleanIndexCompaction(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        if "aten.index.Tensor" in str(func) or "aten.nonzero" in str(func):
            raise AssertionError(f"unexpected CUDA-style mask compaction op: {func}")
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize(
    ("window_rows", "context_rows", "output_rows", "q_start", "history_frames"),
    ((60, 52, 15, 13, 64), (128, 120, 32, 30, 64),
     (240, 232, 60, 58, 64), (640, 632, 160, 158, 79)),
)
def test_mel_window_parameter_sizes_frontend_workspace_and_q2_geometry(
    window_rows: int,
    context_rows: int,
    output_rows: int,
    q_start: int,
    history_frames: int,
) -> None:
    frontend = ListenerMicrobatchAudioFrontend(
        _TinyAudioModel(),
        lambda _wave, _tail: (np.zeros((8, 128), dtype=np.float32), np.zeros(160, dtype=np.float32)),
        device="cpu",
        dtype=torch.float32,
        mel_window_rows=window_rows,
    )

    workspace = frontend._workspace(2)
    assert frontend.mel_window_rows == window_rows
    assert frontend.mel_context_rows == context_rows
    assert frontend.tower_output_rows == output_rows
    assert frontend.current_q_start == q_start
    assert frontend.context_frames == history_frames
    assert workspace.cpu_features.shape == (2, window_rows, 128)
    assert tuple(workspace.features.shape) == (2, window_rows, 128)


@pytest.mark.parametrize("window_rows", (60, 128, 240, 640))
def test_user_mel_window_uses_configured_prior_rows(window_rows: int) -> None:
    history = (np.arange(8 * 100 * 128, dtype=np.float32).reshape(100, 8, 128))
    current = np.full((8, 128), -7.0, dtype=np.float32)

    window = _user_mel_window(history, current, mel_window_rows=window_rows)

    assert window.shape == (window_rows, 128)
    np.testing.assert_array_equal(window[:-8], np.concatenate(history, axis=0)[-(window_rows - 8):])
    np.testing.assert_array_equal(window[-8:], current)


@pytest.mark.parametrize("window_rows", (7, 9, 62, 10, True, 60.0))
def test_mel_window_parameter_rejects_nonmultiples_and_too_short(window_rows) -> None:
    with pytest.raises(ValueError, match="multiple of 4 and at least 8"):
        validate_mel_window_rows(window_rows)


@pytest.mark.parametrize("window_rows", (60, 128, 240, 640))
def test_audio_tower_q2_backend_accepts_matching_window_and_q_start(window_rows: int) -> None:
    backend = object.__new__(GemmaAudioTowerBackend)
    backend.device = torch.device("cpu")
    backend.dtype = torch.float32
    backend.mel_window_rows = window_rows
    features = torch.zeros((1, window_rows, 128), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        backend.run_audio_tower_q2(features, mel_window_geometry(window_rows).q_start)

    with pytest.raises(ValueError, match="q_start="):
        backend.run_audio_tower_q2(features, mel_window_geometry(window_rows).q_start + 1)


def test_full_valid_q2_graph_snapshot_slices_without_mask_compaction() -> None:
    pooled = torch.arange(60 * 1536, dtype=torch.float32).reshape(1, 60, 1536)
    output_mask = torch.ones((1, 60), dtype=torch.bool)
    with _RejectBooleanIndexCompaction():
        selected = _select_audio_tower_rows(
            pooled,
            output_mask,
            selected_q_start=58,
            all_valid_mask=True,
            hidden_size=1536,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    torch.testing.assert_close(selected, pooled[:, 58:60, :].clone(), rtol=0, atol=0)


def test_audio_tower_snapshot_preserves_general_masked_row_selection() -> None:
    pooled = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    output_mask = torch.tensor([[True, False, True, True, False]])
    selected = _select_audio_tower_rows(
        pooled,
        output_mask,
        selected_q_start=1,
        all_valid_mask=False,
        hidden_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(selected, pooled[:, 2:4, :].clone(), rtol=0, atol=0)


def test_q2_frontend_keeps_exactly_52_user_context_rows_and_eight_new_rows() -> None:
    history = np.arange(8 * 8 * 128, dtype=np.float32).reshape(8, 8, 128)
    current = np.full((8, 128), 999.0, dtype=np.float32)

    window = _user_mel_window(tuple(history), current, mel_window_rows=60)

    assert window.shape == (60, 128)
    np.testing.assert_array_equal(window[:52], np.concatenate(history, axis=0)[-52:])
    np.testing.assert_array_equal(window[52:], current)
    assert MEL_WINDOW_ROWS == 240
    assert SOFT_TOKENS_PER_FRAME == LISTENER_Q_PER_FRAME == 2


def test_q2_startup_window_uses_only_available_user_context() -> None:
    current = np.full((8, 128), 7.0, dtype=np.float32)

    window = _user_mel_window((), current, mel_window_rows=60)

    assert window.shape == (8, 128)
    np.testing.assert_array_equal(window, current)


def test_native_control6_keeps_base_1536_and_adds_phase_only_to_lora_delta() -> None:
    torch.manual_seed(31)
    base_rows = torch.randn(6, HIDDEN_WIDTH)
    adapter = NativeControl6LoRA(base_rows, rank=4)
    hidden = torch.randn(3, HIDDEN_WIDTH)
    phase = torch.randn(3, 4)
    with torch.no_grad():
        adapter.lora_B.normal_()
        logits_with_phase = adapter(hidden, phase_t=phase)
        logits_with_zero_phase = adapter(hidden, phase_t=torch.zeros_like(phase))

    assert tuple(adapter.base_rows.shape) == (6, 1536)
    assert tuple(adapter.lora_A.shape) == (4, 1540)
    assert tuple(logits_with_phase.shape) == (3, 6)
    assert not torch.equal(logits_with_phase, logits_with_zero_phase)


def test_legacy_1536_column_adapter_migrates_in_memory_with_zero_phase_effect() -> None:
    torch.manual_seed(47)
    base_rows = torch.randn(6, HIDDEN_WIDTH)
    adapter = NativeControl6LoRA(base_rows, rank=4)
    old_a = torch.randn(4, HIDDEN_WIDTH)
    old_b = torch.randn(6, 4)
    migrated = adapter.load_adapter_state_dict({"lora_A": old_a, "lora_B": old_b})
    hidden = torch.randn(3, HIDDEN_WIDTH)
    phase = torch.randn(3, 4)

    actual = adapter(hidden, phase_t=phase)
    expected = F.linear(hidden, base_rows) + (
        F.linear(F.linear(hidden, old_a), old_b) * adapter.scaling
    )

    assert migrated is True
    assert torch.equal(adapter.lora_A[:, :HIDDEN_WIDTH], old_a)
    assert torch.equal(adapter.lora_A[:, HIDDEN_WIDTH:], torch.zeros((4, 4)))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_phase_t_uses_timestamped_assistant_prior_frames_and_expected_order() -> None:
    user = np.full(1280, 0.1, dtype=np.float32)
    assistant_prior = np.full(1280, 0.01, dtype=np.float32)
    assistant_preceding = np.full(1280, 0.001, dtype=np.float32)

    phase = native_control6_phase_t(
        listener_frame_id=10,
        user_pcm_current=user,
        assistant_pcm_prior=assistant_prior,
        assistant_prior_frame_id=9,
        assistant_pcm_preceding=assistant_preceding,
        assistant_preceding_frame_id=8,
    )

    np.testing.assert_allclose(phase, [-2.0, 1.0, 1.0 / 6.0, -1.0], atol=1e-6)


def test_phase_t_rejects_future_or_nonadjacent_assistant_reference() -> None:
    with np.testing.assert_raises_regex(ValueError, "t-1"):
        native_control6_phase_t(
            listener_frame_id=10,
            user_pcm_current=np.zeros(1280, dtype=np.float32),
            assistant_pcm_prior=np.zeros(1280, dtype=np.float32),
            assistant_prior_frame_id=10,
            assistant_pcm_preceding=None,
            assistant_preceding_frame_id=None,
        )


class _TinyEmbedTokens:
    weight = SimpleNamespace(dtype=torch.float32)

    def __call__(self, token_ids):
        return torch.zeros((*token_ids.shape, HIDDEN_WIDTH), dtype=torch.float32)


class _TinyLanguageModel:
    embed_tokens = _TinyEmbedTokens()

    def get_per_layer_inputs(self, _token_ids, embeds):
        return embeds


class _TinyCache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self, _layer: int = 0) -> int:
        return self.length


def test_text_context_rebuild_replays_only_observed_q2_frames_and_reset_clears_history() -> None:
    backend = object.__new__(GemmaAudioTowerBackend)
    backend.gemma_model = SimpleNamespace(language_model=_TinyLanguageModel())
    backend._use_native_lm = False
    backend.device = torch.device("cpu")
    backend.hidden_size = HIDDEN_WIDTH
    backend._frame_pad_per_layer_inputs = lambda _lm: None
    seen: list[tuple[tuple[int, ...], tuple[int, ...] | None]] = []

    def forward(self, *, embeds, positions, state, **_kwargs):
        seen.append((tuple(embeds.shape), tuple(positions.tolist())))
        prior = 0 if state.gemma_lm_cache is None else state.gemma_lm_cache.get_seq_length()
        state.gemma_lm_cache = _TinyCache(prior + int(embeds.shape[1]))
        return torch.zeros((1, embeds.shape[1], HIDDEN_WIDTH), dtype=torch.float32)

    backend._forward_incremental_lm = MethodType(forward, backend)
    state = GemmaListenerState(next_frame_id=7)
    state.gemma_lm_cache = _TinyCache(40)
    state.gemma_lm_text_context = (1,)
    state.gemma_lm_audio_soft_history.extend(
        ((4, torch.full((2, HIDDEN_WIDTH), 4.0)),
         (6, torch.full((2, HIDDEN_WIDTH), 6.0)))
    )
    ids = torch.zeros((1, AUDIO_START_AFTER_TEXT), dtype=torch.long)
    mask = torch.ones_like(ids)

    hidden, queries = GemmaAudioTowerBackend._run_incremental_lm(
        backend,
        input_ids=ids,
        attention_mask=mask,
        audio_soft=torch.full((1, 2, HIDDEN_WIDTH), 7.0),
        state=state,
        text_ids=(2,),
    )

    assert [shape for shape, _positions in seen] == [
        (1, AUDIO_START_AFTER_TEXT, HIDDEN_WIDTH),
        (1, 2, HIDDEN_WIDTH),
        (1, 2, HIDDEN_WIDTH),
        (1, 2, HIDDEN_WIDTH),
    ]
    assert state.gemma_lm_source_frame_ids == [4, 6, 7]
    assert state.gemma_lm_audio_frames == 3
    assert state.gemma_lm_last_source_frame_id == 7
    assert tuple(hidden.shape) == (1, HIDDEN_WIDTH)
    assert tuple(queries.shape) == (1, 2, HIDDEN_WIDTH)
    assert 5 not in state.gemma_lm_source_frame_ids

    state.reset()
    assert not state.gemma_lm_audio_soft_history


def test_non_native_steady_q2_tick_uses_current_audio_rows() -> None:
    backend = object.__new__(GemmaAudioTowerBackend)
    backend.gemma_model = SimpleNamespace(language_model=_TinyLanguageModel())
    backend._use_native_lm = False
    backend.device = torch.device("cpu")
    backend.hidden_size = HIDDEN_WIDTH
    backend._frame_pad_per_layer_inputs = lambda _lm: None
    seen = {}

    def forward(self, *, embeds, positions, state, **_kwargs):
        seen["shape"] = tuple(embeds.shape)
        seen["positions"] = tuple(positions.tolist())
        prior = state.gemma_lm_cache.get_seq_length()
        state.gemma_lm_cache = _TinyCache(prior + int(embeds.shape[1]))
        return torch.zeros((1, embeds.shape[1], HIDDEN_WIDTH), dtype=torch.float32)

    backend._forward_incremental_lm = MethodType(forward, backend)
    state = GemmaListenerState(next_frame_id=7)
    state.gemma_lm_cache = _TinyCache(AUDIO_START_AFTER_TEXT + 4)
    state.gemma_lm_audio_frames = 2
    state.gemma_lm_source_frame_ids = [5, 6]
    state.gemma_lm_last_source_frame_id = 6
    state.gemma_lm_text_context = (2,)
    ids = torch.zeros((1, AUDIO_START_AFTER_TEXT), dtype=torch.long)

    hidden, queries = GemmaAudioTowerBackend._run_incremental_lm(
        backend,
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        audio_soft=torch.full((1, 2, HIDDEN_WIDTH), 7.0),
        state=state,
        text_ids=(2,),
    )

    assert seen == {
        "shape": (1, 2, HIDDEN_WIDTH),
        "positions": (AUDIO_START_AFTER_TEXT + 4, AUDIO_START_AFTER_TEXT + 5),
    }
    assert state.gemma_lm_audio_frames == 3
    assert state.gemma_lm_last_source_frame_id == 7
    assert tuple(hidden.shape) == (1, HIDDEN_WIDTH)
    assert tuple(queries.shape) == (1, 2, HIDDEN_WIDTH)
