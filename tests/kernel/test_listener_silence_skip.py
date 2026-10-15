from __future__ import annotations

import numpy as np
import pytest

from duplex_kernel.listener import (
    DEFAULT_FRAME_SAMPLES,
    GemmaListener,
    GemmaListenerState,
    ListenerBackendOutput,
    _gemma_lm_cache_can_advance,
)


class _Backend:
    def __init__(self) -> None:
        self.encode_calls = 0
        self.silence_calls = 0

    def encode(self, _user_pcm, _reference, _state):
        self.encode_calls += 1
        value = float(self.encode_calls)
        return ListenerBackendOutput(
            gemma_hidden_t=np.full((1, 4), value, dtype=np.float32),
            gemma_audio_queries_t=np.full((1, 4, 3), value, dtype=np.float32),
        )

    def advance_silent_frame(self, state):
        self.silence_calls += 1
        assert np.count_nonzero(state.user_pcm_history[-1]) == 0
        state.audio_tower_rebuild_pending = True


def test_exact_zero_skip_advances_absolute_input_and_keeps_latent_provenance() -> None:
    backend = _Backend()
    listener = GemmaListener(backend=backend, enable_silence_skip=True)
    state = GemmaListenerState()

    first = listener.step(np.ones(DEFAULT_FRAME_SAMPLES, dtype=np.float32), None, state)
    assert first.fresh is True
    assert first.source_frame_id == 0
    state.gemma_lm_cache = object()
    state.gemma_lm_audio_frames = 1
    state.gemma_lm_source_frame_ids = [0]
    state.gemma_lm_last_source_frame_id = 0
    kv_cache = state.gemma_lm_cache

    skipped = listener.step(
        np.zeros(DEFAULT_FRAME_SAMPLES, dtype=np.float32),
        None,
        state,
        allow_silence_skip=True,
    )

    assert backend.encode_calls == 1
    assert backend.silence_calls == 1
    assert skipped.frame_id == 1
    assert skipped.source_frame_id == 0
    assert skipped.fresh is False
    assert skipped.silence_skipped is True
    assert skipped.source_available_at_ns == first.source_available_at_ns
    assert state.next_frame_id == 2
    assert state.listener_input_frames == 2
    assert state.total_samples == 2 * DEFAULT_FRAME_SAMPLES
    assert state.gemma_lm_skipped_silence_frames == 1
    assert state.gemma_lm_audio_frames == 1
    assert state.gemma_lm_source_frame_ids == [0]
    assert state.gemma_lm_cache is kv_cache
    assert state.audio_tower_rebuild_pending is True
    assert _gemma_lm_cache_can_advance(state) is True
    np.testing.assert_array_equal(state.user_pcm_history[-1], np.zeros(DEFAULT_FRAME_SAMPLES))


@pytest.mark.parametrize("allow_skip", [False, True])
def test_nonzero_pcm_never_skips_even_when_skip_is_allowed(allow_skip: bool) -> None:
    backend = _Backend()
    listener = GemmaListener(backend=backend, enable_silence_skip=True)
    state = GemmaListenerState()
    listener.step(np.ones(DEFAULT_FRAME_SAMPLES, dtype=np.float32), None, state)

    result = listener.step(
        np.full(DEFAULT_FRAME_SAMPLES, 1e-12, dtype=np.float32),
        None,
        state,
        allow_silence_skip=allow_skip,
    )

    assert result.fresh is True
    assert result.frame_id == result.source_frame_id == 1
    assert backend.encode_calls == 2
    assert backend.silence_calls == 0
