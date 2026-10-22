from __future__ import annotations

from types import SimpleNamespace
import threading

import pytest
import torch

from duplex_kernel.audio_context_capture import capture_audio_context_hidden
from duplex_kernel import config
from duplex_kernel.engine import PipelineEngine


class _MaskModel:
    def __init__(self, audio_mask: torch.Tensor) -> None:
        self.audio_mask = audio_mask

    def get_placeholder_mask(self, *, input_ids):
        assert tuple(input_ids.shape) == tuple(self.audio_mask.shape)
        empty = torch.zeros_like(input_ids, dtype=torch.bool)
        return empty, empty, self.audio_mask


def test_capture_derives_80ms_frames_from_40ms_mono_tokens_and_clones() -> None:
    ids = torch.arange(22, dtype=torch.long).reshape(1, 22)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 4:12] = True
    mask[:, 14:22] = True
    # With two cached text tokens, prefill_hidden covers only the 20-token suffix.
    hidden = torch.arange(20 * 1536, dtype=torch.float32).reshape(1, 20, 1536)
    thinker = SimpleNamespace(model=_MaskModel(mask))

    snapshot = capture_audio_context_hidden(
        thinker,
        input_ids=ids,
        prefill_hidden=hidden,
        past_length=2,
        audio_ms_per_token=40,
        source_pcm_samples=79_360,
    )

    assert snapshot.audio_placeholder_count == 16
    assert snapshot.selected_frame_count == 8
    assert snapshot.audio_ms_per_token == 40
    assert snapshot.raw_frame_period_ms == 80
    assert snapshot.source_pcm_samples == 79_360
    # Runs [4, 12) and [14, 22) each restart two-slot mono grouping.
    expected_positions = torch.tensor([2, 4, 6, 8, 12, 14, 16, 18])
    assert torch.equal(snapshot.hidden, hidden.index_select(1, expected_positions))
    hidden.fill_(-7.0)
    assert torch.equal(snapshot.hidden[:, 0], torch.arange(2 * 1536, 3 * 1536).reshape(1, 1536))


class _AudioPrefillThinker:
    def __init__(self, hidden: torch.Tensor, audio_mask: torch.Tensor) -> None:
        self.model = _MaskModel(audio_mask)
        self.hidden = hidden

    def __call__(self, **kwargs):
        assert kwargs["use_cache"] is True
        return SimpleNamespace(
            hidden_states=(self.hidden,),
            past_key_values=object(),
        )


def test_audio_prefill_returns_private_snapshot_from_same_forward() -> None:
    engine = object.__new__(PipelineEngine)
    full_ids = torch.arange(22, dtype=torch.long).reshape(1, 22)
    audio_mask = torch.zeros_like(full_ids, dtype=torch.bool)
    audio_mask[:, 4:12] = True
    audio_mask[:, 14:22] = True
    prefill_hidden = torch.arange(20 * 1536, dtype=torch.float32).reshape(1, 20, 1536)
    engine.thinker = _AudioPrefillThinker(prefill_hidden, audio_mask)
    engine._thinker_wrapper = SimpleNamespace(processor=SimpleNamespace(audio_ms_per_token=40))
    engine.device = "cpu"
    engine._thinker_lock = threading.Lock()
    engine._thinker_audio_forward_active = threading.Event()
    engine._thinker_audio_prefill_lock_hold_max_ms = 0.0

    result = engine._run_thinker_audio_prefill(
        {
            "input_ids": full_ids,
            "attention_mask": torch.ones_like(full_ids),
            "input_features": torch.zeros((1, 4, 8)),
            "input_features_mask": torch.ones((1, 4), dtype=torch.long),
        },
        is_compiled=False,
        static_cache=None,
        past_length=2,
        cache_init_started=None,
        runtime_metrics={
            "thinker_prefill_lock_wait_ms": 0.0,
            "thinker_prefill_lock_hold_ms": 0.0,
            "thinker_prefill_lock_hold_max_ms": 0.0,
        },
        capture_audio_context=True,
        audio_context_source_pcm_samples=79_360,
    )

    assert len(result) == 5
    snapshot = result[4]
    assert snapshot is not None
    assert snapshot.audio_placeholder_count == 16
    assert snapshot.selected_frame_count == 8
    assert snapshot.source_pcm_samples == 79_360
    assert torch.equal(snapshot.hidden[:, 0], prefill_hidden[:, 2])


def test_capture_keeps_partial_final_clock_group_and_rejects_cached_audio_prefix() -> None:
    ids = torch.arange(12, dtype=torch.long).reshape(1, 12)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 3:8] = True
    thinker = SimpleNamespace(model=_MaskModel(mask))
    hidden = torch.zeros((1, 12, 1536))

    partial = capture_audio_context_hidden(
        thinker,
        input_ids=ids,
        prefill_hidden=hidden,
        audio_ms_per_token=40,
    )
    assert partial.audio_placeholder_count == 5
    assert partial.selected_frame_count == 3
    assert torch.equal(partial.hidden, hidden[:, 3:9:2])

    mask.zero_()
    mask[:, 0:4] = True
    with pytest.raises(RuntimeError, match="reused text-only prefix"):
        capture_audio_context_hidden(
            thinker,
            input_ids=ids,
            prefill_hidden=hidden[:, 4:],
            past_length=4,
            audio_ms_per_token=40,
        )


class _RecordingV9Stream:
    raw_dim = 1536

    def __init__(self) -> None:
        self.projector = SimpleNamespace(audio_in=SimpleNamespace(weight=torch.empty((), dtype=torch.float32)))
        self.state_max_len = None
        self.raw = []
        self.anchors = []

    def new_state(self, *, batch_size, device, dtype, max_seq_len):
        self.state_max_len = max_seq_len
        return SimpleNamespace(rope_position=0)

    def step(self, raw, anchor, state):
        self.raw.append(raw.detach().clone())
        self.anchors.append(anchor.detach().clone())
        state.rope_position += 1
        return torch.cat((raw, anchor), dim=-1)[..., :2048]


def test_dual_v9_prefix_keeps_audio_timeline_and_say_anchor_separate() -> None:
    engine = object.__new__(PipelineEngine)
    engine.v9_profile = config.V9_PROFILE_CAUSAL_L0
    engine.v9_recipe = config.V9_CAUSAL_L0_RECIPE
    engine.v9_stream = _RecordingV9Stream()
    raw = torch.full((1, 3, 1536), 2.0)
    anchor = torch.full((1, 1536), 9.0)

    result = engine.v9_prefix_audio_context(raw, anchor)

    assert tuple(result.h_pfx.shape) == (1, 3, 2048)
    assert engine.v9_stream.state_max_len == 3
    assert len(engine.v9_stream.raw) == len(engine.v9_stream.anchors) == 3
    assert all(torch.equal(value, raw[:, index]) for index, value in enumerate(engine.v9_stream.raw))
    assert all(torch.equal(value, anchor) for value in engine.v9_stream.anchors)
    assert torch.equal(result.anchor, anchor)
