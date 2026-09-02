from __future__ import annotations

from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

from duplex_kernel.future_prefix import (
    ConditioningFrame,
    ConditioningFrameBuffer,
    FuturePrefixBuffer,
    FuturePrefixQwen,
    build_causal_prefix_mask,
)
from duplex_kernel.gemma_representation import interleave_duplex_mel
from duplex_kernel.playout import SpeakerPlayoutBuffer


def test_conditioning_buffer_rebases_and_fences_segments() -> None:
    buffer = ConditioningFrameBuffer(max_frames=8)
    for frame_id in range(3):
        buffer.append_values(
            frame_id=frame_id,
            h_pfx=torch.full((1, 2048), float(frame_id)),
            produced_at_ns=frame_id,
            generation_id=1,
            segment_id="old",
        )
    buffer.begin_segment(generation_id=7, segment_id="seg", base_frame_id=1)
    assert buffer.available_through == 1
    assert buffer.snapshot()[0].generation_id == 7
    assert buffer.snapshot()[0].segment_id == "seg"
    assert not buffer.append(ConditioningFrame(3, torch.zeros(1, 2048), 3, 6, "stale"))
    assert buffer.append(ConditioningFrame(3, torch.ones(1, 2048), 3, 7, "seg"))
    assert buffer.available_through == 2
    assert buffer.stats()["frames_rejected_stale"] == 1


def test_conditioning_buffer_wait_honors_cancellation() -> None:
    buffer = ConditioningFrameBuffer()
    buffer.begin_segment(generation_id=1, segment_id="seg", base_frame_id=0)
    assert buffer.wait_for_through(2, stop_check=lambda: True) == ()


def test_conditioning_buffer_wait_has_a_bounded_timeout() -> None:
    buffer = ConditioningFrameBuffer()
    buffer.begin_segment(generation_id=1, segment_id="seg", base_frame_id=0)
    started = time.monotonic()
    assert buffer.wait_for_through(2, timeout_s=0.03) == ()
    assert time.monotonic() - started < 0.5


def test_duplex_mel_interleaves_complete_frame_blocks() -> None:
    user = np.arange(16 * 128, dtype=np.float32).reshape(16, 128)
    assistant = user + 100
    actual = interleave_duplex_mel(user, assistant, mel_per_frame=8)
    expected = np.concatenate((user[:8], assistant[:8], user[8:], assistant[8:]), axis=0)
    np.testing.assert_array_equal(actual, expected)


def test_future_prefix_buffer_enforces_contiguous_ids_and_lookahead() -> None:
    buffer = FuturePrefixBuffer(hidden_dim=4, max_frames=8)
    for frame_id in range(3):
        buffer.push(frame_id, torch.full((4,), float(frame_id)))

    assert buffer.available_through == 2
    assert not buffer.can_emit(0, lookahead_frames=3)
    assert buffer.visible_prefix_end(0, lookahead_frames=3) == 2

    buffer.push(3, torch.full((4,), 3.0))
    assert buffer.can_emit(0, lookahead_frames=3)
    assert buffer.visible_prefix_end(0, lookahead_frames=3) == 3
    assert not buffer.can_emit(1, lookahead_frames=3)

    with pytest.raises(ValueError, match="atteso 4"):
        buffer.push(5, torch.zeros(4))
    with pytest.raises(ValueError, match="2048|hidden"):
        buffer.push(4, torch.zeros(3))

    buffer.cancel()
    assert buffer.available_through == -1
    assert buffer.buffered_frames == 0


def test_future_prefix_mask_hides_everything_beyond_t_plus_l() -> None:
    mask = build_causal_prefix_mask(
        batch_size=1,
        prefix_start=2,
        prefix_frames=5,
        audio_start=7,
        audio_frames=3,
        available_prefix_end=4,
        lookahead_frames=1,
        dtype=torch.float32,
    )
    neg_inf = torch.finfo(torch.float32).min

    # Audio/BOS query j=0 sees prefix 0..1 only; j=1 sees 0..2; j=2 sees 0..3.
    for audio_index, visible_end in enumerate((1, 2, 3)):
        query = 7 + audio_index
        for prefix_index in range(5):
            value = mask[0, 0, query, 2 + prefix_index].item()
            if prefix_index <= visible_end:
                assert value == 0.0
            else:
                assert value == neg_inf


def test_future_prefix_qwen_uses_position_addressable_kv_for_native_backend() -> None:
    class _NativeModel:
        def __init__(self) -> None:
            self.layers = [object()]
            self.rotary_emb = object()
            self.calls: list[dict[str, object]] = []

        def forward(self, *, inputs_embeds, attention_mask, position_ids, cache_position, past_key_values, use_cache):
            self.calls.append({
                "length": int(inputs_embeds.shape[1]),
                "mask": tuple(attention_mask.shape),
                "position": tuple(cache_position.tolist()),
                "has_past": past_key_values is not None,
                "use_cache": use_cache,
            })
            return SimpleNamespace(last_hidden_state=inputs_embeds)

        __call__ = forward

    class _NativeTalker:
        def __init__(self) -> None:
            self.model = _NativeModel()
            self.talker = SimpleNamespace(model=self.model, codec_head=lambda hidden: hidden)

        def build_prefill_parts(self, text: str):
            assert text
            zero = torch.zeros(1, 1, 2048)
            return zero, zero.clone(), zero.clone(), zero.clone()

    talker = _NativeTalker()
    speaker = FuturePrefixQwen(
        talker,
        lookahead_frames=1,
        max_prefix_frames=3,
        device="cpu",
        dtype=torch.float32,
        max_audio_frames=2,
    )
    speaker.begin("test")
    speaker.push_v9_frame(0, torch.zeros(1, 2048))
    speaker.push_v9_frame(1, torch.ones(1, 2048))
    speaker.step_frame()
    assert speaker._slot_kv is True
    assert len(talker.model.calls) == 3  # stable prefill + batched prefix horizon + audio query
    assert talker.model.calls[0]["length"] == 2
    assert talker.model.calls[1]["length"] == 2
    assert talker.model.calls[-1]["length"] == 1
    assert talker.model.calls[-1]["position"] == (5,)
    assert all(call["has_past"] for call in talker.model.calls)


def test_future_prefix_qwen_logits_ignore_prefix_after_horizon() -> None:
    class _FakeModel:
        def __call__(self, *, inputs_embeds, attention_mask, position_ids, use_cache):
            del position_ids, use_cache
            prefix = inputs_embeds[:, 2:6, :]
            # The implementation keeps future dummy rows to preserve the
            # native fixed sequence length; the target is the first audio/BOS
            # query, not the final dummy row.
            allowed = (attention_mask[:, 0, 6, 2:6] == 0).to(prefix.dtype)
            hidden = (prefix * allowed.unsqueeze(-1)).sum(dim=1, keepdim=True)
            return SimpleNamespace(last_hidden_state=hidden)

    class _FakeTalker:
        def __init__(self, values: list[float]):
            self.talker = SimpleNamespace(model=_FakeModel(), codec_head=lambda hidden: hidden)
            self._values = values

        def build_prefill_parts(self, text: str):
            assert text
            zero = torch.zeros(1, 1, 4)
            return zero, zero.clone(), zero.clone(), zero.clone()

    def run(values: list[float]) -> torch.Tensor:
        speaker = FuturePrefixQwen(
            _FakeTalker(values),
            lookahead_frames=1,
            max_prefix_frames=4,
            device="cpu",
            dtype=torch.float32,
        )
        speaker.begin("test")
        for frame_id, value in enumerate(values):
            speaker.push_v9_frame(frame_id, torch.full((4,), value))
        return speaker.step_frame().logits

    baseline = run([1.0, 2.0, 3.0, 4.0])
    changed_future = run([1.0, 2.0, 300.0, 4.0])
    changed_visible = run([1.0, 200.0, 3.0, 4.0])
    torch.testing.assert_close(baseline, changed_future)
    assert not torch.equal(baseline, changed_visible)


def test_cancelled_playout_discards_pending_frames_without_flush() -> None:
    frame = np.zeros(24_000 * 80 // 1000, dtype=np.float32)
    buffer = SpeakerPlayoutBuffer(delay_frames=2)
    buffer.push(frame, frame_id=0)
    buffer.push(frame, frame_id=1)
    buffer.cancel()
    assert buffer.buffered_frames == 0
    assert buffer.flush() == []
