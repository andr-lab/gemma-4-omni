from __future__ import annotations

import asyncio
import queue
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from duplex_kernel.audio_stream import (
    DecodeAccumulator,
    PlaybackWatermarks,
    TensorDecodeAccumulator,
    DeviceRVQCodeRing,
)
from duplex_kernel.models.talker import TalkerModel
from duplex_kernel.reference_buffer import AssistantReferenceBuffer
from duplex_kernel.response_talk_context import ResponseTalkContextPool
from duplex_kernel.server_response_control import (
    _cancel_response_branch_playback,
    _record_rendered_pcm_watermark,
    _response_branch_has_pending_pcm,
    _wait_for_response_playback_drain,
)


def test_playback_drain_rejects_missing_ack_before_its_deadline() -> None:
    branch = {
        "playback_ack_event": asyncio.Event(),
        "playback_segments": {
            "speaker-0": {
                "start_sample": 0,
                "emitted_end_sample": 48_000,
                "client_consumed_end_sample": 0,
            }
        },
    }
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="ACK wait timed out"):
        asyncio.run(_wait_for_response_playback_drain(branch, timeout_s=0.02))
    assert time.monotonic() - started < 0.5


def test_cancellation_drains_response_queues_and_wakes_the_worker() -> None:
    audio = queue.Queue()
    text = queue.Queue()
    audio.put("pcm")
    text.put("delta")
    stopped = asyncio.Event()
    branch = {
        "playback_ack_event": asyncio.Event(),
        "playback_cancel_callback": lambda: (
            audio.queue.clear(), text.queue.clear(), stopped.set()
        ),
    }

    _cancel_response_branch_playback(branch)

    assert audio.empty() and text.empty()
    assert stopped.is_set()
    assert branch["playback_ack_event"].is_set()
    assert branch["playback_cancelled"] is True


def test_cancelled_talk_lease_stays_owned_until_callback_returns() -> None:
    pool = ResponseTalkContextPool([object()])
    lease = pool.acquire("session", "response", generation_id=1)
    lease.callback_started()

    assert lease.cancel()
    assert pool.active_count == 1
    with pytest.raises(TimeoutError, match="admission deadline"):
        pool.acquire("other", "response-2", generation_id=2, timeout=0)

    lease.callback_finished()
    pool.release(lease)
    assert pool.active_count == 0
    assert pool.acquire("other", "response-2", generation_id=2).slot_id == 0


def test_rendered_watermark_does_not_follow_logical_buffer_ack() -> None:
    branch = {
        "playback_segments": {
            "speaker-0": {
                "start_sample": 100,
                "emitted_end_sample": 400,
                "client_consumed_end_sample": 400,
                "rendered_end_sample": 220,
            }
        }
    }
    assert _response_branch_has_pending_pcm(branch)
    _record_rendered_pcm_watermark(
        branch, speaker_segment_id="speaker-0", rendered_end_sample=400
    )
    assert not _response_branch_has_pending_pcm(branch)
    assert branch["playback_segments"]["speaker-0"]["client_consumed_end_sample"] == 400


def test_playback_watermarks_keep_device_submission_and_rendering_separate() -> None:
    watermarks = PlaybackWatermarks()
    watermarks.record_emitted(0, 960)
    watermarks.record_client_buffered(960)
    watermarks.record_rendered(480)

    assert watermarks.client_buffered_end_sample == 960
    assert watermarks.rendered_end_sample == 480
    with pytest.raises(ValueError, match="rendered watermark exceeds"):
        watermarks.record_rendered(961)


def test_assistant_reference_is_selected_by_capture_frame_id() -> None:
    references = AssistantReferenceBuffer(delay_frames=1, max_frames=8)
    references.append(np.full(1280, 0.25, dtype=np.float32), frame_id=41)
    references.append(np.full(1280, 0.75, dtype=np.float32), frame_id=42)

    selected = references.visible_for(listener_frame_id=43)
    assert selected is not None
    assert selected.frame_id == 42
    assert np.all(selected.pcm == np.float32(0.75))


@pytest.mark.parametrize("residual_frames", [1, 3])
def test_code2wav_accumulators_flush_short_and_odd_residuals(residual_frames: int) -> None:
    def decode(codes):
        return np.arange(len(codes) * 4, dtype=np.float32)

    accumulator = DecodeAccumulator(
        decode, chunk_frames=4, frame_samples=4,
        left_context_frames=0, crossfade_samples=0,
    )
    for frame in range(residual_frames):
        assert accumulator.push(frame, [frame] * 16) == []
    (chunk,) = accumulator.flush()

    assert chunk.decoded_frame_ids == tuple(range(residual_frames))
    assert chunk.pcm.size == residual_frames * 4
    assert accumulator.pending_frames == 0


@pytest.mark.parametrize("residual_frames", [1, 3])
def test_tensor_code2wav_accumulator_flushes_short_and_odd_shapes(residual_frames: int) -> None:
    def decode(codes: torch.Tensor) -> torch.Tensor:
        return torch.arange(codes.shape[1] * 4, dtype=torch.float32)

    accumulator = TensorDecodeAccumulator(
        decode,
        history=DeviceRVQCodeRing(capacity=8, history_limit=0),
        chunk_frames=4,
        frame_samples=4,
        left_context_frames=0,
        crossfade_samples=0,
    )
    for frame in range(residual_frames):
        accumulator.push(frame, torch.full((16,), frame, dtype=torch.long))
    (chunk,) = accumulator.flush()

    assert chunk.decoded_frame_ids == tuple(range(residual_frames))
    assert chunk.pcm.size == residual_frames * 4
    assert accumulator.pending_frames == 0


@pytest.mark.parametrize("frame_count", [1, 3])
def test_talker_decode_returns_only_samples_for_short_or_odd_code_shapes(
    frame_count: int,
) -> None:
    class FakeCode2Wav:
        def decode(self, codes):
            assert tuple(codes.shape) == (1, frame_count, 16)
            return SimpleNamespace(
                audio_values=torch.ones((1, frame_count * 1920 + 7))
            )

    model = TalkerModel.__new__(TalkerModel)
    model.code2wav = FakeCode2Wav()
    pcm = model.decode(torch.zeros((1, frame_count, 16), dtype=torch.long))

    assert tuple(pcm.shape) == (frame_count * 1920,)
