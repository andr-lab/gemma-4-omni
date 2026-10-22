import torch

from duplex_kernel.audio_stream import (
    AUDIO_FRAME_SAMPLES,
    DeviceRVQCodeRing,
    TensorDecodeAccumulator,
)
from duplex_kernel.response_talk_context import ResponseTalkContextPool


def _codes(frame_id: int) -> torch.Tensor:
    return torch.arange(16, dtype=torch.long) + (frame_id * 100)


def test_tensor_accumulator_keeps_ordered_device_history_and_flushes_terminal_frame():
    ring = DeviceRVQCodeRing(capacity=5, history_limit=3, device="cpu")
    seen = []

    def decode(codes: torch.Tensor) -> torch.Tensor:
        assert codes.device.type == "cpu"
        assert codes.dtype == torch.long
        ids = (codes[0, :, 0] // 100).tolist()
        seen.append(ids)
        return codes[0, :, 0].repeat_interleave(AUDIO_FRAME_SAMPLES).float()

    accumulator = TensorDecodeAccumulator(
        decode,
        history=ring,
        chunk_frames=2,
        frame_samples=AUDIO_FRAME_SAMPLES,
        left_context_frames=3,
        crossfade_samples=0,
    )

    assert accumulator.push(0, _codes(0)) == []
    assert len(accumulator.push(1, _codes(1))) == 1
    assert len(accumulator.push(2, _codes(2))) == 0
    second = accumulator.push(3, _codes(3))[0]
    assert second.pcm.size == 2 * AUDIO_FRAME_SAMPLES
    assert (second.pcm[0], second.pcm[-1]) == (200.0, 300.0)
    assert second.context_frame_ids == (0, 1)

    assert accumulator.push(4, _codes(4)) == []
    terminal = accumulator.flush(clean_end=True)[0]
    assert terminal.pcm.size == AUDIO_FRAME_SAMPLES
    assert (terminal.pcm[0], terminal.pcm[-1]) == (400.0, 400.0)
    assert terminal.context_frame_ids == (1, 2, 3)
    assert seen == [[0, 1], [0, 1, 2, 3], [1, 2, 3, 4]]

    ordered, frame_ids = ring.window(limit=5)
    assert frame_ids == (0, 1, 2, 3, 4)
    assert ordered[:, 0].tolist() == [0, 100, 200, 300, 400]


def test_tensor_accumulator_cancel_discards_pending_without_decode():
    calls = []
    ring = DeviceRVQCodeRing(capacity=4, history_limit=2, device="cpu")
    accumulator = TensorDecodeAccumulator(
        lambda codes: calls.append(codes),
        history=ring,
        chunk_frames=2,
        frame_samples=AUDIO_FRAME_SAMPLES,
        left_context_frames=2,
        crossfade_samples=0,
    )
    accumulator.push(0, _codes(0))
    accumulator.cancel()
    assert calls == []
    assert accumulator.pending_frames == 0
    assert ring.count == 0


def test_response_slots_own_distinct_rvq_device_rings():
    class Graph:
        device = "cpu"

        def reset(self):
            pass

    pool = ResponseTalkContextPool([Graph(), Graph()])
    first = pool.acquire("session", "response-1", generation_id=1)
    second = pool.acquire("session", "response-2", generation_id=2)
    try:
        assert first.rvq_history is not second.rvq_history
        assert first.rvq_history.device == torch.device("cpu")
        assert second.rvq_history.device == torch.device("cpu")
    finally:
        pool.release(second)
        pool.release(first)


def test_rvq_ring_uses_the_allocated_tensor_device_as_canonical_device():
    ring = DeviceRVQCodeRing(device="cpu")
    assert ring.device == ring._storage.device
    ring.append(0, _codes(0))
    window, frame_ids = ring.window()
    assert window.device == ring._storage.device
    assert frame_ids == (0,)
