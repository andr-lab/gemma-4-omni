from __future__ import annotations

import numpy as np
import pytest

from duplex_kernel.playout import SpeakerPlayoutBuffer


FRAME = 24_000 * 80 // 1000


def pcm(frame_id: int) -> np.ndarray:
    return np.full(FRAME, frame_id / 100.0, dtype=np.float32)


def test_playout_releases_only_after_conditioning_delay() -> None:
    buffer = SpeakerPlayoutBuffer(delay_frames=2)
    for frame_id in range(4):
        buffer.push(pcm(frame_id), frame_id=frame_id)
        released = buffer.release_ready(conditioning_frame_id=frame_id)
        expected = [frame_id - 2] if frame_id >= 2 else []
        assert [frame.frame_id for frame in released] == expected
    assert [frame.frame_id for frame in buffer.flush()] == [2, 3]


def test_playout_does_not_delay_listener_clock_or_drop_chunk_frames() -> None:
    buffer = SpeakerPlayoutBuffer(delay_frames=7)
    generated = np.concatenate([pcm(frame_id) for frame_id in range(8)])
    buffer.push_chunk(generated, first_frame_id=0)
    released = buffer.release_ready(conditioning_frame_id=7)
    assert [frame.frame_id for frame in released] == [0]
    assert buffer.stats()["frames_buffered"] == 7
    assert [frame.frame_id for frame in buffer.flush()] == list(range(1, 8))


def test_playout_rejects_gaps_bad_shape_and_nonfinite_pcm() -> None:
    buffer = SpeakerPlayoutBuffer(delay_frames=1)
    with pytest.raises(ValueError, match="atteso 0"):
        buffer.push(pcm(1), frame_id=1)
    with pytest.raises(ValueError, match="1920"):
        buffer.push(np.zeros(FRAME - 1, dtype=np.float32), frame_id=0)
    with pytest.raises(ValueError, match="non finito"):
        bad = pcm(0)
        bad[0] = np.nan
        buffer.push(bad, frame_id=0)
