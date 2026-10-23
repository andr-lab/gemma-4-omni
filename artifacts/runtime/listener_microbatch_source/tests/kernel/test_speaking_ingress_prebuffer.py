from __future__ import annotations

import numpy as np
import pytest

from duplex_kernel.scheduler import AudioIngressQueue


FRAME_SAMPLES = 1280
FRAME_NS = 80_000_000


def _pcm16_frame(value: int) -> tuple[np.ndarray, bytes]:
    raw = np.full(FRAME_SAMPLES, value, dtype="<i2").tobytes()
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return pcm, raw


def test_speaking_prebuffer_keeps_latest_three_frames_and_reindexes_contiguously() -> None:
    ingress = AudioIngressQueue(max_frames=3)
    pcm, raw = _pcm16_frame(1)
    ingress.put(pcm, capture_start_ns=0, transport_bytes=raw)
    assert ingress.get_nowait().frame_id == 0

    for frame_id in range(1, 4):
        pcm, raw = _pcm16_frame(frame_id + 1)
        ingress.put(
            pcm,
            capture_start_ns=frame_id * FRAME_NS,
            transport_bytes=raw,
        )

    ingress.begin_speaking_prebuffer(max_frames=3)
    assert ingress.qsize == 0
    for frame_id in range(4, 6):
        pcm, raw = _pcm16_frame(frame_id + 1)
        ingress.append(
            pcm,
            capture_start_ns=frame_id * FRAME_NS,
            transport_bytes=raw,
        )

    metrics = ingress.end_speaking_prebuffer()
    assert metrics["limit_frames"] == 3
    assert metrics["replayed_frames"] == 3
    assert metrics["dropped_frames"] == 2
    assert metrics["dropped_spans"]

    replayed = [ingress.get_nowait() for _ in range(3)]
    assert [frame.frame_id for frame in replayed] == [1, 2, 3]
    assert [round(float(frame.pcm[0]) * 32768) for frame in replayed] == [4, 5, 6]
    assert ingress.stats()["out_of_order_input_frames"] == 0


def test_normal_ingress_stays_lossless_and_bounded_outside_speaking_pause() -> None:
    ingress = AudioIngressQueue(max_frames=3)
    pcm, raw = _pcm16_frame(1)
    ingress.append(np.tile(pcm, 4), transport_bytes=raw * 4)
    assert ingress.qsize == 3
    assert ingress.stats()["dropped_input_frames"] == 1


def test_prebuffer_replays_partial_pcm_for_continuity_after_resume() -> None:
    ingress = AudioIngressQueue(max_frames=3)
    ingress.begin_speaking_prebuffer(max_frames=3)
    raw = np.full(FRAME_SAMPLES // 2, 100, dtype="<i2").tobytes()
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    for index in range(7):
        ingress.append(
            pcm,
            capture_start_ns=index * FRAME_NS // 2,
            transport_bytes=raw,
        )
    metrics = ingress.end_speaking_prebuffer()
    assert metrics["replayed_samples"] == 2 * FRAME_SAMPLES + FRAME_SAMPLES // 2
    assert ingress.qsize == 2
    assert ingress.pending_samples == FRAME_SAMPLES // 2

