from __future__ import annotations

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from duplex_kernel.scheduler import (
    AudioIngressQueue,
    DuplexScheduler,
    MonotonicFrameClock,
    VirtualFrameClock,
)
from duplex_kernel.server import create_app
from duplex_kernel.session import SessionManager


FRAME_SAMPLES = 1280


class _FakeTokenizer:
    def apply_chat_template(self, messages, **_):
        return messages[-1]["content"]

    def __call__(self, text, **_):
        return {"input_ids": np.array([[1]])}


class _FakeEngine:
    loaded = True
    tokenizer = _FakeTokenizer()


def test_session_websocket_routes_arbitrary_chunks_to_independent_ingress() -> None:
    sessions = SessionManager()
    client = TestClient(create_app(_FakeEngine(), sessions))
    with client.websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({"type": "session.update", "session": {"turn_policy": "scripted", "scripted_actions": ["LISTEN"]}})
        assert ws.receive_json()["type"] == "session.updated"
        pcm = np.arange(2 * FRAME_SAMPLES, dtype=np.int16).tobytes()
        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")})
        # Force the receive loop to process the preceding append before
        # inspecting the session-owned queue.
        ws.send_json({"type": "session.update", "session": {}})
        assert ws.receive_json()["type"] == "session.updated"
        session = sessions.all()[0]
        assert session.audio_ingress.qsize == 2
        assert session.audio_ingress.stats()["input_frames_received"] == 2
        assert session.audio_ingress.stats()["input_samples_received"] == 2 * FRAME_SAMPLES
        ws.send_json({"type": "session.close"})


def test_audio_ingress_segments_arbitrary_chunks_without_loss() -> None:
    ingress = AudioIngressQueue()
    first = np.arange(500, dtype=np.float32)
    second = np.arange(500, 1280, dtype=np.float32)
    third = np.arange(1440, dtype=np.float32)

    assert ingress.append(first) == []
    frames = ingress.append(second)
    assert [frame.frame_id for frame in frames] == [0]
    assert np.array_equal(frames[0].pcm, np.arange(1280, dtype=np.float32))
    frames = ingress.append(third)
    assert [frame.frame_id for frame in frames] == [1]
    assert ingress.pending_samples == 160

    consumed = [ingress.get_nowait(), ingress.get_nowait()]
    assert [frame.frame_id for frame in consumed] == [0, 1]
    assert np.array_equal(consumed[1].pcm, np.arange(1440, dtype=np.float32)[:1280])
    stats = ingress.stats()
    assert stats["input_samples_received"] == 2720
    assert stats["input_frames_enqueued"] == 2
    assert stats["input_frames_consumed"] == 2
    assert stats["dropped_input_frames"] == 0
    assert stats["duplicate_input_frames"] == 0
    assert stats["out_of_order_input_frames"] == 0


def test_audio_ingress_rejects_non_frame_put_and_nonfinite_samples() -> None:
    ingress = AudioIngressQueue()
    with pytest.raises(ValueError, match="1280"):
        ingress.put(np.zeros(FRAME_SAMPLES - 1, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        ingress.put(np.full(FRAME_SAMPLES, np.nan, dtype=np.float32))


def test_virtual_clock_produces_exact_125_ticks_in_ten_seconds() -> None:
    ingress = AudioIngressQueue()
    ingress.append(np.zeros(125 * FRAME_SAMPLES, dtype=np.float32), capture_start_ns=0)
    scheduler = DuplexScheduler(ingress, clock=VirtualFrameClock())

    ticks = scheduler.run_for(10.0)

    assert len(ticks) == 125
    assert [tick.frame_id for tick in ticks] == list(range(125))
    assert [tick.capture_start_ns for tick in ticks] == [i * 80_000_000 for i in range(125)]
    assert scheduler.stats()["missing_ticks"] == 0
    assert scheduler.stats()["duplicate_ticks"] == 0
    assert scheduler.stats()["out_of_order_ticks"] == 0


def test_virtual_clock_produces_exact_750_ticks_in_sixty_seconds() -> None:
    ingress = AudioIngressQueue()
    ingress.append(np.zeros(750 * FRAME_SAMPLES, dtype=np.float32), capture_start_ns=0)
    scheduler = DuplexScheduler(ingress, clock=VirtualFrameClock())

    scheduler.run_for(60.0)

    metrics = scheduler.stats()
    assert metrics["ticks"] == 750
    assert metrics["input_frames_received"] == 750
    assert metrics["input_frames_consumed"] == 750
    assert metrics["dropped_input_frames"] == 0
    assert metrics["duplicate_input_frames"] == 0
    assert metrics["out_of_order_input_frames"] == 0


def test_input_is_consumed_while_speaker_callback_is_active() -> None:
    ingress = AudioIngressQueue()
    ingress.append(np.zeros(5 * 125 * FRAME_SAMPLES, dtype=np.float32), capture_start_ns=0)
    speaking_frames: list[int] = []

    def speaker(frame, tick) -> None:
        assert frame is not None
        assert tick.speaker_start_ns is not None
        speaking_frames.append(frame.frame_id)

    scheduler = DuplexScheduler(
        ingress,
        clock=VirtualFrameClock(),
        on_speaker=speaker,
    )
    scheduler.run_for(5.04)

    metrics = scheduler.stats()
    assert len(speaking_frames) == 63
    assert metrics["input_frames_during_speaking"] == 63
    assert metrics["dropped_input_frames"] == 0
    assert metrics["input_frames_consumed"] == 63


def test_real_clock_short_run_reports_bounded_jitter() -> None:
    ingress = AudioIngressQueue()
    ingress.append(np.zeros(4 * FRAME_SAMPLES, dtype=np.float32), capture_start_ns=0)
    scheduler = DuplexScheduler(ingress, clock=MonotonicFrameClock())

    scheduler.run_for(4 * 0.08)

    metrics = scheduler.stats()
    assert metrics["ticks"] == 4
    assert metrics["tick_jitter_p99_ms"] <= 20.0
    assert metrics["deadline_miss_rate"] <= 0.005
