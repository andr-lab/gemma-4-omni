from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import queue
import sys
import time
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from client.openai_duplex_client import (
    MIC_FRAME_SAMPLES,
    MIC_SAMPLE_RATE,
    BoundedMicrophoneFrames,
    BoundedPCMPlayback,
    RealtimeAudioDevices,
    decode_response_audio,
    handle_realtime_event,
    send_microphone_frames,
    test_realtime_duplex_websocket as run_live_demo,
    WEBSOCKET_MAX_QUEUE,
    WEBSOCKET_MAX_SIZE,
    WEBSOCKET_WRITE_LIMIT,
)


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.sent_ns: list[int] = []

    async def send(self, message: str) -> None:
        import json

        self.sent.append(json.loads(message))
        self.sent_ns.append(time.monotonic_ns())



def test_microphone_frames_are_causal_80ms_chunks_and_bounded() -> None:
    microphone = BoundedMicrophoneFrames(max_frames=3)
    samples = np.arange(3 * MIC_FRAME_SAMPLES, dtype=np.int16)
    capture_end_ns = time.monotonic_ns()

    microphone.push(samples, capture_end_ns=capture_end_ns)

    first = microphone.get_nowait()
    second = microphone.get_nowait()
    third = microphone.get_nowait()
    assert first.pcm16 == samples[:MIC_FRAME_SAMPLES].astype("<i2").tobytes()
    assert second.pcm16 == samples[MIC_FRAME_SAMPLES : 2 * MIC_FRAME_SAMPLES].astype("<i2").tobytes()
    assert third.pcm16 == samples[2 * MIC_FRAME_SAMPLES :].astype("<i2").tobytes()
    assert first.capture_end_ns < second.capture_end_ns < third.capture_end_ns
    assert second.capture_end_ns - first.capture_end_ns == 80_000_000
    assert third.capture_end_ns - second.capture_end_ns == 80_000_000
    assert MIC_SAMPLE_RATE * 80 // 1000 == MIC_FRAME_SAMPLES

    full = BoundedMicrophoneFrames(max_frames=1)
    with pytest.raises(queue.Full):
        full.push(np.zeros(2 * MIC_FRAME_SAMPLES, dtype=np.int16))
    assert full.qsize == 0, "un frame eccedente deve fallire senza enqueue parziale"


def test_microphone_overflow_is_rejected_before_allocating_a_combined_block(monkeypatch) -> None:
    microphone = BoundedMicrophoneFrames(max_frames=1)
    oversized = np.zeros(2 * MIC_FRAME_SAMPLES, dtype=np.int16)

    def forbidden_concatenate(*_args, **_kwargs):
        raise AssertionError("oversized callback data must be rejected before copying")

    monkeypatch.setattr(np, "concatenate", forbidden_concatenate)
    with pytest.raises(queue.Full):
        microphone.push(oversized)
    assert microphone.qsize == 0


def test_playback_uses_device_demand_and_flushes_stale_generation() -> None:
    playback = BoundedPCMPlayback(max_samples=16)
    playback.begin_response("response-A")
    old_pcm = np.arange(12, dtype=np.float32) / 12
    assert playback.enqueue("response-A", 4, old_pcm)

    first_device_block = np.empty((5, 1), dtype=np.float32)
    assert playback.read_into(first_device_block) == 5
    np.testing.assert_array_equal(first_device_block[:, 0], old_pcm[:5])

    playback.advance_generation(5)
    assert playback.pending_samples == 0
    assert not playback.enqueue("response-A", 4, old_pcm)
    silence = np.ones((4, 1), dtype=np.float32)
    assert playback.read_into(silence) == 0
    np.testing.assert_array_equal(silence, np.zeros_like(silence))

    playback.begin_response("response-B")
    new_pcm = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    assert playback.enqueue("response-B", 5, new_pcm)
    device_block = np.empty((5, 1), dtype=np.float32)
    assert playback.read_into(device_block) == 3
    np.testing.assert_array_equal(device_block[:3, 0], new_pcm)
    np.testing.assert_array_equal(device_block[3:, 0], np.zeros(2, dtype=np.float32))


def test_sequential_completed_responses_keep_queued_pcm_in_order() -> None:
    playback = BoundedPCMPlayback(max_samples=16)
    handle_realtime_event({"type": "response.created", "response_id": "response-A"}, playback)
    handle_realtime_event({"type": "speaker.segment.opened", "generation_id": 1}, playback)
    first = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    assert playback.enqueue("response-A", 1, first)
    handle_realtime_event({"type": "response.done", "response_id": "response-A"}, playback)

    handle_realtime_event({"type": "response.created", "response_id": "response-B"}, playback)
    handle_realtime_event({"type": "speaker.segment.opened", "generation_id": 2}, playback)
    second = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    assert playback.enqueue("response-B", 2, second)

    actual = np.empty((first.size + second.size, 1), dtype=np.float32)
    assert playback.read_into(actual) == actual.shape[0]
    np.testing.assert_array_equal(actual[:, 0], np.concatenate((first, second)))
    assert playback.pending_samples == 0


def test_barge_in_generation_discards_unfinished_pcm_but_not_new_response() -> None:
    playback = BoundedPCMPlayback(max_samples=16)
    handle_realtime_event({"type": "response.created", "response_id": "response-A"}, playback)
    handle_realtime_event({"type": "speaker.segment.opened", "generation_id": 3}, playback)
    stale = np.full(6, 0.25, dtype=np.float32)
    assert playback.enqueue("response-A", 3, stale)

    handle_realtime_event({"type": "speaker.segment.opened", "generation_id": 4}, playback)
    handle_realtime_event({"type": "response.created", "response_id": "response-B"}, playback)
    fresh = np.full(4, -0.5, dtype=np.float32)
    assert playback.pending_samples == 0
    assert playback.enqueue("response-B", 4, fresh)
    handle_realtime_event({
        "type": "response.cancelled",
        "response_id": "response-A",
        "reason": "barge_in",
    }, playback)

    actual = np.empty((fresh.size, 1), dtype=np.float32)
    assert playback.read_into(actual) == fresh.size
    np.testing.assert_array_equal(actual[:, 0], fresh)


def test_microphone_capture_callback_keeps_running_during_speaker_playback() -> None:
    microphone = BoundedMicrophoneFrames(max_frames=2)
    playback = BoundedPCMPlayback(max_samples=MIC_FRAME_SAMPLES)
    playback.begin_response("response-A")
    expected_output = np.full(MIC_FRAME_SAMPLES, 0.25, dtype=np.float32)
    assert playback.enqueue("response-A", 1, expected_output)
    devices = RealtimeAudioDevices(microphone, playback)
    devices._sounddevice = SimpleNamespace(CallbackAbort=RuntimeError)
    captured = np.arange(MIC_FRAME_SAMPLES, dtype=np.int16).reshape(-1, 1)
    output = np.full((MIC_FRAME_SAMPLES, 1), -1.0, dtype=np.float32)
    status = SimpleNamespace(input_overflow=False)

    with ThreadPoolExecutor(max_workers=2) as callbacks:
        input_callback = callbacks.submit(devices._input_callback, captured, len(captured), None, status)
        output_callback = callbacks.submit(devices._output_callback, output, len(output), None, None)
        input_callback.result()
        output_callback.result()

    assert microphone.qsize == 1
    assert microphone.get_nowait().pcm16 == captured[:, 0].astype("<i2").tobytes()
    np.testing.assert_array_equal(output[:, 0], expected_output)
    assert playback.pending_samples == 0


def test_response_cancel_flushes_already_queued_pcm_and_rejects_late_delta() -> None:
    playback = BoundedPCMPlayback(max_samples=32)
    playback.begin_response("response-A")
    pcm = np.ones(20, dtype=np.float32)
    assert playback.enqueue("response-A", 7, pcm)

    playback.cancel_response("response-A")

    assert playback.pending_samples == 0
    assert not playback.enqueue("response-A", 7, pcm)
    out = np.full((8, 1), 9.0, dtype=np.float32)
    assert playback.read_into(out) == 0
    np.testing.assert_array_equal(out, np.zeros_like(out))


def test_playback_capacity_is_bounded_and_overflow_is_visible() -> None:
    playback = BoundedPCMPlayback(max_samples=4)
    playback.begin_response("response-A")

    with pytest.raises(BufferError, match="capacity"):
        playback.enqueue("response-A", 1, np.zeros(5, dtype=np.float32))
    assert playback.pending_samples == 0


def test_audio_delta_decoder_checks_wav_rate_and_sample_timeline() -> None:
    pcm = np.linspace(-0.5, 0.5, 1920, dtype=np.float32)
    pcm16 = np.rint(pcm * 32767).astype("<i2")
    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(pcm16.tobytes())
    event = {
        "type": "response.audio.delta",
        "delta": base64.b64encode(wav_bytes.getvalue()).decode("ascii"),
        "response_id": "response-A",
        "generation_id": 3,
        "speaker_segment_id": "segment-A",
        "packet_id": 12,
        "first_frame_id": 0,
        "last_frame_id": 0,
        "start_sample": 0,
        "end_sample": pcm.size,
        "sample_rate": 24_000,
    }

    packet = decode_response_audio(event)
    assert packet.response_id == "response-A"
    assert packet.generation_id == 3
    assert packet.speaker_segment_id == "segment-A"
    assert packet.start_sample == 0
    assert packet.end_sample == pcm.size
    assert packet.sample_rate == 24_000
    assert packet.pcm.size == pcm.size
    assert np.isfinite(packet.pcm).all()

    bad_rate = {**event, "sample_rate": 16_000}
    with pytest.raises(ValueError, match="sample_rate"):
        decode_response_audio(bad_rate)
    bad_timeline = {**event, "end_sample": event["end_sample"] + 1}
    with pytest.raises(ValueError, match="sample count"):
        decode_response_audio(bad_timeline)


def test_cancel_event_is_response_scoped_and_cannot_flush_new_generation(capsys) -> None:
    playback = BoundedPCMPlayback(max_samples=32)
    handle_realtime_event({"type": "response.created", "response_id": "response-A"}, playback)
    assert playback.enqueue("response-A", 4, np.ones(20, dtype=np.float32))
    handle_realtime_event({"type": "speaker.segment.opened", "generation_id": 5}, playback)
    assert playback.pending_samples == 0
    assert not playback.enqueue("response-A", 4, np.ones(4, dtype=np.float32))

    handle_realtime_event({"type": "response.created", "response_id": "response-B"}, playback)
    assert playback.enqueue("response-B", 5, np.ones(12, dtype=np.float32))
    handle_realtime_event({"type": "response.cancelled", "response_id": "response-A", "reason": "listen"}, playback)
    assert playback.pending_samples == 12
    handle_realtime_event({"type": "response.cancelled", "response_id": "response-B", "reason": "barge_in"}, playback)
    assert playback.pending_samples == 0
    assert "Audio flushed" in capsys.readouterr().out


def test_live_demo_negotiates_bounded_full_duplex_transport_without_hardware(monkeypatch) -> None:
    class FakeRealtimeSocket:
        def __init__(self) -> None:
            wav_bytes = io.BytesIO()
            with wave.open(wav_bytes, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24_000)
                wav.writeframes(np.zeros(1920, dtype="<i2").tobytes())
            self.events = [
                {"type": "session.created"},
                {"type": "session.updated"},
                {"type": "response.created", "response_id": "response-A"},
                {"type": "speaker.segment.opened", "generation_id": 1},
                {
                    "type": "response.audio.delta",
                    "delta": base64.b64encode(wav_bytes.getvalue()).decode("ascii"),
                    "response_id": "response-A",
                    "generation_id": 1,
                    "speaker_segment_id": "segment-A",
                    "packet_id": 0,
                    "first_frame_id": 0,
                    "last_frame_id": 0,
                    "start_sample": 0,
                    "end_sample": 1920,
                    "sample_rate": 24_000,
                },
            ]
            self.sent: list[dict] = []

        async def recv(self) -> str:
            if self.events:
                return json.dumps(self.events.pop(0))
            await asyncio.Future()
            raise AssertionError("unreachable")

        async def send(self, message: str) -> None:
            self.sent.append(json.loads(message))

    class FakeDevices:
        instances = []

        def __init__(self, microphone, playback, **kwargs) -> None:
            self.microphone = microphone
            self.playback = playback
            self.kwargs = kwargs
            self.closed = False
            self.instances.append(self)

        def start(self) -> None:
            self.microphone.push(np.zeros(MIC_FRAME_SAMPLES, dtype=np.int16))

        def raise_if_failed(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    socket = FakeRealtimeSocket()
    connection: dict = {}

    class FakeConnection:
        async def __aenter__(self):
            return socket

        async def __aexit__(self, *_exc):
            return False

    def connect(url: str, **kwargs):
        connection.update(url=url, **kwargs)
        return FakeConnection()

    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=connect))
    monkeypatch.setattr("client.openai_duplex_client.RealtimeAudioDevices", FakeDevices)

    asyncio.run(run_live_demo(
        "http://localhost:8000", duration_s=0.1
    ))

    assert connection["url"] == "ws://localhost:8000/v1/realtime?duplex=1"
    assert connection["max_size"] == WEBSOCKET_MAX_SIZE
    assert connection["max_queue"] == WEBSOCKET_MAX_QUEUE
    assert connection["write_limit"] == WEBSOCKET_WRITE_LIMIT
    session_update = next(message for message in socket.sent if message["type"] == "session.update")
    assert session_update["session"]["turn_policy"] == "turnhead"
    assert "provisional_window_frames" not in session_update["session"]
    assert session_update["session"]["auto_response"] is True
    mic_frames = [message for message in socket.sent if message["type"] == "input_audio_buffer.append"]
    assert len(mic_frames) == 1
    assert len(base64.b64decode(mic_frames[0]["audio"])) == MIC_FRAME_SAMPLES * 2
    assert FakeDevices.instances[-1].playback.pending_samples == 1920
    assert FakeDevices.instances[-1].closed


def test_live_demo_reports_missing_turnhead_checkpoint_before_starting_audio(monkeypatch) -> None:
    class FakeRealtimeSocket:
        def __init__(self) -> None:
            self.events = [
                {"type": "session.created"},
                {
                    "type": "error",
                    "error": {
                        "code": "INVALID_SESSION_UPDATE",
                        "message": "TurnDynamics checkpoint unavailable at checkpoints/turn_dynamics_v1_final.pt",
                    },
                },
            ]
            self.sent: list[dict] = []

        async def recv(self) -> str:
            return json.dumps(self.events.pop(0))

        async def send(self, message: str) -> None:
            self.sent.append(json.loads(message))

    socket = FakeRealtimeSocket()

    class FakeConnection:
        async def __aenter__(self):
            return socket

        async def __aexit__(self, *_exc):
            return False

    monkeypatch.setitem(
        sys.modules,
        "websockets",
        SimpleNamespace(connect=lambda *_args, **_kwargs: FakeConnection()),
    )

    with pytest.raises(RuntimeError, match="TurnDynamics checkpoint unavailable"):
        asyncio.run(run_live_demo("http://localhost:8000", duration_s=0.1))

    update = next(message for message in socket.sent if message["type"] == "session.update")
    assert update["session"]["turn_policy"] == "turnhead"
    assert update["session"]["auto_response"] is True


def test_microphone_sender_runs_paced_while_playout_is_pending() -> None:
    async def exercise() -> None:
        microphone = BoundedMicrophoneFrames(max_frames=3)
        playback = BoundedPCMPlayback(max_samples=MIC_FRAME_SAMPLES)
        playback.begin_response("response-A")
        assert playback.enqueue("response-A", 1, np.ones(MIC_FRAME_SAMPLES, dtype=np.float32))

        microphone.push(np.zeros(2 * MIC_FRAME_SAMPLES, dtype=np.int16))
        websocket = FakeWebSocket()
        stop = asyncio.Event()
        sender = asyncio.create_task(send_microphone_frames(websocket, microphone, stop))
        try:
            for _ in range(200):
                if len(websocket.sent) == 2:
                    break
                await asyncio.sleep(0.005)
            assert len(websocket.sent) == 2
            assert all(message["type"] == "input_audio_buffer.append" for message in websocket.sent)
            assert all(len(base64.b64decode(message["audio"])) == MIC_FRAME_SAMPLES * 2 for message in websocket.sent)
            assert websocket.sent_ns[1] - websocket.sent_ns[0] >= 60_000_000
            # Playback has not been consumed; mic uplink did not wait for it.
            assert playback.pending_samples == MIC_FRAME_SAMPLES
        finally:
            stop.set()
            await asyncio.wait_for(sender, timeout=1.0)

    asyncio.run(exercise())
