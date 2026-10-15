# -*- coding: utf-8 -*-
"""tests/test_duplex_dialogue_harness.py

Unit and contract tests for the two-interlocutor dialogue harness:
1. AudioTimelineReconstructor: timeline assembly, packet deduplication, cancelled packet rejection.
2. AudioBridge: 24 kHz to 16 kHz polyphase resampling, 80 ms pacing, strict loopback prevention (no self-echo).
3. Guardrails: max turns, max duration, silence timeouts.
4. Audio recording (WAV files for A-in, A-out, B-in, B-out) and JSONL event tracing.
"""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path
import numpy as np
import pytest

from client.duplex_dialogue_harness import (
    AudioTimelineReconstructor,
    AudioBridge,
    DialogueRecorder,
    DialogueEventLogger,
    DialogueConfig,
    DialogueOrchestrator,
    resample_24k_to_16k,
    pcm_to_80ms_frames,
    float_to_pcm16_bytes,
    _conditioning_ack_matches_frame,
    FRAME_SAMPLES_16K,
)


def test_resample_24k_to_16k_exact_ratio() -> None:
    # 24,000 samples @ 24 kHz is exactly 1 second.
    # Resampling should yield exactly 16,000 samples @ 16 kHz.
    t_24k = np.linspace(0, 1.0, 24000, endpoint=False, dtype=np.float32)
    tone_24k = np.sin(2 * np.pi * 440.0 * t_24k)

    resampled_16k = resample_24k_to_16k(tone_24k)
    assert len(resampled_16k) == 16000
    assert resampled_16k.dtype == np.float32
    # Verify no clipping / NaN
    assert np.isfinite(resampled_16k).all()
    assert np.max(np.abs(resampled_16k)) <= 1.05


def test_pcm_to_80ms_frames() -> None:
    # 16 kHz * 0.080 s = 1280 samples per frame.
    # 3200 samples = 2 full frames (2560 samples) + 640 remainder.
    pcm = np.arange(3200, dtype=np.float32) / 3200.0
    frames, remainder = pcm_to_80ms_frames(pcm)

    assert len(frames) == 2
    assert len(frames[0]) == 1280
    assert len(frames[1]) == 1280
    assert len(remainder) == 640
    np.testing.assert_array_equal(frames[0], pcm[:1280])
    np.testing.assert_array_equal(frames[1], pcm[1280:2560])
    np.testing.assert_array_equal(remainder, pcm[2560:])


def test_conditioning_ack_requires_a_published_frame_matching_the_ingress_frame() -> None:
    valid_ack = {
        "frame_id": 7,
        "conditioning_frame_id": 7,
        "listener_output_frame_id": 7,
        "conditioning_frame_available": True,
    }
    assert _conditioning_ack_matches_frame(valid_ack, 7)
    assert not _conditioning_ack_matches_frame({**valid_ack, "conditioning_frame_id": 8}, 7)
    assert not _conditioning_ack_matches_frame({**valid_ack, "conditioning_frame_available": False}, 7)
    assert not _conditioning_ack_matches_frame(
        {**valid_ack, "conditioning_frame_id": float("inf")}, 7
    )


def test_timeline_reconstructor_deduplication_and_cancellation() -> None:
    reconstructor = AudioTimelineReconstructor()

    # Packet 1 for response A
    pcm_p1 = np.ones(960, dtype=np.float32) * 0.1
    p1 = reconstructor.add_packet(
        response_id="resp_A",
        generation_id=1,
        packet_id=1,
        first_frame_id=0,
        last_frame_id=1,
        start_sample=0,
        end_sample=960,
        pcm=pcm_p1,
    )
    assert p1 is True

    # Duplicate of Packet 1 must be rejected
    p1_dup = reconstructor.add_packet(
        response_id="resp_A",
        generation_id=1,
        packet_id=1,
        first_frame_id=0,
        last_frame_id=1,
        start_sample=0,
        end_sample=960,
        pcm=pcm_p1,
    )
    assert p1_dup is False
    assert reconstructor.duplicate_packet_count == 1

    # Cancel response A
    reconstructor.cancel_response("resp_A")

    # Packet 2 for cancelled response A must be rejected
    pcm_p2 = np.ones(960, dtype=np.float32) * 0.2
    p2_cancelled = reconstructor.add_packet(
        response_id="resp_A",
        generation_id=1,
        packet_id=2,
        first_frame_id=2,
        last_frame_id=3,
        start_sample=960,
        end_sample=1920,
        pcm=pcm_p2,
    )
    assert p2_cancelled is False
    assert reconstructor.cancelled_packet_count == 1

    # Packet for a new response B must be accepted
    p_b = reconstructor.add_packet(
        response_id="resp_B",
        generation_id=2,
        packet_id=1,
        first_frame_id=0,
        last_frame_id=1,
        start_sample=0,
        end_sample=960,
        pcm=pcm_p1,
    )
    assert p_b is True


def test_audio_bridge_strict_no_echo() -> None:
    bridge = AudioBridge()

    # Audio from A should route strictly to B, NEVER to A
    pcm_a = np.ones(2400, dtype=np.float32) * 0.5  # 100 ms at 24k
    bridge.feed_output(source_speaker="A", pcm_24k=pcm_a)

    frames_for_b = bridge.get_pending_ingress_frames(target_speaker="B")
    assert len(frames_for_b) > 0
    assert bridge.get_pending_ingress_frames(target_speaker="A") == []

    pcm_b = np.ones(2400, dtype=np.float32) * 0.3
    bridge.feed_output(source_speaker="B", pcm_24k=pcm_b)
    assert len(bridge.get_pending_ingress_frames(target_speaker="A")) > 0


def test_audio_bridge_flushes_partial_response_into_padded_frame() -> None:
    bridge = AudioBridge(max_pending_frames=2)
    bridge.feed_output(source_speaker="A", pcm_24k=np.full(2400, 0.5, dtype=np.float32))

    assert bridge.pending_count("B") == 1
    assert bridge.finalize_response("A") == 1
    frames = bridge.get_pending_ingress_frames("B")
    assert len(frames) == 2
    assert frames[1].shape == (FRAME_SAMPLES_16K,)
    assert np.all(frames[1][:320] != 0.0)
    assert np.all(frames[1][320:] == 0.0)


def test_audio_bridge_overflow_fails_without_dropping_frames() -> None:
    bridge = AudioBridge(max_pending_frames=1)
    with pytest.raises(OverflowError, match="AudioBridge.*capacity"):
        bridge.feed_output(source_speaker="A", pcm_24k=np.ones(3840, dtype=np.float32))
    assert bridge.pending_count("B") == 0


def test_audio_bridge_clear_reports_discarded_audio() -> None:
    bridge = AudioBridge(max_pending_frames=4)
    bridge.feed_output(source_speaker="A", pcm_24k=np.ones(2400, dtype=np.float32))

    discarded = bridge.clear_channel("B")
    assert discarded == {"frames": 1, "remainder_samples": 320}
    assert bridge.pending_count("B") == 0


def test_dialogue_recorder_saves_wavs(tmp_path: Path) -> None:
    recorder = DialogueRecorder(output_dir=tmp_path)

    # 1 second of 16 kHz audio for inputs
    audio_16k = np.zeros(16000, dtype=np.float32)
    # 1 second of 24 kHz audio for outputs
    audio_24k = np.zeros(24000, dtype=np.float32)

    recorder.record_input("A", audio_16k)
    recorder.record_output("A", audio_24k)
    recorder.record_input("B", audio_16k)
    recorder.record_output("B", audio_24k)

    saved_paths = recorder.finalize()

    for key, path in saved_paths.items():
        assert path.is_file()
        with wave.open(str(path), "rb") as wf:
            if "output" in key:
                assert wf.getframerate() == 24000
                assert wf.getnframes() == 24000
            else:
                assert wf.getframerate() == 16000
                assert wf.getnframes() == 16000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2


def test_dialogue_event_logger(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    logger = DialogueEventLogger(log_file)

    logger.log("connect", speaker="A", details={"url": "ws://localhost/v1/realtime"})
    logger.log("frame_sent", speaker="A", turn_index=1, details={"frame_id": 0})
    logger.log("audio_received", speaker="B", turn_index=1, details={"packet_id": 1})
    logger.close()

    assert log_file.is_file()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    event0 = json.loads(lines[0])
    assert event0["event_type"] == "connect"
    assert event0["speaker"] == "A"
    assert "timestamp_utc" in event0
    assert "timestamp_ns" in event0


class MockAsyncWebSocket:
    def __init__(self, responses: list | None = None) -> None:
        import asyncio
        self.queue = asyncio.Queue()
        self.sent_messages = []
        self.closed = False
        if responses:
            for r in responses:
                self.queue.put_nowait(r)

    async def recv(self) -> str:
        if self.queue.empty():
            raise EOFError("No more messages")
        return await self.queue.get()

    async def send(self, data: str) -> None:
        self.sent_messages.append(data)

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> MockAsyncWebSocket:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


def _write_pcm16_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(float_to_pcm16_bytes(np.asarray(pcm, dtype=np.float32)))


def _make_mock_wav_delta(response_id: str, pcm_24k: np.ndarray, packet_id: int = 1) -> str:
    import base64
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes((np.clip(pcm_24k, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    b64_wav = base64.b64encode(buf.getvalue()).decode("ascii")
    first_frame = 0
    last_frame = max(0, (pcm_24k.size // 1920) - 1)
    return json.dumps({
        "type": "response.audio.delta",
        "delta": b64_wav,
        "response_id": response_id,
        "packet_id": packet_id,
        "generation_id": 1,
        "speaker_segment_id": "seg_001",
        "first_frame_id": first_frame,
        "last_frame_id": last_frame,
        "start_sample": 0,
        "end_sample": pcm_24k.size,
        "sample_rate": 24000,
    })


@pytest.mark.anyio
async def test_dialogue_orchestrator_mode_a_mock(tmp_path: Path) -> None:
    from dialogue_eval.scenarios import SCENARIO_NAME_NUMBER, generate_speech_for_text

    # Prepare mock server responses for Session
    tone_24k = np.sin(2 * np.pi * 220.0 * np.linspace(0, 0.16, 3840, dtype=np.float32))
    audio_delta = _make_mock_wav_delta("resp_1", tone_24k)

    server_responses = [
        json.dumps({"type": "session.created"}),
        json.dumps({"type": "session.updated"}),
        json.dumps({"type": "response.created", "response_id": "resp_1"}),
        json.dumps({"type": "response.output_text.delta", "delta": "Buongiorno Marco, codice 4829 registrato.", "response_id": "resp_1"}),
        audio_delta,
        json.dumps({"type": "response.done", "response_id": "resp_1"}),
    ]

    mock_ws = MockAsyncWebSocket(server_responses)

    async def mock_connector(url: str, **kwargs) -> MockAsyncWebSocket:
        return mock_ws

    config = DialogueConfig(max_turns=1, silence_timeout_s=0.5)
    orchestrator = DialogueOrchestrator(
        config=config,
        output_dir=tmp_path / "mode_a",
        ws_connect_fn=mock_connector,
    )

    # Provide synthetic PCM for user turn
    user_audio = generate_speech_for_text("Buongiorno, mi chiamo Marco e il mio codice ordine è 4829.")
    audio_inputs = {"turn_1_user": user_audio}

    results = await orchestrator.run_mode_a(
        scenario=SCENARIO_NAME_NUMBER,
        audio_inputs=audio_inputs,
    )

    assert results["status"] == "COMPLETED"
    assert "report" in results
    assert "wav_files" in results
    # Check that audio tracks were recorded
    assert (tmp_path / "mode_a" / "interlocutor_a_input.wav").is_file()
    assert (tmp_path / "mode_a" / "interlocutor_a_output.wav").is_file()
    # Check that JSONL events were written
    assert (tmp_path / "mode_a" / "dialogue_events.jsonl").is_file()
    assert results["report"]["mean_input_wer"] is None
    assert results["report"]["input_asr_available_turns"] == 0
    assert "ASR indipendente non disponibile" in results["markdown_report"]


@pytest.mark.anyio
async def test_dialogue_orchestrator_mode_a_uses_cached_audio_and_independent_asr(tmp_path: Path) -> None:
    from dialogue_eval.evaluator import calculate_wer
    from dialogue_eval.scenarios import DialogueScenario, ScenarioStep

    expected_input = "Buongiorno, mi chiamo Marco e il codice ordine è 4829."
    recognized_input = "Marco, codice ordine 4829."
    scenario = DialogueScenario(
        scenario_id="cached_audio_test",
        title="Cached audio and independent ASR",
        description="A cached utterance must be transcribed from its PCM and scored against the user utterance.",
        system_prompt_a="test",
        system_prompt_b="test",
        steps=[
            ScenarioStep("user_1", "user", expected_input),
            ScenarioStep("assistant_1", "assistant", "Ho registrato il tuo codice."),
        ],
    )
    audio_dir = tmp_path / "scenario_audio_qwen3tts"
    source_pcm = np.linspace(-0.4, 0.4, FRAME_SAMPLES_16K * 2, dtype=np.float32)
    cached_path = audio_dir / "cached_audio_test__user_1.wav"
    _write_pcm16_wav(cached_path, source_pcm, 16000)
    source_pcm = np.frombuffer(cached_path.read_bytes()[44:], dtype="<i2").astype(np.float32) / 32767.0

    tone_24k = np.sin(2 * np.pi * 220.0 * np.linspace(0, 0.16, 3840, dtype=np.float32))
    server_responses = [
        json.dumps({"type": "session.created"}),
        json.dumps({"type": "session.updated"}),
        json.dumps({"type": "response.created", "response_id": "resp_cached"}),
        json.dumps({"type": "response.output_text.delta", "delta": "Ho registrato il tuo codice.", "response_id": "resp_cached"}),
        _make_mock_wav_delta("resp_cached", tone_24k),
        json.dumps({"type": "response.done", "response_id": "resp_cached"}),
    ]
    mock_ws = MockAsyncWebSocket(server_responses)

    async def mock_connector(url: str, **kwargs) -> MockAsyncWebSocket:
        return mock_ws

    transcribed_audio = []

    def fake_asr(pcm: np.ndarray, sample_rate: int) -> str:
        transcribed_audio.append((pcm.copy(), sample_rate))
        return recognized_input

    orchestrator = DialogueOrchestrator(
        config=DialogueConfig(max_turns=1, silence_timeout_s=0.5, frame_pacing_s=0.001),
        output_dir=tmp_path / "cached_mode_a",
        ws_connect_fn=mock_connector,
        scenario_audio_dir=audio_dir,
        asr_transcriber=fake_asr,
    )
    result = await orchestrator.run_mode_a(scenario)

    assert len(transcribed_audio) == 1
    assert transcribed_audio[0][1] == 16000
    np.testing.assert_allclose(transcribed_audio[0][0], source_pcm, atol=1e-7)
    turn = result["report"]["turns"][0]
    assert turn["recognized_input_text"] == recognized_input
    assert turn["input_wer"] == calculate_wer(expected_input, recognized_input)
    assert turn["input_wer"] > 0.0
    events = [
        json.loads(line)
        for line in (tmp_path / "cached_mode_a" / "dialogue_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event["event_type"] == "scenario_audio_loaded"
        and event["details"].get("path") == str(cached_path)
        for event in events
    )
    assert any(
        event["event_type"] == "input_audio_transcribed"
        and event["details"].get("transcript") == recognized_input
        for event in events
    )


@pytest.mark.anyio
async def test_dialogue_orchestrator_mode_b_mock(tmp_path: Path) -> None:
    from dialogue_eval.scenarios import SCENARIO_NAME_NUMBER

    tone_24k = np.sin(2 * np.pi * 300.0 * np.linspace(0, 0.16, 3840, dtype=np.float32))
    ws_a = _TurnGatedRealtimeWebSocket("A", tone_24k)
    ws_b = _TurnGatedRealtimeWebSocket("B", tone_24k)
    connections = [ws_a, ws_b]

    async def connector(url: str, **kwargs):
        return connections.pop(0)

    audio_dir = tmp_path / "scenario_audio_qwen3tts"
    kickoff_pcm = np.full(FRAME_SAMPLES_16K * 2, 0.3, dtype=np.float32)
    kickoff_path = audio_dir / "mode_b_kickoff.wav"
    _write_pcm16_wav(kickoff_path, kickoff_pcm, 16000)

    orchestrator = DialogueOrchestrator(
        config=DialogueConfig(max_turns=2, max_duration_s=30.0, frame_pacing_s=0.001),
        output_dir=tmp_path / "mode_b",
        ws_connect_fn=connector,
        scenario_audio_dir=audio_dir,
    )
    results = await orchestrator.run_mode_b(scenario=SCENARIO_NAME_NUMBER)

    assert results["status"] == "COMPLETED"
    assert results["conversation_status"] == "TURN_TAKING_TRANSPORT_COMPLETE"
    assert [response["speaker"] for response in results["responses"]] == ["A", "B"]
    assert all(response["response_done_ns"] >= response["response_created_ns"] for response in results["responses"])
    assert len(results["turn_transfers"]) == 1
    for track in ["interlocutor_a_input", "interlocutor_a_output", "interlocutor_b_input", "interlocutor_b_output"]:
        assert (tmp_path / "mode_b" / f"{track}.wav").is_file(), f"Missing track: {track}"
    assert (tmp_path / "mode_b" / "dialogue_events.jsonl").is_file()
    assert (tmp_path / "mode_b" / "dialogue_report.md").is_file()
    with wave.open(str(results["wav_files"]["interlocutor_a_input"]), "rb") as wf:
        first_kickoff_samples = np.frombuffer(wf.readframes(kickoff_pcm.size), dtype="<i2")
    with wave.open(str(kickoff_path), "rb") as wf:
        expected_kickoff_samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    np.testing.assert_array_equal(first_kickoff_samples, expected_kickoff_samples)


class _TurnGatedRealtimeWebSocket:
    """Tiny server fake: a commit starts output after the speaker ingests one live frame."""

    def __init__(
        self,
        speaker: str,
        audio_24k: np.ndarray,
        *,
        reject_reused_session_commit: bool = False,
        ack_behavior: str = "valid",
        response_delay_s: float | None = 0.0,
        silence_ack_delay_s: float = 0.01,
        cancel_response: bool = False,
        cancel_then_done: bool = False,
        duplicate_create_after_cancel: bool = False,
    ) -> None:
        import asyncio

        self.speaker = speaker
        self.audio_24k = audio_24k
        self.reject_reused_session_commit = reject_reused_session_commit
        self.ack_behavior = ack_behavior
        self.response_delay_s = response_delay_s
        self.silence_ack_delay_s = silence_ack_delay_s
        self.cancel_response = cancel_response
        self.cancel_then_done = cancel_then_done
        self.duplicate_create_after_cancel = duplicate_create_after_cancel
        self.rejected_reuse_commit = False
        self.queue = asyncio.Queue()
        self.queue.put_nowait(json.dumps({"type": "session.created"}))
        self.sent_messages: list[tuple[dict, int]] = []
        self.ack_opt_in = False
        self.next_frame_id = 0
        self.last_acknowledged_frame_id = -1
        self.acknowledged_frame_ids: list[int] = []
        self.appended_frame_is_silence: list[bool] = []
        self.pending_frame_ids: set[int] = set()
        self.max_unacknowledged_appends = 0
        self.commit_frame_counts: list[int] = []
        self.commit_count = 0
        self._response_emitted = False
        self._pending_response_id: str | None = None
        self._pending_response_text = ""
        self.response_done_at_ns: int | None = None
        self.response_done_times_ns: list[int] = []
        self._ack_tasks = []
        self.closed = False

    async def recv(self) -> str:
        item = await self.queue.get()
        if item is None:
            raise EOFError("websocket closed")
        message = json.loads(item)
        if message.get("type") == "input_audio_buffer.consumed":
            frame_id = int(message["frame_id"])
            if frame_id in self.pending_frame_ids:
                self.pending_frame_ids.remove(frame_id)
                self.last_acknowledged_frame_id = frame_id
                self.acknowledged_frame_ids.append(frame_id)
        return item

    async def send(self, data: str) -> None:
        message = json.loads(data)
        sent_at_ns = __import__("time").monotonic_ns()
        self.sent_messages.append((message, sent_at_ns))
        message_type = message.get("type")
        if message_type == "session.update":
            self.ack_opt_in = message.get("session", {}).get("input_audio_buffer_acknowledgements") is True
            self.conditioning_ack_opt_in = (
                message.get("session", {}).get("input_audio_buffer_conditioning_acknowledgements") is True
            )
            self.queue.put_nowait(json.dumps({"type": "session.updated"}))
        elif message_type == "input_audio_buffer.append":
            frame_id = self.next_frame_id
            self.next_frame_id += 1
            audio_pcm = np.frombuffer(
                __import__("base64").b64decode(message["audio"]), dtype="<i2"
            )
            is_silence = audio_pcm.size > 0 and np.all(audio_pcm == 0)
            self.appended_frame_is_silence.append(bool(is_silence))
            if self.ack_opt_in:
                self.pending_frame_ids.add(frame_id)
                self.max_unacknowledged_appends = max(self.max_unacknowledged_appends, len(self.pending_frame_ids))

                async def acknowledge_after_listener_consumes() -> None:
                    delay_s = self.silence_ack_delay_s if is_silence else 0.01
                    await __import__("asyncio").sleep(delay_s)
                    if self.ack_behavior == "missing":
                        return
                    ack_frame_id = frame_id + 1 if self.ack_behavior == "out_of_order" else frame_id
                    ack = {
                        "type": "input_audio_buffer.consumed",
                        "frame_id": ack_frame_id,
                        "listener_output_frame_id": frame_id,
                    }
                    self.queue.put_nowait(json.dumps(ack))
                    if self.ack_behavior == "duplicate":
                        self.queue.put_nowait(json.dumps(ack))
                    if self.conditioning_ack_opt_in and self.ack_behavior != "missing_conditioning":
                        self.queue.put_nowait(json.dumps({
                            "type": "input_audio_buffer.conditioning_available",
                            "frame_id": frame_id,
                            "listener_output_frame_id": frame_id,
                            "conditioning_frame_available": self.ack_behavior != "no_conditioning",
                            "conditioning_frame_id": (
                                frame_id if self.ack_behavior != "no_conditioning" else None
                            ),
                        }))
                    if (
                        self.commit_count
                        and not self._response_emitted
                        and self.ack_behavior == "valid"
                        and self.response_delay_s == 0.0
                    ):
                        self._emit_response()

                self._ack_tasks.append(__import__("asyncio").create_task(acknowledge_after_listener_consumes()))
        elif message_type == "input_audio_buffer.commit":
            if self.ack_opt_in and self.pending_frame_ids:
                raise AssertionError("commit sent with listener acknowledgements still pending")
            if self.reject_reused_session_commit and self.commit_count >= 1 and not self.rejected_reuse_commit:
                self.rejected_reuse_commit = True
                self.queue.put_nowait(json.dumps({
                    "type": "error",
                    "error": {"code": "RESPONSE_ACTIVE", "message": "response still finalizing"},
                }))
                return
            self.commit_frame_counts.append(self.next_frame_id)
            self.commit_count += 1
            self._response_emitted = False
            self._pending_response_id = f"resp_{self.speaker.lower()}_{self.commit_count}"
            self._pending_response_text = (
                "Buongiorno, codice 4829."
                if self.speaker == "A"
                else "Ciao Marco, confermo il codice 4829."
            )
            self.queue.put_nowait(json.dumps({"type": "response.created", "response_id": self._pending_response_id}))
            if self.cancel_response:
                self.queue.put_nowait(json.dumps({
                    "type": "response.cancelled",
                    "reason": "replan",
                    "response_id": self._pending_response_id,
                }))
                if self.cancel_then_done:
                    self.queue.put_nowait(json.dumps({
                        "type": "response.done",
                        "response_id": self._pending_response_id,
                    }))
                if self.duplicate_create_after_cancel:
                    self.queue.put_nowait(json.dumps({
                        "type": "response.created",
                        "response_id": self._pending_response_id,
                    }))
                    self.queue.put_nowait(json.dumps({
                        "type": "response.done",
                        "response_id": self._pending_response_id,
                    }))
            elif self.response_delay_s == 0.0 and self.ack_behavior == "valid":
                self._emit_response()
            elif self.response_delay_s is not None and self.response_delay_s > 0.0:
                self._ack_tasks.append(__import__("asyncio").create_task(self._emit_response_after_delay()))

    async def _emit_response_after_delay(self) -> None:
        await __import__("asyncio").sleep(float(self.response_delay_s or 0.0))
        if not self.closed and not self._response_emitted:
            self._emit_response()

    def _emit_response(self) -> None:
        self._response_emitted = True
        response_id = self._pending_response_id
        if response_id is None:
            raise AssertionError("accepted commit did not create a response")
        self.queue.put_nowait(json.dumps({
            "type": "response.output_text.delta",
            "delta": self._pending_response_text,
            "response_id": response_id,
        }))
        self.queue.put_nowait(_make_mock_wav_delta(response_id, self.audio_24k))
        self.response_done_at_ns = __import__("time").monotonic_ns()
        self.response_done_times_ns.append(self.response_done_at_ns)
        self.queue.put_nowait(json.dumps({"type": "response.done", "response_id": response_id}))

    async def close(self) -> None:
        self.closed = True
        if self._ack_tasks:
            await __import__("asyncio").gather(*self._ack_tasks, return_exceptions=True)
        self.queue.put_nowait(None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


@pytest.mark.anyio
async def test_mode_b_serializes_turns_and_waits_for_consumed_ingress(tmp_path: Path) -> None:
    from dialogue_eval.scenarios import SCENARIO_NAME_NUMBER

    tone_24k = np.sin(2 * np.pi * 300.0 * np.linspace(0, 0.16, 3840, dtype=np.float32))
    ws_a = _TurnGatedRealtimeWebSocket("A", tone_24k, reject_reused_session_commit=True)
    ws_b = _TurnGatedRealtimeWebSocket("B", tone_24k)
    connections = [ws_a, ws_b]

    async def connector(url: str, **kwargs):
        return connections.pop(0)

    orchestrator = DialogueOrchestrator(
        config=DialogueConfig(max_turns=3, max_duration_s=30.0, frame_pacing_s=0.001),
        output_dir=tmp_path / "mode_b_serialized",
        ws_connect_fn=connector,
    )
    result = await orchestrator.run_mode_b(SCENARIO_NAME_NUMBER)

    assert result["turns_completed"] == 3
    assert [response["speaker"] for response in result["responses"]] == ["A", "B", "A"]
    assert all(response["generated_text"].strip() and response["audio_samples"] > 0 for response in result["responses"])
    assert len(ws_a.commit_frame_counts) == 2
    assert len(ws_b.commit_frame_counts) == 1

    # B receives no audio frames while A is speaking; each frame is paced only
    # after its listener-consumed acknowledgement, before B's commit.
    b_append_times = [
        sent_at_ns
        for message, sent_at_ns in ws_b.sent_messages
        if message["type"] == "input_audio_buffer.append"
    ]
    assert b_append_times
    assert ws_a.response_done_at_ns is not None
    assert ws_b.next_frame_id == len(b_append_times)
    assert ws_a.ack_opt_in and ws_b.ack_opt_in
    assert ws_a.max_unacknowledged_appends == 1
    assert ws_b.max_unacknowledged_appends == 1
    assert ws_a.acknowledged_frame_ids == list(range(ws_a.next_frame_id))
    assert ws_b.acknowledged_frame_ids == list(range(ws_b.next_frame_id))
    assert ws_a.commit_frame_counts[0] < ws_a.commit_frame_counts[1]
    assert len(ws_b.commit_frame_counts) == 1
    b_frames_after_commit = ws_b.appended_frame_is_silence[ws_b.commit_frame_counts[0]:]
    assert all(b_frames_after_commit)
    assert ws_a.rejected_reuse_commit is True
    assert result["turn_transfers"][-1]["commit_attempts"] == 2

    events = [
        json.loads(line)
        for line in (tmp_path / "mode_b_serialized" / "dialogue_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    a_done_index = next(
        index for index, event in enumerate(events)
        if event["event_type"] == "response.done" and event["speaker"] == "A"
    )
    a_done_ns = events[a_done_index]["timestamp_ns"]
    first_b_append_ns = min(b_append_times)
    assert first_b_append_ns >= a_done_ns
    b_bridge_index = next(
        index for index, event in enumerate(events)
        if event["event_type"] == "input_audio_frame_sent"
        and event["speaker"] == "B"
        and event["details"].get("source") == "bridge"
    )
    assert a_done_index < b_bridge_index
    b_done_index = next(
        index for index, event in enumerate(events)
        if event["event_type"] == "response.done" and event["speaker"] == "B"
    )
    a_bridge_index = next(
        index for index, event in enumerate(events)
        if event["event_type"] == "input_audio_frame_sent"
        and event["speaker"] == "A"
        and event["details"].get("source") == "bridge"
    )
    assert b_done_index < a_bridge_index
    assert (tmp_path / "mode_b_serialized" / "interlocutor_a_output.wav").stat().st_size > 44
    assert (tmp_path / "mode_b_serialized" / "interlocutor_b_output.wav").stat().st_size > 44


@pytest.mark.anyio
async def test_mode_b_feeds_conditioning_silence_only_after_response_created(tmp_path: Path) -> None:
    from dialogue_eval.scenarios import SCENARIO_NAME_NUMBER

    tone_24k = np.sin(2 * np.pi * 300.0 * np.linspace(0, 0.16, 3840, dtype=np.float32))
    ws_a = _TurnGatedRealtimeWebSocket(
        "A", tone_24k, response_delay_s=0.05, silence_ack_delay_s=0.2
    )
    ws_b = _TurnGatedRealtimeWebSocket("B", tone_24k)
    connections = [ws_a, ws_b]

    async def connector(url: str, **kwargs):
        return connections.pop(0)

    kickoff_pcm = np.full(FRAME_SAMPLES_16K * 2, 0.3, dtype=np.float32)
    audio_dir = tmp_path / "scenario_audio_qwen3tts"
    _write_pcm16_wav(audio_dir / "mode_b_kickoff.wav", kickoff_pcm, 16000)
    orchestrator = DialogueOrchestrator(
        config=DialogueConfig(max_turns=2, max_duration_s=2.0, frame_pacing_s=0.001),
        output_dir=tmp_path / "mode_b_waits_for_response",
        ws_connect_fn=connector,
        scenario_audio_dir=audio_dir,
    )

    result = await orchestrator.run_mode_b(SCENARIO_NAME_NUMBER)

    assert result["turns_completed"] == 2
    events = [
        json.loads(line)
        for line in (tmp_path / "mode_b_waits_for_response" / "dialogue_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    silence_events = [
        event for event in events
        if event["event_type"] == "conditioning_silence_frame_sent" and event["speaker"] == "A"
    ]
    assert silence_events
    assert ws_a.next_frame_id > 2
    a_created_ns = next(
        event["timestamp_ns"]
        for event in events
        if event["event_type"] == "response.created" and event["speaker"] == "A"
    )
    a_done_ns = next(
        event["timestamp_ns"]
        for event in events
        if event["event_type"] == "response.done" and event["speaker"] == "A"
    )
    assert all(a_created_ns <= event["timestamp_ns"] < a_done_ns for event in silence_events)
    first_b_bridge_ns = next(
        event["timestamp_ns"]
        for event in events
        if event["event_type"] == "input_audio_frame_sent"
        and event["speaker"] == "B"
        and event["details"].get("source") == "bridge"
    )
    assert first_b_bridge_ns >= a_done_ns
    silence_consumed_ns = next(
        event["timestamp_ns"]
        for event in events
        if event["event_type"] == "input_audio_frame_consumed"
        and event["speaker"] == "A"
        and event["details"].get("source") == "conditioning_silence"
    )
    assert first_b_bridge_ns < silence_consumed_ns


@pytest.mark.anyio
async def test_mode_b_max_duration_stops_waiting_for_incomplete_response(tmp_path: Path) -> None:
    from dialogue_eval.scenarios import SCENARIO_NAME_NUMBER

    tone_24k = np.sin(2 * np.pi * 300.0 * np.linspace(0, 0.16, 3840, dtype=np.float32))
    ws_a = _TurnGatedRealtimeWebSocket(
        "A", tone_24k, response_delay_s=None, silence_ack_delay_s=0.5
    )
    ws_b = _TurnGatedRealtimeWebSocket("B", tone_24k)
    connections = [ws_a, ws_b]

    async def connector(url: str, **kwargs):
        return connections.pop(0)

    kickoff_pcm = np.full(FRAME_SAMPLES_16K * 2, 0.3, dtype=np.float32)
    audio_dir = tmp_path / "scenario_audio_qwen3tts"
    _write_pcm16_wav(audio_dir / "mode_b_kickoff.wav", kickoff_pcm, 16000)
    orchestrator = DialogueOrchestrator(
        config=DialogueConfig(max_turns=2, max_duration_s=0.2, frame_pacing_s=0.001),
        output_dir=tmp_path / "mode_b_wait_timeout",
        ws_connect_fn=connector,
        scenario_audio_dir=audio_dir,
    )

    result = await orchestrator.run_mode_b(SCENARIO_NAME_NUMBER)

    assert result["stop_reason"] == "max_duration"
    assert result["turns_completed"] == 0
    assert result["ingress_failures"] == []
    assert ws_a.next_frame_id > 2
    events = [
        json.loads(line)
        for line in (tmp_path / "mode_b_wait_timeout" / "dialogue_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(
        event["event_type"] == "input_audio_buffer.conditioning_available"
        and event["speaker"] == "A"
        for event in events
    )


@pytest.mark.anyio
async def test_mode_b_aborts_a_cancelled_response_after_replan_grace(tmp_path: Path) -> None:
    from dialogue_eval.scenarios import SCENARIO_NAME_NUMBER

    tone_24k = np.sin(2 * np.pi * 300.0 * np.linspace(0, 0.16, 3840, dtype=np.float32))
    ws_a = _TurnGatedRealtimeWebSocket("A", tone_24k, cancel_response=True, cancel_then_done=True)
    ws_b = _TurnGatedRealtimeWebSocket("B", tone_24k)
    connections = [ws_a, ws_b]

    async def connector(url: str, **kwargs):
        return connections.pop(0)

    kickoff_pcm = np.full(FRAME_SAMPLES_16K * 2, 0.3, dtype=np.float32)
    audio_dir = tmp_path / "scenario_audio_qwen3tts"
    _write_pcm16_wav(audio_dir / "mode_b_kickoff.wav", kickoff_pcm, 16000)
    orchestrator = DialogueOrchestrator(
        config=DialogueConfig(
            max_turns=2,
            max_duration_s=0.5,
            silence_timeout_s=0.03,
            frame_pacing_s=0.001,
        ),
        output_dir=tmp_path / "mode_b_cancelled_response",
        ws_connect_fn=connector,
        scenario_audio_dir=audio_dir,
    )

    result = await orchestrator.run_mode_b(SCENARIO_NAME_NUMBER)

    assert result["stop_reason"] == "response_cancelled"
    assert result["turns_completed"] == 0
    assert ws_a.next_frame_id >= 2
    assert all(ws_a.appended_frame_is_silence[2:])


@pytest.mark.anyio
async def test_mode_b_reports_max_duration_when_cancel_replacement_wait_expires_with_run(tmp_path: Path) -> None:
    from dialogue_eval.scenarios import SCENARIO_NAME_NUMBER

    tone_24k = np.sin(2 * np.pi * 300.0 * np.linspace(0, 0.16, 3840, dtype=np.float32))
    ws_a = _TurnGatedRealtimeWebSocket("A", tone_24k, cancel_response=True)
    ws_b = _TurnGatedRealtimeWebSocket("B", tone_24k)
    connections = [ws_a, ws_b]

    async def connector(url: str, **kwargs):
        return connections.pop(0)

    kickoff_pcm = np.full(FRAME_SAMPLES_16K * 2, 0.3, dtype=np.float32)
    audio_dir = tmp_path / "scenario_audio_qwen3tts"
    _write_pcm16_wav(audio_dir / "mode_b_kickoff.wav", kickoff_pcm, 16000)
    orchestrator = DialogueOrchestrator(
        config=DialogueConfig(
            max_turns=2,
            max_duration_s=0.3,
            silence_timeout_s=1.0,
            frame_pacing_s=0.001,
        ),
        output_dir=tmp_path / "mode_b_cancel_max_duration",
        ws_connect_fn=connector,
        scenario_audio_dir=audio_dir,
    )

    result = await orchestrator.run_mode_b(SCENARIO_NAME_NUMBER)

    assert result["stop_reason"] == "max_duration"
    assert result["turns_completed"] == 0
    assert ws_a.next_frame_id >= 2
    assert all(ws_a.appended_frame_is_silence[2:])


@pytest.mark.anyio
async def test_mode_b_fails_closed_on_duplicate_created_after_cancel(tmp_path: Path) -> None:
    from dialogue_eval.scenarios import SCENARIO_NAME_NUMBER

    tone_24k = np.sin(2 * np.pi * 300.0 * np.linspace(0, 0.16, 3840, dtype=np.float32))
    ws_a = _TurnGatedRealtimeWebSocket(
        "A",
        tone_24k,
        cancel_response=True,
        duplicate_create_after_cancel=True,
    )
    ws_b = _TurnGatedRealtimeWebSocket("B", tone_24k)
    connections = [ws_a, ws_b]

    async def connector(url: str, **kwargs):
        return connections.pop(0)

    kickoff_pcm = np.full(FRAME_SAMPLES_16K * 2, 0.3, dtype=np.float32)
    audio_dir = tmp_path / "scenario_audio_qwen3tts"
    _write_pcm16_wav(audio_dir / "mode_b_kickoff.wav", kickoff_pcm, 16000)
    orchestrator = DialogueOrchestrator(
        config=DialogueConfig(max_turns=2, max_duration_s=1.0, frame_pacing_s=0.001),
        output_dir=tmp_path / "mode_b_duplicate_create",
        ws_connect_fn=connector,
        scenario_audio_dir=audio_dir,
    )

    result = await orchestrator.run_mode_b(SCENARIO_NAME_NUMBER)

    assert result["status"] == "FAILED"
    assert result["turns_completed"] == 0
    assert any("duplicate response.created" in error for error in result["ingress_failures"])


@pytest.mark.anyio
async def test_mode_a_stops_ambient_ingress_and_records_server_error(tmp_path: Path) -> None:
    import asyncio
    from dialogue_eval.scenarios import DialogueScenario, ScenarioStep

    class ErrorWebSocket:
        def __init__(self) -> None:
            self.messages = asyncio.Queue()
            self.messages.put_nowait(json.dumps({"type": "session.created"}))
            self.session_updates = []

        async def send(self, raw: str) -> None:
            message = json.loads(raw)
            if message["type"] == "session.update":
                self.session_updates.append(message["session"])
                self.messages.put_nowait(json.dumps({"type": "session.updated"}))
            elif message["type"] == "input_audio_buffer.commit":
                self.messages.put_nowait(json.dumps({
                    "type": "error",
                    "error": {"code": "INFERENCE_ERROR", "message": "synthetic decoder failure"},
                }))

        async def recv(self) -> str:
            return await self.messages.get()

        async def close(self) -> None:
            self.messages.put_nowait(None)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            await self.close()

    ws = ErrorWebSocket()

    async def connector(_url: str, **_kwargs):
        return ws

    scenario = DialogueScenario(
        scenario_id="server_error",
        title="Server error handling",
        description="A server-side inference error ends ambient input immediately.",
        system_prompt_a="test",
        system_prompt_b="test",
        steps=[
            ScenarioStep("user", "user", "hello"),
            ScenarioStep("assistant", "assistant", "hello"),
        ],
    )
    orchestrator = DialogueOrchestrator(
        config=DialogueConfig(frame_pacing_s=0.001, silence_timeout_s=0.05),
        output_dir=tmp_path / "mode_a_server_error",
        ws_connect_fn=connector,
    )
    result = await orchestrator.run_mode_a(
        scenario,
        audio_inputs={"user": np.ones(FRAME_SAMPLES_16K, dtype=np.float32) * 0.1},
    )

    assert ws.session_updates[0]["turn_policy"] == "none"
    assert ws.session_updates[0]["auto_response"] is False
    assert result["status"] == "FAILED"
    assert result["server_errors"] == [{"code": "INFERENCE_ERROR", "message": "synthetic decoder failure"}]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "ack_behavior",
    ["missing", "out_of_order", "duplicate", "no_conditioning", "missing_conditioning"],
)
async def test_mode_b_rejects_missing_or_invalid_conditioning_ack(tmp_path: Path, ack_behavior: str) -> None:
    from dialogue_eval.scenarios import SCENARIO_NAME_NUMBER

    tone_24k = np.ones(3840, dtype=np.float32) * 0.1
    ws_a = _TurnGatedRealtimeWebSocket("A", tone_24k, ack_behavior=ack_behavior)
    ws_b = _TurnGatedRealtimeWebSocket("B", tone_24k)
    connections = [ws_a, ws_b]

    async def connector(url: str, **kwargs):
        return connections.pop(0)

    orchestrator = DialogueOrchestrator(
        config=DialogueConfig(
            max_turns=2,
            max_duration_s=3.0,
            frame_pacing_s=0.001,
            ingress_ack_timeout_s=0.05,
        ),
        output_dir=tmp_path / f"mode_b_bad_ack_{ack_behavior}",
        ws_connect_fn=connector,
    )
    result = await orchestrator.run_mode_b(SCENARIO_NAME_NUMBER)

    assert result["status"] == "FAILED"
    assert result["turns_completed"] == 0
    assert result["ingress_failures"]
