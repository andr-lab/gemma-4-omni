"""Bounded realtime WebSocket probes for the live duplex response service.

``single`` streams a prerecorded microphone track while collecting ACKs and
response audio. ``pair`` routes each participant's generated PCM to the other
participant as it arrives, paced at 80 ms, without waiting for response.done.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import threading
import time
from typing import Any
import wave

import numpy as np
import scipy.signal
import websockets

from client.duplex_dialogue_harness import decode_response_audio, float_to_pcm16_bytes
from duplex_kernel.tts_quality_diagnostics import SYNTHETIC_DIALOGUE_FIXTURE_ID


FRAME_MS = 80
FRAME_SAMPLES = 1280
INPUT_RATE = 16_000
TURN_POLICY = "gemma_native_control6"
OUTPUT_RATE = 24_000
INPUT_CAPACITY = 6
PAIR_AUDIO_QUEUE_CAPACITY = 256
PAIR_ROUTE_QUEUE_CAPACITY_CHUNKS = 8
ENDPOINT_SILENCE_FRAMES = 10
MANUAL_SEED_ACK_TIMEOUT_S = 10.0
_NATIVE_CONTROL6_SELECTION_FIELDS = (
    "native_control6_raw_row_index",
    "native_control6_raw_token_id",
    "native_control6_raw_token",
    "native_control6_selected_row_index",
    "native_control6_selected_token_id",
    "native_control6_selected_token",
    "native_control6_token_selection_scope",
)
_SCHEDULER_TRACE_FIELDS = (
    "scheduler_clock",
    "scheduler_enqueue_ns",
    "scheduler_deadline_ns",
    "scheduler_observed_ns",
    "scheduler_listener_start_ns",
    "scheduler_listener_end_ns",
    "scheduler_enqueue_to_deadline_ms",
    "scheduler_deadline_lateness_ms",
    "scheduler_observed_to_listener_start_ms",
    "scheduler_listener_callback_ms",
)
_SERVER_EVENT_LOG_FIELDS = (
    "legal_actions",
    "policy_id",
    "turn_policy_id",
    "turn_policy_class",
    "native_control6_class_match",
    "native_control6_step_called",
    "native_control6_head_scored",
    "native_control_hidden_supplied",
    "native_control_hidden_frame_id",
    "native_control_no_input_tick",
    "reason",
    "native_action_order",
    "native_action_base_logits",
    "native_action_logits",
    "policy_decision_at_ns",
    "listener_hidden_at_ns",
    "listener_hidden_age_frames",
    "listener_hidden_age_ms",
    "listener_query_reused",
    "listener_query_origin_frame_id",
    "listener_query_age_frames",
    "checkpoint_path",
    "source",
    "user_turn_pending",
    "user_active",
    "voice_run_onset_frame_id",
    "speech_endpoint_frame_id",
    "last_voiced_frame_id",
    "quiet_frames",
    "new_voice_onset",
    "speech_endpoint",
    "peer_turn_active",
    "backchannel_available",
    "pending_plan",
    "pending_user_onset_frame_id",
    "reserved_user_onset_frame_id",
    "last_answered_onset_frame_id",
    "response_generation_id",
    "response_pending",
    "active_response",
    "fresh_native_hidden",
    "native_hidden_frame_id",
    "native_no_input_tick",
    "audio_history",
    "semantic_prefix",
    "control6_context",
    "gate_applied_action",
    "gate_application_scope",
    "server_listener_queue_ms",
    "server_listener_step_ms",
    "server_listener_step_start_ns",
    "server_listener_step_end_ns",
    "server_ack_queue_ms",
    *_NATIVE_CONTROL6_SELECTION_FIELDS,
    *_SCHEDULER_TRACE_FIELDS,
)


def _project_server_event_for_log(event: dict[str, Any]) -> dict[str, Any]:
    """Keep the bounded decision and host-timing fields needed for evidence."""
    kind = str(event.get("type", ""))
    projected: dict[str, Any] = {
        "type": kind,
        "response_id": event.get("response_id"),
        "frame_id": event.get("frame_id"),
        "action": event.get("action"),
        "confidence": event.get("confidence"),
        "accepted": event.get("accepted"),
        "rejection_reason": event.get("rejection_reason"),
        "native_action": event.get("native_action"),
        "native_action_probability": event.get("native_action_probability"),
        "native_action_probabilities": event.get("native_action_probabilities"),
        "selected_action": event.get("selected_action"),
        "selected_action_probability": event.get("selected_action_probability"),
        "listener_frame_id": event.get("listener_frame_id"),
        "feature_contract_input_dim": event.get("feature_contract_input_dim"),
        "acoustic_phase_features_used": event.get("acoustic_phase_features_used"),
        "text_delta": event.get("delta") if kind == "response.output_text.delta" else None,
        "wake_action": event.get("wake_action") if kind == "response.action_selected" else None,
        "response_action": event.get("response_action") if kind == "response.action_selected" else None,
        "deferred_until_endpoint": event.get("deferred_until_endpoint") if kind == "response.action_selected" else None,
        "status": event.get("status") if kind in {"response.action_selected", "response.done"} else None,
        "runtime_metrics": event.get("runtime_metrics") if kind == "response.done" else None,
        "error": event.get("error") if kind == "error" else None,
    }
    for field in _SERVER_EVENT_LOG_FIELDS:
        if field in event:
            projected[field] = event[field]
    if event.get("native_control6_head_scored") is not True:
        for field in _NATIVE_CONTROL6_SELECTION_FIELDS:
            projected.pop(field, None)
    return projected


def _project_input_ack_fields(event: dict[str, Any]) -> dict[str, Any]:
    """Copy server-side ACK timing without inventing unobserved callback ends."""
    fields = (
        "server_listener_queue_ms",
        "server_listener_step_ms",
        "server_listener_step_start_ns",
        "server_listener_step_end_ns",
        "server_ack_queue_ms",
        *_SCHEDULER_TRACE_FIELDS,
    )
    return {field: event[field] for field in fields if field in event}


def read_audio_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width != 2 or channels not in (1, 2):
        raise ValueError(f"expected mono/stereo PCM16 WAV: {path}")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels == 2:
        pcm = pcm.reshape(-1, 2).mean(axis=1)
    if rate != INPUT_RATE:
        divisor = math.gcd(rate, INPUT_RATE)
        pcm = scipy.signal.resample_poly(pcm, INPUT_RATE // divisor, rate // divisor).astype(np.float32)
    return np.clip(pcm, -1.0, 1.0)


def to_frames(pcm: np.ndarray) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for start in range(0, pcm.size, FRAME_SAMPLES):
        frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        chunk = pcm[start:start + FRAME_SAMPLES]
        frame[:chunk.size] = chunk
        frames.append(frame)
    return frames


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="higher"))


def raw_audio_quality(pcm: np.ndarray) -> dict[str, Any]:
    """Measure received float PCM before the response WAV's int16 clipping step."""
    samples = np.asarray(pcm, dtype=np.float64).reshape(-1)
    finite = bool(np.isfinite(samples).all())
    if finite and samples.size:
        peak = round(float(np.max(np.abs(samples))), 8)
        rms = round(float(np.sqrt(np.mean(np.square(samples)))), 8)
    else:
        peak = None
        rms = None
    return {
        "finite": finite,
        "sample_count": int(samples.size),
        "peak": peak,
        "rms": rms,
        "clipping_rail_samples": int(np.count_nonzero((samples <= -1.0) | (samples >= 1.0))),
    }


def build_single_session_update(
    instructions: str,
    *,
    voice_conditioning_mode: str | None = "x_vector_only",
    tts_quality_diagnostics: bool = False,
    diagnostic_seed_sha256: str | None = None,
) -> dict[str, Any]:
    update: dict[str, Any] = {
        "auto_response": True,
        "voice": "gemma",
        "turn_policy": TURN_POLICY,
        "input_audio_buffer_acknowledgements": True,
        "input_audio_transcription": False,
    }
    if voice_conditioning_mode is not None:
        update["voice_conditioning_mode"] = voice_conditioning_mode
    if instructions.strip():
        update["instructions"] = instructions.strip()
    if tts_quality_diagnostics:
        digest = str(diagnostic_seed_sha256 or "").lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("synthetic TTS diagnostic requires the seed WAV SHA-256")
        update["tts_quality_diagnostic_fixture"] = SYNTHETIC_DIALOGUE_FIXTURE_ID
        update["tts_quality_diagnostic_seed_sha256"] = digest
    return update


class SingleStreamProbe:
    def __init__(
        self,
        ws_url: str,
        output_dir: Path,
        instructions: str = "",
        *,
        voice_conditioning_mode: str | None = "x_vector_only",
        tts_quality_diagnostics: bool = False,
    ) -> None:
        self.ws_url = ws_url
        self.output_dir = output_dir
        self.instructions = instructions.strip()
        self.voice_conditioning_mode = voice_conditioning_mode
        self.tts_quality_diagnostics = bool(tts_quality_diagnostics)
        self.diagnostic_seed_sha256: str | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.responses: dict[str, dict[str, object]] = {}
        self.ack_ids: list[int] = []
        self.ack_sent = deque()
        self.ack_latency_ms: list[float] = []
        self.server_listener_queue_ms: list[float] = []
        self.server_listener_step_ms: list[float] = []
        self.server_ack_queue_ms: list[float] = []
        self.input_send_times: list[float] = []
        self.pending_high_water = 0
        self.ack_duplicates = 0
        self.ack_out_of_order = 0
        self.response_active: set[str] = set()
        self.first_response_created = asyncio.Event()
        self.sender_done = asyncio.Event()
        self.errors: list[dict[str, object]] = []
        self.sent_frames = 0
        self.input_frames_during_audio = 0
        self.input_voice_frames_during_audio = 0
        self.input_frames_during_audio_generation = 0
        self.non_silent_frames_during_audio_generation = 0
        self.overflow = False
        self.start_ns = 0

    def log(self, event: dict[str, object]) -> None:
        row = {"at_ms": round((time.monotonic_ns() - self.start_ns) / 1_000_000.0, 3), **event}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def run(self, first_wav: Path, followup_wav: Path | None, duration_s: float) -> dict[str, object]:
        self.start_ns = time.monotonic_ns()
        if self.tts_quality_diagnostics:
            digest = hashlib.sha256()
            with first_wav.open("rb") as seed_handle:
                for block in iter(lambda: seed_handle.read(1024 * 1024), b""):
                    digest.update(block)
            self.diagnostic_seed_sha256 = digest.hexdigest()
        first_frames = to_frames(read_audio_16k(first_wav))
        followup_frames = to_frames(read_audio_16k(followup_wav)) if followup_wav else []
        total_frames = math.ceil(duration_s * 1000.0 / FRAME_MS)
        silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        response_done = asyncio.Event()
        sender_error: str | None = None

        async with websockets.connect(self.ws_url, max_size=32 * 1024 * 1024, ping_interval=20, ping_timeout=20) as ws:
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if first.get("type") != "session.created":
                raise RuntimeError(f"expected session.created, received {first.get('type')}")
            session_id = str(first.get("session_id", ""))
            ack_capacity = int(first.get("input_audio_buffer_capacity_frames", INPUT_CAPACITY))
            session_update = build_single_session_update(
                self.instructions,
                voice_conditioning_mode=self.voice_conditioning_mode,
                tts_quality_diagnostics=self.tts_quality_diagnostics,
                diagnostic_seed_sha256=self.diagnostic_seed_sha256,
            )
            await ws.send(json.dumps({
                "type": "session.update",
                "session": session_update,
            }))
            while True:
                ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                if ready.get("type") == "session.updated":
                    if (ready.get("turn_policy") != TURN_POLICY
                            or ready.get("voice_conditioning_mode") != self.voice_conditioning_mode
                            or (self.voice_conditioning_mode is None
                                and ready.get("voice_conditioning_ablation") is not True)):
                        raise RuntimeError(f"session contract rejected: {ready}")
                    break
                if ready.get("type") == "error":
                    raise RuntimeError(f"session update rejected: {ready.get('error')}")

            async def receive() -> None:
                nonlocal response_done
                expected_ack = 0
                while not self.sender_done.is_set() or self.ack_sent or self.response_active:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    event = json.loads(raw)
                    kind = event.get("type")
                    self.log({"type": kind, "frame_id": event.get("frame_id"), "response_id": event.get("response_id"), "action": event.get("action")})
                    if kind == "input_audio_buffer.consumed":
                        frame_id = int(event.get("frame_id", -1))
                        if frame_id in self.ack_ids:
                            self.ack_duplicates += 1
                            continue
                        if frame_id != expected_ack:
                            self.ack_out_of_order += 1
                        expected_ack = frame_id + 1
                        self.ack_ids.append(frame_id)
                        if self.ack_sent:
                            _local_frame_id, sent_ns = self.ack_sent.popleft()
                            self.ack_latency_ms.append((time.monotonic_ns() - sent_ns) / 1_000_000.0)
                        for key, target in (
                            ("server_listener_queue_ms", self.server_listener_queue_ms),
                            ("server_listener_step_ms", self.server_listener_step_ms),
                            ("server_ack_queue_ms", self.server_ack_queue_ms),
                        ):
                            value = event.get(key)
                            if isinstance(value, (int, float)):
                                target.append(float(value))
                    elif kind == "response.created":
                        response_id = str(event.get("response_id", ""))
                        self.response_active.add(response_id)
                        self.responses[response_id] = {
                            "response_id": response_id,
                            "action": event.get("action"),
                            "created_ns": time.monotonic_ns(),
                            "first_audio_ns": None,
                            "last_audio_ns": None,
                            "done_ns": None,
                            "text": "",
                            "audio": [],
                            "packets": 0,
                        }
                        self.first_response_created.set()
                    elif kind == "response.output_text.delta":
                        response = self.responses.get(str(event.get("response_id", "")))
                        if response is not None:
                            response["text"] = str(response["text"]) + str(event.get("delta", ""))
                    elif kind == "response.audio.delta":
                        response = self.responses.get(str(event.get("response_id", "")))
                        if response is not None:
                            packet = decode_response_audio(event)
                            if response["first_audio_ns"] is None:
                                response["first_audio_ns"] = time.monotonic_ns()
                            response["last_audio_ns"] = time.monotonic_ns()
                            response["audio"].append(packet.pcm)
                            response["packets"] = int(response["packets"]) + 1
                    elif kind in {"response.done", "response.cancelled"}:
                        response = self.responses.get(str(event.get("response_id", "")))
                        if response is not None:
                            response["done_ns"] = time.monotonic_ns()
                        self.response_active.discard(str(event.get("response_id", "")))
                        if not self.response_active and self.first_response_created.is_set():
                            response_done.set()
                    elif kind == "error":
                        self.errors.append(event)

            async def send_frames() -> None:
                nonlocal sender_error
                frames_sent = 0
                next_followup = max(0, min(total_frames, len(first_frames) + 2))
                followup_sent = False
                next_send_at = asyncio.get_running_loop().time()
                while frames_sent < total_frames:
                    if frames_sent < len(first_frames):
                        frame = first_frames[frames_sent]
                    elif followup_frames and not followup_sent and frames_sent >= next_followup and self.first_response_created.is_set():
                        frame = followup_frames[frames_sent - next_followup] if frames_sent - next_followup < len(followup_frames) else silence
                        if frames_sent - next_followup + 1 >= len(followup_frames):
                            followup_sent = True
                    else:
                        frame = silence
                    if len(self.ack_sent) >= ack_capacity:
                        self.overflow = True
                        sender_error = f"local ACK capacity reached ({len(self.ack_sent)}/{ack_capacity}) at frame {frames_sent}"
                        self.log({"type": "probe.input_capacity_reached", "frame_index": frames_sent, "pending": len(self.ack_sent), "capacity": ack_capacity})
                        break
                    now_ns = time.monotonic_ns()
                    if self.response_active:
                        self.input_frames_during_audio += 1
                        if float(np.sqrt(np.mean(frame * frame))) > 0.008:
                            self.input_voice_frames_during_audio += 1
                    generated_audio_active = any(
                        response.get("first_audio_ns") is not None
                        and int(response["first_audio_ns"]) <= now_ns
                        and int(response.get("done_ns") or now_ns) >= now_ns
                        for response in self.responses.values()
                    )
                    if generated_audio_active:
                        self.input_frames_during_audio_generation += 1
                        if float(np.sqrt(np.mean(frame * frame))) > 0.008:
                            self.non_silent_frames_during_audio_generation += 1
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(float_to_pcm16_bytes(frame)).decode("ascii"),
                    }))
                    self.ack_sent.append((frames_sent, now_ns))
                    self.sent_frames += 1
                    self.pending_high_water = max(self.pending_high_water, len(self.ack_sent))
                    self.input_send_times.append(time.monotonic())
                    self.log({"type": "probe.input_frame_sent", "frame_index": frames_sent, "pending": len(self.ack_sent), "rms": round(float(np.sqrt(np.mean(frame * frame))), 5)})
                    frames_sent += 1
                    next_send_at = max(next_send_at + FRAME_MS / 1000.0, asyncio.get_running_loop().time() + FRAME_MS / 1000.0 - 0.005)
                    await asyncio.sleep(max(0.0, next_send_at - asyncio.get_running_loop().time()))
                self.sender_done.set()

            recv_task = asyncio.create_task(receive(), name="probe-receive")
            send_task = asyncio.create_task(send_frames(), name="probe-send")
            try:
                await asyncio.wait_for(send_task, timeout=duration_s + 20)
                await asyncio.wait_for(asyncio.sleep(8), timeout=9)
            except Exception as exc:
                sender_error = sender_error or f"probe wait failed: {type(exc).__name__}: {exc}"
            finally:
                self.sender_done.set()
                try:
                    await ws.send(json.dumps({"type": "session.close"}))
                except Exception:
                    pass
                recv_task.cancel()
                await asyncio.gather(recv_task, return_exceptions=True)

        for response_id, response in self.responses.items():
            chunks = response["audio"]
            audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
            wav_path = self.output_dir / f"response_{response_id}.wav"
            if audio.size:
                pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
                with wave.open(str(wav_path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(OUTPUT_RATE)
                    wav.writeframes(pcm16.tobytes())
            created_ns = response["created_ns"]
            first_audio_ns = response["first_audio_ns"]
            done_ns = response["done_ns"]
            response["audio_samples"] = int(audio.size)
            response["audio_duration_s"] = round(audio.size / OUTPUT_RATE, 3)
            response["first_audio_ms"] = round((first_audio_ns - created_ns) / 1_000_000.0, 1) if first_audio_ns else None
            response["response_wall_ms"] = round((done_ns - created_ns) / 1_000_000.0, 1) if done_ns else None
            response["wav_path"] = str(wav_path) if audio.size else None
            response.pop("audio", None)

        send_gaps_ms = [1000.0 * (b - a) for a, b in zip(self.input_send_times, self.input_send_times[1:])]
        missing_ack_ids = sorted(set(range(self.sent_frames)) - set(self.ack_ids))
        report: dict[str, object] = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "single_realtime_stream",
            "ws_url": self.ws_url,
            "session_id": session_id,
            "input_wav": str(first_wav),
            "followup_wav": str(followup_wav) if followup_wav else None,
            "requested_input_duration_s": duration_s,
            "tts_quality_diagnostics": self.tts_quality_diagnostics,
            "diagnostic_seed_sha256": self.diagnostic_seed_sha256,
            "ack_capacity_frames": ack_capacity,
            "input_frames_sent": self.sent_frames,
            "input_frames_acked": len(self.ack_ids),
            "ack_ids_unique": len(set(self.ack_ids)) == len(self.ack_ids),
            "missing_ack_ids": missing_ack_ids,
            "duplicate_acks": self.ack_duplicates,
            "out_of_order_acks": self.ack_out_of_order,
            "capacity_overflow": self.overflow,
            "pending_high_water_frames": self.pending_high_water,
            "pending_after_close": len(self.ack_sent),
            "ack_latency_ms": {
                "p50": round(quantile(self.ack_latency_ms, 0.50), 2) if self.ack_latency_ms else None,
                "p95": round(quantile(self.ack_latency_ms, 0.95), 2) if self.ack_latency_ms else None,
                "max": round(max(self.ack_latency_ms), 2) if self.ack_latency_ms else None,
            },
            "server_listener_queue_ms": {
                "p50": round(quantile(self.server_listener_queue_ms, 0.50), 2) if self.server_listener_queue_ms else None,
                "p95": round(quantile(self.server_listener_queue_ms, 0.95), 2) if self.server_listener_queue_ms else None,
                "max": round(max(self.server_listener_queue_ms), 2) if self.server_listener_queue_ms else None,
            },
            "server_listener_step_ms": {
                "p50": round(quantile(self.server_listener_step_ms, 0.50), 2) if self.server_listener_step_ms else None,
                "p95": round(quantile(self.server_listener_step_ms, 0.95), 2) if self.server_listener_step_ms else None,
                "max": round(max(self.server_listener_step_ms), 2) if self.server_listener_step_ms else None,
            },
            "server_ack_queue_ms": {
                "p50": round(quantile(self.server_ack_queue_ms, 0.50), 2) if self.server_ack_queue_ms else None,
                "p95": round(quantile(self.server_ack_queue_ms, 0.95), 2) if self.server_ack_queue_ms else None,
                "max": round(max(self.server_ack_queue_ms), 2) if self.server_ack_queue_ms else None,
            },
            "input_send_interval_ms": {
                "p50": round(quantile(send_gaps_ms, 0.50), 2) if send_gaps_ms else None,
                "p95": round(quantile(send_gaps_ms, 0.95), 2) if send_gaps_ms else None,
                "max": round(max(send_gaps_ms), 2) if send_gaps_ms else None,
                "deadline_misses_gt_100ms": sum(value > 100 for value in send_gaps_ms),
            },
            "input_frames_during_response": self.input_frames_during_audio,
            "non_silent_input_frames_during_response": self.input_voice_frames_during_audio,
            "input_frames_during_audio_generation": self.input_frames_during_audio_generation,
            "non_silent_input_frames_during_audio_generation": self.non_silent_frames_during_audio_generation,
            "responses": list(self.responses.values()),
            "errors": self.errors,
            "sender_error": sender_error,
        }
        (self.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


class RealtimePairProbe:
    """Two live sessions connected by a bounded, paced PCM-only audio bridge."""

    _END_UTTERANCE = object()

    def __init__(self, ws_url: str, output_dir: Path, *, max_turns: int, max_duration_s: float,
                 max_created_responses: int | None = None,
                 endpoint_silence_frames: int = ENDPOINT_SILENCE_FRAMES,
                 manual_seed_bootstrap: bool = False,
                 tts_quality_diagnostics: bool = False,
                  voice_conditioning_mode: str | None = "x_vector_only",
                  input_audio_transcription: bool = False,
                  listener_silence_skip: bool = False,
                  cpu_audio_history_archive: bool = False,
                  instructions_a: str = "Sei una persona che organizza una cena con amici. Parla in modo naturale e condividi le tue preferenze.",
                 instructions_b: str = "Sei un assistente conversazionale italiano. Aiuta a scegliere insieme un ristorante e un orario con proposte concise.") -> None:
        self.ws_url = ws_url
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self._log_queue: queue.Queue[dict[str, Any] | None] | None = None
        self._log_thread: threading.Thread | None = None
        # `max_turns` is the target number of completed responses that actually
        # carried audio. Deferred/no-speech responses are reported but do not
        # consume the conversation target. A separate hard creation budget
        # bounds pathological response loops.
        self.max_turns = max_turns
        self.max_created_responses = max_created_responses or max(12, max_turns * 4)
        if self.max_created_responses < 1:
            raise ValueError("max_created_responses must be positive")
        self.max_duration_s = max_duration_s
        self.endpoint_silence_frames = int(endpoint_silence_frames)
        self.manual_seed_bootstrap = bool(manual_seed_bootstrap)
        if self.manual_seed_bootstrap and max_turns < 2:
            raise ValueError("manual_seed_bootstrap requires at least two audible response turns")
        self.tts_quality_diagnostics = bool(tts_quality_diagnostics)
        self.voice_conditioning_mode = voice_conditioning_mode
        self.input_audio_transcription = bool(input_audio_transcription)
        self.listener_silence_skip = bool(listener_silence_skip)
        self.cpu_audio_history_archive = bool(cpu_audio_history_archive)
        self.diagnostic_seed_sha256: str | None = None
        if self.endpoint_silence_frames < 0:
            raise ValueError("endpoint_silence_frames must be non-negative")
        self.instructions = {"A": str(instructions_a), "B": str(instructions_b)}
        self.started_ns = 0
        self.responses: dict[str, dict[str, Any]] = {}
        self.peers: dict[str, dict[str, Any]] = {}
        self.cancelled_response_ids: set[str] = set()
        self.cancellation_fences: dict[str, dict[str, Any]] = {}
        self.errors: list[dict[str, Any]] = []
        self.created = 0
        self.created_after_auto_response_disable_ack = 0
        self.completed = 0
        self.audible_completed = 0
        self.created_budget_exhausted = False
        self.budget_drain_completed = False
        self.created_target_event = asyncio.Event()
        self.target_drain_deadline: float | None = None
        self.target_drain_ready_at: float | None = None
        self.hard_deadline_at: float | None = None
        self.auto_response_disable_task: asyncio.Task | None = None
        self.auto_response_disable_reason: str | None = None
        self.auto_response_disable_started_ns: int | None = None
        self.auto_response_disable_completed_ns: int | None = None
        self.auto_response_disable_confirmed = False
        self.auto_response_disable_requested_snapshot: dict[str, Any] | None = None
        self.auto_response_disable_ack_created_responses: int | None = None
        self.auto_response_disable_ack_active_response_ids: dict[str, list[str]] | None = None
        self.stop_senders_requested = False
        self.pair_drain_completed = False
        self.drain_snapshot_at_timeout: dict[str, Any] | None = None
        self.bridge_stopped = False
        self.overflow = False
        self.stop = asyncio.Event()
        self._last_response_ns = 0

    def _fail_manual_seed_bootstrap(self, peer: dict[str, Any], reason: str,
                                    response_id: str | None = None) -> None:
        if (not peer.get("manual_seed_bootstrap")
                or peer.get("bootstrap_phase") in {"complete", "failed", "disabled"}):
            return
        peer["bootstrap_phase"] = "failed"
        peer["bootstrap_failure_reason"] = str(reason)
        bootstrap_ack = peer.get("bootstrap_auto_response_enable_ack")
        if bootstrap_ack is not None:
            bootstrap_ack.set()
        error = {
            "speaker": peer.get("speaker", "A"),
            "error": f"manual seed bootstrap failed: {reason}",
        }
        if response_id is not None:
            error["response_id"] = str(response_id)
        self.errors.append(error)
        self.log(str(peer.get("speaker", "A")), {
            "type": "probe.manual_seed_bootstrap_failed",
            "reason": str(reason),
            "response_id": str(response_id) if response_id is not None else None,
            "failed_at_ns": time.monotonic_ns(),
        })

    async def _enable_manual_seed_auto_response(self, peer: dict[str, Any]) -> None:
        send_started_at_ns = time.monotonic_ns()
        try:
            await peer["ws"].send(json.dumps({
                "type": "session.update",
                "session": {"auto_response": True},
            }))
        except Exception as exc:
            self._fail_manual_seed_bootstrap(
                peer, f"could not send auto_response=true update: {type(exc).__name__}: {exc}",
                peer.get("bootstrap_response_id"),
            )
            return
        sent_at_ns = time.monotonic_ns()
        peer["bootstrap_auto_response_enable_sent_at_ns"] = sent_at_ns
        self.log(str(peer["speaker"]), {
            "type": "probe.manual_seed_bootstrap_auto_response_enable_sent",
            "send_started_at_ns": send_started_at_ns,
            "sent_at_ns": sent_at_ns,
            "response_id": peer.get("bootstrap_response_id"),
            "auto_response": True,
        })
        try:
            await asyncio.wait_for(
                peer["bootstrap_auto_response_enable_ack"].wait(),
                timeout=MANUAL_SEED_ACK_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            self._fail_manual_seed_bootstrap(
                peer, "session.updated for auto_response=true was not received within the ACK wait limit",
                peer.get("bootstrap_response_id"),
            )
            return
        if peer.get("bootstrap_phase") != "failed":
            peer["bootstrap_phase"] = "complete"

    def _schedule_manual_seed_auto_response(self, peer: dict[str, Any]) -> None:
        if (peer.get("manual_seed_bootstrap")
                and peer.get("bootstrap_response_eligible")
                and peer.get("bootstrap_commit_sent")
                and (peer.get("auto_response_disable_requested") or self.created_budget_exhausted)):
            self._fail_manual_seed_bootstrap(
                peer, "response creation budget already requested auto_response=false",
                peer.get("bootstrap_response_id"),
            )
            return
        if (not peer.get("manual_seed_bootstrap")
                or not peer.get("bootstrap_response_eligible")
                or not peer.get("bootstrap_commit_sent")
                or peer.get("bootstrap_phase") in {"failed", "complete"}
                or peer.get("bootstrap_auto_response_enable_requested")):
            return
        peer["bootstrap_auto_response_enable_requested"] = True
        peer["bootstrap_response_eligible"] = False
        peer["bootstrap_auto_response_enable_task"] = asyncio.create_task(
            self._enable_manual_seed_auto_response(peer),
            name="pair-bootstrap-enable-auto-response-A",
        )

    def log(self, speaker: str, event: dict[str, Any]) -> None:
        row = {
            "at_ms": round((time.monotonic_ns() - self.started_ns) / 1_000_000.0, 3),
            "speaker": speaker,
            **event,
        }
        if self._log_queue is None:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            return
        self._log_queue.put(row)

    def _input_pipeline_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for speaker, peer in self.peers.items():
            snapshot[speaker] = {
                "active_response_ids": sorted(peer.get("active", set())),
                "peer_audio_queue_frames": peer["queue"].qsize(),
                "route_pending_items": int(peer.get("route_pending", 0)),
                "enqueued_pcm_frames": int(peer.get("queue_frames_enqueued", 0)),
                "sent_pcm_frames": int(peer.get("queued_audio_frames_sent", 0)),
                "unsent_pcm_frames": max(
                    0,
                    int(peer.get("queue_frames_enqueued", 0))
                    - int(peer.get("queued_audio_frames_sent", 0)),
                ),
                "pending_ack_frames": len(peer.get("ack_pending", ())),
            }
        return snapshot

    def _request_auto_response_disable(self, reason: str) -> None:
        if self.auto_response_disable_task is not None:
            return
        self.auto_response_disable_reason = str(reason)
        self.auto_response_disable_started_ns = time.monotonic_ns()
        self.auto_response_disable_requested_snapshot = self._input_pipeline_snapshot()
        for peer in self.peers.values():
            peer["auto_response_disable_requested"] = True
        self.log("SYSTEM", {
            "type": "probe.auto_response_disable_requested",
            "reason": self.auto_response_disable_reason,
            "requested_at_ns": self.auto_response_disable_started_ns,
            "created_responses": self.created,
            "active_response_ids": {
                speaker: sorted(peer["active"]) for speaker, peer in self.peers.items()
            },
            "pipeline_snapshot": self.auto_response_disable_requested_snapshot,
        })
        self.auto_response_disable_task = asyncio.create_task(
            self._disable_auto_response_for_peers(),
            name="pair-disable-auto-response",
        )

    async def _disable_auto_response_for_peers(self) -> None:
        peers = [peer for peer in self.peers.values() if hasattr(peer.get("ws"), "send")]
        if not peers:
            return
        results = await asyncio.gather(*(
            peer["ws"].send(json.dumps({
                "type": "session.update",
                "session": {"auto_response": False},
            }))
            for peer in peers
        ), return_exceptions=True)
        send_errors = [
            f"{peer['speaker']}: {type(result).__name__}: {result}"
            for peer, result in zip(peers, results)
            if isinstance(result, BaseException)
        ]
        if send_errors:
            self.errors.append({
                "error": "could not send harness auto_response=false update",
                "details": send_errors,
            })
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*(
                    peer["auto_response_disable_ack"].wait() for peer in peers
                )),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            pass
        unconfirmed = [
            peer["speaker"] for peer in peers
            if not peer["auto_response_disable_confirmed"]
        ]
        self.auto_response_disable_completed_ns = time.monotonic_ns()
        if unconfirmed:
            self.errors.append({
                "error": "session.update auto_response=false was not acknowledged",
                "unconfirmed_speakers": unconfirmed,
            })
            return
        self.auto_response_disable_confirmed = True
        self.auto_response_disable_ack_created_responses = self.created
        self.auto_response_disable_ack_active_response_ids = {
            speaker: sorted(peer["active"]) for speaker, peer in self.peers.items()
        }
        self.log("SYSTEM", {
            "type": "probe.auto_response_disabled_for_harness_drain",
            "reason": self.auto_response_disable_reason,
            "requested_at_ns": self.auto_response_disable_started_ns,
            "acknowledged_at_ns": self.auto_response_disable_completed_ns,
            "created_responses_at_ack": self.created,
            "budget_max_created_responses": self.max_created_responses,
            "active_response_ids": {
                speaker: sorted(peer["active"]) for speaker, peer in self.peers.items()
            },
            "pipeline_snapshot": self._input_pipeline_snapshot(),
        })

    def _start_log_writer(self) -> None:
        if self._log_queue is not None:
            raise RuntimeError("event log writer already running")
        self._log_queue = queue.Queue()

        def write_batches() -> None:
            pending: list[str] = []
            with self.events_path.open("a", encoding="utf-8") as handle:
                while True:
                    row = self._log_queue.get()
                    if row is None:
                        if pending:
                            handle.writelines(pending)
                            pending.clear()
                        handle.flush()
                        return
                    pending.append(json.dumps(row, ensure_ascii=False) + "\n")
                    if len(pending) >= 32:
                        handle.writelines(pending)
                        pending.clear()

        self._log_thread = threading.Thread(
            target=write_batches,
            name="duplex-pair-event-writer",
            daemon=True,
        )
        self._log_thread.start()

    def _stop_log_writer(self) -> None:
        log_queue = self._log_queue
        log_thread = self._log_thread
        if log_queue is None or log_thread is None:
            return
        log_queue.put(None)
        log_thread.join()
        self._log_queue = None
        self._log_thread = None

    async def _open_peer(self, speaker: str) -> dict[str, Any]:
        ws = await websockets.connect(
            self.ws_url, max_size=32 * 1024 * 1024, ping_interval=20, ping_timeout=20,
        )
        first = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if first.get("type") != "session.created":
            await ws.close()
            raise RuntimeError(f"{speaker}: expected session.created, got {first.get('type')}")
        peer: dict[str, Any] = {
            "speaker": speaker,
            "ws": ws,
            "session_id": str(first.get("session_id", "")),
            "ack_capacity": int(first.get("input_audio_buffer_capacity_frames", INPUT_CAPACITY)),
            "queue": asyncio.Queue(maxsize=PAIR_AUDIO_QUEUE_CAPACITY),
            "route_queue": asyncio.Queue(maxsize=PAIR_ROUTE_QUEUE_CAPACITY_CHUNKS),
            "route_pending": 0,
            "route_queue_high_water": 0,
            "queue_high_water": 0,
            "queue_frames_enqueued": 0,
            "queue_frames_sent": 0,
            "send_inflight_response_id": None,
            "queued_audio_frames_sent": 0,
            "endpoint_silence_frames_sent": 0,
            "live_silence_frames_sent": 0,
            "input_samples": [],
            "recent_user_frames": deque(maxlen=5),
            "ack_pending": deque(),
            "ack_wakeup": asyncio.Event(),
            "ack_ids": [],
            "ack_latency_ms": [],
            "ack_latency_by_frame": {},
            "ack_duplicates": 0,
            "ack_out_of_order": 0,
            "input_send_times": [],
            "input_send_gaps_ms": [],
            "input_frame_send_records": [],
            "sender_backpressure_ms": [],
            "pending_high_water": 0,
            "active": set(),
            "responses": [],
            "sender_task": None,
            "receiver_task": None,
            "resample_remainder": np.empty(0, dtype=np.float32),
            "resample_remainder_chunk_ids": np.empty(0, dtype=np.int64),
            "resample_remainder_response_id": None,
            "session_errors": [],
            "stop_sender": False,
            "auto_response_disable_requested": False,
            "auto_response_disable_confirmed": False,
            "auto_response_disable_ack": asyncio.Event(),
            "manual_seed_bootstrap": self.manual_seed_bootstrap and speaker == "A",
            "bootstrap_phase": "seed" if self.manual_seed_bootstrap and speaker == "A" else "disabled",
            "bootstrap_frame_ids": [],
            "bootstrap_seed_frame_count": 0,
            "bootstrap_seed_marker_seen": False,
            "bootstrap_tail_frames_required": 0,
            "bootstrap_tail_frames_scheduled": 0,
            "bootstrap_tail_frames_sent": 0,
            "bootstrap_tail_sent": False,
            "bootstrap_commit_sent": False,
            "bootstrap_commit_sent_at_ns": None,
            "bootstrap_ack_deadline_at": None,
            "bootstrap_response_id": None,
            "bootstrap_response_eligible": False,
            "bootstrap_auto_response_enable_requested": False,
            "bootstrap_auto_response_enable_confirmed": False,
            "bootstrap_auto_response_enable_ack": asyncio.Event(),
            "bootstrap_auto_response_enable_task": None,
            "bootstrap_failure_reason": None,
            "stop_mode_ack_wait_ms": [],
            "frames_discarded_after_response_limit": 0,
            "queue_frames_cancelled_dropped": 0,
        }
        session_config = {
            "instructions": self.instructions[speaker],
            "auto_response": not (self.manual_seed_bootstrap and speaker == "A"),
            "voice": "gemma",
            "turn_policy": TURN_POLICY,
            "input_audio_buffer_acknowledgements": True,
            "input_audio_transcription": self.input_audio_transcription,
            "audio_history_archive_cpu": self.cpu_audio_history_archive,
        }
        if self.voice_conditioning_mode is not None:
            session_config["voice_conditioning_mode"] = self.voice_conditioning_mode
        if self.tts_quality_diagnostics:
            session_config.update({
                "tts_quality_diagnostic_fixture": SYNTHETIC_DIALOGUE_FIXTURE_ID,
                "tts_quality_diagnostic_seed_sha256": self.diagnostic_seed_sha256,
            })
        await ws.send(json.dumps({
            "type": "session.update",
            "session": session_config,
        }))
        self.log(speaker, {
            "type": "probe.session_update_sent",
            "instructions": self.instructions[speaker],
            "voice": "gemma",
            "turn_policy": TURN_POLICY,
            "auto_response": session_config["auto_response"],
        })
        while True:
            ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            kind = ready.get("type")
            if kind == "session.updated":
                if (ready.get("turn_policy") != TURN_POLICY
                        or ready.get("voice_conditioning_mode") != self.voice_conditioning_mode
                        or (self.voice_conditioning_mode is None
                            and ready.get("voice_conditioning_ablation") is not True)
                        or (self.input_audio_transcription
                            and ready.get("input_audio_transcription_enabled") is not True)
                        or (self.cpu_audio_history_archive
                            and ready.get("audio_history_archive_cpu_enabled") is not True)
                        or (self.listener_silence_skip
                            and ready.get("listener_silence_skip_enabled") is not True)):
                    await ws.close()
                    raise RuntimeError(f"{speaker}: session contract rejected: {ready}")
                peer["session_updated"] = ready
                break
            if kind == "error":
                await ws.close()
                raise RuntimeError(f"{speaker}: session update rejected: {ready.get('error')}")
        self.peers[speaker] = peer
        return peer

    async def _queue_frame(
        self,
        receiver: dict[str, Any],
        frame: np.ndarray,
        *,
        source_response_id: str,
        source_chunk_ids: tuple[int, ...] = (),
    ) -> None:
        if self.bridge_stopped:
            return
        queue: asyncio.Queue = receiver["queue"]
        if queue.full():
            self.overflow = True
            raise RuntimeError(f"{receiver['speaker']}: bounded peer audio queue reached {PAIR_AUDIO_QUEUE_CAPACITY} frames")
        item = (
            np.asarray(frame, dtype=np.float32).copy(),
            source_response_id,
            tuple(int(value) for value in source_chunk_ids),
            time.monotonic_ns(),
        )
        queue.put_nowait(item)
        receiver["queue_frames_enqueued"] += 1
        receiver["queue_high_water"] = max(receiver["queue_high_water"], queue.qsize())

    @staticmethod
    def _filter_queue(queue: asyncio.Queue, keep_item) -> list[Any]:
        """Filter an asyncio queue without yielding, balancing its task ledger."""
        retained: list[Any] = []
        dropped: list[Any] = []
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            queue.task_done()
            if keep_item(item):
                retained.append(item)
            else:
                dropped.append(item)
        for item in retained:
            queue.put_nowait(item)
        return dropped

    def _fence_cancelled_response(
        self, sender: dict[str, Any], receiver: dict[str, Any], response_id: str,
    ) -> dict[str, Any]:
        """Fence one canceled response across route, resampler, and peer queues."""
        response_id = str(response_id)
        fence = self.cancellation_fences.get(response_id)
        if fence is None:
            fence = {
                "cancelled_at_ns": time.monotonic_ns(),
                "route_queue_items_dropped": 0,
                "route_queue_audio_items_dropped": 0,
                "peer_queue_items_dropped": 0,
                "peer_queue_audio_frames_dropped": 0,
                "inflight_route_chunks_dropped": 0,
                "inflight_peer_frames_dropped": 0,
                "resample_remainder_samples_dropped": 0,
                "peer_input_frames_committed_before_fence": 0,
                "peer_input_frames_committed_after_fence": 0,
                "last_committed_peer_input_frame_at_ns": None,
            }
            already_sent = [
                item for item in receiver.get("input_frame_send_records", [])
                if item.get("source_response_id") == response_id
            ]
            fence["peer_input_frames_committed_before_fence"] = len(already_sent)
            if already_sent:
                fence["last_committed_peer_input_frame_at_ns"] = max(
                    int(item["sent_at_ns"]) for item in already_sent
                )
            self.cancellation_fences[response_id] = fence
        self.cancelled_response_ids.add(response_id)

        route_drops = self._filter_queue(
            sender["route_queue"],
            lambda item: not (
                isinstance(item, tuple) and len(item) >= 2
                and item[1] == response_id and item[0] in {"audio", "finish"}
            ),
        )
        fence["route_queue_items_dropped"] += len(route_drops)
        audio_route_drops = sum(1 for item in route_drops if item[0] == "audio")
        fence["route_queue_audio_items_dropped"] += audio_route_drops
        sender["route_pending"] = max(0, sender.get("route_pending", 0) - len(route_drops))

        peer_drops = self._filter_queue(
            receiver["queue"],
            lambda item: not (
                isinstance(item, tuple) and item and item[0] is not self._END_UTTERANCE
                and len(item) >= 2 and item[1] == response_id
            ) and not (
                isinstance(item, tuple) and len(item) >= 3
                and item[0] is self._END_UTTERANCE and item[2] == response_id
            ),
        )
        fence["peer_queue_items_dropped"] += len(peer_drops)
        audio_peer_drops = sum(
            1 for item in peer_drops
            if isinstance(item, tuple) and item and item[0] is not self._END_UTTERANCE
            and len(item) >= 2 and item[1] == response_id
        )
        fence["peer_queue_audio_frames_dropped"] += audio_peer_drops
        receiver["queue_frames_cancelled_dropped"] = (
            receiver.get("queue_frames_cancelled_dropped", 0) + audio_peer_drops
        )

        if sender.get("resample_remainder_response_id") == response_id:
            remainder = sender.get("resample_remainder", np.empty(0, dtype=np.float32))
            fence["resample_remainder_samples_dropped"] += int(remainder.size)
            sender["resample_remainder"] = np.empty(0, dtype=np.float32)
            sender["resample_remainder_chunk_ids"] = np.empty(0, dtype=np.int64)
            sender["resample_remainder_response_id"] = None

        # Wake a sender that may have popped this response's frame and be waiting
        # for ACK capacity. It will re-check the fence before sending.
        ack_wakeup = receiver.get("ack_wakeup")
        if ack_wakeup is not None:
            ack_wakeup.set()
        self.log(sender["speaker"], {
            "type": "probe.response_cancel_fence",
            "response_id": response_id,
            "cancelled_at_ns": fence["cancelled_at_ns"],
            "route_queue_audio_items_dropped": audio_route_drops,
            "peer_queue_audio_frames_dropped": audio_peer_drops,
            "route_pending_after_fence": sender.get("route_pending", 0),
        })
        return fence

    def _enqueue_route_item(self, sender: dict[str, Any], item: tuple[Any, ...]) -> None:
        """Keep WebSocket control/ACK reads independent from paced peer forwarding."""
        if self.bridge_stopped:
            return
        response_id = str(item[1]) if len(item) > 1 else ""
        if response_id in self.cancelled_response_ids:
            if item[0] == "audio":
                fence = self.cancellation_fences[response_id]
                fence["late_audio_chunks_after_fence"] = fence.get("late_audio_chunks_after_fence", 0) + 1
            return
        route_queue = sender.get("route_queue")
        if route_queue is None:
            receiver = self.peers.get("B" if sender.get("speaker") == "A" else "A")
            if receiver is not None:
                if item[0] == "audio":
                    _, resp_id, chunk_id, pcm, _ = item
                    asyncio.create_task(self._route_packet(sender, receiver, resp_id, chunk_id, pcm))
                elif item[0] == "finish":
                    _, resp_id, quiet_frames, _ = item
                    asyncio.create_task(self._finish_peer_utterance(sender, receiver, resp_id, quiet_frames=quiet_frames))
            return
        if route_queue.full():
            self.overflow = True
            message = (
                f"{sender['speaker']}: bounded peer route queue reached "
                f"{PAIR_ROUTE_QUEUE_CAPACITY_CHUNKS} chunks"
            )
            self.errors.append({"speaker": sender["speaker"], "error": message})
            self.log(sender["speaker"], {
                "type": "probe.peer_audio_route_overflow",
                "capacity_chunks": PAIR_ROUTE_QUEUE_CAPACITY_CHUNKS,
                "queued_chunks": route_queue.qsize(),
            })
            return
        route_queue.put_nowait(item)
        sender["route_pending"] += 1
        sender["route_queue_high_water"] = max(
            sender["route_queue_high_water"], route_queue.qsize()
        )
        self.log(sender["speaker"], {
            "type": "probe.peer_audio_route_enqueued",
            "route_kind": item[0],
            "response_id": item[1],
            "route_queue_depth": route_queue.qsize(),
            "route_pending": sender["route_pending"],
        })

    async def _route_loop(self, sender: dict[str, Any]) -> None:
        """Resample and forward audio in FIFO order outside the WS reader."""
        receiver = self.peers["B" if sender["speaker"] == "A" else "A"]
        route_queue: asyncio.Queue = sender["route_queue"]
        while True:
            item = await route_queue.get()
            try:
                kind = item[0]
                response_id = str(item[1])
                if kind == "audio":
                    _, _, chunk_id, pcm, received_ns = item
                    if response_id in self.cancelled_response_ids:
                        self.cancellation_fences[response_id]["inflight_route_chunks_dropped"] += 1
                        continue
                    await self._route_packet(
                        sender, receiver, response_id, int(chunk_id), pcm
                    )
                    self.log(sender["speaker"], {
                        "type": "probe.peer_audio_route_completed",
                        "route_kind": "audio",
                        "response_id": response_id,
                        "output_chunk_id": int(chunk_id),
                        "reader_received_at_ns": int(received_ns),
                        "route_completed_at_ns": time.monotonic_ns(),
                        "route_queue_depth": route_queue.qsize(),
                    })
                elif kind == "finish":
                    _, _, quiet_frames, received_ns = item
                    await self._finish_peer_utterance(
                        sender, receiver, response_id, quiet_frames=quiet_frames
                    )
                    self.log(sender["speaker"], {
                        "type": "probe.peer_audio_route_completed",
                        "route_kind": "finish",
                        "response_id": response_id,
                        "reader_received_at_ns": int(received_ns),
                        "route_completed_at_ns": time.monotonic_ns(),
                        "route_queue_depth": route_queue.qsize(),
                    })
                else:
                    raise RuntimeError(f"unknown route queue item kind: {kind}")
            except Exception as exc:
                self.overflow = True
                self.errors.append({
                    "speaker": sender["speaker"],
                    "error": f"peer audio route failed: {type(exc).__name__}: {exc}",
                })
                return
            finally:
                sender["route_pending"] = max(0, sender["route_pending"] - 1)
                route_queue.task_done()

    async def _queue_seed_frames(self, peer: dict[str, Any], frames: list[np.ndarray]) -> None:
        """Feed the initial utterance through the bounded paced sender queue."""
        queue: asyncio.Queue = peer["queue"]
        for frame in frames:
            item = (
                np.asarray(frame, dtype=np.float32).copy(),
                "seed",
                (),
                time.monotonic_ns(),
            )
            # The sender is already running. Awaiting queue capacity applies
            # producer backpressure while it forwards frames at 80 ms cadence.
            await queue.put(item)
            peer["queue_frames_enqueued"] += 1
            peer["queue_high_water"] = max(peer["queue_high_water"], queue.qsize())

    async def _route_packet(
        self,
        sender: dict[str, Any],
        receiver: dict[str, Any],
        response_id: str,
        output_chunk_id: int | np.ndarray,
        pcm_24k: np.ndarray | None = None,
    ) -> None:
        if pcm_24k is None:
            pcm_24k = np.asarray(output_chunk_id, dtype=np.float32)
            output_chunk_id = 0
        # Reaching the audible-response target closes new live-silence ingress,
        # but audio from responses that are already active must still drain.
        if self.bridge_stopped:
            return
        # Output packets are routed immediately. The resample remainder is kept
        # across packets so 80 ms ingress frames are not reset at each delta.
        # The dedicated route worker awaits each resample in FIFO order. The
        # WebSocket reader remains free to process ACKs and control events.
        pcm_16k = await asyncio.to_thread(
            scipy.signal.resample_poly,
            np.asarray(pcm_24k, dtype=np.float32),
            2,
            3,
        )
        # The receive loop can fence this response while resampling is in its
        # worker thread. Do not commit the result after that await.
        if response_id in self.cancelled_response_ids:
            self.cancellation_fences[response_id]["inflight_route_chunks_dropped"] += 1
            return
        pcm_16k = np.asarray(pcm_16k, dtype=np.float32)
        remainder = sender.get("resample_remainder", np.empty(0, dtype=np.float32))
        combined = np.concatenate((remainder, pcm_16k)) if remainder.size else pcm_16k
        chunk_ids = np.full(pcm_16k.size, int(output_chunk_id), dtype=np.int64)
        prior_chunk_ids = sender.get("resample_remainder_chunk_ids", np.empty(0, dtype=np.int64))
        provenance = np.concatenate((prior_chunk_ids, chunk_ids)) if prior_chunk_ids.size else chunk_ids
        count = combined.size // FRAME_SAMPLES
        sender["resample_remainder"] = combined[count * FRAME_SAMPLES:].copy()
        sender["resample_remainder_chunk_ids"] = provenance[count * FRAME_SAMPLES:].copy()
        sender["resample_remainder_response_id"] = response_id if sender["resample_remainder"].size else None
        for index in range(count):
            if response_id in self.cancelled_response_ids:
                self.cancellation_fences[response_id]["inflight_route_chunks_dropped"] += count - index
                return
            start = index * FRAME_SAMPLES
            frame = combined[start:start + FRAME_SAMPLES]
            source_chunks = tuple(int(value) for value in np.unique(
                provenance[start:start + FRAME_SAMPLES]
            ))
            await self._queue_frame(
                receiver,
                frame,
                source_response_id=response_id,
                source_chunk_ids=source_chunks,
            )
            sender["input_routed_to_peer_frames"] = sender.get("input_routed_to_peer_frames", 0) + 1

    async def _finish_peer_utterance(
        self,
        sender: dict[str, Any],
        receiver: dict[str, Any],
        response_id: str,
        *,
        quiet_frames: int | None = None,
    ) -> None:
        if self.bridge_stopped or response_id in self.cancelled_response_ids:
            return
        remainder = sender.get("resample_remainder", np.empty(0, dtype=np.float32))
        if remainder.size:
            padded = np.zeros(FRAME_SAMPLES, dtype=np.float32)
            padded[:remainder.size] = remainder
            prior_chunks = sender.get("resample_remainder_chunk_ids", np.empty(0, dtype=np.int64))
            source_chunks = tuple(int(value) for value in np.unique(prior_chunks))
            await self._queue_frame(
                receiver,
                padded,
                source_response_id=response_id,
                source_chunk_ids=source_chunks,
            )
            sender["input_routed_to_peer_frames"] = sender.get("input_routed_to_peer_frames", 0) + 1
            sender["resample_remainder"] = np.empty(0, dtype=np.float32)
            sender["resample_remainder_chunk_ids"] = np.empty(0, dtype=np.int64)
            sender["resample_remainder_response_id"] = None
        if receiver["queue"].full():
            self.overflow = True
            raise RuntimeError(f"{receiver['speaker']}: bounded peer audio queue reached capacity before endpoint marker")
        if response_id in self.cancelled_response_ids:
            return
        receiver["queue"].put_nowait((
            self._END_UTTERANCE,
            self.endpoint_silence_frames if quiet_frames is None else int(quiet_frames),
            response_id,
        ))
        receiver["queue_high_water"] = max(receiver["queue_high_water"], receiver["queue"].qsize())

    async def _send_loop(self, peer: dict[str, Any]) -> None:
        ws = peer["ws"]
        silence_remaining = 0
        next_send_at = asyncio.get_running_loop().time()
        local_frame_id = 0
        silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        while True:
            if (peer.get("manual_seed_bootstrap")
                    and peer.get("bootstrap_phase") == "awaiting_ack"):
                if peer.get("stop_sender") or self.stop.is_set():
                    return
                frame_ids = list(peer.get("bootstrap_frame_ids", ()))
                if not peer.get("bootstrap_seed_frame_count"):
                    self._fail_manual_seed_bootstrap(peer, "seed audio contained no complete frames")
                    return
                ack_map = peer.get("ack_latency_by_frame", {})
                if not frame_ids or any(frame_id not in ack_map for frame_id in frame_ids):
                    peer["ack_wakeup"].clear()
                    if peer.get("stop_sender") or self.stop.is_set():
                        return
                    if frame_ids and all(frame_id in ack_map for frame_id in frame_ids):
                        continue
                    deadline = peer.get("bootstrap_ack_deadline_at")
                    remaining_s = (
                        max(0.0, deadline - asyncio.get_running_loop().time())
                        if deadline is not None else MANUAL_SEED_ACK_TIMEOUT_S
                    )
                    try:
                        await asyncio.wait_for(peer["ack_wakeup"].wait(), timeout=remaining_s)
                    except asyncio.TimeoutError:
                        self._fail_manual_seed_bootstrap(
                            peer, "seed or endpoint frame ACKs were not received within the ACK wait limit",
                        )
                        return
                    continue
                send_started_at_ns = time.monotonic_ns()
                # The receive loop must be able to bind an exceptionally fast
                # response.created while this WebSocket send is still awaiting
                # its transport flush. The commit is marked sent only below.
                peer["bootstrap_phase"] = "commit_inflight"
                try:
                    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                except Exception as exc:
                    self._fail_manual_seed_bootstrap(
                        peer, f"could not send seed commit: {type(exc).__name__}: {exc}",
                    )
                    return
                sent_at_ns = time.monotonic_ns()
                acknowledgements = [
                    {"frame_id": int(frame_id), **ack_map[int(frame_id)]}
                    for frame_id in frame_ids
                ]
                peer["bootstrap_commit_sent"] = True
                peer["bootstrap_commit_sent_at_ns"] = sent_at_ns
                if peer.get("bootstrap_phase") != "failed":
                    peer["bootstrap_phase"] = "awaiting_response"
                self.log(str(peer["speaker"]), {
                    "type": "probe.manual_seed_bootstrap_commit",
                    "send_started_at_ns": send_started_at_ns,
                    "sent_at_ns": sent_at_ns,
                    "frame_ids": [int(frame_id) for frame_id in frame_ids],
                    "seed_frame_count": int(peer["bootstrap_seed_frame_count"]),
                    "endpoint_silence_frame_count": int(peer["bootstrap_tail_frames_sent"]),
                    "acknowledgements": acknowledgements,
                })
                self._schedule_manual_seed_auto_response(peer)
                continue
            item: Any = None
            try:
                item = peer["queue"].get_nowait()
            except asyncio.QueueEmpty:
                if silence_remaining:
                    frame_metadata: dict[str, Any] = {}
                    if (peer.get("manual_seed_bootstrap")
                            and peer.get("bootstrap_phase") == "tail"
                            and peer.get("bootstrap_seed_marker_seen")
                            and peer.get("bootstrap_tail_frames_scheduled")
                            < peer.get("bootstrap_tail_frames_required")):
                        frame_metadata["manual_seed_bootstrap_tail"] = True
                        peer["bootstrap_tail_frames_scheduled"] += 1
                    item = (silence, "endpoint_silence", (), time.monotonic_ns(), frame_metadata)
                    silence_remaining -= 1
                elif peer["stop_sender"]:
                    if not peer["ack_pending"]:
                        if (
                            not any(target.get("active") for target in self.peers.values())
                            and not any(target.get("route_pending", 0) for target in self.peers.values())
                        ):
                            return
                        await asyncio.sleep(0.005)
                        continue
                    # Even during final drain, accept newly relayed PCM as soon
                    # as the sender has ACK capacity. Waiting only on ACK here
                    # can strand peer audio in the queue for several seconds.
                    peer["ack_wakeup"].clear()
                    get_audio = asyncio.create_task(peer["queue"].get())
                    get_ack = asyncio.create_task(peer["ack_wakeup"].wait())
                    done, pending = await asyncio.wait(
                        (get_audio, get_ack),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    if get_audio in done:
                        item = get_audio.result()
                    else:
                        continue
                else:
                    # A realtime microphone keeps advancing the listener even
                    # when no speech or peer audio is present. Generate one
                    # silence frame at the next 80 ms slot instead of pausing
                    # the peer's input clock after the endpoint marker. Wait
                    # for queued speech until that slot so output arriving in
                    # the meantime is never held behind a silence backlog.
                    until_next_frame = max(0.0, next_send_at - asyncio.get_running_loop().time())
                    if until_next_frame == 0.0:
                        item = (silence, "live_silence", (), time.monotonic_ns())
                    else:
                        try:
                            item = await asyncio.wait_for(peer["queue"].get(), timeout=until_next_frame)
                        except asyncio.TimeoutError:
                            item = (silence, "live_silence", (), time.monotonic_ns())

            if isinstance(item, tuple) and item and item[0] is self._END_UTTERANCE:
                if (peer.get("manual_seed_bootstrap")
                        and peer.get("bootstrap_phase") == "seed"
                        and len(item) > 3
                        and item[3] == "manual_seed_bootstrap"):
                    peer["bootstrap_seed_marker_seen"] = True
                    peer["bootstrap_tail_frames_required"] = max(0, int(item[1]))
                    peer["bootstrap_phase"] = "tail"
                    if peer["bootstrap_tail_frames_required"] == 0:
                        peer["bootstrap_tail_sent"] = True
                        peer["bootstrap_phase"] = "awaiting_ack"
                        peer["bootstrap_ack_deadline_at"] = (
                            asyncio.get_running_loop().time() + MANUAL_SEED_ACK_TIMEOUT_S
                        )
                silence_remaining = (
                    max(silence_remaining, int(item[1]))
                    if self.audible_completed < self.max_turns and not self.created_budget_exhausted else 0
                )
                continue
            if len(item) == 3:
                frame, source_response_id, enqueued_ns = item
                source_chunk_ids = ()
            else:
                frame, source_response_id, source_chunk_ids, enqueued_ns = item[:4]
            frame_metadata = item[4] if len(item) > 4 and isinstance(item[4], dict) else {}
            bootstrap_seed_frame = (
                peer.get("manual_seed_bootstrap")
                and peer.get("bootstrap_phase") == "seed"
                and source_response_id == "seed"
            )
            bootstrap_tail_frame = bool(frame_metadata.get("manual_seed_bootstrap_tail"))
            peer["send_inflight_response_id"] = source_response_id

            while (len(peer["ack_pending"]) >= peer["ack_capacity"]
                   and source_response_id not in self.cancelled_response_ids):
                stalled_at = time.monotonic_ns()
                peer["ack_wakeup"].clear()
                await peer["ack_wakeup"].wait()
                peer["sender_backpressure_ms"].append((time.monotonic_ns() - stalled_at) / 1_000_000.0)
            if source_response_id in self.cancelled_response_ids:
                self.cancellation_fences[source_response_id]["inflight_peer_frames_dropped"] += 1
                peer["queue_frames_cancelled_dropped"] = peer.get("queue_frames_cancelled_dropped", 0) + 1
                peer["send_inflight_response_id"] = None
                continue
            if self.bridge_stopped:
                peer["frames_discarded_after_response_limit"] += 1
                continue
            now = asyncio.get_running_loop().time()
            if now < next_send_at:
                await asyncio.sleep(next_send_at - now)
            if source_response_id in self.cancelled_response_ids:
                self.cancellation_fences[source_response_id]["inflight_peer_frames_dropped"] += 1
                peer["queue_frames_cancelled_dropped"] = peer.get("queue_frames_cancelled_dropped", 0) + 1
                peer["send_inflight_response_id"] = None
                continue

            payload = {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(float_to_pcm16_bytes(frame)).decode("ascii"),
            }
            send_started_ns = time.monotonic_ns()
            # Register before awaiting ws.send(): an ACK may be dispatched while
            # the WebSocket transport is flushing this frame.
            peer["ack_pending"].append((local_frame_id, send_started_ns))
            if bootstrap_seed_frame or bootstrap_tail_frame:
                peer["bootstrap_frame_ids"].append(local_frame_id)
            peer["pending_high_water"] = max(peer["pending_high_water"], len(peer["ack_pending"]))
            try:
                await ws.send(json.dumps(payload))
            except Exception as exc:
                if peer["ack_pending"] and peer["ack_pending"][-1][0] == local_frame_id:
                    peer["ack_pending"].pop()
                if (bootstrap_seed_frame or bootstrap_tail_frame) and peer["bootstrap_frame_ids"][-1] == local_frame_id:
                    peer["bootstrap_frame_ids"].pop()
                if bootstrap_seed_frame or bootstrap_tail_frame:
                    self._fail_manual_seed_bootstrap(
                        peer, f"could not send seed frame {local_frame_id}: {type(exc).__name__}: {exc}",
                    )
                    return
                if peer["stop_sender"]:
                    return
                raise
            sent_ns = time.monotonic_ns()
            if source_response_id in self.cancellation_fences:
                fence = self.cancellation_fences[source_response_id]
                fence["peer_input_frames_committed_after_fence"] += 1
                fence["last_committed_peer_input_frame_at_ns"] = sent_ns
            peer["pending_high_water"] = max(peer["pending_high_water"], len(peer["ack_pending"]))
            peer["input_samples"].append(np.asarray(frame, dtype=np.float32).copy())
            recent = peer["recent_user_frames"]
            current_rms = float(np.sqrt(np.mean(frame * frame)))
            history = list(recent)[-4:] + [np.asarray(frame, dtype=np.float32)]
            observed = np.concatenate(history) if history else np.empty(0, dtype=np.float32)
            past_window = np.zeros(5 * FRAME_SAMPLES, dtype=np.float32)
            if observed.size:
                past_window[-min(observed.size, past_window.size):] = observed[-past_window.size:]
            past_rms = float(np.sqrt(np.mean(past_window * past_window)))
            previous_rms = float(np.sqrt(np.mean(recent[-1] * recent[-1]))) if recent else 0.0
            log_current = math.log10(max(current_rms, 1e-6))
            log_previous = math.log10(max(previous_rms, 1e-6))
            recent.append(np.asarray(frame, dtype=np.float32).copy())
            peer["queue_frames_sent"] += 1
            if bootstrap_seed_frame:
                peer["bootstrap_seed_frame_count"] += 1
            if bootstrap_tail_frame:
                peer["bootstrap_tail_frames_sent"] += 1
                if peer["bootstrap_tail_frames_sent"] >= peer["bootstrap_tail_frames_required"]:
                    peer["bootstrap_tail_sent"] = True
                    peer["bootstrap_phase"] = "awaiting_ack"
                    peer["bootstrap_ack_deadline_at"] = (
                        asyncio.get_running_loop().time() + MANUAL_SEED_ACK_TIMEOUT_S
                    )
            peer["input_send_times"].append(time.monotonic())
            local_audio_response_ids = sorted(
                response_id
                for response_id in peer.get("active", ())
                if (self.responses.get(response_id) or {}).get("first_audio_ns") is not None
            )
            record = {
                "frame_id": local_frame_id,
                "sent_at_ns": sent_ns,
                "send_started_at_ns": send_started_ns,
                "source_response_id": source_response_id,
                "local_audio_response_ids": local_audio_response_ids,
            }
            peer.setdefault("input_frame_send_records", []).append(record)
            peer["send_inflight_response_id"] = None
            self.log(peer["speaker"], {
                "type": "probe.input_frame_sent",
                "frame_id": local_frame_id,
                "sent_at_ns": sent_ns,
                "send_started_at_ns": send_started_ns,
                "source_response_id": source_response_id,
                "source_response_audio_chunk_ids": list(source_chunk_ids),
                "bridge_queue_age_ms": round((sent_ns - enqueued_ns) / 1_000_000.0, 2),
                "pending_acks": len(peer["ack_pending"]),
                "during_local_audio_output": bool(local_audio_response_ids),
                "local_audio_response_ids": local_audio_response_ids,
                "rms": round(current_rms, 5),
                "user_log10_rms_current_80ms": round(float(np.clip(log_current, -6, 0)), 6),
                "user_log10_rms_past_400ms_inclusive": round(float(np.clip(math.log10(max(past_rms, 1e-6)), -6, 0)), 6),
                "user_log_rms_delta_clip6_div6": round(float(np.clip(log_current - log_previous, -6, 6) / 6), 6),
            })
            if local_frame_id:
                peer["input_send_gaps_ms"].append((peer["input_send_times"][-1] - peer["input_send_times"][-2]) * 1000.0)
            local_frame_id += 1
            next_send_at = max(next_send_at + FRAME_MS / 1000.0, asyncio.get_running_loop().time())
            if source_response_id == "live_silence":
                peer["live_silence_frames_sent"] = peer.get("live_silence_frames_sent", 0) + 1
            elif source_response_id == "endpoint_silence":
                peer["endpoint_silence_frames_sent"] = peer.get("endpoint_silence_frames_sent", 0) + 1
            else:
                peer["queued_audio_frames_sent"] = peer.get("queued_audio_frames_sent", 0) + 1

    async def _receive_loop(self, peer: dict[str, Any]) -> None:
        ws = peer["ws"]
        speaker = peer["speaker"]
        receiver = self.peers["B" if speaker == "A" else "A"]
        expected_ack = 0
        while not self.stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                if self.stop.is_set():
                    return
                self.errors.append({"speaker": speaker, "error": f"websocket receive: {type(exc).__name__}: {exc}"})
                if (peer.get("manual_seed_bootstrap")
                        and peer.get("bootstrap_phase") in {"awaiting_response", "commit_inflight", "awaiting_ack"}):
                    self._fail_manual_seed_bootstrap(
                        peer, f"websocket receive failed during bootstrap: {type(exc).__name__}: {exc}",
                        peer.get("bootstrap_response_id"),
                    )
                return
            event = json.loads(raw)
            kind = str(event.get("type", ""))
            normalized_event = _project_server_event_for_log(event)
            self.log(speaker, normalized_event)
            if kind == "input_audio_buffer.consumed":
                ack_id = int(event.get("frame_id", -1))
                if ack_id in peer["ack_ids"]:
                    peer["ack_duplicates"] += 1
                    continue
                if ack_id != expected_ack:
                    peer["ack_out_of_order"] += 1
                expected_ack = ack_id + 1
                peer["ack_ids"].append(ack_id)
                ack_received_ns = time.monotonic_ns()
                local_frame_id = None
                client_ack_latency_ms = None
                if peer["ack_pending"]:
                    local_frame_id, send_started_ns = peer["ack_pending"].popleft()
                    client_ack_latency_ms = (ack_received_ns - send_started_ns) / 1_000_000.0
                    peer["ack_latency_ms"].append(client_ack_latency_ms)
                    peer["ack_latency_by_frame"][int(local_frame_id)] = {
                        "frame_id": int(local_frame_id),
                        "ack_frame_id": ack_id,
                        "ack_received_at_ns": ack_received_ns,
                        "latency_ms": round(client_ack_latency_ms, 2),
                    }
                ack_event = {
                    "type": "probe.input_frame_ack",
                    "ack_frame_id": ack_id,
                    "client_frame_id": local_frame_id,
                    "ack_received_at_ns": ack_received_ns,
                    "client_ack_latency_ms": round(client_ack_latency_ms, 2)
                    if client_ack_latency_ms is not None else None,
                    "pending_acks_after_ack": len(peer["ack_pending"]),
                }
                ack_event.update(_project_input_ack_fields(normalized_event))
                self.log(speaker, ack_event)
                peer["ack_wakeup"].set()
            elif kind == "session.updated":
                if peer.get("bootstrap_auto_response_enable_requested"):
                    peer["bootstrap_auto_response_enable_confirmed"] = True
                    peer["bootstrap_auto_response_enable_ack"].set()
                    peer["session_updated"] = event
                    self.log(speaker, {
                        "type": "probe.manual_seed_bootstrap_auto_response_enable_ack",
                        "acknowledged_at_ns": time.monotonic_ns(),
                        "response_id": peer.get("bootstrap_response_id"),
                        "auto_response": event.get("auto_response", True),
                    })
                if peer["auto_response_disable_requested"]:
                    peer["auto_response_disable_confirmed"] = True
                    peer["auto_response_disable_ack"].set()
                    self.log(speaker, {
                        "type": "probe.auto_response_disable_ack",
                        "acknowledged_at_ns": time.monotonic_ns(),
                        "created_responses": self.created,
                    })
            elif kind == "response.created":
                self.created += 1
                if self.auto_response_disable_confirmed:
                    self.created_after_auto_response_disable_ack += 1
                    self.log(speaker, {
                        "type": "probe.response_created_observed_after_auto_response_disable_ack",
                        "response_id": str(event.get("response_id", "")),
                        "created_responses": self.created,
                        "disable_acknowledged_at_ns": self.auto_response_disable_completed_ns,
                    })
                response_id = str(event.get("response_id", ""))
                if (peer.get("manual_seed_bootstrap")
                        and peer.get("bootstrap_phase") in {"commit_inflight", "awaiting_response"}
                        and peer.get("bootstrap_response_id") is None):
                    peer["bootstrap_response_id"] = response_id
                    peer["bootstrap_response_created_at_ns"] = time.monotonic_ns()
                    self.log(speaker, {
                        "type": "probe.manual_seed_bootstrap_response_bound",
                        "response_id": response_id,
                        "created_at_ns": peer["bootstrap_response_created_at_ns"],
                        "action": event.get("action"),
                    })
                response = {
                    "speaker": speaker,
                    "response_id": response_id,
                    "action": event.get("action"),
                    "created_ns": time.monotonic_ns(),
                    "first_audio_ns": None,
                    "last_audio_ns": None,
                    "done_ns": None,
                    "text": "",
                    "audio": [],
                    "audio_packets": 0,
                    "audio_samples": 0,
                    "wake_action": None,
                    "response_action": None,
                    "deferred_until_endpoint": False,
                    "terminal_status": None,
                    "audio_archive_status": "incomplete",
                }
                self.responses[response_id] = response
                peer["responses"].append(response_id)
                peer["active"].add(response_id)
                self._last_response_ns = time.monotonic_ns()
                if self.created >= self.max_created_responses and not self.created_budget_exhausted:
                    self.created_budget_exhausted = True
                    self._request_auto_response_disable("max_created_responses")
                    self.log("SYSTEM", {
                        "type": "probe.response_creation_budget_reached",
                        "created_responses": self.created,
                        "max_created_responses": self.max_created_responses,
                        "audible_completed_responses": self.audible_completed,
                        "target_audible_responses": self.max_turns,
                        "in_flight_responses": sum(len(target["active"]) for target in self.peers.values()),
                        "pipeline_snapshot": self._input_pipeline_snapshot(),
                    })
                if self.created > self.max_created_responses:
                    self.log("SYSTEM", {
                        "type": "probe.response_creation_budget_overshoot",
                        "created_responses": self.created,
                        "max_created_responses": self.max_created_responses,
                        "created_over_budget": self.created - self.max_created_responses,
                        "in_flight_responses": sum(len(target["active"]) for target in self.peers.values()),
                        "auto_response_disable_requested": self.auto_response_disable_task is not None,
                        "auto_response_disable_confirmed": self.auto_response_disable_confirmed,
                        "pipeline_snapshot": self._input_pipeline_snapshot(),
                    })
            elif kind == "response.action_selected":
                response = self.responses.get(str(event.get("response_id", "")))
                if response is not None:
                    response["wake_action"] = event.get("wake_action")
                    response["response_action"] = event.get("response_action")
                    response["deferred_until_endpoint"] = bool(event.get("deferred_until_endpoint", False))
            elif kind == "response.output_text.delta":
                response = self.responses.get(str(event.get("response_id", "")))
                if response is not None:
                    response["text"] += str(event.get("delta", ""))
            elif kind == "response.audio.delta":
                response_id = str(event.get("response_id", ""))
                response = self.responses.get(response_id)
                if response is not None:
                    packet = decode_response_audio(event)
                    now_ns = time.monotonic_ns()
                    if response["first_audio_ns"] is None:
                        response["first_audio_ns"] = now_ns
                    response["last_audio_ns"] = now_ns
                    response["audio"].append(packet.pcm.copy())
                    output_chunk_id = int(response["audio_packets"])
                    response["audio_packets"] += 1
                    response["audio_samples"] += int(packet.pcm.size)
                    response["audio_chunk_ids"] = response.get("audio_chunk_ids", []) + [output_chunk_id]
                    self.log(speaker, {
                        "type": "probe.response_audio_chunk",
                        "response_id": response_id,
                        "output_chunk_id": output_chunk_id,
                        "packet_id": int(packet.packet_id),
                        "server_packet_id": int(packet.packet_id),
                        "generation_id": int(packet.generation_id),
                        "first_frame_id": int(packet.first_frame_id),
                        "last_frame_id": int(packet.last_frame_id),
                        "start_sample": int(packet.start_sample),
                        "end_sample": int(packet.end_sample),
                        **({"server_emit_started_at_ns": int(event["server_emit_started_at_ns"])}
                           if event.get("server_emit_started_at_ns") is not None else {}),
                        "raw_source_audio_quality": event.get("raw_source_audio_quality"),
                        "received_at_ns": now_ns,
                        "sample_count": int(packet.pcm.size),
                        "sample_rate_hz": int(packet.sample_rate),
                    })
                    self._enqueue_route_item(peer, (
                        "audio", response_id, output_chunk_id,
                        packet.pcm.copy(), now_ns,
                    ))
            elif kind in {"response.done", "response.cancelled"}:
                response_id = str(event.get("response_id", ""))
                response = self.responses.get(response_id)
                cancellation = None
                if kind == "response.cancelled":
                    cancellation = self._fence_cancelled_response(peer, receiver, response_id)
                if response is not None:
                    response["done_ns"] = time.monotonic_ns()
                    response["terminal_event"] = kind
                    response["terminal_status"] = event.get("status")
                    response["runtime_metrics"] = event.get("runtime_metrics")
                    if kind == "response.cancelled":
                        response["relay_cancellation"] = cancellation
                        response["audio_archive_status"] = "partial_generated_audio_on_cancellation"
                    peer["active"].discard(response_id)
                    if (peer.get("manual_seed_bootstrap")
                            and peer.get("bootstrap_phase") in {"commit_inflight", "awaiting_response"}
                            and response_id == peer.get("bootstrap_response_id")):
                        effective_action = str(
                            response.get("response_action") or response.get("action") or ""
                        ).upper()
                        terminal_status = event.get("status")
                        terminal_reason = (
                            (event.get("runtime_metrics") or {}).get("talker", {}).get("terminated_by")
                            if isinstance(event.get("runtime_metrics"), dict) else None
                        )
                        bootstrap_ok = (
                            kind == "response.done"
                            and terminal_status in (None, "completed")
                            and not event.get("reason")
                            and terminal_reason in (None, "eos")
                            and effective_action == "SPEAK"
                            and int(response.get("audio_samples", 0)) > 0
                        )
                        if bootstrap_ok:
                            peer["bootstrap_response_eligible"] = True
                            self._schedule_manual_seed_auto_response(peer)
                        else:
                            self._fail_manual_seed_bootstrap(
                                peer,
                                "initial response was not a completed audible SPEAK",
                                response_id,
                            )
                    self.completed += int(kind == "response.done")
                    if kind == "response.done" and int(response.get("audio_samples", 0)) > 0:
                        self.audible_completed += 1
                        if self.audible_completed >= self.max_turns and not self.created_target_event.is_set():
                            self.target_drain_deadline = (
                                self.hard_deadline_at
                                if self.hard_deadline_at is not None
                                else asyncio.get_running_loop().time() + self.max_duration_s
                            )
                            self.target_drain_ready_at = asyncio.get_running_loop().time()
                            self.created_target_event.set()
                            self._request_auto_response_disable("audible_response_target")
                            self.log("SYSTEM", {
                                "type": "probe.audible_response_target_reached",
                                "audible_completed_responses": self.audible_completed,
                                "target_audible_responses": self.max_turns,
                                "created_responses": self.created,
                                "pipeline_snapshot": self._input_pipeline_snapshot(),
                            })
                    self._last_response_ns = time.monotonic_ns()
                    if kind == "response.done":
                        response["audio_archive_status"] = "completed_response_audio"
                        is_target = self.audible_completed >= self.max_turns
                        self._enqueue_route_item(peer, (
                            "finish", response_id,
                            0 if is_target else self.endpoint_silence_frames,
                            time.monotonic_ns(),
                        ))
            elif kind == "error":
                peer["session_errors"].append(event)
                self.errors.append({"speaker": speaker, **event})
                if (peer.get("manual_seed_bootstrap")
                        and peer.get("bootstrap_phase") in {"awaiting_response", "commit_inflight", "awaiting_ack"}):
                    self._fail_manual_seed_bootstrap(
                        peer, f"server error during bootstrap: {event.get('error', event)}",
                        peer.get("bootstrap_response_id"),
                    )
                if peer["auto_response_disable_requested"]:
                    peer["auto_response_disable_ack"].set()

    async def run(self, seed_wav: Path) -> dict[str, Any]:
        self.started_ns = time.monotonic_ns()
        self.events_path.write_text("", encoding="utf-8")
        self._start_log_writer()
        digest = hashlib.sha256()
        with seed_wav.open("rb") as seed_handle:
            for block in iter(lambda: seed_handle.read(1024 * 1024), b""):
                digest.update(block)
        seed_wav_sha256 = digest.hexdigest()
        if self.tts_quality_diagnostics:
            self.diagnostic_seed_sha256 = seed_wav_sha256
        peer_a = await self._open_peer("A")
        peer_b = await self._open_peer("B")
        for peer in (peer_a, peer_b):
            peer["input_routed_to_peer_frames"] = 0
            peer["route_task"] = asyncio.create_task(
                self._route_loop(peer), name=f"pair-route-{peer['speaker']}"
            )
        for peer in (peer_a, peer_b):
            peer["sender_task"] = asyncio.create_task(self._send_loop(peer), name=f"pair-send-{peer['speaker']}")
        for peer in (peer_a, peer_b):
            peer["receiver_task"] = asyncio.create_task(self._receive_loop(peer), name=f"pair-recv-{peer['speaker']}")

        loop = asyncio.get_running_loop()
        hard_deadline = loop.time() + self.max_duration_s
        self.hard_deadline_at = hard_deadline
        seed_frames = to_frames(read_audio_16k(seed_wav))
        await self._queue_seed_frames(peer_a, seed_frames)
        # The seed is an input utterance for A, so its endpoint marker belongs
        # on A's queue (the response bridge uses the opposite participant).
        seed_marker = (
            (self._END_UTTERANCE, self.endpoint_silence_frames, None, "manual_seed_bootstrap")
            if self.manual_seed_bootstrap
            else (self._END_UTTERANCE, self.endpoint_silence_frames)
        )
        await peer_a["queue"].put(seed_marker)
        if self.manual_seed_bootstrap:
            self.log("A", {
                "type": "probe.manual_seed_bootstrap_seed_queued",
                "queued_at_ns": time.monotonic_ns(),
                "seed_frame_count": len(seed_frames),
                "endpoint_silence_frame_count": self.endpoint_silence_frames,
                "auto_response_initial": False,
                "peer_b_auto_response_initial": True,
            })

        try:
            while loop.time() < hard_deadline:
                drain_requested = self.created_target_event.is_set() or self.created_budget_exhausted
                if drain_requested:
                    queues_empty = all(peer["queue"].empty() for peer in (peer_a, peer_b))
                    routes_drained = all(peer["route_pending"] == 0 for peer in (peer_a, peer_b))
                    no_active = not peer_a["active"] and not peer_b["active"]
                    queued_pcm_sent = all(
                        peer["queued_audio_frames_sent"]
                        + peer.get("queue_frames_cancelled_dropped", 0)
                        + peer["frames_discarded_after_response_limit"]
                        >= peer["queue_frames_enqueued"]
                        for peer in (peer_a, peer_b)
                    )
                    no_pending_acks = all(not peer["ack_pending"] for peer in (peer_a, peer_b))
                    if (
                        self.auto_response_disable_confirmed
                        and queues_empty and routes_drained and no_active and queued_pcm_sent
                    ):
                        if not self.stop_senders_requested:
                            self.stop_senders_requested = True
                            for peer in (peer_a, peer_b):
                                peer["stop_sender"] = True
                                peer["ack_wakeup"].set()
                            self.log("SYSTEM", {
                                "type": "probe.input_stop_requested_after_natural_eos",
                                "requested_at_ns": time.monotonic_ns(),
                                "created_responses": self.created,
                                "pending_ack_frames": sum(len(peer["ack_pending"]) for peer in (peer_a, peer_b)),
                                "pipeline_snapshot": self._input_pipeline_snapshot(),
                            })
                    if self.stop_senders_requested and no_pending_acks:
                        self.pair_drain_completed = True
                        self.budget_drain_completed = self.created_budget_exhausted
                        self.log("SYSTEM", {
                            "type": "probe.response_drain_completed",
                            "created_responses": self.created,
                            "audible_completed_responses": self.audible_completed,
                            "target_audible_responses": self.max_turns,
                            "auto_response_disable_reason": self.auto_response_disable_reason,
                            "completed_at_ns": time.monotonic_ns(),
                            "pipeline_snapshot": self._input_pipeline_snapshot(),
                        })
                        break
                if self.errors or self.overflow:
                    break
                await asyncio.sleep(0.05)
            if (not self.pair_drain_completed and not self.errors and not self.overflow
                    and self.manual_seed_bootstrap
                    and peer_a.get("bootstrap_phase") not in {"complete", "failed"}):
                self._fail_manual_seed_bootstrap(
                    peer_a, "global run deadline expired before initial SPEAK bootstrap completed",
                    peer_a.get("bootstrap_response_id"),
                )
            if not self.pair_drain_completed and not self.errors and not self.overflow:
                self.drain_snapshot_at_timeout = self._input_pipeline_snapshot()
                self.errors.append({
                    "error": "pair reached configured wall deadline before natural drain; result is partial",
                    "audible_completed_responses": self.audible_completed,
                    "target_audible_responses": self.max_turns,
                    "created_responses": self.created,
                    "max_created_responses": self.max_created_responses,
                    "created_budget_exhausted": self.created_budget_exhausted,
                    "pipeline_snapshot": self.drain_snapshot_at_timeout,
                })
        finally:
            for peer in (peer_a, peer_b):
                peer["stop_sender"] = True
                peer["ack_wakeup"].set()
            sender_tasks = [peer["sender_task"] for peer in (peer_a, peer_b)]
            cleanup_budget_s = (
                2.0 if self.pair_drain_completed
                else max(0.0, min(2.0, hard_deadline - loop.time()))
            )
            try:
                if cleanup_budget_s > 0:
                    await asyncio.wait_for(
                        asyncio.gather(*sender_tasks, return_exceptions=True),
                        timeout=cleanup_budget_s,
                    )
                else:
                    pending_senders = [task for task in sender_tasks if not task.done()]
                    if pending_senders:
                        raise asyncio.TimeoutError
                    await asyncio.gather(*sender_tasks, return_exceptions=True)
            except asyncio.TimeoutError:
                pending_snapshot = self._input_pipeline_snapshot()
                self.errors.append({
                    "error": "sender tasks remained blocked at the shared harness cleanup deadline",
                    "pipeline_snapshot": pending_snapshot,
                })
                for task in (peer["sender_task"] for peer in (peer_a, peer_b)):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*(peer["sender_task"] for peer in (peer_a, peer_b)), return_exceptions=True)
            if self.auto_response_disable_task is not None and not self.auto_response_disable_task.done():
                self.auto_response_disable_task.cancel()
                await asyncio.gather(self.auto_response_disable_task, return_exceptions=True)
            bootstrap_tasks = [
                peer.get("bootstrap_auto_response_enable_task")
                for peer in (peer_a, peer_b)
                if peer.get("bootstrap_auto_response_enable_task") is not None
                and not peer["bootstrap_auto_response_enable_task"].done()
            ]
            for task in bootstrap_tasks:
                task.cancel()
            if bootstrap_tasks:
                await asyncio.gather(*bootstrap_tasks, return_exceptions=True)
            active_at_close = {
                speaker: sorted(peer["active"])
                for speaker, peer in self.peers.items()
                if peer["active"]
            }
            if active_at_close:
                self.log("SYSTEM", {
                    "type": "probe.session_close_with_active_responses",
                    "active_response_ids": active_at_close,
                    "close_at_ns": time.monotonic_ns(),
                    "completion_status": "partial",
                })
            self.stop.set()
            for peer in (peer_a, peer_b):
                try:
                    await peer["ws"].send(json.dumps({"type": "session.close"}))
                except Exception:
                    pass
            for peer in (peer_a, peer_b):
                peer["receiver_task"].cancel()
            await asyncio.gather(*(peer["receiver_task"] for peer in (peer_a, peer_b)), return_exceptions=True)
            for peer in (peer_a, peer_b):
                peer["route_task"].cancel()
            await asyncio.gather(*(peer["route_task"] for peer in (peer_a, peer_b)), return_exceptions=True)
            await asyncio.gather(*(peer["ws"].close() for peer in (peer_a, peer_b)), return_exceptions=True)
            await asyncio.to_thread(self._stop_log_writer)

        elapsed_s = (time.monotonic_ns() - self.started_ns) / 1_000_000_000.0
        reports: list[dict[str, Any]] = []
        for peer in (peer_a, peer_b):
            speaker = peer["speaker"]
            input_pcm = np.concatenate(peer["input_samples"]) if peer["input_samples"] else np.empty(0, dtype=np.float32)
            input_path = self.output_dir / f"{speaker}_input.wav"
            if input_pcm.size:
                pcm16 = (np.clip(input_pcm, -1.0, 1.0) * 32767).astype("<i2")
                with wave.open(str(input_path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(INPUT_RATE)
                    wav.writeframes(pcm16.tobytes())
            ack_latencies = peer["ack_latency_ms"]
            send_gaps = peer["input_send_gaps_ms"]
            overlap_records = [
                item for item in peer["input_frame_send_records"]
                if item["local_audio_response_ids"]
            ]
            first_overlap = min(overlap_records, key=lambda item: item["sent_at_ns"]) if overlap_records else None
            first_overlap_ack = (
                peer["ack_latency_by_frame"].get(first_overlap["frame_id"])
                if first_overlap is not None else None
            )
            reports.append({
                "speaker": speaker,
                "session_id": peer["session_id"],
                "session_updated": peer["session_updated"],
                "input_frames_sent": peer["queue_frames_sent"],
                "input_frames_acked": len(peer["ack_ids"]),
                "ack_ids_unique": len(set(peer["ack_ids"])) == len(peer["ack_ids"]),
                "ack_ids_out_of_order": peer["ack_out_of_order"],
                "ack_duplicates": peer["ack_duplicates"],
                "missing_ack_count": max(0, peer["queue_frames_sent"] - len(peer["ack_ids"])),
                "pending_acks_at_close": len(peer["ack_pending"]),
                "ack_capacity_frames": peer["ack_capacity"],
                "ack_pending_high_water": peer["pending_high_water"],
                "ack_latency_ms": {
                    "p50": round(quantile(ack_latencies, .50), 2) if ack_latencies else None,
                    "p95": round(quantile(ack_latencies, .95), 2) if ack_latencies else None,
                    "max": round(max(ack_latencies), 2) if ack_latencies else None,
                },
                "input_frames_during_local_audio_output": len(overlap_records),
                "first_input_during_local_audio_output": ({
                    "frame_id": first_overlap["frame_id"],
                    "sent_at_ns": first_overlap["sent_at_ns"],
                    "local_audio_response_ids": first_overlap["local_audio_response_ids"],
                    "ack_latency_ms": first_overlap_ack.get("latency_ms") if first_overlap_ack else None,
                } if first_overlap is not None else None),
                "input_send_gap_ms": {
                    "p50": round(quantile(send_gaps, .50), 2) if send_gaps else None,
                    "p95": round(quantile(send_gaps, .95), 2) if send_gaps else None,
                    "max": round(max(send_gaps), 2) if send_gaps else None,
                    "over_100ms": sum(value > 100 for value in send_gaps),
                },
                "bridge_queue": {
                    "frames_enqueued": peer["queue_frames_enqueued"],
                    "frames_sent": peer["queue_frames_sent"],
                    "queued_audio_frames_sent": peer.get("queued_audio_frames_sent", 0),
                    "frames_cancelled_dropped": peer.get("queue_frames_cancelled_dropped", 0),
                    "endpoint_silence_frames_sent": peer.get("endpoint_silence_frames_sent", 0),
                    "live_silence_frames_sent": peer.get("live_silence_frames_sent", 0),
                    "high_water": peer["queue_high_water"],
                    "capacity": PAIR_AUDIO_QUEUE_CAPACITY,
                },
                "peer_route_queue": {
                    "capacity_chunks": PAIR_ROUTE_QUEUE_CAPACITY_CHUNKS,
                    "high_water_chunks": peer["route_queue_high_water"],
                    "pending_at_close": peer["route_pending"],
                },
                "speaker_backpressure_wait_ms": {
                    "p50": round(quantile(peer["sender_backpressure_ms"], .50), 2) if peer["sender_backpressure_ms"] else 0,
                    "max": round(max(peer["sender_backpressure_ms"]), 2) if peer["sender_backpressure_ms"] else 0,
                },
                "routed_audio_frames_to_peer": peer["input_routed_to_peer_frames"],
                "frames_discarded_after_response_limit": peer["frames_discarded_after_response_limit"],
                "input_wav": str(input_path) if input_pcm.size else None,
                "session_errors": peer["session_errors"],
            })

        response_reports = []
        for response_id, response in self.responses.items():
            chunks = response["audio"]
            audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
            wav_path = self.output_dir / f"response_{response['speaker']}_{response_id}.wav"
            if audio.size:
                pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
                with wave.open(str(wav_path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(OUTPUT_RATE)
                    wav.writeframes(pcm16.tobytes())
            created_ns = response["created_ns"]
            first_ns = response["first_audio_ns"]
            done_ns = response["done_ns"]
            response_peer = self.peers.get(response["speaker"], {})
            overlap_records = [
                item for item in response_peer.get("input_frame_send_records", [])
                if response_id in item["local_audio_response_ids"]
            ]
            first_overlap = min(overlap_records, key=lambda item: item["sent_at_ns"]) if overlap_records else None
            first_overlap_ack = (
                response_peer.get("ack_latency_by_frame", {}).get(first_overlap["frame_id"])
                if first_overlap is not None else None
            )
            relay_peer = self.peers.get("B" if response["speaker"] == "A" else "A", {})
            response_input_records = [
                item for item in relay_peer.get("input_frame_send_records", [])
                if item.get("source_response_id") == response_id
            ]
            relay_cancellation = self.cancellation_fences.get(response_id)
            if relay_cancellation is not None:
                relay_cancellation = dict(relay_cancellation)
                last_commit_ns = relay_cancellation.get("last_committed_peer_input_frame_at_ns")
                cancelled_at_ns = relay_cancellation["cancelled_at_ns"]
                relay_cancellation["peer_input_frames_sent_total"] = len(response_input_records)
                relay_cancellation["last_commit_relative_to_fence_ms"] = (
                    round((last_commit_ns - cancelled_at_ns) / 1_000_000.0, 2)
                    if last_commit_ns is not None else None
                )
            response_reports.append({
                "speaker": response["speaker"],
                "response_id": response_id,
                "action": response["action"],
                "wake_action": response.get("wake_action"),
                "response_action": response.get("response_action"),
                "deferred_until_endpoint": response.get("deferred_until_endpoint", False),
                "say": response["text"],
                "terminal_event": response.get("terminal_event"),
                "terminal_status": response.get("terminal_status"),
                "audio_archive_status": response.get("audio_archive_status", "incomplete"),
                "relay_cancellation": relay_cancellation,
                "runtime_metrics": response.get("runtime_metrics"),
                "audio_packets": response["audio_packets"],
                "audio_chunk_ids": response.get("audio_chunk_ids", []),
                "audio_samples": int(audio.size),
                "audio_duration_s": round(audio.size / OUTPUT_RATE, 3),
                "audio_rms": round(float(np.sqrt(np.mean(audio * audio))), 6) if audio.size else 0.0,
                "raw_audio_quality": raw_audio_quality(audio),
                "first_pcm_ms_from_created": round((first_ns - created_ns) / 1_000_000, 1) if first_ns else None,
                "first_audio_at_ns": first_ns,
                "input_frames_during_audio_output": len(overlap_records),
                "first_input_during_audio_output": ({
                    "frame_id": first_overlap["frame_id"],
                    "sent_at_ns": first_overlap["sent_at_ns"],
                    "onset_to_send_ms": round((first_overlap["sent_at_ns"] - first_ns) / 1_000_000, 1)
                    if first_ns else None,
                    "ack_latency_ms": first_overlap_ack.get("latency_ms") if first_overlap_ack else None,
                } if first_overlap is not None else None),
                "generation_wall_ms": round((done_ns - created_ns) / 1_000_000, 1) if done_ns else None,
                "wav_path": str(wav_path) if audio.size else None,
            })
            response["audio"] = []

        report: dict[str, Any] = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "realtime_audio_peer_pair",
            "ws_url": self.ws_url,
            "seed_wav": str(seed_wav),
            "seed_wav_sha256": seed_wav_sha256,
            "routing": "A response.audio.delta -> B input at 80ms cadence; B -> A likewise",
            "runtime_asr_enabled": bool(self.input_audio_transcription and all(
                peer.get("session_updated", {}).get("input_audio_transcription_enabled") is True
                for peer in (peer_a, peer_b)
            )),
            "input_audio_transcription_requested": self.input_audio_transcription,
            "cpu_audio_history_archive_requested": self.cpu_audio_history_archive,
            "cpu_audio_history_archive_enabled": all(
                peer.get("session_updated", {}).get("audio_history_archive_cpu_enabled") is True
                for peer in (peer_a, peer_b)
            ),
            "listener_silence_skip_requested": self.listener_silence_skip,
            "listener_silence_skip_enabled": all(
                peer.get("session_updated", {}).get("listener_silence_skip_enabled") is True
                for peer in (peer_a, peer_b)
            ),
            "worker_asr": {
                "live_input_transcription": self.input_audio_transcription,
                "cpu_audio_history_archive": self.cpu_audio_history_archive,
                "archive_completion_source": "response runtime metrics.audio_history_archive",
            },
            "tts_quality_diagnostics": self.tts_quality_diagnostics,
            "diagnostic_seed_sha256": self.diagnostic_seed_sha256,
            "turn_policy": TURN_POLICY,
            "voice_conditioning_mode": self.voice_conditioning_mode,
            "voice_conditioning_ablation_requested": self.voice_conditioning_mode is None,
            "elapsed_s": round(elapsed_s, 2),
            "created_responses": self.created,
            "target_audible_responses": self.max_turns,
            "audible_completed_responses": self.audible_completed,
            "max_created_responses": self.max_created_responses,
            "created_budget_exhausted": self.created_budget_exhausted,
            "budget_drain_completed": self.budget_drain_completed,
            "pair_completion_status": "complete" if self.pair_drain_completed and not self.errors else "partial",
            "pair_drain_completed": self.pair_drain_completed,
            "auto_response_disable": {
                "reason": self.auto_response_disable_reason,
                "requested_at_ns": self.auto_response_disable_started_ns,
                "acknowledged_at_ns": self.auto_response_disable_completed_ns,
                "acknowledged": self.auto_response_disable_confirmed,
                "created_responses_at_ack": self.auto_response_disable_ack_created_responses,
                "active_response_ids_at_ack": self.auto_response_disable_ack_active_response_ids,
                "pipeline_snapshot_at_request": self.auto_response_disable_requested_snapshot,
            },
            "pipeline_snapshot_at_timeout": self.drain_snapshot_at_timeout,
            "responses_created_over_hard_budget": max(0, self.created - self.max_created_responses),
            "responses_created_after_auto_response_disable_ack": self.created_after_auto_response_disable_ack,
            "completed_responses": self.completed,
            "max_duration_s": self.max_duration_s,
            "pair_audio_queue_capacity_frames": PAIR_AUDIO_QUEUE_CAPACITY,
            "pair_route_queue_capacity_chunks": PAIR_ROUTE_QUEUE_CAPACITY_CHUNKS,
            "event_log_writer": "batched_thread_fifo",
            "resampler": "asyncio_to_thread_fifo_route_worker_per_speaker",
            "endpoint_silence_frames": self.endpoint_silence_frames,
            "manual_seed_bootstrap": {
                "enabled": self.manual_seed_bootstrap,
                "peer_a": {
                    "phase": peer_a.get("bootstrap_phase"),
                    "failure_reason": peer_a.get("bootstrap_failure_reason"),
                    "commit_sent": peer_a.get("bootstrap_commit_sent", False),
                    "commit_sent_at_ns": peer_a.get("bootstrap_commit_sent_at_ns"),
                    "frame_ids": peer_a.get("bootstrap_frame_ids", []),
                    "response_id": peer_a.get("bootstrap_response_id"),
                    "auto_response_enable_confirmed": peer_a.get("bootstrap_auto_response_enable_confirmed", False),
                },
                "peer_b": {"phase": peer_b.get("bootstrap_phase"), "commit_sent": False},
            },
            "capacity_overflow": self.overflow,
            "errors": self.errors,
            "participants": reports,
            "responses": response_reports,
            "events_jsonl": str(self.events_path),
        }
        (self.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


def main() -> int:
    global TURN_POLICY
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "pair"), default="single")
    parser.add_argument("--turn-policy", choices=("gemma_native_control6", "duplex_control6"),
                        default=TURN_POLICY)
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8000/v1/realtime?duplex=1")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--followup-audio", type=Path)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--instructions", default="")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-created-responses", type=int)
    parser.add_argument("--max-duration-s", type=float, default=120.0)
    parser.add_argument("--manual-seed-bootstrap", action="store_true",
                        help="in pair mode, manually commit only A's initial seed after all seed and endpoint frames are ACKed")
    parser.add_argument("--tts-quality-diagnostics", action="store_true",
                        help="Opt-in capture of at most two synthetic-fixture SAY/V9/RVQ arrays on the server")
    parser.add_argument("--voice-conditioning-mode", choices=("x_vector_only", "none"), default="x_vector_only",
                        help="session voice mode; none omits profile/ref fields and expects an unconditioned server session")
    parser.add_argument("--input-audio-transcription", action="store_true",
                        help="request live input transcription when the server native ASR runtime is enabled")
    parser.add_argument("--cpu-audio-history-archive", action="store_true",
                        help="require the server's cached offline CPU archive worker for completed real input audio")
    parser.add_argument("--listener-silence-skip", action="store_true",
                        help="require server acknowledgement of the opt-in exact-zero listener skip")
    parser.add_argument("--endpoint-silence-frames", type=int, default=ENDPOINT_SILENCE_FRAMES)
    parser.add_argument("--instructions-a", default="Sei una persona che organizza una cena con amici. Parla in modo naturale e condividi le tue preferenze.")
    parser.add_argument("--instructions-b", default="Sei un assistente conversazionale italiano. Aiuta a scegliere insieme un ristorante e un orario con proposte concise.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    TURN_POLICY = args.turn_policy
    if args.manual_seed_bootstrap and args.mode != "pair":
        parser.error("--manual-seed-bootstrap is available only with --mode pair")
    if args.mode == "single":
        probe = SingleStreamProbe(
            args.ws_url,
            args.output_dir,
            instructions=args.instructions,
            voice_conditioning_mode=(None if args.voice_conditioning_mode == "none" else args.voice_conditioning_mode),
            tts_quality_diagnostics=args.tts_quality_diagnostics,
        )
        report = asyncio.run(probe.run(args.audio, args.followup_audio, args.duration_s))
        failed = bool(report["capacity_overflow"] or report["missing_ack_ids"] or report["errors"])
    else:
        if (args.max_turns < 1 or args.max_duration_s <= 0 or args.endpoint_silence_frames < 0
                or (args.manual_seed_bootstrap and args.max_turns < 2)):
            parser.error("pair mode requires positive turns/duration, non-negative endpoint silence frames, and at least two turns for manual seed bootstrap")
        probe = RealtimePairProbe(
            args.ws_url,
            args.output_dir,
            max_turns=args.max_turns,
            max_created_responses=args.max_created_responses,
            max_duration_s=args.max_duration_s,
            endpoint_silence_frames=args.endpoint_silence_frames,
            manual_seed_bootstrap=args.manual_seed_bootstrap,
            tts_quality_diagnostics=args.tts_quality_diagnostics,
            voice_conditioning_mode=(None if args.voice_conditioning_mode == "none" else args.voice_conditioning_mode),
            input_audio_transcription=args.input_audio_transcription,
            listener_silence_skip=args.listener_silence_skip,
            cpu_audio_history_archive=args.cpu_audio_history_archive,
            instructions_a=args.instructions_a,
            instructions_b=args.instructions_b,
        )
        try:
            report = asyncio.run(probe.run(args.audio))
        finally:
            probe._stop_log_writer()
        failed = bool(report["capacity_overflow"] or report["responses_created_over_hard_budget"] or report["errors"] or
                      report["audible_completed_responses"] < report["target_audible_responses"] or any(
            peer["missing_ack_count"] or peer["pending_acks_at_close"] or peer["ack_duplicates"]
            for peer in report["participants"]
        ))
    if args.mode == "single":
        keys = (
            "mode", "session_id", "input_frames_sent", "input_frames_acked", "missing_ack_ids",
            "capacity_overflow", "pending_high_water_frames", "ack_latency_ms", "input_send_interval_ms",
            "input_frames_during_response", "non_silent_input_frames_during_response", "responses", "errors", "sender_error"
        )
    else:
        keys = (
            "mode", "elapsed_s", "created_responses", "target_audible_responses",
            "audible_completed_responses", "max_created_responses", "created_budget_exhausted",
            "responses_created_over_hard_budget", "completed_responses", "capacity_overflow",
            "manual_seed_bootstrap", "participants", "responses", "errors",
        )
    print(json.dumps({key: report[key] for key in keys}, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
