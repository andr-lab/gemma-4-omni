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
ACK_CREDIT_WAIT_TIMEOUT_S = 10.0
FINAL_ACK_DRAIN_TIMEOUT_S = 10.0
PAIR_AUDIO_QUEUE_CAPACITY = 256
PAIR_ROUTE_QUEUE_CAPACITY_CHUNKS = 8
ENDPOINT_SILENCE_FRAMES = 10
_NATIVE_CONTROL6_SELECTION_FIELDS = (
    "policy_fallback_reason",
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
    "scheduler_dispatch_mode",
    "listener_batch_id",
    "listener_batch_size",
    "listener_batch_start_ns",
    "listener_batch_end_ns",
    "listener_batch_queue_wait_ms",
    "listener_batch_compute_ms",
    "listener_batch_model_lock_wait_ms",
    "listener_batch_model_lock_hold_ms",
    "listener_batch_error",
    "listener_batch_host_lock_timeline",
    "listener_batch_capacity_growth_ms",
    "listener_batch_frontend_encode_ms",
    "listener_batch_mel_prepare_ms",
    "listener_batch_frontend_copy_ms",
    "listener_batch_frontend_copy_cuda_ms",
    "listener_batch_audio_tower_lock_wait_ms",
    "listener_batch_audio_tower_lock_hold_ms",
    "listener_batch_audio_tower_workspace_lock_wait_ms",
    "listener_batch_audio_tower_workspace_lock_hold_ms",
    "listener_batch_audio_tower_row_copy_cuda_ms",
    "listener_batch_audio_tower_graph_submit_ms",
    "listener_batch_audio_tower_cuda_ms",
    "listener_batch_audio_tower_batch_cuda_span_ms",
    "listener_batch_audio_tower_execution_mode",
    "listener_batch_semantic_prefix_request_ms",
    "listener_batch_semantic_prefix_build_ms",
    "listener_batch_semantic_prefix_replay_ms",
    "listener_batch_semantic_prefix_import_ms",
    "listener_batch_semantic_prefix_refresh_event",
    "listener_batch_semantic_prefix_refresh_event_id",
    "listener_batch_semantic_prefix_refresh_revision",
    "listener_batch_semantic_prefix_len",
    "listener_batch_native_submit_ms",
    "listener_batch_native_cuda_ms",
    "listener_batch_native_control6_raw_score_ms",
    "listener_batch_native_output_health_ms",
    "listener_batch_native_output_health_checked",
    "listener_batch_native_output_health_status",
    "listener_batch_native_output_health_nonfinite",
    "listener_batch_native_output_health_generation",
    "listener_batch_native_output_health_session_generation",
    "listener_batch_native_output_health_state_epoch_id",
    "listener_batch_native_output_health_row_frame_id",
    "listener_batch_native_output_health_row_state_version",
    "listener_batch_native_output_health_row_sequence_position",
    "listener_batch_native_output_health_row_cache_position",
    "listener_batch_native_output_health_native_context_id",
    "listener_batch_control6_policy_ms",
    "listener_batch_cuda_completion_wait_ms",
    "scheduler_policy_start_ns",
    "scheduler_policy_end_ns",
    "scheduler_speaker_start_ns",
    "scheduler_speaker_end_ns",
    "scheduler_policy_callback_wall_ms",
    "scheduler_speaker_callback_wall_ms",
    "scheduler_dispatch_start_ns",
    "scheduler_dispatch_end_ns",
    "scheduler_row_callback_entry_ns",
    "scheduler_timeline_request_ns",
    "scheduler_timeline_capture_lock_acquired_ns",
    "scheduler_timeline_model_lock_acquired_ns",
    "scheduler_timeline_snapshot_ns",
    "scheduler_timeline_age_ms",
    "scheduler_timeline_status",
    "scheduler_timeline_reason",
    "scheduler_commit_microbatch_output_start_ns",
    "scheduler_commit_microbatch_output_end_ns",
    "scheduler_commit_state_lock_acquired_ns",
    "scheduler_commit_state_lock_released_ns",
    "scheduler_commit_state_lock_wait_ms",
    "scheduler_commit_state_lock_hold_ms",
    "scheduler_policy_decision_at_ns",
    "scheduler_deadline_to_policy_decision_ms",
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
        "effective_action": event.get("effective_action"),
        "intent_source": event.get("intent_source"),
        "intent_source_frame_id": event.get("intent_source_frame_id"),
        "intent_endpoint_frame_id": event.get("intent_endpoint_frame_id"),
        "policy_decision_frame_id": event.get("policy_decision_frame_id"),
        "policy_decision_at_ns": event.get("policy_decision_at_ns"),
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


def _read_bargein_frames(path: Path) -> list[np.ndarray]:
    return to_frames(read_audio_16k(path))


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="higher"))


def _input_frame_dispatch_deadline(
    *,
    stream_start_loop: float,
    frame_index: int,
    previous_dispatch_loop: float | None,
) -> float:
    """Keep scheduled 80 ms source cadence without compressing a delayed backlog."""
    period_s = FRAME_MS / 1000.0
    scheduled_capture = stream_start_loop + frame_index * period_s
    if previous_dispatch_loop is None:
        return scheduled_capture
    return max(scheduled_capture, previous_dispatch_loop + period_s)


def _followup_after_response_done_due(
    response_done_ns: int | None,
    now_ns: int,
    delay_ms: int,
) -> bool:
    """Return whether an opt-in follow-up may start after a natural response.done."""
    if response_done_ns is None:
        return False
    return now_ns >= response_done_ns + int(delay_ms) * 1_000_000


def _manual_seed_bootstrap_terminal_status(event_kind: str, event: dict[str, Any]) -> str:
    """Normalize this server's omitted status on a successful response.done."""
    if event_kind == "response.cancelled":
        return "cancelled"
    status = event.get("status")
    if status is None:
        reason = event.get("reason")
        error = event.get("error")
        if event_kind == "response.done" and reason in (None, "") and error in (None, ""):
            return "completed"
        return "missing"
    return str(status).strip().lower()


def _manual_seed_bootstrap_response_succeeded(
    event_kind: str,
    event: dict[str, Any],
    response: dict[str, Any],
) -> bool:
    """Check the raw terminal event against the server's response.done contract."""
    if event_kind != "response.done":
        return False
    if _manual_seed_bootstrap_terminal_status(event_kind, event) != "completed":
        return False
    if event.get("reason") not in (None, "") or event.get("error") not in (None, ""):
        return False
    action = str(response.get("response_action") or response.get("action") or "").upper()
    if action != "SPEAK" or int(response.get("audio_samples", 0)) <= 0:
        return False
    runtime_metrics = event.get("runtime_metrics")
    if isinstance(runtime_metrics, dict):
        talker_metrics = runtime_metrics.get("talker")
        if isinstance(talker_metrics, dict):
            terminated_by = talker_metrics.get("terminated_by")
            if terminated_by is not None and terminated_by != "eos":
                return False
    return True


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
        self.ack_capacity_wait_ms: list[float] = []
        self.ack_credit_timeouts = 0
        self.final_ack_drain_timeouts = 0
        self.input_send_lag_ms: list[float] = []
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
        self.start_barrier: asyncio.Event | None = None
        self.prepared_event: asyncio.Event | None = None

    def log(self, event: dict[str, object]) -> None:
        row = {"at_ms": round((time.monotonic_ns() - self.start_ns) / 1_000_000.0, 3), **event}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def _wait_for_ack_capacity(
        self,
        ack_capacity: int,
        capacity_available: asyncio.Event,
        *,
        timeout_s: float = ACK_CREDIT_WAIT_TIMEOUT_S,
        stop_event: asyncio.Event | None = None,
    ) -> float:
        if len(self.ack_sent) < ack_capacity:
            return 0.0
        started_ns = time.monotonic_ns()
        deadline = asyncio.get_running_loop().time() + timeout_s
        pending = len(self.ack_sent)
        self.log({
            "type": "probe.input_capacity_wait",
            "pending": pending,
            "capacity": ack_capacity,
        })
        while len(self.ack_sent) >= ack_capacity:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("ACK credit wait stopped because the WebSocket receiver ended")
            capacity_available.clear()
            # ACK reception and this sender share one event loop. Recheck after
            # clear so a credit released at the boundary cannot be lost.
            if len(self.ack_sent) < ack_capacity:
                break
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("ACK credit wait stopped because the WebSocket receiver ended")
            remaining_s = deadline - asyncio.get_running_loop().time()
            if remaining_s <= 0:
                break
            try:
                await asyncio.wait_for(capacity_available.wait(), timeout=remaining_s)
            except asyncio.TimeoutError:
                break
        if len(self.ack_sent) >= ack_capacity:
            waited_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
            self.ack_capacity_wait_ms.append(waited_ms)
            self.ack_credit_timeouts += 1
            self.log({
                "type": "probe.input_capacity_timeout",
                "pending": len(self.ack_sent),
                "capacity": ack_capacity,
                "wait_ms": round(waited_ms, 3),
            })
            raise TimeoutError(
                f"ACK credit wait timed out after {waited_ms:.1f} ms "
                f"({len(self.ack_sent)}/{ack_capacity} pending)"
            )
        waited_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
        self.ack_capacity_wait_ms.append(waited_ms)
        self.log({
            "type": "probe.input_capacity_released",
            "pending": len(self.ack_sent),
            "capacity": ack_capacity,
            "wait_ms": round(waited_ms, 3),
        })
        return waited_ms

    async def _send_with_ack_tracking(
        self,
        ws,
        payload: str,
        frame_id: int,
        sent_ns: int,
        *,
        pending_acks_drained: asyncio.Event | None = None,
    ) -> None:
        pending = (frame_id, sent_ns)
        self.ack_sent.append(pending)
        if pending_acks_drained is not None:
            pending_acks_drained.clear()
        try:
            await ws.send(payload)
        except BaseException:
            try:
                self.ack_sent.remove(pending)
            except ValueError:
                pass
            if pending_acks_drained is not None and not self.ack_sent:
                pending_acks_drained.set()
            raise

    async def _wait_for_final_ack_drain(
        self,
        pending_acks_drained: asyncio.Event,
        *,
        timeout_s: float = FINAL_ACK_DRAIN_TIMEOUT_S,
        stop_event: asyncio.Event | None = None,
    ) -> bool:
        if not self.ack_sent:
            return True
        if stop_event is not None and stop_event.is_set():
            return False
        pending_acks_drained.clear()
        # ACK reception and this waiter share one event loop. Recheck after
        # clear so an ACK at the boundary cannot leave this wait asleep.
        if not self.ack_sent:
            pending_acks_drained.set()
            return True
        if stop_event is not None and stop_event.is_set():
            return False
        try:
            await asyncio.wait_for(pending_acks_drained.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            self.final_ack_drain_timeouts += 1
            self.log({
                "type": "probe.final_ack_drain_timeout",
                "pending": len(self.ack_sent),
                "timeout_s": timeout_s,
            })
            return False
        return not self.ack_sent

    async def _wait_for_response_completion(
        self,
        first_response_created: asyncio.Event,
        response_done: asyncio.Event,
        *,
        trigger_timeout_s: float = 8.0,
        completion_timeout_s: float = 30.0,
    ) -> bool:
        try:
            await asyncio.wait_for(first_response_created.wait(), timeout=trigger_timeout_s)
        except asyncio.TimeoutError:
            return False
        try:
            await asyncio.wait_for(response_done.wait(), timeout=completion_timeout_s)
        except asyncio.TimeoutError:
            return True
        return False

    async def run(
        self,
        first_wav: Path,
        followup_wav: Path | None,
        duration_s: float,
        *,
        followup_after_response_done_ms: int | None = None,
        start_barrier: asyncio.Event | None = None,
        prepared_event: asyncio.Event | None = None,
    ) -> dict[str, object]:
        self.start_ns = time.monotonic_ns()
        self.start_barrier = start_barrier
        self.prepared_event = prepared_event
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
        first_response_done_ns: int | None = None
        followup_first_frame_index: int | None = None
        followup_first_sent_ns: int | None = None
        followup_last_sent_ns: int | None = None
        followup_frames_sent = 0
        response_completion_timed_out = False
        final_ack_drain_complete = False
        sender_error: str | None = None

        async with websockets.connect(self.ws_url, max_size=32 * 1024 * 1024, ping_interval=20, ping_timeout=20) as ws:
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if first.get("type") != "session.created":
                raise RuntimeError(f"expected session.created, received {first.get('type')}")
            session_id = str(first.get("session_id", ""))
            ack_capacity = int(first.get("input_audio_buffer_capacity_frames", INPUT_CAPACITY))
            ack_capacity_available = asyncio.Event()
            ack_capacity_available.set()
            pending_acks_drained = asyncio.Event()
            pending_acks_drained.set()
            sender_stop = asyncio.Event()
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
                    if self.prepared_event is not None:
                        self.prepared_event.set()
                    break
                if ready.get("type") == "error":
                    raise RuntimeError(f"session update rejected: {ready.get('error')}")

            async def receive() -> None:
                nonlocal response_done, sender_error, first_response_done_ns
                expected_ack = 0
                while not self.sender_done.is_set() or self.ack_sent or self.response_active:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as exc:
                        sender_error = sender_error or (
                            f"WebSocket receiver stopped: {type(exc).__name__}"
                        )
                        sender_stop.set()
                        ack_capacity_available.set()
                        pending_acks_drained.set()
                        break
                    event = json.loads(raw)
                    kind = event.get("type")
                    self.log(_project_server_event_for_log(event))
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
                            if len(self.ack_sent) < ack_capacity:
                                ack_capacity_available.set()
                            if not self.ack_sent:
                                pending_acks_drained.set()
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
                        if (followup_after_response_done_ms is not None
                                and followup_first_frame_index is not None):
                            response_done.clear()
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
                        if kind == "response.done" and first_response_done_ns is None:
                            first_response_done_ns = time.monotonic_ns()
                            self.log({
                                "type": "probe.first_response_done",
                                "response_id": str(event.get("response_id", "")),
                                "response_done_at_ns": first_response_done_ns,
                            })
                        self.response_active.discard(str(event.get("response_id", "")))
                        if not self.response_active and self.first_response_created.is_set():
                            response_done.set()
                    elif kind == "error":
                        self.errors.append(event)
                        sender_error = sender_error or f"server reported an error: {event.get('error')}"
                        sender_stop.set()
                        ack_capacity_available.set()
                        pending_acks_drained.set()
                        break

            async def send_frames() -> None:
                nonlocal sender_error, followup_first_frame_index, followup_first_sent_ns
                nonlocal followup_last_sent_ns, followup_frames_sent
                if self.start_barrier is not None:
                    await self.start_barrier.wait()
                frames_sent = 0
                next_followup = max(0, min(total_frames, len(first_frames) + 2))
                followup_sent = False
                loop = asyncio.get_running_loop()
                capture_origin = loop.time()
                capture_origin_ns = time.monotonic_ns()
                previous_dispatch_loop: float | None = None
                try:
                    while frames_sent < total_frames:
                        if sender_stop.is_set():
                            sender_error = sender_error or "WebSocket receiver stopped before source playback completed"
                            break
                        if frames_sent < len(first_frames):
                            frame = first_frames[frames_sent]
                            source_kind = "initial_wav"
                            source_frame_index = frames_sent
                        elif (followup_frames and not followup_sent
                              and followup_after_response_done_ms is not None):
                            now_for_followup_ns = time.monotonic_ns()
                            if _followup_after_response_done_due(
                                first_response_done_ns,
                                now_for_followup_ns,
                                followup_after_response_done_ms,
                            ):
                                if followup_first_frame_index is None:
                                    followup_first_frame_index = frames_sent
                                    response_done.clear()
                                    self.log({
                                        "type": "probe.followup_delay_elapsed",
                                        "trigger_response_done_at_ns": first_response_done_ns,
                                        "configured_delay_ms": followup_after_response_done_ms,
                                        "due_at_ns": first_response_done_ns + followup_after_response_done_ms * 1_000_000,
                                        "first_followup_frame_index": frames_sent,
                                    })
                                source_frame_index = frames_sent - followup_first_frame_index
                                if source_frame_index < len(followup_frames):
                                    frame = followup_frames[source_frame_index]
                                    source_kind = "followup_wav"
                                    if source_frame_index + 1 >= len(followup_frames):
                                        followup_sent = True
                                else:
                                    frame = silence
                                    source_kind = "silence"
                            else:
                                frame = silence
                                source_kind = "silence"
                                source_frame_index = None
                        elif (followup_frames and not followup_sent and frames_sent >= next_followup
                              and self.first_response_created.is_set()):
                            source_frame_index = frames_sent - next_followup
                            frame = followup_frames[source_frame_index] if source_frame_index < len(followup_frames) else silence
                            source_kind = "followup_wav" if source_frame_index < len(followup_frames) else "silence"
                            if source_frame_index + 1 >= len(followup_frames):
                                followup_sent = True
                        else:
                            frame = silence
                            source_kind = "silence"
                            source_frame_index = None
                        try:
                            await self._wait_for_ack_capacity(
                                ack_capacity,
                                ack_capacity_available,
                                stop_event=sender_stop,
                            )
                        except (TimeoutError, RuntimeError) as exc:
                            sender_error = sender_error or str(exc)
                            break
                        if sender_stop.is_set():
                            sender_error = sender_error or "WebSocket receiver stopped before source playback completed"
                            break

                        dispatch_deadline = _input_frame_dispatch_deadline(
                            stream_start_loop=capture_origin,
                            frame_index=frames_sent,
                            previous_dispatch_loop=previous_dispatch_loop,
                        )
                        wait_s = dispatch_deadline - loop.time()
                        if wait_s > 0:
                            await asyncio.sleep(wait_s)
                        if sender_stop.is_set():
                            sender_error = sender_error or "WebSocket receiver stopped before source playback completed"
                            break

                        scheduled_capture_at_ns = capture_origin_ns + frames_sent * FRAME_MS * 1_000_000
                        actual_send_at = loop.time()
                        now_ns = time.monotonic_ns()
                        capture_lag_ms = max(0.0, (now_ns - scheduled_capture_at_ns) / 1_000_000.0)
                        self.input_send_lag_ms.append(capture_lag_ms)
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
                        pending_payload = json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(float_to_pcm16_bytes(frame)).decode("ascii"),
                        })
                        await self._send_with_ack_tracking(
                            ws,
                            pending_payload,
                            frames_sent,
                            now_ns,
                            pending_acks_drained=pending_acks_drained,
                        )
                        self.sent_frames += 1
                        self.pending_high_water = max(self.pending_high_water, len(self.ack_sent))
                        self.input_send_times.append(actual_send_at)
                        self.log({
                            "type": "probe.input_frame_sent",
                            "frame_index": frames_sent,
                            "source_kind": source_kind,
                            "source_frame_index": source_frame_index,
                            "pending": len(self.ack_sent),
                            "rms": round(float(np.sqrt(np.mean(frame * frame))), 5),
                            "scheduled_capture_at_ns": scheduled_capture_at_ns,
                            "dispatch_started_at_ns": now_ns,
                            "capture_to_dispatch_lag_ms": round(capture_lag_ms, 3),
                        })
                        if source_kind == "followup_wav":
                            followup_frames_sent += 1
                            if followup_first_sent_ns is None:
                                followup_first_sent_ns = now_ns
                                self.log({
                                    "type": "probe.followup_audio_started",
                                    "trigger_response_done_at_ns": first_response_done_ns,
                                    "configured_delay_ms": followup_after_response_done_ms,
                                    "first_sent_at_ns": now_ns,
                                    "actual_delay_ms": round((now_ns - first_response_done_ns) / 1_000_000.0, 3)
                                    if first_response_done_ns is not None else None,
                                    "first_input_frame_index": frames_sent,
                                    "source_frame_count": len(followup_frames),
                                })
                            if followup_frames_sent == len(followup_frames):
                                followup_last_sent_ns = now_ns
                                self.log({
                                    "type": "probe.followup_audio_completed",
                                    "last_sent_at_ns": now_ns,
                                    "frames_sent": followup_frames_sent,
                                })
                        previous_dispatch_loop = actual_send_at
                        frames_sent += 1
                finally:
                    self.sender_done.set()

            recv_task = asyncio.create_task(receive(), name="probe-receive")
            send_task = asyncio.create_task(send_frames(), name="probe-send")
            try:
                await asyncio.wait_for(send_task, timeout=duration_s + 20)
                response_completion_timed_out = await self._wait_for_response_completion(
                    self.first_response_created,
                    response_done,
                )
            except Exception as exc:
                sender_error = sender_error or f"probe wait failed: {type(exc).__name__}: {exc}"
            finally:
                self.sender_done.set()
                try:
                    final_ack_drain_complete = await self._wait_for_final_ack_drain(
                        pending_acks_drained,
                        stop_event=sender_stop,
                    )
                except Exception as exc:
                    sender_error = sender_error or (
                        f"final ACK drain failed: {type(exc).__name__}: {exc}"
                    )
                if not final_ack_drain_complete and not sender_error:
                    reason = (
                        "timed out"
                        if self.final_ack_drain_timeouts
                        else "stopped before all ACKs arrived"
                    )
                    sender_error = (
                        f"final ACK drain {reason} with {len(self.ack_sent)} pending frame(s)"
                    )
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
            "followup_timing": {
                "mode": "after_first_response_done" if followup_after_response_done_ms is not None else (
                    "legacy_after_response_created" if followup_wav else "none"
                ),
                "configured_delay_ms": followup_after_response_done_ms,
                "first_response_done_ns": first_response_done_ns,
                "first_followup_input_frame_index": followup_first_frame_index,
                "first_followup_sent_at_ns": followup_first_sent_ns,
                "actual_delay_ms": round((followup_first_sent_ns - first_response_done_ns) / 1_000_000.0, 3)
                if first_response_done_ns is not None and followup_first_sent_ns is not None else None,
                "last_followup_sent_at_ns": followup_last_sent_ns,
                "followup_frames_sent": followup_frames_sent,
            },
            "requested_input_duration_s": duration_s,
            "source_timing_basis": "prerecorded_wav_scheduled_at_80ms_intervals",
            "requested_input_frames": total_frames,
            "tts_quality_diagnostics": self.tts_quality_diagnostics,
            "diagnostic_seed_sha256": self.diagnostic_seed_sha256,
            "ack_capacity_frames": ack_capacity,
            "input_frames_sent": self.sent_frames,
            "input_frames_not_dispatched": max(0, total_frames - self.sent_frames),
            "source_playback_complete": self.sent_frames == total_frames,
            "input_frames_acked": len(self.ack_ids),
            "ack_ids_unique": len(set(self.ack_ids)) == len(self.ack_ids),
            "missing_ack_ids": missing_ack_ids,
            "duplicate_acks": self.ack_duplicates,
            "out_of_order_acks": self.ack_out_of_order,
            "capacity_overflow": self.overflow,
            "pending_high_water_frames": self.pending_high_water,
            "pending_after_close": len(self.ack_sent),
            "final_ack_drain_complete": final_ack_drain_complete,
            "final_ack_drain_timed_out": self.final_ack_drain_timeouts > 0,
            "ack_credit_timeout_count": self.ack_credit_timeouts,
            "ack_latency_ms": {
                "p50": round(quantile(self.ack_latency_ms, 0.50), 2) if self.ack_latency_ms else None,
                "p95": round(quantile(self.ack_latency_ms, 0.95), 2) if self.ack_latency_ms else None,
                "max": round(max(self.ack_latency_ms), 2) if self.ack_latency_ms else None,
            },
            "ack_capacity_backpressure_ms": {
                "wait_count": len(self.ack_capacity_wait_ms),
                "p50": round(quantile(self.ack_capacity_wait_ms, 0.50), 2) if self.ack_capacity_wait_ms else 0.0,
                "p95": round(quantile(self.ack_capacity_wait_ms, 0.95), 2) if self.ack_capacity_wait_ms else 0.0,
                "max": round(max(self.ack_capacity_wait_ms), 2) if self.ack_capacity_wait_ms else 0.0,
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
            "input_send_lag_ms": {
                "p50": round(quantile(self.input_send_lag_ms, 0.50), 2) if self.input_send_lag_ms else None,
                "p95": round(quantile(self.input_send_lag_ms, 0.95), 2) if self.input_send_lag_ms else None,
                "max": round(max(self.input_send_lag_ms), 2) if self.input_send_lag_ms else None,
                "frames_over_80ms": sum(value > 80 for value in self.input_send_lag_ms),
            },
            "input_frames_during_response": self.input_frames_during_audio,
            "non_silent_input_frames_during_response": self.input_voice_frames_during_audio,
            "input_frames_during_audio_generation": self.input_frames_during_audio_generation,
            "non_silent_input_frames_during_audio_generation": self.non_silent_frames_during_audio_generation,
            "responses": list(self.responses.values()),
            "response_completion_timed_out": response_completion_timed_out,
            "errors": self.errors,
            "sender_error": sender_error,
        }
        (self.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


_LISTENER_TICK_METRICS = (
    "scheduler_observed_to_listener_start_ms",
    "scheduler_listener_callback_ms",
    "scheduler_speaker_callback_wall_ms",
    "tick_deadline_to_listener_end_ms",
    "tick_deadline_to_policy_decision_ms",
)

_LISTENER_BATCH_METRICS = (
    "listener_batch_queue_wait_ms",
    "listener_batch_capacity_growth_ms",
    "listener_batch_frontend_encode_ms",
    "listener_batch_mel_prepare_ms",
    "listener_batch_frontend_copy_ms",
    "listener_batch_frontend_copy_cuda_ms",
    "listener_batch_audio_tower_lock_wait_ms",
    "listener_batch_audio_tower_lock_hold_ms",
    "listener_batch_audio_tower_workspace_lock_wait_ms",
    "listener_batch_audio_tower_workspace_lock_hold_ms",
    "listener_batch_audio_tower_row_copy_cuda_ms",
    "listener_batch_audio_tower_graph_submit_ms",
    "listener_batch_audio_tower_cuda_ms",
    "listener_batch_audio_tower_batch_cuda_span_ms",
    "listener_batch_semantic_prefix_request_ms",
    "listener_batch_semantic_prefix_build_ms",
    "listener_batch_semantic_prefix_replay_ms",
    "listener_batch_semantic_prefix_import_ms",
    "listener_batch_native_submit_ms",
    "listener_batch_native_cuda_ms",
    "listener_batch_native_control6_raw_score_ms",
    "listener_batch_native_output_health_ms",
    "listener_batch_cuda_completion_wait_ms",
    "listener_batch_compute_ms",
    "listener_batch_model_lock_wait_ms",
    "listener_batch_model_lock_hold_ms",
    "listener_batch_control6_policy_ms",
)


def summarize_listener_tick_events(
    event_files: list[Path], *, warmup_ticks_per_session: int = 5,
) -> dict[str, Any]:
    """Summarize actual server tick stamps, separating each session's warmup."""
    if warmup_ticks_per_session < 0:
        raise ValueError("warmup_ticks_per_session must be non-negative")
    groups: dict[str, list[dict[str, Any]]] = {"warmup": [], "steady_state": []}
    row_batch_sizes: dict[str, int] = {}
    row_dispatch_modes: dict[str, int] = {}
    errors: list[dict[str, Any]] = []
    batch_acks: dict[str, list[dict[str, Any]]] = {}
    batch_decisions: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    response_origins: dict[tuple[str, str], dict[str, Any]] = {}
    response_audio_counts: dict[tuple[str, str], int] = {}
    rows_without_batch_id = 0
    total_ticks = 0
    for path in event_files:
        events = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        decisions: dict[int, list[int]] = {}
        for event in events:
            if event.get("type") in {"duplex_control6.decision", "turn_policy.decision"}:
                frame_id = event.get("frame_id")
                decided = event.get("policy_decision_at_ns")
                if isinstance(frame_id, int) and isinstance(decided, int):
                    decisions.setdefault(frame_id, []).append(decided)
                batch_id = event.get("listener_batch_id")
                if isinstance(batch_id, str) and batch_id:
                    row_key = (str(path), int(frame_id)) if isinstance(frame_id, int) else (str(path), -1)
                    candidate = {
                        "event_file": str(path),
                        "frame_id": frame_id,
                        "decision_at_ns": decided,
                        "event_type": event.get("type"),
                        "turn_policy_id": event.get("turn_policy_id"),
                        "turn_policy_class": event.get("turn_policy_class"),
                        "source": event.get("source"),
                        "native_control6_step_called": event.get("native_control6_step_called"),
                        "native_control6_head_scored": event.get("native_control6_head_scored"),
                        "native_control6_class_match": event.get("native_control6_class_match"),
                        "native_control6_raw_row_index": event.get("native_control6_raw_row_index"),
                        "native_control6_raw_token_id": event.get("native_control6_raw_token_id"),
                        "native_control6_raw_token": event.get("native_control6_raw_token"),
                        "native_control6_selected_row_index": event.get("native_control6_selected_row_index"),
                        "native_control6_selected_token_id": event.get("native_control6_selected_token_id"),
                        "native_control6_selected_token": event.get("native_control6_selected_token"),
                        "native_control6_token_selection_scope": event.get("native_control6_token_selection_scope"),
                        "legal_actions": event.get("legal_actions"),
                        "legal_action_probabilities": event.get("legal_action_probabilities"),
                        "native_action_order": event.get("native_action_order"),
                        "native_action_base_logits": event.get("native_action_base_logits"),
                        "native_action": event.get("native_action"),
                        "native_action_probability": event.get("native_action_probability"),
                        "native_action_probabilities": event.get("native_action_probabilities"),
                        "native_action_logits": event.get("native_action_logits"),
                        "selected_action": event.get("selected_action"),
                        "selected_action_probability": event.get("selected_action_probability"),
                        "action": event.get("action"),
                        "accepted": event.get("accepted"),
                        "gate_applied_action": event.get("gate_applied_action"),
                        "checkpoint_path": event.get("checkpoint_path"),
                        "hidden_contract": event.get("hidden_contract"),
                        "native_control_hidden_frame_id": event.get("native_control_hidden_frame_id"),
                        "listener_frame_id": event.get("listener_frame_id"),
                    }
                    current = batch_decisions.setdefault(batch_id, {}).get(row_key)
                    current_time = current.get("decision_at_ns") if current else None
                    if current is None or (isinstance(decided, int) and (not isinstance(current_time, int) or decided > current_time)):
                        batch_decisions[batch_id][row_key] = candidate
            elif event.get("type") == "response.created":
                response_id = event.get("response_id")
                if isinstance(response_id, str) and response_id:
                    response_origins[(str(path), response_id)] = {
                        "event_file": str(path),
                        "response_id": response_id,
                        "action": event.get("effective_action", event.get("action")),
                        "intent_source": event.get("intent_source"),
                        "policy_decision_frame_id": event.get("policy_decision_frame_id"),
                        "policy_decision_at_ns": event.get("policy_decision_at_ns"),
                    }
            elif event.get("type") == "response.audio.delta":
                response_id = event.get("response_id")
                if isinstance(response_id, str) and response_id:
                    response_key = (str(path), response_id)
                    response_audio_counts[response_key] = response_audio_counts.get(response_key, 0) + 1
        ticks = [event for event in events if event.get("type") == "input_audio_buffer.consumed"]
        total_ticks += len(ticks)
        for index, event in enumerate(ticks):
            row = dict(event)
            deadline = row.get("scheduler_deadline_ns")
            listener_end = row.get("scheduler_listener_end_ns")
            frame_id = row.get("frame_id")
            if isinstance(deadline, int) and isinstance(listener_end, int):
                row["tick_deadline_to_listener_end_ms"] = (listener_end - deadline) / 1_000_000.0
            if isinstance(deadline, int) and isinstance(frame_id, int):
                decisions_for_tick = decisions.get(frame_id, ())
                if decisions_for_tick:
                    row["tick_deadline_to_policy_decision_ms"] = (
                        min(decisions_for_tick) - deadline
                    ) / 1_000_000.0
            group = "warmup" if index < warmup_ticks_per_session else "steady_state"
            groups[group].append(row)
            batch_size = row.get("listener_batch_size")
            if isinstance(batch_size, int):
                key = str(batch_size)
                row_batch_sizes[key] = row_batch_sizes.get(key, 0) + 1
            dispatch_mode = row.get("scheduler_dispatch_mode")
            if isinstance(dispatch_mode, str):
                row_dispatch_modes[dispatch_mode] = row_dispatch_modes.get(dispatch_mode, 0) + 1
            batch_id = row.get("listener_batch_id")
            if isinstance(batch_id, str) and batch_id:
                row_key = (str(path), frame_id) if isinstance(frame_id, int) else (str(path), -1)
                batch_row = dict(row)
                batch_row["_warmup"] = index < warmup_ticks_per_session
                current_rows = batch_acks.setdefault(batch_id, [])
                if not any(
                    item.get("_row_key") == row_key for item in current_rows
                ):
                    batch_row["_row_key"] = row_key
                    current_rows.append(batch_row)
            else:
                rows_without_batch_id += 1
            if row.get("listener_batch_error"):
                errors.append({"event_file": str(path), "frame_id": frame_id,
                               "error": str(row["listener_batch_error"])})

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for name in _LISTENER_TICK_METRICS:
            values = [
                float(row[name]) for row in rows
                if isinstance(row.get(name), (int, float))
                and math.isfinite(float(row[name]))
            ]
            if values:
                metrics[name] = {
                    "samples": len(values),
                    "p50_ms": quantile(values, 0.50),
                    "p95_ms": quantile(values, 0.95),
                    "p99_ms": quantile(values, 0.99),
                    "max_ms": max(values),
                    "over_80ms": sum(value > FRAME_MS for value in values),
                }
        return {"ticks": len(rows), "metrics": metrics}

    batches: list[dict[str, Any]] = []
    for batch_id, ack_rows in batch_acks.items():
        if not ack_rows:
            continue
        expected_sizes = {
            int(row["listener_batch_size"])
            for row in ack_rows
            if isinstance(row.get("listener_batch_size"), int)
        }
        expected_size = max(expected_sizes) if expected_sizes else None
        deadlines = {
            int(row["scheduler_deadline_ns"])
            for row in ack_rows
            if isinstance(row.get("scheduler_deadline_ns"), int)
        }
        deadline_ns = min(deadlines) if deadlines else None
        decisions_for_batch = sorted(
            batch_decisions.get(batch_id, {}).values(),
            key=lambda row: (
                row.get("decision_at_ns") if isinstance(row.get("decision_at_ns"), int) else -1,
                str(row.get("event_file", "")),
                row.get("frame_id") if isinstance(row.get("frame_id"), int) else -1,
            ),
        )
        decision_stamps = [
            int(row["decision_at_ns"]) for row in decisions_for_batch
            if isinstance(row.get("decision_at_ns"), int)
        ]
        decision_row_keys = set(batch_decisions.get(batch_id, {}))
        ack_row_keys = {row["_row_key"] for row in ack_rows}
        missing_decision_rows = ack_row_keys - decision_row_keys
        extra_decision_rows = decision_row_keys - ack_row_keys
        decisions_complete = (
            expected_size is not None
            and len(ack_rows) == expected_size
            and not missing_decision_rows
            and not extra_decision_rows
            and len(decision_stamps) == expected_size
        )
        last_decision_ns = max(decision_stamps) if decision_stamps else None
        deadline_to_last = (
            round((last_decision_ns - deadline_ns) / 1_000_000.0, 3)
            if decisions_complete
            and len(deadlines) == 1
            and isinstance(deadline_ns, int)
            and isinstance(last_decision_ns, int)
            else None
        )
        phase_metrics: dict[str, float] = {}
        inconsistent_phases: list[str] = []
        for metric_name in _LISTENER_BATCH_METRICS:
            phase_values = [row.get(metric_name) for row in ack_rows]
            if all(value is None for value in phase_values):
                continue
            if metric_name in {
                "listener_batch_semantic_prefix_build_ms",
                "listener_batch_semantic_prefix_replay_ms",
                "listener_batch_semantic_prefix_import_ms",
            }:
                executed_values = [value for value in phase_values if value is not None]
                if any(not isinstance(value, (int, float)) for value in executed_values):
                    inconsistent_phases.append(metric_name)
                    continue
                numeric_values = [float(value) for value in executed_values]
                if not all(math.isfinite(value) for value in numeric_values):
                    inconsistent_phases.append(metric_name)
                    continue
                # Prefix phases are per-row work. Sum only rows which executed
                # the phase; a missing value means no work on that row.
                phase_metrics[metric_name] = sum(numeric_values)
                continue
            if any(not isinstance(value, (int, float)) for value in phase_values):
                inconsistent_phases.append(metric_name)
                continue
            numeric_values = [float(value) for value in phase_values]
            if not all(math.isfinite(value) for value in numeric_values):
                inconsistent_phases.append(metric_name)
                continue
            if any(value != numeric_values[0] for value in numeric_values[1:]):
                inconsistent_phases.append(metric_name)
                continue
            phase_metrics[metric_name] = numeric_values[0]
        health_rows = [
            row for row in ack_rows
            if any(
                key in row
                for key in (
                    "listener_batch_native_output_health_checked",
                    "listener_batch_native_output_health_status",
                    "listener_batch_native_output_health_generation",
                    "listener_batch_native_output_health_session_generation",
                )
            )
        ]
        native_output_health = None
        if health_rows:
            health_checked = [
                row.get("listener_batch_native_output_health_checked") is True
                for row in health_rows
            ]
            health_nonfinite = [
                row.get("listener_batch_native_output_health_nonfinite") is True
                for row in health_rows
            ]
            health_statuses = sorted({
                str(row.get("listener_batch_native_output_health_status"))
                for row in health_rows
                if isinstance(row.get("listener_batch_native_output_health_status"), str)
            })
            raw_health_generations = [
                row.get("listener_batch_native_output_health_generation")
                for row in health_rows
            ]
            health_generations = sorted({
                int(value)
                for value in raw_health_generations
                if isinstance(value, int) and not isinstance(value, bool) and value > 0
            })
            health_generations_complete = all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in raw_health_generations
            )
            def has_valid_health_provenance(row: dict[str, Any]) -> bool:
                return (
                    isinstance(row.get("listener_batch_native_output_health_row_frame_id"), int)
                    and not isinstance(row.get("listener_batch_native_output_health_row_frame_id"), bool)
                    and row.get("listener_batch_native_output_health_row_frame_id") == row.get("frame_id")
                    and isinstance(row.get("listener_batch_native_output_health_row_state_version"), int)
                    and not isinstance(row.get("listener_batch_native_output_health_row_state_version"), bool)
                    and row.get("listener_batch_native_output_health_row_state_version") >= 0
                    and isinstance(row.get("listener_batch_native_output_health_session_generation"), int)
                    and not isinstance(row.get("listener_batch_native_output_health_session_generation"), bool)
                    and row.get("listener_batch_native_output_health_session_generation") >= 0
                    and isinstance(row.get("listener_batch_native_output_health_state_epoch_id"), str)
                    and bool(row.get("listener_batch_native_output_health_state_epoch_id"))
                    and isinstance(row.get("listener_batch_native_output_health_row_sequence_position"), int)
                    and not isinstance(row.get("listener_batch_native_output_health_row_sequence_position"), bool)
                    and row.get("listener_batch_native_output_health_row_sequence_position") >= 0
                    and isinstance(row.get("listener_batch_native_output_health_row_cache_position"), int)
                    and not isinstance(row.get("listener_batch_native_output_health_row_cache_position"), bool)
                    and row.get("listener_batch_native_output_health_row_cache_position") >= 0
                    and isinstance(row.get("listener_batch_native_output_health_native_context_id"), int)
                    and not isinstance(row.get("listener_batch_native_output_health_native_context_id"), bool)
                    and row.get("listener_batch_native_output_health_native_context_id") > 0
                )

            health_provenance_valid = [
                has_valid_health_provenance(row) for row in health_rows
            ]
            health_rows_match_frames = all(
                isinstance(row.get("listener_batch_native_output_health_row_frame_id"), int)
                and not isinstance(row.get("listener_batch_native_output_health_row_frame_id"), bool)
                and row.get("listener_batch_native_output_health_row_frame_id") == row.get("frame_id")
                for row in health_rows
            )
            health_statuses_complete = all(
                isinstance(row.get("listener_batch_native_output_health_status"), str)
                and row.get("listener_batch_native_output_health_status")
                in ("healthy", "nonfinite_guard")
                for row in health_rows
            )
            health_values_match_status = all(
                isinstance(row.get("listener_batch_native_output_health_nonfinite"), bool)
                and row.get("listener_batch_native_output_health_nonfinite")
                == (row.get("listener_batch_native_output_health_status") == "nonfinite_guard")
                for row in health_rows
            )
            native_output_health = {
                "checked_rows": sum(health_checked),
                "unchecked_rows": len(health_rows) - sum(health_checked),
                "nonfinite_rows": sum(health_nonfinite),
                "valid_provenance_rows": sum(health_provenance_valid),
                "invalid_provenance_rows": len(health_provenance_valid) - sum(health_provenance_valid),
                "statuses": health_statuses,
                "generations": health_generations,
                "consistent": (
                    all(health_checked)
                    and health_generations_complete
                    and len(health_generations) == 1
                    and len(raw_health_generations) == len(health_rows)
                    and bool(health_statuses)
                    and health_statuses_complete
                    and health_rows_match_frames
                    and all(health_provenance_valid)
                    and health_values_match_status
                ),
            }
        host_lock_timeline_snapshots = [
            row["listener_batch_host_lock_timeline"]
            for row in ack_rows
            if isinstance(row.get("listener_batch_host_lock_timeline"), dict)
        ]
        semantic_prefix_refresh_events = [
            {
                "event_file": str(row.get("_row_key", ("",))[0]),
                "frame_id": row.get("frame_id"),
                "event_id": row.get("listener_batch_semantic_prefix_refresh_event_id"),
                "revision": row.get("listener_batch_semantic_prefix_refresh_revision"),
                "prefix_len": row.get("listener_batch_semantic_prefix_len"),
            }
            for row in ack_rows
            if row.get("listener_batch_semantic_prefix_refresh_event") is True
        ]
        lateness_values = [
            float(row["scheduler_deadline_lateness_ms"]) for row in ack_rows
            if isinstance(row.get("scheduler_deadline_lateness_ms"), (int, float))
            and math.isfinite(float(row["scheduler_deadline_lateness_ms"]))
        ]
        warmup_flags = [bool(row.get("_warmup")) for row in ack_rows]
        warmup_class = (
            "warmup" if all(warmup_flags) else "steady_state" if not any(warmup_flags) else "mixed"
        )
        observed_ack_rows = len(ack_rows)
        size_consistent = len(expected_sizes) <= 1
        ack_rows_complete = (
            expected_size is not None
            and size_consistent
            and observed_ack_rows == expected_size
        )
        batch_error_values = sorted({
            str(row["listener_batch_error"]) for row in ack_rows
            if row.get("listener_batch_error")
        })
        complete = (
            ack_rows_complete
            and decisions_complete
            and len(deadlines) == 1
            and not batch_error_values
        )
        batches.append({
            "listener_batch_id": batch_id,
            "warmup_class": warmup_class,
            "expected_batch_size": expected_size,
            "expected_batch_size_consistent": size_consistent,
            "observed_ack_rows": observed_ack_rows,
            "missing_ack_rows": max(0, expected_size - observed_ack_rows) if expected_size is not None else None,
            "ack_rows_complete": ack_rows_complete,
            "complete": complete,
            "deadline_ns": deadline_ns,
            "deadline_stamps_consistent": len(deadlines) == 1,
            "decision_rows": len(decision_stamps),
            "decision_coverage_complete": decisions_complete,
            "missing_decision_rows": len(missing_decision_rows),
            "extra_decision_rows": len(extra_decision_rows),
            "last_policy_decision_at_ns": last_decision_ns,
            "tick_deadline_to_last_policy_decision_ms": deadline_to_last,
            "scheduler_deadline_lateness_ms": max(lateness_values) if lateness_values else None,
            "late_rows_at_dispatch": sum(value > 0 for value in lateness_values),
            "listener_batch_phase_metrics": phase_metrics,
            "semantic_prefix_refresh_events": semantic_prefix_refresh_events,
            "native_output_health": native_output_health,
            "host_lock_timeline_snapshot_count": len(host_lock_timeline_snapshots),
            "host_lock_timeline_snapshot": (
                host_lock_timeline_snapshots[0] if host_lock_timeline_snapshots else None
            ),
            "inconsistent_shared_phase_fields": sorted(inconsistent_phases),
            "dispatch_modes": sorted({
                str(row["scheduler_dispatch_mode"]) for row in ack_rows
                if isinstance(row.get("scheduler_dispatch_mode"), str)
            }),
            "errors": batch_error_values,
            "policy_decisions": decisions_for_batch,
        })
    batches.sort(key=lambda row: (row.get("deadline_ns") or -1, row["listener_batch_id"]))

    def summarize_batches(rows: list[dict[str, Any]]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for name in (*_LISTENER_BATCH_METRICS, "scheduler_deadline_lateness_ms", "tick_deadline_to_last_policy_decision_ms"):
            values: list[float] = []
            for row in rows:
                value = (
                    row.get("listener_batch_phase_metrics", {}).get(name)
                    if name in _LISTENER_BATCH_METRICS
                    else row.get(name)
                )
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
            if values:
                metrics[name] = {
                    "samples": len(values),
                    "p50_ms": quantile(values, 0.50),
                    "p95_ms": quantile(values, 0.95),
                    "p99_ms": quantile(values, 0.99),
                    "max_ms": max(values),
                    "over_80ms": sum(value > FRAME_MS for value in values),
                }
        return {
            "batches": len(rows),
            "incomplete_batches": sum(not row["complete"] for row in rows),
            "incomplete_decision_coverage_batches": sum(not row["decision_coverage_complete"] for row in rows),
            "late_batches": sum(
                isinstance(row.get("scheduler_deadline_lateness_ms"), (int, float))
                and row["scheduler_deadline_lateness_ms"] > 0 for row in rows
            ),
            "metrics": metrics,
        }

    batch_groups = {
        name: summarize_batches([row for row in batches if row["warmup_class"] == name])
        for name in ("warmup", "steady_state", "mixed")
    }
    batch_size_histogram: dict[str, int] = {}
    dispatch_modes: dict[str, int] = {}
    for row in batches:
        size = row.get("expected_batch_size")
        if isinstance(size, int):
            batch_size_histogram[str(size)] = batch_size_histogram.get(str(size), 0) + 1
        for mode in row["dispatch_modes"]:
            dispatch_modes[mode] = dispatch_modes.get(mode, 0) + 1

    response_audio_provenance: list[dict[str, Any]] = []
    for response_key, audio_delta_count in response_audio_counts.items():
        origin = response_origins.get(response_key, {
            "event_file": response_key[0], "response_id": response_key[1]
        })
        has_policy_decision = (
            isinstance(origin.get("policy_decision_at_ns"), int)
            or isinstance(origin.get("policy_decision_frame_id"), int)
        )
        response_audio_provenance.append({
            **origin,
            "audio_delta_events": audio_delta_count,
            "trigger": "automatic_policy" if has_policy_decision else "explicit_api",
            "trigger_basis": "response.created policy decision provenance",
        })
    response_audio_provenance.sort(key=lambda row: (row.get("event_file", ""), row.get("response_id", "")))

    return {
        "clock": "server time.monotonic_ns",
        "warmup_ticks_per_session": warmup_ticks_per_session,
        "total_ticks": total_ticks,
        "batch_size_histogram": batch_size_histogram,
        "dispatch_mode_histogram": dispatch_modes,
        "row_batch_size_histogram": row_batch_sizes,
        "row_dispatch_mode_histogram": row_dispatch_modes,
        "rows_without_listener_batch_id": rows_without_batch_id,
        "batch_timing": {
            "total_batches": len(batches),
            "warmup_ticks_per_session": warmup_ticks_per_session,
            "batch_size_histogram": batch_size_histogram,
            "dispatch_mode_histogram": dispatch_modes,
            "all": summarize_batches(batches),
            **batch_groups,
            "batches": batches,
        },
        "response_audio_provenance": response_audio_provenance,
        "batch_errors": errors,
        "warmup": summarize(groups["warmup"]),
        "steady_state": summarize(groups["steady_state"]),
        "definition": (
            "deadline-to-listener-end includes scheduler lateness and listener callback; "
            "deadline-to-policy-decision joins the per-frame server monotonic decision stamp; "
            "batch deadline-to-last-policy-decision uses the latest decision across every acknowledged row. "
            "Batch-shared phase timings are counted once only when every ACK copy agrees. "
            "Semantic-prefix build, replay, and import timings are per-row sums of phases that actually ran; "
            "rows without that work are omitted. "
            "Warmup is classified per session; mixed-age batches are reported separately. "
            "Response audio trigger is automatic_policy when response.created carries policy-decision provenance, "
            "otherwise explicit_api, matching the server's two response creation paths."
        ),
    }


async def run_concurrent_single_sessions(
    ws_url: str,
    output_dir: Path,
    audio: Path,
    followup_audio: Path | None,
    duration_s: float,
    *,
    session_count: int,
    instructions: str = "",
    warmup_ticks_per_session: int = 5,
) -> dict[str, Any]:
    """Start N independent real audio conversations on one shared barrier."""
    if session_count < 1:
        raise ValueError("session_count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    start_barrier = asyncio.Event()
    prepared = [asyncio.Event() for _ in range(session_count)]
    probes = [
        SingleStreamProbe(
            ws_url,
            output_dir / f"session_{index + 1:02d}",
            instructions=(f"{instructions.strip()}\n" if instructions.strip() else "")
                         + f"Client {index + 1}: keep your context distinct from other sessions.",
        )
        for index in range(session_count)
    ]
    tasks = [
        asyncio.create_task(
            probe.run(
                audio, followup_audio, duration_s,
                start_barrier=start_barrier, prepared_event=prepared[index],
            ),
            name=f"listener-benchmark-session-{index + 1}",
        )
        for index, probe in enumerate(probes)
    ]
    try:
        ready_waiters = {
            asyncio.create_task(event.wait(), name=f"listener-benchmark-ready-{index + 1}")
            for index, event in enumerate(prepared)
        }
        deadline = asyncio.get_running_loop().time() + 30.0
        try:
            while not all(event.is_set() for event in prepared):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("concurrent listener clients did not become ready within 30 seconds")
                done, _pending = await asyncio.wait(
                    [*tasks, *ready_waiters],
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise TimeoutError("concurrent listener clients did not become ready within 30 seconds")
                for index, task in enumerate(tasks):
                    if not task.done():
                        continue
                    if task.cancelled():
                        raise RuntimeError(
                            f"listener benchmark session {index + 1} was cancelled before the shared start barrier"
                        )
                    error = task.exception()
                    if error is not None:
                        raise RuntimeError(
                            f"listener benchmark session {index + 1} failed before the shared start barrier: {error}"
                        ) from error
                    if not prepared[index].is_set():
                        raise RuntimeError(
                            f"listener benchmark session {index + 1} exited before the shared start barrier"
                        )
                ready_waiters.difference_update(done)
        finally:
            for waiter in ready_waiters:
                waiter.cancel()
            await asyncio.gather(*ready_waiters, return_exceptions=True)
        barrier_release_ns = time.monotonic_ns()
        start_barrier.set()
        session_reports = await asyncio.gather(*tasks)
    except BaseException:
        start_barrier.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    event_files = [output_dir / f"session_{index + 1:02d}" / "events.jsonl"
                   for index in range(session_count)]
    responses = [
        {
            "session_index": index + 1,
            "session_id": row.get("session_id"),
            "response_id": response.get("response_id"),
            "text": response.get("text", ""),
            "audio_samples": response.get("audio_samples", 0),
            "audio_duration_s": response.get("audio_duration_s", 0.0),
            "wav_path": response.get("wav_path"),
        }
        for index, row in enumerate(session_reports)
        for response in row.get("responses", [])
    ]
    report: dict[str, Any] = {
        "mode": "concurrent_realtime_audio_sessions",
        "ws_url": ws_url,
        "session_count": session_count,
        "shared_input_wav": str(audio),
        "shared_followup_wav": str(followup_audio) if followup_audio else None,
        "duration_s": duration_s,
        "barrier_release_monotonic_ns": barrier_release_ns,
        "final_ack_drain_complete": all(
            row.get("final_ack_drain_complete") is True
            and row.get("pending_after_close") == 0
            for row in session_reports
        ),
        "pending_after_close": sum(
            int(row.get("pending_after_close", 0) or 0) for row in session_reports
        ),
        "session_reports": session_reports,
        "tick_timing": summarize_listener_tick_events(
            event_files, warmup_ticks_per_session=warmup_ticks_per_session,
        ),
        "responses": responses,
        "overlap": {
            "input_frames_during_audio_generation": sum(
                int(row.get("input_frames_during_audio_generation", 0)) for row in session_reports
            ),
            "non_silent_input_frames_during_audio_generation": sum(
                int(row.get("non_silent_input_frames_during_audio_generation", 0))
                for row in session_reports
            ),
        },
        "all_sessions_healthy": all(
            not row.get("capacity_overflow") and not row.get("missing_ack_ids")
            and not row.get("errors") and not row.get("sender_error")
            and row.get("source_playback_complete") is True
            and row.get("final_ack_drain_complete") is True
            and row.get("pending_after_close") == 0
            and row.get("input_frames_acked") == row.get("input_frames_sent")
            for row in session_reports
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
                 bargein_wav: Path | None = None,
                 bargein_target: str = "B",
                 bargein_delay_ms: int = 500,
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
        self.bargein_wav = Path(bargein_wav) if bargein_wav is not None else None
        self.bargein_target = str(bargein_target).upper()
        self.bargein_delay_ms = int(bargein_delay_ms)
        if self.bargein_target not in {"A", "B"}:
            raise ValueError("bargein_target must be A or B")
        if self.bargein_delay_ms < 0:
            raise ValueError("bargein_delay_ms must be non-negative")
        self.bargein_scheduled = False
        self.bargein_task: asyncio.Task | None = None
        self.bargein_injections: dict[str, dict[str, Any]] = {}
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
                timeout=ACK_CREDIT_WAIT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            self._fail_manual_seed_bootstrap(
                peer, "session.updated for auto_response=true was not received within the ACK wait limit",
                peer.get("bootstrap_response_id"),
            )
            return
        if peer.get("bootstrap_phase") != "failed":
            peer["bootstrap_phase"] = "complete"

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
        bootstrap_updates = [
            peer["bootstrap_auto_response_enable_ack"].wait()
            for peer in peers
            if peer.get("bootstrap_auto_response_enable_requested")
            and not peer.get("bootstrap_auto_response_enable_confirmed")
        ]
        if bootstrap_updates:
            await asyncio.gather(*bootstrap_updates)
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
                        if deadline is not None else ACK_CREDIT_WAIT_TIMEOUT_S
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
                            asyncio.get_running_loop().time() + ACK_CREDIT_WAIT_TIMEOUT_S
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
            source_kind = frame_metadata.get("source_kind")
            injection_id = frame_metadata.get("injection_id")
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
            if source_kind == "injection":
                peer.setdefault("input_frame_provenance", {})[local_frame_id] = {
                    key: value for key, value in frame_metadata.items()
                    if key != "frame_index"
                }
            peer["pending_high_water"] = max(peer["pending_high_water"], len(peer["ack_pending"]))
            try:
                await ws.send(json.dumps(payload))
            except Exception as exc:
                if peer["ack_pending"] and peer["ack_pending"][-1][0] == local_frame_id:
                    peer["ack_pending"].pop()
                peer.setdefault("input_frame_provenance", {}).pop(local_frame_id, None)
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
                        asyncio.get_running_loop().time() + ACK_CREDIT_WAIT_TIMEOUT_S
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
                **({
                    "source_kind": "injection",
                    "injection_id": injection_id,
                    "trigger_response_id": frame_metadata.get("trigger_response_id"),
                } if source_kind == "injection" else {}),
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
                **({
                    "source_kind": "injection",
                    "injection_id": injection_id,
                    "trigger_response_id": frame_metadata.get("trigger_response_id"),
                    "generation_id": frame_metadata.get("generation_id"),
                    "delay_ms": frame_metadata.get("delay_ms"),
                    "trigger_received_at_ns": frame_metadata.get("trigger_received_at_ns"),
                    "frame_index": frame_metadata.get("frame_index"),
                    "frame_count": frame_metadata.get("frame_count"),
                    "enqueued_at_ns": frame_metadata.get("enqueued_at_ns"),
                } if source_kind == "injection" else {}),
                "bridge_queue_age_ms": round((sent_ns - enqueued_ns) / 1_000_000.0, 2),
                "pending_acks": len(peer["ack_pending"]),
                "during_local_audio_output": bool(local_audio_response_ids),
                "local_audio_response_ids": local_audio_response_ids,
                "rms": round(current_rms, 5),
                "user_log10_rms_current_80ms": round(float(np.clip(log_current, -6, 0)), 6),
                "user_log10_rms_past_400ms_inclusive": round(float(np.clip(math.log10(max(past_rms, 1e-6)), -6, 0)), 6),
                "user_log_rms_delta_clip6_div6": round(float(np.clip(log_current - log_previous, -6, 6) / 6), 6),
            })
            if source_kind == "injection":
                injection = self.bargein_injections.get(str(injection_id))
                if injection is not None:
                    injection["frames_sent"] += 1
                    injection["first_sent_at_ns"] = injection["first_sent_at_ns"] or sent_ns
                    injection["last_sent_at_ns"] = sent_ns
                    if injection["frames_sent"] == injection.get("frame_count"):
                        injection["status"] = "sent"
                        self.log(peer["speaker"], {
                            "type": "probe.bargein_injection_sent",
                            "source_kind": "injection",
                            "injection_id": str(injection_id),
                            "trigger_response_id": injection["trigger_response_id"],
                            "generation_id": injection["generation_id"],
                            "trigger_received_at_ns": injection["trigger_received_at_ns"],
                            "delay_ms": injection["delay_ms"],
                            "enqueue_started_at_ns": injection.get("enqueue_started_at_ns"),
                            "enqueued_at_ns": injection.get("enqueued_at_ns"),
                            "first_sent_at_ns": injection["first_sent_at_ns"],
                            "last_sent_at_ns": injection["last_sent_at_ns"],
                            "frame_count": injection["frame_count"],
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
                return
            event_received_ns = time.monotonic_ns()
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
                if local_frame_id is not None:
                    ack_event.update(
                        peer.setdefault("input_frame_provenance", {}).pop(local_frame_id, {})
                    )
                ack_event.update(_project_input_ack_fields(normalized_event))
                self.log(speaker, ack_event)
                peer["ack_wakeup"].set()
            elif kind == "session.updated":
                if (peer.get("bootstrap_phase") == "enabling_auto_response"
                        and peer.get("bootstrap_auto_response_enable_requested")
                        and not peer.get("bootstrap_auto_response_enable_confirmed")):
                    peer["bootstrap_auto_response_enable_confirmed"] = True
                    peer["bootstrap_auto_response_enable_ack"].set()
                    self.log(speaker, {
                        "type": "probe.manual_seed_bootstrap_auto_response_enable_ack",
                        "acknowledged_at_ns": time.monotonic_ns(),
                        "response_id": peer.get("bootstrap_response_id"),
                        "auto_response": True,
                    })
                elif peer["auto_response_disable_requested"]:
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
                        and peer.get("bootstrap_phase") == "awaiting_response"):
                    peer["bootstrap_response_id"] = response_id
                    peer["bootstrap_phase"] = "response_created"
                elif (peer.get("manual_seed_bootstrap")
                      and peer.get("bootstrap_phase") == "response_created"
                      and response_id != peer.get("bootstrap_response_id")):
                    self._fail_manual_seed_bootstrap(
                        peer, "more than one response was created for the one-shot seed commit", response_id,
                    )
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
                        "event_received_at_ns": event_received_ns,
                        "sample_count": int(packet.pcm.size),
                        "sample_rate_hz": int(packet.sample_rate),
                    })
                    self._enqueue_route_item(peer, (
                        "audio", response_id, output_chunk_id,
                        packet.pcm.copy(), now_ns,
                    ))
                    if (
                        self.bargein_wav is not None
                        and speaker == self.bargein_target
                        and packet.pcm.size > 0
                        and not self.bargein_scheduled
                    ):
                        self.bargein_scheduled = True
                        injection_id = (
                            f"bargein:{speaker}:{response_id}:generation-{int(packet.generation_id)}"
                        )
                        injection = {
                            "source_kind": "injection",
                            "injection_id": injection_id,
                            "target_speaker": speaker,
                            "trigger_response_id": response_id,
                            "generation_id": int(packet.generation_id),
                            "trigger_received_at_ns": event_received_ns,
                            "delay_ms": self.bargein_delay_ms,
                            "frame_count": None,
                            "frames_enqueued": 0,
                            "frames_sent": 0,
                            "enqueue_started_at_ns": None,
                            "enqueued_at_ns": None,
                            "first_sent_at_ns": None,
                            "last_sent_at_ns": None,
                            "status": "scheduled",
                        }
                        self.bargein_injections[injection_id] = injection
                        self.log(speaker, {
                            "type": "probe.bargein_injection_scheduled",
                            "source_kind": "injection",
                            "injection_id": injection_id,
                            "trigger_response_id": response_id,
                            "generation_id": int(packet.generation_id),
                            "target_speaker": speaker,
                            "trigger_received_at_ns": event_received_ns,
                            "delay_ms": self.bargein_delay_ms,
                        })
                        self.bargein_task = asyncio.create_task(
                            self._inject_bargein(peer, injection),
                            name=f"pair-bargein-{speaker}-{response_id}",
                        )
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
                    if (peer.get("manual_seed_bootstrap")
                            and response_id == peer.get("bootstrap_response_id")):
                        status = _manual_seed_bootstrap_terminal_status(kind, event)
                        action = str(response.get("response_action") or response.get("action") or "").upper()
                        audio_samples = int(response.get("audio_samples", 0))
                        if not _manual_seed_bootstrap_response_succeeded(kind, event, response):
                            self._fail_manual_seed_bootstrap(
                                peer,
                                "first response was not a completed audible SPEAK "
                                f"(event={kind}, status={status or 'missing'}, action={action or 'missing'}, "
                                f"audio_samples={audio_samples}, reason={event.get('reason') or event.get('error') or 'none'})",
                                response_id,
                            )
                        elif (self.created_budget_exhausted
                              or peer.get("auto_response_disable_requested", False)):
                            self._fail_manual_seed_bootstrap(
                                peer, "harness response drain began before automatic conversation could resume",
                                response_id,
                            )
                        else:
                            peer["bootstrap_phase"] = "enabling_auto_response"
                            peer["bootstrap_auto_response_enable_requested"] = True
                            peer["bootstrap_auto_response_enable_ack"].clear()
                            peer["bootstrap_auto_response_enable_task"] = asyncio.create_task(
                                self._enable_manual_seed_auto_response(peer),
                                name="pair-enable-auto-response-after-seed",
                            )
            elif kind == "error":
                peer["session_errors"].append(event)
                self.errors.append({"speaker": speaker, **event})
                if (peer.get("manual_seed_bootstrap")
                        and peer.get("bootstrap_phase") not in {"complete", "failed", "disabled"}):
                    self._fail_manual_seed_bootstrap(
                        peer, f"server error: {event.get('error') or event}",
                        peer.get("bootstrap_response_id"),
                    )
                if peer["auto_response_disable_requested"]:
                    peer["auto_response_disable_ack"].set()

    async def _inject_bargein(self, peer: dict[str, Any], injection: dict[str, Any]) -> None:
        """Wait from the triggering receive timestamp, then queue frames on that peer."""
        try:
            frames = await asyncio.to_thread(_read_bargein_frames, self.bargein_wav)
            if not frames:
                raise ValueError(f"barge-in WAV contains no audio frames: {self.bargein_wav}")
            injection["frame_count"] = len(frames)

            deadline_ns = injection["trigger_received_at_ns"] + injection["delay_ms"] * 1_000_000
            while True:
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    break
                await asyncio.sleep(remaining_ns / 1_000_000_000.0)

            loop = asyncio.get_running_loop()
            queue_deadline = loop.time() + ACK_CREDIT_WAIT_TIMEOUT_S
            injection["enqueue_started_at_ns"] = time.monotonic_ns()
            for frame_index, frame in enumerate(frames):
                metadata = {
                    "source_kind": "injection",
                    "injection_id": injection["injection_id"],
                    "trigger_response_id": injection["trigger_response_id"],
                    "generation_id": injection["generation_id"],
                    "delay_ms": injection["delay_ms"],
                    "trigger_received_at_ns": injection["trigger_received_at_ns"],
                    "frame_index": frame_index,
                    "frame_count": len(frames),
                    "enqueued_at_ns": None,
                }
                while peer["queue"].full():
                    remaining_queue_s = queue_deadline - loop.time()
                    if remaining_queue_s <= 0:
                        raise asyncio.TimeoutError("barge-in queue backpressure exceeded timeout")
                    await asyncio.sleep(min(0.01, remaining_queue_s))
                enqueued_ns = time.monotonic_ns()
                metadata["enqueued_at_ns"] = enqueued_ns
                item = (
                    np.asarray(frame, dtype=np.float32).copy(),
                    injection["injection_id"],
                    (),
                    enqueued_ns,
                    metadata,
                )
                peer["queue"].put_nowait(item)
                injection["enqueued_at_ns"] = enqueued_ns
                injection["frames_enqueued"] += 1
                peer["queue_frames_enqueued"] += 1
                peer["queue_high_water"] = max(peer["queue_high_water"], peer["queue"].qsize())

            if injection["status"] != "sent":
                injection["status"] = "enqueued"
            self.log(peer["speaker"], {
                "type": "probe.bargein_injection_enqueued",
                "source_kind": "injection",
                "injection_id": injection["injection_id"],
                "trigger_response_id": injection["trigger_response_id"],
                "generation_id": injection["generation_id"],
                "target_speaker": peer["speaker"],
                "trigger_received_at_ns": injection["trigger_received_at_ns"],
                "delay_ms": injection["delay_ms"],
                "enqueue_started_at_ns": injection["enqueue_started_at_ns"],
                "enqueued_at_ns": injection["enqueued_at_ns"],
                "frame_count": injection["frame_count"],
            })
        except asyncio.CancelledError:
            injection["status"] = "cancelled"
            raise
        except Exception as exc:
            injection["status"] = "failed"
            injection["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, asyncio.TimeoutError):
                self.overflow = True
            self.errors.append({
                "speaker": peer["speaker"],
                "error": f"barge-in injection failed: {injection['error']}",
                "injection_id": injection["injection_id"],
                "source_kind": "injection",
                "frames_enqueued": injection["frames_enqueued"],
            })
            self.log(peer["speaker"], {
                "type": "probe.bargein_injection_failed",
                "source_kind": "injection",
                "injection_id": injection["injection_id"],
                "trigger_response_id": injection["trigger_response_id"],
                "generation_id": injection["generation_id"],
                "trigger_received_at_ns": injection["trigger_received_at_ns"],
                "delay_ms": injection["delay_ms"],
                "enqueue_started_at_ns": injection["enqueue_started_at_ns"],
                "enqueued_at_ns": injection["enqueued_at_ns"],
                "frame_count": injection["frame_count"],
                "frames_enqueued": injection["frames_enqueued"],
                "error": injection["error"],
            })

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
        await peer_a["queue"].put((
            self._END_UTTERANCE,
            self.endpoint_silence_frames,
            None,
            "manual_seed_bootstrap" if self.manual_seed_bootstrap else None,
        ))

        try:
            while loop.time() < hard_deadline:
                drain_requested = self.created_target_event.is_set() or self.created_budget_exhausted
                if drain_requested:
                    queues_empty = all(peer["queue"].empty() for peer in (peer_a, peer_b))
                    routes_drained = all(peer["route_pending"] == 0 for peer in (peer_a, peer_b))
                    no_active = not peer_a["active"] and not peer_b["active"]
                    no_pending_injection = self.bargein_task is None or self.bargein_task.done()
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
                        and no_pending_injection
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
            if (self.manual_seed_bootstrap
                    and peer_a.get("bootstrap_phase") not in {"complete", "failed"}):
                self._fail_manual_seed_bootstrap(
                    peer_a,
                    "run ended before the initial SPEAK completed and auto_response=true was acknowledged",
                    peer_a.get("bootstrap_response_id"),
                )
            if self.bargein_task is not None and not self.bargein_task.done():
                self.bargein_task.cancel()
                await asyncio.gather(self.bargein_task, return_exceptions=True)
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
            for injection in self.bargein_injections.values():
                if injection["status"] != "sent":
                    self.log(injection["target_speaker"], {
                        "type": "probe.bargein_injection_shutdown",
                        "source_kind": "injection",
                        "injection_id": injection["injection_id"],
                        "trigger_response_id": injection["trigger_response_id"],
                        "generation_id": injection["generation_id"],
                        "trigger_received_at_ns": injection["trigger_received_at_ns"],
                        "delay_ms": injection["delay_ms"],
                        "enqueue_started_at_ns": injection["enqueue_started_at_ns"],
                        "enqueued_at_ns": injection["enqueued_at_ns"],
                        "frame_count": injection["frame_count"],
                        "frames_enqueued": injection["frames_enqueued"],
                        "frames_sent": injection["frames_sent"],
                        "status": (
                            "cancelled_pending_injection"
                            if injection["status"] == "cancelled"
                            else injection["status"]
                        ),
                        "reported_at_ns": time.monotonic_ns(),
                    })
            if self.auto_response_disable_task is not None and not self.auto_response_disable_task.done():
                self.auto_response_disable_task.cancel()
                await asyncio.gather(self.auto_response_disable_task, return_exceptions=True)
            bootstrap_enable_task = peer_a.get("bootstrap_auto_response_enable_task")
            if bootstrap_enable_task is not None:
                if not bootstrap_enable_task.done():
                    bootstrap_enable_task.cancel()
                await asyncio.gather(bootstrap_enable_task, return_exceptions=True)
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
                "phase": peer_a.get("bootstrap_phase"),
                "seed_frame_count": peer_a.get("bootstrap_seed_frame_count", 0),
                "endpoint_silence_frame_count": peer_a.get("bootstrap_tail_frames_sent", 0),
                "commit_sent": peer_a.get("bootstrap_commit_sent", False),
                "commit_sent_at_ns": peer_a.get("bootstrap_commit_sent_at_ns"),
                "response_id": peer_a.get("bootstrap_response_id"),
                "auto_response_enable_confirmed": peer_a.get("bootstrap_auto_response_enable_confirmed", False),
                "failure_reason": peer_a.get("bootstrap_failure_reason"),
            },
            "capacity_overflow": self.overflow,
            "bargein_injections": [dict(injection) for injection in self.bargein_injections.values()],
            "errors": self.errors,
            "participants": reports,
            "responses": response_reports,
            "events_jsonl": str(self.events_path),
        }
        (self.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


def _single_mode_report_failed(report: dict[str, object]) -> bool:
    return bool(
        report.get("capacity_overflow")
        or report.get("missing_ack_ids")
        or report.get("errors")
        or report.get("sender_error")
        or not report.get("final_ack_drain_complete", False)
        or not report.get("source_playback_complete", False)
        or report.get("input_frames_not_dispatched", 0) != 0
    )


def main() -> int:
    global TURN_POLICY
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "pair"), default="single")
    parser.add_argument("--turn-policy", choices=("gemma_native_control6", "duplex_control6", "turnhead"),
                        default=TURN_POLICY)
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8000/v1/realtime?duplex=1")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--followup-audio", type=Path)
    parser.add_argument(
        "--followup-after-response-done-ms", type=int,
        help="opt-in: send follow-up audio this many milliseconds after first natural response.done",
    )
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--instructions", default="")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument(
        "--manual-seed-bootstrap", action="store_true",
        help="opt-in: commit the initial A seed once after its FIFO frame ACKs, then resume auto responses after its SPEAK",
    )
    parser.add_argument("--max-created-responses", type=int)
    parser.add_argument("--max-duration-s", type=float, default=120.0)
    parser.add_argument(
        "--concurrent-sessions", type=int,
        help="run this many independent single-session conversations from one synchronized start barrier",
    )
    parser.add_argument(
        "--warmup-ticks", type=int, default=5,
        help="per-session initial consumed ticks reported separately from steady state in concurrent mode",
    )
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
    parser.add_argument("--bargein-wav", type=Path,
                        help="opt-in WAV input injected into one pair participant after its first audio delta")
    parser.add_argument("--bargein-target", choices=("A", "B"), default="B")
    parser.add_argument("--bargein-delay-ms", type=int, default=500)
    parser.add_argument("--instructions-a", default="Sei una persona che organizza una cena con amici. Parla in modo naturale e condividi le tue preferenze.")
    parser.add_argument("--instructions-b", default="Sei un assistente conversazionale italiano. Aiuta a scegliere insieme un ristorante e un orario con proposte concise.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    TURN_POLICY = args.turn_policy
    if args.concurrent_sessions is not None and args.concurrent_sessions < 1:
        parser.error("--concurrent-sessions must be positive")
    if args.warmup_ticks < 0:
        parser.error("--warmup-ticks must be non-negative")
    if args.mode == "pair" and args.concurrent_sessions is not None:
        parser.error("--concurrent-sessions is only available in single mode")
    if args.manual_seed_bootstrap and args.mode != "pair":
        parser.error("--manual-seed-bootstrap is only available in pair mode")
    if args.manual_seed_bootstrap and args.max_turns < 2:
        parser.error("--manual-seed-bootstrap requires --max-turns of at least 2")
    if args.bargein_wav is not None and args.mode != "pair":
        parser.error("--bargein-wav is only available in pair mode")
    if args.bargein_wav is not None and args.bargein_delay_ms < 0:
        parser.error("--bargein-delay-ms must be non-negative")
    if args.followup_after_response_done_ms is not None:
        if args.followup_after_response_done_ms < 0:
            parser.error("--followup-after-response-done-ms must be non-negative")
        if args.mode != "single" or args.concurrent_sessions is not None:
            parser.error("--followup-after-response-done-ms is only available in single non-concurrent mode")
        if args.followup_audio is None:
            parser.error("--followup-after-response-done-ms requires --followup-audio")
    if args.mode == "single" and args.concurrent_sessions is not None:
        report = asyncio.run(run_concurrent_single_sessions(
            args.ws_url,
            args.output_dir,
            args.audio,
            args.followup_audio,
            args.duration_s,
            session_count=args.concurrent_sessions,
            instructions=args.instructions,
            warmup_ticks_per_session=args.warmup_ticks,
        ))
        failed = not bool(report["all_sessions_healthy"])
    elif args.mode == "single":
        probe = SingleStreamProbe(
            args.ws_url,
            args.output_dir,
            instructions=args.instructions,
            voice_conditioning_mode=(None if args.voice_conditioning_mode == "none" else args.voice_conditioning_mode),
            tts_quality_diagnostics=args.tts_quality_diagnostics,
        )
        report = asyncio.run(probe.run(
            args.audio,
            args.followup_audio,
            args.duration_s,
            followup_after_response_done_ms=args.followup_after_response_done_ms,
        ))
        failed = _single_mode_report_failed(report)
    else:
        if args.max_turns < 1 or args.max_duration_s <= 0 or args.endpoint_silence_frames < 0:
            parser.error("pair mode requires positive turns/duration and non-negative endpoint silence frames")
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
            bargein_wav=args.bargein_wav,
            bargein_target=args.bargein_target,
            bargein_delay_ms=args.bargein_delay_ms,
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
    if args.mode == "single" and args.concurrent_sessions is not None:
        keys = (
            "mode", "session_count", "all_sessions_healthy", "final_ack_drain_complete",
            "pending_after_close", "tick_timing", "overlap", "responses",
        )
    elif args.mode == "single":
        keys = (
            "mode", "session_id", "input_frames_sent", "input_frames_acked", "missing_ack_ids",
            "input_frames_not_dispatched", "source_playback_complete",
            "final_ack_drain_complete", "final_ack_drain_timed_out", "pending_after_close",
            "capacity_overflow", "pending_high_water_frames", "ack_latency_ms", "input_send_interval_ms",
            "input_frames_during_response", "non_silent_input_frames_during_response", "responses", "errors", "sender_error"
        )
    else:
        keys = (
            "mode", "elapsed_s", "created_responses", "target_audible_responses",
            "audible_completed_responses", "max_created_responses", "created_budget_exhausted",
            "responses_created_over_hard_budget", "completed_responses", "capacity_overflow",
            "participants", "responses", "errors",
        )
    print(json.dumps({key: report[key] for key in keys}, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
