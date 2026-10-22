"""OpenAI-compatible HTTP/SSE and realtime duplex WebSocket server."""
from __future__ import annotations
import asyncio, base64, hashlib, io, json, logging, os, queue, re, tempfile, threading, time, uuid
import itertools
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, deque
from math import gcd
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from duplex_kernel.audio_stream import (
    CODEC_SAMPLE_RATE,
    NETWORK_PACKET_FRAMES,
    AudioPacket,
    NetworkPacketizer,
    PacketSequenceNumber,
    iter_audio_packets,
)
from duplex_kernel.config import (
    ARIA_BACKCHANNEL_MAX_FRAMES,
    ARIA_MAX_FRAMES,
    CODEC_SAMPLE_RATE,
    DEFAULT_VOICE,
    FIRST_INPUT_SCHEDULER_ANCHOR_ENABLED,
    LISTENER_PAIR_LOCK_HANDOFF_ENV,
    NATIVE_LISTENER_CUDA_PROFILE_ENABLED,
    NATIVE_LISTENER_CUDA_PROFILE_DIRECTORY_ENV,
    NATIVE_CONTROL6_FUSED_READBACK_ENABLED,
    NATIVE_CONTROL6_FUSED_READBACK_ENV,
    READY_FRAME_SCHEDULER_DISPATCH_ENABLED,
    V9_TRAINING_REFERENCE_SHA256,
    USER_SAMPLE_RATE,
    TURNHEAD_CKPT,
)
from duplex_kernel.conditioning_provenance import (
    CONDITIONING_PATHS,
    make_trace_row,
    pcm_frame_provenance,
    runtime_source_fingerprint,
)
from duplex_kernel.listener import (
    GemmaAudioTowerBackend,
    GemmaListener,
    capture_host_timeline_nonblocking,
)
from duplex_kernel.playout import PlayoutFrame, SpeakerPlayoutBuffer
from duplex_kernel.scheduler import DIRECT_V2_INGRESS_QUEUE_FRAMES, REALTIME_INGRESS_QUEUE_FRAMES
from duplex_kernel.shared_listener_scheduler import SharedListenerInput, SharedListenerTickDispatcher
from duplex_kernel.microbatch_contract import ListenerFrame
from duplex_kernel.listener_microbatch_serving import (
    _semantic_prefix_request_matches_applied_row,
    commit_initial_semantic_prefix_state,
    initial_listener_session_state,
    make_listener_microbatch_worker,
    make_unified_prefix_import,
    prepare_initial_semantic_prefix_async,
    semantic_prefix_importer,
)
from duplex_kernel.turn_policy import (
    ControlAction,
    ControlState,
    GateEvent,
    NoOpPolicy,
    PolicyDecision,
    PolicyInput,
    ProvisionalFramePolicy,
    ScriptedPolicy,
    SpeakerGate,
)
from duplex_kernel.turnhead_adapter import TurnHeadPolicy
from duplex_kernel.gemma_policy import GemmaPolicy
from duplex_kernel.duplex_control6 import DuplexControl6Policy, legal_control_actions
from duplex_kernel.gemma_native_control6_policy import (
    GemmaNativeControl6Policy,
    legal_control_actions as native_control6_legal_actions,
)
from duplex_kernel.gemma_native_control6_policy import is_controller6_policy
from duplex_kernel.native_control6_diagnostics import (
    attach_native_control6_zero_output_diagnostic,
    native_control6_validate_listener_output_health,
    native_control6_verify_scored_output_health,
)
from duplex_kernel.native_control6_runtime import (
    build_native_control6_semantic_prefix_request,
    prepare_native_control6_semantic_prefix,
    record_native_control6_semantic_audio,
)
from duplex_kernel.listener_attestation import (
    LISTENER_ATTESTATION_DIRECTORY_ENV,
    write_listener_attestation_from_environment,
)
from duplex_kernel.native_control6_prompt import (
    NATIVE_CONTROL6_PREFIX_MAX_TOKENS,
    native_control6_semantic_prefix_metadata_is_supported,
    validate_native_control6_prefix_ids,
)
from training.duplex_qlora.native_control6 import ACTIONS as NATIVE_CONTROL6_ACTIONS
from duplex_kernel.engine import ListenerBacklogAdmission
from duplex_kernel.unified_native_decoder import UnifiedNativeDecoder
from duplex_kernel.models.thinker import normalize_eos_token_ids
from duplex_kernel.action_conditioning import (
    action_control_marker_text,
    action_conditioned_messages,
    action_conditioning_instruction,
)

from duplex_kernel.audio_response_conditioning import (
    DEFAULT_MAX_AUDIO_MS,
    build_audio_response_messages,
    build_audio_response_protocol_v2_messages,
    build_controller_guided_say_v1_messages,
    causal_audio_window,
)
from duplex_kernel.audio_response_protocol import (
    AUDIO_RESPONSE_ACTIONS,
    AUDIO_RESPONSE_PROTOCOL_V2,
    AUDIO_RESPONSE_PROTOCOL_ENV,
    CONTROLLER_GUIDED_SAY_V1,
    CONTROLLER_GUIDED_SAY_COMPAT_ENV,
    AUDIO_RESPONSE_V2_FAILURE_TRACE_MAX_RECORDS,
    AUDIO_RESPONSE_V2_MAX_ACTION_TOKENS,
    AUDIO_RESPONSE_V2_MAX_HEARD_TOKENS,
    AUDIO_RESPONSE_V2_MAX_NEW_TOKENS,
    AUDIO_RESPONSE_V2_MAX_SAY_TOKENS,
    AudioResponseV2ActionConstraint,
    AudioResponseProtocolError,
    AudioResponseV2Collector,
    audio_response_actions_for_wake,
    audio_response_disposition_for_actions,
    controller_backchannel_candidate_allowed,
    controller_backchannel_should_stop,
    controller_guided_response_action,
    audio_response_v2_say_token_char_spans,
    finish_audio_response_v2_with_say_hidden,
    normalize_audio_response_protocol,
    resolve_audio_response_protocol,
    slice_controller_guided_say_hidden,
)
from duplex_kernel.response_intent_local import (
    align_generated_response_hidden_to_qwen_tokens,
    align_hidden_states_by_char_overlap,
    qwen_assistant_text_offsets,
)
from duplex_kernel.backchannel_guard import (
    BackchannelAttemptLatch,
    peer_audio_event_key_for_frame,
    validate_backchannel_phrase,
)
from duplex_kernel.tts_quality_diagnostics import (
    DIAGNOSTIC_MAX_ACOUSTIC_FRAME_BUDGET,
    TTSQualityDiagnosticCapture,
)
from duplex_kernel.session import (
    DuplexSession,
    GemmaTokenTickPacer,
    GenerationToken,
    SessionManager,
    SpokenTokenPacket,
)
from duplex_kernel.live_say import LiveQwenSayStream, TurnLocalQwenSayStream
from duplex_kernel.tts_units import SpokenTextUnit, SpokenTextUnitAssembler
from duplex_kernel.cpu_audio_archive import CPU_AUDIO_ARCHIVE_WORKER
from duplex_kernel.native_partial_transcription import (
    NativePartialTranscriber,
    NativeTranscriptSnapshot,
    should_request_partial_snapshot,
)
from duplex_kernel.tool_parser import extract_tool_calls, tool_system_prompt
from duplex_kernel.voices import resolve_voice_conditioning

logger = logging.getLogger("duplex_kernel.server")


@dataclass(frozen=True)
class _ClosedTTSSegment:
    """Event-loop fence emitted after one text-owned Qwen unit is fully drained."""

    speaker_segment_id: str
    source_tokens: tuple[dict[str, Any], ...]
_NATIVE_CONTROL6_TRACE_ENABLED = os.environ.get("DUPLEX_NATIVE_CONTROL6_TRACE", "0").strip() == "1"
_NATIVE_CONTROL6_DEBUG_INPUT_DIRECTORY = (
    os.environ.get("DUPLEX_NATIVE_CONTROL6_DEBUG_INPUT_DIRECTORY", "").strip() or None
)
_NATIVE_CONTROL6_DEBUG_INPUT_MAX_FRAMES = 256
_AUDIO_HISTORY_ARCHIVE_DEFER_ENV = "DUPLEX_AUDIO_HISTORY_ARCHIVE_DEFER"
_LISTENER_MICROBATCH_ENABLED_ENV = "DUPLEX_LISTENER_MICROBATCH"
_UNIFIED_KV_ENABLED_ENV = "DUPLEX_UNIFIED_KV"
_SEMANTIC_SHADOW_ENABLED_ENV = "DUPLEX_LISTENER_MICROBATCH_SEMANTIC_SHADOW"
NATIVE_LISTENER_PROFILE_STOP_TIMEOUT_S = 1.0
_NATIVE_CONTROL6_SELECTION_TRACE_FIELDS = (
    "policy_fallback_reason",
    "native_control6_effective_candidate",
    "native_control6_effective_order",
    "native_control6_effective_actions",
    "native_control6_effective_members",
    "native_control6_legal_effective_mass",
    "native_control6_selected_effective_probability",
    "native_control6_hysteresis_count",
    "native_control6_hysteresis_pending",
    "native_control6_hysteresis_applied_action",
    "native_action_raw_probabilities",
    "native_action_effective_order",
    "native_action_effective_probabilities",
    "native_action_legal_effective_probabilities",
    "native_action_legal_effective_mass",
    "native_control6_raw_row_index",
    "native_control6_raw_token_id",
    "native_control6_raw_token",
    "native_control6_selected_row_index",
    "native_control6_selected_token_id",
    "native_control6_selected_token",
)


def _attach_native_control6_trace_diagnostic(
    trace: dict[str, Any],
    *,
    native_listener: Any,
    audio_soft: Any,
    q4_output: Any,
    prior_health_result: dict[str, Any] | None = None,
) -> None:
    """Attach optional output diagnostics only for an actually scored head."""
    attach_native_control6_zero_output_diagnostic(
        trace,
        enabled=_NATIVE_CONTROL6_TRACE_ENABLED,
        native_policy=True,
        head_scored=bool(trace.get("native_control6_head_scored")),
        native_listener=native_listener,
        audio_soft=audio_soft,
        q4_output=q4_output,
        prior_health_result=prior_health_result,
    )


def _native_control6_event_annotations(
    decision: Any,
    *,
    head_scored: bool,
    accepted: bool,
) -> dict[str, Any]:
    """Return selected-row diagnostics and final gate disposition for an event.

    Raw/selected token IDs name the six projected Control6 action rows. They
    are not an autoregressive full-vocabulary token sample. Scheduler fallback
    decisions have no scored row and therefore carry no row/token candidate.
    Gate acceptance describes control-plane acceptance, not audible output.
    """
    fields: dict[str, Any] = {
        "gate_applied_action": (
            str(getattr(decision.action, "value", decision.action)) if accepted else None
        ),
        "gate_application_scope": "control_plane_acceptance_not_audible_output",
    }
    fields.update(_native_control6_selection_fields(decision, head_scored=head_scored))
    return fields


def _native_control6_debug_input_row(
    *,
    policy_input: Any,
    decision: Any,
    accepted: bool,
    phase_profile: str,
) -> dict[str, Any] | None:
    """Copy one bounded hidden/phase decision input for opt-in offline comparison."""
    diagnostics = getattr(decision, "diagnostics", None)
    debug_input = None
    if isinstance(diagnostics, dict):
        debug_input = diagnostics.pop("_native_control6_debug_input", None)
    if debug_input is not None:
        if not isinstance(debug_input, Mapping):
            raise TypeError("native Control6 debug input payload must be a mapping")
        if (
            debug_input.get("frame_id") != int(policy_input.frame_id)
            or debug_input.get("generation_id") != int(policy_input.generation_id)
            or debug_input.get("phase_profile") != phase_profile
        ):
            raise ValueError("native Control6 debug input does not match its policy tick")
        hidden = debug_input.get("hidden")
        phase = debug_input.get("phase_t")
    else:
        hidden = getattr(policy_input, "native_control_hidden", None)
        phase = getattr(policy_input, "native_control_phase_t", None)
    if hidden is None or phase is None:
        return None

    def to_cpu_float32(value: Any) -> np.ndarray:
        if callable(getattr(value, "detach", None)):
            value = value.detach()
        if callable(getattr(value, "to", None)):
            value = value.to(device="cpu", dtype=torch.float32)
        return np.asarray(value, dtype=np.float32).reshape(-1)

    hidden_values = to_cpu_float32(hidden)
    phase_values = to_cpu_float32(phase)
    if hidden_values.shape != (1536,) or phase_values.shape != (4,):
        return None
    if not np.isfinite(hidden_values).all() or not np.isfinite(phase_values).all():
        return None
    user_active = (
        bool(diagnostics.get("native_control6_user_activity_detected"))
        if isinstance(diagnostics, dict)
        else False
    )
    return {
        "frame_id": int(policy_input.frame_id),
        "generation_id": int(policy_input.generation_id),
        "hidden": hidden_values.copy(),
        "phase_t": phase_values.copy(),
        "action": str(getattr(decision.action, "value", decision.action)),
        "accepted": bool(accepted),
        "speaker_active": bool(policy_input.speaker_active),
        "user_activity_detected": user_active,
        "phase_profile": str(phase_profile),
    }


def _write_native_control6_debug_input_sidecar(
    directory: str | Path | None,
    *,
    session_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> str | None:
    """Write a small binary sidecar; never expand per-frame JSON with hidden vectors."""
    if directory is None or not rows:
        return None
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{session_id}.npz"
    if destination.exists():
        raise FileExistsError(f"native Control6 debug sidecar already exists: {destination}")
    if len(rows) > _NATIVE_CONTROL6_DEBUG_INPUT_MAX_FRAMES:
        raise ValueError("native Control6 debug sidecar exceeds the 256-frame bound")
    arrays = {
        "frame_ids": np.asarray([int(row["frame_id"]) for row in rows], dtype=np.int64),
        "generation_ids": np.asarray([int(row["generation_id"]) for row in rows], dtype=np.int64),
        "hidden": np.stack([np.asarray(row["hidden"], dtype=np.float32) for row in rows]),
        "phase_t": np.stack([np.asarray(row["phase_t"], dtype=np.float32) for row in rows]),
        "action": np.asarray([str(row["action"]) for row in rows], dtype="U24"),
        "accepted": np.asarray([bool(row["accepted"]) for row in rows], dtype=np.bool_),
        "speaker_active": np.asarray([bool(row["speaker_active"]) for row in rows], dtype=np.bool_),
        "user_activity_detected": np.asarray(
            [bool(row["user_activity_detected"]) for row in rows], dtype=np.bool_
        ),
        "phase_profile": np.asarray([str(row["phase_profile"]) for row in rows], dtype="U24"),
        "session_id": np.asarray(str(session_id), dtype="U128"),
        "schema_version": np.asarray(1, dtype=np.int64),
    }
    temporary = output_dir / f".{session_id}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return str(destination)


def _native_control6_selection_fields(
    decision: Any,
    *,
    head_scored: bool,
) -> dict[str, Any]:
    """Project already-computed six-row selection metadata without tensor reads."""
    fields: dict[str, Any] = {}
    diagnostics = getattr(decision, "diagnostics", None)
    if head_scored and isinstance(diagnostics, dict):
        for name in _NATIVE_CONTROL6_SELECTION_TRACE_FIELDS:
            if name in diagnostics:
                fields[name] = diagnostics[name]
        fields["native_control6_token_selection_scope"] = (
            "six_projected_rows_collapsed_to_physical_actions_not_full_vocabulary_sample"
        )
    return fields


def scheduler_tick_trace_metadata(
    tick: Any, *, include_host_lock_timeline: bool = False
) -> dict[str, Any]:
    """Expose scheduler's host-monotonic phase stamps and signed residuals."""
    fields = (
        ("enqueue_ns", "scheduler_enqueue_ns"),
        ("deadline_ns", "scheduler_deadline_ns"),
        ("observed_ns", "scheduler_observed_ns"),
        ("listener_start_ns", "scheduler_listener_start_ns"),
        ("listener_end_ns", "scheduler_listener_end_ns"),
    )
    stamps: dict[str, int | None] = {}
    for source, target in fields:
        value = getattr(tick, source, None)
        stamps[target] = int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    def delta_ms(later: int | None, earlier: int | None) -> float | None:
        if later is None or earlier is None:
            return None
        return round((later - earlier) / 1_000_000.0, 3)

    result = {
        "scheduler_clock": "time.monotonic_ns",
        **stamps,
        "scheduler_enqueue_to_deadline_ms": delta_ms(
            stamps["scheduler_deadline_ns"], stamps["scheduler_enqueue_ns"]
        ),
        "scheduler_deadline_lateness_ms": delta_ms(
            stamps["scheduler_observed_ns"], stamps["scheduler_deadline_ns"]
        ),
        "scheduler_observed_to_listener_start_ms": delta_ms(
            stamps["scheduler_listener_start_ns"], stamps["scheduler_observed_ns"]
        ),
        "scheduler_listener_callback_ms": delta_ms(
            stamps["scheduler_listener_end_ns"], stamps["scheduler_listener_start_ns"]
        ),
    }

    def optional_stamp(name: str) -> int | None:
        value = getattr(tick, name, None)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    result.update({
        "scheduler_dispatch_start_ns": optional_stamp("scheduler_dispatch_start_ns"),
        "scheduler_dispatch_end_ns": optional_stamp("scheduler_dispatch_end_ns"),
        "scheduler_row_callback_entry_ns": optional_stamp(
            "microbatch_row_callback_entry_ns"
        ),
        "scheduler_timeline_request_ns": optional_stamp("microbatch_timeline_request_ns"),
        "scheduler_timeline_capture_lock_acquired_ns": optional_stamp(
            "microbatch_timeline_capture_lock_acquired_ns"
        ),
        "scheduler_timeline_model_lock_acquired_ns": optional_stamp(
            "microbatch_timeline_model_lock_acquired_ns"
        ),
        "scheduler_timeline_snapshot_ns": optional_stamp("microbatch_timeline_snapshot_ns"),
        "scheduler_timeline_age_ms": getattr(tick, "microbatch_timeline_age_ms", None),
        "scheduler_timeline_status": getattr(tick, "microbatch_timeline_status", None),
        "scheduler_timeline_reason": getattr(tick, "microbatch_timeline_reason", None),
        "scheduler_commit_microbatch_output_start_ns": optional_stamp(
            "microbatch_commit_start_ns"
        ),
        "scheduler_commit_microbatch_output_end_ns": optional_stamp(
            "microbatch_commit_end_ns"
        ),
        "scheduler_commit_state_lock_acquired_ns": optional_stamp(
            "microbatch_commit_state_lock_acquired_ns"
        ),
        "scheduler_commit_state_lock_released_ns": optional_stamp(
            "microbatch_commit_state_lock_released_ns"
        ),
        "scheduler_commit_state_lock_wait_ms": getattr(
            tick, "microbatch_commit_state_lock_wait_ms", None
        ),
        "scheduler_commit_state_lock_hold_ms": getattr(
            tick, "microbatch_commit_state_lock_hold_ms", None
        ),
        "scheduler_policy_decision_at_ns": optional_stamp("policy_decision_at_ns"),
        "scheduler_deadline_to_policy_decision_ms": delta_ms(
            optional_stamp("policy_decision_at_ns"), stamps["scheduler_deadline_ns"]
        ),
    })
    dispatch_mode = getattr(tick, "dispatch_mode", None)
    if isinstance(dispatch_mode, str) and dispatch_mode:
        result["scheduler_dispatch_mode"] = dispatch_mode
    batch_id = getattr(tick, "listener_batch_id", None)
    if isinstance(batch_id, str) and batch_id:
        result.update({
            "listener_batch_id": batch_id,
            "listener_batch_size": getattr(tick, "listener_batch_size", None),
            "listener_batch_start_ns": getattr(tick, "listener_batch_started_ns", None),
            "listener_batch_end_ns": getattr(tick, "listener_batch_finished_ns", None),
            "listener_batch_queue_wait_ms": getattr(tick, "microbatch_queue_wait_ms", None),
            "listener_batch_compute_ms": getattr(tick, "microbatch_compute_ms", None),
            "listener_batch_model_lock_wait_ms": getattr(
                tick, "microbatch_model_lock_wait_ms", None
            ),
            "listener_batch_model_lock_hold_ms": getattr(
                tick, "microbatch_model_lock_hold_ms", None
            ),
            "listener_batch_capacity_growth_ms": getattr(
                tick, "microbatch_capacity_growth_ms", None
            ),
            "listener_batch_frontend_encode_ms": getattr(
                tick, "microbatch_frontend_encode_ms", None
            ),
            "listener_batch_mel_prepare_ms": getattr(tick, "microbatch_mel_prepare_ms", None),
            "listener_batch_frontend_copy_ms": getattr(tick, "microbatch_frontend_copy_ms", None),
            "listener_batch_frontend_copy_cuda_ms": getattr(
                tick, "microbatch_frontend_copy_cuda_ms", None
            ),
            "listener_batch_audio_tower_lock_wait_ms": getattr(
                tick, "microbatch_audio_tower_lock_wait_ms", None
            ),
            "listener_batch_audio_tower_lock_hold_ms": getattr(
                tick, "microbatch_audio_tower_lock_hold_ms", None
            ),
            "listener_batch_audio_tower_workspace_lock_wait_ms": getattr(
                tick, "microbatch_audio_tower_workspace_lock_wait_ms", None
            ),
            "listener_batch_audio_tower_workspace_lock_hold_ms": getattr(
                tick, "microbatch_audio_tower_workspace_lock_hold_ms", None
            ),
            "listener_batch_audio_tower_row_copy_cuda_ms": getattr(
                tick, "microbatch_audio_tower_row_copy_cuda_ms", None
            ),
            "listener_batch_audio_tower_graph_submit_ms": getattr(
                tick, "microbatch_audio_tower_graph_submit_ms", None
            ),
            "listener_batch_audio_tower_cuda_ms": getattr(
                tick, "microbatch_audio_tower_cuda_ms", None
            ),
            "listener_batch_audio_tower_batch_cuda_span_ms": getattr(
                tick, "microbatch_audio_tower_batch_cuda_span_ms", None
            ),
            "listener_batch_audio_tower_execution_mode": getattr(
                tick, "microbatch_audio_tower_execution_mode", None
            ),
            "listener_batch_native_submit_ms": getattr(
                tick, "microbatch_native_submit_ms", None
            ),
            "listener_batch_native_cuda_ms": getattr(
                tick, "microbatch_native_cuda_ms", None
            ),
            "listener_batch_native_control6_raw_score_ms": getattr(
                tick, "microbatch_native_control6_raw_score_ms", None
            ),
            "listener_batch_decision_view_fork_count": getattr(
                tick, "microbatch_decision_view_fork_count", None
            ),
            "listener_batch_decision_view_replayed_q2_count": getattr(
                tick, "microbatch_decision_view_replayed_q2_count", None
            ),
            "listener_batch_decision_view_cache_hit": getattr(
                tick, "microbatch_decision_view_cache_hit", None
            ),
            "listener_batch_decision_view_ms": getattr(
                tick, "microbatch_decision_view_ms", None
            ),
            "listener_batch_decision_view_context_mode": getattr(
                tick, "microbatch_decision_view_context_mode", None
            ),
            "listener_batch_native_output_health_ms": getattr(
                tick, "microbatch_native_output_health_ms", None
            ),
            "listener_batch_native_output_health_checked": getattr(
                tick, "microbatch_native_output_health_checked", None
            ),
            "listener_batch_native_output_health_status": getattr(
                tick, "microbatch_native_output_health_status", None
            ),
            "listener_batch_native_output_health_nonfinite": getattr(
                tick, "microbatch_native_output_health_nonfinite", None
            ),
            "listener_batch_native_output_health_generation": getattr(
                tick, "microbatch_native_output_health_generation", None
            ),
            "listener_batch_native_output_health_session_generation": getattr(
                tick, "microbatch_native_output_health_session_generation", None
            ),
            "listener_batch_native_output_health_state_epoch_id": getattr(
                tick, "microbatch_native_output_health_state_epoch_id", None
            ),
            "listener_batch_native_output_health_row_frame_id": getattr(
                tick, "microbatch_native_output_health_row_frame_id", None
            ),
            "listener_batch_native_output_health_row_state_version": getattr(
                tick, "microbatch_native_output_health_row_state_version", None
            ),
            "listener_batch_native_output_health_row_sequence_position": getattr(
                tick, "microbatch_native_output_health_row_sequence_position", None
            ),
            "listener_batch_native_output_health_row_cache_position": getattr(
                tick, "microbatch_native_output_health_row_cache_position", None
            ),
            "listener_batch_native_output_health_native_context_id": getattr(
                tick, "microbatch_native_output_health_native_context_id", None
            ),
            "listener_batch_semantic_prefix_request_ms": getattr(
                tick, "microbatch_semantic_prefix_request_ms", None
            ),
            "listener_batch_semantic_prefix_build_ms": getattr(
                tick, "microbatch_semantic_prefix_build_ms", None
            ),
            "listener_batch_semantic_prefix_replay_ms": getattr(
                tick, "microbatch_semantic_prefix_replay_ms", None
            ),
            "listener_batch_semantic_prefix_import_ms": getattr(
                tick, "microbatch_semantic_prefix_import_ms", None
            ),
            "listener_batch_semantic_prefix_refresh_event": getattr(
                tick, "microbatch_semantic_prefix_refresh_event", None
            ),
            "listener_batch_semantic_prefix_refresh_event_id": getattr(
                tick, "microbatch_semantic_prefix_refresh_event_id", None
            ),
            "listener_batch_semantic_prefix_refresh_revision": getattr(
                tick, "microbatch_semantic_prefix_refresh_revision", None
            ),
            "listener_batch_semantic_prefix_len": getattr(
                tick, "microbatch_semantic_prefix_len", None
            ),
            "listener_batch_control6_policy_ms": getattr(
                tick, "microbatch_control6_policy_ms", None
            ),
            "listener_batch_cuda_completion_wait_ms": getattr(
                tick, "microbatch_cuda_completion_wait_ms", None
            ),
            "listener_batch_error": getattr(tick, "microbatch_error", None),
        })
        if include_host_lock_timeline:
            timeline = getattr(tick, "microbatch_host_lock_timeline", None)
            if isinstance(timeline, dict):
                result["listener_batch_host_lock_timeline"] = timeline
    return result


def native_q4_cuda_profile_host_context(session_id: str, frame: Any, tick: Any) -> dict[str, Any]:
    """Capture existing host-monotonic frame/tick stamps for a sampled native Q4 join."""
    def stamp(name: str) -> int | None:
        value = getattr(tick, name, None)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    return {
        "session_id": str(session_id),
        "frame_id": int(frame.frame_id),
        "capture_start_ns": int(frame.capture_start_ns),
        "capture_end_ns": int(frame.capture_end_ns),
        "enqueue_ns": int(frame.enqueue_ns),
        "deadline_ns": stamp("deadline_ns"),
        "observed_ns": stamp("observed_ns"),
        "scheduler_listener_start_ns": stamp("listener_start_ns"),
        "scheduler_listener_end_ns": stamp("listener_end_ns"),
        "listener_step_start_ns": None,
        "listener_step_end_ns": None,
        "listener_callback_end_ns": None,
        "shared_lm_lock_wait_ms": None,
        "shared_lm_lock_host_hold_ms": None,
        "clock": "time.monotonic_ns",
    }


def attach_native_q4_lock_metrics(
    host_context: dict[str, Any], listener_state: Any,
) -> None:
    """Attach already-measured listener lock durations to its sampled host row."""
    host_context["shared_lm_lock_wait_ms"] = getattr(
        listener_state, "last_lock_wait_ms", None
    )
    host_context["shared_lm_lock_host_hold_ms"] = getattr(
        listener_state, "last_lock_hold_ms", None
    )

_FRAME_RESIDUAL_RUNTIME_METRIC_TYPES = {
    "frame_residual_adapter": str,
    "adapter_mode": str,
    "frame_residual_checkpoint": str,
    "frame_residual_calls": int,
    "frame_residual_tail_hold_calls": int,
    "frame_residual_qwen_q1_only": bool,
    "frame_residual_prefix_tokens_added": int,
    "frame_residual_learned_scale": float,
    "frame_residual_delta_sq_sum": float,
    "frame_residual_base_sq_sum": float,
    "frame_residual_rms": float,
    "frame_residual_base_rms": float,
    "frame_residual_rms_to_base_ratio": float,
    "frame_residual_live_first_last_hidden_cosine": float,
    "qwen_route": str,
    "qwen_prefix_slots_added": int,
    "qwen_semantic_text_unchanged": bool,
    "source": str,
}

_FRAME_RESIDUAL_LIVE_TRACE_FIELDS = {
    "qwen_frame_step",
    "gemma_token_id",
    "gemma_kv_position",
    "qwen_token_id",
    "qwen_packet_token_index",
    "qwen_packet_token_count",
    "tail_hold",
    "gemma_hidden_norm",
    "gemma_hidden_cosine_previous_packet",
}


def _filter_frame_residual_live_hidden_trace(value: Any) -> list[dict[str, Any]]:
    """Keep bounded scalar provenance for live Q1 steps; never expose tensors."""
    if not isinstance(value, list):
        return []
    retained: list[dict[str, Any]] = []
    integer_fields = {
        "qwen_frame_step", "qwen_token_id",
    }
    optional_integer_fields = {
        "gemma_token_id", "gemma_kv_position",
        "qwen_packet_token_index", "qwen_packet_token_count",
    }
    for entry in value[:256]:
        if not isinstance(entry, dict) or set(entry) != _FRAME_RESIDUAL_LIVE_TRACE_FIELDS:
            continue
        if type(entry.get("tail_hold")) is not bool:
            continue
        if any(
            type(entry.get(key)) is not int or entry[key] < 0
            for key in integer_fields
        ):
            continue
        if any(
            entry.get(key) is not None
            and (type(entry[key]) is not int or entry[key] < 0)
            for key in optional_integer_fields
        ):
            continue
        norm = entry.get("gemma_hidden_norm")
        cosine = entry.get("gemma_hidden_cosine_previous_packet")
        if type(norm) is not float or not np.isfinite(norm) or norm < 0:
            continue
        if cosine is not None and (
            type(cosine) is not float or not np.isfinite(cosine) or not -1.000001 <= cosine <= 1.000001
        ):
            continue
        clean = {key: entry[key] for key in _FRAME_RESIDUAL_LIVE_TRACE_FIELDS}
        # A first packet has no previous packet for a cosine comparison.
        clean["gemma_hidden_cosine_previous_packet"] = cosine
        retained.append(clean)
    return retained


def _filter_turn_local_speech_runtime_metrics(values: dict[str, Any]) -> dict[str, Any]:
    """Expose only bounded FIFO, token pacing and PCM queue evidence."""
    retained: dict[str, Any] = {}
    integer_limits = {
        "turn_local_fifo_capacity": (4, 4),
        "turn_local_fifo_remaining_packets": (0, 4),
        "turn_local_fifo_peak_packets": (0, 4),
        "turn_local_last_vocalized_kv_position": (0, 2**63 - 1),
        "realtime_pcm_packet_queue_capacity": (4, 4),
        "speaker_playout_buffer_max_frames": (1, 4096),
        "native_q1_import_audio_frame_capacity": (1, 4096),
        "native_q1_live_frame_cap": (1, 4096),
    }
    for key, (minimum, maximum) in integer_limits.items():
        value = values.get(key)
        if type(value) is int and minimum <= value <= maximum:
            retained[key] = value
    playback_ack = values.get("turn_local_vocalized_cursor_is_playback_ack")
    if type(playback_ack) is bool and playback_ack is False:
        retained["turn_local_vocalized_cursor_is_playback_ack"] = False

    pacer = values.get("live_gemma_token_pacer")
    pacer_fields = {"tick_ms", "max_tokens_per_tick", "total_tokens", "peak_tokens_in_tick"}
    if isinstance(pacer, dict) and set(pacer) == pacer_fields:
        if all(type(pacer.get(key)) is int for key in pacer_fields):
            if (
                pacer["tick_ms"] == 80
                and pacer["max_tokens_per_tick"] == 2
                and pacer["total_tokens"] >= 0
                and 0 <= pacer["peak_tokens_in_tick"] <= 2
            ):
                retained["live_gemma_token_pacer"] = dict(pacer)
    return retained


def _format_turn_local_speech_summary(
    session_prefix: str,
    values: dict[str, Any],
) -> str:
    """Format one log row from turn-local metrics without retaining live state."""
    pacer = values.get("live_gemma_token_pacer")
    trace = values.get("frame_residual_live_hidden_trace")
    return (
        "[turn-local speech summary] session={} fifo_capacity={} fifo_peak={} "
        "fifo_remaining={} last_vocalized_kv={} pacer_tokens={} "
        "pacer_peak_per_tick={} tick_ms={} max_per_tick={} pcm_queue_capacity={} "
        "q1_frames={} tail_hold_frames={} live_frame_cap={} hidden_trace_entries={} "
        "first_last_hidden_cosine={}"
    ).format(
        str(session_prefix),
        values.get("turn_local_fifo_capacity"),
        values.get("turn_local_fifo_peak_packets"),
        values.get("turn_local_fifo_remaining_packets"),
        values.get("turn_local_last_vocalized_kv_position"),
        pacer.get("total_tokens") if isinstance(pacer, dict) else None,
        pacer.get("peak_tokens_in_tick") if isinstance(pacer, dict) else None,
        pacer.get("tick_ms") if isinstance(pacer, dict) else None,
        pacer.get("max_tokens_per_tick") if isinstance(pacer, dict) else None,
        values.get("realtime_pcm_packet_queue_capacity"),
        values.get("frame_residual_calls"),
        values.get("frame_residual_tail_hold_calls"),
        values.get("native_q1_live_frame_cap"),
        len(trace) if isinstance(trace, list) else 0,
        values.get("frame_residual_live_first_last_hidden_cosine"),
    )


def _submit_or_defer_audio_history_archive(
    *,
    defer_offline: bool,
    archive_id: int,
    response_id: str,
    session_id: str,
    snapshot: dict[str, Any],
    deferred_at_monotonic_ns: int,
    worker_submit: Callable[[np.ndarray], Any],
    update_metrics: Callable[[str, str, dict[str, Any]], None],
) -> tuple[Any | None, bool]:
    """Submit archive ASR or retain the reservation for offline processing."""
    if not defer_offline:
        return worker_submit(snapshot["pcm"]), False
    update_metrics(response_id, "audio_history_archive", {
        "status": "deferred_offline",
        "archive_id": int(archive_id),
        "session_id": str(session_id),
        "source": snapshot.get("source"),
        "response_created_at_monotonic_ns": snapshot.get("response_created_at_monotonic_ns"),
        "reserved_at_monotonic_ns": snapshot.get("reserved_at_monotonic_ns"),
        "deferred_at_monotonic_ns": int(deferred_at_monotonic_ns),
        "input_first_frame_id": snapshot.get("first_frame_id"),
        "input_last_frame_id": snapshot.get("last_frame_id"),
        "input_audio_samples": int(snapshot["pcm"].size),
        "sample_rate": snapshot.get("sample_rate"),
        "transcript_committed": False,
    })
    return None, True


def _filter_frame_residual_runtime_metrics(values: dict[str, Any]) -> dict[str, Any]:
    """Retain only typed, finite adapter/Q1 observability fields."""
    retained: dict[str, Any] = {}
    for key, value in values.items():
        expected_type = _FRAME_RESIDUAL_RUNTIME_METRIC_TYPES.get(key)
        if expected_type is None or type(value) is not expected_type:
            continue
        if expected_type is float and not np.isfinite(value):
            continue
        if key == "frame_residual_live_first_last_hidden_cosine" and not -1.000001 <= value <= 1.000001:
            continue
        retained[key] = value
    trace = _filter_frame_residual_live_hidden_trace(
        values.get("frame_residual_live_hidden_trace")
    )
    if trace:
        retained["frame_residual_live_hidden_trace"] = trace
    return retained


def _filter_native_thinker_lock_metrics(values: Any) -> dict[str, Any]:
    """Expose explicit lock provenance for the unified native Thinker route."""
    if not isinstance(values, dict):
        return {}
    retained: dict[str, Any] = {}
    wait_ms = values.get("thinker_decode_lock_wait_ms")
    if type(wait_ms) in {int, float} and np.isfinite(wait_ms) and wait_ms >= 0:
        retained["thinker_decode_lock_wait_ms"] = float(wait_ms)
    acquire_count = values.get("thinker_decode_lock_acquire_count")
    if type(acquire_count) is int and acquire_count >= 0:
        retained["thinker_decode_lock_acquire_count"] = acquire_count
    provenance = values.get("thinker_decode_lock_wait_provenance")
    if provenance == "native_no_python_decode_lock":
        retained["thinker_decode_lock_wait_provenance"] = provenance
    return retained


def _retain_archive_transcript_commitment(
    measured: dict[str, Any],
    *,
    phase: str,
    values: Mapping[str, Any],
) -> None:
    """Preserve the archive deferral boolean through response-metric filtering."""
    committed = values.get("transcript_committed")
    if phase == "audio_history_archive" and type(committed) is bool:
        measured["transcript_committed"] = committed


def native_listener_cuda_profile_effective_directory(directory: str | Path | None) -> str | None:
    """Return the canonical profile output path used by health and sidecar writes."""
    if directory is None or str(directory) == "":
        return None
    return str(Path(directory).expanduser().resolve())


async def _quiesce_scheduler_for_native_profile(scheduler: Any) -> dict[str, Any]:
    """Stop a profiled session scheduler off-loop before taking close snapshots."""
    stop_error = None
    try:
        await asyncio.to_thread(
            scheduler.stop,
            timeout=NATIVE_LISTENER_PROFILE_STOP_TIMEOUT_S,
        )
    except Exception as exc:
        stop_error = f"{type(exc).__name__}: {exc}"
    callbacks_active = bool(getattr(scheduler, "callbacks_active", False))
    scheduler_running = bool(getattr(scheduler, "is_running", False))
    if stop_error is not None:
        status = "incomplete_scheduler_stop_error"
    elif callbacks_active:
        status = "incomplete_callbacks_active"
    elif scheduler_running:
        status = "incomplete_scheduler_active"
    else:
        status = "complete"
    return {
        "status": status,
        "stop_timeout_s": NATIVE_LISTENER_PROFILE_STOP_TIMEOUT_S,
        "scheduler_running": scheduler_running,
        "callbacks_active": callbacks_active,
        "stop_error": stop_error,
        "measurement": "host_scheduler_lifecycle_state",
    }


def write_native_listener_cuda_profile_sidecar(
    *,
    session_id: str,
    profile: Mapping[str, Any],
    host_tick_timings: list[dict[str, Any]],
    model_lock_timeline: Mapping[str, Any] | None = None,
    directory: str | Path,
) -> str | None:
    """Atomically persist bounded native Q4 samples with their host frame join rows."""
    if profile.get("enabled") is not True:
        return None
    effective_directory = native_listener_cuda_profile_effective_directory(directory)
    if effective_directory is None:
        return None
    target_directory = Path(effective_directory)
    target_directory.mkdir(parents=True, exist_ok=True)
    session_key = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:24]
    target = target_directory / f"native_q4_cuda_profile_{session_key}.json"
    payload = {
        "schema_version": 1,
        "session_id": str(session_id),
        "profile": dict(profile),
        "host_tick_timings": [dict(row) for row in host_tick_timings if isinstance(row, dict)],
        "sampled_host_tick_timings": [
            {
                "session_id": str(session_id),
                **dict(sample["host_context"]),
            }
            for sample in [
                *profile.get("samples", []),
                *profile.get("pending_samples", []),
            ]
            if isinstance(sample, dict)
            and isinstance(sample.get("host_context"), dict)
            and sample["host_context"].get("session_id") == str(session_id)
        ],
        "join_key": ["session_id", "frame_id"],
    }
    if model_lock_timeline is not None:
        payload["model_lock_timeline"] = dict(model_lock_timeline)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target_directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        try:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        except OSError:
            logger.exception("native listener CUDA profile temporary file cleanup failed")
    return str(target)


def _filter_native_talker_q1_graph_runtime(values: Any) -> dict[str, Any]:
    """Keep only bounded scalar graph evidence; never serialize device pointers or tensors."""
    if not isinstance(values, dict):
        return {}
    retained: dict[str, Any] = {}
    bool_fields = {
        "enabled", "captured", "device_cursor", "stream_handle_nonnull",
    }
    int_fields = {
        "capture_count", "replays", "startup_validation_replays", "serving_replays",
        "abi_version", "device_cursor_actual", "device_cursor_error",
        "cache_capacity", "max_audio_frames",
    }
    for key in bool_fields:
        value = values.get(key)
        if type(value) is bool:
            retained[key] = value
    for key in int_fields:
        value = values.get(key)
        if type(value) is int and 0 <= value <= 10_000_000:
            retained[key] = value
    shape = values.get("input_shape")
    if (
        isinstance(shape, list)
        and len(shape) == 3
        and all(type(item) is int and 0 < item <= 131_072 for item in shape)
    ):
        retained["input_shape"] = list(shape)
    dtype = values.get("input_dtype")
    if isinstance(dtype, str) and len(dtype) <= 32:
        retained["input_dtype"] = dtype
    digest = values.get("library_sha256")
    if isinstance(digest, str) and len(digest) == 64 and all(
        char in "0123456789abcdefABCDEF" for char in digest
    ):
        retained["library_sha256"] = digest.lower()
    for key in ("failure", "reason"):
        value = values.get(key)
        if isinstance(value, str) and len(value) <= 200:
            retained[key] = value
    return retained


def _response_admission_allowed_locked(
    session: Any, *, automatic: bool, closing: bool
) -> bool:
    """Check response admission while the caller holds session._generation_lock."""
    if closing:
        return False
    if not automatic or bool(session.auto_response):
        return True
    session.policy_response_pending = False
    if session.control_state in {ControlState.SPEAKING, ControlState.BACKCHANNELING}:
        session.control_state = ControlState.LISTENING
    session.speaker_segment_id = None
    session.speaker_segment = None
    return False


def _empty_controller_say_think_from_eos(
    exc: BaseException,
    *,
    prompt_len: int,
) -> dict[str, Any] | None:
    """Convert a natural empty EOS into the controller's no-speech result."""
    if (
        getattr(exc, "empty_eos", False) is not True
        or getattr(exc, "terminal_reason", None) != "eos"
    ):
        return None
    runtime_metrics = getattr(exc, "runtime_metrics", {})
    return {
        "text": "",
        "gen_ids": [],
        "hidden": None,
        "hidden_prompt_len": int(prompt_len),
        "prompt_len": int(prompt_len),
        "runtime_metrics": dict(runtime_metrics) if isinstance(runtime_metrics, dict) else {},
        "terminal_reason": "eos",
        "cancelled": False,
    }


def _controller_guided_say_generation_options(
    *,
    action: Any,
    controller_guided_say: bool,
    direct_audio_candidate: bool,
    response_protocol: str | None,
) -> dict[str, int]:
    """Return the production request budgets for the selected response route."""
    action_value = getattr(action, "value", action)
    normalized_action = str(action_value).strip().upper() if action_value is not None else None
    if normalized_action == ControlAction.BACKCHANNEL.value:
        return {"max_tokens": 2, "_min_new_tokens": 0}
    is_vocal_action = normalized_action in {ControlAction.SPEAK.value, ControlAction.BACKCHANNEL.value}
    if direct_audio_candidate and response_protocol == AUDIO_RESPONSE_PROTOCOL_V2:
        max_tokens = AUDIO_RESPONSE_V2_MAX_NEW_TOKENS
    elif controller_guided_say and is_vocal_action:
        max_tokens = 256
    elif direct_audio_candidate:
        max_tokens = 64
    elif normalized_action == ControlAction.BACKCHANNEL.value:
        max_tokens = 16
    else:
        max_tokens = 64
    min_tokens = 0
    if controller_guided_say and is_vocal_action:
        min_tokens = 1
    elif not direct_audio_candidate and normalized_action == ControlAction.BACKCHANNEL.value:
        min_tokens = 1
    return {"max_tokens": int(max_tokens), "_min_new_tokens": int(min_tokens)}


def _response_uses_direct_audio_candidate(
    *,
    candidate_enabled: bool,
    committed_audio: np.ndarray,
    intent_origin: dict[str, Any] | None,
) -> bool:
    """Select direct-audio inference only for a response with an audio origin."""
    return bool(
        candidate_enabled
        and (intent_origin is not None or np.asarray(committed_audio).size > 0)
    )


def _require_direct_audio_context(context: dict[str, Any] | None) -> dict[str, Any]:
    """Fail closed when an audio-triggered response lost its causal PCM snapshot."""
    if not isinstance(context, dict) or context.get("_direct_audio_pcm") is None:
        raise RuntimeError("direct_audio_context_missing: snapshot PCM causale non disponibile")
    return context


MODEL_ID = "gemma-4-omni-duplex"
REALTIME_SCHEDULER_EVENT_CAPACITY = 16
REALTIME_AUDIO_PACKET_CAPACITY = 4
REALTIME_TEXT_CHUNK_CAPACITY = 32
EMPTY_ASR_SPEAK_COOLDOWN_FRAMES = 25
BACKCHANNEL_MAX_VOICE_AGE_FRAMES = 10
POLICY_FRAME_MS = 80
POLICY_FRAME_SAMPLES = USER_SAMPLE_RATE * POLICY_FRAME_MS // 1000
DEFAULT_SPEAK_ENDPOINT_QUIET_MS = 800
NATIVE_PARTIAL_ASR_ENV = "DUPLEX_NATIVE_PARTIAL_ASR"
DUPLEX_CONTROLLER_GUIDED_SAY_ENV = CONTROLLER_GUIDED_SAY_COMPAT_ENV
_ACTION_TOKEN_MAP = {
    "SPEAK": "<unused90>",
    "BACKCHANNEL": "<unused93>",
    "REPLAN": "<unused94>",
}


def _native_control6_policy_hidden_for_tick(
    *, user_pcm: Any, speaker_active: bool, listener_hidden: Any | None
) -> tuple[Any | None, bool]:
    """Treat exact-zero ingress during active playback as CONTINUE-only input.

    The listener hidden remains available to the Qwen conditioning buffer; this
    filter only prevents the policy from interpreting synthetic/empty ingress
    as a fresh reason to replan its current response.
    """
    pcm = np.asarray(user_pcm)
    no_input_tick = bool(
        speaker_active
        and pcm.shape == (POLICY_FRAME_SAMPLES,)
        and np.count_nonzero(pcm) == 0
    )
    return (None if no_input_tick else listener_hidden), no_input_tick


def _native_control6_speaker_active(
    *,
    active_response: bool,
    control_state: ControlState,
    pending_pcm_playback: bool = False,
) -> bool:
    """Resolve the one native Control6 speaking state for the whole policy tick."""
    return bool(
        active_response
        or pending_pcm_playback
        or control_state in {
            ControlState.SPEAKING,
            ControlState.BACKCHANNELING,
            ControlState.STOPPING,
        }
    )


def _apply_native_control6_pending_pcm_yield_gate(
    speaker_gate: SpeakerGate,
    decision: PolicyDecision,
    session: Any,
    pending_pcm_branches: Sequence[Mapping[str, Any]],
) -> GateEvent:
    """Reconcile stale LISTENING state with a matching, still-pending PCM branch.

    The registry branch is authoritative for physical playback. Only a native
    YIELD for the current session generation and speaker segment may repair a
    stale LISTENING control state; the ordinary SpeakerGate still performs the
    transition and its stop/segment-close side effects.
    """
    lock = getattr(session, "_generation_lock", None)
    if (
        decision.action != ControlAction.YIELD
        or lock is None
        or not isinstance(getattr(session, "control_state", None), ControlState)
        or session.control_state != ControlState.LISTENING
    ):
        return speaker_gate.apply(decision, session)

    with lock:
        current_generation = int(getattr(session, "session_generation_id", -1))
        current_segment = getattr(session, "speaker_segment_id", None)
        decision_generation = decision.generation_id
        has_matching_pending_pcm = any(
            bool(str(branch.get("response_id", "")))
            and branch.get("session_generation_id") == current_generation
            and branch.get("speaker_segment_id") == current_segment
            and _response_branch_has_pending_pcm(branch)
            for branch in pending_pcm_branches
        )
        if (
            decision_generation == current_generation
            and session.control_state == ControlState.LISTENING
            and has_matching_pending_pcm
        ):
            session.control_state = ControlState.SPEAKING
        return speaker_gate.apply(decision, session)


def _response_branch_has_pending_pcm(
    branch: Mapping[str, Any] | None,
    *,
    speaker_segment_id: str | None = None,
) -> bool:
    """Whether emitted response PCM lacks a complete client-consumption ACK.

    The response/task lifecycle can finish before playback. Keep that output
    physically active until every emitted sample range is ACKed or explicitly
    cancelled; queue ownership and `finalized` do not erase audible PCM.
    """
    if not isinstance(branch, Mapping):
        return False
    lock = branch.get("spoken_prefix_lock")
    if lock is None:
        return _response_branch_has_pending_pcm_locked(
            branch,
            speaker_segment_id=speaker_segment_id,
        )
    with lock:
        return _response_branch_has_pending_pcm_locked(
            branch,
            speaker_segment_id=speaker_segment_id,
        )


def _response_branch_has_pending_pcm_locked(
    branch: Mapping[str, Any],
    *,
    speaker_segment_id: str | None = None,
) -> bool:
    if branch.get("playback_cancelled") is True:
        return False
    segments = branch.get("playback_segments")
    if not isinstance(segments, Mapping):
        return False
    consumed = set(branch.get("client_consumed_segments", ()))
    for segment_id, segment in segments.items():
        if not isinstance(segment, Mapping) or str(segment_id) in consumed:
            continue
        if (
            speaker_segment_id is not None
            and segment.get("speaker_segment_id") != speaker_segment_id
        ):
            continue
        start = segment.get("start_sample")
        emitted_end = segment.get("emitted_end_sample")
        if (
            isinstance(start, int) and not isinstance(start, bool)
            and isinstance(emitted_end, int) and not isinstance(emitted_end, bool)
            and emitted_end > start
        ):
            return True
    return False


def _pending_pcm_response_branches(
    branches: Mapping[str, Any] | None,
    *,
    registry_lock: Any | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return registered response branches with emitted, unconsumed PCM."""
    if not isinstance(branches, Mapping):
        return ()
    if registry_lock is None:
        snapshot = tuple(branches.values())
    else:
        with registry_lock:
            snapshot = tuple(branches.values())
    return tuple(
        branch
        for branch in snapshot
        if isinstance(branch, dict) and _response_branch_has_pending_pcm(branch)
    )


def _cancel_response_branch_playback(branch: dict[str, Any]) -> None:
    """Mark a branch cancelled and wake any finalizer waiting on its PCM ACK."""
    def wake() -> None:
        callback = branch.get("playback_ack_wakeup")
        if callable(callback):
            callback()
            return
        event = branch.get("playback_ack_event")
        if isinstance(event, asyncio.Event):
            event.set()

    lock = branch.get("spoken_prefix_lock")
    if lock is None:
        branch["playback_cancelled"] = True
        wake()
        return
    with lock:
        branch["playback_cancelled"] = True
    wake()


async def _wait_for_response_playback_drain(branch: dict[str, Any]) -> None:
    """Wait without holding branch/transition locks until ACK or cancellation."""
    event = branch.get("playback_ack_event")
    if not isinstance(event, asyncio.Event):
        return
    lock = branch.get("spoken_prefix_lock")
    while True:
        if lock is None:
            event.clear()
            if not _response_branch_has_pending_pcm(branch):
                return
        else:
            with lock:
                event.clear()
                if not _response_branch_has_pending_pcm_locked(branch):
                    return
        await event.wait()


def _strict_env_switch(name: str, *, default: bool = False) -> bool:
    """Read one explicit 0/1 switch without silently accepting typos."""
    value = os.environ.get(name)
    if value is None:
        return default
    if value == "0":
        return False
    if value == "1":
        return True
    raise RuntimeError(f"{name} must be exactly '0' or '1'")


def _listener_frame_is_voiced(pcm: Any) -> bool:
    """Match the realtime endpoint tracker's causal RMS/peak thresholds."""
    frame = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if not frame.size or not np.isfinite(frame).all():
        return False
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    peak = float(np.max(np.abs(frame)))
    return rms >= 0.001 and peak >= 0.005


def _unified_listener_frame_contract(
    pcm: Any,
    *,
    assistant_audio_queue_open: bool,
    user_input_pending: bool,
) -> tuple[int, str | None, bool]:
    """Return logical-Q, ephemeral reason, and next-turn audio queue policy."""
    if assistant_audio_queue_open:
        return 0, "assistant_turn_open", True
    if not user_input_pending and not _listener_frame_is_voiced(pcm):
        return 0, "prespeech_silence", False
    return 2, None, False


def _unified_response_owns_shared_audio_queue(
    response_branch: Mapping[str, Any] | None,
) -> bool:
    """Q0 belongs to the current shared response branch once its KV queue is open."""
    return bool(
        isinstance(response_branch, Mapping)
        and response_branch.get("queue_open") is True
        and response_branch.get("ephemeral_private") is not True
    )


def _shared_listener_response_branch(
    binding: Mapping[str, Any] | None,
    response_id: Any,
) -> Mapping[str, Any] | None:
    """Resolve playback state from the session binding shared with the listener callback."""
    if not isinstance(binding, Mapping) or response_id is None:
        return None
    branches = binding.get("playback_ack_branches")
    if not isinstance(branches, Mapping):
        return None
    branch = branches.get(str(response_id))
    return branch if isinstance(branch, Mapping) else None


def _shared_listener_frame_queue_contract(
    binding: Mapping[str, Any] | None,
    pcm: Any,
    *,
    response_id: Any,
    user_input_pending: bool,
) -> tuple[int, str | None, bool]:
    """Apply Q/queue ownership from the current response branch, including prompt prefill."""
    response_branch = _shared_listener_response_branch(binding, response_id)
    assistant_audio_queue_open = _unified_response_owns_shared_audio_queue(
        response_branch,
    )
    return _unified_listener_frame_contract(
        pcm,
        assistant_audio_queue_open=assistant_audio_queue_open,
        user_input_pending=user_input_pending,
    )


def _clear_shared_listener_response_branches(
    binding: dict[str, Any] | None,
    branches: dict[str, dict[str, Any]],
) -> None:
    """Detach only the registry owned by this websocket from its session binding."""
    if binding is not None and binding.get("playback_ack_branches") is branches:
        binding.pop("playback_ack_branches", None)
    branches.clear()


def _attach_shared_listener_response_branches(
    binding: dict[str, Any] | None,
    branches: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Attach the websocket-owned branch registry after session registration."""
    if binding is None:
        raise RuntimeError("shared listener response registry requires a registered session")
    binding["playback_ack_branches"] = branches
    return binding


def _validate_private_transition_chunk(
    transition_token_ids: Sequence[int],
    current_offset: int,
    token_ids: Sequence[int],
    *,
    source_cache_position: int,
    current_cache_position: int,
) -> int:
    """Validate one logical Q<=8 prefill chunk against the exact prompt and private cursor."""
    prompt_ids = tuple(int(value) for value in transition_token_ids)
    chunk = tuple(int(value) for value in token_ids)
    offset = int(current_offset)
    if not chunk or len(chunk) > 8 or offset < 0 or offset + len(chunk) > len(prompt_ids):
        raise RuntimeError("private transition chunk is outside the original Thinker transition")
    if int(current_cache_position) != int(source_cache_position) + offset:
        raise RuntimeError("private transition cursor differs from its contiguous Q2 prompt offset")
    if chunk != prompt_ids[offset:offset + len(chunk)]:
        raise RuntimeError("private transition chunk differs from the original Thinker transition")
    return offset + len(chunk)


def _native_control6_endpoint_bypass_enabled(policy: Any) -> bool:
    """Only native-row Control6 owns the early endpoint bypass path."""
    return isinstance(policy, GemmaNativeControl6Policy)


def _native_control6_decision_has_valid_input(
    *,
    fresh_hidden: bool,
    no_input_tick: bool,
    speaker_active: bool,
    action: Any,
    legal_actions: Any,
) -> bool:
    """Allow CONTINUE without a hidden only for exact-zero active playback ticks."""
    if fresh_hidden:
        return True
    action_name = str(getattr(action, "value", action)).strip().upper()
    legal = {str(getattr(item, "value", item)).strip().upper() for item in (legal_actions or ())}
    return bool(
        no_input_tick
        and speaker_active
        and action_name == ControlAction.CONTINUE.value
        and action_name in legal
    )


def _native_control6_listener_source_frame_id(listener_output: Any | None) -> int | None:
    """Return a validated listener source frame without trusting stale output fields."""
    if listener_output is None:
        return None
    source_frame_id = getattr(listener_output, "source_frame_id", None)
    if source_frame_id is None:
        source_frame_id = getattr(listener_output, "frame_id", None)
    if isinstance(source_frame_id, bool) or not isinstance(source_frame_id, (int, np.integer)):
        return None
    source_frame_id = int(source_frame_id)
    return source_frame_id if source_frame_id >= 0 else None


def _native_control6_listener_output_is_fresh_for_frame(
    listener_output: Any | None,
    frame_id: int,
    *,
    source_frame_id: int | None = None,
) -> bool:
    """Require both output and source frame identities to match the policy tick."""
    if listener_output is None or source_frame_id is None:
        return False
    output_frame_id = getattr(listener_output, "frame_id", None)
    if isinstance(output_frame_id, bool) or not isinstance(output_frame_id, (int, np.integer)):
        return False
    return bool(
        int(output_frame_id) == int(frame_id)
        and source_frame_id == int(frame_id)
        and getattr(listener_output, "fresh", False)
    )


def _normalize_chat_modalities(value: Any) -> list[str]:
    """Return the API default and reject malformed modality arrays safely."""
    modalities = ["text", "audio"] if value is None else value
    if (
        not isinstance(modalities, list)
        or not modalities
        or any(not isinstance(item, str) or item not in {"text", "audio"} for item in modalities)
    ):
        raise ValueError("modalities deve contenere text e/o audio")
    return list(dict.fromkeys(modalities))


def _verify_v9_training_reference(conditioning: dict[str, Any]) -> Path:
    """Verify both the pinned metadata and the reference WAV bytes on disk."""
    reference_path = Path(str(conditioning.get("reference_wav_path") or "")).expanduser()
    if not reference_path.is_file():
        raise ValueError("il reference voice training V9 non è disponibile")
    declared_sha256 = str(conditioning.get("reference_wav_sha256") or "").lower()
    digest = hashlib.sha256()
    try:
        with reference_path.open("rb") as reference_file:
            for block in iter(lambda: reference_file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError("impossibile leggere il reference voice training V9") from exc
    actual_sha256 = digest.hexdigest()
    if (
        declared_sha256 != V9_TRAINING_REFERENCE_SHA256
        or actual_sha256 != V9_TRAINING_REFERENCE_SHA256
    ):
        raise ValueError("lo SHA del reference voice non corrisponde al training V9")
    return reference_path
RESPONSE_QUEUE_BACKPRESSURE_TIMEOUT_S = 2.0
RESPONSE_QUEUE_BACKPRESSURE_RETRY_S = 0.005


class ResponseOutputQueueFull(queue.Full):
    """A bounded response queue remained full until its worker deadline."""

    def __init__(self, *, queue_kind, burst_items, wait_ms, attempts):
        super().__init__(f"{queue_kind} response output queue full after {wait_ms:.1f} ms")
        self.queue_kind = queue_kind
        self.burst_items = int(burst_items)
        self.wait_ms = float(wait_ms)
        self.attempts = int(attempts)


def _put_response_queue_bounded(
    try_put, *, queue_kind, burst_items,
    timeout_s=RESPONSE_QUEUE_BACKPRESSURE_TIMEOUT_S,
    retry_s=RESPONSE_QUEUE_BACKPRESSURE_RETRY_S,
    cancel_reason="generation_fenced",
):
    """Retry a nonblocking producer while the event loop drains the bounded queue."""
    if queue_kind not in {"audio", "text"}:
        raise ValueError("queue_kind must be audio or text")
    if burst_items < 0 or timeout_s < 0 or retry_s <= 0:
        raise ValueError("invalid bounded response queue limits")
    started = time.monotonic()
    deadline = started + float(timeout_s)
    attempts = 0
    while True:
        attempts += 1
        outcome = try_put()
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if outcome is True:
            return {
                "status": "enqueued", "queue_kind": queue_kind,
                "burst_items": int(burst_items), "wait_ms": round(elapsed_ms, 3),
                "attempts": attempts,
            }
        if outcome is False:
            return {
                "status": "cancelled", "queue_kind": queue_kind,
                "burst_items": int(burst_items), "wait_ms": round(elapsed_ms, 3),
                "attempts": attempts, "cancel_reason": str(cancel_reason),
            }
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ResponseOutputQueueFull(
                queue_kind=queue_kind, burst_items=burst_items,
                wait_ms=(time.monotonic() - started) * 1000.0, attempts=attempts,
            )
        time.sleep(min(float(retry_s), remaining))




def _configured_audio_response_protocol(engine: Any) -> str | None:
    del engine
    return resolve_audio_response_protocol()


def _require_unified_stable_audio_protocol(protocol: Any) -> str:
    """Unified KV requires a prompt contract that stays stable after import."""
    normalized = normalize_audio_response_protocol(protocol)
    if normalized not in {AUDIO_RESPONSE_PROTOCOL_V2, CONTROLLER_GUIDED_SAY_V1}:
        raise RuntimeError(
            "DUPLEX_UNIFIED_KV requires a supported stable audio response protocol"
        )
    return normalized


def _tokenize_template_text(tokenizer: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise RuntimeError("unified prompt requires the loaded Gemma chat template")
    if not callable(getattr(tokenizer, "encode", None)):
        raise RuntimeError("unified prompt requires tokenizer.encode")
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=bool(add_generation_prompt),
    )
    token_ids = tokenizer.encode(str(rendered), add_special_tokens=False)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if not isinstance(token_ids, list) or (token_ids and isinstance(token_ids[0], list)):
        raise TypeError("tokenizer.encode must return one flat token-ID sequence")
    result: list[int] = []
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise TypeError("tokenizer.encode returned an invalid token ID")
        result.append(int(token_id))
    return result


def _unified_audio_template_ids(engine: Any, session: DuplexSession, *, add_generation_prompt: bool) -> tuple[list[int], int, int, int]:
    """Return tokenized one-user audio template and Gemma BOA/EOA IDs."""
    tokenizer = getattr(engine, "tokenizer", None)
    processor = getattr(getattr(engine, "_thinker_wrapper", None), "processor", None)
    audio_marker = (
        getattr(processor, "audio_token", None)
        or getattr(tokenizer, "audio_token", None)
        or "<|audio|>"
    )
    audio_marker = str(audio_marker).strip()
    if not audio_marker:
        raise RuntimeError("unified audio template requires the processor audio marker")
    token_ids = _tokenize_template_text(
        tokenizer,
        [{"role": "user", "content": audio_marker}],
        add_generation_prompt=add_generation_prompt,
    )
    marker_id_values = tokenizer.encode(audio_marker, add_special_tokens=False)
    if hasattr(marker_id_values, "tolist"):
        marker_id_values = marker_id_values.tolist()
    if isinstance(marker_id_values, tuple):
        marker_id_values = list(marker_id_values)
    if not isinstance(marker_id_values, (tuple, list)) or len(marker_id_values) != 1:
        raise RuntimeError("processor audio marker must map to exactly one tokenizer token")
    marker_id = int(marker_id_values[0])
    marker_positions = [index for index, value in enumerate(token_ids) if value == marker_id]
    if len(marker_positions) != 1:
        raise RuntimeError("one-user chat template must contain exactly one audio marker")
    backend = getattr(getattr(session, "listener", None), "backend", None)
    config = getattr(getattr(engine, "_thinker_wrapper", None), "config", None)
    boa_token_id = getattr(backend, "boa_token_id", None)
    if boa_token_id is None:
        boa_token_id = getattr(config, "boa_token_id", None)
    eoa_token_id = getattr(backend, "eoa_token_id", None)
    if eoa_token_id is None:
        eoa_token_id = getattr(config, "eoa_token_id", None)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (boa_token_id, eoa_token_id)):
        raise RuntimeError("unified audio template requires configured Gemma BOA and EOA token IDs")
    return token_ids, marker_positions[0], int(boa_token_id), int(eoa_token_id)


def _unified_turn_end_token_id(engine: Any) -> int:
    """Read the chat turn-end ID from the loaded tokenizer's own metadata."""
    tokenizer = getattr(engine, "tokenizer", None)
    candidates: list[int] = []
    for name in ("eot_id", "eot_token_id"):
        value = getattr(tokenizer, name, None)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"loaded tokenizer exposes an invalid {name}")
        candidates.append(int(value))
    if not candidates:
        eot_token = getattr(tokenizer, "eot_token", None)
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if isinstance(eot_token, str) and eot_token and callable(convert):
            value = convert(eot_token)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError("loaded tokenizer cannot resolve its configured EOT token")
            candidates.append(int(value))
    if not candidates:
        raise RuntimeError("unified audio template requires the loaded tokenizer's EOT token ID")
    if len(set(candidates)) != 1:
        raise RuntimeError("loaded tokenizer exposes conflicting EOT token IDs")
    return candidates[0]


def _validate_unified_audio_turn_end_suffix(
    engine: Any,
    suffix_ids: Sequence[int],
) -> int:
    """Require one configured EOT while preserving every template tail token."""
    eot_token_id = _unified_turn_end_token_id(engine)
    if sum(int(token_id) == eot_token_id for token_id in suffix_ids) != 1:
        raise RuntimeError(
            "unified audio template suffix must contain exactly one configured EOT token"
        )
    return eot_token_id


def _build_unified_audio_response_suffix(engine: Any, session: DuplexSession) -> tuple[int, ...]:
    """Build EOA + user turn end + assistant generation header without audio preprocessing."""
    token_ids, marker_position, _boa_token_id, eoa_token_id = _unified_audio_template_ids(
        engine, session, add_generation_prompt=True
    )
    template_suffix = token_ids[marker_position + 1 :]
    _validate_unified_audio_turn_end_suffix(engine, template_suffix)
    return (eoa_token_id, *template_suffix)


def _build_unified_backchannel_response_suffix(
    engine: Any,
    session: DuplexSession,
) -> tuple[int, ...]:
    """Build a private backchannel suffix without changing the imported prompt."""
    protocol = _require_unified_stable_audio_protocol(
        _configured_audio_response_protocol(engine)
    )
    if protocol == CONTROLLER_GUIDED_SAY_V1:
        # V1 is zero-prefill: action choice comes from the controller, while
        # the already-imported system prompt only requests spoken text. Adding
        # an action-specific instruction here would create a hidden late prompt.
        return _build_unified_audio_response_suffix(engine, session)
    token_ids, marker_position, _boa_token_id, eoa_token_id = _unified_audio_template_ids(
        engine, session, add_generation_prompt=True
    )
    template_suffix = tuple(int(value) for value in token_ids[marker_position + 1 :])
    _validate_unified_audio_turn_end_suffix(engine, template_suffix)
    tokenizer = getattr(engine, "tokenizer", None)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("private backchannel requires the loaded Thinker tokenizer")
    instruction = (
        " Rispondi soltanto con un brevissimo cenno di ascolto in italiano. "
        "Non aggiungere spiegazioni."
    )
    instruction_ids = encode(instruction, add_special_tokens=False)
    if hasattr(instruction_ids, "tolist"):
        instruction_ids = instruction_ids.tolist()
    if isinstance(instruction_ids, tuple):
        instruction_ids = list(instruction_ids)
    if (
        not isinstance(instruction_ids, (list, tuple))
        or not instruction_ids
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in instruction_ids)
    ):
        raise RuntimeError("loaded Thinker tokenizer returned invalid private backchannel prompt IDs")
    return (int(eoa_token_id), *(int(value) for value in instruction_ids), *template_suffix)


def _build_unified_next_user_header(engine: Any, session: DuplexSession) -> tuple[int, ...]:
    """Build the next user header through BOA, excluding the global BOS token."""
    token_ids, marker_position, boa_token_id, _eoa_token_id = _unified_audio_template_ids(
        engine, session, add_generation_prompt=False
    )
    header = token_ids[:marker_position]
    backend = getattr(getattr(session, "listener", None), "backend", None)
    config = getattr(getattr(engine, "_thinker_wrapper", None), "config", None)
    bos_token_id = getattr(backend, "bos_token_id", None)
    if bos_token_id is None:
        bos_token_id = getattr(config, "bos_token_id", None)
    if header and isinstance(bos_token_id, int) and header[0] == int(bos_token_id):
        header = header[1:]
    if not header:
        raise RuntimeError("unified next-user header template is empty")
    return (*header, boa_token_id)


def _unified_end_of_turn_token_id(engine: Any, session: DuplexSession) -> int:
    """Resolve the single EOT in the active template, allowing trailing whitespace IDs."""
    token_ids, marker_position, _boa_token_id, _eoa_token_id = _unified_audio_template_ids(
        engine, session, add_generation_prompt=False
    )
    suffix = token_ids[marker_position + 1 :]
    return _validate_unified_audio_turn_end_suffix(engine, suffix)


async def _finalize_unified_native_cache_branch(
    worker: Any,
    session_id: str,
    branch: dict[str, Any],
    *,
    history_committed: bool,
    close_ids: tuple[int, ...],
    retry_timeout_s: float = 2.0,
) -> Any:
    """Close or roll back one native assistant branch, then drain queued audio.

    The caller holds the per-session turn-transition lock. A successful close
    consumes the branch's reserved suffix budget; any failed close attempts a
    rollback to the user-turn boundary before reporting the error, so a
    partially appended close suffix is never left as a live branch.
    """
    if branch.get("ephemeral_private"):
        handle = branch.get("ephemeral_decode_handle")
        release = getattr(worker, "end_ephemeral_decode", None)
        if handle is None or not callable(release):
            raise RuntimeError("private native branch lacks its ephemeral release contract")
        release_future = asyncio.wrap_future(release(handle))
        try:
            await asyncio.shield(release_future)
        except asyncio.CancelledError:
            # Keep the native row alive until the owner has completed release;
            # cancellation must not turn this slot into a use-after-free.
            await asyncio.shield(release_future)
            raise
        branch["cache_position"] = int(getattr(handle, "cache_position", branch["cache_position"]))
        branch["finalized"] = True
        branch["private_slot_released"] = True
        branch["history_committed"] = False
        return None

    listener_generation_id = int(branch["listener_generation_id"])
    turn_start = int(branch["turn_start_cache_position"])
    cache_cursor = int(branch["cache_position"])

    async def current_snapshot() -> Any:
        snapshot = await asyncio.wrap_future(worker.session_state(session_id))
        if int(snapshot.generation_id) != listener_generation_id:
            raise RuntimeError("unified finalizer lost its listener generation")
        if int(snapshot.cache_position) < turn_start:
            raise RuntimeError("unified finalizer observed a cursor before the user-turn boundary")
        return snapshot

    async def rollback_to_turn_start() -> None:
        nonlocal cache_cursor
        rollback_deadline = asyncio.get_running_loop().time() + retry_timeout_s
        while True:
            latest = await current_snapshot()
            cache_cursor = int(latest.cache_position)
            try:
                rolled_back = await asyncio.wrap_future(worker.rollback_assistant_branch(
                    session_id,
                    expected_generation_id=listener_generation_id,
                    expected_cache_position=cache_cursor,
                    target_cache_position=turn_start,
                ))
                cache_cursor = int(rolled_back.cache_position)
                branch["cache_position"] = cache_cursor
                break
            except RuntimeError as exc:
                if (
                    str(exc) not in {
                        "assistant rollback requires a completed listener frame watermark",
                        "assistant rollback cannot cross a shared listener tick",
                    }
                    or asyncio.get_running_loop().time() >= rollback_deadline
                ):
                    raise
                await asyncio.sleep(0.002)

    async def drain_after_rollback(*, user_turn_already_open: bool) -> Any:
        nonlocal cache_cursor
        drain_deadline = asyncio.get_running_loop().time() + retry_timeout_s
        while True:
            try:
                drained = await asyncio.wrap_future(worker.drain_queued_audio(
                    session_id,
                    expected_generation_id=listener_generation_id,
                    expected_cache_position=cache_cursor,
                    user_turn_already_open=user_turn_already_open,
                ))
                branch["cache_position"] = int(drained.cache_position)
                branch["queue_open"] = False
                branch["finalized"] = True
                branch["rollback_recovered"] = True
                return drained
            except RuntimeError as exc:
                if (
                    str(exc) not in {
                        "queued-audio drain needs the assistant Q0 watermark to complete; retry after the prior tick",
                        "queued-audio drain cannot overlap a shared listener tick",
                    }
                    or asyncio.get_running_loop().time() >= drain_deadline
                ):
                    raise
                await asyncio.sleep(0.002)

    async def append_spoken_prefix_after_rewind() -> None:
        nonlocal cache_cursor
        transition_ids = tuple(
            int(value) for value in branch.get("transition_token_ids", ())
        )
        speech_ids = tuple(
            int(value) for value in branch.get("spoken_prefix_token_ids", ())
        )
        if not transition_ids:
            raise RuntimeError(
                "spoken-prefix KV rebuild has no original assistant transition tokens"
            )
        if not speech_ids:
            raise RuntimeError(
                "spoken-prefix KV rebuild has no exact Gemma token IDs from a consumed segment"
            )
        rebuild_and_close_ids = (*transition_ids, *speech_ids, *tuple(close_ids))
        reserve_deadline = asyncio.get_running_loop().time() + retry_timeout_s
        while True:
            try:
                await asyncio.wrap_future(worker.reserve_tail_tokens(
                    session_id,
                    expected_generation_id=listener_generation_id,
                    expected_cache_position=cache_cursor,
                    token_count=len(rebuild_and_close_ids) + 2,
                ))
                break
            except RuntimeError as exc:
                if (
                    str(exc) not in {
                        "tail reservation requires a completed listener frame watermark",
                        "tail reservation cannot overlap a shared listener tick",
                    }
                    or asyncio.get_running_loop().time() >= reserve_deadline
                ):
                    raise
                await asyncio.sleep(0.002)

        for start in range(0, len(rebuild_and_close_ids), 2):
            token_group = tuple(
                int(value) for value in rebuild_and_close_ids[start : start + 2]
            )
            append_started_ns = time.monotonic_ns()
            while True:
                owner_future = worker.append_text_tokens(
                    session_id,
                    token_group,
                    expected_generation_id=listener_generation_id,
                    expected_cache_position=cache_cursor,
                    consume_reserved_tokens=len(token_group),
                )
                try:
                    await asyncio.shield(asyncio.wrap_future(owner_future))
                    break
                except asyncio.CancelledError:
                    await asyncio.shield(asyncio.wrap_future(owner_future))
                    cache_cursor += len(token_group)
                    branch["cache_position"] = cache_cursor
                    raise
                except RuntimeError as exc:
                    if (
                        str(exc) != "text append requires completed listener frame watermark"
                        or time.monotonic_ns() - append_started_ns
                        >= int(retry_timeout_s * 1e9)
                    ):
                        raise
                    latest = await current_snapshot()
                    if int(latest.cache_position) != cache_cursor:
                        raise RuntimeError(
                            "spoken-prefix KV rebuild observed an untracked cache append"
                        ) from exc
                    await asyncio.sleep(0.002)
            cache_cursor += len(token_group)
            branch["cache_position"] = cache_cursor
        expected_exclusive_cursor = branch.get("spoken_prefix_exclusive_cursor")
        rebuilt_prefix_cursor = cache_cursor - len(tuple(close_ids))
        if (expected_exclusive_cursor is not None
                and rebuilt_prefix_cursor != int(expected_exclusive_cursor)):
            raise RuntimeError(
                "rebuilt spoken prefix does not end at its acknowledged exclusive KV cursor"
            )

    async def rollback_and_drain() -> Any:
        await rollback_to_turn_start()
        acknowledged_text = str(branch.get("spoken_prefix_text", ""))
        if acknowledged_text:
            try:
                if not tuple(branch.get("spoken_prefix_token_ids", ())):
                    raise RuntimeError(
                        "client-consumed spoken prefix has text but no original Gemma token IDs"
                    )
                await append_spoken_prefix_after_rewind()
                drained = await drain_after_rollback(user_turn_already_open=False)
            except BaseException as rebuild_error:
                # Preserve the user turn boundary when canonical reconstruction
                # cannot fit. The caller sees an error and can quarantine the
                # branch; acknowledged text is never silently represented by raw JSON.
                await rollback_to_turn_start()
                drained = await drain_after_rollback(user_turn_already_open=True)
                branch["history_committed"] = False
                branch["spoken_prefix_rebuilt"] = False
                raise RuntimeError(
                    "spoken-prefix KV rebuild failed; branch was rolled back to the open user turn"
                ) from rebuild_error
            branch["spoken_prefix_rebuilt"] = True
            branch["history_committed"] = False
            return drained
        drained = await drain_after_rollback(user_turn_already_open=True)
        branch["history_committed"] = False
        branch["spoken_prefix_rebuilt"] = False
        return drained

    if not history_committed:
        drained = await rollback_and_drain()
        return drained

    close_error: BaseException | None = None
    try:
        latest = await current_snapshot()
        latest_cursor = int(latest.cache_position)
        if latest_cursor < cache_cursor:
            raise RuntimeError("unified finalizer observed a backward cache cursor")
        if latest_cursor > cache_cursor:
            raise RuntimeError(
                "unified finalizer observed an untracked cache append after Thinker completion"
            )
        cache_cursor = latest_cursor
        branch["cache_position"] = cache_cursor
        # Reassert capacity against the post-decode cursor and all queued Q2
        # frames before the first close token can mutate native cache.
        reserve_deadline = asyncio.get_running_loop().time() + retry_timeout_s
        while True:
            try:
                await asyncio.wrap_future(worker.reserve_tail_tokens(
                    session_id,
                    expected_generation_id=listener_generation_id,
                    expected_cache_position=cache_cursor,
                    token_count=len(close_ids) + 2,
                ))
                break
            except RuntimeError as exc:
                if (
                    str(exc) not in {
                        "tail reservation requires a completed listener frame watermark",
                        "tail reservation cannot overlap a shared listener tick",
                    }
                    or asyncio.get_running_loop().time() >= reserve_deadline
                ):
                    raise
                await asyncio.sleep(0.002)
        for start in range(0, len(close_ids), 2):
            token_group = tuple(int(value) for value in close_ids[start : start + 2])
            owner_future = worker.append_text_tokens(
                session_id,
                token_group,
                expected_generation_id=listener_generation_id,
                expected_cache_position=cache_cursor,
                consume_reserved_tokens=len(token_group),
            )
            try:
                await asyncio.shield(asyncio.wrap_future(owner_future))
            except asyncio.CancelledError:
                await asyncio.shield(asyncio.wrap_future(owner_future))
                cache_cursor += len(token_group)
                branch["cache_position"] = cache_cursor
                raise
            cache_cursor += len(token_group)
            branch["cache_position"] = cache_cursor
    except BaseException as exc:
        close_error = exc

    if close_error is not None:
        try:
            await rollback_and_drain()
        except BaseException as recovery_error:
            raise RuntimeError(
                "unified native close failed and assistant-branch rollback/audio drain also failed; "
                "the session must remain quarantined"
            ) from recovery_error
        raise RuntimeError(
            "unified native close failed; assistant branch was rolled back and queued audio was "
            "drained before the session was quarantined"
        ) from close_error

    drain_deadline = asyncio.get_running_loop().time() + retry_timeout_s
    while True:
        try:
            drained = await asyncio.wrap_future(worker.drain_queued_audio(
                session_id,
                expected_generation_id=listener_generation_id,
                expected_cache_position=cache_cursor,
                user_turn_already_open=False,
            ))
            branch["cache_position"] = int(drained.cache_position)
            branch["queue_open"] = False
            branch["finalized"] = True
            branch["history_committed"] = True
            return drained
        except RuntimeError as exc:
            if (
                str(exc) not in {
                    "queued-audio drain needs the assistant Q0 watermark to complete; retry after the prior tick",
                    "queued-audio drain cannot overlap a shared listener tick",
                }
                or asyncio.get_running_loop().time() >= drain_deadline
            ):
                try:
                    await rollback_and_drain()
                except BaseException as recovery_error:
                    raise RuntimeError(
                        "unified native audio drain failed after close; rollback/audio recovery also "
                        "failed and the session must remain quarantined"
                    ) from recovery_error
                raise RuntimeError(
                    "unified native audio drain failed after close; branch was rolled back and "
                    "queued audio was drained before the session was quarantined"
                ) from exc
            await asyncio.sleep(0.002)


def _consume_unified_native_text_append_result(
    result: Any,
    *,
    timings_ns: dict[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Join the native LM-head stream before exposing its tensors to sampling.

    The worker records ``ready_event`` after the LM head on its listener
    stream. A stream wait keeps the dependency asynchronous; record_stream
    keeps the allocator from recycling either output while this consumer
    stream is still reading it.
    """
    try:
        logits = result.logits
        hidden = result.final_norm_hidden
        ready_event = result.ready_event
    except AttributeError as exc:
        raise RuntimeError(
            "unified native text append must return logits, final_norm_hidden, and ready_event"
        ) from exc

    logits_device = getattr(logits, "device", None)
    logits_is_cuda = bool(
        getattr(logits, "is_cuda", getattr(logits_device, "type", None) == "cuda")
    )
    hidden_device = getattr(hidden, "device", logits_device)
    hidden_is_cuda = bool(
        getattr(hidden, "is_cuda", getattr(hidden_device, "type", None) == "cuda")
    )
    if logits_is_cuda != hidden_is_cuda or (
        logits_is_cuda and hidden_device != logits_device
    ):
        raise RuntimeError("unified native text outputs must share one CUDA device")
    if not logits_is_cuda:
        if ready_event is not None:
            raise RuntimeError("CPU unified native text outputs cannot carry a CUDA ready_event")
        return logits, hidden
    if ready_event is None:
        raise RuntimeError("CUDA unified native text append is missing its ready_event")

    consumer_stream = torch.cuda.current_stream(device=logits_device)
    if timings_ns is not None:
        timings_ns["consumer_wait_event_started_ns"] = time.monotonic_ns()
    consumer_stream.wait_event(ready_event)
    if timings_ns is not None:
        timings_ns["consumer_wait_event_enqueued_ns"] = time.monotonic_ns()
    logits.record_stream(consumer_stream)
    hidden.record_stream(consumer_stream)
    if timings_ns is not None:
        timings_ns["consumer_record_stream_returned_ns"] = time.monotonic_ns()
    return logits, hidden


def _run_unified_native_thinker(
    engine: Any,
    session: DuplexSession,
    request: dict[str, Any],
    *,
    worker: Any,
    transition_token_ids: tuple[int, ...],
    think_kwargs: dict[str, Any],
    generation_id: int | None,
    branch: dict[str, Any],
) -> dict[str, Any]:
    """Decode the Thinker suffix through the listener's native cache owner."""
    listener_generation_id = int(branch["listener_generation_id"])
    private_handle = branch.get("ephemeral_decode_handle")
    private_transition_ids = tuple(int(value) for value in transition_token_ids)
    private_transition_offset = 0
    private_transition_remaining = (
        len(private_transition_ids) if private_handle is not None else 0
    )
    cache_cursor = int(
        private_handle.source_cache_position
        if private_handle is not None
        else branch["cache_position"]
    )
    current = worker.session_state(session.session_id).result(timeout=5.0)
    if int(current.generation_id) != listener_generation_id:
        raise RuntimeError("unified Thinker branch lost its listener generation")
    if private_handle is not None:
        if (
            private_handle.parent_session_id != session.session_id
            or int(private_handle.parent_generation_id) != listener_generation_id
            or int(private_handle.source_cache_position) != cache_cursor
        ):
            raise RuntimeError("private Thinker branch identity differs from its listener source")
        branch["cache_position"] = cache_cursor
        branch["turn_start_cache_position"] = cache_cursor
    elif int(branch.get("native_token_append_count", 0)) == 0:
        cache_cursor = int(current.cache_position)
        branch["cache_position"] = cache_cursor
        branch["turn_start_cache_position"] = cache_cursor
    branch["transition_token_ids"] = private_transition_ids
    branch["spoken_prefix_base_cursor"] = cache_cursor + len(private_transition_ids)
    if not tuple(branch.get("client_consumed_segments", ())):
        branch["spoken_prefix_exclusive_cursor"] = int(branch["spoken_prefix_base_cursor"])
    token_timings: dict[str, int] = {}
    # Unified-native decode submits token work to its owning queue and does not
    # acquire the shared Python Thinker/Gemma model lock in this call chain.
    # Queue/Future wait is a different measurement and is reported separately.
    thinker_metrics: dict[str, Any] = {
        "thinker_decode_lock_wait_ms": 0.0,
        "thinker_decode_lock_acquire_count": 0,
        "thinker_decode_lock_wait_provenance": "native_no_python_decode_lock",
    }
    stop_check = think_kwargs.get("stop_check")
    response_cancel_event = request.get("_response_cancel_event")
    native_profile: dict[str, Any] = {
        "clock": "server monotonic_ns",
        "policy_decision_at_ns": branch.get("policy_decision_at_ns"),
        "queue_open_requested_ns": branch.get("queue_open_requested_ns"),
        "queue_open_ready_ns": branch.get("queue_open_ready_ns"),
        "append_steps": [],
        "sampling_steps": [],
    }
    branch["unified_native_profile"] = native_profile

    def adopt_append_cursor(result: Any, token_count: int) -> None:
        nonlocal cache_cursor
        result_cursor = getattr(result, "cache_position", None)
        first_append = int(branch.get("native_token_append_count", 0)) == 0
        if result_cursor is None:
            committed_cursor = cache_cursor + token_count
        else:
            committed_cursor = int(result_cursor)
        if first_append:
            actual_start_cursor = committed_cursor - token_count
            if actual_start_cursor < cache_cursor:
                raise RuntimeError("unified text append returned a cursor before its request")
            rebase_delta = actual_start_cursor - cache_cursor
            if rebase_delta:
                for key in (
                    "turn_start_cache_position",
                    "spoken_prefix_base_cursor",
                    "spoken_prefix_exclusive_cursor",
                ):
                    value = branch.get(key)
                    if isinstance(value, int):
                        branch[key] = value + rebase_delta
        cache_cursor = committed_cursor
        branch["cache_position"] = cache_cursor
        branch["native_token_append_count"] = int(
            branch.get("native_token_append_count", 0)
        ) + token_count

    def publish_profile() -> None:
        if thinker_metrics:
            token_timings["thinker_runtime_metrics"] = dict(thinker_metrics)
        token_timings["unified_native_profile"] = {
            **native_profile,
            "append_steps": [dict(step) for step in native_profile["append_steps"]],
            "sampling_steps": [dict(step) for step in native_profile["sampling_steps"]],
        }
        callback = request.get("_on_thinker_metrics")
        if callable(callback):
            callback(dict(token_timings))

    async def forward_native_tokens(token_ids: list[int]):
        nonlocal cache_cursor, private_transition_remaining, private_transition_offset
        retry_deadline = time.monotonic() + 2.0
        append_call_ns = time.monotonic_ns()
        if private_handle is not None:
            is_private_transition = private_transition_remaining > 0
            next_transition_offset = private_transition_offset
            if is_private_transition:
                next_transition_offset = _validate_private_transition_chunk(
                    private_transition_ids,
                    private_transition_offset,
                    token_ids,
                    source_cache_position=int(private_handle.source_cache_position),
                    current_cache_position=cache_cursor,
                )
                owner_future = worker.append_private_transition_tokens(
                    private_handle,
                    token_ids,
                    expected_cache_position=cache_cursor,
                )
            else:
                if private_transition_offset != len(private_transition_ids):
                    raise RuntimeError("private decode sampled before the original Thinker transition completed")
                owner_future = worker.append_private_tokens(
                    private_handle,
                    token_ids,
                    expected_cache_position=cache_cursor,
                )
            wrapped = asyncio.wrap_future(owner_future)
            try:
                result = await asyncio.shield(wrapped)
            except asyncio.CancelledError:
                result = await asyncio.shield(wrapped)
                cache_cursor = int(result.cache_position)
                branch["cache_position"] = cache_cursor
                if is_private_transition:
                    private_transition_offset = next_transition_offset
                    private_transition_remaining = len(private_transition_ids) - private_transition_offset
                raise
            cache_cursor = int(result.cache_position)
            branch["cache_position"] = cache_cursor
            if is_private_transition:
                private_transition_offset = next_transition_offset
                private_transition_remaining = len(private_transition_ids) - private_transition_offset
            received_ns = time.monotonic_ns()
            step_profile: dict[str, Any] = {
                "phase": "private_transition" if is_private_transition else "private_generated_token",
                "logical_token_count": len(token_ids),
                "append_call_ns": append_call_ns,
                "future_result_received_ns": received_ns,
                "owner_future_wait_ns": max(0, received_ns - append_call_ns),
                "private_cache_position": cache_cursor,
            }
            worker_timings = getattr(result, "timings_ns", None)
            if isinstance(worker_timings, Mapping):
                step_profile.update({str(key): int(value) for key, value in worker_timings.items()})
            pair = _consume_unified_native_text_append_result(result, timings_ns=step_profile)
            step_profile["consumer_returned_ns"] = time.monotonic_ns()
            native_profile["append_steps"].append(step_profile)
            return pair
        while True:
            prepared_protocol = None
            if (
                branch.get("response_protocol") == CONTROLLER_GUIDED_SAY_V1
                and int(branch.get("native_token_append_count", 0)) == 0
                and tuple(int(value) for value in token_ids) == private_transition_ids
            ):
                prepared_protocol = CONTROLLER_GUIDED_SAY_V1
            append_kwargs = {
                "expected_generation_id": listener_generation_id,
                "expected_cache_position": cache_cursor,
                "allow_initial_cursor_rebase": (
                    int(branch.get("native_token_append_count", 0)) == 0
                ),
                "append_scope_id": str(branch.get("response_id") or "") or None,
                "response_cancel_event": response_cancel_event,
            }
            if prepared_protocol is not None:
                append_kwargs["prepared_protocol"] = prepared_protocol
            owner_future = worker.append_text_tokens(
                session.session_id,
                token_ids,
                **append_kwargs,
            )
            wrapped = asyncio.wrap_future(owner_future)
            try:
                result = await asyncio.shield(wrapped)
            except asyncio.CancelledError:
                if response_cancel_event is not None:
                    response_cancel_event.set()
                try:
                    result = await asyncio.shield(wrapped)
                except RuntimeError as append_error:
                    if str(append_error) == "text append response canceled before native KV mutation":
                        raise asyncio.CancelledError from append_error
                    raise
                adopt_append_cursor(result, len(token_ids))
                raise
            except RuntimeError as exc:
                if str(exc) == "text append requires completed listener frame watermark":
                    if time.monotonic() >= retry_deadline or (
                        callable(stop_check) and stop_check()
                    ):
                        raise
                    latest = worker.session_state(session.session_id).result(timeout=5.0)
                    if int(latest.generation_id) != listener_generation_id:
                        raise RuntimeError(
                            "unified Thinker branch lost its listener generation"
                        ) from exc
                    latest_cursor = int(latest.cache_position)
                    latest_valid_history_tokens = int(
                        getattr(latest, "valid_history_tokens", latest_cursor)
                    )
                    initial_append = int(branch.get("native_token_append_count", 0)) == 0
                    if (
                        latest_cursor < cache_cursor
                        or (not initial_append and latest_cursor != cache_cursor)
                        or latest_valid_history_tokens != latest_cursor
                    ):
                        raise RuntimeError(
                            "unified listener cache cursor changed incoherently while waiting "
                            "for a frame watermark: "
                            f"expected_cache_position={cache_cursor}, "
                            f"actual_cache_position={latest_cursor}, "
                            f"actual_valid_history_tokens={latest_valid_history_tokens}"
                        ) from exc
                    await asyncio.sleep(0.002)
                    continue
                raise
            adopt_append_cursor(result, len(token_ids))
            received_ns = time.monotonic_ns()
            step_profile: dict[str, Any] = {
                "phase": "suffix" if not branch.get("first_logits_ready") else "generated_token",
                "logical_token_count": len(token_ids),
                "append_call_ns": append_call_ns,
                "future_result_received_ns": received_ns,
                "owner_future_wait_ns": max(0, received_ns - append_call_ns),
            }
            worker_timings = getattr(result, "timings_ns", None)
            if isinstance(worker_timings, Mapping):
                step_profile.update({str(key): int(value) for key, value in worker_timings.items()})
                owner_started_ns = worker_timings.get("owner_started_ns")
                if isinstance(owner_started_ns, int):
                    step_profile["owner_queue_wait_ns"] = max(0, owner_started_ns - append_call_ns)
            transition_cache_hit = getattr(result, "prepared_transition_cache_hit", None)
            if transition_cache_hit is not None:
                step_profile["prepared_transition_cache_hit"] = bool(transition_cache_hit)
            pair = _consume_unified_native_text_append_result(
                result,
                timings_ns=step_profile,
            )
            step_profile["consumer_returned_ns"] = time.monotonic_ns()
            native_profile["append_steps"].append(step_profile)
            return pair

    async def on_first_logits_ready(event: Any) -> None:
        branch["first_logits_ready"] = True
        token_timings["thinker_first_logits_ready_at_ns"] = int(event.ready_ns)
        native_profile["first_logits_callback_ns"] = time.monotonic_ns()
        decision_ns = native_profile.get("policy_decision_at_ns")
        if isinstance(decision_ns, int):
            first_logits_ms = max(0.0, (int(event.ready_ns) - decision_ns) / 1_000_000.0)
            token_timings["first_logits_after_decision_ms"] = first_logits_ms
            native_profile["first_logits_after_decision_ms"] = first_logits_ms
        publish_profile()

    async def on_native_token(
        token_id: int,
        _hidden: torch.Tensor,
        timing: Any,
    ) -> None:
        record = request.get("_on_thinker_token_id")
        if callable(record):
            record(int(token_id))
        if "first_thinker_token_at_ns" not in token_timings:
            token_timings["first_thinker_token_at_ns"] = time.monotonic_ns()
            token_timings["first_thinker_token_hidden_ready_at_ns"] = int(
                timing.token_hidden_ready_ns
            )
            token_timings["first_thinker_token_sampled_at_ns"] = int(timing.sampled_ns)
            token_timings["first_thinker_token_forward_started_at_ns"] = int(
                timing.token_forward_started_ns
            )
            decision_ns = native_profile.get("policy_decision_at_ns")
            if isinstance(decision_ns, int):
                sampled_ms = max(0.0, (int(timing.sampled_ns) - decision_ns) / 1_000_000.0)
                packet_ready_ms = max(
                    0.0,
                    (int(timing.token_hidden_ready_ns) - decision_ns) / 1_000_000.0,
                )
                token_timings["first_token_sampled_ms"] = sampled_ms
                token_timings["first_packet_hidden_ready_ms"] = packet_ready_ms
                # TTFT is measured at the first token's own post-forward hidden,
                # never at the earlier sample or the preceding token's hidden.
                token_timings["ttft_ms"] = packet_ready_ms
                native_profile["first_token_sampled_ms"] = sampled_ms
                native_profile["first_packet_hidden_ready_ms"] = packet_ready_ms
                native_profile["ttft_ms"] = packet_ready_ms
            native_profile["first_token_callback_ns"] = int(
                token_timings["first_thinker_token_at_ns"]
            )
            # This callback exposes the post-forward token to the response
            # owner; the websocket action is emitted later by the protocol.
            native_profile["first_token_model_callback_ns"] = time.monotonic_ns()
            publish_profile()

    raw_candidate_guard = think_kwargs.get("on_candidate_token")
    candidate_guard_trace: list[dict[str, Any]] = []
    if callable(raw_candidate_guard):
        def observed_candidate_guard(token_id: int, text: str, terminal: bool) -> bool:
            try:
                accepted = bool(raw_candidate_guard(token_id, text, terminal))
            except BaseException as exc:
                candidate_guard_trace.append({
                    "token_id": int(token_id),
                    "text": str(text)[:120],
                    "terminal": bool(terminal),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                raise
            candidate_guard_trace.append({
                "token_id": int(token_id),
                "text": str(text)[:120],
                "terminal": bool(terminal),
                "accepted": accepted,
            })
            return accepted

        think_kwargs["on_candidate_token"] = observed_candidate_guard

    sampler = engine.build_thinker_sampler(
        temperature=float(think_kwargs["temperature"]),
        top_p=float(think_kwargs["top_p"]),
        repetition_penalty=float(request.get("repetition_penalty", 1.15)),
        min_new_tokens=int(think_kwargs["min_new_tokens"]),
        action=None,
        runtime_metrics=thinker_metrics,
        on_generation_start=think_kwargs.get("on_generation_start"),
        on_generated_token=think_kwargs.get("on_generated_token"),
        on_candidate_token=think_kwargs.get("on_candidate_token"),
        on_candidate_fallback=think_kwargs.get("on_candidate_fallback"),
        on_action_logits_probe=think_kwargs.get("on_action_logits_probe"),
        state=session,
        generation_id=(int(generation_id) if generation_id is not None else None),
    )

    def _append_sampling_profile(
        stage: str,
        started_ns: int,
        *,
        candidate: Any = None,
        history_length: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        finished_ns = time.monotonic_ns()
        record: dict[str, Any] = {
            "stage": stage,
            "started_ns": int(started_ns),
            "finished_ns": finished_ns,
            "duration_ns": max(0, finished_ns - int(started_ns)),
        }
        if history_length is not None:
            record["generated_history_length"] = int(history_length)
        if isinstance(candidate, (tuple, list)) and len(candidate) == 3:
            record["candidate_id"] = int(candidate[1])
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"
        native_profile["sampling_steps"].append(record)
        if error is not None:
            publish_profile()

    raw_sample_token = sampler.sample_token

    def profiled_sample_token(logits: torch.Tensor, generated_ids: Sequence[int]):
        started_ns = time.monotonic_ns()
        try:
            candidate = raw_sample_token(logits, generated_ids)
        except BaseException as exc:
            _append_sampling_profile(
                "sample_token",
                started_ns,
                history_length=len(generated_ids),
                error=exc,
            )
            raise
        _append_sampling_profile(
            "sample_token",
            started_ns,
            candidate=candidate,
            history_length=len(generated_ids),
        )
        return candidate

    sampler.sample_token = profiled_sample_token
    raw_sample_initial = sampler.sample_initial

    def profiled_sample_initial(logits: torch.Tensor, generated_ids: Sequence[int] = ()):
        started_ns = time.monotonic_ns()
        try:
            candidate = raw_sample_initial(logits, generated_ids)
        except BaseException as exc:
            _append_sampling_profile(
                "sample_initial",
                started_ns,
                history_length=len(generated_ids),
                error=exc,
            )
            raise
        _append_sampling_profile(
            "sample_initial",
            started_ns,
            candidate=candidate,
            history_length=len(generated_ids),
        )
        return candidate

    sampler.sample_initial = profiled_sample_initial
    raw_select_valid_candidate = sampler.select_valid_candidate

    def profiled_select_valid_candidate(candidate: Any, **kwargs: Any):
        candidate_guard_trace.clear()
        started_ns = time.monotonic_ns()
        generated_ids = kwargs.get("generated_ids", ())
        try:
            selected = raw_select_valid_candidate(candidate, **kwargs)
        except BaseException as exc:
            diagnostic: dict[str, Any] = {
                "stage": "select_valid_candidate",
                "started_ns": int(started_ns),
                "finished_ns": time.monotonic_ns(),
                "generated_ids": [int(value) for value in generated_ids],
                "guard_outcomes": list(candidate_guard_trace),
                "error": f"{type(exc).__name__}: {exc}",
            }
            if isinstance(candidate, (tuple, list)) and len(candidate) == 3:
                diagnostic["sampled_candidate_id"] = int(candidate[1])
                scores = candidate[2]
                if isinstance(scores, torch.Tensor) and scores.numel():
                    count = min(64, int(scores.numel()))
                    top_scores, top_ids = torch.topk(scores.reshape(-1), k=count)
                    diagnostic["top64_ids"] = [int(value) for value in top_ids.detach().cpu().tolist()]
                    diagnostic["top64_scores"] = [
                        float(value) for value in top_scores.detach().float().cpu().tolist()
                    ]
            snapshot_callback = request.get("_audio_response_v2_diagnostic_snapshot")
            if callable(snapshot_callback):
                try:
                    diagnostic["protocol_state"] = snapshot_callback()
                except BaseException as snapshot_error:
                    diagnostic["protocol_state_error"] = (
                        f"{type(snapshot_error).__name__}: {snapshot_error}"
                    )
            native_profile["first_guard_failure"] = diagnostic
            thinker_metrics.update({
                key: value for key, value in sampler.runtime_metrics.items()
                if isinstance(key, str)
            })
            finished_ns = int(diagnostic["finished_ns"])
            diagnostic["duration_ns"] = max(0, finished_ns - started_ns)
            publish_profile()
            raise
        _append_sampling_profile(
            "select_valid_candidate",
            started_ns,
            candidate=selected,
            history_length=len(generated_ids),
        )
        return selected

    sampler.select_valid_candidate = profiled_select_valid_candidate
    decoder = UnifiedNativeDecoder(
        getattr(engine, "thinker", None),
        forward_tokens=forward_native_tokens,
        sampler=sampler,
        monotonic_ns=time.monotonic_ns,
        eos_token_ids=sampler.eos_token_ids,
    )
    decoded = asyncio.run(decoder.decode(
        transition_token_ids=transition_token_ids,
        max_new_tokens=int(think_kwargs["max_new_tokens"]),
        stop_check=think_kwargs.get("stop_check"),
        first_logits_ready=on_first_logits_ready,
        on_token=on_native_token,
    ))
    publish_profile()
    # Preserve engine.think's raw trace contract: EOS is observable for
    # provenance, but the decoder never forwards it or gives it Talker hidden.
    if decoded.termination_reason == "eos" and decoded.terminal_token_id is not None:
        record = request.get("_on_thinker_token_id")
        if callable(record):
            record(int(decoded.terminal_token_id))
    generated_ids = [int(item.token_id) for item in decoded.generated_tokens]
    generated_hidden = [item.hidden for item in decoded.generated_tokens]
    if generated_hidden:
        hidden = torch.cat(generated_hidden, dim=1)
    else:
        thinker_config = getattr(getattr(engine, "thinker", None), "config", None)
        text_config = (
            thinker_config.get_text_config(decoder=True)
            if callable(getattr(thinker_config, "get_text_config", None))
            else thinker_config
        )
        hidden_size = int(getattr(text_config, "hidden_size", 1536))
        hidden = torch.empty(
            (1, 0, hidden_size),
            device=getattr(engine, "device", "cpu"),
            dtype=getattr(engine, "dtype", torch.float32),
        )
    protocol_text = str(engine.tokenizer.decode(generated_ids, skip_special_tokens=True))
    branch["generated_token_count"] = int(decoded.committed_count)
    return {
        "text": protocol_text,
        "protocol_text": protocol_text,
        "gen_ids": generated_ids,
        "hidden": hidden,
        "hidden_prompt_len": 0,
        "prompt_len": 0,
        "past_key_values": None,
        "terminal_reason": decoded.termination_reason,
        "cancelled": decoded.termination_reason == "cancelled",
        "runtime_metrics": {
            **thinker_metrics,
            "unified_native_profile": {
                **native_profile,
                "append_steps": [dict(step) for step in native_profile["append_steps"]],
                "sampling_steps": [dict(step) for step in native_profile["sampling_steps"]],
            },
            "unified_native_cache": {
                "listener_generation_id": listener_generation_id,
                "turn_start_cache_position": int(branch["turn_start_cache_position"]),
                "cache_position_after_decode": cache_cursor,
                "transition_token_count": int(decoded.transition_token_count),
                "committed_generated_tokens": int(decoded.committed_count),
                "terminal_reason": decoded.termination_reason,
                "owner_action": decoded.owner_action,
            },
        },
    }


def _response_projector_mode(engine: Any) -> str:
    if bool(getattr(engine, "ablate_response_projector", False)):
        return "plain_qwen_text_only_no_prefix_kv"
    adapter = getattr(engine, "frame_residual_adapter", None)
    if adapter is not None:
        return str(
            getattr(
                engine,
                "frame_residual_adapter_architecture",
                getattr(adapter, "architecture_id", "frame_residual_adapter_loaded"),
            )
        )
    return "live_v9_projector"


_AUDIO_RESPONSE_V2_FAILURE_TRACE_LOCK = threading.Lock()
_AUDIO_RESPONSE_V2_FAILURE_TRACE_RECORDS: deque[dict[str, Any]] = deque(
    maxlen=AUDIO_RESPONSE_V2_FAILURE_TRACE_MAX_RECORDS
)
_AUDIO_RESPONSE_V2_ACTION_PROBE_LOCK = threading.Lock()
_AUDIO_RESPONSE_V2_ACTION_PROBE_CLAIMED = False
_TTS_QUALITY_DIAGNOSTIC_CAPTURE = TTSQualityDiagnosticCapture(
    output_dir=".tmp/recovery_probe/e2e_pair_diagnostics",
    max_records=8,
)


def _claim_audio_response_v2_action_probe() -> bool:
    """Enable one private, action-token-only diagnostic per server process."""
    global _AUDIO_RESPONSE_V2_ACTION_PROBE_CLAIMED
    if os.environ.get("DUPLEX_V2_ACTION_LOGIT_PROBE_ONCE") != "1":
        return False
    with _AUDIO_RESPONSE_V2_ACTION_PROBE_LOCK:
        if _AUDIO_RESPONSE_V2_ACTION_PROBE_CLAIMED:
            return False
        _AUDIO_RESPONSE_V2_ACTION_PROBE_CLAIMED = True
        return True


def _is_v2_no_speech_result(
    *,
    response_protocol: str | None,
    modalities: Any,
    wav: Any,
    talker_metrics: Any,
    cancelled: bool = False,
) -> bool:
    """Recognize a successful v2 response whose Talker emitted no CB0/PCM."""
    if (
        response_protocol != AUDIO_RESPONSE_PROTOCOL_V2
        or "audio" not in (modalities or ["text"])
        or cancelled
        or wav is None
        or not isinstance(talker_metrics, dict)
    ):
        return False
    pcm = np.asarray(wav).reshape(-1)
    generated_cb0_frames = talker_metrics.get("generated_cb0_frames")
    return pcm.size == 0 and isinstance(generated_cb0_frames, int) and generated_cb0_frames == 0


def _module_inference_freeze_status(module: Any) -> dict[str, Any] | None:
    if module is None:
        return None
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return None
    values = list(parameters())
    return {
        "training": bool(getattr(module, "training", False)),
        "parameter_count": len(values),
        "requires_grad_count": sum(bool(parameter.requires_grad) for parameter in values),
    }


def _realtime_inference_failure_done_event(
    response_id: str,
    runtime_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build the terminal event paired with a realtime INFERENCE_ERROR."""
    return {
        "type": "response.done",
        "response_id": str(response_id),
        "status": "failed",
        "reason": "inference_error",
        "runtime_metrics": runtime_metrics,
    }


def _record_audio_response_v2_private_failure(
    *,
    collector: AudioResponseV2Collector,
    reason: str,
    response_id: str | None,
    action: str | None,
    budget: dict[str, int] | None,
) -> None:
    """Persist only bounded token IDs and parser counts, never raw text/audio."""
    record = {
        "timestamp_unix_ms": int(time.time() * 1000),
        "response_id": str(response_id or ""),
        "selected_action": str(action or ""),
        "budget": dict(budget) if budget is not None else None,
        **collector.private_failure_snapshot(reason=reason),
    }
    with _AUDIO_RESPONSE_V2_FAILURE_TRACE_LOCK:
        _AUDIO_RESPONSE_V2_FAILURE_TRACE_RECORDS.append(record)
        path = Path(__file__).resolve().parent.parent / ".tmp" / "audio_response_v2_private_failures.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        payload = "".join(
            json.dumps(item, ensure_ascii=True, separators=(",", ":")) + "\n"
            for item in _AUDIO_RESPONSE_V2_FAILURE_TRACE_RECORDS
        )
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(path)


def _record_turn_local_vocalized_prefix(
    branch: dict[str, Any] | None,
    generation_id: int,
    kv_position: int | None,
    text: str,
) -> None:
    """Publish an ACKed text prefix only to its owning response generation.

    A missing cursor is the frozen empty FIFO snapshot before any Qwen ACK.
    It must not manufacture a KV position or an assistant prefix.
    """
    if not isinstance(branch, dict):
        return
    lock = branch.get("spoken_prefix_lock")
    if lock is None:
        return
    with lock:
        if int(branch.get("session_generation_id", -1)) != int(generation_id):
            return
        normalized_text = str(text)
        if kv_position is None:
            if normalized_text:
                raise RuntimeError("spoken text has no acknowledged KV cursor")
            return
        previous = branch.get("spoken_prefix_kv_position")
        if previous is not None and int(kv_position) < int(previous):
            raise RuntimeError("turn-local spoken-prefix KV cursor moved backwards")
        branch["spoken_prefix_kv_position"] = int(kv_position)
        branch["spoken_prefix_text"] = normalized_text


def _record_client_consumed_pcm(
    branch: dict[str, Any] | None,
    acknowledgement: dict[str, Any],
) -> bool:
    """Record one complete, client-consumed PCM segment against its exact marker.

    The ACK sequence is a client event counter; it is deliberately independent
    from AudioPacket.packet_id.  The only accepted range is the complete range
    previously announced by ``response.audio.segment_done``.  This signal
    describes logical client/device-buffer consumption, never acoustic output.
    """
    if not isinstance(branch, dict) or not isinstance(acknowledgement, dict):
        return False
    if acknowledgement.get("type") != "client_consumed_pcm":
        return False
    response_id = str(branch.get("response_id", ""))
    segment_id = acknowledgement.get("speaker_segment_id")
    generation_value = acknowledgement.get("generation_id")
    start_sample = acknowledgement.get("start_sample")
    end_sample = acknowledgement.get("end_sample")
    sample_rate = acknowledgement.get("sample_rate")
    sequence = acknowledgement.get("sequence")
    if (
        isinstance(generation_value, bool) or not isinstance(generation_value, int)
        or isinstance(start_sample, bool) or not isinstance(start_sample, int)
        or isinstance(end_sample, bool) or not isinstance(end_sample, int)
        or isinstance(sample_rate, bool) or not isinstance(sample_rate, int)
        or isinstance(sequence, bool) or not isinstance(sequence, int)
    ):
        return False
    if (
        not response_id
        or acknowledgement.get("response_id") != response_id
        or generation_value != branch.get("session_generation_id")
        or not isinstance(segment_id, str)
        or not segment_id
        or acknowledgement.get("consumption_semantics") != "logical_client_device_buffer"
        or start_sample < 0
        or end_sample <= start_sample
        or sample_rate != CODEC_SAMPLE_RATE
        or sequence < 0
        or sequence <= int(branch.get("client_consumed_pcm_sequence", -1))
    ):
        return False
    segments = branch.get("playback_segments")
    if not isinstance(segments, dict):
        return False
    segment = segments.get(segment_id)
    if not isinstance(segment, dict):
        return False
    if (
        segment.get("response_id") != response_id
        or segment.get("generation_id") != generation_value
        or segment.get("speaker_segment_id") != segment_id
        or segment.get("sample_rate") != sample_rate
        or segment.get("start_sample") != start_sample
        or segment.get("end_sample") != end_sample
        or segment.get("emitted_end_sample") != end_sample
        or segment.get("done_emitted") is not True
    ):
        return False
    completed = set(branch.get("client_consumed_segments", ()))
    if segment_id in completed:
        return False
    if not _commit_client_consumed_segment(branch, segment_id):
        return False
    branch["client_consumed_pcm_sequence"] = sequence
    branch["client_consumed_segments"] = tuple((*completed, segment_id))
    branch.setdefault("client_consumed_pcm_events", []).append({
        "response_id": response_id,
        "generation_id": generation_value,
        "speaker_segment_id": segment_id,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "sample_rate": sample_rate,
        "sequence": sequence,
        "consumption_semantics": "logical_client_device_buffer",
    })
    return True


def _commit_client_consumed_segment(branch: dict[str, Any], segment_id: str) -> bool:
    segments = branch.get("playback_segments")
    segment = segments.get(segment_id) if isinstance(segments, dict) else None
    if not isinstance(segment, dict):
        return False
    completed = set(branch.get("client_consumed_segments", ()))
    if segment_id in completed:
        return False
    packets = tuple(segment.get("token_packets", ()))
    cursor = int(branch.get(
        "spoken_prefix_exclusive_cursor",
        branch.get("spoken_prefix_base_cursor", branch.get("turn_start_cache_position", 0)),
    ))
    prefix_ids = list(int(value) for value in branch.get("spoken_prefix_token_ids", ()))
    prefix_parts = list(branch.get("spoken_prefix_text_parts", ()))
    for packet in packets:
        if not isinstance(packet, dict):
            return False
        token_id = packet.get("token_id")
        token_cursor = packet.get("exclusive_kv_cursor")
        text_piece = packet.get("text_piece")
        if (isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
                or isinstance(token_cursor, bool) or not isinstance(token_cursor, int)
                or token_cursor != cursor + 1 or not isinstance(text_piece, str)):
            return False
        prefix_ids.append(token_id)
        prefix_parts.append(text_piece)
        cursor = token_cursor
    branch["spoken_prefix_token_ids"] = tuple(prefix_ids)
    branch["spoken_prefix_exclusive_cursor"] = cursor
    branch["spoken_prefix_text_parts"] = tuple(prefix_parts)
    branch["spoken_prefix_text"] = "".join(prefix_parts)
    branch["client_consumed_pcm_alignment"] = "complete_segment_only; intra_segment_alignment_missing"
    branch.setdefault("client_consumed_pcm_metrics", {})["intra_segment_alignment_missing"] = True
    return True


def _record_emitted_pcm_packet(branch: dict[str, Any], packet: Any) -> None:
    """Track exact server-emitted sample ranges per speaker subsegment."""
    if not isinstance(branch, dict):
        return
    try:
        response_id = str(branch["response_id"])
        generation_id = int(branch["session_generation_id"])
        segment_id = str(packet.speaker_segment_id)
        packet_generation = int(packet.generation_id)
        start_sample = int(packet.start_sample)
        end_sample = int(packet.end_sample)
        packet_id = int(packet.packet_id)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("emitted PCM packet has no complete segment identity") from exc
    if (
        not response_id or not segment_id or packet_generation != generation_id
        or start_sample < 0 or end_sample <= start_sample or packet_id < 0
    ):
        raise RuntimeError("emitted PCM packet has an invalid sample range or identity")
    segments = branch.setdefault("playback_segments", {})
    if not isinstance(segments, dict):
        raise RuntimeError("playback segment registry is not a mapping")
    segment = segments.setdefault(segment_id, {
        "response_id": response_id,
        "generation_id": generation_id,
        "speaker_segment_id": segment_id,
        "start_sample": start_sample,
        "end_sample": None,
        "emitted_end_sample": start_sample,
        "sample_rate": CODEC_SAMPLE_RATE,
        "done_emitted": False,
        "token_packets": (),
        "packet_ids": [],
    })
    if (
        segment.get("response_id") != response_id
        or segment.get("generation_id") != generation_id
        or segment.get("speaker_segment_id") != segment_id
        or segment.get("sample_rate") != CODEC_SAMPLE_RATE
        or segment.get("done_emitted") is True
        or start_sample != int(segment.get("emitted_end_sample", -1))
    ):
        raise RuntimeError("emitted PCM packet is stale, non-contiguous, or follows segment_done")
    segment["emitted_end_sample"] = end_sample
    segment["packet_ids"].append(packet_id)


def _complete_client_pcm_segment(
    branch: dict[str, Any],
    *,
    speaker_segment_id: str,
    token_packets: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any] | None:
    """Close one emitted unit and return its wire marker with exact sample bounds."""
    if not isinstance(branch, dict):
        return None
    segment_id = str(speaker_segment_id)
    segments = branch.get("playback_segments")
    segment = segments.get(segment_id) if isinstance(segments, dict) else None
    if not isinstance(segment, dict) or segment.get("done_emitted") is True:
        return None
    start_sample = segment.get("start_sample")
    end_sample = segment.get("emitted_end_sample")
    generation_id = segment.get("generation_id")
    response_id = segment.get("response_id")
    if (
        isinstance(start_sample, bool) or not isinstance(start_sample, int)
        or isinstance(end_sample, bool) or not isinstance(end_sample, int)
        or end_sample <= start_sample
        or generation_id != branch.get("session_generation_id")
        or response_id != branch.get("response_id")
    ):
        return None
    owned_packets = tuple(dict(packet) for packet in token_packets)
    segment.update({
        "end_sample": end_sample,
        "done_emitted": True,
        "token_packets": owned_packets,
    })
    return {
        "type": "response.audio.segment_done",
        "response_id": str(response_id),
        "generation_id": int(generation_id),
        "speaker_segment_id": segment_id,
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
        "sample_rate": CODEC_SAMPLE_RATE,
    }


def _require_closed_tts_codec_eos(metrics: Mapping[str, Any]) -> None:
    """Require sampled codec EOS before a closed TTS unit can advance KV."""
    if not isinstance(metrics, Mapping) or metrics.get("eos_detected") is not True:
        raise RuntimeError("Qwen Talker closed-unit segment did not reach codec EOS")


def _turn_local_vocalized_prefix_snapshot(branch: dict[str, Any]) -> tuple[int | None, str]:
    lock = branch.get("spoken_prefix_lock")
    if lock is None:
        return None, str(branch.get("spoken_prefix_text", ""))
    with lock:
        position = branch.get("spoken_prefix_kv_position")
        return (None if position is None else int(position), str(branch.get("spoken_prefix_text", "")))


def _wait_for_listener_backlog_admission(
    queue_depth: Callable[[], int],
    generation_is_current: Callable[[], bool],
    *,
    callbacks_active: bool = False,
    max_wait_s: float = 0.160,
    poll_interval_s: float = 0.0005,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
) -> ListenerBacklogAdmission:
    """Wait briefly for queued listener work, not host callback bookkeeping.

    The listener-priority model lock arbitrates actual GPU work. The scheduler's
    callbacks_active flag includes CPU/event-loop work and can remain active for
    the whole microphone stream, so it must not gate a Thinker prefill.
    """
    del callbacks_active  # Diagnostic state only; the bounded queue is authoritative.
    deadline = now() + max_wait_s
    while True:
        if not generation_is_current():
            return ListenerBacklogAdmission.ABORTED
        if queue_depth() <= 1:
            return ListenerBacklogAdmission.ADMITTED
        remaining = deadline - now()
        if remaining <= 0:
            # Admit one model quantum; the shared fair lock yields the next turn
            # to any listener waiter before another compute quantum is granted.
            return ListenerBacklogAdmission.ADMITTED
        sleep(min(poll_interval_s, remaining))


def _snapshot_direct_policy_audio(
    session: DuplexSession,
    *,
    action: ControlAction,
    first_frame_id: int,
    last_frame_id: int,
    user_active: bool | None = None,
    completed_utterance: bool = False,
    source_cutoff_sample_id: int | None = None,
) -> dict[str, Any]:
    """Return a complete causal policy-audio span with its absolute timeline."""
    first = int(first_frame_id)
    last = int(last_frame_id)
    is_active = (
        action == ControlAction.BACKCHANNEL
        if user_active is None
        else bool(user_active)
    )
    if first < 0 or last < first:
        raise _direct_audio_snapshot_error(
            session,
            "span audio utente diretto non valido",
            requested_first=first,
            requested_last=last,
            active=is_active,
        )
    consumed_through = int(session.policy_audio_consumed_through_frame_id)
    fresh_first = max(first, consumed_through + 1) if is_active else first
    if fresh_first > last:
        raise _direct_audio_snapshot_error(
            session,
            "snapshot PCM e span causale non allineati "
            f"(pcm=0, expected=0, first={first}, last={last}, "
            f"consumed_through={consumed_through})",
            requested_first=first,
            requested_last=last,
            active=is_active,
        )
    max_snapshot_frames = max(1, int(DEFAULT_MAX_AUDIO_MS) // POLICY_FRAME_MS)
    use_completed_pin = bool(
        action == ControlAction.SPEAK and completed_utterance and not is_active
    )
    actual_first = (
        first
        if use_completed_pin
        else max(fresh_first, last - max_snapshot_frames + 1)
    )
    target_start_ms = float((last + 1) * POLICY_FRAME_MS)
    user_start_ms = float(actual_first * POLICY_FRAME_MS)
    user_end_ms = None if is_active else float((last + 1) * POLICY_FRAME_MS)
    try:
        window = causal_audio_window(
            action=action.value,
            target_start_ms=target_start_ms,
            current_user_start_ms=user_start_ms,
            current_user_end_ms=user_end_ms,
            current_user_active=True if is_active else False,
            allow_active_speak=bool(action == ControlAction.SPEAK and is_active),
            allow_active_replan=bool(action == ControlAction.REPLAN and is_active),
        )
    except ValueError as exc:
        raise _direct_audio_snapshot_error(
            session,
            str(exc),
            requested_first=first,
            requested_last=last,
            active=is_active,
        ) from exc
    if use_completed_pin:
        pcm = (
            None
            if source_cutoff_sample_id is None
            else session.snapshot_completed_utterance_audio(
                actual_first,
                last,
                source_cutoff_sample_id=int(source_cutoff_sample_id),
                generation_id=int(session.session_generation_id),
            )
        )
        if pcm is None:
            raise _direct_audio_snapshot_error(
                session,
                "completed utterance pin non disponibile per span, generatione o cutoff esatti",
                requested_first=first,
                requested_last=last,
                active=is_active,
                snapshot_first=actual_first,
            )
    else:
        pcm = session.snapshot_policy_audio_range(actual_first, last)
    expected_samples = int(round(
        (float(window["audio_end_ms"]) - float(window["audio_start_ms"]))
        * USER_SAMPLE_RATE / 1000.0
    ))
    if pcm.size != expected_samples:
        raise _direct_audio_snapshot_error(
            session,
            "snapshot PCM e span causale non allineati "
            f"(pcm={pcm.size}, expected={expected_samples}, first={first}, last={last})",
            requested_first=first,
            requested_last=last,
            active=is_active,
            snapshot_first=actual_first,
        )
    return {
        "_direct_audio_pcm": pcm,
        "_direct_audio_first_frame_id": actual_first,
        "_direct_audio_requested_first_frame_id": first,
        "_direct_audio_prefix_frames_clipped": actual_first - fresh_first,
        "_direct_audio_consumed_prefix_frames": fresh_first - first,
        "_direct_audio_sample_count": int(pcm.size),
        "_direct_audio_last_frame_id": last,
        "_direct_audio_target_frame_id": last,
        "_direct_audio_start_ms": float(window["audio_start_ms"]),
        "_direct_audio_end_ms": float(window["audio_end_ms"]),
        "_direct_audio_active": is_active,
        "_direct_audio_window": dict(window),
    }


def _manual_commit_speak_owner_for_span(
    session: DuplexSession,
    first_frame_id: int,
    last_frame_id: int,
    *,
    direct_audio_candidate: bool,
    auto_response: bool,
    committed_audio_size: int,
) -> tuple[bool, int | None]:
    """Resolve manual SPEAK once and bind it only to an onset in its audio span."""
    manual_speak = bool(
        direct_audio_candidate and int(committed_audio_size) > 0 and not auto_response
    )
    if not manual_speak:
        return False, None
    first = int(first_frame_id)
    last = int(last_frame_id)
    if first < 0 or last < first:
        return True, None
    with session._generation_lock:
        onset = session.pending_user_onset_frame_id
        if onset is None or not first <= onset <= last:
            return True, None
        if not session.reserve_user_onset_for_response(onset):
            return True, None
        return True, int(onset)


class DirectAudioSnapshotError(ValueError):
    """A fail-closed direct-audio snapshot error with bounded frame diagnostics."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def _direct_audio_snapshot_error(
    session: DuplexSession,
    message: str,
    *,
    requested_first: int,
    requested_last: int,
    active: bool,
    snapshot_first: int | None = None,
) -> DirectAudioSnapshotError:
    """Build bounded scalar diagnostics without exposing or copying PCM."""
    max_snapshot_frames = max(1, int(DEFAULT_MAX_AUDIO_MS) // POLICY_FRAME_MS)
    with session._generation_lock:
        retained_ids = session.policy_audio_frame_ids
        retained_count = len(retained_ids)
        retained_first = int(retained_ids[0]) if retained_count else None
        retained_last = int(retained_ids[-1]) if retained_count else None
        consumed_through = int(session.policy_audio_consumed_through_frame_id)
        fresh_first = (
            max(requested_first, consumed_through + 1)
            if active
            else requested_first
        )
        effective_first = snapshot_first if snapshot_first is not None else max(
            fresh_first,
            requested_last - max_snapshot_frames + 1,
        )
        bounded_last = min(requested_last, effective_first + max_snapshot_frames - 1)
        expected_ids = (
            range(effective_first, bounded_last + 1)
            if effective_first <= bounded_last
            else range(0)
        )
        expected_count = len(expected_ids)
        frame_counts = Counter(
            int(frame_id)
            for frame_id in retained_ids
            if effective_first <= int(frame_id) <= bounded_last
        )
        missing_count = sum(1 for frame_id in expected_ids if frame_counts.get(frame_id, 0) == 0)
        missing_frame_ids = [
            frame_id
            for frame_id in expected_ids
            if frame_counts.get(frame_id, 0) == 0
        ][:16]
        return DirectAudioSnapshotError(message, {
            "onset_frame_id": requested_first,
            "requested_first_frame_id": requested_first,
            "requested_last_frame_id": requested_last,
            "active": bool(active),
            "consumed_through_frame_id": consumed_through,
            "retained_first_frame_id": retained_first,
            "retained_last_frame_id": retained_last,
            "retained_frame_count": retained_count,
            "snapshot_first_frame_id": effective_first,
            "snapshot_last_frame_id": bounded_last if expected_count else None,
            "expected_snapshot_frame_count": expected_count,
            "missing_frame_count": missing_count,
            "missing_frame_ids": missing_frame_ids,
            "missing_frame_ids_truncated": missing_count > len(missing_frame_ids),
        })


def _direct_audio_snapshot_event_fields(error: BaseException) -> dict[str, Any]:
    diagnostics = getattr(error, "diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        return {}
    return {f"direct_audio_snapshot_{key}": value for key, value in diagnostics.items()}


def _resolve_p7_session_voice(
    engine: Any,
    *,
    voice: str | None,
    voice_style: str | None = None,
) -> dict[str, Any] | None:
    """Resolve and validate the session voice before the mic gate is opened."""
    enforce_training_voice = bool(getattr(engine, "enforce_v9_training_voice", False))
    ablate_voice_conditioning = bool(getattr(engine, "ablate_session_voice_conditioning", False))
    if enforce_training_voice and ablate_voice_conditioning:
        raise ValueError("ablation voce e enforcement V9 sono incompatibili")
    if ablate_voice_conditioning:
        return None
    talker = getattr(engine, "talker", None)
    if talker is None:
        if enforce_training_voice:
            raise ValueError("enforcement V9 richiede un Talker caricato")
        # Lightweight protocol tests and text-only embedding engines do not
        # have a P7 Talker to validate.
        return None
    if enforce_training_voice:
        default_conditioning = getattr(talker, "default_voice_conditioning", None)
        if not isinstance(default_conditioning, dict):
            raise ValueError("il conditioning x-vector training V9 non è caricato")
        conditioning = dict(default_conditioning)
        reference_path = _verify_v9_training_reference(conditioning)
        requested_profile = resolve_voice_conditioning(
            voice or DEFAULT_VOICE,
            voice_style=voice_style,
        )
        if not isinstance(requested_profile, dict) or requested_profile.get("mode") != "x_vector_only":
            raise ValueError("il profilo voce predefinito deve essere x_vector_only")
        training_embedding = conditioning.get("speaker_embedding")
        profile_embedding = requested_profile.get("speaker_embedding")
        if (
            not isinstance(training_embedding, torch.Tensor)
            or not isinstance(profile_embedding, torch.Tensor)
            or training_embedding.numel() != profile_embedding.numel()
            or not torch.allclose(
                training_embedding.detach().float().cpu().reshape(-1),
                profile_embedding.detach().float().cpu().reshape(-1),
                rtol=1e-5,
                atol=1e-6,
            )
        ):
            raise ValueError(
                "il profilo x-vector richiesto non coincide con il reference voice V9 bloccato"
            )
        conditioning["speaker_embedding"] = profile_embedding.detach().float().cpu().contiguous()
        conditioning["conditioning_source"] = "training_default"
    else:
        conditioning = resolve_voice_conditioning(voice, voice_style=voice_style)
        if conditioning is None:
            conditioning = getattr(talker, "default_voice_conditioning", None)
    if not isinstance(conditioning, dict) or conditioning.get("mode") != "x_vector_only":
        raise ValueError("P7 richiede un profilo voce x_vector_only valido")
    embedding = conditioning.get("speaker_embedding")
    if not isinstance(embedding, torch.Tensor):
        raise ValueError("speaker_embedding deve essere un tensor PyTorch")
    hidden_size = int(getattr(getattr(talker, "tcfg", None), "hidden_size", 0))
    if hidden_size <= 0 or embedding.numel() != hidden_size:
        raise ValueError(
            f"speaker_embedding dim {embedding.numel()} != Talker hidden size {hidden_size}"
        )
    if not bool(torch.isfinite(embedding).all()):
        raise ValueError("speaker_embedding contiene valori non finiti")
    if conditioning.get("ref_code") is not None:
        raise ValueError("P7 x-vector-only non deve contenere ref_code")
    return conditioning


def _resolve_request_voice_conditioning(
    engine: Any,
    sess: Any,
    request: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve per-request/session voice, unless the startup ablation is active."""
    enforce_training_voice = bool(getattr(engine, "enforce_v9_training_voice", False))
    ablate_voice_conditioning = bool(getattr(engine, "ablate_session_voice_conditioning", False))
    if enforce_training_voice and ablate_voice_conditioning:
        raise ValueError("ablation voce e enforcement V9 sono incompatibili")
    if ablate_voice_conditioning:
        return None
    if enforce_training_voice:
        conditioning = getattr(sess, "voice_conditioning", None)
        if (
            not isinstance(conditioning, dict)
            or conditioning.get("conditioning_source") != "training_default"
            or conditioning.get("mode") != "x_vector_only"
        ):
            raise ValueError("session voice non è il training default V9 verificato")
        _verify_v9_training_reference(conditioning)
        return conditioning
    explicit_voice = request.get("voice") or request.get("speaker")
    if (
        explicit_voice is None
        and not request.get("voice_style")
        and not request.get("ref_audio")
        and getattr(sess, "voice_conditioning", None) is not None
    ):
        return sess.voice_conditioning
    return resolve_voice_conditioning(
        explicit_voice or getattr(sess, "voice", None) or DEFAULT_VOICE,
        request.get("voice_style") or sess.voice_style,
        request.get("ref_audio"), request.get("ref_text"),
    )


@contextmanager
def _diagnostic_talker_rng_scope(engine: Any):
    """Expose startup seed metadata without mutating process-global RNG state."""
    yield getattr(engine, "diagnostic_talker_seed", None)


def _append_native_action_token(chat: str, action: str | None) -> str:
    """Append the fixed native action token to an assistant generation prompt."""
    if action is None:
        return chat
    normalized_action = str(action).strip().upper()
    token = _ACTION_TOKEN_MAP.get(normalized_action)
    if token and not chat.rstrip().endswith(token):
        return chat + token + " "
    return chat


def _prepare_direct_audio_processor_inputs(
    engine: Any,
    *,
    language: Any,
    history_messages: list[dict[str, Any]],
    pcm: np.ndarray,
    response_protocol: str | None = None,
    session_instructions: str | None = None,
    controller_action: str | None = None,
) -> tuple[list[int], dict[str, Any], float]:
    """Build processor-native Gemma multimodal input without a transcript/action label."""
    processor = getattr(getattr(engine, "_thinker_wrapper", None), "processor", None)
    if processor is None or not callable(processor):
        raise RuntimeError("processor audio Gemma non disponibile per il Thinker diretto")
    audio_marker = (
        getattr(processor, "audio_token", None)
        or getattr(engine.tokenizer, "audio_token", None)
        or "<|audio|>"
    )
    completed_history = [
        message for message in history_messages
        if message.get("role") in {"user", "assistant"}
    ]
    normalized_protocol = normalize_audio_response_protocol(response_protocol)
    if normalized_protocol == AUDIO_RESPONSE_PROTOCOL_V2:
        prompt_messages = build_audio_response_protocol_v2_messages(
            history_messages=completed_history,
            audio_token=str(audio_marker),
            session_instructions=session_instructions,
        )
    elif normalized_protocol == CONTROLLER_GUIDED_SAY_V1:
        prompt_messages = build_controller_guided_say_v1_messages(
            language=language,
            history_messages=completed_history,
            audio_token=str(audio_marker),
            session_instructions=session_instructions,
            action=controller_action or "",
        )
    else:
        prompt_messages = build_audio_response_messages(
            language=language,
            history_messages=completed_history,
            audio_token=str(audio_marker),
        )
    chat = engine.tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    if normalized_protocol != CONTROLLER_GUIDED_SAY_V1:
        chat = _append_native_action_token(chat, controller_action)
    started = time.perf_counter()
    processor_audio_inputs = processor(
        text=[chat],
        audio=[np.asarray(pcm, dtype=np.float32).reshape(-1)],
        sampling_rate=USER_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    if not isinstance(processor_audio_inputs, dict):
        processor_audio_inputs = dict(processor_audio_inputs)
    required_audio_keys = {"input_ids", "input_features", "input_features_mask"}
    if not required_audio_keys.issubset(processor_audio_inputs):
        raise RuntimeError(
            "processor Gemma non ha prodotto le chiavi audio richieste: "
            f"{sorted(required_audio_keys - set(processor_audio_inputs))}"
        )
    input_ids = processor_audio_inputs["input_ids"]
    if getattr(input_ids, "ndim", None) != 2 or input_ids.shape[0] != 1:
        raise RuntimeError("processor Gemma deve produrre input_ids con batch 1")
    ids = [int(value) for value in input_ids[0].tolist()]
    tensors = {
        key: processor_audio_inputs[key]
        for key in ("input_ids", "attention_mask", "input_features", "input_features_mask")
        if key in processor_audio_inputs
    }
    return ids, tensors, (time.perf_counter() - started) * 1000.0


def _build_unified_listener_prefix_token_ids(
    engine: Any,
    session: DuplexSession,
    *,
    instruction_override: str | None = None,
) -> tuple[int, ...]:
    """Serialize the real session system/history/user-audio prefix through BOA.

    This only uses the tokenizer and the processor's audio marker. It does not
    call the audio processor, audio tower, or any cache forward.
    """
    tokenizer = getattr(engine, "tokenizer", None)
    processor = getattr(getattr(engine, "_thinker_wrapper", None), "processor", None)
    if tokenizer is None or not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise RuntimeError("unified listener prefix requires the loaded Gemma tokenizer")
    if not callable(getattr(tokenizer, "encode", None)):
        raise RuntimeError("unified listener prefix requires tokenizer.encode")
    audio_marker = (
        getattr(processor, "audio_token", None)
        or getattr(tokenizer, "audio_token", None)
        or "<|audio|>"
    )
    audio_marker = str(audio_marker).strip()
    if not audio_marker:
        raise RuntimeError("unified listener prefix requires the processor's audio marker")
    history = [
        dict(message)
        for message in getattr(session, "history", ())
        if isinstance(message, dict)
        and message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]
    protocol = _require_unified_stable_audio_protocol(
        _configured_audio_response_protocol(engine)
    )
    session_instructions = (
        getattr(session, "instructions", None)
        if instruction_override is None
        else instruction_override
    )
    if protocol == CONTROLLER_GUIDED_SAY_V1:
        messages = build_audio_response_messages(
            language=getattr(session, "language", None),
            history_messages=history,
            audio_token=audio_marker,
        )
        static_say_instruction = (
            "Il controller ha già selezionato l'azione della risposta. Genera soltanto "
            "il testo da pronunciare, senza etichette o struttura JSON. Non riclassificare "
            "l'azione e non inventare parole che l'audio non supporta."
        )
        stable_system_parts = [
            part for part in (str(session_instructions or "").strip(), static_say_instruction)
            if part
        ]
        messages[0]["content"] = "\n\n".join(stable_system_parts)
    else:
        messages = build_audio_response_protocol_v2_messages(
            history_messages=history,
            audio_token=audio_marker,
            session_instructions=session_instructions,
        )

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    token_ids = tokenizer.encode(str(rendered), add_special_tokens=False)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if not isinstance(token_ids, list) or (token_ids and isinstance(token_ids[0], list)):
        raise TypeError("tokenizer.encode must return one flat token-ID sequence")
    token_ids = [int(token_id) for token_id in token_ids]
    marker_ids = tokenizer.encode(audio_marker, add_special_tokens=False)
    if hasattr(marker_ids, "tolist"):
        marker_ids = marker_ids.tolist()
    if isinstance(marker_ids, tuple):
        marker_ids = list(marker_ids)
    if not isinstance(marker_ids, list) or len(marker_ids) != 1:
        raise RuntimeError("processor audio marker must map to one tokenizer token")
    marker_id = int(marker_ids[0])
    marker_positions = [index for index, token_id in enumerate(token_ids) if token_id == marker_id]
    if not marker_positions:
        raise RuntimeError("chat template did not preserve the final user audio marker")
    boa_token_id = getattr(getattr(session, "listener", None), "backend", None)
    boa_token_id = getattr(boa_token_id, "boa_token_id", None)
    if boa_token_id is None:
        model_config = getattr(getattr(engine, "_thinker_wrapper", None), "config", None)
        boa_token_id = getattr(model_config, "boa_token_id", None)
    if isinstance(boa_token_id, bool) or not isinstance(boa_token_id, int) or boa_token_id < 0:
        raise RuntimeError("unified listener prefix requires Gemma's configured BOA token ID")
    try:
        return validate_native_control6_prefix_ids(
            token_ids[:marker_positions[-1]] + [int(boa_token_id)],
            boa_token_id=int(boa_token_id),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "unified listener system/user/BOA prefix exceeds native prefix capacity: "
            f"{len(token_ids[:marker_positions[-1]]) + 1}>{NATIVE_CONTROL6_PREFIX_MAX_TOKENS}"
        ) from exc


def _unified_instruction_refresh_ineligible_reason(
    session: DuplexSession,
    binding: Mapping[str, Any],
    dispatcher: Any,
) -> str | None:
    """Require a truly untouched frame-zero session before changing its text prefix."""
    state = getattr(session, "listener_state", None)
    if state is None:
        return "unified listener state is unavailable"
    if dispatcher is None or not callable(getattr(dispatcher, "has_pending_work", None)):
        return "unified listener dispatcher is unavailable"
    if binding.get("generation") is None or binding.get("batch_cache_position") is None:
        return "unified listener generation or cache cursor is unavailable"

    with session._generation_lock:
        if session.active_response or session.policy_response_pending:
            return "unified KV instructions can only change before an assistant turn starts"
    with state._lock:
        if (
            int(state.next_frame_id) != 0
            or int(state.listener_input_frames) != 0
            or int(state.listener_state_version) != 0
            or int(state.gemma_lm_audio_frames) != 0
            or bool(state.semantic_audio_soft_history)
        ):
            return "unified KV instructions can only change before the first audio frame"
    ingress = getattr(session, "audio_ingress", None)
    stats_method = getattr(ingress, "stats", None)
    if not callable(stats_method):
        return "unified audio ingress state is unavailable"
    ingress_stats = stats_method()
    if not isinstance(ingress_stats, dict):
        return "unified audio ingress state is unavailable"
    if (
        int(ingress_stats.get("input_samples_received", 0)) != 0
        or int(ingress_stats.get("input_frames_enqueued", 0)) != 0
        or int(getattr(ingress, "qsize", 0)) != 0
        or int(getattr(ingress, "pending_samples", 0)) != 0
    ):
        return "unified KV instructions can only change before audio is admitted"
    if dispatcher.has_pending_work(session.session_id):
        return "unified KV instructions cannot change while a listener frame is pending"
    return None


async def _refresh_unified_prefix_for_instruction_update(
    engine: Any,
    session: DuplexSession,
    *,
    instructions: str,
    worker: Any,
    binding: dict[str, Any],
    dispatcher: Any,
) -> dict[str, Any]:
    """Atomically replace a unified frame-zero prefix for a changed instruction."""
    requested = str(instructions)
    old_instructions = str(session.instructions)
    if requested == old_instructions:
        return {"status": "unchanged", "prefix_len": int(binding.get("batch_cache_position", 0))}
    reason = _unified_instruction_refresh_ineligible_reason(session, binding, dispatcher)
    if reason is not None:
        raise ValueError(reason)
    refresh = getattr(worker, "refresh_unified_prefix_before_audio", None)
    if not callable(refresh):
        raise RuntimeError("unified listener owner does not support instruction-prefix refresh")

    with session._generation_lock:
        if str(session.instructions) != old_instructions:
            raise RuntimeError("session instructions changed while preparing the unified prefix")
        old_revision = int(session.semantic_context_revision)
        prefix_token_ids = _build_unified_listener_prefix_token_ids(
            engine,
            session,
            instruction_override=requested,
        )
        prefix_import = make_unified_prefix_import(
            session.listener.backend,
            prefix_token_ids,
        )
        expected_generation_id = int(binding["generation"])
        expected_cache_position = int(binding["batch_cache_position"])
        if not session.update_instructions(requested):
            raise RuntimeError("session instructions changed before prefix refresh")
        new_revision = int(session.semantic_context_revision)

    def rollback_instruction_revision() -> None:
        with session._generation_lock:
            if (
                str(session.instructions) == requested
                and int(session.semantic_context_revision) == new_revision
            ):
                session.instructions = old_instructions
                session.semantic_context_revision = old_revision

    def commit_prefix_binding() -> None:
        binding["batch_cache_position"] = len(prefix_token_ids)
        binding["semantic_prefix_importer"] = prefix_import

    owner_future = None
    try:
        owner_future = asyncio.wrap_future(refresh(
            session.session_id,
            prefix_token_ids,
            new_revision=new_revision,
            expected_generation_id=expected_generation_id,
            expected_cache_position=expected_cache_position,
            prefix_import=prefix_import,
        ))
        await asyncio.shield(owner_future)
    except asyncio.CancelledError:
        if owner_future is None:
            rollback_instruction_revision()
            raise
        try:
            await asyncio.shield(owner_future)
        except BaseException:
            rollback_instruction_revision()
            raise
        commit_prefix_binding()
        raise
    except BaseException:
        rollback_instruction_revision()
        raise
    commit_prefix_binding()
    return {
        "status": "refreshed",
        "prefix_len": len(prefix_token_ids),
        "revision": new_revision,
    }


def _apply_direct_replan_transaction(
    session: DuplexSession,
    speaker_gate: SpeakerGate,
    decision: PolicyDecision,
    *,
    first_frame_id: int,
    last_frame_id: int,
    user_active: bool,
) -> tuple[GateEvent, dict[str, Any]]:
    """Snapshot a real contiguous correction before invalidating its old response."""
    if decision.action != ControlAction.REPLAN:
        raise ValueError("transactional snapshot richiede REPLAN")
    with session._generation_lock:
        context = _snapshot_direct_policy_audio(
            session,
            action=ControlAction.REPLAN,
            first_frame_id=int(first_frame_id),
            last_frame_id=int(last_frame_id),
            user_active=bool(user_active),
        )
        # SpeakerGate invalidates the previous generation. It is deliberately
        # applied only after the exact input PCM and its frame origin are fixed.
        event = speaker_gate.apply(decision, session)
        if not event.accepted:
            return event, context
        return event, context


def _claim_response_cancel_notification(session: DuplexSession, response_id: str | None) -> bool:
    """Claim one cancellation terminal for the response currently owned by session."""
    if not response_id:
        return False
    with session._generation_lock:
        if session.response_id != response_id or session.response_cancel_notified:
            return False
        session.response_cancel_notified = True
        return True


def _claim_pending_yielded_v2_pcm_cancel(
    session: DuplexSession,
    branch: dict[str, Any],
    *,
    response_id: str,
    generation_id: int,
    speaker_segment_id: str,
) -> str:
    """Claim one YIELD terminal only while this exact response still owns PCM.

    Lock order matches the pending-PCM policy gate (session, then playback
    branch), so an ACK cannot race the terminal claim and a replacement
    response cannot inherit an older worker's cancellation intent.
    """
    if not response_id:
        return "stale"
    session_lock = getattr(session, "_generation_lock", None)
    branch_lock = branch.get("spoken_prefix_lock")
    if session_lock is None or branch_lock is None:
        return "stale"
    with session_lock:
        if (
            getattr(session, "response_id", None) != response_id
            or int(getattr(session, "session_generation_id", -1)) != int(generation_id)
        ):
            return "stale"
        if bool(getattr(session, "response_cancel_notified", False)):
            return "already_notified"
        with branch_lock:
            if (
                branch.get("response_id") != response_id
                or branch.get("session_generation_id") != int(generation_id)
                or branch.get("speaker_segment_id") != speaker_segment_id
            ):
                return "stale"
            if branch.get("playback_cancelled") is True:
                return "already_cancelled"
            if not _response_branch_has_pending_pcm_locked(
                branch,
                speaker_segment_id=speaker_segment_id,
            ):
                return "acknowledged"
            session.response_cancel_notified = True
            branch["playback_cancelled"] = True
    _cancel_response_branch_playback(branch)
    return "claimed"


def _direct_audio_prompt_history(
    messages: list[dict[str, Any]],
    *,
    history_is_committed: bool,
) -> list[dict[str, Any]]:
    """Exclude an API-supplied current transcript; preserve canonical session history."""
    history = [message for message in messages if message.get("role") in {"user", "assistant"}]
    if not history_is_committed and history and history[-1].get("role") == "user":
        history = history[:-1]
    return history


class _PrivateControlLeakError(RuntimeError):
    """A generated response echoed its ephemeral control-event prompt text."""

    def __init__(self, action: str, raw_text: str) -> None:
        super().__init__("generated output contains the private control-event marker")
        self.action = str(action)
        self.raw_text = str(raw_text)


def _listener_evidence_role(runtime_profile: str, *, native_runtime_enabled: bool) -> str:
    """Classify evidence from the live listener path, independent of Git state."""
    if runtime_profile == "production" and native_runtime_enabled:
        return "microphone_causality_evidence"
    return "fixture_causality_diagnostic"


def _yield_admission_rejection(
    *,
    active_response: bool,
    emitted_audio_frames: int,
    has_post_watermark_voice: bool,
) -> str | None:
    """Require audible response output and new user speech before YIELD."""
    if not active_response:
        return None
    if int(emitted_audio_frames) <= 0:
        return "assistant_audio_not_started"
    if not has_post_watermark_voice:
        return "no_new_user_audio"
    return None


_DEFERRED_NATIVE_SPEAK_SOURCE = "thinker_response_action_deferred_to_endpoint"


def _advance_pending_speak_on_new_onset(
    pending_speak: dict[str, Any] | None,
    onset_frame_id: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Retarget legacy pending speech, but fence a v2 decision to its source run."""
    if pending_speak is None or pending_speak.get("endpoint_frame_id") is None:
        return pending_speak, None
    if pending_speak.get("source") == _DEFERRED_NATIVE_SPEAK_SOURCE:
        return None, "dropped_stale_deferred_native_intent"
    updated = dict(pending_speak)
    updated["endpoint_frame_id"] = None
    updated["voice_run_onset_frame_id"] = int(onset_frame_id)
    return updated, "retargeted_pending_speak"


class _ResponseAudioAdmissionState:
    """Response watermark and sent-frame count guarded by policy_causality_lock."""

    def __init__(self) -> None:
        self.latest_input_frame_id = -1
        self.input_watermark = -1
        self.emitted_audio_frames = 0

    def update_input_frame_locked(self, frame_id: int) -> None:
        self.latest_input_frame_id = max(self.latest_input_frame_id, int(frame_id))

    def mark_audio_sent_locked(self, frames: int) -> None:
        if self.emitted_audio_frames == 0:
            self.input_watermark = self.latest_input_frame_id
        self.emitted_audio_frames += max(0, int(frames))

    def reset_locked(self) -> None:
        self.input_watermark = -1
        self.emitted_audio_frames = 0

    def snapshot_locked(self, speech_tracker: "_CausalUserSpeechTracker") -> tuple[int, bool]:
        has_post_watermark_voice = (
            self.input_watermark >= 0
            and speech_tracker.has_onset_after(self.input_watermark)
        )
        return self.emitted_audio_frames, has_post_watermark_voice


def _configured_speak_endpoint_frames() -> int:
    raw = os.environ.get("DUPLEX_SPEAK_ENDPOINT_QUIET_MS", str(DEFAULT_SPEAK_ENDPOINT_QUIET_MS))
    try:
        quiet_ms = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("DUPLEX_SPEAK_ENDPOINT_QUIET_MS deve essere un multiplo positivo di 80 ms") from exc
    if quiet_ms <= 0 or quiet_ms % POLICY_FRAME_MS:
        raise ValueError("DUPLEX_SPEAK_ENDPOINT_QUIET_MS deve essere un multiplo positivo di 80 ms")
    return quiet_ms // POLICY_FRAME_MS


class _CausalUserSpeechTracker:
    """Track voiced runs and endpoints from current, already-consumed PCM frames."""

    def __init__(self, *, endpoint_quiet_frames: int) -> None:
        if isinstance(endpoint_quiet_frames, bool) or int(endpoint_quiet_frames) < 1:
            raise ValueError("endpoint_quiet_frames must be a positive integer")
        self.endpoint_quiet_frames = int(endpoint_quiet_frames)
        self.last_voiced_frame_id: int | None = None
        self.run_onset_frame_id: int | None = None
        self.voice_run_active = False
        self.quiet_frames = 0
        self.endpoint_frame_id: int | None = None
        self.endpoint_source_cutoff_sample_id: int | None = None
        self.speech_complete = False

    def observe(self, frame_id: int, pcm: np.ndarray) -> tuple[bool, bool]:
        frame = np.asarray(pcm, dtype=np.float32).reshape(-1)
        voiced = False
        if frame.size and np.isfinite(frame).all():
            rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
            peak = float(np.max(np.abs(frame)))
            voiced = rms >= 0.001 and peak >= 0.005
        new_onset = voiced and not self.voice_run_active
        endpoint = False
        if voiced:
            if new_onset:
                self.run_onset_frame_id = int(frame_id)
                self.speech_complete = False
                self.endpoint_frame_id = None
                self.endpoint_source_cutoff_sample_id = None
            self.voice_run_active = True
            self.quiet_frames = 0
            self.last_voiced_frame_id = int(frame_id)
        elif self.voice_run_active:
            self.quiet_frames += 1
            if self.quiet_frames >= self.endpoint_quiet_frames:
                self.voice_run_active = False
                self.speech_complete = True
                self.endpoint_frame_id = int(frame_id)
                self.endpoint_source_cutoff_sample_id = (
                    int(frame_id) * int(frame.size) + int(frame.size)
                )
                endpoint = True
        return bool(new_onset), endpoint

    def has_onset_after(self, watermark_frame_id: int) -> bool:
        return (
            self.run_onset_frame_id is not None
            and self.run_onset_frame_id > int(watermark_frame_id)
        )

    def consume_endpoint(self) -> None:
        """Prevent one completed utterance from authorizing multiple SPEAKs."""
        self.speech_complete = False
        self.endpoint_frame_id = None
        self.endpoint_source_cutoff_sample_id = None


class _SchedulerEventBuffer:
    """Bounded thread-to-async bridge with one coalesced loop notification."""

    OVERFLOW = object()

    def __init__(self, loop: asyncio.AbstractEventLoop, *, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("scheduler event capacity must be positive")
        self._loop = loop
        self._maxsize = int(maxsize)
        self._events: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()
        self._wake = asyncio.Event()
        self._wake_scheduled = False
        self._closed = False
        self._overflowed = False

    @property
    def qsize(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def overflowed(self) -> bool:
        with self._lock:
            return self._overflowed

    def _schedule_wake(self) -> bool:
        try:
            self._loop.call_soon_threadsafe(self._wake.set)
        except RuntimeError:
            with self._lock:
                self._closed = True
                self._events.clear()
            return False
        return True

    def put_nowait(self, event: dict[str, Any]) -> bool:
        notify = False
        overflowed_now = False
        with self._lock:
            if self._closed or self._overflowed:
                return False
            if len(self._events) >= self._maxsize:
                self._overflowed = True
                overflowed_now = True
            else:
                self._events.append(dict(event))
            if not self._wake_scheduled:
                self._wake_scheduled = True
                notify = True
        notified = not notify or self._schedule_wake()
        return notified and not overflowed_now

    async def get(self) -> dict[str, Any] | object | None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            with self._lock:
                if self._overflowed:
                    self._events.clear()
                    return self.OVERFLOW
                if self._closed:
                    self._events.clear()
                    self._wake_scheduled = False
                    return None
                if self._events:
                    event = self._events.popleft()
                    if self._events:
                        # Keep the wake level-triggered while draining queued
                        # events; producers need not enqueue per-event callbacks.
                        self._wake.set()
                    else:
                        self._wake_scheduled = False
                    return event
                self._wake_scheduled = False

    def close(self) -> None:
        notify = False
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if not self._wake_scheduled:
                self._wake_scheduled = True
                notify = True
        if notify:
            self._schedule_wake()


def _silence_skip_inputs_are_exact_zero(user_pcm: Any, self_reference_pcm: Any) -> bool:
    """Skip only when both the microphone and visible self-audio windows are exact zero."""
    if user_pcm is None or self_reference_pcm is None:
        return False
    return not np.count_nonzero(np.asarray(user_pcm)) and not np.count_nonzero(
        np.asarray(self_reference_pcm)
    )


def _user_turn_is_unanswered(
    pending_onset_frame_id: int | None,
    last_answered_onset_frame_id: int | None,
) -> bool:
    return bool(
        pending_onset_frame_id is not None
        and (
            last_answered_onset_frame_id is None
            or int(pending_onset_frame_id) > int(last_answered_onset_frame_id)
        )
    )


def _silence_skip_is_true_idle(
    *,
    candidate: bool,
    query_coherent: bool,
    unanswered_user_turn: bool = False,
    active_response: bool = False,
    response_pending: bool = False,
    pending_plan: bool = False,
    control_phase_idle: bool = True,
) -> bool:
    """Permit exact-zero reuse only when no conversational work is outstanding."""
    return bool(
        candidate
        and query_coherent
        and not (
            unanswered_user_turn
            or active_response
            or response_pending
            or pending_plan
            or not control_phase_idle
        )
    )


def _silence_skip_transition_is_still_idle(
    *,
    snapshot_generation_id: int,
    current_generation_id: int,
    active_response: bool,
    response_pending: bool,
    control_state: Any,
) -> bool:
    """Reject cached listener work across any response-generation boundary."""
    phase = str(getattr(control_state, "value", control_state)).strip().upper()
    return bool(
        int(snapshot_generation_id) == int(current_generation_id)
        and not active_response
        and not response_pending
        and phase == ControlState.LISTENING.value
    )


def _scheduler_cuda_stream_context(stream: Any):
    if stream is None:
        return nullcontext()
    return torch.cuda.stream(stream)


def _listener_v9_stream_for_engine(engine: Any) -> Any | None:
    """Legacy listener-to-Qwen V9 stream is unused by frame-residual Q1 responses."""
    if getattr(engine, "frame_residual_adapter", None) is not None:
        return None
    return getattr(engine, "v9_stream", None)


def _silence_skip_query_is_coherent(session: DuplexSession, frame_id: int) -> tuple[bool, str | None]:
    """Only reuse a listener query when its hidden and frame provenance still match."""
    query = getattr(session, "listener_audio_queries", None)
    if not isinstance(query, torch.Tensor):
        return False, "query_unavailable"
    origin = getattr(session, "listener_audio_frame_id", None)
    if not isinstance(origin, int) or isinstance(origin, bool) or origin >= int(frame_id):
        return False, "query_origin_not_older"
    hidden = getattr(session, "listener_hidden", None)
    if not isinstance(hidden, torch.Tensor):
        return False, "hidden_unavailable"
    if getattr(session, "listener_hidden_frame_id", None) != origin:
        return False, "query_hidden_frame_mismatch"
    state_origin = getattr(getattr(session, "listener_state", None), "last_hidden_source_frame_id", None)
    if state_origin != origin:
        return False, "query_listener_state_frame_mismatch"
    return True, None


def _wav_b64(wav: np.ndarray, sr: int = CODEC_SAMPLE_RATE) -> str:
    import soundfile as sf
    buf = io.BytesIO(); sf.write(buf, np.asarray(wav, dtype=np.float32), sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _raw_source_audio_quality(pcm: np.ndarray) -> dict[str, Any]:
    """Measure source float PCM before WAV serialization can clip or quantize it."""
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


def _alignment_failure_diagnostics(
    *,
    generated_ids: list[int] | tuple[int, ...],
    visible_generated_ids: list[int],
    generated_text: str,
    hidden_prompt_len: int,
    raw_hidden: Any,
    say_hidden: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    """Describe a failed text/hidden alignment without weakening its gate."""
    raw_ids = [int(token_id) for token_id in generated_ids]
    pieces: list[dict[str, Any]] = []
    decode = getattr(tokenizer, "decode", None)
    for token_id in raw_ids:
        try:
            piece = str(decode([token_id], skip_special_tokens=False)) if callable(decode) else None
        except Exception as exc:
            piece = f"<decode-error:{type(exc).__name__}:{exc}>"
        pieces.append({"token_id": token_id, "piece": piece})

    def shape_of(value: Any) -> list[int] | None:
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        try:
            return [int(dim) for dim in shape]
        except (TypeError, ValueError):
            return None

    raw_shape = shape_of(raw_hidden)
    say_shape = shape_of(say_hidden)
    return {
        "generated_ids": raw_ids,
        "visible_generated_ids": [int(token_id) for token_id in visible_generated_ids],
        "generated_text_repr": repr(str(generated_text)),
        "per_token_decode_pieces": pieces,
        "hidden_prompt_len": int(hidden_prompt_len),
        "raw_hidden_shape": raw_shape,
        "raw_hidden_token_count": raw_shape[-2] if raw_shape and len(raw_shape) >= 2 else None,
        "say_hidden_shape": say_shape,
        "say_hidden_token_count": say_shape[-2] if say_shape and len(say_shape) >= 2 else None,
        "visible_generated_token_count": len(visible_generated_ids),
    }


def _align_generated_response_hidden_with_diagnostics(
    *,
    response_hidden: Any,
    visible_generated_ids: list[int],
    gemma_tokenizer: Any,
    qwen_tokenizer: Any,
    qwen_processor: Any,
    generated_text: str,
    generated_ids: list[int] | tuple[int, ...],
    hidden_prompt_len: int,
    raw_hidden: Any,
):
    try:
        return align_generated_response_hidden_to_qwen_tokens(
            response_hidden,
            visible_generated_ids,
            gemma_tokenizer=gemma_tokenizer,
            qwen_tokenizer=qwen_tokenizer,
            qwen_processor=qwen_processor,
            text=generated_text,
        )
    except Exception:
        try:
            diagnostics = _alignment_failure_diagnostics(
                generated_ids=generated_ids,
                visible_generated_ids=visible_generated_ids,
                generated_text=generated_text,
                hidden_prompt_len=hidden_prompt_len,
                raw_hidden=raw_hidden,
                say_hidden=response_hidden,
                tokenizer=gemma_tokenizer,
            )
            logger.exception(
                "h_txt generated-response alignment failed; diagnostics=%s",
                json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
            )
        except Exception as diagnostic_error:
            try:
                logger.exception(
                    "h_txt generated-response alignment failed; diagnostics unavailable: %s",
                    diagnostic_error,
                )
            except Exception:
                pass
        raise


def _message_text(content: Any) -> str:
    if isinstance(content, str): return content
    if isinstance(content, list):
        return " ".join(str(x.get("text", "")) for x in content if isinstance(x, dict)).strip()
    return ""


def _drain_queue(buffer: queue.Queue[Any] | None) -> int:
    if buffer is None:
        return 0
    drained = 0
    while True:
        try:
            buffer.get_nowait()
        except queue.Empty:
            return drained
        drained += 1


def _discard_fenced_response_output(
    session: Any,
    token: Any,
    *,
    playout: Any,
    audio_queue: queue.Queue[Any],
    text_queue: queue.Queue[Any],
    output_cancel: threading.Event,
    cancel_reason: dict[str, str | None],
) -> bool:
    """Drop queued tail audio when a destructive fence supersedes a safe YIELD."""
    if session.is_generation_current(token):
        return False
    output_cancel.set()
    with session._generation_lock:
        reason = getattr(session, "stop_reason", None)
    cancel_reason["value"] = str(reason or "generation_fenced")
    if playout is not None:
        playout.cancel()
    _drain_queue(audio_queue)
    _drain_queue(text_queue)
    return True


def apply_speaker_gate_event(
    event: GateEvent,
    *,
    engine: Any,
    playout: Any | None = None,
    packetizer: Any | None = None,
    audio_queue: queue.Queue[Any] | None = None,
    text_queue: queue.Queue[Any] | None = None,
    synthesis_active: bool = True,
    session: DuplexSession | None = None,
    expected_generation_id: int | None = None,
    stop_generation: Callable[[], Any] | None = None,
) -> dict[str, int | bool]:
    """Apply the action-specific stop contract at the speaker boundary.

    ``stop_synthesis`` and ``flush_audio_buffer`` are intentionally separate:
    LISTEN observes the active producer, YIELD stops it but drains accepted
    audio, and REPLAN/HARD_STOP discard pending old-generation output. The
    production scheduler passes the ``GateEvent`` returned by ``SpeakerGate.apply``;
    keeping this boundary typed prevents a raw policy decision from bypassing
    the FSM acceptance checks.
    """
    if not isinstance(event, GateEvent):
        raise TypeError("event deve essere GateEvent")
    action = event.action
    if expected_generation_id is not None and event.generation_id != expected_generation_id:
        raise ValueError("GateEvent generation fence incoerente")
    if session is not None and event.generation_id is not None and not session.is_generation_current(
        generation_id=event.generation_id,
        speaker_segment_id=event.speaker_segment_id,
    ):
        return {"accepted": False, "stop_synthesis": False, "flush_audio_buffer": False, "drained": 0}
    stop_synthesis = bool(event.accepted and event.stop_synthesis)
    flush_audio = bool(event.accepted and event.flush_audio_buffer)
    accepted = bool(event.accepted)

    if not accepted:
        return {"accepted": False, "stop_synthesis": False, "flush_audio_buffer": False, "drained": 0}
    active = event.previous_state in {
        ControlState.SPEAKING,
        ControlState.BACKCHANNELING,
        ControlState.STOPPING,
    }
    expected_stop = action == ControlAction.HARD_STOP or (
        active and action in {ControlAction.YIELD, ControlAction.REPLAN}
    )
    expected_flush = action == ControlAction.HARD_STOP or (
        active and action == ControlAction.REPLAN
    )
    if stop_synthesis != expected_stop or flush_audio != expected_flush:
        raise ValueError("GateEvent viola il contratto stop/flush P6")

    if stop_synthesis and synthesis_active:
        if stop_generation is not None:
            stop_generation()
        elif hasattr(engine, "stop_generation"):
            engine.stop_generation()
    drained = 0
    if flush_audio:
        stale_generation = event.stale_generation_id
        stale_segment = event.speaker_segment_id
        if playout is not None:
            if stale_generation is not None and stale_segment and hasattr(playout, "cancel_segment"):
                playout.cancel_segment(
                    generation_id=stale_generation,
                    speaker_segment_id=stale_segment,
                )
            elif hasattr(playout, "cancel"):
                # Compatibility fallback for P6 test doubles only. Production
                # SpeakerPlayoutBuffer always takes the segment-scoped branch.
                playout.cancel()
        if packetizer is not None and stale_generation is not None and stale_segment:
            if hasattr(packetizer, "cancel_segment"):
                packetizer.cancel_segment(
                    generation_id=stale_generation,
                    speaker_segment_id=stale_segment,
                )
        drained += _drain_queue(audio_queue)
        drained += _drain_queue(text_queue)
    return {
        "accepted": True,
        "stop_synthesis": stop_synthesis,
        "flush_audio_buffer": flush_audio,
        "drained": drained,
        "action": action.value,
    }


class _RealtimeResponseControl:
    """Thread-shared handles used by scheduler STOP/FLUSH callbacks."""

    def __init__(
        self,
        token: GenerationToken,
        playout: SpeakerPlayoutBuffer,
        packetizer: NetworkPacketizer,
        audio_queue: queue.Queue[AudioPacket],
        text_queue: queue.Queue[str],
    ) -> None:
        self.token = token
        self.playout = playout
        self.packetizer = packetizer
        self.audio_queue = audio_queue
        self.text_queue = text_queue
        self.segment_opened = False
        self.segment_close_requested = False
        self.segment_advances = 0
        self.backchannel_requested = False


class _EngineSynthesisOwner:
    """Fence legacy global stop and concurrent response-local cancellation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: tuple[str, str] | None = None
        self._parallel_owners: dict[tuple[str, str], Callable[[], Any]] = {}

    def claim(
        self,
        session_id: str,
        response_id: str,
        *,
        allow_concurrent: bool = False,
        cancel: Callable[[], Any] | None = None,
    ) -> None:
        owner = (str(session_id), str(response_id))
        with self._lock:
            if allow_concurrent:
                if self._owner is not None or len(self._parallel_owners) >= 2:
                    raise RuntimeError("capacità synthesis parallela esaurita")
                if owner in self._parallel_owners:
                    raise RuntimeError(f"synthesis già posseduta da {owner!r}")
                if cancel is None:
                    raise ValueError("parallel synthesis requires a response-local cancel callback")
                self._parallel_owners[owner] = cancel
                return
            if self._owner is not None or self._parallel_owners:
                raise RuntimeError(f"engine synthesis già posseduta da {self._owner!r}")
            self._owner = owner

    def release(self, session_id: str, response_id: str) -> None:
        owner = (str(session_id), str(response_id))
        with self._lock:
            if self._owner == owner:
                self._owner = None
            self._parallel_owners.pop(owner, None)

    def is_owned(self, session_id: str, response_id: str | None) -> bool:
        if response_id is None:
            return False
        with self._lock:
            owner = (str(session_id), str(response_id))
            return self._owner == owner or owner in self._parallel_owners

    def stop_if_owned(self, session_id: str, response_id: str, stop: Callable[[], Any]) -> bool:
        """Make owner validation and the shared engine stop one atomic action."""
        with self._lock:
            owner = (str(session_id), str(response_id))
            if owner in self._parallel_owners:
                self._parallel_owners[owner]()
                return True
            if self._owner != owner:
                return False
            stop()
            return True


class _InferenceExecutionGate:
    """Allow two pooled responses together while excluding legacy callers."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._legacy_active = False
        self._pooled_active = 0

    @contextmanager
    def legacy(self):
        with self._condition:
            while self._legacy_active or self._pooled_active:
                self._condition.wait()
            self._legacy_active = True
        try:
            yield
        finally:
            with self._condition:
                self._legacy_active = False
                self._condition.notify_all()

    @contextmanager
    def pooled(self):
        with self._condition:
            while self._legacy_active:
                self._condition.wait()
            self._pooled_active += 1
        try:
            yield
        finally:
            with self._condition:
                self._pooled_active -= 1
                self._condition.notify_all()


def create_app(engine: Any, session_mgr: SessionManager | None = None) -> FastAPI:
    app = FastAPI(title="Gemma-4-Omni Duplex Kernel", version="0.4.0")
    sessions = session_mgr or SessionManager()
    native_listener_cuda_profile_directory = native_listener_cuda_profile_effective_directory(
        os.environ.get(NATIVE_LISTENER_CUDA_PROFILE_DIRECTORY_ENV)
    )
    # Only the engine's trusted runtime profile can select compatibility mode;
    # the app factory has no public production bypass switch.
    enforce_production_listener = bool(getattr(engine, "enforce_production_listener", True))
    listener_silence_skip_enabled = bool(getattr(engine, "enable_listener_silence_skip", False))
    unified_kv_enabled = _strict_env_switch(_UNIFIED_KV_ENABLED_ENV)
    shared_listener_microbatch_enabled = unified_kv_enabled or (
        os.environ.get(_LISTENER_MICROBATCH_ENABLED_ENV, "0").strip() == "1"
    )
    if unified_kv_enabled:
        _require_unified_stable_audio_protocol(_configured_audio_response_protocol(engine))
        # The unified path has one native cache owner. The legacy semantic
        # shadow and fair listener handoff both create competing cache/lock
        # paths, so close them before the shared worker is constructed.
        os.environ[_SEMANTIC_SHADOW_ENABLED_ENV] = "0"
        thinker_lock = getattr(engine, "_thinker_lock", None)
        if hasattr(thinker_lock, "listener_pair_lock_handoff_enabled"):
            thinker_lock.listener_pair_lock_handoff_enabled = False

    def _native_semantic_prefix_enabled() -> bool:
        return bool(
            not unified_kv_enabled
            and getattr(engine, "native_control6_semantic_prefix_enabled", True)
        )
    inference_lock = threading.Lock()  # GPU model is single-stream by design.
    inference_execution_gate = _InferenceExecutionGate()
    engine_synthesis_owner = _EngineSynthesisOwner()
    shared_listener_lifecycle_lock = asyncio.Lock()
    shared_listener_worker_holder: dict[str, Any] = {}
    shared_listener_dispatcher_holder: dict[str, Any] = {}
    shared_listener_bindings: dict[str, dict[str, Any]] = {}
    shared_listener_context_ids = itertools.count(1)
    shared_listener_timeline_capture_lock = threading.Lock()
    shared_listener_timeline_batch_state: dict[str, Any] = {}

    def _capture_shared_listener_host_timeline(row: SharedListenerInput) -> dict[str, Any] | None:
        """Make at most one best-effort, nonblocking snapshot attempt per batch."""
        batch_id = getattr(row.tick, "listener_batch_id", None)
        if not isinstance(batch_id, str) or not batch_id:
            return None
        backend = shared_listener_worker_holder.get("backend")
        model_lock = getattr(backend, "_lock", None)
        return capture_host_timeline_nonblocking(
            shared_listener_timeline_capture_lock,
            model_lock,
            batch_id=batch_id,
            batch_size=getattr(row.tick, "listener_batch_size", None),
            batch_state=shared_listener_timeline_batch_state,
        )

    @contextmanager
    def _shared_listener_priority_context(rows: tuple[SharedListenerInput, ...]):
        backend = shared_listener_worker_holder.get("backend")
        if backend is None:
            raise RuntimeError("shared listener backend is not bound")
        for row in rows:
            binding = shared_listener_bindings.get(row.session_id)
            if binding is None or getattr(binding["session"].listener, "backend", None) is not backend:
                raise RuntimeError("shared listener batch contains an unbound or incompatible session")
        waiter_priority = getattr(backend, "_listener_waiter_priority", None)
        registered = getattr(waiter_priority, "registered", None)
        if callable(registered):
            with registered():
                yield
            return
        waiting = getattr(backend, "_waiting_for_lm", None)
        if waiting is not None:
            waiting.set()
        try:
            yield
        finally:
            if waiting is not None:
                waiting.clear()

    def _shared_listener_batch_rows(rows: tuple[SharedListenerInput, ...]):
        worker = shared_listener_worker_holder.get("worker")
        if worker is None:
            raise RuntimeError("shared listener worker is not running")
        backend = shared_listener_worker_holder.get("backend")
        if backend is None:
            raise RuntimeError("shared listener backend is not bound")
        frames: list[ListenerFrame] = []
        for row in rows:
            binding = shared_listener_bindings.get(row.session_id)
            if binding is None:
                raise RuntimeError(f"shared listener session is no longer registered: {row.session_id}")
            sess = binding["session"]
            if getattr(sess.listener, "backend", None) is not backend:
                raise RuntimeError("shared listener sessions must use one loaded Gemma backend")
            frame = row.frame
            if int(frame.frame_id) != int(sess.listener_state.next_input_frame_id):
                raise RuntimeError(
                    f"ingress/listener cursor mismatch for {row.session_id}: "
                    f"frame={frame.frame_id}, cursor={sess.listener_state.next_input_frame_id}"
                )
            policy_context_provider = binding.get("policy_context_provider")
            policy_context = (
                policy_context_provider(frame, row.tick)
                if callable(policy_context_provider)
                else None
            )
            if policy_context is not None:
                row.tick._policy_prehead_context = policy_context
            if (
                isinstance(sess.turn_policy, GemmaNativeControl6Policy)
                and _native_semantic_prefix_enabled()
            ):
                pending_provider = binding.get("pending_speak_provider")
                pending_speak = pending_provider() if callable(pending_provider) else None
                prefix_request_started_ns = time.perf_counter_ns()
                request = build_native_control6_semantic_prefix_request(
                    engine,
                    sess,
                    current_frame_id=int(frame.frame_id),
                    pending_speak=pending_speak,
                )
                row.tick.microbatch_semantic_prefix_request_ms = round(
                    max(0, time.perf_counter_ns() - prefix_request_started_ns) / 1_000_000.0,
                    3,
                )
                if request is not None:
                    binding["semantic_request_holder"]["request"] = request
                    binding["semantic_request_holder"]["listener_generation_id"] = int(
                        row.listener_generation_id
                    )
                    if not worker.stage_semantic_prefix(
                        request,
                        listener_generation_id=int(row.listener_generation_id),
                        sequence_position=int(request.listener_state_version),
                    ):
                        raise RuntimeError(
                            "semantic prefix shadow worker is not accepting this session snapshot"
                        )
                else:
                    with sess._generation_lock:
                        desired_revision = int(sess.semantic_context_revision)
                        with sess.listener_state._lock:
                            applied_revision = sess.listener_state.semantic_context_revision
                    if applied_revision is None or int(applied_revision) != desired_revision:
                        worker.invalidate_semantic_prefix(row.session_id)
            reference = sess.assistant_reference_buffer.visible_for(
                listener_frame_id=int(frame.frame_id)
            )
            logical_q_len = 2
            ephemeral_reason = None
            queue_audio_for_next_turn = False
            user_input_pending = False
            if unified_kv_enabled:
                with sess._generation_lock:
                    current_response_id = sess.response_id
                    user_input_pending = bool(
                        sess.pending_user_onset_frame_id is not None
                        or sess.reserved_user_onset_frame_id is not None
                    )
                (
                    logical_q_len,
                    ephemeral_reason,
                    queue_audio_for_next_turn,
                ) = _shared_listener_frame_queue_contract(
                    binding,
                    frame.pcm,
                    response_id=current_response_id,
                    user_input_pending=user_input_pending,
                )
                if os.environ.get("DUPLEX_NATIVE_CONTROL6_TRACE") == "1":
                    trace_count = int(binding.get("_control6_queue_trace_count", 0))
                    if trace_count < 30:
                        branch = _shared_listener_response_branch(binding, current_response_id)
                        logger.info(
                            "[control6_queue_handoff] callback seq=%d session=%s frame=%d "
                            "response=%s branch_found=%s queue_open=%s private=%s "
                            "logical_q=%d queue_next=%s user_pending=%s",
                            trace_count + 1,
                            row.session_id,
                            int(frame.frame_id),
                            current_response_id,
                            branch is not None,
                            branch.get("queue_open") if branch is not None else None,
                            branch.get("ephemeral_private") if branch is not None else None,
                            int(logical_q_len),
                            bool(queue_audio_for_next_turn),
                            bool(user_input_pending),
                        )
                        binding["_control6_queue_trace_count"] = trace_count + 1
            binding["references"][int(frame.frame_id)] = reference
            frames.append(ListenerFrame(
                session_id=row.session_id,
                generation_id=row.listener_generation_id,
                state_epoch_id=str(row.state_epoch_id or sess.listener_state.state_epoch_id),
                frame_id=int(frame.frame_id),
                ready_at_ns=int(frame.enqueue_ns),
                pcm=frame.pcm,
                logical_q_len=logical_q_len,
                ephemeral_reason=ephemeral_reason,
                queue_audio_for_next_turn=queue_audio_for_next_turn,
                reference_pcm=None if reference is None else reference.pcm,
                reference_frame_id=None if reference is None else int(reference.frame_id),
                policy_context=policy_context,
            ))
        return worker.submit_tick(frames)

    def _shared_listener_batch(rows: tuple[SharedListenerInput, ...]):
        if not unified_kv_enabled or not rows:
            return _shared_listener_batch_rows(rows)
        locks_by_id: dict[int, Any] = {}
        for row in rows:
            binding = shared_listener_bindings.get(row.session_id)
            lock = None if binding is None else binding.get("unified_turn_transition_lock")
            if lock is not None:
                locks_by_id[id(lock)] = lock
        locks = [locks_by_id[key] for key in sorted(locks_by_id)]
        for lock in locks:
            lock.acquire()
        try:
            return _shared_listener_batch_rows(rows)
        finally:
            for lock in reversed(locks):
                lock.release()

    def _shared_listener_row_result(
        row: SharedListenerInput,
        output: Any,
        _receipt: Any,
    ) -> None:
        row.tick.microbatch_row_callback_entry_ns = time.monotonic_ns()
        binding = shared_listener_bindings.get(row.session_id)
        if binding is None:
            raise RuntimeError(f"shared listener result has no live session: {row.session_id}")
        reference = binding["references"].pop(int(row.frame.frame_id), None)
        row.tick.microbatch_output = output
        row.tick.microbatch_reference = reference
        host_timeline = _capture_shared_listener_host_timeline(row)
        if host_timeline is not None:
            row.tick.microbatch_host_lock_timeline = host_timeline
            row.tick.microbatch_timeline_request_ns = host_timeline.get(
                "request_monotonic_ns"
            )
            row.tick.microbatch_timeline_capture_lock_acquired_ns = host_timeline.get(
                "capture_lock_acquired_monotonic_ns"
            )
            row.tick.microbatch_timeline_model_lock_acquired_ns = host_timeline.get(
                "model_lock_acquired_monotonic_ns"
            )
            row.tick.microbatch_timeline_snapshot_ns = host_timeline.get(
                "snapshot_monotonic_ns"
            )
            row.tick.microbatch_timeline_age_ms = host_timeline.get("age_ms")
            row.tick.microbatch_timeline_status = host_timeline.get("status")
            row.tick.microbatch_timeline_reason = host_timeline.get("reason")
        row.tick.microbatch_model_lock_wait_ms = getattr(
            output, "listener_model_lock_wait_ms", None
        )
        row.tick.microbatch_model_lock_hold_ms = getattr(
            output, "listener_model_lock_hold_ms", None
        )
        row.tick.microbatch_capacity_growth_ms = getattr(output, "capacity_growth_ms", None)
        row.tick.microbatch_frontend_encode_ms = getattr(output, "frontend_encode_ms", None)
        row.tick.microbatch_mel_prepare_ms = getattr(output, "mel_prepare_ms", None)
        row.tick.microbatch_frontend_copy_ms = getattr(output, "frontend_copy_ms", None)
        row.tick.microbatch_frontend_copy_cuda_ms = getattr(
            output, "frontend_copy_cuda_ms", None
        )
        row.tick.microbatch_audio_tower_lock_wait_ms = getattr(
            output, "audio_tower_lock_wait_ms", None
        )
        row.tick.microbatch_audio_tower_lock_hold_ms = getattr(
            output, "audio_tower_lock_hold_ms", None
        )
        row.tick.microbatch_audio_tower_workspace_lock_wait_ms = getattr(
            output, "audio_tower_workspace_lock_wait_ms", None
        )
        row.tick.microbatch_audio_tower_workspace_lock_hold_ms = getattr(
            output, "audio_tower_workspace_lock_hold_ms", None
        )
        row.tick.microbatch_audio_tower_row_copy_cuda_ms = getattr(
            output, "audio_tower_row_copy_cuda_ms", None
        )
        row.tick.microbatch_audio_tower_graph_submit_ms = getattr(
            output, "audio_tower_graph_submit_ms", None
        )
        row.tick.microbatch_audio_tower_cuda_ms = getattr(output, "audio_tower_cuda_ms", None)
        row.tick.microbatch_audio_tower_batch_cuda_span_ms = getattr(
            output, "audio_tower_batch_cuda_span_ms", None
        )
        row.tick.microbatch_audio_tower_execution_mode = getattr(
            output, "audio_tower_execution_mode", None
        )
        row.tick.microbatch_native_submit_ms = getattr(output, "native_submit_ms", None)
        row.tick.microbatch_native_cuda_ms = getattr(output, "native_cuda_ms", None)
        row.tick.microbatch_native_control6_raw_score_ms = getattr(
            output, "native_control6_raw_score_ms", None
        )
        decision_view = getattr(output, "decision_view", None)
        row.tick.microbatch_decision_view_fork_count = getattr(
            decision_view, "fork_count", None
        )
        row.tick.microbatch_decision_view_replayed_q2_count = getattr(
            decision_view, "replayed_q2_count", None
        )
        row.tick.microbatch_decision_view_cache_hit = getattr(
            decision_view, "cache_hit", None
        )
        row.tick.microbatch_decision_view_ms = getattr(
            decision_view, "decision_view_ms", None
        )
        row.tick.microbatch_decision_view_context_mode = getattr(
            decision_view, "context_mode", None
        )
        row.tick.microbatch_native_output_health_ms = getattr(
            output, "native_output_health_ms", None
        )
        health = getattr(output, "native_output_health", None)
        health_row = getattr(health, "row", None)
        if health is not None:
            row.tick.microbatch_native_output_health_checked = getattr(health, "checked", None)
            row.tick.microbatch_native_output_health_status = getattr(health, "status", None)
            row.tick.microbatch_native_output_health_nonfinite = getattr(health, "nonfinite", None)
            row.tick.microbatch_native_output_health_generation = getattr(health, "generation", None)
            row.tick.microbatch_native_output_health_session_generation = getattr(
                health_row, "session_generation", None
            )
            row.tick.microbatch_native_output_health_state_epoch_id = getattr(
                health_row, "state_epoch_id", None
            )
            row.tick.microbatch_native_output_health_row_frame_id = getattr(
                health_row, "frame_id", None
            )
            row.tick.microbatch_native_output_health_row_state_version = getattr(
                health_row, "listener_state_version", None
            )
            row.tick.microbatch_native_output_health_row_sequence_position = getattr(
                health_row, "sequence_position", None
            )
            row.tick.microbatch_native_output_health_row_cache_position = getattr(
                health_row, "cache_position", None
            )
            row.tick.microbatch_native_output_health_native_context_id = getattr(
                health_row, "native_context_id", None
            )
        row.tick.microbatch_semantic_prefix_build_ms = getattr(
            output, "semantic_prefix_build_ms", None
        )
        row.tick.microbatch_semantic_prefix_replay_ms = getattr(
            output, "semantic_prefix_replay_ms", None
        )
        row.tick.microbatch_semantic_prefix_import_ms = getattr(
            output, "semantic_prefix_import_ms", None
        )
        row.tick.microbatch_semantic_prefix_refresh_event = getattr(
            output, "semantic_prefix_refresh_event", False
        )
        row.tick.microbatch_semantic_prefix_refresh_event_id = getattr(
            output, "semantic_prefix_refresh_event_id", None
        )
        row.tick.microbatch_semantic_prefix_refresh_revision = getattr(
            output, "semantic_prefix_refresh_revision", None
        )
        row.tick.microbatch_semantic_prefix_len = getattr(
            output, "semantic_prefix_len", None
        )
        row.tick.microbatch_control6_policy_ms = getattr(output, "control6_policy_ms", None)
        row.tick.microbatch_cuda_completion_wait_ms = getattr(
            output, "cuda_completion_wait_ms", None
        )
        call_started_ns = time.monotonic_ns()
        scheduler = binding["session"].scheduler
        scheduler.dispatch_shared_frame(row.frame, row.tick)
        scheduler_error = getattr(scheduler, "error", None)
        if scheduler_error is not None:
            raise RuntimeError(
                f"shared listener row fanout failed for {row.session_id}: {scheduler_error}"
            ) from scheduler_error
        call_finished_ns = time.monotonic_ns()
        record_stage = binding.get("record_scheduler_stage")
        if callable(record_stage):
            record_stage(
                "scheduler_tick_wall_ms",
                max(0, call_finished_ns - call_started_ns) / 1_000_000.0,
            )
            processing_ns = max(0, call_finished_ns - int(row.tick.observed_ns or call_started_ns))
            record_stage("scheduler_tick_processing_wall_ms", processing_ns / 1_000_000.0)
            spans = getattr(row.tick, "_scheduler_callback_wall_ns", {})
            callback_ns = sum(max(0, int(value)) for value in spans.values()) if isinstance(spans, dict) else 0
            record_stage(
                "scheduler_tick_outside_callbacks_wall_ms",
                max(0, processing_ns - callback_ns) / 1_000_000.0,
            )
            if row.tick.microbatch_queue_wait_ms is not None:
                record_stage("shared_listener_queue_wait_ms", row.tick.microbatch_queue_wait_ms)
            if row.tick.microbatch_compute_ms is not None:
                record_stage("shared_listener_batch_compute_ms", row.tick.microbatch_compute_ms)
            if row.tick.microbatch_model_lock_wait_ms is not None:
                record_stage(
                    "shared_listener_model_lock_wait_ms",
                    row.tick.microbatch_model_lock_wait_ms,
                )
            if row.tick.microbatch_model_lock_hold_ms is not None:
                record_stage(
                    "shared_listener_model_lock_hold_ms",
                    row.tick.microbatch_model_lock_hold_ms,
                )
            for metric_name, metric_value in (
                ("shared_listener_capacity_growth_ms", row.tick.microbatch_capacity_growth_ms),
                ("shared_listener_frontend_encode_ms", row.tick.microbatch_frontend_encode_ms),
                ("shared_listener_mel_prepare_ms", row.tick.microbatch_mel_prepare_ms),
                ("shared_listener_frontend_copy_ms", row.tick.microbatch_frontend_copy_ms),
                (
                    "shared_listener_frontend_copy_cuda_ms",
                    row.tick.microbatch_frontend_copy_cuda_ms,
                ),
                (
                    "shared_listener_audio_tower_lock_wait_ms",
                    row.tick.microbatch_audio_tower_lock_wait_ms,
                ),
                (
                    "shared_listener_audio_tower_lock_hold_ms",
                    row.tick.microbatch_audio_tower_lock_hold_ms,
                ),
                (
                    "shared_listener_audio_tower_workspace_lock_wait_ms",
                    row.tick.microbatch_audio_tower_workspace_lock_wait_ms,
                ),
                (
                    "shared_listener_audio_tower_workspace_lock_hold_ms",
                    row.tick.microbatch_audio_tower_workspace_lock_hold_ms,
                ),
                (
                    "shared_listener_audio_tower_row_copy_cuda_ms",
                    row.tick.microbatch_audio_tower_row_copy_cuda_ms,
                ),
                (
                    "shared_listener_audio_tower_graph_submit_ms",
                    row.tick.microbatch_audio_tower_graph_submit_ms,
                ),
                ("shared_listener_audio_tower_cuda_ms", row.tick.microbatch_audio_tower_cuda_ms),
                (
                    "shared_listener_audio_tower_batch_cuda_span_ms",
                    row.tick.microbatch_audio_tower_batch_cuda_span_ms,
                ),
                ("shared_listener_native_submit_ms", row.tick.microbatch_native_submit_ms),
                ("shared_listener_native_cuda_ms", row.tick.microbatch_native_cuda_ms),
                (
                    "shared_listener_decision_view_ms",
                    row.tick.microbatch_decision_view_ms,
                ),
                (
                    "shared_listener_semantic_prefix_request_ms",
                    row.tick.microbatch_semantic_prefix_request_ms,
                ),
                (
                    "shared_listener_native_control6_raw_score_ms",
                    row.tick.microbatch_native_control6_raw_score_ms,
                ),
                (
                    "shared_listener_semantic_prefix_build_ms",
                    row.tick.microbatch_semantic_prefix_build_ms,
                ),
                (
                    "shared_listener_semantic_prefix_replay_ms",
                    row.tick.microbatch_semantic_prefix_replay_ms,
                ),
                (
                    "shared_listener_semantic_prefix_import_ms",
                    row.tick.microbatch_semantic_prefix_import_ms,
                ),
                ("shared_listener_control6_policy_ms", row.tick.microbatch_control6_policy_ms),
                (
                    "shared_listener_cuda_completion_wait_ms",
                    row.tick.microbatch_cuda_completion_wait_ms,
                ),
            ):
                if metric_value is not None:
                    record_stage(metric_name, metric_value)

    def _shared_listener_row_error(
        row: SharedListenerInput,
        error: BaseException,
        _receipt: Any,
    ) -> None:
        binding = shared_listener_bindings.get(row.session_id)
        if binding is None:
            return
        binding["references"].pop(int(row.frame.frame_id), None)
        scheduler = binding["session"].scheduler
        on_error = getattr(scheduler, "on_error", None)
        if callable(on_error):
            on_error(error)

    def _initial_semantic_prefix_ineligible_reason(
        sess: DuplexSession,
        binding: Mapping[str, Any],
        dispatcher: SharedListenerTickDispatcher,
        request: Any,
    ) -> str | None:
        """Recheck the frame-zero admission preconditions without CUDA work."""
        state = sess.listener_state
        ingress = sess.audio_ingress
        if request.current_frame_id != 0 or request.replay_q2 or request.replay_frame_ids:
            return "not_a_frame_zero_empty_history_request"
        with state._lock:
            if (
                request.state_epoch_id != state.state_epoch_id
                or request.listener_state_version != state.listener_state_version
                or state.next_frame_id != 0
                or state.listener_input_frames != 0
                or state.listener_state_version != 0
                or state.gemma_lm_audio_frames != 0
                or state.semantic_audio_soft_history
            ):
                return "listener_state_is_not_pristine"
        if dispatcher.has_pending_work(sess.session_id):
            return "dispatcher_has_pending_or_inflight_frame"
        ingress_stats = ingress.stats() if callable(getattr(ingress, "stats", None)) else None
        if not isinstance(ingress_stats, dict):
            return "ingress_state_unavailable"
        if (
            int(ingress_stats.get("input_samples_received", 0)) != 0
            or int(ingress_stats.get("input_frames_enqueued", 0)) != 0
            or int(getattr(ingress, "qsize", 0)) != 0
            or int(getattr(ingress, "pending_samples", 0)) != 0
        ):
            return "audio_arrived_before_initial_prefix_commit"
        with sess._generation_lock:
            if int(sess.semantic_context_revision) != request.semantic_context_revision:
                return "semantic_context_revision_changed"
            if sess.active_response or sess.policy_response_pending:
                return "response_or_policy_action_is_pending"
        if binding.get("generation") is None:
            return "listener_generation_binding_unavailable"
        return None

    async def _prime_shared_listener_initial_prefix(
        sess: DuplexSession,
    ) -> dict[str, Any]:
        """Move the initial semantic KV build off the GPU owner and import by CAS."""
        started_ns = time.perf_counter_ns()
        result: dict[str, Any] = {
            "status": "not_required",
            "reason": None,
            "duration_ms": 0.0,
            "prefix_len": None,
            "revision": None,
            "prefix_build_ms": None,
            "prefix_replay_ms": None,
            "import_ms": None,
            "audio_samples_received": 0,
            "audio_queue_depth_frames": 0,
            "audio_pending_samples": 0,
        }
        binding = shared_listener_bindings.get(sess.session_id)
        dispatcher = shared_listener_dispatcher_holder.get("dispatcher")
        worker = shared_listener_worker_holder.get("worker")
        if (
            binding is None
            or dispatcher is None
            or worker is None
            or not isinstance(sess.turn_policy, GemmaNativeControl6Policy)
            or not _native_semantic_prefix_enabled()
        ):
            return result

        def capture_audio_admission_state() -> None:
            stats = sess.audio_ingress.stats()
            result["audio_samples_received"] = int(stats.get("input_samples_received", 0))
            result["audio_queue_depth_frames"] = int(stats.get("queue_depth_frames", 0))
            result["audio_pending_samples"] = int(sess.audio_ingress.pending_samples)

        capture_audio_admission_state()

        pending_provider = binding.get("pending_speak_provider")
        request = build_native_control6_semantic_prefix_request(
            engine,
            sess,
            current_frame_id=int(sess.listener_state.next_input_frame_id),
            pending_speak=(pending_provider() if callable(pending_provider) else None),
        )
        if request is None:
            with sess._generation_lock:
                revision = int(sess.semantic_context_revision)
                busy = bool(sess.active_response or sess.policy_response_pending)
            with sess.listener_state._lock:
                already_current = (
                    sess.listener_state.semantic_context_revision == revision
                )
            result["status"] = (
                "already_current" if already_current else "deferred_to_first_frame"
            )
            if not already_current:
                result["reason"] = (
                    "response_or_policy_action_is_pending"
                    if busy else "semantic_prefix_snapshot_not_safe"
                )
            result["duration_ms"] = round(
                max(0, time.perf_counter_ns() - started_ns) / 1_000_000.0, 3
            )
            return result

        reason = _initial_semantic_prefix_ineligible_reason(
            sess, binding, dispatcher, request
        )
        if reason is not None:
            result.update(status="deferred_to_first_frame", reason=reason)
            capture_audio_admission_state()
            result["duration_ms"] = round(
                max(0, time.perf_counter_ns() - started_ns) / 1_000_000.0, 3
            )
            return result

        listener_state = sess.listener_state
        with listener_state._lock:
            expected_cache_position = int(binding.get("batch_cache_position", 34))
        prepared = await prepare_initial_semantic_prefix_async(
            sess.listener.backend,
            getattr(sess.listener.backend, "_lock", None),
            request,
            listener_generation_id=int(binding["generation"]),
            expected_cache_position=expected_cache_position,
        )

        # Preparation runs outside the owner so other conversations can keep
        # dispatching. Recheck every source that could make this snapshot stale
        # before queueing its private-slot import.
        reason = _initial_semantic_prefix_ineligible_reason(
            sess, binding, dispatcher, request
        )
        if reason is not None:
            result.update(status="deferred_to_first_frame", reason=reason)
            capture_audio_admission_state()
            result["duration_ms"] = round(
                max(0, time.perf_counter_ns() - started_ns) / 1_000_000.0, 3
            )
            return result

        owner_result = await asyncio.wrap_future(
            worker.prime_initial_prefix(sess.session_id, prepared)
        )
        if owner_result.status != "primed":
            result.update(
                status="deferred_to_first_frame",
                reason=owner_result.reason or "owner_compare_and_swap_rejected",
            )
            capture_audio_admission_state()
            result["duration_ms"] = round(
                max(0, time.perf_counter_ns() - started_ns) / 1_000_000.0, 3
            )
            return result

        # The owner has imported this cache even if session context changed
        # while its command was queued. Keep the native cursor truthful, but
        # only publish the request as applied after one atomic revision/cursor
        # compare-and-swap under generation -> listener-state lock order.
        binding["batch_cache_position"] = int(request.cache_position)
        applied = commit_initial_semantic_prefix_state(sess, request, prepared)
        holder = binding["semantic_request_holder"]
        if applied:
            holder["request"] = request
            holder["listener_generation_id"] = int(binding["generation"])
            mark_applied = getattr(worker, "mark_semantic_prefix_applied", None)
            if callable(mark_applied):
                mark_applied(
                    request,
                    listener_generation_id=int(binding["generation"]),
                )
        elif holder.get("request") is request:
            holder["request"] = None
        result.update(
            status="primed" if applied else "ineligible",
            reason=None if applied else "context_changed_after_owner_import",
            prefix_len=int(request.prefix_len),
            revision=int(request.semantic_context_revision),
            import_ms=owner_result.import_ms,
            prefix_build_ms=prepared.prefix.prefix_build_ms,
            prefix_replay_ms=prepared.prefix.replay_ms,
        )
        capture_audio_admission_state()
        result["duration_ms"] = round(
            max(0, time.perf_counter_ns() - started_ns) / 1_000_000.0, 3
        )
        return result

    async def _register_shared_listener_session(
        sess: DuplexSession,
        *,
        record_scheduler_stage: Callable[[str, float | None], None],
        pending_speak_provider: Callable[[], dict[str, Any] | None] | None = None,
        policy_context_provider: Callable[[Any, Any], Any | None] | None = None,
    ) -> None:
        async with shared_listener_lifecycle_lock:
            if sess.listener is None or sess.listener_state is None:
                raise RuntimeError("shared listener batching requires an initialized listener")
            worker = shared_listener_worker_holder.get("worker")
            dispatcher = shared_listener_dispatcher_holder.get("dispatcher")
            semantic_request_holder: dict[str, Any] = {"request": None}
            prefix_importer = None
            initial_cache_position = 34
            if unified_kv_enabled:
                prefix_token_ids = _build_unified_listener_prefix_token_ids(engine, sess)
                initial_cache_position = len(prefix_token_ids)
                prefix_importer = make_unified_prefix_import(
                    sess.listener.backend,
                    prefix_token_ids,
                )
            elif (
                isinstance(sess.turn_policy, GemmaNativeControl6Policy)
                and _native_semantic_prefix_enabled()
            ):
                initial_request = build_native_control6_semantic_prefix_request(
                    engine,
                    sess,
                    current_frame_id=int(sess.listener_state.next_input_frame_id),
                    pending_speak=(
                        pending_speak_provider() if callable(pending_speak_provider) else None
                    ),
                )
                if initial_request is None:
                    raise RuntimeError(
                        "native Control6 semantic prefix is not safe to snapshot at listener registration"
                    )
                semantic_request_holder["request"] = initial_request
                initial_cache_position = initial_request.cache_position
                prefix_importer = semantic_prefix_importer(
                    sess.listener.backend,
                    lambda: semantic_request_holder["request"],
                )
            if worker is None:
                worker = make_listener_microbatch_worker(
                    sess.listener,
                    native_control6_policy=(
                        sess.turn_policy
                        if isinstance(sess.turn_policy, GemmaNativeControl6Policy)
                        else None
                    ),
                    semantic_prefix_enabled=_native_semantic_prefix_enabled(),
                    unified_kv_enabled=unified_kv_enabled,
                )
                await asyncio.wrap_future(worker.start())
                dispatcher = SharedListenerTickDispatcher(
                    _shared_listener_batch,
                    on_result=_shared_listener_row_result,
                    on_error=_shared_listener_row_error,
                    batch_context=_shared_listener_priority_context,
                )
                shared_listener_worker_holder["worker"] = worker
                shared_listener_worker_holder["backend"] = sess.listener.backend
                shared_listener_dispatcher_holder["dispatcher"] = dispatcher
                dispatcher.start()
            elif getattr(sess.listener, "backend", None) is not shared_listener_worker_holder.get("backend"):
                raise RuntimeError("shared listener sessions must use one loaded Gemma backend")
            runtime_info = getattr(
                sess.listener.backend, "_listener_microbatch_runtime_info", {}
            )
            if isinstance(sess.turn_policy, GemmaNativeControl6Policy):
                if not isinstance(runtime_info, dict) or runtime_info.get("stream_q_len") != 2:
                    raise RuntimeError("shared native Control6 requires the Q=2 listener runtime")
            assert dispatcher is not None
            generation = 1  # stable for this listener state epoch; turn generations are separate
            semantic_request_holder["listener_generation_id"] = generation
            worker_registered = False
            dispatcher_registered = False
            try:
                batch_state = initial_listener_session_state(
                    sess,
                    worker.runtime_signature,
                    native_context_id=next(shared_listener_context_ids),
                    generation_id=generation,
                    cache_position=initial_cache_position,
                )
                await asyncio.wrap_future(worker.register_session(
                    batch_state,
                    prefix_import=prefix_importer,
                ))
                worker_registered = True
                if (
                    unified_kv_enabled
                    and _configured_audio_response_protocol(engine) == CONTROLLER_GUIDED_SAY_V1
                ):
                    transition_ids = _build_unified_audio_response_suffix(engine, sess)
                    prepared_transition = await asyncio.wrap_future(
                        worker.prepare_text_token_inputs(
                            transition_ids,
                            protocol=CONTROLLER_GUIDED_SAY_V1,
                        )
                    )
                    if (
                        int(prepared_transition.get("logical_q_len", -1)) != len(transition_ids)
                        or int(prepared_transition.get("physical_q_len", -1))
                        != max(2, len(transition_ids) + (len(transition_ids) & 1))
                    ):
                        raise RuntimeError("prewarmed V1 transition has incompatible physical-Q metadata")
                    logger.info(
                        "[unified_transition_prewarm] session=%s logical_q=%d physical_q=%d "
                        "cache_size=%d cache_misses=%d",
                        sess.session_id[:8],
                        len(transition_ids),
                        int(prepared_transition["physical_q_len"]),
                        int(prepared_transition["cache_size"]),
                        int(prepared_transition["cache_misses"]),
                    )
                dispatcher.register_session(
                    sess.session_id,
                    sess.audio_ingress,
                    listener_generation_id=generation,
                    state_epoch_id=sess.listener_state.state_epoch_id,
                )
                dispatcher_registered = True
            except BaseException:
                if dispatcher_registered:
                    await asyncio.to_thread(
                        dispatcher.unregister_session,
                        sess.session_id,
                        wait_for_inflight=True,
                        timeout=30.0,
                    )
                if worker_registered:
                    await asyncio.wrap_future(worker.close_session(sess.session_id))
                if not dispatcher.active_session_ids:
                    await asyncio.to_thread(dispatcher.stop, timeout=2.0)
                    try:
                        await asyncio.wrap_future(worker.stop(drain=False))
                    finally:
                        shared_listener_dispatcher_holder.clear()
                        shared_listener_worker_holder.clear()
                raise
            shared_listener_bindings[sess.session_id] = {
                "session": sess,
                "generation": generation,
                "batch_cache_position": int(initial_cache_position),
                "unified_turn_transition_lock": threading.Lock(),
                "references": {},
                "record_scheduler_stage": record_scheduler_stage,
                "pending_speak_provider": pending_speak_provider,
                "policy_context_provider": policy_context_provider,
                "semantic_request_holder": semantic_request_holder,
                "semantic_prefix_importer": prefix_importer,
            }
            sess.conditioning_trace_runtime_provenance = dict(
                getattr(sess.listener.backend, "_listener_microbatch_runtime_info", {})
            )
            sess.conditioning_trace_runtime_provenance.setdefault("enabled", True)

    async def _unregister_shared_listener_session(sess: DuplexSession) -> None:
        async with shared_listener_lifecycle_lock:
            dispatcher = shared_listener_dispatcher_holder.get("dispatcher")
            worker = shared_listener_worker_holder.get("worker")
            if dispatcher is None or worker is None:
                return
            await asyncio.to_thread(
                dispatcher.unregister_session,
                sess.session_id,
                wait_for_inflight=True,
                timeout=30.0,
            )
            shared_listener_bindings.pop(sess.session_id, None)
            try:
                await asyncio.wrap_future(worker.close_session(sess.session_id))
            except BaseException:
                # A poisoned runtime cannot release an individual slot; stop()
                # still closes its shared native context and is reported below.
                logger.exception("shared listener slot cleanup failed for %s", sess.session_id)
            if not dispatcher.active_session_ids:
                await asyncio.to_thread(dispatcher.stop, timeout=2.0)
                try:
                    await asyncio.wrap_future(worker.stop(drain=False))
                finally:
                    shared_listener_dispatcher_holder.clear()
                    shared_listener_worker_holder.clear()

    @contextmanager
    def _inference_execution_scope(pooled: bool):
        if pooled:
            with inference_execution_gate.pooled():
                yield
        else:
            with inference_execution_gate.legacy():
                with inference_lock:
                    yield

    @contextmanager
    def _response_talk_context(
        pooled: bool,
        *,
        session_id: str,
        response_id: str,
        generation_id: int,
    ):
        pool = getattr(engine, "response_talk_context_pool", None)
        if not pooled or pool is None:
            yield None
            return
        lease = pool.acquire(
            session_id,
            response_id,
            generation_id=generation_id,
        )
        try:
            yield lease
        finally:
            pool.release(lease)

    def _stop_owned_synthesis(session_id: str, response_id: str | None) -> bool:
        stop = getattr(engine, "stop_generation", None)
        if response_id is None or not callable(stop):
            return False
        return engine_synthesis_owner.stop_if_owned(session_id, response_id, stop)

    speaker_gate = SpeakerGate()

    def _projector_parameter_sha256(projector: Any) -> str:
        digest = hashlib.sha256()
        for name, parameter in projector.named_parameters():
            digest.update(str(name).encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _make_policy(name: str, cfg: dict[str, Any] | None = None) -> Any:
        cfg = cfg or {}
        native_enabled = bool(getattr(engine, "enable_gemma_native_control6_policy", False))
        if native_enabled and name != "gemma_native_control6":
            raise ValueError("native Control6 è richiesto e non ammette fallback a una head esterna")
        if name == "gemma_native_control6":
            runtime = getattr(engine, "native_control6_runtime", None)
            checkpoint = getattr(engine, "native_control6_adapter_path", None)
            native_report = getattr(engine, "native_control6_report", None)
            if not native_enabled or runtime is None or not checkpoint:
                raise ValueError("gemma_native_control6 richiesto ma adapter nativo non caricato")
            return GemmaNativeControl6Policy(
                runtime,
                checkpoint_path=str(checkpoint),
                trace_scores=_NATIVE_CONTROL6_TRACE_ENABLED,
                fused_readback=NATIVE_CONTROL6_FUSED_READBACK_ENABLED,
                phase_profile=(
                    str(native_report.get("native_control6_phase_profile", "legacy"))
                    if isinstance(native_report, dict)
                    else "legacy"
                ),
            )
        if name == "none":
            return NoOpPolicy()
        if name == "scripted":
            return ScriptedPolicy(cfg.get("scripted_actions", [ControlAction.LISTEN.value]))
        if name == "provisional":
            return ProvisionalFramePolicy(window_frames=cfg.get("provisional_window_frames", 25))
        if name == "gemma":
            return GemmaPolicy()
        if name == "duplex_control6":
            runtime = getattr(engine, "duplex_control6_runtime", None)
            if not bool(getattr(engine, "enable_duplex_control6", False)) or runtime is None:
                raise ValueError(
                    "turn_policy=duplex_control6 richiesto ma adapter/head non caricati; "
                    "avviare con --enable-duplex-control6 e asset validi"
                )
            return DuplexControl6Policy(runtime)
        if name != "turnhead":
            raise ValueError(f"turn_policy non valida: {name}")
        return TurnHeadPolicy(device=getattr(engine, "device", "cuda"), checkpoint=TURNHEAD_CKPT)

    def _require_loaded_turnhead(policy: TurnHeadPolicy) -> None:
        if policy.loaded:
            return
        load_error = getattr(getattr(policy, "runtime", None), "load_error", None)
        detail = f": {load_error}" if load_error else ""
        raise ValueError(
            "TurnDynamics checkpoint unavailable at "
            f"{TURNHEAD_CKPT!r}; set GEMMA4_OMNI_TURNHEAD_CKPT to a valid v1 checkpoint{detail}"
        )

    def _ensure_policy(sess: DuplexSession, *, prefer_bundle_control6: bool = False) -> None:
        if sess.turn_policy is None:
            if bool(getattr(engine, "enable_gemma_native_control6_policy", False)):
                sess.set_turn_policy(_make_policy("gemma_native_control6"))
                return
            runtime = getattr(engine, "duplex_control6_runtime", None)
            bundle_control6_ready = (
                prefer_bundle_control6
                and bool(getattr(engine, "duplex_response_bundle_dir", None))
                and bool(getattr(engine, "enable_duplex_control6", False))
                and isinstance(runtime, dict)
                and isinstance(runtime.get("bundle_id"), str)
                and bool(runtime.get("bundle_id"))
                and runtime.get("head") is not None
                and isinstance(runtime.get("actions"), list)
                and isinstance(runtime.get("feature_contract"), dict)
            )
            if bundle_control6_ready:
                sess.set_turn_policy(_make_policy("duplex_control6"))
                return
            # Tests/embedded runtimes may inject a per-session policy factory;
            # the legacy serving default remains the real TurnHead policy.
            factory = getattr(engine, "default_turn_policy_factory", None)
            sess.set_turn_policy(factory() if callable(factory) else _make_policy("turnhead"))

    def _native_action_selection_mode() -> str:
        direct_audio_enabled = os.environ.get("DUPLEX_DIRECT_AUDIO_CANDIDATE", "0").strip() == "1"
        protocol = _configured_audio_response_protocol(engine)
        if not direct_audio_enabled:
            return "disabled"
        return "controller_authoritative" if protocol == CONTROLLER_GUIDED_SAY_V1 else (
            "wake_scoped" if protocol == AUDIO_RESPONSE_PROTOCOL_V2 else "disabled"
        )

    def _policy_session_metadata(
        policy: Any,
        session: DuplexSession | None = None,
    ) -> dict[str, Any]:
        if isinstance(policy, GemmaNativeControl6Policy):
            metadata = getattr(engine, "native_control6_semantic_prefix_metadata", None)
            prefix_enabled = bool(
                _native_semantic_prefix_enabled()
                and native_control6_semantic_prefix_metadata_is_supported(metadata)
            )
            result = {
                "turn_policy": GemmaNativeControl6Policy.policy_id,
                "policy_id": GemmaNativeControl6Policy.policy_id,
                "policy_actions": list(NATIVE_CONTROL6_ACTIONS),
                "policy_input_contract": "ListenerOutput.gemma_hidden_t:1536:current_frame",
                "policy_base_rows_validated": True,
                "response_lora_applied": False,
                "native_action_selection": _native_action_selection_mode(),
                "semantic_prefix_enabled": prefix_enabled,
                "semantic_prefix_schema_version": (
                    metadata.get("semantic_prefix_schema_version") if prefix_enabled else None
                ),
                "semantic_prefix_contract": (
                    metadata.get("semantic_prefix_contract") if prefix_enabled else None
                ),
                "semantic_prefix_capacity_tokens": (
                    metadata.get("semantic_prefix_capacity") if prefix_enabled else None
                ),
                "semantic_prefix_runtime_capacity_tokens": (
                    NATIVE_CONTROL6_PREFIX_MAX_TOKENS if prefix_enabled else None
                ),
            }
            if session is not None:
                state = getattr(session, "listener_state", None)
                applied_revision = getattr(state, "semantic_context_revision", None)
                result.update({
                    "semantic_prefix_actual_len": (
                        int(getattr(state, "gemma_lm_prefix_len", 0)) if state is not None else None
                    ),
                    "semantic_prefix_committed_history_revision": int(
                        getattr(session, "semantic_context_revision", 0)
                    ),
                    "semantic_prefix_applied_history_revision": (
                        int(applied_revision) if applied_revision is not None else None
                    ),
                    "semantic_prefix_context_current": (
                        applied_revision is not None
                        and int(applied_revision) == int(getattr(session, "semantic_context_revision", 0))
                    ),
                })
            return result
        if isinstance(policy, DuplexControl6Policy):
            runtime = policy.runtime
            feature_contract = runtime.get("feature_contract")
            feature_contract = feature_contract if isinstance(feature_contract, dict) else {}
            head = runtime.get("head")
            metadata: dict[str, Any] = {
                "turn_policy": "duplex_control6",
                "policy_id": "duplex_control6",
                "policy_actions": [str(action) for action in runtime.get("actions", [])],
                "native_action_selection": _native_action_selection_mode(),
                "response_projector_mode": _response_projector_mode(engine),
            }
            bundle_id = runtime.get("bundle_id")
            if isinstance(bundle_id, str) and bundle_id:
                metadata["bundle_id"] = bundle_id
                metadata["response_lora_applied"] = bool(runtime.get("response_lora_applied", True))
            architecture = getattr(head, "architecture", None)
            if isinstance(architecture, str) and architecture:
                metadata["control_head_architecture"] = architecture
            input_dim = feature_contract.get("input_dim", getattr(head, "input_dim", None))
            if isinstance(input_dim, int) and not isinstance(input_dim, bool):
                metadata["control_head_input_dim"] = input_dim
            return metadata
        if isinstance(policy, TurnHeadPolicy):
            return {
                "turn_policy": "turnhead",
                "policy_id": "turnhead",
                "native_action_selection": _native_action_selection_mode(),
            }
        policy_id = getattr(policy, "policy_id", None)
        if not isinstance(policy_id, str) or not policy_id:
            policy_id = {
                NoOpPolicy: "none",
                ScriptedPolicy: "scripted",
                ProvisionalFramePolicy: "provisional",
                GemmaPolicy: "gemma",
            }.get(type(policy), type(policy).__name__.lower())
        return {
            "turn_policy": policy_id,
            "policy_id": policy_id,
            "native_action_selection": _native_action_selection_mode(),
        }

    def _ensure_listener(sess: DuplexSession, *, runtime_metrics: dict[str, Any] | None = None) -> None:
        if sess.listener is not None:
            shared = sess.listener
        else:
            shared = getattr(engine, "listener", None)
            if shared is None:
                wrapper = getattr(engine, "_thinker_wrapper", None)
                if wrapper is not None:
                    try:
                        shared = GemmaListener.from_thinker(wrapper, runtime_profile="production")
                    except (TypeError, ValueError, RuntimeError) as exc:
                        logger.warning("[listener] non disponibile: %s", exc)
        if (
            not shared_listener_microbatch_enabled
            and sess.listener_state.gemma_lm_native is None
            and shared is not None
            and hasattr(shared, "prime_steady_state")
        ):
            if sess.listener_state.listener_input_frames == 0:
                prime_started = time.perf_counter()
                prime_succeeded = False
                try:
                    # This performs the canonical prefix prefill and creates the
                    # real native context before the websocket is admitted.
                    shared.prime_steady_state(sess.listener_state)
                    prime_succeeded = True
                except BaseException as exc:
                    if enforce_production_listener:
                        raise RuntimeError("listener production nativo non prewarmabile") from exc
                    logger.warning("[listener] prewarm nativo dev/test non riuscito: %s", exc)
                finally:
                    if runtime_metrics is not None:
                        listener_metrics = runtime_metrics.setdefault("listener", {})
                        listener_metrics["canonical_prime_calls"] = int(
                            listener_metrics.get("canonical_prime_calls", 0)
                        ) + 1
                        listener_metrics["canonical_prime_ms"] = round(
                            float(listener_metrics.get("canonical_prime_ms", 0.0))
                            + (time.perf_counter() - prime_started) * 1000.0,
                            3,
                        )
                        listener_metrics["canonical_prime_succeeded"] = prime_succeeded
        if enforce_production_listener:
            if type(shared) is not GemmaListener:
                raise RuntimeError("listener production causale non disponibile o di tipo inatteso")
            if getattr(shared, "runtime_profile", None) != "production":
                raise RuntimeError("listener production richiesto: profilo incompatibile")
            if type(getattr(shared, "backend", None)) is not GemmaAudioTowerBackend:
                raise RuntimeError("listener production richiesto: backend Gemma nativo concreto assente")
            if int(getattr(shared, "sample_rate", -1)) != USER_SAMPLE_RATE or int(getattr(shared, "frame_duration_ms", -1)) != 80:
                raise RuntimeError("listener production richiesto: clock 16 kHz/80 ms incompatibile")
            if getattr(shared, "text_token_ids", ()):
                raise RuntimeError("listener production richiesto: text conditioning non separato")
            backend = shared.backend
            if not bool(getattr(backend, "_use_native_lm", False)):
                raise RuntimeError("listener production richiesto: backend Gemma nativo assente")
            if sess.listener_state.gemma_lm_native is None and not shared_listener_microbatch_enabled:
                raise RuntimeError("listener production richiesto: contesto nativo mancante su sessione attiva")
            if not shared_listener_microbatch_enabled:
                try:
                    runtime_info = backend.native_runtime_info()
                    from duplex_kernel.native_listener import assert_production_native_runtime
                    assert_production_native_runtime(runtime_info)
                except (AttributeError, RuntimeError) as exc:
                    raise RuntimeError("listener production privo di attestazione nativa runtime") from exc
                sess.conditioning_trace_runtime_provenance = dict(runtime_info)
        elif shared is not None and hasattr(shared, "backend"):
            try:
                runtime_info = shared.backend.native_runtime_info()
                sess.conditioning_trace_runtime_provenance = dict(runtime_info)
            except Exception:
                pass
        sess.listener = shared

    def _configure_playout(sess: DuplexSession, value: Any | None) -> None:
        if value is None:
            return
        delay = int(value)
        if delay < 0 or delay > 64:
            raise ValueError("playout_delay_frames deve essere compreso tra 0 e 64")
        sess.speaker_playout_delay_frames = delay

    def get_session(request: dict[str, Any]) -> tuple[DuplexSession, bool]:
        sid = request.get("session_id")
        requested_voice = request.get("voice") or request.get("speaker")
        resolved_voice = str(requested_voice or DEFAULT_VOICE).strip() or DEFAULT_VOICE
        if sid:
            sess = sessions.get(str(sid))
            if sess is None: raise ValueError("session_id non valido o scaduto")
            try:
                if requested_voice or not str(getattr(sess, "voice", "")).strip():
                    sess.voice = resolved_voice
                    sess.voice_conditioning = _resolve_p7_session_voice(
                        engine,
                        voice=sess.voice,
                        voice_style=request.get("voice_style") or sess.voice_style,
                    )
                elif getattr(sess, "voice_conditioning", None) is None:
                    sess.voice = resolved_voice
                    sess.voice_conditioning = _resolve_p7_session_voice(engine, voice=sess.voice)
                sess.voice_style = request.get("voice_style") or sess.voice_style
                _configure_playout(sess, request.get("playout_delay_frames"))
                _ensure_policy(sess)
                _ensure_listener(sess)
            except BaseException:
                # A session that cannot be attested must not remain reusable
                # after a failed reconnect/startup attempt.
                sessions.close(sess.session_id)
                raise
            return sess, False
        sess = sessions.create_session(
            instructions=str(request.get("instructions", "")),
            voice=resolved_voice,
            language=str(request.get("language", "Italian")),
        )
        try:
            sess.voice_conditioning = _resolve_p7_session_voice(
                engine,
                voice=sess.voice,
                voice_style=request.get("voice_style"),
            )
            sess.voice_style = request.get("voice_style") or sess.voice_style
            _configure_playout(sess, request.get("playout_delay_frames"))
            _ensure_policy(sess)
            _ensure_listener(sess)
        except BaseException:
            sessions.close(sess.session_id)
            raise
        return sess, True

    def run_response(
        sess: DuplexSession,
        messages: list[dict[str, Any]],
        request: dict[str, Any],
        *,
        action: str | None = None,
        history_is_committed: bool = False,
    ) -> dict[str, Any]:
        clean = [m for m in messages if isinstance(m, dict) and m.get("role") in {"system", "user", "assistant", "tool"}]
        if sess.instructions and not any(m.get("role") == "system" for m in clean):
            clean.insert(0, {"role": "system", "content": sess.instructions})
        tools = request.get("tools") or sess.tools
        if tools:
            clean.insert(0, {"role": "system", "content": tool_system_prompt(tools)})
        generation_id = request.get("_generation_id")
        if generation_id is not None and not sess.is_generation_current(generation_id=generation_id):
            sess.stale_session_mutations += 1
            return {"text": "", "tool_calls": [], "wav": None, "elapsed": 0.0, "cancelled": True}
        normalized_action = str(action).strip().upper() if action is not None else None
        direct_audio_pcm = request.get("_direct_audio_pcm")
        direct_audio = direct_audio_pcm is not None
        audio_requested = "audio" in (request.get("modalities") or ["text"])
        response_protocol = _configured_audio_response_protocol(engine) if direct_audio else None
        unified_native = bool(unified_kv_enabled and direct_audio)
        unified_branch = request.get("_unified_native_branch") if unified_native else None
        private_backchannel_active = bool(
            unified_native
            and normalized_action == ControlAction.BACKCHANNEL.value
            and isinstance(unified_branch, dict)
            and unified_branch.get("ephemeral_decode_handle") is not None
        )
        if unified_native and (
            not isinstance(unified_branch, dict)
            or not (
                unified_branch.get("queue_open")
                or unified_branch.get("ephemeral_decode_handle") is not None
            )
            or response_protocol not in {AUDIO_RESPONSE_PROTOCOL_V2, CONTROLLER_GUIDED_SAY_V1}
        ):
            raise RuntimeError("unified native direct-audio response lacks an open v1/v2 cache branch")
        controller_guided_say = response_protocol == CONTROLLER_GUIDED_SAY_V1
        live_say_active = bool(
            controller_guided_say
            and direct_audio
            and audio_requested
            and normalized_action in {
                ControlAction.SPEAK.value,
                ControlAction.BACKCHANNEL.value,
            }
        )
        frame_residual_adapter = getattr(engine, "frame_residual_adapter", None)
        frame_residual_enabled = bool(getattr(engine, "frame_residual_adapter_enabled", False))
        frame_residual_candidate = bool(
            frame_residual_enabled
            and frame_residual_adapter is not None
            and audio_requested
            and normalized_action in {
                ControlAction.SPEAK.value,
                ControlAction.BACKCHANNEL.value,
            }
        )
        live_v2_say_active = bool(
            response_protocol == AUDIO_RESPONSE_PROTOCOL_V2
            and unified_native
            and direct_audio
            and audio_requested
            and normalized_action == ControlAction.SPEAK.value
        )
        v9_conditioning_source = getattr(engine, "v9_conditioning_source", "live_generated")
        if v9_conditioning_source == "audio_prefill_frame_raw" and not controller_guided_say:
            raise RuntimeError(
                "audio_prefill_frame_raw richiede input audio diretto e controller_guided_say_v1"
            )
        if controller_guided_say:
            normalized_action = controller_guided_response_action(normalized_action)
        response_allowed_actions = (
            audio_response_actions_for_wake(normalized_action)
            if response_protocol == AUDIO_RESPONSE_PROTOCOL_V2
            else None
        )
        private_control_markers = (
            tuple(
                action_control_marker_text(candidate)
                for candidate in ("BACKCHANNEL", "SPEAK", "REPLAN")
            )
            if normalized_action is not None and not direct_audio
            else ()
        )
        private_control_marker_suffix_length = max(map(len, private_control_markers), default=0)
        action_instruction = (
            action_conditioning_instruction(normalized_action) if normalized_action is not None else None
        )
        # La sessione è la proprietaria del rolling context: il prompt non
        # cresce senza limite e non dipende da una lista globale del server.
        # Anche questo piccolo commit di stato è protetto: una REPLAN non può
        # lasciare il contesto A in una risposta B.
        with sess._generation_lock:
            if generation_id is not None and not sess.is_generation_current(generation_id=generation_id):
                sess.stale_session_mutations += 1
                return {"text": "", "tool_calls": [], "wav": None, "elapsed": 0.0, "cancelled": True}
            sess.thinker_ctx.system_prompt = sess.instructions
            sess.thinker_ctx.set_messages(clean)
            clean = sess.thinker_ctx.messages()
        prompt_messages = [dict(message) for message in clean]
        if direct_audio:
            if tools:
                raise RuntimeError("il candidato audio diretto non supporta tools senza prompt di training verificato")
            history_messages = _direct_audio_prompt_history(
                clean,
                history_is_committed=history_is_committed,
            )
        elif normalized_action is not None:
            history_messages = [
                message for message in clean if message.get("role") in {"user", "assistant"}
            ]
            prompt_messages = action_conditioned_messages(
                sess.instructions,
                normalized_action,
                history_messages,
            )
            auxiliary_system_messages = [
                dict(message)
                for message in clean
                if message.get("role") == "system"
                and _message_text(message.get("content", "")).strip() != str(sess.instructions).strip()
            ]
            prompt_messages[1:1] = auxiliary_system_messages
        processor_audio_inputs = None
        audio_prepare_ms = 0.0
        if unified_native:
            ids = list(
                _build_unified_backchannel_response_suffix(engine, sess)
                if private_backchannel_active
                else _build_unified_audio_response_suffix(engine, sess)
            )
            processor_audio_inputs = None
            audio_prepare_ms = 0.0
        elif direct_audio:
            ids, processor_audio_inputs, audio_prepare_ms = _prepare_direct_audio_processor_inputs(
                engine,
                language=request.get("language") or sess.language,
                history_messages=history_messages,
                pcm=np.asarray(direct_audio_pcm, dtype=np.float32),
                response_protocol=response_protocol,
                session_instructions=(
                    sess.instructions
                    if response_protocol in {AUDIO_RESPONSE_PROTOCOL_V2, CONTROLLER_GUIDED_SAY_V1}
                    else None
                ),
                controller_action=normalized_action if controller_guided_say else None,
            )
        else:
            chat = engine.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            chat = _append_native_action_token(chat, normalized_action)
            ids = engine.tokenizer(chat, return_tensors="pt")["input_ids"][0].tolist()
        cond = _resolve_request_voice_conditioning(engine, sess, request)
        live_say_stream: Any | None = None
        live_say_thread: threading.Thread | None = None
        live_say_thread_lock = threading.Lock()
        live_say_cancel = threading.Event()
        live_tts_unit_assembler: SpokenTextUnitAssembler | None = None
        live_tts_unit_queue: queue.Queue[Any] | None = None
        live_tts_unit_sentinel = object()
        live_tts_unit_metrics: list[dict[str, Any]] = []
        live_tts_unit_wavs: list[np.ndarray] = []
        live_tts_visible_text_parts: list[str] = []
        live_tts_active_stream: dict[str, TurnLocalQwenSayStream | None] = {"value": None}
        live_say_eos_text_id: int | None = None
        live_say_result: dict[str, Any] = {}
        live_say_timing: dict[str, int | None] = {
            "consumer_started_at_ns": None,
            "thinker_returned_at_ns": None,
        }
        thinker_first_token_id: int | None = None
        live_say_tokenizer = None
        if live_say_active or live_v2_say_active:
            talker_wrapper = getattr(engine, "talker", None)
            processor = getattr(talker_wrapper, "proc", None)
            live_say_tokenizer = getattr(processor, "tokenizer", None)
            eos_text_id = getattr(getattr(talker_wrapper, "cfg", None), "tts_eos_token_id", None)
            if live_say_tokenizer is None or (live_say_active and processor is None):
                raise RuntimeError("live controller-guided SAY requires native Qwen tokenizer and processor")
            if isinstance(eos_text_id, bool) or not isinstance(eos_text_id, int):
                raise RuntimeError("live controller-guided SAY requires native Qwen text EOS token id")
            live_say_eos_text_id = int(eos_text_id)
            if live_say_active:
                live_say_stream = LiveQwenSayStream(
                    qwen_tokenizer=live_say_tokenizer,
                    qwen_processor=processor,
                    text_eos_token_id=eos_text_id,
                    max_tokens=ARIA_MAX_FRAMES,
                    queue_capacity=16,
                )
            elif live_v2_say_active:
                live_tts_unit_assembler = SpokenTextUnitAssembler(max_words=2)
                live_tts_unit_queue = queue.Queue(maxsize=2)

            def live_say_talker_worker() -> None:
                try:
                    if live_v2_say_active:
                        if live_tts_unit_queue is None or live_say_tokenizer is None:
                            raise RuntimeError("closed-unit Qwen worker is not initialized")
                        live_say_timing["consumer_started_at_ns"] = time.perf_counter_ns()
                        while not live_say_cancel.is_set():
                            try:
                                unit_item = live_tts_unit_queue.get(timeout=0.05)
                            except queue.Empty:
                                continue
                            if unit_item is live_tts_unit_sentinel:
                                break
                            if not isinstance(unit_item, SpokenTextUnit):
                                raise TypeError("closed-unit queue contains a non-SpokenTextUnit item")
                            unit = unit_item
                            unit_index = len(live_tts_unit_metrics)
                            unit_segment_id = (
                                str(request.get("_generation_token").speaker_segment_id)
                                if unit_index == 0 and request.get("_generation_token") is not None
                                else f"{request.get('_generation_token').speaker_segment_id}.tts{unit_index}"
                            )
                            begin_segment = request.get("_begin_audio_segment")
                            finish_segment = request.get("_finish_audio_segment")
                            if not callable(begin_segment) or not callable(finish_segment):
                                raise RuntimeError("closed TTS unit lacks audio segment callbacks")
                            if unit_index:
                                sess.speech_fifo.begin_segment(generation_id=int(generation_id))
                            begin_segment(unit_segment_id)
                            unit_stream = TurnLocalQwenSayStream(
                                qwen_tokenizer=live_say_tokenizer,
                                text_eos_token_id=int(live_say_eos_text_id),
                                fifo=sess.speech_fifo,
                                generation_id=int(generation_id),
                                max_tokens=ARIA_MAX_FRAMES,
                                on_packet_consumed=lambda cursor, text: (
                                    unified_branch.update({
                                        "qwen_forward_ack_exclusive_cursor": int(cursor),
                                        "qwen_forward_ack_text": str(text),
                                    })
                                    if isinstance(unified_branch, dict) else None
                                ),
                            )
                            live_tts_active_stream["value"] = unit_stream
                            unit_feed_error: list[BaseException] = []

                            def feed_closed_unit() -> None:
                                try:
                                    for packet in unit.token_packets:
                                        unit_stream.feed_packet(
                                            packet,
                                            stop_check=lambda: bool(
                                                live_say_cancel.is_set()
                                                or sess.stop_requested
                                                or (
                                                    request.get("_stop_check")
                                                    and request["_stop_check"]()
                                                )
                                            ),
                                        )
                                    unit_stream.finish(final_text=unit.text)
                                except BaseException as exc:
                                    unit_feed_error.append(exc)
                                    unit_stream.fail(exc)

                            feed_thread = threading.Thread(
                                target=feed_closed_unit,
                                name=f"closed-say-feed-{unit_index}",
                                daemon=True,
                            )
                            feed_thread.start()
                            try:
                                initial_token = unit_stream.take_initial_token(
                                    stop_check=lambda: bool(
                                        live_say_cancel.is_set()
                                        or sess.stop_requested
                                        or (
                                            request.get("_stop_check")
                                            and request["_stop_check"]()
                                        )
                                    ),
                                )
                            except BaseException:
                                unit_stream.cancel()
                                feed_thread.join(timeout=2.0)
                                raise
                            unit_metrics: dict[str, Any] = {
                                "unit_index": unit_index,
                                "speaker_segment_id": unit_segment_id,
                                "text": unit.text,
                                "source_token_ids": list(unit.token_ids),
                                "exclusive_kv_cursor": int(unit.exclusive_kv_cursor),
                                "started_at_ns": time.perf_counter_ns(),
                            }
                            h_pfx = torch.empty((1, 0, 2048), device=engine.device, dtype=engine.dtype)
                            talker_kwargs: dict[str, Any] = {
                                "n_words": max(1, len(unit.text.split())),
                                "voice_conditioning": cond,
                                "stop_check": lambda: bool(
                                    live_say_cancel.is_set()
                                    or sess.stop_requested
                                    or (request.get("_stop_check") and request["_stop_check"]())
                                ),
                                "state": sess,
                                "on_audio_chunk": request.get("_on_audio_chunk"),
                                "generation_id": generation_id,
                                "talk_context": request.get("_talk_context"),
                                "metrics": unit_metrics,
                                "listener_admission_wait": listener_admission_wait,
                                "raw_text_token_ids": [int(initial_token.token_id)],
                                "include_text_eos": False,
                                "live_qwen_text_stream": unit_stream,
                                "max_frames": ARIA_BACKCHANNEL_MAX_FRAMES if normalized_action == ControlAction.BACKCHANNEL.value else ARIA_MAX_FRAMES,
                            }
                            if frame_residual_candidate:
                                talker_kwargs["live_frame_residual_hidden_stream"] = unit_stream
                                talker_kwargs["frame_residual_response_hidden"] = None
                            if normalized_action == ControlAction.BACKCHANNEL.value:
                                talker_kwargs["min_frames"] = 2
                            try:
                                with _diagnostic_talker_rng_scope(engine):
                                    wav = engine.talk(unit.text, h_pfx, **talker_kwargs)
                            except BaseException as exc:
                                unit_stream.fail(exc)
                                feed_thread.join(timeout=2.0)
                                raise
                            feed_thread.join(timeout=5.0)
                            if feed_thread.is_alive():
                                unit_stream.cancel()
                                feed_thread.join(timeout=2.0)
                                raise RuntimeError(
                                    "closed TTS unit producer remained blocked after Talker completion"
                                )
                            if unit_feed_error:
                                raise unit_feed_error[0]
                            unit_metrics["ended_at_ns"] = time.perf_counter_ns()
                            if live_say_cancel.is_set() or sess.stop_requested:
                                unit_stream.cancel()
                                break
                            if not unit_stream.codec_eos_allowed:
                                raise RuntimeError(
                                    "Qwen Talker ended before text EOS, codec EOS, and closed-unit drain"
                                )
                            # ``codec_eos_allowed`` only says that the source
                            # text stream sent EOS and drained.  It does not
                            # prove that the Talker sampled its codec EOS:
                            # reaching the acoustic frame budget can return a
                            # normal WAV without a terminal codec token.  A
                            # segment marker advances the canonical Gemma KV
                            # watermark, so only the Talker's explicit EOS
                            # metric is sufficient ownership evidence.
                            _require_closed_tts_codec_eos(unit_metrics)
                            source_tokens = tuple({
                                "token_id": int(source.token_id),
                                "exclusive_kv_cursor": int(source.exclusive_kv_cursor),
                                "text_piece": str(source.text_piece),
                            } for source in unit.source_tokens)
                            finish_segment(unit_segment_id, source_tokens)
                            live_tts_active_stream["value"] = None
                            live_tts_unit_metrics.append(unit_metrics)
                            if isinstance(wav, np.ndarray) and wav.size:
                                live_tts_unit_wavs.append(np.asarray(wav, dtype=np.float32).reshape(-1))
                        live_say_result.update({
                            "wav": (
                                np.concatenate(live_tts_unit_wavs)
                                if live_tts_unit_wavs else np.zeros(0, dtype=np.float32)
                            ),
                            "metrics": {
                                "closed_tts_unit_count": len(live_tts_unit_metrics),
                                "closed_tts_units": [dict(row) for row in live_tts_unit_metrics],
                                "unit_boundary_policy": "max_2_words_at_original_gemma_token_boundary",
                                "unit_text_eos_and_codec_drain": True,
                                "live_qwen_token_cursor_start": 1,
                                "live_qwen_text_eos_consumed": bool(
                                    live_tts_unit_metrics
                                    and all(bool(row.get("eos_detected")) for row in live_tts_unit_metrics)
                                ),
                                "response_wav_concatenated_from_closed_units": True,
                            },
                        })
                        return

                    assert live_say_stream is not None
                    live_say_timing["consumer_started_at_ns"] = time.perf_counter_ns()
                    initial_token = live_say_stream.take_initial_token(
                        stop_check=lambda: bool(
                            live_say_cancel.is_set()
                            or sess.stop_requested
                            or (request.get("_stop_check") and request["_stop_check"]())
                        ),
                    )
                    talker_metrics: dict[str, Any] = {}
                    h_pfx = torch.empty((1, 0, 2048), device=engine.device, dtype=engine.dtype)
                    prefill_text = live_say_stream.visible_text
                    if not prefill_text:
                        raise RuntimeError("live Qwen token 0 was published without visible SAY text")
                    talker_kwargs: dict[str, Any] = {
                        "n_words": 1,
                        "voice_conditioning": cond,
                        "stop_check": lambda: bool(
                            live_say_cancel.is_set()
                            or sess.stop_requested
                            or (request.get("_stop_check") and request["_stop_check"]())
                        ),
                        "state": sess,
                        "on_audio_chunk": request.get("_on_audio_chunk"),
                        "generation_id": generation_id,
                        "talk_context": request.get("_talk_context"),
                        "metrics": talker_metrics,
                        "listener_admission_wait": listener_admission_wait,
                        "raw_text_token_ids": [int(initial_token.token_id)],
                        "include_text_eos": False,
                        "live_qwen_text_stream": live_say_stream,
                        "max_frames": (
                            ARIA_BACKCHANNEL_MAX_FRAMES
                            if normalized_action == ControlAction.BACKCHANNEL.value
                            else ARIA_MAX_FRAMES
                        ),
                    }
                    if frame_residual_candidate:
                        talker_kwargs["live_frame_residual_hidden_stream"] = live_say_stream
                        talker_kwargs["frame_residual_response_hidden"] = None
                    if normalized_action == ControlAction.BACKCHANNEL.value:
                        talker_kwargs["min_frames"] = 2
                    with _diagnostic_talker_rng_scope(engine):
                        wav = engine.talk(prefill_text, h_pfx, **talker_kwargs)
                    if not live_say_stream.codec_eos_allowed and not live_say_cancel.is_set():
                        raise RuntimeError("live Qwen Talker ended before native text EOS and tail drain")
                    live_say_result.update({"wav": wav, "metrics": talker_metrics})
                    if live_say_cancel.is_set():
                        live_say_stream.cancel()
                except BaseException as exc:
                    active_stream = live_tts_active_stream.get("value")
                    if live_say_cancel.is_set() or sess.stop_requested:
                        if active_stream is not None:
                            active_stream.cancel()
                        return
                    live_say_result["error"] = exc
                    live_say_cancel.set()
                    if active_stream is not None:
                        active_stream.fail(exc)
                    elif live_say_stream is not None:
                        live_say_stream.fail(exc)
                    elif live_v2_say_active:
                        sess.speech_fifo.clear()

            def start_live_say_talker() -> None:
                nonlocal live_say_thread
                with live_say_thread_lock:
                    if live_say_thread is not None:
                        return
                    live_say_thread = threading.Thread(
                        target=live_say_talker_worker,
                        name=f"controller-guided-say-{getattr(sess, 'session_id', 'session')}",
                        daemon=True,
                    )
                    live_say_thread.start()

            def enqueue_live_v2_unit(unit: SpokenTextUnit) -> None:
                if live_tts_unit_queue is None:
                    raise RuntimeError("closed TTS unit queue is unavailable")
                while True:
                    if live_say_cancel.is_set() or sess.stop_requested:
                        raise RuntimeError("closed TTS unit enqueue cancelled")
                    if request.get("_stop_check") and request["_stop_check"]():
                        raise RuntimeError("closed TTS unit enqueue lost its response generation")
                    try:
                        live_tts_unit_queue.put(unit, timeout=0.05)
                        break
                    except queue.Full:
                        continue
                start_live_say_talker()

            def finish_live_v2_units() -> None:
                if live_tts_unit_assembler is None or live_tts_unit_queue is None:
                    raise RuntimeError("closed TTS unit assembler is unavailable")
                for unit in live_tts_unit_assembler.finish():
                    enqueue_live_v2_unit(unit)
                if live_say_thread is None:
                    if live_tts_unit_queue.empty():
                        return
                    start_live_say_talker()
                while True:
                    try:
                        live_tts_unit_queue.put(live_tts_unit_sentinel, timeout=0.05)
                        break
                    except queue.Full:
                        if live_say_cancel.is_set():
                            return
                if live_say_thread is not None:
                    live_say_thread.join()
                    if live_say_thread.is_alive():
                        raise RuntimeError("closed TTS unit worker did not stop after EOS")

            def cancel_live_v2_units(error: BaseException | None = None) -> None:
                live_say_cancel.set()
                if error is not None and live_tts_active_stream.get("value") is not None:
                    live_tts_active_stream["value"].fail(error)
                else:
                    sess.speech_fifo.clear()
                if live_tts_unit_queue is not None:
                    while True:
                        try:
                            live_tts_unit_queue.get_nowait()
                        except queue.Empty:
                            break
                    try:
                        live_tts_unit_queue.put_nowait(live_tts_unit_sentinel)
                    except queue.Full:
                        pass
                # Cancellation is a fence, not a synchronous drain: a YIELD
                # must return control without waiting for the Talker lease.
                # FIFO clear wakes a blocked unit producer; generation checks
                # reject any late PCM callback from the daemon worker.

            def live_say_generated_token(
                _token_index: int,
                _token_id: int,
                token_text: str,
                token_hidden: Any,
            ) -> None:
                nonlocal thinker_first_token_id
                if thinker_first_token_id is None:
                    thinker_first_token_id = int(_token_id)
                assert live_say_stream is not None
                stream_hidden = token_hidden
                producer_event = None
                if unified_native:
                    if not isinstance(unified_branch, dict):
                        raise RuntimeError("unified SAY token lacks its native KV branch")
                    if not isinstance(token_hidden, torch.Tensor):
                        raise TypeError("Gemma SAY hidden deve essere Tensor")
                    if token_hidden.ndim == 3 and tuple(token_hidden.shape[:2]) == (1, 1):
                        packet_hidden = token_hidden[:, 0, :]
                    elif token_hidden.ndim == 2:
                        packet_hidden = token_hidden
                    else:
                        raise ValueError(
                            f"Gemma SAY hidden shape non valida: {tuple(token_hidden.shape)}"
                        )
                    if tuple(packet_hidden.shape) != (1, 1536):
                        raise ValueError(
                            f"Gemma SAY packet hidden shape non valida: {tuple(packet_hidden.shape)}"
                        )
                    packet = SpokenTokenPacket(
                        token_id=int(_token_id),
                        text_piece=str(token_text),
                        gemma_hidden_1536=packet_hidden,
                        exclusive_kv_cursor=int(unified_branch["cache_position"]),
                    )
                    producer_event = packet.producer_event
                    with unified_branch["spoken_prefix_lock"]:
                        unified_branch.setdefault("say_token_packets", []).append(packet)
                        unified_branch["segment_token_packets"].append({
                            "token_id": packet.token_id,
                            "exclusive_kv_cursor": packet.exclusive_kv_cursor,
                            "text_piece": packet.text_piece,
                        })
                    stream_hidden = packet.gemma_hidden_1536.unsqueeze(1)
                live_say_stream.feed_piece(
                    token_text,
                    stream_hidden,
                    producer_event=producer_event,
                )
                if token_text and live_say_stream.visible_text:
                    start_live_say_talker()

        diagnostic_talker_seed = getattr(engine, "diagnostic_talker_seed", None)
        streamed_audio: list[np.ndarray] = []
        duplex_realtime = bool(request.get("_duplex_realtime"))
        stream_requested = callable(request.get("_on_audio_chunk")) and not tools
        backchannel_say_phrases = (
            tuple(getattr(engine, "_backchannel_say_phrases", ()))
            if direct_audio and response_protocol == AUDIO_RESPONSE_PROTOCOL_V2
            and response_allowed_actions is not None
            and "BACKCHANNEL" in response_allowed_actions
            else ()
        )
        response_v2_collector = (
            AudioResponseV2Collector(
                allowed_actions=response_allowed_actions,
                backchannel_say_phrases=backchannel_say_phrases,
            )
            if (
                direct_audio
                and response_protocol == AUDIO_RESPONSE_PROTOCOL_V2
                and not private_backchannel_active
            )
            else None
        )
        response_v2_action_constraint = (
            AudioResponseV2ActionConstraint(
                expected_action=None,
                allowed_actions=response_allowed_actions,
                tokenizer=engine.tokenizer,
                enforce_json_structure=True,
            )
            if response_v2_collector is not None else None
        )
        live_v2_token_pacer = GemmaTokenTickPacer() if live_v2_say_active else None
        live_v2_vocal_started = False
        runtime_callback = request.get("_runtime_metrics_callback")
        held_marker_suffix = ""
        raw_stream_text: list[str] = []

        def render_visible_text_chunk(chunk_text: str, chunk_hidden: Any, chunk_prompt_len: int) -> None:
            if not chunk_text or not stream_requested:
                return
            if not duplex_realtime:
                chunk_h = engine.v9_prefix(chunk_hidden, chunk_prompt_len)
                with sess._generation_lock:
                    if sess.turn_policy is not None and (generation_id is None or sess.is_generation_current(generation_id=generation_id)):
                        with sess._policy_lock:
                            if sess.turn_policy is not None:
                                sess.turn_policy.update_hidden(chunk_h)
                    elif generation_id is not None:
                        sess.stale_session_mutations += 1
            if request.get("_generation_token") is not None:
                sess.record_synthesized_text(request["_generation_token"], chunk_text)
            if callable(request.get("_on_text_chunk")):
                request["_on_text_chunk"](chunk_text)
            # Audio is synthesized once, after Thinker has committed the full
            # response hidden sequence.  Synthesizing each text chunk would
            # reintroduce the old one-shot Qwen path and cannot enforce P4D's
            # frame lookahead.

        def render_text_chunk(chunk_text: str, chunk_hidden: Any, chunk_prompt_len: int) -> None:
            nonlocal held_marker_suffix
            if not chunk_text or not stream_requested:
                return
            if not private_control_markers:
                render_visible_text_chunk(chunk_text, chunk_hidden, chunk_prompt_len)
                return
            raw_stream_text.append(str(chunk_text))
            combined = held_marker_suffix + str(chunk_text)
            if any(marker in combined for marker in private_control_markers):
                raise _PrivateControlLeakError(normalized_action or "", "".join(raw_stream_text))
            # Keep enough trailing characters to recognize the exact marker
            # even when its bytes are split across tokenizer text callbacks.
            visible_length = max(0, len(combined) - private_control_marker_suffix_length + 1)
            if visible_length:
                render_visible_text_chunk(combined[:visible_length], chunk_hidden, chunk_prompt_len)
            held_marker_suffix = combined[visible_length:]

        think_kwargs: dict[str, Any] = {
            "max_new_tokens": (
                2
                if private_backchannel_active
                else
                AUDIO_RESPONSE_V2_MAX_NEW_TOKENS
                if response_v2_collector is not None
                else int(request.get("max_tokens", 64))
            ),
            "min_new_tokens": int(request.get("_min_new_tokens", 0)),
            "temperature": float(request.get("temperature", 0.7)),
            "top_p": float(request.get("top_p", 0.9)),
            "stop_check": request.get("_stop_check"),
            "state": sess,
            "generation_id": generation_id,
            "on_text_chunk": (
                render_text_chunk
                if stream_requested
                and response_v2_collector is None
                and not controller_guided_say
                else None
            ),
            "on_token_id": (
                None
                if response_v2_collector is not None or controller_guided_say
                else request.get("_on_thinker_token_id")
            ),
            "action": None if direct_audio else normalized_action,
            "response_protocol": response_protocol if direct_audio else None,
            "input_audio": processor_audio_inputs,
            "graceful_sentence_completion": bool(
                controller_guided_say and normalized_action == ControlAction.SPEAK.value
            ),
        }
        if controller_guided_say and normalized_action == ControlAction.BACKCHANNEL.value:
            thinker_model = getattr(engine, "thinker", None)
            generation_eos = getattr(
                getattr(thinker_model, "generation_config", None), "eos_token_id", None
            )
            model_eos = getattr(getattr(thinker_model, "config", None), "eos_token_id", None)
            eos_value = generation_eos if generation_eos is not None else model_eos
            if eos_value is None:
                eos_ids = normalize_eos_token_ids(getattr(engine.tokenizer, "eos_token_id", None))
                eos_ids.update(normalize_eos_token_ids(getattr(engine.tokenizer, "eot_id", None)))
            else:
                eos_ids = normalize_eos_token_ids(eos_value)
            if not eos_ids:
                raise RuntimeError("controller_guided_backchannel_requires_thinker_eos")
            controller_backchannel_eos_id = min(eos_ids)
            controller_backchannel_text: list[str] = []

            def controller_backchannel_generated_token(
                _token_index: int,
                _token_id: int,
                token_text: str,
                token_hidden: Any,
            ) -> tuple[int, str] | None:
                controller_backchannel_text.append(str(token_text))
                if live_say_active:
                    live_say_generated_token(
                        _token_index,
                        _token_id,
                        token_text,
                        token_hidden,
                    )
                if controller_backchannel_should_stop("".join(controller_backchannel_text)):
                    return controller_backchannel_eos_id, "structure"
                return None

            def controller_backchannel_candidate(
                _token_id: int,
                token_text: str,
                terminal: bool,
            ) -> bool:
                return controller_backchannel_candidate_allowed(
                    "".join(controller_backchannel_text),
                    token_text,
                    terminal=terminal,
                )

            def controller_backchannel_fallback(
                _rejected: list[tuple[int, str, bool]],
            ) -> int | None:
                if controller_backchannel_candidate_allowed(
                    "".join(controller_backchannel_text), "", terminal=True
                ):
                    return controller_backchannel_eos_id
                return None

            think_kwargs["on_generated_token"] = controller_backchannel_generated_token
            think_kwargs["on_candidate_token"] = controller_backchannel_candidate
            think_kwargs["on_candidate_fallback"] = controller_backchannel_fallback
        elif live_say_active:
            think_kwargs["on_generated_token"] = live_say_generated_token
        v9_dual_audio_context_enabled = bool(
            response_v2_collector is not None
            and getattr(engine, "v9_dual_audio_context_experimental", False)
        )
        v9_audio_prefill_frame_raw_enabled = bool(
            controller_guided_say
            and v9_conditioning_source == "audio_prefill_frame_raw"
        )
        if v9_audio_prefill_frame_raw_enabled and v9_dual_audio_context_enabled:
            raise RuntimeError("audio_prefill_frame_raw e V9 dual context sono mutuamente esclusivi")
        if v9_dual_audio_context_enabled:
            think_kwargs["capture_audio_context"] = True
            think_kwargs["audio_context_source_pcm_samples"] = int(
                np.asarray(direct_audio_pcm).size
            )
        if v9_audio_prefill_frame_raw_enabled:
            think_kwargs["capture_audio_prefill_frame_raw"] = True
            think_kwargs["audio_context_source_pcm_samples"] = int(
                np.asarray(direct_audio_pcm).size
            )
        listener_admission_wait = request.get("_listener_admission_wait")
        if direct_audio and callable(listener_admission_wait):
            think_kwargs["listener_admission_wait"] = listener_admission_wait
        if response_v2_collector is not None or controller_guided_say:
            think_kwargs["disable_thinker_cadence_pacing"] = True
        response_v2_budget: dict[str, int] | None = None
        if response_v2_collector is not None:
            assert response_v2_action_constraint is not None
            try:
                response_v2_budget = response_v2_action_constraint.preflight_generation_budget(
                    response_v2_collector,
                    max_new_tokens=int(think_kwargs["max_new_tokens"]),
                    max_heard_tokens=AUDIO_RESPONSE_V2_MAX_HEARD_TOKENS,
                    max_action_tokens=AUDIO_RESPONSE_V2_MAX_ACTION_TOKENS,
                    max_say_tokens=AUDIO_RESPONSE_V2_MAX_SAY_TOKENS,
                )
                response_v2_collector.set_token_caps(
                    heard=response_v2_budget["heard_token_cap"],
                    action=response_v2_budget["action_token_cap"],
                    say=response_v2_budget["say_token_cap"],
                )
            except AudioResponseProtocolError as exc:
                _record_audio_response_v2_private_failure(
                    collector=response_v2_collector,
                    reason=exc.reason,
                    response_id=(
                        request.get("_generation_token").response_id
                        if request.get("_generation_token") is not None else None
                    ),
                    action=normalized_action,
                    budget=None,
                )
                response_v2_collector.cancel()
                raise

            def start_v2_generation() -> int | None:
                assert response_v2_action_constraint is not None
                if (
                    (generation_id is not None and not sess.is_generation_current(generation_id=generation_id))
                    or (request.get("_stop_check") and request["_stop_check"]())
                ):
                    response_v2_collector.cancel()
                    return None
                return response_v2_action_constraint.initial_token_id()

            def collect_v2_token_core(
                token_index: int,
                token_id: int,
                token_text: str,
                token_hidden: Any,
            ) -> tuple[int | tuple[int, str] | None, str]:
                nonlocal thinker_first_token_id
                if thinker_first_token_id is None:
                    thinker_first_token_id = int(token_id)
                if (
                    (generation_id is not None and not sess.is_generation_current(generation_id=generation_id))
                    or (request.get("_stop_check") and request["_stop_check"]())
                ):
                    response_v2_collector.cancel()
                    return None, ""
                visible_say_delta = response_v2_collector.feed_token(
                    token_index=token_index,
                    token_id=token_id,
                    text=token_text,
                    hidden=token_hidden,
                )
                assert response_v2_action_constraint is not None
                forced_id = response_v2_action_constraint.next_token_id(response_v2_collector)
                if forced_id is None:
                    return None, visible_say_delta
                return (
                    forced_id,
                    str(response_v2_action_constraint.last_forced_kind or "structure"),
                ), visible_say_delta

            def collect_v2_token(
                token_index: int,
                token_id: int,
                token_text: str,
                token_hidden: Any,
            ) -> int | tuple[int, str] | None:
                forced, _visible_say_delta = collect_v2_token_core(
                    token_index, token_id, token_text, token_hidden
                )
                return forced

            async def collect_v2_token_live(
                token_index: int,
                token_id: int,
                token_text: str,
                token_hidden: Any,
            ) -> int | tuple[int, str] | None:
                nonlocal live_v2_vocal_started
                assert live_v2_token_pacer is not None
                forced, visible_say_delta = collect_v2_token_core(
                    token_index, token_id, token_text, token_hidden
                )
                action = response_v2_collector.parser.action
                if live_v2_say_active:
                    if not isinstance(unified_branch, dict):
                        raise RuntimeError("live v2 SAY richiede l'owner unified KV del turno")
                    if live_tts_unit_assembler is None:
                        raise RuntimeError("live v2 SAY lacks its closed-unit assembler")
                    exclusive_kv_cursor = int(unified_branch["cache_position"])
                    source_text = (
                        str(visible_say_delta)
                        if action in {ControlAction.SPEAK.value, ControlAction.BACKCHANNEL.value}
                        else ""
                    )
                    vocal_text_started = bool(source_text.strip())
                    await live_v2_token_pacer.acquire(
                        vocal_started=(live_v2_vocal_started or vocal_text_started),
                        fifo_pending_packets=len(sess.speech_fifo),
                        stop_check=lambda: bool(
                            (generation_id is not None and not sess.is_generation_current(generation_id=generation_id))
                            or (request.get("_stop_check") and request["_stop_check"]())
                        ),
                    )
                    if source_text:
                        live_tts_visible_text_parts.append(source_text)
                    if vocal_text_started:
                        live_v2_vocal_started = True
                    speech_packet: SpokenTokenPacket | None = None
                    if source_text:
                        hidden = token_hidden
                        if not isinstance(hidden, torch.Tensor):
                            raise TypeError("Gemma v2 post-token hidden deve essere Tensor")
                        if hidden.ndim == 3 and tuple(hidden.shape[:2]) == (1, 1):
                            hidden = hidden[:, 0, :]
                        if tuple(hidden.shape) != (1, 1536):
                            raise ValueError(
                                f"Gemma v2 post-token hidden shape non valida: {tuple(hidden.shape)}"
                            )
                        speech_packet = SpokenTokenPacket(
                            token_id=int(token_id),
                            text_piece=source_text,
                            gemma_hidden_1536=hidden,
                            exclusive_kv_cursor=exclusive_kv_cursor,
                        )
                        with unified_branch["spoken_prefix_lock"]:
                            unified_branch["segment_token_packets"].append({
                                "token_id": int(token_id),
                                "exclusive_kv_cursor": exclusive_kv_cursor,
                                "text_piece": source_text,
                            })
                    completed_units = live_tts_unit_assembler.push_source(
                        token_id=int(token_id),
                        exclusive_kv_cursor=exclusive_kv_cursor,
                        text_piece=source_text,
                        speech_packet=speech_packet,
                    )
                    for unit in completed_units:
                        await asyncio.to_thread(enqueue_live_v2_unit, unit)
                return forced

            think_kwargs["on_generation_start"] = start_v2_generation
            think_kwargs["on_generated_token"] = (
                collect_v2_token_live if live_v2_say_active else collect_v2_token
            )
            if (
                normalized_action == ControlAction.BACKCHANNEL.value
                and _claim_audio_response_v2_action_probe()
            ):
                def log_pre_dfa_action_probe(probe: dict[str, Any]) -> None:
                    label_token_ids = probe.get("action_label_token_ids", {})
                    label_first_logits = probe.get("action_label_first_token_logits", {})
                    raw_id = int(probe.get("raw_argmax_token_id", -1))
                    sampled_id = int(probe.get("sampled_token_id", -1))
                    raw_labels = list(probe.get("raw_argmax_action_candidates", []))
                    sampled_labels = [
                        label for label in AUDIO_RESPONSE_ACTIONS
                        if label_token_ids.get(label)
                        and int(label_token_ids[label][0]) == sampled_id
                    ]
                    wake_ids = label_token_ids.get(normalized_action, [])
                    record = {
                        "response_id": (
                            request.get("_generation_token").response_id
                            if request.get("_generation_token") is not None else "unknown"
                        ),
                        "wake_action": normalized_action,
                        "allowed_response_actions": list(response_allowed_actions or ()),
                        "raw_argmax_token_id": raw_id,
                        "raw_argmax_action_candidates": raw_labels,
                        "sampled_token_id": sampled_id,
                        "sampled_action_candidates": sampled_labels,
                        "predecode_replacement_token_id": int(probe.get("forced_token_id", -1)),
                        "wake_label_first_token_id": (
                            int(wake_ids[0]) if wake_ids else None
                        ),
                        "wake_label_first_token_logit": label_first_logits.get(normalized_action),
                        "first_label_logits": label_first_logits,
                        "best_first_logit_action_candidates": list(
                            probe.get("best_first_logit_action_candidates", [])
                        ),
                        "action_label_token_ids": label_token_ids,
                    }
                    logger.info(
                        "audio_response_v2_action_probe %s",
                        json.dumps(record, ensure_ascii=True, separators=(",", ":")),
                    )

                think_kwargs["on_action_logits_probe"] = log_pre_dfa_action_probe
            think_kwargs["on_candidate_token"] = (
                lambda _token_id, token_text, terminal: response_v2_action_constraint.candidate_within_generation_budget(
                    response_v2_collector,
                    token_text,
                    terminal=terminal,
                    generated_tokens=response_v2_collector.generated_token_count,
                    budget=response_v2_budget or {},
                )
            )
            action_pool_fallbacks = [0]

            def fallback_v2_candidate(rejected):
                assert response_v2_budget is not None
                action_ids = response_v2_action_constraint.action_value_candidate_token_ids(
                    response_v2_collector
                )
                if action_ids:
                    action_pool_fallbacks[0] += 1
                    return action_ids
                closure_id = response_v2_action_constraint.bounded_field_closure_token_id(
                    response_v2_collector,
                    rejected,
                    budget=response_v2_budget,
                )
                if closure_id is not None:
                    return closure_id
                closure_id = response_v2_action_constraint.bounded_say_closure_token_id(
                    response_v2_collector,
                    rejected,
                    max_say_words=4,
                )
                if closure_id is not None:
                    return closure_id
                if response_v2_collector.backchannel_say_lexicon_active:
                    return response_v2_action_constraint.backchannel_say_candidate_token_ids(
                        response_v2_collector
                    )
                return None

            think_kwargs["on_candidate_fallback"] = fallback_v2_candidate

        def stop_live_say_worker(error: BaseException | None = None) -> None:
            if live_v2_say_active:
                cancel_live_v2_units(error)
                return
            if live_say_stream is None:
                return
            live_say_cancel.set()
            if error is None:
                live_say_stream.cancel()
            else:
                live_say_stream.fail(error)
            if live_say_thread is not None and live_say_thread.is_alive():
                live_say_thread.join(timeout=2.0)
                if live_say_thread.is_alive():
                    raise RuntimeError("live SAY worker did not stop within the bounded join")

        try:
            if unified_native:
                think = _run_unified_native_thinker(
                    engine,
                    sess,
                    request,
                    worker=unified_branch["worker"],
                    transition_token_ids=tuple(int(value) for value in ids),
                    think_kwargs=think_kwargs,
                    generation_id=(int(generation_id) if generation_id is not None else None),
                    branch=unified_branch,
                )
            else:
                think = engine.think(ids, **think_kwargs)
        except Exception as exc:
            empty_eos_think = (
                _empty_controller_say_think_from_eos(exc, prompt_len=len(ids))
                if controller_guided_say else None
            )
            if empty_eos_think is not None:
                think = empty_eos_think
            else:
                stop_live_say_worker(exc)
                if response_v2_collector is not None:
                    reason = (
                        exc.reason if isinstance(exc, AudioResponseProtocolError)
                        else "token_decode_not_prefix_stable"
                        if isinstance(exc, RuntimeError)
                        and str(exc).startswith("Thinker token decode is not prefix-stable")
                        else None
                    )
                    if reason is not None:
                        _record_audio_response_v2_private_failure(
                            collector=response_v2_collector,
                            reason=reason,
                            response_id=(
                                request.get("_generation_token").response_id
                                if request.get("_generation_token") is not None else None
                            ),
                            action=normalized_action,
                            budget=response_v2_budget,
                        )
                    response_v2_collector.cancel()
                    if reason is not None:
                        protocol_metrics = {
                            "response_protocol": AUDIO_RESPONSE_PROTOCOL_V2,
                            "status": "rejected",
                            "failure_reason": reason,
                        }
                        if callable(runtime_callback):
                            runtime_callback("audio_response_protocol", protocol_metrics)
                        logger.warning(
                            "[audio response protocol] rejected response_id=%s reason=%s",
                            request.get("_generation_token").response_id
                            if request.get("_generation_token") is not None else "unknown",
                            reason,
                        )
                raise
        if (
            unified_native
            and isinstance(unified_branch, dict)
            and unified_branch.get("queue_open") is True
            and not unified_branch.get("ephemeral_private")
            and not think.get("cancelled")
            and think.get("terminal_reason") == "eos"
            and getattr(unified_branch.get("worker"), "q2_decision_prefill_supported", False) is True
        ):
            prefill_future = unified_branch["worker"].request_q2_decision_prefill(
                sess.session_id,
                expected_generation_id=int(unified_branch["listener_generation_id"]),
                response_id=str(unified_branch["response_id"]),
            )
            prefill_future.add_done_callback(
                lambda completed: (
                    None if completed.cancelled() else completed.exception()
                )
            )
        if live_say_stream is not None:
            live_say_timing["thinker_returned_at_ns"] = time.perf_counter_ns()
        if live_say_active:
            if think.get("cancelled"):
                stop_live_say_worker()
            elif think.get("terminal_reason") != "eos":
                incomplete = RuntimeError(
                    "controller-guided live SAY requires Thinker EOS before Qwen text flush"
                )
                stop_live_say_worker(incomplete)
                raise incomplete
            else:
                try:
                    final_callback_text = re.sub(
                        r"\s*<unused\d+>\s*", " ", str(think.get("text", ""))
                    )
                    final_callback_text = re.sub(r"[ \t]+", " ", final_callback_text).strip()
                    if final_callback_text.strip():
                        live_say_stream.finish(final_text=final_callback_text)
                        if live_say_thread is not None:
                            live_say_thread.join()
                        if "error" in live_say_result:
                            raise live_say_result["error"]
                    else:
                        stop_live_say_worker()
                except BaseException as exc:
                    stop_live_say_worker(exc)
                    raise
        runtime_metrics: dict[str, Any] = {
            "thinker": dict(think.get("runtime_metrics", {})),
        }
        if controller_guided_say:
            runtime_metrics["heard_transcript"] = {
                "status": "absent_from_controller_guided_say_v1",
                "source": None,
                "note": "this response protocol does not generate an ASR transcript",
            }
        if response_v2_collector is not None and "action_pool_fallbacks" in locals():
            action_pool_overrides = min(
                int(action_pool_fallbacks[0]),
                int(runtime_metrics["thinker"].get(
                    "audio_response_protocol_v2_lexicon_pool_overrides", 0
                )),
            )
            if action_pool_overrides:
                # The engine uses a shared pool-selection counter for legacy
                # BACKCHANNEL lexicon fallbacks. Split out selected-action
                # grammar fallback so the protocol telemetry stays truthful.
                runtime_metrics["thinker"]["audio_response_protocol_v2_lexicon_pool_overrides"] -= action_pool_overrides
                runtime_metrics["thinker"]["audio_response_protocol_v2_action_pool_overrides"] = action_pool_overrides
        if direct_audio:
            runtime_metrics["direct_audio"] = {
                "processor_prepare_ms": audio_prepare_ms,
                "pcm_samples": int(np.asarray(direct_audio_pcm).size),
                "sampling_rate": USER_SAMPLE_RATE,
                "prompt_action_instruction": False,
            }
            if callable(request.get("_runtime_metrics_callback")):
                request["_runtime_metrics_callback"]("direct_audio", runtime_metrics["direct_audio"])
        if isinstance(think.get("terminal_reason"), str):
            runtime_metrics["thinker"]["terminal_reason"] = think["terminal_reason"]
        if callable(runtime_callback):
            runtime_callback("thinker", runtime_metrics["thinker"])
        generated_text = re.sub(r"\s*<unused\d+>\s*", " ", str(think["text"]))
        generated_text = re.sub(r"[ \t]+", " ", generated_text).strip()
        response_hidden = think.get("hidden")
        response_prompt_len = int(think.get("hidden_prompt_len", think.get("prompt_len", len(ids))))
        response_action_value: str | None = None
        frame_residual_response_hidden: torch.Tensor | None = None
        frame_residual_alignment_metrics: dict[str, Any] | None = None
        if private_backchannel_active:
            generated_ids = think.get("gen_ids")
            if not isinstance(generated_ids, (tuple, list)) or len(generated_ids) > 2:
                raise RuntimeError("private backchannel exceeded its two sampled-token budget")
            response_action_value = ControlAction.BACKCHANNEL.value
            response_prompt_len = 0
            runtime_metrics["private_backchannel_decode"] = {
                "status": "decoded",
                "forked_native_kv": True,
                "generated_token_ids": [int(value) for value in generated_ids],
                "sampled_text_token_limit": 2,
                "canonical_listener_mutated": False,
            }

        def empty_vocal_say_result() -> dict[str, Any]:
            no_speech_metrics = {
                "status": "no_speech",
                "audio_samples": 0,
                "reason": "empty_say",
            }
            runtime_metrics["audio_output"] = no_speech_metrics
            if callable(runtime_callback):
                runtime_callback("audio_output", no_speech_metrics)
            return {
                "text": "",
                "tool_calls": [],
                "wav": None,
                "elapsed": 0.0,
                "status": "no_speech",
                "no_speech": True,
                "history_messages": clean[-16:],
                "runtime_metrics": runtime_metrics,
                "_skip_history_commit": True,
            }

        if controller_guided_say:
            if (
                think.get("cancelled")
                or (generation_id is not None and not sess.is_generation_current(generation_id=generation_id))
                or (request.get("_stop_check") and request["_stop_check"]())
            ):
                return {
                    "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
                    "cancelled": True, "runtime_metrics": runtime_metrics,
                }
            response_action_value = controller_guided_response_action(normalized_action)
            if not generated_text.strip():
                return empty_vocal_say_result()
            generated_ids = think.get("gen_ids", [])
            decoded_text_before_trim = generated_text
            if not isinstance(generated_ids, (list, tuple)):
                raise AudioResponseProtocolError("say_hidden_token_ids_invalid")
            raw_response_hidden = response_hidden
            raw_response_prompt_len = response_prompt_len
            response_hidden = slice_controller_guided_say_hidden(
                response_hidden,
                hidden_prompt_len=response_prompt_len,
                generated_token_ids=generated_ids,
                excluded_token_ids=set(getattr(engine.tokenizer, "all_special_ids", ()) or ()) | set(range(256000, 256007)),
            )
            if not generated_text.strip():
                raise AudioResponseProtocolError("say_empty")
            response_prompt_len = 0
            generated_text = re.sub(r"\s*<unused\d+>\s*", " ", generated_text)
            generated_text = re.sub(r"[ \t]+", " ", generated_text).strip()
            generated_say_hidden_tokens = int(response_hidden.shape[1])
            if live_say_active and frame_residual_candidate:
                frame_residual_alignment_metrics = {
                    "source": "same_forward_live_gemma_hidden",
                    "source_protocol": CONTROLLER_GUIDED_SAY_V1,
                    "response_action": normalized_action,
                    "qwen_raw_text_tokens": int(
                        live_say_stream.observed_qwen_token_count
                        if live_say_stream is not None else 0
                    ),
                    "alignment": "causal_character_overlap_mean",
                    "qwen_route": "plain_native_audio_q1",
                    "qwen_prefix_slots_added": 0,
                }
            elif bool(getattr(engine, "frame_residual_adapter_enabled", False)):
                special_ids = {int(token_id) for token_id in (getattr(engine.tokenizer, "all_special_ids", ()) or ())} | set(range(256000, 256007))
                visible_generated_ids = [int(token_id) for token_id in generated_ids if int(token_id) not in special_ids]
                processor = getattr(getattr(engine, "talker", None), "proc", None)
                qwen_tokenizer = getattr(processor, "tokenizer", None)
                if processor is None or qwen_tokenizer is None:
                    raise RuntimeError("frame_residual_qwen_text_alignment_dependencies_missing")
                frame_residual_response_hidden, frame_residual_alignment_metrics = _align_generated_response_hidden_with_diagnostics(
                    response_hidden=response_hidden,
                    visible_generated_ids=visible_generated_ids,
                    gemma_tokenizer=engine.tokenizer,
                    qwen_tokenizer=qwen_tokenizer,
                    qwen_processor=processor,
                    generated_text=generated_text,
                    generated_ids=generated_ids,
                    hidden_prompt_len=raw_response_prompt_len,
                    raw_hidden=raw_response_hidden,
                )
            if response_action_value == ControlAction.BACKCHANNEL.value:
                backchannel_ok, backchannel_reason, backchannel_word_count = (
                    validate_backchannel_phrase(generated_text, max_words=4)
                )
                controller_metrics = {
                    "response_protocol": CONTROLLER_GUIDED_SAY_V1,
                    "first_thinker_token_id": thinker_first_token_id,
                    "status": "backchannel_say_accepted" if backchannel_ok else "backchannel_say_rejected",
                    "wake_action": normalized_action,
                    "response_action": response_action_value,
                    "say_word_count": backchannel_word_count,
                    "say_hidden_tokens": generated_say_hidden_tokens,
                    "guard_reason": backchannel_reason,
                    "text_rewritten": False,
                }
                runtime_metrics["audio_response_protocol"] = controller_metrics
                if callable(runtime_callback):
                    runtime_callback("audio_response_protocol", controller_metrics)
                if not backchannel_ok:
                    return {
                        "text": "",
                        "tool_calls": [],
                        "wav": None,
                        "elapsed": 0.0,
                        "status": "backchannel_say_rejected",
                        "runtime_metrics": runtime_metrics,
                        "history_messages": clean[-16:],
                        "_skip_history_commit": True,
                    }
            replay_metrics: dict[str, Any] | None = None
            if getattr(engine, "v9_conditioning_source", "live_generated") == "v9_text_replay":
                replay_result = engine.controller_say_v9_input(
                    response_hidden,
                    response_prompt_len,
                    generated_text,
                    stop_check=request.get("_stop_check"),
                )
                if replay_result is None:
                    return {
                        "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
                        "cancelled": True, "runtime_metrics": runtime_metrics,
                    }
                response_hidden, response_prompt_len, replay_metrics = replay_result
            controller_metrics = {
                "response_protocol": CONTROLLER_GUIDED_SAY_V1,
                "first_thinker_token_id": thinker_first_token_id,
                "status": "controller_action_fixed",
                "wake_action": normalized_action,
                "response_action": response_action_value,
                "selected_action": response_action_value,
                "deferred_until_endpoint": False,
                "say_chars": len(generated_text),
                "say_hidden_tokens": generated_say_hidden_tokens,
                "thinker_hidden_prompt_len": int(think.get("hidden_prompt_len", len(ids))),
                "generated_token_count": len(generated_ids),
                "hidden_source": "post_token_generated_say_only",
                "v9_conditioning_source": getattr(engine, "v9_conditioning_source", "live_generated"),
            }
            if replay_metrics is not None:
                controller_metrics.update({
                    "hidden_source": "text_only_teacher_forced_replay",
                    "decoded_text_before_trim": decoded_text_before_trim,
                    "say_text_after_trim": generated_text,
                    "generated_token_ids": [int(token_id) for token_id in generated_ids],
                    "text_replay": replay_metrics,
                })
            runtime_metrics["audio_response_protocol"] = controller_metrics
            if callable(runtime_callback):
                runtime_callback("audio_response_protocol", controller_metrics)
            think["text"] = generated_text
            think["hidden"] = response_hidden
            think["hidden_prompt_len"] = response_prompt_len
            think["prompt_len"] = response_prompt_len
            think["gen_ids"] = []
            think["past_key_values"] = None
            if stream_requested:
                render_visible_text_chunk(
                    generated_text,
                    response_hidden[:, response_prompt_len:, :],
                    0,
                )
        if response_v2_collector is not None:
            if (
                think.get("cancelled")
                or (generation_id is not None and not sess.is_generation_current(generation_id=generation_id))
                or (request.get("_stop_check") and request["_stop_check"]())
            ):
                if live_v2_say_active:
                    stop_live_say_worker()
                response_v2_collector.cancel()
                return {
                    "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
                    "cancelled": True, "runtime_metrics": runtime_metrics,
                }
            try:
                protocol_text = think.get("protocol_text")
                if not isinstance(protocol_text, str):
                    raise AudioResponseProtocolError("protocol_text_missing")
                parsed_response, response_hidden = finish_audio_response_v2_with_say_hidden(
                    response_v2_collector,
                    expected_text=protocol_text,
                )
                response_action_value = str(parsed_response["action"])
            except AudioResponseProtocolError as exc:
                if live_v2_say_active:
                    stop_live_say_worker(exc)
                _record_audio_response_v2_private_failure(
                    collector=response_v2_collector,
                    reason=exc.reason,
                    response_id=(
                        request.get("_generation_token").response_id
                        if request.get("_generation_token") is not None else None
                    ),
                    action=normalized_action,
                    budget=response_v2_budget,
                )
                response_v2_collector.cancel()
                protocol_metrics = {
                    "response_protocol": AUDIO_RESPONSE_PROTOCOL_V2,
                    "first_thinker_token_id": thinker_first_token_id,
                    "status": "rejected",
                    "failure_reason": exc.reason,
                }
                runtime_metrics["audio_response_protocol"] = protocol_metrics
                if callable(runtime_callback):
                    runtime_callback("audio_response_protocol", protocol_metrics)
                logger.warning(
                    "[audio response protocol] rejected response_id=%s reason=%s",
                    request.get("_generation_token").response_id
                    if request.get("_generation_token") is not None else "unknown",
                    exc.reason,
                )
                raise
            parsed_callback = request.get("_on_audio_response_v2_parsed")
            parsed_disposition = None
            if callable(parsed_callback):
                parsed_disposition = parsed_callback(
                    dict(parsed_response),
                    bool(response_v2_collector.heard_truncated),
                )
            if parsed_disposition == "deferred_until_endpoint":
                if live_v2_say_active:
                    stop_live_say_worker()
                runtime_metrics["audio_response_protocol"] = {
                    "response_protocol": AUDIO_RESPONSE_PROTOCOL_V2,
                    "first_thinker_token_id": thinker_first_token_id,
                    "status": "deferred_until_endpoint",
                    "wake_action": normalized_action or "",
                    "response_action": parsed_response["action"],
                    "selected_action": parsed_response["action"],
                    "action_constraint": "native_choice_with_wake_allowlist",
                    "action_forced": False,
                    "heard_truncated": bool(response_v2_collector.heard_truncated),
                    "say_chars": 0,
                    "say_hidden_tokens": 0,
                }
                if callable(runtime_callback):
                    runtime_callback("audio_response_protocol", runtime_metrics["audio_response_protocol"])
                return {
                    "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
                    "status": "deferred_until_endpoint",
                    "response_action": parsed_response["action"],
                    "runtime_metrics": runtime_metrics,
                    "history_messages": clean[-16:],
                    "_skip_history_commit": True,
                }
            if parsed_disposition == "stale_generation":
                if live_v2_say_active:
                    stop_live_say_worker()
                return {
                    "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
                    "cancelled": True, "runtime_metrics": runtime_metrics,
                    "_skip_history_commit": True,
                }
            if live_v2_say_active:
                validated_say = str(parsed_response["say"])
                if (
                    parsed_response["action"] in {
                        ControlAction.SPEAK.value,
                        ControlAction.BACKCHANNEL.value,
                    }
                    and validated_say
                ):
                    if "".join(live_tts_visible_text_parts) != validated_say:
                        raise RuntimeError(
                            "closed-unit Gemma SAY source differs from validated v2 text"
                        )
                    if live_tts_unit_assembler is None:
                        raise RuntimeError(
                            "validated v2 SAY did not initialize its closed-unit assembler"
                        )
                    try:
                        finish_live_v2_units()
                        if "error" in live_say_result:
                            raise live_say_result["error"]
                    except BaseException as exc:
                        stop_live_say_worker(exc)
                        raise
                else:
                    stop_live_say_worker()
            say_tokens = response_v2_collector.say_tokens
            response_prompt_len = 0
            generated_text = parsed_response["say"]
            input_transcript_callback = request.get("_on_input_transcript_completed")
            input_transcript_context = request.get("_input_transcript_context")
            if (
                parsed_response["action"] in {"SPEAK", "REPLAN"}
                and not response_v2_collector.heard_truncated
                and isinstance(input_transcript_context, dict)
                and callable(input_transcript_callback)
            ):
                input_transcript_callback(
                    str(parsed_response["heard"]),
                    dict(input_transcript_context),
                )
            generated_token_count = len(think.get("gen_ids", []))
            runtime_metrics["audio_response_protocol"] = {
                "response_protocol": AUDIO_RESPONSE_PROTOCOL_V2,
                "first_thinker_token_id": thinker_first_token_id,
                "status": "valid",
                "response_action": parsed_response["action"],
                "wake_action": normalized_action or "",
                "selected_action": parsed_response["action"],
                "allowed_response_actions": list(response_allowed_actions or AUDIO_RESPONSE_ACTIONS),
                "action_constraint": "native_choice_with_wake_allowlist" if response_allowed_actions else "unconstrained_native",
                "action_forced": bool(
                    response_v2_action_constraint
                    and response_v2_action_constraint.forced_action_token_count
                ),
                "forced_action_token_count": (
                    response_v2_action_constraint.forced_action_token_count
                    if response_v2_action_constraint is not None else 0
                ),
                "forced_action_override_count": int(
                    think.get("runtime_metrics", {}).get("forced_action_override_count", 0)
                ),
                "forced_structure_token_count": int(
                    response_v2_action_constraint.forced_structure_token_count
                    if response_v2_action_constraint is not None else 0
                ),
                "forced_structure_override_count": int(
                    think.get("runtime_metrics", {}).get("forced_structure_override_count", 0)
                ),
                "structure_forced": bool(
                    response_v2_action_constraint
                    and response_v2_action_constraint.forced_structure_token_count
                ),
                "bounded_output": bool(
                    response_v2_action_constraint
                    and response_v2_action_constraint.bounded_output
                ),
                "generation_budget": dict(response_v2_budget or {}),
                "heard_field_tokens": int(response_v2_collector.heard_field_token_count),
                "action_field_tokens": int(response_v2_collector.action_field_token_count),
                "say_field_tokens": int(response_v2_collector.say_field_token_count),
                "heard_truncated": bool(response_v2_collector.heard_truncated),
                "say_truncated": bool(response_v2_collector.say_truncated),
                "predecode_candidate_checks": int(
                    think.get("runtime_metrics", {}).get(
                        "audio_response_protocol_v2_candidate_checks", 0
                    )
                ),
                "predecode_candidate_replacements": int(
                    think.get("runtime_metrics", {}).get(
                        "audio_response_protocol_v2_predecode_replacements", 0
                    )
                ),
                "predecode_bounded_closures": int(
                    think.get("runtime_metrics", {}).get(
                        "audio_response_protocol_v2_bounded_closures", 0
                    )
                ),
                "backchannel_say_lexicon_pool_overrides": int(
                    think.get("runtime_metrics", {}).get(
                        "audio_response_protocol_v2_lexicon_pool_overrides", 0
                    )
                ),
                "backchannel_say_lexicon_constrained": bool(
                    response_v2_collector.backchannel_say_lexicon_active
                ),
                "backchannel_say_lexicon_phrase_count": int(
                    response_v2_collector.backchannel_say_lexicon_phrase_count
                ),
                "backchannel_say_lexicon_rejections": int(
                    response_v2_collector.backchannel_say_lexicon_rejections
                ),
                "say_chars": len(generated_text),
                "output_token_count": generated_token_count,
                "say_hidden_tokens": len(say_tokens),
                "say_token_spans": [
                    {
                        "token_index": int(token.token_index),
                        "token_id": int(token.token_id),
                        "char_start": int(token.char_span[0]),
                        "char_end": int(token.char_span[1]),
                    }
                    for token in say_tokens
                ],
            }
            native_action_logprob = think.get("runtime_metrics", {}).get("native_action_logprob")
            if isinstance(native_action_logprob, (int, float)) and not isinstance(native_action_logprob, bool):
                runtime_metrics["audio_response_protocol"]["native_action_logprob"] = float(native_action_logprob)
            if callable(runtime_callback):
                runtime_callback("audio_response_protocol", runtime_metrics["audio_response_protocol"])
            think["text"] = generated_text
            think["hidden"] = response_hidden
            think["hidden_prompt_len"] = 0
            think["gen_ids"] = []
            think["past_key_values"] = None
            if stream_requested:
                render_visible_text_chunk(generated_text, response_hidden, 0)
        if any(marker in generated_text for marker in private_control_markers):
            raise _PrivateControlLeakError(normalized_action or "", generated_text)
        if held_marker_suffix:
            render_visible_text_chunk(
                held_marker_suffix,
                think.get("hidden"),
                int(think.get("prompt_len", len(ids))),
            )
            held_marker_suffix = ""
        if request.get("_stop_check") and request["_stop_check"]():
            if live_v2_say_active:
                stop_live_say_worker()
            return {
                "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
                "cancelled": True, "runtime_metrics": runtime_metrics,
            }
        calls = extract_tool_calls(think["text"], tools) if tools else []
        if calls:
            return {"text": "", "tool_calls": calls, "wav": None, "elapsed": 0.0}
        text = generated_text
        is_vocal_action = normalized_action in {
            ControlAction.SPEAK.value,
            ControlAction.BACKCHANNEL.value,
        }
        if is_vocal_action and not text.strip():
            if live_v2_say_active:
                stop_live_say_worker()
            return empty_vocal_say_result()
        if request.get("_generation_token") is not None:
            sess.record_planned_text(request["_generation_token"], text)
        is_backchannel_action = normalized_action == ControlAction.BACKCHANNEL.value
        frame_residual_adapter = getattr(engine, "frame_residual_adapter", None)
        frame_residual_enabled = bool(getattr(engine, "frame_residual_adapter_enabled", False))
        frame_residual_candidate = bool(
            frame_residual_enabled
            and frame_residual_adapter is not None
            and audio_requested
            and (is_vocal_action or not direct_audio)
        )
        if (is_vocal_action or frame_residual_candidate) and audio_requested:
            if frame_residual_enabled != (frame_residual_adapter is not None):
                raise RuntimeError("frame_residual_adapter_load_state_inconsistent")
            if (
                frame_residual_candidate
                and response_v2_collector is not None
                and not live_v2_say_active
            ):
                say_tokens = response_v2_collector.say_tokens
                say_spans = audio_response_v2_say_token_char_spans(
                    say_tokens,
                    generated_text,
                )
                processor = getattr(getattr(engine, "talker", None), "proc", None)
                qwen_tokenizer = getattr(processor, "tokenizer", None)
                if processor is None or qwen_tokenizer is None:
                    raise RuntimeError("frame_residual_qwen_text_alignment_dependencies_missing")
                qwen_token_ids, qwen_offsets = qwen_assistant_text_offsets(
                    generated_text,
                    tokenizer=qwen_tokenizer,
                    processor=processor,
                )
                frame_residual_response_hidden = align_hidden_states_by_char_overlap(
                    response_hidden,
                    say_spans,
                    qwen_offsets,
                )
                frame_residual_alignment_metrics = {
                    "source": "validated_v2_say_posttoken_hidden",
                    "source_protocol": AUDIO_RESPONSE_PROTOCOL_V2,
                    "response_action": str(parsed_response["action"]),
                    "gemma_say_hidden_tokens": int(response_hidden.shape[1]),
                    "qwen_raw_text_tokens": len(qwen_token_ids),
                    "text_chars": len(generated_text),
                    "alignment": "collector_say_span_character_overlap_mean",
                }
            if (
                frame_residual_candidate
                and frame_residual_response_hidden is None
                and not (live_say_active or live_v2_say_active)
            ):
                if any(marker in generated_text for marker in private_control_markers):
                    raise _PrivateControlLeakError(normalized_action or "", generated_text)
                if controller_guided_say:
                    raise RuntimeError("frame_residual_response_hidden_alignment_missing")
                generated_ids = think.get("gen_ids")
                if not isinstance(generated_ids, (list, tuple)) or not generated_ids:
                    raise RuntimeError("frame_residual_generated_say_token_ids_missing")
                raw_response_hidden = response_hidden
                raw_response_prompt_len = response_prompt_len
                response_hidden = slice_controller_guided_say_hidden(
                    response_hidden,
                    hidden_prompt_len=response_prompt_len,
                    generated_token_ids=generated_ids,
                    excluded_token_ids=set(getattr(engine.tokenizer, "all_special_ids", ()) or ()) | set(range(256000, 256007)),
                )
                response_prompt_len = 0
                special_ids = {
                    int(token_id)
                    for token_id in (getattr(engine.tokenizer, "all_special_ids", ()) or ())
                } | set(range(256000, 256007))
                visible_generated_ids = [
                    int(token_id) for token_id in generated_ids if int(token_id) not in special_ids
                ]
                processor = getattr(getattr(engine, "talker", None), "proc", None)
                qwen_tokenizer = getattr(processor, "tokenizer", None)
                if processor is None or qwen_tokenizer is None:
                    raise RuntimeError("frame_residual_qwen_text_alignment_dependencies_missing")
                frame_residual_response_hidden, frame_residual_alignment_metrics = (
                    _align_generated_response_hidden_with_diagnostics(
                        response_hidden=response_hidden,
                        visible_generated_ids=visible_generated_ids,
                        gemma_tokenizer=engine.tokenizer,
                        qwen_tokenizer=qwen_tokenizer,
                        qwen_processor=processor,
                        generated_text=generated_text,
                        generated_ids=generated_ids,
                        hidden_prompt_len=raw_response_prompt_len,
                        raw_hidden=raw_response_hidden,
                    )
                )
                if frame_residual_response_hidden is None or int(frame_residual_response_hidden.shape[1]) < 1:
                    raise RuntimeError("frame_residual_response_hidden_alignment_empty")
                frame_residual_alignment_metrics = dict(frame_residual_alignment_metrics or {})
                frame_residual_alignment_metrics.update({
                    "source": "gemma_assistant_response_posttoken_hidden",
                    "source_protocol": response_protocol or "text_action_conditioned",
                    "response_action": normalized_action,
                })
        if frame_residual_candidate and getattr(engine, "response_intent_local_projector", None) is not None:
            raise RuntimeError("frame residual Q1 adapter cannot share a response with a prefix projector")
        is_backchannel = is_backchannel_action and not direct_audio and not frame_residual_candidate
        if is_backchannel and not text.strip():
            runtime_metrics["backchannel"] = {
                "backchannel_done_reason": "empty_say",
                "status": "empty",
                "talker_started": False,
            }
            return {
                "text": "",
                "tool_calls": [],
                "wav": None,
                "elapsed": 0.0,
                "status": "no_speech",
                "runtime_metrics": runtime_metrics,
            }
        response_text_conditioning = None
        if frame_residual_candidate:
            h_pfx = torch.empty((1, 0, 2048), device=engine.device, dtype=engine.dtype)
            frame_residual_alignment_metrics = dict(frame_residual_alignment_metrics or {})
            frame_residual_architecture = str(
                getattr(frame_residual_adapter, "architecture_id", "unknown")
            )
            frame_residual_alignment_metrics.update({
                "candidate": (
                    "frame_residual_q1_query_film_v2"
                    if frame_residual_architecture == "response_hidden_q1_query_film_v2"
                    else "frame_residual_q1_input_v1"
                ),
                "adapter_mode": frame_residual_architecture,
                "qwen_semantic_text_unchanged": True,
                "qwen_prefix_slots_added": 0,
                "qwen_route": "plain_native_audio_q1",
                "checkpoint": getattr(engine, "frame_residual_adapter_resolved_path", None),
            })
            runtime_metrics["frame_residual_adapter"] = frame_residual_alignment_metrics
            runtime_metrics["v9_prefix_route"] = {
                "source": "frame_residual_q1_no_prefix",
                "prefix_frame_count": 0,
            }
            policy_hidden = sess.listener_hidden if duplex_realtime else None
            if callable(runtime_callback):
                runtime_callback("frame_residual_adapter", frame_residual_alignment_metrics)
        elif is_backchannel and not controller_guided_say:
            with sess._generation_lock:
                listener_hidden = sess.listener_hidden
            hidden_dim = getattr(listener_hidden, "shape", ())[-1] if listener_hidden is not None else None
            if hidden_dim != 2048 or not callable(getattr(engine, "talk", None)):
                return {
                    "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
                    "skip_reason": "backchannel_conditioning_unavailable",
                    "runtime_metrics": runtime_metrics,
                }
            hidden_rank = len(getattr(listener_hidden, "shape", ()))
            if hidden_rank == 2:
                h_pfx = (
                    listener_hidden.unsqueeze(1)
                    if callable(getattr(listener_hidden, "unsqueeze", None))
                    else np.expand_dims(np.asarray(listener_hidden), axis=1)
                )
            else:
                h_pfx = listener_hidden
            policy_hidden = listener_hidden
        elif duplex_realtime and response_v2_collector is None and not controller_guided_say:
            h_pfx = engine.v9_prefix(
                response_hidden, response_prompt_len
            )
            policy_hidden = sess.listener_hidden if sess.listener_hidden is not None else h_pfx
        elif response_v2_collector is not None or controller_guided_say:
            # Match the known stateless Qwen path: public SAY V9 is prefetched
            # once before BOS, and subsequent codec frames are conditioned by
            # Qwen's own audio-query hidden states.  The live listener remains
            # available to policy/cancellation but is never a phonetic Talker
            # input for protocol v2.
            if live_say_active or live_v2_say_active:
                h_pfx = torch.empty((1, 0, 2048), device=engine.device, dtype=engine.dtype)
                runtime_metrics["v9_prefix_route"] = {
                    "source": "controller_guided_live_qwen_text_no_prefix",
                    "prefix_frame_count": 0,
                    "prefix_kv_slots": 0,
                }
                if frame_residual_candidate:
                    runtime_metrics["frame_residual_adapter"] = dict(
                        frame_residual_alignment_metrics or {}
                    )
                    runtime_metrics["frame_residual_adapter"].update({
                        "candidate": "frame_residual_q1_query_film_v2"
                        if str(getattr(frame_residual_adapter, "architecture_id", ""))
                        == "response_hidden_q1_query_film_v2"
                        else "frame_residual_q1_input_v1",
                        "adapter_mode": str(
                            getattr(frame_residual_adapter, "architecture_id", "unknown")
                        ),
                        "qwen_semantic_text_unchanged": True,
                        "qwen_route": "plain_native_audio_q1",
                        "checkpoint": getattr(
                            engine, "frame_residual_adapter_resolved_path", None
                        ),
                    })
                    if callable(runtime_callback):
                        runtime_callback(
                            "frame_residual_adapter",
                            runtime_metrics["frame_residual_adapter"],
                        )
                policy_hidden = sess.listener_hidden if duplex_realtime else None
            elif frame_residual_candidate:
                if frame_residual_response_hidden is None:
                    raise RuntimeError("frame_residual_response_hidden_alignment_missing")
                h_pfx = torch.empty((1, 0, 2048), device=engine.device, dtype=engine.dtype)
                frame_residual_alignment_metrics = dict(frame_residual_alignment_metrics or {})
                frame_residual_architecture = str(
                    getattr(frame_residual_adapter, "architecture_id", "unknown")
                )
                frame_residual_alignment_metrics.update({
                    "candidate": (
                        "frame_residual_q1_query_film_v2"
                        if frame_residual_architecture == "response_hidden_q1_query_film_v2"
                        else "frame_residual_q1_input_v1"
                    ),
                    "adapter_mode": frame_residual_architecture,
                    "qwen_semantic_text_unchanged": True, "qwen_prefix_slots_added": 0,
                    "qwen_route": "plain_native_audio_q1",
                    "checkpoint": getattr(engine, "frame_residual_adapter_resolved_path", None),
                })
                runtime_metrics["frame_residual_adapter"] = frame_residual_alignment_metrics
                if callable(runtime_callback):
                    runtime_callback("frame_residual_adapter", frame_residual_alignment_metrics)
                runtime_metrics["v9_prefix_route"] = {"source": "frame_residual_q1_no_prefix", "prefix_frame_count": 0}
            elif controller_guided_say and bool(getattr(engine, "ablate_response_projector", False)):
                # Genuine plain-text Qwen control: text embeddings and BOS
                # remain in engine.talk's prefill; no projector KV slot exists.
                h_pfx = torch.empty((1, 0, 2048), device=engine.device, dtype=engine.dtype)
                ablation_metrics = {
                    "mode": "plain_qwen_text_only_no_prefix_kv",
                    "prefix_kv_slots": 0,
                    "qwen_text_prefill_unchanged": True,
                    "qwen_trailing_text_hidden_unchanged": True,
                    "talker_backend": getattr(engine, "talker_q1_backend", None),
                    "response_action": response_action_value,
                }
                runtime_metrics["response_projector_ablation"] = ablation_metrics
                runtime_metrics["v9_prefix_route"] = {
                    "source": "explicit_no_projector_ablation",
                    "prefix_frame_count": 0,
                    "prefix_kv_slots": 0,
                }
                if callable(runtime_callback):
                    runtime_callback("response_projector_ablation", ablation_metrics)
            elif controller_guided_say and getattr(engine, "response_intent_local_projector", None) is not None:
                from duplex_kernel.response_intent_local import build_local_response_prefix

                special_ids = {
                    int(token_id)
                    for token_id in (getattr(engine.tokenizer, "all_special_ids", ()) or ())
                }
                local_token_ids = [
                    int(token_id) for token_id in generated_ids if int(token_id) not in special_ids
                ]
                qwen_processor = getattr(engine.talker, "proc", None)
                qwen_tokenizer = getattr(qwen_processor, "tokenizer", None)
                h_pfx, local_cursor_metrics = build_local_response_prefix(
                    response_hidden,
                    local_token_ids,
                    gemma_tokenizer=engine.tokenizer,
                    qwen_tokenizer=qwen_tokenizer,
                    qwen_processor=qwen_processor,
                    text=generated_text,
                    projector=engine.response_intent_local_projector,
                )
                local_cursor_metrics.update({
                    "checkpoint": getattr(engine, "response_intent_local_checkpoint_path", None),
                    "response_action": response_action_value,
                    "say_hidden_tokens": int(response_hidden.shape[1]),
                    "generated_non_special_token_ids": len(local_token_ids),
                    "qwen_text_part_unchanged": True,
                    "alignment_scope": "character overlap to Qwen token-step cursor; not acoustic word timing",
                })
                runtime_metrics["response_intent_local_cursor"] = local_cursor_metrics
                runtime_metrics["v9_prefix_route"] = {
                    "source": "gemma_response_local_hidden_by_qwen_text_cursor",
                    "projector": "frozen response_intent_local_cursor_v1 linear",
                    "prefix_frame_count": int(h_pfx.shape[1]),
                }
                if callable(runtime_callback):
                    runtime_callback("response_intent_local_cursor", local_cursor_metrics)
            elif v9_audio_prefill_frame_raw_enabled:
                audio_snapshot = think.get("audio_context_snapshot")
                audio_context_hidden = getattr(audio_snapshot, "hidden", None)
                if audio_context_hidden is None:
                    raise RuntimeError(
                        "audio_prefill_frame_raw_capture_missing: refusing SAY-hidden fallback"
                    )
                if int(getattr(audio_snapshot, "audio_placeholder_run_count", 0)) != 1:
                    raise RuntimeError("audio_prefill_frame_raw_requires_exactly_one_user_audio_run")
                source_pcm_samples = getattr(audio_snapshot, "source_pcm_samples", None)
                if source_pcm_samples is None:
                    raise RuntimeError("audio_prefill_frame_raw_source_pcm_length_missing")
                expected_raw_frames = (
                    int(source_pcm_samples) + POLICY_FRAME_SAMPLES - 1
                ) // POLICY_FRAME_SAMPLES
                if int(audio_context_hidden.shape[1]) != expected_raw_frames:
                    raise RuntimeError(
                        "audio_prefill_frame_raw_source_frame_count_mismatch: "
                        f"selected={int(audio_context_hidden.shape[1])} "
                        f"expected={expected_raw_frames}"
                    )
                project = getattr(engine, "v9_prefix_stateful", None)
                if not callable(project):
                    raise RuntimeError("audio_prefill_frame_raw requires v9_prefix_stateful")
                projection_kwargs = {}
                if getattr(engine, "response_talk_context_pool", None) is not None:
                    projection_kwargs = {
                        "listener_admission_wait": listener_admission_wait,
                        "listener_priority": True,
                        "stop_check": lambda: bool(
                            sess.stop_requested
                            or (generation_id is not None and not sess.is_generation_current(
                                generation_id=generation_id
                            ))
                        ),
                        "metrics": runtime_metrics,
                    }
                # prompt_len=0 makes the first selected 80 ms audio state the
                # causal V9 anchor. SAY text/IDs remain untouched for Qwen.
                prefix_sequence = project(audio_context_hidden, 0, **projection_kwargs)
                h_pfx = getattr(prefix_sequence, "h_pfx", None)
                if not isinstance(h_pfx, torch.Tensor) or int(h_pfx.shape[1]) != int(audio_context_hidden.shape[1]):
                    raise RuntimeError("audio_prefill_frame_raw_v9_prefix_alignment_mismatch")
                direct_origin = request.get("_direct_audio_context")
                if not isinstance(direct_origin, dict):
                    direct_origin = {}
                raw_audio_v9_metrics = {
                    "branch": "audio_prefill_frame_raw",
                    "projector": "IncrementalV9Projector.step",
                    "source": "current_user_audio_placeholder_final_lm_hidden",
                    "source_gap": "training_user_plus_delayed_self_interleaved_vs_runtime_user_only",
                    "audio_placeholder_count": int(audio_snapshot.audio_placeholder_count),
                    "audio_placeholder_run_count": int(audio_snapshot.audio_placeholder_run_count),
                    "audio_placeholder_run_start": int(audio_snapshot.audio_placeholder_run_start),
                    "audio_placeholder_run_end": int(audio_snapshot.audio_placeholder_run_end),
                    "audio_ms_per_token": int(audio_snapshot.audio_ms_per_token),
                    "raw_frame_period_ms": int(audio_snapshot.raw_frame_period_ms),
                    "audio_frame_count": int(audio_context_hidden.shape[1]),
                    "prefix_frame_count": int(h_pfx.shape[1]),
                    "source_first_frame_id": direct_origin.get("_direct_audio_first_frame_id"),
                    "source_last_frame_id": direct_origin.get("_direct_audio_last_frame_id"),
                    "source_frame_ids": (
                        list(range(
                            int(direct_origin["_direct_audio_first_frame_id"]),
                            int(direct_origin["_direct_audio_first_frame_id"])
                            + int(audio_context_hidden.shape[1]),
                        ))
                        if direct_origin.get("_direct_audio_first_frame_id") is not None
                        else None
                    ),
                    "say_text_preserved": True,
                }
                runtime_metrics["v9_prefix_route"] = raw_audio_v9_metrics
                if callable(runtime_callback):
                    runtime_callback("v9_prefix_route", raw_audio_v9_metrics)
            elif v9_dual_audio_context_enabled:
                audio_snapshot = think.get("audio_context_snapshot")
                audio_context_hidden = getattr(audio_snapshot, "hidden", None)
                if audio_context_hidden is None:
                    raise RuntimeError(
                        "v9_dual_audio_context_capture_missing: refusing SAY-only fallback"
                    )
                if not isinstance(response_hidden, torch.Tensor) or response_hidden.ndim != 3:
                    raise RuntimeError("v9_dual_audio_context_say_hidden_invalid")
                semantic_anchor_hidden = response_hidden.mean(dim=1)
                v9_prefix_sequence = engine.v9_prefix_audio_context(
                    audio_context_hidden,
                    semantic_anchor_hidden,
                )
                h_pfx = v9_prefix_sequence.h_pfx
                dual_v9_metrics = {
                    "branch": "experimental_audio_positions_plus_say_anchor",
                    "audio_placeholder_count": int(
                        getattr(audio_snapshot, "audio_placeholder_count", 0)
                    ),
                    "audio_frame_count": int(audio_context_hidden.shape[1]),
                    "audio_ms_per_token": int(
                        getattr(audio_snapshot, "audio_ms_per_token", 0)
                    ),
                    "raw_frame_period_ms": int(
                        getattr(audio_snapshot, "raw_frame_period_ms", 0)
                    ),
                    "source_pcm_samples": getattr(
                        audio_snapshot, "source_pcm_samples", None
                    ),
                    "say_hidden_token_count": int(response_hidden.shape[1]),
                    "projected_prefix_frames": int(h_pfx.shape[1]),
                }
                runtime_metrics["v9_dual_audio_context"] = dual_v9_metrics
                if callable(runtime_callback):
                    runtime_callback("v9_dual_audio_context", dual_v9_metrics)
            else:
                if controller_guided_say:
                    project = getattr(engine, "v9_prefix_stateful", None)
                    if not callable(project):
                        raise RuntimeError(
                            "controller-guided SAY requires fresh v9_prefix_stateful projection"
                        )
                    projection_kwargs = {}
                    if (
                        response_v2_collector is None
                        and getattr(engine, "response_talk_context_pool", None) is not None
                    ):
                        projection_kwargs = {
                            "listener_admission_wait": listener_admission_wait,
                            "listener_priority": True,
                            "stop_check": lambda: bool(
                                sess.stop_requested
                                or (generation_id is not None and not sess.is_generation_current(
                                    generation_id=generation_id
                                ))
                            ),
                            "metrics": runtime_metrics,
                        }
                    prefix_sequence = project(
                        response_hidden, response_prompt_len, **projection_kwargs
                    )
                    h_pfx = getattr(prefix_sequence, "h_pfx", None)
                    if not isinstance(h_pfx, torch.Tensor):
                        raise RuntimeError("v9_prefix_stateful did not return tensor h_pfx")
                    runtime_metrics["v9_prefix_route"] = {
                        "source": "generated_say_hidden",
                        "projector": "IncrementalV9Projector.step",
                        "raw_token_count": int(response_hidden.shape[1]),
                        "prefix_frame_count": int(h_pfx.shape[1]),
                    }
                else:
                    h_pfx = engine.v9_prefix(response_hidden, response_prompt_len)
            policy_hidden = sess.listener_hidden if duplex_realtime else h_pfx
        else:
            h_pfx = engine.v9_prefix(
                response_hidden, response_prompt_len
            )
            policy_hidden = h_pfx
        controller_stateful_pool = bool(
            controller_guided_say
            and not live_say_active
            and not frame_residual_candidate
            and not bool(getattr(engine, "ablate_response_projector", False))
            and response_v2_collector is None
            and getattr(engine, "response_talk_context_pool", None) is not None
        )
        use_stateful_speaker = (
            duplex_realtime
            and not frame_residual_candidate
            and response_v2_collector is None
            and callable(getattr(engine, "talk_stateful", None))
            and controller_stateful_pool
        )
        if policy_hidden is not None:
            with sess._generation_lock:
                if sess.turn_policy is not None and (generation_id is None or sess.is_generation_current(generation_id=generation_id)):
                    with sess._policy_lock:
                        if sess.turn_policy is not None:
                            sess.turn_policy.update_hidden(policy_hidden)
                elif generation_id is not None:
                    sess.stale_session_mutations += 1
        wav = None
        if "audio" in (request.get("modalities") or ["text"]):
            if live_say_active or live_v2_say_active:
                if live_say_thread is None:
                    raise RuntimeError("live SAY ended without starting its Qwen Talker consumer")
                if "error" in live_say_result:
                    raise live_say_result["error"]
                wav = live_say_result.get("wav")
                talker_metrics = dict(live_say_result.get("metrics") or {})
                talker_metrics.update({
                    "response_text_conditioning_source": (
                        (
                            "audio_response_protocol_v2.live_turn_local_qwen_text_tail"
                            if live_v2_say_active
                            else "controller_guided_say_v1.live_native_qwen_text_tail"
                        )
                    ),
                    "live_qwen_token_cursor_start": 1,
                    "live_qwen_text_eos_consumed": bool(
                        live_say_stream.codec_eos_allowed
                        if live_say_stream is not None else False
                    ),
                    "live_say_consumer_started_at_ns": live_say_timing.get(
                        "consumer_started_at_ns"
                    ),
                    "live_say_thinker_returned_at_ns": live_say_timing.get(
                        "thinker_returned_at_ns"
                    ),
                    "live_say_consumer_started_before_thinker_return": bool(
                        live_say_timing.get("consumer_started_at_ns") is not None
                        and live_say_timing.get("thinker_returned_at_ns") is not None
                        and int(live_say_timing["consumer_started_at_ns"])
                        < int(live_say_timing["thinker_returned_at_ns"])
                    ),
                    "listener_v9_talker_injection_frames": 0,
                })
                if live_v2_say_active:
                    talker_metrics.update({
                        "turn_local_fifo_capacity": int(sess.speech_fifo.capacity),
                        "turn_local_fifo_remaining_packets": len(sess.speech_fifo),
                        "turn_local_fifo_peak_packets": int(sess.speech_fifo.peak_depth),
                        "turn_local_last_vocalized_kv_position": (
                            sess.speech_fifo.last_vocalized_kv_position
                        ),
                        "turn_local_vocalized_cursor_is_playback_ack": False,
                        "live_gemma_token_pacer": (
                            live_v2_token_pacer.snapshot()
                            if live_v2_token_pacer is not None else {}
                        ),
                        "realtime_pcm_packet_queue_capacity": REALTIME_AUDIO_PACKET_CAPACITY,
                        "speaker_playout_buffer_max_frames": 256,
                    })
                    logger.info("%s", _format_turn_local_speech_summary(
                        sess.session_id[:8], talker_metrics
                    ))
                runtime_metrics["talker"] = talker_metrics
                if frame_residual_candidate:
                    residual_fields = {
                        key: value for key, value in talker_metrics.items()
                        if key.startswith("frame_residual_")
                    }
                    residual_fields.update({
                        "status": (
                            "applied"
                            if int(talker_metrics.get("frame_residual_calls", 0)) > 0
                            else "no_q1_audio_steps"
                        ),
                        "qwen_route": "plain_native_audio_q1",
                        "qwen_prefix_slots_added": 0,
                        "qwen_semantic_text_unchanged": True,
                        "source": "same_forward_live_gemma_hidden",
                    })
                    runtime_metrics["frame_residual_adapter"] = residual_fields
                    if callable(runtime_callback):
                        runtime_callback("frame_residual_adapter", residual_fields)
                if callable(runtime_callback):
                    runtime_callback("talker", talker_metrics)
            elif is_backchannel and not use_stateful_speaker:
                talker_metrics: dict[str, Any] = (
                    {
                        "diagnostic_rng_seed": int(diagnostic_talker_seed),
                        "diagnostic_rng_scope": "startup_global_not_per_response",
                    }
                    if diagnostic_talker_seed is not None else {}
                )
                with _diagnostic_talker_rng_scope(engine):
                    wav = engine.talk(
                        text,
                        h_pfx,
                        n_words=max(1, len(text.split())),
                        voice_conditioning=cond,
                        stop_check=lambda: bool(request.get("_stop_check") and request["_stop_check"]()),
                        state=sess,
                        on_audio_chunk=request.get("_on_audio_chunk"),
                        generation_id=generation_id,
                        min_frames=2,
                        metrics=talker_metrics,
                    )
                runtime_metrics["talker"] = talker_metrics
                if frame_residual_candidate:
                    residual_fields = {k: v for k, v in talker_metrics.items() if k.startswith("frame_residual_")}
                    residual_fields.update({"status": "applied" if int(talker_metrics.get("frame_residual_calls", 0)) > 0 else "loaded_but_not_applied", "qwen_route": "plain_native_audio_q1", "qwen_prefix_slots_added": 0, "qwen_semantic_text_unchanged": True})
                    runtime_metrics["frame_residual_adapter"] = residual_fields
                    if callable(runtime_callback):
                        runtime_callback("frame_residual_adapter", residual_fields)
                if callable(runtime_callback):
                    runtime_callback("talker", talker_metrics)
            elif use_stateful_speaker:
                talker_metrics = (
                    {
                        "diagnostic_rng_seed": int(diagnostic_talker_seed),
                        "diagnostic_rng_scope": "startup_global_not_per_response",
                    }
                    if diagnostic_talker_seed is not None else {}
                )
                with _diagnostic_talker_rng_scope(engine):
                    wav = engine.talk_stateful(
                        text,
                        h_pfx,
                        n_words=max(1, len(text.split())),
                        voice_conditioning=cond,
                        stop_check=lambda: (
                            sess.stop_requested or bool(
                                request.get("_stop_check")
                                and request["_stop_check"]()
                            )
                        ),
                        state=sess,
                        on_audio_chunk=request.get("_on_audio_chunk"),
                        on_audio_chunk_with_horizon=request.get("_on_audio_chunk_with_horizon"),
                        generation_id=generation_id,
                        response_text_conditioning=response_text_conditioning,
                        conditioning_buffer=(
                            None if controller_stateful_pool else sess.conditioning_buffer
                        ),
                        talk_context=request.get("_talk_context"),
                        min_frames=(
                            2 if controller_guided_say and is_backchannel_action else 48
                        ),
                        max_frames=(
                            ARIA_BACKCHANNEL_MAX_FRAMES if controller_guided_say and is_backchannel_action else None
                        ),
                        listener_admission_wait=listener_admission_wait,
                        metrics=talker_metrics,
                    )
                runtime_metrics["talker"] = talker_metrics
                if callable(runtime_callback):
                    runtime_callback("talker", talker_metrics)
            elif stream_requested and streamed_audio:
                wav = np.concatenate(streamed_audio)
            else:
                talker_metrics = (
                    {
                        "diagnostic_rng_seed": int(diagnostic_talker_seed),
                        "diagnostic_rng_scope": "startup_global_not_per_response",
                    }
                    if diagnostic_talker_seed is not None else {}
                )
                quality_diagnostic = None
                if (
                    direct_audio
                    and (response_v2_collector is not None or controller_guided_say)
                    and response_action_value == normalized_action
                    and response_action_value in {
                        ControlAction.SPEAK.value,
                        ControlAction.BACKCHANNEL.value,
                    }
                ):
                    diagnostic_slot = _TTS_QUALITY_DIAGNOSTIC_CAPTURE.claim(
                        getattr(sess, "tts_quality_diagnostic_fixture", None)
                    )
                    if diagnostic_slot is not None:
                        generation_token = request.get("_generation_token")
                        speaker_embedding = (
                            cond.get("speaker_embedding")
                            if isinstance(cond, dict) else None
                        )
                        hidden_shape = list(getattr(response_hidden, "shape", ()))
                        prefix_shape = list(getattr(h_pfx, "shape", ()))
                        quality_diagnostic = {
                            "say_hidden": response_hidden,
                            "max_acoustic_frame_budget": (
                                DIAGNOSTIC_MAX_ACOUSTIC_FRAME_BUDGET
                                if v9_dual_audio_context_enabled else None
                            ),
                            "audio_context_hidden": (
                                audio_context_hidden if v9_dual_audio_context_enabled else None
                            ),
                            "semantic_anchor_hidden": (
                                semantic_anchor_hidden if v9_dual_audio_context_enabled else None
                            ),
                            "metadata": {
                                "response_id": getattr(generation_token, "response_id", None),
                                "session_id": getattr(sess, "session_id", None),
                                "generation_id": generation_id,
                                "session_generation_id": getattr(generation_token, "session_generation_id", None),
                                "thinker_runtime_metrics": {
                                    key: value
                                    for key, value in runtime_metrics.get("thinker", {}).items()
                                    if isinstance(value, (str, bool, int))
                                    or (isinstance(value, float) and np.isfinite(value))
                                },
                                "fixture_id": sess.tts_quality_diagnostic_fixture,
                                "seed_audio_sha256": sess.tts_quality_diagnostic_seed_sha256,
                                "say_text": text,
                                "say_token_spans": runtime_metrics.get(
                                    "audio_response_protocol", {}
                                ).get("say_token_spans", []),
                                "model_input_token_count": len(ids),
                                "thinker_hidden_prompt_len_before_say_extraction": int(
                                    think.get("hidden_prompt_len", think.get("prompt_len", len(ids)))
                                ),
                                "response_prompt_len_for_v9": int(response_prompt_len),
                                "v9_profile": getattr(engine, "v9_profile", None),
                                "v9_recipe": getattr(engine, "v9_recipe", None),
                                "v9_raw_input_token_count": int(hidden_shape[-2]) if len(hidden_shape) >= 2 else 0,
                                "v9_prefix_shape": prefix_shape,
                                "v9_prefix_frames": int(prefix_shape[-2]) if len(prefix_shape) >= 2 else 0,
                                "v9_dual_audio_context": (
                                    dict(dual_v9_metrics) if v9_dual_audio_context_enabled else None
                                ),
                                "wake_action": normalized_action,
                                "response_action": response_action_value,
                                "voice_profile": (
                                    None
                                    if getattr(engine, "ablate_session_voice_conditioning", False)
                                    else sess.voice
                                ),
                                "voice_style": sess.voice_style,
                                "voice_conditioning_ablation": bool(
                                    getattr(engine, "ablate_session_voice_conditioning", False)
                                ),
                                "voice_conditioning_mode": (
                                    cond.get("mode") if isinstance(cond, dict) else None
                                ),
                                "diagnostic_talker_seed": (
                                    int(diagnostic_talker_seed)
                                    if diagnostic_talker_seed is not None else None
                                ),
                                "voice_embedding_dim": (
                                    int(speaker_embedding.shape[-1])
                                    if getattr(speaker_embedding, "shape", None) else None
                                ),
                            },
                            "capture": lambda record, slot=diagnostic_slot: (
                                _TTS_QUALITY_DIAGNOSTIC_CAPTURE.write(slot, record)
                            ),
                        }
                talker_admission_kwargs = (
                    {"listener_admission_wait": listener_admission_wait}
                    if (response_v2_collector is not None or controller_guided_say or frame_residual_candidate)
                    and callable(listener_admission_wait)
                    else {}
                )
                if quality_diagnostic is not None:
                    talker_admission_kwargs["quality_diagnostic"] = quality_diagnostic
                if (controller_guided_say or frame_residual_candidate) and is_backchannel_action:
                    talker_admission_kwargs["min_frames"] = 2
                    talker_admission_kwargs["max_frames"] = ARIA_BACKCHANNEL_MAX_FRAMES
                if frame_residual_candidate:
                    if frame_residual_response_hidden is None:
                        raise RuntimeError("frame_residual_response_hidden_alignment_missing")
                    talker_admission_kwargs["frame_residual_response_hidden"] = frame_residual_response_hidden
                with _diagnostic_talker_rng_scope(engine):
                    wav = engine.talk(text, h_pfx, n_words=max(1, len(text.split())), voice_conditioning=cond,
                                       stop_check=lambda: (
                                           sess.stop_requested or bool(
                                               request.get("_stop_check")
                                               and request["_stop_check"]()
                                           )
                                       ), state=sess,
                                       on_audio_chunk=request.get("_on_audio_chunk"), generation_id=generation_id,
                                       metrics=talker_metrics,
                                       talk_context=request.get("_talk_context"),
                                       **talker_admission_kwargs)
                if frame_residual_candidate:
                    residual_fields = {
                        key: value for key, value in talker_metrics.items()
                        if key.startswith("frame_residual_")
                    }
                    residual_calls = int(talker_metrics.get("frame_residual_calls", 0))
                    residual_fields.update({
                        "status": "applied" if residual_calls > 0 else "no_q1_audio_steps",
                        "qwen_route": "plain_native_audio_q1",
                        "qwen_prefix_slots_added": 0,
                        "qwen_semantic_text_unchanged": True,
                        "source": (
                            "validated_v2_say_posttoken_hidden"
                            if response_v2_collector is not None
                            else "gemma_assistant_response_posttoken_hidden"
                        ),
                    })
                    runtime_metrics["frame_residual_adapter"] = residual_fields
                    if callable(runtime_callback):
                        runtime_callback("frame_residual_adapter", residual_fields)
                    talker_metrics["response_text_conditioning_source"] = (
                        "gemma_assistant_response_hidden_q1_input_residual"
                    )
                if response_v2_collector is not None or controller_guided_say:
                    talker_metrics.update({
                        "response_text_conditioning_source": (
                            f"{response_protocol}.say_hidden_q1_input_residual"
                            if frame_residual_candidate and response_v2_collector is not None else
                            "controller_guided_say_v1.response_hidden_q1_input_residual"
                            if frame_residual_candidate else
                            f"{CONTROLLER_GUIDED_SAY_V1}.generated_say_hidden_v9.stateless_prefill"
                            if controller_guided_say else
                            "audio_response_protocol_v2.say_hidden_v9.stateless_prefill"
                        ),
                        "listener_v9_talker_injection_frames": 0,
                    })
                runtime_metrics["talker"] = talker_metrics
                if callable(runtime_callback):
                    runtime_callback("talker", talker_metrics)
        talker_runtime = runtime_metrics.get("talker")
        talker_cancelled = (
            isinstance(talker_runtime, dict)
            and talker_runtime.get("cancelled") is True
        )
        generation_stale = (
            generation_id is not None
            and not sess.is_generation_current(generation_id=generation_id)
        )
        if (
            talker_cancelled
            or generation_stale
            or (request.get("_stop_check") and request["_stop_check"]())
        ):
            return {
                "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
                "cancelled": True, "runtime_metrics": runtime_metrics,
            }
        if _is_v2_no_speech_result(
            response_protocol=response_protocol,
            modalities=request.get("modalities"),
            wav=wav,
            talker_metrics=talker_runtime,
            cancelled=talker_cancelled,
        ):
            no_speech_metrics = {
                "status": "no_speech",
                "audio_samples": 0,
                "reason": str(
                    talker_runtime.get("no_speech_reason")
                    or (
                        "codec_eos_before_first_frame"
                        if talker_runtime.get("eos_detected") is True
                        and talker_runtime.get("eos_frame_id") == 0
                        else "talker_emitted_no_audio"
                    )
                ),
            }
            runtime_metrics["audio_output"] = no_speech_metrics
            if callable(runtime_callback):
                runtime_callback("audio_output", no_speech_metrics)
            return {
                "text": "",
                "tool_calls": [],
                "wav": None,
                "elapsed": 0.0,
                "status": "no_speech",
                "no_speech": True,
                "history_messages": clean[-16:],
                "runtime_metrics": runtime_metrics,
                "_skip_history_commit": True,
            }
        if request.get("_defer_history"):
            history_messages = clean[-16:]
        else:
            with sess._generation_lock:
                if generation_id is None or sess.is_generation_current(generation_id=generation_id):
                    sess.history = clean[-16:] + [{"role": "assistant", "content": text}]
                    sess.note_semantic_context_changed()
                else:
                    sess.stale_history_writes += 1
        return {"text": text, "wav": wav, "tool_calls": [], "elapsed": 0.0,
                "history_messages": clean[-16:], "runtime_metrics": runtime_metrics,
                "_skip_history_commit": bool(
                    direct_audio
                    and (
                        response_action_value == ControlAction.BACKCHANNEL.value
                        if response_v2_collector is not None
                        else normalized_action == ControlAction.BACKCHANNEL.value
                    )
                )}

    def sse_event(payload: dict[str, Any]) -> str:
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    async def _legacy_talk_from_wav(wav_bytes: bytes) -> np.ndarray:
        if not bool(getattr(engine, "loaded", False)):
            raise RuntimeError("engine non caricato")
        if not bool(getattr(engine, "frame_residual_adapter_enabled", False)) or getattr(
            engine, "frame_residual_adapter", None
        ) is None:
            raise RuntimeError("frame residual query FiLM v2 non caricato")
        if not wav_bytes or len(wav_bytes) > 24_000_000:
            raise ValueError("WAV vuoto o superiore al limite di 24 MB")
        try:
            import soundfile as sf
            decoded, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
        except Exception as exc:
            raise ValueError("upload non è un WAV leggibile") from exc
        if sample_rate < 8000 or sample_rate > 48000:
            raise ValueError("sample rate WAV deve essere compreso tra 8 e 48 kHz")
        pcm = np.asarray(decoded, dtype=np.float32).mean(axis=1)
        if pcm.size == 0 or pcm.size > int(sample_rate * 30):
            raise ValueError("il WAV deve durare tra 80 ms e 30 secondi")
        if not np.isfinite(pcm).all():
            raise ValueError("il WAV contiene valori non finiti")
        if int(sample_rate) != USER_SAMPLE_RATE:
            divisor = gcd(int(sample_rate), USER_SAMPLE_RATE)
            from scipy.signal import resample_poly

            pcm = resample_poly(
                pcm,
                USER_SAMPLE_RATE // divisor,
                int(sample_rate) // divisor,
            ).astype(np.float32, copy=False)
        if pcm.size < int(round(0.080 * USER_SAMPLE_RATE)):
            raise ValueError("il WAV deve durare almeno 80 ms")

        request = {
            "voice": DEFAULT_VOICE,
            "language": "Italian",
            "turn_policy": "none",
            "modalities": ["text", "audio"],
            "_direct_audio_pcm": np.asarray(pcm, dtype=np.float32),
        }
        sess, ephemeral = get_session(request)
        try:
            sess.voice = DEFAULT_VOICE
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _run_legacy_audio_response(sess, request),
            )
            wav = result.get("wav") if isinstance(result, dict) else None
            if wav is None or np.asarray(wav).size == 0:
                raise RuntimeError("engine.talk non ha prodotto audio")
            return np.asarray(wav, dtype=np.float32).reshape(-1)
        finally:
            if ephemeral:
                sessions.close(sess.session_id)

    def _run_legacy_audio_response(sess: DuplexSession, request: dict[str, Any]) -> dict[str, Any]:
        with _inference_execution_scope(False):
            return run_response(
                sess,
                [{"role": "user", "content": ""}],
                request,
                action=ControlAction.SPEAK.value,
            )

    @app.post("/talk", response_class=Response)
    async def talk_legacy(file: UploadFile = File(...)) -> Response:
        """Accept a user WAV and return the conditioned 24 kHz agent WAV."""
        raw = await file.read()
        try:
            wav = await _legacy_talk_from_wav(raw)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        try:
            import soundfile as sf

            output = io.BytesIO()
            sf.write(output, wav, CODEC_SAMPLE_RATE, format="WAV")
        except Exception:
            logger.exception("legacy /talk WAV encoding failed")
            return JSONResponse({"error": "impossibile codificare WAV di risposta"}, status_code=500)
        return Response(
            content=output.getvalue(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=agent_response.wav"},
        )

    @app.post("/talk_json")
    async def talk_json_legacy(payload: dict[str, Any]) -> Any:
        """Accept ``wav_b64`` and return a 24 kHz WAV encoded as base64."""
        encoded = payload.get("wav_b64")
        if not isinstance(encoded, str) or not encoded:
            return JSONResponse({"error": "wav_b64 deve essere una stringa base64"}, status_code=422)
        try:
            raw = base64.b64decode(encoded, validate=True)
            wav = await _legacy_talk_from_wav(raw)
            return {"wav_b64": _wav_b64(wav), "sr": CODEC_SAMPLE_RATE}
        except (ValueError, base64.binascii.Error) as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        except Exception:
            logger.exception("legacy /talk_json inference failed")
            return JSONResponse({"error": "inferenza legacy fallita"}, status_code=500)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        active_sessions = sessions.all()
        shared_listener_backend = shared_listener_worker_holder.get("backend")
        shared_listener_runtime_info = getattr(
            shared_listener_backend, "_listener_microbatch_runtime_info", {}
        )
        if not isinstance(shared_listener_runtime_info, dict):
            shared_listener_runtime_info = {}
        shared_listener_worker = shared_listener_worker_holder.get("worker")
        shadow_status_provider = getattr(
            shared_listener_worker, "semantic_shadow_status", None
        )
        semantic_shadow_status = (
            dict(shadow_status_provider)
            if isinstance(shadow_status_provider, dict)
            else {}
        )
        audio_forward_event = getattr(engine, "_thinker_audio_forward_active", None)
        talk_context_pool = getattr(engine, "response_talk_context_pool", None)
        semantic_prefix_metadata = getattr(engine, "native_control6_semantic_prefix_metadata", None)
        semantic_prefix_metadata_valid = native_control6_semantic_prefix_metadata_is_supported(
            semantic_prefix_metadata
        )
        semantic_context_sessions: list[dict[str, Any]] = []
        for session in active_sessions:
            state = getattr(session, "listener_state", None)
            if state is None:
                continue
            committed_revision = int(getattr(session, "semantic_context_revision", 0))
            applied_revision = getattr(state, "semantic_context_revision", None)
            semantic_context_sessions.append({
                "actual_prefix_len": int(getattr(state, "gemma_lm_prefix_len", 0)),
                "prefix_capacity_tokens": (
                    NATIVE_CONTROL6_PREFIX_MAX_TOKENS if semantic_prefix_metadata_valid else None
                ),
                "committed_history_revision": committed_revision,
                "applied_history_revision": (
                    int(applied_revision) if applied_revision is not None else None
                ),
                "context_current": (
                    applied_revision is not None and int(applied_revision) == committed_revision
                ),
                "native_prime_applied": applied_revision is not None,
                "source_frame_id": getattr(state, "gemma_lm_last_source_frame_id", None),
            })
        semantic_prefix_enabled = bool(
            getattr(engine, "enable_gemma_native_control6_policy", False)
            and _native_semantic_prefix_enabled()
            and semantic_prefix_metadata_valid
        )
        frame_residual_adapter = getattr(engine, "frame_residual_adapter", None)
        listener_compute_lock = getattr(engine, "_thinker_lock", None)
        listener_pair_lock_handoff_enabled = bool(
            getattr(listener_compute_lock, "listener_pair_lock_handoff_enabled", False)
        )
        listener_host_timeline_enabled = bool(
            getattr(listener_compute_lock, "host_timeline_enabled", False)
        )
        learned_scale_value = getattr(frame_residual_adapter, "scale", None)
        try:
            learned_scale = float(
                learned_scale_value.detach().float().cpu().item()
                if isinstance(learned_scale_value, torch.Tensor)
                else learned_scale_value
            )
        except (TypeError, ValueError, RuntimeError):
            learned_scale = None
        return {
            "status": "ok",
            "engine_loaded": bool(getattr(engine, "loaded", False)),
            "unified_kv_enabled": unified_kv_enabled,
            "first_input_scheduler_anchor_enabled": FIRST_INPUT_SCHEDULER_ANCHOR_ENABLED,
            "ready_frame_scheduler_dispatch_enabled": READY_FRAME_SCHEDULER_DISPATCH_ENABLED,
            "listener_pair_lock_handoff_setting": os.environ.get(
                LISTENER_PAIR_LOCK_HANDOFF_ENV
            ),
            "listener_pair_lock_handoff_enabled": listener_pair_lock_handoff_enabled,
            "native_listener_cuda_profile_enabled": NATIVE_LISTENER_CUDA_PROFILE_ENABLED,
            "native_control6_fused_readback_setting": os.environ.get(
                NATIVE_CONTROL6_FUSED_READBACK_ENV, "0"
            ),
            "native_control6_fused_readback_enabled": NATIVE_CONTROL6_FUSED_READBACK_ENABLED,
            "listener_model_lock_host_timeline_enabled": listener_host_timeline_enabled,
            "native_listener_cuda_profile_directory": native_listener_cuda_profile_directory,
            "listener_microbatch": {
                "enabled": shared_listener_microbatch_enabled,
                "runtime_ready": bool(shared_listener_runtime_info.get("enabled", False)),
                "gemm_graphs_setting": os.environ.get(
                    "DUPLEX_LISTENER_MICROBATCH_GEMM_GRAPHS", "0"
                ),
                "gemm_graphs_enabled": bool(
                    shared_listener_runtime_info.get("gemm_graphs_enabled", False)
                ),
                "gemm_graph_count": int(
                    shared_listener_runtime_info.get("gemm_graph_count", 0)
                ),
                "allocated_capacity": shared_listener_runtime_info.get(
                    "allocated_capacity"
                ),
                "semantic_shadow_setting": os.environ.get(
                    "DUPLEX_LISTENER_MICROBATCH_SEMANTIC_SHADOW", "0"
                ),
                "semantic_shadow_enabled": bool(
                    shared_listener_runtime_info.get("semantic_shadow_enabled", False)
                ),
                "semantic_shadow_ready": bool(
                    shared_listener_runtime_info.get("semantic_shadow_ready", False)
                ),
                "semantic_shadow_capacity": shared_listener_runtime_info.get(
                    "semantic_shadow_capacity"
                ),
                "semantic_shadow_gemm_graphs_enabled": bool(
                    shared_listener_runtime_info.get(
                        "semantic_shadow_gemm_graphs_enabled", False
                    )
                ),
                "semantic_shadow_gemm_graph_count": int(
                    shared_listener_runtime_info.get(
                        "semantic_shadow_gemm_graph_count", 0
                    )
                ),
                "semantic_shadow_status": semantic_shadow_status,
            },
            "first_input_scheduler_anchor_sessions": [
                {
                    "session_id": session.session_id,
                    "enabled": bool(session.scheduler.anchor_first_input),
                    "applied": session.scheduler.first_input_anchor_applied,
                    "epoch_ns": session.scheduler.first_input_anchor_ns,
                }
                for session in active_sessions
            ],
            "ready_frame_scheduler_dispatch_sessions": [
                {
                    "session_id": session.session_id,
                    "enabled": bool(session.scheduler.ready_frame_dispatch),
                }
                for session in active_sessions
            ],
            "frame_residual_adapter": {
                "enabled": bool(getattr(engine, "frame_residual_adapter_enabled", False)),
                "architecture": getattr(
                    engine,
                    "frame_residual_adapter_architecture",
                    getattr(frame_residual_adapter, "architecture_id", None),
                ),
                "mode": getattr(frame_residual_adapter, "architecture_id", None),
                "checkpoint": (
                    getattr(engine, "frame_residual_adapter_resolved_path", None)
                    or getattr(engine, "frame_residual_adapter_checkpoint_path", None)
                ),
                "learned_scale": learned_scale,
                "default_voice": DEFAULT_VOICE,
            },
            "model": MODEL_ID,
            "response_protocol": _configured_audio_response_protocol(engine),
            "direct_audio_candidate": os.environ.get("DUPLEX_DIRECT_AUDIO_CANDIDATE", "0").strip() == "1",
            "audio_history_archive_defer_offline": (
                os.environ.get(_AUDIO_HISTORY_ARCHIVE_DEFER_ENV, "0").strip() == "1"
            ),
            "audio_history_archive_cpu_startup_available": (
                os.environ.get("DUPLEX_AUDIO_HISTORY_ARCHIVE_CPU", "0").strip() == "1"
                and bool(getattr(CPU_AUDIO_ARCHIVE_WORKER, "available", False))
            ),
            "native_action_selection": _native_action_selection_mode(),
            "native_control6_semantic_prefix": {
                "enabled": semantic_prefix_enabled,
                "checkpoint_metadata_valid": semantic_prefix_metadata_valid,
                "schema_version": (
                    semantic_prefix_metadata.get("semantic_prefix_schema_version")
                    if semantic_prefix_metadata_valid else None
                ),
                "contract": (
                    semantic_prefix_metadata.get("semantic_prefix_contract")
                    if semantic_prefix_metadata_valid else None
                ),
                "boa_token_id": (
                    semantic_prefix_metadata.get("semantic_prefix_boa_token_id")
                    if semantic_prefix_metadata_valid else None
                ),
                "prefix_capacity_tokens": (
                    semantic_prefix_metadata.get("semantic_prefix_capacity")
                    if semantic_prefix_metadata_valid else None
                ),
                "active_session_contexts": semantic_context_sessions,
            },
            "direct_v2_audio_warmup_status": getattr(engine, "direct_v2_audio_warmup_status", None),
            "frozen_audio_modules": {
                "talker": _module_inference_freeze_status(
                    getattr(getattr(engine, "talker", None), "talker", None)
                ),
                "subtalker": _module_inference_freeze_status(
                    getattr(
                        getattr(getattr(engine, "talker", None), "talker", None),
                        "code_predictor",
                        None,
                    )
                ),
                "code2wav": _module_inference_freeze_status(
                    getattr(getattr(engine, "talker", None), "code2wav", None)
                ),
                "v9": _module_inference_freeze_status(getattr(engine, "v9", None)),
                "causal_conv": _module_inference_freeze_status(getattr(engine, "causal_conv", None)),
            },
            "talker_q1_backend": getattr(engine, "talker_q1_backend", "eager"),
            "talker_q1_compile_status": getattr(engine, "talker_q1_compile_status", "disabled"),
            "talker_q1_native_status": getattr(engine, "talker_q1_native_status", "disabled"),
            "controller_guided_talk_context_pool": {
                "enabled": talk_context_pool is not None,
                "capacity": int(getattr(talk_context_pool, "capacity", 0)),
                "active": int(getattr(talk_context_pool, "active_count", 0)),
                "memory": getattr(engine, "response_talk_context_memory", None),
            },
            "active_realtime_sessions": len(active_sessions),
            "listener_backlog_frames": max(
                (session.audio_ingress.qsize for session in active_sessions), default=0
            ),
            "listener_callbacks_active": any(
                session.scheduler.callbacks_active for session in active_sessions
            ),
            "active_responses": sum(bool(session.active_response) for session in active_sessions),
            "thinker_audio_forward_active": bool(
                audio_forward_event.is_set()
                if callable(getattr(audio_forward_event, "is_set", None))
                else False
            ),
            "thinker_audio_prefill_lock_hold_max_ms": getattr(
                engine, "_thinker_audio_prefill_lock_hold_max_ms", None
            ),
        }

    @app.get("/", include_in_schema=False)
    async def realtime_ui():
        return FileResponse(Path(__file__).resolve().parent.parent / "server" / "static" / "index.html")

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "duplex_kernel"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict[str, Any]):
        if not getattr(engine, "loaded", False):
            return JSONResponse({"error": {"code": "ENGINE_NOT_READY", "message": "engine non caricato"}}, status_code=503)
        try:
            request["modalities"] = _normalize_chat_modalities(request.get("modalities"))
        except ValueError:
            return JSONResponse(
                {"error": {"code": "VALIDATION_ERROR", "message": "modalities deve contenere text e/o audio"}},
                status_code=422,
            )
        if "audio" in request["modalities"] and hasattr(engine, "frame_residual_adapter_enabled") and (
            not bool(getattr(engine, "frame_residual_adapter_enabled", False))
            or getattr(engine, "frame_residual_adapter", None) is None
        ):
            return JSONResponse(
                {"error": {"code": "APPROVED_CONDITIONING_NOT_READY", "message": "frame residual query FiLM v2 non caricato"}},
                status_code=503,
            )
        messages = request.get("messages")
        try:
            max_tokens = int(request.get("max_tokens", 64)); temperature = float(request.get("temperature", 0.7)); top_p = float(request.get("top_p", 0.9))
            if max_tokens < 1 or not 0.0 < temperature or not 0.0 < top_p <= 1.0:
                raise ValueError
        except (TypeError, ValueError):
            return JSONResponse({"error": {"code": "VALIDATION_ERROR", "message": "max_tokens/temperature/top_p non validi"}}, status_code=422)
        if not isinstance(messages, list) or not messages:
            return JSONResponse({"error": {"code": "VALIDATION_ERROR", "message": "messages deve essere una lista non vuota"}}, status_code=422)
        try: sess, ephemeral = get_session(request)
        except (ValueError, RuntimeError) as exc:
            return JSONResponse({"error": {"code": "SESSION_ERROR", "message": str(exc)}}, status_code=422)
        sess.tools = request.get("tools") or sess.tools
        sess.voice = str(request.get("voice") or request.get("speaker") or sess.voice or DEFAULT_VOICE)
        sess.voice_style = request.get("voice_style") or sess.voice_style
        sess.ref_text = request.get("ref_text") or sess.ref_text
        loop = asyncio.get_running_loop()
        with sess._generation_lock:
            context_messages = (list(sess.history) + messages) if request.get("session_id") else messages
        def infer():
            with _inference_execution_scope(False):
                return run_response(sess, context_messages, request)
        try: result = await loop.run_in_executor(None, infer)
        except Exception:
            if ephemeral: sessions.close(sess.session_id)
            logger.exception("richiesta fallita")
            return JSONResponse({"error": {"code": "INFERENCE_ERROR", "message": "inferenza fallita"}}, status_code=500)
        if ephemeral: result["session_id"] = sess.session_id
        if result["tool_calls"]:
            msg = {"role": "assistant", "content": None, "tool_calls": result["tool_calls"]}
            body = {"id": "chatcmpl_" + uuid.uuid4().hex, "object": "chat.completion", "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}], "session_id": sess.session_id}
            return JSONResponse(body)
        if not request.get("stream"):
            msg = {"role": "assistant", "content": result["text"]}
            if result["wav"] is not None: msg["audio"] = {"data": _wav_b64(result["wav"]), "expires_at": int(time.time()) + 86400}
            if result.get("status") == "no_speech": msg["audio_status"] = "no_speech"
            body = {"id": "chatcmpl_" + uuid.uuid4().hex, "object": "chat.completion", "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]}
            if result.get("status") == "no_speech": body["audio_status"] = "no_speech"
            if ephemeral: body["session_id"] = sess.session_id
            return JSONResponse(body)

        def generate():
            if result["tool_calls"]:
                for index, call in enumerate(result["tool_calls"]):
                    yield sse_event({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, **call}]}}]})
                yield sse_event({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
                yield "data: [DONE]\n\n"; return
            words = result["text"].split(" ")
            for i, word in enumerate(words):
                yield sse_event({"choices": [{"index": 0, "delta": {"content": word + (" " if i + 1 < len(words) else "")}}]})
            wav = result["wav"]
            if wav is not None:
                for packet in iter_audio_packets(
                    wav,
                    generation_id=0,
                    speaker_segment_id="http-sse",
                    packet_frames=NETWORK_PACKET_FRAMES,
                    sample_rate=CODEC_SAMPLE_RATE,
                ):
                    yield sse_event({"choices": [{"index": 0, "delta": {"audio": {
                        "chunk": _wav_b64(packet.pcm),
                        "packet_id": packet.packet_id,
                        "generation_id": packet.generation_id,
                        "speaker_segment_id": packet.speaker_segment_id,
                        "first_frame_id": packet.first_frame_id,
                        "last_frame_id": packet.last_frame_id,
                        "start_sample": packet.start_sample,
                        "end_sample": packet.end_sample,
                        "sample_rate": CODEC_SAMPLE_RATE,
                    }}}]})
            yield sse_event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket, duplex: int = 0):
        if not duplex:
            await ws.close(code=1008, reason="usa /v1/realtime?duplex=1"); return
        await ws.accept()
        direct_audio_candidate = os.environ.get("DUPLEX_DIRECT_AUDIO_CANDIDATE", "0").strip() == "1"
        response_protocol = _configured_audio_response_protocol(engine) if direct_audio_candidate else None
        ingress_capacity_frames = (
            DIRECT_V2_INGRESS_QUEUE_FRAMES
            if direct_audio_candidate
            and response_protocol in {AUDIO_RESPONSE_PROTOCOL_V2, CONTROLLER_GUIDED_SAY_V1}
            else REALTIME_INGRESS_QUEUE_FRAMES
        )
        sess = sessions.create_session(
            voice=DEFAULT_VOICE,
            audio_ingress_capacity_frames=ingress_capacity_frames,
        )
        native_control6_debug_rows: list[dict[str, Any]] = []
        native_control6_debug_frame_ids: set[int] = set()
        try:
            sess.voice_conditioning = _resolve_p7_session_voice(engine, voice=DEFAULT_VOICE)
        except BaseException:
            sessions.close(sess.session_id)
            raise
        session_runtime_metrics: dict[str, Any] = {
            "listener": {"canonical_prime_calls": 0},
            "scheduler": {
                "first_input_anchor_enabled": bool(sess.scheduler.anchor_first_input),
                "first_input_anchor_applied": False,
                "ready_frame_dispatch_enabled": bool(sess.scheduler.ready_frame_dispatch),
            },
        }
        listener_timing_lock = threading.Lock()
        scheduler_stage_samples: dict[str, deque[float]] = {}
        scheduler_stage_totals: dict[str, dict[str, float | int]] = {}

        def record_scheduler_stage(name: str, elapsed_ms: float | None) -> None:
            if elapsed_ms is None:
                return
            value = float(elapsed_ms)
            if not np.isfinite(value) or value < 0:
                return
            with listener_timing_lock:
                scheduler_stage_samples.setdefault(name, deque(maxlen=200)).append(value)
                totals = scheduler_stage_totals.setdefault(
                    name, {"count": 0, "total_ms": 0.0, "max_ms": 0.0}
                )
                totals["count"] = int(totals["count"]) + 1
                totals["total_ms"] = float(totals["total_ms"]) + value
                totals["max_ms"] = max(float(totals["max_ms"]), value)

        def scheduler_stage_summary() -> dict[str, dict[str, Any]]:
            with listener_timing_lock:
                summary: dict[str, dict[str, Any]] = {}
                for name, totals in scheduler_stage_totals.items():
                    ordered = sorted(scheduler_stage_samples.get(name, ()))
                    if not ordered:
                        continue
                    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
                    summary[name] = {
                        "count": int(totals["count"]),
                        "total_ms": round(float(totals["total_ms"]), 3),
                        "max_ms": round(float(totals["max_ms"]), 3),
                        "p95_ms": round(ordered[p95_index], 3),
                        "reservoir_size": len(ordered),
                        "reservoir_capacity": 200,
                        "measurement": "host_monotonic_wall_time",
                    }
                return summary
        try:
            _ensure_policy(
                sess,
                prefer_bundle_control6=(
                    direct_audio_candidate
                    and response_protocol in {AUDIO_RESPONSE_PROTOCOL_V2, CONTROLLER_GUIDED_SAY_V1}
                ),
            )
            _ensure_listener(sess, runtime_metrics=session_runtime_metrics)
        except BaseException:
            sessions.close(sess.session_id)
            raise
        response_task: asyncio.Task | None = None
        playback_ack_branches: dict[str, dict[str, Any]] = {}
        playback_ack_branches_lock = threading.RLock()
        response_binding: dict[str, Any] | None = None
        # Archival ASR is process-shared CPU work; no per-session model copy or
        # inference_lock use. The session owns two bounded FIFO PCM slots.
        audio_archive_lock = threading.RLock()
        audio_archive_futures: dict[int, Any] = {}
        # One transport sequence spans every response/segment on this WS.
        packet_sequence = PacketSequenceNumber()
        closing = False
        native_listener_output_faulted = threading.Event()
        loop = asyncio.get_running_loop()

        def new_playback_ack_state() -> dict[str, Any]:
            ack_event = asyncio.Event()
            return {
                "playback_ack_event": ack_event,
                "playback_ack_wakeup": lambda: loop.call_soon_threadsafe(ack_event.set),
                "playback_cancelled": False,
                "input_audio_watermark_frame_id": None,
            }

        def listener_admission_wait(generation_token: GenerationToken | None = None) -> None:
            """Yield to queued listener work with bounded direct-audio prefill waits."""
            if not (
                direct_audio_candidate
                and response_protocol in {AUDIO_RESPONSE_PROTOCOL_V2, CONTROLLER_GUIDED_SAY_V1}
            ):
                return
            def generation_is_current() -> bool:
                if closing or sess.stop_requested:
                    return False
                return generation_token is None or sess.is_generation_current(generation_token)

            admission = _wait_for_listener_backlog_admission(
                lambda: sess.audio_ingress.qsize,
                generation_is_current,
                callbacks_active=sess.scheduler.callbacks_active,
            )
            return admission

        speech_tracker = _CausalUserSpeechTracker(
            endpoint_quiet_frames=_configured_speak_endpoint_frames()
        )
        peer_audio_event_key: tuple[str, int] | None = None
        backchannel_attempt_latch = BackchannelAttemptLatch()
        scheduler_events = _SchedulerEventBuffer(
            loop,
            maxsize=REALTIME_SCHEDULER_EVENT_CAPACITY,
        )
        native_partial_asr_available = (
            os.environ.get(NATIVE_PARTIAL_ASR_ENV, "0").strip() == "1"
            and callable(getattr(engine, "transcribe_audio", None))
        )
        archive_cpu_startup_available = (
            os.environ.get("DUPLEX_AUDIO_HISTORY_ARCHIVE_CPU", "0").strip() == "1"
            and bool(getattr(CPU_AUDIO_ARCHIVE_WORKER, "available", False))
        )
        audio_history_archive_defer_offline = (
            os.environ.get(_AUDIO_HISTORY_ARCHIVE_DEFER_ENV, "0").strip() == "1"
        )
        audio_history_archive_cpu_enabled = False
        native_partial_asr_enabled = threading.Event()
        transcript_state_lock = threading.RLock()
        transcript_state: dict[str, Any] = {
            "epoch": 0,
            "utterance_id": None,
            "revision": 0,
            "completed": False,
        }
        partial_last_snapshot_frame: int | None = None

        def transcript_snapshot_is_current(snapshot: NativeTranscriptSnapshot) -> bool:
            with transcript_state_lock:
                state_matches = (
                    transcript_state["utterance_id"] == snapshot.utterance_id
                    and int(transcript_state["epoch"]) == int(snapshot.epoch_id)
                    and native_partial_asr_enabled.is_set()
                )
                if not state_matches:
                    return False
                # Assistant output pauses native ASR work; it does not make
                # the user's still-current utterance stale. Transcript epoch
                # and opt-out are the invalidation fences.
                return not closing

        def publish_input_transcript(snapshot: NativeTranscriptSnapshot, transcript: str) -> None:
            event_type = (
                "input_audio_transcription.completed"
                if snapshot.final else "input_audio_transcription.partial"
            )
            with transcript_state_lock:
                if (
                    not native_partial_asr_enabled.is_set()
                    or transcript_state["utterance_id"] != snapshot.utterance_id
                    or int(transcript_state["epoch"]) != int(snapshot.epoch_id)
                ):
                    return
                if closing:
                    return
                if snapshot.final and transcript_state["completed"]:
                    return
                transcript_state["revision"] = int(transcript_state["revision"]) + 1
                revision = int(transcript_state["revision"])
                if snapshot.final:
                    transcript_state["completed"] = True
                scheduler_events.put_nowait({
                    "type": event_type,
                    "utterance_id": snapshot.utterance_id,
                    "revision": revision,
                    "transcript": str(transcript),
                    "first_frame_id": int(snapshot.first_frame_id),
                    "last_frame_id": int(snapshot.last_frame_id),
                    "_transcript_epoch": int(snapshot.epoch_id),
                    "_transcript_utterance_id": snapshot.utterance_id,
                    "_transcript_allow_response_active": True,
                    "_transcript_source": "native_asr_worker",
                })

        def publish_completed_input_transcript(
            transcript: str,
            *,
            utterance_id: str,
            first_frame_id: int,
            last_frame_id: int,
            generation_id: int,
            epoch_id: int,
            source: str = "heard_action_say_v2",
        ) -> None:
            text = str(transcript).strip()
            if not utterance_id or not native_partial_asr_enabled.is_set():
                return
            with transcript_state_lock:
                if (
                    not native_partial_asr_enabled.is_set()
                    or transcript_state["utterance_id"] != utterance_id
                    or int(transcript_state["epoch"]) != int(epoch_id)
                    or transcript_state["completed"]
                ):
                    return
                with sess._generation_lock:
                    if generation_id != sess.session_generation_id or closing:
                        return
                transcript_state["revision"] = int(transcript_state["revision"]) + 1
                revision = int(transcript_state["revision"])
                transcript_state["completed"] = True
                scheduler_events.put_nowait({
                    "type": "input_audio_transcription.completed",
                    "utterance_id": utterance_id,
                    "revision": revision,
                    "transcript": text,
                    "first_frame_id": int(first_frame_id),
                    "last_frame_id": int(last_frame_id),
                    "_generation_fence": int(generation_id),
                    "_transcript_epoch": int(epoch_id),
                    "_transcript_utterance_id": utterance_id,
                    "_transcript_allow_response_active": True,
                    "_transcript_source": str(source),
                })

        def transcribe_live_snapshot(pcm: np.ndarray, *, stop_check: Callable[[], bool]) -> Any:
            transcribe = engine.transcribe_audio
            # Older injected test engines may still expose the original one-arg
            # callable. Production PipelineEngine accepts the cooperative stop
            # check added for decoder-token cancellation.
            try:
                import inspect
                parameters = inspect.signature(transcribe).parameters.values()
                supports_stop = any(
                    parameter.name == "stop_check"
                    or parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
            except (TypeError, ValueError):
                supports_stop = True
            if supports_stop:
                return transcribe(pcm, stop_check=stop_check)
            return transcribe(pcm)

        native_partial_worker = (
            NativePartialTranscriber(
                transcribe=transcribe_live_snapshot,
                publish=publish_input_transcript,
                is_current=transcript_snapshot_is_current,
                max_samples=USER_SAMPLE_RATE * 30,
            )
            if native_partial_asr_available else None
        )

        def advance_transcript_epoch(*, reset_utterance: bool = False) -> int:
            # Mutate the epoch before waiting for the worker lock. The sender
            # uses the same lock across its final eligibility check and send.
            with transcript_state_lock:
                epoch = int(transcript_state["epoch"]) + 1
                transcript_state["epoch"] = epoch
                if reset_utterance:
                    transcript_state.update({
                        "utterance_id": None,
                        "revision": 0,
                        "completed": False,
                    })
            if native_partial_worker is not None:
                native_partial_worker.cancel()
            return epoch

        def set_native_partial_asr_enabled(enabled: bool) -> None:
            with transcript_state_lock:
                if enabled:
                    native_partial_asr_enabled.set()
                else:
                    native_partial_asr_enabled.clear()
                    transcript_state["epoch"] = int(transcript_state["epoch"]) + 1
                    transcript_state.update({
                        "utterance_id": None,
                        "revision": 0,
                        "completed": False,
                    })
            if not enabled and native_partial_worker is not None:
                native_partial_worker.cancel()
            elif enabled and native_partial_worker is not None:
                with sess._generation_lock:
                    response_busy = bool(
                        sess.active_response
                        or sess.policy_response_pending
                        or sess.control_state in {
                            ControlState.SPEAKING,
                            ControlState.BACKCHANNELING,
                            ControlState.STOPPING,
                        }
                    )
                if response_busy:
                    native_partial_worker.pause()
                else:
                    native_partial_worker.resume()

        def begin_transcript_utterance(frame_id: int) -> tuple[str, int]:
            with transcript_state_lock:
                epoch = int(transcript_state["epoch"]) + 1
                utterance_id = f"utt_{int(frame_id)}_{uuid.uuid4().hex[:8]}"
                transcript_state.update({
                    "epoch": epoch,
                    "utterance_id": utterance_id,
                    "revision": 0,
                    "completed": False,
                })
            if native_partial_worker is not None:
                native_partial_worker.cancel()
            return utterance_id, epoch
        ws_send_lock = asyncio.Lock()
        policy_cooldown_lock = threading.Lock()
        latest_policy_frame_id = -1
        empty_asr_cooldown_until_frame = -1
        policy_causality_lock = threading.Lock()
        response_audio_admission = _ResponseAudioAdmissionState()
        pending_speak: dict[str, Any] | None = None
        last_listener_phase_key: str | None = None
        pending_response_speak_handoffs: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        replan_edge_latched = False
        replan_non_event_ticks = 0
        replan_edge_generation_id: int | None = None
        replan_handoff_pending = False
        replan_endpoint_waiters: dict[int, tuple[int, threading.Event]] = {}
        replan_audio_snapshots: dict[int, np.ndarray] = {}
        replan_audio_contexts: dict[int, dict[str, Any]] = {}
        replan_endpoint_lock = threading.Lock()
        response_controls: dict[str, _RealtimeResponseControl] = {}
        response_controls_lock = threading.RLock()
        response_runtime_metrics: dict[str, dict[str, Any]] = {}
        response_runtime_metrics_lock = threading.Lock()
        replan_cancel_waiters: dict[int, threading.Event] = {}
        replan_cancel_waiters_lock = threading.Lock()
        input_ack_enabled = threading.Event()
        input_conditioning_ack_enabled = threading.Event()

        def update_response_runtime_metrics(response_id: str, phase: str, values: Any) -> None:
            if not isinstance(values, dict):
                return
            cache_numeric_fields = {
                "thinker_cache_hit",
                "thinker_cache_reused_tokens",
                "thinker_cache_copy_tokens",
                "thinker_cache_copy_bytes",
                "thinker_cache_snapshot_storage_bytes",
                "thinker_cache_branch_storage_bytes",
                "thinker_audio_prefill",
            }
            cache_string_fields = {
                "thinker_cache_reason",
                "thinker_cache_publish",
                "thinker_cache_snapshot_reason",
                "thinker_cache_storage_reason",
                "thinker_cache_init_method",
            }
            protocol_fields = {
                "response_protocol",
                "first_thinker_token_id",
                "wake_action",
                "response_action",
                "selected_action",
                "allowed_response_actions",
                "say_chars",
                "output_token_count",
                "say_hidden_tokens",
                "say_token_spans",
                "action_constraint",
                "action_forced",
                "native_action_logprob",
                "forced_action_token_count",
                "forced_action_override_count",
                "forced_structure_token_count",
                "forced_structure_override_count",
                "structure_forced",
                "bounded_output",
                "predecode_candidate_checks",
                "predecode_candidate_replacements",
                "predecode_bounded_closures",
                "talker_q1_backend",
                "compiled_talker_q1_calls",
                "native_talker_q1_calls",
                "virtual_prefix_mode",
                "prefix_key_log_gate",
                "code2wav_backend",
                "code2wav_graph_captured",
                "code2wav_graph_window_calls",
                "code2wav_eager_calls",
                "subtalker_graph_backend",
                "subtalker_graph_captured",
                "subtalker_graph_calls",
                "terminated_by",
                "backchannel_done_reason",
                "backchannel_peer_event_id",
                "assistant_text_committed",
                "talker_started",
                "backchannel_say_lexicon_pool_overrides",
                "backchannel_say_lexicon_constrained",
                "backchannel_say_lexicon_phrase_count",
                "backchannel_say_lexicon_rejections",
                "live_say_consumer_started_at_ns",
                "live_say_thinker_returned_at_ns",
                "live_say_consumer_started_before_thinker_return",
                "failure_reason",
                "status",
                "queue_kind",
                "cancel_reason",
            }
            measured = {
                key: value for key, value in values.items()
                if isinstance(key, str) and (
                    key.endswith(("_ms", "_steps", "_chunks", "_frames", "_samples", "_monotonic_ns"))
                    or key in {
                        "decode_steps", "generated_cb0_frames", "generated_pcm_samples",
                        "terminal_token_id", "terminal_reason",
                    }
                    or key in cache_numeric_fields
                    or key in cache_string_fields
                    or key in protocol_fields
                    or key in {"burst_items", "attempts"}
                ) and (
                    isinstance(value, (int, float, bool))
                    or (key in {"terminal_reason", "terminated_by", "backchannel_done_reason", "backchannel_peer_event_id"} and isinstance(value, str))
                    or (key in cache_string_fields and isinstance(value, str))
                    or (key in {"queue_kind", "cancel_reason"} and isinstance(value, str))
                    or (key in {"response_protocol", "wake_action", "response_action", "selected_action", "failure_reason", "status", "coverage"} and isinstance(value, str))
                    or (
                        key == "allowed_response_actions"
                        and isinstance(value, list)
                        and all(item in AUDIO_RESPONSE_ACTIONS for item in value)
                    )
                    or (key in {"talker_q1_backend", "code2wav_backend", "subtalker_graph_backend", "virtual_prefix_mode"} and isinstance(value, str))
                    or (key in {"say_chars", "output_token_count", "say_hidden_tokens"} and isinstance(value, int) and not isinstance(value, bool))
                    or (key == "compiled_talker_q1_calls" and isinstance(value, int) and not isinstance(value, bool))
                    or (key == "native_talker_q1_calls" and isinstance(value, int) and not isinstance(value, bool))
                    or (key in {"forced_action_token_count", "forced_action_override_count"} and isinstance(value, int) and not isinstance(value, bool))
                    or (key in {"forced_structure_token_count", "forced_structure_override_count"} and isinstance(value, int) and not isinstance(value, bool))
                    or (key == "backchannel_say_lexicon_constrained" and isinstance(value, bool))
                    or (key in {"backchannel_say_lexicon_phrase_count", "backchannel_say_lexicon_rejections", "backchannel_say_lexicon_pool_overrides"} and isinstance(value, int) and not isinstance(value, bool))
                    or (key == "action_forced" and isinstance(value, bool))
                    or (key == "structure_forced" and isinstance(value, bool))
                    or (key == "bounded_output" and isinstance(value, bool))
                    or (key in {"assistant_text_committed", "talker_started"} and isinstance(value, bool))
                    or (key == "live_say_consumer_started_before_thinker_return" and isinstance(value, bool))
                    or (
                        key in {
                            "predecode_candidate_checks",
                            "predecode_candidate_replacements",
                            "predecode_bounded_closures",
                        }
                        and isinstance(value, int) and not isinstance(value, bool)
                    )
                    or (key == "native_action_logprob" and isinstance(value, (int, float)) and not isinstance(value, bool))
                    or (key == "action_constraint" and isinstance(value, str))
                    or (
                        key == "say_token_spans"
                        and isinstance(value, list)
                        and all(
                            isinstance(span, dict)
                            and set(span) == {"token_index", "token_id", "char_start", "char_end"}
                            and all(isinstance(item, int) and not isinstance(item, bool) for item in span.values())
                            for span in value
                        )
                    )
                    or (key == "status" and isinstance(value, str))
                    or (key == "failure_reason" and isinstance(value, str))
                )
            }
            if phase == "thinker":
                measured.update(_filter_native_thinker_lock_metrics(values))
            _retain_archive_transcript_commitment(measured, phase=phase, values=values)
            measured.update(_filter_frame_residual_runtime_metrics(values))
            measured.update(_filter_turn_local_speech_runtime_metrics(values))
            graph_requested = values.get("native_talker_q1_graph_requested")
            if type(graph_requested) is bool:
                measured["native_talker_q1_graph_requested"] = graph_requested
            graph_calls = values.get("native_talker_q1_graph_calls")
            if type(graph_calls) is int and 0 <= graph_calls <= 10_000_000:
                measured["native_talker_q1_graph_calls"] = graph_calls
            graph_runtime = _filter_native_talker_q1_graph_runtime(
                values.get("native_talker_q1_graph_runtime")
            )
            if graph_runtime:
                measured["native_talker_q1_graph_runtime"] = graph_runtime
            with response_runtime_metrics_lock:
                response_runtime_metrics.setdefault(response_id, {})[phase] = measured

        def start_audio_history_archive(archive_id: int, response_id: str) -> bool:
            archive_id = int(archive_id)
            with audio_archive_lock:
                if archive_id in audio_archive_futures:
                    return True
            snapshot = sess.get_audio_history_archive(archive_id)
            if not isinstance(snapshot, dict):
                update_response_runtime_metrics(response_id, "audio_history_archive", {
                    "status": "stale",
                    "archive_id": archive_id,
                    "session_id": sess.session_id,
                })
                return False
            if (
                snapshot.get("submitted_at_monotonic_ns") is not None
                or snapshot.get("completed_at_monotonic_ns") is not None
                or snapshot.get("status") in {"transcript_ready", "empty", "failed"}
            ):
                return True
            pcm = snapshot.get("pcm")
            if snapshot.get("source") != "post_response_asr" or not isinstance(pcm, np.ndarray):
                return False
            submit_started_at_ns = time.monotonic_ns()
            try:
                future, deferred = _submit_or_defer_audio_history_archive(
                    defer_offline=audio_history_archive_defer_offline,
                    archive_id=archive_id,
                    response_id=response_id,
                    session_id=sess.session_id,
                    snapshot=snapshot,
                    deferred_at_monotonic_ns=submit_started_at_ns,
                    worker_submit=CPU_AUDIO_ARCHIVE_WORKER.submit,
                    update_metrics=update_response_runtime_metrics,
                )
            except Exception as exc:
                sess.fail_audio_history_archive(archive_id, str(exc))
                update_response_runtime_metrics(response_id, "audio_history_archive", {
                    "status": "failed_cpu_submit",
                    "failure_reason": str(exc)[:200],
                    "archive_id": archive_id,
                    "session_id": sess.session_id,
                    "source": snapshot.get("source"),
                    "response_created_at_monotonic_ns": snapshot.get("response_created_at_monotonic_ns"),
                    "reserved_at_monotonic_ns": snapshot.get("reserved_at_monotonic_ns"),
                    "submit_started_at_monotonic_ns": submit_started_at_ns,
                    "submitted_at_monotonic_ns": None,
                })
                return False
            if deferred:
                return False
            if future is None:
                sess.fail_audio_history_archive(archive_id, "bounded_cpu_archive_queue_full")
                update_response_runtime_metrics(response_id, "audio_history_archive", {
                    "status": "skipped_backpressure",
                    "archive_id": archive_id,
                    "session_id": sess.session_id,
                    "source": snapshot.get("source"),
                    "response_created_at_monotonic_ns": snapshot.get("response_created_at_monotonic_ns"),
                    "reserved_at_monotonic_ns": snapshot.get("reserved_at_monotonic_ns"),
                    "submit_started_at_monotonic_ns": submit_started_at_ns,
                    "submitted_at_monotonic_ns": None,
                    "input_first_frame_id": snapshot.get("first_frame_id"),
                    "input_last_frame_id": snapshot.get("last_frame_id"),
                    "created_at_monotonic_ns": snapshot.get("created_at_monotonic_ns"),
                })
                return False
            submitted_at_ns = time.monotonic_ns()
            with sess._generation_lock:
                submitted = sess.mark_audio_history_archive_submitted(
                    archive_id,
                    submit_started_at_monotonic_ns=submit_started_at_ns,
                    submitted_at_monotonic_ns=submitted_at_ns,
                )
                if not submitted:
                    future.cancel()
                    update_response_runtime_metrics(response_id, "audio_history_archive", {
                        "status": "stale_before_submit",
                        "failure_reason": "session_closed_or_archive_missing",
                        "archive_id": archive_id,
                        "session_id": sess.session_id,
                        "response_created_at_monotonic_ns": snapshot.get("response_created_at_monotonic_ns"),
                        "reserved_at_monotonic_ns": snapshot.get("reserved_at_monotonic_ns"),
                        "submit_started_at_monotonic_ns": submit_started_at_ns,
                        "submitted_at_monotonic_ns": submitted_at_ns,
                    })
                    return False
                with audio_archive_lock:
                    audio_archive_futures[archive_id] = future
            update_response_runtime_metrics(response_id, "audio_history_archive", {
                "status": "submitted",
                "archive_id": archive_id,
                "session_id": sess.session_id,
                "source": snapshot.get("source"),
                "response_created_at_monotonic_ns": snapshot.get("response_created_at_monotonic_ns"),
                "reserved_at_monotonic_ns": snapshot.get("reserved_at_monotonic_ns"),
                "submit_started_at_monotonic_ns": submit_started_at_ns,
                "submitted_at_monotonic_ns": submitted_at_ns,
                "input_first_frame_id": snapshot.get("first_frame_id"),
                "input_last_frame_id": snapshot.get("last_frame_id"),
                "created_at_monotonic_ns": snapshot.get("created_at_monotonic_ns"),
            })

            def finish_archive(done: Any) -> None:
                completed_at_ns = time.monotonic_ns()
                try:
                    result = done.result()
                    text = str(result.get("text", "")).strip()
                    committed = sess.complete_audio_history_archive(archive_id, text)
                    status = "ready" if committed and text else "empty" if committed else "stale"
                    report = {
                        **{key: value for key, value in result.items() if key != "text"},
                        "archive_total_ms": (completed_at_ns - submit_started_at_ns) / 1_000_000.0,
                        "archive_submit_to_complete_ms": (completed_at_ns - submitted_at_ns) / 1_000_000.0,
                        "archive_reservation_to_submit_ms": (
                            (submitted_at_ns - int(snapshot.get("reserved_at_monotonic_ns", submitted_at_ns)))
                            / 1_000_000.0
                        ),
                        "status": status,
                        "archive_id": archive_id,
                        "session_id": sess.session_id,
                        "source": snapshot.get("source"),
                        "response_created_at_monotonic_ns": snapshot.get("response_created_at_monotonic_ns"),
                        "reserved_at_monotonic_ns": snapshot.get("reserved_at_monotonic_ns"),
                        "submit_started_at_monotonic_ns": submit_started_at_ns,
                        "submitted_at_monotonic_ns": submitted_at_ns,
                        "completed_at_monotonic_ns": completed_at_ns,
                        "input_first_frame_id": snapshot.get("first_frame_id"),
                        "input_last_frame_id": snapshot.get("last_frame_id"),
                        "created_at_monotonic_ns": snapshot.get("created_at_monotonic_ns"),
                        "sample_rate": snapshot.get("sample_rate"),
                    }
                    if not committed:
                        with sess._generation_lock:
                            report["failure_reason"] = (
                                "session_closed" if sess._closed else "archive_reservation_missing"
                            )
                    update_response_runtime_metrics(response_id, "audio_history_archive", report)
                except Exception as exc:
                    sess.fail_audio_history_archive(archive_id, str(exc))
                    update_response_runtime_metrics(response_id, "audio_history_archive", {
                        "archive_total_ms": (completed_at_ns - submit_started_at_ns) / 1_000_000.0,
                        "archive_submit_to_complete_ms": (completed_at_ns - submitted_at_ns) / 1_000_000.0,
                        "status": "failed",
                        "archive_id": archive_id,
                        "session_id": sess.session_id,
                        "source": snapshot.get("source"),
                        "response_created_at_monotonic_ns": snapshot.get("response_created_at_monotonic_ns"),
                        "reserved_at_monotonic_ns": snapshot.get("reserved_at_monotonic_ns"),
                        "submit_started_at_monotonic_ns": submit_started_at_ns,
                        "submitted_at_monotonic_ns": submitted_at_ns,
                        "completed_at_monotonic_ns": completed_at_ns,
                        "input_first_frame_id": snapshot.get("first_frame_id"),
                        "input_last_frame_id": snapshot.get("last_frame_id"),
                        "failure_reason": str(exc)[:200],
                    })
                finally:
                    with audio_archive_lock:
                        if audio_archive_futures.get(archive_id) is done:
                            audio_archive_futures.pop(archive_id, None)

            future.add_done_callback(finish_archive)
            return True

        def resolve_previous_audio_history_archive(response_id: str) -> None:
            """Observe archive progress without blocking response admission."""
            if not direct_audio_candidate:
                return
            with sess._generation_lock:
                pending = sess.pending_audio_history_archive
                pending_id = int(pending.get("archive_id", -1)) if isinstance(pending, dict) else None
                pending_status = str(pending.get("status", "pending")) if isinstance(pending, dict) else "none"
            if pending_id is None:
                return
            with audio_archive_lock:
                future = audio_archive_futures.get(pending_id)
            update_response_runtime_metrics(response_id, "audio_history_archive_wait", {
                "wait_ms": 0.0,
                "status": "running" if future is not None and not future.done() else pending_status,
                "archive_id": pending_id,
                "response_admission_blocked": False,
            })

        def initialize_generation_trace(
            response_id: str,
            action: ControlAction | None,
            history_roles: list[str] | None = None,
        ) -> None:
            if action is None:
                return
            instruction = None if direct_audio_candidate else action_conditioning_instruction(action.value)
            with response_runtime_metrics_lock:
                response_runtime_metrics.setdefault(response_id, {})["generation_trace"] = {
                    "conditioned_action": action.value,
                    "action_instruction": instruction,
                    "prompt_action_instruction": not direct_audio_candidate,
                    "completed_history_roles": list(history_roles or []),
                    "thinker_token_ids": [],
                    "first_thinker_token_id": None,
                    "terminal_reason": None,
                }

        def record_thinker_token_id(response_id: str, token_id: int) -> None:
            with response_runtime_metrics_lock:
                trace = response_runtime_metrics.setdefault(response_id, {}).get("generation_trace")
                if isinstance(trace, dict):
                    token_ids = trace.setdefault("thinker_token_ids", [])
                    if not token_ids:
                        trace["first_thinker_token_id"] = int(token_id)
                    token_ids.append(int(token_id))

        def runtime_metrics_event(response_id: str, result: Any | None = None) -> dict[str, Any]:
            if isinstance(result, dict):
                for phase, values in (result.get("runtime_metrics") or {}).items():
                    if phase == "thinker" and isinstance(values, dict):
                        terminal_reason = values.get("terminal_reason")
                        if isinstance(terminal_reason, str):
                            with response_runtime_metrics_lock:
                                trace = response_runtime_metrics.setdefault(response_id, {}).get("generation_trace")
                                if isinstance(trace, dict) and trace.get("terminal_reason") is None:
                                    trace["terminal_reason"] = terminal_reason
                    update_response_runtime_metrics(response_id, str(phase), values)
            with response_runtime_metrics_lock:
                phases = {
                    phase: dict(values) if isinstance(values, dict) else values
                    for phase, values in response_runtime_metrics.get(response_id, {}).items()
                }
            phases["scheduler"] = sess.scheduler.stats()
            return phases

        def mark_cancelled_response_playback(payload: Mapping[str, Any]) -> None:
            if payload.get("type") not in {"response.cancelled", "audio.cancelled"}:
                return
            response_value = payload.get("response_id")
            if not isinstance(response_value, str) or not response_value:
                return
            with playback_ack_branches_lock:
                branch = playback_ack_branches.get(response_value)
            if branch is not None:
                _cancel_response_branch_playback(branch)

        async def send_ws(
            payload: dict[str, Any],
            token: GenerationToken | None = None,
            generation_id: int | None = None,
            transcript_epoch: int | None = None,
            transcript_utterance_id: str | None = None,
            transcript_allow_response_active: bool = False,
        ) -> bool:
            """Serialize WS writers and fence response/scheduler payloads."""
            async with ws_send_lock:
                if transcript_epoch is not None:
                    # Hold the transcript lock through transport handoff so an
                    # opt-out, commit, new utterance, or response fence cannot
                    # race between this final check and sending private text.
                    with transcript_state_lock:
                        if (
                            not native_partial_asr_enabled.is_set()
                            or int(transcript_state["epoch"]) != int(transcript_epoch)
                            or transcript_state["utterance_id"] != transcript_utterance_id
                        ):
                            return True  # stale event deliberately dropped
                        with sess._generation_lock:
                            if (
                                generation_id is not None
                                and not sess.is_generation_current(generation_id=generation_id)
                            ):
                                return True
                            response_busy = bool(
                                sess.active_response
                                or sess.policy_response_pending
                                or sess.control_state in {
                                    ControlState.SPEAKING,
                                    ControlState.BACKCHANNELING,
                                    ControlState.STOPPING,
                                }
                            )
                        if response_busy and not transcript_allow_response_active:
                            return True
                        try:
                            await asyncio.wait_for(ws.send_json(payload), timeout=0.08)
                        except asyncio.TimeoutError:
                            logger.warning("[realtime %s] WebSocket send timeout", sess.session_id[:8])
                            return False
                        mark_cancelled_response_playback(payload)
                        return True
                # ws_send_lock serializes every write. Only generation-fenced
                # outputs take the control lock; ingress notifications have no
                # response owner and must remain deliverable while V9 holds it.
                # Never keep the control lock across network backpressure.
                if token is not None or generation_id is not None:
                    with sess._generation_lock:
                        if token is not None and not sess.is_generation_current(token):
                            return False
                        if token is None and generation_id is not None and not sess.is_generation_current(generation_id=generation_id):
                            return False
                try:
                    await asyncio.wait_for(ws.send_json(payload), timeout=0.08)
                except asyncio.TimeoutError:
                    logger.warning("[realtime %s] WebSocket send timeout", sess.session_id[:8])
                    return False
                mark_cancelled_response_playback(payload)
                return True

        async def publish_scheduler_events() -> None:
            nonlocal closing
            while True:
                event = await scheduler_events.get()
                if event is None:
                    return
                if event is scheduler_events.OVERFLOW:
                    closing = True
                    logger.error(
                        "[realtime %s] scheduler event overload; closing connection",
                        sess.session_id[:8],
                    )
                    try:
                        await send_ws({
                            "type": "error",
                            "error": {
                                "code": "SCHEDULER_EVENT_OVERLOAD",
                                "message": "Realtime event backlog exceeded capacity; reconnect the session",
                            },
                        })
                        await ws.close(code=1013, reason="scheduler event overload")
                    except (WebSocketDisconnect, RuntimeError):
                        pass
                    return
                try:
                    event = dict(event)
                    generation_id = event.pop("_generation_fence", None)
                    transcript_epoch = event.pop("_transcript_epoch", None)
                    transcript_utterance_id = event.pop("_transcript_utterance_id", None)
                    ack_enqueued_ns = event.pop("_ack_enqueued_ns", None)
                    transcript_allow_response_active = bool(
                        event.pop("_transcript_allow_response_active", False)
                    )
                    transcript_source = event.pop("_transcript_source", None)
                    if transcript_source is not None:
                        logger.info(
                            "[realtime %s] input transcript source=%s type=%s revision=%s frames=%s-%s",
                            sess.session_id[:8], transcript_source, event.get("type"),
                            event.get("revision"), event.get("first_frame_id"),
                            event.get("last_frame_id"),
                        )
                    replan_followup_generation = event.pop("_replan_followup_generation", None)
                    scheduler_failed = event.get("type") == "error"
                    if scheduler_failed:
                        closing = True
                    if event.get("type") == "input_audio_buffer.consumed" and isinstance(ack_enqueued_ns, int):
                        event["server_ack_queue_ms"] = round(
                            max(0, time.monotonic_ns() - ack_enqueued_ns) / 1_000_000.0,
                            3,
                        )
                    sent = await send_ws(
                        event,
                        generation_id=generation_id,
                        transcript_epoch=transcript_epoch,
                        transcript_utterance_id=transcript_utterance_id,
                        transcript_allow_response_active=transcript_allow_response_active,
                    )
                    if not sent and (
                        generation_id is None
                        or sess.is_generation_current(generation_id=generation_id)
                    ):
                        logger.warning("[realtime %s] scheduler event send failed; closing connection", sess.session_id[:8])
                        closing = True
                        await ws.close(code=1013, reason="realtime event transport stalled")
                        return
                    if sent and scheduler_failed:
                        await ws.close(code=1011, reason="scheduler failed")
                        return
                    if sent and replan_followup_generation is not None:
                        with replan_cancel_waiters_lock:
                            waiter = replan_cancel_waiters.pop(int(replan_followup_generation), None)
                        if waiter is not None:
                            waiter.set()
                except (WebSocketDisconnect, RuntimeError):
                    return

        def scheduler_listener(frame: Any, tick: Any) -> None:
            """Consume every complete mic frame before invoking the policy."""
            nonlocal last_listener_phase_key
            if native_listener_output_faulted.is_set():
                return
            if frame is None or sess.listener is None:
                return
            conditioning_frame_published = False
            conditioning_frame_unavailable_reason = "not_published"
            conditioning_segment: str | None = None
            consumed_event: dict[str, Any] | None = None
            with sess._generation_lock:
                listener_generation = sess.session_generation_id
            reference = (
                getattr(tick, "microbatch_reference", None)
                if shared_listener_microbatch_enabled
                else sess.assistant_reference_buffer.visible_for(
                    listener_frame_id=sess.listener_state.next_input_frame_id
                )
            )
            phase_key = str(getattr(sess.control_state, "value", sess.control_state))
            phase_changed = last_listener_phase_key is not None and phase_key != last_listener_phase_key
            last_listener_phase_key = phase_key
            silence_skip_candidate = bool(
                listener_silence_skip_enabled
                and not shared_listener_microbatch_enabled
                and _silence_skip_inputs_are_exact_zero(
                    frame.pcm, getattr(reference, "pcm", None)
                )
                and not bool(speech_tracker.voice_run_active)
                and not phase_changed
            )
            query_coherent, query_fallback_reason = (
                _silence_skip_query_is_coherent(sess, int(frame.frame_id))
                if silence_skip_candidate else (False, None)
            )
            silence_skip_transition_lock_acquired = False
            with sess._generation_lock:
                unanswered_user_turn = _user_turn_is_unanswered(
                    sess.pending_user_onset_frame_id,
                    sess.last_answered_onset_frame_id,
                )
                active_response_for_skip = bool(sess.active_response)
                response_pending_for_skip = bool(sess.policy_response_pending)
                control_phase_idle_for_skip = sess.control_state == ControlState.LISTENING
            with replan_endpoint_lock:
                replan_endpoint_pending = bool(replan_endpoint_waiters)
            pending_plan_for_skip = bool(
                pending_speak is not None
                or replan_handoff_pending
                or not pending_response_speak_handoffs.empty()
                or replan_endpoint_pending
            )
            allow_silence_skip = _silence_skip_is_true_idle(
                candidate=silence_skip_candidate,
                query_coherent=query_coherent,
                unanswered_user_turn=unanswered_user_turn,
                active_response=active_response_for_skip,
                response_pending=response_pending_for_skip,
                pending_plan=pending_plan_for_skip,
                control_phase_idle=control_phase_idle_for_skip,
            )
            silence_skip_fallback_reason = None
            if allow_silence_skip:
                # Serialize the cached-output fast path with begin_response().
                # Once a response boundary starts, following ingress ticks must
                # use the full listener/V9 conditioning path.
                sess._silence_skip_transition_lock.acquire()
                silence_skip_transition_lock_acquired = True
                with sess._generation_lock:
                    allow_silence_skip = _silence_skip_transition_is_still_idle(
                        snapshot_generation_id=int(listener_generation),
                        current_generation_id=int(sess.session_generation_id),
                        active_response=bool(sess.active_response),
                        response_pending=bool(sess.policy_response_pending),
                        control_state=sess.control_state,
                    )
                if not allow_silence_skip:
                    sess._silence_skip_transition_lock.release()
                    silence_skip_transition_lock_acquired = False
                    silence_skip_fallback_reason = "response_transition_started"
            if silence_skip_candidate and not query_coherent:
                silence_skip_fallback_reason = query_fallback_reason or "query_hidden_incoherent"
                logger.info(
                    "[realtime %s] silence-skip fallback frame=%d reason=%s; run full listener",
                    sess.session_id[:8], int(frame.frame_id), silence_skip_fallback_reason,
                )
            elif silence_skip_candidate and not allow_silence_skip:
                if silence_skip_fallback_reason is None:
                    silence_skip_fallback_reason = "conversation_work_pending"
                logger.info(
                    "[realtime %s] silence-skip fallback frame=%d reason=%s; run full listener",
                    sess.session_id[:8], int(frame.frame_id), silence_skip_fallback_reason,
                )
            listener_callback_started = time.perf_counter()
            listener_step_wall_ms = None
            listener_step_succeeded = False
            listener_step_started_ns: int | None = None
            listener_step_ended_ns: int | None = None
            q4_profile_host_context: dict[str, Any] | None = None
            output = None
            try:
                if not allow_silence_skip and not shared_listener_microbatch_enabled:
                    prepare_native_control6_semantic_prefix(
                        engine,
                        sess,
                        current_frame_id=int(frame.frame_id),
                        pending_speak=pending_speak,
                    )
                listener_step_started = time.perf_counter()
                listener_step_started_ns = (
                    int(tick.listener_batch_started_ns)
                    if shared_listener_microbatch_enabled
                    and isinstance(getattr(tick, "listener_batch_started_ns", None), int)
                    else time.monotonic_ns()
                )
                if NATIVE_LISTENER_CUDA_PROFILE_ENABLED and not shared_listener_microbatch_enabled:
                    q4_profile_host_context = native_q4_cuda_profile_host_context(
                        sess.session_id, frame, tick
                    )
                    q4_profile_host_context["listener_step_start_ns"] = listener_step_started_ns
                    sess.listener_state._native_q4_profile_host_context = q4_profile_host_context
                if shared_listener_microbatch_enabled:
                    row_output = getattr(tick, "microbatch_output", None)
                    if row_output is None:
                        raise RuntimeError("shared tick reached listener callback without its batch row")
                    commit_timing: dict[str, int] = {}
                    tick.microbatch_commit_start_ns = time.monotonic_ns()
                    try:
                        output = sess.listener.commit_microbatch_output(
                            frame.pcm,
                            reference,
                            sess.listener_state,
                            row_output,
                            source_available_at_ns=int(
                                tick.listener_batch_finished_ns
                                or tick.listener_end_ns
                                or time.monotonic_ns()
                            ),
                            timing_sink=commit_timing,
                        )
                        # The output's semantic revision has been imported into
                        # the primary native slot, and commit_microbatch_output
                        # has now committed its cursor/revision CAS. Only here
                        # may the shadow retire matching staged work.
                        if row_output.semantic_context_revision is not None:
                            binding = shared_listener_bindings.get(sess.session_id)
                            worker = shared_listener_worker_holder.get("worker")
                            mark_applied = getattr(
                                worker, "mark_semantic_prefix_applied", None
                            )
                            if callable(mark_applied):
                                request = (
                                    binding.get("semantic_request_holder", {}).get("request")
                                    if binding is not None else None
                                )
                                request_generation = (
                                    binding.get("semantic_request_holder", {}).get(
                                        "listener_generation_id"
                                    )
                                    if binding is not None else None
                                )
                                if not _semantic_prefix_request_matches_applied_row(
                                    request, request_generation, row_output
                                ):
                                    request = row_output
                                mark_applied(
                                    request,
                                    listener_generation_id=int(row_output.generation_id),
                                    current_frame_id=int(row_output.frame_id),
                                    listener_state_version=int(
                                        row_output.listener_state_version
                                    ),
                                )
                    finally:
                        tick.microbatch_commit_end_ns = time.monotonic_ns()
                        wait_ns = commit_timing.get("state_lock_wait_ns")
                        hold_ns = commit_timing.get("state_lock_hold_ns")
                        tick.microbatch_commit_state_lock_wait_ms = (
                            round(wait_ns / 1_000_000.0, 3)
                            if isinstance(wait_ns, int) else None
                        )
                        tick.microbatch_commit_state_lock_hold_ms = (
                            round(hold_ns / 1_000_000.0, 3)
                            if isinstance(hold_ns, int) else None
                        )
                        tick.microbatch_commit_state_lock_acquired_ns = commit_timing.get(
                            "state_lock_acquired_ns"
                        )
                        tick.microbatch_commit_state_lock_released_ns = commit_timing.get(
                            "state_lock_released_ns"
                        )
                else:
                    output = sess.listener.step(
                        frame.pcm,
                        reference,
                        sess.listener_state,
                        allow_silence_skip=allow_silence_skip,
                    )
                if silence_skip_transition_lock_acquired:
                    sess._silence_skip_transition_lock.release()
                    silence_skip_transition_lock_acquired = False
                listener_step_ended_ns = (
                    int(tick.listener_batch_finished_ns)
                    if shared_listener_microbatch_enabled
                    and isinstance(getattr(tick, "listener_batch_finished_ns", None), int)
                    else time.monotonic_ns()
                )
                listener_step_wall_ms = (
                    tick.microbatch_compute_ms
                    if shared_listener_microbatch_enabled
                    else round((time.perf_counter() - listener_step_started) * 1000.0, 3)
                )
                listener_step_succeeded = True
                # The ingress acknowledgement describes the current PCM frame
                # consumed by the listener frontend.  It must not wait for a
                # Gemma query, a policy decision, or a V9 conditioning output:
                # bootstrap ticks may advance the input state without producing
                # any of those artifacts yet.
                if input_ack_enabled.is_set():
                    consumed_event = {
                        "type": "input_audio_buffer.consumed",
                        "frame_id": int(frame.frame_id),
                        "server_listener_queue_ms": round(
                            max(0, time.monotonic_ns() - int(frame.enqueue_ns)) / 1_000_000.0,
                            3,
                        ),
                        "server_listener_step_ms": listener_step_wall_ms,
                        "server_listener_step_start_ns": listener_step_started_ns,
                        "server_listener_step_end_ns": listener_step_ended_ns,
                        **scheduler_tick_trace_metadata(
                            tick, include_host_lock_timeline=True
                        ),
                        "_ack_enqueued_ns": time.monotonic_ns(),
                    }
                    if output is not None:
                        output_frame_id = getattr(output, "frame_id", None)
                        if isinstance(output_frame_id, int) and not isinstance(output_frame_id, bool):
                            consumed_event["listener_output_frame_id"] = int(output_frame_id)
                        consumed_event["listener_output_source_frame_id"] = getattr(
                            output, "source_frame_id", output_frame_id
                        )
                        consumed_event["listener_output_fresh"] = bool(getattr(output, "fresh", True))
                        consumed_event["listener_silence_skipped"] = bool(
                            getattr(output, "silence_skipped", False)
                        )
                        logical_q_len = getattr(output, "logical_q_len", None)
                        ephemeral_reason = getattr(output, "ephemeral_reason", None)
                        if ephemeral_reason is not None:
                            consumed_event["ephemeral_reason"] = str(ephemeral_reason)
                        consumed_event["queue_audio_for_next_turn"] = bool(
                            getattr(output, "queue_audio_for_next_turn", False)
                        )
                        cache_after = getattr(output, "listener_cache_position", None)
                        valid_after = getattr(output, "listener_valid_history_tokens", None)
                        if isinstance(logical_q_len, int) and not isinstance(logical_q_len, bool):
                            consumed_event["logical_q_len"] = logical_q_len
                            if isinstance(cache_after, int) and not isinstance(cache_after, bool):
                                consumed_event["listener_cache_position_after"] = cache_after
                                consumed_event["listener_cache_position_before"] = (
                                    cache_after - logical_q_len
                                )
                            if isinstance(valid_after, int) and not isinstance(valid_after, bool):
                                consumed_event["listener_valid_history_tokens_after"] = valid_after
                                consumed_event["listener_valid_history_tokens_before"] = (
                                    valid_after - logical_q_len
                                )
                        for field in (
                            "kv_rollback_tokens",
                            "kv_rollback_enqueue_ms",
                            "native_cache_current_position_after",
                            "native_cache_valid_kv_len_after",
                        ):
                            value = getattr(output, field, None)
                            if value is not None:
                                consumed_event[field] = value
                    if silence_skip_fallback_reason is not None:
                        consumed_event["listener_silence_skip_fallback_reason"] = silence_skip_fallback_reason
                    scheduler_events.put_nowait(consumed_event)
                if output is None:
                    conditioning_frame_unavailable_reason = "listener_output_unavailable"
                    # A listener warmup frame is consumed even though it has no
                    # hidden/query to publish. Clear prior-frame data so the
                    # policy cannot mistake stale features for this input tick.
                    with sess._generation_lock:
                        if listener_generation == sess.session_generation_id:
                            sess.mark_listener_frame(int(frame.frame_id))
                            sess.listener_hidden = None
                            sess.listener_hidden_frame_id = None
                            sess.listener_hidden_at_ns = None
                            sess.listener_audio_queries = None
                            sess.listener_audio_frame_id = None
                            sess.last_listener_output = None
                        else:
                            sess.stale_session_mutations += 1
                    return
                if not bool(getattr(output, "fresh", True)):
                    conditioning_frame_unavailable_reason = "listener_output_not_fresh"
                    source_frame_id = getattr(output, "source_frame_id", None)
                    source_available_at_ns = getattr(output, "source_available_at_ns", None)
                    with sess._generation_lock:
                        if listener_generation != sess.session_generation_id:
                            sess.stale_session_mutations += 1
                            return
                        if (
                            not bool(getattr(output, "silence_skipped", False))
                            or not isinstance(source_frame_id, int)
                            or source_frame_id >= int(frame.frame_id)
                            or sess.listener_audio_queries is None
                            or sess.listener_audio_frame_id != source_frame_id
                        ):
                            scheduler_error(RuntimeError("silence-skip senza query/hidden sorgente coerenti"))
                            return
                        sess.mark_listener_frame(int(frame.frame_id))
                        sess.listener_hidden = output.gemma_hidden_t
                        sess.listener_hidden_frame_id = source_frame_id
                        sess.listener_hidden_at_ns = source_available_at_ns
                        sess.last_listener_output = output
                    return
                record_native_control6_semantic_audio(
                    engine,
                    sess,
                    output,
                    expected_session_generation=listener_generation,
                )
                with sess._generation_lock:
                    if listener_generation != sess.session_generation_id:
                        sess.stale_session_mutations += 1
                        return
                    if isinstance(sess.turn_policy, DuplexControl6Policy):
                        queries = getattr(output, "gemma_audio_queries_t", None)
                        if callable(getattr(queries, "detach", None)):
                            sess.listener_audio_queries = queries.detach().clone()
                            sess.listener_audio_frame_id = int(output.frame_id)
                        else:
                            sess.listener_audio_queries = None
                            sess.listener_audio_frame_id = None
                            logger.debug(
                                "[realtime %s] listener consumed frame=%d without fresh Control6 query",
                                sess.session_id[:8], int(frame.frame_id),
                            )
                v9_stream = _listener_v9_stream_for_engine(engine)
                if v9_stream is None:
                    conditioning_frame_unavailable_reason = "v9_stream_unavailable"
                listener_hidden = output.gemma_hidden_t
                trace_evidence_role = "fixture_causality_diagnostic"
                trace_model_provenance: dict[str, Any] = {
                    "model": "Gemma4Model",
                    "frozen": True,
                    "runtime_profile": getattr(sess.listener, "runtime_profile", "unknown"),
                    "text_context": "separate_plan",
                    "native_attestation_observed": False,
                }
                if sess.conditioning_trace_enabled:
                    if v9_stream is not None and sess.conditioning_trace_projector_provenance is None:
                        projector = getattr(v9_stream, "projector", None)
                        if projector is not None and hasattr(projector, "named_parameters"):
                            sess.conditioning_trace_projector_provenance = {
                                "used": True,
                                "frozen": all(not parameter.requires_grad for parameter in projector.parameters()),
                                "training": bool(projector.training),
                                "optimizer_membership": "serving_no_optimizer",
                                "parameter_sha256_before": _projector_parameter_sha256(projector),
                            }
                    with sess._generation_lock:
                        if sess.conditioning_trace_source_fingerprint is None:
                            sess.conditioning_trace_source_fingerprint = runtime_source_fingerprint()
                        if sess.conditioning_trace_runtime_provenance is None:
                            try:
                                native_backend = getattr(sess.listener, "backend", None)
                                runtime_info = native_backend.native_runtime_info()
                            except (AttributeError, RuntimeError) as exc:
                                runtime_info = {"enabled": False, "error": str(exc)}
                            sess.conditioning_trace_runtime_provenance = dict(runtime_info)
                        trace_model_provenance["native_runtime"] = dict(sess.conditioning_trace_runtime_provenance)
                        trace_model_provenance["native_attestation_observed"] = bool(
                            sess.conditioning_trace_runtime_provenance.get("enabled") is True
                        )
                        trace_evidence_role = _listener_evidence_role(
                            trace_model_provenance["runtime_profile"],
                            native_runtime_enabled=trace_model_provenance["native_attestation_observed"],
                        )
                live_input_provenance = pcm_frame_provenance(frame.pcm, sample_rate_hz=USER_SAMPLE_RATE)
                live_input_provenance.update({
                    "audio_frame_id": int(frame.frame_id),
                    "listener_output_frame_id": int(output.frame_id),
                    "state_epoch_id": sess.listener_state.state_epoch_id,
                    "state_lineage_digest": sess.listener_state.state_lineage_digest,
                    "capture_timestamp_source": frame.capture_timestamp_source,
                    "capture_start_ns": int(frame.capture_start_ns),
                    "capture_end_ns": int(frame.capture_end_ns),
                    "transport_sha256": frame.transport_sha256,
                    "transport_serialization": frame.transport_serialization,
                    "assistant_reference_frame_id": None if reference is None else int(reference.frame_id),
                    "assistant_reference_emitted_at_ns": None if reference is None else int(reference.emitted_at_ns),
                    "assistant_reference_pcm_sha256": None if reference is None else pcm_frame_provenance(
                        reference.pcm, sample_rate_hz=int(reference.sample_rate)
                    )["sha256"],
                    "assistant_reference_sample_count": None if reference is None else int(reference.pcm.size),
                    "assistant_reference_sample_rate_hz": None if reference is None else int(reference.sample_rate),
                })
                live_anchor = (
                    {
                        "name": "first_live_listener_hidden_of_current_turn",
                        "frame_id": 0,
                        "source": "session.v9_text_anchor",
                    }
                    if v9_stream is not None else "none_without_v9"
                )
                if sess.conditioning_trace_enabled:
                    sess.record_conditioning_trace(
                        make_trace_row(
                            source_path=CONDITIONING_PATHS["C"],
                            sample_id=sess.conditioning_trace_sample_id,
                            stream_id=sess.session_id,
                            frame_id=int(frame.frame_id),
                            timestamp_ns=int(frame.capture_end_ns),
                            available_at_ns=int(frame.capture_end_ns),
                            available_through_frame=int(frame.frame_id),
                            token_type="audio_feature_token",
                            evidence_role=trace_evidence_role,
                            mode="live_causal_listener",
                            prompt_visibility="not_visible",
                            text_visibility="separate_plan_not_in_listener_hidden",
                            frontend="Gemma4AudioFeatureExtractor.incremental_left160_current1280_sentinel1",
                            hidden_layer="Gemma4Model.language_model.last_hidden_state.first_audio_feature_of_current_frame",
                            hidden_stage="pre_causal_conv",
                            hidden_kind="h_raw",
                            hidden=listener_hidden,
                            causal_conv_stage="not_applied",
                            anchor=live_anchor,
                            availability_kind="AudioFrame.capture_end_ns_current_frame_only",
                            future_pcm_visible=False,
                            state_version=output.listener_state_version,
                            state_lineage="GemmaListenerState.listener_state_version",
                            generation_id=listener_generation,
                            segment_id=sess.speaker_segment_id,
                            source_availability_evidence="DuplexScheduler AudioFrame.capture_end_ns",
                            input_provenance=live_input_provenance,
                            source_fingerprint=sess.conditioning_trace_source_fingerprint,
                            model_provenance=trace_model_provenance,
                            projector_provenance={"used": False, "next_stage": "IncrementalV9Projector.step"},
                        ),
                        expected_generation_id=listener_generation,
                    )
                if v9_stream is not None:
                    import torch
                    # Generation fencing and conditioning append are one
                    # transaction.  begin_segment() takes the same lock before
                    # rebasing the buffer; therefore a frame cannot be read as
                    # generation G and appended after G has been replaced.
                    with sess._generation_lock:
                        if listener_generation != sess.session_generation_id:
                            sess.stale_session_mutations += 1
                            conditioning_frame_unavailable_reason = "session_generation_changed"
                            return
                        conditioning_generation = listener_generation
                        conditioning_segment = sess.speaker_segment_id or ""
                        with sess.v9_state_lock:
                            if sess.v9_state is None:
                                v9_parameter = v9_stream.projector.audio_in.weight
                                sess.v9_state = v9_stream.new_state(
                                    batch_size=1,
                                    device=v9_parameter.device,
                                    dtype=v9_parameter.dtype,
                                    max_seq_len=v9_stream.max_seq_len,
                                )
                                # P4D causal anchor: freeze the first Gemma
                                # pre-conv hidden received at t=0.  It is available
                                # without looking into future frames and remains
                                # fixed for the whole V9 sequence.
                                sess.v9_text_anchor = torch.as_tensor(
                                    output.gemma_hidden_t,
                                    device=sess.v9_state.device,
                                    dtype=sess.v9_state.dtype,
                                ).reshape(1, v9_stream.raw_dim).clone()
                            listener_hidden = v9_stream.step(
                                output.gemma_hidden_t, sess.v9_text_anchor, sess.v9_state, commit=False
                            )
                            # Commit the externally visible frame counter only
                            # after the corresponding hidden is published. This
                            # removes the frames_processed/listener_hidden race
                            # without reversing the session/V9 lock order.
                            sess.listener_hidden = listener_hidden
                            sess.listener_hidden_frame_id = int(output.frame_id)
                            sess.listener_hidden_at_ns = time.monotonic_ns()
                            sess.v9_state.frames_processed += 1
                            if sess.conditioning_trace_enabled and getattr(listener_hidden, "shape", ())[-1:] == (2048,):
                                projector_provenance = dict(sess.conditioning_trace_projector_provenance or {"used": True})
                                if "parameter_sha256_after" not in projector_provenance:
                                    projector = getattr(v9_stream, "projector", None)
                                    if projector is not None and hasattr(projector, "named_parameters"):
                                        projector_provenance["parameter_sha256_after"] = _projector_parameter_sha256(projector)
                                        sess.conditioning_trace_projector_provenance = projector_provenance
                                sess.record_conditioning_trace(
                                    make_trace_row(
                                        source_path=CONDITIONING_PATHS["C"],
                                        sample_id=sess.conditioning_trace_sample_id,
                                        stream_id=sess.session_id,
                                        frame_id=int(frame.frame_id),
                                        timestamp_ns=int(frame.capture_end_ns),
                                        available_at_ns=int(frame.capture_end_ns),
                                        available_through_frame=int(frame.frame_id),
                                        token_type="v9_conditioning_frame",
                                        evidence_role=trace_evidence_role,
                                        mode="live_causal_listener",
                                        prompt_visibility="not_visible",
                                        text_visibility="separate_plan_not_in_listener_hidden",
                                        frontend="Gemma4AudioFeatureExtractor.incremental_left160_current1280_sentinel1",
                                        hidden_layer="IncrementalV9Projector.final_norm",
                                        hidden_stage="post_causal_conv_projector_output",
                                        hidden_kind="h_pfx",
                                        hidden=listener_hidden,
                                        causal_conv_stage="inside_IncrementalV9Projector.step_before_projector",
                                        anchor={
                                            "name": "first_live_listener_hidden_of_current_turn",
                                            "frame_id": 0,
                                            "source": "session.v9_text_anchor",
                                        },
                                        availability_kind="AudioFrame.capture_end_ns_current_frame_only",
                                        future_pcm_visible=False,
                                        state_version=output.listener_state_version,
                                        state_lineage="GemmaListenerState.listener_state_version",
                                        generation_id=listener_generation,
                                        segment_id=conditioning_segment or None,
                                        source_availability_evidence="DuplexScheduler AudioFrame.capture_end_ns",
                                        input_provenance=live_input_provenance,
                                        source_fingerprint=sess.conditioning_trace_source_fingerprint,
                                        model_provenance=trace_model_provenance,
                                        projector_provenance={
                                            **projector_provenance,
                                            "profile": getattr(v9_stream.projector, "_v9_profile", "unknown"),
                                        },
                                    ),
                                    expected_generation_id=listener_generation,
                                )
                            # Production V9 is fixed at 2048 dims. Tiny CPU
                            # projectors used by the kernel contract tests do not
                            # represent a consumable Qwen conditioning frame.
                            if getattr(listener_hidden, "shape", ())[-1:] != (2048,):
                                conditioning_frame_unavailable_reason = "conditioning_hidden_dim_invalid"
                            elif (
                                not isinstance(getattr(output, "frame_id", None), int)
                                or isinstance(output.frame_id, bool)
                                or int(output.frame_id) != int(frame.frame_id)
                            ):
                                conditioning_frame_unavailable_reason = "listener_output_frame_mismatch"
                            else:
                                conditioning_frame_published = bool(sess.conditioning_buffer.append_values(
                                    frame_id=output.frame_id,
                                    h_pfx=listener_hidden,
                                    produced_at_ns=time.monotonic_ns(),
                                    generation_id=conditioning_generation,
                                    segment_id=conditioning_segment,
                                ))
                                conditioning_frame_unavailable_reason = (
                                    None if conditioning_frame_published else "conditioning_buffer_rejected"
                                )
            except Exception as exc:
                conditioning_frame_unavailable_reason = "listener_or_conditioning_error"
                logger.exception("[realtime %s] listener/V9 failed on frame %d", sess.session_id[:8], frame.frame_id)
                scheduler_error(exc)
                return
            finally:
                if silence_skip_transition_lock_acquired:
                    sess._silence_skip_transition_lock.release()
                    silence_skip_transition_lock_acquired = False
                state = sess.listener_state
                if q4_profile_host_context is not None:
                    q4_profile_host_context["listener_step_end_ns"] = listener_step_ended_ns
                    q4_profile_host_context["listener_callback_end_ns"] = time.monotonic_ns()
                    attach_native_q4_lock_metrics(q4_profile_host_context, state)
                    if getattr(state, "_native_q4_profile_host_context", None) is q4_profile_host_context:
                        state._native_q4_profile_host_context = None
                timing = {
                    "frame_id": int(frame.frame_id),
                    "capture_start_ns": int(frame.capture_start_ns),
                    "capture_end_ns": int(frame.capture_end_ns),
                    "enqueue_ns": int(frame.enqueue_ns),
                    "deadline_ns": getattr(tick, "deadline_ns", None),
                    "observed_ns": getattr(tick, "observed_ns", None),
                    "listener_step_start_ns": listener_step_started_ns,
                    "listener_step_end_ns": listener_step_ended_ns,
                    "listener_step_wall_ms": listener_step_wall_ms,
                    "listener_callback_wall_ms": round((time.perf_counter() - listener_callback_started) * 1000.0, 3),
                    "shared_lm_lock_wait_ms": getattr(state, "last_lock_wait_ms", None),
                    "shared_lm_lock_host_hold_ms": getattr(state, "last_lock_hold_ms", None),
                    "silence_skipped": bool(getattr(output, "silence_skipped", False)),
                    "hidden_fresh": bool(getattr(output, "fresh", False)),
                    "step_succeeded": listener_step_succeeded,
                }
                with listener_timing_lock:
                    rows = session_runtime_metrics["listener"].setdefault("tick_timings", [])
                    rows.append(timing)
                    if len(rows) > 128:
                        del rows[:len(rows) - 128]
                record_scheduler_stage("listener_step_wall_ms", listener_step_wall_ms)
                record_scheduler_stage(
                    "listener_frontend_wall_ms", getattr(state, "last_frontend_wall_ms", None)
                )
                record_scheduler_stage(
                    "listener_audio_tower_call_wall_ms",
                    getattr(state, "last_audio_tower_call_wall_ms", None),
                )
                record_scheduler_stage(
                    "listener_lm_lock_wait_wall_ms", getattr(state, "last_lock_wait_ms", None)
                )
                record_scheduler_stage(
                    "listener_lm_lock_hold_wall_ms", getattr(state, "last_lock_hold_ms", None)
                )
                if consumed_event is not None and input_conditioning_ack_enabled.is_set():
                    output_frame_id = getattr(output, "frame_id", None) if output is not None else None
                    conditioning_frame_id = (
                        int(output_frame_id)
                        if conditioning_frame_published
                        and isinstance(output_frame_id, int)
                        and not isinstance(output_frame_id, bool)
                        else None
                    )
                    conditioning_event: dict[str, Any] = {
                        "type": "input_audio_buffer.conditioning_available",
                        "frame_id": int(frame.frame_id),
                        "listener_output_frame_id": (
                            int(output_frame_id)
                            if isinstance(output_frame_id, int) and not isinstance(output_frame_id, bool)
                            else None
                        ),
                        "conditioning_frame_available": conditioning_frame_id == int(frame.frame_id),
                        "conditioning_frame_id": conditioning_frame_id,
                        "conditioning_generation_id": int(listener_generation),
                        "conditioning_segment_id": conditioning_segment or None,
                    }
                    if not conditioning_event["conditioning_frame_available"]:
                        conditioning_event["reason"] = conditioning_frame_unavailable_reason
                    scheduler_events.put_nowait(conditioning_event)
            with sess._generation_lock:
                if listener_generation != sess.session_generation_id:
                    sess.stale_session_mutations += 1
                    return
                sess.mark_listener_frame(output.frame_id)
                sess.listener_hidden = listener_hidden
                sess.listener_hidden_frame_id = int(
                    output.source_frame_id if output.source_frame_id is not None else output.frame_id
                )
                sess.listener_hidden_at_ns = int(
                    output.source_available_at_ns
                    if output.source_available_at_ns is not None else time.monotonic_ns()
                )
                sess.last_listener_output = output

        def _prepare_policy_tick_state(frame: Any, tick: Any) -> dict[str, Any]:
            """Apply the per-frame mutable scheduler updates exactly once.

            Shared native Control6 needs its immutable pre-head snapshot before
            the owner worker evaluates this frame. The scheduler callback then
            reuses this record after the listener result is committed.
            """
            nonlocal pending_speak, peer_audio_event_key, latest_policy_frame_id
            nonlocal partial_last_snapshot_frame
            with policy_cooldown_lock:
                latest_policy_frame_id = max(latest_policy_frame_id, int(frame.frame_id))
            new_voice_onset, speech_endpoint = speech_tracker.observe(
                int(frame.frame_id), frame.pcm
            )
            if new_voice_onset:
                sess.observe_user_onset(int(frame.frame_id))
            peer_audio_event_key = peer_audio_event_key_for_frame(
                peer_audio_event_key,
                int(frame.frame_id),
                new_voice_onset=new_voice_onset,
            )
            if new_voice_onset:
                begin_transcript_utterance(int(frame.frame_id))
                partial_last_snapshot_frame = None
                pending_speak, onset_intent_change = _advance_pending_speak_on_new_onset(
                    pending_speak,
                    int(frame.frame_id),
                )
                if onset_intent_change == "dropped_stale_deferred_native_intent":
                    logger.info(
                        "[realtime %s] drop deferred native SPEAK after a new utterance onset frame=%d",
                        sess.session_id[:8],
                        int(frame.frame_id),
                    )
                elif onset_intent_change == "retargeted_pending_speak":
                    logger.info(
                        "[realtime %s] pending SPEAK waits resumed voice run onset=%d",
                        sess.session_id[:8],
                        int(frame.frame_id),
                    )
            while True:
                try:
                    deferred_intent = pending_response_speak_handoffs.get_nowait()
                except queue.Empty:
                    break
                with sess._generation_lock:
                    intent_is_current = int(deferred_intent.get("generation_id", -1)) == int(
                        sess.session_generation_id
                    )
                onset_frame_id = deferred_intent.get("voice_run_onset_frame_id")
                if (
                    not intent_is_current
                    or onset_frame_id is None
                    or speech_tracker.run_onset_frame_id is None
                    or int(onset_frame_id) != int(speech_tracker.run_onset_frame_id)
                ):
                    logger.info(
                        "[realtime %s] drop deferred Thinker SPEAK handoff: stale utterance/generation",
                        sess.session_id[:8],
                    )
                    continue
                if pending_speak is None or int(
                    pending_speak.get("voice_run_onset_frame_id", -1)
                ) == int(onset_frame_id):
                    deferred_endpoint = (
                        int(speech_tracker.endpoint_frame_id)
                        if speech_tracker.speech_complete
                        and speech_tracker.endpoint_frame_id is not None
                        else None
                    )
                    pending_speak = {
                        "source_frame_id": int(deferred_intent.get("source_frame_id", frame.frame_id)),
                        "confidence": float(deferred_intent.get("confidence", 0.0)),
                        "source": _DEFERRED_NATIVE_SPEAK_SOURCE,
                        "voice_run_onset_frame_id": int(onset_frame_id),
                        "endpoint_frame_id": deferred_endpoint,
                        "endpoint_source_cutoff_sample_id": (
                            int(speech_tracker.endpoint_source_cutoff_sample_id)
                            if deferred_endpoint is not None
                            and speech_tracker.endpoint_source_cutoff_sample_id is not None
                            else None
                        ),
                    }
                    logger.info(
                        "[realtime %s] deferred native SPEAK waiting endpoint onset=%d endpoint=%s",
                        sess.session_id[:8],
                        int(onset_frame_id),
                        pending_speak.get("endpoint_frame_id"),
                    )
            with policy_causality_lock:
                response_audio_admission.update_input_frame_locked(int(frame.frame_id))
            with sess._generation_lock:
                policy = sess.turn_policy
                policy_revision = sess._policy_revision
                turn_generation_id = int(sess.session_generation_id)
                if policy is not None:
                    sess.append_policy_audio(frame.frame_id, frame.pcm)
                    if (
                        speech_endpoint
                        and speech_tracker.run_onset_frame_id is not None
                        and speech_tracker.endpoint_source_cutoff_sample_id is not None
                    ):
                        sess.pin_completed_utterance_audio(
                            int(speech_tracker.run_onset_frame_id),
                            int(frame.frame_id),
                            source_cutoff_sample_id=int(
                                speech_tracker.endpoint_source_cutoff_sample_id
                            ),
                            generation_id=int(sess.session_generation_id),
                        )
            native_user_previous_pcm = None
            native_user_previous_frame_id = None
            if (
                isinstance(policy, GemmaNativeControl6Policy)
                and policy.phase_profile == "native1540"
            ):
                (
                    native_user_previous_pcm,
                    native_user_previous_frame_id,
                ) = sess.observe_native_control6_user_pcm(
                    int(frame.frame_id), np.asarray(frame.pcm, dtype=np.float32)
                )
            prepared = {
                "new_voice_onset": bool(new_voice_onset),
                "speech_endpoint": bool(speech_endpoint),
                "policy": policy,
                "policy_revision": int(policy_revision),
                "turn_generation_id": turn_generation_id,
                "audio_appended": policy is not None,
                "native_user_previous_pcm": native_user_previous_pcm,
                "native_user_previous_frame_id": native_user_previous_frame_id,
            }
            tick._policy_tick_prepared = prepared
            return prepared

        def _shared_native_policy_context(frame: Any, tick: Any) -> Any | None:
            prepared = _prepare_policy_tick_state(frame, tick)
            policy = prepared["policy"]
            if not isinstance(policy, GemmaNativeControl6Policy):
                return None
            from duplex_kernel.gemma_native_control6_batch_policy import PolicyPreHeadContext
            from duplex_kernel.native_control6_phase import (
                native_control6_native1540_phase_t,
                native_control6_phase_t,
            )

            with sess._generation_lock:
                turn_generation_id = int(prepared["turn_generation_id"])
                pending_onset = sess.pending_user_onset_frame_id
                last_answered_onset = sess.last_answered_onset_frame_id
                response_pending = bool(sess.policy_response_pending)
                active_response = bool(sess.active_response)
                control_state = sess.control_state
                speaker_segment_id = sess.speaker_segment_id
            speaker_active = _native_control6_speaker_active(
                active_response=active_response,
                control_state=control_state,
                pending_pcm_playback=bool(_pending_pcm_response_branches(
                    playback_ack_branches,
                    registry_lock=playback_ack_branches_lock,
                )),
            )
            user_turn_pending = _user_turn_is_unanswered(
                pending_onset, last_answered_onset
            )
            peer_turn_active = bool(speech_tracker.voice_run_active)
            legal_actions = native_control6_legal_actions(
                speaker_active=speaker_active,
                user_turn_pending=user_turn_pending,
                peer_turn_active=peer_turn_active,
            )
            frame_id = int(frame.frame_id)
            visible_assistant = {
                int(item.frame_id): item
                for item in sess.assistant_reference_buffer.visible_frames_for(
                    listener_frame_id=frame_id
                )
            }
            assistant_prior = visible_assistant.get(frame_id - 1)
            assistant_preceding = visible_assistant.get(frame_id - 2)
            if policy.phase_profile == "native1540":
                phase_t = native_control6_native1540_phase_t(
                    listener_frame_id=frame_id,
                    user_pcm_current=frame.pcm,
                    user_pcm_previous=prepared["native_user_previous_pcm"],
                    user_previous_frame_id=prepared["native_user_previous_frame_id"],
                    assistant_pcm_prior=(None if assistant_prior is None else assistant_prior.pcm),
                    assistant_prior_frame_id=(None if assistant_prior is None else assistant_prior.frame_id),
                )
            else:
                phase_t = native_control6_phase_t(
                    listener_frame_id=frame_id,
                    user_pcm_current=frame.pcm,
                    assistant_pcm_prior=(None if assistant_prior is None else assistant_prior.pcm),
                    assistant_prior_frame_id=(None if assistant_prior is None else assistant_prior.frame_id),
                    assistant_pcm_preceding=(
                        None if assistant_preceding is None else assistant_preceding.pcm
                    ),
                    assistant_preceding_frame_id=(
                        None if assistant_preceding is None else assistant_preceding.frame_id
                    ),
                )
            no_input_tick = bool(
                speaker_active
                and np.asarray(frame.pcm).shape == (POLICY_FRAME_SAMPLES,)
                and np.count_nonzero(frame.pcm) == 0
            )
            state = PolicyInput(
                session_id=sess.session_id,
                frame_id=frame_id,
                user_pcm=frame.pcm,
                speaker_active=speaker_active,
                generation_id=turn_generation_id,
                peer_turn_active=peer_turn_active,
                speaker_segment_id=speaker_segment_id,
                legal_actions=tuple(legal_actions),
                native_control_phase_t=phase_t,
                native_control_assistant_pcm_prior=(
                    None if assistant_prior is None else assistant_prior.pcm
                ),
                native_control_assistant_prior_frame_id=(
                    None if assistant_prior is None else int(assistant_prior.frame_id)
                ),
                native_control_no_input_tick=no_input_tick,
            )
            binding = shared_listener_bindings.get(sess.session_id)
            if binding is None:
                raise RuntimeError("native policy snapshot has no shared listener registration")
            return PolicyPreHeadContext(
                policy=policy,
                policy_input=state,
                tick_id=(sess.session_id, turn_generation_id, frame_id),
                captured_at_ns=time.monotonic_ns(),
                listener_generation_id=int(binding["generation"]),
            )

        def scheduler_policy(frame: Any, tick: Any) -> None:
            """Policy callback invoked only by DuplexScheduler's tick thread."""
            nonlocal pending_speak
            nonlocal peer_audio_event_key
            nonlocal latest_policy_frame_id, replan_edge_latched, replan_non_event_ticks
            nonlocal replan_edge_generation_id
            nonlocal replan_handoff_pending
            nonlocal partial_last_snapshot_frame
            if native_listener_output_faulted.is_set():
                return
            if frame is None:
                return
            prepared = getattr(tick, "_policy_tick_prepared", None)
            if not isinstance(prepared, dict):
                prepared = _prepare_policy_tick_state(frame, tick)
            new_voice_onset = bool(prepared["new_voice_onset"])
            speech_endpoint = bool(prepared["speech_endpoint"])
            policy = prepared["policy"]
            policy_revision = int(prepared["policy_revision"])
            if policy is None:
                return
            with sess._generation_lock:
                if (
                    sess._policy_revision != policy_revision
                    or sess.turn_policy is not policy
                    or sess.session_generation_id != int(prepared["turn_generation_id"])
                ):
                    # The immutable pre-head context was captured before the
                    # worker call; a session mutation after enqueue must not
                    # make the scheduler run policy against a different state.
                    sess.stale_session_mutations += 1
                    return
            if not bool(prepared.get("audio_appended")):
                with sess._generation_lock:
                    sess.append_policy_audio(frame.frame_id, frame.pcm)
                    if (
                        speech_endpoint
                        and speech_tracker.run_onset_frame_id is not None
                        and speech_tracker.endpoint_source_cutoff_sample_id is not None
                    ):
                        sess.pin_completed_utterance_audio(
                            int(speech_tracker.run_onset_frame_id),
                            int(frame.frame_id),
                            source_cutoff_sample_id=int(
                                speech_tracker.endpoint_source_cutoff_sample_id
                            ),
                            generation_id=int(sess.session_generation_id),
                        )
            acoustic_user_history = None
            acoustic_assistant_reference = None
            if (
                isinstance(policy, DuplexControl6Policy)
                and policy.requires_acoustic_phase
            ):
                current_listener_output = sess.last_listener_output
                current_frame_id = int(frame.frame_id)
                query_reused = bool(
                    current_listener_output is not None
                    and getattr(current_listener_output, "silence_skipped", False)
                    and int(getattr(current_listener_output, "frame_id", -1)) == current_frame_id
                    and int(getattr(current_listener_output, "source_frame_id", -1))
                    == int(sess.listener_audio_frame_id if sess.listener_audio_frame_id is not None else -2)
                )
                with sess.listener_state._lock:
                    observed_user_frames = tuple(sess.listener_state.user_pcm_history)
                if (
                    current_listener_output is None
                    or int(getattr(current_listener_output, "frame_id", -1)) != current_frame_id
                    or sess.listener_audio_queries is None
                    or (
                        not query_reused
                        and sess.listener_audio_frame_id != current_frame_id
                    )
                    or (
                        query_reused
                        and sess.listener_audio_frame_id is not None
                        and sess.listener_audio_frame_id >= current_frame_id
                    )
                    or not observed_user_frames
                    or any(np.asarray(value).shape != (POLICY_FRAME_SAMPLES,) for value in observed_user_frames)
                    or not np.array_equal(np.asarray(observed_user_frames[-1]), np.asarray(frame.pcm))
                ):
                    logger.debug(
                        "[realtime %s] skip acoustic Control6 frame=%d: causal PCM snapshot mismatch",
                        sess.session_id[:8],
                        int(frame.frame_id),
                    )
                    return
                # The listener step has appended this tick to its rolling
                # history. Reconstruct the head input from at most four
                # prior observed frames plus the scheduler's current PCM so
                # the 400 ms window is explicitly anchored at frame t.
                prior_user_frames = observed_user_frames[-5:-1]
                acoustic_user_history = np.concatenate((
                    *(np.asarray(value, dtype=np.float32) for value in prior_user_frames),
                    np.asarray(frame.pcm, dtype=np.float32),
                )).copy()
                reference = sess.assistant_reference_buffer.visible_for(
                    listener_frame_id=current_frame_id
                )
                acoustic_assistant_reference = (
                    None if reference is None else np.asarray(reference.pcm, dtype=np.float32).copy()
                )
            legal_actions: tuple[str, ...] | None = None
            pending_onset: int | None = None
            last_answered_onset: int | None = None
            response_pending_for_policy = False
            active_response_for_policy = False
            native_speaker_active_for_policy = False
            user_turn_pending = False
            pending_plan = False
            peer_turn_active = bool(speech_tracker.voice_run_active)
            backchannel_available = False
            if is_controller6_policy(policy):
                with sess._generation_lock:
                    pending_onset = sess.pending_user_onset_frame_id
                    last_answered_onset = sess.last_answered_onset_frame_id
                    response_pending_for_policy = bool(sess.policy_response_pending)
                    active_response_for_policy = bool(sess.active_response)
                    control_state_for_policy = sess.control_state
                if isinstance(policy, GemmaNativeControl6Policy):
                    native_speaker_active_for_policy = _native_control6_speaker_active(
                        active_response=active_response_for_policy,
                        control_state=control_state_for_policy,
                        pending_pcm_playback=bool(_pending_pcm_response_branches(
                            playback_ack_branches,
                            registry_lock=playback_ack_branches_lock,
                        )),
                    )
                user_turn_pending = _user_turn_is_unanswered(
                    pending_onset, last_answered_onset
                )
                pending_plan = bool(
                    pending_speak is not None
                    and pending_onset is not None
                    and int(pending_speak.get("voice_run_onset_frame_id", -1))
                    == int(pending_onset)
                )
                backchannel_available = bool(
                    peer_audio_event_key is not None
                    and not backchannel_attempt_latch.has_attempted(peer_audio_event_key)
                    and not active_response_for_policy
                    and not response_pending_for_policy
                )
                if isinstance(policy, GemmaNativeControl6Policy):
                    legal_actions = native_control6_legal_actions(
                        speaker_active=native_speaker_active_for_policy,
                        user_turn_pending=user_turn_pending,
                        peer_turn_active=peer_turn_active,
                    )
                else:
                    legal_actions = legal_control_actions(
                        active_response=active_response_for_policy,
                        response_pending=response_pending_for_policy,
                        user_turn_pending=user_turn_pending,
                        peer_turn_active=peer_turn_active,
                        backchannel_available=backchannel_available,
                        renderer_plan_pending=bool(
                            response_pending_for_policy or pending_plan
                        ),
                    )
            native_control_hidden = None
            native_control_hidden_frame_id = None
            native_control_hidden_at_ns = None
            native_control_phase = None
            native_control_assistant_pcm_prior = None
            native_control_assistant_prior_frame_id = None
            native_control_no_input_tick = False
            native_control6_raw_scores = None
            native_listener_generation = None
            if isinstance(policy, GemmaNativeControl6Policy):
                from duplex_kernel.native_control6_phase import (
                    native_control6_native1540_phase_t,
                    native_control6_phase_t,
                )

                current_frame_id = int(frame.frame_id)
                visible_assistant = {
                    int(item.frame_id): item
                    for item in sess.assistant_reference_buffer.visible_frames_for(
                        listener_frame_id=current_frame_id
                    )
                }
                assistant_prior = visible_assistant.get(current_frame_id - 1)
                assistant_preceding = visible_assistant.get(current_frame_id - 2)
                native_control_assistant_pcm_prior = (
                    None if assistant_prior is None else assistant_prior.pcm
                )
                native_control_assistant_prior_frame_id = (
                    None if assistant_prior is None else int(assistant_prior.frame_id)
                )
                if policy.phase_profile == "native1540":
                    native_control_phase = native_control6_native1540_phase_t(
                        listener_frame_id=current_frame_id,
                        user_pcm_current=frame.pcm,
                        user_pcm_previous=prepared["native_user_previous_pcm"],
                        user_previous_frame_id=prepared["native_user_previous_frame_id"],
                        assistant_pcm_prior=(None if assistant_prior is None else assistant_prior.pcm),
                        assistant_prior_frame_id=(None if assistant_prior is None else assistant_prior.frame_id),
                    )
                else:
                    native_control_phase = native_control6_phase_t(
                        listener_frame_id=current_frame_id,
                        user_pcm_current=frame.pcm,
                        assistant_pcm_prior=(None if assistant_prior is None else assistant_prior.pcm),
                        assistant_prior_frame_id=(None if assistant_prior is None else assistant_prior.frame_id),
                        assistant_pcm_preceding=(
                            None if assistant_preceding is None else assistant_preceding.pcm
                        ),
                        assistant_preceding_frame_id=(
                            None if assistant_preceding is None else assistant_preceding.frame_id
                        ),
                    )
                with sess._generation_lock:
                    native_output = sess.last_listener_output
                source_frame = _native_control6_listener_source_frame_id(native_output)
                if (
                    native_output is not None
                    and getattr(native_output, "frame_id", None) == int(frame.frame_id)
                ):
                    if _native_control6_listener_output_is_fresh_for_frame(
                        native_output,
                        int(frame.frame_id),
                        source_frame_id=source_frame,
                    ):
                        native_control_hidden = getattr(native_output, "gemma_hidden_t", None)
                        native_control_hidden_frame_id = int(source_frame)
                        available_at = getattr(native_output, "source_available_at_ns", None)
                        native_control_hidden_at_ns = int(available_at) if available_at is not None else None
                        candidate_scores = getattr(
                            native_output, "native_control6_raw_scores", None
                        )
                        if candidate_scores is not None:
                            if (
                                getattr(candidate_scores, "session_id", None) != sess.session_id
                                or getattr(candidate_scores, "frame_id", None) != int(frame.frame_id)
                                or getattr(candidate_scores, "session_generation", None)
                                != getattr(native_output, "listener_generation_id", None)
                                or getattr(candidate_scores, "state_epoch_id", None)
                                != getattr(native_output, "listener_state_epoch_id", None)
                                or getattr(candidate_scores, "listener_state_version", None)
                                != int(native_output.listener_state_version) - 1
                                or getattr(candidate_scores, "sequence_position", None)
                                != int(getattr(native_output, "listener_sequence_position", -1)) - 1
                                or getattr(candidate_scores, "cache_position", None)
                                != int(getattr(native_output, "listener_cache_position", -1)) - 2
                                or getattr(candidate_scores, "checkpoint_path", None)
                                != policy.checkpoint_path
                            ):
                                raise RuntimeError(
                                    "batched native Control6 score provenance does not match the fresh listener row"
                                )
                            native_control6_raw_scores = candidate_scores
                            native_listener_generation = int(
                                native_output.listener_generation_id
                            )
                    elif (
                        bool(getattr(native_output, "silence_skipped", False))
                        and source_frame is not None
                        and source_frame < int(frame.frame_id)
                    ):
                        current_pcm = np.asarray(frame.pcm)
                        reference = sess.assistant_reference_buffer.visible_for(listener_frame_id=int(frame.frame_id))
                        reference_pcm = None if reference is None else np.asarray(reference.pcm)
                        native_control_no_input_tick = bool(
                            current_pcm.shape == (POLICY_FRAME_SAMPLES,)
                            and reference_pcm is not None
                            and reference_pcm.shape == (POLICY_FRAME_SAMPLES,)
                            and np.count_nonzero(current_pcm) == 0
                            and np.count_nonzero(reference_pcm) == 0
                        )
                native_control_hidden, exact_zero_active_tick = (
                    _native_control6_policy_hidden_for_tick(
                        user_pcm=frame.pcm,
                        speaker_active=native_speaker_active_for_policy,
                        listener_hidden=native_control_hidden,
                    )
                )
                if exact_zero_active_tick:
                    native_control_hidden_frame_id = None
                    native_control_hidden_at_ns = None
                    native_control_no_input_tick = True
            policy_input = PolicyInput(
                session_id=sess.session_id,
                frame_id=frame.frame_id,
                user_pcm=frame.pcm,
                speaker_active=(
                    native_speaker_active_for_policy
                    if isinstance(policy, GemmaNativeControl6Policy)
                    else (
                        sess.active_response
                        or sess.policy_response_pending
                        or sess.control_state in {
                            ControlState.SPEAKING,
                            ControlState.BACKCHANNELING,
                            ControlState.STOPPING,
                        }
                    )
                ),
                generation_id=sess.session_generation_id,
                peer_turn_active=peer_turn_active,
                speaker_segment_id=sess.speaker_segment_id,
                listener_hidden=(
                    sess.listener_hidden
                    if _listener_v9_stream_for_engine(engine) is not None
                    and getattr(sess.listener_hidden, "shape", ())[-1:] == (2048,)
                    else None
                ),
                listener_audio_queries=(
                    sess.listener_audio_queries if isinstance(policy, DuplexControl6Policy) else None
                ),
                listener_frame_id=(
                    sess.listener_audio_frame_id if isinstance(policy, DuplexControl6Policy) else None
                ),
                listener_query_reused=bool(
                    isinstance(policy, DuplexControl6Policy)
                    and sess.last_listener_output is not None
                    and getattr(sess.last_listener_output, "silence_skipped", False)
                    and int(getattr(sess.last_listener_output, "frame_id", -1)) == int(frame.frame_id)
                ),
                listener_hidden_frame_id=sess.listener_hidden_frame_id,
                listener_hidden_at_ns=sess.listener_hidden_at_ns,
                user_pcm_history=acoustic_user_history,
                assistant_reference_pcm=acoustic_assistant_reference,
                legal_actions=legal_actions,
                native_control_hidden=native_control_hidden,
                native_control_hidden_frame_id=native_control_hidden_frame_id,
                native_control_hidden_at_ns=native_control_hidden_at_ns,
                native_control_phase_t=native_control_phase,
                native_control_assistant_pcm_prior=native_control_assistant_pcm_prior,
                native_control_assistant_prior_frame_id=native_control_assistant_prior_frame_id,
                native_control_no_input_tick=native_control_no_input_tick,
                native_control6_raw_scores=native_control6_raw_scores,
                native_listener_generation=native_listener_generation,
            )
            prehead_context = getattr(tick, "_policy_prehead_context", None)
            if shared_listener_microbatch_enabled and isinstance(
                policy, GemmaNativeControl6Policy
            ):
                from duplex_kernel.gemma_native_control6_batch_policy import PolicyPreHeadContext

                if not isinstance(prehead_context, PolicyPreHeadContext):
                    raise RuntimeError(
                        "shared native Control6 tick reached policy callback without its immutable pre-head context"
                    )
                binding = shared_listener_bindings.get(sess.session_id)
                if binding is None:
                    raise RuntimeError("shared native Control6 decision lost its listener registration")
                expected_tick = (
                    sess.session_id,
                    int(sess.session_generation_id),
                    int(frame.frame_id),
                )
                if (
                    prehead_context.policy is not policy
                    or prehead_context.tick_id != expected_tick
                    or int(prehead_context.listener_generation_id)
                    != int(binding.get("generation", -1))
                ):
                    raise RuntimeError(
                        "shared native Control6 pre-head context is stale or belongs to another listener generation"
                    )
                policy_input = prehead_context.policy_input
                legal_actions = tuple(policy_input.legal_actions or ())
                native_speaker_active_for_policy = bool(policy_input.speaker_active)
                native_control_no_input_tick = bool(policy_input.native_control_no_input_tick)
            with policy_causality_lock:
                if (
                    replan_edge_latched
                    and replan_edge_generation_id != int(policy_input.generation_id)
                ):
                    # A destructive cancel or other generation fence ends the
                    # edge's scope. Never carry a REPLAN latch into fresh work.
                    replan_edge_latched = False
                    replan_non_event_ticks = 0
                    replan_edge_generation_id = None
            if native_partial_worker is not None:
                if policy_input.speaker_active:
                    native_partial_worker.pause()
                else:
                    native_partial_worker.resume()
                if native_partial_asr_enabled.is_set():
                    onset_frame_id = speech_tracker.run_onset_frame_id
                    with transcript_state_lock:
                        utterance_id = transcript_state["utterance_id"]
                    if utterance_id is None and speech_tracker.voice_run_active and onset_frame_id is not None:
                        # The client may opt in after speech already began.
                        utterance_id, _ = begin_transcript_utterance(int(onset_frame_id))
                    snapshot_last_frame = (
                        int(speech_tracker.last_voiced_frame_id)
                        if speech_endpoint and speech_tracker.last_voiced_frame_id is not None
                        else int(frame.frame_id)
                    )
                    if utterance_id and onset_frame_id is not None and (
                        speech_endpoint
                        or should_request_partial_snapshot(
                            enabled=True,
                            voice_active=bool(speech_tracker.voice_run_active),
                            # Snapshot capture is cheap CPU/buffer work. The
                            # worker itself remains paused while response GPU
                            # work is active and keeps only the newest audio.
                            response_busy=False,
                            frame_id=int(frame.frame_id),
                            onset_frame_id=int(onset_frame_id),
                            last_snapshot_frame_id=partial_last_snapshot_frame,
                        )
                    ):
                        max_snapshot_frames = max(
                            1,
                            (USER_SAMPLE_RATE * 30) // POLICY_FRAME_SAMPLES,
                        )
                        first_snapshot_frame = max(
                            int(onset_frame_id),
                            snapshot_last_frame - max_snapshot_frames + 1,
                        )
                        snapshot_pcm = sess.snapshot_policy_audio_range(
                            first_snapshot_frame,
                            snapshot_last_frame,
                        )
                        if snapshot_pcm.size:
                            with transcript_state_lock:
                                snapshot_generation_id = int(policy_input.generation_id)
                                snapshot_epoch_id = int(transcript_state["epoch"])
                            submitted = native_partial_worker.submit(NativeTranscriptSnapshot(
                                utterance_id=str(utterance_id),
                                pcm=snapshot_pcm,
                                first_frame_id=first_snapshot_frame,
                                last_frame_id=snapshot_last_frame,
                                generation_id=snapshot_generation_id,
                                final=bool(speech_endpoint),
                                epoch_id=snapshot_epoch_id,
                            ))
                            if submitted:
                                partial_last_snapshot_frame = int(frame.frame_id)
            if speech_endpoint and speech_tracker.run_onset_frame_id is not None:
                if (
                    pending_speak is not None
                    and pending_speak.get("voice_run_onset_frame_id")
                    == int(speech_tracker.run_onset_frame_id)
                ):
                    pending_speak["endpoint_frame_id"] = int(frame.frame_id)
                    pending_speak["endpoint_source_cutoff_sample_id"] = (
                        speech_tracker.endpoint_source_cutoff_sample_id
                    )
                # Snapshot the correction through its exact endpoint frame in
                # the scheduler thread. Later microphone frames cannot leak
                # into the pending REPLAN, even if the event loop is delayed.
                with replan_endpoint_lock:
                    matching = [
                        generation_id
                        for generation_id, (onset_frame_id, _waiter) in replan_endpoint_waiters.items()
                        if onset_frame_id == int(speech_tracker.run_onset_frame_id)
                    ]
                for generation_id in matching:
                    direct_replan_context: dict[str, Any] | None = {
                        "endpoint_frame_id": int(frame.frame_id),
                    }
                    if direct_audio_candidate:
                        try:
                            direct_replan_context = _snapshot_direct_policy_audio(
                                sess,
                                action=ControlAction.REPLAN,
                                first_frame_id=int(speech_tracker.run_onset_frame_id),
                                last_frame_id=int(frame.frame_id),
                            )
                        except ValueError as exc:
                            direct_replan_context = {
                                "_direct_audio_error": str(exc)[:200],
                                "endpoint_frame_id": int(frame.frame_id),
                            }
                    if direct_replan_context is None:
                        direct_replan_context = {}
                    committed_endpoint_audio = sess.take_policy_audio_through_frame(int(frame.frame_id))
                    sess.release_completed_utterance_audio(
                        int(speech_tracker.run_onset_frame_id),
                        int(frame.frame_id),
                        source_cutoff_sample_id=speech_tracker.endpoint_source_cutoff_sample_id,
                        generation_id=int(sess.session_generation_id),
                    )
                    speech_tracker.consume_endpoint()
                    with replan_endpoint_lock:
                        waiting = replan_endpoint_waiters.pop(generation_id, None)
                        if waiting is not None:
                            replan_audio_snapshots[generation_id] = committed_endpoint_audio
                            if direct_replan_context is not None:
                                replan_audio_contexts[generation_id] = direct_replan_context
                            waiting[1].set()
            if isinstance(policy, DuplexControl6Policy) and (
                policy_input.listener_audio_queries is None
                or policy_input.listener_frame_id is None
                or (
                    int(policy_input.listener_frame_id) != int(policy_input.frame_id)
                    and not policy_input.listener_query_reused
                )
            ):
                logger.debug(
                    "[realtime %s] skip Control6 frame=%d: missing/fresh listener query (listener_frame=%s)",
                    sess.session_id[:8],
                    int(policy_input.frame_id),
                    policy_input.listener_frame_id,
                )
                return
            # Policy/model work can be synchronous and slower than an 80 ms
            # tick. Serialize it independently so mic admission does not wait
            # on the session generation lock for the duration of this call.
            native_output_health_result: dict[str, Any] = {
                "checked": False,
                "status": "not_scored",
            }
            decision = None
            native_policy_error: Exception | None = None
            worker_native_hidden_valid = False
            worker_native_output = None
            policy_started = time.perf_counter()
            try:
                try:
                    if (
                        shared_listener_microbatch_enabled
                        and isinstance(policy, GemmaNativeControl6Policy)
                    ):
                        worker_native_output = getattr(tick, "microbatch_output", None)
                        if worker_native_output is None:
                            raise RuntimeError(
                                "shared native Control6 decision has no committed worker output"
                            )
                        decision = getattr(worker_native_output, "policy_decision", None)
                        if decision is None:
                            raise RuntimeError(
                                "shared native Control6 worker returned no policy decision"
                            )
                        diagnostics = decision.diagnostics
                        provenance = (
                            diagnostics.get("native_output_provenance")
                            if isinstance(diagnostics, dict)
                            else None
                        )
                        binding = shared_listener_bindings.get(sess.session_id)
                        expected_listener_generation = (
                            int(binding.get("generation", -1)) if binding is not None else -1
                        )
                        if (
                            binding is None
                            or decision.session != sess.session_id
                            or decision.generation_id != policy_input.generation_id
                            or decision.frame_id != int(policy_input.frame_id)
                            or decision.listener_frame_id != int(policy_input.frame_id)
                            or not isinstance(provenance, dict)
                            or provenance.get("canonical_session_id") != sess.session_id
                            or provenance.get("listener_generation_id") != expected_listener_generation
                            or provenance.get("turn_generation_id") != policy_input.generation_id
                            or provenance.get("state_epoch_id") != str(sess.listener_state.state_epoch_id)
                            or provenance.get("frame_id") != int(policy_input.frame_id)
                            or provenance.get("listener_state_version")
                            != int(worker_native_output.listener_state_version) - 1
                        ):
                            raise RuntimeError(
                                "shared native Control6 decision provenance differs from its active turn/listener frame"
                            )
                        view = getattr(worker_native_output, "decision_view", None)
                        selected_health = (
                            getattr(view, "native_output_health", None)
                            if view is not None
                            else getattr(worker_native_output, "native_output_health", None)
                        )
                        if view is not None:
                            source_output = getattr(view, "native_output", None)
                            source_manifest = getattr(view, "manifest", None)
                            source_row = (
                                source_manifest.rows[0]
                                if source_manifest is not None
                                and getattr(source_manifest, "rows", None)
                                else None
                            )
                            selected_manifest_id = getattr(source_manifest, "manifest_id", None)
                            selected_output_generation = getattr(source_output, "generation", None)
                        else:
                            source_output = None
                            source_manifest = None
                            source_row = getattr(selected_health, "row", None)
                            selected_manifest_id = provenance.get("manifest_id")
                            selected_output_generation = provenance.get("output_generation")
                        if (
                            selected_health is None
                            or not bool(getattr(selected_health, "checked", False))
                            or bool(getattr(selected_health, "nonfinite", True))
                            or getattr(selected_health, "status", None) != "healthy"
                            or not isinstance(selected_manifest_id, str)
                            or not selected_manifest_id
                            or isinstance(selected_output_generation, bool)
                            or not isinstance(selected_output_generation, int)
                            or getattr(selected_health, "generation", None)
                            != selected_output_generation
                            or getattr(selected_health, "row", None) != source_row
                            or source_row is None
                            or source_row.session_generation != expected_listener_generation
                            or source_row.state_epoch_id != str(sess.listener_state.state_epoch_id)
                            or source_row.frame_id != int(policy_input.frame_id)
                            or source_row.listener_state_version
                            != int(worker_native_output.listener_state_version) - 1
                        ):
                            raise RuntimeError(
                                "selected native Control6 decision row lacks matching healthy output provenance"
                            )
                        if view is not None:
                            view_source_row = getattr(view, "source_row", None)
                            fork_row = getattr(view, "fork_row", None)
                            if (
                                view_source_row is None
                                or fork_row is None
                                or view_source_row.session_id != sess.session_id
                                or view_source_row.session_generation != expected_listener_generation
                                or view_source_row.state_epoch_id != str(sess.listener_state.state_epoch_id)
                                or view_source_row.frame_id != int(policy_input.frame_id)
                                or source_row.session_id != fork_row.session_id
                                or not source_row.session_id.startswith("__q2view__")
                                or source_row.logical_q_len != 2
                                or not bool(worker_native_output.queue_audio_for_next_turn)
                            ):
                                raise RuntimeError(
                                    "temporary Q2 decision view is not bound to the current canonical speaking frame"
                                )
                        native_output_health_result = {
                            "checked": True,
                            "status": "healthy",
                            "manifest_id": selected_manifest_id,
                            "output_generation": selected_output_generation,
                            "native_session_id": source_row.session_id,
                            "frame_id": source_row.frame_id,
                            "listener_state_version": source_row.listener_state_version,
                        }
                        native_control_hidden = (
                            getattr(view, "gemma_hidden_t", None)
                            if view is not None
                            else getattr(worker_native_output, "gemma_hidden_t", None)
                        )
                        native_control_hidden_frame_id = int(policy_input.frame_id)
                        native_control_hidden_at_ns = time.monotonic_ns()
                        worker_native_hidden_valid = not bool(
                            policy_input.native_control_no_input_tick
                        )
                    else:
                        decision = sess.run_turn_policy(
                            policy, policy_revision, policy_input
                        )
                except (RuntimeError, ValueError) as exc:
                    if not (
                        isinstance(policy, GemmaNativeControl6Policy)
                        or getattr(policy, "policy_id", None) == "gemma_native_control6"
                    ):
                        raise
                    native_policy_error = exc
            finally:
                record_scheduler_stage(
                    "policy_wall_ms", (time.perf_counter() - policy_started) * 1000.0
                )
            fresh_native_hidden = bool(
                worker_native_hidden_valid
                or (
                    policy_input.native_control_hidden is not None
                    and policy_input.native_control_hidden_frame_id is not None
                    and int(policy_input.native_control_hidden_frame_id)
                    == int(policy_input.frame_id)
                )
            )
            if isinstance(policy, GemmaNativeControl6Policy) and fresh_native_hidden:
                if shared_listener_microbatch_enabled:
                    if worker_native_output is None:
                        binding = shared_listener_bindings.get(sess.session_id)
                        native_output_health_result = native_control6_validate_listener_output_health(
                            native_output,
                            session_id=sess.session_id,
                            generation_id=int(binding.get("generation", -1)) if binding else -1,
                            state_epoch_id=str(sess.listener_state.state_epoch_id),
                            frame_id=int(policy_input.frame_id),
                        )
                else:
                    native_output_health_result = native_control6_verify_scored_output_health(
                        getattr(sess.listener_state, "gemma_lm_native", None),
                        native_policy=True,
                        fresh_hidden=True,
                    )
                if native_output_health_result.get("status") != "healthy":
                    fail_native_listener_output_health(
                        int(policy_input.frame_id), native_output_health_result
                    )
                    return
            if (
                decision is not None
                and isinstance(policy, GemmaNativeControl6Policy)
                and not _native_control6_decision_has_valid_input(
                    fresh_hidden=fresh_native_hidden,
                    no_input_tick=bool(policy_input.native_control_no_input_tick),
                    speaker_active=bool(policy_input.speaker_active),
                    action=getattr(decision, "action", None),
                    legal_actions=legal_actions,
                )
            ):
                native_policy_error = RuntimeError(
                    "native Control6 action lacks a fresh listener hidden for this frame"
                )
                decision = None
            if native_policy_error is not None:
                exc = native_policy_error
                logger.warning("[realtime %s] native Control6 action rejected frame=%d: %s",
                               sess.session_id[:8], int(policy_input.frame_id), exc)
                try:
                    scheduler_events.put_nowait({"type": "turn_policy.action_rejected",
                        "policy_id": "gemma_native_control6", "frame_id": int(policy_input.frame_id),
                        "turn_policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
                        "native_control6_class_match": isinstance(policy, GemmaNativeControl6Policy),
                        "native_control6_step_called": True,
                        "native_control6_head_scored": False,
                        "native_control_hidden_supplied": fresh_native_hidden,
                        "native_control_hidden_frame_id": native_control_hidden_frame_id,
                        "native_control_no_input_tick": bool(policy_input.native_control_no_input_tick),
                        "legal_actions": list(legal_actions or ()),
                        "user_turn_pending": bool(user_turn_pending),
                        "pending_user_onset_frame_id": pending_onset,
                        "last_answered_onset_frame_id": last_answered_onset,
                        "reason": str(exc)[:240]})
                except queue.Full:
                    pass
                return
            if decision is None:
                return
            native_control6_class_match = isinstance(policy, GemmaNativeControl6Policy)
            native_control6_step_called = bool(
                native_control6_class_match
                or (
                    isinstance(decision.diagnostics, dict)
                    and decision.diagnostics.get("policy_id") == "gemma_native_control6"
                )
            )
            native_control6_head_scored = bool(
                native_control6_step_called
                and fresh_native_hidden
                and not (
                    isinstance(decision.diagnostics, dict)
                    and decision.diagnostics.get("no_input_tick") is True
                )
            )
            with sess._generation_lock:
                if (
                    policy_revision != sess._policy_revision
                    or policy is not sess.turn_policy
                    or sess.session_generation_id != policy_input.generation_id
                ):
                    sess.stale_session_mutations += 1
                    return
                if (
                    decision.frame_id != policy_input.frame_id
                    or (
                        decision.generation_id is not None
                        and decision.generation_id != policy_input.generation_id
                    )
                ):
                    sess.stale_session_mutations += 1
                    return
                policy_decision_at_ns = time.monotonic_ns()
                tick.policy_decision_at_ns = policy_decision_at_ns
                trace = {
                    "action": decision.action.value,
                    "confidence": float(decision.confidence),
                    "turn_policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
                    "turn_policy_id": str(getattr(policy, "policy_id", type(policy).__name__)),
                    "native_control6_class_match": native_control6_class_match,
                    "native_control6_step_called": native_control6_step_called,
                    "native_control6_head_scored": native_control6_head_scored,
                    "native_control_hidden_supplied": fresh_native_hidden,
                    "native_control_hidden_frame_id": native_control_hidden_frame_id,
                    "frame_id": int(decision.frame_id),
                    "policy_decision_at_ns": policy_decision_at_ns,
                    "listener_hidden_frame_id": policy_input.listener_hidden_frame_id,
                    "listener_hidden_at_ns": policy_input.listener_hidden_at_ns,
                    "listener_hidden_age_frames": (
                        int(policy_input.frame_id) - int(policy_input.listener_hidden_frame_id)
                        if policy_input.listener_hidden_frame_id is not None else None
                    ),
                    "listener_hidden_age_ms": (
                        round(max(0, time.monotonic_ns() - int(policy_input.listener_hidden_at_ns)) / 1_000_000, 3)
                        if policy_input.listener_hidden_at_ns is not None else None
                    ),
                    "listener_query_reused": bool(policy_input.listener_query_reused),
                    "listener_query_age_frames": (
                        int(policy_input.frame_id) - int(policy_input.listener_frame_id)
                        if policy_input.listener_frame_id is not None else None
                    ),
                    "listener_frame_id": int(
                        decision.listener_frame_id
                        if decision.listener_frame_id is not None
                        else (
                            policy_input.listener_frame_id
                            if policy_input.listener_frame_id is not None
                            else policy_input.frame_id
                        )
                    ),
                    "session": str(decision.session or sess.session_id),
                    "source": str(decision.source),
                }
                trace.update(scheduler_tick_trace_metadata(tick))
                if is_controller6_policy(policy) and isinstance(decision.diagnostics, dict):
                    for field in (
                        "legal_actions",
                        "legal_action_probabilities",
                        "native_action",
                        "native_action_probability",
                        "native_action_probabilities",
                        "native_action_order",
                        "native_action_base_logits",
                        "native_action_logits",
                        "selected_action",
                        "selected_action_probability",
                        "base_action",
                        "policy_fallback_reason",
                        "checkpoint_path",
                        "hidden_contract",
                        "listener_hidden_frame_id",
                        "listener_hidden_at_ns",
                        "response_lora_applied",
                        "native_output_provenance",
                        "decision_view_fallback_reason",
                        "calibration_applied",
                        "calibration_margin",
                        "calibration_threshold",
                        "feature_contract_input_dim",
                        "acoustic_phase_features_used",
                        "listener_query_reused",
                        "listener_query_origin_frame_id",
                        "listener_query_age_frames",
                    ):
                        if field in decision.diagnostics:
                            trace[field] = decision.diagnostics[field]
                trace.update(_native_control6_selection_fields(
                    decision,
                    head_scored=native_control6_head_scored,
                ))
                if _NATIVE_CONTROL6_TRACE_ENABLED and isinstance(policy, GemmaNativeControl6Policy):
                    listener_state_for_trace = getattr(sess, "listener_state", None)
                    native_output_for_trace = locals().get("native_output")
                    _attach_native_control6_trace_diagnostic(
                        trace,
                        native_listener=getattr(listener_state_for_trace, "gemma_lm_native", None),
                        audio_soft=native_control_hidden,
                        q4_output=getattr(native_output_for_trace, "gemma_audio_queries_t", None),
                        prior_health_result=native_output_health_result,
                    )
                if is_controller6_policy(policy) and _NATIVE_CONTROL6_TRACE_ENABLED:
                    with sess._generation_lock:
                        history_ids = tuple(int(value) for value in sess.policy_audio_frame_ids)
                        history_samples = int(sess.policy_audio_samples)
                        reserved_onset = sess.reserved_user_onset_frame_id
                        current_generation = int(sess.session_generation_id)
                        current_revision = int(sess.semantic_context_revision)
                        listener_state = sess.listener_state
                        applied_revision = getattr(listener_state, "semantic_context_revision", None)
                        prefix_length = int(getattr(listener_state, "gemma_lm_prefix_len", 0))
                    prefix_metadata = getattr(engine, "native_control6_semantic_prefix_metadata", None)
                    prefix_metadata = prefix_metadata if isinstance(prefix_metadata, dict) else {}
                    trace["control6_context"] = {
                        "legal_actions": list(legal_actions or ()),
                        "user_turn_pending": bool(user_turn_pending),
                        "user_active": bool(speech_tracker.voice_run_active),
                        "voice_run_onset_frame_id": speech_tracker.run_onset_frame_id,
                        "speech_endpoint_frame_id": speech_tracker.endpoint_frame_id,
                        "last_voiced_frame_id": speech_tracker.last_voiced_frame_id,
                        "quiet_frames": int(speech_tracker.quiet_frames),
                        "new_voice_onset": bool(new_voice_onset),
                        "speech_endpoint": bool(speech_endpoint),
                        "peer_turn_active": bool(peer_turn_active),
                        "backchannel_available": bool(backchannel_available),
                        "pending_plan": bool(pending_plan),
                        "pending_user_onset_frame_id": pending_onset,
                        "reserved_user_onset_frame_id": reserved_onset,
                        "last_answered_onset_frame_id": last_answered_onset,
                        "response_generation_id": current_generation,
                        "response_pending": bool(response_pending_for_policy),
                        "active_response": bool(active_response_for_policy),
                        "fresh_native_hidden": policy_input.native_control_hidden is not None,
                        "native_hidden_frame_id": policy_input.native_control_hidden_frame_id,
                        "native_no_input_tick": bool(policy_input.native_control_no_input_tick),
                        "audio_history": {
                            "first_frame_id": history_ids[0] if history_ids else None,
                            "last_frame_id": history_ids[-1] if history_ids else None,
                            "frame_count": len(history_ids),
                            "sample_count": history_samples,
                            "contiguous": all(
                                right == left + 1 for left, right in zip(history_ids, history_ids[1:])
                            ),
                        },
                        "semantic_prefix": {
                            "schema_version": prefix_metadata.get("semantic_prefix_schema_version"),
                            "contract": prefix_metadata.get("semantic_prefix_contract"),
                            "length": prefix_length,
                            "capacity": prefix_metadata.get("semantic_prefix_capacity"),
                            "committed_history_revision": current_revision,
                            "applied_history_revision": applied_revision,
                            "revision_age": (
                                current_revision - int(applied_revision)
                                if applied_revision is not None else None
                            ),
                        },
                    }
                trace_decision = (
                    is_controller6_policy(policy)
                    or isinstance(policy, ScriptedPolicy)
                    or _NATIVE_CONTROL6_TRACE_ENABLED
                )

                def publish_decision(
                    *,
                    accepted: bool,
                    rejection_reason: str | None = None,
                    snapshot_error: BaseException | None = None,
                ) -> None:
                    if (
                        _NATIVE_CONTROL6_DEBUG_INPUT_DIRECTORY is not None
                        and isinstance(policy, GemmaNativeControl6Policy)
                        and fresh_native_hidden
                        and len(native_control6_debug_rows) < _NATIVE_CONTROL6_DEBUG_INPUT_MAX_FRAMES
                        and int(policy_input.frame_id) not in native_control6_debug_frame_ids
                    ):
                        try:
                            debug_row = _native_control6_debug_input_row(
                                policy_input=policy_input,
                                decision=decision,
                                accepted=accepted,
                                phase_profile=policy.phase_profile,
                            )
                            if debug_row is not None:
                                native_control6_debug_rows.append(debug_row)
                                native_control6_debug_frame_ids.add(int(policy_input.frame_id))
                        except Exception:
                            logger.exception(
                                "native Control6 debug input capture failed for frame=%d session=%s",
                                int(policy_input.frame_id), sess.session_id,
                            )
                    if (
                        is_controller6_policy(policy)
                        and direct_audio_candidate
                        and audio_history_archive_cpu_enabled
                        and speech_endpoint
                        and speech_tracker.run_onset_frame_id is not None
                        and not (accepted and decision.action == ControlAction.SPEAK)
                    ):
                        onset_id = int(speech_tracker.run_onset_frame_id)
                        with sess._generation_lock:
                            turn_already_answered = (
                                sess.last_answered_onset_frame_id is not None
                                and onset_id <= int(sess.last_answered_onset_frame_id)
                            )
                            turn_reserved = sess.reserved_user_onset_frame_id == onset_id
                            archive_base = [dict(message) for message in sess.history]
                        if not turn_already_answered and not turn_reserved:
                            endpoint_id = int(frame.frame_id)
                            endpoint_pcm = sess.snapshot_policy_audio_range(onset_id, endpoint_id)
                            if endpoint_pcm.size:
                                try:
                                    archive_id = sess.reserve_audio_history_archive(
                                        pcm=endpoint_pcm,
                                        first_frame_id=onset_id,
                                        last_frame_id=endpoint_id,
                                        base_messages=archive_base,
                                        source="post_response_asr",
                                        coverage="complete",
                                    )
                                    if archive_id is not None:
                                        archive_metric_id = f"archive_{sess.session_id[:8]}_{onset_id}"
                                        submitted = start_audio_history_archive(
                                            int(archive_id), archive_metric_id
                                        )
                                        logger.info(
                                            "[realtime %s] endpoint archive onset=%d endpoint=%d coverage=complete submitted=%s",
                                            sess.session_id[:8], onset_id, endpoint_id, submitted,
                                        )
                                except Exception:
                                    logger.exception(
                                        "[realtime %s] endpoint archive failed onset=%d",
                                        sess.session_id[:8], onset_id,
                                    )
                    if not trace_decision:
                        return
                    event = {"type": "duplex_control6.decision", **trace, "accepted": bool(accepted)}
                    event.update(_native_control6_event_annotations(
                        decision,
                        head_scored=native_control6_head_scored,
                        accepted=accepted,
                    ))
                    if snapshot_error is not None:
                        event.update(_direct_audio_snapshot_event_fields(snapshot_error))
                    if rejection_reason is not None:
                        event["rejection_reason"] = rejection_reason
                    scheduler_events.put_nowait(event)

                pending_pcm_branches = _pending_pcm_response_branches(
                    playback_ack_branches,
                    registry_lock=playback_ack_branches_lock,
                )
                active_for_causal_gate = bool(
                    sess.active_response
                    or sess.policy_response_pending
                    or pending_pcm_branches
                    or sess.control_state in {
                        ControlState.SPEAKING,
                        ControlState.BACKCHANNELING,
                        ControlState.STOPPING,
                    }
                )
                if (
                    decision.action == ControlAction.YIELD
                    and pending_speak is not None
                    and pending_speak.get("source") == _DEFERRED_NATIVE_SPEAK_SOURCE
                ):
                    # A fresh native safe-boundary decision cancels, rather
                    # than being overwritten by, an older endpoint handoff.
                    pending_speak = None
                rejection_reason: str | None = None
                if decision.action == ControlAction.REPLAN and not active_for_causal_gate:
                    rejection_reason = "replan_without_active_response"
                pending_speak_ready = bool(
                    not is_controller6_policy(policy)
                    and pending_speak is not None
                    and not active_for_causal_gate
                    and pending_speak.get("endpoint_frame_id") is not None
                    and speech_tracker.speech_complete
                    and decision.action not in {ControlAction.REPLAN, ControlAction.HARD_STOP}
                )
                if decision.action == ControlAction.BACKCHANNEL:
                    if peer_audio_event_key is None:
                        rejection_reason = "no_peer_speech_event"
                    elif backchannel_attempt_latch.has_attempted(peer_audio_event_key):
                        rejection_reason = "backchannel_event_latched"
                    with policy_causality_lock:
                        last_voiced_frame = speech_tracker.last_voiced_frame_id
                    if rejection_reason is None and (
                        last_voiced_frame is None
                        or int(policy_input.frame_id) - last_voiced_frame > BACKCHANNEL_MAX_VOICE_AGE_FRAMES
                    ) and not pending_speak_ready:
                        rejection_reason = "no_recent_user_audio"
                with policy_causality_lock:
                    if decision.action == ControlAction.REPLAN:
                        if replan_edge_latched:
                            rejection_reason = "replan_edge_latched"
                        if rejection_reason is None and replan_handoff_pending:
                            rejection_reason = "replan_handoff_pending"
                    elif replan_edge_latched:
                        replan_non_event_ticks += 1
                        if replan_non_event_ticks >= 2:
                            replan_edge_latched = False
                            replan_non_event_ticks = 0
                            replan_edge_generation_id = None
                    if rejection_reason is None and active_for_causal_gate:
                        emitted_audio_frames, has_post_watermark_voice = (
                            response_audio_admission.snapshot_locked(speech_tracker)
                        )
                        if decision.action == ControlAction.YIELD:
                            pending_branch_voice = False
                            for pending_branch in pending_pcm_branches:
                                branch_lock = pending_branch.get("spoken_prefix_lock")
                                if branch_lock is None:
                                    watermark = pending_branch.get("input_audio_watermark_frame_id")
                                else:
                                    with branch_lock:
                                        watermark = pending_branch.get("input_audio_watermark_frame_id")
                                if (
                                    isinstance(watermark, int)
                                    and not isinstance(watermark, bool)
                                    and speech_tracker.has_onset_after(watermark)
                                ):
                                    pending_branch_voice = True
                                    break
                            if pending_pcm_branches:
                                emitted_audio_frames = max(1, int(emitted_audio_frames))
                                has_post_watermark_voice = bool(
                                    has_post_watermark_voice or pending_branch_voice
                                )
                            rejection_reason = _yield_admission_rejection(
                                active_response=True,
                                emitted_audio_frames=emitted_audio_frames,
                                has_post_watermark_voice=has_post_watermark_voice,
                            )
                        elif decision.action == ControlAction.REPLAN:
                            # A pending renderer may be replanned before its first PCM frame.
                            if emitted_audio_frames > 0 and not has_post_watermark_voice:
                                rejection_reason = "no_new_user_onset"
                if rejection_reason is not None:
                    publish_decision(accepted=False, rejection_reason=rejection_reason)
                    return
                if trace_decision:
                    logger.info("[duplex_control6.trace] %s", json.dumps(trace, ensure_ascii=False, sort_keys=True))
                if decision.action in {ControlAction.SPEAK, ControlAction.BACKCHANNEL} and (
                    sess.active_response or sess.policy_response_pending
                ):
                    if (
                        decision.action == ControlAction.SPEAK
                        and sess.control_state == ControlState.BACKCHANNELING
                        and speech_tracker.run_onset_frame_id is not None
                        and (speech_tracker.voice_run_active or speech_tracker.speech_complete)
                    ):
                        if pending_speak is None:
                            pending_speak = {
                                "source_frame_id": int(decision.frame_id),
                                "confidence": float(decision.confidence),
                                "source": str(decision.source),
                                "voice_run_onset_frame_id": int(speech_tracker.run_onset_frame_id),
                                "endpoint_frame_id": (
                                    None
                                    if speech_tracker.endpoint_frame_id is None
                                    else int(speech_tracker.endpoint_frame_id)
                                ),
                                "endpoint_source_cutoff_sample_id": (
                                    speech_tracker.endpoint_source_cutoff_sample_id
                                ),
                            }
                        publish_decision(accepted=False, rejection_reason="deferred_until_backchannel_done")
                    else:
                        publish_decision(accepted=False, rejection_reason="response_already_active")
                    return
                if decision.action == ControlAction.HARD_STOP:
                    pending_speak = None
                elif isinstance(policy, GemmaNativeControl6Policy) and decision.action == ControlAction.SPEAK:
                    # Native Control6 may take the turn while the peer is still
                    # speaking; do not replace its decision with a deferred
                    # endpoint-triggered SPEAK from an earlier tick.
                    pending_speak = None
                deferred_speak = None
                gate_decision = decision
                if decision.action == ControlAction.SPEAK:
                    if speech_tracker.voice_run_active and not isinstance(
                        policy, GemmaNativeControl6Policy
                    ):
                        if pending_speak is None:
                            pending_speak = {
                                "source_frame_id": int(decision.frame_id),
                                "confidence": float(decision.confidence),
                                "source": str(decision.source),
                                "voice_run_onset_frame_id": int(speech_tracker.run_onset_frame_id),
                                "endpoint_frame_id": None,
                            }
                        publish_decision(accepted=False, rejection_reason="speech_endpoint_pending")
                        return
                    if not is_controller6_policy(policy) and (
                        pending_speak is None
                        and (
                            speech_tracker.last_voiced_frame_id is None
                            or not speech_tracker.speech_complete
                        )
                    ):
                        publish_decision(accepted=False, rejection_reason="no_new_user_utterance")
                        return
                if pending_speak_ready and not isinstance(policy, GemmaNativeControl6Policy):
                    deferred_speak = dict(pending_speak)
                    gate_decision = PolicyDecision(
                        action=ControlAction.SPEAK,
                        frame_id=int(deferred_speak["endpoint_frame_id"]),
                        confidence=float(deferred_speak["confidence"]),
                        reason="deferred_until_causal_speech_endpoint",
                        generation_id=int(policy_input.generation_id),
                        listener_frame_id=(
                            None if policy_input.listener_frame_id is None
                            else int(policy_input.listener_frame_id)
                        ),
                        source="deferred_speech_endpoint",
                        session=str(policy_input.session_id),
                    )
                if gate_decision.action == ControlAction.SPEAK:
                    with policy_cooldown_lock:
                        if policy_input.frame_id < empty_asr_cooldown_until_frame:
                            publish_decision(accepted=False, rejection_reason="empty_asr_cooldown")
                            return
                    if not sess.policy_audio_samples:
                        publish_decision(accepted=False, rejection_reason="no_committed_user_audio")
                        return
                prepared_direct_replan_context: dict[str, Any] | None = None
                if (
                    direct_audio_candidate
                    and gate_decision.action == ControlAction.REPLAN
                    and active_for_causal_gate
                ):
                    onset_frame_id = speech_tracker.run_onset_frame_id
                    if onset_frame_id is None:
                        publish_decision(accepted=False, rejection_reason="direct_audio_missing_onset")
                        return
                    try:
                        gate, prepared_direct_replan_context = _apply_direct_replan_transaction(
                            sess,
                            speaker_gate,
                            gate_decision,
                            first_frame_id=int(onset_frame_id),
                            last_frame_id=int(policy_input.frame_id),
                            user_active=bool(speech_tracker.voice_run_active),
                        )
                    except ValueError as exc:
                        publish_decision(
                            accepted=False,
                            rejection_reason=f"direct_audio_snapshot_invalid:{str(exc)[:80]}",
                            snapshot_error=exc,
                        )
                        return
                else:
                    if (
                        isinstance(policy, GemmaNativeControl6Policy)
                        and gate_decision.action == ControlAction.YIELD
                    ):
                        gate = _apply_native_control6_pending_pcm_yield_gate(
                            speaker_gate,
                            gate_decision,
                            sess,
                            pending_pcm_branches,
                        )
                    else:
                        gate = speaker_gate.apply(gate_decision, sess)
                if not gate.accepted:
                    publish_decision(accepted=False, rejection_reason="speaker_gate_rejected")
                    return
                if gate.action == ControlAction.REPLAN:
                    pending_speak = None
            if native_partial_worker is not None and (
                gate.action in {
                    ControlAction.SPEAK,
                    ControlAction.BACKCHANNEL,
                    ControlAction.REPLAN,
                    ControlAction.YIELD,
                } or gate.stop_synthesis
            ):
                native_partial_worker.pause()
            logger.info("[realtime %s] scheduler action=%s frame=%d active=%s", sess.session_id[:8], gate.action.value, frame.frame_id, sess.active_response)

            # The scheduler owns the decision, while the response worker owns
            # Qwen/PCM.  Revalidate the gate result before touching the
            # response registry: a new response may have begun after policy
            # evaluation but before this callback reached the worker.
            with sess._generation_lock:
                if (
                    gate.generation_id != sess.session_generation_id
                    or gate.speaker_segment_id != sess.speaker_segment_id
                ):
                    sess.stale_session_mutations += 1
                    publish_decision(accepted=False, rejection_reason="stale_generation")
                    return
                event_response_id = sess.response_id
            # The shared control record connects them without moving
            # inference into the scheduler thread.
            with response_controls_lock:
                control = response_controls.get(event_response_id or "")
                if control is not None:
                    control.segment_opened = control.segment_opened or gate.open_segment
                    control.segment_close_requested = control.segment_close_requested or gate.close_segment
                    control.segment_advances += int(gate.advance_segment)
                    control.backchannel_requested = control.backchannel_requested or gate.trigger_backchannel
                owns_synthesis = engine_synthesis_owner.is_owned(sess.session_id, event_response_id)
                gate_result = apply_speaker_gate_event(
                    gate,
                    engine=engine,
                    playout=None if control is None else control.playout,
                    packetizer=None if control is None else control.packetizer,
                    audio_queue=None if control is None else control.audio_queue,
                    text_queue=None if control is None else control.text_queue,
                    synthesis_active=owns_synthesis,
                    session=sess,
                    expected_generation_id=gate.generation_id,
                    stop_generation=lambda: _stop_owned_synthesis(sess.session_id, event_response_id),
                )
                if gate_result.get("accepted") is not True:
                    sess.stale_session_mutations += 1
                    publish_decision(accepted=False, rejection_reason="speaker_gate_application_rejected")
                    return

            if gate.action == ControlAction.YIELD and pending_pcm_branches:
                for pending_branch in pending_pcm_branches:
                    pending_response_id = str(pending_branch.get("response_id", ""))
                    if not pending_response_id:
                        continue
                    with response_controls_lock:
                        branch_worker_still_running = pending_response_id in response_controls
                    if not branch_worker_still_running:
                        # The response task may already have sent response.done
                        # while its PCM remains in the client device buffer.
                        # Cancel only that exact playback identity; the native
                        # finalizer uses its own ACK frontier and current listener
                        # generation after this wakeup.
                        scheduler_events.put_nowait({
                            "type": "response.cancelled",
                            "reason": "yield",
                            "response_id": pending_response_id,
                            "generation_id": int(
                                pending_branch.get("session_generation_id", -1)
                            ),
                        })

            policy_launch: tuple[
                ControlAction,
                np.ndarray,
                threading.Event | None,
                threading.Event | None,
                dict[str, Any] | None,
            ] | None = None
            replan_cancel_event: threading.Event | None = None
            replan_endpoint_waiter: threading.Event | None = None
            with sess._generation_lock:
                if (
                    sess.session_generation_id != gate.generation_id
                    or sess.speaker_segment_id != gate.speaker_segment_id
                ):
                    sess.stale_session_mutations += 1
                    publish_decision(accepted=False, rejection_reason="stale_generation")
                    return
                event_generation = gate.generation_id
                gate_response_id = event_response_id
                if gate.open_segment and gate.action in {ControlAction.SPEAK, ControlAction.BACKCHANNEL}:
                    if sess.active_response or sess.policy_response_pending:
                        sess.stale_session_mutations += 1
                        publish_decision(accepted=False, rejection_reason="response_already_active")
                        return
                    if not _response_admission_allowed_locked(sess, automatic=True, closing=closing):
                        sess.stale_session_mutations += 1
                        publish_decision(accepted=False, rejection_reason="auto_response_disabled")
                        return
                    committed = np.empty(0, dtype=np.float32)
                    direct_audio_context: dict[str, Any] | None = None
                    if gate.action == ControlAction.SPEAK:
                        controller6_speak = _native_control6_endpoint_bypass_enabled(policy)
                        onset_frame_id = (
                            int(sess.pending_user_onset_frame_id)
                            if controller6_speak and sess.pending_user_onset_frame_id is not None
                            else int(deferred_speak["voice_run_onset_frame_id"])
                            if deferred_speak is not None
                            else speech_tracker.run_onset_frame_id
                        )
                        endpoint_frame_id = (
                            int(speech_tracker.endpoint_frame_id)
                            if controller6_speak
                            and speech_tracker.speech_complete
                            and speech_tracker.endpoint_frame_id is not None
                            and speech_tracker.run_onset_frame_id == onset_frame_id
                            else int(deferred_speak["endpoint_frame_id"])
                            if deferred_speak is not None
                            else speech_tracker.endpoint_frame_id
                        )
                        endpoint_source_cutoff_sample_id = (
                            int(speech_tracker.endpoint_source_cutoff_sample_id)
                            if endpoint_frame_id is not None
                            and speech_tracker.endpoint_frame_id == endpoint_frame_id
                            and speech_tracker.endpoint_source_cutoff_sample_id is not None
                            else int(deferred_speak["endpoint_source_cutoff_sample_id"])
                            if deferred_speak is not None
                            and deferred_speak.get("endpoint_source_cutoff_sample_id") is not None
                            else int(pending_speak["endpoint_source_cutoff_sample_id"])
                            if pending_speak is not None
                            and pending_speak.get("endpoint_source_cutoff_sample_id") is not None
                            else None
                        )
                        if direct_audio_candidate:
                            if onset_frame_id is None:
                                publish_decision(accepted=False, rejection_reason="direct_audio_missing_onset")
                                return
                            final_frame_id = (
                                int(endpoint_frame_id)
                                if endpoint_frame_id is not None
                                else int(policy_input.frame_id)
                            )
                            try:
                                direct_audio_context = _snapshot_direct_policy_audio(
                                    sess,
                                    action=ControlAction.SPEAK,
                                    first_frame_id=int(onset_frame_id),
                                    last_frame_id=final_frame_id,
                                    user_active=bool(
                                        controller6_speak and speech_tracker.voice_run_active
                                    ),
                                    completed_utterance=endpoint_frame_id is not None,
                                    source_cutoff_sample_id=endpoint_source_cutoff_sample_id,
                                )
                            except ValueError as exc:
                                sess.control_state = ControlState.LISTENING
                                publish_decision(
                                    accepted=False,
                                    rejection_reason=f"direct_audio_snapshot_invalid:{str(exc)[:80]}",
                                    snapshot_error=exc,
                                )
                                return
                        if controller6_speak:
                            committed = sess.take_policy_audio_through_frame(
                                int(policy_input.frame_id)
                            )
                        elif endpoint_frame_id is None:
                            committed = sess.take_policy_audio()
                        else:
                            committed = sess.take_policy_audio_through_frame(int(endpoint_frame_id))
                        if not committed.size:
                            sess.control_state = ControlState.LISTENING
                            publish_decision(accepted=False, rejection_reason="no_committed_user_audio")
                            return
                        with sess._pending_audio_lock:
                            if sess.pending_audio.size >= committed.size:
                                sess.pending_audio = sess.pending_audio[committed.size:]
                        if endpoint_frame_id is not None:
                            sess.release_completed_utterance_audio(
                                int(onset_frame_id),
                                int(endpoint_frame_id),
                                source_cutoff_sample_id=endpoint_source_cutoff_sample_id,
                                generation_id=int(sess.session_generation_id),
                            )
                            speech_tracker.consume_endpoint()
                    elif direct_audio_candidate:
                        onset_frame_id = speech_tracker.run_onset_frame_id
                        if onset_frame_id is None:
                            publish_decision(accepted=False, rejection_reason="direct_audio_missing_onset")
                            return
                        try:
                            direct_audio_context = _snapshot_direct_policy_audio(
                                sess,
                                action=ControlAction.BACKCHANNEL,
                                first_frame_id=int(onset_frame_id),
                                last_frame_id=int(policy_input.frame_id),
                            )
                        except ValueError as exc:
                            sess.control_state = ControlState.LISTENING
                            publish_decision(
                                accepted=False,
                                rejection_reason=f"direct_audio_snapshot_invalid:{str(exc)[:80]}",
                                snapshot_error=exc,
                            )
                            return
                        committed = np.asarray(
                            direct_audio_context["_direct_audio_pcm"], dtype=np.float32
                        )
                    if gate.action == ControlAction.BACKCHANNEL:
                        if not backchannel_attempt_latch.mark_attempted(peer_audio_event_key):
                            sess.control_state = ControlState.LISTENING
                            publish_decision(accepted=False, rejection_reason="backchannel_event_latched")
                            return
                    sess.policy_response_pending = True
                    intent_origin = None
                    if deferred_speak is not None:
                        intent_origin = {
                            "source_frame_id": int(deferred_speak["source_frame_id"]),
                            "endpoint_frame_id": int(deferred_speak["endpoint_frame_id"]),
                            "source": str(deferred_speak["source"]),
                        }
                    elif direct_audio_candidate:
                        origin_frame = int(policy_input.frame_id)
                        endpoint_for_event = int(
                            direct_audio_context["_direct_audio_last_frame_id"]
                            if direct_audio_context is not None
                            else origin_frame
                        )
                        intent_origin = {
                            "source_frame_id": origin_frame,
                            "endpoint_frame_id": endpoint_for_event,
                            "source": "direct_audio_policy",
                        }
                        if gate.action == ControlAction.SPEAK:
                            intent_origin["_user_turn_onset_frame_id"] = (
                                int(sess.pending_user_onset_frame_id)
                                if sess.pending_user_onset_frame_id is not None
                                else int(direct_audio_context["_direct_audio_first_frame_id"])
                            )
                            intent_origin["_user_turn_endpoint_complete"] = bool(
                                speech_tracker.speech_complete
                            )
                    elif gate.action == ControlAction.SPEAK:
                        intent_origin = {
                            "source_frame_id": int(decision.frame_id),
                            "endpoint_frame_id": int(
                                speech_tracker.endpoint_frame_id
                                if speech_tracker.endpoint_frame_id is not None
                                else policy_input.frame_id
                            ),
                            "source": str(decision.source),
                        }
                    if gate.action == ControlAction.BACKCHANNEL:
                        trigger_frame = int(decision.frame_id)
                        intent_origin = {
                            "source_frame_id": trigger_frame,
                            "endpoint_frame_id": int(policy_input.frame_id),
                            "source": str(decision.source),
                            **(intent_origin or {}),
                        }
                    if direct_audio_context is not None:
                        intent_origin = {**(intent_origin or {}), **direct_audio_context}
                    if gate.action == ControlAction.BACKCHANNEL and peer_audio_event_key is not None:
                        intent_origin = {
                            **(intent_origin or {}),
                            "backchannel_peer_event_id": (
                                f"{peer_audio_event_key[0]}:{peer_audio_event_key[1]}"
                            ),
                        }
                    if intent_origin is not None:
                        intent_origin = dict(intent_origin)
                        intent_origin.update({
                            "policy_decision_frame_id": int(decision.frame_id),
                            "policy_decision_at_ns": int(trace["policy_decision_at_ns"]),
                            "policy_input_listener_frame_id": (
                                None if policy_input.listener_frame_id is None
                                else int(policy_input.listener_frame_id)
                            ),
                            "policy_input_listener_hidden_frame_id": (
                                None if policy_input.listener_hidden_frame_id is None
                                else int(policy_input.listener_hidden_frame_id)
                            ),
                            "policy_input_listener_hidden_at_ns": (
                                None if policy_input.listener_hidden_at_ns is None
                                else int(policy_input.listener_hidden_at_ns)
                            ),
                        })
                    if gate.action in {ControlAction.SPEAK, ControlAction.REPLAN}:
                        with transcript_state_lock:
                            input_utterance_id = transcript_state["utterance_id"]
                            input_transcript_epoch = int(transcript_state["epoch"])
                        input_first_frame_id = speech_tracker.run_onset_frame_id
                        input_last_frame_id = (
                            speech_tracker.endpoint_frame_id
                            if gate.action == ControlAction.SPEAK
                            and speech_tracker.endpoint_frame_id is not None
                            else speech_tracker.last_voiced_frame_id
                        )
                        if (
                            input_utterance_id
                            and input_first_frame_id is not None
                            and input_last_frame_id is not None
                        ):
                            intent_origin = dict(intent_origin or {})
                            intent_origin["_input_transcript_context"] = {
                                "utterance_id": str(input_utterance_id),
                                "epoch_id": input_transcript_epoch,
                                "first_frame_id": int(input_first_frame_id),
                                "last_frame_id": int(input_last_frame_id),
                            }
                    policy_launch = (gate.action, committed, None, None, intent_origin)
                    with policy_causality_lock:
                        response_audio_admission.reset_locked()
                        replan_handoff_pending = False
                elif (
                    gate.action == ControlAction.REPLAN
                    and gate.stop_synthesis
                    and gate.previous_state in {ControlState.SPEAKING, ControlState.BACKCHANNELING}
                    and gate_response_id
                ):
                    # Stop the old response now, but wait for the correction's
                    # speech endpoint before freezing audio for ASR. The
                    # scheduler snapshots through that exact frame so later
                    # ingress cannot leak into the REPLAN prompt.
                    wait_for_endpoint = bool(
                        speech_tracker.voice_run_active and not direct_audio_candidate
                    )
                    committed = np.empty(0, dtype=np.float32)
                    intent_origin = {
                        "source_frame_id": int(decision.frame_id),
                        "endpoint_frame_id": int(
                            speech_tracker.endpoint_frame_id
                            if speech_tracker.endpoint_frame_id is not None
                            else decision.frame_id
                        ),
                        "source": str(decision.source),
                    }
                    replan_onset_frame_id = speech_tracker.run_onset_frame_id
                    if wait_for_endpoint:
                        onset_frame_id = replan_onset_frame_id
                        if onset_frame_id is None:
                            publish_decision(accepted=False, rejection_reason="missing_user_onset")
                            return
                        replan_endpoint_waiter = threading.Event()
                        with replan_endpoint_lock:
                            replan_endpoint_waiters[event_generation] = (
                                int(onset_frame_id),
                                replan_endpoint_waiter,
                            )
                        if direct_audio_candidate:
                            intent_origin = {
                                "source_frame_id": int(policy_input.frame_id),
                                "endpoint_frame_id": int(policy_input.frame_id),
                                "source": "direct_audio_replan",
                            }
                    else:
                        if direct_audio_candidate:
                            if prepared_direct_replan_context is None:
                                raise RuntimeError("REPLAN senza snapshot diretto prenotato")
                            direct_audio_context = prepared_direct_replan_context
                            intent_origin = {
                                "source_frame_id": int(policy_input.frame_id),
                                "endpoint_frame_id": int(policy_input.frame_id),
                                "source": "direct_audio_replan",
                                **direct_audio_context,
                            }
                            committed = np.asarray(
                                direct_audio_context["_direct_audio_pcm"], dtype=np.float32
                            )
                            # Drop retained frames through the materialized cutoff;
                            # the bounded direct snapshot remains the response source.
                            sess.take_policy_audio_through_frame(int(policy_input.frame_id))
                        else:
                            committed = sess.take_policy_audio_through_frame(int(policy_input.frame_id))
                        if committed.size:
                            with sess._pending_audio_lock:
                                if sess.pending_audio.size >= committed.size:
                                    sess.pending_audio = sess.pending_audio[committed.size:]
                        if (
                            speech_tracker.run_onset_frame_id is not None
                            and speech_tracker.endpoint_frame_id is not None
                        ):
                            sess.release_completed_utterance_audio(
                                int(speech_tracker.run_onset_frame_id),
                                int(speech_tracker.endpoint_frame_id),
                                source_cutoff_sample_id=speech_tracker.endpoint_source_cutoff_sample_id,
                                generation_id=int(sess.session_generation_id),
                            )
                        speech_tracker.consume_endpoint()
                    with transcript_state_lock:
                        input_utterance_id = transcript_state["utterance_id"]
                        input_transcript_epoch = int(transcript_state["epoch"])
                    if (
                        input_utterance_id
                        and replan_onset_frame_id is not None
                        and speech_tracker.last_voiced_frame_id is not None
                    ):
                        intent_origin = dict(intent_origin or {})
                        intent_origin["_input_transcript_context"] = {
                            "utterance_id": str(input_utterance_id),
                            "epoch_id": input_transcript_epoch,
                            "first_frame_id": int(replan_onset_frame_id),
                            "last_frame_id": int(speech_tracker.last_voiced_frame_id),
                        }
                    sess.policy_response_pending = True
                    replan_cancel_event = threading.Event()
                    with policy_causality_lock:
                        # Consume the edge only after the decision passed the
                        # causal gate, SpeakerGate, and generation revalidation.
                        # An early raw REPLAN must not suppress a later valid one.
                        replan_edge_latched = True
                        replan_non_event_ticks = 0
                        replan_edge_generation_id = int(event_generation)
                        replan_handoff_pending = True
                    with replan_cancel_waiters_lock:
                        replan_cancel_waiters[event_generation] = replan_cancel_event
                    policy_launch = (
                        ControlAction.REPLAN,
                        committed,
                        replan_cancel_event,
                        replan_endpoint_waiter,
                        intent_origin,
                    )
            if deferred_speak is None:
                publish_decision(accepted=True)
            else:
                publish_decision(accepted=False, rejection_reason="deferred_speak_dispatched")
            if gate.open_segment:
                scheduler_events.put_nowait({
                    "type": "speaker.segment.opened",
                    "action": gate.action.value.lower(),
                    "generation_id": event_generation,
                    "response_id": gate_response_id,
                    "_generation_fence": event_generation,
                })
            if gate.trigger_backchannel:
                scheduler_events.put_nowait({
                    "type": "speaker.backchannel.requested",
                    "generation_id": event_generation,
                    "response_id": gate_response_id,
                    "_generation_fence": event_generation,
                })
            if policy_launch is not None:
                # The accepted SPEAK decision consumed this input turn.
                # Clear its audio evidence before later frames can authorize
                # another response. Scripted policies are action sequences
                # across ticks: fence the decision without rewinding the script.
                if isinstance(policy, ScriptedPolicy):
                    sess.set_turn_policy(policy)
                else:
                    sess.reset_turn_policy()
            if deferred_speak is not None and policy_launch is not None:
                pending_speak = None
            if policy_launch is not None:
                loop.call_soon_threadsafe(
                    launch_policy_response,
                    policy_launch[0],
                    policy_launch[1],
                    event_generation,
                    gate.speaker_segment_id,
                    policy_launch[2],
                    policy_launch[3],
                    policy_launch[4],
                )

            # YIELD is a safe-boundary completion, not a cancellation event.
            # Destructive actions invalidate the generation and notify once.
            with sess._generation_lock:
                response_active = sess.active_response
            if (
                not gate.stop_synthesis
                or gate.action == ControlAction.YIELD
                or (not response_active and control is None)
            ):
                return
            with sess._generation_lock:
                if not _claim_response_cancel_notification(sess, gate_response_id):
                    return
                event = {
                    "type": "response.cancelled",
                    "reason": gate.action.value.lower(),
                    "response_id": gate_response_id,
                    "_generation_fence": gate.generation_id,
                }
                if replan_cancel_event is not None:
                    event["_replan_followup_generation"] = event_generation
                if gate_response_id:
                    event["runtime_metrics"] = runtime_metrics_event(gate_response_id)
            scheduler_events.put_nowait(event)

        def scheduler_error(exc: BaseException) -> None:
            logger.error(
                "[realtime %s] scheduler callback failed",
                sess.session_id[:8],
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            scheduler_events.put_nowait(
                {"type": "error", "error": {"code": "SCHEDULER_ERROR", "message": str(exc)[:240]}}
            )

        def fail_native_listener_output_health(
            frame_id: int, health_result: dict[str, Any]
        ) -> None:
            nonlocal closing
            if native_listener_output_faulted.is_set():
                return
            native_listener_output_faulted.set()
            closing = True
            sess.invalidate_generation("native_listener_output_health_fault")
            error = {
                "code": "NATIVE_LISTENER_OUTPUT_FAULT",
                "message": str(health_result.get("error") or health_result.get("status"))[:240],
                "terminal": True,
                "frame_id": int(frame_id),
                "health_status": str(health_result.get("status", "unknown")),
            }
            logger.error(
                "[realtime %s] terminal native listener output fault frame=%d status=%s",
                sess.session_id[:8], int(frame_id), error["health_status"],
            )
            scheduler_events.put_nowait({"type": "error", "error": error})

        scheduler_cuda_stream = (
            torch.cuda.Stream(device=torch.cuda.current_device())
            if torch.cuda.is_available() else None
        )
        if scheduler_cuda_stream is not None:
            session_runtime_metrics["scheduler_cuda_stream"] = {
                "enabled": True,
                "device": str(scheduler_cuda_stream.device),
                "handle": int(scheduler_cuda_stream.cuda_stream),
                "ownership": "per_session_listener_policy",
            }

        def timed_scheduler_callback(stage_name: str, callback: Callable[[Any, Any], Any]):
            def wrapped(frame: Any, tick: Any) -> Any:
                started_ns = time.monotonic_ns()
                if stage_name == "scheduler_listener_callback_wall_ms":
                    tick.scheduler_dispatch_start_ns = started_ns
                try:
                    with _scheduler_cuda_stream_context(scheduler_cuda_stream):
                        return callback(frame, tick)
                finally:
                    finished_ns = time.monotonic_ns()
                    elapsed_ns = max(0, finished_ns - started_ns)
                    if stage_name == "scheduler_listener_callback_wall_ms":
                        tick.scheduler_dispatch_end_ns = finished_ns
                    record_scheduler_stage(stage_name, elapsed_ns / 1_000_000.0)
                    spans = getattr(tick, "_scheduler_callback_wall_ns", None)
                    if not isinstance(spans, dict):
                        spans = {}
                        setattr(tick, "_scheduler_callback_wall_ns", spans)
                    spans[stage_name] = elapsed_ns
            return wrapped

        # Keep phase timing around the entire callback, including prologue,
        # session-lock waits, and epilogue; inner stage timers remain unchanged.
        sess.scheduler.on_listener = timed_scheduler_callback(
            "scheduler_listener_callback_wall_ms", scheduler_listener
        )
        sess.scheduler.on_policy = timed_scheduler_callback(
            "scheduler_policy_callback_wall_ms", scheduler_policy
        )
        if sess.scheduler.on_speaker is not None:
            sess.scheduler.on_speaker = timed_scheduler_callback(
                "scheduler_speaker_callback_wall_ms", sess.scheduler.on_speaker
            )
        sess.scheduler.on_error = scheduler_error

        scheduler_tick = sess.scheduler.tick
        def timed_scheduler_tick() -> Any:
            call_started_ns = time.monotonic_ns()
            tick = None
            try:
                tick = scheduler_tick()
                return tick
            finally:
                call_finished_ns = time.monotonic_ns()
                record_scheduler_stage(
                    "scheduler_tick_wall_ms",
                    max(0, call_finished_ns - call_started_ns) / 1_000_000.0,
                )
                if tick is not None and isinstance(getattr(tick, "observed_ns", None), int):
                    processing_ns = max(0, call_finished_ns - int(tick.observed_ns))
                    record_scheduler_stage(
                        "scheduler_tick_processing_wall_ms", processing_ns / 1_000_000.0
                    )
                    spans = getattr(tick, "_scheduler_callback_wall_ns", {})
                    callback_ns = sum(
                        max(0, int(value)) for value in spans.values()
                    ) if isinstance(spans, dict) else 0
                    outside_ns = max(0, processing_ns - callback_ns)
                    record_scheduler_stage(
                        "scheduler_tick_outside_callbacks_wall_ms", outside_ns / 1_000_000.0
                    )
        sess.scheduler.tick = timed_scheduler_tick
        if shared_listener_microbatch_enabled:
            await _register_shared_listener_session(
                sess,
                record_scheduler_stage=record_scheduler_stage,
                pending_speak_provider=lambda: pending_speak,
                policy_context_provider=(
                    _shared_native_policy_context
                    if isinstance(sess.turn_policy, GemmaNativeControl6Policy)
                    else None
                ),
            )
            response_binding = _attach_shared_listener_response_branches(
                shared_listener_bindings.get(sess.session_id),
                playback_ack_branches,
            )
            session_runtime_metrics["scheduler"]["shared_listener_microbatch_enabled"] = True
            session_runtime_metrics["scheduler"]["listener_tick_period_ms"] = 80
            session_runtime_metrics["scheduler"]["listener_batch_size_policy"] = (
                "all_ready_real_rows"
            )
        else:
            sess.scheduler.start(name=f"duplex-scheduler-{sess.session_id[:8]}")
        scheduler_event_task = asyncio.create_task(publish_scheduler_events())

        async def produce_response(
            committed_audio: np.ndarray,
            *,
            action: ControlAction | None = None,
            intent_origin: dict[str, Any] | None = None,
            automatic_policy_response: bool = False,
        ) -> None:
            nonlocal response_task, closing, replan_handoff_pending
            if automatic_policy_response:
                onset_frame_id = (
                    int(intent_origin["_user_turn_onset_frame_id"])
                    if action == ControlAction.SPEAK
                    and isinstance(intent_origin, dict)
                    and intent_origin.get("_user_turn_onset_frame_id") is not None
                    else None
                )
                with sess._generation_lock:
                    admitted = _response_admission_allowed_locked(
                        sess, automatic=True, closing=closing
                    )
                    if not admitted and onset_frame_id is not None:
                        sess.cancel_user_onset_response(onset_frame_id)
                if not admitted:
                    return
            response_id = "resp_" + uuid.uuid4().hex[:12]
            response_event_loop = asyncio.get_running_loop()
            archive_id_holder: dict[str, int | None] = {"archive_id": None}
            archive_reservation_attempted = {"value": False}
            archive_assistant_finalized = {"value": False}
            archive_heard_finalized = {"value": False}
            archive_cpu_submitted = {"value": False}
            response_done_sent = False
            user_turn_completed = False
            user_turn_onset_frame_id = (
                int(intent_origin["_user_turn_onset_frame_id"])
                if action == ControlAction.SPEAK
                and isinstance(intent_origin, dict)
                and intent_origin.get("_user_turn_onset_frame_id") is not None
                else None
            )
            response_created_at_monotonic_ns: int | None = None
            response_uses_direct_audio = _response_uses_direct_audio_candidate(
                candidate_enabled=direct_audio_candidate,
                committed_audio=committed_audio,
                intent_origin=intent_origin,
            )
            response_protocol = (
                _configured_audio_response_protocol(engine)
                if response_uses_direct_audio else None
            )
            direct_audio_v2_response = (
                response_uses_direct_audio and response_protocol == AUDIO_RESPONSE_PROTOCOL_V2
            )
            controller_guided_say_response = (
                response_uses_direct_audio and response_protocol == CONTROLLER_GUIDED_SAY_V1
            )
            direct_audio_context = (
                intent_origin
                if response_uses_direct_audio and isinstance(intent_origin, dict)
                and intent_origin.get("_direct_audio_pcm") is not None
                else None
            )
            with response_runtime_metrics_lock:
                response_runtime_metrics.setdefault(response_id, {})
            with sess._generation_lock:
                completed_history_roles = [
                    str(message.get("role"))
                    for message in sess.history
                    if message.get("role") in {"user", "assistant"}
                ]
            initialize_generation_trace(response_id, action, completed_history_roles)
            started = time.perf_counter()
            token = sess.begin_response(
                response_id,
                preserve_pending_user_turn_audio=(action == ControlAction.BACKCHANNEL),
                preserve_newer_pending_user_turn_audio_after_onset_frame_id=(
                    user_turn_onset_frame_id
                    if action == ControlAction.SPEAK
                    else None
                ),
            )
            playback_branch = {
                "response_id": response_id,
                "session_generation_id": int(token.session_generation_id),
                "speaker_segment_id": str(token.speaker_segment_id),
                "playback_segments": {},
                "segment_token_packets": [],
                "spoken_prefix_token_ids": (),
                "spoken_prefix_exclusive_cursor": 0,
                "spoken_prefix_text": "",
                "client_consumed_pcm_sequence": -1,
                "client_consumed_segments": (),
                "spoken_prefix_lock": threading.Lock(),
                **new_playback_ack_state(),
            }
            with playback_ack_branches_lock:
                playback_ack_branches[response_id] = playback_branch
                while len(playback_ack_branches) > 8:
                    playback_ack_branches.pop(next(iter(playback_ack_branches)))
            unified_native_branch: dict[str, Any] | None = None
            unified_transition_lock: Any | None = None
            unified_close_ids: tuple[int, ...] | None = None
            if unified_kv_enabled and response_uses_direct_audio:
                binding = shared_listener_bindings.get(sess.session_id)
                worker = shared_listener_worker_holder.get("worker")
                if binding is None or worker is None:
                    sess.end_response(token)
                    await send_ws({
                        "type": "error",
                        "response_id": response_id,
                        "error": {
                            "code": "UNIFIED_NATIVE_CACHE_UNAVAILABLE",
                            "message": "unified direct-audio response requires its registered native listener owner",
                        },
                    })
                    return
                unified_transition_lock = binding["unified_turn_transition_lock"]
                unified_transition_lock.acquire()
                try:
                    if action == ControlAction.BACKCHANNEL:
                        begin_deadline = asyncio.get_running_loop().time() + 2.0
                        private_handle = None
                        while private_handle is None:
                            if closing or not sess.is_generation_current(token):
                                raise RuntimeError(
                                    "private backchannel fork lost the response generation"
                                )
                            listener_snapshot = await asyncio.wrap_future(
                                worker.session_state(sess.session_id)
                            )
                            try:
                                private_handle = await asyncio.wrap_future(
                                    worker.begin_ephemeral_decode(
                                        sess.session_id,
                                        expected_generation_id=int(listener_snapshot.generation_id),
                                        expected_cache_position=int(listener_snapshot.cache_position),
                                    )
                                )
                            except RuntimeError as exc:
                                if (
                                    str(exc) not in {
                                        "private decode fork requires a completed listener frame watermark",
                                        "unified native operation lost its generation/cache-position compare-and-swap",
                                    }
                                    or asyncio.get_running_loop().time() >= begin_deadline
                                ):
                                    raise
                                await asyncio.sleep(0.002)
                        unified_native_branch = {
                            "worker": worker,
                            "response_protocol": _configured_audio_response_protocol(engine),
                            "response_id": response_id,
                            "speaker_segment_id": str(token.speaker_segment_id),
                            "session_generation_id": int(token.session_generation_id),
                            "listener_generation_id": int(private_handle.parent_generation_id),
                            "cache_position": int(private_handle.source_cache_position),
                            "turn_start_cache_position": int(private_handle.source_cache_position),
                            "playback_segments": {},
                            "client_consumed_pcm_sequence": -1,
                            "client_consumed_segments": (),
                            "spoken_prefix_lock": threading.Lock(),
                            "queue_open": False,
                            "ephemeral_private": True,
                            "ephemeral_decode_handle": private_handle,
                            "finalized": False,
                            "ended_response": False,
                            "transition_lock": unified_transition_lock,
                            "close_ids": (),
                            "policy_decision_at_ns": (
                                int(intent_origin["policy_decision_at_ns"])
                                if isinstance(intent_origin, dict)
                                and intent_origin.get("policy_decision_at_ns") is not None
                                else None
                            ),
                            "queue_open_requested_ns": None,
                            "queue_open_ready_ns": time.monotonic_ns(),
                            "generated_token_count": 0,
                            "native_token_append_count": 0,
                            **new_playback_ack_state(),
                        }
                        with playback_ack_branches_lock:
                            playback_ack_branches[response_id] = unified_native_branch
                    else:
                        unified_close_ids = (
                            _unified_end_of_turn_token_id(engine, sess),
                            *_build_unified_next_user_header(engine, sess),
                        )
                        queue_open_requested_ns = time.monotonic_ns()
                        begin_deadline = asyncio.get_running_loop().time() + 2.0
                        while True:
                            if closing or not sess.is_generation_current(token):
                                raise RuntimeError("unified cache queue opening lost the response generation")
                            listener_snapshot = await asyncio.wrap_future(
                                worker.session_state(sess.session_id)
                            )
                            try:
                                opened_snapshot = await asyncio.wrap_future(
                                    worker.begin_assistant_audio_queue(
                                        sess.session_id,
                                        expected_generation_id=int(listener_snapshot.generation_id),
                                        expected_cache_position=int(listener_snapshot.cache_position),
                                        reserve_tail_tokens=len(unified_close_ids) + 2,
                                        response_id=response_id,
                                    )
                                )
                            except RuntimeError as exc:
                                if (
                                    str(exc) not in {
                                        "assistant audio queue must open at an idle frame boundary",
                                        "unified native operation lost its generation/cache-position compare-and-swap",
                                    }
                                    or asyncio.get_running_loop().time() >= begin_deadline
                                ):
                                    raise
                                await asyncio.sleep(0.002)
                                continue
                            unified_native_branch = {
                                "worker": worker,
                                "response_protocol": response_protocol,
                                "response_id": response_id,
                                "speaker_segment_id": str(token.speaker_segment_id),
                                "session_generation_id": int(token.session_generation_id),
                                "listener_generation_id": int(opened_snapshot.generation_id),
                                "cache_position": int(opened_snapshot.cache_position),
                                "turn_start_cache_position": int(opened_snapshot.cache_position),
                                "queue_open": True,
                                "finalized": False,
                                "ended_response": False,
                                "history_committed": False,
                                "transition_lock": unified_transition_lock,
                                "close_ids": unified_close_ids,
                                "reserved_close_tokens": len(unified_close_ids) + 2,
                                "transition_token_ids": (),
                                "spoken_prefix_text": "",
                                "spoken_prefix_kv_position": None,
                                "spoken_prefix_token_ids": (),
                                "spoken_prefix_exclusive_cursor": int(opened_snapshot.cache_position),
                                "spoken_prefix_text_parts": (),
                                "playback_segments": {},
                                "segment_token_packets": [],
                                "client_consumed_pcm_sequence": -1,
                                "client_consumed_segments": (),
                                "client_consumed_pcm_events": [],
                                "spoken_prefix_rebuilt": False,
                                "spoken_prefix_lock": threading.Lock(),
                                "generated_token_count": 0,
                                "native_token_append_count": 0,
                                "policy_decision_at_ns": (
                                    int(intent_origin["policy_decision_at_ns"])
                                    if isinstance(intent_origin, dict)
                                    and intent_origin.get("policy_decision_at_ns") is not None
                                    else None
                                ),
                                "queue_open_requested_ns": queue_open_requested_ns,
                                "queue_open_ready_ns": time.monotonic_ns(),
                                **new_playback_ack_state(),
                            }
                            with playback_ack_branches_lock:
                                playback_ack_branches[response_id] = unified_native_branch
                            if os.environ.get("DUPLEX_NATIVE_CONTROL6_TRACE") == "1":
                                logger.info(
                                    "[control6_queue_handoff] server queue_open_published "
                                    "session=%s response=%s listener_generation=%d cursor=%d "
                                    "next_frame_id=%d branch_queue_open=%s",
                                    sess.session_id,
                                    response_id,
                                    int(opened_snapshot.generation_id),
                                    int(opened_snapshot.cache_position),
                                    int(opened_snapshot.next_frame_id),
                                    unified_native_branch.get("queue_open"),
                                )
                            break
                except BaseException as exc:
                    unified_transition_lock.release()
                    sess.end_response(token)
                    if user_turn_onset_frame_id is not None:
                        sess.cancel_user_onset_response(user_turn_onset_frame_id)
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    logger.exception(
                        "[realtime %s %s] failed to open unified native audio queue",
                        sess.session_id[:8],
                        response_id,
                    )
                    await send_ws({
                        "type": "error",
                        "response_id": response_id,
                        "error": {
                            "code": "UNIFIED_NATIVE_CACHE_UNAVAILABLE",
                            "message": str(exc)[:240],
                        },
                    })
                    return

            async def finalize_unified_native_cache(*, history_committed: bool) -> None:
                branch = unified_native_branch
                if branch is None or branch.get("finalized"):
                    return
                transition_lock = branch["transition_lock"]
                transition_lock.acquire()
                try:
                    worker = branch["worker"]
                    if int(branch.get("client_consumed_pcm_sequence", -1)) < 0:
                        branch["client_playback_limitation"] = (
                            "no client_consumed_pcm ACK; only the last confirmed complete-segment prefix is retained"
                        )
                        update_response_runtime_metrics(response_id, "client_consumed_pcm", {
                            "status": "no_playback_ack",
                            "prefix_policy": "retain_last_confirmed_prefix_only",
                            "logical_client_consumption_only": True,
                        })
                    await _finalize_unified_native_cache_branch(
                        worker,
                        sess.session_id,
                        branch,
                        history_committed=history_committed,
                        close_ids=tuple(branch["close_ids"]),
                    )
                finally:
                    if not branch.get("ended_response"):
                        sess.end_response(token)
                        branch["ended_response"] = True
                    transition_lock.release()

            async def finalize_unified_native_cache_after_playback() -> None:
                branch = unified_native_branch
                if branch is None:
                    return
                # Wait outside the transition lock so ACK ingress and listener
                # frames continue while the playback frontier advances.
                await _wait_for_response_playback_drain(branch)
                try:
                    await finalize_unified_native_cache(
                        history_committed=bool(branch.get("history_committed"))
                        and branch.get("playback_cancelled") is not True
                    )
                except Exception as exc:
                    logger.exception(
                        "[realtime %s %s] deferred unified native cache finalization failed",
                        sess.session_id[:8],
                        response_id,
                    )
                    fail_native_listener_output_health(
                        -1,
                        {
                            "status": "unified_cache_finalization_failed",
                            "error": str(exc),
                        },
                    )
                    if not branch.get("ended_response"):
                        sess.end_response(token)
                        branch["ended_response"] = True

            def response_listener_admission_wait() -> None:
                listener_admission_wait(token)

            def handle_v2_response_action(
                parsed: dict[str, Any],
                heard_truncated: bool,
                *,
                base_messages: list[dict[str, Any]],
            ) -> str:
                if not direct_audio_v2_response:
                    return "accepted"
                with sess._generation_lock:
                    if not sess.is_generation_current(token):
                        return "stale_generation"
                response_action = str(parsed.get("action", "")).strip().upper()
                wake_action = action.value if action is not None else None
                disposition = audio_response_disposition_for_actions(
                    wake_action,
                    response_action,
                )
                if disposition == "defer_until_endpoint":
                    if direct_audio_context is None:
                        raise AudioResponseProtocolError("deferred_speak_missing_audio_snapshot")
                    onset_frame_id = direct_audio_context.get("_direct_audio_first_frame_id")
                    source_frame_id = direct_audio_context.get("_direct_audio_last_frame_id")
                    if onset_frame_id is None or source_frame_id is None:
                        raise AudioResponseProtocolError("deferred_speak_missing_utterance_fence")
                    pending_response_speak_handoffs.put_nowait({
                        "generation_id": int(token.session_generation_id),
                        "voice_run_onset_frame_id": int(onset_frame_id),
                        "source_frame_id": int(source_frame_id),
                        "confidence": 0.0,
                    })

                if disposition == "archive_completed_user_turn":
                    if direct_audio_context is None:
                        raise AudioResponseProtocolError("response_archive_missing_audio_snapshot")
                    cpu_archive_for_direct_v2 = bool(audio_history_archive_cpu_enabled)
                    if archive_id_holder["archive_id"] is None:
                        archive_id_holder["archive_id"] = sess.reserve_audio_history_archive(
                            pcm=(
                                np.asarray(
                                    direct_audio_context["_direct_audio_pcm"], dtype=np.float32
                                ).reshape(-1).copy()
                                if cpu_archive_for_direct_v2 else None
                            ),
                            first_frame_id=int(direct_audio_context["_direct_audio_first_frame_id"]),
                            last_frame_id=int(direct_audio_context["_direct_audio_last_frame_id"]),
                            base_messages=base_messages,
                            source=(
                                "post_response_asr" if cpu_archive_for_direct_v2
                                else "heard_action_say_v2"
                            ),
                            response_created_at_monotonic_ns=response_created_at_monotonic_ns,
                            coverage=(
                                "partial" if bool(direct_audio_context.get("_direct_audio_active"))
                                else "complete"
                            ),
                        )
                    if archive_id_holder["archive_id"] is None:
                        archive_heard_finalized["value"] = True
                        update_response_runtime_metrics(response_id, "audio_history_archive", {
                            "status": "skipped_backpressure",
                            "source": "heard_action_say_v2",
                            "session_id": sess.session_id,
                            "response_action": response_action,
                        })
                        return "accepted"
                    if cpu_archive_for_direct_v2:
                        archive_id = int(archive_id_holder["archive_id"])
                        archive_cpu_submitted["value"] = start_audio_history_archive(
                            archive_id, response_id
                        )
                        update_response_runtime_metrics(response_id, "audio_history_archive_cpu_submit", {
                            "status": (
                                "submitted_cpu" if archive_cpu_submitted["value"]
                                else "failed_cpu_submit"
                            ),
                            "source": "post_response_asr",
                            "archive_id": archive_id,
                            "session_id": sess.session_id,
                            "response_action": response_action,
                            "input_first_frame_id": int(
                                direct_audio_context["_direct_audio_first_frame_id"]
                            ),
                            "input_last_frame_id": int(
                                direct_audio_context["_direct_audio_last_frame_id"]
                            ),
                            "input_audio_samples": int(
                                np.asarray(direct_audio_context["_direct_audio_pcm"]).size
                            ),
                            "model_heard_text_used_for_archive": False,
                        })
                    else:
                        heard_value = str(parsed.get("heard", "")).strip()
                        transcript_is_complete = bool(heard_value) and not heard_truncated
                        with sess._generation_lock:
                            if not sess.is_generation_current(token):
                                return "stale_generation"
                            archived = sess.complete_audio_history_archive(
                                int(archive_id_holder["archive_id"]),
                                heard_value if transcript_is_complete else "",
                            )
                        archive_heard_finalized["value"] = bool(archived)
                        update_response_runtime_metrics(response_id, "audio_history_archive", {
                            "status": "ready" if archived and transcript_is_complete else "empty" if archived else "stale",
                            "source": "heard_action_say_v2",
                            "response_action": response_action,
                            "heard_truncated": bool(heard_truncated),
                            "transcript_committed": bool(archived and transcript_is_complete),
                        })

                action_event = {
                    "type": "response.action_selected",
                    "response_id": response_id,
                    "wake_action": wake_action or "",
                    "response_action": response_action,
                    "deferred_until_endpoint": disposition == "defer_until_endpoint",
                }
                sent = asyncio.run_coroutine_threadsafe(
                    send_ws(action_event, token),
                    response_event_loop,
                ).result(timeout=5.0)
                if not sent:
                    return "stale_generation"
                if disposition == "defer_until_endpoint":
                    logger.info(
                        "[realtime %s %s] native response SPEAK deferred to endpoint wake=%s onset=%d",
                        sess.session_id[:8],
                        response_id,
                        wake_action,
                        int(direct_audio_context["_direct_audio_first_frame_id"]),
                    )
                    return "deferred_until_endpoint"
                return "accepted"

            if native_partial_worker is not None:
                native_partial_worker.pause()
            with policy_causality_lock:
                replan_handoff_pending = False
                response_audio_admission.reset_locked()
            with sess._generation_lock:
                sess.active_response = True
                if action is not None and sess.speaker_segment is not None:
                    sess.speaker_segment.control_action = action.value
                if action == ControlAction.BACKCHANNEL:
                    sess.control_state = ControlState.BACKCHANNELING
                    if sess.speaker_segment is not None:
                        sess.speaker_segment.control_opened = True
                        sess.speaker_segment.control_action = ControlAction.BACKCHANNEL.value
                        sess.speaker_segment.control_frames_remaining = 4
            if unified_transition_lock is not None:
                unified_transition_lock.release()
            logger.info("[realtime %s] start %s audio_samples=%d auto=%s", sess.session_id[:8], response_id, committed_audio.size, sess.auto_response)
            playout: SpeakerPlayoutBuffer | None = None
            packetizer: NetworkPacketizer | None = None
            control: _RealtimeResponseControl | None = None
            future: asyncio.Future[Any] | None = None
            response_output_cancel = threading.Event()
            response_output_cancel_reason: dict[str, str | None] = {"value": None}
            try:
                audio_queue: queue.Queue[AudioPacket] = queue.Queue(maxsize=REALTIME_AUDIO_PACKET_CAPACITY)
                text_queue: queue.Queue[str] = queue.Queue(maxsize=REALTIME_TEXT_CHUNK_CAPACITY)
                playout = SpeakerPlayoutBuffer(delay_frames=sess.speaker_playout_delay_frames)
                playout.begin_segment(
                    generation_id=token.session_generation_id,
                    speaker_segment_id=token.speaker_segment_id,
                )
                packetizer = NetworkPacketizer(
                    sequence=packet_sequence,
                    packet_frames=NETWORK_PACKET_FRAMES,
                    sample_rate=CODEC_SAMPLE_RATE,
                )
                control = _RealtimeResponseControl(token, playout, packetizer, audio_queue, text_queue)
                with response_controls_lock:
                    response_controls[response_id] = control
                response_created_event: dict[str, Any] = {
                    "type": "response.created",
                    "response_id": response_id,
                    "response_created_at_ns": time.monotonic_ns(),
                }
                response_voice_conditioning = getattr(sess, "voice_conditioning", None)
                if isinstance(response_voice_conditioning, dict):
                    for field in (
                        "conditioning_source", "mode", "reference_wav_path",
                        "reference_wav_sha256",
                    ):
                        value = response_voice_conditioning.get(field)
                        if isinstance(value, str) and value:
                            response_created_event[
                                "voice_conditioning_mode" if field == "mode" else field
                            ] = value
                response_created_at_monotonic_ns = int(
                    response_created_event["response_created_at_ns"]
                )
                response_created_event["response_created_at_monotonic_ns"] = (
                    response_created_at_monotonic_ns
                )
                with sess._generation_lock:
                    listener_hidden_frame_id = sess.listener_hidden_frame_id
                    listener_hidden_at_ns = sess.listener_hidden_at_ns
                response_created_event["listener_hidden_frame_id"] = listener_hidden_frame_id
                response_created_event["listener_hidden_at_ns"] = listener_hidden_at_ns
                if listener_hidden_at_ns is not None:
                    response_created_event["listener_hidden_age_ms"] = round(
                        max(0, response_created_event["response_created_at_ns"] - int(listener_hidden_at_ns))
                        / 1_000_000.0,
                        3,
                    )
                diagnostic_talker_seed = getattr(engine, "diagnostic_talker_seed", None)
                if diagnostic_talker_seed is not None:
                    response_created_event["diagnostic_talker_seed"] = int(diagnostic_talker_seed)
                    response_created_event["diagnostic_rng_scope"] = "startup_global_not_per_response"
                if action is not None:
                    response_created_event["action"] = action.value
                    response_created_event["effective_action"] = action.value
                if intent_origin is not None:
                    response_created_event.update({
                        "intent_source_frame_id": int(intent_origin["source_frame_id"]),
                        "intent_endpoint_frame_id": int(intent_origin["endpoint_frame_id"]),
                        "intent_source": str(intent_origin["source"]),
                    })
                    for field in (
                        "policy_decision_frame_id",
                        "policy_decision_at_ns",
                        "policy_input_listener_frame_id",
                        "policy_input_listener_hidden_frame_id",
                        "policy_input_listener_hidden_at_ns",
                    ):
                        if intent_origin.get(field) is not None:
                            response_created_event[field] = int(intent_origin[field])
                    transcript_context = intent_origin.get("_input_transcript_context")
                    if isinstance(transcript_context, dict):
                        response_created_event.update({
                            "input_utterance_id": str(transcript_context.get("utterance_id", "")),
                            "input_first_frame_id": int(transcript_context.get("first_frame_id", 0)),
                            "input_last_frame_id": int(transcript_context.get("last_frame_id", 0)),
                        })
                await send_ws(response_created_event, token)
                if response_uses_direct_audio:
                    resolve_previous_audio_history_archive(response_id)
                if (
                    response_uses_direct_audio
                    and not direct_audio_v2_response
                    and action == ControlAction.SPEAK
                    and direct_audio_context is not None
                ):
                    archive_reservation_attempted["value"] = True
                    with sess._generation_lock:
                        archive_base_messages = [dict(message) for message in sess.history]
                    archive_pcm = np.asarray(
                        direct_audio_context["_direct_audio_pcm"], dtype=np.float32
                    ).reshape(-1).copy()
                    archive_id_holder["archive_id"] = sess.reserve_audio_history_archive(
                        pcm=archive_pcm,
                        first_frame_id=int(direct_audio_context["_direct_audio_first_frame_id"]),
                        last_frame_id=int(direct_audio_context["_direct_audio_last_frame_id"]),
                        base_messages=archive_base_messages,
                        source="post_response_asr",
                        response_created_at_monotonic_ns=response_created_at_monotonic_ns,
                        coverage=(
                            "partial" if bool(direct_audio_context.get("_direct_audio_active"))
                            else "complete"
                        ),
                    )
                    archive_snapshot = (
                        sess.get_audio_history_archive(int(archive_id_holder["archive_id"]))
                        if archive_id_holder["archive_id"] is not None else None
                    )
                    update_response_runtime_metrics(response_id, "audio_history_archive", {
                        "status": "reserved" if archive_id_holder["archive_id"] is not None else "skipped_backpressure",
                        "source": "post_response_asr",
                        "coverage": (
                            "partial" if bool(direct_audio_context.get("_direct_audio_active"))
                            else "complete"
                        ),
                        "response_created_at_monotonic_ns": response_created_at_monotonic_ns,
                        "reserved_at_monotonic_ns": (
                            archive_snapshot.get("reserved_at_monotonic_ns")
                            if isinstance(archive_snapshot, dict) else None
                        ),
                        "input_audio_samples": int(archive_pcm.size),
                        "input_first_frame_id": int(direct_audio_context["_direct_audio_first_frame_id"]),
                        "input_last_frame_id": int(direct_audio_context["_direct_audio_last_frame_id"]),
                    })
                    if archive_id_holder["archive_id"] is not None:
                        archive_cpu_submitted["value"] = start_audio_history_archive(
                            int(archive_id_holder["archive_id"]), response_id
                        )
                if controller_guided_say_response and action is not None:
                    policy_decision_at_ns = (
                        intent_origin.get("policy_decision_at_ns")
                        if isinstance(intent_origin, dict) else None
                    )
                    await send_ws({
                        "type": "response.action_selected",
                        "response_id": response_id,
                        "wake_action": action.value,
                        "response_action": action.value,
                        "deferred_until_endpoint": False,
                        "selection_source": "80ms_controller",
                        "response_protocol": CONTROLLER_GUIDED_SAY_V1,
                        **({"policy_decision_at_ns": int(policy_decision_at_ns)}
                           if isinstance(policy_decision_at_ns, int) else {}),
                    }, token)
                next_generated_frame_id = 0
                streamed_chunks = 0
                first_tts_audio_chunk_pcm_at_ns: int | None = None
                active_output_segment_id = {"value": str(token.speaker_segment_id)}
                def packetize_frames_as_packets(frames: list[PlayoutFrame]) -> list[AudioPacket]:
                    if packetizer is None:
                        raise RuntimeError("packetizer non inizializzato")
                    packets: list[AudioPacket] = []
                    for frame in frames:
                        packets.extend(packetizer.push(frame))
                    return packets

                def cancel_response_output(reason: str) -> None:
                    response_output_cancel.set()
                    with sess._generation_lock:
                        if sess.is_generation_current(token):
                            response_output_cancel_reason["value"] = str(reason)
                            sess.stop_requested = True

                def discard_fenced_output() -> bool:
                    return _discard_fenced_response_output(
                        sess,
                        token,
                        playout=playout,
                        audio_queue=audio_queue,
                        text_queue=text_queue,
                        output_cancel=response_output_cancel,
                        cancel_reason=response_output_cancel_reason,
                    )

                def queue_response_output(
                    item: Any,
                    *,
                    queue_kind: str,
                    output_queue: queue.Queue[Any],
                    burst_items: int,
                ) -> bool:
                    def try_put() -> bool | None:
                        with sess._generation_lock:
                            if response_output_cancel.is_set():
                                return False
                            if not sess.is_generation_current(token):
                                return False
                            try:
                                output_queue.put_nowait(item)
                            except queue.Full:
                                return None
                            return True

                    metrics = _put_response_queue_bounded(
                        try_put,
                        queue_kind=queue_kind,
                        burst_items=burst_items,
                        cancel_reason=(
                            str(response_output_cancel_reason["value"])
                            if response_output_cancel_reason["value"]
                            else "generation_fenced"
                        ),
                    )
                    update_response_runtime_metrics(response_id, "output_backpressure", metrics)
                    if metrics["status"] == "cancelled":
                        response_output_cancel.set()
                        return False
                    return True

                def on_audio_chunk_with_horizon(chunk: np.ndarray, conditioning_frame_id: int) -> None:
                    nonlocal next_generated_frame_id, first_tts_audio_chunk_pcm_at_ns
                    if first_tts_audio_chunk_pcm_at_ns is None:
                        # Same monotonic clock as Control6's policy_decision_at_ns.
                        # This latch is before playout, packetization, and WS relay.
                        first_tts_audio_chunk_pcm_at_ns = time.monotonic_ns()
                    array = np.asarray(chunk, dtype=np.float32)
                    packets_to_enqueue: list[AudioPacket] = []
                    def enqueue_ready() -> None:
                        nonlocal next_generated_frame_id
                        generated = playout.push_chunk(
                            array,
                            first_frame_id=next_generated_frame_id,
                            generation_id=token.session_generation_id,
                            speaker_segment_id=str(active_output_segment_id["value"]),
                        )
                        next_generated_frame_id += len(generated)
                        ready = playout.release_ready(conditioning_frame_id=int(conditioning_frame_id))
                        packets_to_enqueue.extend(packetize_frames_as_packets(ready))
                    if not sess.guarded_write(token, "audio", enqueue_ready):
                        return
                    burst_items = len(packets_to_enqueue)
                    for packet in packets_to_enqueue:
                        if not queue_response_output(
                            packet, queue_kind="audio", output_queue=audio_queue,
                            burst_items=burst_items,
                        ):
                            return

                def on_audio_chunk(chunk: np.ndarray) -> None:
                    # Compatibility callback for the legacy one-shot path.  The
                    # stateful path supplies the real V9 horizon explicitly.
                    horizon = max(-1, next_generated_frame_id - 1)
                    on_audio_chunk_with_horizon(chunk, horizon)

                def begin_audio_segment(segment_id: str) -> None:
                    nonlocal next_generated_frame_id
                    normalized_segment_id = str(segment_id)
                    if not normalized_segment_id:
                        raise ValueError("closed TTS segment requires an identity")
                    if normalized_segment_id == active_output_segment_id["value"]:
                        return
                    if playout is None or packetizer is None:
                        raise RuntimeError("closed TTS segment has no playout or packetizer")
                    playout.begin_segment(
                        generation_id=token.session_generation_id,
                        speaker_segment_id=normalized_segment_id,
                    )
                    active_output_segment_id["value"] = normalized_segment_id
                    next_generated_frame_id = 0

                def finish_audio_segment(
                    segment_id: str,
                    source_tokens: Sequence[Mapping[str, Any]],
                ) -> None:
                    normalized_segment_id = str(segment_id)
                    if normalized_segment_id != active_output_segment_id["value"]:
                        raise RuntimeError("closed TTS segment does not own the active playout")
                    if playout is None or packetizer is None:
                        raise RuntimeError("closed TTS segment has no playout or packetizer")
                    packets = packetize_frames_as_packets(
                        playout.flush_segment(
                            generation_id=token.session_generation_id,
                            speaker_segment_id=normalized_segment_id,
                        )
                    )
                    packets.extend(packetizer.flush_segment(
                        token.session_generation_id,
                        normalized_segment_id,
                    ))
                    burst_items = len(packets) + 1
                    for packet in packets:
                        if not queue_response_output(
                            packet,
                            queue_kind="audio",
                            output_queue=audio_queue,
                            burst_items=burst_items,
                        ):
                            return
                    marker = _ClosedTTSSegment(
                        speaker_segment_id=normalized_segment_id,
                        source_tokens=tuple(dict(row) for row in source_tokens),
                    )
                    queue_response_output(
                        marker,
                        queue_kind="audio",
                        output_queue=audio_queue,
                        burst_items=burst_items,
                    )

                def on_text_chunk(chunk: str) -> None:
                    cleaned_chunk = re.sub(r"<unused\d+>", " ", str(chunk))
                    if cleaned_chunk:
                        queue_response_output(
                            cleaned_chunk, queue_kind="text", output_queue=text_queue, burst_items=1,
                        )

                def record_emitted_reference(chunk: np.ndarray) -> bool:
                    """Put only current-generation audio sent to the client on self-listen."""
                    array = np.asarray(chunk, dtype=np.float32).reshape(-1)
                    with sess._generation_lock:
                        if not sess.is_generation_current(token):
                            return False
                        normalized_samples = int(round(array.size * USER_SAMPLE_RATE / CODEC_SAMPLE_RATE))
                        frame_samples = sess.assistant_reference_buffer.frame_samples
                        if normalized_samples <= 0 or normalized_samples % frame_samples:
                            logger.error("[realtime %s %s] emitted audio non allineato: %d samples", sess.session_id[:8], response_id, array.size)
                            return False
                        count = normalized_samples // frame_samples
                        first_id = sess.reserve_assistant_reference_frames(count)
                        sess.assistant_reference_buffer.append_chunk(
                            array,
                            first_frame_id=first_id,
                            emitted_at_ns=time.monotonic_ns(),
                            sample_rate=CODEC_SAMPLE_RATE,
                        )
                        return True

                async def emit_audio_packet(packet: AudioPacket) -> bool:
                    nonlocal streamed_chunks
                    server_emit_started_at_ns = time.monotonic_ns()
                    payload = {
                        "type": "response.audio.delta",
                        "delta": _wav_b64(packet.pcm),
                        "response_id": response_id,
                        "server_emit_started_at_ns": server_emit_started_at_ns,
                        "raw_source_audio_quality": _raw_source_audio_quality(packet.pcm),
                        "packet_id": packet.packet_id,
                        "sequence": packet.packet_id,
                        "generation_id": packet.generation_id,
                        "speaker_segment_id": packet.speaker_segment_id,
                        "first_frame_id": packet.first_frame_id,
                        "last_frame_id": packet.last_frame_id,
                        "start_sample": packet.start_sample,
                        "end_sample": packet.end_sample,
                        "sample_rate": CODEC_SAMPLE_RATE,
                    }
                    if not await send_ws(payload, token):
                        if not discard_fenced_output():
                            cancel_response_output("websocket_audio_send_failed")
                        return False
                    if discard_fenced_output():
                        return False
                    if not record_emitted_reference(packet.pcm):
                        discard_fenced_output()
                        return False
                    frame_count = packet.last_frame_id - packet.first_frame_id + 1
                    if not sess.record_emitted_audio(token, frame_count):
                        discard_fenced_output()
                        return False
                    with policy_causality_lock:
                        response_audio_admission.mark_audio_sent_locked(frame_count)
                        input_watermark_frame_id = response_audio_admission.input_watermark
                    with playback_ack_branches_lock:
                        ack_branch = playback_ack_branches.get(response_id)
                    if ack_branch is not None:
                        with ack_branch.get("spoken_prefix_lock", nullcontext()):
                            if ack_branch.get("input_audio_watermark_frame_id") is None:
                                ack_branch["input_audio_watermark_frame_id"] = input_watermark_frame_id
                            _record_emitted_pcm_packet(ack_branch, packet)
                    streamed_chunks += 1
                    return True

                async def emit_audio_queue_item(item: Any) -> bool:
                    if isinstance(item, AudioPacket):
                        return await emit_audio_packet(item)
                    if not isinstance(item, _ClosedTTSSegment):
                        raise TypeError("realtime audio queue contains an unsupported item")
                    with playback_ack_branches_lock:
                        ack_branch = playback_ack_branches.get(response_id)
                    if ack_branch is None:
                        return True
                    with ack_branch.get("spoken_prefix_lock", nullcontext()):
                        marker = _complete_client_pcm_segment(
                            ack_branch,
                            speaker_segment_id=item.speaker_segment_id,
                            token_packets=item.source_tokens,
                        )
                    return marker is None or await send_ws(marker, token)

                async def emit_text_chunk(chunk: str) -> bool:
                    sent = await send_ws(
                        {"type": "response.output_text.delta", "delta": chunk, "response_id": response_id},
                        token,
                    )
                    if not sent:
                        if not discard_fenced_output():
                            cancel_response_output("websocket_text_send_failed")
                        return False
                    if discard_fenced_output():
                        return False
                    if not sess.record_emitted_text(token, chunk):
                        discard_fenced_output()
                        return False
                    return True

                policy_decision_at_ns = (
                    int(intent_origin["policy_decision_at_ns"])
                    if isinstance(intent_origin, dict)
                    and intent_origin.get("policy_decision_at_ns") is not None
                    else None
                )

                def publish_thinker_metrics(values: dict[str, Any]) -> None:
                    metrics = dict(values)
                    if policy_decision_at_ns is not None:
                        metrics["policy_decision_at_ns"] = policy_decision_at_ns
                    update_response_runtime_metrics(response_id, "thinker_metrics", metrics)
                    event = {
                        "type": "response.thinker_metrics",
                        "response_id": response_id,
                        **metrics,
                    }
                    sent = asyncio.run_coroutine_threadsafe(
                        send_ws(event, token), response_event_loop
                    ).result(timeout=5.0)
                    if not sent:
                        raise RuntimeError("thinker timing event lost its response generation")

                def infer_ws():
                    nonlocal empty_asr_cooldown_until_frame
                    frame_residual_v2_response = bool(
                        direct_audio_v2_response
                        and bool(getattr(engine, "frame_residual_adapter_enabled", False))
                        and getattr(engine, "frame_residual_adapter", None) is not None
                    )
                    use_talk_context_pool = bool(
                        (controller_guided_say_response or frame_residual_v2_response)
                        and getattr(engine, "response_talk_context_pool", None) is not None
                    )
                    with _response_talk_context(
                        use_talk_context_pool,
                        session_id=sess.session_id,
                        response_id=response_id,
                        generation_id=int(token.session_generation_id),
                    ) as talk_context, _inference_execution_scope(use_talk_context_pool):
                        engine_synthesis_owner.claim(
                            sess.session_id,
                            response_id,
                            allow_concurrent=use_talk_context_pool,
                            cancel=(talk_context.cancel if talk_context is not None else None),
                        )
                        try:
                            if not sess.is_generation_current(token):
                                return {"text": "", "tool_calls": [], "wav": None, "cancelled": True}
                            with sess._generation_lock:
                                history = list(sess.history)
                            asr_metrics: dict[str, Any] = {}
                            thinker_audio: np.ndarray | None = None
                            if response_uses_direct_audio:
                                _require_direct_audio_context(direct_audio_context)
                                thinker_audio = np.asarray(
                                    direct_audio_context["_direct_audio_pcm"], dtype=np.float32
                                ).reshape(-1)
                                if not thinker_audio.size or not np.isfinite(thinker_audio).all():
                                    raise RuntimeError("direct_audio_context_invalid: PCM vuoto o non finito")
                                if (
                                    action != ControlAction.BACKCHANNEL
                                    and not direct_audio_v2_response
                                    and not archive_reservation_attempted["value"]
                                ):
                                    archive_reservation_attempted["value"] = True
                                    with sess._generation_lock:
                                        if not sess.is_generation_current(token):
                                            return {"text": "", "tool_calls": [], "wav": None, "cancelled": True}
                                        archive_id_holder["archive_id"] = sess.reserve_audio_history_archive(
                                            pcm=thinker_audio,
                                            first_frame_id=int(direct_audio_context["_direct_audio_first_frame_id"]),
                                            last_frame_id=int(direct_audio_context["_direct_audio_last_frame_id"]),
                                            base_messages=history,
                                            source="post_response_asr",
                                            response_created_at_monotonic_ns=response_created_at_monotonic_ns,
                                            coverage=(
                                                "partial" if bool(direct_audio_context.get("_direct_audio_active"))
                                                else "complete"
                                            ),
                                        )
                                    archive_snapshot = (
                                        sess.get_audio_history_archive(int(archive_id_holder["archive_id"]))
                                        if archive_id_holder["archive_id"] is not None else None
                                    )
                                    update_response_runtime_metrics(response_id, "audio_history_archive", {
                                        "status": "reserved" if archive_id_holder["archive_id"] is not None else "skipped_backpressure",
                                        "source": "post_response_asr",
                                        "coverage": (
                                            "partial" if bool(direct_audio_context.get("_direct_audio_active"))
                                            else "complete"
                                        ),
                                        "response_created_at_monotonic_ns": response_created_at_monotonic_ns,
                                        "reserved_at_monotonic_ns": (
                                            archive_snapshot.get("reserved_at_monotonic_ns")
                                            if isinstance(archive_snapshot, dict) else None
                                        ),
                                        "input_audio_samples": int(thinker_audio.size),
                                        "input_first_frame_id": int(direct_audio_context["_direct_audio_first_frame_id"]),
                                        "input_last_frame_id": int(direct_audio_context["_direct_audio_last_frame_id"]),
                                    })
                                    if archive_id_holder["archive_id"] is not None:
                                        archive_cpu_submitted["value"] = start_audio_history_archive(
                                            int(archive_id_holder["archive_id"]), response_id
                                        )
                            if committed_audio.size and action != ControlAction.BACKCHANNEL and not response_uses_direct_audio:
                                transcribe = getattr(engine, "transcribe_audio", None)
                                if not callable(transcribe):
                                    raise RuntimeError("engine.transcribe_audio non disponibile per il turno audio")
                                transcription = transcribe(committed_audio)
                                asr_metrics = (
                                    transcription.get("phase_timings_ms", {})
                                    if isinstance(transcription, dict) else {}
                                )
                                user_text = str(
                                    transcription.get("text", "") if isinstance(transcription, dict) else transcription
                                ).strip()
                                transcript_context = (
                                    intent_origin.get("_input_transcript_context")
                                    if isinstance(intent_origin, dict) else None
                                )
                                if isinstance(transcript_context, dict):
                                    publish_completed_input_transcript(
                                        user_text,
                                        utterance_id=str(transcript_context.get("utterance_id", "")),
                                        first_frame_id=int(transcript_context.get("first_frame_id", 0)),
                                        last_frame_id=int(transcript_context.get("last_frame_id", 0)),
                                        generation_id=int(token.session_generation_id),
                                        epoch_id=int(transcript_context.get("epoch_id", -1)),
                                        source="post_response_asr",
                                    )
                                update_response_runtime_metrics(response_id, "asr", asr_metrics)
                                if not user_text:
                                    with sess._generation_lock:
                                        if not sess.is_generation_current(token):
                                            return {"text": "", "tool_calls": [], "wav": None, "cancelled": True}
                                    # Empty ASR is not VAD: keep classifying every
                                    # tick, but briefly suppress repeat SPEAK->ASR
                                    # launches while the head continues receiving
                                    # causal audio/features.
                                    with policy_cooldown_lock:
                                        empty_asr_cooldown_until_frame = max(
                                            empty_asr_cooldown_until_frame,
                                            latest_policy_frame_id + EMPTY_ASR_SPEAK_COOLDOWN_FRAMES,
                                        )
                                    return {"text": "", "tool_calls": [], "wav": None,
                                            "skip_reason": "empty_transcript",
                                            "runtime_metrics": {"asr": asr_metrics}}
                                with sess._generation_lock:
                                    if not sess.is_generation_current(token):
                                        return {"text": "", "tool_calls": [], "wav": None, "cancelled": True}
                                    history = list(sess.history)
                                    history.append({"role": "user", "content": user_text})
                                    history = history[-16:]
                                    # Keep a committed user turn even if barge-in
                                    # cancels the subsequent Thinker/Talker work.
                                    sess.history = [dict(message) for message in history]
                                    sess.note_semantic_context_changed()
                            if action == ControlAction.SPEAK and not committed_audio.size and thinker_audio is None:
                                return {"text": "", "tool_calls": [], "wav": None,
                                        "skip_reason": "empty_input",
                                        "runtime_metrics": {"asr": asr_metrics}}
                            has_user_context = any(
                                message.get("role") == "user" and _message_text(message.get("content", "")).strip()
                                for message in history
                            )
                            if not has_user_context and action != ControlAction.BACKCHANNEL and thinker_audio is None:
                                return {"text": "", "tool_calls": [], "wav": None,
                                        "skip_reason": "empty_input"}
                            logger.info("[realtime %s %s] Thinker/Talker start (listener già attivo)", sess.session_id[:8], response_id)
                            result = run_response(
                                sess,
                                history,
                                {
                                    "modalities": ["text", "audio"],
                                    "tools": sess.tools,
                                    "voice": sess.voice,
                                    "voice_style": sess.voice_style,
                                    "language": sess.language,
                                    "ref_text": sess.ref_text,
                                    **_controller_guided_say_generation_options(
                                        action=action,
                                        controller_guided_say=controller_guided_say_response,
                                        direct_audio_candidate=response_uses_direct_audio,
                                        response_protocol=response_protocol,
                                    ),
                                    "_direct_audio_pcm": thinker_audio,
                                    "_direct_audio_context": (
                                        dict(direct_audio_context)
                                        if isinstance(direct_audio_context, dict)
                                        else None
                                    ),
                                    "_generation_id": token.session_generation_id,
                                    "_generation_token": token,
                                    "_unified_native_branch": unified_native_branch,
                                    "_response_cancel_event": response_output_cancel,
                                    "_on_thinker_metrics": publish_thinker_metrics,
                                    "_talk_context": talk_context,
                                    "_input_transcript_context": (
                                        dict(intent_origin["_input_transcript_context"])
                                        if isinstance(intent_origin, dict)
                                        and isinstance(intent_origin.get("_input_transcript_context"), dict)
                                        else None
                                    ),
                                    "_on_input_transcript_completed": (
                                        lambda heard, context: publish_completed_input_transcript(
                                            heard,
                                            utterance_id=str(context.get("utterance_id", "")),
                                            first_frame_id=int(context.get("first_frame_id", 0)),
                                            last_frame_id=int(context.get("last_frame_id", 0)),
                                            generation_id=int(token.session_generation_id),
                                            epoch_id=int(context.get("epoch_id", -1)),
                                            source="heard_action_say_v2",
                                        )
                                    ),
                                    "_on_audio_response_v2_parsed": (
                                        lambda parsed, heard_truncated: handle_v2_response_action(
                                            parsed,
                                            heard_truncated,
                                            base_messages=history,
                                        )
                                        if direct_audio_v2_response else None
                                    ),
                                    "_defer_history": True,
                                    "_stop_check": lambda: (
                                        sess.stop_requested or response_output_cancel.is_set()
                                    ),
                                    "_on_audio_chunk": on_audio_chunk,
                                    "_on_audio_chunk_with_horizon": on_audio_chunk_with_horizon,
                                    "_begin_audio_segment": begin_audio_segment,
                                    "_finish_audio_segment": finish_audio_segment,
                                    "_on_text_chunk": on_text_chunk,
                                    "_on_thinker_token_id": lambda token_id: record_thinker_token_id(response_id, token_id),
                                    "_listener_admission_wait": (
                                        response_listener_admission_wait
                                        if response_protocol in {AUDIO_RESPONSE_PROTOCOL_V2, CONTROLLER_GUIDED_SAY_V1}
                                        else None
                                    ),
                                    "_duplex_realtime": True,
                                    "_runtime_metrics_callback": lambda phase, values: update_response_runtime_metrics(response_id, phase, values),
                                },
                                action=action.value if action is not None else None,
                                history_is_committed=True,
                            )
                            if archive_id_holder["archive_id"] is not None:
                                if (
                                    direct_audio_v2_response
                                    and not audio_history_archive_cpu_enabled
                                    and not archive_heard_finalized["value"]
                                ):
                                    archive_heard_finalized["value"] = sess.complete_audio_history_archive(
                                        int(archive_id_holder["archive_id"]),
                                        "",
                                    )
                                    archive_assistant_finalized["value"] = bool(
                                        archive_heard_finalized["value"]
                                    )
                            if (
                                response_uses_direct_audio
                                and not direct_audio_v2_response
                                and action == ControlAction.BACKCHANNEL
                            ):
                                result["_skip_history_commit"] = True
                            if committed_audio.size and asr_metrics:
                                result.setdefault("runtime_metrics", {})["asr"] = asr_metrics
                            logger.info("[realtime %s %s] Thinker/Talker done elapsed=%.2fs", sess.session_id[:8], response_id, time.perf_counter() - started)
                            return result
                        finally:
                            engine_synthesis_owner.release(sess.session_id, response_id)
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(None, infer_ws)
                deadline = loop.time() + 180.0
                while not future.done():
                    if loop.time() >= deadline:
                        with sess._generation_lock:
                            sess.stop_requested = True
                        sess.invalidate_generation("realtime_timeout")
                        _stop_owned_synthesis(sess.session_id, response_id)
                        logger.error("[realtime %s %s] timeout dopo %.1fs", sess.session_id[:8], response_id, time.perf_counter() - started)
                        await send_ws({"type": "error", "error": {"code": "INFERENCE_TIMEOUT", "message": "risposta oltre 180 secondi"}, "response_id": response_id})
                        return
                    try:
                        chunk_wav = audio_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.005)
                    else:
                        if not await emit_audio_queue_item(chunk_wav):
                            return
                    try:
                        chunk_text = text_queue.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        if not await emit_text_chunk(chunk_text):
                            return
                result = await future
                current = sess.is_generation_current(token)
                yielded = current and sess.stop_reason == ControlAction.YIELD.value
                if yielded:
                    with response_runtime_metrics_lock:
                        trace = response_runtime_metrics.setdefault(response_id, {}).get("generation_trace")
                        if isinstance(trace, dict):
                            trace["terminal_reason"] = (
                                "yielded_before_first_token"
                                if not trace.get("thinker_token_ids")
                                else "yielded_after_token"
                            )
                if current and result.get("skip_reason"):
                    playout.cancel()
                    _drain_queue(audio_queue)
                    _drain_queue(text_queue)
                    cancelled_event = {
                        "type": "response.cancelled", "reason": result["skip_reason"],
                        "response_id": response_id,
                        "runtime_metrics": runtime_metrics_event(response_id, result),
                    }
                    await send_ws(
                        cancelled_event, token,
                    )
                    return
                if not current:
                    discard_fenced_output()
                    if _claim_response_cancel_notification(sess, response_id):
                        with sess._generation_lock:
                            cancellation_generation = int(sess.session_generation_id)
                        await send_ws({
                            "type": "response.cancelled",
                            "reason": "generation_replaced_before_completion",
                            "response_id": response_id,
                            "runtime_metrics": runtime_metrics_event(response_id),
                        }, generation_id=cancellation_generation)
                    return
                if result.get("cancelled") and not yielded:
                    # A cooperative cancellation must not flush pending output.
                    playout.cancel()
                    _drain_queue(audio_queue)
                    _drain_queue(text_queue)
                    if current and not sess.response_cancel_notified:
                        await send_ws({"type": "response.cancelled", "reason": "cooperative_stop",
                                       "response_id": response_id,
                                       "runtime_metrics": runtime_metrics_event(response_id, result)}, token)
                    return
                yielded_pcm_branch = None
                if yielded and direct_audio_v2_response:
                    with playback_ack_branches_lock:
                        candidate_branch = playback_ack_branches.get(response_id)
                    if (
                        isinstance(candidate_branch, dict)
                        and candidate_branch.get("response_id") == response_id
                        and candidate_branch.get("session_generation_id")
                        == int(token.session_generation_id)
                        and candidate_branch.get("speaker_segment_id")
                        == str(token.speaker_segment_id)
                        and _response_branch_has_pending_pcm(
                            candidate_branch,
                            speaker_segment_id=str(token.speaker_segment_id),
                        )
                    ):
                        yielded_pcm_branch = candidate_branch
                if yielded_pcm_branch is not None:
                    # Only fully closed, text-owned TTS units may survive YIELD.
                    # Keep queued packets through the last segment marker and
                    # discard any PCM from an unfinished unit without flushing it.
                    queued_audio: list[Any] = []
                    while True:
                        try:
                            queued_audio.append(audio_queue.get_nowait())
                        except queue.Empty:
                            break
                    last_closed_index = max(
                        (
                            index for index, item in enumerate(queued_audio)
                            if isinstance(item, _ClosedTTSSegment)
                        ),
                        default=-1,
                    )
                    for item in queued_audio[: last_closed_index + 1]:
                        if not await emit_audio_queue_item(item):
                            return
                    active_segment = str(active_output_segment_id["value"])
                    if playout is not None:
                        playout.cancel_segment(
                            generation_id=token.session_generation_id,
                            speaker_segment_id=active_segment,
                        )
                    if packetizer is not None:
                        packetizer.cancel_segment(
                            token.session_generation_id,
                            active_segment,
                        )
                    _drain_queue(text_queue)
                    cancel_disposition = _claim_pending_yielded_v2_pcm_cancel(
                        sess,
                        yielded_pcm_branch,
                        response_id=response_id,
                        generation_id=int(token.session_generation_id),
                        speaker_segment_id=str(token.speaker_segment_id),
                    )
                    if cancel_disposition in {
                        "claimed",
                        "already_notified",
                        "already_cancelled",
                    }:
                        if cancel_disposition == "claimed":
                            await send_ws({
                                "type": "response.cancelled",
                                "reason": "yield",
                                "response_id": response_id,
                                "generation_id": token.session_generation_id,
                                "runtime_metrics": runtime_metrics_event(response_id, result),
                            }, token)
                        return
                # YIELD is different: release accepted packets before flush output.
                while True:
                    try:
                        chunk_wav = audio_queue.get_nowait()
                    except queue.Empty:
                        break
                    if not await emit_audio_queue_item(chunk_wav):
                        return
                if discard_fenced_output():
                    return
                for packet in packetize_frames_as_packets(
                    playout.flush_segment(
                        generation_id=token.session_generation_id,
                        speaker_segment_id=token.speaker_segment_id,
                    )
                ):
                    if not await emit_audio_packet(packet):
                        return
                if packetizer is None:
                    raise RuntimeError("packetizer non inizializzato")
                for packet in packetizer.flush_segment(
                    token.session_generation_id, token.speaker_segment_id,
                ):
                    if not await emit_audio_packet(packet):
                        return
                while True:
                    try:
                        chunk_text = text_queue.get_nowait()
                    except queue.Empty:
                        break
                    if not await emit_text_chunk(chunk_text):
                        return
                if discard_fenced_output():
                    return
                if result.get("cancelled") and not yielded:
                    if not sess.response_cancel_notified:
                        await send_ws({"type": "response.cancelled", "reason": "cooperative_stop",
                                       "response_id": response_id,
                                       "runtime_metrics": runtime_metrics_event(response_id, result)}, token)
                    return
                if result["tool_calls"]:
                    for call in result["tool_calls"]:
                        fn = call["function"]
                        await send_ws({"type": "response.function_call_arguments.done", "call_id": call["id"], "name": fn["name"], "arguments": fn["arguments"]}, token)
                    return
                if not sess.speaker_segment or not sess.speaker_segment.emitted_text:
                    if result["text"] and not await emit_text_chunk(result["text"]):
                        return
                result_text = re.sub(r"\s*<unused\d+>\s*", " ", str(result.get("text", "")))
                result_text = re.sub(r"[ \t]+", " ", result_text).strip()
                if (
                    not response_uses_direct_audio
                    and not result.get("_skip_history_commit")
                    and archive_id_holder["archive_id"] is None
                ):
                    sess.commit_history(token, result.get("history_messages", []), result_text)
                if result.get("wav") is not None and streamed_chunks == 0:
                    # A complete legacy WAV is still transported through the
                    # same atomic-frame/packetizer path; it is not chunked by a
                    # fixed wall-clock duration.
                    for packet in iter_audio_packets(
                        result["wav"],
                        generation_id=token.session_generation_id,
                        speaker_segment_id=token.speaker_segment_id,
                        sequence=packet_sequence,
                        packet_frames=NETWORK_PACKET_FRAMES,
                        sample_rate=CODEC_SAMPLE_RATE,
                    ):
                        if not await emit_audio_packet(packet):
                            return
                response_done_at_monotonic_ns = time.monotonic_ns()
                with playback_ack_branches_lock:
                    ack_branch = playback_ack_branches.get(response_id)
                if ack_branch is not None:
                    response_audio_complete = bool(
                        not result.get("cancelled")
                        and not yielded
                        and result.get("status") not in {"no_speech", "deferred_until_endpoint"}
                    )
                    if response_audio_complete:
                        # Until the live SAY producer exposes closed text-owned
                        # TTS units, this response-level segment deliberately
                        # carries no Gemma token ownership. The marker is still
                        # useful to the client playback queue, but an ACK cannot
                        # advance the canonical KV prefix without that mapping.
                        with ack_branch.get("spoken_prefix_lock", nullcontext()):
                            marker = _complete_client_pcm_segment(
                                ack_branch,
                                speaker_segment_id=str(token.speaker_segment_id),
                            )
                        if marker is not None and not await send_ws(marker, token):
                            return
                    update_response_runtime_metrics(response_id, "client_consumed_pcm", {
                        "alignment": "complete_closed_tts_unit_only",
                        "text_owned_tts_units_available": False,
                        "logical_client_consumption_only": True,
                        "confirmed_segments": list(ack_branch.get("client_consumed_segments", ())),
                        "confirmed_exclusive_kv_cursor": ack_branch.get(
                            "spoken_prefix_exclusive_cursor"
                        ),
                    })
                if response_uses_direct_audio and not archive_assistant_finalized["value"]:
                    with sess._generation_lock:
                        segment = sess.speaker_segment
                        emitted_assistant_text = (
                            str(segment.emitted_text).strip()
                            if segment is not None
                            and segment.segment_id == token.speaker_segment_id
                            else ""
                        )
                        emitted_audio_frames = (
                            int(segment.emitted_audio_frames)
                            if segment is not None
                            and segment.segment_id == token.speaker_segment_id
                            else 0
                        )
                    v2_metrics = (
                        (result.get("runtime_metrics") or {}).get("audio_response_protocol")
                        if isinstance(result.get("runtime_metrics"), dict) else None
                    )
                    unified_speak_valid = bool(
                        not unified_kv_enabled
                        or (
                            action == ControlAction.SPEAK
                            and isinstance(v2_metrics, dict)
                            and v2_metrics.get("response_protocol") == AUDIO_RESPONSE_PROTOCOL_V2
                            and v2_metrics.get("status") == "valid"
                            and str(v2_metrics.get("response_action", "")).upper()
                            == ControlAction.SPEAK.value
                            and not result.get("_skip_history_commit")
                        )
                    )
                    assistant_completed = bool(
                        emitted_assistant_text
                        and emitted_audio_frames > 0
                        and not result.get("cancelled")
                        and not result.get("tool_calls")
                        and not yielded
                        and result.get("status") != "no_speech"
                        and unified_speak_valid
                    )
                    assistant_status = (
                        "completed" if assistant_completed
                        else "cancelled" if result.get("cancelled") or yielded
                        else "absent"
                    )
                    committed_assistant_text = (
                        emitted_assistant_text if assistant_completed else ""
                    )
                    archive_id_value = archive_id_holder["archive_id"]
                    if assistant_completed:
                        archived_assistant = sess.commit_completed_assistant_response(
                            token,
                            response_id,
                            committed_assistant_text,
                            archive_id=(
                                None if archive_id_value is None else int(archive_id_value)
                            ),
                            user_onset_frame_id=user_turn_onset_frame_id,
                            response_done_at_monotonic_ns=response_done_at_monotonic_ns,
                        )
                    elif archive_id_value is not None:
                        archived_assistant = sess.update_audio_history_archive_assistant(
                            int(archive_id_value),
                            "",
                            status=assistant_status,
                            response_done_at_monotonic_ns=response_done_at_monotonic_ns,
                        )
                    else:
                        archived_assistant = False
                    archive_assistant_finalized["value"] = bool(archived_assistant)
                    if unified_native_branch is not None:
                        unified_native_branch["history_committed"] = bool(
                            assistant_completed
                            and archived_assistant
                            and unified_speak_valid
                        )
                    update_response_runtime_metrics(response_id, "audio_history_archive_assistant", {
                        "status": assistant_status,
                        "assistant_text_committed": bool(archived_assistant and committed_assistant_text),
                        "response_done_at_monotonic_ns": response_done_at_monotonic_ns,
                    })
                done_event = {
                    "type": "response.done",
                    "response_id": response_id,
                    "response_done_at_monotonic_ns": response_done_at_monotonic_ns,
                    "runtime_metrics": runtime_metrics_event(response_id, result),
                }
                if first_tts_audio_chunk_pcm_at_ns is not None:
                    done_event["t_first_audio_chunk_pcm_at_ns"] = int(
                        first_tts_audio_chunk_pcm_at_ns
                    )
                if result.get("status") == "no_speech":
                    done_event["status"] = "no_speech"
                    done_event["audio_samples"] = 0
                elif result.get("status") == "deferred_until_endpoint":
                    done_event["status"] = "deferred_until_endpoint"
                response_done_sent = await send_ws(done_event, token)
                if (
                    response_done_sent
                    and action == ControlAction.SPEAK
                    and user_turn_onset_frame_id is not None
                    and streamed_chunks > 0
                    and result.get("status") != "no_speech"
                    and not result.get("cancelled")
                    and not yielded
                ):
                    user_turn_completed = sess.complete_user_onset_response(
                        user_turn_onset_frame_id
                    )
                logger.info("[realtime %s %s] done total=%.2fs audio_chunks=%d", sess.session_id[:8], response_id, time.perf_counter() - started, streamed_chunks)
                if result.get("status") != "deferred_until_endpoint":
                    with sess._generation_lock:
                        with sess._pending_audio_lock:
                            queued_audio = sess.pending_audio.copy()
                            if action == ControlAction.BACKCHANNEL:
                                sess.reset_backchannel_response()
                            elif action == ControlAction.SPEAK:
                                preserved_pending_pin = sess.reset_speak_response(
                                    consumed_user_onset_frame_id=user_turn_onset_frame_id,
                                )
                                logger.info(
                                    "[realtime %s %s] SPEAK audio cleanup owner=%s pending=%s newer_pin_preserved=%s",
                                    sess.session_id[:8],
                                    response_id,
                                    user_turn_onset_frame_id,
                                    sess.pending_user_onset_frame_id,
                                    preserved_pending_pin,
                                )
                            else:
                                sess.reset_turn()
                            sess.pending_audio = queued_audio
            except asyncio.CancelledError:
                with sess._generation_lock:
                    sess.stop_requested = True
                sess.invalidate_generation("realtime_task_cancelled")
                _stop_owned_synthesis(sess.session_id, response_id)
                if future is not None and not future.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(future), timeout=2.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        logger.warning("[realtime %s %s] worker non terminato entro cleanup", sess.session_id[:8], response_id)
                    except Exception:
                        logger.exception("[realtime %s %s] worker cleanup failed", sess.session_id[:8], response_id)
                raise
            except ResponseOutputQueueFull as exc:
                response_output_cancel.set()
                with sess._generation_lock:
                    if sess.is_generation_current(token):
                        sess.stop_requested = True
                output_metrics = {
                    "status": "timeout", "queue_kind": exc.queue_kind,
                    "burst_items": exc.burst_items, "wait_ms": round(exc.wait_ms, 3),
                    "attempts": exc.attempts, "cancel_reason": "bounded_queue_timeout",
                }
                update_response_runtime_metrics(response_id, "output_backpressure", output_metrics)
                if playout is not None:
                    playout.cancel()
                _drain_queue(audio_queue)
                _drain_queue(text_queue)
                logger.error(
                    "[realtime %s %s] outbound %s queue saturated burst=%d wait_ms=%.1f attempts=%d",
                    sess.session_id[:8], response_id, exc.queue_kind,
                    exc.burst_items, exc.wait_ms, exc.attempts,
                )
                await send_ws({
                    "type": "error",
                    "error": {
                        "code": "OUTPUT_OVERLOAD",
                        "message": (
                            f"Realtime {exc.queue_kind} output remained backlogged for "
                            f"{exc.wait_ms:.0f} ms; response cancelled"
                        ),
                    },
                    "response_id": response_id,
                    "runtime_metrics": runtime_metrics_event(response_id),
                }, token)
            except queue.Full:
                logger.exception(
                    "[realtime %s %s] unexpected outbound queue saturation",
                    sess.session_id[:8], response_id,
                )
                await send_ws({
                    "type": "error",
                    "error": {
                        "code": "OUTPUT_OVERLOAD",
                        "message": "Realtime output queue unexpectedly exceeded capacity; response cancelled",
                    },
                    "response_id": response_id,
                    "runtime_metrics": runtime_metrics_event(response_id),
                }, token)
            except _PrivateControlLeakError as exc:
                with response_runtime_metrics_lock:
                    trace = response_runtime_metrics.setdefault(response_id, {}).get("generation_trace")
                    if isinstance(trace, dict):
                        trace["outcome"] = "private_control_leak"
                        trace["raw_text"] = exc.raw_text
                if playout is not None:
                    playout.cancel()
                _drain_queue(audio_queue)
                _drain_queue(text_queue)
                if packetizer is not None:
                    packetizer.cancel_all()
                with sess._generation_lock:
                    current = sess.is_generation_current(token)
                    if current:
                        sess.stop_requested = True
                        sess.response_cancel_notified = True
                if current:
                    await send_ws({
                        "type": "error",
                        "error": {
                            "code": "PRIVATE_CONTROL_LEAK",
                            "message": "Generated private control event; response rejected before speech synthesis.",
                        },
                        "response_id": response_id,
                    }, token)
                    await send_ws({
                        "type": "response.cancelled",
                        "reason": "private_control_leak",
                        "response_id": response_id,
                        "runtime_metrics": runtime_metrics_event(response_id),
                    }, token)
                logger.warning(
                    "[realtime %s %s] rejected generated private control marker action=%s raw_chars=%d",
                    sess.session_id[:8],
                    response_id,
                    exc.action,
                    len(exc.raw_text),
                )
            except Exception as exc:
                logger.exception("realtime response failed")
                error_detail: dict[str, Any] = {
                    "code": "INFERENCE_ERROR",
                    "message": str(exc)[:240],
                }
                terminal_reason = getattr(exc, "terminal_reason", None)
                terminal_token_id = getattr(exc, "terminal_token_id", None)
                if isinstance(terminal_reason, str):
                    error_detail["terminal_reason"] = terminal_reason
                if isinstance(terminal_token_id, int):
                    error_detail["terminal_token_id"] = int(terminal_token_id)
                error_payload: dict[str, Any] = {
                    "type": "error",
                    "error": error_detail,
                    "response_id": response_id,
                }
                failure_metrics = getattr(exc, "runtime_metrics", None)
                if isinstance(failure_metrics, dict):
                    error_payload["runtime_metrics"] = runtime_metrics_event(
                        response_id,
                        {"runtime_metrics": {"thinker": failure_metrics}},
                    )
                else:
                    error_payload["runtime_metrics"] = runtime_metrics_event(response_id)
                await send_ws(error_payload, token)
                if not response_done_sent:
                    await send_ws(
                        _realtime_inference_failure_done_event(
                            response_id,
                            runtime_metrics_event(response_id),
                        ),
                        token,
                    )
            finally:
                if user_turn_onset_frame_id is not None and not user_turn_completed:
                    sess.cancel_user_onset_response(user_turn_onset_frame_id)
                if archive_id_holder["archive_id"] is not None:
                    archive_id = int(archive_id_holder["archive_id"])
                    if (
                        direct_audio_v2_response
                        and not audio_history_archive_cpu_enabled
                        and not archive_heard_finalized["value"]
                    ):
                        archive_heard_finalized["value"] = sess.complete_audio_history_archive(
                            archive_id,
                            "",
                        )
                    # A cancelled/failed response has no verified transcript;
                    # complete it as empty so session history stays unchanged.
                    if not archive_assistant_finalized["value"]:
                        spoken_prefix = ""
                        _spoken_position = None
                        if direct_audio_v2_response and unified_native_branch is not None:
                            _spoken_position = unified_native_branch.get(
                                "spoken_prefix_exclusive_cursor"
                            )
                            spoken_prefix = str(
                                unified_native_branch.get("spoken_prefix_text", "")
                            )
                            if int(unified_native_branch.get("client_consumed_pcm_sequence", -1)) < 0:
                                unified_native_branch["client_playback_limitation"] = (
                                    "no client_consumed_pcm ACK; only the last confirmed complete-segment prefix is retained"
                                )
                                update_response_runtime_metrics(response_id, "client_consumed_pcm", {
                                    "status": "no_playback_ack",
                                    "prefix_policy": "retain_last_confirmed_prefix_only",
                                    "logical_client_consumption_only": True,
                                })
                        if spoken_prefix:
                            assistant_status = "partial"
                        else:
                            with sess._generation_lock:
                                assistant_status = (
                                    "cancelled"
                                    if sess.stop_requested
                                    or sess.stop_reason in {
                                        ControlAction.YIELD.value,
                                        ControlAction.REPLAN.value,
                                        ControlAction.HARD_STOP.value,
                                    }
                                    else "absent"
                                )
                        archive_assistant_finalized["value"] = sess.update_audio_history_archive_assistant(
                            archive_id,
                            spoken_prefix,
                            status=assistant_status,
                        )
                        update_response_runtime_metrics(response_id, "audio_history_archive_assistant", {
                            "status": assistant_status,
                            "assistant_text_committed": bool(spoken_prefix),
                            "spoken_prefix_kv_position": (
                                _spoken_position if spoken_prefix else None
                            ),
                        })
                # A failed response ends this worker, not the input/listener
                # generation. Invalidating the session here can race a valid
                # scheduler_listener callback and drop its listener output and
                # consumed ACK even though the mic frame was processed.
                # Response-owned playback queues still get cleared below.
                if playout is not None:
                    playout.cancel()
                if packetizer is not None:
                    packetizer.cancel_all()
                with response_controls_lock:
                    if response_controls.get(response_id) is control:
                        response_controls.pop(response_id, None)
                if unified_native_branch is not None:
                    if _response_branch_has_pending_pcm(unified_native_branch):
                        # Keep the shared queue and response ownership open until
                        # emitted PCM reaches its exact client-consumed frontier.
                        # The independent receive loop remains free to process ACKs.
                        finalize_task = asyncio.create_task(
                            finalize_unified_native_cache_after_playback()
                        )
                        unified_native_branch["deferred_playback_finalizer"] = finalize_task
                    else:
                        finalize_task = asyncio.create_task(finalize_unified_native_cache(
                            history_committed=bool(unified_native_branch.get("history_committed"))
                        ))
                        try:
                            await asyncio.shield(finalize_task)
                        except asyncio.CancelledError:
                            try:
                                await asyncio.shield(finalize_task)
                            except Exception:
                                logger.exception(
                                    "[realtime %s %s] unified native cache cleanup failed during cancellation",
                                    sess.session_id[:8],
                                    response_id,
                                )
                        except Exception as exc:
                            logger.exception(
                                "[realtime %s %s] unified native cache finalization failed",
                                sess.session_id[:8],
                                response_id,
                            )
                            fail_native_listener_output_health(
                                -1,
                                {
                                    "status": "unified_cache_finalization_failed",
                                    "error": str(exc),
                                },
                            )
                            if not unified_native_branch.get("ended_response"):
                                sess.end_response(token)
                                unified_native_branch["ended_response"] = True
                elif not (unified_native_branch and unified_native_branch.get("ended_response")):
                    sess.end_response(token)
                if native_partial_worker is not None and native_partial_asr_enabled.is_set():
                    native_partial_worker.resume()

        def launch_policy_response(
            action: ControlAction,
            committed_audio: np.ndarray,
            generation_id: int,
            speaker_segment_id: str | None,
            replan_cancel_event: threading.Event | None = None,
            replan_endpoint_waiter: threading.Event | None = None,
            intent_origin: dict[str, Any] | None = None,
        ) -> None:
            nonlocal response_task, replan_handoff_pending
            if replan_cancel_event is not None and not replan_cancel_event.is_set():
                async def wait_for_cancellation_event() -> None:
                    while not closing and not replan_cancel_event.is_set():
                        await asyncio.sleep(0.001)
                    if not closing and replan_cancel_event.is_set():
                        launch_policy_response(
                            action,
                            committed_audio,
                            generation_id,
                            speaker_segment_id,
                            None,
                            replan_endpoint_waiter,
                            intent_origin,
                        )
                asyncio.create_task(wait_for_cancellation_event())
                return
            with sess._generation_lock:
                if (
                    closing
                    or not sess.policy_response_pending
                    or not sess.is_generation_current(
                        generation_id=generation_id,
                        speaker_segment_id=speaker_segment_id,
                    )
                ):
                    return
            if response_task is not None and not response_task.done():
                async def wait_for_previous() -> None:
                    while response_task is not None and not response_task.done() and not closing:
                        await asyncio.sleep(0.005)
                    if action == ControlAction.REPLAN:
                        with sess._generation_lock:
                            if (
                                not closing
                                and sess.is_generation_current(
                                    generation_id=generation_id,
                                    speaker_segment_id=speaker_segment_id,
                                )
                            ):
                                # end_response clears this flag on the stale
                                # worker; restore it only for the queued replan.
                                sess.policy_response_pending = True
                    launch_policy_response(
                        action,
                        committed_audio,
                        generation_id,
                        speaker_segment_id,
                        None,
                        replan_endpoint_waiter,
                        intent_origin,
                    )
                asyncio.create_task(wait_for_previous())
                return
            if replan_endpoint_waiter is not None:
                if not replan_endpoint_waiter.is_set():
                    async def wait_for_correction_endpoint() -> None:
                        endpoint_intent_origin = intent_origin
                        while not closing and not replan_endpoint_waiter.is_set():
                            await asyncio.sleep(0.005)
                        if closing or not replan_endpoint_waiter.is_set():
                            return
                        with replan_endpoint_lock:
                            endpoint_audio = replan_audio_snapshots.pop(
                                generation_id,
                                np.empty(0, dtype=np.float32),
                            )
                            endpoint_context = replan_audio_contexts.pop(generation_id, None)
                            replan_endpoint_waiters.pop(generation_id, None)
                        if isinstance(endpoint_context, dict):
                            if endpoint_context.get("_direct_audio_pcm") is not None:
                                endpoint_audio = np.asarray(
                                    endpoint_context["_direct_audio_pcm"], dtype=np.float32
                                )
                            endpoint_intent_origin = {**(endpoint_intent_origin or {}), **endpoint_context}
                            endpoint_intent_origin["endpoint_frame_id"] = int(
                                endpoint_context.get(
                                    "_direct_audio_last_frame_id",
                                    endpoint_context.get(
                                        "endpoint_frame_id",
                                        endpoint_intent_origin.get("endpoint_frame_id", 0),
                                    ),
                                )
                            )
                        with sess._generation_lock:
                            if (
                                closing
                                or not sess.is_generation_current(
                                    generation_id=generation_id,
                                    speaker_segment_id=speaker_segment_id,
                                )
                            ):
                                return
                            if endpoint_audio.size:
                                with sess._pending_audio_lock:
                                    if sess.pending_audio.size >= endpoint_audio.size:
                                        sess.pending_audio = sess.pending_audio[endpoint_audio.size:]
                        launch_policy_response(
                            action,
                            endpoint_audio,
                            generation_id,
                            speaker_segment_id,
                            None,
                            None,
                            endpoint_intent_origin,
                        )
                    asyncio.create_task(wait_for_correction_endpoint())
                    return
                with replan_endpoint_lock:
                    committed_audio = replan_audio_snapshots.pop(
                        generation_id,
                        np.empty(0, dtype=np.float32),
                    )
                    endpoint_context = replan_audio_contexts.pop(generation_id, None)
                    replan_endpoint_waiters.pop(generation_id, None)
                if isinstance(endpoint_context, dict):
                    if endpoint_context.get("_direct_audio_pcm") is not None:
                        committed_audio = np.asarray(
                            endpoint_context["_direct_audio_pcm"], dtype=np.float32
                        )
                    intent_origin = {**(intent_origin or {}), **endpoint_context}
                    intent_origin["endpoint_frame_id"] = int(
                        endpoint_context.get(
                            "_direct_audio_last_frame_id",
                            intent_origin.get("endpoint_frame_id", 0),
                        )
                    )
            if action == ControlAction.REPLAN:
                with sess._generation_lock:
                    if (
                        not closing
                        and sess.is_generation_current(
                            generation_id=generation_id,
                            speaker_segment_id=speaker_segment_id,
                        )
                        and not sess.active_response
                    ):
                        # REPLAN changes control/listening state; it is not a
                        # renderable response action. The old response worker
                        # may have cleared this flag while draining, so release
                        # the handoff here after cancellation/endpoint wait.
                        sess.policy_response_pending = False
                        if sess.control_state == ControlState.STOPPING:
                            sess.control_state = ControlState.LISTENING
                with policy_causality_lock:
                    replan_handoff_pending = False
                logger.info(
                    "[realtime %s] REPLAN handoff complete; listening for next renderable action",
                    sess.session_id[:8],
                )
                return
            with sess._generation_lock:
                if not _response_admission_allowed_locked(
                    sess, automatic=True, closing=closing
                ):
                    return
                if closing or sess.active_response or not sess.policy_response_pending or not sess.is_generation_current(
                    generation_id=generation_id,
                    speaker_segment_id=speaker_segment_id,
                ):
                    return
                user_turn_onset = (
                    int(intent_origin["_user_turn_onset_frame_id"])
                    if action == ControlAction.SPEAK
                    and isinstance(intent_origin, dict)
                    and intent_origin.get("_user_turn_onset_frame_id") is not None
                    else None
                )
                if user_turn_onset is not None and not sess.reserve_user_onset_for_response(
                    user_turn_onset
                ):
                    sess.policy_response_pending = False
                    if sess.control_state == ControlState.SPEAKING:
                        sess.control_state = ControlState.LISTENING
                    logger.info(
                        "[realtime %s] SPEAK reservation rejected for superseded onset=%d",
                        sess.session_id[:8], user_turn_onset,
                    )
                    return
            response_task = asyncio.create_task(
                produce_response(
                    committed_audio,
                    action=action,
                    intent_origin=intent_origin,
                    automatic_policy_response=True,
                )
            )

        try:
            await send_ws({
                "type": "session.created",
                "session_id": sess.session_id,
                "input_audio_buffer_capacity_frames": sess.audio_ingress.max_frames,
                "runtime_metrics": session_runtime_metrics,
                **_policy_session_metadata(sess.turn_policy, sess),
            })
            while True:
                try:
                    ev = json.loads(await ws.receive_text())
                    if not isinstance(ev, dict): raise ValueError("evento deve essere un oggetto JSON")
                except (json.JSONDecodeError, ValueError) as exc:
                    await send_ws({"type": "error", "error": {"code": "INVALID_EVENT", "message": str(exc)}})
                    continue
                et = ev.get("type")
                if et == "client_consumed_pcm":
                    ack_response_id = ev.get("response_id")
                    with playback_ack_branches_lock:
                        branch = playback_ack_branches.get(str(ack_response_id or ""))
                    if branch is None or not sess.is_generation_current(
                        generation_id=ev.get("generation_id")
                        if isinstance(ev.get("generation_id"), int)
                        and not isinstance(ev.get("generation_id"), bool)
                        else None,
                        response_id=str(ack_response_id or ""),
                    ):
                        accepted = False
                    else:
                        with branch.get("spoken_prefix_lock", nullcontext()):
                            accepted = _record_client_consumed_pcm(branch, ev)
                    if accepted and not _response_branch_has_pending_pcm(branch):
                        ack_event = branch.get("playback_ack_event")
                        if isinstance(ack_event, asyncio.Event):
                            ack_event.set()
                    if not accepted:
                        await send_ws({
                            "type": "error",
                            "response_id": str(ack_response_id or ""),
                            "error": {
                                "code": "STALE_CLIENT_CONSUMPTION_ACK",
                            "message": "client_consumed_pcm identity, sequence, or closed segment range is stale or mismatched",
                            },
                        })
                    else:
                        update_response_runtime_metrics(str(ack_response_id), "client_consumed_pcm", {
                            "status": "accepted",
                            "logical_client_consumption_only": True,
                            "response_id": str(ack_response_id),
                            "generation_id": int(ev["generation_id"]),
                            "speaker_segment_id": str(ev["speaker_segment_id"]),
                            "start_sample": int(ev["start_sample"]),
                            "end_sample": int(ev["end_sample"]),
                            "sample_rate": int(ev["sample_rate"]),
                            "sequence": int(ev["sequence"]),
                            "consumption_semantics": "logical_client_device_buffer",
                            "confirmed_exclusive_kv_cursor": branch.get(
                                "spoken_prefix_exclusive_cursor"
                            ),
                        })
                elif et == "session.update":
                    cfg = ev.get("session", {})
                    extra = cfg.get("extra_body", {}) or {}
                    requested_instructions = str(cfg.get("instructions", sess.instructions))
                    instructions_changed = requested_instructions != str(sess.instructions)
                    if unified_kv_enabled and instructions_changed:
                        binding = shared_listener_bindings.get(sess.session_id)
                        dispatcher = shared_listener_dispatcher_holder.get("dispatcher")
                        reason = (
                            "unified listener session is unavailable"
                            if binding is None
                            else _unified_instruction_refresh_ineligible_reason(
                                sess,
                                binding,
                                dispatcher,
                            )
                        )
                        if reason is not None:
                            await send_ws({
                                "type": "error",
                                "error": {
                                    "code": "INVALID_SESSION_UPDATE",
                                    "message": reason[:240],
                                },
                            })
                            continue
                    requested_voice = str(cfg.get("voice", sess.voice))
                    requested_voice_style = cfg.get("voice_style", sess.voice_style)
                    requested_auto_response = bool(
                        cfg.get("auto_response", extra.get("auto_response", sess.auto_response))
                    )
                    requested_input_ack = cfg.get(
                        "input_audio_buffer_acknowledgements",
                        extra.get("input_audio_buffer_acknowledgements"),
                    )
                    requested_conditioning_ack = cfg.get(
                        "input_audio_buffer_conditioning_acknowledgements",
                        extra.get("input_audio_buffer_conditioning_acknowledgements"),
                    )
                    requested_input_transcription = cfg.get(
                        "input_audio_transcription",
                        extra.get("input_audio_transcription", False),
                    )
                    requested_audio_history_archive_cpu = cfg.get(
                        "audio_history_archive_cpu",
                        extra.get("audio_history_archive_cpu", False),
                    )
                    if type(requested_input_transcription) is not bool:
                        await send_ws({
                            "type": "error",
                            "error": {
                                "code": "INVALID_SESSION_UPDATE",
                                "message": "input_audio_transcription deve essere booleano",
                            },
                        })
                        continue
                    if type(requested_audio_history_archive_cpu) is not bool:
                        await send_ws({
                            "type": "error",
                            "error": {
                                "code": "INVALID_SESSION_UPDATE",
                                "message": "audio_history_archive_cpu deve essere booleano",
                            },
                        })
                        continue
                    if requested_input_ack is not None and type(requested_input_ack) is not bool:
                        await send_ws({
                            "type": "error",
                            "error": {
                                "code": "INVALID_SESSION_UPDATE",
                                "message": "input_audio_buffer_acknowledgements deve essere booleano",
                            },
                        })
                        continue
                    if requested_conditioning_ack is not None and type(requested_conditioning_ack) is not bool:
                        await send_ws({
                            "type": "error",
                            "error": {
                                "code": "INVALID_SESSION_UPDATE",
                                "message": "input_audio_buffer_conditioning_acknowledgements deve essere booleano",
                            },
                        })
                        continue
                    effective_input_ack = (
                        bool(requested_input_ack)
                        if requested_input_ack is not None
                        else input_ack_enabled.is_set()
                    )
                    if requested_conditioning_ack is True and not effective_input_ack:
                        await send_ws({
                            "type": "error",
                            "error": {
                                "code": "INVALID_SESSION_UPDATE",
                                "message": (
                                    "input_audio_buffer_conditioning_acknowledgements richiede "
                                    "input_audio_buffer_acknowledgements=true"
                                ),
                            },
                        })
                        continue
                    requested_voice_mode = cfg.get("voice_conditioning_mode")
                    if requested_voice_mode not in {None, "x_vector_only"}:
                        prepared_voice_conditioning = None
                        voice_error: Exception | None = ValueError(
                            "P7 supporta esclusivamente voice conditioning x_vector_only"
                        )
                    else:
                        try:
                            prepared_voice_conditioning = _resolve_p7_session_voice(
                                engine,
                                voice=requested_voice,
                                voice_style=requested_voice_style,
                            )
                            voice_error = None
                        except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
                            prepared_voice_conditioning = None
                            voice_error = exc
                    if voice_error is not None:
                        # A rejected voice profile must not leave an automatic
                        # SPEAK pending behind the mic gate or repeatedly spawn
                        # responses that can never pass the P7 prefill.
                        sess.auto_response = False
                        input_ack_enabled.clear()
                        input_conditioning_ack_enabled.clear()
                        set_native_partial_asr_enabled(False)
                        pending_speak = None
                        with policy_causality_lock:
                            replan_handoff_pending = False
                        with sess._generation_lock:
                            sess.policy_response_pending = False
                            response_to_stop = sess.response_id
                            has_active_response = bool(sess.active_response)
                        if has_active_response:
                            sess.invalidate_generation("invalid_voice_conditioning")
                            _stop_owned_synthesis(sess.session_id, response_to_stop)
                            if response_task and not response_task.done():
                                response_task.cancel()
                                await asyncio.gather(response_task, return_exceptions=True)
                        await send_ws({
                            "type": "error",
                            "error": {
                                "code": "VOICE_CONDITIONING_ERROR",
                                "message": str(voice_error)[:240],
                                "terminal": True,
                            },
                        })
                        continue
                    policy_name_present = "turn_policy" in cfg or "turn_policy" in extra
                    policy_name = cfg.get("turn_policy") if "turn_policy" in cfg else extra.get("turn_policy")
                    selected_policy = None
                    try:
                        with sess._generation_lock:
                            if policy_name_present:
                                if not isinstance(policy_name, str) or not policy_name.strip():
                                    raise ValueError("turn_policy deve essere un nome supportato non vuoto")
                                policy_name = policy_name.strip()
                                if policy_name == "turnhead" and isinstance(sess.turn_policy, TurnHeadPolicy):
                                    selected_policy = sess.turn_policy
                                elif policy_name == "duplex_control6" and isinstance(
                                    sess.turn_policy, DuplexControl6Policy
                                ):
                                    selected_policy = sess.turn_policy
                                else:
                                    selected_policy = _make_policy(policy_name, cfg)
                                if isinstance(selected_policy, TurnHeadPolicy):
                                    _require_loaded_turnhead(selected_policy)
                            elif (
                                requested_auto_response
                                and isinstance(sess.turn_policy, TurnHeadPolicy)
                                and not sess.turn_policy.loaded
                            ):
                                _require_loaded_turnhead(sess.turn_policy)
                    except (TypeError, ValueError) as exc:
                        await send_ws({"type": "error", "error": {"code": "INVALID_SESSION_UPDATE", "message": str(exc)}})
                        continue
                    trace_enabled = cfg.get("conditioning_trace_enabled", extra.get("conditioning_trace_enabled"))
                    trace_sample_id = cfg.get("conditioning_trace_sample_id", extra.get("conditioning_trace_sample_id"))
                    if trace_sample_id is not None and not str(trace_sample_id).strip():
                        await send_ws({"type": "error", "error": {"code": "INVALID_SESSION_UPDATE", "message": "conditioning_trace_sample_id vuoto"}})
                        continue
                    with sess._generation_lock:
                        sample_id_locked = (
                            trace_sample_id is not None
                            and sess.conditioning_trace_sample_id_locked
                            and str(trace_sample_id) != sess.conditioning_trace_sample_id
                        )
                    if sample_id_locked:
                        await send_ws({"type": "error", "error": {"code": "INVALID_SESSION_UPDATE", "message": "conditioning_trace_sample_id già congelato"}})
                        continue
                    if trace_enabled is not None or trace_sample_id is not None:
                        with sess._generation_lock:
                            if trace_enabled is not None:
                                sess.conditioning_trace_enabled = bool(trace_enabled)
                            if trace_sample_id is not None:
                                sess.conditioning_trace_sample_id = str(trace_sample_id)
                    try:
                        _configure_playout(sess, cfg.get("playout_delay_frames", extra.get("playout_delay_frames")))
                    except (TypeError, ValueError) as exc:
                        await send_ws({"type": "error", "error": {"code": "INVALID_SESSION_UPDATE", "message": str(exc)}})
                        continue
                    if selected_policy is not None:
                        with sess._generation_lock:
                            sess.set_turn_policy(selected_policy)
                    if isinstance(selected_policy, DuplexControl6Policy):
                        runtime = selected_policy.runtime
                        loaded_event = {
                            "type": "model_loaded",
                            "policy_id": "duplex_control6",
                            "base_path": runtime["base_path"],
                            "adapter_path": runtime["adapter_path"],
                            "control_head": runtime["control_head"],
                            "control_head_architecture": getattr(runtime.get("head"), "architecture", None),
                            "control_head_input_dim": runtime.get("feature_contract", {}).get("input_dim"),
                            "actions": list(runtime["actions"]),
                            "feature_contract": dict(runtime["feature_contract"]),
                        }
                        if runtime.get("bundle_id"):
                            loaded_event["bundle_id"] = runtime["bundle_id"]
                            loaded_event["base_model_id"] = runtime["base_model_id"]
                            loaded_event["response_lora_applied"] = bool(
                                runtime.get("response_lora_applied", True)
                            )
                            loaded_event["feature_source"] = dict(runtime["feature_source"])
                            loaded_event["runtime_abi"] = dict(runtime["runtime_abi"])
                        else:
                            loaded_event["run_id"] = runtime["run_id"]
                        await send_ws(loaded_event)
                    if unified_kv_enabled and instructions_changed:
                        binding = shared_listener_bindings.get(sess.session_id)
                        dispatcher = shared_listener_dispatcher_holder.get("dispatcher")
                        worker = shared_listener_worker_holder.get("worker")
                        try:
                            if binding is None:
                                raise RuntimeError("unified listener session is unavailable")
                            await _refresh_unified_prefix_for_instruction_update(
                                engine,
                                sess,
                                instructions=requested_instructions,
                                worker=worker,
                                binding=binding,
                                dispatcher=dispatcher,
                            )
                        except Exception as exc:
                            await send_ws({
                                "type": "error",
                                "error": {
                                    "code": "INVALID_SESSION_UPDATE",
                                    "message": str(exc)[:240],
                                },
                            })
                            continue
                    else:
                        sess.update_instructions(requested_instructions)
                    diagnostic_fixture = cfg.get("tts_quality_diagnostic_fixture")
                    sess.tts_quality_diagnostic_fixture = (
                        str(diagnostic_fixture)
                        if isinstance(diagnostic_fixture, str)
                        else None
                    )
                    diagnostic_seed_sha256 = cfg.get("tts_quality_diagnostic_seed_sha256")
                    sess.tts_quality_diagnostic_seed_sha256 = (
                        diagnostic_seed_sha256.lower()
                        if isinstance(diagnostic_seed_sha256, str)
                        and len(diagnostic_seed_sha256) == 64
                        and all(char in "0123456789abcdefABCDEF" for char in diagnostic_seed_sha256)
                        else None
                    )
                    sess.voice = requested_voice
                    sess.voice_style = requested_voice_style
                    sess.voice_conditioning = prepared_voice_conditioning
                    sess.tools = cfg.get("tools", sess.tools) or []
                    with sess._generation_lock:
                        sess.auto_response = requested_auto_response
                    input_transcription_enabled = bool(
                        requested_input_transcription and native_partial_asr_available
                    )
                    set_native_partial_asr_enabled(input_transcription_enabled)
                    audio_history_archive_cpu_enabled = bool(
                        requested_audio_history_archive_cpu and archive_cpu_startup_available
                    )
                    if requested_input_ack is not None:
                        if requested_input_ack:
                            input_ack_enabled.set()
                        else:
                            input_ack_enabled.clear()
                            input_conditioning_ack_enabled.clear()
                    if requested_conditioning_ack is not None:
                        if requested_conditioning_ack:
                            input_conditioning_ack_enabled.set()
                        else:
                            input_conditioning_ack_enabled.clear()
                    initial_prefix_admission = await _prime_shared_listener_initial_prefix(sess)
                    await send_ws({
                        "type": "session.updated",
                        "session_id": sess.session_id,
                        "native_control6_semantic_prefix_initial_admission": (
                            initial_prefix_admission.get("status")
                        ),
                        "native_control6_semantic_prefix_initial_admission_reason": (
                            initial_prefix_admission.get("reason")
                        ),
                        "native_control6_semantic_prefix_initial_admission_ms": (
                            initial_prefix_admission.get("duration_ms")
                        ),
                        "native_control6_semantic_prefix_initial_prefix_len": (
                            initial_prefix_admission.get("prefix_len")
                        ),
                        "native_control6_semantic_prefix_initial_revision": (
                            initial_prefix_admission.get("revision")
                        ),
                        "native_control6_semantic_prefix_initial_build_ms": (
                            initial_prefix_admission.get("prefix_build_ms")
                        ),
                        "native_control6_semantic_prefix_initial_replay_ms": (
                            initial_prefix_admission.get("prefix_replay_ms")
                        ),
                        "native_control6_semantic_prefix_initial_import_ms": (
                            initial_prefix_admission.get("import_ms")
                        ),
                        "native_control6_semantic_prefix_initial_audio_samples_received": (
                            initial_prefix_admission.get("audio_samples_received")
                        ),
                        "native_control6_semantic_prefix_initial_audio_queue_depth_frames": (
                            initial_prefix_admission.get("audio_queue_depth_frames")
                        ),
                        "native_control6_semantic_prefix_initial_audio_pending_samples": (
                            initial_prefix_admission.get("audio_pending_samples")
                        ),
                        "input_audio_transcription_enabled": input_transcription_enabled,
                        "audio_history_archive_cpu_enabled": audio_history_archive_cpu_enabled,
                        "audio_history_archive_defer_offline": audio_history_archive_defer_offline,
                        "input_audio_buffer_conditioning_acknowledgements": (
                            input_conditioning_ack_enabled.is_set()
                        ),
                        "listener_silence_skip_enabled": listener_silence_skip_enabled,
                        "input_audio_buffer_capacity_frames": sess.audio_ingress.max_frames,
                        "voice": sess.voice,
                        "response_projector_mode": _response_projector_mode(engine),
                        "voice_conditioning_mode": (
                            prepared_voice_conditioning.get("mode")
                            if isinstance(prepared_voice_conditioning, dict)
                            else None
                        ),
                        "conditioning_source": (
                            prepared_voice_conditioning.get("conditioning_source")
                            if isinstance(prepared_voice_conditioning, dict)
                            else None
                        ),
                        "reference_wav_path": (
                            prepared_voice_conditioning.get("reference_wav_path")
                            if isinstance(prepared_voice_conditioning, dict)
                            else None
                        ),
                        "reference_wav_sha256": (
                            prepared_voice_conditioning.get("reference_wav_sha256")
                            if isinstance(prepared_voice_conditioning, dict)
                            else None
                        ),
                        "voice_conditioning_ablation": bool(
                            getattr(engine, "ablate_session_voice_conditioning", False)
                        ),
                        **_policy_session_metadata(sess.turn_policy, sess),
                    })
                elif et == "conversation.item.create":
                    item = ev.get("item", {})
                    with sess._generation_lock:
                        sess.history.append({"role": item.get("role", "user"), "content": _message_text(item.get("content", ""))})
                    await send_ws({"type": "conversation.item.created"})
                elif et == "input_audio_buffer.append":
                    try:
                        raw = base64.b64decode(ev.get("audio", ""), validate=True)
                        if not raw or len(raw) % 2:
                            raise ValueError("audio PCM16 vuoto o con lunghezza dispari")
                        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                    except (ValueError, TypeError) as exc:
                        await send_ws({"type": "error", "error": {"code": "INVALID_AUDIO", "message": str(exc)}})
                        continue
                    # Keep input admission off the generation/control lock so a
                    # slow synchronous listener/policy callback cannot stall the
                    # WebSocket receive loop before bounded-queue overload is seen.
                    ingress_overloaded = False
                    with sess._pending_audio_lock:
                        pending_cap = USER_SAMPLE_RATE * 30
                        mode_b_realtime = sess.turn_policy is not None
                        # Mode B is a continuous microphone stream: retain only
                        # the newest 30 seconds instead of applying Mode A's
                        # per-turn lifetime cap to silence and later utterances.
                        # A single oversized append remains invalid in either
                        # mode so callers cannot bypass the bounded ingress.
                        over_limit = pcm.size > pending_cap or (
                            not mode_b_realtime and sess.pending_audio.size + pcm.size > pending_cap
                        )
                        if not over_limit:
                            try:
                                sess.audio_ingress.append(pcm, transport_bytes=raw)
                            except queue.Full:
                                ingress_overloaded = True
                            else:
                                pending_audio = np.concatenate((sess.pending_audio, pcm))
                                if mode_b_realtime and pending_audio.size > pending_cap:
                                    pending_audio = pending_audio[-pending_cap:]
                                sess.pending_audio = pending_audio
                    if over_limit:
                        await send_ws({"type": "error", "error": {"code": "AUDIO_LIMIT", "message": "massimo 30 secondi per turno"}}); continue
                    if ingress_overloaded:
                        stats = sess.audio_ingress.stats()
                        logger.warning(
                            "[realtime %s] microphone ingress saturated depth=%d capacity=%d rejected=%d",
                            sess.session_id[:8],
                            sess.audio_ingress.qsize,
                            sess.audio_ingress.max_frames,
                            stats["dropped_input_frames"],
                        )
                        closing = True
                        sess.invalidate_generation("input_audio_overload")
                        await send_ws({
                            "type": "error",
                            "error": {
                                "code": "INPUT_AUDIO_OVERLOAD",
                                "message": (
                                    f"Microphone ingress exceeded {sess.audio_ingress.max_frames} queued frames; "
                                    "the incoming audio was rejected, queued audio will not be replayed, "
                                    "and the session is closing"
                                ),
                            },
                            "rejected_frames": stats["dropped_input_frames"],
                        })
                        try:
                            await ws.close(code=1013, reason="microphone ingress overload")
                        except (WebSocketDisconnect, RuntimeError):
                            pass
                        break
                    sess.touch()
                elif et in {"input_audio_buffer.commit", "response.create"}:
                    with sess._generation_lock:
                        response_busy = sess.active_response or sess.policy_response_pending
                    if (response_task and not response_task.done()) or response_busy:
                        await send_ws({"type": "error", "error": {"code": "RESPONSE_ACTIVE", "message": "response già attiva"}})
                        continue
                    if (
                        pending_speak is not None
                        and pending_speak.get("source") == _DEFERRED_NATIVE_SPEAK_SOURCE
                    ):
                        # An explicit manual commit starts a different turn;
                        # it cannot inherit an automatic v2 action from an
                        # earlier, incomplete utterance.
                        pending_speak = None
                    while True:
                        try:
                            pending_response_speak_handoffs.get_nowait()
                        except queue.Empty:
                            break
                    # Freeze the turn and exclude already-enqueued frames from
                    # the next policy-owned audio snapshot.  The listener still
                    # consumes those frames on its continuous 80 ms clock.
                    with sess._generation_lock:
                        with sess._pending_audio_lock:
                            committed_audio = sess.pending_audio.copy()
                            sess.pending_audio = np.empty(0, dtype=np.float32)
                        committed_sample_count = int(sess.audio_ingress.stats()["input_samples_received"])
                        first_manual_sample = max(
                            0, committed_sample_count - int(committed_audio.size)
                        )
                        first_manual_frame = first_manual_sample // POLICY_FRAME_SAMPLES
                        last_manual_frame = max(
                            first_manual_frame,
                            (committed_sample_count + POLICY_FRAME_SAMPLES - 1)
                            // POLICY_FRAME_SAMPLES - 1,
                        )
                        manual_commit_speaks, manual_user_turn_onset_frame_id = (
                            _manual_commit_speak_owner_for_span(
                                sess,
                                first_manual_frame,
                                last_manual_frame,
                                direct_audio_candidate=direct_audio_candidate,
                                auto_response=bool(sess.auto_response),
                                committed_audio_size=int(committed_audio.size),
                            )
                        )
                        sess.reset_policy_audio(committed_sample_count=committed_sample_count)
                        sess.reset_turn_policy()
                        sess.policy_response_pending = True
                        committed_through_frame = (
                            (committed_sample_count + POLICY_FRAME_SAMPLES - 1) // POLICY_FRAME_SAMPLES - 1
                        )
                        with policy_causality_lock:
                            response_audio_admission.reset_locked()
                            replan_handoff_pending = False
                    if committed_audio.size:
                        first_committed_frame = max(
                            0,
                            (committed_sample_count - int(committed_audio.size)) // POLICY_FRAME_SAMPLES,
                        )
                        manual_utterance_id, manual_transcript_epoch = begin_transcript_utterance(
                            first_committed_frame
                        )
                    else:
                        manual_transcript_epoch = advance_transcript_epoch(reset_utterance=True)
                        manual_utterance_id = ""
                    # Non-blocking: the receive loop remains live during ASR,
                    # Thinker/V9/Talker and can continue capturing barge-in.
                    manual_audio_context = None
                    if direct_audio_candidate and committed_audio.size:
                        first_sample_id = max(0, committed_sample_count - int(committed_audio.size))
                        first_frame_id = first_sample_id // POLICY_FRAME_SAMPLES
                        last_frame_id = max(
                            first_frame_id,
                            (committed_sample_count + POLICY_FRAME_SAMPLES - 1) // POLICY_FRAME_SAMPLES - 1,
                        )
                        manual_audio_context = {
                            "source_frame_id": int(last_frame_id),
                            "endpoint_frame_id": int(last_frame_id),
                            "source": "manual_audio_commit",
                            "_direct_audio_pcm": np.array(committed_audio, dtype=np.float32, copy=True),
                            "_direct_audio_first_frame_id": int(first_frame_id),
                            "_direct_audio_last_frame_id": int(last_frame_id),
                            "_direct_audio_target_frame_id": int(committed_through_frame),
                            "_direct_audio_start_ms": first_sample_id * 1000.0 / USER_SAMPLE_RATE,
                            "_direct_audio_end_ms": committed_sample_count * 1000.0 / USER_SAMPLE_RATE,
                            "_direct_audio_active": False,
                            "_direct_audio_manual_commit": True,
                        }
                    if committed_audio.size:
                        manual_audio_context = dict(manual_audio_context or {})
                        manual_audio_context.setdefault(
                            "source_frame_id", int(first_committed_frame)
                        )
                        manual_audio_context.setdefault(
                            "endpoint_frame_id",
                            int(
                                max(
                                    first_committed_frame,
                                    (committed_sample_count + POLICY_FRAME_SAMPLES - 1)
                                    // POLICY_FRAME_SAMPLES - 1,
                                )
                            ),
                        )
                        manual_audio_context.setdefault("source", "manual_audio_commit")
                        if manual_user_turn_onset_frame_id is not None:
                            manual_audio_context["_user_turn_onset_frame_id"] = (
                                manual_user_turn_onset_frame_id
                            )
                            manual_audio_context["_user_turn_endpoint_complete"] = True
                        manual_audio_context["_input_transcript_context"] = {
                            "utterance_id": manual_utterance_id,
                            "epoch_id": int(manual_transcript_epoch),
                            "first_frame_id": int(first_committed_frame),
                            "last_frame_id": int(
                                max(
                                    first_committed_frame,
                                    (committed_sample_count + POLICY_FRAME_SAMPLES - 1)
                                    // POLICY_FRAME_SAMPLES - 1,
                                )
                            ),
                        }
                    response_task = asyncio.create_task(
                        produce_response(
                            committed_audio,
                            action=(
                                ControlAction.SPEAK
                                if manual_commit_speaks
                                else None
                            ),
                            intent_origin=manual_audio_context,
                        )
                    )
                elif et == "session.close": break
        except (WebSocketDisconnect, asyncio.CancelledError): pass
        finally:
            closing = True
            with playback_ack_branches_lock:
                playback_branches_to_close = tuple(playback_ack_branches.values())
            for playback_branch in playback_branches_to_close:
                _cancel_response_branch_playback(playback_branch)
            set_native_partial_asr_enabled(False)
            with sess._generation_lock:
                sess.policy_response_pending = False
                sess.stop_requested = True
            sess.invalidate_generation("websocket_closed")
            if native_partial_worker is not None:
                native_partial_worker.close()
            with sess._generation_lock:
                response_to_stop = sess.response_id
            _stop_owned_synthesis(sess.session_id, response_to_stop)
            if response_task and not response_task.done():
                response_task.cancel()
                await asyncio.gather(response_task, return_exceptions=True)
            deferred_finalizers = tuple(
                task
                for playback_branch in playback_branches_to_close
                if isinstance((task := playback_branch.get("deferred_playback_finalizer")), asyncio.Task)
                and not task.done()
            )
            if deferred_finalizers:
                await asyncio.gather(*deferred_finalizers, return_exceptions=True)
            scheduler_events.close()
            if scheduler_event_task and not scheduler_event_task.done():
                scheduler_event_task.cancel()
                await asyncio.gather(scheduler_event_task, return_exceptions=True)
            if shared_listener_microbatch_enabled:
                await _unregister_shared_listener_session(sess)
            with playback_ack_branches_lock:
                _clear_shared_listener_response_branches(
                    response_binding,
                    playback_ack_branches,
                )
            profile_host_finalization = None
            if NATIVE_LISTENER_CUDA_PROFILE_ENABLED:
                # The listener callback writes its final host stamps after Q4
                # returns. Keep the loop alive while the bounded scheduler join
                # runs elsewhere, so callback cleanup can finish without
                # blocking the WebSocket event loop.
                profile_host_finalization = await _quiesce_scheduler_for_native_profile(
                    sess.scheduler
                )
            with response_runtime_metrics_lock:
                final_response_metrics = {
                    response_id: {phase: dict(values) for phase, values in phases.items()}
                    for response_id, phases in response_runtime_metrics.items()
                }
            with listener_timing_lock:
                final_session_metrics = {
                    "listener": {
                        **session_runtime_metrics["listener"],
                        "tick_timings": list(session_runtime_metrics["listener"].get("tick_timings", [])),
                    }
                }
            if profile_host_finalization is not None:
                final_session_metrics["listener"]["native_q4_cuda_profile_host_finalization"] = dict(
                    profile_host_finalization
                )
            if (
                profile_host_finalization is not None
                and profile_host_finalization.get("status") != "complete"
            ):
                final_session_metrics["listener"]["audio_tower_cuda_timing"] = {
                    "available": False,
                    "reason": "scheduler_not_quiescent_for_profile_snapshot",
                }
            else:
                try:
                    audio_tower_timing_summary = getattr(
                        sess.listener_state, "audio_tower_cuda_timing_summary", None
                    )
                    if callable(audio_tower_timing_summary):
                        final_session_metrics["listener"]["audio_tower_cuda_timing"] = audio_tower_timing_summary()
                except Exception as exc:
                    final_session_metrics["listener"]["audio_tower_cuda_timing"] = {
                        "available": False,
                        "reason": type(exc).__name__,
                    }
            final_session_metrics["scheduler_stage_wall_ms"] = scheduler_stage_summary()
            final_session_metrics["scheduler"] = sess.scheduler.stats()
            if _NATIVE_CONTROL6_DEBUG_INPUT_DIRECTORY is not None:
                sidecar_path = None
                sidecar_error = None
                try:
                    sidecar_path = _write_native_control6_debug_input_sidecar(
                        _NATIVE_CONTROL6_DEBUG_INPUT_DIRECTORY,
                        session_id=sess.session_id,
                        rows=native_control6_debug_rows,
                    )
                except Exception as exc:
                    sidecar_error = f"{type(exc).__name__}: {exc}"
                    logger.exception(
                        "native Control6 debug input sidecar write failed for session %s",
                        sess.session_id,
                    )
                final_session_metrics["listener"]["native_control6_debug_inputs"] = {
                    "enabled": True,
                    "max_frames": _NATIVE_CONTROL6_DEBUG_INPUT_MAX_FRAMES,
                    "captured_frames": len(native_control6_debug_rows),
                    "sidecar_path": sidecar_path,
                    "error": sidecar_error,
                }
            final_session_metrics["scheduler_cuda_stream"] = session_runtime_metrics.get(
                "scheduler_cuda_stream", {"enabled": False, "ownership": "cpu_fallback"}
            )
            try:
                if (
                    profile_host_finalization is not None
                    and profile_host_finalization.get("status") != "complete"
                ):
                    info = {
                        "available": False,
                        "reason": "scheduler_not_quiescent_for_profile_snapshot",
                    }
                else:
                    native_listener = getattr(sess.listener_state, "gemma_lm_native", None)
                    info = native_listener.native_runtime_info() if callable(getattr(native_listener, "native_runtime_info", None)) else {"available": False, "reason": "native_runtime_info_unavailable"}
                final_session_metrics["listener"]["native_runtime_info_at_close"] = info
            except Exception as exc:
                final_session_metrics["listener"]["native_runtime_info_at_close"] = {"available": False, "reason": type(exc).__name__}
            runtime_info_at_close = final_session_metrics["listener"]["native_runtime_info_at_close"]
            if (
                profile_host_finalization is not None
                and profile_host_finalization.get("status") != "complete"
            ):
                profile_summary = {
                    "schema_version": 1,
                    "enabled": True,
                    "status": "incomplete",
                    "session_id": sess.session_id,
                    "reason": "scheduler_not_quiescent_for_profile_snapshot",
                    "native_event_snapshot_status": "not_taken_while_scheduler_active",
                    "sampled_event_pair_count": None,
                    "completed_sample_count": None,
                    "pending_sample_count": None,
                    "pending_frame_ids": [],
                    "pending_samples": [],
                    "samples": [],
                    # Do not read the callback-owned context while a timed-out
                    # callback may still be mutating it. These are explicit
                    # unknowns, not a partial/final timing record.
                    "active_callback_host_context": {
                        "status": "unknown_while_scheduler_active",
                        "listener_step_end_ns": None,
                        "listener_callback_end_ns": None,
                        "shared_lm_lock_wait_ms": None,
                        "shared_lm_lock_host_hold_ms": None,
                    },
                }
            else:
                native_listener = getattr(sess.listener_state, "gemma_lm_native", None)
                profile_summary_fn = getattr(native_listener, "native_q4_cuda_profile_summary", None)
                if callable(profile_summary_fn):
                    try:
                        profile_summary = profile_summary_fn(
                            session_id=sess.session_id,
                            runtime_info=runtime_info_at_close,
                        )
                    except Exception as exc:
                        profile_summary = {
                            "schema_version": 1,
                            "enabled": NATIVE_LISTENER_CUDA_PROFILE_ENABLED,
                            "status": "failed",
                            "session_id": sess.session_id,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                else:
                    profile_summary = {
                        "schema_version": 1,
                        "enabled": NATIVE_LISTENER_CUDA_PROFILE_ENABLED,
                        "status": "unavailable" if NATIVE_LISTENER_CUDA_PROFILE_ENABLED else "not_requested",
                        "session_id": sess.session_id,
                        "reason": "native_listener_context_unavailable",
                    }
            if profile_host_finalization is not None:
                profile_summary["host_finalization"] = dict(profile_host_finalization)
            final_session_metrics["listener"]["native_q4_cuda_profile"] = profile_summary
            profile_directory = native_listener_cuda_profile_directory
            if profile_directory:
                try:
                    model_lock_timeline = None
                    if NATIVE_LISTENER_CUDA_PROFILE_ENABLED:
                        shared_model_lock = getattr(engine, "_thinker_lock", None)
                        snapshot_timeline = getattr(
                            shared_model_lock, "host_timeline_snapshot", None
                        )
                        if callable(snapshot_timeline):
                            try:
                                model_lock_timeline = snapshot_timeline()
                            except Exception as exc:
                                model_lock_timeline = {
                                    "enabled": True,
                                    "status": "unavailable",
                                    "scope": "shared_model_lock",
                                    "reason": type(exc).__name__,
                                }
                        else:
                            model_lock_timeline = {
                                "enabled": True,
                                "status": "unavailable",
                                "scope": "shared_model_lock",
                                "reason": "timeline_snapshot_unavailable",
                            }
                    sidecar_path = write_native_listener_cuda_profile_sidecar(
                        session_id=sess.session_id,
                        profile=profile_summary,
                        host_tick_timings=final_session_metrics["listener"].get("tick_timings", []),
                        model_lock_timeline=model_lock_timeline,
                        directory=profile_directory,
                    )
                    if sidecar_path is not None:
                        profile_summary["sidecar_path"] = sidecar_path
                except Exception as exc:
                    profile_summary["sidecar_status"] = "failed"
                    profile_summary["sidecar_error"] = f"{type(exc).__name__}: {exc}"
                    logger.exception(
                        "native listener CUDA profile sidecar write failed for session %s",
                        sess.session_id,
                    )
            if os.environ.get(LISTENER_ATTESTATION_DIRECTORY_ENV) is not None:
                try:
                    write_listener_attestation_from_environment(sess.session_id, runtime_info_at_close)
                except Exception:
                    logger.exception(
                        "native listener session attestation sidecar write failed for session %s",
                        sess.session_id,
                    )
            logger.info(
                "[realtime_session_metrics] %s",
                json.dumps({
                    "session_id": sess.session_id,
                    "scheduler": sess.scheduler.stats(),
                    "session": final_session_metrics,
                    "responses": final_response_metrics,
                }, ensure_ascii=False, allow_nan=False),
            )
            sessions.close(sess.session_id)
            with audio_archive_lock:
                pending_archive_futures = list(audio_archive_futures.values())
                audio_archive_futures.clear()
            for pending_archive_future in pending_archive_futures:
                pending_archive_future.cancel()
    return app


def run(engine: Any, *, host: str = "0.0.0.0", port: int = 8092) -> None:
    import uvicorn
    uvicorn.run(create_app(engine), host=host, port=port, log_level="info")

__all__ = ["apply_speaker_gate_event", "create_app", "run"]
