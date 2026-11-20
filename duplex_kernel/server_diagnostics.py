"""Diagnostic, sidecar, and metric helpers for the duplex server."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from duplex_kernel.tts_quality_diagnostics import TTSQualityDiagnosticCapture

from duplex_kernel.native_control6_diagnostics import attach_native_control6_zero_output_diagnostic

logger = logging.getLogger("duplex_kernel.server")

_NATIVE_CONTROL6_INVALIDATION_LOG_LOCK = threading.Lock()
_NATIVE_CONTROL6_INVALIDATION_LOG_STATE: dict[str, tuple[str, int]] = {}

_NATIVE_CONTROL6_TRACE_ENABLED = os.environ.get("DUPLEX_NATIVE_CONTROL6_TRACE", "0").strip() == "1"

_NATIVE_CONTROL6_DEBUG_INPUT_MAX_FRAMES = 256

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
    "native_action_order",
    "native_action_logits",
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
    "native_control6_peer_turn_active",
    "native_control6_user_rms",
    "native_control6_user_activity_detected",
    "native_control6_speech_rms_floor",
    "native_control6_early_speak_activity_active",
    "native_control6_early_speak_confidence",
    "native_control6_early_speak_confidence_source",
    "native_control6_early_speak_confidence_threshold",
    "native_control6_early_speak_gated",
    "native_control6_endpoint_speak_pending",
    "native_control6_endpoint_speak_ready",
    "native_control6_endpoint_speak_released",
    "native_control6_endpoint_speak_turn_onset_frame_id",
    "native_control6_minimum_backchannel_confidence",
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
    import duplex_kernel.server as server
    enabled = getattr(server, "_NATIVE_CONTROL6_TRACE_ENABLED", _NATIVE_CONTROL6_TRACE_ENABLED)
    attach_native_control6_zero_output_diagnostic(
        trace,
        enabled=enabled,
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
    diagnostics = getattr(decision, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        for name in (
            "policy_evaluation_state",
            "native_control6_head_scored",
            "decision_view_unavailable_reason",
        ):
            if name in diagnostics:
                fields[name] = diagnostics[name]
    return fields


def _capture_native_control6_policy_error(
    decision: Any, error: Exception
) -> tuple[None, Exception]:
    """Invalidate a decision at the first native-policy validation failure."""
    return None, error


def _validate_native_control6_worker_decision_provenance(
    decision: Any,
    provenance: Any,
    *,
    binding_available: bool,
    canonical_session_id: str,
    expected_listener_generation_id: int,
    expected_turn_generation_id: int,
    expected_state_epoch_id: str,
    expected_frame_id: int,
    expected_listener_state_version: int,
) -> None:
    """Raise the first worker decision provenance mismatch without fallback text."""
    if (
        not binding_available
        or decision.session != canonical_session_id
        or decision.generation_id != expected_turn_generation_id
        or decision.frame_id != expected_frame_id
        or decision.listener_frame_id != expected_frame_id
        or not isinstance(provenance, Mapping)
        or provenance.get("canonical_session_id") != canonical_session_id
        or provenance.get("listener_generation_id") != expected_listener_generation_id
        or provenance.get("turn_generation_id") != expected_turn_generation_id
        or provenance.get("state_epoch_id") != expected_state_epoch_id
        or provenance.get("frame_id") != expected_frame_id
        or provenance.get("listener_state_version") != expected_listener_state_version
    ):
        raise RuntimeError(
            "shared native Control6 decision provenance differs from its active turn/listener frame"
        )


def _native_control6_policy_evaluation_state(
    *,
    decision: Any,
    head_scored: bool,
    no_input_tick: bool = False,
    error: BaseException | None = None,
    decision_view_unavailable_reason: str | None = None,
) -> str:
    """Classify whether Control6 scored, used its exact-zero path, or was unavailable."""
    if error is not None or decision_view_unavailable_reason:
        return "unavailable"
    diagnostics = getattr(decision, "diagnostics", None)
    if isinstance(diagnostics, Mapping) and (
        diagnostics.get("policy_head_skipped") is True
        or diagnostics.get("decision_view_unavailable_reason")
    ):
        return "unavailable"
    if head_scored is True:
        return "scored"
    no_input_diagnostic = (
        isinstance(diagnostics, Mapping)
        and diagnostics.get("no_input_tick") is True
    )
    if (
        no_input_tick is True
        and no_input_diagnostic
        and str(getattr(getattr(decision, "action", None), "value", getattr(decision, "action", "")))
        == "CONTINUE"
    ):
        return "exact_zero_continue"
    return "unavailable"


def _native_control6_policy_error_diagnostics(
    error: BaseException,
    *,
    identity: Mapping[str, Any],
    decision_view_unavailable_reason: str | None,
    invalidation_repeat_count: int,
) -> dict[str, Any]:
    """Build CPU-only rejection evidence while retaining unabridged exception text."""
    error_detail = str(error)
    return {
        **dict(identity),
        "policy_evaluation_state": "unavailable",
        "native_control6_head_scored": False,
        "decision_view_unavailable_reason": decision_view_unavailable_reason,
        "original_cause": repr(error),
        "error_detail": error_detail,
        "reason": error_detail[:240],
        "invalidation_repeat_count": int(invalidation_repeat_count),
    }


def _record_native_control6_policy_invalidation(
    *,
    session_id: str,
    state_epoch_id: str,
    frame_id: int,
    reason: str,
) -> int:
    """Warn on the first invalidation per session epoch and debug-log repeats."""
    session_key = str(session_id)
    epoch_key = str(state_epoch_id)
    with _NATIVE_CONTROL6_INVALIDATION_LOG_LOCK:
        previous = _NATIVE_CONTROL6_INVALIDATION_LOG_STATE.get(session_key)
        if previous is None or previous[0] != epoch_key:
            repeat_count = 0
            _NATIVE_CONTROL6_INVALIDATION_LOG_STATE[session_key] = (epoch_key, 0)
            first_in_epoch = True
        else:
            repeat_count = previous[1] + 1
            _NATIVE_CONTROL6_INVALIDATION_LOG_STATE[session_key] = (
                epoch_key,
                repeat_count,
            )
            first_in_epoch = False

        if first_in_epoch:
            logger.warning(
                "[realtime %s] first native Control6 policy invalidation epoch=%s frame=%d: %s",
                session_key[:8],
                epoch_key,
                int(frame_id),
                reason,
            )
        else:
            logger.debug(
                "[realtime %s] repeated native Control6 policy invalidation epoch=%s frame=%d repeat=%d: %s",
                session_key[:8],
                epoch_key,
                int(frame_id),
                repeat_count,
                reason,
            )
    return repeat_count

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
        "scheduler_unified_turn_transition_lock_wait_ms": getattr(
            tick, "microbatch_unified_transition_lock_wait_ms", None
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
            "listener_batch_unified_turn_transition_lock_wait_ms": getattr(
                tick, "microbatch_unified_transition_lock_wait_ms", None
            ),
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



from duplex_kernel import server_metrics as _server_metrics

globals().update({name: getattr(_server_metrics, name) for name in _server_metrics.__all__})
