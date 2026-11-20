"""Bounded runtime metrics, timing, and emitted-audio diagnostics."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from duplex_kernel.tts_quality_diagnostics import TTSQualityDiagnosticCapture

logger = logging.getLogger("duplex_kernel.server")

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

NATIVE_LISTENER_PROFILE_STOP_TIMEOUT_S = 1.0

_TTS_QUALITY_DIAGNOSTIC_CAPTURE = TTSQualityDiagnosticCapture(
    output_dir=".tmp/recovery_probe/e2e_pair_diagnostics",
    max_records=8,
)

def _acquire_shared_transition_lock_with_timing(
    lock: Any,
    tick: Any,
    *,
    now_ns: Callable[[], int] | None = None,
) -> float:
    """Acquire the shared KV transition lock and record this row's wait."""
    clock = time.perf_counter_ns if now_ns is None else now_ns
    started_ns = int(clock())
    lock.acquire()
    elapsed_ms = round(max(0, int(clock()) - started_ns) / 1_000_000.0, 3)
    tick.microbatch_unified_transition_lock_wait_ms = elapsed_ms
    return elapsed_ms

def _queued_audio_drain_metric_fields(result: Any) -> dict[str, Any]:
    """Flatten successful worker drain provenance for response metrics/events."""
    metrics = getattr(result, "queued_audio_drain_metrics", None)
    if metrics is None:
        return {}
    source_ids = getattr(metrics, "source_frame_ids", ())
    elapsed_ms = float(getattr(metrics, "elapsed_ms", 0.0))
    frame_count = int(getattr(metrics, "frame_count", 0))
    owner_wait_ms = getattr(metrics, "owner_wait_ms", None)
    if owner_wait_ms is None:
        owner_wait_ms = getattr(metrics, "drain_queue_wait_ms", 0.0)
    return {
        "queued_audio_drain_ms": elapsed_ms,
        "queued_audio_drain_frames": frame_count,
        "queued_audio_drain_tokens": int(getattr(metrics, "token_count", 0)),
        "queued_audio_retained_context_source": str(
            getattr(metrics, "retained_context_source", "unknown")
        ),
        "queued_audio_source_frame_ids": [int(value) for value in source_ids],
        "queued_audio_source_queue_epochs": [
            None if value is None else int(value)
            for value in getattr(metrics, "source_queue_epochs", ())
        ],
        "queued_audio_response_id": getattr(metrics, "queue_response_id", None),
        "queued_audio_epoch": getattr(metrics, "queue_epoch", None),
        "queued_audio_first_frame_id": getattr(metrics, "queue_first_frame_id", None),
        "drain_queue_wait_ms": float(getattr(metrics, "drain_queue_wait_ms", 0.0)),
        "owner_wait_ms": float(owner_wait_ms),
        "drain_replay_forward_ms": float(getattr(metrics, "drain_replay_forward_ms", 0.0)),
        "retained_frames_count": int(
            getattr(metrics, "retained_frames_count", frame_count)
        ),
        "drain_total_elapsed_ms": float(
            getattr(metrics, "drain_total_elapsed_ms", elapsed_ms)
        ),
        "drain_transition_lock_wait_ms": float(
            getattr(metrics, "drain_transition_lock_wait_ms", 0.0)
        ),
        "drain_completion_wait_ms": float(
            getattr(metrics, "drain_completion_wait_ms", 0.0)
        ),
    }

def _unified_native_finalizer_event(
    response_id: str,
    *,
    status: str,
    rollback_metrics: Mapping[str, Any] | None = None,
    drain_metrics: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a tokenless event for finalizer work that may outlive response.done."""
    phases: dict[str, Any] = {}
    if isinstance(rollback_metrics, Mapping):
        phases["unified_native_rollback"] = dict(rollback_metrics)
    finalizer = dict(drain_metrics) if isinstance(drain_metrics, Mapping) else {}
    finalizer["status"] = str(status)
    if error:
        finalizer["failure_reason"] = str(error)[:240]
    phases["unified_native_finalizer"] = finalizer
    return {
        "type": "response.unified_native_finalized",
        "response_id": str(response_id),
        "status": str(status),
        "runtime_metrics": phases,
    }

def _filter_unified_native_event_metrics(values: Any) -> dict[str, Any]:
    """Keep the small, typed set of clocks and rollback/drain fields in WS metrics."""
    if not isinstance(values, Mapping):
        return {}
    numeric_fields = {
        "first_qwen_id_published_at_ns",
        "qwen_dynamic_prefill_model_entry_at_ns",
        "qwen_first_id_to_prefill_entry_ms",
        "queued_audio_drain_ms",
        "queued_audio_drain_frames",
        "queued_audio_drain_tokens",
        "queued_audio_first_frame_id",
        "drain_queue_wait_ms",
        "owner_wait_ms",
        "drain_replay_forward_ms",
        "retained_frames_count",
        "drain_total_elapsed_ms",
        "drain_transition_lock_wait_ms",
        "drain_completion_wait_ms",
        "drain_queue_to_finalizer_event_ms",
        "queued_audio_queue_open_epoch",
        "queued_audio_queue_close_epoch",
        "physical_rewind_tokens",
        "committed_generated_tokens",
        "sampled_generated_tokens",
        "unappended_terminal_tokens",
        "transition_tokens_replayed",
        "acknowledged_source_tokens_replayed",
        "acknowledged_source_exclusive_cursor",
        "client_consumed_pcm_ack_sequence",
        "close_tokens_replayed",
        "cache_position_before_rewind",
        "cache_position_after_rebuild",
        "net_cache_cursor_tokens_removed_after_rebuild",
    }
    string_fields = {
        "qwen_prefill_clock",
        "queued_audio_retained_context_source",
        "queued_audio_response_id",
        "queued_audio_epoch",
        "acknowledged_source_ack_provenance",
        "status",
        "failure_reason",
    }
    list_fields = {
        "queued_audio_source_frame_ids",
        "queued_audio_source_queue_epochs",
        "acknowledged_source_token_ids_replayed",
    }
    result: dict[str, Any] = {}
    for key in numeric_fields:
        value = values.get(key)
        if value is None:
            if key in values:
                result[key] = None
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = value
    for key in string_fields:
        value = values.get(key)
        if isinstance(value, str):
            result[key] = value[:240]
    for key in list_fields:
        value = values.get(key)
        if isinstance(value, list):
            compact_value = value[-32:]
            if key == "queued_audio_source_queue_epochs":
                if all(
                    item is None
                    or (isinstance(item, int) and not isinstance(item, bool) and item >= 0)
                    for item in compact_value
                ):
                    result[key] = list(compact_value)
            elif all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 0
                for item in compact_value
            ):
                result[key] = list(compact_value)
    return result


def _compact_unified_native_terminal_metrics(values: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(values, Mapping):
        return {}
    phases: dict[str, dict[str, Any]] = {}
    for phase in (
        "qwen_prefill",
        "unified_native_rollback",
        "unified_native_finalizer",
    ):
        compact_phase = _filter_unified_native_event_metrics(values.get(phase))
        if compact_phase:
            phases[phase] = compact_phase
    return phases

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

def _wait_for_emitted_tts_segment(
    emitted_events: Mapping[str, threading.Event],
    segment_id: str,
    *,
    stop_check: Callable[[], bool] | None = None,
    poll_interval_s: float = 0.05,
) -> bool:
    """Wait until the queued close marker passed the PCM emitter."""
    event = emitted_events.get(str(segment_id))
    if event is None:
        return False
    while not event.wait(max(0.001, float(poll_interval_s))):
        if stop_check is not None and stop_check():
            return False
    return not (stop_check is not None and stop_check())

def _signal_emitted_tts_segment(
    emitted_events: Mapping[str, threading.Event],
    segment_id: str,
) -> bool:
    event = emitted_events.get(str(segment_id))
    if event is None:
        return False
    event.set()
    return True

def _cancel_tts_segment_emission_waiters(
    emitted_events: dict[str, threading.Event],
) -> None:
    for event in tuple(emitted_events.values()):
        event.set()
    emitted_events.clear()

async def _send_unified_native_finalizer_event(
    send: Callable[[dict[str, Any]], Any],
    response_id: str,
    *,
    status: str,
    rollback_metrics: Mapping[str, Any] | None = None,
    drain_metrics: Mapping[str, Any] | None = None,
    runtime_metrics: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> Any:
    """Send the finalizer event without a response-generation fence."""
    event = _unified_native_finalizer_event(
        response_id,
        status=status,
        rollback_metrics=rollback_metrics,
        drain_metrics=drain_metrics,
        error=error,
    )
    compact_phases = {
        phase: compact_phase
        for phase, values in event["runtime_metrics"].items()
        if (compact_phase := _filter_unified_native_event_metrics(values))
    }
    event["runtime_metrics"] = compact_phases
    if isinstance(runtime_metrics, Mapping):
        compact_runtime_metrics = _filter_unified_native_event_metrics(runtime_metrics)
        for phase, values in runtime_metrics.items():
            if isinstance(values, Mapping):
                compact_phase = _filter_unified_native_event_metrics(values)
                if compact_phase:
                    compact_runtime_metrics[str(phase)[:64]] = compact_phase
        if compact_runtime_metrics:
            event["runtime_metrics"].update(compact_runtime_metrics)
    event_size = len(json.dumps(event, separators=(",", ":")).encode("utf-8"))
    if event_size > 2_048:
        phases = event["runtime_metrics"]
        preferred_phases = {
            "qwen_prefill",
            "unified_native_rollback",
            "unified_native_finalizer",
        }
        prune_order = [phase for phase in phases if phase not in preferred_phases]
        prune_order.extend(
            phase
            for phase in (
                "qwen_prefill",
                "unified_native_rollback",
                "unified_native_finalizer",
            )
            if phase in phases
        )
        for phase in prune_order:
            values = phases.get(phase)
            if not isinstance(values, dict):
                continue
            for key in tuple(values):
                if phase == "unified_native_finalizer" and key == "status":
                    continue
                member_size = len(
                    json.dumps(key, separators=(",", ":")).encode("utf-8")
                ) + 1 + len(
                    json.dumps(values[key], separators=(",", ":")).encode("utf-8")
                )
                event_size -= member_size + (1 if len(values) > 1 else 0)
                del values[key]
                if not values and phase != "unified_native_finalizer":
                    phase_size = len(
                        json.dumps(phase, separators=(",", ":")).encode("utf-8")
                    ) + 3
                    event_size -= phase_size + (1 if len(phases) > 1 else 0)
                    del phases[phase]
                if event_size <= 2_048:
                    break
            if event_size <= 2_048:
                break
    return await send(event)

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

class _PrivateControlLeakError(RuntimeError):
    """A generated response echoed its ephemeral control-event prompt text."""

    def __init__(self, action: str, raw_text: str) -> None:
        super().__init__("generated output contains the private control-event marker")
        self.action = str(action)
        self.raw_text = str(raw_text)

__all__ = (
    '_acquire_shared_transition_lock_with_timing',
    '_queued_audio_drain_metric_fields',
    '_unified_native_finalizer_event',
    '_filter_unified_native_event_metrics',
    '_compact_unified_native_terminal_metrics',
    'native_q4_cuda_profile_host_context',
    'attach_native_q4_lock_metrics',
    '_filter_frame_residual_live_hidden_trace',
    '_filter_turn_local_speech_runtime_metrics',
    '_format_turn_local_speech_summary',
    '_submit_or_defer_audio_history_archive',
    '_filter_frame_residual_runtime_metrics',
    '_filter_native_thinker_lock_metrics',
    '_retain_archive_transcript_commitment',
    'native_listener_cuda_profile_effective_directory',
    'write_native_listener_cuda_profile_sidecar',
    '_filter_native_talker_q1_graph_runtime',
    '_alignment_failure_diagnostics',
    '_wait_for_emitted_tts_segment',
    '_signal_emitted_tts_segment',
    '_cancel_tts_segment_emission_waiters',
    '_send_unified_native_finalizer_event',
    '_quiesce_scheduler_for_native_profile',
    '_PrivateControlLeakError',
    '_TTS_QUALITY_DIAGNOSTIC_CAPTURE',
    '_FRAME_RESIDUAL_RUNTIME_METRIC_TYPES',
    '_FRAME_RESIDUAL_LIVE_TRACE_FIELDS',
    'NATIVE_LISTENER_PROFILE_STOP_TIMEOUT_S',
)

