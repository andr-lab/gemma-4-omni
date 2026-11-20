"""Realtime listener callback helpers used by the HTTP/WebSocket server."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from duplex_kernel.server_diagnostics import scheduler_tick_trace_metadata


@dataclass(frozen=True)
class ListenerSilenceSkipEvaluation:
    candidate: bool
    query_coherent: bool
    allow_silence_skip: bool
    fallback_reason: str | None
    transition_lock_acquired: bool


def _prepare_listener_tick_reference(
    sess: Any,
    tick: Any,
    shared_listener_microbatch_enabled: bool,
) -> Any:
    if shared_listener_microbatch_enabled:
        return getattr(tick, "microbatch_reference", None)
    return sess.assistant_reference_buffer.visible_for(
        listener_frame_id=sess.listener_state.next_input_frame_id
    )


def _evaluate_listener_silence_skip(
    sess: Any,
    frame: Any,
    reference: Any,
    speech_tracker: Any,
    phase_changed: bool,
    allow_silence_skip: bool,
    *,
    shared_listener_microbatch_enabled: bool,
    listener_generation: int,
    pending_plan: bool,
) -> ListenerSilenceSkipEvaluation:
    # A silent frame still advances the acoustic and canonical Q2 context.
    del sess, frame, reference, speech_tracker, phase_changed, allow_silence_skip
    del shared_listener_microbatch_enabled, listener_generation, pending_plan
    return ListenerSilenceSkipEvaluation(
        candidate=False, query_coherent=False, allow_silence_skip=False,
        fallback_reason=None, transition_lock_acquired=False,
    )


def _record_listener_commit_timing(tick: Any, commit_timing: dict[str, int]) -> None:
    tick.microbatch_commit_end_ns = time.monotonic_ns()
    wait_ns = commit_timing.get("state_lock_wait_ns")
    hold_ns = commit_timing.get("state_lock_hold_ns")
    tick.microbatch_commit_state_lock_wait_ms = (
        round(wait_ns / 1_000_000.0, 3) if isinstance(wait_ns, int) else None
    )
    tick.microbatch_commit_state_lock_hold_ms = (
        round(hold_ns / 1_000_000.0, 3) if isinstance(hold_ns, int) else None
    )
    tick.microbatch_commit_state_lock_acquired_ns = commit_timing.get(
        "state_lock_acquired_ns"
    )
    tick.microbatch_commit_state_lock_released_ns = commit_timing.get(
        "state_lock_released_ns"
    )
    prepare_ms = commit_timing.get("prepare_ms")
    prepare_ns = commit_timing.get("prepare_ns")
    if isinstance(prepare_ms, (int, float)) and not isinstance(prepare_ms, bool):
        tick.microbatch_commit_prepare_ms = round(float(prepare_ms), 3)
    elif isinstance(prepare_ns, (int, float)) and not isinstance(prepare_ns, bool):
        tick.microbatch_commit_prepare_ms = round(prepare_ns / 1_000_000.0, 3)


def _publish_listener_conditioning_row(
    sess: Any,
    row_output: Any,
    request: Any,
    *,
    request_generation_id: Any,
    worker: Any,
) -> None:
    del sess
    if getattr(row_output, "semantic_context_revision", None) is None:
        return
    mark_applied = getattr(worker, "mark_semantic_prefix_applied", None)
    if not callable(mark_applied):
        return
    from duplex_kernel.listener_microbatch_serving import (
        _semantic_prefix_request_matches_applied_row,
    )

    if not _semantic_prefix_request_matches_applied_row(
        request, request_generation_id, row_output
    ):
        request = row_output
    mark_applied(
        request,
        listener_generation_id=int(row_output.generation_id),
        current_frame_id=int(row_output.frame_id),
        listener_state_version=int(row_output.listener_state_version),
    )


def _build_input_audio_consumed_event(
    sess: Any,
    frame: Any,
    packet_pcm: Any,
    output: Any,
    *,
    tick: Any,
    listener_step_wall_ms: float | None,
    listener_step_started_ns: int | None,
    listener_step_ended_ns: int | None,
    silence_skip_fallback_reason: str | None,
) -> dict[str, Any]:
    del sess, packet_pcm
    output_frame_id = getattr(output, "frame_id", None)
    event: dict[str, Any] = {
        "type": "input_audio_buffer.consumed",
        "frame_id": int(frame.frame_id),
        "server_listener_queue_ms": round(
            max(0, time.monotonic_ns() - int(frame.enqueue_ns)) / 1_000_000.0,
            3,
        ),
        "server_listener_step_ms": listener_step_wall_ms,
        "server_listener_step_start_ns": listener_step_started_ns,
        "server_listener_step_end_ns": listener_step_ended_ns,
        **scheduler_tick_trace_metadata(tick, include_host_lock_timeline=True),
        "_ack_enqueued_ns": time.monotonic_ns(),
    }
    if output is not None:
        if isinstance(output_frame_id, int) and not isinstance(output_frame_id, bool):
            event["listener_output_frame_id"] = int(output_frame_id)
        event["listener_output_source_frame_id"] = getattr(
            output, "source_frame_id", output_frame_id
        )
        event["listener_output_fresh"] = bool(getattr(output, "fresh", True))
        event["listener_silence_skipped"] = bool(
            getattr(output, "silence_skipped", False)
        )
        logical_q_len = getattr(output, "logical_q_len", None)
        ephemeral_reason = getattr(output, "ephemeral_reason", None)
        if ephemeral_reason is not None:
            event["ephemeral_reason"] = str(ephemeral_reason)
        event["queue_audio_for_next_turn"] = bool(
            getattr(output, "queue_audio_for_next_turn", False)
        )
        cache_after = getattr(output, "listener_cache_position", None)
        valid_after = getattr(output, "listener_valid_history_tokens", None)
        if isinstance(logical_q_len, int) and not isinstance(logical_q_len, bool):
            event["logical_q_len"] = logical_q_len
            if isinstance(cache_after, int) and not isinstance(cache_after, bool):
                event["listener_cache_position_after"] = cache_after
                event["listener_cache_position_before"] = cache_after - logical_q_len
            if isinstance(valid_after, int) and not isinstance(valid_after, bool):
                event["listener_valid_history_tokens_after"] = valid_after
                event["listener_valid_history_tokens_before"] = valid_after - logical_q_len
        for field in (
            "kv_rollback_tokens",
            "kv_rollback_enqueue_ms",
            "native_cache_current_position_after",
            "native_cache_valid_kv_len_after",
        ):
            value = getattr(output, field, None)
            if value is not None:
                event[field] = value
    if silence_skip_fallback_reason is not None:
        event["listener_silence_skip_fallback_reason"] = silence_skip_fallback_reason
    return event
