"""Response admission, native Control6 gating, and pending PCM helpers."""
from __future__ import annotations

import asyncio
import time
from typing import Any, Mapping, Sequence

import numpy as np

from duplex_kernel.backchannel_guard import BACKCHANNEL_MAX_TEXT_TOKENS
from duplex_kernel.audio_response_protocol import (
    AUDIO_RESPONSE_PROTOCOL_V2,
    AUDIO_RESPONSE_V2_MAX_NEW_TOKENS,
)
from duplex_kernel.audio_stream import CODEC_SAMPLE_RATE
from duplex_kernel.config import ARIA_HARD_MAX_FRAMES
from duplex_kernel.server_state import POLICY_FRAME_SAMPLES
from duplex_kernel.turn_policy import (
    ControlAction,
    ControlState,
    GateEvent,
    PolicyDecision,
    SpeakerGate,
)

LONG_RESPONSE_PROFILE = "long"
STANDARD_RESPONSE_PROFILE = "standard"
LONG_RESPONSE_MAX_TEXT_TOKENS = 768
LONG_RESPONSE_MAX_CODEC_FRAMES = ARIA_HARD_MAX_FRAMES
PLAYBACK_ACK_FINAL_GRACE_SECONDS = 2.0


def _normalize_response_profile(value: Any) -> str:
    """Normalize an explicit response profile without inspecting its content."""
    if value is None:
        return STANDARD_RESPONSE_PROFILE
    if not isinstance(value, str):
        raise ValueError("response_profile deve essere 'standard' o 'long'")
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"standard", "ordinary", "default"}:
        return STANDARD_RESPONSE_PROFILE
    if normalized in {"long", "long_response"}:
        return LONG_RESPONSE_PROFILE
    raise ValueError("response_profile deve essere 'standard' o 'long'")


def should_emit_final_text_fallback(*, published_delta_count: int, result_text: str) -> bool:
    """Emit final text only when this response published no incremental deltas."""
    return published_delta_count == 0 and bool(result_text)



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

def _controller_guided_say_generation_options(
    *,
    action: Any,
    controller_guided_say: bool,
    direct_audio_candidate: bool,
    response_protocol: str | None,
    response_profile: str | None = None,
) -> dict[str, int]:
    """Return the production request budgets for the selected response route."""
    normalized_profile = _normalize_response_profile(response_profile)
    action_value = getattr(action, "value", action)
    normalized_action = str(action_value).strip().upper() if action_value is not None else None
    v2_audio_response = (
        direct_audio_candidate and response_protocol == AUDIO_RESPONSE_PROTOCOL_V2
    )
    if normalized_action == ControlAction.BACKCHANNEL.value and not v2_audio_response:
        return {"max_tokens": BACKCHANNEL_MAX_TEXT_TOKENS, "_min_new_tokens": 0}
    if normalized_profile == LONG_RESPONSE_PROFILE:
        return {"max_tokens": LONG_RESPONSE_MAX_TEXT_TOKENS, "_min_new_tokens": 1}
    is_vocal_action = normalized_action in {ControlAction.SPEAK.value, ControlAction.BACKCHANNEL.value}
    if v2_audio_response:
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

def _native_control6_policy_hidden_for_tick(
    *, user_pcm: Any, speaker_active: bool, listener_hidden: Any | None
) -> tuple[Any | None, bool]:
    """Keep the genuine current-frame hidden, including exact-zero PCM."""
    del user_pcm, speaker_active
    return listener_hidden, False


def _request_policy_response_revision(
    session: Any,
    decision: PolicyDecision,
    branch: Mapping[str, Any] | None,
) -> bool:
    """Authorize a candidate only from REPLAN for the current live response."""
    if decision.action is not ControlAction.REPLAN or not isinstance(branch, Mapping):
        return False
    with session._generation_lock:
        if (
            decision.session != session.session_id
            or decision.generation_id != session.session_generation_id
            or not session.active_response
            or session.control_state is not ControlState.SPEAKING
            or branch.get("response_id") != session.response_id
            or branch.get("session_generation_id") != decision.generation_id
            or branch.get("revisioned_live_say") is not True
        ):
            return False
        with branch["spoken_prefix_lock"]:
            if branch.get("playback_cancelled") or branch.get("finalized"):
                return False
            request_revision = getattr(branch.get("revision_coordinator"), "request_revision", None)
            return bool(callable(request_revision) and request_revision(int(decision.frame_id)))


def _native_control6_speaker_active(
    *,
    active_response: bool,
    control_state: ControlState,
    pending_pcm_playback: bool = False,
    closing: bool = False,
) -> bool:
    """Resolve the one native Control6 speaking state for the whole policy tick."""
    if closing and not pending_pcm_playback:
        return False
    return bool(
        active_response
        or pending_pcm_playback
        or control_state in {
            ControlState.SPEAKING,
            ControlState.BACKCHANNELING,
            ControlState.STOPPING,
        }
    )


def _native_control6_worker_fail_closed_context(
    *,
    diagnostics: Any,
    decision: PolicyDecision,
    speaker_active: bool,
    active_response: bool,
    response_branch: Mapping[str, Any] | None,
) -> bool:
    """Whether a Q2 context loss should stop an actively speaking response.

    A response branch that is closing, has drained playback, or has no pending
    client PCM cannot represent loss of the live speaking context.
    """
    branch_is_draining_or_closing = False
    if isinstance(response_branch, dict):
        branch_is_draining_or_closing = bool(
            response_branch.get("closing")
            or response_branch.get("playback_drained")
            or not _response_branch_has_pending_pcm(response_branch)
        )
    return bool(
        isinstance(diagnostics, dict)
        and diagnostics.get("system_fail_closed_action") is True
        and decision.reason == "stable_q2_context_unavailable_fail_closed"
        and decision.action is ControlAction.HARD_STOP
        and speaker_active
        and active_response
        and not branch_is_draining_or_closing
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
    cancellation_deadline = branch.get("playback_ack_final_deadline_monotonic")
    if (
        branch.get("playback_cancelled") is True
        and isinstance(cancellation_deadline, (int, float))
        and time.monotonic() >= float(cancellation_deadline)
    ):
        return False
    segments = branch.get("playback_segments")
    if not isinstance(segments, Mapping):
        return False
    for segment_id, segment in segments.items():
        if not isinstance(segment, Mapping):
            continue
        if (
            speaker_segment_id is not None
            and segment.get("speaker_segment_id") != speaker_segment_id
        ):
            continue
        start = segment.get("start_sample")
        emitted_end = segment.get("emitted_end_sample")
        rendered_end = segment.get("rendered_end_sample")
        # A browser ACK says that PCM entered its output callback. When a
        # renderer watermark exists, use it for speaker activity and retain
        # the logical ACK frontier as a separate history/rollback fact.
        consumed_end = (
            rendered_end
            if isinstance(rendered_end, int) and not isinstance(rendered_end, bool)
            else segment.get("client_consumed_end_sample", start)
        )
        if (
            isinstance(start, int) and not isinstance(start, bool)
            and isinstance(emitted_end, int) and not isinstance(emitted_end, bool)
            and isinstance(consumed_end, int) and not isinstance(consumed_end, bool)
            and emitted_end > consumed_end
        ):
            return True
    return False


def _record_rendered_pcm_watermark(
    branch: dict[str, Any],
    *,
    speaker_segment_id: str,
    rendered_end_sample: int,
) -> bool:
    """Advance the physical renderer frontier without changing the client ACK."""
    segment_id = str(speaker_segment_id)
    value = rendered_end_sample
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return False
    lock = branch.get("spoken_prefix_lock")
    context = lock if lock is not None else _NullLock()
    with context:
        segments = branch.get("playback_segments")
        segment = segments.get(segment_id) if isinstance(segments, Mapping) else None
        if not isinstance(segment, dict):
            return False
        start = segment.get("start_sample")
        emitted_end = segment.get("emitted_end_sample")
        buffered_end = segment.get("client_consumed_end_sample", start)
        previous = segment.get("rendered_end_sample", start)
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (start, emitted_end, buffered_end, previous)
        ):
            return False
        if not (int(start) <= int(previous) <= value <= int(buffered_end) <= int(emitted_end)):
            return False
        segment["rendered_end_sample"] = value
        segment["rendered_pcm_samples"] = value - int(start)
        return True


class _NullLock:
    """Context manager for branches without a response lock."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: Any) -> None:
        return None


def _response_branch_ack_retained(
    branch: Mapping[str, Any],
    *,
    now_monotonic: float | None = None,
) -> bool:
    """Keep branch ACK identity while PCM is pending or cancellation can finalize.

    The cancellation grace interval is independent of pending PCM: the browser
    may report its final device-buffer watermark after cancellation removed the
    server-side pending range. Once the interval expires, the normal pending
    PCM predicate also fails closed for cancelled branches.
    """
    if not isinstance(branch, Mapping):
        return False
    now = time.monotonic() if now_monotonic is None else float(now_monotonic)
    deadline = branch.get("playback_ack_final_deadline_monotonic")
    if branch.get("playback_cancelled") is True:
        return isinstance(deadline, (int, float)) and now <= float(deadline)
    return _response_branch_has_pending_pcm(branch)

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
    """Cancel response output, wake waiters, and retain final client ACKs."""
    callback = None

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
        if branch.get("playback_cancelled") is not True:
            cancelled_at = time.monotonic()
            branch["playback_cancelled_at_monotonic"] = cancelled_at
            branch["playback_ack_final_deadline_monotonic"] = (
                cancelled_at + PLAYBACK_ACK_FINAL_GRACE_SECONDS
            )
        branch["playback_cancelled"] = True
        if branch.get("playback_cancel_notified") is not True:
            branch["playback_cancel_notified"] = True
            callback = branch.get("playback_cancel_callback")
    else:
        with lock:
            if branch.get("playback_cancelled") is not True:
                cancelled_at = time.monotonic()
                branch["playback_cancelled_at_monotonic"] = cancelled_at
                branch["playback_ack_final_deadline_monotonic"] = (
                    cancelled_at + PLAYBACK_ACK_FINAL_GRACE_SECONDS
                )
            branch["playback_cancelled"] = True
            if branch.get("playback_cancel_notified") is not True:
                branch["playback_cancel_notified"] = True
                callback = branch.get("playback_cancel_callback")
    try:
        if callable(callback):
            callback()
    except Exception as exc:
        branch["playback_cancel_callback_error"] = f"{type(exc).__name__}: {exc}"[:240]
    finally:
        wake()

def _playback_drain_timeout_s(branch: Mapping[str, Any]) -> float:
    """Bound normal ACK waits by duration; cancellation waits at most two seconds."""
    if branch.get("playback_cancelled") is True:
        deadline = branch.get("playback_ack_final_deadline_monotonic")
        if isinstance(deadline, (int, float)):
            return max(0.001, min(2.0, float(deadline) - time.monotonic()))
        return 0.001
    segments = branch.get("playback_segments")
    consumed = set(branch.get("client_consumed_segments", ()))
    pending_samples = 0
    if isinstance(segments, Mapping):
        for segment_id, segment in segments.items():
            if str(segment_id) in consumed or not isinstance(segment, Mapping):
                continue
            start = segment.get("start_sample")
            end = segment.get("emitted_end_sample")
            if (
                isinstance(start, int) and not isinstance(start, bool)
                and isinstance(end, int) and not isinstance(end, bool)
                and end > start
            ):
                pending_samples += end - start
    return max(5.0, min(180.0, pending_samples / CODEC_SAMPLE_RATE + 5.0))


async def _wait_for_response_playback_drain(
    branch: dict[str, Any],
    *,
    timeout_s: float | None = None,
) -> None:
    """Wait for logical client consumption, fail closed on missing or late ACKs.

    Cancellation is a separate terminal condition and never advances the
    consumed prefix.  The timeout covers the outstanding PCM duration plus a
    five-second delivery allowance, capped at three minutes.
    """
    lock = branch.get("spoken_prefix_lock")
    event = branch.get("playback_ack_event")
    if lock is None:
        if not _response_branch_has_pending_pcm(branch):
            return
    else:
        with lock:
            if not _response_branch_has_pending_pcm_locked(branch):
                return
    if not isinstance(event, asyncio.Event):
        raise RuntimeError("client playback drain has pending PCM but no ACK event")
    budget = _playback_drain_timeout_s(branch) if timeout_s is None else float(timeout_s)
    if not (budget > 0.0):
        raise ValueError("client playback drain timeout must be positive")
    deadline = asyncio.get_running_loop().time() + budget
    cancellation_wait = branch.get("playback_cancelled") is True
    while True:
        if branch.get("playback_cancelled") is True:
            cancellation_wait = True
            final_deadline = branch.get("playback_ack_final_deadline_monotonic")
            if isinstance(final_deadline, (int, float)):
                cancellation_budget = max(0.0, float(final_deadline) - time.monotonic())
                deadline = min(
                    deadline,
                    asyncio.get_running_loop().time() + cancellation_budget,
                )
        if lock is None:
            event.clear()
            if not _response_branch_has_pending_pcm(branch):
                return
        else:
            with lock:
                event.clear()
                if not _response_branch_has_pending_pcm_locked(branch):
                    return
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        try:
            await asyncio.wait_for(event.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            break
    cancellation_wait = branch.get("playback_cancelled") is True
    if lock is None:
        still_pending = _response_branch_has_pending_pcm(branch)
        cancellation_wait = cancellation_wait or branch.get("playback_cancelled") is True
    else:
        with lock:
            still_pending = _response_branch_has_pending_pcm_locked(branch)
            cancellation_wait = cancellation_wait or branch.get("playback_cancelled") is True
    if still_pending and not cancellation_wait:
        raise RuntimeError(
            f"client playback ACK wait timed out after {budget:.3f}s; pending PCM was not committed"
        )
    cancellation_unconfirmed = False
    if cancellation_wait:
        segments = branch.get("playback_segments")
        if isinstance(segments, Mapping):
            cancellation_unconfirmed = any(
                isinstance(segment, Mapping)
                and segment.get("client_consumed_final") is not True
                for segment in segments.values()
            )
    if (still_pending or cancellation_unconfirmed) and cancellation_wait:
        branch["client_playback_ack_missing"] = True
        branch["client_playback_ack_status"] = "cancelled_final_ack_timeout"
