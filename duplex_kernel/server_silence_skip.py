"""Unified prefix refresh and transactional REPLAN helpers."""
from __future__ import annotations

import asyncio
import sys
import time
from contextlib import nullcontext
from typing import Any, Mapping

from duplex_kernel.audio_response_conditioning import (
    build_audio_response_messages,
    build_audio_response_protocol_v2_messages,
)
from duplex_kernel.audio_response_protocol import CONTROLLER_GUIDED_SAY_V1
from duplex_kernel.listener_microbatch_serving import make_unified_prefix_import
from duplex_kernel.native_control6_prompt import (
    NATIVE_CONTROL6_PREFIX_MAX_TOKENS,
    validate_native_control6_prefix_ids,
)
from duplex_kernel.server_audio_snapshot import (
    _cancel_response_branch_playback,
    _snapshot_direct_policy_audio,
)
from duplex_kernel.server_response_control import PLAYBACK_ACK_FINAL_GRACE_SECONDS
from duplex_kernel.server_unified import (
    _configured_audio_response_protocol,
    _require_unified_stable_audio_protocol,
)
from duplex_kernel.session import DuplexSession
from duplex_kernel.turn_policy import ControlAction, GateEvent, PolicyDecision, SpeakerGate



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
        new_revision = old_revision + 1
        prefix_token_ids = _build_unified_listener_prefix_token_ids(
            engine,
            session,
            instruction_override=requested,
        )
        server_module = sys.modules.get("duplex_kernel.server")
        prefix_import_factory = getattr(
            server_module, "make_unified_prefix_import", make_unified_prefix_import
        )
        prefix_import = prefix_import_factory(
            session.listener.backend,
            prefix_token_ids,
            semantic_context_revision=new_revision,
        )
        expected_generation_id = int(binding["generation"])
        expected_cache_position = int(binding["batch_cache_position"])
        if not session.update_instructions(requested, notify_callback=False):
            raise RuntimeError("session instructions changed before prefix refresh")
        if int(session.semantic_context_revision) != new_revision:
            session.instructions = old_instructions
            session.semantic_context_revision = old_revision
            raise RuntimeError("session instruction refresh changed an unexpected semantic revision")

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
        session.notify_history_context_changed()
        raise
    except BaseException:
        rollback_instruction_revision()
        raise
    commit_prefix_binding()
    session.notify_history_context_changed()
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

def _claim_branch_yield_cancel(
    session: Any,
    branch: dict[str, Any],
    *,
    response_id: str,
    generation_id: int,
    speaker_segment_id: str,
) -> bool:
    """Claim one-shot YIELD cancellation on a playback branch and cancel playback."""
    if not response_id or not isinstance(branch, dict):
        return False
    session_lock = getattr(session, "_generation_lock", None)
    branch_lock = branch.get("spoken_prefix_lock")
    claimed = False
    with session_lock if session_lock is not None else nullcontext():
        with branch_lock if branch_lock is not None else nullcontext():
            if (
                branch.get("response_id") != response_id
                or int(branch.get("session_generation_id", -1)) != int(generation_id)
                or str(branch.get("speaker_segment_id", "")) != str(speaker_segment_id)
            ):
                return False
            if branch.get("playback_cancelled") is not True:
                cancelled_at = time.monotonic()
                branch["playback_cancelled_at_monotonic"] = cancelled_at
                branch["playback_ack_final_deadline_monotonic"] = (
                    cancelled_at + PLAYBACK_ACK_FINAL_GRACE_SECONDS
                )
            branch["playback_cancelled"] = True
            branch["cancellation_reason"] = "yield"
            claimed = not bool(branch.get("yield_cancel_enqueued", False))
            if claimed:
                branch["yield_cancel_enqueued"] = True
            if (
                getattr(session, "response_id", None) == response_id
                and int(getattr(session, "session_generation_id", -1)) == int(generation_id)
            ):
                session.response_cancel_notified = True
    _cancel_response_branch_playback(branch)
    return claimed


def _claim_pending_yielded_v2_pcm_cancel(
    session: DuplexSession,
    branch: dict[str, Any],
    *,
    response_id: str,
    generation_id: int,
    speaker_segment_id: str,
) -> str:
    """Backward-compatible status wrapper for the shared YIELD cancel claim."""
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
        if _claim_branch_yield_cancel(
            session,
            branch,
            response_id=response_id,
            generation_id=generation_id,
            speaker_segment_id=speaker_segment_id,
        ):
            return "claimed"
        with branch_lock:
            if (
                branch.get("response_id") != response_id
                or int(branch.get("session_generation_id", -1)) != int(generation_id)
                or str(branch.get("speaker_segment_id", "")) != str(speaker_segment_id)
            ):
                return "stale"
            return "already_cancelled"

from duplex_kernel.server_silence_skip_helpers import (
    _DEFERRED_NATIVE_SPEAK_SOURCE,
    _audio_packet_matches_emission_owner,
    _advance_pending_speak_on_new_onset,
    _listener_evidence_role,
    _listener_v9_stream_for_engine,
    _raw_source_audio_quality,
    _release_pcm_packet_then_ack,
    _scheduler_cuda_stream_context,
    _silence_skip_inputs_are_exact_zero,
    _silence_skip_is_true_idle,
    _silence_skip_query_is_coherent,
    _silence_skip_transition_is_still_idle,
    _user_turn_is_unanswered,
    _wav_b64,
    _yield_admission_rejection,
)
