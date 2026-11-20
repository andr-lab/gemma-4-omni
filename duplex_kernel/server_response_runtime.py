from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import queue
import re
import threading
import time
import unicodedata
import uuid
from dataclasses import replace
from contextlib import contextmanager, nullcontext
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from duplex_kernel.action_conditioning import (
    action_conditioned_messages,
    action_conditioning_instruction,
    action_control_marker_text,
)
from duplex_kernel.audio_response_conditioning import (
    build_audio_response_messages,
    build_audio_response_protocol_v2_messages,
    build_controller_guided_say_v1_messages,
)
from duplex_kernel.audio_response_protocol import (
    AUDIO_RESPONSE_ACTIONS,
    AUDIO_RESPONSE_PROTOCOL_V2,
    AUDIO_RESPONSE_V2_FAILURE_TRACE_MAX_RECORDS,
    AUDIO_RESPONSE_V2_MAX_ACTION_TOKENS,
    AUDIO_RESPONSE_V2_MAX_HEARD_TOKENS,
    AUDIO_RESPONSE_V2_MAX_NEW_TOKENS,
    AUDIO_RESPONSE_V2_MAX_SAY_TOKENS,
    CONTROLLER_GUIDED_SAY_V1,
    AudioResponseProtocolError,
    AudioResponseV2ActionConstraint,
    AudioResponseV2Collector,
    audio_response_actions_for_wake,
    audio_response_v2_say_token_char_spans,
    controller_backchannel_candidate_allowed,
    controller_backchannel_should_stop,
    controller_guided_response_action,
    finish_audio_response_v2_with_say_hidden,
    normalize_audio_response_protocol,
    resolve_audio_response_protocol,
    slice_controller_guided_say_hidden,
)
from duplex_kernel.backchannel_guard import (
    BACKCHANNEL_MAX_TEXT_TOKENS,
    validate_backchannel_phrase,
)
from duplex_kernel.config import (
    ARIA_BACKCHANNEL_MAX_FRAMES,
    ARIA_MAX_FRAMES,
    DEFAULT_VOICE,
    USER_SAMPLE_RATE,
)
from duplex_kernel.live_say import LiveQwenSayStream, LiveSayCancelled, TurnLocalQwenSayStream
from duplex_kernel.models.thinker import normalize_eos_token_ids
from duplex_kernel.response_intent_local import (
    align_generated_response_hidden_to_qwen_tokens,
    align_hidden_states_by_char_overlap,
    qwen_assistant_text_offsets,
)
from duplex_kernel.server_diagnostics import (
    _PrivateControlLeakError,
    _TTS_QUALITY_DIAGNOSTIC_CAPTURE,
    _alignment_failure_diagnostics,
    _cancel_tts_segment_emission_waiters,
    _format_turn_local_speech_summary,
    _wait_for_emitted_tts_segment,
)
from duplex_kernel.tts_quality_diagnostics import DIAGNOSTIC_MAX_ACOUSTIC_FRAME_BUDGET
from duplex_kernel.server_speech_ack import (
    _mark_unified_say_packet_committed,
    _publish_acknowledged_speech_prefix,
    _record_unified_native_profile_timestamp,
    _register_response_speech_ack_streams,
    _require_closed_tts_codec_eos,
    _validate_unified_say_packet_frontier,
    _wait_for_qwen_dynamic_prefill_model_entry,
)
from duplex_kernel.response_revision_runtime import create_live_say_revision_coordinator
from duplex_kernel.server_state import (
    POLICY_FRAME_SAMPLES,
    REALTIME_AUDIO_PACKET_CAPACITY,
    _verify_v9_training_reference,
)
from duplex_kernel.server_response_control import (
    LONG_RESPONSE_MAX_CODEC_FRAMES,
    LONG_RESPONSE_MAX_TEXT_TOKENS,
    LONG_RESPONSE_PROFILE,
    _normalize_response_profile,
)
from duplex_kernel.server_unified import (
    _build_unified_audio_response_suffix,
    _build_unified_backchannel_response_suffix,
    _configured_audio_response_protocol,
    _run_unified_native_thinker,
    _unified_audio_template_ids,
    _unified_turn_end_token_id,
    _validate_unified_audio_turn_end_suffix,
)
from duplex_kernel.session import DuplexSession, GemmaTokenTickPacer, SpokenTokenPacket
from duplex_kernel.tool_parser import extract_tool_calls, tool_system_prompt
from duplex_kernel.tts_units import SpokenTextUnit, SpokenTextUnitAssembler
from duplex_kernel.turn_policy import ControlAction
from duplex_kernel.voices import resolve_voice_conditioning
from duplex_kernel.thinker_sampler import CandidateFallbackPool
from duplex_kernel.unified_native_decoder import NativeDecodeSessionStatus

logger = logging.getLogger("duplex_kernel.server_response")


def _resume_adopted_controller(
    adopted_controller: Mapping[str, Any],
    unified_branch: Mapping[str, Any],
) -> dict[str, Any] | None:
    branch_revision_id = str(unified_branch.get("revision_id", ""))
    adopted_revision_id = str(adopted_controller.get("revision_id", ""))
    adopted_session = adopted_controller.get("session")
    session_status = getattr(adopted_session, "status", None)
    session_error = (
        adopted_controller.get("fatal_error")
        or getattr(adopted_session, "fatal_error", None)
    )
    is_registered_adoption = bool(
        unified_branch.get("_adopted_native_session_controller") is adopted_controller
        and branch_revision_id
        and adopted_revision_id == branch_revision_id
        and str(unified_branch.get("_streaming_revision_id", branch_revision_id))
        == branch_revision_id
        and adopted_controller.get("promoted") is True
        and adopted_session is not None
        and not isinstance(session_error, BaseException)
        and session_status is not NativeDecodeSessionStatus.FAILED
        and not bool(getattr(adopted_session, "_failed", False))
    )
    if not is_registered_adoption:
        if isinstance(session_error, BaseException):
            raise session_error
        raise RuntimeError("adopted native controller is not current and healthy")

    if bool(getattr(adopted_session, "_closed", False)):
        final_result = adopted_controller.get("final_result")
        if adopted_controller.get("done") and isinstance(final_result, Mapping):
            result = dict(final_result)
            result["revision_handoff"] = True
            return result
        raise RuntimeError("adopted native controller is closed without a final result")

    resume = adopted_controller.get("run_to_completion")
    if not callable(resume):
        raise RuntimeError("adopted native controller has no completion entrypoint")
    try:
        think = resume()
        if not isinstance(think, Mapping):
            raise TypeError("adopted controller result must be a mapping")
        result = dict(think)
        result["revision_handoff"] = True
        return result
    except Exception as exc:
        logger.warning("adopted controller resume failed: %s", exc)
        session_error = (
            adopted_controller.get("fatal_error")
            or getattr(adopted_session, "fatal_error", None)
        )
        if (
            isinstance(session_error, BaseException)
            or getattr(adopted_session, "status", None) is NativeDecodeSessionStatus.FAILED
            or bool(getattr(adopted_session, "_failed", False))
        ):
            if isinstance(session_error, BaseException):
                raise session_error
            raise
        adopted_think = unified_branch.get("_last_adopted_think")
        if isinstance(adopted_think, Mapping):
            result = dict(adopted_think)
            result["revision_handoff"] = True
            return result
        raise


def should_stream_thinker_text(
    *,
    stream_requested: bool,
    unified_native: bool,
    tts_text_consumed_callback: Any,
) -> bool:
    """Avoid publishing a second text stream when unified Qwen owns publication."""
    return bool(
        stream_requested
        and not (unified_native and callable(tts_text_consumed_callback))
    )


def _unified_direct_audio_cache_branch_is_live(
    branch: Any,
    response_protocol: str | None,
) -> bool:
    """Accept a private decode owner for either stable audio response protocol."""
    if not isinstance(branch, dict) or response_protocol not in {
        AUDIO_RESPONSE_PROTOCOL_V2,
        CONTROLLER_GUIDED_SAY_V1,
    }:
        return False
    if branch.get("queue_open") or branch.get("ephemeral_decode_handle") is not None:
        return True
    return branch.get("response_decode_handle") is not None


def _configure_native_live_say_callbacks(
    think_kwargs: dict[str, Any],
    *,
    frame_residual_active: bool,
    packet_callback: Any,
    committed_callback: Any,
) -> None:
    """Use each token's post-forward hidden when live FiLM needs it."""
    think_kwargs["anticipate_first_token"] = not bool(frame_residual_active)
    think_kwargs["on_token_packet"] = packet_callback
    think_kwargs["on_token_committed"] = committed_callback
    think_kwargs["on_generated_token"] = None

from duplex_kernel.server_response_builder import (
    _empty_controller_say_think_from_eos,
    _claim_audio_response_v2_action_probe,
    _is_v2_no_speech_result,
    _record_audio_response_v2_private_failure,
    _resolve_request_voice_conditioning,
    _diagnostic_talker_rng_scope,
    _append_native_action_token,
    _prepare_direct_audio_processor_inputs,
    _direct_audio_prompt_history,
    _align_generated_response_hidden_with_diagnostics,
    _qwen_text_token_surfaces,
    _backchannel_budget_without_eos,
    _publish_backchannel_tokens_after_eos,
    _message_text,
)

def _response_generation_max_new_tokens(
    *,
    response_v2_enabled: bool,
    private_backchannel_active: bool,
    backchannel_action: bool,
    requested_max_tokens: int,
) -> int:
    if response_v2_enabled:
        return min(int(requested_max_tokens), AUDIO_RESPONSE_V2_MAX_NEW_TOKENS)
    if private_backchannel_active:
        return BACKCHANNEL_MAX_TEXT_TOKENS
    if backchannel_action:
        return min(int(requested_max_tokens), BACKCHANNEL_MAX_TEXT_TOKENS)
    return int(requested_max_tokens)


def _decode_backchannel_source_text(think: Mapping[str, Any], tokenizer: Any) -> str | None:
    """Reconstruct the validated phrase from the original generated Gemma IDs."""
    protocol_text = think.get("protocol_text")
    token_ids = think.get("gen_ids")
    decode = getattr(tokenizer, "decode", None)
    if not isinstance(token_ids, (tuple, list)) or not callable(decode):
        return None
    try:
        value = decode(token_ids, skip_special_tokens=True)
    except TypeError:
        try:
            value = decode(token_ids)
        except Exception:
            return None
    except Exception:
        return None
    if not isinstance(value, str):
        return None
    if isinstance(protocol_text, str) and protocol_text != value:
        return None
    return value


def _response_completion_diagnostics(
    *,
    token_limit_requested: int,
    token_limit: int,
    token_limit_effective: int | None = None,
    token_limit_cap_reason: str | None = None,
    tokens_generated: int,
    text_terminal_reason: str | None,
    codec_metrics: Mapping[str, Any] | None,
    audio_emitted: bool,
    cancellation_reason: str | None = None,
) -> dict[str, Any]:
    """Combine text and codec terminal evidence without rewriting either source."""
    metrics = codec_metrics if isinstance(codec_metrics, Mapping) else {}
    units = metrics.get("closed_tts_units")
    unit_metrics = [item for item in units if isinstance(item, Mapping)] if isinstance(units, list) else []

    def first_int(keys: Sequence[str]) -> int | None:
        for key in keys:
            value = metrics.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return int(value)
        return None

    codec_frame_limit = first_int(("codec_frame_limit", "effective_max_frames", "configured_max_frames"))
    codec_frames_produced = first_int(("codec_frames_produced", "generated_cb0_frames", "rvq_frames"))
    if unit_metrics:
        unit_limits = [
            int(item[key])
            for item in unit_metrics
            for key in ("codec_frame_limit", "effective_max_frames", "configured_max_frames")
            if isinstance(item.get(key), int) and not isinstance(item.get(key), bool) and item[key] >= 0
        ]
        unit_frames = [
            int(item[key])
            for item in unit_metrics
            for key in ("codec_frames_produced", "generated_cb0_frames", "rvq_frames")
            if isinstance(item.get(key), int) and not isinstance(item.get(key), bool) and item[key] >= 0
        ]
        if codec_frame_limit is None and unit_limits:
            codec_frame_limit = sum(unit_limits)
        if codec_frames_produced is None and unit_frames:
            codec_frames_produced = sum(unit_frames)

    eos_detected = metrics.get("eos_detected")
    if not isinstance(eos_detected, bool) and unit_metrics:
        unit_eos = [item.get("eos_detected") for item in unit_metrics]
        if unit_eos and all(isinstance(value, bool) for value in unit_eos):
            eos_detected = all(unit_eos)
    if not isinstance(eos_detected, bool):
        eos_detected = False

    raw_text_reason = str(text_terminal_reason) if isinstance(text_terminal_reason, str) else None
    normalized_cancellation = (
        str(cancellation_reason).strip()
        if isinstance(cancellation_reason, str) and cancellation_reason.strip()
        else None
    )
    if normalized_cancellation is None and raw_text_reason in {
        "cancelled", "cancelled_before_first_token", "cancelled_after_token",
        "cancelled_before_token_forward",
    }:
        normalized_cancellation = raw_text_reason

    token_limited = raw_text_reason in {
        "length", "budget_exhausted", "budget_partial", "sentence_boundary_after_budget",
        "producer_limit",
    }
    codec_limited = bool(
        not eos_detected
        and (
            metrics.get("terminated_by_budget") is True
            or any(item.get("terminated_by_budget") is True for item in unit_metrics)
            or (
                codec_frame_limit is not None
                and codec_frames_produced is not None
                and codec_frame_limit > 0
                and codec_frames_produced >= codec_frame_limit
            )
        )
    )
    if normalized_cancellation is not None or metrics.get("cancelled") is True:
        finish_reason = "cancelled"
    elif token_limited or codec_limited:
        finish_reason = "length"
    elif raw_text_reason in {"eos", "stop"} and (not audio_emitted or eos_detected):
        finish_reason = "stop"
    else:
        # Missing or contradictory EOS evidence cannot certify natural completion.
        finish_reason = "length"

    return {
        "token_limit_requested": max(0, int(token_limit_requested)),
        "token_limit": max(0, int(token_limit if token_limit_effective is None else token_limit_effective)),
        "token_limit_effective": max(0, int(token_limit if token_limit_effective is None else token_limit_effective)),
        "token_limit_cap_reason": token_limit_cap_reason,
        "tokens_generated": max(0, int(tokens_generated)),
        "text_terminal_reason": raw_text_reason,
        "codec_frame_limit": codec_frame_limit,
        "codec_frames_produced": codec_frames_produced or 0,
        "eos_detected": bool(eos_detected),
        "cancellation_reason": normalized_cancellation,
        "finish_reason": finish_reason,
    }


def _thinker_callback_admission(
    *,
    action: str | None,
    stream_requested: bool,
    response_v2_active: bool,
    controller_guided_say: bool,
) -> tuple[bool, bool]:
    """Decide which raw Thinker callbacks may escape before response validation."""
    if str(action or "").strip().upper() == ControlAction.BACKCHANNEL.value:
        return False, False
    return (
        bool(stream_requested and not response_v2_active and not controller_guided_say),
        bool(not response_v2_active and not controller_guided_say),
    )


def run_response(
    engine: Any,
    sess: DuplexSession,
    messages: list[dict[str, Any]],
    request: dict[str, Any],
    *,
    action: str | None = None,
    history_is_committed: bool = False,
    unified_kv_enabled: bool = False,
) -> dict[str, Any]:
    clean = [m for m in messages if isinstance(m, dict) and m.get("role") in {"system", "user", "assistant", "tool"}]
    listener_admission_wait = request.get("_listener_admission_wait")
    listener_compute_quantum = request.get("_listener_compute_quantum")
    if sess.instructions and not any(m.get("role") == "system" for m in clean):
        clean.insert(0, {"role": "system", "content": sess.instructions})
    tools = request.get("tools") or sess.tools
    stream_requested = callable(request.get("_on_audio_chunk")) and not tools
    if tools:
        clean.insert(0, {"role": "system", "content": tool_system_prompt(tools)})
    generation_id = request.get("_generation_id")
    response_profile = _normalize_response_profile(
        request.get("response_profile", getattr(sess, "response_profile", "standard"))
    )
    requested_token_limit = max(
        0,
        int(request.get(
            "max_tokens",
            LONG_RESPONSE_MAX_TEXT_TOKENS
            if response_profile == LONG_RESPONSE_PROFILE else 64,
        )),
    )
    profile_token_budget = min(
        requested_token_limit,
        LONG_RESPONSE_MAX_TEXT_TOKENS
        if response_profile == LONG_RESPONSE_PROFILE
        else requested_token_limit,
    )
    response_codec_frame_limit = (
        LONG_RESPONSE_MAX_CODEC_FRAMES
        if response_profile == LONG_RESPONSE_PROFILE
        else ARIA_MAX_FRAMES
    )

    def preflight_cancelled_result() -> dict[str, Any]:
        reason = request.get("_cancellation_reason") or getattr(sess, "stop_reason", None) or "cancelled"
        completion = _response_completion_diagnostics(
            token_limit_requested=requested_token_limit,
            token_limit=profile_token_budget,
            token_limit_effective=profile_token_budget,
            token_limit_cap_reason=(
                "profile_max"
                if profile_token_budget < requested_token_limit else None
            ),
            tokens_generated=0,
            text_terminal_reason=None,
            codec_metrics={},
            audio_emitted=False,
            cancellation_reason=str(reason),
        )
        return {
            "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
            "cancelled": True, "finish_reason": completion["finish_reason"],
            "response_profile": response_profile,
            "runtime_metrics": {"response_completion": completion},
        }

    if generation_id is not None and not sess.is_generation_current(generation_id=generation_id):
        sess.stale_session_mutations += 1
        return preflight_cancelled_result()
    normalized_action = str(action).strip().upper() if action is not None else None
    raw_audio_chunk_callback = request.get("_on_audio_chunk")
    raw_audio_chunk_with_horizon_callback = request.get("_on_audio_chunk_with_horizon")
    backchannel_pcm_emitted = False

    def note_backchannel_pcm(pcm: Any) -> None:
        nonlocal backchannel_pcm_emitted
        if normalized_action == ControlAction.BACKCHANNEL.value and pcm is not None:
            try:
                backchannel_pcm_emitted = backchannel_pcm_emitted or bool(
                    np.asarray(pcm).size
                )
            except (TypeError, ValueError):
                numel = getattr(pcm, "numel", None)
                if callable(numel):
                    backchannel_pcm_emitted = backchannel_pcm_emitted or int(numel()) > 0

    def tracked_audio_chunk_callback(*args: Any, **kwargs: Any) -> Any:
        note_backchannel_pcm(args[0] if args else kwargs.get("pcm"))
        if callable(raw_audio_chunk_callback):
            return raw_audio_chunk_callback(*args, **kwargs)
        return None

    def tracked_audio_chunk_with_horizon_callback(*args: Any, **kwargs: Any) -> Any:
        note_backchannel_pcm(args[0] if args else kwargs.get("pcm"))
        if callable(raw_audio_chunk_with_horizon_callback):
            return raw_audio_chunk_with_horizon_callback(*args, **kwargs)
        return None

    audio_chunk_callback = tracked_audio_chunk_callback if callable(raw_audio_chunk_callback) else None
    audio_chunk_with_horizon_callback = (
        tracked_audio_chunk_with_horizon_callback
        if callable(raw_audio_chunk_with_horizon_callback) else None
    )
    controller_backchannel_text: list[str] = []
    controller_backchannel_packets: list[tuple[int, int, str, Any]] = []
    response_id = str(
        request.get("_response_id")
        or request.get("response_id")
        or f"resp_{uuid.uuid4().hex[:12]}"
    )
    runtime_metrics_cb = request.get("_runtime_metrics_callback")
    def update_response_runtime_metrics(resp_id: str, phase: str, values: Any) -> None:
        if callable(runtime_metrics_cb):
            try:
                runtime_metrics_cb(phase, values)
            except Exception:
                pass
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
    if unified_native and not _unified_direct_audio_cache_branch_is_live(
        unified_branch,
        response_protocol,
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
    nonunified_v1_source_stream = bool(
        live_say_active
        and not unified_native
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
            return preflight_cancelled_result()
        sess.thinker_ctx.system_prompt = sess.instructions
        sess.thinker_ctx.set_messages(clean)
        clean = sess.thinker_ctx.messages()
    completed_history = [
        dict(message)
        for message in clean
        if message.get("role") in {"user", "assistant"}
    ][-16:]
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
    live_say_publication_lock = threading.RLock()
    live_say_cancel = threading.Event()
    talker_limit_reached = threading.Event()
    live_qwen_token_capacity = (
        max(ARIA_MAX_FRAMES, profile_token_budget * 4)
        if response_profile == LONG_RESPONSE_PROFILE
        else ARIA_MAX_FRAMES
    )
    live_qwen_source_pieces = max(2_048, live_qwen_token_capacity * 4)
    live_qwen_source_chars = max(16_384, live_qwen_token_capacity * 32)
    live_qwen_retokenization_work = max(
        1_000_000,
        int(2_000_000 * ((live_qwen_token_capacity / 512) ** 2)),
    )
    live_tts_unit_assembler: SpokenTextUnitAssembler | None = None
    live_tts_unit_queue: queue.Queue[Any] | None = None
    live_tts_unit_sentinel = object()
    live_tts_unit_metrics: list[dict[str, Any]] = []
    live_tts_unit_wavs: list[np.ndarray] = []
    live_tts_visible_text_parts: list[str] = []
    live_tts_active_stream: dict[str, TurnLocalQwenSayStream | None] = {"value": None}
    live_tts_streams_by_segment: dict[str, TurnLocalQwenSayStream] = {}
    live_tts_segment_emitted_events: dict[str, threading.Event] = {}
    live_say_eos_text_id: int | None = None
    live_say_result: dict[str, Any] = {}
    live_say_prefill_entry = threading.Event()
    live_say_timing: dict[str, int | None] = {
        "consumer_started_at_ns": None,
        "thinker_returned_at_ns": None,
    }
    thinker_first_token_id: int | None = None
    streamed_live_chars = 0
    live_say_tokenizer = None
    response_ack_branch = (
        unified_branch
        if isinstance(unified_branch, dict)
        else request.get("_playback_ack_branch")
    )
    nonunified_v2_source_rows: tuple[dict[str, Any], ...] = ()
    revision_coordinator_holder: dict[str, Any] = {"value": None}
    revisioned_live_say = bool(
        unified_native
        and normalized_action == ControlAction.SPEAK.value
        and response_protocol == CONTROLLER_GUIDED_SAY_V1
        and isinstance(unified_branch, dict)
    )
    active_say_revision_id = (
        str(unified_branch.get("revision_id", "r0"))
        if isinstance(unified_branch, dict) else "r0"
    )
    if revisioned_live_say and isinstance(unified_branch, dict):
        unified_branch.setdefault("_streaming_revision_id", active_say_revision_id)
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
            cache_epoch = (
                response_ack_branch.get("cache_epoch_id")
                if isinstance(response_ack_branch, dict)
                and response_ack_branch.get("cache_epoch_id") is not None
                else request.get("_cache_epoch_id")
            )
            tts_consumption_callback = (
                request.get("_on_tts_text_consumed")
                if unified_native
                and stream_requested
                and callable(request.get("_on_tts_text_consumed"))
                else None
            )

            def on_live_tts_text_consumed(frontier: Any, cumulative_text: str) -> None:
                provenance_branch = (
                    unified_branch
                    if isinstance(unified_branch, dict)
                    else response_ack_branch
                    if nonunified_v1_source_stream and isinstance(response_ack_branch, dict)
                    else None
                )
                if not isinstance(provenance_branch, dict):
                    return
                ledger = provenance_branch.get("response_event_ledger")
                record_consumed = getattr(ledger, "record_tts_consumed", None)
                source_sequence = getattr(frontier, "frozen_source_sequence", None)
                if callable(record_consumed):
                    if (
                        isinstance(source_sequence, bool)
                        or not isinstance(source_sequence, int)
                        or source_sequence < 0
                    ):
                        return
                    if nonunified_v1_source_stream and provenance_branch is response_ack_branch:
                        source_rows = tuple(
                            provenance_branch.get("nonunified_v1_source_tokens", ())
                        )
                        rows_by_sequence = {
                            int(row["source_token_seq"]): row
                            for row in source_rows
                            if isinstance(row, Mapping)
                            and isinstance(row.get("source_token_seq"), int)
                            and not isinstance(row.get("source_token_seq"), bool)
                        }
                        sequences = tuple(getattr(frontier, "source_sequences", ()))
                        token_ids = tuple(getattr(frontier, "source_token_ids", ()))
                        if len(sequences) != len(token_ids):
                            raise RuntimeError("Qwen V1 source frontier has mismatched token identities")
                        for sequence, token_id in zip(sequences, token_ids, strict=True):
                            row = rows_by_sequence.get(int(sequence))
                            if row is None or int(row["token_id"]) != int(token_id):
                                raise RuntimeError("Qwen V1 source frontier changed original token identity")
                        covered_rows = tuple(
                            rows_by_sequence[index]
                            for index in sorted(rows_by_sequence)
                            if index <= int(source_sequence)
                        )
                        if tuple(
                            int(row["source_token_seq"]) for row in covered_rows
                        ) != tuple(range(int(source_sequence) + 1)):
                            return
                        expected_text = unicodedata.normalize(
                            "NFC",
                            "".join(str(row["text_piece"]) for row in covered_rows),
                        )
                        if str(cumulative_text) != expected_text:
                            return
                    record_consumed(int(source_sequence))
                if callable(tts_consumption_callback):
                    tts_consumption_callback(frontier, str(cumulative_text))

            def on_live_tts_text_reserved(consumption: Any) -> None:
                if not isinstance(unified_branch, dict):
                    raise RuntimeError("Qwen text reservation lacks its native response branch")
                ledger = unified_branch.get("response_event_ledger")
                record_reserved = getattr(ledger, "record_tts_reserved", None)
                source_sequences = tuple(getattr(consumption, "source_sequences", ()))
                if callable(record_reserved):
                    if not source_sequences:
                        raise RuntimeError("Qwen reservation has no logical source token")
                    record_reserved(max(int(sequence) for sequence in source_sequences))

            def on_live_tts_eos_consumed() -> None:
                coordinator = revision_coordinator_holder.get("value")
                note_consumed = getattr(coordinator, "note_eos_consumed", None)
                if callable(note_consumed):
                    note_consumed()
                elif nonunified_v1_source_stream and isinstance(response_ack_branch, dict):
                    ledger = response_ack_branch.get("response_event_ledger")
                    record_consumed = getattr(ledger, "record_tts_consumed", None)
                    source_rows = tuple(
                        response_ack_branch.get("nonunified_v1_source_tokens", ())
                    )
                    if not callable(record_consumed) or not source_rows:
                        return
                    frontier = live_say_stream.source_frontier_snapshot()
                    if frontier.qwen_consumed_count != len(live_say_stream.published_qwen_ids):
                        return
                    expected_text = unicodedata.normalize(
                        "NFC",
                        "".join(str(row["text_piece"]) for row in source_rows),
                    )
                    if frontier.cumulative_source_text != expected_text:
                        return
                    final_sequence = int(source_rows[-1]["source_token_seq"])
                    record_consumed(final_sequence)

            live_say_stream = LiveQwenSayStream(
                qwen_tokenizer=live_say_tokenizer,
                qwen_processor=processor,
                text_eos_token_id=eos_text_id,
                max_tokens=live_qwen_token_capacity,
                queue_capacity=16,
                max_source_chars=live_qwen_source_chars,
                max_source_pieces=live_qwen_source_pieces,
                max_retokenization_work=live_qwen_retokenization_work,
                generation_id=int(generation_id),
                cache_epoch=cache_epoch,
                on_tts_text_consumed=(
                    on_live_tts_text_consumed
                    if unified_native
                    or (
                        nonunified_v1_source_stream
                        and isinstance(response_ack_branch, dict)
                        and response_ack_branch.get("response_event_ledger") is not None
                    )
                    else None
                ),
                on_tts_text_reserved=(
                    on_live_tts_text_reserved
                    if unified_native
                    else None
                ),
                on_tts_eos_consumed=(
                    on_live_tts_eos_consumed
                    if unified_native
                    or (
                        nonunified_v1_source_stream
                        and isinstance(response_ack_branch, dict)
                        and response_ack_branch.get("response_event_ledger") is not None
                    )
                    else None
                ),
                on_packet_emitted=(
                    lambda cursor, text, **provenance: _publish_acknowledged_speech_prefix(
                        unified_branch,
                        int(generation_id),
                        int(cursor),
                        str(text),
                        source_token_seq=provenance.get("source_token_seq"),
                        revision_id=provenance.get("revision_id"),
                        cache_epoch_id=provenance.get("cache_epoch_id"),
                    )
                    if (
                        unified_native
                        and normalized_action == ControlAction.SPEAK.value
                    )
                    else None
                ),
            )
        elif live_v2_say_active:
            live_tts_unit_assembler = SpokenTextUnitAssembler(max_words=2)
            live_tts_unit_queue = queue.Queue(maxsize=2)
        generation_token = request.get("_generation_token")
        if isinstance(response_ack_branch, dict) and generation_token is not None:
            if not _register_response_speech_ack_streams(
                response_ack_branch,
                response_id=str(generation_token.response_id),
                generation_id=int(generation_token.session_generation_id),
                response_segment_id=str(generation_token.speaker_segment_id),
                live_stream=live_say_stream,
                closed_unit_streams=live_tts_streams_by_segment,
                closed_unit_events=live_tts_segment_emitted_events,
                required=bool(live_say_active or live_v2_say_active),
            ):
                raise RuntimeError(
                    "response speech ACK resources do not match the active response identity"
                )

        def live_say_talker_worker() -> None:
            _record_unified_native_profile_timestamp(
                unified_branch, "live_talker_worker_started_ns"
            )
            try:
                if live_v2_say_active:
                    if live_tts_unit_queue is None or live_say_tokenizer is None:
                        raise RuntimeError("closed-unit Qwen worker is not initialized")
                    live_say_timing["consumer_started_at_ns"] = time.perf_counter_ns()
                    previous_unit_segment_id: str | None = None
                    while not live_say_cancel.is_set() and not talker_limit_reached.is_set():
                        try:
                            unit_item = live_tts_unit_queue.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        if unit_item is live_tts_unit_sentinel:
                            break
                        if not isinstance(unit_item, SpokenTextUnit):
                            raise TypeError("closed-unit queue contains a non-SpokenTextUnit item")
                        unit = unit_item
                        used_codec_frames = sum(
                            int(item.get("generated_cb0_frames", 0))
                            for item in live_tts_unit_metrics
                            if isinstance(item.get("generated_cb0_frames", 0), int)
                            and not isinstance(item.get("generated_cb0_frames", 0), bool)
                        )
                        unit_frame_limit = (
                            ARIA_BACKCHANNEL_MAX_FRAMES
                            if normalized_action == ControlAction.BACKCHANNEL.value
                            else max(0, response_codec_frame_limit - used_codec_frames)
                        )
                        if unit_frame_limit <= 0:
                            talker_limit_reached.set()
                            break
                        unit_index = len(live_tts_unit_metrics)
                        unit_segment_id = (
                            str(request.get("_generation_token").speaker_segment_id)
                            if unit_index == 0 and request.get("_generation_token") is not None
                            else f"{request.get('_generation_token').speaker_segment_id}.tts{unit_index}"
                        )
                        if previous_unit_segment_id is not None:
                            if not _wait_for_emitted_tts_segment(
                                live_tts_segment_emitted_events,
                                previous_unit_segment_id,
                                stop_check=lambda: bool(
                                    live_say_cancel.is_set()
                                    or sess.stop_requested
                                    or (request.get("_stop_check") and request["_stop_check"]())
                                ),
                            ):
                                raise RuntimeError(
                                    "closed TTS segment PCM release cancelled before next unit"
                                )
                            live_tts_segment_emitted_events.pop(
                                previous_unit_segment_id, None
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
                            max_tokens=live_qwen_token_capacity,
                            max_source_chars=live_qwen_source_chars,
                            on_packet_consumed=lambda cursor, text: (
                                unified_branch.update({
                                    "qwen_forward_ack_exclusive_cursor": int(cursor),
                                    "qwen_forward_ack_text": str(text),
                                })
                                if isinstance(unified_branch, dict) else None
                            ),
                        )
                        live_tts_active_stream["value"] = unit_stream
                        live_tts_streams_by_segment[unit_segment_id] = unit_stream
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
                            "on_audio_chunk": audio_chunk_callback,
                            "generation_id": generation_id,
                            "talk_context": request.get("_talk_context"),
                            "metrics": unit_metrics,
                            "listener_admission_wait": listener_admission_wait,
                            "listener_compute_quantum": (
                                listener_compute_quantum
                                if callable(listener_compute_quantum)
                                else None
                            ),
                            "raw_text_token_ids": [int(initial_token.token_id)],
                            "include_text_eos": False,
                            "live_qwen_text_stream": unit_stream,
                            "max_frames": unit_frame_limit,
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
                        used_codec_frames = sum(
                            int(item.get("generated_cb0_frames", 0))
                            for item in live_tts_unit_metrics
                            if isinstance(item.get("generated_cb0_frames", 0), int)
                            and not isinstance(item.get("generated_cb0_frames", 0), bool)
                        )
                        total_codec_frames = used_codec_frames + int(
                            unit_metrics.get("generated_cb0_frames", 0)
                        )
                        unit_limit_reached = bool(
                            unit_metrics.get("terminated_by_budget") is True
                            or (
                                total_codec_frames >= response_codec_frame_limit
                                and unit_metrics.get("eos_detected") is not True
                            )
                        )
                        if unit_limit_reached:
                            talker_limit_reached.set()
                            # The closed-unit producer can be blocked on a full
                            # FIFO after the Talker reaches its acoustic cap.
                            # Canceling this unit wakes it before join and keeps
                            # the decoder stop signal independent of cleanup.
                            unit_stream.cancel()
                        feed_thread.join(timeout=5.0)
                        if feed_thread.is_alive():
                            unit_stream.cancel()
                            feed_thread.join(timeout=2.0)
                            raise RuntimeError(
                                "closed TTS unit producer remained blocked after Talker completion"
                            )
                        if unit_feed_error and not (
                            unit_limit_reached
                            and all(isinstance(error, LiveSayCancelled) for error in unit_feed_error)
                        ):
                            raise unit_feed_error[0]
                        unit_metrics["ended_at_ns"] = time.perf_counter_ns()
                        if live_say_cancel.is_set() or sess.stop_requested:
                            unit_stream.cancel()
                            break
                        if not unit_stream.codec_eos_allowed and not unit_limit_reached:
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
                        if unit_metrics.get("eos_detected") is True:
                            unit_metrics["terminal_reason"] = "eos"
                        elif unit_metrics.get("terminated_by_budget") is True:
                            # A frame cap is an orderly truncated unit. Keep the
                            # sampled-codec EOS flag false and let completion
                            # diagnostics report length instead of raising.
                            unit_metrics["terminal_reason"] = "length"
                        else:
                            _require_closed_tts_codec_eos(unit_metrics)
                        vocalized_cursor = getattr(
                            unit_stream.fifo, "last_vocalized_kv_position", None
                        )
                        source_tokens = tuple({
                            "token_id": int(source.token_id),
                            "exclusive_kv_cursor": int(source.exclusive_kv_cursor),
                            "text_piece": str(source.text_piece),
                            "token_index": int(source_index),
                            "termination_reason": str(unit_metrics["terminal_reason"]),
                        } for source_index, source in enumerate(unit.source_tokens)
                          if vocalized_cursor is not None
                          and int(source.exclusive_kv_cursor) <= int(vocalized_cursor))
                        finish_segment(unit_segment_id, source_tokens)
                        previous_unit_segment_id = unit_segment_id
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
                _record_unified_native_profile_timestamp(
                    unified_branch, "live_talker_take_initial_started_ns"
                )
                initial_token = live_say_stream.take_initial_token(
                    stop_check=lambda: bool(
                        live_say_cancel.is_set()
                        or sess.stop_requested
                        or (request.get("_stop_check") and request["_stop_check"]())
                    ),
                )
                _record_unified_native_profile_timestamp(
                    unified_branch, "live_talker_initial_token_ready_ns"
                )
                talker_metrics: dict[str, Any] = {}
                first_qwen_id_ns = live_say_result.get("first_qwen_id_published_at_ns")
                if isinstance(first_qwen_id_ns, int):
                    talker_metrics["first_qwen_id_published_at_ns"] = first_qwen_id_ns
                talker_metrics["qwen_prefill_clock"] = "server monotonic_ns"
                h_pfx = torch.empty((1, 0, 2048), device=engine.device, dtype=engine.dtype)
                prefill_text = live_say_stream.initial_prefill_text
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
                    "on_audio_chunk": audio_chunk_callback,
                    "generation_id": generation_id,
                    "talk_context": request.get("_talk_context"),
                    "metrics": talker_metrics,
                    "listener_admission_wait": listener_admission_wait,
                    "listener_compute_quantum": (
                        listener_compute_quantum
                        if callable(listener_compute_quantum)
                        else None
                    ),
                    "raw_text_token_ids": [int(initial_token.token_id)],
                    "include_text_eos": False,
                    "live_qwen_text_stream": live_say_stream,
                    "max_frames": (
                        ARIA_BACKCHANNEL_MAX_FRAMES
                        if normalized_action == ControlAction.BACKCHANNEL.value
                        else response_codec_frame_limit
                    ),
                }
                def on_dynamic_prefill_model_entry() -> None:
                    entry_ns = time.monotonic_ns()
                    live_say_result["qwen_dynamic_prefill_model_entry_at_ns"] = entry_ns
                    talker_metrics["qwen_dynamic_prefill_model_entry_at_ns"] = entry_ns
                    talker_metrics["qwen_prefill_clock"] = "server monotonic_ns"
                    first_id_ns = live_say_result.get("first_qwen_id_published_at_ns")
                    prefill_metrics: dict[str, Any] = {
                        "qwen_dynamic_prefill_model_entry_at_ns": entry_ns,
                        "qwen_prefill_clock": "server monotonic_ns",
                    }
                    if isinstance(first_id_ns, int):
                        elapsed_ms = (entry_ns - first_id_ns) / 1_000_000.0
                        talker_metrics["qwen_first_id_to_prefill_entry_ms"] = elapsed_ms
                        prefill_metrics.update({
                            "first_qwen_id_published_at_ns": int(first_id_ns),
                            "qwen_first_id_to_prefill_entry_ms": elapsed_ms,
                        })
                    update_response_runtime_metrics(
                        response_id, "qwen_prefill", prefill_metrics
                    )
                    _record_unified_native_profile_timestamp(
                        unified_branch,
                        "qwen_dynamic_prefill_model_entry_at_ns",
                        entry_ns,
                    )
                    if isinstance(unified_branch, dict):
                        profile = unified_branch.get("unified_native_profile")
                        profile_lock = unified_branch.get("unified_native_profile_lock")
                        if isinstance(profile, dict):
                            if profile_lock is None:
                                first_id_ns = profile.get("first_qwen_id_published_at_ns")
                                if isinstance(first_id_ns, int):
                                    profile["qwen_first_id_to_prefill_entry_ms"] = max(
                                        0.0,
                                        (entry_ns - first_id_ns) / 1_000_000.0,
                                    )
                            else:
                                with profile_lock:
                                    first_id_ns = profile.get("first_qwen_id_published_at_ns")
                                    if isinstance(first_id_ns, int):
                                        profile["qwen_first_id_to_prefill_entry_ms"] = (
                                            entry_ns - first_id_ns
                                        ) / 1_000_000.0
                    live_say_prefill_entry.set()

                talker_kwargs["on_dynamic_prefill_model_entry"] = (
                    on_dynamic_prefill_model_entry
                )
                if frame_residual_candidate:
                    talker_kwargs["live_frame_residual_hidden_stream"] = live_say_stream
                    talker_kwargs["frame_residual_response_hidden"] = None
                if normalized_action == ControlAction.BACKCHANNEL.value:
                    talker_kwargs["min_frames"] = 2
                with _diagnostic_talker_rng_scope(engine):
                    wav = engine.talk(prefill_text, h_pfx, **talker_kwargs)
                codec_frames = talker_metrics.get("generated_cb0_frames")
                if (
                    talker_metrics.get("terminated_by_budget") is True
                    or (
                        isinstance(codec_frames, int)
                        and not isinstance(codec_frames, bool)
                        and codec_frames >= response_codec_frame_limit
                        and talker_metrics.get("eos_detected") is not True
                    )
                ):
                    talker_limit_reached.set()
                if (
                    not live_say_stream.codec_eos_allowed
                    and not live_say_cancel.is_set()
                    and not talker_limit_reached.is_set()
                ):
                    raise RuntimeError("live Qwen Talker ended before native text EOS and tail drain")
                live_say_result.update({"wav": wav, "metrics": talker_metrics})
                if live_say_cancel.is_set():
                    live_say_stream.cancel()
            except BaseException as exc:
                active_stream = live_tts_active_stream.get("value")
                if live_say_cancel.is_set() or sess.stop_requested:
                    if active_stream is not None:
                        active_stream.cancel()
                    elif live_say_stream is not None:
                        live_say_stream.cancel()
                    return
                live_say_result["error"] = exc
                live_say_cancel.set()
                if active_stream is not None:
                    active_stream.fail(exc)
                elif live_say_stream is not None:
                    live_say_stream.fail(exc)
                elif live_v2_say_active:
                    sess.speech_fifo.clear()
            finally:
                # Wake token-0 publication when talk() fails before reaching
                # dynamic t.model; the waiter then propagates the recorded error.
                live_say_prefill_entry.set()

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
            _cancel_tts_segment_emission_waiters(live_tts_segment_emitted_events)
            live_tts_streams_by_segment.clear()
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

        def publish_revisioned_live_say_pending(
            *,
            token_index: int,
            source_token_seq: int,
            revision_id: str,
            packet: Any,
            record: Mapping[str, Any],
        ) -> None:
            if not revisioned_live_say or live_say_stream is None or not isinstance(unified_branch, dict):
                raise RuntimeError("deferred Qwen publication requires the revisioned V1 branch")
            generation_lock = getattr(sess, "_generation_lock", None)
            branch_lock = unified_branch.get("spoken_prefix_lock")
            stream_condition = getattr(live_say_stream, "_condition", None)
            ledger = unified_branch.get("response_event_ledger")
            ledger_lock = getattr(ledger, "_lock", None)
            if any(lock is None for lock in (generation_lock, branch_lock, stream_condition, ledger_lock)):
                raise RuntimeError("deferred Qwen publication lacks its transaction locks")
            response_cancel_event = request.get("_response_cancel_event")

            def stopped() -> bool:
                return bool(
                    live_say_cancel.is_set()
                    or sess.stop_requested
                    or (
                        callable(getattr(response_cancel_event, "is_set", None))
                        and response_cancel_event.is_set()
                    )
                    or int(getattr(sess, "session_generation_id", -1)) != int(generation_id)
                    or str(unified_branch.get("_streaming_revision_id", revision_id))
                    != revision_id
                )

            def commit_if_current(publish):
                with generation_lock:
                    if (
                        generation_id is None
                        or not sess.is_generation_current(generation_id=generation_id)
                        or str(getattr(sess, "response_id", "")) != str(response_id)
                        or not bool(getattr(sess, "active_response", False))
                        or live_say_cancel.is_set()
                        or bool(getattr(sess, "stop_requested", False))
                        or (
                            callable(getattr(response_cancel_event, "is_set", None))
                            and response_cancel_event.is_set()
                        )
                    ):
                        return False
                    with branch_lock:
                        with stream_condition:
                            with ledger_lock:
                                packets = unified_branch.get("say_token_packets")
                                records = unified_branch.get("segment_token_packets")
                                if (
                                    str(unified_branch.get("response_id", ""))
                                    != str(response_id)
                                    or int(unified_branch.get("session_generation_id", -1))
                                    != int(generation_id)
                                    or str(unified_branch.get("revision_id", "r0"))
                                    != revision_id
                                    or str(unified_branch.get("_streaming_revision_id", ""))
                                    != revision_id
                                    or not isinstance(packets, list)
                                    or not isinstance(records, list)
                                    or token_index < 0
                                    or token_index >= len(packets)
                                    or token_index >= len(records)
                                    or packets[token_index] is not packet
                                    or records[token_index] is not record
                                    or source_token_seq not in ledger._source_tokens
                                    or source_token_seq in ledger._retired_source_sequences
                                ):
                                    return False
                                # This callback publishes only if a slot is
                                # already free; it never waits under these locks.
                                return publish()

            live_say_stream.publish_deferred(
                stop_check=stopped,
                publication_guard=commit_if_current,
            )

        def live_say_generated_token(
            _token_index: int,
            _token_id: int,
            token_text: str,
            token_hidden: Any,
            *,
            exclusive_kv_cursor: int | None = None,
            source_token_seq: int | None = None,
            hidden_provenance: str = "token",
            kv_committed: bool = True,
            source_group: tuple[tuple[int, int], ...] | None = None,
        ) -> None:
            nonlocal thinker_first_token_id, streamed_live_chars
            if thinker_first_token_id is None:
                thinker_first_token_id = int(_token_id)
            assert live_say_stream is not None
            stream_hidden = token_hidden
            producer_event = None
            nonunified_v1_source_sequence: int | None = None
            if nonunified_v1_source_stream and isinstance(response_ack_branch, dict):
                ledger = response_ack_branch.get("response_event_ledger")
                record_generated = getattr(ledger, "record_generated_tokens", None)
                snapshot_source = getattr(ledger, "source_frontier_snapshot", None)
                if callable(record_generated) and callable(snapshot_source):
                    next_sequence = int(snapshot_source()["next_source_sequence"])
                    source_row = {
                        "source_token_seq": next_sequence,
                        "token_id": int(_token_id),
                        "text_piece": str(token_text),
                    }
                    recorded_sequences = record_generated([source_row])
                    if recorded_sequences != (next_sequence,):
                        raise RuntimeError("non-unified V1 source token sequence changed during publication")
                    nonunified_v1_source_sequence = next_sequence
                    with response_ack_branch.get("spoken_prefix_lock", nullcontext()):
                        source_rows = response_ack_branch.setdefault(
                            "nonunified_v1_source_tokens", []
                        )
                        if not isinstance(source_rows, list):
                            raise RuntimeError("non-unified V1 source token rows are not mutable")
                        source_rows.append(dict(source_row))
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
                if revisioned_live_say and (
                    isinstance(source_token_seq, bool)
                    or not isinstance(source_token_seq, int)
                    or source_token_seq < 0
                ):
                    raise RuntimeError("revisioned V1 SAY token lacks its logical source sequence")
                token_revision_id = str(unified_branch.get("revision_id", "r0"))
                token_cache_epoch_id = (
                    unified_branch.get("cache_epoch_id")
                    if "cache_epoch_id" in unified_branch else cache_epoch
                )
                if revisioned_live_say and token_revision_id != str(
                    unified_branch.get("_streaming_revision_id", active_say_revision_id)
                ):
                    raise LiveSayCancelled("active Thinker revision was superseded before token publication")
                base_say_cursor = int(
                    unified_branch.get("spoken_prefix_base_cursor")
                    or unified_branch.get("cache_position")
                    or 0
                )
                packet = SpokenTokenPacket(
                    token_id=int(_token_id),
                    text_piece=str(token_text),
                    gemma_hidden_1536=packet_hidden,
                    exclusive_kv_cursor=int(
                        base_say_cursor + int(_token_index) + 1
                        if exclusive_kv_cursor is None
                        else exclusive_kv_cursor
                    ),
                    hidden_provenance=str(hidden_provenance),
                    kv_committed=bool(kv_committed),
                )
                producer_event = packet.producer_event
                record = {
                        "token_id": packet.token_id,
                        "exclusive_kv_cursor": packet.exclusive_kv_cursor,
                        "text_piece": packet.text_piece,
                        "token_index": int(_token_index),
                        "source_token_seq": (
                            int(source_token_seq)
                            if source_token_seq is not None else None
                        ),
                        "revision_id": token_revision_id,
                        "cache_epoch_id": token_cache_epoch_id,
                        "physical_kv_provenance": [{
                            "revision_id": token_revision_id,
                            "cache_epoch_id": token_cache_epoch_id,
                            "physical_kv_cursor": int(packet.exclusive_kv_cursor),
                        }],
                        "hidden_provenance": packet.hidden_provenance,
                        "kv_committed": packet.kv_committed,
                    }
                stream_hidden = packet.gemma_hidden_1536.unsqueeze(1)
            if token_text and stream_requested and not (
                unified_native
                and callable(request.get("_on_tts_text_consumed"))
            ):
                render_visible_text_chunk(token_text, stream_hidden, 0)
                streamed_live_chars += len(token_text)
            if unified_native:
                resolved_cursor = (
                    packet.exclusive_kv_cursor
                    if packet is not None
                    else exclusive_kv_cursor
                )
                if resolved_cursor is None:
                    raise RuntimeError("unified V1 SAY packet lacks its exclusive KV cursor")
                if revisioned_live_say:
                    generation_lock = getattr(sess, "_generation_lock", None)
                    stream_condition = getattr(live_say_stream, "_condition", None)
                    ledger = unified_branch.get("response_event_ledger")
                    ledger_lock = getattr(ledger, "_lock", None)
                    branch_lock = unified_branch.get("spoken_prefix_lock")
                    if any(lock is None for lock in (generation_lock, stream_condition, ledger_lock, branch_lock)):
                        raise RuntimeError("revisioned V1 source commit lacks its transaction locks")
                    source_row = {
                        "source_token_seq": int(source_token_seq),
                        "token_id": int(_token_id),
                        "text_piece": str(token_text),
                        "exclusive_kv_cursor": int(resolved_cursor),
                        "revision_id": token_revision_id,
                        "cache_epoch_id": token_cache_epoch_id,
                        "physical_kv_provenance": list(record["physical_kv_provenance"]),
                    }
                    with generation_lock:
                        if (
                            generation_id is None
                            or not sess.is_generation_current(generation_id=generation_id)
                            or str(unified_branch.get("revision_id", "r0")) != token_revision_id
                            or str(unified_branch.get("_streaming_revision_id", token_revision_id))
                            != token_revision_id
                        ):
                            raise LiveSayCancelled("active Thinker revision is no longer current")
                        with branch_lock:
                            with stream_condition:
                                with ledger_lock:
                                    packet_records = unified_branch.get("say_token_packets")
                                    segment_records = unified_branch.get("segment_token_packets")
                                    if not isinstance(packet_records, list) or not isinstance(segment_records, list):
                                        raise LiveSayCancelled("active V1 source branch was adopted by a candidate")
                                    if len(packet_records) != len(segment_records):
                                        raise RuntimeError("unified SAY packet and segment record counts differ")
                                    if int(_token_index) != len(packet_records):
                                        raise RuntimeError(
                                            "V1 Gemma SAY token indices are not contiguous: "
                                            f"token_index={int(_token_index)}, packet_count={len(packet_records)}"
                                        )
                                    snapshot = ledger.source_frontier_snapshot()
                                    if (
                                        int(source_token_seq) <= int(snapshot["frozen_source_sequence"])
                                        or int(source_token_seq) in ledger._source_tokens
                                        or int(source_token_seq) in ledger._retired_source_sequences
                                    ):
                                        raise RuntimeError("active V1 token logical source sequence is not appendable")
                                    if producer_event is not None:
                                        device = getattr(stream_hidden, "device", None)
                                        if (
                                            device is not None
                                            and getattr(device, "type", None) == "cuda"
                                            and torch.cuda.is_available()
                                        ):
                                            current_stream = torch.cuda.current_stream(device=device)
                                            current_stream.wait_event(producer_event)
                                            packet_hidden.record_stream(current_stream)
                                    sealed_initial = live_say_stream.feed_piece(
                                        token_text,
                                        stream_hidden,
                                        token_index=int(_token_index),
                                        source_token_id=int(_token_id),
                                        source_token_seq=int(source_token_seq),
                                        source_group=source_group,
                                        exclusive_kv_cursor=int(resolved_cursor),
                                        revision_id=token_revision_id,
                                        producer_event=None,
                                        kv_committed=bool(kv_committed),
                                        seal_initial_segment=normalized_action == ControlAction.SPEAK.value,
                                        defer_publication=True,
                                    )
                                    ledger.record_generated_tokens(
                                        [source_row],
                                        revision_id=token_revision_id,
                                        cache_epoch_id=(
                                            None if token_cache_epoch_id is None
                                            else str(token_cache_epoch_id)
                                        ),
                                    )
                                    packet_records.append(packet)
                                    segment_records.append(record)
                else:
                    with unified_branch["spoken_prefix_lock"]:
                        packet_records = unified_branch.get("say_token_packets")
                        segment_records = unified_branch.get("segment_token_packets")
                        if not isinstance(packet_records, list) or not isinstance(segment_records, list):
                            raise RuntimeError(
                                "unified SAY branch lacks initialized token packet records"
                            )
                        if len(packet_records) != len(segment_records):
                            raise RuntimeError(
                                "unified SAY packet and segment record counts differ"
                            )
                        if exclusive_kv_cursor is not None and int(_token_index) != len(packet_records):
                            raise RuntimeError(
                                "V1 Gemma SAY token indices are not contiguous: "
                                f"token_index={int(_token_index)}, packet_count={len(packet_records)}"
                            )
                        packet_records.append(packet)
                        segment_records.append(record)
                    sealed_initial = live_say_stream.feed_piece(
                        token_text,
                        stream_hidden,
                        token_index=int(_token_index),
                        source_token_id=int(_token_id),
                        source_token_seq=source_token_seq,
                        source_group=source_group,
                        exclusive_kv_cursor=int(resolved_cursor),
                        revision_id=token_revision_id,
                        producer_event=producer_event,
                        kv_committed=bool(kv_committed),
                        seal_initial_segment=normalized_action == ControlAction.SPEAK.value,
                    )
            else:
                if nonunified_v1_source_sequence is not None:
                    sealed_initial = live_say_stream.feed_piece(
                        token_text,
                        stream_hidden,
                        producer_event=producer_event,
                        token_index=int(_token_index),
                        source_token_id=int(_token_id),
                        source_token_seq=int(nonunified_v1_source_sequence),
                        kv_committed=bool(kv_committed),
                    )
                else:
                    sealed_initial = live_say_stream.feed_piece(
                        token_text,
                        stream_hidden,
                        producer_event=producer_event,
                        token_index=(int(_token_index) if exclusive_kv_cursor is not None else None),
                        exclusive_kv_cursor=exclusive_kv_cursor,
                        kv_committed=bool(kv_committed),
                    )
            if revisioned_live_say:
                publish_revisioned_live_say_pending(
                    token_index=int(_token_index),
                    source_token_seq=int(source_token_seq),
                    revision_id=token_revision_id,
                    packet=packet,
                    record=record,
                )

            first_qwen_id_ns = live_say_stream.first_qwen_id_published_at_ns
            if isinstance(first_qwen_id_ns, int):
                live_say_result.setdefault(
                    "first_qwen_id_published_at_ns", int(first_qwen_id_ns)
                )
                update_response_runtime_metrics(response_id, "qwen_prefill", {
                    "first_qwen_id_published_at_ns": int(first_qwen_id_ns),
                    "qwen_prefill_clock": "server monotonic_ns",
                })
                _record_unified_native_profile_timestamp(
                    unified_branch,
                    "first_qwen_id_published_at_ns",
                    int(first_qwen_id_ns),
                )
            if live_say_stream.published_qwen_ids:
                start_live_say_talker()
            if sealed_initial:
                _wait_for_qwen_dynamic_prefill_model_entry(
                    live_say_prefill_entry,
                    live_say_result,
                    cancelled=lambda: bool(live_say_cancel.is_set() or sess.stop_requested),
                    stopped=lambda: bool(
                        request.get("_stop_check") and request["_stop_check"]()
                    ),
                )

        def live_say_native_token_packet(generated: Any) -> None:
            if not unified_native or normalized_action != ControlAction.SPEAK.value:
                raise RuntimeError("anticipated Gemma token packet is only valid for unified SPEAK")
            if not isinstance(unified_branch, dict) or live_say_stream is None:
                raise RuntimeError("anticipated V1 SPEAK packet lacks its live native branch")
            token_index = int(generated.token_index)
            expected_cursor = _validate_unified_say_packet_frontier(
                unified_branch,
                token_index,
                int(unified_branch["spoken_prefix_base_cursor"]) + token_index + 1,
                kv_committed=bool(generated.kv_committed),
            )
            live_say_generated_token(
                token_index,
                int(generated.token_id),
                str(generated.token_text),
                generated.hidden,
                exclusive_kv_cursor=expected_cursor,
                source_token_seq=int(generated.logical_source_sequence),
                hidden_provenance=str(generated.hidden_provenance),
                kv_committed=bool(generated.kv_committed),
                source_group=tuple(getattr(generated, "text_source_sequences", ())) or None,
            )
            release_pause = unified_branch.get("release_listener_burst_pause")
            if token_index >= 1 and bool(generated.kv_committed) and callable(release_pause):
                release_pause("second_token_commit")

        def live_say_native_token_committed(generated: Any) -> None:
            if not isinstance(unified_branch, dict) or live_say_stream is None:
                raise RuntimeError("committed V1 SPEAK packet lacks its live native branch")
            token_index = int(generated.token_index)
            expected_cursor = _validate_unified_say_packet_frontier(
                unified_branch,
                token_index,
                int(unified_branch["spoken_prefix_base_cursor"]) + token_index + 1,
                kv_committed=True,
            )
            _mark_unified_say_packet_committed(
                unified_branch,
                live_say_stream,
                token_index,
                expected_cursor,
                generation_id=int(live_say_stream.generation_id),
                segment_id=int(live_say_stream.segment_id),
            )

        def live_say_decoded_text_flush(text_suffix: str, *details: Any) -> None:
            nonlocal streamed_live_chars
            if not text_suffix:
                return
            source_group = None
            if len(details) == 1:
                generated = details[0]
                token_index = int(getattr(generated, "token_index", -1))
                token_id = int(getattr(generated, "token_id", -1))
                token_hidden = getattr(generated, "hidden", None)
                source_group = tuple(
                    getattr(generated, "text_source_sequences", ())
                ) or None
            elif len(details) == 3:
                token_index, token_id, token_hidden = details
                token_index = int(token_index)
                token_id = int(token_id)
            else:
                raise TypeError("decoded text flush callback has an unsupported signature")
            if not isinstance(token_hidden, torch.Tensor):
                raise TypeError("decoded text flush requires the post-token Gemma hidden")
            if live_say_active and live_say_stream is not None:
                if unified_native:
                    if not isinstance(unified_branch, dict) or token_index < 0:
                        raise RuntimeError("final V1 text flush lacks its native source token")
                    cursor = int(unified_branch["spoken_prefix_base_cursor"]) + token_index + 1
                    if revisioned_live_say:
                        source_token_seq = (
                            int(getattr(details[0], "logical_source_sequence"))
                            if len(details) == 1 else None
                        )
                        if source_token_seq is None:
                            raise RuntimeError("final V1 text flush lacks its logical source sequence")
                        if str(unified_branch.get("revision_id", "r0")) != str(
                            unified_branch.get("_streaming_revision_id", active_say_revision_id)
                        ):
                            return
                        source_text = str(getattr(details[0], "token_text", ""))
                        token_revision_id = str(unified_branch.get("revision_id", "r0"))
                        token_cache_epoch_id = (
                            unified_branch.get("cache_epoch_id")
                            if "cache_epoch_id" in unified_branch else cache_epoch
                        )
                        ledger = unified_branch.get("response_event_ledger")
                        ledger_lock = getattr(ledger, "_lock", None)
                        branch_lock = unified_branch.get("spoken_prefix_lock")
                        if ledger_lock is None or branch_lock is None:
                            raise RuntimeError("final V1 text flush lacks source transaction locks")
                        if (
                            int(getattr(live_say_stream, "generation_id", -1))
                            != int(generation_id if generation_id is not None else -2)
                            or int(getattr(ledger, "generation_id", -1))
                            != int(generation_id if generation_id is not None else -2)
                        ):
                            raise RuntimeError("final V1 text flush crossed its response generation")
                        stream_plan = None
                        with sess._generation_lock:
                            if (
                                generation_id is None
                                or not sess.is_generation_current(generation_id=generation_id)
                                or str(unified_branch.get("revision_id", "r0"))
                                != token_revision_id
                                or str(unified_branch.get("_streaming_revision_id", token_revision_id))
                                != token_revision_id
                            ):
                                return
                            with branch_lock:
                                with live_say_stream._condition:
                                    packet_records = unified_branch.get("say_token_packets")
                                    segment_records = unified_branch.get("segment_token_packets")
                                    if not isinstance(packet_records, list) or not isinstance(segment_records, list):
                                        return
                                    row_index = next((
                                        index
                                        for index, record in enumerate(segment_records)
                                        if isinstance(record, Mapping)
                                        and record.get("source_token_seq") == source_token_seq
                                    ), None)
                                    if row_index is None or row_index >= len(packet_records):
                                        return
                                    updated_packet = replace(
                                        packet_records[row_index], text_piece=source_text
                                    )
                                    updated_record = dict(segment_records[row_index])
                                    updated_record["text_piece"] = source_text
                                    next_packets = list(packet_records)
                                    next_records = list(segment_records)
                                    next_packets[row_index] = updated_packet
                                    next_records[row_index] = updated_record
                                    stream_plan = live_say_stream.prepare_final_text_flush_locked(
                                        str(text_suffix),
                                        token_hidden,
                                        token_index=token_index,
                                        source_token_id=token_id,
                                        source_token_seq=source_token_seq,
                                        source_group=source_group,
                                        exclusive_kv_cursor=cursor,
                                        revision_id=token_revision_id,
                                        seal_initial_segment=normalized_action == ControlAction.SPEAK.value,
                                    )
                                    with ledger_lock:
                                        ledger_plan = ledger.prepare_unconsumed_source_text_locked(
                                            source_token_seq,
                                            source_text,
                                            expected_revision_id=token_revision_id,
                                            expected_cache_epoch_id=(
                                                None if token_cache_epoch_id is None
                                                else str(token_cache_epoch_id)
                                            ),
                                        )
                                        ledger.install_unconsumed_source_text_locked(ledger_plan)
                                        live_say_stream.install_final_text_flush_locked(stream_plan)
                                        unified_branch["say_token_packets"] = next_packets
                                        unified_branch["segment_token_packets"] = next_records
                                        live_say_stream._condition.notify_all()
                        if stream_plan is not None:
                            live_say_stream.dispatch_final_text_flush_emissions(stream_plan)
                            publish_revisioned_live_say_pending(
                                token_index=int(row_index),
                                source_token_seq=int(source_token_seq),
                                revision_id=token_revision_id,
                                packet=updated_packet,
                                record=updated_record,
                            )
                        if stream_requested and not callable(request.get("_on_tts_text_consumed")):
                            render_visible_text_chunk(str(text_suffix), token_hidden, 0)
                            streamed_live_chars += len(text_suffix)
                        return
                    _validate_unified_say_packet_frontier(
                        unified_branch,
                        token_index,
                        cursor,
                        kv_committed=True,
                    )
                    live_say_stream.feed_final_text_piece(
                        str(text_suffix),
                        token_hidden,
                        token_index=token_index,
                        source_token_id=token_id,
                        source_token_seq=(
                            int(getattr(details[0], "logical_source_sequence"))
                            if len(details) == 1 else None
                        ),
                        source_group=source_group,
                        exclusive_kv_cursor=cursor,
                        revision_id=str(unified_branch.get("revision_id", "r0")),
                        seal_initial_segment=normalized_action == ControlAction.SPEAK.value,
                    )
                else:
                    live_say_stream.feed_piece(str(text_suffix), token_hidden)
            if stream_requested and not (
                unified_native
                and callable(request.get("_on_tts_text_consumed"))
            ):
                render_visible_text_chunk(str(text_suffix), token_hidden, 0)
                streamed_live_chars += len(str(text_suffix))

    diagnostic_talker_seed = getattr(engine, "diagnostic_talker_seed", None)
    streamed_audio: list[np.ndarray] = []
    duplex_realtime = bool(request.get("_duplex_realtime"))
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

    allow_thinker_text_callback, allow_thinker_token_id_callback = (
        _thinker_callback_admission(
            action=normalized_action,
            stream_requested=stream_requested,
            response_v2_active=response_v2_collector is not None,
            controller_guided_say=controller_guided_say,
        )
    )
    think_kwargs: dict[str, Any] = {
        "max_new_tokens": _response_generation_max_new_tokens(
            response_v2_enabled=response_v2_collector is not None,
            private_backchannel_active=private_backchannel_active,
            backchannel_action=normalized_action == ControlAction.BACKCHANNEL.value,
            requested_max_tokens=profile_token_budget,
        ),
        "min_new_tokens": int(request.get("_min_new_tokens", 0)),
        "temperature": float(request.get("temperature", 0.7)),
        "top_p": float(request.get("top_p", 0.9)),
        "stop_check": request.get("_stop_check"),
        "state": sess,
        "generation_id": generation_id,
        "on_text_chunk": (
            render_text_chunk
            if allow_thinker_text_callback
            else None
        ),
        "on_token_id": (
            request.get("_on_thinker_token_id")
            if allow_thinker_token_id_callback
            else None
        ),
        "action": None if direct_audio else normalized_action,
        "response_protocol": response_protocol if direct_audio else None,
        "input_audio": processor_audio_inputs,
        "graceful_sentence_completion": bool(
            (
                controller_guided_say
                and normalized_action == ControlAction.SPEAK.value
            )
            or (
                unified_native
                and response_profile == LONG_RESPONSE_PROFILE
                and normalized_action != ControlAction.BACKCHANNEL.value
            )
        ),
    }
    if live_say_active:
        think_kwargs["on_decoded_text_flush"] = live_say_decoded_text_flush
    route_token_budget = int(think_kwargs["max_new_tokens"])
    think_kwargs["_token_limit_effective"] = route_token_budget
    if route_token_budget < profile_token_budget:
        think_kwargs["_token_limit_cap_reason"] = (
            "response_protocol_limit"
            if response_v2_collector is not None
            else "backchannel_limit"
            if normalized_action == ControlAction.BACKCHANNEL.value
            else "generation_route_limit"
        )
    elif profile_token_budget < requested_token_limit:
        think_kwargs["_token_limit_cap_reason"] = "profile_max"
    if unified_native and think_kwargs["graceful_sentence_completion"]:
        think_kwargs["sentence_completion_max_tokens"] = 32
    if revisioned_live_say:
        think_kwargs["max_new_tokens"] = min(
            768, int(think_kwargs["max_new_tokens"])
        )
        think_kwargs["sentence_completion_max_tokens"] = min(
            int(think_kwargs.get("sentence_completion_max_tokens", 0)),
            max(0, 768 - int(think_kwargs["max_new_tokens"])),
        )
    if unified_native and (live_say_active or live_v2_say_active):
        think_kwargs["producer_limit_check"] = talker_limit_reached.is_set
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

        def controller_backchannel_generated_token(
            token_index: int,
            token_id: int,
            token_text: str,
            token_hidden: Any,
        ) -> tuple[int, str] | None:
            controller_backchannel_text.append(str(token_text))
            if live_say_active:
                controller_backchannel_packets.append((
                    int(token_index),
                    int(token_id),
                    str(token_text),
                    token_hidden,
                ))
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
        if unified_native and normalized_action == ControlAction.SPEAK.value:
            _configure_native_live_say_callbacks(
                think_kwargs,
                frame_residual_active=bool(frame_residual_candidate),
                packet_callback=live_say_native_token_packet,
                committed_callback=live_say_native_token_committed,
            )
        else:
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
            if (
                response_v2_collector.parser.action == ControlAction.BACKCHANNEL.value
                and response_v2_budget is not None
            ):
                backchannel_say_cap = min(
                    int(response_v2_budget["say_token_cap"]),
                    BACKCHANNEL_MAX_TEXT_TOKENS,
                )
                if backchannel_say_cap != int(response_v2_budget["say_token_cap"]):
                    response_v2_budget["say_token_cap"] = backchannel_say_cap
                    response_v2_collector.set_token_caps(
                        heard=int(response_v2_budget["heard_token_cap"]),
                        action=int(response_v2_budget["action_token_cap"]),
                        say=backchannel_say_cap,
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
        def fallback_v2_candidate(rejected):
            assert response_v2_budget is not None
            action_ids = response_v2_action_constraint.action_value_candidate_token_ids(
                response_v2_collector
            )
            if action_ids:
                return CandidateFallbackPool(action_ids, metric="action_pool")
            closure_id = response_v2_action_constraint.bounded_field_closure_token_id(
                response_v2_collector,
                rejected,
                budget=response_v2_budget,
                proposal=True,
            )
            if closure_id is not None:
                return closure_id
            closure_id = response_v2_action_constraint.bounded_say_closure_token_id(
                response_v2_collector,
                rejected,
                max_say_words=4,
                proposal=True,
            )
            if closure_id is not None:
                return closure_id
            if response_v2_collector.backchannel_say_lexicon_active:
                return response_v2_action_constraint.backchannel_say_candidate_token_ids(
                    response_v2_collector
                )
            return None

        think_kwargs["on_candidate_fallback"] = fallback_v2_candidate

    def remove_revision_coordinator(*, cancel: bool, reason: str = "response_complete") -> None:
        coordinator = revision_coordinator_holder.get("value")
        if coordinator is None:
            return
        if cancel:
            coordinator.cancel_all(reason)
        if not coordinator.join(timeout_s=2.0):
            raise RuntimeError("response revision producer did not stop within the bounded join")
        if isinstance(unified_branch, dict):
            branch_lock = unified_branch.get("spoken_prefix_lock")
            if branch_lock is not None:
                with branch_lock:
                    if unified_branch.get("revision_coordinator") is coordinator:
                        unified_branch.pop("revision_coordinator", None)
        if revision_coordinator_holder.get("value") is coordinator:
            revision_coordinator_holder["value"] = None

    def stop_live_say_worker(error: BaseException | None = None) -> None:
        with live_say_publication_lock:
            live_say_cancel.set()
        remove_revision_coordinator(
            cancel=True,
            reason="live_say_stopped" if error is None else "live_say_failed",
        )
        if live_v2_say_active:
            cancel_live_v2_units(error)
            return
        if live_say_stream is None:
            return
        if error is None:
            live_say_stream.cancel()
        else:
            live_say_stream.fail(error)
        if live_say_thread is not None and live_say_thread.is_alive():
            live_say_thread.join(timeout=2.0)
            if live_say_thread.is_alive():
                raise RuntimeError("live SAY worker did not stop within the bounded join")

    def backchannel_cancelled_now() -> bool:
        return bool(
            live_say_cancel.is_set()
            or getattr(sess, "stop_requested", False)
            or (generation_id is not None and not sess.is_generation_current(generation_id=generation_id))
            or (request.get("_stop_check") and request["_stop_check"]())
        )

    if revisioned_live_say:
        if generation_id is None or not isinstance(unified_branch, dict):
            raise RuntimeError("revisioned V1 SPEAK lacks its response generation")
        coordinator = create_live_say_revision_coordinator(
            engine=engine,
            session=sess,
            request=request,
            worker=unified_branch["worker"],
            branch=unified_branch,
            stream=live_say_stream,
            ledger=unified_branch.get("response_event_ledger"),
            transition_token_ids=tuple(int(value) for value in ids),
            think_kwargs=think_kwargs,
            response_protocol=str(response_protocol),
            generation_id=int(generation_id),
            response_id=str(response_id),
            update_metrics=lambda phase, values: update_response_runtime_metrics(
                response_id, phase, values
            ),
        )
        revision_coordinator_holder["value"] = coordinator

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
        if isinstance(exc, LiveSayCancelled):
            stop_live_say_worker()
            raise
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
    if revisioned_live_say:
        coordinator = revision_coordinator_holder.get("value")
        if coordinator is not None:
            if bool(think.get("cancelled")):
                coordinator.cancel_all("response_cancelled")
            else:
                coordinator.request_eos()
                while True:
                    if live_say_stream.eos_consumed:
                        coordinator.note_eos_consumed()
                    if not coordinator.wait_until_idle(timeout_s=0.05):
                        if sess.stop_requested or (
                            request.get("_stop_check") and request["_stop_check"]()
                        ):
                            coordinator.cancel_all("response_cancelled_during_revision_drain")
                            break
                        continue
                    if coordinator.claim_finalize_if_idle():
                        break
                    if coordinator.closed or sess.stop_requested:
                        coordinator.cancel_all("response_cancelled_during_revision_drain")
                        break
                adopted_controller = unified_branch.get(
                    "_adopted_native_session_controller"
                )
                if (
                    isinstance(adopted_controller, dict)
                    and str(unified_branch.get("revision_id", "r0")) != active_say_revision_id
                ):
                    resumed = _resume_adopted_controller(
                        adopted_controller,
                        unified_branch,
                    )
                    if resumed is not None:
                        think = resumed
                adopted_think = unified_branch.get("_last_adopted_think")
                if (
                    isinstance(adopted_think, Mapping)
                    and str(unified_branch.get("revision_id", "r0")) != active_say_revision_id
                    and not isinstance(adopted_controller, dict)
                ):
                    think = dict(adopted_think)
                    think["revision_handoff"] = True
        remove_revision_coordinator(cancel=False)
    # Keep the original Thinker count before controller-guided paths replace
    # gen_ids with an empty list after consuming their hidden state.
    thinker_generated_token_count = len(think.get("gen_ids", ())) if isinstance(
        think.get("gen_ids"), (list, tuple)
    ) else 0
    if revisioned_live_say and bool(think.get("revision_handoff")):
        thinker_generated_token_count = int(
            unified_branch.get("_generated_budget_used", thinker_generated_token_count)
        )
    thinker_runtime_metrics = think.get("runtime_metrics")
    raw_forced_structure_overrides = (
        thinker_runtime_metrics.get(
            "forced_structure_override_count",
            thinker_runtime_metrics.get("forced_structure_overrides", 0),
        )
        if isinstance(thinker_runtime_metrics, Mapping) else 0
    )
    forced_structure_overrides = (
        int(raw_forced_structure_overrides)
        if isinstance(raw_forced_structure_overrides, int)
        and not isinstance(raw_forced_structure_overrides, bool)
        else 0
    )
    if (
        normalized_action == ControlAction.BACKCHANNEL.value
        and forced_structure_overrides > 0
        and think.get("terminal_reason") in {"eos", "stop"}
    ):
        # A callback-injected EOS is a structural stop even though the decoder
        # loop sees an EOS token ID. Do not report it as model-sampled EOS.
        think["terminal_reason"] = "structure"
        if isinstance(thinker_runtime_metrics, dict):
            thinker_runtime_metrics["terminal_reason"] = "structure"

    def backchannel_lifecycle_metrics(
        *,
        omission_reason: str | None = None,
        pcm_present: bool = False,
        generated_token_count: int | None = None,
    ) -> dict[str, Any]:
        terminal_reason = think.get("terminal_reason")
        terminal = str(terminal_reason) if isinstance(terminal_reason, str) else None
        return {
            "terminal_reason": terminal,
            "eos_sampled": terminal == "eos",
            "stop_imposed": terminal if terminal in {
                "length", "budget_exhausted", "structure", "budget_partial",
                "sentence_boundary_after_budget",
            } else None,
            "generated_token_count": (
                thinker_generated_token_count
                if generated_token_count is None else max(0, int(generated_token_count))
            ),
            "omission_reason": omission_reason,
            "pcm_present": bool(pcm_present),
        }

    def early_cancel_completion_metrics() -> tuple[dict[str, Any], dict[str, Any]]:
        early_metrics: dict[str, Any] = {
            "thinker": dict(think.get("runtime_metrics", {})),
        }
        if isinstance(think.get("terminal_reason"), str):
            early_metrics["thinker"]["terminal_reason"] = think["terminal_reason"]
        explicit_cancel_reason = request.get("_cancellation_reason") or getattr(
            sess, "stop_reason", None
        )
        if explicit_cancel_reason is None:
            explicit_cancel_reason = "cancelled"
        completion = _response_completion_diagnostics(
            token_limit_requested=requested_token_limit,
            token_limit=int(think_kwargs["max_new_tokens"]),
            token_limit_effective=int(
                think_kwargs.get("_token_limit_effective", think_kwargs["max_new_tokens"])
            ),
            token_limit_cap_reason=(
                str(think_kwargs["_token_limit_cap_reason"])
                if think_kwargs.get("_token_limit_cap_reason") is not None
                else "profile_max"
                if profile_token_budget < requested_token_limit
                else None
            ),
            tokens_generated=thinker_generated_token_count,
            text_terminal_reason=think.get("terminal_reason"),
            codec_metrics={},
            audio_emitted=False,
            cancellation_reason=str(explicit_cancel_reason),
        )
        early_metrics["response_completion"] = completion
        if callable(runtime_callback):
            runtime_callback("thinker", early_metrics["thinker"])
            runtime_callback("response_completion", completion)
        return early_metrics, completion

    if (
        unified_native
        and isinstance(unified_branch, dict)
        and unified_branch.get("queue_open") is True
        and not unified_branch.get("ephemeral_private")
        and not think.get("cancelled")
        and think.get("terminal_reason") in {"eos", "length", "budget_exhausted"}
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
    backchannel_cancelled = bool(think.get("cancelled"))
    if normalized_action == ControlAction.BACKCHANNEL.value:
        backchannel_cancelled = bool(backchannel_cancelled or backchannel_cancelled_now())
    backchannel_skip_reason = (
        _backchannel_budget_without_eos(
            normalized_action,
            private=private_backchannel_active,
            controller_guided=controller_guided_say,
            terminal_reason=think.get("terminal_reason"),
            forced_structure_overrides=forced_structure_overrides,
            cancelled=backchannel_cancelled,
            buffered_tokens=controller_backchannel_packets or controller_backchannel_text,
        )
        if response_v2_collector is None else None
    )
    original_generated_ids = think.get("gen_ids")
    if (
        normalized_action == ControlAction.BACKCHANNEL.value
        and response_v2_collector is None
        and backchannel_skip_reason is None
        and (
            not isinstance(original_generated_ids, (tuple, list))
            or not original_generated_ids
            or len(original_generated_ids) > BACKCHANNEL_MAX_TEXT_TOKENS
        )
    ):
        backchannel_skip_reason = "backchannel_rejected"
    backchannel_validation: tuple[bool, str, int] | None = None
    final_callback_text = re.sub(
        r"\s*<unused\d+>\s*", " ", str(think.get("text", ""))
    )
    final_callback_text = re.sub(r"[ \t]+", " ", final_callback_text).strip()
    if normalized_action == ControlAction.BACKCHANNEL.value and response_v2_collector is None:
        if backchannel_skip_reason is not None:
            stop_live_say_worker()
            backchannel_metrics = {
                "status": "not_emitted",
                "reason": backchannel_skip_reason,
                "terminal_reason": str(think.get("terminal_reason")),
                "generated_token_ids": [int(value) for value in think.get("gen_ids", ())],
                "sampled_text_token_limit": BACKCHANNEL_MAX_TEXT_TOKENS,
                "audio_flushed": False,
                "canonical_listener_mutated": False,
                **backchannel_lifecycle_metrics(omission_reason=backchannel_skip_reason),
            }
            metric_name = (
                "private_backchannel_decode"
                if private_backchannel_active
                else "controller_backchannel_decode"
            )
            skipped_metrics = {
                "thinker": dict(think.get("runtime_metrics", {})),
                metric_name: backchannel_metrics,
            }
            skipped_completion, skipped_diagnostics = early_cancel_completion_metrics()
            skipped_metrics.update(skipped_completion)
            if callable(runtime_callback):
                runtime_callback(metric_name, backchannel_metrics)
            return {
                "text": "",
                "tool_calls": [],
                "wav": None,
                "elapsed": 0.0,
                "status": "backchannel_say_rejected",
                "guard_reason": backchannel_skip_reason,
                "skip_reason": backchannel_skip_reason,
                "cancelled": backchannel_skip_reason == "cancelled",
                "finish_reason": skipped_diagnostics["finish_reason"],
                "runtime_metrics": skipped_metrics,
                "_skip_history_commit": True,
            }

        source_backchannel_text = _decode_backchannel_source_text(think, engine.tokenizer)
        if source_backchannel_text is None:
            backchannel_validation = (False, "source_token_text_unavailable", 0)
        elif (
            controller_backchannel_text
            and "".join(controller_backchannel_text) != source_backchannel_text
        ):
            backchannel_validation = (False, "source_token_text_mismatch", 0)
        elif (
            controller_backchannel_packets
            and [int(packet[1]) for packet in controller_backchannel_packets]
            != [int(value) for value in original_generated_ids]
        ):
            backchannel_validation = (False, "source_token_ids_mismatch", 0)
        else:
            backchannel_validation = validate_backchannel_phrase(
                source_backchannel_text, max_words=4
            )
        backchannel_ok, backchannel_reason, backchannel_word_count = backchannel_validation
        if not backchannel_ok:
            stop_live_say_worker()
            controller_metrics = {
                "response_protocol": response_protocol or CONTROLLER_GUIDED_SAY_V1,
                "first_thinker_token_id": thinker_first_token_id,
                "status": "backchannel_say_rejected",
                "wake_action": normalized_action,
                "response_action": ControlAction.BACKCHANNEL.value,
                "say_word_count": backchannel_word_count,
                "guard_reason": backchannel_reason,
                "text_rewritten": False,
            }
            rejected_metrics = {
                "thinker": dict(think.get("runtime_metrics", {})),
                "audio_response_protocol": controller_metrics,
                (
                    "private_backchannel_decode"
                    if private_backchannel_active else "controller_backchannel_decode"
                ): {
                    "status": "not_emitted",
                    "reason": "backchannel_rejected",
                    "guard_reason": backchannel_reason,
                    "generated_token_ids": [
                        int(value) for value in original_generated_ids
                    ] if isinstance(original_generated_ids, (tuple, list)) else [],
                    "sampled_text_token_limit": BACKCHANNEL_MAX_TEXT_TOKENS,
                    "audio_flushed": False,
                    "canonical_listener_mutated": False,
                    **backchannel_lifecycle_metrics(omission_reason="backchannel_rejected"),
                },
            }
            rejected_completion, rejected_diagnostics = early_cancel_completion_metrics()
            rejected_metrics.update(rejected_completion)
            if callable(runtime_callback):
                runtime_callback("audio_response_protocol", controller_metrics)
                runtime_callback(
                    "private_backchannel_decode"
                    if private_backchannel_active else "controller_backchannel_decode",
                    rejected_metrics[
                        "private_backchannel_decode"
                        if private_backchannel_active else "controller_backchannel_decode"
                    ],
                )
            return {
                "text": "",
                "tool_calls": [],
                "wav": None,
                "elapsed": 0.0,
                "status": "backchannel_say_rejected",
                "guard_reason": backchannel_reason,
                "skip_reason": "backchannel_rejected",
                "finish_reason": rejected_diagnostics["finish_reason"],
                "runtime_metrics": rejected_metrics,
                "_skip_history_commit": True,
            }
    if live_say_active:
        if think.get("cancelled"):
            stop_live_say_worker()
        elif think.get("terminal_reason") not in {
            "eos", "stop", "length", "budget_exhausted",
        }:
            incomplete = RuntimeError(
                "controller-guided live SAY requires Thinker EOS before Qwen text flush"
            )
            stop_live_say_worker(incomplete)
            raise incomplete
        else:
            try:
                if controller_guided_say and normalized_action == ControlAction.BACKCHANNEL.value:
                    if live_say_stream is None:
                        raise RuntimeError(
                            "controller-guided backchannel reached Thinker EOS without its live SAY stream"
                        )

                    def publish_validated_backchannel_batch(
                        packets: tuple[tuple[int, int, str, Any], ...],
                    ) -> None:
                        with live_say_publication_lock:
                            if think.get("cancelled") or backchannel_cancelled_now():
                                raise LiveSayCancelled(
                                    "backchannel cancelled before Live SAY token publication"
                                )
                            base_cursor = int(
                                unified_branch.get("spoken_prefix_base_cursor")
                                or unified_branch.get("cache_position")
                                or 0
                            ) if isinstance(unified_branch, dict) else 0
                            for token_index, token_id, token_text, hidden in packets:
                                packet_cursor = (
                                    {"exclusive_kv_cursor": base_cursor + token_index + 1}
                                    if unified_native else {}
                                )
                                live_say_generated_token(
                                    token_index,
                                    token_id,
                                    token_text,
                                    hidden,
                                    **packet_cursor,
                                )

                    published_backchannel_tokens = _publish_backchannel_tokens_after_eos(
                        controller_backchannel_packets,
                        terminal_reason=think.get("terminal_reason"),
                        forced_structure_overrides=forced_structure_overrides,
                        publish_batch=publish_validated_backchannel_batch,
                    )
                    if not published_backchannel_tokens:
                        stop_live_say_worker()
                        rejected_metrics, completion = early_cancel_completion_metrics()
                        omission_reason = (
                            "cancelled"
                            if think.get("cancelled") or backchannel_cancelled_now()
                            else "backchannel_tokens_unpublished"
                        )
                        rejected_metrics["controller_backchannel_decode"] = {
                            "status": "not_emitted",
                            "reason": omission_reason,
                            "audio_flushed": False,
                            "generated_token_ids": [
                                int(value) for value in original_generated_ids
                            ] if isinstance(original_generated_ids, (tuple, list)) else [],
                            **backchannel_lifecycle_metrics(omission_reason=omission_reason),
                        }
                        return {
                            "text": "",
                            "tool_calls": [],
                            "wav": None,
                            "elapsed": 0.0,
                            "status": "backchannel_say_rejected",
                            "guard_reason": omission_reason,
                            "skip_reason": omission_reason,
                            "finish_reason": completion["finish_reason"],
                            "runtime_metrics": rejected_metrics,
                            "_skip_history_commit": True,
                        }
                if normalized_action == ControlAction.BACKCHANNEL.value and not (
                    backchannel_validation and backchannel_validation[0]
                ):
                    raise RuntimeError("backchannel_reached_live_say_without_phrase_validation")
                if final_callback_text.strip():
                    live_say_stream.finish(final_text=final_callback_text)
                    start_live_say_talker()
                    if live_say_thread is not None:
                        live_say_thread.join()
                    if "error" in live_say_result:
                        raise live_say_result["error"]
                else:
                    stop_live_say_worker()
            except LiveSayCancelled as exc:
                response_cancel_event = request.get("_response_cancel_event")
                stop_check = request.get("_stop_check")
                cancellation_confirmed = bool(
                    getattr(sess, "stop_requested", False)
                    or think.get("cancelled")
                    or (
                        response_cancel_event is not None
                        and response_cancel_event.is_set()
                    )
                    or (
                        generation_id is not None
                        and not sess.is_generation_current(generation_id=generation_id)
                    )
                    or (callable(stop_check) and stop_check())
                )
                if cancellation_confirmed:
                    stop_live_say_worker()
                    return {
                        "text": final_callback_text or "",
                        "tool_calls": [],
                        "wav": None,
                        "elapsed": 0.0,
                        "cancelled": True,
                        "finish_reason": "cancelled",
                        "_skip_history_commit": True,
                        "runtime_metrics": (
                            runtime_metrics if "runtime_metrics" in locals() else {}
                        ),
                    }
                stop_live_say_worker(exc)
                raise
            except BaseException as exc:
                stop_live_say_worker(exc)
                raise
    runtime_metrics: dict[str, Any] = {
        "thinker": dict(think.get("runtime_metrics", {})),
    }
    if (
        normalized_action == ControlAction.BACKCHANNEL.value
        and response_v2_collector is None
    ):
        backchannel_metric_name = (
            "private_backchannel_decode"
            if private_backchannel_active else "controller_backchannel_decode"
        )
        runtime_metrics[backchannel_metric_name] = {
            "status": "validated",
            "reason": None,
            "generated_token_ids": [
                int(value) for value in original_generated_ids
            ] if isinstance(original_generated_ids, (tuple, list)) else [],
            "sampled_text_token_limit": BACKCHANNEL_MAX_TEXT_TOKENS,
            "audio_flushed": False,
            "canonical_listener_mutated": False,
            **backchannel_lifecycle_metrics(),
        }
    if controller_guided_say:
        runtime_metrics["heard_transcript"] = {
            "status": "absent_from_controller_guided_say_v1",
            "source": None,
            "note": "this response protocol does not generate an ASR transcript",
        }
    if response_v2_collector is not None:
        action_pool_overrides = min(
            int(runtime_metrics["thinker"].get(
                "audio_response_protocol_v2_action_pool_overrides", 0
            )),
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

    def current_cancellation_reason() -> str:
        explicit = request.get("_cancellation_reason") or getattr(sess, "stop_reason", None)
        if explicit is not None:
            return str(explicit)
        raw_reason = think.get("terminal_reason")
        if isinstance(raw_reason, str) and raw_reason.startswith("cancelled"):
            return raw_reason
        return "cancelled"

    def record_response_completion(
        *,
        codec_metrics: Mapping[str, Any] | None = None,
        response_wav: Any = None,
        cancellation_reason: str | None = None,
    ) -> dict[str, Any]:
        completion_codec_metrics = dict(codec_metrics) if isinstance(codec_metrics, Mapping) else {}
        closed_units = completion_codec_metrics.get("closed_tts_units")
        has_audio = bool(
            isinstance(response_wav, np.ndarray) and response_wav.size > 0
        ) or bool(
            isinstance(completion_codec_metrics.get("generated_cb0_frames"), int)
            and completion_codec_metrics.get("generated_cb0_frames", 0) > 0
        ) or bool(
            isinstance(closed_units, list)
            and any(
                isinstance(unit, Mapping)
                and isinstance(unit.get("generated_cb0_frames"), int)
                and unit.get("generated_cb0_frames", 0) > 0
                for unit in closed_units
            )
        )
        if has_audio:
            produced = completion_codec_metrics.get("generated_cb0_frames")
            if not isinstance(produced, int) or isinstance(produced, bool):
                units = completion_codec_metrics.get("closed_tts_units")
                has_unit_frames = isinstance(units, list) and any(
                    isinstance(unit, Mapping)
                    and isinstance(unit.get("generated_cb0_frames"), int)
                    and not isinstance(unit.get("generated_cb0_frames"), bool)
                    for unit in units
                )
                if not has_unit_frames:
                    completion_codec_metrics["codec_frames_produced"] = int(response_wav.size // 1920)
            if not any(
                isinstance(completion_codec_metrics.get(key), int)
                and not isinstance(completion_codec_metrics.get(key), bool)
                for key in ("codec_frame_limit", "effective_max_frames", "configured_max_frames")
            ) and not completion_codec_metrics.get("closed_tts_units"):
                completion_codec_metrics["codec_frame_limit"] = (
                    ARIA_BACKCHANNEL_MAX_FRAMES
                    if normalized_action == ControlAction.BACKCHANNEL.value
                    else response_codec_frame_limit
                )
        completion = _response_completion_diagnostics(
            token_limit_requested=requested_token_limit,
            token_limit=int(think_kwargs["max_new_tokens"]),
            token_limit_effective=int(
                think_kwargs.get("_token_limit_effective", think_kwargs["max_new_tokens"])
            ),
            token_limit_cap_reason=(
                str(think_kwargs["_token_limit_cap_reason"])
                if think_kwargs.get("_token_limit_cap_reason") is not None
                else "profile_max"
                if profile_token_budget < requested_token_limit
                else None
            ),
            tokens_generated=thinker_generated_token_count,
            text_terminal_reason=think.get("terminal_reason"),
            codec_metrics=completion_codec_metrics,
            audio_emitted=has_audio,
            cancellation_reason=cancellation_reason,
        )
        completion["response_profile"] = response_profile
        completion["codec_frame_limit_requested"] = int(response_codec_frame_limit)
        completion["codec_frame_limit_effective"] = int(response_codec_frame_limit)
        completion["acoustic_limit_reached"] = bool(
            completion_codec_metrics.get("terminated_by_budget") is True
            or any(
                isinstance(unit, Mapping) and unit.get("terminated_by_budget") is True
                for unit in (
                    completion_codec_metrics.get("closed_tts_units")
                    if isinstance(completion_codec_metrics.get("closed_tts_units"), list)
                    else ()
                )
            )
            or (
                completion.get("codec_frames_produced", 0) >= response_codec_frame_limit
                and not completion.get("eos_detected")
            )
        )
        if isinstance(response_ack_branch, dict):
            playback_segments = response_ack_branch.get("playback_segments")
            consumed_segments = set(response_ack_branch.get("client_consumed_segments", ()))
            consumed_events = {
                str(item.get("speaker_segment_id")): item
                for item in response_ack_branch.get("client_consumed_pcm_events", ())
                if isinstance(item, Mapping)
            }
            unit_reasons = {
                str(item.get("speaker_segment_id")): str(item.get("terminal_reason"))
                for item in (
                    completion_codec_metrics.get("closed_tts_units")
                    if isinstance(completion_codec_metrics.get("closed_tts_units"), list)
                    else ()
                )
                if isinstance(item, Mapping) and item.get("terminal_reason") is not None
            }
            segment_diagnostics: list[dict[str, Any]] = []
            if isinstance(playback_segments, Mapping):
                for segment_id, segment in playback_segments.items():
                    if not isinstance(segment, Mapping):
                        continue
                    start_sample = segment.get("start_sample")
                    end_sample = segment.get("emitted_end_sample")
                    token_packets = segment.get("token_packets")
                    packets = [item for item in token_packets if isinstance(item, Mapping)] \
                        if isinstance(token_packets, (tuple, list)) else []
                    ack = consumed_events.get(str(segment_id))
                    emitted_samples = (
                        max(0, int(end_sample) - int(start_sample))
                        if isinstance(start_sample, int) and isinstance(end_sample, int)
                        else 0
                    )
                    segment_diagnostics.append({
                        "speaker_segment_id": str(segment_id),
                        "original_token_ids": [
                            int(item["token_id"])
                            for item in packets
                            if isinstance(item.get("token_id"), int)
                            and not isinstance(item.get("token_id"), bool)
                        ],
                        "original_token_cursor_start": (
                            int(packets[0]["exclusive_kv_cursor"]) - 1
                            if packets and isinstance(packets[0].get("exclusive_kv_cursor"), int)
                            else None
                        ),
                        "original_token_cursor_end": (
                            int(packets[-1]["exclusive_kv_cursor"])
                            if packets and isinstance(packets[-1].get("exclusive_kv_cursor"), int)
                            else None
                        ),
                        "pcm_emitted_start_sample": start_sample,
                        "pcm_emitted_end_sample": end_sample,
                        "pcm_samples_emitted": emitted_samples,
                        "pcm_consumed_start_sample": ack.get("start_sample") if ack else None,
                        "pcm_consumed_end_sample": ack.get("end_sample") if ack else None,
                        "pcm_samples_consumed": (
                            int(ack["end_sample"]) - int(ack["start_sample"])
                            if ack else 0
                        ),
                        "client_consumed": str(segment_id) in consumed_segments,
                        "termination_reason": unit_reasons.get(
                            str(segment_id),
                            "length" if completion["acoustic_limit_reached"] else
                            "cancelled" if completion["finish_reason"] == "cancelled" else
                            "eos" if completion["eos_detected"] else "unknown",
                        ),
                    })
            completion["audio_segments"] = segment_diagnostics
        runtime_metrics["response_completion"] = completion
        if isinstance(codec_metrics, dict):
            codec_reason = None
            if completion["finish_reason"] == "cancelled":
                codec_reason = "cancelled"
            elif (
                not completion["eos_detected"]
                and completion["codec_frame_limit"] is not None
                and completion["codec_frames_produced"] >= completion["codec_frame_limit"]
            ) or codec_metrics.get("terminated_by_budget") is True or any(
                isinstance(unit, Mapping) and unit.get("terminated_by_budget") is True
                for unit in (
                    codec_metrics.get("closed_tts_units")
                    if isinstance(codec_metrics.get("closed_tts_units"), list) else ()
                )
            ):
                codec_reason = "length"
            elif completion["eos_detected"]:
                codec_reason = "eos"
            if codec_reason is not None:
                codec_metrics.setdefault("terminal_reason", codec_reason)
                if callable(runtime_callback):
                    runtime_callback("talker", codec_metrics)
        if callable(runtime_callback):
            runtime_callback("response_completion", completion)
        return completion

    generated_text = re.sub(r"\s*<unused\d+>\s*", " ", str(think["text"]))
    generated_text = re.sub(r"[ \t]+", " ", generated_text).strip()
    response_hidden = think.get("hidden")
    response_prompt_len = int(think.get("hidden_prompt_len", think.get("prompt_len", len(ids))))
    response_action_value: str | None = None
    frame_residual_response_hidden: torch.Tensor | None = None
    frame_residual_response_texts: tuple[str, ...] | None = None
    frame_residual_alignment_metrics: dict[str, Any] | None = None
    if private_backchannel_active:
        generated_ids = think.get("gen_ids")
        if not isinstance(generated_ids, (tuple, list)) or len(generated_ids) > BACKCHANNEL_MAX_TEXT_TOKENS:
            raise RuntimeError(
                "private backchannel exceeded its six sampled-text-token budget"
            )
        response_action_value = ControlAction.BACKCHANNEL.value
        response_prompt_len = 0
        runtime_metrics["private_backchannel_decode"] = {
            "status": "decoded",
            "forked_native_kv": True,
            "generated_token_ids": [int(value) for value in generated_ids],
            "sampled_text_token_limit": BACKCHANNEL_MAX_TEXT_TOKENS,
            "canonical_listener_mutated": False,
            **backchannel_lifecycle_metrics(),
        }

    def empty_vocal_say_result() -> dict[str, Any]:
        completion = record_response_completion()
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
            "finish_reason": completion["finish_reason"],
            "history_messages": completed_history,
            "runtime_metrics": runtime_metrics,
            "_skip_history_commit": True,
        }

    if controller_guided_say:
        if (
            think.get("cancelled")
            or (generation_id is not None and not sess.is_generation_current(generation_id=generation_id))
            or (request.get("_stop_check") and request["_stop_check"]())
        ):
            if normalized_action == ControlAction.BACKCHANNEL.value:
                backchannel_metric_name = (
                    "private_backchannel_decode"
                    if private_backchannel_active else "controller_backchannel_decode"
                )
                metrics = runtime_metrics.setdefault(backchannel_metric_name, {})
                metrics.update(backchannel_lifecycle_metrics(omission_reason="cancelled"))
                metrics.update({
                    "status": "not_emitted",
                    "reason": "cancelled",
                    "audio_flushed": False,
                })
                if callable(runtime_callback):
                    runtime_callback(backchannel_metric_name, dict(metrics))
            completion = record_response_completion(
                cancellation_reason=current_cancellation_reason()
            )
            return {
                "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
                "cancelled": True, "finish_reason": completion["finish_reason"],
                "runtime_metrics": runtime_metrics,
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
            _qwen_token_ids, qwen_offsets = qwen_assistant_text_offsets(
                generated_text,
                tokenizer=qwen_tokenizer,
                processor=processor,
            )
            frame_residual_response_texts = _qwen_text_token_surfaces(
                generated_text, qwen_offsets
            )
            if len(frame_residual_response_texts) != int(frame_residual_response_hidden.shape[1]):
                raise RuntimeError("frame_residual_response_text_hidden_alignment_mismatch")
        if response_action_value == ControlAction.BACKCHANNEL.value:
            backchannel_ok, backchannel_reason, backchannel_word_count = (
                backchannel_validation
                if backchannel_validation is not None
                else validate_backchannel_phrase(generated_text, max_words=4)
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
                completion = record_response_completion()
                return {
                    "text": "",
                    "tool_calls": [],
                    "wav": None,
                    "elapsed": 0.0,
                    "status": "backchannel_say_rejected",
                    "finish_reason": completion["finish_reason"],
                    "runtime_metrics": runtime_metrics,
                    "history_messages": completed_history,
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
                if normalized_action == ControlAction.BACKCHANNEL.value:
                    backchannel_metric_name = (
                        "private_backchannel_decode"
                        if private_backchannel_active else "controller_backchannel_decode"
                    )
                    metrics = runtime_metrics.setdefault(backchannel_metric_name, {})
                    metrics.update(backchannel_lifecycle_metrics(omission_reason="cancelled"))
                    metrics.update({
                        "status": "not_emitted",
                        "reason": "cancelled",
                        "audio_flushed": False,
                    })
                    if callable(runtime_callback):
                        runtime_callback(backchannel_metric_name, dict(metrics))
                completion = record_response_completion(
                    cancellation_reason=current_cancellation_reason()
                )
                return {
                    "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
                    "cancelled": True, "finish_reason": completion["finish_reason"],
                    "runtime_metrics": runtime_metrics,
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
        if should_stream_thinker_text(
            stream_requested=stream_requested,
            unified_native=unified_native,
            tts_text_consumed_callback=request.get("_on_tts_text_consumed"),
        ):
            remaining = (
                generated_text[streamed_live_chars:]
                if streamed_live_chars > 0
                else generated_text
            )
            if remaining:
                render_visible_text_chunk(
                    remaining,
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
            completion = record_response_completion(
                cancellation_reason=current_cancellation_reason()
            )
            return {
                "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
                "cancelled": True, "finish_reason": completion["finish_reason"],
                "runtime_metrics": runtime_metrics,
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
        if response_action_value == ControlAction.BACKCHANNEL.value:
            say_tokens = response_v2_collector.say_tokens
            say_token_ids = [int(token.token_id) for token in say_tokens]
            omission_reason = _backchannel_budget_without_eos(
                ControlAction.BACKCHANNEL.value,
                private=False,
                controller_guided=False,
                terminal_reason=think.get("terminal_reason"),
                forced_structure_overrides=forced_structure_overrides,
                cancelled=bool(think.get("cancelled") or backchannel_cancelled_now()),
                buffered_tokens=say_tokens,
            )
            say_validation = validate_backchannel_phrase(
                str(parsed_response.get("say", "")), max_words=4
            )
            if omission_reason is None and (
                len(say_token_ids) > BACKCHANNEL_MAX_TEXT_TOKENS
                or not say_validation[0]
            ):
                omission_reason = "backchannel_rejected"
            backchannel_metric_name = "controller_backchannel_decode"
            backchannel_metrics = {
                "status": "not_emitted" if omission_reason else "validated",
                "reason": omission_reason,
                "guard_reason": say_validation[1] if not say_validation[0] else None,
                "terminal_reason": str(think.get("terminal_reason")),
                "generated_token_ids": say_token_ids,
                "sampled_text_token_limit": BACKCHANNEL_MAX_TEXT_TOKENS,
                "audio_flushed": False,
                "canonical_listener_mutated": False,
                **backchannel_lifecycle_metrics(
                    omission_reason=omission_reason,
                    generated_token_count=len(say_token_ids),
                ),
            }
            runtime_metrics[backchannel_metric_name] = backchannel_metrics
            if callable(runtime_callback):
                runtime_callback(backchannel_metric_name, dict(backchannel_metrics))
            if omission_reason is not None:
                response_v2_collector.cancel()
                completion = record_response_completion()
                return {
                    "text": "",
                    "tool_calls": [],
                    "wav": None,
                    "elapsed": 0.0,
                    "status": "backchannel_say_rejected",
                    "guard_reason": backchannel_metrics["guard_reason"],
                    "skip_reason": omission_reason,
                    "finish_reason": completion["finish_reason"],
                    "runtime_metrics": runtime_metrics,
                    "history_messages": completed_history,
                    "_skip_history_commit": True,
                }
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
            completion = record_response_completion()
            return {
                "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
                "status": "deferred_until_endpoint",
                "response_action": parsed_response["action"],
                "finish_reason": completion["finish_reason"],
                "runtime_metrics": runtime_metrics,
                "history_messages": completed_history,
                "_skip_history_commit": True,
            }
        if parsed_disposition == "stale_generation":
            if live_v2_say_active:
                stop_live_say_worker()
            completion = record_response_completion(
                cancellation_reason=current_cancellation_reason()
            )
            return {
                "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
                "cancelled": True, "finish_reason": completion["finish_reason"],
                "runtime_metrics": runtime_metrics,
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
        nonunified_v2_source_rows = ()
        if (
            direct_audio
            and response_v2_collector is not None
            and not unified_native
            and parsed_response["action"] in {
                ControlAction.SPEAK.value,
                ControlAction.BACKCHANNEL.value,
            }
            and generated_text
        ):
            rows: list[dict[str, Any]] = []
            text_cursor = 0
            for token in say_tokens:
                start, end = token.char_span
                if (
                    isinstance(start, bool)
                    or isinstance(end, bool)
                    or not isinstance(start, int)
                    or not isinstance(end, int)
                    or start != text_cursor
                    or end < start
                    or end > len(generated_text)
                ):
                    raise RuntimeError(
                        "validated v2 SAY source token spans do not cover decoded text"
                    )
                rows.append({
                    "token_id": int(token.token_id),
                    "text_piece": generated_text[start:end],
                })
                text_cursor = end
            if not rows or text_cursor != len(generated_text):
                raise RuntimeError(
                    "validated v2 SAY source tokens do not cover decoded text"
                )
            nonunified_v2_source_rows = tuple(rows)

        def record_nonunified_v2_tts_ingestion() -> None:
            if not nonunified_v2_source_rows:
                return
            branch = response_ack_branch
            ledger = (
                branch.get("response_event_ledger")
                if isinstance(branch, dict) else None
            )
            if ledger is None:
                if request.get("_generation_token") is None:
                    return
                raise RuntimeError(
                    "non-unified v2 SAY has no response-owned provenance ledger"
                )
            record_generated = getattr(ledger, "record_generated_tokens", None)
            record_consumed = getattr(ledger, "record_tts_consumed", None)
            source_frontier = getattr(ledger, "source_frontier_snapshot", None)
            if not callable(record_generated) or not callable(record_consumed):
                raise RuntimeError(
                    "non-unified v2 SAY ledger lacks source or TTS-consumption recording"
                )
            if not callable(source_frontier):
                raise RuntimeError(
                    "non-unified v2 SAY ledger lacks a logical source sequence frontier"
                )
            frontier = source_frontier()
            next_source_sequence = frontier.get("next_source_sequence")
            if (
                isinstance(next_source_sequence, bool)
                or not isinstance(next_source_sequence, int)
                or next_source_sequence < 0
            ):
                raise RuntimeError(
                    "non-unified v2 SAY ledger has an invalid logical source sequence frontier"
                )
            occurrence_rows = tuple(
                {**row, "source_token_seq": int(next_source_sequence) + index}
                for index, row in enumerate(nonunified_v2_source_rows)
            )
            source_sequences = record_generated(occurrence_rows)
            if len(source_sequences) != len(nonunified_v2_source_rows):
                raise RuntimeError("non-unified v2 SAY source ledger count mismatch")
            source_packets = tuple(
                {**row, "source_token_seq": int(source_seq)}
                for row, source_seq in zip(
                    nonunified_v2_source_rows, source_sequences, strict=True
                )
            )
            record_consumed(max(int(source_seq) for source_seq in source_sequences))
            if isinstance(branch, dict):
                branch["one_shot_v2_tts_source_tokens"] = source_packets

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
        completion = record_response_completion(
            cancellation_reason=current_cancellation_reason()
        )
        return {
            "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
            "cancelled": True, "finish_reason": completion["finish_reason"],
            "runtime_metrics": runtime_metrics,
        }
    calls = extract_tool_calls(think["text"], tools) if tools else []
    if calls:
        completion = record_response_completion()
        return {
            "text": "", "tool_calls": calls, "wav": None, "elapsed": 0.0,
            "finish_reason": completion["finish_reason"],
            "runtime_metrics": runtime_metrics,
        }
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
            frame_residual_response_texts = _qwen_text_token_surfaces(
                generated_text, qwen_offsets
            )
            frame_residual_response_hidden = align_hidden_states_by_char_overlap(
                response_hidden,
                say_spans,
                qwen_offsets,
            )
            if len(frame_residual_response_texts) != int(frame_residual_response_hidden.shape[1]):
                raise RuntimeError("frame_residual_response_text_hidden_alignment_mismatch")
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
            _qwen_token_ids, qwen_offsets = qwen_assistant_text_offsets(
                generated_text,
                tokenizer=qwen_tokenizer,
                processor=processor,
            )
            frame_residual_response_texts = _qwen_text_token_surfaces(
                generated_text, qwen_offsets
            )
            if frame_residual_response_hidden is None or int(frame_residual_response_hidden.shape[1]) < 1:
                raise RuntimeError("frame_residual_response_hidden_alignment_empty")
            if len(frame_residual_response_texts) != int(frame_residual_response_hidden.shape[1]):
                raise RuntimeError("frame_residual_response_text_hidden_alignment_mismatch")
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
        completion = record_response_completion()
        return {
            "text": "",
            "tool_calls": [],
            "wav": None,
            "elapsed": 0.0,
            "status": "no_speech",
            "finish_reason": completion["finish_reason"],
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
            completion = record_response_completion(
                cancellation_reason="backchannel_conditioning_unavailable"
            )
            return {
                "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
                "skip_reason": "backchannel_conditioning_unavailable",
                "cancelled": True, "finish_reason": completion["finish_reason"],
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
                    on_audio_chunk=audio_chunk_callback,
                    generation_id=generation_id,
                    min_frames=2,
                    metrics=talker_metrics,
                    **(
                        {
                            "listener_admission_wait": listener_admission_wait,
                            "listener_compute_quantum": listener_compute_quantum,
                        }
                        if direct_audio and callable(listener_admission_wait)
                        and callable(listener_compute_quantum)
                        else {}
                    ),
                    **(
                        {"on_text_input_prefill_complete": record_nonunified_v2_tts_ingestion}
                        if nonunified_v2_source_rows else {}
                    ),
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
                    on_audio_chunk=audio_chunk_callback,
                    on_audio_chunk_with_horizon=audio_chunk_with_horizon_callback,
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
                        ARIA_BACKCHANNEL_MAX_FRAMES
                        if controller_guided_say and is_backchannel_action
                        else response_codec_frame_limit
                        if response_profile == LONG_RESPONSE_PROFILE
                        else None
                    ),
                    listener_admission_wait=listener_admission_wait,
                    listener_compute_quantum=(
                        listener_compute_quantum
                        if callable(listener_compute_quantum) else None
                    ),
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
            if callable(listener_compute_quantum) and callable(listener_admission_wait):
                talker_admission_kwargs["listener_compute_quantum"] = listener_compute_quantum
            if quality_diagnostic is not None:
                talker_admission_kwargs["quality_diagnostic"] = quality_diagnostic
            if nonunified_v2_source_rows:
                talker_admission_kwargs["on_text_input_prefill_complete"] = (
                    record_nonunified_v2_tts_ingestion
                )
            if (controller_guided_say or frame_residual_candidate) and is_backchannel_action:
                talker_admission_kwargs["min_frames"] = 2
                talker_admission_kwargs["max_frames"] = ARIA_BACKCHANNEL_MAX_FRAMES
            elif response_profile == LONG_RESPONSE_PROFILE:
                talker_admission_kwargs["max_frames"] = response_codec_frame_limit
            if frame_residual_candidate:
                if frame_residual_response_hidden is None:
                    raise RuntimeError("frame_residual_response_hidden_alignment_missing")
                talker_admission_kwargs["frame_residual_response_hidden"] = frame_residual_response_hidden
                if frame_residual_response_texts is not None:
                    if len(frame_residual_response_texts) != int(frame_residual_response_hidden.shape[1]):
                        raise RuntimeError("frame_residual_response_text_hidden_alignment_mismatch")
                    talker_admission_kwargs["frame_residual_response_texts"] = frame_residual_response_texts
            with _diagnostic_talker_rng_scope(engine):
                wav = engine.talk(text, h_pfx, n_words=max(1, len(text.split())), voice_conditioning=cond,
                                   stop_check=lambda: (
                                       sess.stop_requested or bool(
                                           request.get("_stop_check")
                                           and request["_stop_check"]()
                                       )
                                   ), state=sess,
                                   on_audio_chunk=audio_chunk_callback, generation_id=generation_id,
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
    completion_cancelled = bool(
        talker_cancelled
        or think.get("cancelled")
        or generation_stale
        or (request.get("_stop_check") and request["_stop_check"]())
    )
    completion_diagnostics = record_response_completion(
        codec_metrics=talker_runtime if isinstance(talker_runtime, Mapping) else None,
        response_wav=wav,
        cancellation_reason=current_cancellation_reason() if completion_cancelled else None,
    )
    if (
        response_action_value == ControlAction.BACKCHANNEL.value
        or (
            response_v2_collector is None
            and normalized_action == ControlAction.BACKCHANNEL.value
        )
    ):
        backchannel_metric_name = (
            "private_backchannel_decode"
            if private_backchannel_active else "controller_backchannel_decode"
        )
        backchannel_metrics = runtime_metrics.setdefault(backchannel_metric_name, {})
        wav_has_pcm = False
        if wav is not None:
            try:
                wav_has_pcm = bool(np.asarray(wav).size)
            except (TypeError, ValueError):
                wav_has_pcm = False
        codec_has_pcm = any(
            isinstance(talker_runtime, Mapping)
            and isinstance(talker_runtime.get(key), int)
            and not isinstance(talker_runtime.get(key), bool)
            and int(talker_runtime[key]) > 0
            for key in (
                "generated_cb0_frames", "codec_frames_produced", "rvq_frames",
            )
        )
        pcm_present = bool(backchannel_pcm_emitted or wav_has_pcm or codec_has_pcm)
        backchannel_metrics.update(backchannel_lifecycle_metrics(pcm_present=pcm_present))
        backchannel_metrics["pcm_present"] = pcm_present
        backchannel_metrics["audio_flushed"] = pcm_present
        if callable(runtime_callback):
            runtime_callback(backchannel_metric_name, dict(backchannel_metrics))
    if (
        talker_cancelled
        or generation_stale
        or (request.get("_stop_check") and request["_stop_check"]())
    ):
        return {
            "text": "", "tool_calls": [], "wav": None, "elapsed": 0.0,
            "cancelled": True, "finish_reason": "cancelled",
            "runtime_metrics": runtime_metrics,
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
            "finish_reason": completion_diagnostics["finish_reason"],
            "history_messages": completed_history,
            "runtime_metrics": runtime_metrics,
            "_skip_history_commit": True,
        }
    if request.get("_defer_history"):
        history_messages = completed_history
    else:
        history_changed = False
        with sess._generation_lock:
            if generation_id is None or sess.is_generation_current(generation_id=generation_id):
                history_changed = sess._replace_history_locked(
                    completed_history + [{"role": "assistant", "content": text}]
                )
                if history_changed:
                    sess.note_semantic_context_changed()
            else:
                sess.stale_history_writes += 1
        if history_changed:
            sess.notify_history_context_changed()
    return {"text": text, "wav": wav, "tool_calls": [], "elapsed": 0.0,
            "finish_reason": completion_diagnostics["finish_reason"],
            "history_messages": completed_history, "runtime_metrics": runtime_metrics,
            "_skip_history_commit": bool(
                direct_audio
                and (
                    response_action_value == ControlAction.BACKCHANNEL.value
                    if response_v2_collector is not None
                    else normalized_action == ControlAction.BACKCHANNEL.value
                )
            )}
