"""OpenAI-compatible HTTP/SSE and realtime duplex WebSocket server."""
from __future__ import annotations
import asyncio, base64, copy, gc, hashlib, io, json, logging, os, queue, re, sys, tempfile, threading, time, uuid
import itertools
from dataclasses import dataclass, replace
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, OrderedDict, deque
from datetime import datetime, timezone
from math import gcd
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence
import numpy as np
import torch
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
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
    AUDIO_FRAME_SAMPLES,
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
from duplex_kernel.gpu_admission import (
    DEFAULT_GPU_ADMISSION_CAPACITY,
    GPU_ADMISSION_CAPACITY_ENV,
    GpuAdmissionASGIMiddleware,
    GpuAdmissionLease,
    GpuAdmissionLimiter,
    parse_gpu_admission_capacity,
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
from duplex_kernel.live_say import LiveSayCancelled
from duplex_kernel.playout import PlayoutFrame, SpeakerPlayoutBuffer
from duplex_kernel.scheduler import DIRECT_V2_INGRESS_QUEUE_FRAMES, REALTIME_INGRESS_QUEUE_FRAMES
from duplex_kernel.shared_listener_scheduler import SharedListenerInput, SharedListenerTickDispatcher
from duplex_kernel.microbatch_contract import ListenerFrame
from duplex_kernel.listener_microbatch_serving import (
    commit_initial_semantic_prefix_state,
    initial_listener_session_state,
    make_listener_microbatch_worker,
    make_unified_prefix_import,
    prepare_initial_semantic_prefix_async,
    semantic_prefix_importer,
)
from duplex_kernel.listener_microbatch_worker import (
    ResponseDecodePurpose,
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
from duplex_kernel.engine_types import ListenerBacklogAdmission
from duplex_kernel.tts_quality_diagnostics import TTSQualityDiagnosticCapture
from duplex_kernel.listener_microbatch_serving import _semantic_prefix_request_matches_applied_row
from duplex_kernel.server_response_builder import (
    _ACTION_TOKEN_MAP,
    _AUDIO_RESPONSE_V2_ACTION_PROBE_LOCK,
    _AUDIO_RESPONSE_V2_ACTION_PROBE_CLAIMED,
)
from duplex_kernel.tts_quality_diagnostics import (
    DIAGNOSTIC_MAX_ACOUSTIC_FRAME_BUDGET,
)
from duplex_kernel.session import (
    DuplexSession,
    GemmaTokenTickPacer,
    GenerationToken,
    SessionManager,
    SpokenTokenPacket,
)
from duplex_kernel.server_state import (
    _CausalUserSpeechTracker,
    DEFAULT_SPEAK_ENDPOINT_QUIET_MS,
    _EngineSynthesisOwner,
    _InferenceExecutionGate,
    POLICY_FRAME_MS,
    _RealtimeResponseControl,
    _ResponseAudioAdmissionState,
    _SchedulerEventBuffer,
    _configured_speak_endpoint_frames,
    POLICY_FRAME_SAMPLES,
    REALTIME_AUDIO_PACKET_CAPACITY,
    _verify_v9_training_reference,
)
from duplex_kernel.runtime_source_diagnostics import collect_runtime_source_diagnostics
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
from duplex_kernel.server_diagnostics import (
    NATIVE_LISTENER_PROFILE_STOP_TIMEOUT_S,
    _FRAME_RESIDUAL_LIVE_TRACE_FIELDS,
    _FRAME_RESIDUAL_RUNTIME_METRIC_TYPES,
    _NATIVE_CONTROL6_DEBUG_INPUT_MAX_FRAMES,
    _NATIVE_CONTROL6_SELECTION_TRACE_FIELDS,
    _NATIVE_CONTROL6_TRACE_ENABLED,
    _acquire_shared_transition_lock_with_timing,
    _attach_native_control6_trace_diagnostic,
    _filter_frame_residual_live_hidden_trace,
    _filter_frame_residual_runtime_metrics,
    _compact_unified_native_terminal_metrics,
    _filter_native_talker_q1_graph_runtime,
    _filter_native_thinker_lock_metrics,
    _filter_turn_local_speech_runtime_metrics,
    _filter_unified_native_event_metrics,
    _format_turn_local_speech_summary,
    _native_control6_debug_input_row,
    _native_control6_event_annotations,
    _capture_native_control6_policy_error,
    _native_control6_policy_evaluation_state,
    _native_control6_policy_error_diagnostics,
    _validate_native_control6_worker_decision_provenance,
    _record_native_control6_policy_invalidation,
    _native_control6_selection_fields,
    _queued_audio_drain_metric_fields,
    _PrivateControlLeakError,
    _TTS_QUALITY_DIAGNOSTIC_CAPTURE,
    _alignment_failure_diagnostics,
    _cancel_tts_segment_emission_waiters,
    _signal_emitted_tts_segment,
    _wait_for_emitted_tts_segment,
    _quiesce_scheduler_for_native_profile,
    _retain_archive_transcript_commitment,
    _send_unified_native_finalizer_event,
    _submit_or_defer_audio_history_archive,
    _unified_native_finalizer_event,
    attach_native_q4_lock_metrics,
    native_listener_cuda_profile_effective_directory,
    native_q4_cuda_profile_host_context,
    scheduler_tick_trace_metadata,
    write_native_listener_cuda_profile_sidecar,
    _write_native_control6_debug_input_sidecar,
)

from duplex_kernel.server_unified import (
    _strict_env_switch,
    _listener_frame_is_voiced,
    _unified_listener_frame_contract,
    _unified_response_owns_shared_audio_queue,
    _shared_listener_response_branch,
    _shared_listener_frame_queue_contract,
    _shared_listener_queue_voice_activity,
    _shared_listener_queue_retention_reason,
    _shared_listener_queue_onset_frame_id,
    _clear_shared_listener_response_branches,
    _attach_shared_listener_response_branches,
    _validate_private_transition_chunk,
    _configured_audio_response_protocol,
    _require_unified_stable_audio_protocol,
    _tokenize_template_text,
    _unified_audio_template_ids,
    _unified_turn_end_token_id,
    _validate_unified_audio_turn_end_suffix,
    _build_unified_audio_response_suffix,
    _prewarm_unified_v1_transition,
    _unified_v1_transition_prewarm_key,
    _prewarm_unified_v1_transition_after_import,
    _build_unified_backchannel_response_suffix,
    _build_unified_next_user_header,
    _unified_end_of_turn_token_id,
    _finalize_unified_native_cache_branch,
    _consume_unified_native_text_append_result,
    _run_unified_native_thinker,
)
from duplex_kernel.server_unified_cache import (
    _commit_deferred_acoustic_history_archive,
    _deferred_acoustic_history_archive_status,
    _prepare_deferred_acoustic_history_finalization,
)

from duplex_kernel.server_response import (
    _align_generated_response_hidden_with_diagnostics,
    _append_native_action_token,
    _backchannel_budget_without_eos,
    _claim_audio_response_v2_action_probe,
    _diagnostic_talker_rng_scope,
    _direct_audio_prompt_history,
    _empty_controller_say_think_from_eos,
    _is_v2_no_speech_result,
    _message_text,
    _prepare_direct_audio_processor_inputs,
    _publish_backchannel_tokens_after_eos,
    _qwen_text_token_surfaces,
    _record_audio_response_v2_private_failure,
    _resolve_request_voice_conditioning,
    run_response as _server_response_run_response,
)
run_response = _server_response_run_response

from duplex_kernel.server_speech_ack import (
    _ClosedTTSSegment,
    _validate_unified_say_packet_frontier,
    _publish_thinker_metrics_event,
    _record_server_ack_lock_wait_ms,
    _record_unified_native_profile_timestamp,
    _wait_for_qwen_dynamic_prefill_model_entry,
    _mark_unified_say_packet_committed,
    _feed_unified_v1_say_packet,
    _publish_acknowledged_speech_prefix,
    _client_consumed_v1_token_packets,
    _record_client_consumed_pcm,
    _canonical_client_consumed_pcm_ack,
    _client_consumed_pcm_ack_duplicate_status,
    _commit_client_consumed_segment,
    _record_emitted_pcm_packet,
    _complete_client_pcm_segment,
    _require_closed_tts_codec_eos,
    _retire_response_speech_ack_streams,
    _turn_local_vocalized_prefix_snapshot,
    _wait_for_listener_backlog_admission,
    _ack_emitted_audio_packet,
    _response_speech_ack_lock,
    _register_response_speech_ack_streams,
    _response_speech_ack_resources,
    _make_finish_audio_segment_callback,
    _retire_response_speech_ack_streams,
    _ack_response_speech_packet,
)


from duplex_kernel.server_audio_snapshot import (
    RESPONSE_QUEUE_BACKPRESSURE_RETRY_S,
    RESPONSE_QUEUE_BACKPRESSURE_TIMEOUT_S,
    DirectAudioSnapshotError,
    ResponseOutputQueueFull,
    _apply_native_control6_pending_pcm_yield_gate,
    _cancel_response_branch_playback,
    _controller_guided_say_generation_options,
    _direct_audio_snapshot_error,
    _direct_audio_snapshot_event_fields,
    _manual_commit_speak_owner_for_span,
    _module_inference_freeze_status,
    _native_control6_decision_has_valid_input,
    _native_control6_endpoint_bypass_enabled,
    _native_control6_listener_output_is_fresh_for_frame,
    _native_control6_listener_source_frame_id,
    _native_control6_policy_hidden_for_tick,
    _native_control6_speaker_active,
    _normalize_chat_modalities,
    _pending_pcm_response_branches,
    _put_response_queue_bounded,
    _record_turn_local_vocalized_prefix,
    _realtime_inference_failure_done_event,
    _require_direct_audio_context,
    _resolve_p7_session_voice,
    _response_admission_allowed_locked,
    _response_branch_has_pending_pcm,
    _response_branch_has_pending_pcm_locked,
    _response_projector_mode,
    _response_uses_direct_audio_candidate,
    _snapshot_direct_policy_audio,
    _snapshot_speaking_branch_context,
    _wait_for_response_playback_drain,
)
from duplex_kernel.server_response_control import (
    _native_control6_worker_fail_closed_context,
    _normalize_response_profile,
    _request_policy_response_revision,
    _response_branch_ack_retained,
    should_emit_final_text_fallback,
)
from duplex_kernel.server_listener_compaction import SessionListenerCompactionAdapter
from duplex_kernel.server_silence_skip import (
    _DEFERRED_NATIVE_SPEAK_SOURCE,
    _advance_pending_speak_on_new_onset,
    _apply_direct_replan_transaction,
    _audio_packet_matches_emission_owner,
    _build_unified_listener_prefix_token_ids,
    _claim_branch_yield_cancel,
    _claim_pending_yielded_v2_pcm_cancel,
    _claim_response_cancel_notification,
    _listener_evidence_role,
    _listener_v9_stream_for_engine,
    _raw_source_audio_quality,
    _refresh_unified_prefix_for_instruction_update,
    _release_pcm_packet_then_ack,
    _scheduler_cuda_stream_context,
    _silence_skip_inputs_are_exact_zero,
    _silence_skip_is_true_idle,
    _silence_skip_query_is_coherent,
    _silence_skip_transition_is_still_idle,
    _unified_instruction_refresh_ineligible_reason,
    _user_turn_is_unanswered,
    _wav_b64,
    _yield_admission_rejection,
)
from duplex_kernel.server_realtime_scheduler import (
    _build_input_audio_consumed_event,
    _evaluate_listener_silence_skip,
    _prepare_listener_tick_reference,
    _publish_listener_conditioning_row,
    _record_listener_commit_timing,
)
from duplex_kernel.server_realtime_policy import (
    _classify_control6_policy_action,
    _control6_transition_rejection,
    _prepare_policy_tick_state_impl,
    _shared_native_policy_context_impl,
)
try:
    from tools.acceptance_telemetry import SAMPLE_INTERVAL_SECONDS
except ImportError:
    from acceptance_telemetry import SAMPLE_INTERVAL_SECONDS
logger = logging.getLogger("duplex_kernel.server")


def _queue_consumed_ack_before_deferred_work(
    scheduler_events: Any,
    consumed_event: dict[str, Any] | None,
    deferred_work: Callable[[], Any] | None = None,
) -> None:
    """Queue a committed listener ACK before running secondary publication work."""
    if consumed_event is not None:
        scheduler_events.put_nowait(consumed_event)
    if callable(deferred_work):
        deferred_work()


@dataclass(frozen=True, slots=True)
class GcPauseSample:
    sequence: int
    generation: int
    start_ns: int
    end_ns: int
    collected: int
    uncollectable: int


_gc_pause_tracker_state = getattr(gc, "_gemma_duplex_pause_tracker", None)
if not isinstance(_gc_pause_tracker_state, dict):
    _gc_pause_tracker_state = {
        "ring": deque(maxlen=256),
        "starts": {},
        "sequence": 0,
        "active_sessions": set(),
        "session_lock": threading.Lock(),
        "gc_was_enabled": None,
    }
    setattr(gc, "_gemma_duplex_pause_tracker", _gc_pause_tracker_state)
gc_pause_ring: deque[GcPauseSample] = _gc_pause_tracker_state["ring"]
active_duplex_sessions: set[str] = _gc_pause_tracker_state["active_sessions"]
_duplex_gc_session_lock: threading.Lock = _gc_pause_tracker_state["session_lock"]


gc_pause_callback = _gc_pause_tracker_state.get("callback")
if not callable(gc_pause_callback):
    def gc_pause_callback(phase: str, info: Mapping[str, Any]) -> None:
        if phase == "start":
            generation = int(info["generation"])
            _gc_pause_tracker_state["starts"][generation] = time.perf_counter_ns()
            return
        if phase != "stop":
            return
        generation = int(info["generation"])
        start_ns = _gc_pause_tracker_state["starts"].pop(generation, None)
        end_ns = time.perf_counter_ns()
        if start_ns is None:
            return
        _gc_pause_tracker_state["sequence"] += 1
        gc_pause_ring.append(GcPauseSample(
            sequence=_gc_pause_tracker_state["sequence"],
            generation=generation,
            start_ns=start_ns,
            end_ns=end_ns,
            collected=int(info.get("collected", 0)),
            uncollectable=int(info.get("uncollectable", 0)),
        ))

    _gc_pause_tracker_state["callback"] = gc_pause_callback
if not any(callback is gc_pause_callback for callback in gc.callbacks):
    gc.callbacks.append(gc_pause_callback)


def gc_pause_snapshot() -> tuple[GcPauseSample, ...]:
    """Return a stable snapshot of the process-level bounded pause ring."""
    return tuple(gc_pause_ring.copy())


def activate_duplex_session_gc(session_id: str) -> None:
    """Disable automatic cyclic GC while any realtime duplex session is live."""
    normalized_id = str(session_id)
    with _duplex_gc_session_lock:
        if normalized_id in active_duplex_sessions:
            return
        if not active_duplex_sessions:
            _gc_pause_tracker_state["gc_was_enabled"] = gc.isenabled()
            if _gc_pause_tracker_state["gc_was_enabled"]:
                gc.disable()
        active_duplex_sessions.add(normalized_id)


def finish_duplex_session_gc(session_id: str) -> bool:
    """Collect synchronously after the last torn-down duplex session closes."""
    normalized_id = str(session_id)
    with _duplex_gc_session_lock:
        if normalized_id not in active_duplex_sessions:
            return False
        active_duplex_sessions.remove(normalized_id)
        if active_duplex_sessions:
            return False
        gc_was_enabled = _gc_pause_tracker_state["gc_was_enabled"]
        _gc_pause_tracker_state["gc_was_enabled"] = None
        try:
            gc.collect()
        finally:
            if gc_was_enabled:
                gc.enable()
        return True


_NATIVE_CONTROL6_DEBUG_INPUT_DIRECTORY = (
    os.environ.get("DUPLEX_NATIVE_CONTROL6_DEBUG_INPUT_DIRECTORY", "").strip() or None
)
_AUDIO_HISTORY_ARCHIVE_DEFER_ENV = "DUPLEX_AUDIO_HISTORY_ARCHIVE_DEFER"
_LISTENER_MICROBATCH_ENABLED_ENV = "DUPLEX_LISTENER_MICROBATCH"
_UNIFIED_KV_ENABLED_ENV = "DUPLEX_UNIFIED_KV"
_SEMANTIC_SHADOW_ENABLED_ENV = "DUPLEX_LISTENER_MICROBATCH_SEMANTIC_SHADOW"


MODEL_ID = "gemma-4-omni-duplex"
REALTIME_SCHEDULER_EVENT_CAPACITY = 16
REALTIME_TEXT_CHUNK_CAPACITY = 32
EMPTY_ASR_SPEAK_COOLDOWN_FRAMES = 25
BACKCHANNEL_MAX_VOICE_AGE_FRAMES = 10
NATIVE_PARTIAL_ASR_ENV = "DUPLEX_NATIVE_PARTIAL_ASR"
DUPLEX_CONTROLLER_GUIDED_SAY_ENV = CONTROLLER_GUIDED_SAY_COMPAT_ENV


_AUDIO_RESPONSE_V2_FAILURE_TRACE_RECORDS: deque[dict[str, Any]] = deque(
    maxlen=AUDIO_RESPONSE_V2_FAILURE_TRACE_MAX_RECORDS
)


def _legacy_audio_chunk_conditioning_horizon(
    next_frame_id: int,
    pcm_sample_count: int,
) -> int:
    """Use the final frame in a newly produced legacy PCM chunk as its horizon."""
    frame_id = int(next_frame_id)
    sample_count = int(pcm_sample_count)
    if frame_id < 0:
        raise ValueError("next_frame_id non può essere negativo")
    if sample_count <= 0 or sample_count % AUDIO_FRAME_SAMPLES:
        raise ValueError("legacy PCM chunk non allineato ai frame da 80 ms")
    return frame_id + sample_count // AUDIO_FRAME_SAMPLES - 1


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
    """Drop response-owned output after cancellation or a generation fence."""
    with session._generation_lock:
        generation_current = session.is_generation_current(token)
        if generation_current and not output_cancel.is_set():
            return False
        output_cancel.set()
        reason = cancel_reason.get("value") or getattr(session, "stop_reason", None)
        cancel_reason["value"] = str(reason or "generation_fenced")
        if playout is not None:
            playout.cancel()
        _drain_queue(audio_queue)
        _drain_queue(text_queue)
    return True


async def _begin_private_backchannel_decode(
    worker: Any,
    session_id: str,
    *,
    can_continue: Callable[[], bool],
    transition_lock: Any,
    retry_timeout_s: float = 2.0,
) -> Any:
    """Create the CAS-fenced private fork before taking the shared transition lock."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + float(retry_timeout_s)
    private_handle = None
    lock_acquired = False
    try:
        while private_handle is None:
            if not can_continue():
                raise RuntimeError("private backchannel fork lost the response generation")
            listener_snapshot = await asyncio.wrap_future(worker.session_state(session_id))
            try:
                private_handle = await asyncio.wrap_future(
                    worker.begin_ephemeral_decode(
                        session_id,
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
                    or loop.time() >= deadline
                ):
                    raise
                await asyncio.sleep(0.002)

        while not transition_lock.acquire(blocking=False):
            if not can_continue():
                raise RuntimeError("private backchannel fork lost the response generation")
            if loop.time() >= deadline:
                raise RuntimeError("unified transition lock wait exceeded retry timeout")
            await asyncio.sleep(0.002)
        lock_acquired = True
        return private_handle
    except BaseException:
        if private_handle is not None and not lock_acquired:
            end_ephemeral_decode = getattr(worker, "end_ephemeral_decode", None)
            if callable(end_ephemeral_decode):
                try:
                    await asyncio.wrap_future(end_ephemeral_decode(private_handle))
                except BaseException:
                    logger.exception("private backchannel fork cleanup failed after transition wait")
        raise


def _response_completion_metrics_event(
    response_id: str,
    runtime_metrics: Mapping[str, Any],
    *,
    cancellation_reason: str,
) -> dict[str, Any]:
    """Expose final cancellation diagnostics without reviving stale output."""
    metrics = dict(runtime_metrics)
    completion = metrics.get("response_completion")
    if isinstance(completion, Mapping):
        completion = dict(completion)
        completion["finish_reason"] = "cancelled"
        completion["cancellation_reason"] = str(cancellation_reason)
        metrics["response_completion"] = completion
    return {
        "type": "response.runtime_metrics",
        "response_id": str(response_id),
        "finish_reason": "cancelled",
        "runtime_metrics": metrics,
    }


async def _send_response_completion_metrics(
    send_ws: Callable[..., Any],
    response_id: str,
    runtime_metrics: Mapping[str, Any],
    *,
    cancellation_reason: str,
) -> bool:
    """Send a diagnostic-only completion record without a stale generation token."""
    event = _response_completion_metrics_event(
        response_id,
        runtime_metrics,
        cancellation_reason=cancellation_reason,
    )
    return bool(await send_ws(event))


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
    output_cancel: threading.Event | None = None,
    cancel_reason: dict[str, str | None] | None = None,
) -> dict[str, int | bool]:
    """Apply the action-specific stop contract at the speaker boundary.

    ``stop_synthesis`` and ``flush_audio_buffer`` remain separate gate facts.
    YIELD stops the producer and drops server-buffered response output at the
    accepted boundary; REPLAN/HARD_STOP also discard pending old-generation
    output. The production scheduler passes the typed ``GateEvent`` returned
    by ``SpeakerGate.apply``.
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
    if action == ControlAction.YIELD:
        generation_lock = getattr(session, "_generation_lock", None)
        with generation_lock if generation_lock is not None else nullcontext():
            if output_cancel is not None:
                output_cancel.set()
            if cancel_reason is not None:
                cancel_reason["value"] = action.value.lower()
            generation_id = event.generation_id
            speaker_segment_id = event.speaker_segment_id
            if playout is not None:
                if (
                    generation_id is not None
                    and speaker_segment_id
                    and hasattr(playout, "cancel_segment")
                ):
                    playout.cancel_segment(
                        generation_id=generation_id,
                        speaker_segment_id=speaker_segment_id,
                    )
                elif hasattr(playout, "cancel"):
                    playout.cancel()
            if packetizer is not None:
                if (
                    generation_id is not None
                    and speaker_segment_id
                    and hasattr(packetizer, "cancel_segment")
                ):
                    packetizer.cancel_segment(
                        generation_id=generation_id,
                        speaker_segment_id=speaker_segment_id,
                    )
                elif hasattr(packetizer, "cancel_all"):
                    packetizer.cancel_all()
            drained += _drain_queue(audio_queue)
            drained += _drain_queue(text_queue)
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


async def _finalize_after_playback_drain(
    branch: dict[str, Any],
    *,
    wait_for_drain: Callable[[dict[str, Any]], Any],
    finalize: Callable[..., Any],
    record_drain_failure: Callable[[Exception, str], Any],
) -> Exception | None:
    """Finalize normally after ACK drain, or recover through ACK-prefix rollback."""
    try:
        await wait_for_drain(branch)
    except Exception as drain_error:
        failure = f"{type(drain_error).__name__}: {drain_error}"[:240]
        branch["playback_drain_failure"] = failure
        await record_drain_failure(drain_error, failure)
        # A timeout never turns unacknowledged generated tokens into history.
        # The native finalizer rebuilds only complete acknowledged segments.
        await finalize(history_committed=False)
        return drain_error
    branch["playback_drained"] = True
    branch["closing"] = True
    await finalize(
        history_committed=bool(branch.get("history_committed"))
        and branch.get("playback_cancelled") is not True
    )
    return None


class TerminalResponseArchive:
    """Thread-safe FIFO archive bounded to lightweight finalized response records."""

    _SCALAR_FIELDS = (
        "generation_id",
        "speaker_segment_id",
        "finalized",
        "ended_response",
        "finalizer_task_done",
        "finalizer_task_status",
        "playback_cancelled",
        "playback_drained",
        "playback_drain_failure",
        "playback_pending",
        "client_consumed_pcm_sequence",
        "spoken_prefix_exclusive_cursor",
        "kv_cache_position",
        "kv_cache_source",
        "finalized_at_utc",
    )
    _ALLOCATOR_FIELDS = (
        "status",
        "device",
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    _SEGMENT_FIELDS = (
        "speaker_segment_id",
        "start_sample",
        "end_sample",
        "emitted_end_sample",
        "done_emitted",
    )

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = max(1, int(capacity))
        self._lock = threading.RLock()
        self._records: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self.eviction_count: int = 0

    @staticmethod
    def _scalar(value: Any) -> Any:
        return value if value is None or type(value) in (str, int, float, bool) else None

    @classmethod
    def _lightweight_record(
        cls,
        session_id: str,
        response_id: str,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        record = {
            "session_id": str(session_id),
            "response_id": str(response_id),
        }
        record.update({
            key: cls._scalar(data.get(key))
            for key in cls._SCALAR_FIELDS
            if key in data
        })
        consumed_segments = data.get("client_consumed_segments")
        if isinstance(consumed_segments, (list, tuple)):
            record["client_consumed_segments"] = [
                cls._scalar(value)
                for value in consumed_segments
                if value is None or type(value) in (str, int, float, bool)
            ]
        playback_segments = data.get("playback_segments")
        if isinstance(playback_segments, (list, tuple)):
            record["playback_segments"] = [
                {
                    key: cls._scalar(segment.get(key))
                    for key in cls._SEGMENT_FIELDS
                    if key in segment
                }
                for segment in playback_segments
                if isinstance(segment, Mapping)
            ]
        allocator = data.get("allocator")
        if isinstance(allocator, Mapping):
            record["allocator"] = {
                key: cls._scalar(allocator.get(key))
                for key in cls._ALLOCATOR_FIELDS
                if key in allocator
            }
        return record

    def record_finalized(
        self,
        session_id: str,
        response_id: str,
        data: dict[str, Any],
    ) -> None:
        session_key = str(session_id)
        response_key = str(response_id)
        key = (session_key, response_key)
        record = self._lightweight_record(session_key, response_key, data)
        with self._lock:
            self._records[key] = record
            while len(self._records) > self.capacity:
                self._records.popitem(last=False)
                self.eviction_count += 1

    def get_for_session(self, session_id: str) -> list[dict[str, Any]]:
        session_key = str(session_id)
        with self._lock:
            return [
                copy.deepcopy(record)
                for (record_session_id, _), record in self._records.items()
                if record_session_id == session_key
            ]

    def get_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [copy.deepcopy(record) for record in self._records.values()]


def _terminal_response_allocator_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "status": "unavailable",
        "device": None,
        "allocated_bytes": None,
        "reserved_bytes": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    cuda = getattr(torch, "cuda", None)
    is_initialized = getattr(cuda, "is_initialized", None)
    try:
        if callable(is_initialized) and bool(is_initialized()):
            device = int(cuda.current_device())
            snapshot.update({
                "status": "available",
                "device": f"cuda:{device}",
                "allocated_bytes": int(cuda.memory_allocated(device)),
                "reserved_bytes": int(cuda.memory_reserved(device)),
                "peak_allocated_bytes": int(cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(cuda.max_memory_reserved(device)),
            })
    except Exception as exc:
        snapshot.update({
            "status": "fatal",
            "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
        })
    return snapshot


def _terminal_response_segments(branch: Mapping[str, Any]) -> list[dict[str, Any]]:
    segments = branch.get("playback_segments")
    if isinstance(segments, Mapping):
        rows = []
        for segment_id, segment in segments.items():
            if not isinstance(segment, Mapping):
                continue
            rows.append({
                "speaker_segment_id": str(segment_id),
                "start_sample": segment.get("start_sample"),
                "end_sample": segment.get("end_sample"),
                "emitted_end_sample": segment.get("emitted_end_sample"),
                "done_emitted": bool(segment.get("done_emitted", False)),
            })
        return rows
    if isinstance(segments, (list, tuple)):
        return [dict(segment) for segment in segments if isinstance(segment, Mapping)]
    return []


def _attach_terminal_response_finalizer(
    archive: TerminalResponseArchive,
    session: Any,
    response_id: str,
    branch: dict[str, Any],
    finalizer_task: Any,
) -> None:
    """Publish a scalar snapshot as soon as the operational finalizer task ends."""
    def publish(completed_task: Any) -> None:
        try:
            if completed_task.cancelled():
                task_status = "cancelled"
            else:
                task_status = "failed" if completed_task.exception() is not None else "completed"
        except asyncio.CancelledError:
            task_status = "cancelled"
        except Exception:
            task_status = "failed"

        branch_lock = branch.get("spoken_prefix_lock")
        lock = branch_lock if callable(getattr(branch_lock, "__enter__", None)) else nullcontext()
        try:
            with lock:
                kv_cache_position = getattr(
                    getattr(session, "kv_cache", None), "current_position", None
                )
                record = {
                    "generation_id": branch.get("session_generation_id"),
                    "speaker_segment_id": branch.get("speaker_segment_id"),
                    "finalized": True,
                    "ended_response": True,
                    "finalizer_task_done": True,
                    "finalizer_task_status": task_status,
                    "playback_cancelled": branch.get("playback_cancelled") is True,
                    "playback_drained": bool(
                        branch.get("playback_drained") is True
                        or (
                            task_status == "completed"
                            and branch.get("playback_drain_failure") is None
                        )
                    ),
                    "playback_drain_failure": branch.get("playback_drain_failure"),
                    "playback_pending": False,
                    "client_consumed_pcm_sequence": branch.get(
                        "client_consumed_pcm_sequence"
                    ),
                    "client_consumed_segments": branch.get(
                        "client_consumed_segments", ()
                    ),
                    "spoken_prefix_exclusive_cursor": branch.get(
                        "spoken_prefix_exclusive_cursor"
                    ),
                    "kv_cache_position": kv_cache_position,
                    "kv_cache_source": (
                        "session.kv_cache.current_position"
                        if kv_cache_position is not None else None
                    ),
                    "playback_segments": _terminal_response_segments(branch),
                    "finalized_at_utc": datetime.now(timezone.utc)
                    .isoformat(timespec="milliseconds")
                    .replace("+00:00", "Z"),
                }
            record["allocator"] = _terminal_response_allocator_snapshot()
            archive.record_finalized(
                str(session.session_id), str(response_id), record
            )
        except Exception:
            logger.exception(
                "[realtime %s %s] failed to archive terminal response telemetry",
                str(getattr(session, "session_id", "unknown"))[:8],
                response_id,
            )

    finalizer_task.add_done_callback(publish)


def create_app(engine: Any, session_mgr: SessionManager | None = None) -> FastAPI:
    app = FastAPI(title="Gemma-4-Omni Duplex Kernel", version="0.4.0")
    sessions = session_mgr or SessionManager()
    app.state.terminal_response_archive = TerminalResponseArchive(capacity=64)
    configured_gpu_admission_value = getattr(engine, "gpu_admission_capacity", None)
    if configured_gpu_admission_value is None:
        configured_gpu_admission_value = os.environ.get(
            GPU_ADMISSION_CAPACITY_ENV,
            str(DEFAULT_GPU_ADMISSION_CAPACITY),
        )
        gpu_admission_source = (
            GPU_ADMISSION_CAPACITY_ENV
            if GPU_ADMISSION_CAPACITY_ENV in os.environ else "default"
        )
    else:
        gpu_admission_source = str(
            getattr(engine, "gpu_admission_capacity_source", "launch")
        )
    gpu_admission_limiter = GpuAdmissionLimiter(
        parse_gpu_admission_capacity(configured_gpu_admission_value),
        source=gpu_admission_source,
    )
    configured_talk_pool = getattr(engine, "response_talk_context_pool", None)
    talk_pool_capacity = (
        getattr(
            configured_talk_pool,
            "admission_capacity",
            getattr(configured_talk_pool, "capacity", None),
        )
        if configured_talk_pool is not None else None
    )
    gpu_admission_config_error = None
    if configured_talk_pool is not None and (
        isinstance(talk_pool_capacity, bool)
        or not isinstance(talk_pool_capacity, int)
        or talk_pool_capacity < gpu_admission_limiter.capacity
    ):
        gpu_admission_config_error = "talk_context_pool_below_gpu_admission_capacity"

    def _acquire_gpu_admission(
        kind: str,
    ) -> tuple[GpuAdmissionLease | None, str | None]:
        if gpu_admission_config_error is not None:
            gpu_admission_limiter.record_rejection()
            return None, gpu_admission_config_error
        lease = gpu_admission_limiter.try_acquire(kind)
        return lease, None if lease is not None else "capacity"

    app.state.gpu_admission_limiter = gpu_admission_limiter
    app.add_middleware(
        GpuAdmissionASGIMiddleware,
        acquire=_acquire_gpu_admission,
        inference_paths=("/talk", "/talk_json", "/v1/chat/completions"),
    )
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
    shared_listener_runtime_status: dict[str, Any] = {
        "error": None,
        "shutting_down": False,
    }
    shared_listener_bindings: dict[str, dict[str, Any]] = {}
    shared_listener_compaction_adapter = SessionListenerCompactionAdapter(
        engine,
        unified_kv_enabled=unified_kv_enabled,
    )
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
            queue_vad_context = policy_context
            if unified_kv_enabled and queue_vad_context is None:
                vad_context_provider = binding.get("causal_vad_context_provider")
                if callable(vad_context_provider):
                    queue_vad_context = vad_context_provider(frame, row.tick)
            voice_active = _shared_listener_queue_voice_activity(
                queue_vad_context,
                binding.get("voice_activity_provider"),
            )
            queue_audio_reason = _shared_listener_queue_retention_reason(
                queue_vad_context,
                voice_active,
            )
            queue_audio_onset_frame_id = _shared_listener_queue_onset_frame_id(
                queue_vad_context,
            )
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
            frame_queue_epoch: int | None = None
            frame_queue_reason: str | None = None
            frame_queue_onset_frame_id: int | None = None
            user_input_pending = False
            if unified_kv_enabled:
                with sess._generation_lock:
                    current_response_id = sess.response_id
                    user_input_pending = _user_turn_is_unanswered(
                        sess.pending_user_onset_frame_id,
                        (
                            sess.reserved_user_onset_frame_id
                            if sess.reserved_user_onset_frame_id is not None
                            else sess.last_answered_onset_frame_id
                        ),
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
                    voice_active=voice_active,
                )
                if queue_audio_for_next_turn:
                    frame_queue_reason = queue_audio_reason
                    frame_queue_onset_frame_id = queue_audio_onset_frame_id
                queue_branch = _shared_listener_response_branch(
                    binding, current_response_id,
                )
                branch_epoch = (
                    queue_branch.get("queue_boundary_epoch")
                    if queue_branch is not None else None
                )
                if isinstance(branch_epoch, int) and not isinstance(branch_epoch, bool):
                    frame_queue_epoch = int(branch_epoch)
                if os.environ.get("DUPLEX_NATIVE_CONTROL6_TRACE") == "1":
                    trace_count = int(binding.get("_control6_queue_trace_count", 0))
                    if trace_count < 30:
                        queue_epoch = (
                            queue_branch.get("queue_boundary_epoch")
                            if queue_branch is not None else None
                        )
                        logger.info(
                            "[control6_queue_handoff] callback seq=%d session=%s frame=%d "
                            "response=%s branch_found=%s queue_open=%s queue_epoch=%s private=%s "
                            "logical_q=%d queue_next=%s user_pending=%s",
                            trace_count + 1,
                            row.session_id,
                            int(frame.frame_id),
                            current_response_id,
                            queue_branch is not None,
                            queue_branch.get("queue_open") if queue_branch is not None else None,
                            queue_epoch,
                            queue_branch.get("ephemeral_private") if queue_branch is not None else None,
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
                queue_epoch=frame_queue_epoch,
                queue_audio_reason=frame_queue_reason,
                queue_audio_onset_frame_id=frame_queue_onset_frame_id,
                reference_pcm=None if reference is None else reference.pcm,
                reference_frame_id=None if reference is None else int(reference.frame_id),
                policy_context=policy_context,
            ))
        dispatcher = shared_listener_dispatcher_holder.get("dispatcher")
        cuda_dependencies = (
            dispatcher.active_compute_dependencies()
            if dispatcher is not None else ()
        )
        if cuda_dependencies:
            return worker.submit_tick(frames, cuda_dependencies=cuda_dependencies)
        return worker.submit_tick(frames)

    def _shared_listener_batch(rows: tuple[SharedListenerInput, ...]):
        if not unified_kv_enabled or not rows:
            return _shared_listener_batch_rows(rows)
        locks_by_id: dict[int, tuple[Any, list[Any]]] = {}
        for row in rows:
            binding = shared_listener_bindings.get(row.session_id)
            lock = None if binding is None else binding.get("unified_turn_transition_lock")
            if lock is not None:
                entry = locks_by_id.setdefault(id(lock), (lock, []))
                entry[1].append(row.tick)
        locks = [locks_by_id[key] for key in sorted(locks_by_id)]
        acquired_locks: list[Any] = []
        try:
            # Frame classification and worker submission share the finalizer's
            # queue-close lock, so every frame is enqueued on one side of the
            # close-and-drain command boundary.
            for lock, ticks in locks:
                wait_ms = _acquire_shared_transition_lock_with_timing(lock, ticks[0])
                acquired_locks.append(lock)
                for tick in ticks[1:]:
                    tick.microbatch_unified_transition_lock_wait_ms = wait_ms
            return _shared_listener_batch_rows(rows)
        finally:
            for lock in reversed(acquired_locks):
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
        row.tick.microbatch_worker_queue_wait_ms = getattr(
            output, "worker_queue_wait_ms", None
        )
        row.tick.microbatch_worker_wall_ms = getattr(output, "worker_wall_ms", None)
        if binding.get("unified_v1_transition_prewarm_enabled"):
            prefix_key = _unified_v1_transition_prewarm_key(output)
            if binding.get("unified_v1_transition_prewarm_key") != prefix_key:
                worker = shared_listener_worker_holder.get("worker")
                if worker is None:
                    raise RuntimeError("shared listener worker disappeared before V1 transition prewarm")
                prewarm_started_ns = time.perf_counter_ns()
                prepared_transition = _prewarm_unified_v1_transition_after_import(
                    worker,
                    binding=binding,
                    output=output,
                    session_id=row.session_id,
                    transition_ids=_build_unified_audio_response_suffix(
                        engine, binding["session"]
                    ),
                )
                if prepared_transition is not None:
                    prewarm_ms = max(
                        0, time.perf_counter_ns() - prewarm_started_ns
                    ) / 1_000_000.0
                    record_stage = binding.get("record_scheduler_stage")
                    if callable(record_stage):
                        record_stage("unified_v1_transition_prewarm_ms", prewarm_ms)
                    logger.info(
                        "[unified_transition_prewarm] session=%s logical_q=%d physical_q=%d "
                        "cache_size=%d cache_misses=%d prepare_ms=%.3f native_roundtrip_ms=%.3f "
                        "canonical_cursor=%d private_cursor=%d",
                        row.session_id[:8],
                        int(prepared_transition["logical_q_len"]),
                        int(prepared_transition["physical_q_len"]),
                        int(prepared_transition["cache_size"]),
                        int(prepared_transition["cache_misses"]),
                        float(prepared_transition["prepare_ms"]),
                        float(prepared_transition["native_roundtrip_ms"]),
                        int(prepared_transition["canonical_cache_position"]),
                        int(prepared_transition["private_cache_position"]),
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
            if row.tick.microbatch_service_ms is not None:
                record_stage("shared_listener_batch_service_ms", row.tick.microbatch_service_ms)
            if row.tick.microbatch_compute_ms is not None:
                # Backward-compatible alias; this is dispatcher service wall time.
                record_stage("shared_listener_batch_compute_ms", row.tick.microbatch_compute_ms)
            if row.tick.microbatch_worker_queue_wait_ms is not None:
                record_stage(
                    "shared_listener_worker_queue_wait_ms",
                    row.tick.microbatch_worker_queue_wait_ms,
                )
            if row.tick.microbatch_worker_wall_ms is not None:
                record_stage(
                    "shared_listener_worker_wall_ms",
                    row.tick.microbatch_worker_wall_ms,
                )
            if row.tick.microbatch_unified_transition_lock_wait_ms is not None:
                record_stage(
                    "shared_listener_unified_turn_transition_lock_wait_ms",
                    row.tick.microbatch_unified_transition_lock_wait_ms,
                )
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

    def _shared_listener_runtime_is_healthy() -> bool:
        worker = shared_listener_worker_holder.get("worker")
        dispatcher = shared_listener_dispatcher_holder.get("dispatcher")
        worker_thread = getattr(worker, "_thread", None)
        worker_thread_is_alive = bool(
            worker_thread is not None
            and callable(getattr(worker_thread, "is_alive", None))
            and worker_thread.is_alive()
        )
        return bool(
            worker is not None
            and worker_thread_is_alive
            and getattr(worker, "owner_thread_id", None) is not None
            and getattr(worker, "fatal_error", None) is None
            and dispatcher is not None
            and bool(getattr(dispatcher, "is_running", False))
            and getattr(dispatcher, "fatal_error", None) is None
        )

    def _shared_listener_allocator_counters() -> dict[str, Any]:
        cuda = getattr(torch, "cuda", None)
        initialized = getattr(cuda, "is_initialized", None)
        try:
            if not callable(initialized) or not initialized():
                return {
                    "status": "unavailable",
                    "allocated_bytes": None,
                    "reserved_bytes": None,
                }
            device = int(cuda.current_device())
            return {
                "status": "available",
                "device": device,
                "allocated_bytes": int(cuda.memory_allocated(device)),
                "reserved_bytes": int(cuda.memory_reserved(device)),
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                "allocated_bytes": None,
                "reserved_bytes": None,
            }

    async def _stop_shared_listener_runtime_locked(*, timeout_s: float = 2.0) -> None:
        dispatcher = shared_listener_dispatcher_holder.get("dispatcher")
        worker = shared_listener_worker_holder.get("worker")
        try:
            if dispatcher is not None:
                await asyncio.to_thread(dispatcher.stop, timeout=timeout_s)
        finally:
            try:
                if worker is not None:
                    await asyncio.wrap_future(worker.stop(drain=False))
            finally:
                shared_listener_dispatcher_holder.clear()
                shared_listener_worker_holder.clear()

    async def _prepare_and_request_shared_listener_compaction(
        sess: DuplexSession,
        snapshot: Any,
        binding: dict[str, Any],
    ) -> None:
        """Tokenize a detached snapshot off-loop, then request background rebase."""
        try:
            prepared = await asyncio.to_thread(
                shared_listener_compaction_adapter.publish_snapshot,
                sess,
                snapshot,
            )
            if not prepared or shared_listener_bindings.get(sess.session_id) is not binding:
                return
            worker = shared_listener_worker_holder.get("worker")
            if worker is None:
                return
            future = worker.request_listener_compaction(
                sess.session_id,
                expected_generation_id=int(binding["generation"]),
                reason="confirmed_history_commit",
                history_revision=getattr(snapshot, "history_revision"),
            )
            await asyncio.wrap_future(future)
        except Exception:
            logger.exception(
                "confirmed listener history compaction request failed for %s",
                sess.session_id,
            )

    def _queue_shared_listener_compaction_snapshot(
        sess: DuplexSession,
        snapshot: Any | None = None,
    ) -> None:
        binding = shared_listener_bindings.get(sess.session_id)
        if binding is None:
            return
        try:
            if snapshot is None:
                snapshot = sess.snapshot_history_with_provenance()
            if not shared_listener_compaction_adapter.mark_pending(sess.session_id, snapshot):
                return
            loop = binding.get("event_loop")
            if loop is None or loop.is_closed():
                return

            def start_task() -> None:
                if shared_listener_bindings.get(sess.session_id) is not binding:
                    return
                task = loop.create_task(
                    _prepare_and_request_shared_listener_compaction(
                        sess,
                        snapshot,
                        binding,
                    )
                )
                tasks = binding.setdefault("compaction_tasks", set())
                tasks.add(task)
                task.add_done_callback(tasks.discard)

            loop.call_soon_threadsafe(start_task)
        except Exception:
            logger.exception(
                "could not queue confirmed listener history snapshot for %s",
                sess.session_id,
            )

    def _shared_listener_archive_history_callback(
        sess: DuplexSession,
        snapshot: Any,
    ) -> None:
        _queue_shared_listener_compaction_snapshot(sess, snapshot)

    def _shared_listener_response_history_callback(
        sess: DuplexSession,
        _projection: Any,
    ) -> None:
        _queue_shared_listener_compaction_snapshot(sess)

    async def _ensure_shared_listener_runtime() -> tuple[Any, Any]:
        """Start and retain the service-level listener worker and dispatcher."""
        async with shared_listener_lifecycle_lock:
            if shared_listener_runtime_status.get("shutting_down"):
                raise RuntimeError("shared listener runtime is shutting down")
            if _shared_listener_runtime_is_healthy():
                return (
                    shared_listener_worker_holder["worker"],
                    shared_listener_dispatcher_holder["dispatcher"],
                )
            if shared_listener_bindings:
                raise RuntimeError(
                    "shared listener runtime stopped while sessions remain registered"
                )
            if shared_listener_worker_holder or shared_listener_dispatcher_holder:
                await _stop_shared_listener_runtime_locked()

            try:
                listener = getattr(engine, "listener", None)
                if listener is None:
                    raise RuntimeError(
                        "shared listener runtime requires the loaded engine.listener"
                    )
                backend = getattr(listener, "backend", None)
                if backend is None:
                    raise RuntimeError(
                        "shared listener runtime requires engine.listener.backend"
                    )
                native_control6_policy = (
                    _make_policy("gemma_native_control6")
                    if bool(getattr(engine, "enable_gemma_native_control6_policy", False))
                    else None
                )

                def commit_shared_listener_cache_identity(
                    session_id: str,
                    cache_epoch_id: int,
                    native_semantic_context_revision: int | None,
                    applied_history_revision: str | int | None,
                    commit_owner_metadata: Callable[[], None],
                ) -> None:
                    binding = shared_listener_bindings.get(str(session_id))
                    if binding is None:
                        raise RuntimeError(
                            "listener cache promotion lost its registered session binding"
                        )
                    sess = binding["session"]
                    state = sess.listener_state
                    # This lock order is shared with candidate adoption in A1.
                    # The callback is CPU-only: the copy/event fence has
                    # completed before the worker calls it.
                    with sess._generation_lock:
                        with state._lock:
                            commit_owner_metadata()
                            state.cache_epoch_id = int(cache_epoch_id)
                            state.native_semantic_context_revision = (
                                None
                                if native_semantic_context_revision is None
                                else int(native_semantic_context_revision)
                            )
                            state.applied_history_revision = applied_history_revision
                            if isinstance(applied_history_revision, int) and not isinstance(
                                applied_history_revision, bool
                            ):
                                state.semantic_context_revision = int(applied_history_revision)

                worker = make_listener_microbatch_worker(
                    listener,
                    native_control6_policy=native_control6_policy,
                    semantic_prefix_enabled=_native_semantic_prefix_enabled(),
                    unified_kv_enabled=unified_kv_enabled,
                    gc_pause_snapshot_provider=gc_pause_snapshot,
                    on_cache_identity_commit=commit_shared_listener_cache_identity,
                    compaction_request_builder=(
                        shared_listener_compaction_adapter.build_context
                    ),
                )
                shared_listener_worker_holder.update({"worker": worker, "backend": backend})
                logger.info(
                    "shared listener allocator before worker.start: %s",
                    _shared_listener_allocator_counters(),
                )
                await asyncio.wrap_future(worker.start())
                logger.info(
                    "shared listener allocator after worker.start: %s",
                    _shared_listener_allocator_counters(),
                )
                dispatcher = SharedListenerTickDispatcher(
                    _shared_listener_batch,
                    on_result=_shared_listener_row_result,
                    on_error=_shared_listener_row_error,
                    batch_context=_shared_listener_priority_context,
                )
                shared_listener_dispatcher_holder["dispatcher"] = dispatcher
                dispatcher.start()
                shared_listener_runtime_status["error"] = None
                return worker, dispatcher
            except BaseException as exc:
                shared_listener_runtime_status["error"] = (
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                )
                logger.exception("shared listener runtime initialization failed")
                try:
                    await _stop_shared_listener_runtime_locked()
                except BaseException:
                    logger.exception("failed to clean up partial shared listener runtime")
                raise

    async def _register_shared_listener_session(
        sess: DuplexSession,
        *,
        record_scheduler_stage: Callable[[str, float | None], None],
        pending_speak_provider: Callable[[], dict[str, Any] | None] | None = None,
        policy_context_provider: Callable[[Any, Any], Any | None] | None = None,
        voice_activity_provider: Callable[[], bool] | None = None,
        causal_vad_context_provider: Callable[[Any, Any], Any | None] | None = None,
    ) -> None:
        await _ensure_shared_listener_runtime()
        async with shared_listener_lifecycle_lock:
            if sess.listener is None or sess.listener_state is None:
                raise RuntimeError("shared listener batching requires an initialized listener")
            worker = shared_listener_worker_holder.get("worker")
            dispatcher = shared_listener_dispatcher_holder.get("dispatcher")
            if worker is None or dispatcher is None or not _shared_listener_runtime_is_healthy():
                raise RuntimeError(
                    shared_listener_runtime_status.get("error")
                    or "shared listener runtime is not ready"
                )
            if getattr(sess.listener, "backend", None) is not shared_listener_worker_holder.get("backend"):
                raise RuntimeError("shared listener sessions must use one loaded Gemma backend")
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
            compaction_registered = False
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
                dispatcher.register_session(
                    sess.session_id,
                    sess.audio_ingress,
                    listener_generation_id=generation,
                    state_epoch_id=sess.listener_state.state_epoch_id,
                )
                dispatcher_registered = True
                await asyncio.to_thread(
                    shared_listener_compaction_adapter.register_session,
                    sess,
                    generation_id=generation,
                )
                compaction_registered = True
                sess.audio_ingress.set_on_frame_ready(
                    lambda session_id=sess.session_id, registered_dispatcher=dispatcher: (
                        registered_dispatcher.notify_input_ready(session_id)
                    )
                )
                if sess.audio_ingress.qsize:
                    dispatcher.notify_input_ready(sess.session_id)
            except BaseException:
                sess.audio_ingress.set_on_frame_ready(None)
                if dispatcher_registered:
                    await asyncio.to_thread(
                        dispatcher.unregister_session,
                        sess.session_id,
                        wait_for_inflight=True,
                        timeout=30.0,
                    )
                if worker_registered:
                    await asyncio.wrap_future(worker.close_session(sess.session_id))
                if compaction_registered:
                    shared_listener_compaction_adapter.unregister_session(sess.session_id)
                raise
            shared_listener_bindings[sess.session_id] = {
                "session": sess,
                "generation": generation,
                "batch_cache_position": int(initial_cache_position),
                "unified_turn_transition_lock": threading.Lock(),
                "queue_epoch_state": {"value": 0},
                "references": {},
                "record_scheduler_stage": record_scheduler_stage,
                "pending_speak_provider": pending_speak_provider,
                "policy_context_provider": policy_context_provider,
                "voice_activity_provider": voice_activity_provider,
                "causal_vad_context_provider": causal_vad_context_provider,
                "semantic_request_holder": semantic_request_holder,
                "semantic_prefix_importer": prefix_importer,
                "event_loop": asyncio.get_running_loop(),
                "unified_v1_transition_prewarm_enabled": (
                    unified_kv_enabled
                    and _configured_audio_response_protocol(engine) == CONTROLLER_GUIDED_SAY_V1
                ),
                "unified_v1_transition_prewarm_key": None,
                "compaction_tasks": set(),
            }
            sess.set_history_archive_commit_callback(
                _shared_listener_archive_history_callback
            )
            sess.set_response_ledger_history_callback(
                _shared_listener_response_history_callback
            )
            sess.conditioning_trace_runtime_provenance = dict(
                getattr(sess.listener.backend, "_listener_microbatch_runtime_info", {})
            )
            sess.conditioning_trace_runtime_provenance.setdefault("enabled", True)

    async def _unregister_shared_listener_session(sess: DuplexSession) -> None:
        async with shared_listener_lifecycle_lock:
            sess.audio_ingress.set_on_frame_ready(None)
            sess.set_history_archive_commit_callback(None)
            sess.set_response_ledger_history_callback(None)
            shared_listener_compaction_adapter.unregister_session(sess.session_id)
            dispatcher = shared_listener_dispatcher_holder.get("dispatcher")
            worker = shared_listener_worker_holder.get("worker")
            try:
                if dispatcher is not None:
                    await asyncio.to_thread(
                        dispatcher.unregister_session,
                        sess.session_id,
                        wait_for_inflight=True,
                        timeout=30.0,
                    )
            finally:
                binding = shared_listener_bindings.pop(sess.session_id, None)
                if binding is not None:
                    for task in tuple(binding.get("compaction_tasks", ())):
                        task.cancel()
                if worker is not None:
                    try:
                        await asyncio.wrap_future(worker.close_session(sess.session_id))
                    except BaseException:
                        logger.exception(
                            "shared listener slot cleanup failed for %s", sess.session_id
                        )

    @contextmanager
    def _inference_execution_scope(pooled: bool):
        if pooled:
            with inference_execution_gate.pooled():
                yield
        else:
            with inference_execution_gate.legacy():
                with inference_lock:
                    yield

    def _submit_admitted_inference(
        loop: asyncio.AbstractEventLoop,
        frontend_lease: GpuAdmissionLease | None,
        function: Callable[[], Any],
    ) -> asyncio.Future[Any]:
        """Keep a producer reference until its executor thread actually exits."""
        producer_lease = (
            None if frontend_lease is None else frontend_lease.retain()
        )
        lifecycle_lock = threading.Lock()
        producer_state = {"started": False, "cancelled_before_start": False}

        def run_producer() -> Any:
            with lifecycle_lock:
                if producer_state["cancelled_before_start"]:
                    return None
                producer_state["started"] = True
            try:
                return function()
            finally:
                if producer_lease is not None:
                    producer_lease.release()

        try:
            future = loop.run_in_executor(None, run_producer)
        except BaseException:
            if producer_lease is not None:
                producer_lease.release()
            raise

        if producer_lease is not None:
            def release_if_cancelled_before_start(done: asyncio.Future[Any]) -> None:
                if done.cancelled():
                    with lifecycle_lock:
                        if not producer_state["started"]:
                            producer_state["cancelled_before_start"] = True
                            should_release = True
                        else:
                            should_release = False
                    if should_release:
                        producer_lease.release()
                    return
                if not done.cancelled():
                    try:
                        done.exception()
                    except BaseException:
                        pass

            future.add_done_callback(release_if_cancelled_before_start)
        return future

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
                minimum_transition_confidence=float(os.environ.get("GEMMA_CONTROL6_MIN_CONFIDENCE", 0.55)),
                early_speak_confidence_threshold=float(os.environ.get("GEMMA_CONTROL6_EARLY_SPEAK_THRESHOLD", 0.80)),
                high_confidence_single_tick_threshold=float(os.environ.get("GEMMA_CONTROL6_HIGH_CONF_THRESHOLD", 0.85)),
                speech_rms_floor=float(os.environ.get("GEMMA_CONTROL6_SPEECH_RMS_FLOOR", 0.002)),
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
        return _server_response_run_response(
            engine,
            sess,
            messages,
            request,
            action=action,
            history_is_committed=history_is_committed,
            unified_kv_enabled=unified_kv_enabled,
        )

    def sse_event(payload: dict[str, Any]) -> str:
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    async def _legacy_talk_from_wav(
        wav_bytes: bytes,
        *,
        admission_lease: GpuAdmissionLease | None = None,
    ) -> np.ndarray:
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
        deferred_session_close = False
        try:
            sess.voice = DEFAULT_VOICE
            inference_future = _submit_admitted_inference(
                asyncio.get_running_loop(),
                admission_lease,
                lambda: _run_legacy_audio_response(sess, request),
            )
            try:
                result = await asyncio.shield(inference_future)
            except asyncio.CancelledError:
                if ephemeral:
                    deferred_session_close = True
                    def close_session_after_inference(_done: asyncio.Future[Any]) -> None:
                        try:
                            _done.exception()
                        except BaseException:
                            pass
                        sessions.close(sess.session_id)

                    inference_future.add_done_callback(close_session_after_inference)
                raise
            wav = result.get("wav") if isinstance(result, dict) else None
            if wav is None or np.asarray(wav).size == 0:
                raise RuntimeError("engine.talk non ha prodotto audio")
            return np.asarray(wav, dtype=np.float32).reshape(-1)
        finally:
            if ephemeral and not deferred_session_close:
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
    async def talk_legacy(request: Request, file: UploadFile = File(...)) -> Response:
        """Accept a user WAV and return the conditioned 24 kHz agent WAV."""
        raw = await file.read()
        try:
            wav = await _legacy_talk_from_wav(
                raw,
                admission_lease=getattr(request.state, "gpu_admission_lease", None),
            )
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
    async def talk_json_legacy(payload: dict[str, Any], http_request: Request) -> Any:
        """Accept ``wav_b64`` and return a 24 kHz WAV encoded as base64."""
        encoded = payload.get("wav_b64")
        if not isinstance(encoded, str) or not encoded:
            return JSONResponse({"error": "wav_b64 deve essere una stringa base64"}, status_code=422)
        try:
            raw = base64.b64decode(encoded, validate=True)
            wav = await _legacy_talk_from_wav(
                raw,
                admission_lease=getattr(http_request.state, "gpu_admission_lease", None),
            )
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
        shared_listener_runtime_ready = bool(
            shared_listener_microbatch_enabled
            and shared_listener_runtime_info.get("enabled", False)
            and _shared_listener_runtime_is_healthy()
        )
        shared_listener_runtime_error = shared_listener_runtime_status.get("error")
        if shared_listener_microbatch_enabled and not shared_listener_runtime_ready:
            if shared_listener_runtime_error is None:
                runtime_fault = (
                    getattr(shared_listener_worker, "fatal_error", None)
                    or getattr(
                        shared_listener_dispatcher_holder.get("dispatcher"),
                        "fatal_error",
                        None,
                    )
                )
                shared_listener_runtime_error = (
                    f"{type(runtime_fault).__name__}: {str(runtime_fault)[:300]}"
                    if runtime_fault is not None
                    else "shared listener worker and dispatcher are not running"
                )
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
        frame_residual_enabled = bool(
            getattr(engine, "frame_residual_adapter_enabled", False)
        )
        learned_scale = None
        learned_scale_status = "disabled" if not frame_residual_enabled else "unobserved"
        frame_residual_status = "disabled" if not frame_residual_enabled else "unobserved"
        if frame_residual_enabled and frame_residual_adapter is None:
            frame_residual_status = "degraded"
        elif frame_residual_enabled and frame_residual_adapter is not None:
            # The adapter's ``scale`` property evaluates a CUDA tensor during
            # normal inference. Health is a CPU-only endpoint, so inspect only
            # its backing parameter when that parameter is already on CPU.
            parameters = getattr(frame_residual_adapter, "_parameters", None)
            scale_logit = (
                parameters.get("scale_logit")
                if isinstance(parameters, dict)
                else None
            )
            cached_scale = None
            scale_attributes = getattr(frame_residual_adapter, "__dict__", None)
            if isinstance(scale_attributes, dict):
                cached_scale = scale_attributes.get("scale")
            try:
                scale_logit_device = getattr(
                    getattr(scale_logit, "device", None), "type", None
                )
                if scale_logit is not None and scale_logit_device != "cpu":
                    learned_scale_status = "not_sampled_cpu_only"
                elif isinstance(scale_logit, torch.Tensor):
                    if scale_logit.device.type == "cpu":
                        max_scale = float(getattr(frame_residual_adapter, "max_scale"))
                        learned_scale = float(
                            (torch.sigmoid(scale_logit.detach().float()) * max_scale).item()
                        )
                        learned_scale_status = (
                            "observed_cpu" if np.isfinite(learned_scale) else "invalid_cpu"
                        )
                    else:
                        learned_scale_status = "not_sampled_cpu_only"
                elif isinstance(cached_scale, torch.Tensor):
                    if cached_scale.device.type == "cpu":
                        learned_scale = float(cached_scale.detach().float().item())
                        learned_scale_status = (
                            "observed_cpu" if np.isfinite(learned_scale) else "invalid_cpu"
                        )
                    else:
                        learned_scale_status = "not_sampled_cpu_only"
                elif isinstance(cached_scale, (int, float)) and not isinstance(cached_scale, bool):
                    learned_scale = float(cached_scale)
                    learned_scale_status = (
                        "observed_cpu" if np.isfinite(learned_scale) else "invalid_cpu"
                    )
            except (AttributeError, TypeError, ValueError, RuntimeError):
                learned_scale = None
                learned_scale_status = "read_error"
            if learned_scale_status == "invalid_cpu":
                learned_scale = None
                frame_residual_status = "fatal"
            elif learned_scale_status == "read_error":
                frame_residual_status = "degraded"
            elif learned_scale_status == "observed_cpu":
                frame_residual_status = "ok"
            else:
                # An intentionally unsampled CUDA resident scale is unknown,
                # not a service failure. Keep its availability in the
                # learned_scale_status field without reading the device.
                frame_residual_status = "unobserved"
        engine_loaded = bool(getattr(engine, "loaded", False))
        health_status = (
            "fatal"
            if frame_residual_status == "fatal"
            else "degraded"
            if (
                not engine_loaded
                or frame_residual_status == "degraded"
                or (shared_listener_microbatch_enabled and not shared_listener_runtime_ready)
                or gpu_admission_config_error is not None
            )
            else "ok"
        )
        context_pool = getattr(engine, "response_talk_context_pool", None)
        configured_context_capacity = getattr(engine, "controller_guided_talk_contexts", 1)
        if (
            isinstance(configured_context_capacity, bool)
            or not isinstance(configured_context_capacity, int)
            or configured_context_capacity < 0
        ):
            configured_context_capacity = 1
        runtime_native_binding = next((
            sys.modules[name]
            for name in (
                "duplex_kernel.native_qwen_talker",
                "duplex_kernel.native_listener_microbatch",
                "duplex_kernel.native_control6_runtime",
            )
            if name in sys.modules
        ), None)
        runtime_sources = collect_runtime_source_diagnostics({
            "server": sys.modules.get(__name__),
            "listener": sys.modules.get("duplex_kernel.listener"),
            "frontend": sys.modules.get("duplex_kernel.listener_microbatch_serving"),
            "worker": sys.modules.get("duplex_kernel.listener_microbatch_worker"),
            "session": sys.modules.get("duplex_kernel.session"),
            "native_binding": runtime_native_binding,
        })
        pool_capacity = getattr(
            context_pool,
            "admission_capacity",
            getattr(context_pool, "capacity", None),
        ) if context_pool is not None else None
        gpu_capacity = {
            "configured_response_contexts": int(configured_context_capacity),
            "pool_capacity": int(pool_capacity) if isinstance(pool_capacity, int) else None,
            "active_contexts": int(getattr(context_pool, "active_count", 0)) if context_pool is not None else 0,
            **gpu_admission_limiter.snapshot(),
            "talk_context_compatible": gpu_admission_config_error is None,
            "configuration_error": gpu_admission_config_error,
        }
        return {
            "status": health_status,
            "ready": health_status == "ok",
            "engine_loaded": engine_loaded,
            "unified_kv_enabled": unified_kv_enabled,
            "first_input_scheduler_anchor_enabled": FIRST_INPUT_SCHEDULER_ANCHOR_ENABLED,
            "ready_frame_scheduler_dispatch_enabled": READY_FRAME_SCHEDULER_DISPATCH_ENABLED,
            "listener_pair_lock_handoff_setting": os.environ.get(
                LISTENER_PAIR_LOCK_HANDOFF_ENV
            ),
            "listener_pair_lock_handoff_enabled": listener_pair_lock_handoff_enabled,
            "native_listener_cuda_profile_enabled": NATIVE_LISTENER_CUDA_PROFILE_ENABLED,
            "native_control6_fused_readback_setting": os.environ.get(
                NATIVE_CONTROL6_FUSED_READBACK_ENV, "1"
            ),
            "native_control6_fused_readback_enabled": NATIVE_CONTROL6_FUSED_READBACK_ENABLED,
            "listener_model_lock_host_timeline_enabled": listener_host_timeline_enabled,
            "native_listener_cuda_profile_directory": native_listener_cuda_profile_directory,
            "listener_microbatch": {
                "enabled": shared_listener_microbatch_enabled,
                "runtime_ready": shared_listener_runtime_ready,
                "error": shared_listener_runtime_error,
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
                "enabled": frame_residual_enabled,
                "status": frame_residual_status,
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
                "learned_scale_status": learned_scale_status,
                "default_voice": DEFAULT_VOICE,
            },
            "model": MODEL_ID,
            "runtime_sources": runtime_sources,
            "gpu_capacity": gpu_capacity,
            "response_protocol": _configured_audio_response_protocol(engine),
            "direct_audio_candidate": os.environ.get("DUPLEX_DIRECT_AUDIO_CANDIDATE", "0").strip() == "1",
            "audio_history_archive_defer_offline": (
                os.environ.get(_AUDIO_HISTORY_ARCHIVE_DEFER_ENV, "1").strip() == "1"
            ),
            "audio_history_archive_cpu_startup_available": (
                False
            ),
            "audio_history_archive_cpu_realtime_bypassed": True,
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

    @app.get("/debug/runtime-telemetry")
    async def runtime_telemetry(
        session_id: str | None = None,
        response_id: str | None = None,
    ) -> Any:
        """Read host-side allocator counters and current response branch state when opted in."""
        if os.environ.get("DUPLEX_RUNTIME_TELEMETRY") != "1":
            return JSONResponse(
                {"status": "disabled", "reason": "set DUPLEX_RUNTIME_TELEMETRY=1 to enable"},
                status_code=404,
            )

        sampled_at_ns = time.monotonic_ns()
        allocator: dict[str, Any] = {
            "status": "unavailable",
            "device": None,
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "sampled_at_monotonic_ns": sampled_at_ns,
        }
        cuda = getattr(torch, "cuda", None)
        is_initialized = getattr(cuda, "is_initialized", None)
        try:
            if callable(is_initialized) and bool(is_initialized()):
                device = int(cuda.current_device())
                allocator.update({
                    "status": "available",
                    "device": f"cuda:{device}",
                    "allocated_bytes": int(cuda.memory_allocated(device)),
                    "reserved_bytes": int(cuda.memory_reserved(device)),
                    # These are process allocator high-water counters; this
                    # endpoint never resets them or synchronizes a CUDA stream.
                    "peak_allocated_bytes": int(cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(cuda.max_memory_reserved(device)),
                })
        except Exception as exc:
            allocator.update({
                "status": "fatal",
                "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
            })

        session_snapshots: list[dict[str, Any]] = []
        archived_by_session: dict[str, list[dict[str, Any]]] = {}
        for archived_record in app.state.terminal_response_archive.get_all():
            archived_session_id = str(archived_record.get("session_id", ""))
            if session_id is not None and archived_session_id != str(session_id):
                continue
            if response_id is not None and str(archived_record.get("response_id")) != str(response_id):
                continue
            archived_by_session.setdefault(archived_session_id, []).append(archived_record)

        active_session_ids: set[str] = set()
        for session in sessions.all():
            if session_id is not None and str(session.session_id) != str(session_id):
                continue
            active_session_ids.add(str(session.session_id))
            with session._generation_lock:
                current_response_id = session.response_id
                current_generation_id = int(session.session_generation_id)
                current_speaker_segment_id = session.speaker_segment_id
                active_response = bool(session.active_response)
                response_cancel_notified = bool(session.response_cancel_notified)
                session_kv_position = getattr(
                    getattr(session, "kv_cache", None), "current_position", None
                )
            branch_registry = getattr(session, "_playback_ack_branches", None)
            registry_lock = getattr(session, "_playback_ack_branches_lock", nullcontext())
            if callable(getattr(registry_lock, "__enter__", None)):
                with registry_lock:
                    branches = list(branch_registry.items()) if isinstance(branch_registry, dict) else []
            else:
                branches = list(branch_registry.items()) if isinstance(branch_registry, dict) else []
            if response_id is not None:
                branches = [item for item in branches if str(item[0]) == str(response_id)]

            branch_snapshots: list[dict[str, Any]] = []
            for branch_key, branch in branches:
                branch_lock = branch.get("spoken_prefix_lock")
                with branch_lock if callable(getattr(branch_lock, "__enter__", None)) else nullcontext():
                    finalizer_task = branch.get("deferred_playback_finalizer")
                    finalizer_done = bool(
                        callable(getattr(finalizer_task, "done", None))
                        and finalizer_task.done()
                    )
                    if finalizer_task is None:
                        finalizer_task_status = "not_scheduled"
                    elif not finalizer_done:
                        finalizer_task_status = "pending"
                    else:
                        try:
                            finalizer_task_status = (
                                "failed"
                                if finalizer_task.exception() is not None
                                else "completed"
                            )
                        except asyncio.CancelledError:
                            finalizer_task_status = "cancelled"
                    segments = branch.get("playback_segments")
                    segment_ranges = [
                        {
                            "speaker_segment_id": str(segment_id),
                            "start_sample": segment.get("start_sample"),
                            "end_sample": segment.get("end_sample"),
                            "emitted_end_sample": segment.get("emitted_end_sample"),
                            "done_emitted": bool(segment.get("done_emitted", False)),
                        }
                        for segment_id, segment in segments.items()
                        if isinstance(segments, Mapping) and isinstance(segment, Mapping)
                    ] if isinstance(segments, Mapping) else []
                    branch_snapshot = {
                        "response_id": str(branch.get("response_id", branch_key)),
                        "generation_id": branch.get("session_generation_id"),
                        "speaker_segment_id": branch.get("speaker_segment_id"),
                        "matches_current_generation": bool(
                            branch.get("response_id") == current_response_id
                            and branch.get("session_generation_id") == current_generation_id
                        ),
                        "playback_cancelled": bool(branch.get("playback_cancelled", False)),
                        "finalized": branch.get("finalized"),
                        "ended_response": branch.get("ended_response"),
                        "finalizer_task_done": finalizer_done,
                        "finalizer_task_status": finalizer_task_status,
                        "client_consumed_pcm_sequence": branch.get(
                            "client_consumed_pcm_sequence"
                        ),
                        "client_consumed_segments": list(
                            branch.get("client_consumed_segments", ())
                        ),
                        "playback_segments": segment_ranges,
                        "spoken_prefix_exclusive_cursor": branch.get(
                            "spoken_prefix_exclusive_cursor"
                        ),
                        "client_consumed_prefix_exclusive_cursor": branch.get(
                            "client_consumed_prefix_exclusive_cursor"
                        ),
                        "client_consumed_prefix_text": branch.get(
                            "client_consumed_prefix_text", ""
                        ),
                        "kv_cache_position": session_kv_position,
                        "kv_cache_source": "session.kv_cache.current_position",
                    }
                # This helper acquires spoken_prefix_lock itself.
                branch_snapshot["playback_pending"] = bool(
                    _response_branch_has_pending_pcm(branch)
                )
                branch_snapshots.append(branch_snapshot)

            authoritative_kv_position = session_kv_position
            authoritative_kv_source = (
                "session.kv_cache.current_position"
                if session_kv_position is not None else None
            )
            worker = next(
                (branch.get("worker") for _, branch in branches if branch.get("worker") is not None),
                None,
            )
            worker_cache_snapshot = "unavailable"
            if worker is not None:
                try:
                    state = await asyncio.wait_for(
                        asyncio.wrap_future(worker.session_state(session.session_id)),
                        timeout=0.25,
                    )
                    authoritative_kv_position = int(state.cache_position)
                    authoritative_kv_source = "listener_microbatch_worker.session_state.cache_position"
                    worker_cache_snapshot = "available"
                except Exception as exc:
                    worker_cache_snapshot = f"unavailable:{type(exc).__name__}"
            branch_by_response_id = {
                str(branch_snapshot.get("response_id")): branch_snapshot
                for branch_snapshot in branch_snapshots
            }
            for archived_record in archived_by_session.get(str(session.session_id), ()):
                archived_response_id = str(archived_record.get("response_id", ""))
                existing_branch = branch_by_response_id.get(archived_response_id)
                if existing_branch is None:
                    archived_branch = dict(archived_record)
                    branch_snapshots.append(archived_branch)
                    branch_by_response_id[archived_response_id] = archived_branch
                else:
                    existing_branch.update(archived_record)
            if response_id is not None and not branch_snapshots:
                continue
            session_snapshots.append({
                "session_id": str(session.session_id),
                "response_id": current_response_id,
                "generation_id": current_generation_id,
                "speaker_segment_id": current_speaker_segment_id,
                "active_response": active_response,
                "response_cancel_notified": response_cancel_notified,
                "kv_cache_position": authoritative_kv_position,
                "kv_cache_source": authoritative_kv_source,
                "listener_worker_snapshot_status": worker_cache_snapshot,
                "branches": branch_snapshots,
            })
        for archived_session_id, archived_records in archived_by_session.items():
            if archived_session_id in active_session_ids or not archived_records:
                continue
            terminal_branches = [dict(record) for record in archived_records]
            last_branch = terminal_branches[-1]
            session_snapshots.append({
                "session_id": archived_session_id,
                "response_id": None,
                "generation_id": last_branch.get("generation_id"),
                "speaker_segment_id": last_branch.get("speaker_segment_id"),
                "active_response": False,
                "response_cancel_notified": False,
                "kv_cache_position": last_branch.get("kv_cache_position"),
                "kv_cache_source": last_branch.get("kv_cache_source"),
                "listener_worker_snapshot_status": "unavailable",
                "branches": terminal_branches,
            })
        return {
            "status": "ok" if allocator["status"] != "fatal" else "fatal",
            "process_id": os.getpid(),
            "sampled_at_monotonic_ns": sampled_at_ns,
            "sample_interval_ms": int(SAMPLE_INTERVAL_SECONDS * 1000),
            "cuda_allocator": allocator,
            "sessions": session_snapshots,
        }

    @app.get("/", include_in_schema=False)
    async def realtime_ui():
        return FileResponse(
            Path(__file__).resolve().parent.parent / "server" / "static" / "index.html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "duplex_kernel"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict[str, Any], http_request: Request):
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
        http_admission_lease = getattr(
            http_request.state, "gpu_admission_lease", None
        )
        with sess._generation_lock:
            context_messages = (list(sess.history) + messages) if request.get("session_id") else messages

        if request.get("stream"):
            response_id = "chatcmpl_" + uuid.uuid4().hex
            try:
                generation_token = sess.begin_response(response_id)
            except (RuntimeError, ValueError) as exc:
                if ephemeral:
                    sessions.close(sess.session_id)
                return JSONResponse(
                    {"error": {"code": "SESSION_ERROR", "message": str(exc)}},
                    status_code=422,
                )

            cancel_event = threading.Event()
            producer_done = threading.Event()
            event_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=16)
            producer_state: dict[str, Any] = {"result": None, "error": None}
            tools_requested = bool(request.get("tools") or sess.tools)
            modalities = set(request["modalities"])
            prior_stop_check = request.get("_stop_check")
            prior_cancel_event = request.get("_response_cancel_event")
            prior_text_callback = request.get("_on_text_chunk")
            prior_audio_callback = request.get("_on_audio_chunk")
            prior_audio_horizon_callback = request.get("_on_audio_chunk_with_horizon")
            text_lock = threading.Lock()
            audio_lock = threading.Lock()
            streamed_text_parts: list[str] = []
            audio_packetizer = NetworkPacketizer(
                packet_frames=NETWORK_PACKET_FRAMES,
                sample_rate=CODEC_SAMPLE_RATE,
            )
            audio_state = {"next_frame_id": 0, "callback_samples": 0}

            def should_stop() -> bool:
                prior_cancelled = bool(
                    callable(getattr(prior_cancel_event, "is_set", None))
                    and prior_cancel_event.is_set()
                )
                return bool(
                    cancel_event.is_set()
                    or getattr(sess, "stop_requested", False)
                    or prior_cancelled
                    or (callable(prior_stop_check) and prior_stop_check())
                )

            request["_response_id"] = response_id
            request["_generation_token"] = generation_token
            request["_generation_id"] = generation_token.session_generation_id
            request["_response_cancel_event"] = cancel_event
            request["_stop_check"] = should_stop

            def publish(payload: dict[str, Any]) -> bool:
                event = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    **payload,
                }
                if ephemeral:
                    event["session_id"] = sess.session_id
                while not should_stop():
                    try:
                        event_queue.put(event, timeout=0.05)
                        return True
                    except queue.Full:
                        continue
                return False

            def publish_text(chunk: Any) -> None:
                if tools_requested or chunk is None:
                    return
                text = str(chunk)
                if not text:
                    return
                if callable(prior_text_callback):
                    prior_text_callback(text)
                with text_lock:
                    streamed_text_parts.append(text)
                publish({"choices": [{"index": 0, "delta": {"content": text}}]})

            def publish_audio(chunk: Any, last_frame_id: int | None = None) -> None:
                if tools_requested or "audio" not in modalities or chunk is None:
                    return
                pcm = np.asarray(chunk, dtype=np.float32).reshape(-1)
                if pcm.size == 0:
                    return
                if pcm.size % AUDIO_FRAME_SAMPLES:
                    raise ValueError("PCM callback HTTP deve contenere frame audio completi")
                frame_count = int(pcm.size // AUDIO_FRAME_SAMPLES)
                with audio_lock:
                    next_frame_id = int(audio_state["next_frame_id"])
                    first_frame_id = (
                        int(last_frame_id) - frame_count + 1
                        if last_frame_id is not None
                        else next_frame_id
                    )
                    if first_frame_id != next_frame_id:
                        raise ValueError(
                            f"frame audio HTTP atteso {next_frame_id}, ricevuto {first_frame_id}"
                        )
                    audio_state["next_frame_id"] = first_frame_id + frame_count
                    audio_state["callback_samples"] = int(audio_state["callback_samples"]) + int(pcm.size)
                    packets = []
                    for index in range(frame_count):
                        start = index * AUDIO_FRAME_SAMPLES
                        frame = PlayoutFrame(
                            frame_id=first_frame_id + index,
                            generation_id=generation_token.session_generation_id,
                            speaker_segment_id=generation_token.speaker_segment_id,
                            pcm=pcm[start:start + AUDIO_FRAME_SAMPLES],
                        )
                        packets.extend(audio_packetizer.push(frame))
                for packet in packets:
                    publish({"choices": [{"index": 0, "delta": {"audio": {
                        "chunk": _wav_b64(packet.pcm, sr=CODEC_SAMPLE_RATE),
                        "format": "wav",
                        "mime_type": "audio/wav",
                        "packet_id": packet.packet_id,
                        "generation_id": packet.generation_id,
                        "speaker_segment_id": packet.speaker_segment_id,
                        "first_frame_id": packet.first_frame_id,
                        "last_frame_id": packet.last_frame_id,
                        "start_sample": packet.start_sample,
                        "end_sample": packet.end_sample,
                        "sample_rate": CODEC_SAMPLE_RATE,
                    }}}]})

            def on_audio_chunk(chunk: Any, *args: Any, **kwargs: Any) -> None:
                if callable(prior_audio_callback):
                    prior_audio_callback(chunk, *args, **kwargs)
                publish_audio(chunk)

            def on_audio_chunk_with_horizon(chunk: Any, conditioning_frame_id: Any, *args: Any, **kwargs: Any) -> None:
                if callable(prior_audio_horizon_callback):
                    prior_audio_horizon_callback(chunk, conditioning_frame_id, *args, **kwargs)
                elif callable(prior_audio_callback):
                    prior_audio_callback(chunk)
                publish_audio(chunk, int(conditioning_frame_id))

            request["_on_text_chunk"] = publish_text
            # run_response uses _on_audio_chunk to enable its text callback path;
            # the horizon callback carries the original decoded frame IDs.
            request["_on_audio_chunk"] = on_audio_chunk
            request["_on_audio_chunk_with_horizon"] = on_audio_chunk_with_horizon

            def infer_stream() -> None:
                try:
                    with _inference_execution_scope(False):
                        result = run_response(sess, context_messages, request)
                    producer_state["result"] = result
                    if tools_requested:
                        calls = result.get("tool_calls") or []
                        if calls:
                            tool_delta = [
                                {"index": index, **call}
                                for index, call in enumerate(calls)
                            ]
                            publish({"choices": [{
                                "index": 0,
                                "delta": {"tool_calls": tool_delta},
                                "finish_reason": "tool_calls",
                            }]})
                        else:
                            final_text = str(result.get("text") or "")
                            if final_text:
                                publish({"choices": [{
                                    "index": 0,
                                    "delta": {"content": final_text},
                                }]})
                            publish({"choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": result.get("finish_reason") or "stop",
                            }]})
                        return

                    final_text = str(result.get("text") or "")
                    with text_lock:
                        streamed_text = "".join(streamed_text_parts)
                    if final_text and not streamed_text:
                        publish({"choices": [{"index": 0, "delta": {"content": final_text}}]})
                    elif final_text.startswith(streamed_text) and len(final_text) > len(streamed_text):
                        publish({"choices": [{"index": 0, "delta": {
                            "content": final_text[len(streamed_text):]
                        }}]})

                    with audio_lock:
                        callback_samples = int(audio_state["callback_samples"])
                        if callback_samples:
                            packets = audio_packetizer.flush_segment(
                                generation_token.session_generation_id,
                                generation_token.speaker_segment_id,
                            )
                        else:
                            packets = []
                    if callback_samples:
                        for packet in packets:
                            publish({"choices": [{"index": 0, "delta": {"audio": {
                                "chunk": _wav_b64(packet.pcm, sr=CODEC_SAMPLE_RATE),
                                "format": "wav",
                                "mime_type": "audio/wav",
                                "packet_id": packet.packet_id,
                                "generation_id": packet.generation_id,
                                "speaker_segment_id": packet.speaker_segment_id,
                                "first_frame_id": packet.first_frame_id,
                                "last_frame_id": packet.last_frame_id,
                                "start_sample": packet.start_sample,
                                "end_sample": packet.end_sample,
                                "sample_rate": CODEC_SAMPLE_RATE,
                            }}}]})
                    elif "audio" in modalities and result.get("wav") is not None:
                        for packet in iter_audio_packets(
                            result["wav"],
                            generation_id=generation_token.session_generation_id,
                            speaker_segment_id=generation_token.speaker_segment_id,
                            packet_frames=NETWORK_PACKET_FRAMES,
                            sample_rate=CODEC_SAMPLE_RATE,
                        ):
                            publish({"choices": [{"index": 0, "delta": {"audio": {
                                "chunk": _wav_b64(packet.pcm, sr=CODEC_SAMPLE_RATE),
                                "format": "wav",
                                "mime_type": "audio/wav",
                                "packet_id": packet.packet_id,
                                "generation_id": packet.generation_id,
                                "speaker_segment_id": packet.speaker_segment_id,
                                "first_frame_id": packet.first_frame_id,
                                "last_frame_id": packet.last_frame_id,
                                "start_sample": packet.start_sample,
                                "end_sample": packet.end_sample,
                                "sample_rate": CODEC_SAMPLE_RATE,
                            }}}]})
                    publish({"choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": result.get("finish_reason") or "stop",
                    }]})
                except Exception as exc:
                    producer_state["error"] = exc
                finally:
                    try:
                        sess.end_response(generation_token)
                    except Exception:
                        logger.exception("chat completion response cleanup failed")
                    producer_done.set()

            producer_state["future"] = _submit_admitted_inference(
                loop,
                http_admission_lease,
                infer_stream,
            )
            if ephemeral:
                def close_ephemeral_after_cancel_or_error(done: asyncio.Future[Any]) -> None:
                    should_close = bool(
                        cancel_event.is_set() or producer_state["error"] is not None
                    )
                    try:
                        should_close = should_close or done.cancelled() or done.exception() is not None
                    except BaseException:
                        should_close = True
                    if should_close:
                        sessions.close(sess.session_id)

                producer_state["future"].add_done_callback(
                    close_ephemeral_after_cancel_or_error
                )

            async def generate_stream():
                done_sent = False
                try:
                    while True:
                        try:
                            payload = event_queue.get_nowait()
                        except queue.Empty:
                            if producer_done.is_set():
                                break
                            await asyncio.sleep(0.01)
                            continue
                        yield sse_event(payload)

                    error = producer_state.get("error")
                    if error is not None:
                        logger.error(
                            "chat completion stream failed",
                            exc_info=(type(error), error, error.__traceback__),
                        )
                        yield sse_event({"error": {
                            "code": "INFERENCE_ERROR",
                            "message": "inferenza fallita",
                        }})
                        yield "data: [DONE]\n\n"
                        done_sent = True
                    elif not tools_requested:
                        result = producer_state.get("result") or {}
                        if result.get("status") == "no_speech":
                            yield sse_event({"audio_status": "no_speech"})
                        yield "data: [DONE]\n\n"
                        done_sent = True
                    if not done_sent:
                        yield "data: [DONE]\n\n"
                except asyncio.CancelledError:
                    cancel_event.set()
                    raise
                except Exception as exc:
                    cancel_event.set()
                    logger.exception("chat completion SSE send failed")
                    yield sse_event({"error": {
                        "code": "INFERENCE_ERROR",
                        "message": "inferenza fallita",
                    }})
                    if not done_sent:
                        yield "data: [DONE]\n\n"
                finally:
                    if not producer_done.is_set():
                        cancel_event.set()

            class _HTTPChatStreamingResponse(StreamingResponse):
                async def __call__(self, scope, receive, send):
                    try:
                        await super().__call__(scope, receive, send)
                    finally:
                        if not producer_done.is_set():
                            cancel_event.set()

            return _HTTPChatStreamingResponse(generate_stream(), media_type="text/event-stream")

        def infer():
            with _inference_execution_scope(False):
                return run_response(sess, context_messages, request)
        try:
            inference_future = _submit_admitted_inference(
                loop,
                http_admission_lease,
                infer,
            )
            result = await asyncio.shield(inference_future)
        except asyncio.CancelledError:
            if ephemeral:
                def close_session_after_inference(_done: asyncio.Future[Any]) -> None:
                    try:
                        _done.exception()
                    except BaseException:
                        pass
                    sessions.close(sess.session_id)

                inference_future.add_done_callback(close_session_after_inference)
            raise
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
            body = {"id": "chatcmpl_" + uuid.uuid4().hex, "object": "chat.completion", "choices": [{"index": 0, "message": msg, "finish_reason": result.get("finish_reason") or "stop"}]}
            if result.get("status") == "no_speech": body["audio_status"] = "no_speech"
            if ephemeral: body["session_id"] = sess.session_id
            return JSONResponse(body)
        return JSONResponse({"error": {"code": "INFERENCE_ERROR", "message": "invalid streaming state"}}, status_code=500)

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
        gpu_admission_lease, gpu_admission_rejection = _acquire_gpu_admission("realtime")
        if gpu_admission_lease is None:
            unavailable = gpu_admission_rejection not in {None, "capacity"}
            error_code = (
                "GPU_ADMISSION_UNAVAILABLE"
                if unavailable else "GPU_CAPACITY_EXCEEDED"
            )
            try:
                await ws.send_json({
                    "type": "error",
                    "error": {
                        "code": error_code,
                        "message": (
                            "GPU inference capacity is unavailable"
                            if unavailable else "GPU inference capacity is full"
                        ),
                    },
                })
            except Exception:
                pass
            try:
                await ws.close(code=1013, reason=error_code)
            except Exception:
                pass
            return
        try:
            sess = sessions.create_session(
                voice=DEFAULT_VOICE,
                audio_ingress_capacity_frames=ingress_capacity_frames,
            )
        except BaseException:
            gpu_admission_lease.release()
            raise
        try:
            activate_duplex_session_gc(sess.session_id)
        except BaseException:
            try:
                sessions.close(sess.session_id)
            finally:
                gpu_admission_lease.release()
            raise
        native_control6_debug_rows: list[dict[str, Any]] = []
        native_control6_debug_frame_ids: set[int] = set()
        try:
            sess.voice_conditioning = _resolve_p7_session_voice(engine, voice=DEFAULT_VOICE)
        except BaseException:
            try:
                sessions.close(sess.session_id)
            finally:
                try:
                    finish_duplex_session_gc(sess.session_id)
                finally:
                    gpu_admission_lease.release()
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
            try:
                sessions.close(sess.session_id)
            finally:
                try:
                    finish_duplex_session_gc(sess.session_id)
                finally:
                    gpu_admission_lease.release()
            raise
        response_task: asyncio.Task | None = None
        playback_ack_branches: dict[str, dict[str, Any]] = {}
        playback_ack_branches_lock = threading.RLock()
        sess._playback_ack_branches = playback_ack_branches
        sess._playback_ack_branches_lock = playback_ack_branches_lock
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
                "playback_ack_wakeup": lambda ev=ack_event: loop.call_soon_threadsafe(ev.set),
                "playback_cancelled": False,
                "input_audio_watermark_frame_id": None,
            }

        def register_playback_ack_branch(response_id: str, branch: dict[str, Any]) -> None:
            """Keep active/cancel-retained ACK metadata; prune only drained/expired branches."""
            now = time.monotonic()
            with playback_ack_branches_lock:
                playback_ack_branches[str(response_id)] = branch
                snapshot = tuple(playback_ack_branches.items())
            snapshot_by_key = dict(snapshot)
            removable_keys: list[str] = []
            for key, candidate in snapshot:
                if key == str(response_id):
                    continue
                if not _response_branch_ack_retained(
                    candidate,
                    now_monotonic=now,
                ):
                    removable_keys.append(key)
            with playback_ack_branches_lock:
                for key in removable_keys:
                    if key != str(response_id) and playback_ack_branches.get(key) is snapshot_by_key.get(key):
                        playback_ack_branches.pop(key, None)
                while len(playback_ack_branches) > 64:
                    removable = next((key for key in removable_keys if key in playback_ack_branches), None)
                    if removable is None:
                        # Never evict an unexpired branch that may still accept
                        # its final cancellation watermark.
                        break
                    playback_ack_branches.pop(removable, None)
                    removable_keys.remove(removable)

        def listener_admission_wait(
            generation_token: GenerationToken | None = None,
        ) -> ListenerBacklogAdmission:
            """Wait for queued or dispatcher-owned listener work before Talker compute."""
            if not (
                direct_audio_candidate
                and response_protocol in {AUDIO_RESPONSE_PROTOCOL_V2, CONTROLLER_GUIDED_SAY_V1}
            ):
                return ListenerBacklogAdmission.ADMITTED
            def generation_is_current() -> bool:
                if closing or sess.stop_requested:
                    return False
                return generation_token is None or sess.is_generation_current(generation_token)

            if not generation_is_current():
                return ListenerBacklogAdmission.ABORTED
            if not shared_listener_microbatch_enabled:
                return ListenerBacklogAdmission.ADMITTED
            dispatcher = shared_listener_dispatcher_holder.get("dispatcher")
            if dispatcher is None:
                return ListenerBacklogAdmission.ABORTED

            admission = _wait_for_listener_backlog_admission(
                lambda: sess.audio_ingress.qsize,
                generation_is_current,
                callbacks_active=sess.scheduler.callbacks_active,
                wait_for_listener_clear=lambda is_current: dispatcher.wait_for_listener_admission(
                    sess.session_id,
                    sess.audio_ingress,
                    is_current,
                ),
            )
            return admission

        def listener_ingress_publication():
            """Close the queue-publish race against Talker permit acquisition."""
            if not (direct_audio_candidate and shared_listener_microbatch_enabled):
                return nullcontext()
            dispatcher = shared_listener_dispatcher_holder.get("dispatcher")
            if dispatcher is None:
                return nullcontext()
            return dispatcher.ingress_publication(sess.session_id)

        def listener_compute_quantum(
            generation_token: GenerationToken | None = None,
        ):
            """Hold one device quantum permit until its stream event is published."""
            if not (
                direct_audio_candidate
                and response_protocol in {AUDIO_RESPONSE_PROTOCOL_V2, CONTROLLER_GUIDED_SAY_V1}
                and shared_listener_microbatch_enabled
            ):
                return nullcontext(None)

            def generation_is_current() -> bool:
                if closing or sess.stop_requested:
                    return False
                return generation_token is None or sess.is_generation_current(generation_token)

            dispatcher = shared_listener_dispatcher_holder.get("dispatcher")
            if dispatcher is None:
                return nullcontext(False)
            return dispatcher.compute_quantum(generation_is_current)

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
        # Faster-Whisper remains available to offline archive tooling. Realtime
        # requests never load or submit work to its process-shared CPU worker.
        archive_cpu_startup_available = False
        audio_history_archive_defer_offline = (
            os.environ.get(_AUDIO_HISTORY_ARCHIVE_DEFER_ENV, "1").strip() == "1"
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
            measured.update(_filter_unified_native_event_metrics(values))
            if phase == "thinker":
                measured.update(_filter_native_thinker_lock_metrics(values))
            _retain_archive_transcript_commitment(measured, phase=phase, values=values)
            measured.update(_filter_frame_residual_runtime_metrics(values))
            measured.update(_filter_turn_local_speech_runtime_metrics(values))
            if phase == "response_completion":
                completion_fields = {
                    "token_limit_requested", "token_limit", "tokens_generated",
                    "text_terminal_reason", "codec_frame_limit",
                    "codec_frames_produced", "eos_detected",
                    "cancellation_reason", "finish_reason",
                }
                for key in completion_fields:
                    value = values.get(key)
                    if key in {"token_limit_requested", "token_limit", "tokens_generated", "codec_frame_limit", "codec_frames_produced"}:
                        valid = value is None or (
                            isinstance(value, int) and not isinstance(value, bool) and value >= 0
                        )
                    elif key in {"text_terminal_reason", "cancellation_reason", "finish_reason"}:
                        valid = value is None or isinstance(value, str)
                    else:
                        valid = isinstance(value, bool)
                    if key in values and valid:
                        measured[key] = value
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
                    committed = sess.complete_audio_history_archive(
                        archive_id,
                        text,
                        asr_provenance=(
                            {key: result.get(key) for key in (
                                "model", "device", "compute_type", "cpu_threads", "language"
                            )}
                            if isinstance(result, Mapping) else None
                        ),
                    )
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

        def unified_native_terminal_metrics_event(response_id: str) -> dict[str, dict[str, Any]]:
            with response_runtime_metrics_lock:
                phases = response_runtime_metrics.get(response_id, {})
                return _compact_unified_native_terminal_metrics(phases)

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
            first_pcm_trace_callback: Callable[[str, int], None] | None = None,
        ) -> bool:
            """Serialize WS writers and fence response/scheduler payloads."""
            ack_lock_requested_ns = (
                time.monotonic_ns()
                if payload.get("type") == "input_audio_buffer.consumed"
                else None
            )
            async with ws_send_lock:
                if ack_lock_requested_ns is not None:
                    _record_server_ack_lock_wait_ms(
                        payload,
                        lock_requested_ns=ack_lock_requested_ns,
                        lock_acquired_ns=time.monotonic_ns(),
                    )
                if payload.get("type") == "response.done":
                    done_response_id = payload.get("response_id")
                    if isinstance(done_response_id, str) and done_response_id:
                        with playback_ack_branches_lock:
                            done_branch = playback_ack_branches.get(done_response_id)
                        if isinstance(done_branch, dict) and (
                            done_branch.get("playback_cancelled") is True
                            or done_branch.get("yield_cancel_enqueued") is True
                        ):
                            return True
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
                        if (
                            token is not None
                            and sess.stop_reason == ControlAction.YIELD.value
                            and payload.get("type") in {
                                "response.audio.delta",
                                "response.audio.segment_done",
                                "response.output_text.delta",
                            }
                        ):
                            return False
                try:
                    if first_pcm_trace_callback is not None:
                        first_pcm_trace_callback("ws_lock_acquired", time.monotonic_ns())
                        first_pcm_trace_callback("send_json_started", time.monotonic_ns())
                    await asyncio.wait_for(ws.send_json(payload), timeout=0.08)
                except asyncio.TimeoutError:
                    if first_pcm_trace_callback is not None:
                        first_pcm_trace_callback("send_json_timeout", time.monotonic_ns())
                    logger.warning("[realtime %s] WebSocket send timeout", sess.session_id[:8])
                    return False
                if first_pcm_trace_callback is not None:
                    first_pcm_trace_callback("send_json_completed", time.monotonic_ns())
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
            deferred_conditioning_publish: Callable[[], Any] | None = None
            with sess._generation_lock:
                listener_generation = sess.session_generation_id
            reference = _prepare_listener_tick_reference(
                sess, tick, shared_listener_microbatch_enabled
            )
            phase_key = str(getattr(sess.control_state, "value", sess.control_state))
            phase_changed = last_listener_phase_key is not None and phase_key != last_listener_phase_key
            last_listener_phase_key = phase_key
            with replan_endpoint_lock:
                replan_endpoint_pending = bool(replan_endpoint_waiters)
            pending_plan_for_skip = bool(
                pending_speak is not None
                or replan_handoff_pending
                or not pending_response_speak_handoffs.empty()
                or replan_endpoint_pending
            )
            silence_skip = _evaluate_listener_silence_skip(
                sess,
                frame,
                reference,
                speech_tracker,
                phase_changed,
                listener_silence_skip_enabled,
                shared_listener_microbatch_enabled=shared_listener_microbatch_enabled,
                listener_generation=int(listener_generation),
                pending_plan=pending_plan_for_skip,
            )
            silence_skip_candidate = silence_skip.candidate
            query_coherent = silence_skip.query_coherent
            allow_silence_skip = silence_skip.allow_silence_skip
            silence_skip_transition_lock_acquired = silence_skip.transition_lock_acquired
            silence_skip_fallback_reason = silence_skip.fallback_reason
            if silence_skip_candidate and not query_coherent:
                logger.info(
                    "[realtime %s] silence-skip fallback frame=%d reason=%s; run full listener",
                    sess.session_id[:8], int(frame.frame_id), silence_skip_fallback_reason,
                )
            elif silence_skip_candidate and not allow_silence_skip:
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
                            row_worker = shared_listener_worker_holder.get("worker")

                            def publish_conditioning_row(
                                *,
                                target_session=sess,
                                target_row=row_output,
                                target_request=request,
                                target_generation=request_generation,
                                target_worker=row_worker,
                            ) -> None:
                                _publish_listener_conditioning_row(
                                    target_session,
                                    target_row,
                                    target_request,
                                    request_generation_id=target_generation,
                                    worker=target_worker,
                                )

                            deferred_conditioning_publish = publish_conditioning_row
                    finally:
                        _record_listener_commit_timing(tick, commit_timing)
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
                    consumed_event = _build_input_audio_consumed_event(
                        sess,
                        frame,
                        frame.pcm,
                        output,
                        tick=tick,
                        listener_step_wall_ms=listener_step_wall_ms,
                        listener_step_started_ns=listener_step_started_ns,
                        listener_step_ended_ns=listener_step_ended_ns,
                        silence_skip_fallback_reason=silence_skip_fallback_reason,
                    )
                _queue_consumed_ack_before_deferred_work(
                    scheduler_events,
                    consumed_event,
                    deferred_conditioning_publish,
                )
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
            """Prepare one immutable policy tick record and update its closure state."""
            nonlocal pending_speak, peer_audio_event_key, latest_policy_frame_id
            nonlocal partial_last_snapshot_frame
            (
                prepared,
                pending_speak,
                peer_audio_event_key,
                latest_policy_frame_id,
                partial_last_snapshot_frame,
            ) = _prepare_policy_tick_state_impl(
                frame,
                tick,
                sess=sess,
                speech_tracker=speech_tracker,
                policy_cooldown_lock=policy_cooldown_lock,
                response_audio_admission=response_audio_admission,
                policy_causality_lock=policy_causality_lock,
                pending_response_speak_handoffs=pending_response_speak_handoffs,
                begin_transcript_utterance=begin_transcript_utterance,
                pending_speak=pending_speak,
                peer_audio_event_key=peer_audio_event_key,
                latest_policy_frame_id=latest_policy_frame_id,
                partial_last_snapshot_frame=partial_last_snapshot_frame,
            )
            if shared_listener_microbatch_enabled:
                shared_listener_compaction_adapter.record_session_input_frame_classification(
                    sess,
                    frame,
                    bool(prepared["input_frame_voiced"]),
                )
            return prepared

        def _shared_listener_causal_vad_context(frame: Any, tick: Any) -> Any:
            prepared = getattr(tick, "_policy_tick_prepared", None)
            if not isinstance(prepared, dict):
                prepared = _prepare_policy_tick_state(frame, tick)
            onset_frame_id = speech_tracker.run_onset_frame_id
            return SimpleNamespace(
                policy_input=SimpleNamespace(
                    peer_turn_active=bool(speech_tracker.voice_run_active),
                ),
                speech_endpoint=bool(prepared["speech_endpoint"]),
                new_voice_onset=bool(prepared["new_voice_onset"]),
                voice_run_onset_frame_id=(
                    None if onset_frame_id is None else int(onset_frame_id)
                ),
            )

        def _shared_native_policy_context(frame: Any, tick: Any) -> Any | None:
            return _shared_native_policy_context_impl(
                frame,
                tick,
                sess=sess,
                prepare_policy_tick_state=_prepare_policy_tick_state,
                speech_tracker=speech_tracker,
                playback_ack_branches=playback_ack_branches,
                playback_ack_branches_lock=playback_ack_branches_lock,
                shared_listener_bindings=shared_listener_bindings,
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
            endpoint_complete = bool(
                speech_tracker.speech_complete
                and not speech_tracker.voice_run_active
            )
            user_endpoint_onset_frame_id = (
                int(speech_tracker.run_onset_frame_id)
                if endpoint_complete and speech_tracker.run_onset_frame_id is not None
                else None
            )
            user_endpoint_frame_id = (
                int(speech_tracker.endpoint_frame_id)
                if endpoint_complete and speech_tracker.endpoint_frame_id is not None
                else None
            )
            backchannel_available = False
            if is_controller6_policy(policy):
                with sess._generation_lock:
                    pending_onset = sess.pending_user_onset_frame_id
                    last_answered_onset = sess.last_answered_onset_frame_id
                    response_pending_for_policy = bool(sess.policy_response_pending)
                    active_response_for_policy = bool(sess.active_response)
                    control_state_for_policy = sess.control_state
                if isinstance(policy, GemmaNativeControl6Policy):
                    with playback_ack_branches_lock:
                        response_branch = (
                            playback_ack_branches.get(sess.response_id)
                            if sess.response_id is not None
                            else None
                        )
                    branch_closing = bool(
                        isinstance(response_branch, dict)
                        and (
                            response_branch.get("closing")
                            or response_branch.get("playback_drained")
                        )
                    )
                    native_speaker_active_for_policy = _native_control6_speaker_active(
                        active_response=active_response_for_policy,
                        control_state=control_state_for_policy,
                        pending_pcm_playback=bool(_pending_pcm_response_branches(
                            playback_ack_branches,
                            registry_lock=playback_ack_branches_lock,
                        )),
                        closing=branch_closing,
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
                user_turn_onset_frame_id=(
                    int(pending_onset)
                    if user_turn_pending and pending_onset is not None
                    else None
                ),
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
                user_endpoint_onset_frame_id=user_endpoint_onset_frame_id,
                user_endpoint_frame_id=user_endpoint_frame_id,
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
            rejected_policy_diagnostics: dict[str, Any] = {}
            rejected_policy_identity: dict[str, Any] = {}
            worker_native_hidden_valid = False
            worker_native_output = None
            worker_decision_view_unavailable_reason: str | None = None
            worker_fail_closed_context = False
            evaluated_snapshot = None
            expected_state_epoch_id: str | None = None
            expected_listener_state_version: int | None = None
            binding = (
                shared_listener_bindings.get(sess.session_id)
                if shared_listener_microbatch_enabled
                else None
            )
            expected_listener_generation = (
                int(binding.get("generation", -1)) if binding is not None else -1
            )
            provenance = None
            view = None
            view_source_row = None
            source_row = None
            fork_row = None
            selected_health = None

            def capture_native_policy_error(exc: Exception) -> None:
                nonlocal decision, native_policy_error
                nonlocal rejected_policy_diagnostics, rejected_policy_identity
                if native_policy_error is not None:
                    return
                diagnostics = getattr(decision, "diagnostics", None)
                if isinstance(diagnostics, dict):
                    rejected_policy_diagnostics = dict(diagnostics)
                rejected_policy_identity = {
                    "session_id": getattr(decision, "session", None),
                    "generation_id": getattr(decision, "generation_id", None),
                    "frame_id": getattr(decision, "frame_id", None),
                    "listener_frame_id": getattr(decision, "listener_frame_id", None),
                }
                decision, native_policy_error = _capture_native_control6_policy_error(
                    decision, exc
                )

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
                        evaluated_snapshot = getattr(
                            worker_native_output, "state_snapshot", None
                        )
                        if evaluated_snapshot is None:
                            raise RuntimeError(
                                "shared native Control6 output lacks the evaluated-frame snapshot"
                            )
                        evaluated_frame_id = int(evaluated_snapshot.next_frame_id) - 1
                        if evaluated_frame_id != int(policy_input.frame_id):
                            raise RuntimeError(
                                "shared native Control6 snapshot does not identify the evaluated frame"
                            )
                        expected_state_epoch_id = str(evaluated_snapshot.state_epoch_id)
                        expected_listener_state_version = int(evaluated_snapshot.state_version) - 1
                        snapshot_listener_generation = int(evaluated_snapshot.generation_id)
                        if snapshot_listener_generation != expected_listener_generation:
                            raise RuntimeError(
                                "shared native Control6 frame snapshot generation differs from its binding"
                            )
                        worker_decision_view_unavailable_reason = getattr(
                            worker_native_output,
                            "decision_view_unavailable_reason",
                            None,
                        )
                        decision = getattr(worker_native_output, "policy_decision", None)
                        if decision is None:
                            raise RuntimeError(
                                "shared native Control6 worker returned no policy decision"
                            )
                        diagnostics = decision.diagnostics
                        if (
                            isinstance(worker_decision_view_unavailable_reason, str)
                            and worker_decision_view_unavailable_reason
                        ):
                            decision_diagnostics = (
                                dict(diagnostics) if isinstance(diagnostics, dict) else {}
                            )
                            decision_diagnostics[
                                "decision_view_unavailable_reason"
                            ] = worker_decision_view_unavailable_reason
                            decision = replace(
                                decision, diagnostics=decision_diagnostics
                            )
                            diagnostics = decision.diagnostics
                        elif isinstance(diagnostics, dict):
                            candidate_reason = diagnostics.get(
                                "decision_view_unavailable_reason"
                            )
                            if isinstance(candidate_reason, str) and candidate_reason:
                                worker_decision_view_unavailable_reason = candidate_reason
                        with playback_ack_branches_lock:
                            response_branch = (
                                playback_ack_branches.get(sess.response_id)
                                if sess.response_id is not None
                                else None
                            )
                        worker_fail_closed_context = (
                            _native_control6_worker_fail_closed_context(
                                diagnostics=diagnostics,
                                decision=decision,
                                speaker_active=policy_input.speaker_active,
                                active_response=active_response_for_policy,
                                response_branch=response_branch,
                            )
                        )
                        provenance = (
                            diagnostics.get("native_output_provenance")
                            if isinstance(diagnostics, dict)
                            else None
                        )
                        view = getattr(worker_native_output, "decision_view", None)
                        view_source_row = (
                            getattr(view, "source_row", None) if view is not None else None
                        )
                        fork_row = (
                            getattr(view, "fork_row", None) if view is not None else None
                        )
                        _validate_native_control6_worker_decision_provenance(
                            decision,
                            provenance,
                            binding_available=binding is not None,
                            canonical_session_id=sess.session_id,
                            expected_listener_generation_id=expected_listener_generation,
                            expected_turn_generation_id=int(policy_input.generation_id),
                            expected_state_epoch_id=expected_state_epoch_id,
                            expected_frame_id=int(policy_input.frame_id),
                            expected_listener_state_version=expected_listener_state_version,
                        )
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
                            or source_row.state_epoch_id != expected_state_epoch_id
                            or source_row.frame_id != int(policy_input.frame_id)
                            or source_row.listener_state_version
                            != expected_listener_state_version
                        ):
                            raise RuntimeError(
                                "selected native Control6 decision row lacks matching healthy output provenance"
                            )
                        if view is not None:
                            if (
                                view_source_row is None
                                or fork_row is None
                                or view_source_row.session_id != sess.session_id
                                or view_source_row.session_generation != expected_listener_generation
                                or view_source_row.state_epoch_id != expected_state_epoch_id
                                or view_source_row.frame_id != int(policy_input.frame_id)
                                or source_row.session_id != fork_row.session_id
                                or not source_row.session_id.startswith("__q2view__")
                                or source_row.logical_q_len != 2
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
                        worker_native_hidden_valid = bool(
                            not worker_fail_closed_context
                            and not policy_input.native_control_no_input_tick
                        )
                        if worker_fail_closed_context:
                            native_control_hidden = None
                    else:
                        decision = sess.run_turn_policy(
                            policy, policy_revision, policy_input
                        )
                except Exception as exc:
                    if not (
                        isinstance(policy, GemmaNativeControl6Policy)
                        or getattr(policy, "policy_id", None) == "gemma_native_control6"
                    ):
                        raise
                    capture_native_policy_error(exc)
            finally:
                record_scheduler_stage(
                    "policy_wall_ms", (time.perf_counter() - policy_started) * 1000.0
                )
            fresh_native_hidden = False
            if native_policy_error is None:
                try:
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
                                native_output_health_result = native_control6_validate_listener_output_health(
                                    native_output,
                                    session_id=sess.session_id,
                                    generation_id=expected_listener_generation,
                                    state_epoch_id=expected_state_epoch_id,
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
                        and not worker_fail_closed_context
                        and not _native_control6_decision_has_valid_input(
                            fresh_hidden=fresh_native_hidden,
                            no_input_tick=bool(policy_input.native_control_no_input_tick),
                            speaker_active=bool(policy_input.speaker_active),
                            action=getattr(decision, "action", None),
                            legal_actions=legal_actions,
                        )
                    ):
                        capture_native_policy_error(RuntimeError(
                            "native Control6 action lacks a fresh listener hidden for this frame"
                        ))
                except Exception as exc:
                    if not (
                        isinstance(policy, GemmaNativeControl6Policy)
                        or getattr(policy, "policy_id", None) == "gemma_native_control6"
                    ):
                        raise
                    capture_native_policy_error(exc)
            if native_policy_error is not None:
                exc = native_policy_error
                source_row = source_row or getattr(selected_health, "row", None)
                if view is not None:
                    fork_row = fork_row or getattr(view, "fork_row", None)
                    view_source_row = view_source_row or getattr(view, "source_row", None)
                    source_row = source_row or view_source_row
                provenance_for_error = (
                    provenance
                    if isinstance(provenance, dict)
                    else rejected_policy_diagnostics.get("native_output_provenance")
                )
                if not isinstance(provenance_for_error, dict):
                    provenance_for_error = {}

                def first_available(*values: Any) -> Any:
                    return next((value for value in values if value is not None), None)

                worker_listener_generation = getattr(
                    worker_native_output, "generation_id", None
                )
                worker_state_epoch = getattr(worker_native_output, "state_epoch_id", None)
                worker_frame_id = getattr(worker_native_output, "frame_id", None)
                identity = {
                    "session_id": str(sess.session_id),
                    "observed_session_id": first_available(
                        rejected_policy_identity.get("session_id"),
                        getattr(worker_native_output, "session_id", None),
                        provenance_for_error.get("canonical_session_id"),
                    ),
                    "listener_generation_id": {
                        "expected": expected_listener_generation,
                        "observed": first_available(
                            worker_listener_generation,
                            provenance_for_error.get("listener_generation_id"),
                            getattr(source_row, "session_generation", None),
                        ),
                    },
                    "turn_generation_id": {
                        "expected": int(policy_input.generation_id),
                        "observed": first_available(
                            rejected_policy_identity.get("generation_id"),
                            provenance_for_error.get("turn_generation_id"),
                        ),
                    },
                    "state_epoch_id": {
                        "expected": expected_state_epoch_id or getattr(
                            worker_native_output, "state_epoch_id", None
                        ),
                        "observed": first_available(
                            worker_state_epoch,
                            provenance_for_error.get("state_epoch_id"),
                            getattr(source_row, "state_epoch_id", None),
                        ),
                    },
                    "frame_id": {
                        "expected": int(policy_input.frame_id),
                        "observed": first_available(
                            rejected_policy_identity.get("frame_id"),
                            worker_frame_id,
                            provenance_for_error.get("frame_id"),
                        ),
                    },
                    "listener_state_version": {
                        "expected": expected_listener_state_version,
                        "observed": first_available(
                            getattr(source_row, "listener_state_version", None),
                            provenance_for_error.get("listener_state_version"),
                        ),
                    },
                    "cache_position": {
                        "expected": getattr(
                            view_source_row,
                            "cache_position",
                            getattr(source_row, "cache_position", None),
                        ),
                        "observed": first_available(
                            getattr(fork_row, "cache_position", None),
                            getattr(source_row, "cache_position", None),
                        ),
                    },
                }
                invalidation_repeat_count = _record_native_control6_policy_invalidation(
                    session_id=str(sess.session_id),
                    state_epoch_id=str(
                        expected_state_epoch_id
                        or getattr(worker_native_output, "state_epoch_id", "unknown")
                    ),
                    frame_id=int(policy_input.frame_id),
                    reason=str(exc),
                )
                rejection_diagnostics = _native_control6_policy_error_diagnostics(
                    exc,
                    identity=identity,
                    decision_view_unavailable_reason=(
                        worker_decision_view_unavailable_reason
                    ),
                    invalidation_repeat_count=invalidation_repeat_count,
                )
                rejected_decision_diagnostics = {
                    key: rejected_policy_diagnostics[key]
                    for key in (
                        "policy_id",
                        "policy_head_skipped",
                        "native_output_provenance",
                        "decision_view_fallback_reason",
                        "decision_view_unavailable_reason",
                    )
                    if key in rejected_policy_diagnostics
                }
                try:
                    scheduler_events.put_nowait({"type": "turn_policy.action_rejected",
                        "policy_id": "gemma_native_control6", "frame_id": int(policy_input.frame_id),
                        "turn_policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
                        "native_control6_class_match": isinstance(policy, GemmaNativeControl6Policy),
                        "native_control6_step_called": bool(
                            getattr(worker_native_output, "policy_decision", None) is not None
                            or rejected_policy_diagnostics.get("policy_id") == "gemma_native_control6"
                        ),
                        "native_control6_head_scored": False,
                        "policy_evaluation_state": "unavailable",
                        "native_control_hidden_supplied": bool(
                            worker_native_hidden_valid
                            or policy_input.native_control_hidden is not None
                        ),
                        "native_control_hidden_frame_id": native_control_hidden_frame_id,
                        "native_control_no_input_tick": bool(policy_input.native_control_no_input_tick),
                        "legal_actions": list(legal_actions or ()),
                        "user_turn_pending": bool(user_turn_pending),
                        "pending_user_onset_frame_id": pending_onset,
                        "last_answered_onset_frame_id": last_answered_onset,
                        **rejection_diagnostics,
                        "diagnostics": {
                            **rejection_diagnostics,
                            "decision_diagnostics": rejected_decision_diagnostics,
                        }})
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
                and not bool(policy_input.native_control_no_input_tick)
                and not (
                    isinstance(decision.diagnostics, dict)
                    and decision.diagnostics.get("no_input_tick") is True
                )
                and not (
                    isinstance(decision.diagnostics, dict)
                    and (
                        decision.diagnostics.get("policy_head_skipped") is True
                        or decision.diagnostics.get("decision_view_unavailable_reason")
                    )
                )
            )
            if worker_decision_view_unavailable_reason is None and isinstance(
                decision.diagnostics, dict
            ):
                candidate_view_reason = decision.diagnostics.get(
                    "decision_view_unavailable_reason"
                )
                if isinstance(candidate_view_reason, str) and candidate_view_reason:
                    worker_decision_view_unavailable_reason = candidate_view_reason
            policy_evaluation_state = _native_control6_policy_evaluation_state(
                decision=decision,
                head_scored=native_control6_head_scored,
                no_input_tick=bool(policy_input.native_control_no_input_tick),
                decision_view_unavailable_reason=(
                    worker_decision_view_unavailable_reason
                ),
            )
            decision_diagnostics = (
                dict(decision.diagnostics)
                if isinstance(decision.diagnostics, dict)
                else {}
            )
            decision_diagnostics.update({
                "native_control6_head_scored": native_control6_head_scored,
                "policy_evaluation_state": policy_evaluation_state,
            })
            if worker_decision_view_unavailable_reason is not None:
                decision_diagnostics[
                    "decision_view_unavailable_reason"
                ] = worker_decision_view_unavailable_reason
            decision = replace(decision, diagnostics=decision_diagnostics)
            if worker_fail_closed_context:
                with sess._generation_lock:
                    if (
                        policy_revision != sess._policy_revision
                        or policy is not sess.turn_policy
                        or sess.session_generation_id != policy_input.generation_id
                        or not sess.active_response
                    ):
                        sess.stale_session_mutations += 1
                        return
                    fail_closed_response_id = sess.response_id
                try:
                    scheduler_events.put_nowait({
                        "type": "error",
                        "error": {
                            "code": "CONTROL6_CONTEXT_UNAVAILABLE",
                            "message": (
                                "Control6 Q2 context was lost during SPEAK; "
                                "the active response is stopping"
                            ),
                            "details": worker_decision_view_unavailable_reason,
                        },
                        "response_id": fail_closed_response_id,
                        "frame_id": int(policy_input.frame_id),
                    })
                except queue.Full:
                    pass
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
                    "policy_evaluation_state": policy_evaluation_state,
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
                        "decision_view_unavailable_reason",
                        "policy_evaluation_state",
                        "native_control6_head_scored",
                        "calibration_applied",
                        "calibration_margin",
                        "calibration_threshold",
                        "feature_contract_input_dim",
                        "acoustic_phase_features_used",
                        "listener_query_reused",
                        "listener_query_origin_frame_id",
                        "listener_query_age_frames",
                        "gc_pause_ms",
                        "gc_pause_overlapped",
                        "listener_batch_control6_policy_wall_start_ns",
                        "listener_batch_control6_policy_wall_end_ns",
                        "listener_batch_control6_policy_thread_cpu_start_ns",
                        "listener_batch_control6_policy_thread_cpu_end_ns",
                        "native_control6_batch_policy_wall_start_ns",
                        "native_control6_batch_policy_wall_end_ns",
                        "native_control6_batch_policy_thread_cpu_start_ns",
                        "native_control6_batch_policy_thread_cpu_end_ns",
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
                control6_action = _classify_control6_policy_action(
                    policy, decision.action
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
                        has_post_response_voice = (
                            response_audio_admission.has_post_response_voice_locked(
                                speech_tracker
                            )
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
                                    and (
                                        speech_tracker.has_onset_after(watermark)
                                        or (
                                            speech_tracker.last_voiced_frame_id is not None
                                            and speech_tracker.last_voiced_frame_id > watermark
                                        )
                                    )
                                ):
                                    pending_branch_voice = True
                                    break
                            if pending_pcm_branches:
                                emitted_audio_frames = max(1, int(emitted_audio_frames))
                                has_post_watermark_voice = bool(
                                    has_post_watermark_voice or pending_branch_voice
                                )
                        if rejection_reason is None:
                            rejection_reason = _control6_transition_rejection(
                                decision.action,
                                active_for_causal_gate=active_for_causal_gate,
                                emitted_audio_frames=emitted_audio_frames,
                                has_post_watermark_voice=has_post_watermark_voice,
                                has_post_response_voice=has_post_response_voice,
                            )
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
                elif (
                    isinstance(policy, GemmaNativeControl6Policy)
                    and control6_action == ControlAction.SPEAK
                ):
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
                if unified_kv_enabled and gate_decision.action is ControlAction.REPLAN:
                    with playback_ack_branches_lock:
                        revision_branch = playback_ack_branches.get(sess.response_id or "")
                    revision_requested = _request_policy_response_revision(
                        sess, gate_decision, revision_branch,
                    )
                    if revision_requested:
                        pending_speak = None
                        with policy_causality_lock:
                            replan_edge_latched = True
                            replan_non_event_ticks = 0
                            replan_edge_generation_id = int(policy_input.generation_id)
                    publish_decision(
                        accepted=revision_requested,
                        rejection_reason=None if revision_requested else "revision_branch_unavailable",
                    )
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
            if gate.action in {ControlAction.YIELD, ControlAction.HARD_STOP} and event_response_id:
                with playback_ack_branches_lock:
                    cancel_branch = playback_ack_branches.get(event_response_id)
                if isinstance(cancel_branch, dict):
                    coordinator = cancel_branch.get("revision_coordinator")
                    cancel_all = getattr(coordinator, "cancel_all", None)
                    if callable(cancel_all):
                        cancel_all(gate.action.value.lower())
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
                    output_cancel=None if control is None else control.output_cancel,
                    cancel_reason=None if control is None else control.cancel_reason,
                )
                if gate_result.get("accepted") is not True:
                    sess.stale_session_mutations += 1
                    publish_decision(accepted=False, rejection_reason="speaker_gate_application_rejected")
                    return

            if gate.action == ControlAction.YIELD:
                target_branches = []
                target_response_ids: set[str] = set()
                for pending_branch in pending_pcm_branches:
                    if not isinstance(pending_branch, dict):
                        continue
                    pending_response_id = str(pending_branch.get("response_id", ""))
                    if pending_response_id and pending_response_id not in target_response_ids:
                        target_branches.append(pending_branch)
                        target_response_ids.add(pending_response_id)
                if event_response_id:
                    with playback_ack_branches_lock:
                        current_branch = playback_ack_branches.get(event_response_id)
                    if (
                        isinstance(current_branch, dict)
                        and current_branch.get("response_id") == event_response_id
                        and int(current_branch.get("session_generation_id", -1))
                        == int(gate.generation_id)
                        and event_response_id not in target_response_ids
                    ):
                        target_branches.append(current_branch)
                        target_response_ids.add(event_response_id)

                for target_branch in target_branches:
                    branch_response_id = str(target_branch.get("response_id", ""))
                    if not branch_response_id:
                        continue
                    branch_generation_id = int(
                        target_branch.get("session_generation_id", -1)
                    )
                    if _claim_branch_yield_cancel(
                        sess,
                        target_branch,
                        response_id=branch_response_id,
                        generation_id=branch_generation_id,
                        speaker_segment_id=str(
                            target_branch.get("speaker_segment_id", "")
                        ),
                    ):
                        scheduler_events.put_nowait({
                            "type": "response.cancelled",
                            "reason": "yield",
                            "finish_reason": "cancelled",
                            "response_id": branch_response_id,
                            "generation_id": branch_generation_id,
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
                cancellation_reason = gate.action.value.lower()
                event = {
                    "type": "response.cancelled",
                    "reason": cancellation_reason,
                    "finish_reason": "cancelled",
                    "response_id": gate_response_id,
                    "_generation_fence": gate.generation_id,
                }
                if gate_response_id:
                    update_response_runtime_metrics(gate_response_id, "response_completion", {
                        "token_limit_requested": None,
                        "token_limit": None,
                        "tokens_generated": None,
                        "text_terminal_reason": None,
                        "codec_frame_limit": None,
                        "codec_frames_produced": 0,
                        "eos_detected": False,
                        "cancellation_reason": cancellation_reason,
                        "finish_reason": "cancelled",
                    })
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

        scheduler_cuda_stream = None

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
        scheduler_event_task: asyncio.Task | None = None
        try:
            # Allocate this before binding or starting the scheduler, but keep
            # it inside the startup guard so CUDA initialization failures also
            # retire the session and any listener resources created above.
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
            if shared_listener_microbatch_enabled:
                await _register_shared_listener_session(
                    sess,
                    record_scheduler_stage=record_scheduler_stage,
                    pending_speak_provider=lambda: pending_speak,
                    voice_activity_provider=lambda: bool(speech_tracker.voice_run_active),
                    policy_context_provider=(
                        _shared_native_policy_context
                        if isinstance(sess.turn_policy, GemmaNativeControl6Policy)
                        else None
                    ),
                    causal_vad_context_provider=_shared_listener_causal_vad_context,
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
        except BaseException:
            # This setup used to run before the websocket's main try/finally.
            # Retire only this session's resources while preserving a shared
            # worker that still has other registered sessions.
            closing = True
            scheduler_events.close()
            if scheduler_event_task is not None:
                scheduler_event_task.cancel()
                await asyncio.gather(scheduler_event_task, return_exceptions=True)
            if shared_listener_microbatch_enabled:
                if sess.session_id in shared_listener_bindings:
                    try:
                        await _unregister_shared_listener_session(sess)
                    except BaseException:
                        logger.exception("failed to unregister listener after realtime startup failure")
            else:
                try:
                    await asyncio.to_thread(sess.scheduler.stop, timeout=2.0)
                except BaseException:
                    logger.exception("failed to stop scheduler after realtime startup failure")
            if native_partial_worker is not None:
                try:
                    native_partial_worker.close()
                except BaseException:
                    logger.exception("failed to close partial ASR worker after realtime startup failure")
            try:
                sess.invalidate_generation("realtime_startup_failed")
                sessions.close(sess.session_id)
            except BaseException:
                logger.exception("failed to close session after realtime startup failure")
            finally:
                try:
                    finish_duplex_session_gc(sess.session_id)
                finally:
                    gpu_admission_lease.release()
            raise

        async def produce_response(
            committed_audio: np.ndarray,
            *,
            action: ControlAction | None = None,
            intent_origin: dict[str, Any] | None = None,
            automatic_policy_response: bool = False,
            response_profile: str | None = None,
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
            selected_response_profile = _normalize_response_profile(
                sess.response_profile if response_profile is None else response_profile
            )
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
            response_done_at_monotonic_ns: int | None = None
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
            with playback_ack_branches_lock:
                prior_playback_branches = tuple(
                    branch for prior_id, branch in playback_ack_branches.items()
                    if prior_id != response_id
                )
            for prior_branch in prior_playback_branches:
                if _response_branch_has_pending_pcm(prior_branch):
                    prior_branch.setdefault("cancellation_reason", "superseded_by_new_response")
                    _cancel_response_branch_playback(prior_branch)
            token = sess.begin_response(
                response_id,
                preserve_pending_user_turn_audio=(action == ControlAction.BACKCHANNEL),
                preserve_newer_pending_user_turn_audio_after_onset_frame_id=(
                    user_turn_onset_frame_id
                    if action == ControlAction.SPEAK
                    else None
                ),
            )
            response_event_ledger = sess.get_response_event_ledger(response_id)
            playback_branch = {
                "response_id": response_id,
                "session_generation_id": int(token.session_generation_id),
                "speaker_segment_id": str(token.speaker_segment_id),
                "speech_ack_retired": False,
                "playback_segments": {},
                "segment_token_packets": [],
                "spoken_prefix_token_ids": (),
                "spoken_prefix_exclusive_cursor": 0,
                "spoken_prefix_source_seq": -1,
                "spoken_prefix_text": "",
                "client_consumed_pcm_sequence": -1,
                "client_consumed_segments": (),
                "response_event_ledger": response_event_ledger,
                "spoken_prefix_lock": threading.Lock(),
                **new_playback_ack_state(),
            }
            register_playback_ack_branch(response_id, playback_branch)
            unified_native_branch: dict[str, Any] | None = None
            unified_transition_lock: Any | None = None
            unified_transition_lock_acquired = False
            unified_close_ids: tuple[int, ...] | None = None
            listener_dispatcher = shared_listener_dispatcher_holder.get("dispatcher")
            listener_dispatch_pause: dict[str, Any] = {"held": False, "released": False}
            listener_dispatch_pause_lock = threading.Lock()
            revisioned_v1_speak = False

            def release_listener_burst_pause(reason: str) -> bool:
                with listener_dispatch_pause_lock:
                    if not listener_dispatch_pause["held"]:
                        return False
                    listener_dispatch_pause["held"] = False
                    listener_dispatch_pause["released"] = True
                resumed = bool(
                    listener_dispatcher is not None
                    and listener_dispatcher.resume_session(sess.session_id)
                )
                if unified_native_branch is not None:
                    unified_native_branch["listener_dispatch_pause_release_reason"] = str(reason)
                    unified_native_branch["listener_dispatch_pause_resumed"] = resumed
                update_response_runtime_metrics(response_id, "listener_burst_dispatch_pause", {
                    "status": "released" if resumed else "already_unregistered",
                    "reason": str(reason),
                    "release_at_ns": time.monotonic_ns(),
                })
                return resumed

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
                revisioned_v1_speak = bool(
                    action == ControlAction.SPEAK
                    and response_protocol == CONTROLLER_GUIDED_SAY_V1
                )
                try:
                    if action == ControlAction.SPEAK:
                        if listener_dispatcher is None:
                            raise RuntimeError(
                                "unified SPEAK requires a registered shared listener dispatcher"
                            )
                    if action == ControlAction.BACKCHANNEL:
                        private_handle = await _begin_private_backchannel_decode(
                            worker,
                            sess.session_id,
                            can_continue=lambda: (
                                not closing and sess.is_generation_current(token)
                            ),
                            transition_lock=unified_transition_lock,
                        )
                        unified_transition_lock_acquired = True
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
                            "say_token_packets": [],
                            "segment_token_packets": [],
                            "client_consumed_pcm_sequence": -1,
                            "client_consumed_segments": (),
                            "response_event_ledger": response_event_ledger,
                            "spoken_prefix_lock": threading.Lock(),
                            "queue_open": False,
                            "ephemeral_private": True,
                            "ephemeral_decode_handle": private_handle,
                            "finalized": False,
                            "ended_response": False,
                            "transition_lock": unified_transition_lock,
                            "close_ids": (),
                            "transition_token_ids": tuple(
                                _build_unified_backchannel_response_suffix(engine, sess)
                            ),
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
                        register_playback_ack_branch(response_id, unified_native_branch)
                    elif action == ControlAction.SPEAK:
                        listener_state = await asyncio.wrap_future(
                            worker.session_state(sess.session_id)
                        )
                        if not sess.is_generation_current(token):
                            raise RuntimeError("revisioned SPEAK lost its response generation")
                        response_handle = None
                        active_revision_id = "r0"
                        begin_deadline = asyncio.get_running_loop().time() + 2.0
                        while True:
                            if closing or not sess.is_generation_current(token):
                                raise RuntimeError("revisioned SPEAK lost its response generation")
                            try:
                                response_handle = await asyncio.wrap_future(
                                    worker.begin_response_decode(
                                        sess.session_id,
                                        expected_generation_id=int(listener_state.generation_id),
                                        response_id=response_id,
                                        response_generation_id=int(token.session_generation_id),
                                        revision_id=active_revision_id,
                                        purpose=ResponseDecodePurpose.for_action(action),
                                        source_watermark=None,
                                        max_tokens=768,
                                        prompt_token_budget=774,
                                    )
                                )
                            except RuntimeError as exc:
                                if (
                                    str(exc) not in {
                                        "response fork requires a completed listener frame watermark",
                                        "response fork source native row no longer matches its completed listener watermark",
                                    }
                                    or asyncio.get_running_loop().time() >= begin_deadline
                                ):
                                    raise
                                await asyncio.sleep(0.002)
                                listener_state = await asyncio.wrap_future(
                                    worker.session_state(sess.session_id)
                                )
                                continue
                            with sess._generation_lock:
                                if not sess.is_generation_current(token):
                                    raise RuntimeError("revisioned SPEAK lost its response generation")
                                history_revision = int(sess.semantic_context_revision)
                                mirror_cache_epoch = getattr(
                                    sess.listener_state, "cache_epoch_id", None
                                )
                                mirror_native_revision = getattr(
                                    sess.listener_state,
                                    "native_semantic_context_revision",
                                    None,
                                )
                                mirror_history_revision = getattr(
                                    sess.listener_state, "applied_history_revision", None
                                )
                                handle_context_current = (
                                    isinstance(mirror_cache_epoch, int)
                                    and not isinstance(mirror_cache_epoch, bool)
                                    and mirror_cache_epoch >= 0
                                    and int(response_handle.source_cache_epoch_id)
                                    == mirror_cache_epoch
                                    and int(response_handle.source_history_revision)
                                    == history_revision
                                    and mirror_native_revision == history_revision
                                    and mirror_history_revision == history_revision
                                )
                            if handle_context_current:
                                break
                            await asyncio.shield(asyncio.wrap_future(
                                worker.end_response_decode(response_handle)
                            ))
                            response_handle = None
                            if asyncio.get_running_loop().time() >= begin_deadline:
                                raise RuntimeError(
                                    "revisioned SPEAK listener cache has not applied current history"
                                )
                            await asyncio.sleep(0.01)
                            listener_state = await asyncio.wrap_future(
                                worker.session_state(sess.session_id)
                            )
                        with sess._generation_lock:
                            if not sess.is_generation_current(token):
                                raise RuntimeError("revisioned SPEAK lost its response generation")
                            history_revision = int(sess.semantic_context_revision)
                        branch_cache_epoch = int(response_handle.source_cache_epoch_id)
                        unified_native_branch = {
                            "worker": worker,
                            "response_protocol": response_protocol,
                            "response_action": ControlAction.SPEAK.value,
                            "revisioned_live_say": revisioned_v1_speak,
                            "response_id": response_id,
                            "response_generation_id": int(token.session_generation_id),
                            "speaker_segment_id": str(token.speaker_segment_id),
                            "session_generation_id": int(token.session_generation_id),
                            "listener_generation_id": int(response_handle.parent_generation_id),
                            "listener_state_epoch_id": str(response_handle.parent_state_epoch_id),
                            "source_frame_id": int(response_handle.source_frame_id),
                            "source_state_version": int(response_handle.source_state_version),
                            "source_sequence_position": int(response_handle.source_sequence_position),
                            "cache_position": int(response_handle.source_cache_position),
                            "turn_start_cache_position": int(response_handle.source_cache_position),
                            "spoken_prefix_base_cursor": int(response_handle.source_cache_position),
                            "spoken_prefix_source_seq": -1,
                            "queue_open": False,
                            "queue_epoch_state": None,
                            "queue_open_epoch": None,
                            "queue_close_epoch": None,
                            "response_decode_handle": response_handle,
                            "response_decode_purpose": ResponseDecodePurpose.ACTIVE.value,
                            "revision_id": active_revision_id,
                            "cache_epoch_id": branch_cache_epoch,
                            "listener_cache_epoch_id": branch_cache_epoch,
                            "applied_history_revision": history_revision,
                            "history_revision": history_revision,
                            "transition_token_ids": tuple(
                                int(value)
                                for value in _build_unified_audio_response_suffix(engine, sess)
                            ),
                            "playback_segments": {},
                            "say_token_packets": [],
                            "segment_token_packets": [],
                            "client_consumed_pcm_sequence": -1,
                            "client_consumed_segments": (),
                            "response_event_ledger": response_event_ledger,
                            "spoken_prefix_lock": threading.RLock(),
                            "generated_token_count": 0,
                            "native_token_append_count": 0,
                            "_generated_budget_used": 0,
                            "_next_revision_number": 1,
                            "revision_metrics": {
                                "adopted_revisions": 0,
                                "last_adaptation_lag_frames": None,
                                "last_adopted_watermark_frame_id": None,
                            },
                            "finalized": False,
                            "ended_response": False,
                            "history_committed": False,
                            "transition_lock": None,
                            "close_ids": (),
                            "policy_decision_at_ns": (
                                int(intent_origin["policy_decision_at_ns"])
                                if isinstance(intent_origin, dict)
                                and intent_origin.get("policy_decision_at_ns") is not None
                                else None
                            ),
                            "queue_open_requested_ns": None,
                            "queue_open_ready_ns": time.monotonic_ns(),
                            "direct_audio_context": None,
                            "acoustic_state_snapshot": None,
                            "listener_dispatch_pause_active": False,
                            **new_playback_ack_state(),
                        }
                        (
                            unified_native_branch["direct_audio_context"],
                            unified_native_branch["acoustic_state_snapshot"],
                        ) = _snapshot_speaking_branch_context(
                            direct_audio_context,
                            sess.listener_state,
                        )
                        register_playback_ack_branch(response_id, unified_native_branch)
                    else:
                        raise RuntimeError("REPLAN requires the current response revision coordinator")
                except BaseException as exc:
                    if unified_transition_lock_acquired:
                        unified_transition_lock.release()
                        unified_transition_lock_acquired = False
                    release_listener_burst_pause("queue_open_failed")
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

            def commit_deferred_acoustic_history() -> bool:
                branch = unified_native_branch or playback_branch
                if (
                    branch is None
                    or not isinstance(branch.get("deferred_acoustic_history_commit"), Mapping)
                    or branch.get("deferred_acoustic_history_committed") is True
                ):
                    return False
                committed, acknowledged_text, acknowledged_cursor, done_at_ns = (
                    _commit_deferred_acoustic_history_archive(
                        branch,
                        sess,
                        response_token=token,
                        response_id=response_id,
                        user_onset_frame_id=user_turn_onset_frame_id,
                    )
                )
                archive_assistant_finalized["value"] = bool(committed)
                update_response_runtime_metrics(response_id, "audio_history_archive_assistant", {
                    "status": _deferred_acoustic_history_archive_status(
                        branch, acknowledged_text
                    ),
                    "assistant_text_committed": bool(committed and acknowledged_text),
                    "history_prefix_source": "client_consumed_pcm_ack",
                    "client_consumed_prefix_exclusive_cursor": acknowledged_cursor,
                    "response_done_at_monotonic_ns": done_at_ns,
                })
                return bool(committed)

            async def finalize_nonunified_audio_history_after_playback() -> None:
                branch = unified_native_branch or playback_branch
                if branch is None or not isinstance(
                    branch.get("deferred_acoustic_history_commit"), Mapping
                ):
                    return
                try:
                    await _wait_for_response_playback_drain(branch)
                    branch["playback_drained"] = True
                except Exception as exc:
                    branch["client_playback_ack_missing"] = True
                    branch["client_playback_ack_status"] = (
                        "cancelled_final_ack_timeout"
                        if branch.get("playback_cancelled") is True
                        else "playback_ack_timeout"
                    )
                    branch["playback_drain_failure"] = f"{type(exc).__name__}: {exc}"[:240]
                    update_response_runtime_metrics(response_id, "client_playback_drain", {
                        "status": branch["client_playback_ack_status"],
                        "failure_reason": branch["playback_drain_failure"],
                        "logical_client_consumption_only": True,
                    })
                _prepare_deferred_acoustic_history_finalization(
                    branch,
                    history_committed=False,
                    commit_deferred_acoustic_history=commit_deferred_acoustic_history,
                )

            async def finalize_unified_native_cache(*, history_committed: bool) -> None:
                branch = unified_native_branch
                if branch is None:
                    return
                branch["closing"] = True
                if branch.get("finalized"):
                    return
                finalizer_status = "completed"
                finalizer_error: str | None = None
                try:
                    worker = branch["worker"]
                    history_committed = _prepare_deferred_acoustic_history_finalization(
                        branch,
                        history_committed=history_committed,
                        commit_deferred_acoustic_history=commit_deferred_acoustic_history,
                    )
                    if (
                        branch.get("playback_drained")
                        or not _response_branch_has_pending_pcm(branch)
                    ):
                        with sess._generation_lock:
                            if sess.control_state in {ControlState.SPEAKING, ControlState.STOPPING}:
                                sess.control_state = ControlState.LISTENING
                    if int(branch.get("client_consumed_pcm_sequence", -1)) < 0:
                        branch["client_playback_limitation"] = (
                            "no client_consumed_pcm ACK; only the last confirmed complete-segment prefix is retained"
                        )
                        update_response_runtime_metrics(response_id, "client_consumed_pcm", {
                            "status": "no_playback_ack",
                            "prefix_policy": "retain_last_confirmed_prefix_only",
                            "logical_client_consumption_only": True,
                        })
                    drained = await _finalize_unified_native_cache_branch(
                        worker,
                        sess.session_id,
                        branch,
                        history_committed=history_committed,
                        close_ids=tuple(branch["close_ids"]),
                    )
                    branch.setdefault(
                        "queued_audio_drain_metrics",
                        _queued_audio_drain_metric_fields(drained),
                    )
                except BaseException as exc:
                    finalizer_status = (
                        "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
                    )
                    finalizer_error = f"{type(exc).__name__}: {exc}"[:240]
                    raise
                finally:
                    playback_drain_failure = branch.get("playback_drain_failure")
                    if playback_drain_failure and finalizer_status == "completed":
                        finalizer_status = "failed"
                        finalizer_error = str(playback_drain_failure)[:240]
                    commit_deferred_acoustic_history()
                    rollback_metrics = branch.get("rollback_metrics")
                    if isinstance(rollback_metrics, Mapping):
                        update_response_runtime_metrics(
                            response_id,
                            "unified_native_rollback",
                            dict(rollback_metrics),
                        )
                    if not branch.get("ended_response"):
                        sess.end_response(token)
                        branch["ended_response"] = True
                    drain_metrics = branch.get("queued_audio_drain_metrics")
                    finalizer_metrics = (
                        dict(drain_metrics) if isinstance(drain_metrics, Mapping) else {}
                    )
                    queue_closed_at_ns = branch.get("queue_closed_at_ns")
                    if isinstance(queue_closed_at_ns, int):
                        finalizer_metrics["drain_queue_to_finalizer_event_ms"] = max(
                            0, time.monotonic_ns() - queue_closed_at_ns
                        ) / 1_000_000.0
                        branch["queued_audio_drain_metrics"] = dict(finalizer_metrics)
                    finalizer_metrics["status"] = finalizer_status
                    if finalizer_error:
                        finalizer_metrics["failure_reason"] = finalizer_error
                    update_response_runtime_metrics(
                        response_id,
                        "unified_native_finalizer",
                        finalizer_metrics,
                    )
                    # Send without the response token: YIELD/cancellation commonly
                    # invalidates it before this owner-thread finalizer completes.
                    try:
                        await asyncio.shield(_send_unified_native_finalizer_event(
                            send_ws,
                            response_id,
                            status=finalizer_status,
                            rollback_metrics=(
                                rollback_metrics if isinstance(rollback_metrics, Mapping) else None
                            ),
                            drain_metrics=(
                                drain_metrics if isinstance(drain_metrics, Mapping) else None
                            ),
                            runtime_metrics=unified_native_terminal_metrics_event(response_id),
                            error=finalizer_error,
                        ))
                    except Exception:
                        logger.exception(
                            "[realtime %s %s] failed to publish unified native finalizer metrics",
                            sess.session_id[:8],
                            response_id,
                        )

            async def finalize_unified_native_cache_after_playback() -> None:
                branch = unified_native_branch
                if branch is None:
                    return
                async def record_drain_failure(
                    _error: Exception, failure: str
                ) -> None:
                    commit_deferred_acoustic_history()
                    update_response_runtime_metrics(response_id, "client_playback_drain", {
                        "status": "failed",
                        "failure_reason": failure,
                        "prefix_policy": "rebuild_only_acknowledged_complete_segments",
                        "client_consumed_pcm_sequence": int(
                            branch.get("client_consumed_pcm_sequence", -1)
                        ),
                    })

                try:
                    drain_error = await _finalize_after_playback_drain(
                        branch,
                        wait_for_drain=_wait_for_response_playback_drain,
                        finalize=finalize_unified_native_cache,
                        record_drain_failure=record_drain_failure,
                    )
                except Exception as exc:
                    logger.exception(
                        "[realtime %s %s] playback drain/cache finalization failed",
                        sess.session_id[:8],
                        response_id,
                    )
                    rollback_failed = branch.get("playback_drain_failure") is not None
                    failure_status = (
                        "playback_drain_rollback_failed"
                        if rollback_failed else "unified_cache_finalization_failed"
                    )
                    fail_native_listener_output_health(-1, {
                        "status": failure_status,
                        "error": str(exc),
                        "drain_error": branch.get("playback_drain_failure"),
                    })
                    if not branch.get("ended_response"):
                        sess.end_response(token)
                        branch["ended_response"] = True
                    await send_ws({
                        "type": "error",
                        "response_id": response_id,
                        "error": {
                            "code": (
                                "CLIENT_PLAYBACK_DRAIN_ROLLBACK_FAILED"
                                if rollback_failed
                                else "UNIFIED_NATIVE_CACHE_FINALIZATION_FAILED"
                            ),
                            "message": str(exc)[:240],
                        },
                        "runtime_metrics": runtime_metrics_event(response_id),
                    })
                    return
                if drain_error is not None:
                    branch_cancelled = bool(
                        branch.get("playback_cancelled") is True
                        or getattr(sess, "response_cancel_notified", False)
                        or str(getattr(sess, "stop_reason", "")).upper()
                        == ControlAction.YIELD.value
                        or str(branch.get("cancellation_reason", "")).upper()
                        == ControlAction.YIELD.value
                    )
                    if not branch_cancelled:
                        await send_ws({
                            "type": "error",
                            "response_id": response_id,
                            "error": {
                                "code": "CLIENT_PLAYBACK_DRAIN_FAILED",
                                "message": branch.get("playback_drain_failure"),
                            },
                            "runtime_metrics": runtime_metrics_event(response_id),
                        })

            def response_listener_admission_wait() -> None:
                listener_admission_wait(token)

            def response_listener_compute_quantum():
                return listener_compute_quantum(token)

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
                        archived = sess.complete_audio_history_archive(
                            int(archive_id_holder["archive_id"]),
                            heard_value if transcript_is_complete else "",
                            expected_generation_id=int(token.session_generation_id),
                        )
                        if not archived and not sess.is_generation_current(token):
                            return "stale_generation"
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
            if unified_transition_lock is not None and unified_transition_lock_acquired:
                unified_transition_lock.release()
                unified_transition_lock_acquired = False
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
                control = _RealtimeResponseControl(
                    token,
                    playout,
                    packetizer,
                    audio_queue,
                    text_queue,
                    output_cancel=response_output_cancel,
                    cancel_reason=response_output_cancel_reason,
                )
                with response_controls_lock:
                    response_controls[response_id] = control
                response_created_event: dict[str, Any] = {
                    "type": "response.created",
                    "response_id": response_id,
                    "response_created_at_ns": time.monotonic_ns(),
                    "response_profile": selected_response_profile,
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
                playback_pacing_start_ns: int | None = None
                cumulative_pcm_samples_emitted: int = 0
                first_tts_audio_chunk_pcm_at_ns: int | None = None
                first_pcm_trace_lock = threading.Lock()
                first_pcm_trace: dict[str, Any] = {}
                active_output_segment_id = {"value": str(token.speaker_segment_id)}

                def record_first_pcm_trace(
                    boundary: str,
                    at_ns: int | None = None,
                    **metadata: Any,
                ) -> None:
                    """Keep monotonic first-PCM boundary times in response metrics only."""
                    key = f"first_pcm_{boundary}_ns"
                    timestamp = time.monotonic_ns() if at_ns is None else int(at_ns)
                    with first_pcm_trace_lock:
                        if key in first_pcm_trace:
                            return
                        first_pcm_trace.setdefault("clock", "monotonic_ns")
                        first_pcm_trace.setdefault("response_id", response_id)
                        first_pcm_trace[key] = timestamp
                        for name, value in metadata.items():
                            if value is not None:
                                first_pcm_trace.setdefault(name, value)
                        snapshot = dict(first_pcm_trace)
                    with response_runtime_metrics_lock:
                        response_runtime_metrics.setdefault(response_id, {})[
                            "first_pcm_trace"
                        ] = snapshot

                def record_first_pcm_packet_identity(packet: AudioPacket) -> None:
                    record_first_pcm_trace(
                        "packet_identity",
                        packet_id=int(packet.packet_id),
                        generation_id=int(packet.generation_id),
                        speaker_segment_id=str(packet.speaker_segment_id),
                        first_frame_id=int(packet.first_frame_id),
                        last_frame_id=int(packet.last_frame_id),
                        start_sample=int(packet.start_sample),
                        end_sample=int(packet.end_sample),
                    )

                def record_first_pcm_consumer_get(item: Any) -> None:
                    if not isinstance(item, AudioPacket):
                        return
                    with first_pcm_trace_lock:
                        expected_packet_id = first_pcm_trace.get("packet_id")
                    if expected_packet_id is not None and int(item.packet_id) == int(expected_packet_id):
                        record_first_pcm_trace("consumer_get")

                def packetize_frames_as_packets(frames: list[PlayoutFrame]) -> list[AudioPacket]:
                    if packetizer is None:
                        raise RuntimeError("packetizer non inizializzato")
                    packets: list[AudioPacket] = []
                    for frame in frames:
                        packets.extend(packetizer.push(frame))
                    return packets

                def drain_cancelled_response_output() -> None:
                    """Stop the producer path and discard every queued response item."""
                    response_output_cancel.set()
                    with sess._generation_lock:
                        if sess.is_generation_current(token):
                            sess.stop_requested = True
                    if playout is not None:
                        playout.cancel()
                    if packetizer is not None:
                        packetizer.cancel_all()
                    _drain_queue(audio_queue)
                    _drain_queue(text_queue)

                for output_branch in (playback_branch, unified_native_branch):
                    if isinstance(output_branch, dict):
                        output_branch["playback_cancel_callback"] = drain_cancelled_response_output

                def cancel_response_output(reason: str) -> None:
                    response_output_cancel_reason["value"] = str(reason)
                    with playback_ack_branches_lock:
                        cancelled_branch = playback_ack_branches.get(response_id)
                    if isinstance(cancelled_branch, dict):
                        cancelled_branch["cancellation_reason"] = str(reason)
                        _cancel_response_branch_playback(cancelled_branch)
                    else:
                        drain_cancelled_response_output()

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
                            if (
                                response_output_cancel.is_set()
                                or sess.stop_reason == ControlAction.YIELD.value
                            ):
                                return False
                            if not sess.is_generation_current(token):
                                return False
                            try:
                                output_queue.put_nowait(item)
                            except queue.Full:
                                return None
                            if queue_kind == "audio" and isinstance(item, AudioPacket):
                                with first_pcm_trace_lock:
                                    first_packet_id = first_pcm_trace.get("packet_id")
                                if first_packet_id is not None and int(item.packet_id) == int(first_packet_id):
                                    record_first_pcm_trace("queue_put")
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
                    callback_entry_ns = time.monotonic_ns()
                    if first_tts_audio_chunk_pcm_at_ns is None:
                        # Same monotonic clock as Control6's policy_decision_at_ns.
                        # This latch is before playout, packetization, and WS relay.
                        first_tts_audio_chunk_pcm_at_ns = callback_entry_ns
                    array = np.asarray(chunk, dtype=np.float32)
                    if response_event_ledger is not None:
                        response_event_ledger.record_pcm_synthesized(int(array.size))
                    record_first_pcm_trace(
                        "callback_entry",
                        callback_entry_ns,
                        conditioning_frame_id=int(conditioning_frame_id),
                        pcm_samples=int(array.size),
                    )
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
                        if ready:
                            record_first_pcm_trace(
                                "playout_release",
                                frame_count=len(ready),
                                first_frame_id=int(ready[0].frame_id),
                                last_frame_id=int(ready[-1].frame_id),
                            )
                        packets = packetize_frames_as_packets(ready)
                        if packets:
                            packets_to_enqueue.extend(packets)
                            first_packet = packets[0]
                            record_first_pcm_packet_identity(first_packet)
                            record_first_pcm_trace(
                                "packetizer_ready",
                                frame_count=sum(
                                    packet.last_frame_id - packet.first_frame_id + 1
                                    for packet in packets
                                ),
                            )
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
                    array = np.asarray(chunk, dtype=np.float32).reshape(-1)
                    horizon = _legacy_audio_chunk_conditioning_horizon(
                        next_generated_frame_id,
                        int(array.size),
                    )
                    on_audio_chunk_with_horizon(array, horizon)

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

                ack_resources_branch = (
                    unified_native_branch
                    if isinstance(unified_native_branch, dict)
                    else playback_branch
                )
                finish_audio_segment = _make_finish_audio_segment_callback(
                    active_output_segment_id=active_output_segment_id,
                    playout=playout,
                    packetizer=packetizer,
                    ack_resources_branch=ack_resources_branch,
                    response_id=response_id,
                    generation_id=int(token.session_generation_id),
                    response_segment_id=str(token.speaker_segment_id),
                    packetize_frames_as_packets=packetize_frames_as_packets,
                    queue_response_output=queue_response_output,
                    audio_queue=audio_queue,
                )

                def on_text_chunk(chunk: str) -> None:
                    cleaned_chunk = re.sub(r"<unused\d+>", " ", str(chunk))
                    if cleaned_chunk:
                        queue_response_output(
                            cleaned_chunk, queue_kind="text", output_queue=text_queue, burst_items=1,
                        )

                def on_tts_text_consumed(frontier: Any, cumulative_source_text: str) -> None:
                    """Publish V1 text only after its Qwen token forward commits."""
                    cumulative = str(cumulative_source_text)
                    already_public = str(getattr(frontier, "already_public_text", ""))
                    if not cumulative.startswith(already_public):
                        raise RuntimeError(
                            "Qwen-consumed text no longer extends the public response prefix"
                        )
                    delta = cumulative[len(already_public):]
                    if not delta:
                        return
                    if not sess.record_synthesized_text(token, delta):
                        return
                    on_text_chunk(delta)

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
                    async def release_pcm(released_packet: AudioPacket) -> bool:
                        nonlocal streamed_chunks, playback_pacing_start_ns, cumulative_pcm_samples_emitted
                        now_ns = time.monotonic_ns()
                        if playback_pacing_start_ns is None:
                            playback_pacing_start_ns = now_ns
                        else:
                            audio_duration_s = cumulative_pcm_samples_emitted / CODEC_SAMPLE_RATE
                            elapsed_s = (now_ns - playback_pacing_start_ns) / 1_000_000_000.0
                            lead_s = audio_duration_s - elapsed_s
                            MAX_CLIENT_PLAYBACK_LEAD_S = 2.0
                            if lead_s > MAX_CLIENT_PLAYBACK_LEAD_S:
                                throttle_s = lead_s - MAX_CLIENT_PLAYBACK_LEAD_S
                                while throttle_s > 0.005 and not discard_fenced_output():
                                    sleep_chunk = min(throttle_s, 0.05)
                                    await asyncio.sleep(sleep_chunk)
                                    throttle_s -= sleep_chunk
                                if discard_fenced_output():
                                    return False
                        server_emit_started_at_ns = time.monotonic_ns()
                        payload = {
                            "type": "response.audio.delta",
                            "delta": _wav_b64(released_packet.pcm),
                            "response_id": response_id,
                            "server_emit_started_at_ns": server_emit_started_at_ns,
                            "raw_source_audio_quality": _raw_source_audio_quality(released_packet.pcm),
                            "packet_id": released_packet.packet_id,
                            "sequence": released_packet.packet_id,
                            "generation_id": released_packet.generation_id,
                            "speaker_segment_id": released_packet.speaker_segment_id,
                            "first_frame_id": released_packet.first_frame_id,
                            "last_frame_id": released_packet.last_frame_id,
                            "start_sample": released_packet.start_sample,
                            "end_sample": released_packet.end_sample,
                            "sample_rate": CODEC_SAMPLE_RATE,
                        }
                        with first_pcm_trace_lock:
                            is_first_pcm_packet = (
                                first_pcm_trace.get("packet_id")
                                == int(released_packet.packet_id)
                            )
                        if not await send_ws(
                            payload,
                            token,
                            first_pcm_trace_callback=(
                                record_first_pcm_trace if is_first_pcm_packet else None
                            ),
                        ):
                            if not discard_fenced_output():
                                cancel_response_output("websocket_audio_send_failed")
                            return False
                        if discard_fenced_output():
                            return False
                        if not record_emitted_reference(released_packet.pcm):
                            discard_fenced_output()
                            return False
                        frame_count = released_packet.last_frame_id - released_packet.first_frame_id + 1
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
                                _record_emitted_pcm_packet(ack_branch, released_packet)
                        cumulative_pcm_samples_emitted += int(released_packet.pcm.size)
                        streamed_chunks += 1
                        return True

                    def acknowledge_released_pcm(released_packet: AudioPacket) -> bool:
                        ack_result = {"ok": False}

                        def advance_speech_watermark() -> None:
                            with playback_ack_branches_lock:
                                ack_branch = playback_ack_branches.get(response_id)
                            ack_result["ok"] = _ack_response_speech_packet(
                                ack_branch,
                                released_packet,
                                response_id=response_id,
                                generation_id=int(token.session_generation_id),
                                response_segment_id=str(token.speaker_segment_id),
                            )

                        if not sess.guarded_write(
                            token, "speech_pcm_ack", advance_speech_watermark
                        ):
                            discard_fenced_output()
                            return False
                        if not ack_result["ok"]:
                            cancel_response_output("speech_pcm_ack_identity_mismatch")
                            return False
                        return True

                    def validate_pcm_packet(released_packet: AudioPacket) -> bool:
                        if not _audio_packet_matches_emission_owner(
                            released_packet,
                            generation_id=int(token.session_generation_id),
                            speaker_segment_id=str(active_output_segment_id["value"]),
                        ):
                            if not discard_fenced_output():
                                cancel_response_output("audio_packet_identity_mismatch")
                            return False
                        return not discard_fenced_output()

                    return await _release_pcm_packet_then_ack(
                        packet,
                        validate=validate_pcm_packet,
                        release=release_pcm,
                        acknowledge=acknowledge_released_pcm,
                    )

                async def emit_audio_queue_item(item: Any) -> bool:
                    if isinstance(item, AudioPacket):
                        return await emit_audio_packet(item)
                    if not isinstance(item, _ClosedTTSSegment):
                        raise TypeError("realtime audio queue contains an unsupported item")
                    with playback_ack_branches_lock:
                        ack_branch = playback_ack_branches.get(response_id)
                    ack_resources_branch = (
                        ack_branch
                        if ack_branch is not None
                        else (
                            unified_native_branch
                            if isinstance(unified_native_branch, dict)
                            else playback_branch
                        )
                    )
                    ack_resources = _response_speech_ack_resources(
                        ack_resources_branch,
                        response_id=response_id,
                        generation_id=int(token.session_generation_id),
                        response_segment_id=str(token.speaker_segment_id),
                    )
                    if ack_branch is None:
                        if ack_resources is not None:
                            ack_resources["closed_unit_streams"].pop(
                                item.speaker_segment_id, None
                            )
                            _signal_emitted_tts_segment(
                                ack_resources["closed_unit_events"], item.speaker_segment_id
                            )
                        return True
                    if ack_resources is None:
                        cancel_response_output("speech_ack_branch_identity_mismatch")
                        return False
                    with ack_branch.get("spoken_prefix_lock", nullcontext()):
                        marker = _complete_client_pcm_segment(
                            ack_branch,
                            speaker_segment_id=item.speaker_segment_id,
                            token_packets=item.source_tokens,
                        )
                    sent = marker is None or await send_ws(marker, token)
                    if sent:
                        ack_resources["closed_unit_streams"].pop(item.speaker_segment_id, None)
                        _signal_emitted_tts_segment(
                            ack_resources["closed_unit_events"], item.speaker_segment_id
                        )
                    return sent

                published_text_delta_count = 0

                async def emit_text_chunk(chunk: str) -> bool:
                    nonlocal published_text_delta_count
                    sent = await send_ws(
                        {"type": "response.output_text.delta", "delta": chunk, "response_id": response_id},
                        token,
                    )
                    if not sent:
                        if not discard_fenced_output():
                            cancel_response_output("websocket_text_send_failed")
                        return False
                    if chunk:
                        published_text_delta_count += 1
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
                    def send_event(event: dict[str, Any]) -> bool:
                        async def send_if_current() -> bool:
                            if response_output_cancel.is_set():
                                return False
                            return await send_ws(event, token)

                        return asyncio.run_coroutine_threadsafe(
                            send_if_current(), response_event_loop
                        ).result(timeout=5.0)

                    _publish_thinker_metrics_event(
                        values,
                        response_id=response_id,
                        policy_decision_at_ns=policy_decision_at_ns,
                        session=sess,
                        token=token,
                        cancellation_event=response_output_cancel,
                        update_runtime_metrics=update_response_runtime_metrics,
                        send_event=send_event,
                    )

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
                                    sess._append_history_items_locked([
                                        {"role": "user", "content": user_text}
                                    ])
                                    # Keep a committed user turn even if barge-in
                                    # cancels the subsequent Thinker/Talker work.
                                    history = [dict(message) for message in sess.history]
                                    sess.note_semantic_context_changed()
                                sess.notify_history_context_changed()
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
                                    "response_profile": selected_response_profile,
                                    **_controller_guided_say_generation_options(
                                        action=action,
                                        controller_guided_say=controller_guided_say_response,
                                        direct_audio_candidate=response_uses_direct_audio,
                                        response_protocol=response_protocol,
                                        response_profile=selected_response_profile,
                                    ),
                                    "_direct_audio_pcm": thinker_audio,
                                    "_response_id": response_id,
                                    "_direct_audio_context": (
                                        dict(direct_audio_context)
                                        if isinstance(direct_audio_context, dict)
                                        else None
                                    ),
                                    "_generation_id": token.session_generation_id,
                                    "_generation_token": token,
                                    "_unified_native_branch": unified_native_branch,
                                    "_cache_epoch_id": (
                                        unified_native_branch.get("cache_epoch_id")
                                        if isinstance(unified_native_branch, dict)
                                        else None
                                    ),
                                    "_on_tts_text_consumed": (
                                        on_tts_text_consumed
                                        if revisioned_v1_speak else None
                                    ),
                                    "_playback_ack_branch": (
                                        unified_native_branch
                                        if isinstance(unified_native_branch, dict)
                                        else playback_branch
                                    ),
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
                                    "_listener_compute_quantum": (
                                        response_listener_compute_quantum
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
                timeout_limit_s = float(
                    os.environ.get(
                        "DUPLEX_INFERENCE_TIMEOUT_SECONDS",
                        "300.0" if selected_response_profile == "long" else "180.0",
                    )
                )
                deadline = loop.time() + timeout_limit_s
                while not future.done():
                    if loop.time() >= deadline:
                        with sess._generation_lock:
                            sess.stop_requested = True
                        sess.invalidate_generation("realtime_timeout")
                        _stop_owned_synthesis(sess.session_id, response_id)
                        logger.error("[realtime %s %s] timeout dopo %.1fs (limite=%.1fs, profile=%s)", sess.session_id[:8], response_id, time.perf_counter() - started, timeout_limit_s, selected_response_profile)
                        await send_ws({"type": "error", "error": {"code": "INFERENCE_TIMEOUT", "message": f"risposta oltre {int(timeout_limit_s)} secondi"}, "response_id": response_id})
                        return
                    try:
                        chunk_wav = audio_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.005)
                    else:
                        record_first_pcm_consumer_get(chunk_wav)
                        if not await emit_audio_queue_item(chunk_wav):
                            break
                    try:
                        chunk_text = text_queue.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        if not await emit_text_chunk(chunk_text):
                            break
                try:
                    result = await future
                except LiveSayCancelled as exc:
                    response_cancelled = bool(
                        sess.stop_requested
                        or any(
                            isinstance(candidate_branch, dict)
                            and candidate_branch.get("playback_cancelled")
                            for candidate_branch in (
                                unified_native_branch,
                                playback_branch,
                            )
                        )
                        or response_output_cancel.is_set()
                        or not sess.is_generation_current(token)
                    )
                    if not response_cancelled:
                        raise
                    result = {
                        "cancelled": True,
                        "skip_reason": "cancelled",
                        "text": "",
                        "wav": None,
                    }
                current = sess.is_generation_current(token)
                yielded = current and (
                    str(sess.stop_reason).upper() == ControlAction.YIELD.value
                    or str(response_output_cancel_reason.get("value", "")).upper()
                    == ControlAction.YIELD.value
                )
                if yielded:
                    with response_runtime_metrics_lock:
                        trace = response_runtime_metrics.setdefault(response_id, {}).get("generation_trace")
                        if isinstance(trace, dict):
                            trace["terminal_reason"] = (
                                "yielded_before_first_token"
                                if not trace.get("thinker_token_ids")
                                else "yielded_after_token"
                            )
                if current and result.get("skip_reason") and not yielded:
                    playout.cancel()
                    _drain_queue(audio_queue)
                    _drain_queue(text_queue)
                    cancelled_event = {
                        "type": "response.cancelled", "reason": result["skip_reason"],
                        "finish_reason": "cancelled",
                        "response_id": response_id,
                        "runtime_metrics": runtime_metrics_event(response_id, result),
                    }
                    await send_ws(
                        cancelled_event, token,
                    )
                    return
                if not current:
                    discard_fenced_output()
                    final_runtime_metrics = runtime_metrics_event(response_id, result)
                    completion_metrics = final_runtime_metrics.get("response_completion")
                    cancellation_reason = (
                        completion_metrics.get("cancellation_reason")
                        if isinstance(completion_metrics, Mapping)
                        else None
                    )
                    if not isinstance(cancellation_reason, str) or not cancellation_reason:
                        cancellation_reason = "generation_replaced_before_completion"
                    if _claim_response_cancel_notification(sess, response_id):
                        with sess._generation_lock:
                            cancellation_generation = int(sess.session_generation_id)
                        await send_ws({
                            "type": "response.cancelled",
                            "reason": "generation_replaced_before_completion",
                            "finish_reason": "cancelled",
                            "response_id": response_id,
                            "runtime_metrics": final_runtime_metrics,
                        }, generation_id=cancellation_generation)
                    await _send_response_completion_metrics(
                        send_ws,
                        response_id,
                        final_runtime_metrics,
                        cancellation_reason=cancellation_reason,
                    )
                    return
                if result.get("cancelled") and not yielded:
                    # A cooperative cancellation must not flush pending output.
                    playout.cancel()
                    _drain_queue(audio_queue)
                    _drain_queue(text_queue)
                    if current and not sess.response_cancel_notified:
                        await send_ws({"type": "response.cancelled", "reason": "cooperative_stop",
                                       "finish_reason": "cancelled",
                                       "response_id": response_id,
                                       "runtime_metrics": runtime_metrics_event(response_id, result)}, token)
                    final_runtime_metrics = runtime_metrics_event(response_id, result)
                    completion_metrics = final_runtime_metrics.get("response_completion")
                    cancellation_reason = (
                        completion_metrics.get("cancellation_reason")
                        if isinstance(completion_metrics, Mapping)
                        else None
                    )
                    if not isinstance(cancellation_reason, str) or not cancellation_reason:
                        cancellation_reason = "cooperative_stop"
                    await _send_response_completion_metrics(
                        send_ws,
                        response_id,
                        final_runtime_metrics,
                        cancellation_reason=cancellation_reason,
                    )
                    return
                if yielded:
                    # The YIELD fence owns every response protocol from here.
                    # No queued packets, text, or segment markers may escape,
                    # including already-closed TTS segments.
                    cancel_response_output("yield")
                    _drain_queue(audio_queue)
                    _drain_queue(text_queue)
                    active_segment = str(
                        active_output_segment_id["value"] or token.speaker_segment_id
                    )
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
                    with playback_ack_branches_lock:
                        candidate_branch = playback_ack_branches.get(response_id)
                    if isinstance(candidate_branch, dict):
                        claimed = _claim_branch_yield_cancel(
                            sess,
                            candidate_branch,
                            response_id=response_id,
                            generation_id=int(token.session_generation_id),
                            speaker_segment_id=str(token.speaker_segment_id),
                        )
                        if claimed:
                            scheduler_events.put_nowait({
                                "type": "response.cancelled",
                                "reason": "yield",
                                "finish_reason": "cancelled",
                                "response_id": response_id,
                                "generation_id": token.session_generation_id,
                                "runtime_metrics": runtime_metrics_event(response_id, result),
                            })
                        return
                    if _claim_response_cancel_notification(sess, response_id):
                        scheduler_events.put_nowait({
                            "type": "response.cancelled",
                            "reason": "yield",
                            "finish_reason": "cancelled",
                            "response_id": response_id,
                            "generation_id": token.session_generation_id,
                            "runtime_metrics": runtime_metrics_event(response_id, result),
                        })
                    return
                # YIELD is different: release accepted packets before flush output.
                while True:
                    try:
                        chunk_wav = audio_queue.get_nowait()
                    except queue.Empty:
                        break
                    record_first_pcm_consumer_get(chunk_wav)
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
                        await send_ws({
                            "type": "response.cancelled",
                            "reason": "cooperative_stop",
                            "finish_reason": "cancelled",
                            "response_id": response_id,
                            "runtime_metrics": runtime_metrics_event(response_id, result),
                        }, token)
                    return
                if result["tool_calls"]:
                    for call in result["tool_calls"]:
                        fn = call["function"]
                        await send_ws({"type": "response.function_call_arguments.done", "call_id": call["id"], "name": fn["name"], "arguments": fn["arguments"]}, token)
                    return
                final_result_text = str(result.get("text", ""))
                if should_emit_final_text_fallback(
                    published_delta_count=published_text_delta_count,
                    result_text=final_result_text,
                ) and not await emit_text_chunk(final_result_text):
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
                        and result.get("status") not in {
                            "no_speech",
                            "deferred_until_endpoint",
                            "backchannel_say_rejected",
                        }
                    )
                    if response_audio_complete:
                        result_runtime_metrics = result.get("runtime_metrics")
                        completion_metrics = (
                            result_runtime_metrics.get("response_completion")
                            if isinstance(result_runtime_metrics, Mapping) else None
                        )
                        response_finish = result.get("finish_reason")
                        if not isinstance(response_finish, str) and isinstance(
                            completion_metrics, Mapping
                        ):
                            response_finish = completion_metrics.get("finish_reason")
                        segment_termination = (
                            "length"
                            if (
                                isinstance(completion_metrics, Mapping)
                                and completion_metrics.get("acoustic_limit_reached") is True
                            ) or response_finish == "length"
                            else "eos"
                        )
                        with ack_branch.get("spoken_prefix_lock", nullcontext()):
                            token_packets = _client_consumed_v1_token_packets(
                                ack_branch,
                                termination_reason=segment_termination,
                            )
                            one_shot_v2_source_tokens = ack_branch.get(
                                "one_shot_v2_tts_source_tokens"
                            )
                            if (
                                direct_audio_v2_response
                                and not unified_kv_enabled
                                and segment_termination == "eos"
                                and isinstance(completion_metrics, Mapping)
                                and completion_metrics.get("eos_detected") is True
                                and isinstance(one_shot_v2_source_tokens, (tuple, list))
                            ):
                                token_packets = tuple(
                                    dict(row) for row in one_shot_v2_source_tokens
                                )
                            v1_source_tokens = ack_branch.get(
                                "nonunified_v1_source_tokens"
                            )
                            if (
                                controller_guided_say_response
                                and not unified_kv_enabled
                                and segment_termination == "eos"
                                and isinstance(completion_metrics, Mapping)
                                and completion_metrics.get("eos_detected") is True
                                and isinstance(v1_source_tokens, (tuple, list))
                            ):
                                v1_ledger = ack_branch.get("response_event_ledger")
                                v1_projection = (
                                    v1_ledger.project()
                                    if callable(getattr(v1_ledger, "project", None))
                                    else None
                                )
                                v1_frontier = (
                                    v1_ledger.source_frontier_snapshot()
                                    if callable(getattr(v1_ledger, "source_frontier_snapshot", None))
                                    else {}
                                )
                                normalized_v1_rows = tuple(
                                    dict(row) for row in v1_source_tokens
                                    if isinstance(row, Mapping)
                                )
                                projected_v1_rows = tuple(
                                    (
                                        int(row["source_token_seq"]),
                                        int(row["token_id"]),
                                        str(row["text_piece"]),
                                    )
                                    for row in getattr(v1_projection, "source_tokens", ())
                                )
                                expected_v1_rows = tuple(
                                    (
                                        int(row["source_token_seq"]),
                                        int(row["token_id"]),
                                        str(row["text_piece"]),
                                    )
                                    for row in normalized_v1_rows
                                )
                                final_source_sequence = (
                                    expected_v1_rows[-1][0]
                                    if expected_v1_rows else -1
                                )
                                if (
                                    expected_v1_rows
                                    and expected_v1_rows == projected_v1_rows
                                    and int(v1_frontier.get("generated_source_sequence", -1))
                                    == final_source_sequence
                                    and int(v1_frontier.get("tts_consumed_source_seq", -1))
                                    == final_source_sequence
                                ):
                                    token_packets = normalized_v1_rows
                            marker = _complete_client_pcm_segment(
                                ack_branch,
                                speaker_segment_id=str(token.speaker_segment_id),
                                token_packets=token_packets,
                            )
                        if marker is not None and not await send_ws(marker, token):
                            return
                    update_response_runtime_metrics(response_id, "client_consumed_pcm", {
                        "alignment": "complete_closed_tts_unit_only",
                        "text_owned_tts_units_available": False,
                        "logical_client_consumption_only": True,
                        "confirmed_segments": list(ack_branch.get("client_consumed_segments", ())),
                        "confirmed_exclusive_kv_cursor": ack_branch.get(
                            "client_consumed_prefix_exclusive_cursor"
                        ),
                    })
                if response_uses_direct_audio and not archive_assistant_finalized["value"]:
                    result_runtime_metrics = result.get("runtime_metrics")
                    completion_metrics = (
                        result_runtime_metrics.get("response_completion")
                        if isinstance(result_runtime_metrics, Mapping) else None
                    )
                    acoustic_limit_reached = bool(
                        isinstance(completion_metrics, Mapping)
                        and completion_metrics.get("acoustic_limit_reached") is True
                    )
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
                    response_finish = result.get("finish_reason")
                    if not isinstance(response_finish, str) and isinstance(
                        completion_metrics, Mapping
                    ):
                        response_finish = completion_metrics.get("finish_reason")
                    nonunified_v2_text_complete = bool(
                        not direct_audio_v2_response
                        or unified_kv_enabled
                        or (
                            isinstance(completion_metrics, Mapping)
                            and completion_metrics.get("eos_detected") is True
                            and not acoustic_limit_reached
                            and response_finish != "length"
                        )
                    )
                    v1_source_rows = (
                        ack_branch.get("nonunified_v1_source_tokens")
                        if isinstance(ack_branch, dict) else None
                    )
                    v1_source_frontier = (
                        ack_branch.get("response_event_ledger").source_frontier_snapshot()
                        if isinstance(ack_branch, dict)
                        and callable(getattr(
                            ack_branch.get("response_event_ledger"),
                            "source_frontier_snapshot",
                            None,
                        ))
                        else {}
                    )
                    v1_final_source_sequence = (
                        int(v1_source_rows[-1]["source_token_seq"])
                        if isinstance(v1_source_rows, (tuple, list))
                        and v1_source_rows
                        and isinstance(v1_source_rows[-1], Mapping)
                        else -1
                    )
                    nonunified_v1_text_complete = bool(
                        not controller_guided_say_response
                        or unified_kv_enabled
                        or (
                            isinstance(completion_metrics, Mapping)
                            and completion_metrics.get("eos_detected") is True
                            and not acoustic_limit_reached
                            and response_finish != "length"
                            and v1_final_source_sequence >= 0
                            and int(v1_source_frontier.get("generated_source_sequence", -1))
                            == v1_final_source_sequence
                            and int(v1_source_frontier.get("tts_consumed_source_seq", -1))
                            == v1_final_source_sequence
                        )
                    )
                    unified_speak_valid = bool(
                        not unified_kv_enabled
                        or (
                            action == ControlAction.SPEAK
                            and isinstance(v2_metrics, dict)
                            and (
                                (
                                    v2_metrics.get("response_protocol") == AUDIO_RESPONSE_PROTOCOL_V2
                                    and v2_metrics.get("status") == "valid"
                                )
                                or (
                                    v2_metrics.get("response_protocol") == CONTROLLER_GUIDED_SAY_V1
                                    and v2_metrics.get("status") in {"valid", "controller_action_fixed"}
                                )
                            )
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
                        and nonunified_v2_text_complete
                        and nonunified_v1_text_complete
                    )
                    ledger = ack_branch.get("response_event_ledger") if ack_branch is not None else response_event_ledger
                    if ledger is not None:
                        ledger.close_response(
                            intended_text=(emitted_assistant_text or result_text),
                            finish_reason=(response_finish if isinstance(response_finish, str) else None),
                            completed=assistant_completed,
                            cancelled=bool(result.get("cancelled") or yielded or not assistant_completed),
                        )
                    ledger_projection = ledger.project() if ledger is not None else None
                    response_has_pcm = bool(
                        ledger_projection is not None
                        and int(ledger_projection.pcm_samples_sent) > 0
                    )
                    defer_acoustic_history = response_has_pcm
                    history_branch = unified_native_branch or ack_branch or playback_branch
                    if defer_acoustic_history and isinstance(history_branch, dict):
                        history_branch["deferred_acoustic_history_commit"] = {
                            "archive_id": archive_id_holder["archive_id"],
                            "response_done_at_monotonic_ns": response_done_at_monotonic_ns,
                        }
                        history_branch["acoustic_limit_reached"] = acoustic_limit_reached
                    assistant_status = (
                        "completed" if assistant_completed
                        else "cancelled" if result.get("cancelled") or yielded
                        else "absent"
                    )
                    committed_assistant_text = (
                        emitted_assistant_text if assistant_completed else ""
                    )
                    archive_id_value = archive_id_holder["archive_id"]
                    if defer_acoustic_history:
                        archived_assistant = False
                        assistant_status = "awaiting_client_consumption_ack"
                    elif assistant_completed:
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
                        "history_prefix_source": (
                            "pending_client_consumption_ack"
                            if defer_acoustic_history else "response_completion"
                        ),
                        "response_done_at_monotonic_ns": response_done_at_monotonic_ns,
                    })
                response_finish_reason = result.get("finish_reason")
                if not isinstance(response_finish_reason, str):
                    result_runtime_metrics = result.get("runtime_metrics")
                    completion_metrics = (
                        result_runtime_metrics.get("response_completion")
                        if isinstance(result_runtime_metrics, Mapping) else None
                    )
                    response_finish_reason = (
                        completion_metrics.get("finish_reason")
                        if isinstance(completion_metrics, Mapping) else None
                    )
                done_event = {
                    "type": "response.done",
                    "response_id": response_id,
                    "response_done_at_monotonic_ns": response_done_at_monotonic_ns,
                    "runtime_metrics": runtime_metrics_event(response_id, result),
                }
                if isinstance(response_finish_reason, str):
                    done_event["finish_reason"] = response_finish_reason
                if first_tts_audio_chunk_pcm_at_ns is not None:
                    done_event["t_first_audio_chunk_pcm_at_ns"] = int(
                        first_tts_audio_chunk_pcm_at_ns
                    )
                if result.get("status") == "no_speech":
                    done_event["status"] = "no_speech"
                    done_event["audio_samples"] = 0
                elif result.get("status") == "deferred_until_endpoint":
                    done_event["status"] = "deferred_until_endpoint"
                if result.get("status") == "backchannel_say_rejected":
                    response_done_sent = await send_ws({
                        "type": "response.cancelled",
                        "response_id": response_id,
                        "reason": "backchannel_rejected",
                        "finish_reason": "cancelled",
                    }, token)
                else:
                    response_done_sent = await send_ws(done_event, token)
                    if not response_done_sent and not sess.is_generation_current(token):
                        final_runtime_metrics = runtime_metrics_event(response_id, result)
                        completion_metrics = final_runtime_metrics.get("response_completion")
                        cancellation_reason = (
                            completion_metrics.get("cancellation_reason")
                            if isinstance(completion_metrics, Mapping)
                            else None
                        )
                        if not isinstance(cancellation_reason, str) or not cancellation_reason:
                            with sess._generation_lock:
                                cancellation_reason = str(
                                    sess.stop_reason or "generation_replaced_before_completion"
                                )
                        await _send_response_completion_metrics(
                            send_ws,
                            response_id,
                            final_runtime_metrics,
                            cancellation_reason=cancellation_reason,
                        )
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
                        "finish_reason": "cancelled",
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
            except LiveSayCancelled as exc:
                logger.info(
                    "[realtime %s %s] live say cancelled cleanly: %s",
                    sess.session_id[:8], response_id, exc,
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
                release_listener_burst_pause("response_finally")
                if user_turn_onset_frame_id is not None and not user_turn_completed:
                    sess.cancel_user_onset_response(user_turn_onset_frame_id)
                history_branch = unified_native_branch or playback_branch
                response_ledger = history_branch.get("response_event_ledger")
                if response_ledger is not None:
                    ledger_snapshot = response_ledger.project()
                    if ledger_snapshot.pcm_samples_sent > 0 and not ledger_snapshot.response_done:
                        with sess._generation_lock:
                            current_segment = sess.speaker_segment
                            fallback_intended_text = (
                                str(current_segment.emitted_text)
                                if current_segment is not None
                                and current_segment.segment_id == token.speaker_segment_id
                                else ""
                            )
                        response_ledger.close_response(
                            intended_text=fallback_intended_text,
                            finish_reason="cancelled",
                            completed=False,
                            cancelled=True,
                        )
                        history_branch.setdefault("deferred_acoustic_history_commit", {
                            "archive_id": archive_id_holder["archive_id"],
                            "response_done_at_monotonic_ns": time.monotonic_ns(),
                        })
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
                    history_branch = unified_native_branch or playback_branch
                    defer_acoustic_history = bool(
                        isinstance(history_branch.get("deferred_acoustic_history_commit"), Mapping)
                    )
                    if not archive_assistant_finalized["value"] and not defer_acoustic_history:
                        spoken_prefix = ""
                        _spoken_position = None
                        if (
                            direct_audio_v2_response or controller_guided_say_response
                        ) and unified_native_branch is not None:
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
                if (
                    response_output_cancel_reason.get("value") == "yield"
                    and unified_native_branch is not None
                ):
                    _cancel_response_branch_playback(unified_native_branch)
                if unified_native_branch is not None:
                    if _response_branch_has_pending_pcm(unified_native_branch):
                        # Keep the shared queue and response ownership open until
                        # emitted PCM reaches its exact client-consumed frontier.
                        # The independent receive loop remains free to process ACKs.
                        finalize_task = asyncio.create_task(
                            finalize_unified_native_cache_after_playback()
                        )
                        unified_native_branch["deferred_playback_finalizer"] = finalize_task
                        _attach_terminal_response_finalizer(
                            app.state.terminal_response_archive,
                            sess,
                            response_id,
                            unified_native_branch,
                            finalize_task,
                        )
                    else:
                        finalize_task = asyncio.create_task(finalize_unified_native_cache(
                            history_committed=(
                                bool(unified_native_branch.get("history_committed"))
                                and unified_native_branch.get("playback_cancelled") is not True
                            )
                        ))
                        unified_native_branch["deferred_playback_finalizer"] = finalize_task
                        _attach_terminal_response_finalizer(
                            app.state.terminal_response_archive,
                            sess,
                            response_id,
                            unified_native_branch,
                            finalize_task,
                        )
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
                    history_branch = unified_native_branch or playback_branch
                    if isinstance(
                        history_branch.get("deferred_acoustic_history_commit"), Mapping
                    ):
                        finalize_task = asyncio.create_task(
                            finalize_nonunified_audio_history_after_playback()
                        )
                        history_branch["deferred_playback_finalizer"] = finalize_task
                        _attach_terminal_response_finalizer(
                            app.state.terminal_response_archive,
                            sess,
                            response_id,
                            history_branch,
                            finalize_task,
                        )
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
                "voice": sess.voice,
                "response_profile": sess.response_profile,
                "voice_conditioning_mode": (
                    sess.voice_conditioning.get("mode")
                    if isinstance(sess.voice_conditioning, Mapping)
                    else getattr(sess.voice_conditioning, "mode", None)
                ),
                "input_audio_buffer_capacity_frames": sess.audio_ingress.max_frames,
                "runtime_metrics": session_runtime_metrics,
                **_policy_session_metadata(sess.turn_policy, sess),
            })
            client_consumed_pcm_last_sequence = -1
            client_consumed_pcm_seen: dict[int, tuple[Any, ...]] = {}
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
                    sequence = ev.get("sequence")
                    if (
                        branch is not None
                        and branch.get("playback_cancelled") is True
                        and isinstance(
                            branch.get("playback_ack_final_deadline_monotonic"),
                            (int, float),
                        )
                        and time.monotonic()
                        > float(branch["playback_ack_final_deadline_monotonic"])
                    ):
                        with playback_ack_branches_lock:
                            if playback_ack_branches.get(str(ack_response_id or "")) is branch:
                                playback_ack_branches.pop(str(ack_response_id or ""), None)
                        branch = None
                    canonical_ack = _canonical_client_consumed_pcm_ack(ev)
                    cached_duplicate = (
                        ev.get("ack_version", 1) == 2
                        and isinstance(sequence, int)
                        and not isinstance(sequence, bool)
                        and client_consumed_pcm_seen.get(sequence) == canonical_ack
                    )
                    branch_duplicate = _client_consumed_pcm_ack_duplicate_status(
                        branch, ev
                    )
                    duplicate = bool(branch is not None and branch_duplicate is True)
                    if branch is None:
                        accepted = False
                        is_current = False
                    elif branch_duplicate is not None:
                        accepted = bool(branch_duplicate)
                        is_current = False
                    elif cached_duplicate:
                        # The connection cache is only an accelerator; an ACK
                        # absent from its retained response branch is unknown.
                        accepted = False
                        is_current = False
                    else:
                        is_current = bool(
                            branch is not None
                            and sess.is_generation_current(
                                generation_id=ev.get("generation_id")
                                if isinstance(ev.get("generation_id"), int)
                                and not isinstance(ev.get("generation_id"), bool)
                                else None,
                                response_id=str(ack_response_id or ""),
                            )
                        )
                        generation_matches_branch = bool(
                            branch is not None
                            and isinstance(ev.get("generation_id"), int)
                            and not isinstance(ev.get("generation_id"), bool)
                            and int(ev["generation_id"])
                            == int(branch.get("session_generation_id", -1))
                        )
                        final_deadline = (
                            branch.get("playback_ack_final_deadline_monotonic")
                            if isinstance(branch, dict) else None
                        )
                        late_cancelled_ack = bool(
                            branch is not None
                            and generation_matches_branch
                            and branch.get("playback_cancelled") is True
                            and ev.get("ack_version") == 2
                            and isinstance(final_deadline, (int, float))
                            and time.monotonic() <= float(final_deadline)
                        )
                        sequence_valid = bool(
                            isinstance(sequence, int)
                            and not isinstance(sequence, bool)
                            and sequence > client_consumed_pcm_last_sequence
                        )
                        if branch is None or not generation_matches_branch or not sequence_valid or not (
                            is_current or late_cancelled_ack
                        ):
                            accepted = False
                        else:
                            with branch.get("spoken_prefix_lock", nullcontext()):
                                accepted = _record_client_consumed_pcm(branch, ev)
                            if accepted:
                                client_consumed_pcm_last_sequence = int(sequence)
                                client_consumed_pcm_seen[int(sequence)] = canonical_ack
                                while len(client_consumed_pcm_seen) > 256:
                                    client_consumed_pcm_seen.pop(
                                        next(iter(client_consumed_pcm_seen))
                                    )
                    if branch is None:
                        logger.warning(
                            "[playback_ack_reject] branch_found=false ack_response_id=%s ev_gen=%s sess_gen=%s",
                            ack_response_id,
                            ev.get("generation_id"),
                            sess.session_generation_id,
                        )
                    if accepted and branch is not None and not duplicate:
                        ack_event = branch.get("playback_ack_event")
                        if isinstance(ack_event, asyncio.Event):
                            ack_event.set()
                    if not accepted:
                        await send_ws({
                            "type": "error",
                            "response_id": str(ack_response_id or ""),
                            "error": {
                                "code": "STALE_CLIENT_CONSUMPTION_ACK",
                            "message": "client_consumed_pcm identity, sequence, or cumulative emitted range is stale or mismatched",
                            },
                        })
                    else:
                        segment_row = (
                            branch.get("playback_segments", {}).get(str(ev["speaker_segment_id"]))
                            if isinstance(branch, dict)
                            and isinstance(branch.get("playback_segments"), Mapping)
                            else None
                        )
                        segment_metrics = None
                        if isinstance(segment_row, Mapping):
                            segment_metrics = {
                                key: segment_row.get(key)
                                for key in (
                                    "original_token_ids",
                                    "original_token_cursor_start",
                                    "original_token_cursor_end",
                                    "start_sample",
                                    "end_sample",
                                    "emitted_end_sample",
                                    "pcm_samples_emitted",
                                    "pcm_samples_consumed",
                                    "pcm_consumed_start_sample",
                                    "pcm_consumed_end_sample",
                                    "termination_reason",
                                    "client_consumed",
                                )
                            }
                        update_response_runtime_metrics(str(ack_response_id), "client_consumed_pcm", {
                            "status": "accepted",
                            "logical_client_consumption_only": True,
                                "ack_version": int(ev.get("ack_version", 1)),
                                "final": bool(ev.get("final", True)),
                                "duplicate": bool(duplicate),
                            "response_id": str(ack_response_id),
                            "generation_id": int(ev["generation_id"]),
                            "speaker_segment_id": str(ev["speaker_segment_id"]),
                            "start_sample": int(ev["start_sample"]),
                            "end_sample": int(ev["end_sample"]),
                            "sample_rate": int(ev["sample_rate"]),
                            "sequence": int(ev["sequence"]),
                            "consumption_semantics": "logical_client_device_buffer",
                            "confirmed_exclusive_kv_cursor": (
                                branch.get("client_consumed_prefix_exclusive_cursor")
                                if isinstance(branch, dict) else None
                            ),
                            "audio_segment": segment_metrics,
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
                    try:
                        requested_response_profile = _normalize_response_profile(
                            cfg.get(
                                "response_profile",
                                extra.get("response_profile", sess.response_profile),
                            )
                        )
                    except ValueError as exc:
                        await send_ws({
                            "type": "error",
                            "error": {"code": "INVALID_SESSION_UPDATE", "message": str(exc)},
                        })
                        continue
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
                        sess.response_profile = requested_response_profile
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
                        "response_profile": sess.response_profile,
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
                        sess._append_history_items_locked([{
                            "role": item.get("role", "user"),
                            "content": _message_text(item.get("content", "")),
                        }])
                        sess._history_mutation_sequence += 1
                        sess.semantic_context_revision += 1
                    sess.notify_history_context_changed()
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
                                with listener_ingress_publication():
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
                    response_profile_override = None
                    profile_was_supplied = "response_profile" in ev
                    profile_value = ev.get("response_profile")
                    nested_response = ev.get("response")
                    if (
                        not profile_was_supplied
                        and isinstance(nested_response, Mapping)
                        and "response_profile" in nested_response
                    ):
                        profile_was_supplied = True
                        profile_value = nested_response.get("response_profile")
                    if profile_was_supplied:
                        try:
                            response_profile_override = _normalize_response_profile(profile_value)
                        except ValueError as exc:
                            await send_ws({
                                "type": "error",
                                "error": {"code": "INVALID_RESPONSE_PROFILE", "message": str(exc)},
                            })
                            continue
                    controller_owned_auto_commit = bool(
                        et == "input_audio_buffer.commit"
                        and direct_audio_candidate
                        and sess.auto_response
                        and response_protocol == CONTROLLER_GUIDED_SAY_V1
                    )
                    with sess._generation_lock:
                        response_busy = sess.active_response or sess.policy_response_pending
                    response_running = bool(
                        (response_task and not response_task.done()) or response_busy
                    )
                    if controller_owned_auto_commit:
                        # V1 is driven by the live listener/controller.  A
                        # client commit acknowledges its continuous input
                        # boundary; it must not consume the pending PCM or
                        # create an action-less response, especially while an
                        # earlier controller response is still active.
                        sess.touch()
                        continue
                    if response_running:
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
                        if not controller_owned_auto_commit:
                            sess.reset_policy_audio(committed_sample_count=committed_sample_count)
                            sess.reset_turn_policy()
                        sess.policy_response_pending = not controller_owned_auto_commit
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
                    if not controller_owned_auto_commit:
                        response_task = asyncio.create_task(
                            produce_response(
                                committed_audio,
                                action=(
                                    ControlAction.SPEAK
                                    if manual_commit_speaks
                                    else None
                                ),
                                intent_origin=manual_audio_context,
                                response_profile=response_profile_override,
                            )
                        )
                elif et == "session.close": break
        except (WebSocketDisconnect, asyncio.CancelledError): pass
        finally:
            closing = True
            try:
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
            finally:
                try:
                    sessions.close(sess.session_id)
                finally:
                    try:
                        with audio_archive_lock:
                            pending_archive_futures = list(audio_archive_futures.values())
                            audio_archive_futures.clear()
                        for pending_archive_future in pending_archive_futures:
                            pending_archive_future.cancel()
                    finally:
                        try:
                            finish_duplex_session_gc(sess.session_id)
                        finally:
                            gpu_admission_lease.release()

    @app.on_event("startup")
    async def _start_shared_listener_runtime() -> None:
        if not shared_listener_microbatch_enabled:
            return
        try:
            await _ensure_shared_listener_runtime()
        except Exception:
            # Let health report the unavailable runtime and its cause; the
            # service remains explicitly unready until initialization succeeds.
            logger.exception("service startup could not initialize shared listener runtime")

    @app.on_event("shutdown")
    async def _shutdown_shared_listener_runtime() -> None:
        async with shared_listener_lifecycle_lock:
            shared_listener_runtime_status["shutting_down"] = True
            try:
                await _stop_shared_listener_runtime_locked(timeout_s=30.0)
            except Exception as exc:
                shared_listener_runtime_status["error"] = (
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                )
                logger.exception("service shutdown failed to stop shared listener runtime")
    return app


def run(engine: Any, *, host: str = "0.0.0.0", port: int = 8092) -> None:
    import uvicorn
    uvicorn.run(create_app(engine), host=host, port=port, log_level="info")

__all__ = ["apply_speaker_gate_event", "create_app", "run"]
