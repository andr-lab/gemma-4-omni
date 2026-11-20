"""Small silence-skip, audio serialization, and packet helpers."""
from __future__ import annotations

import base64
import io
from contextlib import nullcontext
from typing import Any, Callable

import numpy as np
import torch

from duplex_kernel.config import CODEC_SAMPLE_RATE
from duplex_kernel.session import DuplexSession
from duplex_kernel.turn_policy import ControlState

_DEFERRED_NATIVE_SPEAK_SOURCE = "thinker_response_action_deferred_to_endpoint"

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

def _audio_packet_matches_emission_owner(
    packet: Any,
    *,
    generation_id: int,
    speaker_segment_id: str,
) -> bool:
    """Reject buffered PCM that does not belong to this response segment."""
    try:
        return (
            int(packet.generation_id) == int(generation_id)
            and str(packet.speaker_segment_id) == str(speaker_segment_id)
            and bool(str(packet.speaker_segment_id))
        )
    except (AttributeError, TypeError, ValueError):
        return False

async def _release_pcm_packet_then_ack(
    packet: Any,
    *,
    validate: Callable[[Any], bool] | None = None,
    release: Callable[[Any], Any],
    acknowledge: Callable[[Any], Any],
) -> bool:
    """Run the ACK callback only after the PCM release callback succeeds."""
    if validate is not None and validate(packet) is not True:
        return False
    if await release(packet) is not True:
        return False
    return acknowledge(packet) is not False
