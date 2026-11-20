"""Causal direct-audio snapshot and voice-conditioning implementations."""
from __future__ import annotations

import copy
import sys
from collections import Counter
from typing import Any, Mapping

import numpy as np
import torch

from duplex_kernel.audio_response_conditioning import DEFAULT_MAX_AUDIO_MS, causal_audio_window
from duplex_kernel.config import DEFAULT_VOICE, USER_SAMPLE_RATE
from duplex_kernel.server_state import POLICY_FRAME_MS, _verify_v9_training_reference
from duplex_kernel.session import DuplexSession
from duplex_kernel.turn_policy import ControlAction
from duplex_kernel.voices import resolve_voice_conditioning



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

def _snapshot_speaking_branch_context(
    direct_audio_context: dict[str, Any] | None,
    listener_state: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Copy SPEAK metadata and acoustic priors independently of its KV fork."""
    direct_copy = copy.deepcopy(direct_audio_context) if direct_audio_context is not None else None
    lock = getattr(listener_state, "_lock", None)
    if lock is None:
        raise RuntimeError("speaking Q2 branch requires a lockable listener acoustic state")

    def copy_history(value: Any) -> tuple[np.ndarray, ...]:
        return tuple(np.array(item, copy=True) for item in tuple(value or ()))

    with lock:
        acoustic = {
            "state_epoch_id": str(getattr(listener_state, "state_epoch_id", "")),
            "listener_state_version": int(getattr(listener_state, "listener_state_version", 0)),
            "next_frame_id": int(getattr(listener_state, "next_frame_id", 0)),
            "gemma_lm_context_start_frame": int(
                getattr(listener_state, "gemma_lm_context_start_frame", 0)
            ),
            "gemma_lm_source_frame_ids": tuple(
                int(value) for value in getattr(listener_state, "gemma_lm_source_frame_ids", ())
            ),
            "gemma_lm_last_source_frame_id": getattr(
                listener_state, "gemma_lm_last_source_frame_id", None
            ),
            "user_pcm_history": copy_history(getattr(listener_state, "user_pcm_history", ())),
            "assistant_pcm_history": copy_history(
                getattr(listener_state, "assistant_pcm_history", ())
            ),
            "user_mel_history": copy_history(getattr(listener_state, "user_mel_history", ())),
            "assistant_mel_history": copy_history(
                getattr(listener_state, "assistant_mel_history", ())
            ),
            "user_pcm_tail": (
                None if getattr(listener_state, "user_pcm_tail", None) is None
                else np.array(listener_state.user_pcm_tail, copy=True)
            ),
            "assistant_pcm_tail": (
                None if getattr(listener_state, "assistant_pcm_tail", None) is None
                else np.array(listener_state.assistant_pcm_tail, copy=True)
            ),
        }
    return direct_copy, acoustic

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
    server_module = sys.modules.get("duplex_kernel.server")
    voice_resolver = getattr(
        server_module, "resolve_voice_conditioning", resolve_voice_conditioning
    )
    if enforce_training_voice:
        default_conditioning = getattr(talker, "default_voice_conditioning", None)
        if not isinstance(default_conditioning, dict):
            raise ValueError("il conditioning x-vector training V9 non è caricato")
        conditioning = dict(default_conditioning)
        reference_path = _verify_v9_training_reference(conditioning)
        requested_profile = voice_resolver(
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
        conditioning = voice_resolver(voice, voice_style=voice_style)
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
