"""Realtime policy preparation and Control6 transition helpers."""
from __future__ import annotations

import logging
import queue
import time
from typing import Any, Callable

import numpy as np

from duplex_kernel.backchannel_guard import peer_audio_event_key_for_frame
from duplex_kernel.gemma_native_control6_policy import (
    GemmaNativeControl6Policy,
    is_controller6_policy,
    legal_control_actions as native_control6_legal_actions,
)
from duplex_kernel.server_silence_skip import (
    _DEFERRED_NATIVE_SPEAK_SOURCE,
    _advance_pending_speak_on_new_onset,
    _user_turn_is_unanswered,
    _yield_admission_rejection,
)
from duplex_kernel.server_state import POLICY_FRAME_SAMPLES
from duplex_kernel.server_audio_snapshot import (
    _native_control6_speaker_active,
    _pending_pcm_response_branches,
)
from duplex_kernel.turn_policy import ControlAction, PolicyInput


logger = logging.getLogger("duplex_kernel.server")


def _prepare_policy_tick_state_impl(
    frame: Any,
    tick: Any,
    *,
    sess: Any,
    speech_tracker: Any,
    policy_cooldown_lock: Any,
    response_audio_admission: Any,
    policy_causality_lock: Any,
    pending_response_speak_handoffs: Any,
    begin_transcript_utterance: Callable[[int], Any],
    pending_speak: Any,
    peer_audio_event_key: Any,
    latest_policy_frame_id: int,
    partial_last_snapshot_frame: Any,
) -> tuple[dict[str, Any], Any, Any, int, Any]:
    """Apply per-frame scheduler state once, before either listener or policy work."""
    with policy_cooldown_lock:
        latest_policy_frame_id = max(latest_policy_frame_id, int(frame.frame_id))
    new_voice_onset, speech_endpoint = speech_tracker.observe(
        int(frame.frame_id), frame.pcm
    )
    # This is the raw class returned by the current-frame VAD decision. Do not
    # substitute speech-run state here: a quiet frame may occur inside the
    # endpoint hangover while still being explicitly nonvoice input.
    input_frame_voiced = bool(
        speech_tracker.last_voiced_frame_id is not None
        and int(speech_tracker.last_voiced_frame_id) == int(frame.frame_id)
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
        "input_frame_voiced": input_frame_voiced,
        "policy": policy,
        "policy_revision": int(policy_revision),
        "turn_generation_id": turn_generation_id,
        "audio_appended": policy is not None,
        "native_user_previous_pcm": native_user_previous_pcm,
        "native_user_previous_frame_id": native_user_previous_frame_id,
    }
    tick._policy_tick_prepared = prepared
    return (
        prepared,
        pending_speak,
        peer_audio_event_key,
        latest_policy_frame_id,
        partial_last_snapshot_frame,
    )


def _shared_native_policy_context_impl(
    frame: Any,
    tick: Any,
    *,
    sess: Any,
    prepare_policy_tick_state: Callable[[Any, Any], dict[str, Any]],
    speech_tracker: Any,
    playback_ack_branches: Any,
    playback_ack_branches_lock: Any,
    shared_listener_bindings: Any,
) -> Any | None:
    prepared = prepare_policy_tick_state(frame, tick)
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
    endpoint_complete = bool(
        speech_tracker is not None
        and getattr(speech_tracker, "speech_complete", False)
        and not getattr(speech_tracker, "voice_run_active", True)
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
            assistant_prior_frame_id=(
                None if assistant_prior is None else assistant_prior.frame_id
            ),
        )
    else:
        phase_t = native_control6_phase_t(
            listener_frame_id=frame_id,
            user_pcm_current=frame.pcm,
            assistant_pcm_prior=(None if assistant_prior is None else assistant_prior.pcm),
            assistant_prior_frame_id=(
                None if assistant_prior is None else assistant_prior.frame_id
            ),
            assistant_pcm_preceding=(
                None if assistant_preceding is None else assistant_preceding.pcm
            ),
            assistant_preceding_frame_id=(
                None if assistant_preceding is None else assistant_preceding.frame_id
            ),
        )
    state = PolicyInput(
        session_id=sess.session_id,
        frame_id=frame_id,
        user_pcm=frame.pcm,
        speaker_active=speaker_active,
        generation_id=turn_generation_id,
        user_turn_onset_frame_id=(
            int(pending_onset)
            if user_turn_pending and pending_onset is not None
            else None
        ),
        user_endpoint_onset_frame_id=user_endpoint_onset_frame_id,
        user_endpoint_frame_id=user_endpoint_frame_id,
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
        native_control_no_input_tick=False,
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
        # This is the endpoint signal for this exact PCM frame. Keep it
        # separate from peer_turn_active, whose policy meaning remains intact.
        speech_endpoint=bool(prepared["speech_endpoint"]),
        new_voice_onset=bool(prepared["new_voice_onset"]),
        voice_run_onset_frame_id=(
            None
            if speech_tracker.run_onset_frame_id is None
            else int(speech_tracker.run_onset_frame_id)
        ),
    )


def _classify_control6_policy_action(policy: Any, action: Any) -> ControlAction | None:
    if not is_controller6_policy(policy):
        return None
    try:
        return action if isinstance(action, ControlAction) else ControlAction(action)
    except (TypeError, ValueError):
        return None


def _control6_transition_rejection(
    action: Any,
    *,
    active_for_causal_gate: bool,
    emitted_audio_frames: int,
    has_post_watermark_voice: bool,
    has_post_response_voice: bool,
) -> str | None:
    if not active_for_causal_gate:
        return None
    if action == ControlAction.YIELD:
        return _yield_admission_rejection(
            active_response=True,
            emitted_audio_frames=emitted_audio_frames,
            has_post_watermark_voice=has_post_watermark_voice,
        )
    if action == ControlAction.REPLAN and not has_post_response_voice:
        return "no_new_user_audio"
    return None
