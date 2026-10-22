from __future__ import annotations

import numpy as np
import pytest

from duplex_kernel.server import (
    _apply_direct_replan_transaction,
    _claim_response_cancel_notification,
)
from duplex_kernel.session import DuplexSession
from duplex_kernel.turn_policy import ControlAction, ControlState, PolicyDecision, SpeakerGate


SAMPLES_PER_POLICY_FRAME = 1280


def _active_session() -> DuplexSession:
    session = DuplexSession("transactional-replan")
    session.control_state = ControlState.SPEAKING
    session.session_generation_id = 7
    session.response_id = "resp-active"
    session.speaker_segment_id = "segment-active"
    session.active_response = True
    session.policy_response_pending = True
    session.pending_user_onset_frame_id = 10
    session.stop_requested = False
    session.response_cancel_notified = False
    return session


def _decision() -> PolicyDecision:
    return PolicyDecision(
        action=ControlAction.REPLAN,
        frame_id=12,
        confidence=0.9,
        generation_id=7,
        source="test",
        session="transactional-replan",
    )


def test_invalid_replan_snapshot_does_not_mutate_active_response_or_pending_turn() -> None:
    session = _active_session()
    session.append_policy_audio(10, np.zeros(SAMPLES_PER_POLICY_FRAME, dtype=np.float32))
    session.append_policy_audio(12, np.zeros(SAMPLES_PER_POLICY_FRAME, dtype=np.float32))

    with pytest.raises(ValueError, match="snapshot PCM"):
        _apply_direct_replan_transaction(
            session,
            SpeakerGate(),
            _decision(),
            first_frame_id=10,
            last_frame_id=12,
            user_active=True,
        )

    assert session.session_generation_id == 7
    assert session.control_state is ControlState.SPEAKING
    assert session.active_response is True
    assert session.policy_response_pending is True
    assert session.pending_user_onset_frame_id == 10
    assert session.stop_requested is False


def test_valid_active_replan_reserves_exact_pcm_before_invalidating_generation() -> None:
    session = _active_session()
    frames = [np.full(SAMPLES_PER_POLICY_FRAME, i, dtype=np.float32) for i in (10, 11, 12)]
    for frame_id, pcm in zip((10, 11, 12), frames):
        session.append_policy_audio(frame_id, pcm)

    gate_event, context = _apply_direct_replan_transaction(
        session,
        SpeakerGate(),
        _decision(),
        first_frame_id=10,
        last_frame_id=12,
        user_active=True,
    )

    assert gate_event.accepted is True
    assert gate_event.action is ControlAction.REPLAN
    assert gate_event.stale_generation_id == 7
    assert session.session_generation_id == 8
    assert session.control_state is ControlState.STOPPING
    assert context["_direct_audio_first_frame_id"] == 10
    assert context["_direct_audio_last_frame_id"] == 12
    assert context["_direct_audio_sample_count"] == 3 * SAMPLES_PER_POLICY_FRAME
    np.testing.assert_array_equal(context["_direct_audio_pcm"], np.concatenate(frames))


def test_long_replan_uses_only_contiguous_bounded_tail_with_true_origin() -> None:
    session = _active_session()
    session.pending_user_onset_frame_id = 0
    for frame_id in range(401):
        session.append_policy_audio(
            frame_id,
            np.full(SAMPLES_PER_POLICY_FRAME, frame_id % 7, dtype=np.float32),
        )
    decision = _decision()
    decision = PolicyDecision(
        action=decision.action,
        frame_id=400,
        confidence=decision.confidence,
        generation_id=7,
        source=decision.source,
        session=decision.session,
    )

    gate_event, context = _apply_direct_replan_transaction(
        session,
        SpeakerGate(),
        decision,
        first_frame_id=0,
        last_frame_id=400,
        user_active=True,
    )

    assert gate_event.accepted is True
    assert context["_direct_audio_requested_first_frame_id"] == 0
    assert context["_direct_audio_first_frame_id"] == 26
    assert context["_direct_audio_prefix_frames_clipped"] == 26
    assert context["_direct_audio_last_frame_id"] == 400
    assert context["_direct_audio_sample_count"] == 375 * SAMPLES_PER_POLICY_FRAME


def test_cancel_terminal_claim_is_exactly_once_for_response() -> None:
    session = _active_session()

    assert _claim_response_cancel_notification(session, "resp-active") is True
    assert _claim_response_cancel_notification(session, "resp-active") is False
    assert _claim_response_cancel_notification(session, "other-response") is False
