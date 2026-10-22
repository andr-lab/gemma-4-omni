from __future__ import annotations

import numpy as np
import pytest

from duplex_kernel.server import (
    _manual_commit_speak_owner_for_span,
    _snapshot_direct_policy_audio,
)
from duplex_kernel.session import DuplexSession
from duplex_kernel.turn_policy import ControlAction


def test_direct_audio_snapshot_uses_full_completed_turn_and_absolute_frames() -> None:
    session = DuplexSession("direct-audio-speak")
    session.append_policy_audio(11, np.full(1280, 11, dtype=np.float32))
    session.append_policy_audio(12, np.full(1280, 12, dtype=np.float32))

    result = _snapshot_direct_policy_audio(
        session, action=ControlAction.SPEAK, first_frame_id=11, last_frame_id=12,
    )

    assert result["_direct_audio_pcm"].size == 2560
    assert result["_direct_audio_window"]["audio_start_ms"] == 880
    assert result["_direct_audio_window"]["audio_end_ms"] == 1040
    assert result["_direct_audio_active"] is False
    assert list(session.policy_audio_frame_ids) == [11, 12]


def test_direct_backchannel_snapshot_is_active_and_never_committed() -> None:
    session = DuplexSession("direct-audio-backchannel")
    session.append_policy_audio(20, np.full(1280, 20, dtype=np.float32))
    session.append_policy_audio(21, np.full(1280, 21, dtype=np.float32))

    result = _snapshot_direct_policy_audio(
        session, action=ControlAction.BACKCHANNEL, first_frame_id=20, last_frame_id=21,
    )

    assert result["_direct_audio_pcm"].size == 2560
    assert result["_direct_audio_active"] is True
    assert result["_direct_audio_end_ms"] == 1760
    assert not session.history
    assert not session.pending_audio_history_archive


def test_direct_audio_fails_closed_on_missing_frame() -> None:
    session = DuplexSession("direct-audio-gap")
    session.append_policy_audio(2, np.ones(1280, dtype=np.float32))
    session.append_policy_audio(4, np.ones(1280, dtype=np.float32))

    with pytest.raises(ValueError, match="snapshot PCM e span causale"):
        _snapshot_direct_policy_audio(
            session, action=ControlAction.SPEAK, first_frame_id=2, last_frame_id=4,
        )


def test_direct_audio_snapshot_rejection_reports_bounded_retention_diagnostics_only() -> None:
    from duplex_kernel.server import _direct_audio_snapshot_event_fields

    session = DuplexSession("direct-audio-snapshot-diagnostics")
    frame_pcm = np.ones(1280, dtype=np.float32)
    session.append_policy_audio(230, frame_pcm)
    session.append_policy_audio(231, frame_pcm)
    session.append_policy_audio(233, frame_pcm)
    assert session.take_policy_audio_through_frame(230).size == 1280

    with pytest.raises(ValueError, match="snapshot PCM e span causale non allineati") as caught:
        _snapshot_direct_policy_audio(
            session,
            action=ControlAction.REPLAN,
            first_frame_id=230,
            last_frame_id=233,
            user_active=True,
        )

    fields = _direct_audio_snapshot_event_fields(caught.value)
    assert fields == {
        "direct_audio_snapshot_onset_frame_id": 230,
        "direct_audio_snapshot_requested_first_frame_id": 230,
        "direct_audio_snapshot_requested_last_frame_id": 233,
        "direct_audio_snapshot_active": True,
        "direct_audio_snapshot_consumed_through_frame_id": 230,
        "direct_audio_snapshot_retained_first_frame_id": 231,
        "direct_audio_snapshot_retained_last_frame_id": 233,
        "direct_audio_snapshot_retained_frame_count": 2,
        "direct_audio_snapshot_snapshot_first_frame_id": 231,
        "direct_audio_snapshot_snapshot_last_frame_id": 233,
        "direct_audio_snapshot_expected_snapshot_frame_count": 3,
        "direct_audio_snapshot_missing_frame_count": 1,
        "direct_audio_snapshot_missing_frame_ids": [232],
        "direct_audio_snapshot_missing_frame_ids_truncated": False,
    }
    assert not any(isinstance(value, np.ndarray) for value in fields.values())
    assert list(session.policy_audio_frame_ids) == [231, 233]
    assert session.policy_audio_consumed_through_frame_id == 230


def test_direct_audio_snapshot_diagnostic_missing_ids_are_capped() -> None:
    from duplex_kernel.server import _direct_audio_snapshot_event_fields

    session = DuplexSession("direct-audio-snapshot-diagnostics-cap")
    session.append_policy_audio(100, np.ones(1280, dtype=np.float32))

    with pytest.raises(ValueError) as caught:
        _snapshot_direct_policy_audio(
            session,
            action=ControlAction.BACKCHANNEL,
            first_frame_id=100,
            last_frame_id=140,
            user_active=True,
        )

    fields = _direct_audio_snapshot_event_fields(caught.value)
    assert len(fields["direct_audio_snapshot_missing_frame_ids"]) == 16
    assert fields["direct_audio_snapshot_missing_frame_count"] == 40
    assert fields["direct_audio_snapshot_missing_frame_ids_truncated"] is True


def _pin_test_utterance(session: DuplexSession, first: int = 2, last: int = 135) -> np.ndarray:
    for frame_id in range(first, last + 1):
        session.append_policy_audio(frame_id, np.full(1280, frame_id, dtype=np.float32))
    assert session.pin_completed_utterance_audio(
        first,
        last,
        source_cutoff_sample_id=(last + 1) * 1280,
        generation_id=session.session_generation_id,
    )
    return np.concatenate([
        np.full(1280, frame_id, dtype=np.float32)
        for frame_id in range(first, last + 1)
    ])


def test_completed_utterance_pin_survives_rolling_ring_eviction() -> None:
    session = DuplexSession("direct-audio-completed-pin")
    expected = _pin_test_utterance(session)

    for frame_id in range(136, 486):
        session.append_policy_audio(frame_id, np.full(1280, frame_id, dtype=np.float32))

    assert session.pin_completed_utterance_audio(
        2,
        135,
        source_cutoff_sample_id=136 * 1280,
        generation_id=session.session_generation_id,
    )
    result = session.snapshot_completed_utterance_audio(
        2,
        135,
        source_cutoff_sample_id=136 * 1280,
        generation_id=session.session_generation_id,
    )

    assert list(session.policy_audio_frame_ids) == list(range(111, 486))
    assert result is not None
    assert result.size == 171520
    assert np.array_equal(result, expected)

    context = _snapshot_direct_policy_audio(
        session,
        action=ControlAction.SPEAK,
        first_frame_id=2,
        last_frame_id=135,
        completed_utterance=True,
        source_cutoff_sample_id=136 * 1280,
    )
    assert context["_direct_audio_sample_count"] == 171520
    assert np.array_equal(context["_direct_audio_pcm"], expected)


def test_completed_utterance_pin_rejects_wrong_range_generation_and_cutoff() -> None:
    session = DuplexSession("direct-audio-completed-pin-fence")
    _pin_test_utterance(session, first=20, last=21)

    assert session.snapshot_completed_utterance_audio(
        19,
        21,
        source_cutoff_sample_id=22 * 1280,
        generation_id=session.session_generation_id,
    ) is None
    assert session.snapshot_completed_utterance_audio(
        20,
        22,
        source_cutoff_sample_id=23 * 1280,
        generation_id=session.session_generation_id,
    ) is None
    assert session.snapshot_completed_utterance_audio(
        20,
        21,
        source_cutoff_sample_id=22 * 1280,
        generation_id=session.session_generation_id + 1,
    ) is None
    assert session.snapshot_completed_utterance_audio(
        20,
        21,
        source_cutoff_sample_id=21 * 1280,
        generation_id=session.session_generation_id,
    ) is None


def test_completed_utterance_pin_releases_on_new_onset_and_audio_reset() -> None:
    session = DuplexSession("direct-audio-completed-pin-lifecycle")
    _pin_test_utterance(session, first=2, last=3)
    assert session.observe_user_onset(10)
    assert session.snapshot_completed_utterance_audio(
        2, 3, source_cutoff_sample_id=4 * 1280, generation_id=session.session_generation_id,
    ) is None

    _pin_test_utterance(session, first=10, last=11)
    session.reset_policy_audio(committed_sample_count=12 * 1280)
    assert session.snapshot_completed_utterance_audio(
        10, 11, source_cutoff_sample_id=12 * 1280, generation_id=session.session_generation_id,
    ) is None

    _pin_test_utterance(session, first=12, last=13)
    session.invalidate_generation("test_audio_generation_reset")
    assert session.snapshot_completed_utterance_audio(
        12, 13, source_cutoff_sample_id=14 * 1280, generation_id=session.session_generation_id,
    ) is None


def test_completed_utterance_pin_releases_on_accepted_consume_or_close() -> None:
    consumed = DuplexSession("direct-audio-completed-pin-consumed")
    _pin_test_utterance(consumed, first=2, last=3)
    assert consumed.release_completed_utterance_audio(
        2,
        3,
        source_cutoff_sample_id=4 * 1280,
        generation_id=consumed.session_generation_id,
    )
    assert consumed.snapshot_completed_utterance_audio(
        2, 3, source_cutoff_sample_id=4 * 1280, generation_id=consumed.session_generation_id,
    ) is None

    closed = DuplexSession("direct-audio-completed-pin-closed")
    _pin_test_utterance(closed, first=2, last=3)
    closed.close()
    assert closed.snapshot_completed_utterance_audio(
        2, 3, source_cutoff_sample_id=4 * 1280, generation_id=closed.session_generation_id,
    ) is None


def test_completed_utterance_pin_carries_across_backchannel_generation_and_done() -> None:
    session = DuplexSession("direct-audio-backchannel-carry-before")
    assert session.observe_user_onset(2)
    expected = _pin_test_utterance(session, first=2, last=3)
    previous_generation = session.session_generation_id

    token = session.begin_response(
        "resp-backchannel", preserve_pending_user_turn_audio=True,
    )
    assert token.session_generation_id == previous_generation + 1
    assert session.session_generation_id == token.session_generation_id
    assert session.pending_user_onset_frame_id == 2
    assert session.last_answered_onset_frame_id is None
    assert session.snapshot_completed_utterance_audio(
        2, 3, source_cutoff_sample_id=4 * 1280,
        generation_id=token.session_generation_id,
    ) is not None
    assert session.snapshot_completed_utterance_audio(
        2, 3, source_cutoff_sample_id=4 * 1280,
        generation_id=previous_generation,
    ) is None

    session.reset_backchannel_response()
    carried = session.snapshot_completed_utterance_audio(
        2, 3, source_cutoff_sample_id=4 * 1280,
        generation_id=session.session_generation_id,
    )
    assert carried is not None
    assert np.array_equal(carried, expected)
    assert session.pending_user_onset_frame_id == 2
    assert session.last_answered_onset_frame_id is None

    assert session.reserve_user_onset_for_response(2)
    assert session.complete_user_onset_response(2)
    session.reset_turn()
    assert session.pending_user_onset_frame_id is None
    assert session.last_answered_onset_frame_id == 2
    assert session.snapshot_completed_utterance_audio(
        2, 3, source_cutoff_sample_id=4 * 1280,
        generation_id=session.session_generation_id,
    ) is None


def test_completed_utterance_pin_created_during_backchannel_survives_response_done() -> None:
    session = DuplexSession("direct-audio-backchannel-carry-during")
    assert session.observe_user_onset(10)
    token = session.begin_response(
        "resp-backchannel", preserve_pending_user_turn_audio=True,
    )
    expected = _pin_test_utterance(session, first=10, last=11)

    session.reset_backchannel_response()
    carried = session.snapshot_completed_utterance_audio(
        10, 11, source_cutoff_sample_id=12 * 1280,
        generation_id=token.session_generation_id,
    )
    assert carried is not None
    assert np.array_equal(carried, expected)
    assert session.pending_user_onset_frame_id == 10
    assert session.last_answered_onset_frame_id is None


def test_backchannel_carry_rejects_pin_superseded_by_new_onset() -> None:
    session = DuplexSession("direct-audio-backchannel-carry-stale")
    assert session.observe_user_onset(2)
    _pin_test_utterance(session, first=2, last=3)
    assert session.observe_user_onset(5)

    token = session.begin_response(
        "resp-backchannel", preserve_pending_user_turn_audio=True,
    )

    assert session.pending_user_onset_frame_id == 5
    assert session.snapshot_completed_utterance_audio(
        2, 3, source_cutoff_sample_id=4 * 1280,
        generation_id=token.session_generation_id,
    ) is None


def test_speak_completion_preserves_newer_onset_pinned_before_response_begin() -> None:
    session = DuplexSession("direct-audio-speak-preserve-newer-before")
    assert session.observe_user_onset(2)
    assert session.reserve_user_onset_for_response(2)

    # A newer onset and its exact endpoint can arrive after U is selected but
    # before U's response begins. The response may carry U's committed context,
    # but must not consume V's independently pinned audio.
    assert session.observe_user_onset(10)
    expected = _pin_test_utterance(session, first=10, last=11)
    token = session.begin_response(
        "resp-speak-owner-u",
        preserve_newer_pending_user_turn_audio_after_onset_frame_id=2,
    )

    assert session.pending_user_onset_frame_id == 10
    assert session.reserved_user_onset_frame_id == 2
    assert session.complete_user_onset_response(2)
    session.reset_speak_response(consumed_user_onset_frame_id=2)

    actual = session.snapshot_completed_utterance_audio(
        10,
        11,
        source_cutoff_sample_id=12 * 1280,
        generation_id=token.session_generation_id,
    )
    assert actual is not None
    assert np.array_equal(actual, expected)
    assert session.pending_user_onset_frame_id == 10
    assert session.reserved_user_onset_frame_id is None
    assert session.last_answered_onset_frame_id == 2


def test_speak_completion_preserves_newer_endpoint_pinned_during_response() -> None:
    session = DuplexSession("direct-audio-speak-preserve-newer-during")
    assert session.observe_user_onset(2)
    assert session.reserve_user_onset_for_response(2)
    token = session.begin_response(
        "resp-speak-owner-u",
        preserve_newer_pending_user_turn_audio_after_onset_frame_id=2,
    )

    assert session.observe_user_onset(10)
    expected = _pin_test_utterance(session, first=10, last=11)
    assert session.complete_user_onset_response(2)
    session.reset_speak_response(consumed_user_onset_frame_id=2)

    actual = session.snapshot_completed_utterance_audio(
        10,
        11,
        source_cutoff_sample_id=12 * 1280,
        generation_id=token.session_generation_id,
    )
    assert actual is not None
    assert np.array_equal(actual, expected)
    assert session.pending_user_onset_frame_id == 10
    assert session.last_answered_onset_frame_id == 2


def test_speak_cleanup_preserves_newer_pin_when_owned_response_does_not_complete() -> None:
    session = DuplexSession("direct-audio-speak-preserve-newer-after-no-speech")
    assert session.observe_user_onset(2)
    assert session.reserve_user_onset_for_response(2)
    token = session.begin_response(
        "resp-speak-owner-u-no-speech",
        preserve_newer_pending_user_turn_audio_after_onset_frame_id=2,
    )
    assert session.observe_user_onset(10)
    expected = _pin_test_utterance(session, first=10, last=11)

    assert session.reset_speak_response(consumed_user_onset_frame_id=2)
    assert session.cancel_user_onset_response(2)
    actual = session.snapshot_completed_utterance_audio(
        10,
        11,
        source_cutoff_sample_id=12 * 1280,
        generation_id=token.session_generation_id,
    )
    assert actual is not None
    assert np.array_equal(actual, expected)
    assert session.pending_user_onset_frame_id == 10
    assert session.reserved_user_onset_frame_id is None
    assert session.last_answered_onset_frame_id is None


def test_speak_cleanup_does_not_keep_pin_for_answered_or_invalid_pending_onset() -> None:
    session = DuplexSession("direct-audio-speak-does-not-resurrect-stale")
    token = session.begin_response("resp-unowned-speak")
    assert session.observe_user_onset(10)
    _pin_test_utterance(session, first=10, last=11)
    # Simulate a newer onset becoming answered before cleanup; its exact pin
    # remains allocated but no longer has a live owner.
    session.last_answered_onset_frame_id = 10
    session.reset_speak_response()

    assert session.pending_user_onset_frame_id == 10
    assert session.snapshot_completed_utterance_audio(
        10,
        11,
        source_cutoff_sample_id=12 * 1280,
        generation_id=token.session_generation_id,
    ) is None


def test_manual_speak_without_owned_onset_keeps_only_valid_pending_pin() -> None:
    session = DuplexSession("direct-audio-manual-unowned-speak-preserves")
    token = session.begin_response("resp-manual-seed-no-onset")
    assert session.observe_user_onset(10)
    expected = _pin_test_utterance(session, first=10, last=11)

    assert session.reset_speak_response() is True
    actual = session.snapshot_completed_utterance_audio(
        10,
        11,
        source_cutoff_sample_id=12 * 1280,
        generation_id=token.session_generation_id,
    )
    assert actual is not None
    assert np.array_equal(actual, expected)


def test_speak_begin_rejects_newer_pin_with_different_reserved_owner_or_same_onset() -> None:
    mismatched = DuplexSession("direct-audio-speak-owner-mismatch")
    assert mismatched.observe_user_onset(2)
    assert mismatched.reserve_user_onset_for_response(2)
    assert mismatched.observe_user_onset(10)
    _pin_test_utterance(mismatched, first=10, last=11)
    mismatch_token = mismatched.begin_response(
        "resp-speak-mismatched-owner",
        preserve_newer_pending_user_turn_audio_after_onset_frame_id=3,
    )
    assert mismatched.snapshot_completed_utterance_audio(
        10,
        11,
        source_cutoff_sample_id=12 * 1280,
        generation_id=mismatch_token.session_generation_id,
    ) is None

    same_onset = DuplexSession("direct-audio-speak-owner-same-onset")
    assert same_onset.observe_user_onset(2)
    assert same_onset.reserve_user_onset_for_response(2)
    _pin_test_utterance(same_onset, first=2, last=3)
    same_token = same_onset.begin_response(
        "resp-speak-same-owner",
        preserve_newer_pending_user_turn_audio_after_onset_frame_id=2,
    )
    assert same_onset.snapshot_completed_utterance_audio(
        2,
        3,
        source_cutoff_sample_id=4 * 1280,
        generation_id=same_token.session_generation_id,
    ) is None


def test_manual_seed_owner_is_reserved_from_committed_span_and_newer_onset_survives() -> None:
    session = DuplexSession("direct-audio-manual-seed-owner")
    assert session.observe_user_onset(2)
    manual_speaks, seed_owner = _manual_commit_speak_owner_for_span(
        session,
        0,
        5,
        direct_audio_candidate=True,
        auto_response=False,
        committed_audio_size=5 * 1280,
    )
    assert manual_speaks is True
    assert seed_owner == 2

    token = session.begin_response(
        "resp-manual-seed-owner",
        preserve_newer_pending_user_turn_audio_after_onset_frame_id=seed_owner,
    )
    assert session.observe_user_onset(10)
    expected = _pin_test_utterance(session, first=10, last=11)
    assert session.complete_user_onset_response(seed_owner)
    assert session.reset_speak_response(consumed_user_onset_frame_id=seed_owner)

    actual = session.snapshot_completed_utterance_audio(
        10,
        11,
        source_cutoff_sample_id=12 * 1280,
        generation_id=token.session_generation_id,
    )
    assert actual is not None
    assert np.array_equal(actual, expected)
    assert session.pending_user_onset_frame_id == 10
    assert session.last_answered_onset_frame_id == 2
    assert session.reserved_user_onset_frame_id is None


def test_manual_seed_owner_requires_pending_onset_inside_committed_audio_span() -> None:
    inside = DuplexSession("direct-audio-manual-seed-owner-inside")
    assert inside.observe_user_onset(12)
    manual_speaks, owner = _manual_commit_speak_owner_for_span(
        inside,
        10,
        14,
        direct_audio_candidate=True,
        auto_response=False,
        committed_audio_size=1280,
    )
    assert manual_speaks is True
    assert owner == 12
    assert inside.reserved_user_onset_frame_id == 12

    outside = DuplexSession("direct-audio-manual-seed-owner-outside")
    assert outside.observe_user_onset(15)
    manual_speaks, owner = _manual_commit_speak_owner_for_span(
        outside,
        10,
        14,
        direct_audio_candidate=True,
        auto_response=False,
        committed_audio_size=1280,
    )
    assert manual_speaks is True
    assert owner is None
    assert outside.reserved_user_onset_frame_id is None


def test_manual_commit_with_auto_response_does_not_reserve_user_onset() -> None:
    session = DuplexSession("direct-audio-manual-non-speak-commit")
    assert session.observe_user_onset(12)
    manual_speaks, owner = _manual_commit_speak_owner_for_span(
        session,
        10,
        14,
        direct_audio_candidate=True,
        auto_response=True,
        committed_audio_size=1280,
    )
    assert manual_speaks is False
    assert owner is None
    assert session.pending_user_onset_frame_id == 12
    assert session.reserved_user_onset_frame_id is None
