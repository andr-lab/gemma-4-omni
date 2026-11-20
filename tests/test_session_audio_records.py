from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from duplex_kernel.session import DuplexSession


def reserve(session, **kwargs):
    return session.reserve_audio_history_archive(
        pcm=np.arange(1280, dtype=np.float32), first_frame_id=3,
        last_frame_id=3, base_messages=[], state_epoch_id=1, **kwargs,
    )


@pytest.mark.parametrize("transcript", [None, "", "   "])
def test_user_audio_commits_before_asr_and_assistant(transcript):
    session = DuplexSession("audio-only")
    archive_id = reserve(session)
    before = session.snapshot_history_with_provenance()
    assert len(before.history) == 1
    assert before.history[0]["role"] == "user"
    assert before.history[0]["content"] == ""
    assert before.user_audio_spans == (before.history[0]["audio_span"],)
    assert session.pending_audio_history_archive is None
    assert session.complete_audio_history_archive(archive_id, transcript)
    after = session.snapshot_history_with_provenance()
    assert after.history == before.history
    assert after.history_revision == before.history_revision
    assert after.history_mutation_sequence == before.history_mutation_sequence


def test_late_asr_only_updates_diagnostics_without_notification():
    session = DuplexSession("late-asr")
    notifications = []
    session.set_history_archive_commit_callback(lambda _, snapshot: notifications.append(snapshot))
    archive_id = reserve(session)
    before = session.snapshot_history_with_provenance()
    assert session.update_audio_history_archive_assistant(archive_id, "Confermato.")
    confirmed = session.snapshot_history_with_provenance()
    assert session.complete_audio_history_archive(archive_id, "diagnostica", asr_provenance={"model": "test"})
    after = session.snapshot_history_with_provenance()
    assert after.history == confirmed.history
    assert after.history_revision == confirmed.history_revision
    assert session.get_audio_history_archive(archive_id)["transcript"] == "diagnostica"
    assert dict(after.provenance[0].asr_provenance) == {"model": "test"}
    assert len(notifications) == 2
    assert before.assistant_text_spans == ()


def test_assistant_commits_without_audio_or_transcript():
    session = DuplexSession("assistant-first")
    archive_id = session.reserve_audio_history_archive(
        pcm=None, first_frame_id=0, last_frame_id=2, base_messages=[],
        source="heard_action_say_v2",
    )
    assert session.update_audio_history_archive_assistant(archive_id, "Risposta.")
    snapshot = session.snapshot_history_with_provenance()
    assert [item["role"] for item in snapshot.history] == ["assistant"]
    assert snapshot.assistant_text_spans[0].confirmed_text == "Risposta."
    assert snapshot.user_audio_spans == ()


def test_audio_record_owns_immutable_cpu_buffers_and_snapshot_preserves_references():
    from duplex_kernel.session import UserAudioSpan

    session = DuplexSession("integrity")
    pcm = np.ones(1280, dtype=np.float32)
    soft = np.ones((4, 8), dtype=np.float32)
    archive_id = session.reserve_audio_history_archive(
        pcm=pcm, first_frame_id=8, last_frame_id=9, base_messages=[],
        soft_tokens_reference=soft, encoder_profile="q2-test", turn_id=5,
        chunk_id=2, revision=3, turn_open=True,
    )
    pcm[:] = 0
    soft[:] = 0
    span = session.history[0]["audio_span"]
    assert isinstance(span, UserAudioSpan)
    assert (span.turn_id, span.chunk_id, span.revision) == (5, 2, 3)
    assert (span.first_frame_id, span.last_frame_id) == (8, 9)
    assert span.encoder_profile == "q2-test" and span.turn_open
    assert np.all(span.pcm_reference == 1) and np.all(span.soft_tokens_reference == 1)
    with pytest.raises(FrozenInstanceError):
        span.revision = 4
    with pytest.raises(ValueError):
        span.pcm_reference[0] = 7
    with pytest.raises(ValueError):
        span.pcm_reference.setflags(write=True)
    snapshot = session.snapshot_history_with_provenance()
    assert snapshot.user_audio_spans[0] is span
    assert session.get_audio_history_archive(archive_id) is not None


def test_confirmed_assistant_record_without_archive():
    from duplex_kernel.session import AssistantTextSpan

    session = DuplexSession("confirmed")
    token = session.begin_response("response-1")
    assert session.commit_completed_assistant_response(token, "response-1", "Ciao.")
    assert session.commit_completed_assistant_response(token, "response-1", "Ciao.")
    snapshot = session.snapshot_history_with_provenance()
    span = snapshot.assistant_text_spans[0]
    assert isinstance(span, AssistantTextSpan)
    assert span.response_id == "response-1"
    assert span.confirmed_text == "Ciao."
    assert span.source_character_range == (0, 5)
    assert len(snapshot.history) == 1
    with pytest.raises(FrozenInstanceError):
        span.confirmed_text = "Altro"


def test_compaction_eligibility_does_not_require_asr():
    session = DuplexSession("eligibility")
    reserve(session)
    record = session.snapshot_history_with_provenance().provenance[0]
    assert record.asr_provenance == ()
    assert record.compaction_eligible
    assert not replace(record, coverage_complete=False).compaction_eligible
    assert not replace(record, state_epoch_id=None).compaction_eligible


def test_record_revision_changes_history_identity():
    session = DuplexSession("row-identity")
    reserve(session)
    before = session.snapshot_history_with_provenance()
    session.history[0]["audio_span"] = replace(before.user_audio_spans[0], revision=2)
    after = session.snapshot_history_with_provenance()
    assert after.history_item_ids != before.history_item_ids


def test_memory_revision_is_immutable():
    from duplex_kernel.session import MemoryRevision

    revision = MemoryRevision("m2", "m1", ((3, 9),), (2,), "Preferenza confermata", None)
    assert revision.covered_source_ranges == ((3, 9),)
    assert revision.source_revisions == (2,)
    with pytest.raises(FrozenInstanceError):
        revision.semantic_notes = "mutato"


def test_slow_asr_does_not_exhaust_audio_archive_slots():
    session = DuplexSession("slow-asr")
    for frame in range(4):
        archive_id = session.reserve_audio_history_archive(
            pcm=np.ones(1280, dtype=np.float32), first_frame_id=frame,
            last_frame_id=frame, base_messages=[],
        )
        assert archive_id is not None
    assert len(session.snapshot_history_with_provenance().user_audio_spans) == 4
    assert session.audio_history_archive_overflow_count == 0


def test_unarchived_slot_does_not_block_later_audio():
    session = DuplexSession("independent-slots")
    first = session.reserve_audio_history_archive(
        pcm=None, first_frame_id=0, last_frame_id=0, base_messages=[],
        source="heard_action_say_v2",
    )
    second = reserve(session)
    snapshot = session.snapshot_history_with_provenance()
    assert snapshot.user_audio_spans[0].first_frame_id == 3
    assert session.pending_audio_history_archive["archive_id"] == first
    assert session.get_audio_history_archive(second) is not None


def test_soft_tokens_can_commit_without_pcm_or_asr():
    session = DuplexSession("soft-only")
    archive_id = session.reserve_audio_history_archive(
        pcm=None, soft_tokens_reference=np.ones((4, 8), dtype=np.float32),
        encoder_profile="q2", first_frame_id=0, last_frame_id=0,
        base_messages=[],
    )
    span = session.snapshot_history_with_provenance().user_audio_spans[0]
    assert span.pcm_reference is None
    assert span.soft_tokens_reference.shape == (4, 8)
    assert session.pending_audio_history_archive is None
    assert session.fail_audio_history_archive(archive_id, "ASR unavailable")
    assert session.snapshot_history_with_provenance().user_audio_spans == (span,)


def test_assistant_frontier_updates_the_existing_record():
    session = DuplexSession("assistant-frontier")
    archive_id = reserve(session)
    for frontier in (40, 80, 80):
        assert session.update_audio_history_archive_assistant(
            archive_id, "Ciao.", response_id="r1", revision=2,
            pcm_reference=np.ones(frontier, dtype=np.float32),
            rendered_sample_frontier=frontier,
        )
    snapshot = session.snapshot_history_with_provenance()
    assert len(snapshot.history) == 2
    span = snapshot.assistant_text_spans[0]
    assert span.response_id == "r1" and span.revision == 2
    assert span.rendered_sample_frontier == 80
    assert span.pcm_reference.shape == (80,)
    revision = snapshot.history_revision
    assert session.update_audio_history_archive_assistant(archive_id, "Ciao.")
    unchanged = session.snapshot_history_with_provenance()
    assert unchanged.history_revision == revision
    assert unchanged.assistant_text_spans == (span,)


def test_compaction_verifies_represented_content_without_asr_metadata():
    from duplex_kernel.server_listener_compaction import _verified_history_span

    session = DuplexSession("represented-source")
    reserve(session)
    snapshot = session.snapshot_history_with_provenance()
    arguments = dict(
        session_id=session.session_id, message_index=0,
        history_item_ids=snapshot.history_item_ids,
        history_revision=snapshot.history_revision,
        history_mutation_sequence=snapshot.history_mutation_sequence,
    )
    record = snapshot.provenance[0]
    assert record.compaction_eligible and not record.asr_provenance
    assert _verified_history_span(record, messages=snapshot.history, **arguments) is None
    represented = ({"role": "user", "content": "Contenuto conservato nel prefisso."},)
    verified = _verified_history_span(record, messages=represented, **arguments)
    assert verified is not None
    assert (verified.first_frame_id, verified.last_frame_id) == (3, 3)
