from __future__ import annotations

import numpy as np

from duplex_kernel.session import DuplexSession


def test_policy_audio_snapshot_is_contiguous_inclusive_and_non_consuming() -> None:
    session = DuplexSession("audio-snapshot")
    session.append_policy_audio(3, np.full(1280, 3, dtype=np.float32))
    session.append_policy_audio(4, np.full(1280, 4, dtype=np.float32))
    session.append_policy_audio(6, np.full(1280, 6, dtype=np.float32))

    snapshot = session.snapshot_policy_audio_range(3, 4)

    assert snapshot.size == 2560
    assert np.all(snapshot[:1280] == 3)
    assert np.all(snapshot[1280:] == 4)
    assert session.snapshot_policy_audio_range(3, 6).size == 0
    assert list(session.policy_audio_frame_ids) == [3, 4, 6]
    assert session.policy_audio_samples == 3840


def test_policy_audio_snapshot_rejects_duplicate_ids_inside_requested_span() -> None:
    session = DuplexSession("audio-snapshot-duplicate-id")
    session.append_policy_audio(3, np.full(1280, 3, dtype=np.float32))
    session.append_policy_audio(3, np.full(1280, 30, dtype=np.float32))
    session.append_policy_audio(4, np.full(1280, 4, dtype=np.float32))

    assert session.snapshot_policy_audio_range(3, 4).size == 0
    np.testing.assert_array_equal(
        session.snapshot_policy_audio_range(4, 4),
        np.full(1280, 4, dtype=np.float32),
    )


def test_audio_archive_commits_completed_transcript_in_turn_order() -> None:
    session = DuplexSession("audio-archive")
    base = [{"role": "user", "content": "precedente"}, {"role": "assistant", "content": "risposta"}]
    pcm = np.ones(1280, dtype=np.float32)

    archive_id = session.reserve_audio_history_archive(
        pcm=pcm, first_frame_id=10, last_frame_id=10, base_messages=base,
    )
    session.update_audio_history_archive_assistant(archive_id, "risposta nuova")
    pcm[:] = 7
    base[0]["content"] = "mutato"

    assert session.complete_audio_history_archive(archive_id, "domanda archiviata")
    assert session.history == [
        {"role": "user", "content": "precedente"},
        {"role": "assistant", "content": "risposta"},
        {"role": "user", "content": "domanda archiviata"},
        {"role": "assistant", "content": "risposta nuova"},
    ]
    assert session.pending_audio_history_archive is None
    assert not session.complete_audio_history_archive(archive_id, "stale")


def test_audio_archive_commits_acknowledged_partial_assistant_text_after_cancel() -> None:
    session = DuplexSession("audio-archive-partial-assistant")
    archive_id = session.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32),
        first_frame_id=0,
        last_frame_id=0,
        base_messages=[{"role": "user", "content": "precedente"}],
    )

    assert session.complete_audio_history_archive(archive_id, "audio utente")
    assert session.update_audio_history_archive_assistant(
        archive_id,
        "Aspetta!",
        status="partial",
    )

    assert session.history == [
        {"role": "user", "content": "precedente"},
        {"role": "user", "content": "audio utente"},
        {"role": "assistant", "content": "Aspetta!"},
    ]
    assert session.pending_audio_history_archive is None


def test_fast_archive_waits_for_assistant_commit_before_advancing_history() -> None:
    session = DuplexSession("audio-archive-race")
    archive_id = session.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32),
        first_frame_id=0,
        last_frame_id=0,
        base_messages=[{"role": "user", "content": "precedente"}],
    )

    assert session.complete_audio_history_archive(archive_id, "current user")
    assert session.history == []
    assert session.pending_audio_history_archive["status"] == "transcript_ready"
    assert session.update_audio_history_archive_assistant(archive_id, "current assistant")
    assert session.history == [
        {"role": "user", "content": "precedente"},
        {"role": "user", "content": "current user"},
        {"role": "assistant", "content": "current assistant"},
    ]
    assert session.pending_audio_history_archive is None


def test_audio_archive_is_session_fenced_and_skips_empty_transcript() -> None:
    session = DuplexSession("audio-archive-fence")
    archive_id = session.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32),
        first_frame_id=0,
        last_frame_id=0,
        base_messages=[],
    )

    assert not session.complete_audio_history_archive(archive_id + 1, "stale")
    assert session.complete_audio_history_archive(archive_id, " ")
    assert session.pending_audio_history_archive is not None
    assert session.update_audio_history_archive_assistant(archive_id, "", status="absent")
    assert session.pending_audio_history_archive is None
    session.close()
    assert not session.complete_audio_history_archive(archive_id, "dopo close")
    assert session.history == []


def test_cancelled_archive_preserves_real_user_transcript_without_assistant_text() -> None:
    session = DuplexSession("audio-archive-cancelled-assistant")
    archive_id = session.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32),
        first_frame_id=4,
        last_frame_id=4,
        base_messages=[{"role": "assistant", "content": "turno precedente"}],
        response_created_at_monotonic_ns=100,
    )
    assert session.pending_audio_history_archive is not None
    assert session.pending_audio_history_archive["response_created_at_monotonic_ns"] == 100

    assert session.mark_audio_history_archive_submitted(
        archive_id,
        submit_started_at_monotonic_ns=110,
        submitted_at_monotonic_ns=120,
    )
    assert session.complete_audio_history_archive(archive_id, "domanda reale")
    assert session.update_audio_history_archive_assistant(
        archive_id, "", status="cancelled", response_done_at_monotonic_ns=130,
    )

    assert session.history == [
        {"role": "assistant", "content": "turno precedente"},
        {"role": "user", "content": "domanda reale"},
    ]
    assert session.pending_audio_history_archive is None



def test_v2_heard_archive_needs_no_pcm_or_second_transcription() -> None:
    session = DuplexSession("audio-archive-v2-heard")
    archive_id = session.reserve_audio_history_archive(
        pcm=None,
        first_frame_id=3,
        last_frame_id=8,
        base_messages=[{"role": "assistant", "content": "Prima risposta."}],
        source="heard_action_say_v2",
    )

    pending = session.pending_audio_history_archive
    assert pending is not None
    assert pending["pcm"] is None
    assert pending["source"] == "heard_action_say_v2"
    assert session.complete_audio_history_archive(archive_id, "Audio precedente senza trascrizione disponibile.")
    assert session.update_audio_history_archive_assistant(archive_id, "Contesto salvato.")
    assert session.history == [
        {"role": "assistant", "content": "Prima risposta."},
        {"role": "user", "content": "Audio precedente senza trascrizione disponibile."},
        {"role": "assistant", "content": "Contesto salvato."},
    ]


def test_bounded_archives_commit_in_turn_order_without_waiting_for_later_asr():
    session = DuplexSession("archive-queue-order")
    pcm = np.ones(1600, dtype=np.float32)
    first = session.reserve_audio_history_archive(
        pcm=pcm, first_frame_id=0, last_frame_id=0, base_messages=[],
    )
    second = session.reserve_audio_history_archive(
        pcm=pcm, first_frame_id=1, last_frame_id=1, base_messages=[],
    )

    assert session.update_audio_history_archive_assistant(first, "risposta uno")
    assert session.update_audio_history_archive_assistant(second, "risposta due")
    assert session.complete_audio_history_archive(second, "utente due")
    assert session.history == []  # turn two cannot overtake turn one
    assert session.complete_audio_history_archive(first, "utente uno")

    assert session.history == [
        {"role": "user", "content": "utente uno"},
        {"role": "assistant", "content": "risposta uno"},
        {"role": "user", "content": "utente due"},
        {"role": "assistant", "content": "risposta due"},
    ]


def test_empty_user_archive_keeps_completed_assistant_text_exactly_once():
    session = DuplexSession("audio-archive-empty-user-completed-assistant")
    token = session.begin_response("resp-complete")
    archive_id = session.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32),
        first_frame_id=7,
        last_frame_id=7,
        base_messages=[{"role": "assistant", "content": "turno prima"}],
    )

    assert session.complete_audio_history_archive(archive_id, "")
    assert session.pending_audio_history_archive is not None
    assert session.commit_completed_assistant_response(
        token, "resp-complete", "Risposta canonica.", archive_id=archive_id,
        user_onset_frame_id=7,
    )
    assert session.history == [
        {"role": "assistant", "content": "turno prima"},
        {"role": "assistant", "content": "Risposta canonica."},
    ]
    assert session.pending_audio_history_archive is None
    assert session.commit_completed_assistant_response(
        token, "resp-complete", "Risposta canonica.", archive_id=archive_id,
    )
    assert session.history.count({"role": "assistant", "content": "Risposta canonica."}) == 1


def test_completed_assistant_text_commits_when_no_user_archive_exists():
    session = DuplexSession("audio-archive-disabled-assistant-history")
    token = session.begin_response("resp-no-archive")

    assert session.commit_completed_assistant_response(
        token, "resp-no-archive", "Risposta senza ASR.",
    )
    assert session.commit_completed_assistant_response(
        token, "resp-no-archive", "Risposta senza ASR.",
    )
    assert session.history == [{"role": "assistant", "content": "Risposta senza ASR."}]


def test_stale_or_cancelled_response_cannot_commit_assistant_history():
    session = DuplexSession("audio-archive-stale-response")
    token = session.begin_response("resp-stale")
    session.invalidate_generation("yield")

    assert not session.commit_completed_assistant_response(
        token, "resp-stale", "Testo che non deve entrare in history.",
    )
    assert session.history == []


def test_cancelled_empty_archive_never_commits_assistant_text():
    session = DuplexSession("audio-archive-empty-cancelled")
    archive_id = session.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32),
        first_frame_id=2,
        last_frame_id=2,
        base_messages=[],
    )

    assert session.complete_audio_history_archive(archive_id, "")
    assert session.update_audio_history_archive_assistant(
        archive_id, "", status="cancelled",
    )
    assert session.history == []
    assert session.pending_audio_history_archive is None


def _complete_cpu_asr_provenance() -> dict[str, object]:
    return {
        "model": "faster-whisper-small",
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": 4,
        "language": "it",
        "text": "must not be copied into provenance",
        "phase_timings_ms": {"decode": 12.0},
    }


def test_archive_commit_keeps_immutable_cpu_asr_provenance_after_pending_closes():
    session = DuplexSession("archive-provenance")
    archive_id = session.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32),
        first_frame_id=12,
        last_frame_id=14,
        base_messages=[],
        source="post_response_asr",
        timeline_id="timeline-archive-provenance",
        state_epoch_id=9,
        coverage="complete",
    )
    assert archive_id is not None
    assert session.complete_audio_history_archive(
        archive_id,
        "parlami di Torino",
        asr_provenance=_complete_cpu_asr_provenance(),
    )
    assert session.update_audio_history_archive_assistant(archive_id, "Certo.")

    snapshot = session.snapshot_history_with_provenance()
    assert session.pending_audio_history_archive is None
    assert snapshot.session_id == session.session_id
    assert [dict(message) for message in snapshot.history] == [
        {"role": "user", "content": "parlami di Torino"},
        {"role": "assistant", "content": "Certo."},
    ]
    assert len(snapshot.provenance) == 1
    provenance = snapshot.provenance[0]
    assert provenance.archive_id == archive_id
    assert provenance.history_item_id == snapshot.history_item_ids[0]
    assert provenance.message_index == 0
    assert provenance.timeline_id == "timeline-archive-provenance"
    assert provenance.state_epoch_id == 9
    assert (provenance.first_frame_id, provenance.last_frame_id) == (12, 14)
    assert provenance.coverage_complete is True
    assert provenance.transcript_status == "committed"
    assert provenance.history_revision == snapshot.history_revision
    assert provenance.history_mutation_sequence == snapshot.history_mutation_sequence
    assert provenance.compaction_eligible is True
    assert dict(provenance.asr_provenance) == {
        "model": "faster-whisper-small",
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": 4,
        "language": "it",
    }


def test_archive_provenance_without_complete_cpu_asr_is_not_compaction_eligible():
    from types import SimpleNamespace

    missing_asr = DuplexSession("archive-provenance-missing-asr")
    missing_id = missing_asr.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32), first_frame_id=0, last_frame_id=1,
        base_messages=[], source="post_response_asr", coverage="complete",
    )
    assert missing_id is not None
    assert missing_asr.complete_audio_history_archive(missing_id, "trascrizione")
    assert missing_asr.update_audio_history_archive_assistant(missing_id, "risposta")
    assert missing_asr.snapshot_history_with_provenance().provenance[0].compaction_eligible is False

    incomplete = DuplexSession("archive-provenance-partial")
    partial_id = incomplete.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32), first_frame_id=2, last_frame_id=2,
        base_messages=[], source="post_response_asr", coverage="partial",
    )
    assert partial_id is not None
    assert incomplete.complete_audio_history_archive(
        partial_id, "trascrizione parziale", asr_provenance=_complete_cpu_asr_provenance(),
    )
    assert incomplete.update_audio_history_archive_assistant(partial_id, "risposta")
    assert incomplete.snapshot_history_with_provenance().provenance[0].compaction_eligible is False

    heard = DuplexSession("archive-provenance-heard-action")
    heard_id = heard.reserve_audio_history_archive(
        pcm=None, first_frame_id=3, last_frame_id=3, base_messages=[],
        source="heard_action_say_v2", coverage="complete",
    )
    assert heard_id is not None
    assert heard.complete_audio_history_archive(
        heard_id, "testo sentito", asr_provenance=_complete_cpu_asr_provenance(),
    )
    assert heard.update_audio_history_archive_assistant(heard_id, "risposta")
    assert heard.snapshot_history_with_provenance().provenance[0].compaction_eligible is False

    missing_epoch = DuplexSession(
        "archive-provenance-missing-epoch",
        listener_state=SimpleNamespace(state_epoch_id=None),
    )
    epoch_id = missing_epoch.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32), first_frame_id=0, last_frame_id=0,
        base_messages=[], source="post_response_asr", coverage="complete",
    )
    assert epoch_id is not None
    assert missing_epoch.complete_audio_history_archive(
        epoch_id, "senza epoch", asr_provenance=_complete_cpu_asr_provenance(),
    )
    assert missing_epoch.update_audio_history_archive_assistant(epoch_id, "risposta")
    assert missing_epoch.snapshot_history_with_provenance().provenance[0].compaction_eligible is False


def test_history_snapshot_revision_is_atomic_and_session_owned():
    first = DuplexSession("archive-provenance-session-one")
    second = DuplexSession("archive-provenance-session-two")
    archive_id = first.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32), first_frame_id=4, last_frame_id=4,
        base_messages=[], source="post_response_asr", coverage="complete",
    )
    second_archive_id = second.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32), first_frame_id=8, last_frame_id=8,
        base_messages=[], source="post_response_asr", coverage="complete",
    )
    assert archive_id == 1
    assert second_archive_id == archive_id
    assert first.complete_audio_history_archive(
        archive_id, "stessa domanda", asr_provenance=_complete_cpu_asr_provenance(),
    )
    assert first.update_audio_history_archive_assistant(archive_id, "risposta")
    assert second.complete_audio_history_archive(
        second_archive_id, "altra domanda", asr_provenance=_complete_cpu_asr_provenance(),
    )
    assert second.update_audio_history_archive_assistant(second_archive_id, "altra risposta")

    before = first.snapshot_history_with_provenance()
    second_snapshot = second.snapshot_history_with_provenance()
    assert before.provenance[0].history_revision == before.history_revision
    assert before.provenance[0].history_mutation_sequence == before.history_mutation_sequence
    assert before.provenance[0].session_id == first.session_id
    assert second_snapshot.provenance[0].session_id == second.session_id
    assert second_snapshot.provenance[0].archive_id == before.provenance[0].archive_id
    assert second_snapshot.history[0]["content"] == "altra domanda"

    first.update_instructions("Rispondi con una frase.")
    after = first.snapshot_history_with_provenance()
    assert after.history_revision > before.history_revision
    assert after.history_mutation_sequence == before.history_mutation_sequence
    assert after.instructions == "Rispondi con una frase."
    assert after.provenance[0].history_revision == before.history_revision
    assert after.provenance[0].history_item_id == after.history_item_ids[0]


def test_archive_and_instruction_callbacks_receive_snapshots_outside_generation_lock():
    session = DuplexSession("archive-provenance-callback")
    callback_observations: list[tuple[bool, int, str]] = []

    def callback(_session, snapshot) -> None:
        callback_observations.append((
            bool(session._generation_lock._is_owned()),
            snapshot.history_revision,
            snapshot.instructions,
        ))

    session.set_history_archive_commit_callback(callback)
    archive_id = session.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32), first_frame_id=5, last_frame_id=5,
        base_messages=[], source="post_response_asr", coverage="complete",
    )
    assert archive_id is not None
    assert session.complete_audio_history_archive(
        archive_id, "testo", asr_provenance=_complete_cpu_asr_provenance(),
    )
    assert session.update_audio_history_archive_assistant(archive_id, "risposta")
    session.update_instructions("Istruzioni aggiornate.")

    assert len(callback_observations) == 2
    assert callback_observations[0][0] is False
    assert callback_observations[0][2] != "Istruzioni aggiornate."
    assert callback_observations[1] == (
        False,
        session.semantic_context_revision,
        "Istruzioni aggiornate.",
    )


def test_mutated_history_row_loses_its_old_archive_identity():
    session = DuplexSession("archive-provenance-mutated-row")
    archive_id = session.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32), first_frame_id=0, last_frame_id=0,
        base_messages=[], source="post_response_asr", coverage="complete",
    )
    assert archive_id is not None
    assert session.complete_audio_history_archive(
        archive_id, "verified words", asr_provenance=_complete_cpu_asr_provenance(),
    )
    assert session.update_audio_history_archive_assistant(archive_id, "reply")
    original = session.snapshot_history_with_provenance()
    session.history[0]["content"] = "edited words"

    changed = session.snapshot_history_with_provenance()

    assert changed.history[0]["content"] == "edited words"
    assert changed.history_item_ids[0] != original.history_item_ids[0]
    assert changed.provenance == original.provenance
    assert changed.provenance[0].history_item_id not in changed.history_item_ids


def test_audio_timeline_id_tracks_frame_clock_not_response_generation():
    session = DuplexSession("archive-timeline-identity")
    first_timeline = session.timeline_id
    assert session.snapshot_history_with_provenance().timeline_id == first_timeline

    session.invalidate_generation("response_cancelled")
    session.reset_turn()
    assert session.timeline_id == first_timeline

    first_archive = session.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32), first_frame_id=0, last_frame_id=0,
        base_messages=[], source="post_response_asr", coverage="complete",
    )
    assert first_archive is not None
    assert session.complete_audio_history_archive(
        first_archive, "prima frase", asr_provenance=_complete_cpu_asr_provenance(),
    )
    assert session.update_audio_history_archive_assistant(first_archive, "risposta")
    assert session.snapshot_history_with_provenance().provenance[0].timeline_id == first_timeline

    callback_rows = []
    session.set_history_archive_commit_callback(
        lambda current, snapshot: callback_rows.append((
            current._generation_lock._is_owned(), snapshot.timeline_id,
        ))
    )
    second_timeline = session.reset_audio_timeline()
    assert second_timeline != first_timeline
    assert callback_rows == [(False, second_timeline)]
    second_archive = session.reserve_audio_history_archive(
        pcm=np.ones(1280, dtype=np.float32), first_frame_id=0, last_frame_id=0,
        base_messages=[], source="post_response_asr", coverage="complete",
    )
    assert second_archive is not None
    assert session.complete_audio_history_archive(
        second_archive, "seconda timeline", asr_provenance=_complete_cpu_asr_provenance(),
    )
    assert session.update_audio_history_archive_assistant(second_archive, "risposta")
    snapshot = session.snapshot_history_with_provenance()
    assert snapshot.timeline_id == second_timeline
    assert [row.timeline_id for row in snapshot.provenance] == [
        first_timeline,
        second_timeline,
    ]
