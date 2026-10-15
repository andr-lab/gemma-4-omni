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
