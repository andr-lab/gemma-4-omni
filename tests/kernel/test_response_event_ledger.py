from types import SimpleNamespace

import numpy as np
import pytest

from duplex_kernel.response_event_ledger import ResponseEventLedger
from duplex_kernel.server_pcm_tracker import (
    _complete_client_pcm_segment,
    _record_client_consumed_pcm,
    _record_emitted_pcm_packet,
)
from duplex_kernel.session import DuplexSession


def _branch(ledger: ResponseEventLedger, *, end_sample: int = 12):
    branch = {
        "response_id": "r1",
        "session_generation_id": 4,
        "client_consumed_pcm_sequence": -1,
        "client_consumed_segments": (),
        "playback_segments": {},
        "response_event_ledger": ledger,
    }
    packet = SimpleNamespace(
        generation_id=4,
        speaker_segment_id="r1.seg0",
        start_sample=0,
        end_sample=end_sample,
        packet_id=0,
        pcm=np.arange(end_sample, dtype=np.float32) / 10,
    )
    _record_emitted_pcm_packet(branch, packet)
    return branch


def _ack(branch, *, end_sample: int, final: bool, sequence: int = 0):
    return {
        "type": "client_consumed_pcm",
        "ack_version": 2,
        "response_id": "r1",
        "generation_id": 4,
        "speaker_segment_id": "r1.seg0",
        "start_sample": 0,
        "end_sample": end_sample,
        "sample_rate": 24_000,
        "sequence": sequence,
        "consumption_semantics": "logical_client_device_buffer",
        "final": final,
    }


def test_physical_cursor_rebase_preserves_logical_source_identity():
    ledger = ResponseEventLedger(response_id="r1", generation_id=4)
    first = {"source_token_seq": 0, "token_id": 7, "text_piece": "ciao", "exclusive_kv_cursor": 10}
    rebased = {**first, "revision_id": "r1", "cache_epoch_id": "epoch-2", "exclusive_kv_cursor": 3}

    assert ledger.record_generated_tokens([first], revision_id="r0", cache_epoch_id="epoch-1") == (0,)
    assert ledger.record_generated_tokens([rebased], revision_id="r1", cache_epoch_id="epoch-2") == (0,)

    projection = ledger.project()
    assert [row["source_token_seq"] for row in projection.source_tokens] == [0]
    assert projection.source_tokens[0]["physical_kv_cursor"] == 3
    assert projection.source_tokens[0]["physical_kv_provenance"] == [
        {"revision_id": "r0", "cache_epoch_id": "epoch-1", "physical_kv_cursor": 10},
        {"revision_id": "r1", "cache_epoch_id": "epoch-2", "physical_kv_cursor": 3},
    ]
    assert not projection.physical_cursor_contiguous
    assert ledger.source_frontier_snapshot()["generated_source_sequence"] == 0
    with pytest.raises(ValueError, match="mutated"):
        ledger.record_generated_tokens([{**rebased, "token_id": 8}])


def _record_direct_pcm_consumption(ledger, *, on_commit):
    _branch(ledger, end_sample=2)
    ledger.record_pcm_consumed(
        speaker_segment_id="r1.seg0",
        start_sample=0,
        end_sample=2,
        pcm=np.asarray([0.0, 0.1], dtype=np.float32),
        sequence=0,
        final=False,
        on_commit=on_commit,
    )


def test_record_pcm_consumed_invokes_no_argument_commit_without_project(monkeypatch):
    ledger = ResponseEventLedger(response_id="r1", generation_id=4)
    calls = []

    def unexpected_project():
        pytest.fail("PCM consumption ACK should not project the full ledger")

    monkeypatch.setattr(ledger, "project", unexpected_project)
    _record_direct_pcm_consumption(ledger, on_commit=lambda: calls.append("committed"))

    assert calls == ["committed"]


def test_record_pcm_consumed_keeps_projection_callback_compatibility():
    ledger = ResponseEventLedger(response_id="r1", generation_id=4)
    projections = []

    _record_direct_pcm_consumption(ledger, on_commit=projections.append)

    assert len(projections) == 1
    assert projections[0] == ledger.project()


def test_record_pcm_consumed_does_not_retry_internal_callback_type_error():
    ledger = ResponseEventLedger(response_id="r1", generation_id=4)
    calls = []

    def commit_then_raise(_projection=None):
        calls.append("called")
        raise TypeError("callback implementation failed")

    with pytest.raises(TypeError, match="callback implementation failed"):
        _record_direct_pcm_consumption(ledger, on_commit=commit_then_raise)

    assert calls == ["called"]
    assert ledger._pcm_samples_consumed == 0


def test_record_pcm_consumed_does_not_retry_when_callback_signature_is_opaque(monkeypatch):
    ledger = ResponseEventLedger(response_id="r1", generation_id=4)
    calls = []

    class OpaqueCommit:
        @property
        def __signature__(self):
            raise ValueError("signature intentionally unavailable")

        def __call__(self, _projection=None):
            calls.append("called")
            raise TypeError("opaque callback implementation failed")

    def unexpected_project():
        pytest.fail("callback failure should not trigger projection fallback")

    monkeypatch.setattr(ledger, "project", unexpected_project)
    with pytest.raises(TypeError, match="opaque callback implementation failed"):
        _record_direct_pcm_consumption(ledger, on_commit=OpaqueCommit())

    assert calls == ["called"]
    assert ledger._pcm_samples_consumed == 0


def test_final_text_flush_updates_only_mutable_source_text_and_keeps_provenance():
    ledger = ResponseEventLedger(
        response_id="r1",
        generation_id=4,
        revision_id="r0",
        cache_epoch_id="epoch-1",
    )
    ledger.record_generated_tokens([{
        "source_token_seq": 0,
        "token_id": 7,
        "text_piece": "",
        "exclusive_kv_cursor": 10,
        "revision_id": "r0",
        "cache_epoch_id": "epoch-1",
    }])
    before = ledger.project()
    frontier_before = ledger.source_frontier_snapshot()

    with pytest.raises(RuntimeError, match="revision or cache epoch"):
        ledger.update_unconsumed_source_text(
            0,
            "è",
            expected_revision_id="other-revision",
            expected_cache_epoch_id="epoch-1",
        )
    assert ledger.project() == before

    assert ledger.update_unconsumed_source_text(
        0,
        "è",
        expected_revision_id="r0",
        expected_cache_epoch_id="epoch-1",
    ) is True
    after = ledger.project()
    assert after.source_tokens[0]["source_token_seq"] == 0
    assert after.source_tokens[0]["text_piece"] == "è"
    assert after.source_tokens[0]["physical_kv_provenance"] == before.source_tokens[0][
        "physical_kv_provenance"
    ]
    assert ledger.source_frontier_snapshot() == frontier_before

    ledger.record_tts_reserved(0)
    frozen_before = ledger.project()
    with pytest.raises(RuntimeError, match="immutable frontier"):
        ledger.update_unconsumed_source_text(
            0,
            "replacement",
            expected_revision_id="r0",
            expected_cache_epoch_id="epoch-1",
        )
    assert ledger.project() == frozen_before


def test_locked_text_completion_plan_is_read_only_until_assignment_install():
    ledger = ResponseEventLedger(
        response_id="r1",
        generation_id=4,
        revision_id="r0",
        cache_epoch_id="epoch-1",
    )
    ledger.record_generated_tokens([
        {
            "source_token_seq": 0, "token_id": 7, "text_piece": "frozen",
            "exclusive_kv_cursor": 10, "revision_id": "r0", "cache_epoch_id": "epoch-1",
        },
        {
            "source_token_seq": 1, "token_id": 8, "text_piece": "",
            "exclusive_kv_cursor": 11, "revision_id": "r0", "cache_epoch_id": "epoch-1",
        },
    ])
    ledger.record_tts_reserved(0)
    before = ledger.project()
    snapshot = ledger.source_frontier_snapshot()
    with ledger._lock:
        plan = ledger.prepare_unconsumed_source_text_locked(
            1,
            "é",
            expected_revision_id=snapshot["revision_id"],
            expected_cache_epoch_id=snapshot["cache_epoch_id"],
        )
        assert ledger.project() == before
        assert ledger.install_unconsumed_source_text_locked(plan) is True
    after = ledger.project()
    assert after.source_tokens[0]["text_piece"] == "frozen"
    assert after.source_tokens[1]["text_piece"] == "é"
    assert after.source_tokens[1]["physical_kv_provenance"] == before.source_tokens[1][
        "physical_kv_provenance"
    ]
    assert after.logical_source_token_seq == before.logical_source_token_seq

    installed = ledger.project()
    with ledger._lock, pytest.raises(RuntimeError, match="immutable frontier"):
        ledger.prepare_unconsumed_source_text_locked(
            0,
            "changed",
            expected_revision_id=snapshot["revision_id"],
            expected_cache_epoch_id=snapshot["cache_epoch_id"],
        )
    assert ledger.project() == installed


def test_replace_unconsumed_suffix_retires_stale_tokens_and_freezes_reserved_frontier():
    ledger = ResponseEventLedger(response_id="r1", generation_id=4)
    ledger.record_generated_tokens([
        {"source_token_seq": 0, "token_id": 1, "text_piece": "stable"},
        {"source_token_seq": 1, "token_id": 2, "text_piece": "obsolete"},
    ])
    ledger.record_tts_reserved(0)
    snapshot = ledger.source_frontier_snapshot()
    assert snapshot["frozen_source_sequence"] == 0

    replaced = ledger.replace_unconsumed_suffix(
        1,
        [{"token_id": 3, "text_piece": "new"}],
        expected_source_frontier=snapshot["frozen_source_sequence"],
        expected_revision_id=snapshot["revision_id"],
        expected_cache_epoch_id=snapshot["cache_epoch_id"],
        revision_id="r2",
        cache_epoch_id="epoch-2",
        intended_text="stable new",
    )

    assert replaced == (2,)
    projection = ledger.project()
    assert [(row["source_token_seq"], row["text_piece"]) for row in projection.source_tokens] == [
        (0, "stable"), (2, "new")
    ]
    assert projection.intended_text == "stable new"
    with pytest.raises(RuntimeError, match="CAS"):
        ledger.replace_unconsumed_suffix(
            1,
            [],
            expected_source_frontier=0,
            expected_revision_id="r1",
            expected_cache_epoch_id=None,
            revision_id="r3",
        )
    with pytest.raises(ValueError, match="reserved or consumed"):
        refreshed = ledger.source_frontier_snapshot()
        ledger.replace_unconsumed_suffix(
            0,
            [],
            expected_source_frontier=refreshed["frozen_source_sequence"],
            expected_revision_id=refreshed["revision_id"],
            expected_cache_epoch_id=refreshed["cache_epoch_id"],
            revision_id="r3",
        )


def test_published_source_text_is_immutable_even_before_qwen_reservation():
    ledger = ResponseEventLedger(response_id="r1", generation_id=4)
    ledger.record_generated_tokens([
        {"source_token_seq": 0, "token_id": 1, "text_piece": "già "},
        {"source_token_seq": 1, "token_id": 2, "text_piece": "inviato"},
    ])
    ledger.record_tts_published(0)

    snapshot = ledger.source_frontier_snapshot()
    assert snapshot["frozen_source_sequence"] == 0
    assert snapshot["next_source_sequence"] == 2
    with pytest.raises(ValueError, match="reserved or consumed"):
        ledger.replace_unconsumed_suffix(
            0,
            [],
            expected_source_frontier=0,
            expected_revision_id="r0",
            expected_cache_epoch_id=None,
            revision_id="r1",
        )


def test_locked_suffix_plan_prepares_without_mutation_and_installs_by_assignment():
    ledger = ResponseEventLedger(response_id="r1", generation_id=4)
    ledger.record_generated_tokens([
        {"source_token_seq": 0, "token_id": 1, "text_piece": "frozen"},
        {"source_token_seq": 1, "token_id": 2, "text_piece": "old"},
    ])
    ledger.record_tts_published(0)
    snapshot = ledger.source_frontier_snapshot()
    before = ledger.project()

    plan_arguments = {
        "expected_source_frontier": snapshot["frozen_source_sequence"],
        "expected_revision_id": snapshot["revision_id"],
        "expected_cache_epoch_id": snapshot["cache_epoch_id"],
        "expected_tts_published_source_seq": snapshot["tts_published_source_seq"],
        "expected_tts_reserved_source_seq": snapshot["tts_reserved_source_seq"],
        "expected_tts_consumed_source_seq": snapshot["tts_consumed_source_seq"],
        "revision_id": "r1-revision",
        "cache_epoch_id": "cache-2",
        "intended_text": "frozen replacement",
    }
    with ledger._lock:
        plan = ledger.prepare_unconsumed_suffix_locked(
            1,
            [{"token_id": 3, "text_piece": "replacement"}],
            **plan_arguments,
        )
        assert ledger.project() == before
        with pytest.raises(ValueError, match="replacement token ID"):
            ledger.prepare_unconsumed_suffix_locked(
                1,
                [
                    {"token_id": 4, "text_piece": "valid first"},
                    {"token_id": True, "text_piece": "invalid second"},
                ],
                **plan_arguments,
            )
        assert ledger.project() == before
        stale_frontier_arguments = dict(plan_arguments)
        stale_frontier_arguments["expected_tts_reserved_source_seq"] = 1
        with pytest.raises(RuntimeError, match="CAS"):
            ledger.prepare_unconsumed_suffix_locked(
                1,
                [{"token_id": 5, "text_piece": "stale"}],
                **stale_frontier_arguments,
            )
        assert ledger.project() == before
        assert ledger.install_unconsumed_suffix_locked(plan) == (2,)

    after = ledger.project()
    assert [(row["source_token_seq"], row["text_piece"]) for row in after.source_tokens] == [
        (0, "frozen"), (2, "replacement"),
    ]
    assert after.intended_text == "frozen replacement"
    assert ledger.source_frontier_snapshot()["next_source_sequence"] == 3


def test_session_ledgers_capture_input_watermark_and_reject_late_older_history_commit():
    session = DuplexSession("response-ledger-order")
    session.history = [{"role": "user", "content": "domanda"}]
    session.last_listener_frame_id = 9
    session.begin_response("r1")
    older = session.get_response_event_ledger("r1")
    assert older.project().source_input_watermark == 9

    session.last_listener_frame_id = 11
    session.begin_response("r2")
    newer = session.get_response_event_ledger("r2")
    assert newer.project().source_input_watermark == 11

    def complete_response_ledger(ledger, response_id, text):
        token = {"source_token_seq": 0, "token_id": 7, "text_piece": text, "exclusive_kv_cursor": 1}
        ledger.record_generated_tokens([token])
        ledger.record_tts_consumed(0)
        packet = SimpleNamespace(
            generation_id=ledger.generation_id,
            speaker_segment_id=f"{response_id}.seg0",
            start_sample=0,
            end_sample=2,
            packet_id=0,
            pcm=np.asarray([0.1, 0.2], dtype=np.float32),
        )
        ledger.record_pcm_sent(packet)
        ledger.record_closed_tts_unit(
            speaker_segment_id=f"{response_id}.seg0",
            end_sample=2,
            source_tokens=[token],
        )
        ledger.record_pcm_consumed(
            speaker_segment_id=f"{response_id}.seg0",
            start_sample=0,
            end_sample=2,
            pcm=packet.pcm,
            sequence=0,
            final=True,
        )
        ledger.close_response(
            intended_text=text,
            finish_reason="stop",
            completed=True,
        )

    complete_response_ledger(older, "r1", "vecchio")
    complete_response_ledger(newer, "r2", "nuovo")
    assert session.commit_response_ledger_projection(newer)
    history_after_newer_commit = [dict(message) for message in session.history]

    assert not session.commit_response_ledger_projection(older)
    assert session.history == history_after_newer_commit


def test_consumed_pcm_is_exact_partial_prefix_and_repeated_words_are_not_deduplicated():
    ledger = ResponseEventLedger(response_id="r1", generation_id=4)
    ledger.record_generated_tokens([
        {"source_token_seq": 0, "token_id": 5, "text_piece": "ciao ", "exclusive_kv_cursor": 21},
        {"source_token_seq": 1, "token_id": 5, "text_piece": "ciao", "exclusive_kv_cursor": 22},
    ])
    ledger.set_context(revision_id="r0", cache_epoch_id="epoch-1")
    branch = _branch(ledger)
    _complete_client_pcm_segment(
        branch,
        speaker_segment_id="r1.seg0",
        token_packets=[
            {"source_token_seq": 0, "token_id": 5, "text_piece": "ciao ", "exclusive_kv_cursor": 21},
            {"source_token_seq": 1, "token_id": 5, "text_piece": "ciao", "exclusive_kv_cursor": 22},
        ],
    )
    assert _record_client_consumed_pcm(branch, _ack(branch, end_sample=5, final=False))
    ledger.close_response(intended_text="ciao ciao", finish_reason="cancelled", completed=False, cancelled=True)

    projection = ledger.project()
    assert projection.confirmed_text == ""
    assert projection.text_status == "unknown"
    assert projection.retained_pcm_samples == 5
    stored = np.frombuffer(projection.consumed_pcm_chunks[0]["pcm_f32_le"], dtype="<f4")
    np.testing.assert_array_equal(stored, np.arange(5, dtype=np.float32) / 10)


def test_closed_complete_segment_confirms_repeated_text_and_cache_ids_together():
    ledger = ResponseEventLedger(response_id="r1", generation_id=4)
    branch = _branch(ledger)
    tokens = [
        {"source_token_seq": 0, "token_id": 5, "text_piece": "ciao ", "exclusive_kv_cursor": 21},
        {"source_token_seq": 1, "token_id": 5, "text_piece": "ciao", "exclusive_kv_cursor": 22},
    ]
    ledger.record_generated_tokens(tokens)
    ledger.record_tts_consumed(1)
    assert _complete_client_pcm_segment(branch, speaker_segment_id="r1.seg0", token_packets=tokens)
    assert _record_client_consumed_pcm(branch, _ack(branch, end_sample=12, final=True))
    assert branch["client_consumed_prefix_token_ids"] == (5, 5)
    assert branch["client_consumed_prefix_text"] == "ciao ciao"
    ledger.close_response(intended_text="ciao ciao", finish_reason="stop", completed=True)

    projection = ledger.project()
    assert projection.text_status == "completed"
    assert projection.confirmed_text == "ciao ciao"
    assert [row["token_id"] for row in projection.confirmed_source_tokens] == [5, 5]
    assert [row["source_token_seq"] for row in projection.confirmed_source_tokens] == [0, 1]


def test_full_pcm_ack_does_not_complete_text_past_tts_consumed_frontier():
    ledger = ResponseEventLedger(response_id="r1", generation_id=4)
    tokens = [
        {"source_token_seq": 0, "token_id": 5, "text_piece": "first ", "exclusive_kv_cursor": 21},
        {"source_token_seq": 1, "token_id": 6, "text_piece": "second", "exclusive_kv_cursor": 22},
    ]
    ledger.record_generated_tokens(tokens)
    ledger.record_tts_consumed(0)
    branch = _branch(ledger)
    assert _complete_client_pcm_segment(
        branch, speaker_segment_id="r1.seg0", token_packets=tokens[:1]
    )
    assert _record_client_consumed_pcm(branch, _ack(branch, end_sample=12, final=True))
    ledger.close_response(
        intended_text="first second", finish_reason="length", completed=True
    )

    projection = ledger.project()
    assert projection.all_pcm_consumed
    assert projection.tts_consumed_source_seq == 0
    assert [row["source_token_seq"] for row in projection.confirmed_source_tokens] == [0]
    assert projection.confirmed_text == "first "
    assert projection.text_status == "partial"


def test_full_pcm_ack_does_not_complete_text_with_uncovered_closed_unit_suffix():
    ledger = ResponseEventLedger(response_id="r1", generation_id=4)
    tokens = [
        {"source_token_seq": 0, "token_id": 5, "text_piece": "first ", "exclusive_kv_cursor": 21},
        {"source_token_seq": 1, "token_id": 6, "text_piece": "second", "exclusive_kv_cursor": 22},
    ]
    ledger.record_generated_tokens(tokens)
    ledger.record_tts_consumed(1)
    branch = _branch(ledger)
    assert _complete_client_pcm_segment(
        branch, speaker_segment_id="r1.seg0", token_packets=tokens[:1]
    )
    assert _record_client_consumed_pcm(branch, _ack(branch, end_sample=12, final=True))
    ledger.close_response(
        intended_text="first second", finish_reason="length", completed=True
    )

    projection = ledger.project()
    assert projection.all_pcm_consumed
    assert projection.tts_consumed_source_seq == 1
    assert [row["source_token_seq"] for row in projection.confirmed_source_tokens] == [0]
    assert projection.confirmed_text == "first "
    assert projection.text_status == "partial"


def test_length_termination_keeps_pcm_frontier_separate_from_text_completion():
    ledger = ResponseEventLedger(response_id="r1", generation_id=4)
    tokens = [
        {"source_token_seq": 0, "token_id": 5, "text_piece": "first ", "exclusive_kv_cursor": 21},
        {"source_token_seq": 1, "token_id": 6, "text_piece": "second", "exclusive_kv_cursor": 22},
    ]
    ledger.record_generated_tokens(tokens)
    ledger.record_tts_consumed(1)
    branch = _branch(ledger)
    assert _complete_client_pcm_segment(
        branch, speaker_segment_id="r1.seg0", token_packets=tokens
    )
    assert _record_client_consumed_pcm(branch, _ack(branch, end_sample=12, final=True))
    ledger.close_response(
        intended_text="first second", finish_reason="length", completed=True
    )

    projection = ledger.project()
    assert projection.all_pcm_consumed
    assert projection.confirmed_text == "first second"
    assert projection.text_status == "partial"
