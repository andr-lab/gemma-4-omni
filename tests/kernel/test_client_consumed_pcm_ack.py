import asyncio
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from duplex_kernel.server_pcm_tracker import (
    _client_consumed_pcm_ack_duplicate_status,
    _copy_emitted_pcm_range,
    _complete_client_pcm_segment,
    _record_client_consumed_pcm,
    _record_emitted_pcm_packet,
)
from duplex_kernel.server_speech_ack import _client_consumed_v1_token_packets
from duplex_kernel.server_response_control import (
    _cancel_response_branch_playback,
    _native_control6_speaker_active,
    _native_control6_worker_fail_closed_context,
    _response_branch_has_pending_pcm,
    _response_branch_ack_retained,
    _wait_for_response_playback_drain,
)
from duplex_kernel.response_event_ledger import ResponseEventLedger
from duplex_kernel.server_unified_cache import (
    _commit_deferred_acoustic_history_archive,
    _finalize_unified_native_cache_branch,
    _prepare_deferred_acoustic_history_finalization,
)
from duplex_kernel.session import DuplexSession
from duplex_kernel.turn_policy import ControlAction, ControlState, PolicyDecision


def _branch(*, with_event=True):
    result = {
        "response_id": "response-1",
        "session_generation_id": 9,
        "speaker_segment_id": "response-1.seg0",
        "client_consumed_pcm_sequence": -1,
        "client_consumed_segments": (),
        "spoken_prefix_base_cursor": 12,
        "spoken_prefix_exclusive_cursor": 12,
        "spoken_prefix_token_ids": (),
        "spoken_prefix_text_parts": (),
        "spoken_prefix_text": "",
        "client_consumed_prefix_token_ids": (),
        "client_consumed_prefix_exclusive_cursor": 12,
        "client_consumed_prefix_text_parts": (),
        "client_consumed_prefix_text": "",
        "spoken_prefix_lock": threading.RLock(),
    }
    if with_event:
        result["playback_ack_event"] = asyncio.Event()
    return result


def _announce_segment(
    branch,
    *,
    speaker_segment_id="response-1.seg0",
    start_sample=0,
    end_sample=2400,
    token_packets=None,
):
    packet = SimpleNamespace(
        generation_id=9,
        speaker_segment_id=speaker_segment_id,
        start_sample=start_sample,
        end_sample=end_sample,
        packet_id=3,
        pcm=np.linspace(-0.5, 0.5, end_sample - start_sample, dtype=np.float32),
    )
    _record_emitted_pcm_packet(branch, packet)
    marker = _complete_client_pcm_segment(
        branch,
        speaker_segment_id=speaker_segment_id,
        token_packets=token_packets or ({
            "token_id": 101,
            "exclusive_kv_cursor": 13,
            "text_piece": "ciao",
        },),
    )
    assert marker is not None
    return marker


def _ack(marker, *, sequence=0):
    return {
        **marker,
        "type": "client_consumed_pcm",
        "ack_version": 1,
        "sequence": sequence,
        "consumption_semantics": "logical_client_device_buffer",
    }


def _ack_v2(branch, segment_id, *, end_sample, sequence=0, final=False):
    segment = branch["playback_segments"][segment_id]
    return {
        "type": "client_consumed_pcm",
        "ack_version": 2,
        "response_id": branch["response_id"],
        "generation_id": branch["session_generation_id"],
        "speaker_segment_id": segment_id,
        "start_sample": segment["start_sample"],
        "end_sample": end_sample,
        "sample_rate": 24_000,
        "sequence": sequence,
        "consumption_semantics": "logical_client_device_buffer",
        "final": final,
    }


class _FinalizerWorker:
    def __init__(self, *, cursor, generation_id=9):
        self.cursor = cursor
        self.generation_id = generation_id
        self.rollback_targets = []
        self.append_calls = []
        self.drain_open_modes = []

    @staticmethod
    def _done(value=None):
        future = Future()
        future.set_result(value)
        return future

    def session_state(self, _session_id):
        return self._done(SimpleNamespace(
            generation_id=self.generation_id,
            cache_position=self.cursor,
        ))

    def reserve_tail_tokens(
        self, _session_id, *, expected_generation_id, expected_cache_position, token_count
    ):
        assert expected_generation_id == self.generation_id
        assert expected_cache_position == self.cursor
        assert token_count >= 0
        return self._done(None)

    def append_text_tokens(
        self, _session_id, token_ids, *, expected_generation_id, expected_cache_position,
        consume_reserved_tokens,
    ):
        assert expected_generation_id == self.generation_id
        assert expected_cache_position == self.cursor
        token_ids = tuple(token_ids)
        assert consume_reserved_tokens == len(token_ids)
        self.append_calls.append(token_ids)
        self.cursor += len(token_ids)
        return self._done(None)

    def rollback_assistant_branch(
        self, _session_id, *, expected_generation_id, expected_cache_position,
        target_cache_position,
    ):
        assert expected_generation_id == self.generation_id
        assert expected_cache_position == self.cursor
        self.rollback_targets.append(target_cache_position)
        self.cursor = target_cache_position
        return self._done(SimpleNamespace(cache_position=self.cursor))

    def drain_queued_audio(
        self, _session_id, *, expected_generation_id, expected_cache_position,
        user_turn_already_open, **_kwargs,
    ):
        assert expected_generation_id == self.generation_id
        assert expected_cache_position == self.cursor
        self.drain_open_modes.append(user_turn_already_open)
        return self._done(SimpleNamespace(
            cache_position=self.cursor,
            queued_audio_drain_metrics=None,
        ))


class _AudioHistorySession:
    def __init__(self):
        self.archive_updates = []
        self.completed_responses = []

    def update_audio_history_archive_assistant(self, archive_id, text, **kwargs):
        self.archive_updates.append((archive_id, text, kwargs))
        return True

    def commit_completed_assistant_response(self, *args, **kwargs):
        self.completed_responses.append((args, kwargs))
        return True


def _deferred_branch(*, pending_pcm=False):
    branch = _branch()
    branch.update({
        "listener_generation_id": 9,
        "cache_position": 17,
        "turn_start_cache_position": 12,
        "queue_open": True,
        "finalized": False,
        "close_ids": (90, 91),
        "transition_token_ids": (5, 6),
        "deferred_acoustic_history_commit": {
            "archive_id": 41,
            "response_done_at_monotonic_ns": 123456,
        },
        "client_consumed_pcm_sequence": 0,
        "client_consumed_segments": (),
        "client_consumed_prefix_token_ids": (101,),
        "client_consumed_prefix_exclusive_cursor": 15,
        "client_consumed_prefix_text_parts": ("ciao",),
        "client_consumed_prefix_text": "ciao",
        "spoken_prefix_lock": threading.RLock(),
        "playback_segments": ({
            "response-1.seg1": {
                "speaker_segment_id": "response-1.seg1",
                "start_sample": 0,
                "emitted_end_sample": 120,
            },
        } if pending_pcm else {}),
    })
    return branch


def test_complete_segment_ack_commits_only_the_exact_marker_range():
    branch = _branch()
    marker = _announce_segment(branch)
    assert branch["client_consumed_prefix_text"] == ""

    assert _record_client_consumed_pcm(branch, _ack(marker))
    assert branch["client_consumed_segments"] == ("response-1.seg0",)
    assert branch["spoken_prefix_token_ids"] == (101,)
    assert branch["spoken_prefix_exclusive_cursor"] == 13
    assert branch["spoken_prefix_text"] == "ciao"
    assert branch["client_consumed_prefix_token_ids"] == (101,)
    assert branch["client_consumed_prefix_exclusive_cursor"] == 13
    assert branch["client_consumed_prefix_text"] == "ciao"
    assert branch["client_consumed_pcm_events"][0]["consumption_semantics"] == (
        "logical_client_device_buffer"
    )


def test_ack_frontier_is_independent_of_pcm_already_emitted_ahead():
    branch = _branch()
    branch["spoken_prefix_token_ids"] = (202,)
    branch["spoken_prefix_text_parts"] = ("poi",)
    branch["spoken_prefix_text"] = "poi"
    branch["spoken_prefix_exclusive_cursor"] = 14
    marker = _announce_segment(branch)

    assert _record_client_consumed_pcm(branch, _ack(marker))
    assert branch["spoken_prefix_text"] == "poi"
    assert branch["client_consumed_prefix_token_ids"] == (101,)
    assert branch["client_consumed_prefix_exclusive_cursor"] == 13
    assert branch["client_consumed_prefix_text"] == "ciao"


def test_first_segment_commits_after_transition_tokens_advance_spoken_prefix_base():
    branch = _branch()
    branch["spoken_prefix_base_cursor"] = 1238
    branch["spoken_prefix_exclusive_cursor"] = 1238
    branch.pop("client_consumed_prefix_exclusive_cursor")
    marker = _announce_segment(
        branch,
        token_packets=({
            "token_id": 101,
            "exclusive_kv_cursor": 1239,
            "text_piece": "ciao",
        },),
    )

    assert _record_client_consumed_pcm(branch, _ack(marker))
    assert branch["spoken_prefix_exclusive_cursor"] == 1239
    assert branch["client_consumed_prefix_exclusive_cursor"] == 1239
    assert branch["client_consumed_prefix_token_ids"] == (101,)


def test_first_segment_with_stale_pre_transition_client_consumed_cursor_succeeds():
    branch = _branch()
    branch["client_consumed_prefix_exclusive_cursor"] = 797
    branch["spoken_prefix_base_cursor"] = 803
    branch["spoken_prefix_exclusive_cursor"] = 803
    marker = _announce_segment(
        branch,
        token_packets=({
            "token_id": 101,
            "exclusive_kv_cursor": 804,
            "text_piece": "ciao",
        },),
    )

    assert _record_client_consumed_pcm(branch, _ack(marker))
    assert branch["client_consumed_segments"] == ("response-1.seg0",)
    assert branch["client_consumed_prefix_token_ids"] == (101,)
    assert branch["client_consumed_prefix_exclusive_cursor"] == 804


def test_first_segment_uses_first_packet_cursor_when_spoken_base_is_missing():
    branch = _branch()
    branch.pop("spoken_prefix_base_cursor")
    branch.pop("spoken_prefix_exclusive_cursor")
    branch.pop("client_consumed_prefix_exclusive_cursor")
    branch["turn_start_cache_position"] = 7
    marker = _announce_segment(
        branch,
        token_packets=({
            "token_id": 101,
            "exclusive_kv_cursor": 1239,
            "text_piece": "ciao",
        },),
    )

    assert _record_client_consumed_pcm(branch, _ack(marker))
    assert branch["spoken_prefix_exclusive_cursor"] == 1239
    assert branch["client_consumed_prefix_exclusive_cursor"] == 1239


def test_second_contiguous_segment_advances_client_consumed_cursor():
    branch = _branch()
    first_marker = _announce_segment(branch)
    assert _record_client_consumed_pcm(branch, _ack(first_marker))

    second_marker = _announce_segment(
        branch,
        speaker_segment_id="response-1.seg1",
        start_sample=2400,
        end_sample=4800,
        token_packets=({
            "token_id": 102,
            "exclusive_kv_cursor": 14,
            "text_piece": " dopo",
        },),
    )
    assert _record_client_consumed_pcm(branch, _ack(second_marker, sequence=1))
    assert branch["client_consumed_prefix_token_ids"] == (101, 102)
    assert branch["client_consumed_prefix_exclusive_cursor"] == 14
    assert branch["client_consumed_prefix_text"] == "ciao dopo"


@pytest.mark.parametrize(
    ("second_packet", "expected_reason"),
    [
        ({"token_id": 102, "exclusive_kv_cursor": 15, "text_piece": " dopo"}, "non_contiguous_cursor"),
        ({"token_id": -1, "exclusive_kv_cursor": 14, "text_piece": " dopo"}, "invalid_token_id"),
    ],
)
def test_invalid_later_packet_rejects_segment_and_logs_packet_context(
    second_packet, expected_reason, caplog
):
    branch = _branch()
    marker = _announce_segment(
        branch,
        token_packets=(
            {"token_id": 101, "exclusive_kv_cursor": 13, "text_piece": "ciao"},
            second_packet,
        ),
    )

    assert not _record_client_consumed_pcm(branch, _ack(marker))
    assert branch["client_consumed_segments"] == ()
    assert branch["client_consumed_prefix_token_ids"] == ()
    assert branch["client_consumed_prefix_exclusive_cursor"] == 12
    assert "packet_index=1" in caplog.text
    assert f"reason={expected_reason}" in caplog.text
    assert f"token_cursor={second_packet['exclusive_kv_cursor']}" in caplog.text
    assert "expected_cursor=14" in caplog.text
    assert f"token_id={second_packet['token_id']}" in caplog.text
    assert f"text_piece={second_packet['text_piece']!r}" in caplog.text
    assert "segment_id=response-1.seg0" in caplog.text


def test_v1_client_segment_owns_only_committed_tokens_at_the_vocalized_frontier():
    branch = {
        "spoken_prefix_ack_provenance": "v1_live_qwen_frame_ack",
        "spoken_prefix_exclusive_cursor": 13,
        "segment_token_packets": [
            {"token_id": 101, "exclusive_kv_cursor": 13, "text_piece": "ciao", "kv_committed": True},
            {"token_id": 102, "exclusive_kv_cursor": 14, "text_piece": " dopo", "kv_committed": True},
        ],
        "say_token_packets": [
            SimpleNamespace(
                token_id=101,
                exclusive_kv_cursor=13,
                text_piece="ciao",
                kv_committed=True,
            ),
            SimpleNamespace(
                token_id=102,
                exclusive_kv_cursor=14,
                text_piece=" dopo",
                kv_committed=True,
            ),
        ],
    }

    packets = _client_consumed_v1_token_packets(
        branch,
        termination_reason="length",
    )

    assert packets == ({
        "token_id": 101,
        "exclusive_kv_cursor": 13,
        "text_piece": "ciao",
        "source_token_index": 0,
        "termination_reason": "length",
    },)


def test_v1_client_segment_uses_logical_frontier_across_cursor_rebase_and_repeated_text():
    ledger = ResponseEventLedger(response_id="response-1", generation_id=9)
    source_rows = [
        {
            "token_id": 101,
            "text_piece": "ha",
            "source_token_seq": 10,
            "exclusive_kv_cursor": 7,
            "revision_id": "r0",
            "cache_epoch_id": "epoch-a",
        },
        {
            "token_id": 101,
            "text_piece": "ha",
            "source_token_seq": 12,
            "exclusive_kv_cursor": 3,
            "revision_id": "r1",
            "cache_epoch_id": "epoch-b",
        },
    ]
    ledger.record_generated_tokens(source_rows)
    branch = {
        "spoken_prefix_ack_provenance": "v1_live_qwen_frame_ack",
        "spoken_prefix_source_seq": 12,
        "spoken_prefix_exclusive_cursor": 3,
        "response_event_ledger": ledger,
        "segment_token_packets": [
            {
                **row,
                "token_index": index,
                "kv_committed": True,
                "physical_kv_provenance": [{
                    "revision_id": row["revision_id"],
                    "cache_epoch_id": row["cache_epoch_id"],
                    "physical_kv_cursor": row["exclusive_kv_cursor"],
                }],
            }
            for index, row in enumerate(source_rows)
        ],
        "say_token_packets": [
            SimpleNamespace(
                token_id=row["token_id"],
                exclusive_kv_cursor=row["exclusive_kv_cursor"],
                text_piece=row["text_piece"],
                kv_committed=True,
            )
            for row in source_rows
        ],
    }

    packets = _client_consumed_v1_token_packets(branch, termination_reason="length")

    assert [row["source_token_seq"] for row in packets] == [10, 12]
    assert [row["exclusive_kv_cursor"] for row in packets] == [7, 3]
    assert [row["text_piece"] for row in packets] == ["ha", "ha"]


@pytest.mark.parametrize(
    "change",
    [
        {"response_id": "stale-response"},
        {"generation_id": 10},
        {"speaker_segment_id": "response-1.seg1"},
        {"start_sample": 1},
        {"end_sample": 2399},
        {"sample_rate": 16000},
        {"consumption_semantics": "acoustic_output"},
        {"sequence": -1},
    ],
)
def test_stale_or_incomplete_ack_never_advances_prefix(change):
    branch = _branch()
    marker = _announce_segment(branch)
    acknowledgement = _ack(marker)
    acknowledgement.update(change)

    assert not _record_client_consumed_pcm(branch, acknowledgement)
    assert branch["client_consumed_segments"] == ()
    assert branch["spoken_prefix_token_ids"] == ()
    assert branch["spoken_prefix_exclusive_cursor"] == 12


def test_v2_partial_progress_before_marker_preserves_exact_pcm_without_claiming_text():
    ledger = ResponseEventLedger(response_id="response-1", generation_id=9)
    branch = _branch()
    branch["response_event_ledger"] = ledger
    _record_emitted_pcm_packet(branch, SimpleNamespace(
        generation_id=9,
        speaker_segment_id="response-1.seg0",
        start_sample=0,
        end_sample=24,
        packet_id=0,
        pcm=np.linspace(-0.5, 0.5, 24, dtype=np.float32),
    ))
    progress = _ack_v2(
        branch, "response-1.seg0", end_sample=7, sequence=3, final=False
    )

    assert _record_client_consumed_pcm(branch, progress)
    assert branch["playback_segments"]["response-1.seg0"]["client_consumed_end_sample"] == 7
    assert branch.get("client_consumed_prefix_text", "") == ""
    assert branch["playback_segments"]["response-1.seg0"]["pcm_samples_consumed"] == 7
    projection = ledger.project()
    assert projection.retained_pcm_samples == 7
    consumed = np.frombuffer(projection.consumed_pcm_chunks[0]["pcm_f32_le"], dtype="<f4")
    np.testing.assert_array_equal(
        consumed,
        np.linspace(-0.5, 0.5, 24, dtype=np.float32)[:7],
    )


def test_v2_duplicate_is_idempotent_but_mutated_duplicate_and_regression_reject():
    branch = _branch()
    _announce_segment(branch)
    first = _ack_v2(branch, "response-1.seg0", end_sample=500, sequence=1)
    assert _record_client_consumed_pcm(branch, first)
    assert _record_client_consumed_pcm(branch, dict(first))
    assert not _record_client_consumed_pcm(branch, {**first, "end_sample": 501})
    assert not _record_client_consumed_pcm(
        branch,
        _ack_v2(branch, "response-1.seg0", end_sample=499, sequence=2),
    )


def test_v2_duplicate_identity_survives_more_than_300_progress_acks_until_retirement(
    monkeypatch,
):
    ledger = ResponseEventLedger(response_id="response-1", generation_id=9)
    branch = _branch()
    branch["response_event_ledger"] = ledger
    _announce_segment(branch, end_sample=400)

    ledger_records = 0
    original_record_pcm = ledger.record_pcm_consumed

    def counted_record_pcm(**kwargs):
        nonlocal ledger_records
        ledger_records += 1
        return original_record_pcm(**kwargs)

    monkeypatch.setattr(ledger, "record_pcm_consumed", counted_record_pcm)
    acknowledgements = []
    for sequence in range(301):
        acknowledgement = _ack_v2(
            branch,
            "response-1.seg0",
            end_sample=sequence + 1,
            sequence=sequence,
            final=False,
        )
        assert _record_client_consumed_pcm(branch, acknowledgement)
        acknowledgements.append(acknowledgement)

    first = acknowledgements[0]
    assert len(branch["client_consumed_pcm_ack_by_sequence"]) == 301
    assert _client_consumed_pcm_ack_duplicate_status(branch, first) is True
    projection_before_duplicate = ledger.project()
    event_count_before_duplicate = len(branch["client_consumed_pcm_events"])
    prefix_before_duplicate = (
        branch["spoken_prefix_token_ids"],
        branch["spoken_prefix_text"],
        branch["spoken_prefix_exclusive_cursor"],
    )

    assert _record_client_consumed_pcm(branch, dict(first)) is True
    assert ledger.project() == projection_before_duplicate
    assert len(branch["client_consumed_pcm_events"]) == event_count_before_duplicate
    assert ledger_records == 301
    assert (
        branch["spoken_prefix_token_ids"],
        branch["spoken_prefix_text"],
        branch["spoken_prefix_exclusive_cursor"],
    ) == prefix_before_duplicate

    changed_duplicate = {**first, "end_sample": 2}
    assert _client_consumed_pcm_ack_duplicate_status(branch, changed_duplicate) is False
    assert _record_client_consumed_pcm(branch, changed_duplicate) is False
    assert ledger.project() == projection_before_duplicate
    assert _client_consumed_pcm_ack_duplicate_status(None, first) is None
    assert not _record_client_consumed_pcm(None, first)


def test_cancelled_ack_branch_retained_through_final_watermark_grace_without_pending_pcm():
    branch = _branch()
    marker = _announce_segment(branch, end_sample=24)
    _cancel_response_branch_playback(branch)
    deadline = branch["playback_ack_final_deadline_monotonic"]
    assert _response_branch_has_pending_pcm(branch)
    assert _response_branch_ack_retained(branch, now_monotonic=deadline - 0.001)
    assert not _response_branch_ack_retained(branch, now_monotonic=deadline + 0.001)

    segment = branch["playback_segments"][marker["speaker_segment_id"]]
    segment["client_consumed_end_sample"] = segment["emitted_end_sample"]
    assert not _response_branch_has_pending_pcm(branch)
    assert _response_branch_ack_retained(branch, now_monotonic=deadline - 0.001)
    assert not _response_branch_ack_retained(branch, now_monotonic=deadline + 0.001)


def test_progress_ack_during_speak_does_not_commit_assistant_history_before_done_and_full_playback():
    session = DuplexSession("history-ack-gate")
    session.history = [{"role": "user", "content": "domanda"}]
    response_id = "response-history"
    generation = session.begin_response(response_id)
    ledger = session.get_response_event_ledger(response_id)
    assert ledger is not None
    segment_id = generation.speaker_segment_id
    branch = _branch()
    branch.update({
        "response_id": response_id,
        "session_generation_id": generation.session_generation_id,
        "speaker_segment_id": segment_id,
        "spoken_prefix_base_cursor": 12,
        "spoken_prefix_exclusive_cursor": 12,
        "client_consumed_prefix_exclusive_cursor": 12,
        "response_event_ledger": ledger,
    })
    source_token = {
        "source_token_seq": 0,
        "token_id": 101,
        "text_piece": "Ciao",
        "exclusive_kv_cursor": 13,
    }
    ledger.record_generated_tokens([source_token])
    ledger.record_tts_consumed(0)
    pcm = np.linspace(-0.5, 0.5, 12, dtype=np.float32)
    _record_emitted_pcm_packet(branch, SimpleNamespace(
        generation_id=generation.session_generation_id,
        speaker_segment_id=segment_id,
        start_sample=0,
        end_sample=12,
        packet_id=0,
        pcm=pcm,
    ))
    assert _complete_client_pcm_segment(
        branch,
        speaker_segment_id=segment_id,
        token_packets=(source_token,),
    )

    partial = _ack_v2(branch, segment_id, end_sample=4, sequence=0, final=False)
    assert _record_client_consumed_pcm(branch, partial)
    assert session.history == [{"role": "user", "content": "domanda"}]
    assert session.snapshot_history_with_provenance().history == tuple(session.history)

    ledger.close_response(
        intended_text="Ciao",
        finish_reason="stop",
        completed=True,
        cancelled=False,
    )
    # Response DONE without complete playback leaves no assistant history text.
    assert session.commit_response_ledger_projection(ledger)
    assert session.history == [{"role": "user", "content": "domanda"}]

    final = _ack_v2(branch, segment_id, end_sample=12, sequence=1, final=True)
    assert _record_client_consumed_pcm(branch, final)
    assert ledger.project().all_pcm_consumed
    assert session.history == [{"role": "user", "content": "domanda"}]
    assert session.commit_response_ledger_projection(ledger)
    assert session.history == [
        {"role": "user", "content": "domanda"},
        {"role": "assistant", "content": "Ciao"},
    ]


def test_ledger_commit_failure_rolls_back_the_branch_ack_watermark():
    ledger = ResponseEventLedger(response_id="response-1", generation_id=9)
    branch = _branch()
    branch["response_event_ledger"] = ledger
    _announce_segment(branch)

    def fail_record(**_kwargs):
        raise RuntimeError("injected ledger commit failure")

    ledger.record_pcm_consumed = fail_record
    ack = _ack_v2(branch, "response-1.seg0", end_sample=100, sequence=0)

    assert not _record_client_consumed_pcm(branch, ack)
    segment = branch["playback_segments"]["response-1.seg0"]
    assert segment["client_consumed_end_sample"] == 0
    assert branch["client_consumed_pcm_sequence"] == -1
    assert branch["spoken_prefix_token_ids"] == ()
    assert ledger.project().pcm_samples_consumed == 0


def test_ledger_projection_failure_rolls_back_both_ack_and_pcm_frontier(monkeypatch):
    import duplex_kernel.server_pcm_tracker as pcm_tracker

    ledger = ResponseEventLedger(response_id="response-1", generation_id=9)
    branch = _branch()
    branch["response_event_ledger"] = ledger
    _announce_segment(branch)

    def fail_projection(*_args, **_kwargs):
        raise RuntimeError("injected branch projection failure")

    monkeypatch.setattr(pcm_tracker, "_commit_client_consumed_segment", fail_projection)
    ack = _ack_v2(
        branch,
        "response-1.seg0",
        end_sample=2400,
        sequence=0,
        final=True,
    )

    assert not _record_client_consumed_pcm(branch, ack)
    segment = branch["playback_segments"]["response-1.seg0"]
    assert segment["client_consumed_end_sample"] == 0
    assert segment["client_consumed"] is False
    assert branch["client_consumed_pcm_sequence"] == -1
    assert branch.get("client_consumed_pcm_events", []) == []
    assert branch.get("client_consumed_pcm_ack_by_sequence", {}) == {}
    projection = ledger.project()
    assert projection.pcm_samples_consumed == 0
    assert not projection.all_pcm_consumed


def test_v2_final_old_branch_ack_is_accepted_after_cancel_within_retention_window():
    branch = _branch()
    _announce_segment(branch)
    _cancel_response_branch_playback(branch)
    final = _ack_v2(
        branch, "response-1.seg0", end_sample=384, sequence=2, final=True
    )

    assert _record_client_consumed_pcm(branch, final)
    assert branch["playback_segments"]["response-1.seg0"]["client_consumed_end_sample"] == 384
    assert not _record_client_consumed_pcm(branch, {**final, "sequence": 3, "end_sample": 500})


def _cancelled_branch_with_pcm():
    branch = _branch()
    _announce_segment(branch)
    branch["playback_cancelled"] = True
    branch["playback_ack_final_deadline_monotonic"] = time.monotonic() + 5.0
    return branch


def test_v2_progressive_and_final_ack_are_accepted_on_cancelled_branch():
    branch = _cancelled_branch_with_pcm()
    segment_id = "response-1.seg0"

    progress = _ack_v2(
        branch, segment_id, end_sample=1200, sequence=1, final=False
    )
    assert _record_client_consumed_pcm(branch, progress) is True

    segment = branch["playback_segments"][segment_id]
    assert branch["client_consumed_pcm_sequence"] == 1
    assert segment["client_consumed_end_sample"] == 1200
    assert segment["pcm_samples_consumed"] == 1200
    assert segment["client_consumed_final"] is False
    assert segment["client_consumed"] is False
    assert branch["client_consumed_segments"] == ()

    final = _ack_v2(
        branch, segment_id, end_sample=2400, sequence=2, final=True
    )
    assert _record_client_consumed_pcm(branch, final) is True
    assert branch["client_consumed_pcm_sequence"] == 2
    assert segment["client_consumed_end_sample"] == 2400
    assert segment["pcm_samples_consumed"] == 2400
    assert segment["client_consumed_final"] is True
    assert segment["client_consumed"] is True
    assert branch["client_consumed_segments"] == (segment_id,)


@pytest.mark.parametrize(
    ("ack_changes", "expire_deadline"),
    [
        ({"sequence": 0, "end_sample": 800}, False),
        ({"sequence": 2, "generation_id": 10, "end_sample": 800}, False),
        ({"sequence": 2, "end_sample": 800}, True),
    ],
    ids=("regressive-sequence", "wrong-generation", "expired-deadline"),
)
def test_v2_cancelled_branch_rejects_regressive_stale_or_expired_ack(
    ack_changes, expire_deadline
):
    branch = _cancelled_branch_with_pcm()
    segment_id = "response-1.seg0"
    progress = _ack_v2(
        branch, segment_id, end_sample=400, sequence=1, final=False
    )
    assert _record_client_consumed_pcm(branch, progress) is True

    invalid = _ack_v2(
        branch, segment_id, end_sample=800, sequence=2, final=False
    )
    invalid.update(ack_changes)
    if expire_deadline:
        branch["playback_ack_final_deadline_monotonic"] = time.monotonic() - 1.0

    assert _record_client_consumed_pcm(branch, invalid) is False
    assert branch["client_consumed_pcm_sequence"] == 1
    assert branch["playback_segments"][segment_id]["client_consumed_end_sample"] == 400


def test_v2_cancel_final_allows_zero_consumed_samples_at_original_start():
    branch = _branch()
    _announce_segment(branch, start_sample=20, end_sample=32)
    _cancel_response_branch_playback(branch)
    final = _ack_v2(
        branch, "response-1.seg0", end_sample=20, sequence=0, final=True
    )

    assert _record_client_consumed_pcm(branch, final)
    assert branch["playback_segments"]["response-1.seg0"]["pcm_samples_consumed"] == 0


def test_cancel_fences_late_ack_and_further_emission():
    branch = _branch()
    marker = _announce_segment(branch)

    _cancel_response_branch_playback(branch)

    assert not _record_client_consumed_pcm(branch, _ack(marker))
    assert branch["client_consumed_segments"] == ()
    assert branch["spoken_prefix_token_ids"] == ()
    with pytest.raises(RuntimeError, match="cancelled response"):
        _record_emitted_pcm_packet(branch, SimpleNamespace(
            generation_id=9,
            speaker_segment_id="response-1.seg1",
            start_sample=0,
            end_sample=1200,
            packet_id=4,
        ))


def test_playback_drain_wait_is_bounded_and_missing_event_is_explicit():
    branch = _branch(with_event=False)
    _announce_segment(branch)
    with pytest.raises(RuntimeError, match="no ACK event"):
        asyncio.run(_wait_for_response_playback_drain(branch))

    branch = _branch()
    _announce_segment(branch)
    with pytest.raises(RuntimeError, match="ACK wait timed out"):
        asyncio.run(_wait_for_response_playback_drain(branch, timeout_s=0.01))
    assert branch["client_consumed_segments"] == ()
    assert branch["spoken_prefix_token_ids"] == ()


def test_cancellation_wakes_playback_drain_without_fabricating_consumption():
    async def scenario():
        branch = _branch()
        assert isinstance(branch["playback_ack_event"], asyncio.Event)
        _announce_segment(branch)
        waiter = asyncio.create_task(
            _wait_for_response_playback_drain(branch, timeout_s=1.0)
        )
        await asyncio.sleep(0)
        _cancel_response_branch_playback(branch)
        await waiter
        assert branch["client_consumed_segments"] == ()
        assert branch["spoken_prefix_token_ids"] == ()
        assert branch["client_playback_ack_missing"] is True

    asyncio.run(scenario())


def test_playback_drain_completes_after_client_consumed_pcm_ack():
    async def scenario():
        branch = _branch()
        ack_event = branch["playback_ack_event"]
        assert isinstance(ack_event, asyncio.Event)
        marker = _announce_segment(branch)

        waiter = asyncio.create_task(
            _wait_for_response_playback_drain(branch, timeout_s=1.0)
        )
        await asyncio.sleep(0)
        assert not waiter.done()

        with branch["spoken_prefix_lock"]:
            accepted = _record_client_consumed_pcm(branch, _ack(marker))
        assert accepted
        assert branch["playback_segments"]["response-1.seg0"]["client_consumed"] is True
        if not _response_branch_has_pending_pcm(branch):
            ack_event.set()

        await asyncio.wait_for(waiter, timeout=1.0)
        assert branch["client_consumed_segments"] == ("response-1.seg0",)
        assert branch["spoken_prefix_exclusive_cursor"] == 13

    asyncio.run(scenario())


def test_finalization_syncs_consumed_prefix_without_deferred_commit_metadata():
    branch = _deferred_branch()
    branch["deferred_acoustic_history_commit"] = None
    branch["spoken_prefix_token_ids"] = (7,)
    branch["spoken_prefix_exclusive_cursor"] = 13
    branch["spoken_prefix_text_parts"] = ("old",)
    branch["spoken_prefix_text"] = "old"
    commit_calls = []

    history_committed = _prepare_deferred_acoustic_history_finalization(
        branch,
        history_committed=True,
        commit_deferred_acoustic_history=lambda: commit_calls.append(True),
    )

    assert history_committed is True
    assert branch["spoken_prefix_token_ids"] == (101,)
    assert branch["spoken_prefix_exclusive_cursor"] == 15
    assert branch["spoken_prefix_text_parts"] == ("ciao",)
    assert branch["spoken_prefix_text"] == "ciao"
    assert commit_calls == []


def test_native_control6_speaker_is_inactive_when_closing_without_pending_pcm():
    assert _native_control6_speaker_active(
        active_response=True,
        control_state=ControlState.SPEAKING,
        pending_pcm_playback=False,
        closing=True,
    ) is False
    assert _native_control6_speaker_active(
        active_response=True,
        control_state=ControlState.SPEAKING,
        pending_pcm_playback=True,
        closing=True,
    ) is True


@pytest.mark.parametrize("branch_flag", ["closing", "playback_drained"])
def test_draining_response_does_not_emit_context_unavailable_error(branch_flag):
    response_branch = {
        branch_flag: True,
        "playback_segments": {
            "response-1.seg0": {
                "start_sample": 0,
                "emitted_end_sample": 1200,
            },
        },
        "client_consumed_segments": (),
        "spoken_prefix_lock": threading.RLock(),
    }
    decision = PolicyDecision(
        action=ControlAction.HARD_STOP,
        frame_id=1323,
        reason="stable_q2_context_unavailable_fail_closed",
        generation_id=9,
    )

    should_emit_context_error = _native_control6_worker_fail_closed_context(
        diagnostics={"system_fail_closed_action": True},
        decision=decision,
        speaker_active=True,
        active_response=True,
        response_branch=response_branch,
    )

    assert should_emit_context_error is False


def test_complete_consumption_commits_deferred_archive_and_closes_without_rollback():
    branch = _deferred_branch()
    session = _AudioHistorySession()

    history_committed = _prepare_deferred_acoustic_history_finalization(
        branch,
        history_committed=False,
        commit_deferred_acoustic_history=lambda: _commit_deferred_acoustic_history_archive(
            branch,
            session,
            response_token="turn-1",
            response_id="response-1",
            user_onset_frame_id=7,
        ),
    )

    assert history_committed is True
    assert branch["history_committed"] is True
    assert branch["spoken_prefix_token_ids"] == (101,)
    assert branch["spoken_prefix_exclusive_cursor"] == 15
    assert session.archive_updates == [(
        41,
        "ciao",
        {
            "status": "completed",
            "response_done_at_monotonic_ns": 123456,
        },
    )]

    worker = _FinalizerWorker(cursor=17)
    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        "session-1",
        branch,
        history_committed=history_committed,
        close_ids=(90, 91),
    ))

    assert worker.rollback_targets == []
    assert worker.append_calls == [(90, 91)]
    assert worker.drain_open_modes == [False]
    assert branch["finalized"] is True


def test_interrupted_consumption_rolls_back_and_rebuilds_exact_spoken_prefix():
    branch = _deferred_branch(pending_pcm=True)
    session = _AudioHistorySession()

    history_committed = _prepare_deferred_acoustic_history_finalization(
        branch,
        history_committed=True,
        commit_deferred_acoustic_history=lambda: _commit_deferred_acoustic_history_archive(
            branch,
            session,
            response_token="turn-1",
            response_id="response-1",
            user_onset_frame_id=7,
        ),
    )

    assert history_committed is False
    assert branch["history_committed"] is False
    assert session.archive_updates == []
    assert branch["spoken_prefix_token_ids"] == (101,)
    assert branch["spoken_prefix_exclusive_cursor"] == 15

    worker = _FinalizerWorker(cursor=17)
    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        "session-1",
        branch,
        history_committed=history_committed,
        close_ids=(90, 91),
    ))

    assert worker.rollback_targets == [12]
    assert worker.append_calls == [(5, 6), (101, 90), (91,)]
    assert worker.drain_open_modes == [False]
    assert branch["spoken_prefix_rebuilt"] is True
    assert branch["history_committed"] is False

    assert _commit_deferred_acoustic_history_archive(
        branch,
        session,
        response_token="turn-1",
        response_id="response-1",
        user_onset_frame_id=7,
    )[0] is True
    assert session.archive_updates == [(
        41,
        "ciao",
        {
            "status": "partial",
            "response_done_at_monotonic_ns": 123456,
        },
    )]


def test_history_and_cache_prefix_share_the_same_ledger_projection_after_cancel():
    session = DuplexSession("ledger-history-kv")
    session.history = [{"role": "user", "content": "domanda"}]
    token = session.begin_response("r1")
    ledger = session.get_response_event_ledger("r1")
    ledger.record_generated_tokens([
        {"source_token_seq": 0, "token_id": 31, "text_piece": "sì ", "exclusive_kv_cursor": 13},
        {"source_token_seq": 1, "token_id": 32, "text_piece": "grazie", "exclusive_kv_cursor": 14},
    ])
    ledger.record_tts_consumed(0)
    branch = {
        "response_id": "r1",
        "session_generation_id": token.session_generation_id,
        "playback_segments": {},
        "client_consumed_pcm_sequence": -1,
        "client_consumed_segments": (),
        "spoken_prefix_base_cursor": 12,
        "turn_start_cache_position": 12,
        "response_event_ledger": ledger,
        "spoken_prefix_lock": threading.RLock(),
    }
    for segment_id, start, end, source_token in (
        ("r1.seg0", 0, 8, 0),
        ("r1.seg1", 0, 12, 1),
    ):
        _record_emitted_pcm_packet(branch, SimpleNamespace(
            generation_id=token.session_generation_id,
            speaker_segment_id=segment_id,
            start_sample=start,
            end_sample=end,
            packet_id=source_token,
            pcm=np.arange(end - start, dtype=np.float32) / 10,
        ))
        _complete_client_pcm_segment(
            branch,
            speaker_segment_id=segment_id,
            token_packets=({
                "source_token_seq": source_token,
                "token_id": 31 + source_token,
                "text_piece": ("sì " if source_token == 0 else "grazie"),
                "exclusive_kv_cursor": 13 + source_token,
            },),
        )
    assert _record_client_consumed_pcm(branch, _ack_v2(
        branch, "r1.seg0", end_sample=8, sequence=0, final=True
    ))
    _cancel_response_branch_playback(branch)
    assert _record_client_consumed_pcm(branch, _ack_v2(
        branch, "r1.seg1", end_sample=5, sequence=1, final=True
    ))
    ledger.close_response(
        intended_text="sì grazie",
        finish_reason="cancelled",
        completed=False,
        cancelled=True,
    )
    branch["deferred_acoustic_history_commit"] = {
        "archive_id": None,
        "response_done_at_monotonic_ns": 123,
    }

    history_committed = _prepare_deferred_acoustic_history_finalization(
        branch,
        history_committed=False,
        commit_deferred_acoustic_history=lambda: _commit_deferred_acoustic_history_archive(
            branch,
            session,
            response_token=token,
            response_id="r1",
            user_onset_frame_id=None,
        ),
    )

    assert history_committed is False
    assert session.history == [
        {"role": "user", "content": "domanda"},
        {"role": "assistant", "content": "sì "},
    ]
    assert branch["spoken_prefix_token_ids"] == (31,)
    assert branch["spoken_prefix_text"] == "sì "
    assert branch["spoken_prefix_exclusive_cursor"] == 13
