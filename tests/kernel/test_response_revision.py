from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, replace
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from duplex_kernel.live_say import LiveQwenSayStream
from duplex_kernel.listener_microbatch_worker import EphemeralDecodeHandle
from duplex_kernel.response_event_ledger import ResponseEventLedger
from duplex_kernel.response_revision import (
    ResponseRevisionCandidate,
    ResponseRevisionCoordinator,
    ResponseRevisionRequest,
    install_response_revision_transaction,
)
import duplex_kernel.response_revision_runtime as response_revision_runtime
from duplex_kernel.server_speech_ack import _mark_unified_say_packet_committed
from duplex_kernel.server_unified_cache import _finalize_unified_native_cache_branch
from duplex_kernel.session import SpokenTokenPacket


@dataclass(frozen=True)
class _GeneratedPacket:
    token_index: int
    logical_source_sequence: int
    token_id: int
    token_text: str
    hidden: torch.Tensor
    hidden_provenance: str = "token"
    kv_committed: bool = True
    exclusive_kv_cursor: int | None = None
    text_source_sequences: tuple[tuple[int, int], ...] = ()


class _Tokenizer:
    pieces = {17: "Ciao", 18: "é", 15: "!", 16: "?", 30: " ", 99: ""}

    def _spans(self, text):
        spans = []
        cursor = 0
        while cursor < len(text):
            piece = next(
                (value for value in self.pieces.values() if value and text.startswith(value, cursor)),
                None,
            )
            assert piece is not None, (text, cursor)
            spans.append((piece, (cursor, cursor + len(piece))))
            cursor += len(piece)
        return spans

    def __init__(self):
        from types import SimpleNamespace

        self.backend_tokenizer = SimpleNamespace(
            pre_tokenizer=SimpleNamespace(pre_tokenize_str=self._spans)
        )

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False):
        prefix = "<|im_start|>assistant\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        raw = text[len(prefix):-len(suffix)] if add_special_tokens else text
        spans = self._spans(raw)
        token_ids = {piece: token_id for token_id, piece in self.pieces.items() if piece}
        ids = [token_ids[piece] for piece, _span in spans]
        base = len(prefix) if add_special_tokens else 0
        offsets = (
            [(0, 0)] * 3
            + [(base + start, base + end) for _piece, (start, end) in spans]
            + [(0, 0)] * 5
            if add_special_tokens else [(start, end) for _piece, (start, end) in spans]
        )
        return {"input_ids": ids, "offset_mapping": offsets} if return_offsets_mapping else {"input_ids": ids}

    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[int(token_id)] for token_id in ids)


class _Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, *, text, return_tensors):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([self.tokenizer(text)["input_ids"]])}


def _stream():
    tokenizer = _Tokenizer()
    return LiveQwenSayStream(
        qwen_tokenizer=tokenizer,
        qwen_processor=_Processor(tokenizer),
        text_eos_token_id=99,
        max_tokens=16,
        cache_epoch=None,
        generation_id=7,
        segment_id=0,
    )


def _candidate(
    stream,
    ledger,
    *,
    rows=None,
    packets=None,
    packet_records=None,
    revision="r1",
    base_revision="r0",
    source_seq=2,
    token_id=15,
    text="!",
    cache_epoch=None,
):
    row = {
        "token_id": token_id,
        "text_piece": text,
        "hidden": torch.ones((1, 1536)),
        "source_token_seq": source_seq,
        "source_group": ((source_seq, token_id),),
        "exclusive_kv_cursor": 10,
        "revision_id": revision,
        "cache_epoch_id": cache_epoch,
        "physical_kv_provenance": [{
            "revision_id": revision,
            "cache_epoch_id": cache_epoch,
            "physical_kv_cursor": 10,
        }],
        "kv_committed": True,
    }
    packet = SpokenTokenPacket(
        token_id=token_id,
        text_piece=text,
        gemma_hidden_1536=torch.ones((1, 1536)),
        exclusive_kv_cursor=10,
    )
    record = {
        "token_id": token_id,
        "text_piece": text,
        "token_index": 0,
        "source_token_seq": source_seq,
        "revision_id": revision,
        "cache_epoch_id": cache_epoch,
        "physical_kv_provenance": [{
            "revision_id": revision,
            "cache_epoch_id": cache_epoch,
            "physical_kv_cursor": 10,
        }],
        "exclusive_kv_cursor": 10,
        "kv_committed": True,
    }
    watermark = SimpleNamespace(generation_id=3, state_epoch_id="epoch-a", frame_id=5)
    return ResponseRevisionCandidate(
        session_id="session-a",
        response_id="response-a",
        generation_id=7,
        base_revision_id=base_revision,
        revision_id=revision,
        history_revision=2,
        cache_epoch_id=cache_epoch,
        listener_watermark=watermark,
        latest_listener_snapshot=watermark,
        requested_frame_id=6,
        stream_frontier=stream.source_frontier_snapshot(),
        ledger_snapshot=ledger.source_frontier_snapshot(),
        frozen_prefix_token_ids=(),
        source_rows=tuple(rows if rows is not None else (row,)),
        packets=tuple(packets if packets is not None else (packet,)),
        packet_records=tuple(packet_records if packet_records is not None else (record,)),
        cumulative_text=text,
        think_result={"text": "!", "terminal_reason": "eos", "spoken_prefix_base_cursor": 9},
        cache_position=11,
        handle=SimpleNamespace(handle_id="candidate-handle"),
    )


def _transaction_context():
    stream = _stream()
    hidden = torch.full((1, 1, 1536), 3.0)
    stream.feed_piece(
        "Ciao",
        hidden,
        token_index=0,
        source_token_id=17,
        source_token_seq=0,
        exclusive_kv_cursor=4,
        revision_id="r0",
    )
    stream.feed_piece(
        "!",
        torch.full((1, 1, 1536), 4.0),
        token_index=1,
        source_token_id=15,
        source_token_seq=1,
        exclusive_kv_cursor=5,
        revision_id="r0",
    )
    ledger = ResponseEventLedger(response_id="response-a", generation_id=7)
    ledger.record_generated_tokens([{
        "token_id": 17,
        "text_piece": "Ciao",
        "source_token_seq": 0,
        "exclusive_kv_cursor": 4,
        "revision_id": "r0",
        "physical_kv_provenance": [{
            "revision_id": "r0", "cache_epoch_id": None, "physical_kv_cursor": 4,
        }],
    }, {
        "token_id": 15,
        "text_piece": "!",
        "source_token_seq": 1,
        "exclusive_kv_cursor": 5,
        "revision_id": "r0",
        "physical_kv_provenance": [{
            "revision_id": "r0", "cache_epoch_id": None, "physical_kv_cursor": 5,
        }],
    }])
    session = SimpleNamespace(
        _generation_lock=threading.RLock(),
        session_generation_id=7,
        response_id="response-a",
        active_response=True,
        semantic_context_revision=2,
    )
    packet = SpokenTokenPacket(
        token_id=17,
        text_piece="Ciao",
        gemma_hidden_1536=torch.full((1, 1536), 3.0),
        exclusive_kv_cursor=4,
    )
    second_packet = SpokenTokenPacket(
        token_id=15,
        text_piece="!",
        gemma_hidden_1536=torch.full((1, 1536), 4.0),
        exclusive_kv_cursor=5,
    )
    branch = {
        "response_id": "response-a",
        "session_generation_id": 7,
        "revision_id": "r0",
        "cache_epoch_id": None,
        "spoken_prefix_lock": threading.RLock(),
        "say_token_packets": [packet, second_packet],
        "segment_token_packets": [{
            "token_id": 17,
            "text_piece": "Ciao",
            "source_token_seq": 0,
            "revision_id": "r0",
            "cache_epoch_id": None,
            "physical_kv_provenance": [{
                "revision_id": "r0", "cache_epoch_id": None, "physical_kv_cursor": 4,
            }],
            "exclusive_kv_cursor": 4,
            "kv_committed": True,
        }, {
            "token_id": 15,
            "text_piece": "!",
            "source_token_seq": 1,
            "revision_id": "r0",
            "cache_epoch_id": None,
            "physical_kv_provenance": [{
                "revision_id": "r0", "cache_epoch_id": None, "physical_kv_cursor": 5,
            }],
            "exclusive_kv_cursor": 5,
            "kv_committed": True,
        }],
        "spoken_prefix_source_seq": -1,
        "spoken_prefix_text": "",
    }
    return session, branch, stream, ledger


def test_atomic_revision_replaces_unreserved_suffix_and_keeps_frozen_prefix():
    session, branch, stream, ledger = _transaction_context()
    candidate = _candidate(stream, ledger)

    assert install_response_revision_transaction(
        session=session,
        branch=branch,
        stream=stream,
        ledger=ledger,
        candidate=candidate,
    )
    assert branch["revision_id"] == "r1"
    assert [record["source_token_seq"] for record in branch["segment_token_packets"]] == [2]
    assert stream.source_frontier_snapshot().frozen_source_sequence == -1
    assert ledger.source_frontier_snapshot()["generated_source_sequence"] == 2
    assert ledger.project().source_tokens[0]["token_id"] == 15


def test_revisions_retire_unconsumed_suffix_ids_and_reuse_response_budget():
    session, branch, stream, ledger = _transaction_context()

    first = _candidate(stream, ledger, revision="r1", source_seq=2, token_id=15, text="!")
    assert install_response_revision_transaction(
        session=session,
        branch=branch,
        stream=stream,
        ledger=ledger,
        candidate=first,
    )
    assert branch["_generated_budget_used"] == 1
    assert ledger.source_frontier_snapshot()["next_source_sequence"] == 3

    second = _candidate(
        stream,
        ledger,
        revision="r2",
        base_revision="r1",
        source_seq=3,
        token_id=16,
        text="?",
    )
    assert install_response_revision_transaction(
        session=session,
        branch=branch,
        stream=stream,
        ledger=ledger,
        candidate=second,
    )
    snapshot = ledger.source_frontier_snapshot()
    assert branch["_generated_budget_used"] == 1
    assert branch["revision_id"] == "r2"
    assert snapshot["generated_source_sequence"] == 3
    assert snapshot["next_source_sequence"] == 4
    assert ledger.project().source_tokens == ({
        "source_token_seq": 3,
        "token_id": 16,
        "text_piece": "?",
        "physical_kv_cursor": 10,
        "revision_id": "r2",
        "cache_epoch_id": None,
        "physical_kv_provenance": [{
            "revision_id": "r2",
            "cache_epoch_id": None,
            "physical_kv_cursor": 10,
        }],
    },)


def test_compaction_epoch_is_cas_checked_and_adopted_with_revision():
    session, branch, stream, ledger = _transaction_context()
    session.listener_state = SimpleNamespace(
        cache_epoch_id=8,
        native_semantic_context_revision=2,
        applied_history_revision=2,
    )
    watermark = SimpleNamespace(
        generation_id=3,
        state_epoch_id="epoch-a",
        frame_id=5,
        cache_epoch_id=8,
        applied_history_revision=2,
    )
    snapshot = SimpleNamespace(
        generation_id=3,
        state_epoch_id="epoch-a",
        cache_epoch_id=8,
        semantic_context_revision=2,
        applied_history_revision=2,
    )
    candidate = _candidate(stream, ledger, revision="r1", cache_epoch=8)
    candidate = replace(
        candidate,
        listener_watermark=watermark,
        latest_listener_snapshot=snapshot,
    )
    session.listener_state.cache_epoch_id = 9
    before_stream = stream.published_qwen_ids
    before_ledger = ledger.source_frontier_snapshot()
    before_records = tuple(branch["segment_token_packets"])
    assert not install_response_revision_transaction(
        session=session,
        branch=branch,
        stream=stream,
        ledger=ledger,
        candidate=candidate,
    )
    assert stream.published_qwen_ids == before_stream
    assert ledger.source_frontier_snapshot() == before_ledger
    assert tuple(branch["segment_token_packets"]) == before_records

    session.listener_state.cache_epoch_id = 8
    assert install_response_revision_transaction(
        session=session,
        branch=branch,
        stream=stream,
        ledger=ledger,
        candidate=candidate,
    )
    assert stream.cache_epoch == 8
    assert branch["listener_cache_epoch_id"] == 8
    assert branch["history_revision"] == 2
    assert ledger.source_frontier_snapshot()["cache_epoch_id"] == "8"


def test_production_candidate_forks_current_listener_and_adopts_private_suffix(monkeypatch):
    session, branch, stream, ledger = _transaction_context()
    public_events = []
    resumed_packets = []
    resumed_commits = []
    resumed_flushes = []
    stream.on_tts_text_reserved = lambda consumption: ledger.record_tts_reserved(
        max(consumption.source_sequences)
    )

    def on_consumed(frontier, text):
        ledger.record_tts_consumed(frontier.frozen_source_sequence)
        public_events.append((frontier, text))

    def on_resumed_packet(generated):
        assert generated.token_index == 2
        assert generated.logical_source_sequence == 3
        cursor = 11
        row = {
            "token_id": int(generated.token_id),
            "text_piece": str(generated.token_text),
            "source_token_seq": int(generated.logical_source_sequence),
            "exclusive_kv_cursor": cursor,
            "revision_id": "r1",
            "cache_epoch_id": "4",
            "physical_kv_provenance": [{
                "revision_id": "r1",
                "cache_epoch_id": "4",
                "physical_kv_cursor": cursor,
            }],
        }
        with branch["spoken_prefix_lock"]:
            branch["cache_position"] = cursor
            branch["say_token_packets"].append(SpokenTokenPacket(
                token_id=row["token_id"],
                text_piece=row["text_piece"],
                gemma_hidden_1536=generated.hidden.reshape(1, 1536),
                exclusive_kv_cursor=cursor,
                kv_committed=bool(generated.kv_committed),
            ))
            branch["segment_token_packets"].append({
                **row,
                "token_index": int(generated.token_index),
                "hidden_provenance": str(generated.hidden_provenance),
                "kv_committed": bool(generated.kv_committed),
            })
        ledger.record_generated_tokens([row], revision_id="r1", cache_epoch_id="4")
        stream.feed_piece(
            str(generated.token_text),
            generated.hidden,
            token_index=int(generated.token_index),
            source_token_id=int(generated.token_id),
            source_token_seq=int(generated.logical_source_sequence),
            exclusive_kv_cursor=cursor,
            revision_id="r1",
            kv_committed=bool(generated.kv_committed),
        )
        resumed_packets.append(generated)

    def on_resumed_commit(generated):
        _mark_unified_say_packet_committed(
            branch,
            stream,
            int(generated.token_index),
            11,
            generation_id=7,
            segment_id=0,
        )
        resumed_commits.append(generated)

    def on_resumed_flush(suffix, generated):
        resumed_flushes.append((suffix, generated))

    stream.on_tts_text_consumed = on_consumed
    stream.take_initial_token(timeout_s=0.1)
    stream.consume_initial_token()
    assert stream.source_frontier_snapshot().frozen_source_sequence == 0
    assert [text for _frontier, text in public_events] == ["Ciao"]

    session.session_id = "session-a"
    session.stop_requested = False
    session.is_generation_current = lambda *, generation_id: (
        session.session_generation_id == generation_id
    )
    session.listener_state = SimpleNamespace(
        cache_epoch_id=4,
        native_semantic_context_revision=2,
        applied_history_revision=2,
    )
    active_handle = SimpleNamespace(
        handle_id="active-handle",
        purpose="active_response",
        revision_id="r0",
    )
    branch.update({
        "history_revision": 2,
        "listener_generation_id": 3,
        "response_decode_handle": active_handle,
        "revision_metrics": {"adopted_revisions": 0},
        "_next_revision_number": 1,
        "transition_token_ids": (5, 6),
        "spoken_prefix_token_ids": (17,),
    })

    listener_snapshot = SimpleNamespace(
        session_id="session-a",
        generation_id=3,
        state_epoch_id="epoch-a",
        next_frame_id=12,
        state_version=8,
        sequence_position=12,
        cache_position=9,
        cache_epoch_id=4,
        semantic_context_revision=2,
        applied_history_revision=2,
    )
    handle = EphemeralDecodeHandle(
        handle_id="candidate-handle",
        parent_session_id="session-a",
        parent_generation_id=3,
        parent_state_epoch_id="epoch-a",
        source_cache_position=9,
        fork_session_id="candidate-slot",
        fork_native_context_id=101,
        max_tokens=767,
        max_prompt_tokens=774,
        purpose="candidate_response",
        source_frame_id=12,
        source_state_version=8,
        source_sequence_position=12,
        source_cache_epoch_id=4,
        source_history_revision=2,
        response_id="response-a",
        response_generation_id=7,
        revision_id="r1",
    )

    class _Worker:
        def __init__(self):
            self.begins = []
            self.ends = []

        def session_state(self, _session_id):
            future = Future()
            future.set_result(listener_snapshot)
            return future

        def begin_response_decode(self, *args, **kwargs):
            self.begins.append((args, kwargs))
            future = Future()
            future.set_result(handle)
            return future

        def end_response_decode(self, ended_handle):
            self.ends.append(ended_handle.handle_id)
            future = Future()
            future.set_result(None)
            return future

        def promote_response_decode(self, candidate_handle, expected_active, expected_revision):
            assert candidate_handle.handle_id == "candidate-handle"
            assert expected_active is active_handle
            assert expected_revision == "r0"
            self.ends.append(expected_active.handle_id)
            future = Future()
            future.set_result(SimpleNamespace(
                handle_id=candidate_handle.handle_id,
                purpose="active_response",
                revision_id=candidate_handle.revision_id,
                source_cache_position=candidate_handle.source_cache_position,
            ))
            return future

    worker = _Worker()
    leaked_text = []
    request = {
        "_stop_check": lambda: False,
        "_on_text_chunk": leaked_text.append,
        "_cache_epoch_id": 4,
    }

    def fake_native_thinker(
        _engine,
        _session,
        candidate_request,
        *,
        worker,
        transition_token_ids,
        think_kwargs,
        generation_id,
        branch,
    ):
        assert generation_id == 7
        assert transition_token_ids == (5, 6)
        assert candidate_request["_continuation_prefix_token_ids"] == (17,)
        assert candidate_request["_logical_source_offset"] == 2
        assert candidate_request["_generated_budget_used"] == 1
        assert think_kwargs["max_new_tokens"] == 767
        assert think_kwargs["on_text_chunk"] is None
        assert think_kwargs["on_token_id"] is None
        generated = _GeneratedPacket(
            token_index=0,
            logical_source_sequence=2,
            token_id=18,
            token_text="é",
            hidden=torch.ones((1, 1, 1536)),
        )
        think_kwargs["on_token_packet"](generated)
        think_kwargs["on_decoded_text_flush"]("é", generated)
        branch["cache_position"] = 10
        candidate_session = SimpleNamespace(native_handle=handle)
        candidate_session.validate_native_handle_adoption = lambda next_handle: (
            None if next_handle.handle_id == handle.handle_id
            else (_ for _ in ()).throw(RuntimeError("candidate handle identity changed"))
        )
        candidate_session.adopt_native_handle = lambda next_handle: setattr(
            candidate_session, "native_handle", next_handle
        )
        controller = {
            "session": candidate_session,
            "handle_ref": {"value": handle},
            "handoff_state": {},
            "promoted": False,
            "done": False,
            "revision_id": "r1",
            "checkpoint_step": SimpleNamespace(progress=SimpleNamespace(
                generated_token_ids=(18,),
                stable_text="Ciaoé",
                generated_text="é",
            )),
        }

        def resume_candidate():
            continued = _GeneratedPacket(
                token_index=1,
                logical_source_sequence=3,
                token_id=16,
                token_text="!",
                hidden=torch.full((1, 1, 1536), 5.0),
                kv_committed=False,
                exclusive_kv_cursor=11,
                text_source_sequences=((3, 16),),
            )
            think_kwargs["on_token_packet"](continued)
            committed = replace(continued, kv_committed=True)
            think_kwargs["on_token_committed"](committed)
            think_kwargs["on_decoded_text_flush"]("!", committed)
            controller["done"] = True
            result = {
                "text": "Ciaoé!",
                "protocol_text": "Ciaoé!",
                "suffix_text": "é!",
                "gen_ids": [18, 16],
                "terminal_reason": "eos",
                "cancelled": False,
            }
            controller["final_result"] = result
            return result

        controller["run_to_completion"] = resume_candidate
        return {
            "text": "Ciaoé",
            "protocol_text": "Ciaoé",
            "suffix_text": "é",
            "gen_ids": [18],
            "terminal_reason": "length",
            "cancelled": False,
            "spoken_prefix_base_cursor": 9,
            # This is the private sampler's replayed history. The installed
            # branch must retain only the original transition because the
            # frozen prefix is already represented by spoken_prefix_token_ids.
            "transition_token_ids": (5, 6, 17),
            "_native_session_controller": controller,
        }

    monkeypatch.setattr(
        response_revision_runtime, "_run_unified_native_thinker", fake_native_thinker
    )
    coordinator = response_revision_runtime.create_live_say_revision_coordinator(
        engine=SimpleNamespace(device="cpu"),
        session=session,
        request=request,
        worker=worker,
        branch=branch,
        stream=stream,
        ledger=ledger,
        transition_token_ids=(5, 6),
        think_kwargs={
            "max_new_tokens": 768,
            "min_new_tokens": 0,
            "temperature": 0.0,
            "top_p": 1.0,
            "on_token_packet": on_resumed_packet,
            "on_token_committed": on_resumed_commit,
            "on_decoded_text_flush": on_resumed_flush,
        },
        response_protocol="controller_guided_say_v1",
        generation_id=7,
        response_id="response-a",
    )

    assert coordinator.request_revision(10)
    assert coordinator.wait_until_idle(2.0)
    adopted = coordinator.latest_adopted_after("r0")
    assert adopted is not None, coordinator.metrics_snapshot().get("last_error")
    assert branch["revision_id"] == "r1"
    assert branch["_generated_budget_used"] == 2
    assert branch["cache_epoch_id"] == 4
    assert [record["source_token_seq"] for record in branch["segment_token_packets"]] == [0, 2]
    assert [record["token_index"] for record in branch["segment_token_packets"]] == [0, 1]
    assert isinstance(branch["say_token_packets"], list)
    assert isinstance(branch["segment_token_packets"], list)
    assert stream.visible_text == "Ciaoé"
    assert stream._already_public_text == "Ciao"
    assert [text for _frontier, text in public_events] == ["Ciao"]
    assert leaked_text == []
    assert worker.begins[0][1]["purpose"].value == "candidate_response"
    assert worker.begins[0][1]["source_watermark"].cache_epoch_id == 4
    assert worker.ends == ["active-handle"]
    assert branch["response_decode_handle"].purpose == "active_response"
    assert tuple(branch["transition_token_ids"]) == (5, 6)
    rebuilt_prefix_ids = (
        *tuple(branch["transition_token_ids"]),
        *tuple(branch["spoken_prefix_token_ids"]),
    )
    assert rebuilt_prefix_ids == (5, 6, 17)
    branch["say_token_packets"][1] = replace(
        branch["say_token_packets"][1], kv_committed=False
    )
    branch["segment_token_packets"][1]["kv_committed"] = False
    _mark_unified_say_packet_committed(
        branch,
        stream,
        1,
        10,
        generation_id=7,
        segment_id=0,
    )
    assert branch["say_token_packets"][1].kv_committed is True
    assert branch["segment_token_packets"][1]["kv_committed"] is True
    adopted_controller = branch["_adopted_native_session_controller"]
    completed = adopted_controller["run_to_completion"]()
    assert completed["text"] == "Ciaoé!"
    assert [packet.token_index for packet in resumed_packets] == [2]
    assert [packet.token_index for packet in resumed_commits] == [2]
    assert [(suffix, generated.token_index) for suffix, generated in resumed_flushes] == [
        ("!", 2),
    ]
    assert stream.visible_text == "Ciaoé!"
    assert ledger.source_frontier_snapshot()["generated_source_sequence"] == 3
    assert [record["token_index"] for record in branch["segment_token_packets"]] == [
        0, 1, 2,
    ]

    class _RollbackWorker:
        def __init__(self, cursor):
            self.cursor = cursor
            self.appended = []

        @staticmethod
        def _done(value):
            future = Future()
            future.set_result(value)
            return future

        def session_state(self, _session_id):
            return self._done(SimpleNamespace(generation_id=3, cache_position=self.cursor))

        def rollback_assistant_branch(
            self, _session_id, *, expected_generation_id,
            expected_cache_position, target_cache_position,
        ):
            assert expected_generation_id == 3
            assert expected_cache_position == self.cursor
            self.cursor = target_cache_position
            return self._done(SimpleNamespace(cache_position=self.cursor))

        def reserve_tail_tokens(
            self, _session_id, *, expected_generation_id,
            expected_cache_position, token_count,
        ):
            assert expected_generation_id == 3
            assert expected_cache_position == self.cursor
            assert token_count == 7
            return self._done(None)

        def append_text_tokens(
            self, _session_id, token_ids, *, expected_generation_id,
            expected_cache_position, consume_reserved_tokens,
        ):
            assert expected_generation_id == 3
            assert expected_cache_position == self.cursor
            token_ids = tuple(token_ids)
            assert consume_reserved_tokens == len(token_ids)
            self.appended.extend(token_ids)
            self.cursor += len(token_ids)
            return self._done(SimpleNamespace(ready_event=None))

        def drain_queued_audio(
            self, _session_id, *, expected_generation_id,
            expected_cache_position, user_turn_already_open,
        ):
            assert expected_generation_id == 3
            assert expected_cache_position == self.cursor
            assert user_turn_already_open is False
            return self._done(SimpleNamespace(cache_position=self.cursor))

    branch.update({
        "listener_generation_id": 3,
        "turn_start_cache_position": 6,
        "cache_position": 11,
        "spoken_prefix_text": "Ciao",
        "spoken_prefix_exclusive_cursor": 9,
        "spoken_prefix_rebuilt": False,
    })
    rollback_worker = _RollbackWorker(cursor=11)
    asyncio.run(_finalize_unified_native_cache_branch(
        rollback_worker,
        "session-a",
        branch,
        history_committed=False,
        close_ids=(91, 92),
    ))
    assert rollback_worker.appended == [5, 6, 17, 91, 92]
    assert branch["spoken_prefix_rebuilt"] is True


def test_revision_prepare_failure_leaves_stream_ledger_and_branch_unchanged():
    session, branch, stream, ledger = _transaction_context()
    before_stream = stream.published_qwen_ids
    before_ledger = ledger.source_frontier_snapshot()
    before_records = tuple(branch["segment_token_packets"])
    candidate = _candidate(stream, ledger, rows=({"token_id": 15},))

    with pytest.raises((TypeError, ValueError)):
        install_response_revision_transaction(
            session=session,
            branch=branch,
            stream=stream,
            ledger=ledger,
            candidate=candidate,
        )
    assert stream.published_qwen_ids == before_stream
    assert ledger.source_frontier_snapshot() == before_ledger
    assert branch["revision_id"] == "r0"
    assert tuple(branch["segment_token_packets"]) == before_records


def test_owner_promotion_failure_leaves_prepared_python_views_unchanged():
    session, branch, stream, ledger = _transaction_context()
    active_handle = SimpleNamespace(
        handle_id="active-handle", purpose="active_response", revision_id="r0"
    )
    branch["response_decode_handle"] = active_handle
    candidate = replace(
        _candidate(stream, ledger),
        handle=SimpleNamespace(
            handle_id="candidate-handle",
            purpose="candidate_response",
            revision_id="r1",
            source_cache_position=9,
        ),
    )
    before_stream = stream.published_qwen_ids
    before_ledger = ledger.source_frontier_snapshot()
    before_packets = tuple(branch["say_token_packets"])
    before_records = tuple(branch["segment_token_packets"])

    def reject_owner_cas(_candidate):
        raise RuntimeError("owner CAS rejected")

    with pytest.raises(RuntimeError, match="owner CAS rejected"):
        install_response_revision_transaction(
            session=session,
            branch=branch,
            stream=stream,
            ledger=ledger,
            candidate=candidate,
            promote_candidate=reject_owner_cas,
        )
    assert stream.published_qwen_ids == before_stream
    assert ledger.source_frontier_snapshot() == before_ledger
    assert tuple(branch["say_token_packets"]) == before_packets
    assert tuple(branch["segment_token_packets"]) == before_records
    assert branch["response_decode_handle"] is active_handle
    assert branch["revision_id"] == "r0"


def test_reserved_qwen_forward_race_rejects_stale_candidate_before_any_install():
    session, branch, stream, ledger = _transaction_context()
    candidate = _candidate(stream, ledger)
    ledger.record_tts_reserved(0)
    stream.take_initial_token(timeout_s=0.1)
    before_ledger = ledger.source_frontier_snapshot()
    before_records = tuple(branch["segment_token_packets"])

    assert not install_response_revision_transaction(
        session=session,
        branch=branch,
        stream=stream,
        ledger=ledger,
        candidate=candidate,
    )
    assert ledger.source_frontier_snapshot() == before_ledger
    assert branch["revision_id"] == "r0"
    assert tuple(branch["segment_token_packets"]) == before_records


def test_active_unreserved_tail_growth_does_not_invalidate_qwen_frontier_plan():
    stream = _stream()
    stream.feed_piece(
        "Ciao!",
        torch.ones((1, 1, 1536)),
        token_index=0,
        source_token_id=17,
        source_token_seq=0,
        exclusive_kv_cursor=4,
        revision_id="r0",
    )
    expected = stream.source_frontier_snapshot()
    stream.feed_piece(
        "?",
        torch.full((1, 1, 1536), 2.0),
        token_index=1,
        source_token_id=16,
        source_token_seq=1,
        exclusive_kv_cursor=5,
        revision_id="r0",
    )
    row = {
        "token_id": 15,
        "text_piece": "!",
        "hidden": torch.full((1, 1536), 3.0),
        "source_token_seq": 2,
        "source_group": ((2, 15),),
        "exclusive_kv_cursor": 6,
        "revision_id": "r1",
    }
    with stream._condition:
        plan = stream.prepare_unconsumed_suffix_locked(
            [row],
            expected_frontier=expected,
            frozen_prefix_token_ids=(),
            revision_id="r1",
            cache_epoch=None,
        )
    assert plan.expected_frontier == expected
    assert stream.source_frontier_snapshot().frozen_source_sequence == -1


def test_slow_candidate_can_commit_while_voiced_requests_keep_arriving():
    first_started = threading.Event()
    release_first = threading.Event()
    request_three_sent = threading.Event()
    input_active = threading.Event()
    input_active.set()
    adopted_during_input = threading.Event()
    calls = []
    committed = []

    def run_candidate(request, cancelled):
        calls.append(request.frame_id)
        if request.frame_id == 1:
            first_started.set()
            assert release_first.wait(1.0)
            assert not cancelled.is_set()
        return SimpleNamespace(frame_id=request.frame_id, revision_id=f"r{request.frame_id}")

    def commit(candidate):
        committed.append(candidate.frame_id)
        if input_active.is_set():
            adopted_during_input.set()
        return True

    coordinator = ResponseRevisionCoordinator(
        session_id="s1",
        response_id="r1",
        generation_id=4,
        run_candidate=run_candidate,
        commit_candidate=commit,
    )
    assert coordinator.request_revision(1)
    assert first_started.wait(0.5)

    def continue_voiced_requests():
        for frame_id in range(2, 10):
            time.sleep(0.08)
            coordinator.request_revision(frame_id)
            if frame_id == 3:
                request_three_sent.set()
            if adopted_during_input.is_set():
                break
        input_active.clear()

    input_thread = threading.Thread(target=continue_voiced_requests, daemon=True)
    input_thread.start()
    assert request_three_sent.wait(0.5)
    # Keep the captured candidate running beyond the old 80 ms cancellation
    # window while new voiced frames continue to coalesce.
    time.sleep(0.1)
    release_first.set()
    assert coordinator.wait_until_idle(1.0)
    input_thread.join(timeout=0.5)
    assert not input_thread.is_alive()
    assert committed[0] == 1
    assert adopted_during_input.is_set()
    assert calls[-1] >= 3
    metrics = coordinator.metrics_snapshot()
    assert metrics["coalesced_requests"] >= 2
    assert metrics["candidate_cancelled"] == 0
    assert metrics["candidate_committed"] >= 1


def test_applied_history_cas_is_separate_from_native_prefix_revision():
    session, branch, stream, ledger = _transaction_context()
    session.listener_state = SimpleNamespace(
        cache_epoch_id=8,
        native_semantic_context_revision=99,
        applied_history_revision=2,
    )
    watermark = SimpleNamespace(
        generation_id=3,
        state_epoch_id="epoch-a",
        frame_id=5,
        cache_epoch_id=8,
        applied_history_revision=2,
    )
    snapshot = SimpleNamespace(
        generation_id=3,
        state_epoch_id="epoch-a",
        cache_epoch_id=8,
        semantic_context_revision=99,
        applied_history_revision=2,
    )
    candidate = replace(
        _candidate(stream, ledger, revision="r1", cache_epoch=8),
        listener_watermark=watermark,
        latest_listener_snapshot=snapshot,
    )

    assert install_response_revision_transaction(
        session=session,
        branch=branch,
        stream=stream,
        ledger=ledger,
        candidate=candidate,
    )
    assert branch["history_revision"] == 2

    session, branch, stream, ledger = _transaction_context()
    session.listener_state = SimpleNamespace(
        cache_epoch_id=8,
        native_semantic_context_revision=99,
        applied_history_revision=1,
    )
    stale_snapshot = SimpleNamespace(**{**vars(snapshot), "applied_history_revision": 1})
    stale_watermark = SimpleNamespace(**{**vars(watermark), "applied_history_revision": 1})
    stale_candidate = replace(
        candidate,
        base_revision_id="r0",
        revision_id="r1",
        listener_watermark=stale_watermark,
        latest_listener_snapshot=stale_snapshot,
        stream_frontier=stream.source_frontier_snapshot(),
        ledger_snapshot=ledger.source_frontier_snapshot(),
    )
    assert not install_response_revision_transaction(
        session=session,
        branch=branch,
        stream=stream,
        ledger=ledger,
        candidate=stale_candidate,
    )
    assert branch["revision_id"] == "r0"


def test_revision_jobs_are_session_local_and_yield_cancels_pending_publication():
    started = threading.Event()
    release = threading.Event()
    published = []

    def run_candidate(request, cancelled):
        started.set()
        release.wait(1.0)
        return SimpleNamespace(frame_id=request.frame_id)

    coordinator = ResponseRevisionCoordinator(
        session_id="session-a",
        response_id="response-a",
        generation_id=9,
        run_candidate=run_candidate,
        commit_candidate=lambda candidate: published.append(candidate.frame_id) or True,
    )
    isolated = ResponseRevisionCoordinator(
        session_id="session-b",
        response_id="response-b",
        generation_id=9,
        run_candidate=lambda _request, _cancel: None,
        commit_candidate=lambda _candidate: True,
    )
    assert isolated.metrics_snapshot()["requests"] == 0
    assert coordinator.request_revision(2)
    assert started.wait(0.5)
    coordinator.cancel_all("yield")
    release.set()
    assert coordinator.wait_until_idle(1.0)
    assert published == []
    assert not coordinator.request_revision(3)


def test_eos_waits_for_eligible_revision_and_finalize_can_be_claimed_once():
    started = threading.Event()
    release = threading.Event()

    def run_candidate(request, cancelled):
        started.set()
        release.wait(1.0)
        return SimpleNamespace(frame_id=request.frame_id)

    coordinator = ResponseRevisionCoordinator(
        session_id="s1",
        response_id="r1",
        generation_id=4,
        run_candidate=run_candidate,
        commit_candidate=lambda _candidate: True,
    )
    assert coordinator.request_revision(10)
    assert started.wait(0.5)
    coordinator.request_eos()
    assert not coordinator.claim_finalize_if_idle()
    release.set()
    assert coordinator.wait_until_idle(1.0)
    assert coordinator.claim_finalize_if_idle()
    assert not coordinator.claim_finalize_if_idle()
    assert not coordinator.request_revision(11)


def test_stale_candidate_retries_with_latest_snapshot_once():
    calls = []
    commits = []
    refreshed = []

    def run_candidate(request, _cancel):
        calls.append(request.attempt)
        return SimpleNamespace(frame_id=request.frame_id, attempt=request.attempt)

    def refresh(request):
        refreshed.append(request.frame_id)
        return ResponseRevisionRequest(
            session_id=request.session_id,
            response_id=request.response_id,
            generation_id=request.generation_id,
            frame_id=request.frame_id + 1,
            requested_at_ns=time.monotonic_ns(),
        )

    coordinator = ResponseRevisionCoordinator(
        session_id="s1",
        response_id="r1",
        generation_id=4,
        run_candidate=run_candidate,
        commit_candidate=lambda candidate: commits.append(candidate.attempt) or candidate.attempt > 0,
        refresh_request=refresh,
    )
    coordinator.request_revision(10)
    assert coordinator.wait_until_idle(1.0)
    assert calls == [0, 1]
    assert refreshed == [10]
    assert commits == [0, 1]

