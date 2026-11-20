from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
import unicodedata
from concurrent.futures import Future

import pytest
import torch

from duplex_kernel.live_say import LiveQwenSayStream, LiveSayCancelled
from duplex_kernel.response_event_ledger import ResponseEventLedger
from duplex_kernel.response_intent_local import (
    align_generated_response_hidden_to_qwen_tokens,
    align_hidden_states_by_char_overlap,
)
from duplex_kernel.audio_response_protocol import (
    AUDIO_RESPONSE_PROTOCOL_V2,
    CONTROLLER_GUIDED_SAY_V1,
    AudioResponseV2Collector,
    audio_response_v2_say_token_char_spans,
    finish_audio_response_v2_with_say_hidden,
)
from duplex_kernel.server import (
    _configured_audio_response_protocol,
    _finalize_unified_native_cache_branch,
    _mark_unified_say_packet_committed,
    _publish_acknowledged_speech_prefix,
)
from duplex_kernel.session import DuplexSession, SpokenTokenPacket
from duplex_kernel.response_intent_local import qwen_assistant_text_offsets


class _TokenizingQwen:
    """Small framed tokenizer with a split accented subword."""

    def __init__(self) -> None:
        self.pieces = {
            10: "L\u2019", 11: "\u00e9", 12: "lan", 13: "o",
            15: "!", 16: "?", 14: "", 30: " ",
            17: "Ciao", 18: "é", 19: "e", 31: "😀",
        }
        self.backend_tokenizer = SimpleNamespace(
            pre_tokenizer=SimpleNamespace(pre_tokenize_str=self._pre_tokenize_str),
        )

    def _pre_tokenize_str(self, text: str):
        spans = []
        cursor = 0
        while cursor < len(text):
            match = next(
                (
                    piece for _token_id, piece in self.pieces.items()
                    if piece and text.startswith(piece, cursor)
                ),
                None,
            )
            assert match is not None, (text, cursor)
            spans.append((match, (cursor, cursor + len(match))))
            cursor += len(match)
        return spans

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ):
        prefix = "<|im_start|>assistant\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        framed = add_special_tokens
        if framed:
            assert text.startswith(prefix) and text.endswith(suffix)
            raw = text[len(prefix):-len(suffix)]
            offset_base = len(prefix)
        else:
            raw = text
            offset_base = 0
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(raw):
            match = next(
                (
                    (token_id, piece) for token_id, piece in self.pieces.items()
                    if piece and raw.startswith(piece, cursor)
                ),
                None,
            )
            assert match is not None, (raw, cursor)
            token_id, piece = match
            ids.append(token_id)
            offsets.append((offset_base + cursor, offset_base + cursor + len(piece)))
            cursor += len(piece)
        assert cursor == len(raw), (raw, cursor)
        if framed:
            ids = [0, 1, 2] + ids + [20, 21, 22, 23, 24]
            offsets = [(0, 0)] * 3 + offsets + [(0, 0)] * 5
        if return_offsets_mapping:
            return {"input_ids": ids, "offset_mapping": offsets}
        return {"input_ids": ids}

    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[int(token_id)] for token_id in ids)


class _Processor:
    def __init__(self, tokenizer: _TokenizingQwen) -> None:
        self.tokenizer = tokenizer

    def __call__(self, *, text: str, return_tensors: str):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([self.tokenizer(text)["input_ids"]])}


def _new_stream(*, queue_capacity: int = 16, **kwargs) -> LiveQwenSayStream:
    tokenizer = _TokenizingQwen()
    return LiveQwenSayStream(
        qwen_tokenizer=tokenizer,
        qwen_processor=_Processor(tokenizer),
        text_eos_token_id=99,
        max_tokens=16,
        queue_capacity=queue_capacity,
        **kwargs,
    )


def _consume_initial_after_successful_prefill(stream, **kwargs):
    token = stream.take_initial_token(**kwargs)
    stream.consume_initial_token()
    return token


def _consume_trailing_after_successful_q1(stream, **kwargs):
    token = stream.take_trailing_token(**kwargs)
    if token is not None:
        stream.consume_trailing_token()
    return token


@pytest.mark.parametrize("stop_phase", ["before_initial", "finish_full", "after_eos"])
def test_live_say_consumer_stop_releases_response_producer(monkeypatch, stop_phase):
    import duplex_kernel.server_response_runtime as runtime

    stream = _new_stream(queue_capacity=1)
    session = DuplexSession(session_id="live-say-cancel")
    generation = session.begin_response("response-cancel")
    producer_waiting = threading.Event()
    consumer_exited = threading.Event()
    response_done = threading.Event()
    errors = []
    results = []
    original_publish = stream._publish_token

    def observe_publication(*args, **kwargs):
        with stream._condition:
            if not kwargs.get("initial", False) and stream._tail_queue_size_locked() >= 1:
                producer_waiting.set()
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(stream, "_publish_token", observe_publication)
    monkeypatch.setattr(runtime, "LiveQwenSayStream", lambda **_kwargs: stream)
    monkeypatch.setattr(runtime, "_configured_audio_response_protocol", lambda _engine: CONTROLLER_GUIDED_SAY_V1)
    monkeypatch.setattr(runtime, "_prepare_direct_audio_processor_inputs", lambda *_args, **_kwargs: ([1], None, 0.0))
    monkeypatch.setattr(runtime, "_resolve_request_voice_conditioning", lambda *_args, **_kwargs: None)

    def think(_ids, *, on_generated_token, **_kwargs):
        if stop_phase == "before_initial":
            session.stop_requested = True
            return {"text": "", "gen_ids": [], "cancelled": True, "terminal_reason": "cancelled"}
        for index, piece in enumerate(("L’", "é", "lan")):
            on_generated_token(index, index + 10, piece, torch.ones((1, 1, 2)))
        return {"text": "L’élan", "gen_ids": [10, 11, 12], "terminal_reason": "eos"}

    def talk(_text, _hidden, *, live_qwen_text_stream, stop_check, **_kwargs):
        try:
            if stop_phase == "finish_full":
                assert producer_waiting.wait(1.0), "response did not reach bounded EOS publication"
            else:
                live_qwen_text_stream.consume_initial_token()
                while live_qwen_text_stream.take_trailing_token(timeout_s=1.0) is not None:
                    live_qwen_text_stream.consume_trailing_token()
                assert live_qwen_text_stream.eos_consumed
            session.stop_requested = True
            assert stop_check()
            raise LiveSayCancelled("Talker observed YIELD")
        finally:
            consumer_exited.set()

    engine = SimpleNamespace(
        device="cpu", dtype=torch.float32, tokenizer=_TokenizingQwen(),
        talker=SimpleNamespace(proc=_Processor(_TokenizingQwen()), cfg=SimpleNamespace(tts_eos_token_id=99)),
        think=think, talk=talk,
    )

    def run_response():
        try:
            results.append(runtime.run_response(
                engine, session, [{"role": "user", "content": "parla"}],
                {"_generation_id": generation.session_generation_id, "_direct_audio_pcm": [0.0],
                 "modalities": ["audio"], "_stop_check": lambda: session.stop_requested},
                action="SPEAK",
            ))
        except BaseException as exc:
            errors.append(exc)
        finally:
            response_done.set()

    producer = threading.Thread(target=run_response, daemon=True)
    producer.start()
    try:
        if stop_phase != "before_initial":
            assert consumer_exited.wait(2.0), "Talker did not observe cancellation"
        assert response_done.wait(1.0), "consumer exited but response producer remained blocked"
        assert not errors
        assert results
        if stop_phase == "after_eos":
            assert stream.eos_consumed
        else:
            with pytest.raises(LiveSayCancelled):
                stream.take_initial_token(timeout_s=0.1)
    finally:
        stream.cancel()
        producer.join(timeout=2.0)
    assert not producer.is_alive()


def test_revision_replaces_only_unreserved_qwen_tail_and_keeps_source_identity():
    stream = _new_stream()
    old_rows = (
        ("Ciao", 101, 40),
        ("!", 102, 41),
        ("?", 103, 42),
        ("Ciao", 104, 43),
    )
    for index, (piece, cursor, marker) in enumerate(old_rows):
        stream.feed_piece(
            piece,
            torch.full((1, 1, 1536), float(marker)),
            token_index=index,
            source_token_id=100 + index,
            source_token_seq=index,
            source_group=((index, 100 + index),),
            exclusive_kv_cursor=cursor,
            revision_id="r0",
            seal_initial_segment=True,
        )
    assert stream.published_qwen_ids == [17, 15, 16]
    assert _consume_initial_after_successful_prefill(stream).token_id == 17

    frontier = stream.source_frontier_snapshot()
    assert frontier.frozen_source_sequence == 0
    with stream._condition:
        plan = stream.prepare_unconsumed_suffix_locked(
            (
                {
                    "token_id": 201,
                    "text_piece": "?",
                    "hidden": torch.full((1, 1, 1536), 51.0),
                    "source_token_seq": 4,
                    "source_group": ((4, 201),),
                    "exclusive_kv_cursor": 8,
                    "revision_id": "r1",
                },
                {
                    "token_id": 202,
                    "text_piece": "!",
                    "hidden": torch.full((1, 1, 1536), 52.0),
                    "source_token_seq": 5,
                    "source_group": ((5, 202),),
                    "exclusive_kv_cursor": 9,
                    "revision_id": "r1",
                },
            ),
            expected_frontier=frontier,
            frozen_prefix_token_ids=(100,),
            revision_id="r1",
            cache_epoch=None,
        )
        stream.install_unconsumed_suffix_locked(plan)

    assert stream.published_qwen_ids == [17, 16]
    assert stream.source_frontier_snapshot().frozen_source_sequence == 0
    assert stream.take_trailing_token(timeout_s=0.1, q1_index=1) == 16
    reserved = stream._reserved_consumption
    assert reserved is not None
    assert reserved.source_sequences == (4,)
    assert torch.equal(reserved.hidden, torch.full((1, 1, 1536), 51.0))
    stream.finish(final_text=stream.visible_text)
    assert stream.published_qwen_ids == [17, 16, 15]


def test_revision_stream_prepare_failure_leaves_live_queue_unchanged():
    stream = _new_stream()
    stream.feed_piece(
        "Ciao",
        torch.ones((1, 1, 1536)),
        token_index=0,
        source_token_id=100,
        source_token_seq=0,
        source_group=((0, 100),),
        exclusive_kv_cursor=10,
        seal_initial_segment=True,
    )
    before_ids = stream.published_qwen_ids
    before = stream.source_frontier_snapshot()
    with stream._condition:
        with pytest.raises(RuntimeError, match="differs from live source packet"):
            stream.prepare_unconsumed_suffix_locked(
                ({
                    "token_id": 201,
                    "text_piece": "!",
                    "hidden": torch.ones((1, 1, 1536)),
                    "source_token_seq": 1,
                    "source_group": ((1, 201),),
                    "exclusive_kv_cursor": 2,
                },),
                expected_frontier=before,
                frozen_prefix_token_ids=(999,),
                revision_id="r1",
                cache_epoch=None,
            )
    assert stream.published_qwen_ids == before_ids
    assert stream.source_frontier_snapshot() == before


def test_final_text_second_plan_failure_leaves_stream_ledger_and_branch_unchanged():
    session = DuplexSession("final-text-atomic")
    stream = _new_stream(generation_id=0, segment_id=0)
    hidden = torch.ones((1, 1, 1536))
    stream.feed_piece(
        "Ciao",
        hidden,
        token_index=0,
        source_token_id=101,
        source_token_seq=0,
        source_group=((0, 101),),
        exclusive_kv_cursor=10,
        revision_id="r0",
        seal_initial_segment=True,
    )
    ledger = ResponseEventLedger(
        response_id="response-1",
        generation_id=0,
        revision_id="r0",
        cache_epoch_id="epoch-1",
    )
    ledger.record_generated_tokens([{
        "source_token_seq": 0,
        "token_id": 101,
        "text_piece": "Ciao",
        "exclusive_kv_cursor": 10,
        "revision_id": "r0",
        "cache_epoch_id": "epoch-1",
    }])
    old_packets = [SimpleNamespace(text_piece="Ciao")]
    old_records = [{"source_token_seq": 0, "text_piece": "Ciao"}]
    branch = {"say_token_packets": old_packets, "segment_token_packets": old_records}
    branch_lock = threading.RLock()

    stream_before = (
        stream.source_frontier_snapshot(),
        stream.visible_text,
        stream._source_revision,
        stream._source_packets[0]["text_piece"],
        stream._source_packets[0]["final_text_flushed"],
    )
    ledger_before = ledger.project()
    next_packets = [SimpleNamespace(text_piece="Ciao!")]
    next_records = [{"source_token_seq": 0, "text_piece": "Ciao!"}]

    # Match serving lock order: the stream plan is complete before the
    # ledger's second prepare fails its revision check.
    with session._generation_lock:
        with branch_lock:
            with stream._condition:
                stream_plan = stream.prepare_final_text_flush_locked(
                    "!",
                    hidden,
                    token_index=0,
                    source_token_id=101,
                    source_token_seq=0,
                    source_group=((0, 101),),
                    exclusive_kv_cursor=10,
                    revision_id="r0",
                )
                with ledger._lock:
                    with pytest.raises(RuntimeError, match="revision or cache epoch"):
                        ledger.prepare_unconsumed_source_text_locked(
                            0,
                            "Ciao!",
                            expected_revision_id="stale-revision",
                            expected_cache_epoch_id="epoch-1",
                        )
                    assert ledger.project() == ledger_before
                    assert stream.source_frontier_snapshot() == stream_before[0]
                    assert stream.visible_text == stream_before[1]
                    assert stream._source_revision == stream_before[2]
                    assert stream._source_packets[0]["text_piece"] == stream_before[3]
                    assert stream._source_packets[0]["final_text_flushed"] is stream_before[4]
                    assert branch["say_token_packets"] is old_packets
                    assert branch["segment_token_packets"] is old_records

                    ledger_plan = ledger.prepare_unconsumed_source_text_locked(
                        0,
                        "Ciao!",
                        expected_revision_id="r0",
                        expected_cache_epoch_id="epoch-1",
                    )
                    ledger.install_unconsumed_source_text_locked(ledger_plan)
                    stream.install_final_text_flush_locked(stream_plan)
                    branch["say_token_packets"] = next_packets
                    branch["segment_token_packets"] = next_records
                    stream._condition.notify_all()

    assert ledger.project().source_tokens[0]["text_piece"] == "Ciao!"
    assert stream.visible_text == "Ciao!"
    assert branch["say_token_packets"] is next_packets
    assert branch["segment_token_packets"] is next_records


class _ActualQwenBackendTokenizer:
    """CPU differential tokenizer backed by tokenizers' Split/ByteLevel."""

    PREFIX = "<|im_start|>assistant\n"
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
    PATTERN = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
        r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
        r"\s+(?!\S)|\s+"
    )

    def __init__(self, *, added_token: str | None = None) -> None:
        from tokenizers import AddedToken, Regex, pre_tokenizers
        from tokenizers import normalizers

        self.backend_tokenizer = SimpleNamespace(
            normalizer=normalizers.NFC(),
            pre_tokenizer=pre_tokenizers.Sequence([
                pre_tokenizers.Split(Regex(self.PATTERN), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]),
        )
        self.added_tokens_decoder = {}
        self._special_ids: dict[str, int] = {}
        self._segments: dict[str, int] = {}
        self._decoded: dict[int, str] = {}
        if added_token is not None:
            self._special_ids[added_token] = 77
            self._decoded[77] = added_token
            self.added_tokens_decoder[77] = AddedToken(added_token, normalized=False)

    def _body(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(text):
            next_special = min(
                (text.find(content, cursor) for content in self._special_ids if text.find(content, cursor) >= 0),
                default=-1,
            )
            if next_special == cursor:
                content = next(content for content in self._special_ids if text.startswith(content, cursor))
                token_id = self._special_ids[content]
                ids.append(token_id)
                offsets.append((cursor, cursor + len(content)))
                cursor += len(content)
                continue
            end = len(text) if next_special < 0 else next_special
            segment_text = text[cursor:end]
            for _segment, (start, stop) in self.backend_tokenizer.pre_tokenizer.pre_tokenize_str(segment_text):
                surface = segment_text[start:stop]
                token_id = self._segments.setdefault(surface, 1000 + len(self._segments))
                self._decoded[token_id] = surface
                ids.append(token_id)
                offsets.append((cursor + start, cursor + stop))
            cursor = end
        return ids, offsets

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ):
        if add_special_tokens:
            assert text.startswith(self.PREFIX) and text.endswith(self.SUFFIX)
            start = len(self.PREFIX)
            body = text[start:-len(self.SUFFIX)]
            raw_ids, raw_offsets = self._body(body)
            ids = [0, 1, 2] + raw_ids + [20, 21, 22, 23, 24]
            offsets = [(0, 0)] * 3 + [
                (start + first, start + last) for first, last in raw_offsets
            ] + [(0, 0)] * 5
        else:
            ids, offsets = self._body(text)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode(self, ids, **_kwargs):
        return "".join(self._decoded[int(token_id)] for token_id in ids)


class _ActualQwenProcessor:
    def __init__(self, tokenizer: _ActualQwenBackendTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, *, text: str, return_tensors: str):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([self.tokenizer(text)["input_ids"]])}


def _regex_stream(*, added_token: str | None = None, **kwargs) -> LiveQwenSayStream:
    tokenizer = _ActualQwenBackendTokenizer(added_token=added_token)
    return LiveQwenSayStream(
        qwen_tokenizer=tokenizer,
        qwen_processor=_ActualQwenProcessor(tokenizer),
        text_eos_token_id=99,
        max_tokens=128,
        **kwargs,
    )


def test_controller_guided_say_is_default_and_exact_zero_forces_legacy(monkeypatch):
    engine = SimpleNamespace(audio_response_protocol=AUDIO_RESPONSE_PROTOCOL_V2)
    monkeypatch.delenv("DUPLEX_AUDIO_RESPONSE_PROTOCOL", raising=False)
    monkeypatch.delenv("DUPLEX_CONTROLLER_GUIDED_SAY_V1", raising=False)
    assert _configured_audio_response_protocol(engine) == CONTROLLER_GUIDED_SAY_V1
    monkeypatch.setenv("DUPLEX_CONTROLLER_GUIDED_SAY_V1", "0")
    assert _configured_audio_response_protocol(engine) == AUDIO_RESPONSE_PROTOCOL_V2
    monkeypatch.setenv("DUPLEX_CONTROLLER_GUIDED_SAY_V1", "00")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _configured_audio_response_protocol(engine)


def test_protocol_enum_wins_over_compatibility_toggle_and_rejects_unknown(monkeypatch):
    engine = SimpleNamespace(audio_response_protocol=AUDIO_RESPONSE_PROTOCOL_V2)
    monkeypatch.setenv("DUPLEX_AUDIO_RESPONSE_PROTOCOL", AUDIO_RESPONSE_PROTOCOL_V2)
    monkeypatch.setenv("DUPLEX_CONTROLLER_GUIDED_SAY_V1", "1")
    assert _configured_audio_response_protocol(engine) == AUDIO_RESPONSE_PROTOCOL_V2
    monkeypatch.setenv("DUPLEX_AUDIO_RESPONSE_PROTOCOL", "unknown")
    with pytest.raises(ValueError, match="DUPLEX_AUDIO_RESPONSE_PROTOCOL"):
        _configured_audio_response_protocol(engine)


def test_live_qwen_ids_release_causally_once_and_flush_in_full_prefix_order():
    stream = _new_stream()
    pieces = [("L\u2019", 1.0), ("\u00e9", 3.0), ("lan", 5.0), ("o", 7.0)]
    for piece, value in pieces:
        stream.feed_piece(piece, torch.full((1, 1, 2), value))
    stream.finish(final_text="L\u2019\u00e9lano")

    initial = _consume_initial_after_successful_prefill(stream, timeout_s=0.1)
    assert initial.token_id == 10
    assert stream.codec_eos_allowed is False
    tail = [
        _consume_trailing_after_successful_q1(stream, timeout_s=0.1)
        for _ in range(4)
    ]
    assert tail == [11, 12, 13, 99]
    assert _consume_trailing_after_successful_q1(stream, timeout_s=0.1) is None
    assert stream.codec_eos_allowed is True
    assert stream.published_qwen_ids == [10, 11, 12, 13]


def test_live_qwen_reservation_commits_only_after_prefill_or_successful_q1():
    events = []
    stream = _new_stream(
        cache_epoch="cache-epoch-7",
        on_tts_text_consumed=lambda frontier, text: events.append((frontier, text))
    )
    hidden_first = torch.full((1, 1, 2), 1.0)
    hidden_second = torch.full((1, 1, 2), 2.0)
    stream.feed_piece(
        "Ciao", hidden_first, token_index=0, source_token_id=71,
        exclusive_kv_cursor=100,
    )
    stream.feed_piece(
        "😀", hidden_second, token_index=1, source_token_id=72,
        exclusive_kv_cursor=101,
    )
    stream.finish(final_text="Ciao😀")

    initial = stream.take_initial_token(timeout_s=0.1)
    before_prefill = stream.source_frontier_snapshot()
    assert before_prefill.cache_epoch == "cache-epoch-7"
    assert before_prefill.qwen_consumed_count == 0
    assert before_prefill.reserved_qwen_token_id == initial.token_id
    assert stream.reserved_consumption.q1_index is None
    torch.testing.assert_close(stream.reserved_consumption.hidden, stream.published_hidden[0])

    stream.consume_initial_token()
    after_prefill = stream.source_frontier_snapshot()
    assert after_prefill.qwen_consumed_count == 1
    assert after_prefill.reserved_qwen_token_id is None
    assert events[-1][1] == "Ciao"

    trailing_id = stream.take_trailing_token(timeout_s=0.1, q1_index=0)
    reserved = stream.reserved_consumption
    assert trailing_id == 31
    assert reserved.qwen_token_id == trailing_id
    assert reserved.q1_index == 0
    assert reserved.source_sequences == (1,)
    assert reserved.source_token_ids == (72,)
    assert reserved.source_kv_cursors == (101,)
    torch.testing.assert_close(reserved.hidden, stream.published_hidden[1])
    frozen = stream.source_frontier_snapshot()
    assert frozen.qwen_consumed_count == 1
    assert frozen.reserved_qwen_token_id == trailing_id
    assert frozen.frozen_source_sequence == 1
    assert frozen.cache_epoch == "cache-epoch-7"

    stream.consume_trailing_token()
    assert stream.source_frontier_snapshot().qwen_consumed_count == 2
    assert events[-1][1] == "Ciao😀"


def test_live_qwen_logical_source_consumption_keeps_physical_cursor_absent():
    consumed_events = []
    stream = _new_stream(
        on_tts_text_consumed=lambda frontier, text: consumed_events.append((frontier, text))
    )
    stream.feed_piece(
        "Ciao",
        torch.ones((1, 1, 2)),
        token_index=0,
        source_token_id=701,
        source_token_seq=0,
    )
    stream.finish(final_text="Ciao")

    initial = stream.take_initial_token(timeout_s=0.1)
    reservation = stream.reserved_consumption
    assert initial.token_id == 17
    assert reservation is not None
    assert reservation.source_sequences == (0,)
    assert reservation.source_token_ids == (701,)
    assert reservation.source_kv_cursors == ()
    assert stream.source_frontier_snapshot().qwen_consumed_count == 0

    stream.consume_initial_token()
    frontier = stream.source_frontier_snapshot()
    assert frontier.qwen_consumed_count == 1
    assert frontier.source_sequences == (0,)
    assert frontier.source_token_ids == (701,)
    assert frontier.source_kv_cursors == ()
    assert consumed_events[-1][1] == "Ciao"


def test_live_qwen_eos_record_holds_last_consumed_source_hidden():
    stream = _new_stream()
    last_hidden = torch.full((1, 1, 2), 9.0)
    stream.feed_piece(
        "Ciao", last_hidden, token_index=0, source_token_id=88,
        exclusive_kv_cursor=340,
    )
    stream.finish(final_text="Ciao")
    stream.take_initial_token(timeout_s=0.1)
    stream.consume_initial_token()

    assert stream.take_trailing_token(timeout_s=0.1, q1_index=0) == 99
    eos = stream.reserved_consumption
    assert eos.eos is True
    assert eos.tail_hold is True
    assert eos.source_sequences == (0,)
    assert eos.source_token_ids == (88,)
    assert eos.source_kv_cursors == (340,)
    torch.testing.assert_close(eos.hidden, stream.published_hidden[0])
    stream.consume_trailing_token()
    held = stream.hidden_for_frame(1)
    assert held.tail_hold is True
    assert held.gemma_token_id == 88
    assert held.pad is True
    assert held.qwen_token_id is None
    torch.testing.assert_close(held.hidden, stream.published_hidden[0])


def test_live_v1_pcm_ack_waits_for_unified_kv_packet_commit_and_catches_up_multiframe():
    stream = _new_stream()
    committed_prefixes = []
    stream.on_packet_emitted = lambda cursor, text: committed_prefixes.append((cursor, text))
    stream.feed_piece(
        "Ciao",
        torch.ones((1, 1, 2)),
        token_index=0,
        exclusive_kv_cursor=100,
        kv_committed=False,
    )
    stream.finish(final_text="Ciao")
    assert stream.published_qwen_ids == [17]
    assert stream.mark_packet_committed(
        100, generation_id=9, segment_id=0
    ) is None

    # A three-frame PCM packet can be released before the native append. Its
    # frame ACKs stay pending until the source Gemma token commits.
    assert stream.ack_emitted_audio_frame(
        0, generation_id=0, segment_id=0
    ) == ()
    assert stream.ack_emitted_audio_frame(
        1, generation_id=0, segment_id=0
    ) == ()
    assert stream.ack_emitted_audio_frame(
        2, generation_id=0, segment_id=0
    ) == ()
    assert stream.mark_packet_committed(
        100, generation_id=0, segment_id=0
    ) == (17,)
    assert committed_prefixes == [(100, "Ciao")]
    assert stream._last_processed_emitted_frame == 2
    assert stream.ack_emitted_audio_frame(
        3, generation_id=0, segment_id=0
    ) == ()


def test_live_v1_commit_barrier_does_not_skip_an_uncommitted_empty_source_packet():
    stream = _new_stream()
    committed_prefixes = []
    stream.on_packet_emitted = lambda cursor, text: committed_prefixes.append((cursor, text))
    hidden = torch.ones((1, 1, 2))
    stream.feed_piece(
        "",
        hidden,
        token_index=0,
        exclusive_kv_cursor=200,
        kv_committed=False,
    )
    stream.feed_piece(
        "Ciao",
        hidden,
        token_index=1,
        exclusive_kv_cursor=201,
        kv_committed=True,
    )
    stream.finish(final_text="Ciao")
    assert stream.ack_emitted_audio_frame(
        0, generation_id=0, segment_id=0
    ) == ()
    assert stream.mark_packet_committed(
        201, generation_id=0, segment_id=0
    ) is None
    assert stream.mark_packet_committed(
        200, generation_id=0, segment_id=0
    ) == (17,)
    assert committed_prefixes == [(201, "Ciao")]


def test_live_v1_cancel_discards_pcm_ack_waiting_for_packet_commit():
    stream = _new_stream()
    stream.feed_piece(
        "Ciao",
        torch.ones((1, 1, 2)),
        token_index=0,
        exclusive_kv_cursor=300,
        kv_committed=False,
    )
    stream.finish(final_text="Ciao")
    assert stream.ack_emitted_audio_frame(
        0, generation_id=0, segment_id=0
    ) == ()
    stream.cancel()
    assert stream.mark_packet_committed(
        300, generation_id=0, segment_id=0
    ) is None
    assert not stream._pending_emitted_frames
    with pytest.raises(RuntimeError, match="cancelled"):
        stream.ack_emitted_audio_frame(0, generation_id=0, segment_id=0)


def test_live_v1_stale_packet_commit_does_not_mutate_branch_record():
    hidden = torch.zeros((1, 1536))
    branch = {
        "session_generation_id": 4,
        "spoken_prefix_base_cursor": 60,
        "spoken_prefix_lock": threading.RLock(),
        "say_token_packets": [SpokenTokenPacket(
            token_id=21,
            text_piece="Ciao",
            gemma_hidden_1536=hidden,
            exclusive_kv_cursor=61,
            kv_committed=False,
        )],
        "segment_token_packets": [{
            "token_id": 21,
            "text_piece": "Ciao",
            "token_index": 0,
            "exclusive_kv_cursor": 61,
            "kv_committed": False,
        }],
    }
    stream = _new_stream(generation_id=5, segment_id=0)

    assert _mark_unified_say_packet_committed(
        branch, stream, 0, 61, generation_id=4, segment_id=0
    ) is None
    assert branch["say_token_packets"][0].kv_committed is False
    assert branch["segment_token_packets"][0]["kv_committed"] is False


def test_live_v1_frame20_ack_advances_partial_watermark_and_yield_rebuilds_only_heard_prefix():
    class Worker:
        def __init__(self):
            self.cursor = 64
            self.rollback_targets = []
            self.append_calls = []
            self.drain_open_modes = []

        @staticmethod
        def done(value):
            future = Future()
            future.set_result(value)
            return future

        def session_state(self, _session_id):
            return self.done(SimpleNamespace(generation_id=12, cache_position=self.cursor))

        def rollback_assistant_branch(
            self, _session_id, *, expected_generation_id, expected_cache_position,
            target_cache_position,
        ):
            assert expected_generation_id == 12
            assert expected_cache_position == self.cursor
            self.rollback_targets.append(target_cache_position)
            self.cursor = target_cache_position
            return self.done(SimpleNamespace(cache_position=self.cursor))

        def reserve_tail_tokens(
            self, _session_id, *, expected_generation_id, expected_cache_position, token_count,
        ):
            assert expected_generation_id == 12
            assert expected_cache_position == self.cursor
            assert token_count == 7
            return self.done(SimpleNamespace(cache_position=self.cursor))

        def append_text_tokens(
            self, _session_id, token_ids, *, expected_generation_id, expected_cache_position,
            consume_reserved_tokens,
        ):
            assert expected_generation_id == 12
            assert expected_cache_position == self.cursor
            values = tuple(token_ids)
            self.append_calls.append(values)
            self.cursor += len(values)
            return self.done(SimpleNamespace(cache_position=self.cursor))

        def drain_queued_audio(
            self, _session_id, *, expected_generation_id, expected_cache_position,
            user_turn_already_open,
        ):
            assert expected_generation_id == 12
            assert expected_cache_position == self.cursor
            self.drain_open_modes.append(user_turn_already_open)
            return self.done(SimpleNamespace(cache_position=self.cursor))

    spoken_prefix_lock = threading.RLock()
    branch = {
        "listener_generation_id": 12,
        "session_generation_id": 0,
        "cache_position": 64,
        "turn_start_cache_position": 60,
        "spoken_prefix_base_cursor": 62,
        "spoken_prefix_exclusive_cursor": 62,
        "spoken_prefix_token_ids": (),
        "spoken_prefix_text_parts": (),
        "spoken_prefix_text": "",
        "spoken_prefix_kv_position": None,
        "spoken_prefix_lock": spoken_prefix_lock,
        "transition_token_ids": (11, 12),
        "queue_open": True,
        "finalized": False,
        "say_token_packets": [],
        "segment_token_packets": [],
    }
    first_text = "Ciao" * 6  # Six Qwen spans end at PCM frame 20.
    second_text = "L’"
    packet_hidden = torch.zeros((1, 1536))
    branch["say_token_packets"] = [
        SpokenTokenPacket(
            token_id=21,
            text_piece=first_text,
            gemma_hidden_1536=packet_hidden,
            exclusive_kv_cursor=63,
            hidden_provenance="transition",
            kv_committed=False,
        ),
        SpokenTokenPacket(
            token_id=22,
            text_piece=second_text,
            gemma_hidden_1536=packet_hidden,
            exclusive_kv_cursor=64,
            hidden_provenance="token",
            kv_committed=True,
        ),
    ]
    branch["segment_token_packets"] = [
        {
            "token_id": 21, "text_piece": first_text,
            "exclusive_kv_cursor": 63, "token_index": 0,
            "hidden_provenance": "transition", "kv_committed": False,
        },
        {
            "token_id": 22, "text_piece": second_text,
            "exclusive_kv_cursor": 64, "token_index": 1,
            "hidden_provenance": "token", "kv_committed": True,
        },
    ]

    stream = _new_stream(
        generation_id=0,
        segment_id=0,
        on_packet_emitted=lambda cursor, text: _publish_acknowledged_speech_prefix(
            branch, 0, cursor, text
        ),
    )
    source_hidden = torch.zeros((1, 1, 2))
    stream.feed_piece(
        first_text,
        source_hidden,
        token_index=0,
        exclusive_kv_cursor=63,
        kv_committed=False,
    )
    stream.feed_piece(
        second_text,
        source_hidden,
        token_index=1,
        exclusive_kv_cursor=64,
        kv_committed=True,
    )
    stream.finish(final_text=first_text + second_text)

    for frame_id in range(21):
        stream.ack_emitted_audio_frame(frame_id, generation_id=0, segment_id=0)
    assert branch["spoken_prefix_exclusive_cursor"] == 62
    assert branch["spoken_prefix_token_ids"] == ()

    branch["cache_position"] = 62
    with pytest.raises(RuntimeError, match="before its KV cursor was committed"):
        _mark_unified_say_packet_committed(
            branch, stream, 0, 63, generation_id=0, segment_id=0
        )
    assert branch["say_token_packets"][0].kv_committed is False
    assert branch["segment_token_packets"][0]["kv_committed"] is False
    branch["cache_position"] = 64

    _mark_unified_say_packet_committed(
        branch, stream, 0, 63, generation_id=0, segment_id=0
    )
    assert branch["spoken_prefix_exclusive_cursor"] == 63
    assert branch["spoken_prefix_token_ids"] == (21,)
    assert branch["spoken_prefix_text"] == first_text
    assert stream._last_processed_emitted_frame == 20

    worker = Worker()
    asyncio.run(_finalize_unified_native_cache_branch(
        worker,
        "session",
        branch,
        history_committed=False,
        close_ids=(90, 91),
    ))
    assert worker.rollback_targets == [60]
    assert worker.append_calls == [(11, 12), (21, 90), (91,)]
    assert worker.drain_open_modes == [False]
    assert branch["spoken_prefix_rebuilt"] is True


def _logical_ack_branch(source_rows):
    ledger = ResponseEventLedger(response_id="logical-ack", generation_id=7)
    ledger.record_generated_tokens(source_rows)
    packets = []
    records = []
    for row in source_rows:
        token_id = int(row["token_id"])
        text_piece = str(row["text_piece"])
        cursor = int(row["exclusive_kv_cursor"])
        revision_id = str(row["revision_id"])
        cache_epoch_id = row["cache_epoch_id"]
        packet = SpokenTokenPacket(
            token_id=token_id,
            text_piece=text_piece,
            gemma_hidden_1536=torch.zeros((1, 1536)),
            exclusive_kv_cursor=cursor,
            kv_committed=True,
        )
        packets.append(packet)
        records.append({
            **row,
            "kv_committed": True,
            "physical_kv_provenance": [{
                "revision_id": revision_id,
                "cache_epoch_id": cache_epoch_id,
                "physical_kv_cursor": cursor,
            }],
        })
    return {
        "session_generation_id": 7,
        "spoken_prefix_lock": threading.RLock(),
        "spoken_prefix_token_ids": (),
        "spoken_prefix_text_parts": (),
        "spoken_prefix_text": "",
        "spoken_prefix_source_seq": -1,
        "spoken_prefix_exclusive_cursor": 2,
        "say_token_packets": packets,
        "segment_token_packets": records,
        "response_event_ledger": ledger,
    }


def test_v1_pcm_prefix_carries_canonicalized_away_text_into_next_acknowledged_span():
    source_rows = [
        {
            "token_id": 30, "text_piece": " ", "source_token_seq": 0,
            "exclusive_kv_cursor": 5, "revision_id": "r0", "cache_epoch_id": "e0",
        },
        {
            "token_id": 17, "text_piece": "Ciao", "source_token_seq": 1,
            "exclusive_kv_cursor": 6, "revision_id": "r0", "cache_epoch_id": "e0",
        },
        {
            "token_id": 30, "text_piece": " ", "source_token_seq": 2,
            "exclusive_kv_cursor": 7, "revision_id": "r0", "cache_epoch_id": "e0",
        },
    ]
    branch = _logical_ack_branch(source_rows)

    def publish(cursor, text, *, source_token_seq, revision_id, cache_epoch_id):
        _publish_acknowledged_speech_prefix(
            branch,
            7,
            cursor,
            text,
            source_token_seq=source_token_seq,
            revision_id=revision_id,
            cache_epoch_id=cache_epoch_id,
        )

    stream = _new_stream(
        generation_id=7,
        cache_epoch="e0",
        on_packet_emitted=publish,
    )
    hidden = torch.zeros((1, 1, 2))
    stream.feed_piece(
        " ", hidden, token_index=0, source_token_id=30, source_token_seq=0,
        source_group=((0, 30),), exclusive_kv_cursor=5, revision_id="r0",
    )
    stream.feed_piece(
        "Ciao", hidden, token_index=1, source_token_id=17, source_token_seq=1,
        source_group=((1, 17),), exclusive_kv_cursor=6, revision_id="r0",
    )
    stream.feed_piece(
        " ", hidden, token_index=2, source_token_id=30, source_token_seq=2,
        source_group=((2, 30),), exclusive_kv_cursor=7, revision_id="r0",
    )
    stream.finish(final_text="Ciao")

    initial = _consume_initial_after_successful_prefill(stream, timeout_s=0.1)
    assert initial.token_id == 17
    assert branch["spoken_prefix_source_seq"] == -1
    stream.ack_emitted_audio_frame(0, generation_id=7, segment_id=0)

    assert branch["spoken_prefix_source_seq"] == 1
    assert branch["spoken_prefix_token_ids"] == (30, 17)
    assert branch["spoken_prefix_text"] == " Ciao"
    assert branch["spoken_prefix_text_parts"] == (" ", "Ciao")


def test_v1_played_prefix_uses_logical_sequence_across_rebased_cursors_and_gaps():
    branch = _logical_ack_branch([
        {
            "token_id": 31, "text_piece": "a", "source_token_seq": 10,
            "exclusive_kv_cursor": 7, "revision_id": "r0", "cache_epoch_id": "e0",
        },
        {
            "token_id": 32, "text_piece": "b", "source_token_seq": 12,
            "exclusive_kv_cursor": 3, "revision_id": "r1", "cache_epoch_id": "e1",
        },
    ])

    assert _publish_acknowledged_speech_prefix(
        branch, 7, 7, "a", source_token_seq=10, revision_id="r0", cache_epoch_id="e0",
    )
    assert _publish_acknowledged_speech_prefix(
        branch, 7, 3, "ab", source_token_seq=12, revision_id="r1", cache_epoch_id="e1",
    )

    assert branch["spoken_prefix_source_seq"] == 12
    assert branch["spoken_prefix_exclusive_cursor"] == 3
    assert branch["spoken_prefix_token_ids"] == (31, 32)
    assert branch["spoken_prefix_text"] == "ab"
    assert branch["response_event_ledger"].source_frontier_snapshot()[
        "tts_consumed_source_seq"
    ] == -1


def test_v1_played_prefix_preserves_repeated_same_text_as_distinct_source_tokens():
    branch = _logical_ack_branch([
        {
            "token_id": 41, "text_piece": "ha", "source_token_seq": 0,
            "exclusive_kv_cursor": 5, "revision_id": "r0", "cache_epoch_id": "e0",
        },
        {
            "token_id": 41, "text_piece": "ha", "source_token_seq": 1,
            "exclusive_kv_cursor": 6, "revision_id": "r0", "cache_epoch_id": "e0",
        },
    ])

    assert _publish_acknowledged_speech_prefix(
        branch, 7, 6, "haha", source_token_seq=1, revision_id="r0", cache_epoch_id="e0",
    )
    assert branch["spoken_prefix_source_seq"] == 1
    assert branch["spoken_prefix_token_ids"] == (41, 41)
    assert branch["spoken_prefix_text_parts"] == ("ha", "ha")


def test_v1_played_prefix_rejects_unmatched_physical_provenance_without_mutation():
    branch = _logical_ack_branch([
        {
            "token_id": 51, "text_piece": "ok", "source_token_seq": 0,
            "exclusive_kv_cursor": 8, "revision_id": "r0", "cache_epoch_id": "e0",
        },
    ])
    before = (
        branch["spoken_prefix_source_seq"],
        branch["spoken_prefix_token_ids"],
        branch["spoken_prefix_text_parts"],
        branch["spoken_prefix_text"],
    )

    with pytest.raises(RuntimeError, match="exact committed revision packet"):
        _publish_acknowledged_speech_prefix(
            branch, 7, 8, "ok", source_token_seq=0, revision_id="r1", cache_epoch_id="e1",
        )

    assert (
        branch["spoken_prefix_source_seq"],
        branch["spoken_prefix_token_ids"],
        branch["spoken_prefix_text_parts"],
        branch["spoken_prefix_text"],
    ) == before


def test_v1_played_prefix_duplicate_is_idempotent_but_changed_duplicate_is_invalid():
    branch = _logical_ack_branch([
        {
            "token_id": 61, "text_piece": "ok", "source_token_seq": 4,
            "exclusive_kv_cursor": 11, "revision_id": "r0", "cache_epoch_id": "e0",
        },
    ])
    kwargs = {
        "source_token_seq": 4,
        "revision_id": "r0",
        "cache_epoch_id": "e0",
    }

    assert _publish_acknowledged_speech_prefix(branch, 7, 11, "ok", **kwargs)
    assert not _publish_acknowledged_speech_prefix(branch, 7, 11, "ok", **kwargs)
    before = (
        branch["spoken_prefix_source_seq"],
        branch["spoken_prefix_token_ids"],
        branch["spoken_prefix_text_parts"],
        branch["spoken_prefix_text"],
    )
    with pytest.raises(RuntimeError, match="same V1 logical source frontier"):
        _publish_acknowledged_speech_prefix(branch, 7, 11, "not ok", **kwargs)
    assert (
        branch["spoken_prefix_source_seq"],
        branch["spoken_prefix_token_ids"],
        branch["spoken_prefix_text_parts"],
        branch["spoken_prefix_text"],
    ) == before


def test_live_alignment_matches_full_prefix_for_split_accented_pieces():
    stream = _new_stream()
    gemma_pieces = [("L\u2019", 1.0), ("\u00e9", 3.0), ("lan", 5.0), ("o", 7.0)]
    for piece, value in gemma_pieces:
        stream.feed_piece(piece, torch.full((1, 1, 2), value))
    stream.finish(final_text="L\u2019\u00e9lano")

    ids = [1, 2, 3, 4]
    hidden = torch.cat([torch.full((1, 1, 2), value) for _, value in gemma_pieces], dim=1)
    expected, _ = align_generated_response_hidden_to_qwen_tokens(
        hidden,
        ids,
        gemma_tokenizer=SimpleNamespace(
            decode=lambda values, **_kwargs: {1: "L\u2019", 2: "\u00e9", 3: "lan", 4: "o"}[int(values[0])]
        ),
        qwen_tokenizer=stream.qwen_tokenizer,
        qwen_processor=stream.qwen_processor,
        text="L\u2019\u00e9lano",
    )
    actual = torch.cat(stream.published_hidden, dim=1)
    torch.testing.assert_close(actual, expected)


def test_live_stream_canonicalizes_repeated_internal_whitespace_with_source_mapping():
    stream = _new_stream()
    pieces = [
        ("L\u2019", 1.0),
        (" ", 3.0),
        (" ", 5.0),
        ("\u00e9", 7.0),
        ("lan", 9.0),
        ("o", 11.0),
    ]
    for piece, value in pieces:
        stream.feed_piece(piece, torch.full((1, 1, 2), value))
    stream.finish(final_text="L\u2019 \u00e9lano")

    initial = _consume_initial_after_successful_prefill(stream, timeout_s=0.1)
    tail = []
    while (token := _consume_trailing_after_successful_q1(stream, timeout_s=0.1)) is not None:
        tail.append(token)
    assert initial.token_id == 10
    assert tail == [30, 11, 12, 13, 99]
    # The single canonical space is sourced from both Gemma whitespace tokens.
    torch.testing.assert_close(
        stream.published_hidden[1], torch.full((1, 1, 2), 4.0)
    )
    assert stream.codec_eos_allowed is True


def test_live_text_tail_waits_while_open_and_never_substitutes_pad_for_empty_tail():
    stream = _new_stream()
    for piece, value in [("L\u2019", 1.0), ("\u00e9", 3.0)]:
        stream.feed_piece(piece, torch.full((1, 1, 2), value))
    initial = _consume_initial_after_successful_prefill(stream, timeout_s=0.1)
    assert initial.token_id == 10

    result: list[int | None] = []
    finished = threading.Event()

    def consume_one():
        result.append(_consume_trailing_after_successful_q1(stream))
        finished.set()

    worker = threading.Thread(target=consume_one)
    worker.start()
    assert not finished.wait(0.05)
    stream.feed_piece("lan", torch.full((1, 1, 2), 5.0))
    assert finished.wait(0.5)
    worker.join(timeout=0.5)
    assert result == [11]
    assert stream.codec_eos_allowed is False


def test_live_consumer_waits_for_visible_text_and_a_proven_stable_qwen_prefix():
    stream = _new_stream()
    stream.feed_piece(" \t", torch.full((1, 1, 2), 1.0))
    assert stream.visible_text == ""
    with pytest.raises(TimeoutError):
        stream.take_initial_token(timeout_s=0.02)

    stream.feed_piece("L\u2019", torch.full((1, 1, 2), 2.0))
    stream.feed_piece("\u00e9", torch.full((1, 1, 2), 3.0))
    # The current pretokenizer spans close L’ before the open accented token,
    # so only the causally stable first ID is available at this point.
    initial = _consume_initial_after_successful_prefill(stream, timeout_s=0.1)
    assert initial.token_id == 10

    result: list[int | None] = []
    finished = threading.Event()

    def consume_next():
        result.append(_consume_trailing_after_successful_q1(stream, timeout_s=0.5))
        finished.set()

    worker = threading.Thread(target=consume_next)
    worker.start()
    assert not finished.wait(0.05)
    stream.feed_piece("lan", torch.full((1, 1, 2), 4.0))
    assert finished.wait(0.5)
    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert result == [11]
    stream.cancel()


@pytest.mark.parametrize("terminal", ["cancel", "error"])
def test_live_text_tail_cancel_and_error_wake_blocked_producer_and_consumer(terminal: str):
    stream = _new_stream(queue_capacity=1)
    for piece, value in [("L\u2019", 1.0), ("\u00e9", 3.0)]:
        stream.feed_piece(piece, torch.full((1, 1, 2), value))
    assert stream.take_initial_token(timeout_s=0.1).token_id == 10

    producer_errors: list[BaseException] = []
    consumer_errors: list[BaseException] = []
    producer_started = threading.Event()
    producer_filled_queue = threading.Event()
    producer_second_feed_returned = threading.Event()
    consumer_started = threading.Event()

    def blocked_producer():
        producer_started.set()
        try:
            # "lan" closes the accented Qwen segment and fills the one-slot
            # trailing queue. The following "o" stabilizes another token and
            # must block until a Qwen consumer drains that slot.
            stream.feed_piece("lan", torch.full((1, 1, 2), 5.0))
            producer_filled_queue.set()
            stream.feed_piece("o", torch.full((1, 1, 2), 7.0))
            producer_second_feed_returned.set()
        except BaseException as exc:  # captured to assert thread cleanup
            producer_errors.append(exc)

    def blocked_consumer():
        consumer_started.set()
        try:
            empty_stream.hidden_for_frame(0)
        except BaseException as exc:  # captured to assert thread cleanup
            consumer_errors.append(exc)

    empty_stream = _new_stream(queue_capacity=1)

    producer = threading.Thread(target=blocked_producer)
    producer.start()
    assert producer_started.wait(0.2)
    assert producer_filled_queue.wait(0.5), "producer did not fill the bounded tail queue"
    consumer = threading.Thread(target=blocked_consumer)
    consumer.start()
    assert consumer_started.wait(0.2)
    time.sleep(0.05)
    assert producer.is_alive(), "bounded text queue should backpressure the producer"
    assert not producer_second_feed_returned.is_set()
    assert stream._tail_queue_size_locked() == 1
    tail = stream.hidden_for_frame(99)
    assert tail.tail_hold is True, "live audio must reuse the last known token during starvation"
    assert consumer.is_alive(), "an empty live stream should wait for its first aligned token"
    if terminal == "cancel":
        stream.cancel()
        empty_stream.cancel()
    else:
        stream.fail(RuntimeError("producer failed"))
        empty_stream.fail(RuntimeError("producer failed"))
    producer.join(timeout=0.5)
    consumer.join(timeout=0.5)
    assert not producer.is_alive() and not consumer.is_alive()
    assert len(producer_errors) == 1 and len(consumer_errors) == 1


@pytest.mark.parametrize(
    "fence_during_wait", ["none", "cancel", "generation", "revision"]
)
def test_revisioned_source_commit_does_not_hold_transaction_locks_for_queue_wait(
    fence_during_wait: str,
):
    """A full one-token tail must not block the consumer's ledger callback."""
    stream = _new_stream(queue_capacity=1, generation_id=7, segment_id=0)
    ledger = ResponseEventLedger(
        response_id="queue-wait",
        generation_id=7,
        revision_id="r0",
        cache_epoch_id="epoch-0",
    )
    generation_lock = threading.RLock()
    branch_lock = threading.RLock()
    session = SimpleNamespace(
        session_generation_id=7,
        response_id="queue-wait",
        active_response=True,
        stop_requested=False,
    )
    branch = {
        "response_id": "queue-wait",
        "session_generation_id": 7,
        "revision_id": "r0",
        "_streaming_revision_id": "r0",
        "spoken_prefix_lock": branch_lock,
        "say_token_packets": [],
        "segment_token_packets": [],
    }
    cancelled = threading.Event()
    stream.on_tts_text_reserved = lambda consumption: ledger.record_tts_reserved(
        max(consumption.source_sequences)
    )

    def guard_publication(token_index, source_row):
        def publish_if_current(publish):
            with generation_lock:
                with branch_lock:
                    with stream._condition:
                        with ledger._lock:
                            if (
                                session.session_generation_id != 7
                                or not session.active_response
                                or cancelled.is_set()
                                or session.stop_requested
                                or branch["revision_id"] != "r0"
                                or branch["_streaming_revision_id"] != "r0"
                                or branch["segment_token_packets"][token_index]
                                .get("source_token_seq")
                                != source_row["source_token_seq"]
                                or source_row["source_token_seq"] not in ledger._source_tokens
                            ):
                                return False
                            return publish()

        return publish_if_current

    def commit_source(token_index, token_id, source_sequence, text_piece):
        source_row = {
            "source_token_seq": source_sequence,
            "token_id": token_id,
            "text_piece": text_piece,
            "exclusive_kv_cursor": token_index + 1,
            "revision_id": "r0",
            "cache_epoch_id": "epoch-0",
            "physical_kv_provenance": [{
                "revision_id": "r0",
                "cache_epoch_id": "epoch-0",
                "physical_kv_cursor": token_index + 1,
            }],
        }
        packet = SimpleNamespace(token_id=token_id, token_index=token_index)
        record = {
            **source_row,
            "token_index": token_index,
            "hidden_provenance": "test",
            "kv_committed": True,
        }
        with generation_lock:
            with branch_lock:
                with stream._condition:
                    with ledger._lock:
                        assert token_index == len(branch["say_token_packets"])
                        stream.feed_piece(
                            text_piece,
                            torch.full((1, 1, 2), float(token_index + 1)),
                            token_index=token_index,
                            source_token_id=token_id,
                            source_token_seq=source_sequence,
                            source_group=((source_sequence, token_id),),
                            exclusive_kv_cursor=token_index + 1,
                            revision_id="r0",
                            seal_initial_segment=True,
                            defer_publication=True,
                        )
                        ledger.record_generated_tokens(
                            [source_row], revision_id="r0", cache_epoch_id="epoch-0"
                        )
                        branch["say_token_packets"].append(packet)
                        branch["segment_token_packets"].append(record)

        stream.publish_deferred(
            stop_check=lambda: (
                cancelled.is_set()
                or session.stop_requested
                or session.session_generation_id != 7
                or branch["_streaming_revision_id"] != "r0"
            ),
            publication_guard=guard_publication(token_index, source_row),
        )

    # The first source piece is still mutable. The next punctuation seals the
    # initial prefix; its callback must already see the source row in ledger.
    commit_source(0, 100, 0, "Ciao")
    commit_source(1, 101, 1, "!")
    initial = stream.take_initial_token(timeout_s=0.5)
    assert initial.token_id == 17
    assert initial.consumption is not None
    assert initial.consumption.source_sequences == (0,)
    stream.consume_initial_token()

    # Fill the single trailing slot, then let a second producer stage its
    # source row under the same lock order used by the revisioned runtime.
    commit_source(2, 102, 2, "?")
    queue_wait_entered = threading.Event()
    original_wait = stream._wait_for_publication_capacity

    def observe_capacity_wait(pending, *, stop_check):
        queue_wait_entered.set()
        return original_wait(pending, stop_check=stop_check)

    stream._wait_for_publication_capacity = observe_capacity_wait
    producer_errors: list[BaseException] = []
    producer_done = threading.Event()

    def producer():
        try:
            # This piece stabilizes the prior '?' Qwen token, filling the
            # bounded tail only after the ledger transaction has committed.
            commit_source(3, 103, 3, "Ciao")
        except BaseException as exc:
            producer_errors.append(exc)
        finally:
            producer_done.set()

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()
    consumed: list[tuple[int, tuple[int, ...]]] = []
    consumer_errors: list[BaseException] = []
    stop_consumer = threading.Event()

    def consumer():
        try:
            for q1_index in (1, 2):
                token_id = stream.take_trailing_token(
                    timeout_s=1.0,
                    q1_index=q1_index,
                    stop_check=stop_consumer.is_set,
                )
                if token_id is None:
                    break
                reservation = stream.reserved_consumption
                assert reservation is not None
                consumed.append((token_id, reservation.source_sequences))
                stream.consume_trailing_token()
        except BaseException as exc:
            if not (fence_during_wait != "none" and isinstance(exc, LiveSayCancelled)):
                consumer_errors.append(exc)

    consumer_thread = threading.Thread(target=consumer, daemon=True)
    try:
        assert queue_wait_entered.wait(0.5), "producer did not reach the full queue"
        if fence_during_wait == "revision":
            with generation_lock:
                with branch_lock:
                    with stream._condition:
                        with ledger._lock:
                            branch["revision_id"] = "r1"
                            branch["_streaming_revision_id"] = "r1"
        elif fence_during_wait == "generation":
            with generation_lock:
                session.session_generation_id = 8
        elif fence_during_wait == "cancel":
            cancelled.set()
            session.stop_requested = True

        # Let the producer observe the fence before the consumer frees space.
        consumer_thread.start()

        assert producer_done.wait(1.0), "producer held a transaction lock while waiting for capacity"
        if fence_during_wait != "none":
            stop_consumer.set()
            with stream._condition:
                stream._condition.notify_all()
        consumer_thread.join(timeout=1.0)
        assert not consumer_thread.is_alive()
        assert not consumer_errors
        assert consumed[0] == (15, (1,))
        if fence_during_wait != "none":
            assert consumed == [(15, (1,))]
            assert stream.published_qwen_ids == [17, 15]
            assert isinstance(producer_errors[0], LiveSayCancelled)
            assert 16 not in stream.published_qwen_ids
        else:
            assert not producer_errors
            assert consumed == [(15, (1,)), (16, (2,))]
            assert stream.published_qwen_ids == [17, 15, 16]
    finally:
        stream.cancel()
        producer_thread.join(timeout=1.0)
        consumer_thread.join(timeout=1.0)


@pytest.mark.parametrize(
    "pieces",
    [
        ["L’", "e"],
        ["l'", "s"],
        ["Ciao ", "mondo"],
        ["Ciao\r", "\nmondo"],
        ["abc-", "def"],
        ["12", "3456", "78"],
        ["caf", "é"],
        ["cafe", "\u0301"],
    ],
)
def test_incremental_qwen_matches_actual_backend_pretokenizer_at_eos(pieces: list[str]):
    stream = _regex_stream()
    full_text = unicodedata.normalize("NFC", "".join(pieces))
    for index, piece in enumerate(pieces):
        stream.feed_piece(piece, torch.full((1, 1, 2), float(index + 1)))
    stream.finish(final_text=full_text)

    expected_ids, expected_offsets = qwen_assistant_text_offsets(
        full_text,
        tokenizer=stream.qwen_tokenizer,
        processor=stream.qwen_processor,
    )
    initial = _consume_initial_after_successful_prefill(stream, timeout_s=0.2)
    tail: list[int] = []
    while (token := _consume_trailing_after_successful_q1(stream, timeout_s=0.2)) is not None:
        tail.append(token)
    assert [initial.token_id, *tail[:-1]] == expected_ids
    assert stream.published_qwen_ids == expected_ids
    assert stream.published_offsets == expected_offsets
    assert tail[-1] == 99
    assert stream.codec_eos_allowed
    assert len(stream.published_hidden) == len(expected_ids)


def test_nfc_composition_maps_one_qwen_span_to_both_gemma_source_rows():
    stream = _regex_stream()
    stream.feed_piece("e", torch.full((1, 1, 2), 2.0))
    stream.feed_piece("\u0301", torch.full((1, 1, 2), 6.0))
    stream.finish(final_text="e\u0301")
    assert stream.visible_text == "é"
    assert len(stream.published_hidden) == 1
    torch.testing.assert_close(stream.published_hidden[0], torch.full((1, 1, 2), 4.0))


def test_v2_collector_say_hidden_aligns_to_exact_qwen_text_without_private_fields():
    collector = AudioResponseV2Collector(expected_action="SPEAK")
    fragments = [
        '{"heard":"private","action":"SPEAK","say":"Ciao \\u',
        "00",
        "E9 e \\uD83D",
        "\\uDE00\"}",
    ]
    for index, fragment in enumerate(fragments):
        collector.feed_token(
            token_index=index,
            token_id=100 + index,
            text=fragment,
            hidden=torch.full((1, 1, 1), float(index + 1)),
        )
    parsed, say_hidden = finish_audio_response_v2_with_say_hidden(
        collector,
        expected_text="".join(fragments),
    )
    text = parsed["say"]
    say_spans = audio_response_v2_say_token_char_spans(collector.say_tokens, text)
    tokenizer = _TokenizingQwen()
    processor = _Processor(tokenizer)
    qwen_ids, qwen_offsets = qwen_assistant_text_offsets(
        text,
        tokenizer=tokenizer,
        processor=processor,
    )
    aligned = align_hidden_states_by_char_overlap(say_hidden, say_spans, qwen_offsets)

    assert text == "Ciao é e 😀"
    assert len(qwen_ids) == aligned.shape[1]
    assert aligned[0, :, 0].tolist() == [1.0, 1.0, 2.5, 3.0, 3.0, 3.0, 4.0]


def test_split_added_token_prefix_stays_buffered_until_a_following_segment():
    stream = _regex_stream(added_token="<END>")
    for index, piece in enumerate(["ciao ", "<EN", "D>", " mondo", "!"]):
        stream.feed_piece(piece, torch.full((1, 1, 2), float(index + 1)))
        if index < 4:
            assert 77 not in stream.published_qwen_ids
    assert 77 in stream.published_qwen_ids
    stream.finish(final_text="ciao <END> mondo!")
    expected_ids, _ = qwen_assistant_text_offsets(
        "ciao <END> mondo!",
        tokenizer=stream.qwen_tokenizer,
        processor=stream.qwen_processor,
    )
    assert stream.published_qwen_ids == expected_ids
    assert 77 in expected_ids


def test_completed_added_token_at_open_boundary_flushes_only_after_validated_finish():
    stream = _regex_stream(added_token="<END>")
    stream.feed_piece("ciao ", torch.ones((1, 1, 2)))
    stream.feed_piece("<END>", torch.full((1, 1, 2), 2.0))
    assert 77 not in stream.published_qwen_ids

    with pytest.raises(ValueError, match="testo finale canonico"):
        stream.finish(final_text="ciao <END> non-canonico")
    assert 77 not in stream.published_qwen_ids

    validated = _regex_stream(added_token="<END>")
    validated.feed_piece("ciao ", torch.ones((1, 1, 2)))
    validated.feed_piece("<END>", torch.full((1, 1, 2), 2.0))
    assert 77 not in validated.published_qwen_ids
    validated.finish(final_text="ciao <END>")
    assert 77 in validated.published_qwen_ids


def test_open_word_waits_for_following_text_then_releases_without_eos():
    stream = _regex_stream()
    stream.feed_piece("Ciao", torch.ones((1, 1, 2)))
    assert stream.published_qwen_ids == []

    result: list[int] = []
    finished = threading.Event()

    def consume_initial():
        result.append(stream.take_initial_token(timeout_s=1.0).token_id)
        finished.set()

    worker = threading.Thread(target=consume_initial)
    worker.start()
    assert not finished.wait(0.05)
    stream.feed_piece(" mondo", torch.full((1, 1, 2), 2.0))
    assert finished.wait(0.5), "following valid text should close the first word promptly"
    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert result == [stream.published_qwen_ids[0]]
    assert stream.codec_eos_allowed is False
    stream.cancel()


def test_feed_exception_fails_stream_and_wakes_blocked_consumers():
    stream = _regex_stream()
    errors: list[BaseException] = []
    started = threading.Event()

    def waiting_consumer():
        started.set()
        try:
            stream.take_initial_token()
        except BaseException as exc:
            errors.append(exc)

    consumer = threading.Thread(target=waiting_consumer)
    consumer.start()
    assert started.wait(0.2)
    with pytest.raises(TypeError):
        stream.feed_piece("broken", torch.zeros((1, 1)))
    consumer.join(timeout=0.5)
    assert not consumer.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], TypeError)


@pytest.mark.parametrize("mutation", ["id", "offset", "hidden"])
def test_feed_fails_immediately_if_published_alignment_changes(mutation: str):
    stream = _regex_stream()
    stream.feed_piece("Ciao", torch.ones((1, 1, 2)))
    assert stream.published_qwen_ids == []
    stream.feed_piece(" mondo", torch.full((1, 1, 2), 2.0))
    assert len(stream.published_qwen_ids) >= 1
    published_before_mutation = stream.published_qwen_ids
    assert stream.take_initial_token(timeout_s=0.1).token_id == published_before_mutation[0]
    current_alignment = stream._current_alignment

    def changed_alignment(text, pieces, **kwargs):
        ids, offsets, aligned, source_cursors, char_sources = current_alignment(
            text,
            pieces,
            **kwargs,
        )
        ids = list(ids)
        offsets = list(offsets)
        aligned = list(aligned)
        if mutation == "id":
            ids[0] = 999
        elif mutation == "offset":
            offsets[0] = (1, offsets[0][1])
        else:
            aligned[0] = aligned[0] + 1
        return ids, offsets, aligned, source_cursors, char_sources

    stream._current_alignment = changed_alignment
    with pytest.raises(ValueError, match="published Qwen"):
        stream.feed_piece("o", torch.full((1, 1, 2), 7.0))
    assert stream.published_qwen_ids == published_before_mutation
    with pytest.raises(ValueError, match="published Qwen"):
        stream.take_trailing_token(timeout_s=0.1)


def test_retokenization_work_cap_is_cumulative_and_feed_fails_closed():
    stream = _regex_stream(max_retokenization_work=70)
    stream.feed_piece("a", torch.ones((1, 1, 2)))
    with pytest.raises(RuntimeError, match="retokenization work capacity"):
        stream.feed_piece("b", torch.ones((1, 1, 2)))
    with pytest.raises(RuntimeError, match="retokenization work capacity"):
        stream.take_initial_token(timeout_s=0.1)


@pytest.mark.parametrize("wait_kind", ["initial", "trailing"])
def test_blocked_initial_and_trailing_waits_honor_cancellation(wait_kind: str):
    stream = _regex_stream()
    cancelled = threading.Event()
    started = threading.Event()
    errors: list[BaseException] = []

    def wait_for_text():
        started.set()
        try:
            if wait_kind == "initial":
                stream.take_initial_token(stop_check=cancelled.is_set)
            else:
                stream.take_trailing_token(stop_check=cancelled.is_set)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=wait_for_text)
    worker.start()
    assert started.wait(0.2)
    cancelled.set()
    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
