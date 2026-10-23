from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
import unicodedata
from concurrent.futures import Future

import pytest
import torch

from duplex_kernel.live_say import LiveQwenSayStream
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
from duplex_kernel.session import SpokenTokenPacket
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

    def __call__(self, text: str, *, return_offsets_mapping: bool = False):
        prefix = "<|im_start|>assistant\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        assert text.startswith(prefix) and text.endswith(suffix)
        raw = text[len(prefix):-len(suffix)]
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
            offsets.append((len(prefix) + cursor, len(prefix) + cursor + len(piece)))
            cursor += len(piece)
        assert cursor == len(raw), (raw, cursor)
        framed_ids = [0, 1, 2] + ids + [20, 21, 22, 23, 24]
        framed_offsets = [(0, 0)] * 3 + offsets + [(0, 0)] * 5
        if return_offsets_mapping:
            return {"input_ids": framed_ids, "offset_mapping": framed_offsets}
        return {"input_ids": framed_ids}

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

    def __call__(self, text: str, *, return_offsets_mapping: bool = False):
        assert text.startswith(self.PREFIX) and text.endswith(self.SUFFIX)
        start = len(self.PREFIX)
        body = text[start:-len(self.SUFFIX)]
        raw_ids, raw_offsets = self._body(body)
        ids = [0, 1, 2] + raw_ids + [20, 21, 22, 23, 24]
        offsets = [(0, 0)] * 3 + [
            (start + first, start + last) for first, last in raw_offsets
        ] + [(0, 0)] * 5
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

    initial = stream.take_initial_token(timeout_s=0.1)
    assert initial.token_id == 10
    assert stream.codec_eos_allowed is False
    tail = [stream.take_trailing_token(timeout_s=0.1) for _ in range(4)]
    assert tail == [11, 12, 13, 99]
    assert stream.take_trailing_token(timeout_s=0.1) is None
    assert stream.codec_eos_allowed is True
    assert stream.published_qwen_ids == [10, 11, 12, 13]


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
    first_text = "Ciao" * 7  # Seven Qwen spans end at PCM frame 20.
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

    initial = stream.take_initial_token(timeout_s=0.1)
    tail = []
    while (token := stream.take_trailing_token(timeout_s=0.1)) is not None:
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
    assert stream.take_initial_token(timeout_s=0.1).token_id == 10

    result: list[int | None] = []
    finished = threading.Event()

    def consume_one():
        result.append(stream.take_trailing_token())
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
    assert stream.take_initial_token(timeout_s=0.1).token_id == 10

    result: list[int | None] = []
    finished = threading.Event()

    def consume_next():
        result.append(stream.take_trailing_token(timeout_s=0.5))
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
    initial = stream.take_initial_token(timeout_s=0.2)
    tail: list[int] = []
    while (token := stream.take_trailing_token(timeout_s=0.2)) is not None:
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

    def changed_alignment(text, pieces):
        ids, offsets, aligned, source_cursors, char_sources = current_alignment(text, pieces)
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
