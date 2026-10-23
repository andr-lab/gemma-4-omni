"""Causally aligned, bounded text handoff for controller-guided SAY."""
from __future__ import annotations

import threading
import time
import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import torch

from duplex_kernel.response_intent_local import qwen_assistant_text_offsets
from duplex_kernel.session import (
    SpanAlignedSpeechCursor,
    SpeechQwenToken,
    SpokenTokenPacket,
    TurnLocalSpeechFIFO,
)


class LiveSayCancelled(RuntimeError):
    """A response-scoped live SAY producer or consumer was cancelled."""


@dataclass(frozen=True)
class LiveQwenToken:
    token_id: int
    hidden: torch.Tensor


@dataclass(frozen=True)
class LiveQwenHidden:
    hidden: torch.Tensor
    tail_hold: bool = False
    gemma_token_id: int | None = None
    packet_exclusive_kv_cursor: int | None = None
    qwen_token_id: int | None = None
    packet_token_index: int | None = None
    packet_token_count: int | None = None
    producer_event: Any | None = None


class LiveQwenSayStream:
    """Publish stable Qwen IDs and same-prefix Gemma hidden states.

    The Qwen token cursor consumes ID 0 as prefill, then consumes each later
    raw-text token once. The independent FiLM cursor starts at aligned token 0.
    Both cursors are response-owned and bounded by ``max_tokens``.
    """

    def __init__(
        self,
        *,
        qwen_tokenizer: Any,
        qwen_processor: Any,
        text_eos_token_id: int,
        max_tokens: int = 512,
        queue_capacity: int = 16,
        max_source_chars: int = 16_384,
        max_source_pieces: int = 2_048,
        max_retokenization_work: int = 1_000_000,
        generation_id: int = 0,
        segment_id: int = 0,
        on_packet_emitted: Any | None = None,
    ) -> None:
        if isinstance(max_tokens, bool) or int(max_tokens) < 1:
            raise ValueError("max_tokens deve essere positivo")
        if isinstance(queue_capacity, bool) or int(queue_capacity) < 1:
            raise ValueError("queue_capacity deve essere positivo")
        if isinstance(text_eos_token_id, bool) or int(text_eos_token_id) < 0:
            raise ValueError("text_eos_token_id deve essere non negativo")
        if isinstance(max_source_chars, bool) or int(max_source_chars) < 1:
            raise ValueError("max_source_chars deve essere positivo")
        if isinstance(max_source_pieces, bool) or int(max_source_pieces) < 1:
            raise ValueError("max_source_pieces deve essere positivo")
        if isinstance(max_retokenization_work, bool) or int(max_retokenization_work) < 1:
            raise ValueError("max_retokenization_work deve essere positivo")
        self.qwen_tokenizer = qwen_tokenizer
        self.qwen_processor = qwen_processor
        self.text_eos_token_id = int(text_eos_token_id)
        self.max_tokens = int(max_tokens)
        self.queue_capacity = int(queue_capacity)
        self.max_source_chars = int(max_source_chars)
        self.max_source_pieces = int(max_source_pieces)
        self.max_retokenization_work = int(max_retokenization_work)
        self.generation_id = int(generation_id)
        self.segment_id = int(segment_id)
        self.on_packet_emitted = on_packet_emitted
        self.span_cursor = SpanAlignedSpeechCursor(
            generation_id=self.generation_id,
            segment_id=self.segment_id,
        )
        self._condition = threading.Condition(threading.RLock())
        self._pieces: list[tuple[str, torch.Tensor, int | None, int | None]] = []
        self._source_packet_cursors: dict[int, int] = {}
        self._source_cursor_to_index: dict[int, int] = {}
        self._source_packets: dict[int, dict[str, Any]] = {}
        self._source_packet_order: list[int] = []
        self._source_piece_index: dict[int, int] = {}
        self._next_source_packet_to_emit = 0
        self._vocalized_text_parts: list[str] = []
        self._committed_packet_cursors: set[int] = set()
        self._span_packet_cursors: dict[int, tuple[int, ...]] = {}
        self._pending_emitted_frames: deque[int] = deque()
        self._last_received_emitted_frame = -1
        self._last_processed_emitted_frame = -1
        self._last_source_token_index: int | None = None
        self._source_char_count = 0
        self._retokenization_work = 0
        self._published_ids: list[int] = []
        self._published_offsets: list[tuple[int, int]] = []
        self._published_hidden: list[torch.Tensor] = []
        self._published_events: list[Any | None] = []
        self._qwen_cursor = 0
        self._last_qwen_ids: list[int] | None = None
        self._last_qwen_offsets: list[tuple[int, int]] | None = None
        self._last_qwen_hidden: list[torch.Tensor] | None = None
        self._last_prefix = ""
        self._added_token_metadata = self._loaded_added_token_metadata()
        self._added_token_ids = frozenset(item[0] for item in self._added_token_metadata)
        self._error: BaseException | None = None
        self._closed = False
        self._eos_enqueued = False
        self._eos_consumed = False
        self._finish_text: str | None = None

    @staticmethod
    def _validate_hidden(hidden: Any) -> torch.Tensor:
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise TypeError("Gemma hidden live deve essere un Tensor (1,1,D)")
        if int(hidden.shape[0]) != 1 or int(hidden.shape[1]) != 1:
            raise ValueError(f"Gemma hidden live deve avere shape (1,1,D), got {tuple(hidden.shape)}")
        if int(hidden.shape[-1]) <= 0:
            raise ValueError("Gemma hidden live ha una width vuota")
        return hidden.detach().clone()

    def _raise_terminal_locked(self) -> None:
        if self._error is not None:
            raise self._error

    def _current_alignment(self, text: str, pieces: list[tuple[str, torch.Tensor, int | None, int | None]]):
        """Retokenize the canonical visible prefix and align covered Qwen spans."""
        ids, qwen_offsets = qwen_assistant_text_offsets(
            text,
            tokenizer=self.qwen_tokenizer,
            processor=self.qwen_processor,
        )
        canonical, char_sources, piece_hiddens = self._canonical_source_map(pieces)
        if canonical != text:
            raise ValueError("Gemma visible pieces do not reconstruct the canonical Qwen prefix")
        aligned: list[torch.Tensor] = []
        source_cursors: list[tuple[int, ...]] = []
        for qwen_start, qwen_end in qwen_offsets:
            source_indices = sorted({
                source_index
                for char_index in range(int(qwen_start), int(qwen_end))
                for source_index in char_sources[char_index]
            })
            if (
                not source_indices
                or any(not char_sources[index] for index in range(int(qwen_start), int(qwen_end)))
            ):
                raise ValueError("Qwen token span is not fully covered by Gemma text pieces")
            aligned.append(torch.cat(
                [piece_hiddens[index] for index in source_indices], dim=1
            ).mean(dim=1, keepdim=True))
            source_cursors.append(tuple(sorted({
                int(pieces[index][3])
                for index in source_indices
                if pieces[index][3] is not None
            })))
        return ids, qwen_offsets, aligned, source_cursors, char_sources

    def _pretokenizer_segments(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        """Return spans from the actual loaded fast-tokenizer backend."""
        backend = getattr(self.qwen_tokenizer, "backend_tokenizer", None)
        pre_tokenizer = getattr(backend, "pre_tokenizer", None)
        split = getattr(pre_tokenizer, "pre_tokenize_str", None)
        if not callable(split):
            raise ValueError("live Qwen tokenization requires the tokenizer pre-tokenizer")
        try:
            result = [
                (str(segment), (int(span[0]), int(span[1])))
                for segment, span in split(text)
            ]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("Qwen pre-tokenizer returned invalid character spans") from exc
        cursor = 0
        for segment, (start, end) in result:
            if start < cursor or end <= start or end > len(text):
                raise ValueError("Qwen pre-tokenizer returned out-of-range or overlapping spans")
            cursor = end
        return result

    def _loaded_added_token_metadata(
        self,
    ) -> tuple[tuple[int, str, bool, bool, bool, bool], ...]:
        """Read active AddedToken IDs, content, and flags from the tokenizer."""
        decoder = getattr(self.qwen_tokenizer, "added_tokens_decoder", None)
        if not isinstance(decoder, dict):
            getter = getattr(self.qwen_tokenizer, "get_added_tokens_decoder", None)
            decoder = getter() if callable(getter) else {}
        items = decoder.items() if isinstance(decoder, dict) else ()
        metadata: set[tuple[int, str, bool, bool, bool, bool]] = set()
        for token_id, token in items:
            if isinstance(token_id, bool):
                continue
            try:
                token_id = int(token_id)
            except (TypeError, ValueError):
                continue
            content = getattr(token, "content", None)
            lstrip = bool(getattr(token, "lstrip", False))
            if content is None and isinstance(token, dict):
                content = token.get("content")
                lstrip = bool(token.get("lstrip", False))
            if isinstance(content, str) and content:
                rstrip = bool(getattr(token, "rstrip", False))
                normalized = bool(getattr(token, "normalized", True))
                single_word = bool(getattr(token, "single_word", False))
                if isinstance(token, dict):
                    rstrip = bool(token.get("rstrip", False))
                    normalized = bool(token.get("normalized", True))
                    single_word = bool(token.get("single_word", False))
                if normalized:
                    content = unicodedata.normalize("NFC", content)
                metadata.add((token_id, content, lstrip, rstrip, normalized, single_word))
        return tuple(sorted(metadata, key=lambda item: (-len(item[1]), item[1], item[0])))

    def _added_token_suffix_start(self, text: str) -> int | None:
        starts: list[int] = []
        for _token_id, content, lstrip, rstrip, _normalized, single_word in self._added_token_metadata:
            searchable = text.rstrip() if rstrip else text
            for length in range(1, min(len(content), len(searchable)) + 1):
                if not searchable.endswith(content[:length]):
                    continue
                token_start = len(searchable) - length
                if single_word and token_start > 0:
                    previous = searchable[token_start - 1]
                    if previous == "_" or previous.isalnum():
                        continue
                start = token_start
                if lstrip:
                    while start > 0 and searchable[start - 1].isspace():
                        start -= 1
                starts.append(start)
        return min(starts) if starts else None

    @staticmethod
    def _common_prefix_length(left: str, right: str) -> int:
        common = 0
        for old, new in zip(left, right):
            if old != new:
                break
            common += 1
        return common

    def _stable_boundary(self, text: str, segments: list[tuple[str, tuple[int, int]]]) -> int:
        """Use the current open segment, text LCP, and AddedToken candidate."""
        if not self._last_prefix or not segments:
            return 0
        boundary = int(segments[-1][1][0])
        common_chars = self._common_prefix_length(self._last_prefix, text)
        boundary = min(boundary, common_chars)
        added_start = self._added_token_suffix_start(text)
        if added_start is not None:
            boundary = min(boundary, added_start)
        return max(0, min(boundary, len(text)))

    @staticmethod
    def _assert_published_immutable(
        published_ids: list[int],
        published_offsets: list[tuple[int, int]],
        published_hidden: list[torch.Tensor],
        ids: list[int],
        offsets: list[tuple[int, int]],
        aligned: list[torch.Tensor],
    ) -> None:
        count = len(published_ids)
        if ids[:count] != published_ids:
            raise ValueError("published Qwen token IDs changed during incremental tokenization")
        if offsets[:count] != published_offsets:
            raise ValueError("published Qwen token offsets changed during incremental tokenization")
        if len(aligned) < count or any(
            not torch.equal(published_hidden[index], aligned[index])
            for index in range(count)
        ):
            raise ValueError("published Qwen hidden alignment changed during incremental tokenization")

    @staticmethod
    def _canonical_source_map(
        pieces: list[tuple[str, torch.Tensor]],
    ) -> tuple[str, list[list[int]], list[torch.Tensor]]:
        """NFC-normalize text while mapping output characters to source rows."""
        source_chars: list[tuple[str, set[int]]] = []
        source_hiddens: list[torch.Tensor] = []
        for piece_index, piece_row in enumerate(pieces):
            piece, hidden = piece_row[0], piece_row[1]
            source_hiddens.append(hidden)
            source_chars.extend((char, {piece_index}) for char in piece)
        raw_text = "".join(char for char, _source in source_chars)

        # Work in canonical decomposition space so a combining mark supplied
        # by a later Gemma piece remains attached to both source hidden rows.
        decomposed_sources: dict[str, deque[set[int]]] = defaultdict(deque)
        for char, owners in source_chars:
            for component in unicodedata.normalize("NFD", char):
                decomposed_sources[component].append(set(owners))
        nfd_text = unicodedata.normalize("NFD", raw_text)
        nfd_sources: list[set[int]] = []
        for component in nfd_text:
            owners = decomposed_sources[component]
            if not owners:
                raise ValueError("NFC source map could not recover a decomposed character owner")
            nfd_sources.append(owners.popleft())
        nfc_text = unicodedata.normalize("NFC", raw_text)
        normalized_nfc: list[tuple[str, set[int]]] = []
        nfd_cursor = 0
        for char in nfc_text:
            width = len(unicodedata.normalize("NFD", char))
            owners: set[int] = set()
            for source_set in nfd_sources[nfd_cursor:nfd_cursor + width]:
                owners.update(source_set)
            if not owners:
                raise ValueError("NFC output character has no Gemma source piece")
            normalized_nfc.append((char, owners))
            nfd_cursor += width
        if nfd_cursor != len(nfd_sources):
            raise ValueError("NFC source map did not consume the complete NFD source")

        raw_text = nfc_text
        normalized: list[tuple[str, set[int]]] = []
        cursor = 0
        for match in re.finditer(r"\s*<unused\d+>\s*", raw_text):
            normalized.extend(normalized_nfc[cursor:match.start()])
            replaced_sources: set[int] = set()
            for _char, owners in normalized_nfc[match.start():match.end()]:
                replaced_sources.update(owners)
            normalized.append((" ", replaced_sources))
            cursor = match.end()
        normalized.extend(normalized_nfc[cursor:])

        canonical_chars: list[str] = []
        char_sources: list[list[int]] = []
        index = 0
        while index < len(normalized):
            char, piece_indices = normalized[index]
            if char in " \t":
                whitespace_sources: set[int] = set()
                while index < len(normalized) and normalized[index][0] in " \t":
                    whitespace_sources.update(normalized[index][1])
                    index += 1
                canonical_chars.append(" ")
                char_sources.append(sorted(whitespace_sources))
                continue
            canonical_chars.append(char)
            char_sources.append(sorted(piece_indices))
            index += 1

        while canonical_chars and canonical_chars[0].isspace():
            canonical_chars.pop(0)
            char_sources.pop(0)
        while canonical_chars and canonical_chars[-1].isspace():
            canonical_chars.pop()
            char_sources.pop()
        return "".join(canonical_chars), char_sources, source_hiddens

    def _tail_queue_size_locked(self) -> int:
        consumed_tail = max(0, self._qwen_cursor - 1)
        return max(0, len(self._published_ids) - 1 - consumed_tail)

    def _retokenization_charge(self, text_length: int, piece_count: int) -> int:
        """Conservative operation budget for one cumulative alignment pass.

        The charge accounts for NFC/source mapping, loaded tokenizer and
        processor offsets, the actual pre-tokenizer, span-to-hidden alignment,
        prefix LCP/stability scans, and the bounded AddedToken suffix scan.
        Qwen byte-level tokens can expand a Unicode character to four bytes;
        cap that estimate by the configured observed-token limit.
        """
        length = max(0, int(text_length))
        pieces = max(0, int(piece_count))
        max_ids = min(self.max_tokens, 4 * length)
        added_token_work = sum(
            (width := min(len(content), length)) * (width + 1) // 2
            for _token_id, content, _lstrip, _rstrip, _normalized, _single_word
            in self._added_token_metadata
        )
        return (
            10 * length
            + 6 * max_ids
            + 4 * pieces
            + added_token_work
        )

    def _reserve_retokenization_work(self, text_length: int, piece_count: int) -> None:
        charge = self._retokenization_charge(text_length, piece_count)
        with self._condition:
            self._raise_terminal_locked()
            if self._retokenization_work + charge > self.max_retokenization_work:
                raise RuntimeError("live Qwen retokenization work capacity exceeded")
            self._retokenization_work += charge

    def _publish_token(
        self,
        token_id: int,
        offset: tuple[int, int],
        hidden: torch.Tensor,
        *,
        initial: bool = False,
        source_packet_cursors: tuple[int, ...] = (),
    ) -> None:
        owned_hidden = hidden.detach().clone()
        producer_event = None
        if owned_hidden.device.type == "cuda" and torch.cuda.is_available():
            producer_event = torch.cuda.Event(enable_timing=False)
            producer_event.record(torch.cuda.current_stream(device=owned_hidden.device))
        with self._condition:
            while not initial and self._tail_queue_size_locked() >= self.queue_capacity:
                self._raise_terminal_locked()
                if self._closed:
                    raise RuntimeError("live Qwen stream closed while producer was publishing")
                self._condition.wait(0.02)
            self._raise_terminal_locked()
            if self._closed:
                raise RuntimeError("live Qwen stream is closed")
            if len(self._published_ids) >= self.max_tokens:
                raise RuntimeError("live Qwen text token capacity exceeded")
            start, end = int(offset[0]), int(offset[1])
            token_text = self._last_prefix[start:end] if 0 <= start <= end <= len(self._last_prefix) else ""
            span = self.span_cursor.append(
                token_id=int(token_id),
                token_text=token_text,
                hidden=owned_hidden,
                producer_event=producer_event,
            )
            self._span_packet_cursors[span.start_frame] = tuple(source_packet_cursors)
            for cursor in source_packet_cursors:
                source_index = self._source_cursor_to_index.get(int(cursor))
                packet_record = (
                    self._source_packets.get(source_index)
                    if source_index is not None else None
                )
                if packet_record is not None:
                    packet_record["span_sequences"].add(int(span.token_sequence))
            self._published_ids.append(int(token_id))
            self._published_offsets.append((int(offset[0]), int(offset[1])))
            self._published_hidden.append(owned_hidden)
            self._published_events.append(producer_event)
            self._condition.notify_all()

    def feed_piece(
        self,
        text_piece: str,
        gemma_hidden: Any,
        *,
        producer_event: Any | None = None,
        token_index: int | None = None,
        exclusive_kv_cursor: int | None = None,
        kv_committed: bool = True,
    ) -> None:
        try:
            device = getattr(gemma_hidden, "device", None)
            if (
                producer_event is not None
                and device is not None
                and getattr(device, "type", None) == "cuda"
                and torch.cuda.is_available()
            ):
                stream = torch.cuda.current_stream(device=device)
                stream.wait_event(producer_event)
                if isinstance(gemma_hidden, torch.Tensor):
                    gemma_hidden.record_stream(stream)
            self._feed_piece(
                text_piece,
                gemma_hidden,
                token_index=token_index,
                exclusive_kv_cursor=exclusive_kv_cursor,
                kv_committed=kv_committed,
            )
        except BaseException as exc:
            self.fail(exc)
            raise

    def _feed_piece(
        self,
        text_piece: str,
        gemma_hidden: Any,
        *,
        token_index: int | None = None,
        exclusive_kv_cursor: int | None = None,
        kv_committed: bool = True,
    ) -> None:
        if not isinstance(text_piece, str):
            raise TypeError("Thinker token text deve essere string")
        hidden = self._validate_hidden(gemma_hidden)
        if not isinstance(kv_committed, bool):
            raise TypeError("kv_committed must be bool")
        if (token_index is None) != (exclusive_kv_cursor is None):
            raise ValueError("source token index and exclusive KV cursor must be supplied together")
        if token_index is None and not kv_committed:
            raise ValueError("uncommitted source pieces require a token index and KV cursor")
        if token_index is not None:
            if (
                isinstance(token_index, bool)
                or not isinstance(token_index, int)
                or token_index < 0
                or isinstance(exclusive_kv_cursor, bool)
                or not isinstance(exclusive_kv_cursor, int)
                or exclusive_kv_cursor < 0
            ):
                raise ValueError("source token index and KV cursor must be nonnegative integers")
            with self._condition:
                self._raise_terminal_locked()
                if self._closed:
                    raise RuntimeError("live Qwen stream is closed")
                expected_index = 0 if self._last_source_token_index is None else self._last_source_token_index + 1
                if token_index != expected_index:
                    raise RuntimeError("source Gemma token indices must be contiguous and monotonic")
                cursor = int(exclusive_kv_cursor)
                if cursor in self._source_cursor_to_index:
                    raise RuntimeError("source Gemma KV cursor was reused")
                if self._source_packet_cursors and cursor <= max(self._source_packet_cursors.values()):
                    raise RuntimeError("source Gemma KV cursors must be strictly increasing")
                self._source_packet_cursors[token_index] = cursor
                self._source_cursor_to_index[cursor] = token_index
                self._source_packets[token_index] = {
                    "cursor": cursor,
                    "text_piece": text_piece,
                    "committed": kv_committed,
                    "spans_known": False,
                    "span_sequences": set(),
                    "acked_sequences": set(),
                }
                self._source_packet_order.append(token_index)
                if kv_committed:
                    self._committed_packet_cursors.add(cursor)
                self._last_source_token_index = token_index
        if not text_piece:
            with self._condition:
                self._raise_terminal_locked()
                if self._closed:
                    raise RuntimeError("live Qwen stream is closed")
            return
        with self._condition:
            self._raise_terminal_locked()
            if self._closed:
                raise RuntimeError("live Qwen stream is closed")
            if self._source_char_count + len(text_piece) > self.max_source_chars:
                raise RuntimeError("live Qwen source character capacity exceeded")
            if len(self._pieces) + 1 > self.max_source_pieces:
                raise RuntimeError("live Qwen source piece capacity exceeded")
            self._source_char_count += len(text_piece)
            self._pieces.append((text_piece, hidden, token_index, exclusive_kv_cursor))
            if token_index is not None:
                self._source_piece_index[token_index] = len(self._pieces) - 1
            snapshot = list(self._pieces)
        prefix, _char_sources, _source_hiddens = self._canonical_source_map(snapshot)
        if not prefix:
            return
        self._reserve_retokenization_work(len(prefix), len(snapshot))
        ids, offsets, aligned, source_packet_cursors, char_sources = self._current_alignment(
            prefix, snapshot
        )
        segments = self._pretokenizer_segments(prefix)
        if len(ids) > self.max_tokens or len(aligned) > self.max_tokens:
            raise RuntimeError("live Qwen observed token/hidden row capacity exceeded")
        if len(ids) != len(offsets) or len(ids) != len(aligned):
            raise ValueError("live Qwen observed IDs, offsets, and hidden rows are not aligned")
        with self._condition:
            self._raise_terminal_locked()
            self._assert_published_immutable(
                self._published_ids,
                self._published_offsets,
                self._published_hidden,
                ids,
                offsets,
                aligned,
            )
            stable_boundary = self._stable_boundary(prefix, segments)
            stable_count = 0
            for token_id, (_start, end) in zip(ids, offsets):
                if int(end) > stable_boundary:
                    break
                # Equality means this AddedToken touches the open boundary.
                # Wait for context to move the boundary beyond it; EOS may
                # release it only after the full prefix is validated.
                if int(end) == stable_boundary and int(token_id) in self._added_token_ids:
                    break
                stable_count += 1
            if self._last_qwen_ids is not None and self._last_qwen_offsets is not None:
                lcp = 0
                for old_id, old_offset, token_id, offset in zip(
                    self._last_qwen_ids,
                    self._last_qwen_offsets,
                    ids,
                    offsets,
                ):
                    if old_id != token_id or old_offset != offset:
                        break
                    lcp += 1
                stable_count = min(stable_count, lcp)
            stable_source_indices = {
                self._source_cursor_to_index[cursor]
                for token_sources in source_packet_cursors[:stable_count]
                for cursor in token_sources
                if cursor in self._source_cursor_to_index
            }
            latest_stable_source_index = max(stable_source_indices, default=-1)
            source_char_ends: dict[int, int] = {}
            for char_index, source_indices in enumerate(char_sources):
                for source_index in source_indices:
                    source_char_ends[source_index] = char_index + 1
            for token_index, record in self._source_packets.items():
                piece_index = self._source_piece_index.get(token_index)
                if piece_index is not None and piece_index in source_char_ends:
                    if source_char_ends[piece_index] <= stable_boundary:
                        record["spans_known"] = True
                elif token_index < latest_stable_source_index:
                    # Canonicalized-away leading text has no Qwen span once a
                    # later stable token has passed it.
                    record["spans_known"] = True
            self._last_qwen_ids = list(ids)
            self._last_qwen_offsets = list(offsets)
            self._last_qwen_hidden = [hidden.detach().clone() for hidden in aligned]
            self._last_prefix = prefix
            next_index = len(self._published_ids)
            pending = [
                (
                    ids[index], offsets[index], aligned[index], index == 0,
                    source_packet_cursors[index],
                )
                for index in range(next_index, stable_count)
            ]
        for token_id, offset, token_hidden, initial, source_cursors in pending:
            self._publish_token(
                token_id,
                offset,
                token_hidden,
                initial=initial,
                source_packet_cursors=source_cursors,
            )
        with self._condition:
            emitted = self._collect_vocalized_packets_locked()
        self._dispatch_packet_emissions(emitted)

    def finish(self, *, final_text: str) -> None:
        try:
            self._finish(final_text=final_text)
        except BaseException as exc:
            self.fail(exc)
            raise

    def _finish(self, *, final_text: str) -> None:
        if not isinstance(final_text, str) or not final_text.strip():
            raise ValueError("final SAY text deve essere non vuoto")
        with self._condition:
            self._raise_terminal_locked()
            if self._closed:
                raise RuntimeError("live Qwen stream è già chiuso")
            snapshot = list(self._pieces)
        canonical_text, _char_sources, _source_hiddens = self._canonical_source_map(snapshot)
        final_text = unicodedata.normalize("NFC", final_text)
        if canonical_text != final_text:
            raise ValueError("Thinker callback text non corrisponde al testo finale canonico")
        self._reserve_retokenization_work(len(canonical_text), len(snapshot))
        ids, offsets, aligned, source_packet_cursors, _char_sources = self._current_alignment(
            canonical_text, snapshot
        )
        if len(ids) > self.max_tokens or len(aligned) > self.max_tokens:
            raise RuntimeError("live Qwen observed token/hidden row capacity exceeded")
        if len(ids) != len(offsets) or len(ids) != len(aligned):
            raise ValueError("live Qwen final IDs, offsets, and hidden rows are not aligned")
        with self._condition:
            self._raise_terminal_locked()
            self._assert_published_immutable(
                self._published_ids,
                self._published_offsets,
                self._published_hidden,
                ids,
                offsets,
                aligned,
            )
            if self._last_qwen_ids is not None and (
                self._last_qwen_ids != ids or self._last_qwen_offsets != offsets
            ):
                raise ValueError("live Qwen full-prefix IDs or offsets drifted at EOS")
            if self._last_qwen_hidden is not None and (
                len(self._last_qwen_hidden) != len(aligned)
                or any(
                    not torch.equal(previous, current)
                    for previous, current in zip(self._last_qwen_hidden, aligned)
                )
            ):
                raise ValueError("live Qwen full-prefix hidden alignment drifted at EOS")
            published = list(self._published_ids)
        for index in range(len(published), len(ids)):
            self._publish_token(
                ids[index],
                offsets[index],
                aligned[index],
                initial=index == 0,
                source_packet_cursors=source_packet_cursors[index],
            )
        with self._condition:
            self._raise_terminal_locked()
            if self._eos_enqueued:
                raise RuntimeError("native Qwen text EOS è già stato accodato")
            self._finish_text = canonical_text
            for record in self._source_packets.values():
                record["spans_known"] = True
            self._eos_enqueued = True
            self._closed = True
            self.span_cursor.close()
            self._condition.notify_all()
            emitted = self._collect_vocalized_packets_locked()
        self._dispatch_packet_emissions(emitted)

    def _wait_locked(self, predicate, *, timeout_s: float | None, stop_check: Any | None) -> None:
        started = time.monotonic()
        while not predicate():
            self._raise_terminal_locked()
            if stop_check is not None and stop_check():
                raise LiveSayCancelled("live Qwen stream consumer cancelled")
            if self._closed:
                return
            if timeout_s is not None:
                remaining = float(timeout_s) - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("live Qwen stream wait timed out")
                self._condition.wait(min(0.02, remaining))
            else:
                self._condition.wait(0.02)

    def take_initial_token(
        self,
        *,
        timeout_s: float | None = None,
        stop_check: Any | None = None,
    ) -> LiveQwenToken:
        with self._condition:
            self._wait_locked(
                lambda: bool(self._published_ids),
                timeout_s=timeout_s,
                stop_check=stop_check,
            )
            self._raise_terminal_locked()
            if not self._published_ids:
                raise RuntimeError("live Qwen stream closed before text token 0")
            if self._qwen_cursor != 0:
                raise RuntimeError("Qwen initial text token was already consumed")
            self._qwen_cursor = 1
            self._condition.notify_all()
            return LiveQwenToken(self._published_ids[0], self._published_hidden[0])

    def take_trailing_token(
        self,
        *,
        timeout_s: float | None = None,
        stop_check: Any | None = None,
    ) -> int | None:
        with self._condition:
            self._wait_locked(
                lambda: self._qwen_cursor < len(self._published_ids)
                or (self._closed and self._eos_enqueued),
                timeout_s=timeout_s,
                stop_check=stop_check,
            )
            self._raise_terminal_locked()
            if self._qwen_cursor < len(self._published_ids):
                token_id = self._published_ids[self._qwen_cursor]
                self._qwen_cursor += 1
                self._condition.notify_all()
                return token_id
            if self._eos_enqueued and not self._eos_consumed:
                self._eos_consumed = True
                self._condition.notify_all()
                return self.text_eos_token_id
            return None

    def hidden_for_frame(
        self,
        frame_id: int,
        *,
        stop_check: Any | None = None,
    ) -> LiveQwenHidden:
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError("frame_id deve essere un intero non negativo")
        with self._condition:
            self._wait_locked(
                lambda: bool(self.span_cursor.frame_spans) or self._closed,
                timeout_s=None,
                stop_check=stop_check,
            )
            self._raise_terminal_locked()
            if not self.span_cursor.frame_spans:
                raise RuntimeError("live Qwen stream closed before aligned text hidden was available")
            selected = self.span_cursor.select(
                frame_id,
                generation_id=self.generation_id,
                segment_id=self.segment_id,
            )
            span = selected.span
            return LiveQwenHidden(
                span.hidden,
                selected.tail_hold,
                qwen_token_id=int(span.token_id),
                producer_event=span.producer_event,
            )

    def ack_emitted_audio_frame(
        self,
        frame_id: int,
        *,
        generation_id: int,
        segment_id: int,
    ) -> tuple[int, ...]:
        """Return Qwen IDs whose span-start PCM frame is emitted and committed."""
        if int(generation_id) != self.generation_id or int(segment_id) != self.segment_id:
            raise RuntimeError("PCM frame ACK belongs to a stale generation or segment")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")
        with self._condition:
            self._raise_terminal_locked()
            if frame_id != self._last_received_emitted_frame + 1:
                raise RuntimeError("emitted PCM frame ACKs must be contiguous and monotonic")
            self._pending_emitted_frames.append(frame_id)
            self._last_received_emitted_frame = frame_id
            acknowledged = self._flush_pending_emitted_frames_locked()
            emitted = self._collect_vocalized_packets_locked()
        self._dispatch_packet_emissions(emitted)
        return acknowledged

    def _flush_pending_emitted_frames_locked(self) -> tuple[int, ...]:
        acknowledged: list[int] = []
        while self._pending_emitted_frames:
            frame_id = self._pending_emitted_frames[0]
            if frame_id != self._last_processed_emitted_frame + 1:
                raise RuntimeError("pending PCM emission frames lost contiguous order")
            source_cursors = self._span_packet_cursors.get(frame_id, ())
            last_source_index = max(
                (self._source_cursor_to_index[cursor] for cursor in source_cursors),
                default=-1,
            )
            if any(
                cursor not in self._committed_packet_cursors
                for index, cursor in self._source_packet_cursors.items()
                if index <= last_source_index
            ):
                break
            started = self.span_cursor.ack_emitted_frame(
                frame_id,
                generation_id=self.generation_id,
                segment_id=self.segment_id,
            )
            acknowledged.extend(span.token_id for span in started)
            for span in started:
                for cursor in self._span_packet_cursors.get(span.start_frame, ()):
                    source_index = self._source_cursor_to_index.get(cursor)
                    packet_record = (
                        self._source_packets.get(source_index)
                        if source_index is not None else None
                    )
                    if packet_record is not None:
                        packet_record["acked_sequences"].add(int(span.token_sequence))
            self._pending_emitted_frames.popleft()
            self._last_processed_emitted_frame = frame_id
        return tuple(acknowledged)

    def _collect_vocalized_packets_locked(self) -> list[tuple[int, str]]:
        emitted: list[tuple[int, str]] = []
        while self._next_source_packet_to_emit < len(self._source_packet_order):
            token_index = self._source_packet_order[self._next_source_packet_to_emit]
            record = self._source_packets[token_index]
            if not record["spans_known"]:
                break
            span_sequences = record["span_sequences"]
            if not span_sequences:
                # No Qwen token depends on this source text, so it has no PCM
                # span to own and cannot advance the voiced prefix.
                self._next_source_packet_to_emit += 1
                continue
            if (
                not record["committed"]
                or not span_sequences.issubset(record["acked_sequences"])
            ):
                break
            self._vocalized_text_parts.append(str(record["text_piece"]))
            emitted.append((
                int(record["cursor"]),
                "".join(self._vocalized_text_parts),
            ))
            self._next_source_packet_to_emit += 1
        return emitted

    def _dispatch_packet_emissions(self, emitted: list[tuple[int, str]]) -> None:
        callback = self.on_packet_emitted
        if not callable(callback):
            return
        for cursor, text in emitted:
            callback(cursor, text)

    def mark_packet_committed(
        self,
        exclusive_kv_cursor: int,
        *,
        generation_id: int,
        segment_id: int,
    ) -> tuple[int, ...] | None:
        """Release queued PCM ACKs after the matching Gemma packet appends."""
        if (
            isinstance(exclusive_kv_cursor, bool)
            or not isinstance(exclusive_kv_cursor, int)
            or exclusive_kv_cursor < 0
        ):
            raise ValueError("exclusive_kv_cursor must be a nonnegative integer")
        if (
            isinstance(generation_id, bool)
            or int(generation_id) != self.generation_id
            or isinstance(segment_id, bool)
            or int(segment_id) != self.segment_id
        ):
            return None
        with self._condition:
            if self._error is not None:
                return None
            token_index = self._source_cursor_to_index.get(exclusive_kv_cursor)
            if token_index is None:
                raise RuntimeError("committed KV cursor has no source Gemma packet")
            # A commit cannot leap over an earlier uncommitted source token.
            for earlier_index, earlier_cursor in self._source_packet_cursors.items():
                if earlier_index < token_index and earlier_cursor not in self._committed_packet_cursors:
                    return None
            if exclusive_kv_cursor in self._committed_packet_cursors:
                return ()
            self._committed_packet_cursors.add(exclusive_kv_cursor)
            self._source_packets[token_index]["committed"] = True
            acknowledged = self._flush_pending_emitted_frames_locked()
            emitted = self._collect_vocalized_packets_locked()
        self._dispatch_packet_emissions(emitted)
        return acknowledged

    def cancel(self) -> None:
        with self._condition:
            if self._error is not None:
                return
            self._error = LiveSayCancelled("live Qwen stream cancelled")
            self._pending_emitted_frames.clear()
            self._committed_packet_cursors.clear()
            self.span_cursor.close()
            self._condition.notify_all()

    def fail(self, error: BaseException) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("error deve essere un'eccezione")
        with self._condition:
            if self._error is not None:
                return
            self._error = error
            self._pending_emitted_frames.clear()
            self._committed_packet_cursors.clear()
            self.span_cursor.close()
            self._condition.notify_all()

    @property
    def codec_eos_allowed(self) -> bool:
        with self._condition:
            return bool(
                self._closed
                and self._eos_consumed
                and self._qwen_cursor >= len(self._published_ids)
            )

    @property
    def visible_text(self) -> str:
        """Cumulative callback text used only to extend the bounded frame budget."""
        with self._condition:
            text, _char_sources, _source_hiddens = self._canonical_source_map(self._pieces)
            return text

    @property
    def observed_qwen_token_count(self) -> int:
        """Current complete-prefix Qwen token count, including held-back IDs."""
        with self._condition:
            return len(self._last_qwen_ids or ())

    @property
    def published_qwen_ids(self) -> list[int]:
        with self._condition:
            return list(self._published_ids)

    @property
    def published_offsets(self) -> list[tuple[int, int]]:
        with self._condition:
            return list(self._published_offsets)

    @property
    def published_hidden(self) -> list[torch.Tensor]:
        with self._condition:
            return [hidden.detach().clone() for hidden in self._published_hidden]


class TurnLocalQwenSayStream:
    """Incremental v2 SAY consumer backed by the session's only text-ahead FIFO.

    Each complete visible Gemma SAY delta is encoded independently. The exact
    Gemma post-token state is attached to every Qwen ID produced by that delta;
    FIFO capacity therefore counts source Gemma packets, not an unbounded
    retokenized prefix. Independent packet tokenization is accepted only when
    Qwen decode reproduces the source delta byte-for-character.
    """

    def __init__(
        self,
        *,
        qwen_tokenizer: Any,
        text_eos_token_id: int,
        fifo: TurnLocalSpeechFIFO,
        generation_id: int,
        max_tokens: int = 512,
        max_source_chars: int = 16_384,
        on_packet_consumed: Any | None = None,
        on_packet_emitted: Any | None = None,
    ) -> None:
        if isinstance(text_eos_token_id, bool) or int(text_eos_token_id) < 0:
            raise ValueError("text_eos_token_id deve essere non negativo")
        if isinstance(max_tokens, bool) or int(max_tokens) < 1:
            raise ValueError("max_tokens deve essere positivo")
        if isinstance(max_source_chars, bool) or int(max_source_chars) < 1:
            raise ValueError("max_source_chars deve essere positivo")
        if not isinstance(fifo, TurnLocalSpeechFIFO):
            raise TypeError("fifo deve essere TurnLocalSpeechFIFO")
        self.qwen_tokenizer = qwen_tokenizer
        self.text_eos_token_id = int(text_eos_token_id)
        self.fifo = fifo
        self.generation_id = int(generation_id)
        self.segment_id = fifo.segment_id
        self.max_tokens = int(max_tokens)
        self.max_source_chars = int(max_source_chars)
        if on_packet_consumed is not None and on_packet_emitted is not None:
            raise ValueError("specify only one packet emission callback")
        # Keep the historical keyword accepted while moving its trigger to
        # the PCM-emission boundary. New callers should use on_packet_emitted.
        self.on_packet_emitted = on_packet_emitted or on_packet_consumed
        self._condition = threading.Condition(threading.RLock())
        self._visible_parts: list[str] = []
        self._visible_char_count = 0
        self._observed_qwen_tokens = 0
        self._last_hidden: torch.Tensor | None = None
        self._error: BaseException | None = None
        self._closed = False
        self._initial_taken = False
        self._initial_inflight = False
        self._trailing_inflight = False
        self._eos_consumed = False
        self._eos_inflight = False
        self._tail_hold = False
        self._initial_packet: SpokenTokenPacket | None = None
        self._trailing_token: SpeechQwenToken | None = None
        self._last_packet_token: SpeechQwenToken | None = None

    @staticmethod
    def _ids(encoded: Any) -> list[int]:
        if callable(getattr(encoded, "detach", None)):
            encoded = encoded.detach().cpu().reshape(-1).tolist()
        elif callable(getattr(encoded, "tolist", None)):
            encoded = encoded.tolist()
        while isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
            encoded = encoded[0]
        if not isinstance(encoded, (list, tuple)):
            raise TypeError("Qwen tokenizer encode deve restituire una sequenza di ID")
        ids = [int(value) for value in encoded]
        if any(value < 0 for value in ids):
            raise ValueError("Qwen tokenizer ha prodotto un ID negativo")
        return ids

    def _encode_piece(self, text_piece: str) -> tuple[list[int], tuple[str, ...]]:
        encode = getattr(self.qwen_tokenizer, "encode", None)
        decode = getattr(self.qwen_tokenizer, "decode", None)
        if not callable(encode) or not callable(decode):
            raise RuntimeError("Qwen tokenizer deve supportare encode e decode")
        ids = self._ids(encode(text_piece, add_special_tokens=False))
        if len(ids) > self.max_tokens:
            raise RuntimeError("live Qwen packet token capacity exceeded")
        try:
            decoded = decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            decoded = decode(ids)
        if not isinstance(decoded, str) or decoded != text_piece:
            raise RuntimeError("Qwen packet tokenization is not an exact text round trip")
        surfaces = self._offset_surfaces(text_piece, ids)
        if surfaces is None:
            surfaces = self._decoded_prefix_surfaces(text_piece, ids, decode)
        if "".join(surfaces) != text_piece:
            raise RuntimeError("Qwen token surfaces do not reconstruct the packet text")
        return ids, surfaces

    def _offset_surfaces(self, text_piece: str, ids: list[int]) -> tuple[str, ...] | None:
        tokenizer = self.qwen_tokenizer
        if not callable(tokenizer):
            return None
        try:
            encoded = tokenizer(
                text_piece,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            encoded_ids = self._ids(encoded["input_ids"])
            offsets = encoded["offset_mapping"]
            if callable(getattr(offsets, "tolist", None)):
                offsets = offsets.tolist()
            while isinstance(offsets, list) and len(offsets) == 1 and isinstance(offsets[0], list):
                offsets = offsets[0]
            if encoded_ids != ids or not isinstance(offsets, (list, tuple)) or len(offsets) != len(ids):
                return None
            normalized: list[tuple[int, int]] = []
            previous_end = 0
            for item in offsets:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    return None
                start, end = int(item[0]), int(item[1])
                if start < previous_end or end < start or end > len(text_piece):
                    return None
                normalized.append((start, end))
                previous_end = end
            # Preserve skipped whitespace by attaching each gap to the token
            # whose offset ends after it; no packet text is split heuristically.
            surfaces: list[str] = []
            start = 0
            for index, (_token_start, token_end) in enumerate(normalized):
                end = len(text_piece) if index == len(normalized) - 1 else max(start, token_end)
                surfaces.append(text_piece[start:end])
                start = end
            if len(surfaces) == len(ids) and "".join(surfaces) == text_piece:
                return tuple(surfaces)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            return None
        return None

    @staticmethod
    def _decoded_prefix_surfaces(text_piece: str, ids: list[int], decode: Any) -> tuple[str, ...]:
        surfaces: list[str] = []
        previous = ""
        for index in range(1, len(ids) + 1):
            try:
                prefix = decode(
                    ids[:index],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                prefix = decode(ids[:index])
            if not isinstance(prefix, str) or not prefix.startswith(previous):
                raise RuntimeError("Qwen tokenizer cannot expose stable token text surfaces")
            surfaces.append(prefix[len(previous):])
            previous = prefix
        if previous != text_piece:
            raise RuntimeError("Qwen decoded token prefixes do not reconstruct packet text")
        return tuple(surfaces)

    def feed_packet(
        self,
        packet: SpokenTokenPacket,
        *,
        stop_check: Any | None = None,
    ) -> None:
        try:
            self._feed_packet(packet, stop_check=stop_check)
        except BaseException as exc:
            self.fail(exc)
            raise

    def _feed_packet(self, packet: SpokenTokenPacket, *, stop_check: Any | None) -> None:
        if not isinstance(packet, SpokenTokenPacket):
            raise TypeError("packet deve essere SpokenTokenPacket")
        if not packet.text_piece:
            return
        with self._condition:
            if self._error is not None:
                raise self._error
            if self._closed:
                raise RuntimeError("turn-local Qwen SAY stream is closed")
            if self._visible_char_count + len(packet.text_piece) > self.max_source_chars:
                raise RuntimeError("turn-local Qwen SAY text capacity exceeded")
            if self._observed_qwen_tokens >= self.max_tokens:
                raise RuntimeError("turn-local Qwen SAY token capacity exceeded")
        qwen_ids, qwen_token_texts = self._encode_piece(packet.text_piece)
        if not qwen_ids:
            raise RuntimeError("visible Gemma SAY packet encoded to zero Qwen IDs")
        with self._condition:
            if self._error is not None:
                raise self._error
            if self._closed:
                raise RuntimeError("turn-local Qwen SAY stream is closed")
            if self._observed_qwen_tokens + len(qwen_ids) > self.max_tokens:
                raise RuntimeError("turn-local Qwen SAY token capacity exceeded")
        self.fifo.put(
            packet,
            qwen_token_ids=tuple(qwen_ids),
            qwen_token_texts=qwen_token_texts,
            generation_id=self.generation_id,
            segment_id=self.segment_id,
            stop_check=stop_check,
        )
        with self._condition:
            self._visible_parts.append(packet.text_piece)
            self._visible_char_count += len(packet.text_piece)
            self._observed_qwen_tokens += len(qwen_ids)
            self._condition.notify_all()

    def take_initial_token(
        self,
        *,
        timeout_s: float | None = None,
        stop_check: Any | None = None,
    ) -> LiveQwenToken:
        with self._condition:
            if self._initial_taken:
                raise RuntimeError("Qwen initial text token was already consumed")
        token = self.fifo.take_qwen_token(
            generation_id=self.generation_id,
            segment_id=self.segment_id,
            initial=True,
            stop_check=stop_check,
            timeout_s=timeout_s,
        )
        if token is None:
            self._raise_terminal()
            raise RuntimeError("turn-local Qwen SAY stream closed before token 0")
        hidden = self._hidden_3d(token.packet)
        with self._condition:
            self._initial_taken = True
            self._initial_inflight = True
            self._last_hidden = hidden
            self._initial_packet = token.packet
            self._last_packet_token = token
        return LiveQwenToken(token.token_id, hidden)

    def consume_initial_token(self) -> None:
        with self._condition:
            if not self._initial_inflight:
                raise RuntimeError("turn-local Qwen initial token is not in flight")
            packet = self._initial_packet
        if packet is None:
            raise RuntimeError("turn-local Qwen initial packet was lost before ACK")
        self.fifo.consume_qwen_token(
            generation_id=self.generation_id,
            segment_id=self.segment_id,
        )
        with self._condition:
            self._initial_inflight = False
            self._initial_packet = None

    def ack_initial_token(self) -> None:
        """Compatibility alias; this acknowledges Qwen ingestion, not PCM."""
        self.consume_initial_token()

    def take_trailing_token(
        self,
        *,
        timeout_s: float | None = None,
        stop_check: Any | None = None,
    ) -> int | None:
        self._raise_terminal()
        token = self.fifo.take_qwen_token(
            generation_id=self.generation_id,
            segment_id=self.segment_id,
            stop_check=stop_check,
            timeout_s=timeout_s,
        )
        # A failure may race the FIFO wakeup caused by clear(). Propagate it
        # before translating an empty, closed queue into a text EOS token.
        self._raise_terminal()
        if token is not None:
            with self._condition:
                if self._trailing_inflight:
                    raise RuntimeError("turn-local Qwen trailing token is already in flight")
                self._trailing_inflight = True
                self._last_hidden = self._hidden_3d(token.packet)
                self._tail_hold = False
                self._trailing_token = token
                self._last_packet_token = token
            return token.token_id
        with self._condition:
            if not self._closed:
                return None
            if not self._eos_consumed and not self._eos_inflight:
                self._eos_inflight = True
                self._tail_hold = True
                return self.text_eos_token_id
            return None

    def consume_trailing_token(self) -> None:
        with self._condition:
            if self._eos_inflight:
                self._eos_inflight = False
                self._eos_consumed = True
                return
            if not self._trailing_inflight:
                raise RuntimeError("turn-local Qwen trailing token is not in flight")
            token = self._trailing_token
        if token is None:
            raise RuntimeError("turn-local Qwen trailing packet was lost before ACK")
        self.fifo.consume_qwen_token(
            generation_id=self.generation_id,
            segment_id=self.segment_id,
        )
        with self._condition:
            self._trailing_inflight = False
            self._trailing_token = None
    def ack_trailing_token(self) -> None:
        """Compatibility alias; this acknowledges Qwen ingestion, not PCM."""
        self.consume_trailing_token()

    def ack_emitted_audio_frame(
        self,
        frame_id: int,
        *,
        generation_id: int,
        segment_id: int,
    ) -> int | None:
        if int(generation_id) != self.generation_id:
            raise RuntimeError("PCM frame ACK belongs to a stale generation")
        if int(segment_id) != self.segment_id:
            raise RuntimeError("PCM frame ACK belongs to a stale segment")
        cursor = self.fifo.ack_emitted_audio_frame(
            frame_id,
            generation_id=generation_id,
            segment_id=segment_id,
        )
        if cursor is not None and callable(self.on_packet_emitted):
            self.on_packet_emitted(int(cursor), self.fifo.vocalized_text)
        return cursor

    def mark_packet_committed(
        self,
        exclusive_kv_cursor: int,
        *,
        generation_id: int,
        segment_id: int | None = None,
    ) -> int | None:
        """Release PCM-acknowledged text after its Gemma KV append succeeds."""
        effective_segment_id = self.segment_id if segment_id is None else segment_id
        if (
            int(generation_id) != self.generation_id
            or int(effective_segment_id) != self.segment_id
        ):
            return None
        cursor = self.fifo.mark_packet_committed(
            exclusive_kv_cursor,
            generation_id=generation_id,
            segment_id=effective_segment_id,
        )
        if cursor is not None and callable(self.on_packet_emitted):
            self.on_packet_emitted(int(cursor), self.fifo.vocalized_text)
        return cursor

    @staticmethod
    def _hidden_3d(packet: SpokenTokenPacket) -> torch.Tensor:
        hidden = packet.gemma_hidden_1536
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("Gemma packet hidden deve essere Tensor")
        return hidden.unsqueeze(1)

    def hidden_for_frame(self, frame_id: int, *, stop_check: Any | None = None) -> LiveQwenHidden:
        if isinstance(frame_id, bool) or int(frame_id) < 0:
            raise ValueError("frame_id deve essere non negativo")
        self._raise_terminal()
        with self._condition:
            if self.fifo.segment_id != self.segment_id:
                raise RuntimeError("turn-local Qwen hidden requested from a stale segment")
            has_spans = bool(
                self.fifo.span_cursor is not None
                and self.fifo.span_cursor.frame_spans
            )
            closed = self._closed
        if has_spans:
            selected = self.fifo.select_span_for_frame(
                int(frame_id),
                generation_id=self.generation_id,
                segment_id=self.segment_id,
            )
            span = selected.span
            packet = span.packet
            return LiveQwenHidden(
                self._hidden_3d(packet),
                tail_hold=selected.tail_hold,
                gemma_token_id=int(packet.token_id),
                packet_exclusive_kv_cursor=int(packet.exclusive_kv_cursor),
                qwen_token_id=int(span.token_id),
                packet_token_index=span.packet_token_index,
                packet_token_count=span.packet_token_count,
                producer_event=packet.producer_event,
            )
        if closed:
            raise RuntimeError("turn-local Qwen SAY stream closed before a text hidden was consumed")
        # The first acoustic Q1 query follows the first codec frame and the
        # corresponding trailing Qwen text ID. Waiting here would indicate an
        # engine ordering regression, so fail rather than substitute a hidden.
        if stop_check is not None and stop_check():
            raise LiveSayCancelled("turn-local Qwen hidden consumer cancelled")
        raise RuntimeError("Qwen frame residual requested before a packet text ID was consumed")

    def finish(self, *, final_text: str) -> None:
        if not isinstance(final_text, str) or not final_text:
            raise ValueError("final SAY text deve essere non vuoto")
        with self._condition:
            self._raise_terminal()
            if self._closed:
                raise RuntimeError("turn-local Qwen SAY stream is already closed")
            visible = "".join(self._visible_parts)
            if visible != final_text:
                raise ValueError("incremental Gemma SAY differs from validated v2 SAY")
            self._closed = True
            self._condition.notify_all()
        if not self.fifo.close(
            generation_id=self.generation_id,
            segment_id=self.segment_id,
        ):
            raise RuntimeError("turn-local Qwen stream closed a stale segment")

    def cancel(self) -> None:
        with self._condition:
            if self._error is None and not self._closed:
                self._error = LiveSayCancelled("turn-local Qwen SAY stream cancelled")
            self._closed = True
            self._condition.notify_all()
        self.fifo.clear(
            generation_id=self.generation_id,
            segment_id=self.segment_id,
        )

    def fail(self, error: BaseException) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("error deve essere un'eccezione")
        with self._condition:
            if self._error is None:
                self._error = error
            self._closed = True
            self._condition.notify_all()
        self.fifo.clear(
            generation_id=self.generation_id,
            segment_id=self.segment_id,
        )

    def _raise_terminal(self) -> None:
        with self._condition:
            if self._error is not None:
                raise self._error

    @property
    def codec_eos_allowed(self) -> bool:
        with self._condition:
            return bool(
                self._closed
                and self._error is None
                and self._eos_consumed
                and not self._eos_inflight
                and not self._initial_inflight
                and not self._trailing_inflight
                and len(self.fifo) == 0
            )

    @property
    def visible_text(self) -> str:
        with self._condition:
            return "".join(self._visible_parts)

    @property
    def observed_qwen_token_count(self) -> int:
        with self._condition:
            return self._observed_qwen_tokens

    @property
    def published_qwen_ids(self) -> list[int]:
        # Diagnostics deliberately report accepted IDs, not unconsumed queue depth.
        return []
