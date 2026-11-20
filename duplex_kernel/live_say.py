"""Causally aligned, bounded text handoff for controller-guided SAY."""
from __future__ import annotations

import threading
import time
import re
import copy
import inspect
import unicodedata
from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass, field
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
    consumption: QwenTextConsumption | None = field(default=None, repr=False, compare=False)


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
    source_sequences: tuple[int, ...] = ()
    source_token_ids: tuple[int, ...] = ()
    q1_index: int | None = None
    eos: bool = False
    pad: bool = False


@dataclass(frozen=True, eq=False)
class _PendingQwenPublication:
    token_id: int
    offset: tuple[int, int]
    hidden: torch.Tensor
    initial: bool
    source_packet_cursors: tuple[int, ...]


@dataclass(frozen=True)
class QwenTextConsumption:
    """One immutable Qwen text input reserved for prefill or a successful Q1."""

    qwen_token_id: int | None
    source_sequences: tuple[int, ...]
    source_token_ids: tuple[int, ...]
    source_kv_cursors: tuple[int, ...] = field(default=(), repr=False)
    hidden: torch.Tensor | None = field(default=None, repr=False, compare=False)
    producer_event: Any | None = field(default=None, repr=False, compare=False)
    q1_index: int | None = None
    eos: bool = False
    pad: bool = False
    tail_hold: bool = False


@dataclass(frozen=True)
class QwenSourceFrontier:
    """Atomic text-consumption and source-identity snapshot for one live segment."""

    generation_id: int
    segment_id: int
    revision: int
    qwen_consumed_count: int
    reserved_qwen_token_id: int | None
    frozen_source_sequence: int
    source_sequences: tuple[int, ...]
    source_token_ids: tuple[int, ...]
    source_kv_cursors: tuple[int, ...] = field(default=(), repr=False)
    already_public_text: str = ""
    cumulative_source_text: str = ""
    cache_epoch: str | int | None = None


@dataclass(frozen=True)
class LiveQwenSuffixReplacementPlan:
    """Fully prepared stream state for replacing only its unfrozen source tail."""

    expected_frontier: QwenSourceFrontier
    revision_id: str
    cache_epoch: str | int | None
    staged: Any = field(repr=False, compare=False)
    span_cursor: Any = field(repr=False, compare=False)
    span_packet_cursors: dict[int, tuple[int, ...]] = field(repr=False, compare=False)
    committed_packet_cursors: set[int] = field(repr=False, compare=False)
    preserved_qwen_count: int
    source_rows: tuple[dict[str, Any], ...] = field(repr=False, compare=False)


@dataclass(frozen=True)
class LiveQwenFinalTextFlushPlan:
    """Fully validated final decoder text update for an unfrozen source row."""

    expected_source_revision: int
    staged_state: dict[str, Any] = field(repr=False, compare=False)
    post_commit_emissions: tuple[Any, ...] = field(default=(), repr=False, compare=False)


class _PrefixTailSequence:
    """Indexable cumulative view that does not copy an immutable prefix."""

    def __init__(self, prefix: list[Any], tail: list[Any]) -> None:
        self._prefix = prefix
        self._tail = tail

    def __len__(self) -> int:
        return len(self._prefix) + len(self._tail)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index < len(self._prefix):
            return self._prefix[index]
        return self._tail[index - len(self._prefix)]


class _BoundedSequence:
    """Read-only view of the fixed prefix of an append-only list."""

    def __init__(self, items, length: int | None = None) -> None:
        self._items = items
        self._length = len(items) if length is None else int(length)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._length))]
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        return self._items[index]


class _ConcatSequence:
    """Cumulative indexed view over immutable segments without list copies."""

    def __init__(self, parts) -> None:
        self._parts = tuple(part for part in parts if len(part))
        self._ends: list[int] = []
        total = 0
        for part in self._parts:
            total += len(part)
            self._ends.append(total)
        self._length = total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._length))]
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        part_index = bisect_right(self._ends, index)
        start = 0 if part_index == 0 else self._ends[part_index - 1]
        return self._parts[part_index][index - start]

    def __eq__(self, other) -> bool:
        try:
            if len(self) != len(other):
                return False
            return all(left == right for left, right in zip(self, other))
        except TypeError:
            return False


class _SourceCharEndView:
    """Overlay mutable-tail character ends on the stable-prefix mapping."""

    def __init__(
        self,
        prefix: dict[int, int],
        tail: dict[int, int],
        prefix_length: int,
    ) -> None:
        self._prefix = prefix
        self._tail = tail
        self._prefix_length = prefix_length

    def get(self, source_index: int, default=None):
        tail_end = self._tail.get(source_index)
        if tail_end is not None:
            return self._prefix_length + int(tail_end)
        return self._prefix.get(source_index, default)

    def __getitem__(self, source_index: int) -> int:
        value = self.get(source_index)
        if value is None:
            raise KeyError(source_index)
        return value

    def __contains__(self, source_index: int) -> bool:
        return source_index in self._tail or source_index in self._prefix


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
        cache_epoch: str | int | None = None,
        on_packet_emitted: Any | None = None,
        on_tts_text_reserved: Any | None = None,
        on_tts_text_consumed: Any | None = None,
        on_tts_eos_consumed: Any | None = None,
    ) -> None:
        if isinstance(max_tokens, bool) or int(max_tokens) < 1:
            raise ValueError("max_tokens deve essere positivo")
        if isinstance(queue_capacity, bool) or int(queue_capacity) < 1:
            raise ValueError("queue_capacity deve essere positivo")
        if isinstance(text_eos_token_id, bool) or int(text_eos_token_id) < 0:
            raise ValueError("text_eos_token_id deve essere non negativo")
        if cache_epoch is not None and (
            isinstance(cache_epoch, bool)
            or not isinstance(cache_epoch, (int, str))
            or (isinstance(cache_epoch, str) and not cache_epoch)
        ):
            raise ValueError("cache_epoch deve essere un intero, una stringa non vuota o None")
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
        max_source_chars = int(max_source_chars)
        max_source_pieces = int(max_source_pieces)
        max_retokenization_work = int(max_retokenization_work)
        if self.max_tokens > 512:
            if max_source_chars == 16_384:
                max_source_chars = max(16_384, self.max_tokens * 32)
            if max_source_pieces == 2_048:
                max_source_pieces = max(2_048, self.max_tokens * 4)
            if max_retokenization_work == 1_000_000:
                max_retokenization_work = max(
                    1_000_000,
                    int(2_000_000 * ((self.max_tokens / 512) ** 2)),
                )
        self.max_source_chars = max_source_chars
        self.max_source_pieces = max_source_pieces
        self.max_retokenization_work = max_retokenization_work
        self.generation_id = int(generation_id)
        self.segment_id = int(segment_id)
        self.cache_epoch = cache_epoch
        self.on_packet_emitted = on_packet_emitted
        self.on_tts_text_reserved = on_tts_text_reserved
        self.on_tts_text_consumed = on_tts_text_consumed
        self.on_tts_eos_consumed = on_tts_eos_consumed
        self.span_cursor = SpanAlignedSpeechCursor(
            generation_id=self.generation_id,
            segment_id=self.segment_id,
        )
        self._condition = threading.Condition(threading.RLock())
        self._pieces: list[tuple[str, torch.Tensor, int | None, int | None]] = []
        self._source_piece_raw_starts: list[int] = []
        self._source_piece_raw_ends: list[int] = []
        self._source_hiddens: list[torch.Tensor] = []
        self._canonical_cached_raw_end = 0
        self._canonical_cached_text_parts: list[str] = []
        self._canonical_cached_length = 0
        self._canonical_cached_sources: list[list[int]] = []
        self._canonical_cached_raw_ranges: list[tuple[int, int]] = []
        self._canonical_cached_source_char_ends: dict[int, int] = {}
        self._source_packet_cursors: dict[int, int] = {}
        # Physical KV positions are revision-local provenance and can repeat
        # after a private response branch rebases. Stream associations use
        # packet indices and logical source IDs instead.
        self._source_cursor_to_index: dict[int, list[int]] = {}
        self._source_sequence_to_index: dict[int, int] = {}
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
        self._deferred_publications: deque[_PendingQwenPublication] = deque()
        self._first_qwen_id_published_at_ns: int | None = None
        self._published_offsets: list[tuple[int, int]] = []
        self._published_hidden: list[torch.Tensor] = []
        self._published_events: list[Any | None] = []
        self._published_source_sequences: list[tuple[int, ...]] = []
        self._published_source_token_ids: list[tuple[int, ...]] = []
        self._published_source_cursors: list[tuple[int, ...]] = []
        self._published_source_packet_indices: list[tuple[int, ...]] = []
        self._source_revision = 0
        self._qwen_cursor = 0
        self._reserved_consumption: QwenTextConsumption | None = None
        self._reserved_qwen_index: int | None = None
        self._reserved_initial = False
        self._last_consumed_qwen_index: int | None = None
        self._last_consumed_source_sequences: tuple[int, ...] = ()
        self._q1_consumption_mode = False
        self._already_public_text = ""
        self._already_public_source_sequences: tuple[int, ...] = ()
        self._last_qwen_ids: list[int] | None = None
        self._last_qwen_offsets: list[tuple[int, int]] | None = None
        self._last_qwen_hidden: list[torch.Tensor] | None = None
        self._last_qwen_cursors: list[tuple[int, ...]] | None = None
        self._last_qwen_mutable_tail_ids: list[int] | None = None
        self._last_qwen_mutable_tail_offsets: list[tuple[int, int]] | None = None
        self._last_qwen_mutable_tail_hidden: list[torch.Tensor] | None = None
        self._last_qwen_mutable_tail_cursors: list[tuple[int, ...]] | None = None
        self._last_qwen_mutable_tail_base: int | None = None
        self._last_qwen_mutable_tail_text = ""
        self._current_qwen_tail: tuple[Any, ...] | None = None
        self._latest_stable_source_index = -1
        self._last_prefix = ""
        self._last_qwen_suffix_prefix = ""
        self._sealed_initial_text: str | None = None
        self._sealed_initial_id_count = 0
        self._added_token_metadata = self._loaded_added_token_metadata()
        self._added_token_matchers = tuple(
            (item, self._prefix_failure_table(item[1]))
            for item in self._added_token_metadata
        )
        self._added_token_ids = frozenset(item[0] for item in self._added_token_metadata)
        self._max_added_token_chars = max(
            (len(item[1]) for item in self._added_token_metadata),
            default=0,
        )
        self._max_added_token_chars_by_rstrip = {
            rstrip: max(
                len(item[1])
                for item in self._added_token_metadata
                if item[3] == rstrip
            )
            for rstrip in {item[3] for item in self._added_token_metadata}
        }
        self._added_token_trie_by_rstrip = self._build_added_token_tries(
            self._added_token_metadata
        )
        self._added_token_candidate_work_bound = max(
            (
                sum(
                    len(nodes[node_index]["token_ids"])
                    for node_index in self._failure_chain(nodes, state)
                )
                for nodes in self._added_token_trie_by_rstrip.values()
                for state in range(1, len(nodes))
            ),
            default=0,
        )
        self._sealed_initial_ids: list[int] = []
        self._sealed_initial_offsets: list[tuple[int, int]] = []
        self._sealed_initial_hidden: list[torch.Tensor] = []
        self._sealed_initial_cursors: list[tuple[int, ...]] = []
        self._stable_suffix_text = ""
        self._stable_suffix_ids: list[int] = []
        self._stable_suffix_offsets: list[tuple[int, int]] = []
        self._stable_suffix_hidden: list[torch.Tensor] = []
        self._stable_suffix_cursors: list[tuple[int, ...]] = []
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

    def _current_alignment(
        self,
        text: str,
        pieces: list[tuple[str, torch.Tensor, int | None, int | None]],
        *,
        strip_leading: bool = True,
        canonical_data: tuple[str, list[list[int]], list[torch.Tensor]] | None = None,
    ):
        """Retokenize the canonical visible prefix and align covered Qwen spans."""
        ids, qwen_offsets = qwen_assistant_text_offsets(
            text,
            tokenizer=self.qwen_tokenizer,
            processor=self.qwen_processor,
        )
        canonical, char_sources, piece_hiddens = (
            canonical_data
            if canonical_data is not None
            else self._canonical_source_map(pieces, strip_leading=strip_leading)
        )
        if canonical != text:
            raise ValueError("Gemma visible pieces do not reconstruct the canonical Qwen prefix")
        aligned, source_cursors = self._align_token_offsets(
            ids,
            qwen_offsets,
            source_offset=0,
            pieces=pieces,
            char_sources=char_sources,
            piece_hiddens=piece_hiddens,
        )
        return ids, qwen_offsets, aligned, source_cursors, char_sources

    @staticmethod
    def _align_token_offsets(
        token_ids: list[int],
        token_offsets: list[tuple[int, int]],
        *,
        source_offset: int,
        pieces: list[tuple[str, torch.Tensor, int | None, int | None]],
        char_sources: list[list[int]],
        piece_hiddens: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[tuple[int, ...]]]:
        aligned: list[torch.Tensor] = []
        source_cursors: list[tuple[int, ...]] = []
        for start, end in token_offsets:
            start = int(start) + source_offset
            end = int(end) + source_offset
            source_indices = sorted({
                source_index
                for char_index in range(start, end)
                for source_index in char_sources[char_index]
            })
            if (
                not source_indices
                or any(not char_sources[index] for index in range(start, end))
            ):
                raise ValueError("Qwen token span is not fully covered by Gemma text pieces")
            aligned.append(torch.cat(
                [piece_hiddens[index] for index in source_indices], dim=1
            ).mean(dim=1, keepdim=True))
            source_cursors.append(tuple(sorted({
                int(pieces[index][2])
                for index in source_indices
                if pieces[index][2] is not None
            })))
        if len(token_ids) != len(aligned):
            raise ValueError("Qwen token IDs and aligned source rows differ in length")
        return aligned, source_cursors

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

    @staticmethod
    def _prefix_failure_table(pattern: str) -> tuple[int, ...]:
        """Build KMP failure links once for one immutable AddedToken string."""
        failure = [0] * len(pattern)
        matched = 0
        for index in range(1, len(pattern)):
            while matched and pattern[index] != pattern[matched]:
                matched = failure[matched - 1]
            if pattern[index] == pattern[matched]:
                matched += 1
            failure[index] = matched
        return tuple(failure)

    @staticmethod
    def _failure_chain(nodes: list[dict[str, Any]], state: int):
        while state:
            yield state
            state = int(nodes[state]["failure"])

    @staticmethod
    def _build_added_token_tries(
        metadata: tuple[tuple[int, str, bool, bool, bool, bool], ...],
    ) -> dict[bool, list[dict[str, Any]]]:
        """Build bounded suffix automata grouped by AddedToken rstrip behavior."""
        tries: dict[bool, list[dict[str, Any]]] = {}
        for token_id, content, _lstrip, rstrip, _normalized, _single_word in metadata:
            nodes = tries.setdefault(
                rstrip,
                [{"next": {}, "failure": 0, "depth": 0, "token_ids": set()}],
            )
            state = 0
            for char in content:
                next_states = nodes[state]["next"]
                next_state = next_states.get(char)
                if next_state is None:
                    next_state = len(nodes)
                    next_states[char] = next_state
                    nodes.append({
                        "next": {},
                        "failure": 0,
                        "depth": int(nodes[state]["depth"]) + 1,
                        "token_ids": set(),
                    })
                state = int(next_state)
                nodes[state]["token_ids"].add(token_id)

        for nodes in tries.values():
            queue = deque()
            for child in nodes[0]["next"].values():
                queue.append(child)
            while queue:
                parent = queue.popleft()
                for char, child in nodes[parent]["next"].items():
                    fallback = int(nodes[parent]["failure"])
                    while fallback and char not in nodes[fallback]["next"]:
                        fallback = int(nodes[fallback]["failure"])
                    nodes[child]["failure"] = int(nodes[fallback]["next"].get(char, 0))
                    queue.append(child)
        return tries

    @staticmethod
    def _longest_prefix_suffix_length(
        text: str,
        pattern: str,
        failure: tuple[int, ...],
    ) -> int:
        """Find the longest AddedToken prefix matching the text suffix in O(n)."""
        matched = 0
        for char in text[-len(pattern):]:
            while matched and (
                matched == len(pattern) or pattern[matched] != char
            ):
                matched = failure[matched - 1]
            if pattern[matched] == char:
                matched += 1
        return matched

    def _added_token_suffix_work(self, text_length: int) -> int:
        """Upper bound suffix-automaton checks and whitespace flag handling."""
        length = max(0, int(text_length))
        if length == 0 or not self._added_token_metadata:
            return 0
        metadata_chars = sum(len(item[1]) for item in self._added_token_metadata)
        matcher_work = 3 * min(
            length,
            self._max_added_token_chars,
        ) * len(self._added_token_trie_by_rstrip)
        matcher_work += 3 * self._added_token_candidate_work_bound
        whitespace_work = length * (
            int(any(item[2] for item in self._added_token_metadata))
            + int(any(item[3] for item in self._added_token_metadata))
        )
        return 3 * metadata_chars + matcher_work + whitespace_work

    def _added_token_suffix_start(self, text: str) -> int | None:
        current_metadata = self._loaded_added_token_metadata()
        if current_metadata != self._added_token_metadata:
            raise RuntimeError("Qwen AddedToken metadata changed during a live stream")
        if not self._added_token_trie_by_rstrip:
            return None

        metadata_by_id = {item[0]: item for item in self._added_token_metadata}
        starts: list[int] = []
        for rstrip, nodes in self._added_token_trie_by_rstrip.items():
            searchable = text.rstrip() if rstrip else text
            if not searchable:
                continue
            max_chars = self._max_added_token_chars_by_rstrip[rstrip]
            suffix = searchable[max(0, len(searchable) - max_chars):]
            state = 0
            for char in suffix:
                next_state = nodes[state]["next"].get(char)
                while state and next_state is None:
                    state = int(nodes[state]["failure"])
                    next_state = nodes[state]["next"].get(char)
                state = int(next_state) if next_state is not None else 0
            for matched_state in self._failure_chain(nodes, state):
                token_start = len(searchable) - int(nodes[matched_state]["depth"])
                for token_id in nodes[matched_state]["token_ids"]:
                    (_id, _content, lstrip, _rstrip, _normalized, single_word) = metadata_by_id[token_id]
                    if single_word and token_start > 0:
                        previous = searchable[token_start - 1]
                        if previous == "_" or previous.isalnum():
                            continue
                    start = token_start
                    if lstrip:
                        while start > 0 and text[start - 1].isspace():
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

    def _stable_boundary(
        self,
        text: str,
        segments: list[tuple[str, tuple[int, int]]],
        *,
        previous_prefix: str | None = None,
        previous_prefix_offset: int = 0,
    ) -> int:
        """Use the current open segment, text LCP, and AddedToken candidate."""
        previous = self._last_prefix if previous_prefix is None else previous_prefix
        if (not previous and not previous_prefix_offset) or not segments:
            return 0
        boundary = int(segments[-1][1][0])
        prefix_offset = max(0, int(previous_prefix_offset))
        if prefix_offset:
            common_chars = prefix_offset + self._common_prefix_length(
                previous,
                text[prefix_offset:],
            )
        else:
            common_chars = self._common_prefix_length(previous, text)
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
        *,
        strip_leading: bool = True,
        strip_trailing: bool = True,
    ) -> tuple[str, list[list[int]], list[torch.Tensor]]:
        """NFC-normalize text while mapping output characters to source rows."""
        source_hiddens: list[torch.Tensor] = []
        source_chars: list[tuple[str, int, int]] = []
        raw_offset = 0
        for piece_index, piece_row in enumerate(pieces):
            piece, hidden = piece_row[0], piece_row[1]
            source_hiddens.append(hidden)
            source_chars.extend(
                (char, piece_index, raw_offset + char_index)
                for char_index, char in enumerate(piece)
            )
            raw_offset += len(piece)
        canonical_text, char_sources, _raw_ranges, _source_char_ends = (
            LiveQwenSayStream._canonical_source_map_records(
                source_chars,
                source_hiddens,
                strip_leading=strip_leading,
                strip_trailing=strip_trailing,
            )
        )
        return canonical_text, char_sources, source_hiddens

    @staticmethod
    def _canonical_source_map_records(
        source_chars: list[tuple[str, int, int]],
        source_hiddens: list[torch.Tensor],
        *,
        strip_leading: bool = True,
        strip_trailing: bool = True,
    ) -> tuple[
        str,
        list[list[int]],
        list[tuple[int, int]],
        dict[int, int],
    ]:
        """Normalize selected raw characters while retaining global source spans."""
        raw_text = "".join(char for char, _piece_index, _raw_index in source_chars)

        # Work in canonical decomposition space so a combining mark supplied
        # by a later Gemma piece remains attached to both source hidden rows.
        decomposed_sources: dict[str, deque[tuple[int, int]]] = defaultdict(deque)
        for char, piece_index, raw_index in source_chars:
            for component in unicodedata.normalize("NFD", char):
                decomposed_sources[component].append((piece_index, raw_index))
        nfd_text = unicodedata.normalize("NFD", raw_text)
        nfd_sources: list[tuple[int, int]] = []
        for component in nfd_text:
            owners = decomposed_sources[component]
            if not owners:
                raise ValueError("NFC source map could not recover a decomposed character owner")
            nfd_sources.append(owners.popleft())
        nfc_text = unicodedata.normalize("NFC", raw_text)
        normalized_nfc: list[tuple[str, set[int], int, int]] = []
        nfd_cursor = 0
        for char in nfc_text:
            width = len(unicodedata.normalize("NFD", char))
            owners: set[int] = set()
            raw_indices: list[int] = []
            for piece_index, raw_index in nfd_sources[nfd_cursor:nfd_cursor + width]:
                owners.add(piece_index)
                raw_indices.append(raw_index)
            if not owners:
                raise ValueError("NFC output character has no Gemma source piece")
            normalized_nfc.append((char, owners, min(raw_indices), max(raw_indices) + 1))
            nfd_cursor += width
        if nfd_cursor != len(nfd_sources):
            raise ValueError("NFC source map did not consume the complete NFD source")

        raw_text = nfc_text
        normalized: list[tuple[str, set[int], int, int]] = []
        cursor = 0
        for match in re.finditer(r"\s*<unused\d+>\s*", raw_text):
            normalized.extend(normalized_nfc[cursor:match.start()])
            replaced_sources: set[int] = set()
            replaced_raw_indices: list[tuple[int, int]] = []
            for _char, owners, raw_start, raw_end in normalized_nfc[match.start():match.end()]:
                replaced_sources.update(owners)
                replaced_raw_indices.append((raw_start, raw_end))
            normalized.append((
                " ",
                replaced_sources,
                min(item[0] for item in replaced_raw_indices),
                max(item[1] for item in replaced_raw_indices),
            ))
            cursor = match.end()
        normalized.extend(normalized_nfc[cursor:])

        canonical_chars: list[str] = []
        char_sources: list[list[int]] = []
        raw_ranges: list[tuple[int, int]] = []
        index = 0
        while index < len(normalized):
            char, piece_indices, raw_start, raw_end = normalized[index]
            if char in " \t":
                whitespace_sources: set[int] = set()
                whitespace_raw_start = raw_start
                whitespace_raw_end = raw_end
                while index < len(normalized) and normalized[index][0] in " \t":
                    whitespace_sources.update(normalized[index][1])
                    whitespace_raw_start = min(whitespace_raw_start, normalized[index][2])
                    whitespace_raw_end = max(whitespace_raw_end, normalized[index][3])
                    index += 1
                canonical_chars.append(" ")
                char_sources.append(sorted(whitespace_sources))
                raw_ranges.append((whitespace_raw_start, whitespace_raw_end))
                continue
            canonical_chars.append(char)
            char_sources.append(sorted(piece_indices))
            raw_ranges.append((raw_start, raw_end))
            index += 1

        if strip_leading:
            while canonical_chars and canonical_chars[0].isspace():
                canonical_chars.pop(0)
                char_sources.pop(0)
                raw_ranges.pop(0)
        if strip_trailing:
            while canonical_chars and canonical_chars[-1].isspace():
                canonical_chars.pop()
                char_sources.pop()
                raw_ranges.pop()
        source_char_ends: dict[int, int] = {}
        for char_index, source_indices in enumerate(char_sources):
            for source_index in source_indices:
                source_char_ends[source_index] = char_index + 1
        return "".join(canonical_chars), char_sources, raw_ranges, source_char_ends

    def _canonical_source_map_incremental(self):
        """Map only raw source text not covered by the previously stable prefix."""
        cached_text_parts = self._canonical_cached_text_parts
        cached_length = self._canonical_cached_length
        raw_start = self._canonical_cached_raw_end
        source_chars: list[tuple[str, int, int]] = []
        first_piece = bisect_right(self._source_piece_raw_ends, raw_start)
        for piece_index in range(first_piece, len(self._pieces)):
            piece, _hidden, _token_index, _cursor = self._pieces[piece_index]
            piece_start = self._source_piece_raw_starts[piece_index]
            local_start = max(0, raw_start - piece_start)
            source_chars.extend(
                (char, piece_index, piece_start + char_index)
                for char_index, char in enumerate(piece[local_start:], start=local_start)
            )
        tail_text, tail_sources, tail_ranges, tail_source_ends = (
            self._canonical_source_map_records(
                source_chars,
                self._source_hiddens,
                strip_leading=not bool(cached_length),
                strip_trailing=True,
            )
        )
        if cached_length and not self._canonical_cached_sources:
            raise RuntimeError("canonical source cache text has no ownership map")
        source_offset = cached_length
        source_char_ends = _SourceCharEndView(
            self._canonical_cached_source_char_ends,
            tail_source_ends,
            source_offset,
        )
        canonical_text = "".join((*cached_text_parts, tail_text))
        char_sources = _PrefixTailSequence(self._canonical_cached_sources, tail_sources)
        raw_ranges = _PrefixTailSequence(self._canonical_cached_raw_ranges, tail_ranges)
        if len(canonical_text) != len(char_sources) or len(canonical_text) != len(raw_ranges):
            raise ValueError("incremental NFC source-map lengths diverged")
        return (
            (canonical_text, char_sources, self._source_hiddens),
            raw_ranges,
            source_char_ends,
        )

    @staticmethod
    def _safe_canonical_cache_boundary(
        text: str,
        boundary: int,
        segments: list[tuple[str, tuple[int, int]]],
        *,
        source_offset: int = 0,
    ) -> int:
        """Choose a tokenizer boundary outside open Unicode/AddedToken tails."""
        boundary = max(0, min(len(text), int(boundary) - int(source_offset)))
        if boundary <= 0:
            return 0

        # Retain the last normalization cluster only when the candidate cut
        # is at EOS or the following code point can compose across the cut.
        needs_unicode_lookbehind = boundary == len(text)
        if boundary < len(text) and unicodedata.combining(text[boundary]) != 0:
            needs_unicode_lookbehind = True
        if boundary < len(text):
            next_name = unicodedata.name(text[boundary], "")
            previous_name = unicodedata.name(text[boundary - 1], "")
            if (
                ("HANGUL JUNGSEONG" in next_name and "HANGUL CHOSEONG" in previous_name)
                or (
                    "HANGUL JONGSEONG" in next_name
                    and ("HANGUL SYLLABLE" in previous_name or "HANGUL JUNGSEONG" in previous_name)
                )
            ):
                needs_unicode_lookbehind = True
        if needs_unicode_lookbehind:
            cluster_start = boundary - 1
            while cluster_start > 0 and unicodedata.combining(text[cluster_start]) != 0:
                cluster_start -= 1
            boundary = cluster_start

        # A replacement marker or its surrounding whitespace may still grow
        # across the source boundary; leave that full candidate in the tail.
        for marker in re.finditer(r"\s*<unused\d+>\s*", text):
            if marker.start() < boundary <= marker.end() or (
                marker.end() == boundary == len(text)
            ):
                boundary = min(boundary, marker.start())
        marker_prefix = re.search(r"<unused(?:\d*)?$", text[:boundary])
        if marker_prefix is not None:
            boundary = min(boundary, marker_prefix.start())
        marker_start = text.rfind("<", 0, boundary)
        if marker_start >= 0:
            candidate = text[marker_start:boundary]
            if (
                "<unused".startswith(candidate)
                or re.fullmatch(r"<unused\d*", candidate)
            ):
                whitespace_prefix = re.search(r"\s*$", text[:marker_start])
                marker_boundary = (
                    whitespace_prefix.start()
                    if whitespace_prefix is not None and whitespace_prefix.end() == marker_start
                    else marker_start
                )
                boundary = min(boundary, marker_boundary)

        # A collapsed whitespace run remains mutable until a non-space suffix
        # arrives. The full canonical text has trailing spaces stripped, but
        # this also protects internal boundaries at an open whitespace run.
        if boundary > 0 and text[boundary - 1].isspace():
            while boundary > 0 and text[boundary - 1].isspace():
                boundary -= 1

        valid_ends = {
            int(span[1]) - int(source_offset)
            for _segment, span in segments
            if int(span[1]) - int(source_offset) <= boundary
        }
        return max(valid_ends, default=0)

    def _promote_canonical_source_prefix(
        self,
        canonical_data,
        raw_ranges: list[tuple[int, int]],
        stable_boundary: int,
        segments: list[tuple[str, tuple[int, int]]],
        *,
        source_offset: int = 0,
    ) -> int:
        text, char_sources, _hiddens = canonical_data
        safe_boundary = self._safe_canonical_cache_boundary(
            text,
            stable_boundary,
            segments,
            source_offset=source_offset,
        )
        cached_length = self._canonical_cached_length
        if safe_boundary <= cached_length:
            return 0
        new_ranges = raw_ranges[cached_length:safe_boundary]
        if not new_ranges:
            return 0
        raw_end = max(end for _start, end in new_ranges)
        if raw_end < self._canonical_cached_raw_end or raw_end > self._source_char_count:
            raise ValueError("canonical source cache raw boundary is not monotonic")
        for char_index in range(cached_length, safe_boundary):
            for source_index in char_sources[char_index]:
                self._canonical_cached_source_char_ends[source_index] = char_index + 1
        self._canonical_cached_text_parts.append(text[cached_length:safe_boundary])
        self._canonical_cached_length = safe_boundary
        self._canonical_cached_sources.extend(char_sources[cached_length:safe_boundary])
        self._canonical_cached_raw_ranges.extend(new_ranges)
        self._canonical_cached_raw_end = raw_end
        return safe_boundary - cached_length

    def _segmented_alignment(
        self,
        pieces: list[tuple[str, torch.Tensor, int | None, int | None]],
        *,
        initial_text: str | None = None,
        canonical_data: tuple[str, list[list[int]], list[torch.Tensor]] | None = None,
    ):
        """Align an immutable first fragment plus a cumulative suffix stream."""
        if initial_text is None:
            initial_text = self._sealed_initial_text
        if not initial_text:
            raise ValueError("sealed Qwen initial segment is missing its source text")
        full_text, char_sources, piece_hiddens = (
            canonical_data
            if canonical_data is not None
            else self._canonical_source_map(pieces)
        )
        if not full_text.startswith(initial_text):
            raise ValueError("segmented Qwen text does not reconstruct the canonical Gemma prefix")

        initial_length = len(initial_text)
        suffix_text = full_text[initial_length:]

        def align_range(text: str, source_offset: int):
            if not text:
                return [], [], [], []
            token_ids, token_offsets = qwen_assistant_text_offsets(
                text,
                tokenizer=self.qwen_tokenizer,
                processor=self.qwen_processor,
            )
            global_offsets: list[tuple[int, int]] = []
            for start, end in token_offsets:
                start = int(start) + source_offset
                end = int(end) + source_offset
                global_offsets.append((start, end))
            aligned_hidden, aligned_cursors = self._align_token_offsets(
                token_ids,
                token_offsets,
                source_offset=source_offset,
                pieces=pieces,
                char_sources=char_sources,
                piece_hiddens=piece_hiddens,
            )
            return token_ids, global_offsets, aligned_hidden, aligned_cursors

        initial_ids, initial_offsets, initial_hidden, initial_cursors = align_range(
            initial_text, 0
        )
        suffix_ids, suffix_offsets, suffix_hidden, suffix_cursors = align_range(
            suffix_text, initial_length
        )
        ids = [*initial_ids, *suffix_ids]
        offsets = [*initial_offsets, *suffix_offsets]
        aligned = [*initial_hidden, *suffix_hidden]
        source_cursors = [*initial_cursors, *suffix_cursors]
        return (
            full_text,
            ids,
            offsets,
            aligned,
            source_cursors,
            char_sources,
            suffix_text,
            len(initial_ids),
        )

    def _cached_segmented_alignment(
        self,
        pieces: list[tuple[str, torch.Tensor, int | None, int | None]],
        canonical_data: tuple[str, list[list[int]], list[torch.Tensor]],
    ):
        """Reuse tokenizer-stable rows and tokenize only the mutable suffix tail."""
        full_text, char_sources, piece_hiddens = canonical_data
        initial_text = self._sealed_initial_text
        if not initial_text:
            raise ValueError("sealed Qwen initial segment is missing its source text")
        if not full_text.startswith(initial_text):
            raise ValueError("segmented Qwen text does not reconstruct the canonical Gemma prefix")
        suffix_text = full_text[len(initial_text):]
        cached_text = self._stable_suffix_text
        if not suffix_text.startswith(cached_text):
            raise ValueError("tokenizer-stable Qwen suffix cache changed under the source prefix")
        tail_text = suffix_text[len(cached_text):]

        tail_ids: list[int] = []
        tail_offsets: list[tuple[int, int]] = []
        tail_hidden: list[torch.Tensor] = []
        tail_cursors: list[tuple[int, ...]] = []
        if tail_text:
            tail_ids, local_offsets = qwen_assistant_text_offsets(
                tail_text,
                tokenizer=self.qwen_tokenizer,
                processor=self.qwen_processor,
            )
            source_offset = len(initial_text) + len(cached_text)
            tail_offsets = [
                (int(start) + source_offset, int(end) + source_offset)
                for start, end in local_offsets
            ]
            tail_hidden, tail_cursors = self._align_token_offsets(
                tail_ids,
                local_offsets,
                source_offset=source_offset,
                pieces=pieces,
                char_sources=char_sources,
                piece_hiddens=piece_hiddens,
            )

        initial_count = len(self._sealed_initial_ids)
        stable_count = len(self._stable_suffix_ids)
        ids = _ConcatSequence((
            _BoundedSequence(self._sealed_initial_ids, initial_count),
            _BoundedSequence(self._stable_suffix_ids, stable_count),
            tail_ids,
        ))
        offsets = _ConcatSequence((
            _BoundedSequence(self._sealed_initial_offsets, initial_count),
            _BoundedSequence(self._stable_suffix_offsets, stable_count),
            tail_offsets,
        ))
        aligned = _ConcatSequence((
            _BoundedSequence(self._sealed_initial_hidden, initial_count),
            _BoundedSequence(self._stable_suffix_hidden, stable_count),
            tail_hidden,
        ))
        source_cursors = _ConcatSequence((
            _BoundedSequence(self._sealed_initial_cursors, initial_count),
            _BoundedSequence(self._stable_suffix_cursors, stable_count),
            tail_cursors,
        ))
        self._current_qwen_tail = (
            tail_ids,
            tail_offsets,
            tail_hidden,
            tail_cursors,
            initial_count + stable_count,
            tail_text,
        )
        if not ids:
            raise ValueError("segmented Qwen alignment produced no text tokens")
        return (
            full_text,
            ids,
            offsets,
            aligned,
            source_cursors,
            char_sources,
            suffix_text,
            len(self._sealed_initial_ids),
        )

    def _tail_queue_size_locked(self) -> int:
        consumed_tail = max(0, self._qwen_cursor - 1)
        return max(0, len(self._published_ids) - 1 - consumed_tail)

    def _retokenization_charge(
        self,
        text_length: int,
        piece_count: int,
        *,
        tokenization_length: int | None = None,
        token_count_bound: int | None = None,
        added_scan_text_length: int | None = None,
        source_copy_length: int = 0,
    ) -> int:
        """Bound source-map, tokenizer, alignment, and AddedToken work.

        ``text_length`` and ``piece_count`` cover the cumulative NFC/source
        map and its per-piece scans. Tokenizer work is charged from the input
        actually sent to the tokenizer (or a preflight byte-level token bound),
        so a cached stable prefix is not billed as if it were retokenized.
        AddedToken matching uses precomputed KMP tables and has a linear bound.
        """
        length = max(0, int(text_length))
        pieces = max(0, int(piece_count))
        tokenizer_length = length if tokenization_length is None else max(
            0, int(tokenization_length)
        )
        if token_count_bound is None:
            max_ids = min(self.max_tokens, 4 * tokenizer_length)
        else:
            max_ids = min(self.max_tokens, max(0, int(token_count_bound)))
        added_scan_length = length if added_scan_text_length is None else max(
            0, int(added_scan_text_length)
        )
        added_token_work = self._added_token_suffix_work(added_scan_length)
        return (
            10 * length
            + 6 * max_ids
            + 4 * pieces
            + added_token_work
            + max(0, int(source_copy_length))
        )

    def _reserve_retokenization_work(
        self,
        text_length: int,
        piece_count: int,
        *,
        tokenization_length: int | None = None,
        token_count_bound: int | None = None,
        added_scan_text_length: int | None = None,
        source_copy_length: int = 0,
    ) -> None:
        charge = self._retokenization_charge(
            text_length,
            piece_count,
            tokenization_length=tokenization_length,
            token_count_bound=token_count_bound,
            added_scan_text_length=added_scan_text_length,
            source_copy_length=source_copy_length,
        )
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
        wait_for_capacity: bool = True,
        prepared: tuple[torch.Tensor, Any | None] | None = None,
    ) -> bool:
        if prepared is None:
            owned_hidden = hidden.detach().clone()
            producer_event = None
            if owned_hidden.device.type == "cuda" and torch.cuda.is_available():
                producer_event = torch.cuda.Event(enable_timing=False)
                producer_event.record(torch.cuda.current_stream(device=owned_hidden.device))
        else:
            owned_hidden, producer_event = prepared
        with self._condition:
            if wait_for_capacity:
                while not initial and self._tail_queue_size_locked() >= self.queue_capacity:
                    self._raise_terminal_locked()
                    if self._closed:
                        raise RuntimeError("live Qwen stream closed while producer was publishing")
                    self._condition.wait(0.02)
            elif not initial and self._tail_queue_size_locked() >= self.queue_capacity:
                return False
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
            for source_index in source_packet_cursors:
                packet_record = self._source_packets.get(int(source_index))
                if packet_record is not None:
                    packet_record["span_sequences"].add(int(span.token_sequence))
            source_group: dict[int, int] = {}
            source_cursors: set[int] = set()
            source_packet_indices: set[int] = set()
            for packet_index in source_packet_cursors:
                source_index = int(packet_index)
                record = self._source_packets.get(source_index)
                if record is None:
                    continue
                source_packet_indices.add(source_index)
                group = record.get("source_group") or ((source_index, record.get("token_id")),)
                for source_sequence, source_token_id in group:
                    source_group[int(source_sequence)] = int(source_token_id)
                    mapped_index = self._source_sequence_to_index.get(int(source_sequence))
                    source_cursor = (
                        self._source_packet_cursors.get(mapped_index)
                        if mapped_index is not None else None
                    )
                    if source_cursor is not None:
                        source_cursors.add(int(source_cursor))
            is_first_id = not self._published_ids
            self._published_ids.append(int(token_id))
            self._published_offsets.append((int(offset[0]), int(offset[1])))
            self._published_hidden.append(owned_hidden)
            self._published_events.append(producer_event)
            self._published_source_sequences.append(tuple(sorted(source_group)))
            self._published_source_token_ids.append(tuple(
                source_group[index] for index in sorted(source_group)
            ))
            self._published_source_cursors.append(tuple(sorted(source_cursors)))
            self._published_source_packet_indices.append(
                tuple(sorted(source_packet_indices))
            )
            if is_first_id:
                self._first_qwen_id_published_at_ns = time.monotonic_ns()
            self._condition.notify_all()
            return True

    def _wait_for_publication_capacity(
        self,
        pending: _PendingQwenPublication,
        *,
        stop_check: Any | None,
    ) -> bool:
        """Wait for a slot without holding caller-owned transaction locks."""
        while True:
            if stop_check is not None and stop_check():
                with self._condition:
                    self._deferred_publications.clear()
                    self._condition.notify_all()
                raise LiveSayCancelled("live Qwen publication lost its generation or revision")
            with self._condition:
                self._raise_terminal_locked()
                if not self._deferred_publications or self._deferred_publications[0] is not pending:
                    return False
                if self._closed:
                    raise LiveSayCancelled("live Qwen stream closed before deferred publication")
                if pending.initial or self._tail_queue_size_locked() < self.queue_capacity:
                    return True
                # Condition.wait releases this method's RLock before sleeping;
                # no session, branch, or ledger lock is acquired here.
                self._condition.wait(0.02)

    def publish_deferred(
        self,
        *,
        stop_check: Any | None = None,
        publication_guard: Any | None = None,
    ) -> None:
        """Publish staged Qwen IDs after their source ledger transaction commits.

        Capacity waits happen with only the stream's own condition. The guard
        may acquire the session, branch, and ledger locks, but receives a
        nonblocking publish callback so the revision fence is atomic.
        """
        while True:
            with self._condition:
                if not self._deferred_publications:
                    return
                pending = self._deferred_publications[0]
            if not self._wait_for_publication_capacity(pending, stop_check=stop_check):
                raise LiveSayCancelled("deferred Qwen publication was superseded")

            owned_hidden = pending.hidden.detach().clone()
            producer_event = None
            if owned_hidden.device.type == "cuda" and torch.cuda.is_available():
                producer_event = torch.cuda.Event(enable_timing=False)
                producer_event.record(torch.cuda.current_stream(device=owned_hidden.device))
            prepared = (owned_hidden, producer_event)

            def publish_if_slot_available() -> bool | None:
                with self._condition:
                    if not self._deferred_publications or self._deferred_publications[0] is not pending:
                        return False
                    published = self._publish_token(
                        pending.token_id,
                        pending.offset,
                        pending.hidden,
                        initial=pending.initial,
                        source_packet_cursors=pending.source_packet_cursors,
                        wait_for_capacity=False,
                        prepared=prepared,
                    )
                    if not published:
                        return None
                    self._deferred_publications.popleft()
                    self._condition.notify_all()
                    return True

            published = (
                publication_guard(publish_if_slot_available)
                if callable(publication_guard)
                else publish_if_slot_available()
            )
            if published is True:
                continue
            if published is None:
                # A competing nonblocking publisher used the slot after the
                # capacity check. Retry only after another unlocked wait.
                continue
            with self._condition:
                self._deferred_publications.clear()
                self._condition.notify_all()
            raise LiveSayCancelled("live Qwen publication failed its generation or revision fence")

    def feed_piece(
        self,
        text_piece: str,
        gemma_hidden: Any,
        *,
        producer_event: Any | None = None,
        token_index: int | None = None,
        source_token_id: int | None = None,
        source_token_seq: int | None = None,
        source_group: tuple[tuple[int, int], ...] | None = None,
        exclusive_kv_cursor: int | None = None,
        revision_id: str | None = None,
        kv_committed: bool = True,
        seal_initial_segment: bool = False,
        defer_publication: bool = False,
    ) -> bool:
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
            return self._feed_piece(
                text_piece,
                gemma_hidden,
                token_index=token_index,
                source_token_id=source_token_id,
                source_token_seq=source_token_seq,
                source_group=source_group,
                exclusive_kv_cursor=exclusive_kv_cursor,
                revision_id=revision_id,
                kv_committed=kv_committed,
                seal_initial_segment=seal_initial_segment,
                defer_publication=defer_publication,
            )
        except BaseException as exc:
            self.fail(exc)
            raise

    def feed_final_text_piece(
        self,
        text_piece: str,
        gemma_hidden: Any,
        *,
        token_index: int,
        source_token_id: int,
        source_token_seq: int | None = None,
        source_group: tuple[tuple[int, int], ...] | None,
        exclusive_kv_cursor: int,
        revision_id: str | None = None,
        seal_initial_segment: bool = False,
    ) -> bool:
        """Attach final decoder output to its original committed source token."""
        try:
            return self._feed_piece(
                text_piece,
                gemma_hidden,
                token_index=token_index,
                source_token_id=source_token_id,
                source_token_seq=source_token_seq,
                source_group=source_group,
                exclusive_kv_cursor=exclusive_kv_cursor,
                revision_id=revision_id,
                kv_committed=True,
                seal_initial_segment=seal_initial_segment,
                final_text_flush=True,
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
        source_token_id: int | None = None,
        source_token_seq: int | None = None,
        source_group: tuple[tuple[int, int], ...] | None = None,
        exclusive_kv_cursor: int | None = None,
        revision_id: str | None = None,
        kv_committed: bool = True,
        seal_initial_segment: bool = False,
        defer_publication: bool = False,
        final_text_flush: bool = False,
    ) -> bool:
        if not isinstance(text_piece, str):
            raise TypeError("Thinker token text deve essere string")
        if not isinstance(seal_initial_segment, bool):
            raise TypeError("seal_initial_segment must be bool")
        if not isinstance(defer_publication, bool):
            raise TypeError("defer_publication must be bool")
        hidden = self._validate_hidden(gemma_hidden)
        if not isinstance(kv_committed, bool):
            raise TypeError("kv_committed must be bool")
        with self._condition:
            if self._deferred_publications:
                raise RuntimeError(
                    "deferred Qwen publications must be drained before the next source piece"
                )
        if source_token_id is not None and (
            isinstance(source_token_id, bool)
            or not isinstance(source_token_id, int)
            or source_token_id < 0
        ):
            raise ValueError("source_token_id must be a nonnegative integer")
        if source_token_seq is not None and (
            isinstance(source_token_seq, bool)
            or not isinstance(source_token_seq, int)
            or source_token_seq < 0
        ):
            raise ValueError("source_token_seq must be a nonnegative integer")
        normalized_source_group: tuple[tuple[int, int], ...] | None = None
        if source_group is not None:
            normalized: list[tuple[int, int]] = []
            for source_sequence, source_id in source_group:
                if (
                    isinstance(source_sequence, bool)
                    or not isinstance(source_sequence, int)
                    or source_sequence < 0
                    or isinstance(source_id, bool)
                    or not isinstance(source_id, int)
                    or source_id < 0
                ):
                    raise ValueError("source_group entries must be nonnegative integer pairs")
                normalized.append((int(source_sequence), int(source_id)))
            if tuple(sorted({sequence for sequence, _token_id in normalized})) != tuple(
                sequence for sequence, _token_id in normalized
            ):
                raise ValueError("source_group sequences must be unique and ordered")
            normalized_source_group = tuple(normalized)
        if normalized_source_group is None and token_index is not None:
            normalized_source_group = ((
                int(source_token_seq if source_token_seq is not None else token_index),
                int(source_token_id if source_token_id is not None else token_index),
            ),)
        if token_index is None and exclusive_kv_cursor is not None:
            raise ValueError("an exclusive KV cursor requires its source token index")
        if (
            token_index is not None
            and exclusive_kv_cursor is None
            and (source_token_id is None or source_token_seq is None)
        ):
            raise ValueError(
                "logical-only source packets require the original token ID and source sequence"
            )
        if token_index is None and not kv_committed:
            raise ValueError("uncommitted source pieces require a token index and KV cursor")
        if final_text_flush and token_index is None:
            raise ValueError("final text flush requires its original source sequence")
        if token_index is not None and not final_text_flush:
            if (
                isinstance(token_index, bool)
                or not isinstance(token_index, int)
                or token_index < 0
                or (
                    exclusive_kv_cursor is not None
                    and (
                        isinstance(exclusive_kv_cursor, bool)
                        or not isinstance(exclusive_kv_cursor, int)
                        or exclusive_kv_cursor < 0
                    )
                )
            ):
                raise ValueError("source token index and optional KV cursor must be nonnegative integers")
            with self._condition:
                self._raise_terminal_locked()
                if self._closed:
                    raise RuntimeError("live Qwen stream is closed")
                expected_index = 0 if self._last_source_token_index is None else self._last_source_token_index + 1
                if token_index != expected_index:
                    raise RuntimeError("source Gemma token indices must be contiguous and monotonic")
                cursor = (
                    None if exclusive_kv_cursor is None else int(exclusive_kv_cursor)
                )
                if cursor is not None:
                    self._source_packet_cursors[token_index] = cursor
                    self._source_cursor_to_index.setdefault(cursor, []).append(token_index)
                for sequence, source_id in normalized_source_group or ():
                    previous_index = self._source_sequence_to_index.get(int(sequence))
                    if previous_index is not None:
                        previous_record = self._source_packets.get(previous_index)
                        previous_group = dict(
                            previous_record.get("source_group", ())
                        ) if previous_record is not None else {}
                        if previous_group.get(int(sequence)) != int(source_id):
                            raise RuntimeError("logical source sequence changed token identity")
                    else:
                        self._source_sequence_to_index[int(sequence)] = token_index
                self._source_packets[token_index] = {
                    "cursor": cursor,
                    "revision_id": str(revision_id or "r0"),
                    "hidden": hidden,
                    "source_token_seq": (
                        int(source_token_seq)
                        if source_token_seq is not None
                        else max(
                            (int(sequence) for sequence, _source_id in (normalized_source_group or ())),
                            default=int(token_index),
                        )
                    ),
                    "token_id": (
                        int(source_token_id)
                        if source_token_id is not None else int(token_index)
                    ),
                    "source_group": (
                        normalized_source_group or ()
                    ),
                    "text_piece": text_piece,
                    "final_text_flushed": False,
                    "committed": kv_committed,
                    "spans_known": False,
                    "span_sequences": set(),
                    "acked_sequences": set(),
                }
                self._source_packet_order.append(token_index)
                self._source_revision += 1
                if kv_committed:
                    self._committed_packet_cursors.add(int(token_index))
                self._last_source_token_index = token_index
        elif final_text_flush:
            if (
                token_index is None
                or source_token_id is None
                or exclusive_kv_cursor is None
            ):
                raise ValueError("final text flush requires original token ID and KV cursor")
            with self._condition:
                self._raise_terminal_locked()
                if self._closed:
                    raise RuntimeError("live Qwen stream is closed")
                record = self._source_packets.get(int(token_index))
                if record is None or int(record["cursor"]) != int(exclusive_kv_cursor):
                    raise RuntimeError("final text flush does not match its committed source packet")
                if record.get("final_text_flushed"):
                    raise RuntimeError("source packet received a duplicate final text flush")
                group_by_sequence = {
                    int(sequence): int(token_id)
                    for sequence, token_id in record.get("source_group", ())
                }
                supplied_group = normalized_source_group or (
                    (int(token_index), int(source_token_id)),
                )
                for sequence, token_id in supplied_group:
                    previous_id = group_by_sequence.get(int(sequence))
                    if previous_id is not None and previous_id != int(token_id):
                        raise RuntimeError("final text flush changed an original source token ID")
                    group_by_sequence[int(sequence)] = int(token_id)
                record["source_group"] = tuple(sorted(group_by_sequence.items()))
                record["text_piece"] = str(record["text_piece"]) + text_piece
                record["final_text_flushed"] = True
                self._source_revision += 1
        if not text_piece:
            with self._condition:
                self._raise_terminal_locked()
                if self._closed:
                    raise RuntimeError("live Qwen stream is closed")
            return False
        with self._condition:
            self._raise_terminal_locked()
            if self._closed:
                raise RuntimeError("live Qwen stream is closed")
            if self._source_char_count + len(text_piece) > self.max_source_chars:
                raise RuntimeError("live Qwen source character capacity exceeded")
            if len(self._pieces) + 1 > self.max_source_pieces:
                raise RuntimeError("live Qwen source piece capacity exceeded")
            self._source_char_count += len(text_piece)
            self._source_piece_raw_starts.append(self._source_char_count - len(text_piece))
            self._source_piece_raw_ends.append(self._source_char_count)
            self._source_hiddens.append(hidden)
            self._pieces.append((text_piece, hidden, token_index, exclusive_kv_cursor))
            if token_index is not None:
                self._source_piece_index[token_index] = len(self._pieces) - 1
            snapshot = list(self._pieces)

        # Reserve for the uncached NFC/source-map tail and the bounded copies
        # needed to present a cumulative view to the tokenizer/alignment code.
        uncached_source_chars = self._source_char_count - self._canonical_cached_raw_end
        first_uncached_piece = bisect_right(
            self._source_piece_raw_ends,
            self._canonical_cached_raw_end,
        )
        uncached_source_pieces = len(self._pieces) - first_uncached_piece
        self._reserve_retokenization_work(
            uncached_source_chars,
            uncached_source_pieces,
            tokenization_length=0,
            token_count_bound=0,
            added_scan_text_length=0,
            source_copy_length=(
                self._canonical_cached_length
                + 4 * len(self._canonical_cached_text_parts)
                + 3 * uncached_source_chars
            ),
        )
        canonical_data, canonical_raw_ranges, canonical_source_char_ends = (
            self._canonical_source_map_incremental()
        )
        canonical_text, char_sources, _piece_hiddens = canonical_data
        sealed_now = False
        segmented = self._sealed_initial_text is not None
        prepared_alignment = None
        prepared_segments = None
        prepared_stable_boundary = None
        cache_fallback = False
        if seal_initial_segment and not segmented:
            punctuation = ",.?!;:-\u2014\"'()[]"
            apostrophes = {"'", "\u2019"}
            opening_punctuation = "\"'(["
            raw_text = unicodedata.normalize("NFC", "".join(piece[0] for piece in snapshot))
            def apostrophe_is_between_letters(index: int) -> bool:
                return bool(
                    canonical_text[index] in apostrophes
                    and index > 0
                    and index + 1 < len(canonical_text)
                    and canonical_text[index - 1].isalpha()
                    and canonical_text[index + 1].isalpha()
                )

            def apostrophe_is_waiting_for_next_piece(index: int) -> bool:
                return bool(
                    canonical_text[index] in apostrophes
                    and index > 0
                    and index + 1 == len(canonical_text)
                    and canonical_text[index - 1].isalpha()
                )

            has_punctuation = any(
                char in punctuation
                and not (
                    char in apostrophes
                    and (
                        apostrophe_is_between_letters(index)
                        or apostrophe_is_waiting_for_next_piece(index)
                    )
                )
                for index, char in enumerate(canonical_text)
            )
            has_internal_space = any(
                char.isspace() or char == "\u2581" for char in canonical_text
            )
            has_trailing_space = (
                bool(raw_text.strip())
                and raw_text.rstrip() != raw_text
            ) or raw_text.endswith("\u2581")
            prior_text = unicodedata.normalize(
                "NFC", "".join(piece[0] for piece in snapshot[:-1])
            )
            token_starts_with_internal_apostrophe = bool(
                text_piece
                and text_piece[0] in apostrophes
                and prior_text
                and prior_text[-1].isalpha()
                and len(text_piece) > 1
                and text_piece[1].isalpha()
            )
            starts_with_delimiter = bool(text_piece) and (
                text_piece[0].isspace()
                or text_piece[0] == "\u2581"
                or (
                    text_piece[0] in punctuation
                    and not token_starts_with_internal_apostrophe
                    and not (
                        text_piece[0] in apostrophes
                        and canonical_text.endswith(text_piece[0])
                        and len(canonical_text) > 1
                        and canonical_text[-2].isalpha()
                    )
                )
            )
            should_seal = (
                has_punctuation
                or has_internal_space
                or has_trailing_space
                or (token_index is not None and token_index >= 1 and starts_with_delimiter)
            )
            if should_seal and canonical_text:
                boundary = len(canonical_text)
                has_word_character = False
                for index, char in enumerate(canonical_text):
                    if char.isspace() or char == "\u2581":
                        if has_word_character:
                            boundary = index
                            break
                        continue
                    if char in punctuation:
                        if (
                            char in apostrophes
                            and (
                                apostrophe_is_between_letters(index)
                                or apostrophe_is_waiting_for_next_piece(index)
                            )
                        ):
                            has_word_character = True
                            continue
                        if has_word_character:
                            boundary = index + 1
                            break
                        if char in opening_punctuation:
                            continue
                        continue
                    has_word_character = True
                # Keep the first segment only when the loaded pre-tokenizer
                # shows that it cannot merge with the currently open tail.
                seal_context_text = canonical_text + (" " if has_trailing_space else "")
                self._reserve_retokenization_work(
                    0,
                    0,
                    tokenization_length=0,
                    token_count_bound=0,
                    added_scan_text_length=len(seal_context_text),
                )
                initial_segments = self._pretokenizer_segments(seal_context_text)
                tokenizer_stable_boundary = (
                    int(initial_segments[-1][1][0]) if initial_segments else 0
                )
                added_candidate_start = self._added_token_suffix_start(seal_context_text)
                if added_candidate_start is not None:
                    tokenizer_stable_boundary = min(
                        tokenizer_stable_boundary,
                        added_candidate_start,
                    )
                boundary = min(boundary, tokenizer_stable_boundary)
                segment_starts = {0}
                segment_starts.update(
                    int(span[0]) for _segment, span in initial_segments
                )
                boundary = max(
                    (start for start in segment_starts if start <= boundary),
                    default=0,
                )
                canonical_word_prefix = canonical_text[:boundary].rstrip()
                if canonical_word_prefix and has_word_character:
                    try:
                        suffix_candidate = canonical_text[len(canonical_word_prefix):]
                        self._reserve_retokenization_work(
                            0,
                            0,
                            tokenization_length=len(canonical_text),
                            token_count_bound=min(
                                self.max_tokens,
                                4 * len(canonical_text),
                            ),
                            added_scan_text_length=len(suffix_candidate),
                        )
                        prepared_alignment = self._segmented_alignment(
                            snapshot,
                            initial_text=canonical_word_prefix,
                            canonical_data=canonical_data,
                        )
                    except (TypeError, ValueError, IndexError):
                        # Preserve cumulative carry if the prefix cannot be
                        # represented by the loaded Qwen tokenizer.
                        prepared_alignment = None
                    if prepared_alignment is not None:
                        self._sealed_initial_text = canonical_word_prefix
                        self._sealed_initial_id_count = int(prepared_alignment[7])
                        self._sealed_initial_ids = list(
                            prepared_alignment[1][:self._sealed_initial_id_count]
                        )
                        self._sealed_initial_offsets = list(
                            prepared_alignment[2][:self._sealed_initial_id_count]
                        )
                        self._sealed_initial_hidden = list(
                            prepared_alignment[3][:self._sealed_initial_id_count]
                        )
                        self._sealed_initial_cursors = list(
                            prepared_alignment[4][:self._sealed_initial_id_count]
                        )
                        suffix_candidate = prepared_alignment[6]
                        prepared_segments = (
                            self._pretokenizer_segments(suffix_candidate)
                            if suffix_candidate else []
                        )
                        prepared_stable_boundary = self._stable_boundary(
                            suffix_candidate,
                            prepared_segments,
                            previous_prefix=self._last_qwen_suffix_prefix,
                        )
                        segmented = True
                        sealed_now = True

        prefix = canonical_text
        if not prefix:
            return sealed_now
        self._current_qwen_tail = None
        if prepared_alignment is not None:
            (
                prefix,
                ids,
                offsets,
                aligned,
                source_packet_cursors,
                char_sources,
                suffix_text,
                initial_id_count,
            ) = prepared_alignment
        elif segmented:
            suffix_text = prefix[len(self._sealed_initial_text or ""):]
            cached_text = self._stable_suffix_text
            cache_fallback = not suffix_text.startswith(cached_text)
            tail_text = suffix_text if cache_fallback else suffix_text[len(cached_text):]
            initial_text_length = len(self._sealed_initial_text or "")
            self._reserve_retokenization_work(
                0,
                0,
                tokenization_length=len(tail_text) + (initial_text_length if cache_fallback else 0),
                token_count_bound=min(
                    self.max_tokens,
                    4 * (len(tail_text) + (initial_text_length if cache_fallback else 0)),
                ),
                added_scan_text_length=len(suffix_text),
            )
            segments = self._pretokenizer_segments(suffix_text) if suffix_text else []
            if cache_fallback:
                stable_boundary = self._stable_boundary(
                    suffix_text,
                    segments,
                    previous_prefix=suffix_text,
                )
                (
                    prefix,
                    ids,
                    offsets,
                    aligned,
                    source_packet_cursors,
                    char_sources,
                    suffix_text,
                    initial_id_count,
                ) = self._segmented_alignment(
                    snapshot,
                    canonical_data=canonical_data,
                )
                initial_end = initial_id_count
                if (
                    list(ids[:initial_end]) != self._sealed_initial_ids
                    or list(offsets[:initial_end]) != self._sealed_initial_offsets
                    or any(
                        not torch.equal(old, current)
                        for old, current in zip(
                            self._sealed_initial_hidden,
                            aligned[:initial_end],
                        )
                    )
                    or list(source_packet_cursors[:initial_end]) != self._sealed_initial_cursors
                ):
                    raise ValueError("full suffix fallback changed the sealed initial Qwen alignment")
            else:
                stable_boundary = self._stable_boundary(
                    suffix_text,
                    segments,
                    previous_prefix=self._last_qwen_mutable_tail_text,
                    previous_prefix_offset=len(cached_text),
                )
                (
                    prefix,
                    ids,
                    offsets,
                    aligned,
                    source_packet_cursors,
                    char_sources,
                    suffix_text,
                    initial_id_count,
                ) = self._cached_segmented_alignment(snapshot, canonical_data)
        else:
            self._reserve_retokenization_work(
                0,
                0,
                tokenization_length=len(prefix),
                token_count_bound=min(self.max_tokens, 4 * len(prefix)),
                added_scan_text_length=len(prefix),
            )
            ids, offsets, aligned, source_packet_cursors, char_sources = self._current_alignment(
                prefix, snapshot, canonical_data=canonical_data
            )
            suffix_text = prefix
            initial_id_count = 0
            segments = self._pretokenizer_segments(prefix)
            stable_boundary = self._stable_boundary(
                prefix,
                segments,
                previous_prefix=self._last_prefix,
            )
        if prepared_alignment is not None:
            segments = prepared_segments
            stable_boundary = prepared_stable_boundary
            if segments is None or stable_boundary is None:
                raise RuntimeError("prepared initial Qwen alignment is missing its stability boundary")
        stability_text = suffix_text if segmented else prefix
        stable_boundary = self._safe_canonical_cache_boundary(
            stability_text,
            stable_boundary,
            segments,
        )
        if len(ids) > self.max_tokens or len(aligned) > self.max_tokens:
            raise RuntimeError("live Qwen observed token/hidden row capacity exceeded")
        if len(ids) != len(offsets) or len(ids) != len(aligned):
            raise ValueError("live Qwen observed IDs, offsets, and hidden rows are not aligned")
        with self._condition:
            self._raise_terminal_locked()
            if not segmented or prepared_alignment is not None or cache_fallback:
                self._assert_published_immutable(
                    self._published_ids,
                    self._published_offsets,
                    self._published_hidden,
                    ids,
                    offsets,
                    aligned,
                )
            elif not cache_fallback and len(self._published_ids) > (
                len(self._sealed_initial_ids) + len(self._stable_suffix_ids)
            ):
                raise RuntimeError("published Qwen token escaped the immutable alignment cache")
            if cache_fallback:
                self._stable_suffix_text = ""
                self._stable_suffix_ids.clear()
                self._stable_suffix_offsets.clear()
                self._stable_suffix_hidden.clear()
                self._stable_suffix_cursors.clear()
                self._last_qwen_mutable_tail_ids = None
                self._last_qwen_mutable_tail_offsets = None
                self._last_qwen_mutable_tail_hidden = None
                self._last_qwen_mutable_tail_cursors = None
                self._last_qwen_mutable_tail_base = None
                self._last_qwen_mutable_tail_text = ""
                self._current_qwen_tail = None
            stable_global_boundary = (
                stable_boundary + len(self._sealed_initial_text or "")
                if segmented else stable_boundary
            )
            stable_suffix_count = 0
            stable_cache_base = (
                initial_id_count + len(self._stable_suffix_ids) if segmented else 0
            )
            suffix_offset = len(self._sealed_initial_text or "") if segmented else 0
            for token_index in range(stable_cache_base, len(ids)):
                token_id = ids[token_index]
                end = int(offsets[token_index][1]) - suffix_offset
                if int(end) > stable_boundary:
                    break
                # Equality means this AddedToken touches the open boundary.
                # Wait for context to move the boundary beyond it; EOS may
                # release it only after the full prefix is validated.
                if int(end) == stable_boundary and int(token_id) in self._added_token_ids:
                    break
                stable_suffix_count += 1
            stable_count = stable_cache_base + stable_suffix_count
            current_tail = self._current_qwen_tail
            if segmented and self._last_qwen_mutable_tail_ids is not None and current_tail is not None:
                current_base = int(current_tail[4])
                if self._last_qwen_mutable_tail_base != current_base:
                    raise ValueError("Qwen mutable alignment tail moved across its stable cache")
                lcp = 0
                old_tail_ids = self._last_qwen_mutable_tail_ids
                old_tail_offsets = self._last_qwen_mutable_tail_offsets or []
                current_tail_ids = current_tail[0]
                current_tail_offsets = current_tail[1]
                for tail_index in range(min(len(old_tail_ids), len(current_tail_ids))):
                    if (
                        old_tail_ids[tail_index] != current_tail_ids[tail_index]
                        or old_tail_offsets[tail_index] != current_tail_offsets[tail_index]
                    ):
                        break
                    lcp += 1
                stable_count = min(stable_count, current_base + lcp)
            elif (
                self._last_qwen_ids is not None
                and self._last_qwen_offsets is not None
                and not cache_fallback
                and not (segmented and prepared_alignment is not None)
            ):
                # One-time transition into segmented alignment; afterwards
                # only the mutable tail needs differential comparison.
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
            if stable_count < len(self._published_ids):
                raise ValueError("stable Qwen prefix retracted an already published token")
            if segmented:
                cached_count = len(self._stable_suffix_ids)
                stable_cache_count = stable_count - initial_id_count
                if stable_cache_count < cached_count:
                    raise ValueError("tokenizer-stable Qwen cache crossed its committed boundary")
                newly_stable = stable_cache_count - cached_count
                if newly_stable:
                    cache_start = initial_id_count + cached_count
                    cache_end_index = cache_start + newly_stable
                    suffix_offset = len(self._sealed_initial_text or "")
                    if stable_boundary < len(self._stable_suffix_text):
                        raise ValueError("Qwen stable boundary crossed its committed text cache")
                    if not suffix_text.startswith(self._stable_suffix_text):
                        raise ValueError("tokenizer-stable Qwen suffix cache changed under the source prefix")
                    self._stable_suffix_text = suffix_text[:stable_boundary]
                    self._stable_suffix_ids.extend(ids[cache_start:cache_end_index])
                    self._stable_suffix_offsets.extend(offsets[cache_start:cache_end_index])
                    self._stable_suffix_hidden.extend(aligned[cache_start:cache_end_index])
                    self._stable_suffix_cursors.extend(
                        source_packet_cursors[cache_start:cache_end_index]
                    )
            if segmented:
                source_start = 0 if prepared_alignment is not None else stable_cache_base
                stable_cursor_rows = source_packet_cursors[source_start:stable_count]
                latest_stable_source_index = self._latest_stable_source_index
            else:
                stable_cursor_rows = source_packet_cursors[:stable_count]
                latest_stable_source_index = -1
            for token_sources in stable_cursor_rows:
                for cursor in token_sources:
                    latest_stable_source_index = max(
                        latest_stable_source_index,
                        int(cursor),
                    )
            self._latest_stable_source_index = max(
                self._latest_stable_source_index,
                latest_stable_source_index,
            )
            latest_stable_source_index = self._latest_stable_source_index
            for token_index, record in self._source_packets.items():
                if record.get("spans_known"):
                    continue
                piece_index = self._source_piece_index.get(token_index)
                if piece_index is not None and piece_index in canonical_source_char_ends:
                    if canonical_source_char_ends[piece_index] <= stable_global_boundary:
                        record["spans_known"] = True
                elif token_index < latest_stable_source_index:
                    # Canonicalized-away leading text has no Qwen span once a
                    # later stable token has passed it.
                    record["spans_known"] = True
            self._promote_canonical_source_prefix(
                canonical_data,
                canonical_raw_ranges,
                stable_global_boundary,
                segments,
                source_offset=(len(self._sealed_initial_text or "") if segmented else 0),
            )
            if segmented:
                tail_base = len(self._sealed_initial_ids) + len(self._stable_suffix_ids)
                self._last_qwen_mutable_tail_ids = list(ids[tail_base:])
                self._last_qwen_mutable_tail_offsets = list(offsets[tail_base:])
                self._last_qwen_mutable_tail_hidden = list(aligned[tail_base:])
                self._last_qwen_mutable_tail_cursors = list(source_packet_cursors[tail_base:])
                self._last_qwen_mutable_tail_base = tail_base
                self._last_qwen_mutable_tail_text = suffix_text[len(self._stable_suffix_text):]
                self._last_qwen_ids = _ConcatSequence((
                    _BoundedSequence(self._sealed_initial_ids),
                    _BoundedSequence(self._stable_suffix_ids),
                    self._last_qwen_mutable_tail_ids,
                ))
                self._last_qwen_offsets = _ConcatSequence((
                    _BoundedSequence(self._sealed_initial_offsets),
                    _BoundedSequence(self._stable_suffix_offsets),
                    self._last_qwen_mutable_tail_offsets,
                ))
                self._last_qwen_hidden = _ConcatSequence((
                    _BoundedSequence(self._sealed_initial_hidden),
                    _BoundedSequence(self._stable_suffix_hidden),
                    self._last_qwen_mutable_tail_hidden,
                ))
                self._last_qwen_cursors = _ConcatSequence((
                    _BoundedSequence(self._sealed_initial_cursors),
                    _BoundedSequence(self._stable_suffix_cursors),
                    self._last_qwen_mutable_tail_cursors,
                ))
            else:
                self._last_qwen_ids = list(ids)
                self._last_qwen_offsets = list(offsets)
                self._last_qwen_hidden = list(aligned)
                self._last_qwen_cursors = list(source_packet_cursors)
                self._last_qwen_mutable_tail_ids = None
                self._last_qwen_mutable_tail_offsets = None
                self._last_qwen_mutable_tail_hidden = None
                self._last_qwen_mutable_tail_cursors = None
                self._last_qwen_mutable_tail_base = None
                self._last_qwen_mutable_tail_text = ""
            self._last_prefix = prefix
            self._last_qwen_suffix_prefix = suffix_text if segmented else prefix
            next_index = len(self._published_ids)
            pending = [
                (
                    ids[index], offsets[index], aligned[index], index == 0,
                    source_packet_cursors[index],
                )
                for index in range(next_index, stable_count)
            ]
        if defer_publication:
            with self._condition:
                self._deferred_publications.extend(
                    _PendingQwenPublication(
                        int(token_id),
                        (int(offset[0]), int(offset[1])),
                        token_hidden,
                        bool(initial),
                        tuple(source_cursors),
                    )
                    for token_id, offset, token_hidden, initial, source_cursors in pending
                )
        else:
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
        return sealed_now

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
        self._reserve_retokenization_work(
            self._source_char_count,
            len(snapshot),
            tokenization_length=0,
            token_count_bound=0,
            added_scan_text_length=0,
        )
        canonical_data = self._canonical_source_map(snapshot)
        canonical_text, _char_sources, _source_hiddens = canonical_data
        cached_length = self._canonical_cached_length
        source_cache_matches = len(_char_sources) >= cached_length
        text_cursor = 0
        for cached_part in self._canonical_cached_text_parts:
            end = text_cursor + len(cached_part)
            source_cache_matches = source_cache_matches and (
                canonical_text[text_cursor:end] == cached_part
            )
            text_cursor = end
        if not source_cache_matches or _char_sources[:cached_length] != self._canonical_cached_sources:
            raise ValueError("full EOS source map drifted from the tokenizer-stable source cache")
        final_text = unicodedata.normalize("NFC", final_text)
        if canonical_text != final_text:
            raise ValueError("Thinker callback text non corrisponde al testo finale canonico")
        self._reserve_retokenization_work(
            0,
            0,
            tokenization_length=len(canonical_text),
            token_count_bound=min(self.max_tokens, 4 * len(canonical_text)),
            added_scan_text_length=0,
        )
        if self._sealed_initial_text is not None:
            (
                segmented_text,
                ids,
                offsets,
                aligned,
                source_packet_cursors,
                _char_sources,
                suffix_text,
                _initial_id_count,
            ) = self._segmented_alignment(
                snapshot,
                canonical_data=canonical_data,
            )
            if segmented_text != canonical_text:
                raise ValueError("segmented Qwen text differs from the final canonical Gemma text")
        else:
            ids, offsets, aligned, source_packet_cursors, _char_sources = self._current_alignment(
                canonical_text,
                snapshot,
                canonical_data=canonical_data,
            )
            suffix_text = canonical_text
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
            if (
                self._last_qwen_cursors is not None
                and self._last_qwen_cursors != source_packet_cursors
            ):
                raise ValueError("live Qwen source-cursor alignment drifted at EOS")
            if self._last_qwen_ids is not None and self._last_qwen_suffix_prefix != suffix_text:
                raise ValueError("live Qwen cumulative suffix text drifted at EOS")
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
            self._last_qwen_suffix_prefix = suffix_text
            for record in self._source_packets.values():
                record["spans_known"] = True
            self._eos_enqueued = True
            self._closed = True
            self._source_revision += 1
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
            if self._reserved_consumption is not None:
                raise RuntimeError("Qwen text input is already reserved")
            consumption = self._make_consumption_locked(0, q1_index=None)
            self._reserved_consumption = consumption
            self._reserved_qwen_index = 0
            self._reserved_initial = True
            token = LiveQwenToken(
                self._published_ids[0], self._published_hidden[0], consumption
            )
        self._notify_tts_reserved(consumption)
        return token

    def _notify_tts_reserved(self, consumption: QwenTextConsumption) -> None:
        callback = self.on_tts_text_reserved
        if callable(callback):
            callback(consumption)

    @property
    def reserved_consumption(self) -> QwenTextConsumption | None:
        reserved: QwenTextConsumption | None = None
        terminal_eos = False
        with self._condition:
            return self._reserved_consumption

    @property
    def eos_consumed(self) -> bool:
        with self._condition:
            return bool(self._eos_consumed)

    def _make_consumption_locked(
        self,
        index: int,
        *,
        q1_index: int | None,
        eos: bool = False,
        pad: bool = False,
        tail_hold: bool = False,
        qwen_token_id: int | None = None,
    ) -> QwenTextConsumption:
        if index < 0 or index >= len(self._published_ids):
            raise RuntimeError("Qwen consumption references an unpublished token")
        return QwenTextConsumption(
            qwen_token_id=(
                int(qwen_token_id)
                if qwen_token_id is not None else
                None if pad else int(self._published_ids[index])
            ),
            source_sequences=self._published_source_sequences[index],
            source_token_ids=self._published_source_token_ids[index],
            source_kv_cursors=self._published_source_cursors[index],
            hidden=self._published_hidden[index],
            producer_event=self._published_events[index],
            q1_index=q1_index,
            eos=bool(eos),
            pad=bool(pad),
            tail_hold=bool(tail_hold),
        )

    def _source_frontier_locked(self) -> QwenSourceFrontier:
        sequences: dict[int, int] = {}
        cursors: set[int] = set()
        covered_indices = list(range(min(self._qwen_cursor, len(self._published_ids))))
        if (
            self._reserved_consumption is not None
            and self._reserved_qwen_index is not None
            and not self._reserved_consumption.eos
            and not self._reserved_consumption.pad
        ):
            covered_indices.append(self._reserved_qwen_index)
        for index in covered_indices:
            for sequence, token_id in zip(
                self._published_source_sequences[index],
                self._published_source_token_ids[index],
            ):
                sequences[int(sequence)] = int(token_id)
            cursors.update(self._published_source_cursors[index])
        public_ids = self._source_token_ids_for_sequences_locked(
            self._already_public_source_sequences
        )
        for sequence, token_id in zip(self._already_public_source_sequences, public_ids):
            sequences[int(sequence)] = int(token_id)
            packet_index = self._source_sequence_to_index.get(int(sequence))
            cursor = (
                self._source_packet_cursors.get(packet_index)
                if packet_index is not None else None
            )
            if cursor is not None:
                cursors.add(int(cursor))
        frozen_sequence = max(sequences, default=-1)
        char_end = max(
            (int(self._published_offsets[index][1]) for index in covered_indices),
            default=0,
        )
        return QwenSourceFrontier(
            generation_id=self.generation_id,
            segment_id=self.segment_id,
            revision=self._source_revision,
            qwen_consumed_count=int(self._qwen_cursor),
            reserved_qwen_token_id=(
                int(self._reserved_consumption.qwen_token_id)
                if self._reserved_consumption is not None
                and self._reserved_consumption.qwen_token_id is not None
                else None
            ),
            frozen_source_sequence=int(frozen_sequence),
            source_sequences=tuple(sorted(sequences)),
            source_token_ids=tuple(sequences[index] for index in sorted(sequences)),
            source_kv_cursors=tuple(sorted(cursors)),
            already_public_text=self._already_public_text,
            cumulative_source_text=self._last_prefix[:char_end],
            cache_epoch=self.cache_epoch,
        )

    @staticmethod
    def _same_frozen_source_frontier(
        left: QwenSourceFrontier,
        right: QwenSourceFrontier,
    ) -> bool:
        """Compare only immutable source identity, excluding mutable-tail revision."""
        return (
            left.generation_id == right.generation_id
            and left.segment_id == right.segment_id
            and left.qwen_consumed_count == right.qwen_consumed_count
            and left.reserved_qwen_token_id == right.reserved_qwen_token_id
            and left.frozen_source_sequence == right.frozen_source_sequence
            and left.source_sequences == right.source_sequences
            and left.source_token_ids == right.source_token_ids
            and left.source_kv_cursors == right.source_kv_cursors
            and left.already_public_text == right.already_public_text
            and left.cumulative_source_text == right.cumulative_source_text
            and left.cache_epoch == right.cache_epoch
        )

    def _source_token_ids_for_sequences_locked(
        self,
        source_sequences: tuple[int, ...],
    ) -> tuple[int, ...]:
        result: dict[int, int] = {}
        for record in self._source_packets.values():
            for sequence, token_id in record.get("source_group", ()):
                if int(sequence) in source_sequences:
                    result[int(sequence)] = int(token_id)
        return tuple(result[index] for index in source_sequences if index in result)

    def source_frontier_snapshot(self) -> QwenSourceFrontier:
        """Return a lock-consistent text frontier with physical cursors as provenance."""
        with self._condition:
            self._raise_terminal_locked()
            return self._source_frontier_locked()

    def prepare_unconsumed_suffix_locked(
        self,
        replacement_tokens: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        *,
        expected_frontier: QwenSourceFrontier,
        frozen_prefix_token_ids: tuple[int, ...] | list[int],
        revision_id: str,
        cache_epoch: str | int | None,
    ) -> LiveQwenSuffixReplacementPlan:
        """Build a complete replacement queue without changing this stream.

        The caller must hold ``_condition`` and the session generation lock.
        Logical source IDs in ``replacement_tokens`` are immutable ledger IDs;
        source packet and Qwen queue indices in the returned plan are rebuilt
        independently. Only the consumed and currently reserved Qwen prefix
        is retained from the old physical queue.
        """
        owns_lock = getattr(self._condition, "_is_owned", None)
        if callable(owns_lock) and not owns_lock():
            raise RuntimeError("stream replacement prepare requires its condition lock")
        if not isinstance(expected_frontier, QwenSourceFrontier):
            raise TypeError("expected_frontier must be a QwenSourceFrontier")
        if not isinstance(revision_id, str) or not revision_id.strip():
            raise ValueError("revision_id must be non-empty")
        if cache_epoch is not None and (
            isinstance(cache_epoch, bool)
            or not isinstance(cache_epoch, (str, int))
            or (isinstance(cache_epoch, str) and not cache_epoch)
        ):
            raise ValueError("cache_epoch must be a non-empty string, integer, or None")
        self._raise_terminal_locked()
        if self._closed or self._eos_enqueued or self._eos_consumed:
            raise RuntimeError("closed or EOS-consumed Qwen stream cannot accept a revision")
        actual_frontier = self._source_frontier_locked()
        if not self._same_frozen_source_frontier(actual_frontier, expected_frontier):
            raise RuntimeError("live Qwen source frontier changed before revision prepare")
        if expected_frontier.cache_epoch != self.cache_epoch:
            raise RuntimeError("live Qwen cache epoch changed before revision prepare")
        prefix_ids = tuple(int(value) for value in frozen_prefix_token_ids)
        if any(isinstance(value, bool) or value < 0 for value in frozen_prefix_token_ids):
            raise ValueError("frozen prefix contains an invalid Gemma token ID")

        prefix_rows: list[dict[str, Any]] = []
        actual_prefix_ids: list[int] = []
        for packet_index in self._source_packet_order:
            record = self._source_packets[packet_index]
            source_sequence = record.get("source_token_seq")
            if (
                isinstance(source_sequence, bool)
                or not isinstance(source_sequence, int)
                or int(source_sequence) > int(expected_frontier.frozen_source_sequence)
            ):
                continue
            token_id = int(record["token_id"])
            text_piece = str(record["text_piece"])
            hidden = record.get("hidden")
            if hidden is None:
                piece_index = self._source_piece_index.get(int(packet_index))
                if piece_index is None:
                    raise RuntimeError("frozen Qwen source row has no preserved Gemma hidden")
                hidden = self._source_hiddens[piece_index]
            physical_cursor = int(self._source_packet_cursors[int(packet_index)])
            source_group = tuple(
                (int(sequence), int(source_id))
                for sequence, source_id in record.get("source_group", ())
            )
            if not source_group:
                source_group = ((int(source_sequence), token_id),)
            prefix_rows.append({
                "token_id": token_id,
                "text_piece": text_piece,
                "hidden": hidden,
                "source_token_seq": int(source_sequence),
                "source_group": source_group,
                "exclusive_kv_cursor": physical_cursor,
                "revision_id": str(record.get("revision_id", "r0")),
                "kv_committed": bool(record.get("committed", True)),
            })
            actual_prefix_ids.append(token_id)
        if tuple(actual_prefix_ids) != prefix_ids:
            raise RuntimeError("frozen Gemma prefix differs from live source packet identity")

        normalized_replacements: list[dict[str, Any]] = []
        previous_sequence = int(expected_frontier.frozen_source_sequence)
        for row in replacement_tokens:
            if not isinstance(row, dict):
                raise TypeError("replacement source rows must be mappings")
            token_id = row.get("token_id")
            text_piece = row.get("text_piece", "")
            source_sequence = row.get("source_token_seq")
            hidden = row.get("hidden")
            physical_cursor = row.get("exclusive_kv_cursor", row.get("physical_kv_cursor"))
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ValueError("replacement token ID must be a nonnegative integer")
            if not isinstance(text_piece, str):
                raise TypeError("replacement text_piece must be str")
            if isinstance(source_sequence, bool) or not isinstance(source_sequence, int):
                raise ValueError("replacement token must carry an assigned logical source sequence")
            if int(source_sequence) <= previous_sequence:
                raise ValueError("replacement logical source sequences must increase after the frozen prefix")
            if isinstance(physical_cursor, bool) or not isinstance(physical_cursor, int) or physical_cursor < 0:
                raise ValueError("replacement token must carry a physical KV cursor")
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("replacement token must carry a post-forward Gemma hidden")
            source_group_value = row.get("source_group")
            source_group = tuple(
                (int(sequence), int(source_id))
                for sequence, source_id in source_group_value
            ) if source_group_value is not None else ()
            if not source_group:
                source_group = ((int(source_sequence), int(token_id)),)
            normalized_replacements.append({
                "token_id": int(token_id),
                "text_piece": text_piece,
                "hidden": hidden,
                "source_token_seq": int(source_sequence),
                "source_group": source_group,
                "exclusive_kv_cursor": int(physical_cursor),
                "revision_id": str(row.get("revision_id") or revision_id),
                "kv_committed": True,
            })
            previous_sequence = int(source_sequence)

        preserved_qwen_count = int(self._qwen_cursor) + int(
            self._reserved_consumption is not None
            and self._reserved_qwen_index == self._qwen_cursor
            and not self._reserved_consumption.eos
            and not self._reserved_consumption.pad
        )
        if preserved_qwen_count > len(self._published_ids):
            raise RuntimeError("Qwen reservation exceeds the published source queue")
        if len(self.span_cursor._spans) < preserved_qwen_count:
            raise RuntimeError("Qwen span cursor lost a consumed or reserved text token")
        for span in self.span_cursor._spans[preserved_qwen_count:]:
            if int(span.start_frame) <= int(self._last_received_emitted_frame):
                raise RuntimeError("Qwen suffix already emitted PCM and cannot be revised")
        if (
            preserved_qwen_count < len(self.span_cursor._spans)
            and int(self.span_cursor._last_selected_frame)
            >= int(self.span_cursor._spans[preserved_qwen_count].start_frame)
        ):
            raise RuntimeError("Qwen suffix already selected by the Talker and cannot be revised")

        staging = LiveQwenSayStream(
            qwen_tokenizer=self.qwen_tokenizer,
            qwen_processor=self.qwen_processor,
            text_eos_token_id=self.text_eos_token_id,
            max_tokens=self.max_tokens,
            queue_capacity=self.max_tokens,
            max_source_chars=self.max_source_chars,
            max_source_pieces=self.max_source_pieces,
            max_retokenization_work=self.max_retokenization_work,
            generation_id=self.generation_id,
            segment_id=self.segment_id,
            cache_epoch=cache_epoch,
            on_packet_emitted=None,
            on_tts_text_consumed=None,
        )
        all_rows = [*prefix_rows, *normalized_replacements]
        for packet_index, row in enumerate(all_rows):
            hidden = row["hidden"]
            if hidden.ndim == 2:
                hidden = hidden.unsqueeze(1)
            if hidden.ndim != 3 or tuple(hidden.shape[:2]) != (1, 1):
                raise ValueError("replacement Gemma hidden must have shape (1,1,D)")
            staging.feed_piece(
                str(row["text_piece"]),
                hidden,
                token_index=int(packet_index),
                source_token_id=int(row["token_id"]),
                source_token_seq=int(row["source_token_seq"]),
                source_group=tuple(row["source_group"]),
                exclusive_kv_cursor=int(row["exclusive_kv_cursor"]),
                revision_id=str(row["revision_id"]),
                kv_committed=True,
                seal_initial_segment=True,
            )
        if len(staging._published_ids) < preserved_qwen_count:
            raise RuntimeError("candidate retokenization retracts a consumed or reserved Qwen token")
        for index in range(preserved_qwen_count):
            if (
                staging._published_ids[index] != self._published_ids[index]
                or staging._published_offsets[index] != self._published_offsets[index]
                or staging._published_source_sequences[index] != self._published_source_sequences[index]
                or staging._published_source_token_ids[index] != self._published_source_token_ids[index]
                or not torch.equal(staging._published_hidden[index], self._published_hidden[index])
            ):
                raise RuntimeError("candidate retokenization changed a consumed or reserved Qwen token")

        for packet_index in range(len(prefix_rows)):
            old_record = self._source_packets.get(packet_index)
            new_record = staging._source_packets.get(packet_index)
            if old_record is not None and new_record is not None:
                new_record["span_sequences"] = set(old_record.get("span_sequences", ()))
                new_record["acked_sequences"] = set(old_record.get("acked_sequences", ()))
                new_record["spans_known"] = bool(old_record.get("spans_known", False))
        staging._next_source_packet_to_emit = int(self._next_source_packet_to_emit)
        staging._last_source_token_index = (
            len(all_rows) - 1 if all_rows else None
        )
        new_span_cursor = SpanAlignedSpeechCursor(
            generation_id=self.generation_id,
            segment_id=self.segment_id,
        )
        kept_spans = list(self.span_cursor._spans[:preserved_qwen_count])
        new_span_cursor._spans = kept_spans
        new_span_cursor._frame_count = max(
            (int(span.end_frame) for span in kept_spans), default=0
        )
        new_span_cursor._next_sequence = max(
            int(self.span_cursor._next_sequence),
            max((int(span.token_sequence) + 1 for span in kept_spans), default=0),
        )
        new_span_cursor._last_selected_frame = int(self.span_cursor._last_selected_frame)
        new_span_cursor._last_emitted_frame = int(self.span_cursor._last_emitted_frame)
        kept_span_ids = {id(span) for span in kept_spans}
        new_span_cursor._selections = {
            frame_id: selection
            for frame_id, selection in self.span_cursor._selections.items()
            if id(selection.span) in kept_span_ids
        }
        new_span_cursor._closed = False
        replacement_span_map = {
            int(span.start_frame): tuple(
                self._span_packet_cursors.get(
                    int(span.start_frame),
                    staging._span_packet_cursors.get(int(span.start_frame), ()),
                )
            )
            for span in kept_spans
        }
        for qwen_index in range(preserved_qwen_count, len(staging._published_ids)):
            staged_span = staging.span_cursor._spans[qwen_index]
            span = new_span_cursor.append(
                token_id=int(staging._published_ids[qwen_index]),
                token_text=str(staged_span.token_text),
                hidden=staging._published_hidden[qwen_index],
                producer_event=staging._published_events[qwen_index],
            )
            packet_indices = tuple(staging._published_source_packet_indices[qwen_index])
            replacement_span_map[int(span.start_frame)] = packet_indices
            for packet_index in packet_indices:
                record = staging._source_packets.get(int(packet_index))
                if record is not None:
                    record["span_sequences"].add(int(span.token_sequence))

        staging._span_packet_cursors = replacement_span_map
        committed_packet_cursors = {
            int(index)
            for index, record in staging._source_packets.items()
            if record.get("committed") is True
        }
        return LiveQwenSuffixReplacementPlan(
            expected_frontier=expected_frontier,
            revision_id=revision_id,
            cache_epoch=cache_epoch,
            staged=staging,
            span_cursor=new_span_cursor,
            span_packet_cursors=replacement_span_map,
            committed_packet_cursors=committed_packet_cursors,
            preserved_qwen_count=preserved_qwen_count,
            source_rows=tuple(normalized_replacements),
        )

    def has_unconsumed_qwen_tokens_for_suffix(
        self,
        replacement_tokens: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        *,
        expected_frontier: QwenSourceFrontier,
        frozen_prefix_token_ids: tuple[int, ...] | list[int],
        revision_id: str,
        cache_epoch: str | int | None,
    ) -> bool:
        """Check candidate Qwen readiness on a stream snapshot.

        The live condition protects only the source snapshot. Retokenization,
        staging, and Qwen token construction run after the live lock is
        released, so a private response candidate cannot hold up TTS ACKs.
        The final install repeats all frontier checks under the transaction
        locks.
        """
        with self._condition:
            self._raise_terminal_locked()
            actual_frontier = self._source_frontier_locked()
            if not self._same_frozen_source_frontier(actual_frontier, expected_frontier):
                raise RuntimeError("live Qwen frozen frontier changed before readiness check")
            snapshot = copy.copy(self)
            snapshot.__dict__ = {
                key: self._clone_final_text_plan_value(value)
                for key, value in self.__dict__.items()
                if key != "_condition"
            }
            snapshot._condition = threading.Condition(threading.RLock())

        with snapshot._condition:
            plan = snapshot.prepare_unconsumed_suffix_locked(
                replacement_tokens,
                expected_frontier=expected_frontier,
                frozen_prefix_token_ids=frozen_prefix_token_ids,
                revision_id=revision_id,
                cache_epoch=cache_epoch,
            )
        total_tokens = len(plan.staged._last_qwen_ids or ())
        return total_tokens > int(plan.preserved_qwen_count)

    @staticmethod
    def _clone_final_text_plan_value(value: Any) -> Any:
        if isinstance(value, (str, bytes, int, float, bool, type(None), torch.Tensor)):
            return value
        if isinstance(value, list):
            return [LiveQwenSayStream._clone_final_text_plan_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(LiveQwenSayStream._clone_final_text_plan_value(item) for item in value)
        if isinstance(value, set):
            return {LiveQwenSayStream._clone_final_text_plan_value(item) for item in value}
        if isinstance(value, deque):
            return deque(
                (LiveQwenSayStream._clone_final_text_plan_value(item) for item in value),
                maxlen=value.maxlen,
            )
        if isinstance(value, defaultdict):
            return defaultdict(
                value.default_factory,
                {
                    LiveQwenSayStream._clone_final_text_plan_value(key):
                    LiveQwenSayStream._clone_final_text_plan_value(item)
                    for key, item in value.items()
                },
            )
        if isinstance(value, dict):
            return {
                LiveQwenSayStream._clone_final_text_plan_value(key):
                LiveQwenSayStream._clone_final_text_plan_value(item)
                for key, item in value.items()
            }
        if isinstance(value, SpanAlignedSpeechCursor) or isinstance(
            value, (_PrefixTailSequence, _BoundedSequence, _ConcatSequence, _SourceCharEndView)
        ):
            cloned = copy.copy(value)
            cloned.__dict__ = {
                key: LiveQwenSayStream._clone_final_text_plan_value(item)
                for key, item in vars(value).items()
            }
            return cloned
        return value

    def prepare_final_text_flush_locked(
        self,
        text_piece: str,
        gemma_hidden: Any,
        *,
        token_index: int,
        source_token_id: int,
        source_token_seq: int | None,
        source_group: tuple[tuple[int, int], ...] | None,
        exclusive_kv_cursor: int,
        revision_id: str | None = None,
        seal_initial_segment: bool = False,
    ) -> LiveQwenFinalTextFlushPlan:
        """Stage a final text flush without changing the live stream."""
        owns_lock = getattr(self._condition, "_is_owned", None)
        if callable(owns_lock) and not owns_lock():
            raise RuntimeError("final text flush prepare requires its condition lock")
        staged = copy.copy(self)
        staged.__dict__ = {
            key: self._clone_final_text_plan_value(value)
            for key, value in self.__dict__.items()
            if key != "_condition"
        }
        staged._condition = threading.Condition(threading.RLock())
        staged.on_packet_emitted = None
        staged.on_tts_text_reserved = None
        staged.on_tts_text_consumed = None
        staged.on_tts_eos_consumed = None
        post_commit_emissions: list[Any] = []
        staged._dispatch_packet_emissions = lambda values: post_commit_emissions.extend(values)
        staged._feed_piece(
            text_piece,
            gemma_hidden,
            token_index=token_index,
            source_token_id=source_token_id,
            source_token_seq=source_token_seq,
            source_group=source_group,
            exclusive_kv_cursor=exclusive_kv_cursor,
            revision_id=revision_id,
            kv_committed=True,
            seal_initial_segment=seal_initial_segment,
            defer_publication=True,
            final_text_flush=True,
        )
        del staged.__dict__["_dispatch_packet_emissions"]
        staged.on_packet_emitted = self.on_packet_emitted
        staged.on_tts_text_reserved = self.on_tts_text_reserved
        staged.on_tts_text_consumed = self.on_tts_text_consumed
        staged.on_tts_eos_consumed = self.on_tts_eos_consumed
        staged_state = dict(staged.__dict__)
        staged_state["_condition"] = self._condition
        return LiveQwenFinalTextFlushPlan(
            expected_source_revision=int(self._source_revision),
            staged_state=staged_state,
            post_commit_emissions=tuple(post_commit_emissions),
        )

    def install_final_text_flush_locked(
        self,
        plan: LiveQwenFinalTextFlushPlan,
    ) -> None:
        """Install an already prepared final text flush with one assignment."""
        self.__dict__ = plan.staged_state

    def dispatch_final_text_flush_emissions(
        self,
        plan: LiveQwenFinalTextFlushPlan,
    ) -> None:
        """Run staged public ACK callbacks only after every owner lock is released."""
        if plan.post_commit_emissions:
            self._dispatch_packet_emissions(list(plan.post_commit_emissions))

    def install_unconsumed_suffix_locked(
        self,
        plan: LiveQwenSuffixReplacementPlan,
    ) -> None:
        """Install a stream plan using assignments only; caller holds condition."""
        owns_lock = getattr(self._condition, "_is_owned", None)
        if callable(owns_lock) and not owns_lock():
            raise RuntimeError("stream replacement install requires its condition lock")
        if not isinstance(plan, LiveQwenSuffixReplacementPlan):
            raise TypeError("plan must be a LiveQwenSuffixReplacementPlan")
        if not self._same_frozen_source_frontier(
            self._source_frontier_locked(), plan.expected_frontier
        ):
            raise RuntimeError("live Qwen source frontier changed after replacement prepare")
        stage = plan.staged
        self.span_cursor = plan.span_cursor
        self._source_revision = int(self._source_revision) + 1
        self.cache_epoch = plan.cache_epoch
        for name in (
            "_pieces", "_source_piece_raw_starts", "_source_piece_raw_ends",
            "_source_hiddens", "_canonical_cached_raw_end",
            "_canonical_cached_text_parts", "_canonical_cached_length",
            "_canonical_cached_sources", "_canonical_cached_raw_ranges",
            "_canonical_cached_source_char_ends", "_source_packet_cursors",
            "_source_cursor_to_index", "_source_sequence_to_index", "_source_packets",
            "_source_packet_order", "_source_piece_index", "_next_source_packet_to_emit",
            "_last_source_token_index", "_source_char_count", "_retokenization_work", "_published_ids",
            "_published_offsets", "_published_hidden", "_published_events",
            "_published_source_sequences", "_published_source_token_ids",
            "_published_source_cursors", "_published_source_packet_indices",
            "_last_qwen_ids", "_last_qwen_offsets", "_last_qwen_hidden",
            "_last_qwen_cursors", "_last_qwen_mutable_tail_ids",
            "_last_qwen_mutable_tail_offsets", "_last_qwen_mutable_tail_hidden",
            "_last_qwen_mutable_tail_cursors", "_last_qwen_mutable_tail_base",
            "_last_qwen_mutable_tail_text", "_current_qwen_tail",
            "_latest_stable_source_index", "_last_prefix", "_last_qwen_suffix_prefix",
            "_sealed_initial_text", "_sealed_initial_id_count", "_sealed_initial_ids",
            "_sealed_initial_offsets", "_sealed_initial_hidden", "_sealed_initial_cursors",
            "_stable_suffix_text", "_stable_suffix_ids", "_stable_suffix_offsets",
            "_stable_suffix_hidden", "_stable_suffix_cursors",
        ):
            setattr(self, name, getattr(stage, name))
        # Pending IDs belong to the suffix replaced by this transaction.
        self._deferred_publications.clear()
        self._qwen_cursor = int(plan.expected_frontier.qwen_consumed_count)
        self._reserved_consumption = self._reserved_consumption
        self._reserved_qwen_index = (
            self._reserved_qwen_index
            if self._reserved_consumption is not None else None
        )
        self._reserved_initial = self._reserved_initial
        self._span_packet_cursors = plan.span_packet_cursors
        self._committed_packet_cursors = plan.committed_packet_cursors
        self._condition.notify_all()

    def _notify_tts_consumed(self) -> None:
        callback = self.on_tts_text_consumed
        if not callable(callback):
            return
        with self._condition:
            frontier = self._source_frontier_locked()
            text = frontier.cumulative_source_text
            if (
                text == self._already_public_text
                and frontier.source_sequences == self._already_public_source_sequences
            ):
                return
        callback(frontier, text)
        with self._condition:
            if self.generation_id == frontier.generation_id and self.segment_id == frontier.segment_id:
                self._already_public_text = text
                self._already_public_source_sequences = tuple(frontier.source_sequences)

    def take_trailing_token(
        self,
        *,
        timeout_s: float | None = None,
        stop_check: Any | None = None,
        q1_index: int | None = None,
    ) -> int | None:
        if q1_index is not None and (
            isinstance(q1_index, bool) or not isinstance(q1_index, int) or q1_index < 0
        ):
            raise ValueError("q1_index must be a nonnegative integer or None")
        if q1_index is not None:
            self._q1_consumption_mode = True
        reserved: QwenTextConsumption | None = None
        terminal_eos = False
        with self._condition:
            self._wait_locked(
                lambda: self._reserved_consumption is None and (
                    self._qwen_cursor < len(self._published_ids)
                    or (self._closed and self._eos_enqueued)
                ),
                timeout_s=timeout_s,
                stop_check=stop_check,
            )
            self._raise_terminal_locked()
            if self._reserved_consumption is not None:
                raise RuntimeError("Qwen text input is already reserved")
            if self._qwen_cursor < len(self._published_ids):
                index = self._qwen_cursor
                self._reserved_consumption = self._make_consumption_locked(
                    index, q1_index=q1_index
                )
                self._reserved_qwen_index = index
                self._reserved_initial = False
                reserved = self._reserved_consumption
                token_id = int(self._published_ids[index])
            elif self._eos_enqueued and not self._eos_consumed:
                last_valid = self._qwen_cursor - 1
                if last_valid < 0:
                    raise RuntimeError("Qwen EOS cannot be reserved before a valid text token")
                self._reserved_consumption = self._make_consumption_locked(
                    last_valid,
                    q1_index=q1_index,
                    eos=True,
                    tail_hold=True,
                    qwen_token_id=self.text_eos_token_id,
                )
                self._reserved_qwen_index = None
                self._reserved_initial = False
                terminal_eos = True
                token_id = self.text_eos_token_id
            else:
                return None
        if reserved is not None and not terminal_eos:
            self._notify_tts_reserved(reserved)
        return token_id

    def consume_initial_token(self) -> None:
        with self._condition:
            if not self._reserved_initial or self._reserved_consumption is None:
                raise RuntimeError("Qwen initial text token is not in flight")
            if self._qwen_cursor != 0 or self._reserved_qwen_index != 0:
                raise RuntimeError("Qwen initial reservation no longer matches its cursor")
            consumption = self._reserved_consumption
            self._qwen_cursor = 1
            self._last_consumed_qwen_index = 0
            self._last_consumed_source_sequences = consumption.source_sequences
            self._q1_consumption_mode = (
                self._q1_consumption_mode or consumption.q1_index is not None
            )
            self._reserved_consumption = None
            self._reserved_qwen_index = None
            self._reserved_initial = False
            self._condition.notify_all()
        self._notify_tts_consumed()

    def ack_initial_token(self) -> None:
        """Compatibility alias for successful Qwen dynamic prefill."""
        self.consume_initial_token()

    def consume_trailing_token(self) -> None:
        with self._condition:
            consumption = self._reserved_consumption
            if consumption is None or self._reserved_initial:
                raise RuntimeError("Qwen trailing text token is not in flight")
            if consumption.eos:
                self._eos_consumed = True
            elif self._reserved_qwen_index != self._qwen_cursor:
                raise RuntimeError("Qwen trailing reservation no longer matches its cursor")
            else:
                self._last_consumed_qwen_index = self._reserved_qwen_index
                self._last_consumed_source_sequences = consumption.source_sequences
                self._q1_consumption_mode = (
                    self._q1_consumption_mode or consumption.q1_index is not None
                )
                self._qwen_cursor += 1
            self._reserved_consumption = None
            self._reserved_qwen_index = None
            self._condition.notify_all()
        if not consumption.eos:
            self._notify_tts_consumed()
        else:
            callback = self.on_tts_eos_consumed
            if callable(callback):
                callback()

    def ack_trailing_token(self) -> None:
        """Compatibility alias for successful native/eager Q1 ingestion."""
        self.consume_trailing_token()

    def hidden_for_frame(
        self,
        frame_id: int,
        *,
        stop_check: Any | None = None,
    ) -> LiveQwenHidden:
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError("frame_id deve essere un intero non negativo")
        with self._condition:
            consumption = self._reserved_consumption
            exact_q1 = self._q1_consumption_mode or (
                consumption is not None and consumption.q1_index is not None
            )
            if exact_q1:
                if consumption is None:
                    index = self._last_consumed_qwen_index
                    if index is None:
                        raise RuntimeError(
                            "live Qwen stream closed before a valid token was consumed"
                        )
                    consumption = self._make_consumption_locked(
                        index, q1_index=frame_id, pad=True, tail_hold=True
                    )
                if consumption.q1_index is not None and consumption.q1_index != frame_id:
                    raise RuntimeError("live Qwen reservation belongs to a different Q1 index")
                return LiveQwenHidden(
                    consumption.hidden,
                    consumption.tail_hold,
                    gemma_token_id=(
                        consumption.source_token_ids[-1]
                        if consumption.source_token_ids else None
                    ),
                    packet_exclusive_kv_cursor=(
                        consumption.source_kv_cursors[-1]
                        if consumption.source_kv_cursors else None
                    ),
                    qwen_token_id=consumption.qwen_token_id,
                    producer_event=consumption.producer_event,
                    source_sequences=consumption.source_sequences,
                    source_token_ids=consumption.source_token_ids,
                    q1_index=consumption.q1_index,
                    eos=consumption.eos,
                    pad=consumption.pad,
                )
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
            last_source_index = max((int(index) for index in source_cursors), default=-1)
            if any(
                index not in self._committed_packet_cursors
                for index in self._source_packet_cursors
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
                    source_index = int(cursor)
                    packet_record = self._source_packets.get(source_index)
                    if packet_record is not None:
                        packet_record["acked_sequences"].add(int(span.token_sequence))
            self._pending_emitted_frames.popleft()
            self._last_processed_emitted_frame = frame_id
        return tuple(acknowledged)

    def _collect_vocalized_packets_locked(self) -> list[tuple[int, str, int | None, str | None, str | int | None]]:
        emitted: list[tuple[int, str, int | None, str | None, str | int | None]] = []
        while self._next_source_packet_to_emit < len(self._source_packet_order):
            token_index = self._source_packet_order[self._next_source_packet_to_emit]
            record = self._source_packets[token_index]
            if not record["spans_known"]:
                break
            span_sequences = record["span_sequences"]
            if not span_sequences:
                # Canonicalized-away source text has no PCM span of its own.
                # Retain it for the next later span so that an ACK advances a
                # contiguous source prefix without attributing PCM to this row.
                self._vocalized_text_parts.append(str(record["text_piece"]))
                self._next_source_packet_to_emit += 1
                continue
            if (
                not record["committed"]
                or not span_sequences.issubset(record["acked_sequences"])
            ):
                break
            physical_cursor = record.get("cursor")
            if physical_cursor is None:
                # Non-unified V1 retains logical source identity without
                # inventing a physical Gemma KV cursor. PCM ACKs therefore
                # cannot publish a source prefix through this callback.
                self._next_source_packet_to_emit += 1
                continue
            self._vocalized_text_parts.append(str(record["text_piece"]))
            emitted.append((
                int(physical_cursor),
                "".join(self._vocalized_text_parts),
                int(record["source_token_seq"])
                if isinstance(record.get("source_token_seq"), int)
                and not isinstance(record.get("source_token_seq"), bool)
                else None,
                str(record.get("revision_id"))
                if record.get("revision_id") is not None else None,
                self.cache_epoch,
            ))
            self._next_source_packet_to_emit += 1
        return emitted

    def _dispatch_packet_emissions(
        self,
        emitted: list[tuple[int, str, int | None, str | None, str | int | None]],
    ) -> None:
        callback = self.on_packet_emitted
        if not callable(callback):
            return
        try:
            parameters = inspect.signature(callback).parameters
            accepts_provenance = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            ) or {"source_token_seq", "revision_id", "cache_epoch_id"}.issubset(parameters)
        except (TypeError, ValueError):
            accepts_provenance = False
        for cursor, text, source_seq, revision_id, cache_epoch_id in emitted:
            if accepts_provenance:
                callback(
                    cursor,
                    text,
                    source_token_seq=source_seq,
                    revision_id=revision_id,
                    cache_epoch_id=cache_epoch_id,
                )
            else:
                callback(cursor, text)

    def mark_packet_committed(
        self,
        exclusive_kv_cursor: int,
        *,
        generation_id: int,
        segment_id: int,
        source_sequence: int | None = None,
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
            token_index = (
                self._source_sequence_to_index.get(int(source_sequence))
                if source_sequence is not None
                else next(
                    iter(reversed(self._source_cursor_to_index.get(exclusive_kv_cursor, ()))),
                    None,
                )
            )
            if token_index is None:
                raise RuntimeError("committed KV cursor has no source Gemma packet")
            if int(self._source_packet_cursors.get(token_index, -1)) != int(exclusive_kv_cursor):
                raise RuntimeError("committed source packet does not match its revision-local KV cursor")
            # A commit cannot leap over an earlier uncommitted source token.
            for earlier_index in self._source_packet_cursors:
                if earlier_index < token_index and earlier_index not in self._committed_packet_cursors:
                    return None
            if token_index in self._committed_packet_cursors:
                return ()
            self._committed_packet_cursors.add(token_index)
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
            self._deferred_publications.clear()
            # Keep only text IDs already handed to the Talker. Any remaining
            # queue tail must not become consumable after cancellation; the
            # active Talker frame can finish under its stop_check, then the
            # next text/frame read observes the cancellation.
            preserve_until = self._qwen_cursor + int(
                self._reserved_consumption is not None
                and self._reserved_qwen_index == self._qwen_cursor
            )
            del self._published_ids[preserve_until:]
            del self._published_offsets[preserve_until:]
            del self._published_hidden[preserve_until:]
            del self._published_events[preserve_until:]
            del self._published_source_sequences[preserve_until:]
            del self._published_source_token_ids[preserve_until:]
            del self._published_source_cursors[preserve_until:]
            del self._published_source_packet_indices[preserve_until:]
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
            self._deferred_publications.clear()
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
            if self._sealed_initial_text is not None and self._last_qwen_ids is not None:
                return self._last_prefix
            text, _char_sources, _source_hiddens = self._canonical_source_map(self._pieces)
            return text

    @property
    def initial_prefill_text(self) -> str:
        """Text whose first Qwen ID was fixed for the initial live prefill."""
        with self._condition:
            if self._sealed_initial_text is not None:
                return self._sealed_initial_text
            return self.visible_text

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
    def first_qwen_id_published_at_ns(self) -> int | None:
        """Monotonic timestamp captured when the first stable Qwen ID is published."""
        with self._condition:
            return self._first_qwen_id_published_at_ns

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
        on_tts_text_consumed: Any | None = None,
        cache_epoch: str | int | None = None,
    ) -> None:
        if isinstance(text_eos_token_id, bool) or int(text_eos_token_id) < 0:
            raise ValueError("text_eos_token_id deve essere non negativo")
        if isinstance(max_tokens, bool) or int(max_tokens) < 1:
            raise ValueError("max_tokens deve essere positivo")
        if isinstance(max_source_chars, bool) or int(max_source_chars) < 1:
            raise ValueError("max_source_chars deve essere positivo")
        if cache_epoch is not None and (
            isinstance(cache_epoch, bool)
            or not isinstance(cache_epoch, (int, str))
            or (isinstance(cache_epoch, str) and not cache_epoch)
        ):
            raise ValueError("cache_epoch deve essere un intero, una stringa non vuota o None")
        if not isinstance(fifo, TurnLocalSpeechFIFO):
            raise TypeError("fifo deve essere TurnLocalSpeechFIFO")
        self.qwen_tokenizer = qwen_tokenizer
        self.text_eos_token_id = int(text_eos_token_id)
        self.fifo = fifo
        self.generation_id = int(generation_id)
        self.segment_id = fifo.segment_id
        self.cache_epoch = cache_epoch
        self.max_tokens = int(max_tokens)
        self.max_source_chars = int(max_source_chars)
        if self.max_tokens > 512 and self.max_source_chars == 16_384:
            self.max_source_chars = max(16_384, self.max_tokens * 32)
        if on_packet_consumed is not None and on_packet_emitted is not None:
            raise ValueError("specify only one packet emission callback")
        # Keep the historical keyword accepted while moving its trigger to
        # the PCM-emission boundary. New callers should use on_packet_emitted.
        self.on_packet_emitted = on_packet_emitted or on_packet_consumed
        self.on_tts_text_consumed = on_tts_text_consumed
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
        self._next_source_sequence = 0
        self._source_packets: list[tuple[int, int, str, int]] = []
        self._source_sequence_by_packet: dict[int, int] = {}
        self._source_revision = 0
        self._qwen_consumed_count = 0
        self._reserved_consumption: QwenTextConsumption | None = None
        self._reserved_initial = False
        self._last_consumed_consumption: QwenTextConsumption | None = None
        self._consumed_source_sequences: set[int] = set()
        self._q1_consumption_mode = False
        self._already_public_text = ""
        self._already_public_source_sequences: tuple[int, ...] = ()

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
        source_sequence = self._next_source_sequence
        self._next_source_sequence += 1
        self._source_sequence_by_packet[id(packet)] = source_sequence
        self._source_packets.append((
            source_sequence,
            int(packet.token_id),
            str(packet.text_piece),
            int(packet.exclusive_kv_cursor),
        ))
        self._source_revision += 1
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
            consumption = self._make_fifo_consumption_locked(token, q1_index=None)
            self._reserved_consumption = consumption
            self._reserved_initial = True
        return LiveQwenToken(token.token_id, hidden, consumption)

    @property
    def reserved_consumption(self) -> QwenTextConsumption | None:
        with self._condition:
            return self._reserved_consumption

    def _make_fifo_consumption_locked(
        self,
        token: SpeechQwenToken,
        *,
        q1_index: int | None,
        eos: bool = False,
        tail_hold: bool = False,
    ) -> QwenTextConsumption:
        sequence = self._source_sequence_by_packet.get(id(token.packet))
        sequences = () if sequence is None else (int(sequence),)
        source_ids = () if sequence is None else (int(token.packet.token_id),)
        cursors = () if sequence is None else (int(token.packet.exclusive_kv_cursor),)
        return QwenTextConsumption(
            qwen_token_id=int(token.token_id),
            source_sequences=sequences,
            source_token_ids=source_ids,
            source_kv_cursors=cursors,
            hidden=self._hidden_3d(token.packet),
            producer_event=token.packet.producer_event,
            q1_index=q1_index,
            eos=bool(eos),
            tail_hold=bool(tail_hold),
        )

    def source_frontier_snapshot(self) -> QwenSourceFrontier:
        with self._condition:
            sequences = sorted(self._consumed_source_sequences)
            if self._reserved_consumption is not None:
                sequences.extend(self._reserved_consumption.source_sequences)
            sequences = sorted(set(int(value) for value in sequences))
            source_by_sequence = {
                sequence: (token_id, text, cursor)
                for sequence, token_id, text, cursor in self._source_packets
            }
            token_ids = tuple(
                source_by_sequence[sequence][0]
                for sequence in sequences if sequence in source_by_sequence
            )
            cursors = tuple(
                source_by_sequence[sequence][2]
                for sequence in sequences if sequence in source_by_sequence
            )
            frozen = max(sequences, default=-1)
            text = "".join(
                source_by_sequence[sequence][1]
                for sequence in range(frozen + 1)
                if sequence in source_by_sequence
            )
            return QwenSourceFrontier(
                generation_id=self.generation_id,
                segment_id=self.segment_id,
                revision=self._source_revision,
                qwen_consumed_count=self._qwen_consumed_count,
                reserved_qwen_token_id=(
                    self._reserved_consumption.qwen_token_id
                    if self._reserved_consumption is not None
                    else None
                ),
                frozen_source_sequence=frozen,
                source_sequences=tuple(sequences),
                source_token_ids=token_ids,
                source_kv_cursors=cursors,
                already_public_text=self._already_public_text,
                cumulative_source_text=text,
                cache_epoch=self.cache_epoch,
            )

    def _notify_tts_consumed(self, *, packet_complete: bool) -> None:
        callback = self.on_tts_text_consumed
        if not packet_complete or not callable(callback):
            return
        frontier = self.source_frontier_snapshot()
        if (
            frontier.cumulative_source_text == self._already_public_text
            and frontier.source_sequences == self._already_public_source_sequences
        ):
            return
        callback(frontier, frontier.cumulative_source_text)
        with self._condition:
            self._already_public_text = frontier.cumulative_source_text
            self._already_public_source_sequences = tuple(frontier.source_sequences)

    def consume_initial_token(self) -> None:
        with self._condition:
            if not self._initial_inflight:
                raise RuntimeError("turn-local Qwen initial token is not in flight")
            packet = self._initial_packet
            consumption = self._reserved_consumption
        if packet is None:
            raise RuntimeError("turn-local Qwen initial packet was lost before ACK")
        self.fifo.consume_qwen_token(
            generation_id=self.generation_id,
            segment_id=self.segment_id,
        )
        with self._condition:
            self._initial_inflight = False
            self._initial_packet = None
            if consumption is None or not self._reserved_initial:
                raise RuntimeError("turn-local Qwen initial reservation was lost")
            self._qwen_consumed_count += 1
            self._last_consumed_consumption = consumption
            self._consumed_source_sequences.update(consumption.source_sequences)
            self._q1_consumption_mode = (
                self._q1_consumption_mode or consumption.q1_index is not None
            )
            self._reserved_consumption = None
            self._reserved_initial = False
        self._notify_tts_consumed(
            packet_complete=bool(
                self._last_packet_token is not None
                and self._last_packet_token.packet_token_index + 1
                == self._last_packet_token.packet_token_count
            )
        )

    def ack_initial_token(self) -> None:
        """Compatibility alias; this acknowledges Qwen ingestion, not PCM."""
        self.consume_initial_token()

    def take_trailing_token(
        self,
        *,
        timeout_s: float | None = None,
        stop_check: Any | None = None,
        q1_index: int | None = None,
    ) -> int | None:
        if q1_index is not None and (
            isinstance(q1_index, bool) or not isinstance(q1_index, int) or q1_index < 0
        ):
            raise ValueError("q1_index must be a nonnegative integer or None")
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
                if self._trailing_inflight or self._reserved_consumption is not None:
                    raise RuntimeError("turn-local Qwen trailing token is already in flight")
                self._trailing_inflight = True
                self._last_hidden = self._hidden_3d(token.packet)
                self._tail_hold = False
                self._trailing_token = token
                self._last_packet_token = token
                self._reserved_consumption = self._make_fifo_consumption_locked(
                    token, q1_index=q1_index
                )
                self._reserved_initial = False
            return token.token_id
        with self._condition:
            if not self._closed:
                return None
            if not self._eos_consumed and not self._eos_inflight:
                self._eos_inflight = True
                self._tail_hold = True
                if self._last_consumed_consumption is None:
                    raise RuntimeError("turn-local Qwen EOS cannot precede a valid consumed token")
                previous = self._last_consumed_consumption
                self._reserved_consumption = QwenTextConsumption(
                    qwen_token_id=int(self.text_eos_token_id),
                    source_sequences=previous.source_sequences,
                    source_token_ids=previous.source_token_ids,
                    source_kv_cursors=previous.source_kv_cursors,
                    hidden=previous.hidden,
                    producer_event=previous.producer_event,
                    q1_index=q1_index,
                    eos=True,
                    tail_hold=True,
                )
                return self.text_eos_token_id
            return None

    def consume_trailing_token(self) -> None:
        with self._condition:
            if self._eos_inflight:
                self._eos_inflight = False
                self._eos_consumed = True
                self._reserved_consumption = None
                return
            if not self._trailing_inflight:
                raise RuntimeError("turn-local Qwen trailing token is not in flight")
            token = self._trailing_token
            consumption = self._reserved_consumption
        if token is None:
            raise RuntimeError("turn-local Qwen trailing packet was lost before ACK")
        self.fifo.consume_qwen_token(
            generation_id=self.generation_id,
            segment_id=self.segment_id,
        )
        with self._condition:
            self._trailing_inflight = False
            self._trailing_token = None
            if consumption is None:
                raise RuntimeError("turn-local Qwen consumption record was lost")
            self._qwen_consumed_count += 1
            self._last_consumed_consumption = consumption
            self._consumed_source_sequences.update(consumption.source_sequences)
            self._q1_consumption_mode = (
                self._q1_consumption_mode or consumption.q1_index is not None
            )
            self._reserved_consumption = None
        self._notify_tts_consumed(
            packet_complete=bool(
                token.packet_token_index + 1 == token.packet_token_count
            )
        )
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
            consumption = self._reserved_consumption
            exact_q1 = self._q1_consumption_mode or (
                consumption is not None and consumption.q1_index is not None
            )
            if exact_q1:
                if consumption is None:
                    consumption = self._last_consumed_consumption
                    if consumption is None:
                        raise RuntimeError(
                            "turn-local Qwen stream has no consumed text hidden for pad hold"
                        )
                    consumption = QwenTextConsumption(
                        qwen_token_id=None,
                        source_sequences=consumption.source_sequences,
                        source_token_ids=consumption.source_token_ids,
                        source_kv_cursors=consumption.source_kv_cursors,
                        hidden=consumption.hidden,
                        producer_event=consumption.producer_event,
                        q1_index=int(frame_id),
                        pad=True,
                        tail_hold=True,
                    )
                if consumption.q1_index is not None and consumption.q1_index != int(frame_id):
                    raise RuntimeError("turn-local Qwen reservation belongs to a different Q1 index")
                return LiveQwenHidden(
                    consumption.hidden,
                    tail_hold=consumption.tail_hold,
                    gemma_token_id=(
                        consumption.source_token_ids[-1]
                        if consumption.source_token_ids else None
                    ),
                    packet_exclusive_kv_cursor=(
                        consumption.source_kv_cursors[-1]
                        if consumption.source_kv_cursors else None
                    ),
                    qwen_token_id=consumption.qwen_token_id,
                    producer_event=consumption.producer_event,
                    source_sequences=consumption.source_sequences,
                    source_token_ids=consumption.source_token_ids,
                    q1_index=consumption.q1_index,
                    eos=consumption.eos,
                    pad=consumption.pad,
                )
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
