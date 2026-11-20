"""Session-owned logical response ledger for text, cache, and PCM frontiers."""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import threading
from typing import Any, Callable, Mapping, Sequence

import numpy as np

MAX_RETAINED_RESPONSE_PCM_SAMPLES = 24_000 * 30


@dataclass(frozen=True)
class ResponseLedgerProjection:
    response_id: str
    generation_id: int
    revision_id: str
    cache_epoch_id: str | None
    source_input_watermark: int | None
    source_tokens: tuple[Mapping[str, Any], ...]
    confirmed_source_tokens: tuple[Mapping[str, Any], ...]
    generated_exclusive_kv_cursor: int | None
    confirmed_exclusive_kv_cursor: int | None
    physical_cursor_contiguous: bool
    intended_text: str
    confirmed_text: str
    text_status: str
    response_done: bool
    response_cancelled: bool
    finish_reason: str | None
    pcm_samples_synthesized: int
    pcm_samples_sent: int
    pcm_samples_consumed: int
    all_pcm_consumed: bool
    retained_pcm_samples: int
    retained_pcm_truncated: bool
    consumed_pcm_chunks: tuple[Mapping[str, Any], ...]
    logical_source_token_seq: int
    tts_reserved_source_seq: int
    tts_consumed_source_seq: int
    history_order: int
    history_mutation_sequence_at_start: int


@dataclass(frozen=True)
class ResponseLedgerSuffixReplacementPlan:
    """Prevalidated replacement state for a coordinated stream/ledger CAS.

    The private mutable maps are newly allocated and owned only by this plan.
    Callers must not mutate them. Installation transfers them into the ledger
    by assignment while holding ``ResponseEventLedger._lock``.
    """

    expected_source_frontier: int
    expected_revision_id: str
    expected_cache_epoch_id: str | None
    expected_tts_published_source_seq: int
    expected_tts_reserved_source_seq: int
    expected_tts_consumed_source_seq: int
    revision_id: str
    cache_epoch_id: str | None
    sequences: tuple[int, ...]
    _source_tokens: dict[int, dict[str, Any]] = field(repr=False, compare=False)
    _source_identity_to_seq: dict[tuple[Any, ...], int] = field(repr=False, compare=False)
    _retired_source_sequences: set[int] = field(repr=False, compare=False)
    _next_source_seq: int = field(repr=False, compare=False)
    _intended_text: str = field(repr=False, compare=False)


@dataclass(frozen=True)
class ResponseLedgerSourceTextUpdatePlan:
    """Prebuilt mutable-token text completion for a coordinated stream commit."""

    source_token_seq: int
    expected_revision_id: str
    expected_cache_epoch_id: str | None
    changed: bool
    _source_tokens: dict[int, dict[str, Any]] = field(repr=False, compare=False)
    _source_identity_to_seq: dict[tuple[Any, ...], int] = field(repr=False, compare=False)


class ResponseEventLedger:
    """Record causal response events without equating tokens with audio time.

    ``source_token_seq`` is the logical identity/order. Physical KV positions
    are retained as per-revision provenance and never define source contiguity.
    Confirmed text comes only from source rows consumed by TTS and covered by a
    closed TTS unit whose complete PCM range the client consumed. A complete
    PCM watermark alone never proves that every intended text row was spoken.
    """

    def __init__(
        self,
        *,
        response_id: str,
        generation_id: int,
        revision_id: str = "r0",
        cache_epoch_id: str | None = None,
        source_input_watermark: int | None = None,
        history_order: int = 0,
        history_mutation_sequence_at_start: int = 0,
    ) -> None:
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("response_id must be nonempty")
        self.response_id = response_id
        self.generation_id = int(generation_id)
        self.revision_id = str(revision_id)
        self.cache_epoch_id = None if cache_epoch_id is None else str(cache_epoch_id)
        self.source_input_watermark = (
            None if source_input_watermark is None else int(source_input_watermark)
        )
        self.history_order = int(history_order)
        self.history_mutation_sequence_at_start = int(history_mutation_sequence_at_start)
        self._lock = threading.RLock()
        self._source_tokens: dict[int, dict[str, Any]] = {}
        self._source_identity_to_seq: dict[tuple[Any, ...], int] = {}
        self._retired_source_sequences: set[int] = set()
        self._next_source_seq = 0
        self._tts_published_source_seq = -1
        self._tts_reserved_source_seq = -1
        self._tts_consumed_source_seq = -1
        self._segments: dict[str, dict[str, Any]] = {}
        self._pcm_samples_synthesized = 0
        self._pcm_samples_sent = 0
        self._pcm_samples_consumed = 0
        self._retained_pcm_samples = 0
        self._retained_pcm_truncated = False
        self._consumed_pcm_chunks: list[dict[str, Any]] = []
        self._intended_text = ""
        self._response_done = False
        self._response_cancelled = False
        self._finish_reason: str | None = None

    def set_context(
        self,
        *,
        revision_id: str | None = None,
        cache_epoch_id: str | None = None,
        source_input_watermark: int | None = None,
    ) -> None:
        with self._lock:
            if revision_id is not None:
                self.revision_id = str(revision_id)
            if cache_epoch_id is not None:
                self.cache_epoch_id = str(cache_epoch_id)
            if source_input_watermark is not None:
                value = int(source_input_watermark)
                if self.source_input_watermark is not None and value < self.source_input_watermark:
                    raise ValueError("source input watermark cannot move backwards")
                self.source_input_watermark = value

    @staticmethod
    def _token_row(token: Any) -> Mapping[str, Any]:
        if isinstance(token, Mapping):
            return token
        try:
            return vars(token)
        except TypeError as exc:
            raise TypeError("source token must be a mapping or record") from exc

    @staticmethod
    def _physical_provenance(
        *, revision_id: str, cache_epoch_id: str | None, physical_cursor: int | None
    ) -> dict[str, Any]:
        return {
            "revision_id": revision_id,
            "cache_epoch_id": cache_epoch_id,
            "physical_kv_cursor": physical_cursor,
        }

    def _frozen_source_frontier_locked(self) -> int:
        closed = [
            int(row["source_token_seq"])
            for segment in self._segments.values()
            if segment.get("confirmed") is True
            for row in segment.get("closed_tokens", ())
            if isinstance(row.get("source_token_seq"), int)
            and not isinstance(row.get("source_token_seq"), bool)
        ]
        return max(
            self._tts_published_source_seq,
            self._tts_reserved_source_seq,
            self._tts_consumed_source_seq,
            *closed,
            -1,
        )

    def source_frontier_snapshot(self) -> dict[str, Any]:
        """Return the response-local logical source frontiers for CAS callers."""
        with self._lock:
            return {
                "generated_source_sequence": self._next_source_seq - 1,
                "next_source_sequence": self._next_source_seq,
                "frozen_source_sequence": self._frozen_source_frontier_locked(),
                "revision_id": self.revision_id,
                "cache_epoch_id": self.cache_epoch_id,
                "tts_published_source_seq": self._tts_published_source_seq,
                "tts_reserved_source_seq": self._tts_reserved_source_seq,
                "tts_consumed_source_seq": self._tts_consumed_source_seq,
            }

    def record_generated_tokens(
        self,
        tokens: Sequence[Any],
        *,
        revision_id: str | None = None,
        cache_epoch_id: str | None = None,
    ) -> tuple[int, ...]:
        recorded_sequences: list[int] = []
        with self._lock:
            if revision_id is not None:
                self.revision_id = str(revision_id)
            if cache_epoch_id is not None:
                self.cache_epoch_id = str(cache_epoch_id)
            for token in tokens:
                row = self._token_row(token)
                token_id = row.get("token_id")
                text_piece = row.get("text_piece", "")
                physical_cursor = row.get(
                    "exclusive_kv_cursor", row.get("physical_kv_cursor")
                )
                if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                    raise ValueError("source token ID must be a nonnegative integer")
                if not isinstance(text_piece, str):
                    raise TypeError("source token text_piece must be str")
                token_revision = str(row.get("revision_id") or self.revision_id)
                token_epoch_raw = row.get("cache_epoch_id", self.cache_epoch_id)
                token_epoch = None if token_epoch_raw is None else str(token_epoch_raw)
                if physical_cursor is not None and (
                    isinstance(physical_cursor, bool)
                    or not isinstance(physical_cursor, int)
                    or physical_cursor < 0
                ):
                    raise ValueError("physical KV cursor must be a nonnegative integer")

                source_seq = row.get("source_token_seq")
                if source_seq is None:
                    identity = (
                        token_revision,
                        token_epoch,
                        physical_cursor,
                        int(token_id),
                        text_piece,
                    )
                    source_seq = self._source_identity_to_seq.get(identity)
                    if source_seq is None:
                        source_seq = self._next_source_seq
                        self._source_identity_to_seq[identity] = int(source_seq)
                if (
                    isinstance(source_seq, bool)
                    or not isinstance(source_seq, int)
                    or source_seq < 0
                ):
                    raise ValueError("source_token_seq must be a nonnegative integer")
                source_seq = int(source_seq)
                if source_seq in self._retired_source_sequences:
                    raise ValueError("retired logical source sequence cannot be reused")

                existing = self._source_tokens.get(source_seq)
                if existing is not None:
                    # Logical identity survives a KV rebase. Record another
                    # physical provenance row instead of treating the rebase
                    # as a source-token mutation.
                    if existing["token_id"] != int(token_id) or existing["text_piece"] != text_piece:
                        raise ValueError("logical source token sequence was mutated")
                    provenance = self._physical_provenance(
                        revision_id=token_revision,
                        cache_epoch_id=token_epoch,
                        physical_cursor=(
                            None if physical_cursor is None else int(physical_cursor)
                        ),
                    )
                    provenance_rows = existing.setdefault("physical_kv_provenance", [])
                    if provenance not in provenance_rows:
                        provenance_rows.append(provenance)
                    existing["physical_kv_cursor"] = (
                        None if physical_cursor is None else int(physical_cursor)
                    )
                    existing["revision_id"] = token_revision
                    existing["cache_epoch_id"] = token_epoch
                else:
                    if source_seq <= self._frozen_source_frontier_locked():
                        raise ValueError("source token sequence is at or behind an immutable frontier")
                    provenance = self._physical_provenance(
                        revision_id=token_revision,
                        cache_epoch_id=token_epoch,
                        physical_cursor=(
                            None if physical_cursor is None else int(physical_cursor)
                        ),
                    )
                    self._source_tokens[source_seq] = {
                        "source_token_seq": source_seq,
                        "token_id": int(token_id),
                        "text_piece": text_piece,
                        "physical_kv_cursor": (
                            None if physical_cursor is None else int(physical_cursor)
                        ),
                        "revision_id": token_revision,
                        "cache_epoch_id": token_epoch,
                        "physical_kv_provenance": [provenance],
                    }
                self._next_source_seq = max(self._next_source_seq, source_seq + 1)
                recorded_sequences.append(source_seq)
        return tuple(recorded_sequences)

    def replace_unconsumed_suffix(
        self,
        start_source_seq: int,
        replacement_tokens: Sequence[Any],
        *,
        expected_source_frontier: int,
        expected_revision_id: str,
        expected_cache_epoch_id: str | None,
        revision_id: str,
        cache_epoch_id: str | None = None,
        intended_text: str | None = None,
    ) -> tuple[int, ...]:
        """Compatibility wrapper for a standalone suffix revision transaction."""
        with self._lock:
            plan = self.prepare_unconsumed_suffix_locked(
                start_source_seq,
                replacement_tokens,
                expected_source_frontier=expected_source_frontier,
                expected_revision_id=expected_revision_id,
                expected_cache_epoch_id=expected_cache_epoch_id,
                expected_tts_published_source_seq=self._tts_published_source_seq,
                expected_tts_reserved_source_seq=self._tts_reserved_source_seq,
                expected_tts_consumed_source_seq=self._tts_consumed_source_seq,
                revision_id=revision_id,
                cache_epoch_id=cache_epoch_id,
                intended_text=intended_text,
            )
            return self.install_unconsumed_suffix_locked(plan)

    def prepare_unconsumed_suffix_locked(
        self,
        start_source_seq: int,
        replacement_tokens: Sequence[Any],
        *,
        expected_source_frontier: int,
        expected_revision_id: str,
        expected_cache_epoch_id: str | None,
        expected_tts_published_source_seq: int,
        expected_tts_reserved_source_seq: int,
        expected_tts_consumed_source_seq: int,
        revision_id: str,
        cache_epoch_id: str | None = None,
        intended_text: str | None = None,
    ) -> ResponseLedgerSuffixReplacementPlan:
        """Prepare a suffix CAS without mutating ledger state.

        Caller must already hold ``self._lock`` when coordinating this plan
        with another mutable owner. The method intentionally leaves every
        ledger field unchanged if validation or preparation raises.
        """
        if isinstance(start_source_seq, bool) or not isinstance(start_source_seq, int) or start_source_seq < 0:
            raise ValueError("start_source_seq must be a nonnegative integer")
        expected_frontier_values = (
            expected_source_frontier,
            expected_tts_published_source_seq,
            expected_tts_reserved_source_seq,
            expected_tts_consumed_source_seq,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in expected_frontier_values):
            raise ValueError("expected source frontiers must be integers")
        current_frontier = self._frozen_source_frontier_locked()
        if (
            expected_source_frontier != current_frontier
            or str(expected_revision_id) != self.revision_id
            or (
                None if expected_cache_epoch_id is None else str(expected_cache_epoch_id)
            ) != self.cache_epoch_id
            or expected_tts_published_source_seq != self._tts_published_source_seq
            or expected_tts_reserved_source_seq != self._tts_reserved_source_seq
            or expected_tts_consumed_source_seq != self._tts_consumed_source_seq
        ):
            raise RuntimeError("response source frontier CAS lost its revision or cache epoch")
        if start_source_seq <= current_frontier:
            raise ValueError("revision would replace a reserved or consumed source token")
        if start_source_seq != current_frontier + 1:
            raise ValueError("revision must begin immediately after the frozen source frontier")
        for segment in self._segments.values():
            if any(
                isinstance(row.get("source_token_seq"), int)
                and not isinstance(row.get("source_token_seq"), bool)
                and int(row["source_token_seq"]) >= start_source_seq
                for row in segment.get("closed_tokens", ())
            ):
                raise ValueError("revision would replace source tokens linked to a closed TTS unit")
        normalized: list[dict[str, Any]] = []
        candidate_sequences: set[int] = set()
        next_seq = max(self._next_source_seq, start_source_seq)
        last_sequence = start_source_seq - 1
        replacement_revision_id = str(revision_id)
        replacement_cache_epoch_id = None if cache_epoch_id is None else str(cache_epoch_id)
        if intended_text is not None and not isinstance(intended_text, str):
            raise TypeError("intended_text must be str")
        final_intended_text = self._intended_text if intended_text is None else intended_text
        for token in replacement_tokens:
            row = dict(self._token_row(token))
            token_id = row.get("token_id")
            text_piece = row.get("text_piece", "")
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ValueError("replacement token ID must be a nonnegative integer")
            if not isinstance(text_piece, str):
                raise TypeError("replacement token text_piece must be str")
            physical_cursor = row.get(
                "exclusive_kv_cursor", row.get("physical_kv_cursor")
            )
            if physical_cursor is not None and (
                isinstance(physical_cursor, bool)
                or not isinstance(physical_cursor, int)
                or physical_cursor < 0
            ):
                raise ValueError("replacement physical KV cursor is invalid")
            provided_seq = row.get("source_token_seq")
            if provided_seq is not None and (
                isinstance(provided_seq, bool)
                or not isinstance(provided_seq, int)
                or provided_seq < start_source_seq
                or provided_seq in self._retired_source_sequences
                or provided_seq in self._source_tokens
                or provided_seq in candidate_sequences
                or provided_seq <= last_sequence
            ):
                raise ValueError("replacement token has a reused or invalid logical source sequence")
            if provided_seq is None:
                next_seq = max(next_seq, last_sequence + 1)
                while (
                    next_seq in self._retired_source_sequences
                    or next_seq in self._source_tokens
                    or next_seq in candidate_sequences
                ):
                    next_seq += 1
                provided_seq = next_seq
                next_seq += 1
                row["source_token_seq"] = provided_seq
            last_sequence = int(provided_seq)
            next_seq = max(next_seq, last_sequence + 1)
            candidate_sequences.add(last_sequence)
            row["revision_id"] = replacement_revision_id
            row["cache_epoch_id"] = replacement_cache_epoch_id
            normalized.append(row)

        retired = {
            int(source_seq)
            for source_seq in self._source_tokens
            if source_seq >= start_source_seq
        }
        new_source_tokens = dict(self._source_tokens)
        for source_seq in retired:
            new_source_tokens.pop(source_seq, None)
        new_retired_source_sequences = set(self._retired_source_sequences)
        new_retired_source_sequences.update(retired)
        new_source_identity_to_seq = {
            identity: source_seq
            for identity, source_seq in self._source_identity_to_seq.items()
            if source_seq not in new_retired_source_sequences
        }
        sequences: list[int] = []
        for row in normalized:
            source_seq = int(row["source_token_seq"])
            physical_cursor = row.get("exclusive_kv_cursor", row.get("physical_kv_cursor"))
            normalized_cursor = None if physical_cursor is None else int(physical_cursor)
            provenance = self._physical_provenance(
                revision_id=replacement_revision_id,
                cache_epoch_id=replacement_cache_epoch_id,
                physical_cursor=normalized_cursor,
            )
            new_source_tokens[source_seq] = {
                "source_token_seq": source_seq,
                "token_id": int(row["token_id"]),
                "text_piece": str(row["text_piece"]),
                "physical_kv_cursor": normalized_cursor,
                "revision_id": replacement_revision_id,
                "cache_epoch_id": replacement_cache_epoch_id,
                "physical_kv_provenance": [provenance],
            }
            sequences.append(source_seq)

        return ResponseLedgerSuffixReplacementPlan(
            expected_source_frontier=current_frontier,
            expected_revision_id=self.revision_id,
            expected_cache_epoch_id=self.cache_epoch_id,
            expected_tts_published_source_seq=self._tts_published_source_seq,
            expected_tts_reserved_source_seq=self._tts_reserved_source_seq,
            expected_tts_consumed_source_seq=self._tts_consumed_source_seq,
            revision_id=replacement_revision_id,
            cache_epoch_id=replacement_cache_epoch_id,
            sequences=tuple(sequences),
            _source_tokens=new_source_tokens,
            _source_identity_to_seq=new_source_identity_to_seq,
            _retired_source_sequences=new_retired_source_sequences,
            _next_source_seq=next_seq,
            _intended_text=final_intended_text,
        )

    def install_unconsumed_suffix_locked(
        self,
        plan: ResponseLedgerSuffixReplacementPlan,
    ) -> tuple[int, ...]:
        """Install a prepared plan using assignments only; caller holds ``_lock``."""
        self._source_tokens = plan._source_tokens
        self._source_identity_to_seq = plan._source_identity_to_seq
        self._retired_source_sequences = plan._retired_source_sequences
        self._next_source_seq = plan._next_source_seq
        self.revision_id = plan.revision_id
        self.cache_epoch_id = plan.cache_epoch_id
        self._intended_text = plan._intended_text
        return plan.sequences

    def _require_source_token_seq_locked(self, source_token_seq: int, name: str) -> int:
        if isinstance(source_token_seq, bool) or not isinstance(source_token_seq, int) or source_token_seq < 0:
            raise ValueError(f"{name} source frontier must be nonnegative")
        if source_token_seq not in self._source_tokens:
            raise ValueError(f"{name} source frontier references an unknown logical source token")
        return int(source_token_seq)

    def update_unconsumed_source_text(
        self,
        source_token_seq: int,
        text_piece: str,
        *,
        expected_revision_id: str,
        expected_cache_epoch_id: str | None,
    ) -> bool:
        """Finalize one still-mutable token delta without changing its identity.

        This is for a decoder's final incremental-text flush. The logical
        sequence, token ID, and physical KV provenance stay fixed; only the
        token's final text delta may be completed before it is published or
        reserved for TTS.
        """
        if not isinstance(text_piece, str):
            raise TypeError("source text_piece must be str")
        if not isinstance(expected_revision_id, str) or not expected_revision_id:
            raise ValueError("expected revision_id must be nonempty")
        if expected_cache_epoch_id is not None and (
            isinstance(expected_cache_epoch_id, bool)
            or not isinstance(expected_cache_epoch_id, (int, str))
        ):
            raise ValueError("expected cache_epoch_id must be a string, integer, or None")
        with self._lock:
            plan = self.prepare_unconsumed_source_text_locked(
                source_token_seq,
                text_piece,
                expected_revision_id=expected_revision_id,
                expected_cache_epoch_id=expected_cache_epoch_id,
            )
            return self.install_unconsumed_source_text_locked(plan)

    def prepare_unconsumed_source_text_locked(
        self,
        source_token_seq: int,
        text_piece: str,
        *,
        expected_revision_id: str,
        expected_cache_epoch_id: str | int | None,
    ) -> ResponseLedgerSourceTextUpdatePlan:
        """Prepare a text-only source row update; caller holds ``_lock``.

        All frontier, revision, cache, closed-unit, and identity checks happen
        before any ledger mutation. Installation below only assigns the two
        prebuilt registries so it can join the stream's atomic final-text CAS.
        The caller must retain ``_lock`` continuously through the coordinated
        installation; this method is the validation phase, not a standalone
        optimistic plan that can be installed after releasing the lock.
        """
        owns_lock = getattr(self._lock, "_is_owned", None)
        if callable(owns_lock) and not owns_lock():
            raise RuntimeError("source text update prepare requires its ledger lock")
        if not isinstance(text_piece, str):
            raise TypeError("source text_piece must be str")
        if not isinstance(expected_revision_id, str) or not expected_revision_id:
            raise ValueError("expected revision_id must be nonempty")
        if expected_cache_epoch_id is not None and (
            isinstance(expected_cache_epoch_id, bool)
            or not isinstance(expected_cache_epoch_id, (int, str))
        ):
            raise ValueError("expected cache_epoch_id must be a string, integer, or None")
        expected_epoch = (
            None if expected_cache_epoch_id is None else str(expected_cache_epoch_id)
        )
        if (
            self.revision_id != expected_revision_id
            or self.cache_epoch_id != expected_epoch
        ):
            raise RuntimeError("source text update lost its revision or cache epoch")
        source_seq = self._require_source_token_seq_locked(
            source_token_seq, "source text update"
        )
        if source_seq <= self._frozen_source_frontier_locked():
            raise RuntimeError("source text is beyond the immutable frontier")
        if any(
            any(
                isinstance(row.get("source_token_seq"), int)
                and not isinstance(row.get("source_token_seq"), bool)
                and int(row["source_token_seq"]) == source_seq
                for row in segment.get("closed_tokens", ())
            )
            for segment in self._segments.values()
        ):
            raise RuntimeError("source text belongs to a closed TTS unit")

        row = self._source_tokens[source_seq]
        row_revision = str(row.get("revision_id", ""))
        row_epoch_raw = row.get("cache_epoch_id")
        row_epoch = None if row_epoch_raw is None else str(row_epoch_raw)
        if row_revision != expected_revision_id or row_epoch != expected_epoch:
            raise RuntimeError("source text row does not match its revision or cache epoch")
        previous_text = str(row["text_piece"])
        if previous_text == text_piece:
            return ResponseLedgerSourceTextUpdatePlan(
                source_token_seq=source_seq,
                expected_revision_id=expected_revision_id,
                expected_cache_epoch_id=expected_epoch,
                changed=False,
                _source_tokens=self._source_tokens,
                _source_identity_to_seq=self._source_identity_to_seq,
            )

        replacement_row = dict(row)
        replacement_row["text_piece"] = text_piece
        new_source_tokens = dict(self._source_tokens)
        new_source_tokens[source_seq] = replacement_row
        new_identity_map = {
            identity: existing_sequence
            for identity, existing_sequence in self._source_identity_to_seq.items()
            if existing_sequence != source_seq
        }
        new_identity = (
            row_revision,
            row_epoch,
            row.get("physical_kv_cursor"),
            int(row["token_id"]),
            text_piece,
        )
        existing_identity_sequence = new_identity_map.get(new_identity)
        if existing_identity_sequence is None or existing_identity_sequence == source_seq:
            new_identity_map[new_identity] = source_seq
        return ResponseLedgerSourceTextUpdatePlan(
            source_token_seq=source_seq,
            expected_revision_id=expected_revision_id,
            expected_cache_epoch_id=expected_epoch,
            changed=True,
            _source_tokens=new_source_tokens,
            _source_identity_to_seq=new_identity_map,
        )

    def install_unconsumed_source_text_locked(
        self,
        plan: ResponseLedgerSourceTextUpdatePlan,
    ) -> bool:
        """Install a prepared text completion using assignments only.

        The caller holds ``_lock`` continuously from prepare through the
        cross-object install. All validation and allocation belongs in
        ``prepare_unconsumed_source_text_locked``.
        """
        self._source_tokens = plan._source_tokens
        self._source_identity_to_seq = plan._source_identity_to_seq
        return plan.changed

    def record_tts_published(self, source_token_seq: int) -> None:
        """Freeze source text as soon as it is published to the TTS consumer."""
        with self._lock:
            source_seq = self._require_source_token_seq_locked(source_token_seq, "TTS published")
            if source_seq < self._tts_published_source_seq:
                raise ValueError("TTS published source frontier cannot move backwards")
            self._tts_published_source_seq = source_seq

    def record_tts_reserved(self, source_token_seq: int) -> None:
        with self._lock:
            source_seq = self._require_source_token_seq_locked(source_token_seq, "TTS reserved")
            if source_seq < self._tts_reserved_source_seq:
                raise ValueError("TTS reserved source frontier cannot move backwards")
            self._tts_published_source_seq = max(self._tts_published_source_seq, source_seq)
            self._tts_reserved_source_seq = source_seq

    def record_tts_consumed(self, source_token_seq: int) -> None:
        with self._lock:
            source_seq = self._require_source_token_seq_locked(source_token_seq, "TTS consumed")
            if source_seq < self._tts_consumed_source_seq:
                raise ValueError("TTS consumed source frontier cannot move backwards")
            self._tts_published_source_seq = max(self._tts_published_source_seq, source_seq)
            self._tts_reserved_source_seq = max(self._tts_reserved_source_seq, source_seq)
            self._tts_consumed_source_seq = source_seq

    def record_pcm_synthesized(self, sample_count: int) -> None:
        count = int(sample_count)
        if count < 0:
            raise ValueError("synthesized sample count cannot be negative")
        with self._lock:
            self._pcm_samples_synthesized += count

    def record_pcm_sent(self, packet: Any) -> None:
        try:
            segment_id = str(packet.speaker_segment_id)
            start = int(packet.start_sample)
            end = int(packet.end_sample)
            generation = int(packet.generation_id)
            packet_id = int(packet.packet_id)
            pcm = np.asarray(packet.pcm, dtype=np.float32).reshape(-1)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("sent PCM packet lacks sample identity or data") from exc
        if (
            generation != self.generation_id
            or not segment_id
            or start < 0
            or end <= start
            or end - start != pcm.size
            or not np.isfinite(pcm).all()
        ):
            raise ValueError("sent PCM packet has an invalid identity or range")
        with self._lock:
            segment = self._segments.setdefault(segment_id, {
                "start_sample": start,
                "sent_end_sample": start,
                "consumed_end_sample": start,
                "closed_end_sample": None,
                "closed_tokens": (),
                "confirmed": False,
            })
            if segment.get("closed_end_sample") is not None:
                raise ValueError("sent PCM arrived after its TTS unit was closed")
            if start != int(segment["sent_end_sample"]) or start < int(segment["start_sample"]):
                raise ValueError("sent PCM segment range is not contiguous")
            segment["sent_end_sample"] = end
            segment.setdefault("packet_ranges", []).append((packet_id, start, end))
            self._pcm_samples_sent += end - start

    def record_closed_tts_unit(
        self,
        *,
        speaker_segment_id: str,
        end_sample: int,
        source_tokens: Sequence[Mapping[str, Any]],
    ) -> None:
        segment_id = str(speaker_segment_id)
        source_sequences = self.record_generated_tokens(source_tokens)
        if len(source_sequences) != len(source_tokens):
            raise ValueError("closed TTS unit includes an invalid source token")
        normalized_tokens = tuple(
            {**dict(row), "source_token_seq": source_seq}
            for row, source_seq in zip(source_tokens, source_sequences, strict=True)
        )
        with self._lock:
            segment = self._segments.get(segment_id)
            if segment is None or int(segment["sent_end_sample"]) != int(end_sample):
                raise ValueError("closed TTS unit does not match the sent PCM frontier")
            existing_end = segment.get("closed_end_sample")
            existing_tokens = segment.get("closed_tokens", ())
            if existing_end is not None and (
                int(existing_end) != int(end_sample) or tuple(existing_tokens) != normalized_tokens
            ):
                raise ValueError("closed TTS unit identity was reused with different source tokens")
            segment["closed_end_sample"] = int(end_sample)
            segment["closed_tokens"] = normalized_tokens
            segment["confirmed"] = int(segment["consumed_end_sample"]) == int(end_sample)

    def validate_pcm_consumed(
        self,
        *,
        speaker_segment_id: str,
        start_sample: int,
        end_sample: int,
        pcm: np.ndarray,
        sequence: int,
    ) -> bool:
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if (
            isinstance(start_sample, bool)
            or isinstance(end_sample, bool)
            or isinstance(sequence, bool)
            or not isinstance(start_sample, int)
            or not isinstance(end_sample, int)
            or not isinstance(sequence, int)
            or start_sample < 0
            or end_sample < start_sample
            or end_sample - start_sample != samples.size
            or not np.isfinite(samples).all()
            or sequence < 0
        ):
            return False
        with self._lock:
            segment = self._segments.get(str(speaker_segment_id))
            return bool(
                segment is not None
                and start_sample == int(segment["consumed_end_sample"])
                and end_sample <= int(segment["sent_end_sample"])
            )

    def record_pcm_consumed(
        self,
        *,
        speaker_segment_id: str,
        start_sample: int,
        end_sample: int,
        pcm: np.ndarray,
        sequence: int,
        final: bool,
        on_commit: (
            Callable[[], Any] | Callable[[ResponseLedgerProjection], Any] | None
        ) = None,
    ) -> None:
        segment_id = str(speaker_segment_id)
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        with self._lock:
            if not self.validate_pcm_consumed(
                speaker_segment_id=segment_id,
                start_sample=start_sample,
                end_sample=end_sample,
                pcm=samples,
                sequence=sequence,
            ):
                raise ValueError("consumed PCM range is stale, unknown, or beyond sent PCM")
            segment = self._segments[segment_id]
            segment_snapshot = dict(segment)
            pcm_samples_consumed = self._pcm_samples_consumed
            retained_pcm_samples = self._retained_pcm_samples
            retained_pcm_truncated = self._retained_pcm_truncated
            consumed_chunk_count = len(self._consumed_pcm_chunks)
            try:
                segment["consumed_end_sample"] = int(end_sample)
                segment["last_consumption_sequence"] = int(sequence)
                self._pcm_samples_consumed += int(samples.size)
                retained = min(
                    int(samples.size),
                    max(0, MAX_RETAINED_RESPONSE_PCM_SAMPLES - self._retained_pcm_samples),
                )
                if retained:
                    self._consumed_pcm_chunks.append({
                        "speaker_segment_id": segment_id,
                        "start_sample": int(start_sample),
                        "end_sample": int(start_sample + retained),
                        "sample_rate": 24_000,
                        "sequence": int(sequence),
                        "pcm_f32_le": np.asarray(samples[:retained], dtype="<f4").tobytes(order="C"),
                        "consumption_semantics": "logical_client_device_buffer",
                    })
                    self._retained_pcm_samples += retained
                if retained < samples.size:
                    self._retained_pcm_truncated = True
                closed_end = segment.get("closed_end_sample")
                if closed_end is not None and int(end_sample) == int(closed_end):
                    segment["confirmed"] = True
                if final:
                    segment["client_final"] = True
                if on_commit is not None:
                    try:
                        on_commit()
                    except TypeError as callback_error:
                        try:
                            callback_signature = inspect.signature(on_commit)
                        except (TypeError, ValueError):
                            raise callback_error
                        try:
                            callback_signature.bind()
                        except TypeError:
                            on_commit(self.project())
                        else:
                            raise callback_error
            except Exception:
                segment.clear()
                segment.update(segment_snapshot)
                self._pcm_samples_consumed = pcm_samples_consumed
                self._retained_pcm_samples = retained_pcm_samples
                self._retained_pcm_truncated = retained_pcm_truncated
                del self._consumed_pcm_chunks[consumed_chunk_count:]
                raise

    def close_response(
        self,
        *,
        intended_text: str,
        finish_reason: str | None,
        completed: bool,
        cancelled: bool = False,
    ) -> None:
        if not isinstance(intended_text, str):
            raise TypeError("intended_text must be str")
        with self._lock:
            self._intended_text = intended_text
            self._finish_reason = None if finish_reason is None else str(finish_reason)
            self._response_done = bool(completed)
            self._response_cancelled = bool(cancelled)

    def project(self) -> ResponseLedgerProjection:
        with self._lock:
            source_tokens = tuple(
                dict(self._source_tokens[key]) for key in sorted(self._source_tokens)
            )
            all_segments_closed = bool(self._segments) and all(
                segment.get("closed_end_sample") is not None
                for segment in self._segments.values()
            )
            all_pcm_consumed = bool(
                self._response_done
                and not self._response_cancelled
                and self._pcm_samples_sent > 0
                and all_segments_closed
                and self._pcm_samples_consumed == self._pcm_samples_sent
            )
            confirmed_source_seqs = {
                int(row["source_token_seq"])
                for segment in self._segments.values()
                if segment.get("confirmed") is True
                for row in segment.get("closed_tokens", ())
                if isinstance(row.get("source_token_seq"), int)
                and not isinstance(row.get("source_token_seq"), bool)
            }
            confirmed_tokens = tuple(
                row for row in source_tokens
                if (
                    int(row["source_token_seq"]) in confirmed_source_seqs
                    and int(row["source_token_seq"]) <= self._tts_consumed_source_seq
                )
            )
            has_complete_text_source = bool(
                self._finish_reason != "length"
                and (not self._intended_text or source_tokens)
                and all(
                    int(row["source_token_seq"]) in confirmed_source_seqs
                    and int(row["source_token_seq"]) <= self._tts_consumed_source_seq
                    for row in source_tokens
                )
            )
            confirmed_text = (
                self._intended_text
                if all_pcm_consumed and has_complete_text_source
                else "".join(str(row.get("text_piece", "")) for row in confirmed_tokens)
            )
            status = (
                "completed"
                if all_pcm_consumed and has_complete_text_source
                else "partial" if confirmed_text else "unknown" if self._intended_text else "absent"
            )
            physical_rows = [
                (int(row["source_token_seq"]), row.get("physical_kv_cursor"),
                 row.get("revision_id"), row.get("cache_epoch_id"),
                 tuple(row.get("physical_kv_provenance", ())))
                for row in source_tokens
            ]
            physical_cursor_contiguous = bool(physical_rows) and all(
                isinstance(cursor, int)
                and not isinstance(cursor, bool)
                and isinstance(previous_cursor, int)
                and not isinstance(previous_cursor, bool)
                and seq == previous_seq + 1
                and cursor == previous_cursor + 1
                and revision == previous_revision
                and epoch == previous_epoch
                and len({
                    (item.get("revision_id"), item.get("cache_epoch_id"), item.get("physical_kv_cursor"))
                    for item in provenance
                }) <= 1
                and len({
                    (item.get("revision_id"), item.get("cache_epoch_id"), item.get("physical_kv_cursor"))
                    for item in previous_provenance
                }) <= 1
                for (previous_seq, previous_cursor, previous_revision, previous_epoch, previous_provenance),
                    (seq, cursor, revision, epoch, provenance) in zip(physical_rows, physical_rows[1:])
            )
            if len(physical_rows) == 1:
                provenance = physical_rows[0][4]
                physical_cursor_contiguous = bool(
                    isinstance(physical_rows[0][1], int)
                    and len({
                        (item.get("revision_id"), item.get("cache_epoch_id"), item.get("physical_kv_cursor"))
                        for item in provenance
                    }) <= 1
                )
            generated_cursor = next((
                int(row["physical_kv_cursor"])
                for row in reversed(source_tokens)
                if isinstance(row.get("physical_kv_cursor"), int)
                and not isinstance(row.get("physical_kv_cursor"), bool)
            ), None)
            confirmed_cursor = next((
                int(row["physical_kv_cursor"])
                for row in reversed(confirmed_tokens)
                if isinstance(row.get("physical_kv_cursor"), int)
                and not isinstance(row.get("physical_kv_cursor"), bool)
            ), None)
            return ResponseLedgerProjection(
                response_id=self.response_id,
                generation_id=self.generation_id,
                revision_id=self.revision_id,
                cache_epoch_id=self.cache_epoch_id,
                source_input_watermark=self.source_input_watermark,
                source_tokens=source_tokens,
                confirmed_source_tokens=tuple(dict(row) for row in confirmed_tokens),
                generated_exclusive_kv_cursor=generated_cursor,
                confirmed_exclusive_kv_cursor=confirmed_cursor,
                physical_cursor_contiguous=physical_cursor_contiguous,
                intended_text=self._intended_text,
                confirmed_text=confirmed_text,
                text_status=status,
                response_done=self._response_done,
                response_cancelled=self._response_cancelled,
                finish_reason=self._finish_reason,
                pcm_samples_synthesized=self._pcm_samples_synthesized,
                pcm_samples_sent=self._pcm_samples_sent,
                pcm_samples_consumed=self._pcm_samples_consumed,
                all_pcm_consumed=all_pcm_consumed,
                retained_pcm_samples=self._retained_pcm_samples,
                retained_pcm_truncated=self._retained_pcm_truncated,
                consumed_pcm_chunks=tuple(dict(row) for row in self._consumed_pcm_chunks),
                logical_source_token_seq=self._next_source_seq - 1,
                tts_reserved_source_seq=self._tts_reserved_source_seq,
                tts_consumed_source_seq=self._tts_consumed_source_seq,
                history_order=self.history_order,
                history_mutation_sequence_at_start=self.history_mutation_sequence_at_start,
            )
