"""Confirmed-history preparation for native listener cache compaction.

History snapshots and chat-template tokenization are prepared away from the
listener worker. Its synchronous builder selects an immutable prepared prefix,
verifies explicit input-class/ASR coverage before the full recent-Q2 ring, and
checks that the ring ends at the listener watermark.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Mapping

from duplex_kernel.listener_microbatch_worker import (
    CompactionBuilderCapacityError,
    CompactionRequestContext,
    ListenerSessionSnapshot,
    MAX_RECENT_LISTENER_Q2_FRAMES,
)
from duplex_kernel.native_control6_prompt import (
    NATIVE_CONTROL6_PREFIX_MAX_TOKENS,
    NATIVE_CONTROL6_SYSTEM_INSTRUCTION_IT,
    validate_native_control6_prefix_ids,
)
from duplex_kernel.server_silence_skip import _build_unified_listener_prefix_token_ids


MAX_INPUT_CLASSIFICATION_RUNS = 32_768


@dataclass(frozen=True, slots=True)
class _VerifiedSpan:
    message_index: int
    first_frame_id: int
    last_frame_id: int
    timeline_id: str
    state_epoch_id: str
    history_revision: int
    history_mutation_sequence: int


@dataclass(frozen=True, slots=True)
class _PreparedCandidate:
    timeline_id: str
    history_covered_through_frame_id: int | None
    prefix_token_ids: tuple[int, ...]
    included_history: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _PreparedSessionHistory:
    session_id: str
    timeline_id: str
    history_revision: str | int
    history_mutation_sequence: int
    candidates: tuple[_PreparedCandidate, ...]
    asr_spans: tuple[_VerifiedSpan, ...]


@dataclass(frozen=True, slots=True)
class _InputClassificationRun:
    first_frame_id: int
    last_frame_id: int
    voiced: bool
    state_epoch_id: str


@dataclass(slots=True)
class _SessionEntry:
    session: Any
    generation_id: int
    latest_requested_key: tuple[int, int, int | str]
    latest_requested_timeline_id: str
    prepared: _PreparedSessionHistory | None = None
    preparation_failure: tuple[tuple[int, int, int | str], str, bool] | None = None
    input_runs_by_timeline: dict[str, list[_InputClassificationRun]] = field(default_factory=dict)
    last_input_by_timeline: dict[str, tuple[int, bool, str]] = field(default_factory=dict)
    input_classification_errors: dict[str, str] = field(default_factory=dict)


class SessionListenerCompactionAdapter:
    """Prepare confirmed-history prefixes and expose a bounded worker builder.

    ``prepare_snapshot`` performs tokenizer work and must run outside the
    listener worker. ``build_context`` reads immutable prepared data, checks the
    bounded Q2 ring, and scans run-length encoded CPU coverage intervals.
    """

    def __init__(self, engine: Any, *, unified_kv_enabled: bool):
        if engine is None:
            raise ValueError("listener compaction adapter requires the loaded engine")
        if not isinstance(unified_kv_enabled, bool):
            raise TypeError("unified_kv_enabled must be bool")
        self._engine = engine
        self._unified_kv_enabled = unified_kv_enabled
        self._lock = threading.Lock()
        self._tokenizer_lock = threading.Lock()
        self._sessions: dict[str, _SessionEntry] = {}

    def register_session(self, session: Any, *, generation_id: int) -> None:
        session_id = str(getattr(session, "session_id", "")).strip()
        if not session_id:
            raise ValueError("listener compaction session requires a stable session_id")
        if isinstance(generation_id, bool) or not isinstance(generation_id, int) or generation_id < 0:
            raise ValueError("listener compaction generation_id must be non-negative")
        snapshotter = getattr(session, "snapshot_history_with_provenance", None)
        if not callable(snapshotter):
            raise RuntimeError("session must expose snapshot_history_with_provenance()")
        snapshot = snapshotter()
        sequence = _snapshot_sequence(snapshot)
        version_key = _snapshot_order(snapshot)
        timeline_id = _snapshot_timeline_id(snapshot)
        with self._lock:
            self._sessions[session_id] = _SessionEntry(
                session=session,
                generation_id=int(generation_id),
                latest_requested_key=version_key,
                latest_requested_timeline_id=timeline_id,
            )
        self.publish_snapshot(session, snapshot)

    def unregister_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(str(session_id), None)

    def record_input_frame_classification(
        self,
        session_id: str,
        state_epoch_id: str,
        frame_id: int,
        voiced: bool,
        *,
        timeline_id: str | None = None,
    ) -> bool:
        """Retain raw CPU VAD classes across physical epochs in one input timeline."""
        session_id = str(session_id)
        epoch_id = str(state_epoch_id).strip()
        timeline_id = str(timeline_id or session_id).strip()
        if (
            not epoch_id
            or not timeline_id
            or isinstance(frame_id, bool)
            or not isinstance(frame_id, int)
            or frame_id < 0
            or not isinstance(voiced, bool)
        ):
            with self._lock:
                entry = self._sessions.get(session_id)
                if entry is not None:
                    entry.input_classification_errors[timeline_id or "unknown"] = (
                        "invalid_listener_input_classification"
                    )
            return False
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return False
            if timeline_id != entry.latest_requested_timeline_id:
                entry.input_classification_errors[
                    entry.latest_requested_timeline_id
                ] = "listener_input_timeline_snapshot_not_published"
                return False
            runs = entry.input_runs_by_timeline.setdefault(timeline_id, [])
            previous = entry.last_input_by_timeline.get(timeline_id)
            if previous is not None:
                previous_id, previous_voiced, previous_epoch = previous
                if frame_id == previous_id:
                    if previous_voiced != voiced or previous_epoch != epoch_id:
                        entry.input_classification_errors[timeline_id] = (
                            "listener_input_classification_changed_for_frame"
                        )
                        return False
                    return True
                if frame_id != previous_id + 1:
                    entry.input_classification_errors[timeline_id] = (
                        "listener_input_classification_has_unclassified_frames"
                    )
                    return False
            elif frame_id != 0:
                entry.input_classification_errors[timeline_id] = (
                    "listener_input_classification_missing_initial_frames"
                )
                return False
            if (
                runs
                and previous is not None
                and frame_id == previous[0] + 1
                and runs[-1].voiced == voiced
                and runs[-1].state_epoch_id == epoch_id
            ):
                last_run = runs[-1]
                runs[-1] = replace(last_run, last_frame_id=frame_id)
            else:
                runs.append(_InputClassificationRun(frame_id, frame_id, voiced, epoch_id))
                if len(runs) > MAX_INPUT_CLASSIFICATION_RUNS:
                    entry.input_classification_errors[timeline_id] = (
                        "listener_input_classification_capacity_exceeded"
                    )
                    return False
            entry.last_input_by_timeline[timeline_id] = (frame_id, voiced, epoch_id)
            return True

    def record_session_input_frame_classification(
        self,
        session: Any,
        frame: Any,
        voiced: bool,
    ) -> bool:
        """Record scheduler PCM using the owning session's listener identity."""
        session_id = getattr(session, "session_id", None)
        listener_state = getattr(session, "listener_state", None)
        state_epoch_id = getattr(listener_state, "state_epoch_id", None)
        if not isinstance(session_id, str) or not session_id.strip():
            raise RuntimeError("listener input classification requires an owning session ID")
        if not isinstance(state_epoch_id, str) or not state_epoch_id.strip():
            raise RuntimeError("listener input classification requires a current state epoch")
        frame_id = getattr(frame, "frame_id", None)
        timeline_id = getattr(session, "timeline_id", session_id)
        return self.record_input_frame_classification(
            session_id,
            state_epoch_id,
            frame_id,
            voiced,
            timeline_id=timeline_id,
        )

    def mark_pending(self, session_id: str, snapshot: Any) -> bool:
        """Record a newer coherent snapshot before off-thread tokenization starts."""
        session_id = str(session_id)
        version_key = _snapshot_order(snapshot)
        timeline_id = _snapshot_timeline_id(snapshot)
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return False
            if str(getattr(snapshot, "session_id", "")) != session_id:
                raise ValueError("history snapshot belongs to a different session")
            if timeline_id != entry.latest_requested_timeline_id:
                entry.latest_requested_timeline_id = timeline_id
                entry.latest_requested_key = version_key
                entry.prepared = None
                entry.preparation_failure = None
                entry.input_runs_by_timeline.clear()
                entry.last_input_by_timeline.clear()
                entry.input_classification_errors.clear()
                return True
            if version_key <= entry.latest_requested_key:
                return False
            entry.latest_requested_key = version_key
            entry.preparation_failure = None
            return True

    def publish_snapshot(self, session: Any, snapshot: Any) -> bool:
        """Build prefix variants from a detached snapshot; call via ``to_thread``."""
        session_id = str(getattr(snapshot, "session_id", "")).strip()
        timeline_id = _snapshot_timeline_id(snapshot)
        if not session_id or str(getattr(session, "session_id", "")) != session_id:
            raise ValueError("history snapshot/session identity mismatch")
        version_key = _snapshot_order(snapshot)
        try:
            prepared = self._prepare_snapshot(session, snapshot)
        except Exception as exc:
            with self._lock:
                entry = self._sessions.get(session_id)
                if entry is None or entry.session is not session:
                    return False
                if timeline_id != entry.latest_requested_timeline_id:
                    return False
                if version_key < entry.latest_requested_key:
                    return False
                entry.latest_requested_key = max(entry.latest_requested_key, version_key)
                entry.preparation_failure = (
                    version_key,
                    str(exc)[:240] or type(exc).__name__,
                    isinstance(exc, CompactionBuilderCapacityError),
                )
            return True
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None or entry.session is not session:
                return False
            if timeline_id != entry.latest_requested_timeline_id:
                return False
            if version_key < entry.latest_requested_key or (
                entry.prepared is not None
                and _prepared_order(entry.prepared) > version_key
            ):
                return False
            entry.latest_requested_key = max(entry.latest_requested_key, version_key)
            entry.prepared = prepared
            entry.preparation_failure = None
            return True

    def build_context(
        self,
        listener_snapshot: ListenerSessionSnapshot,
        recent_q2: tuple[Any, ...],
        history_revision: str | None,
    ) -> CompactionRequestContext | None:
        """Select a prefix proven to cover every evicted voice frame."""
        session_id = str(getattr(listener_snapshot, "session_id", ""))
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                raise CompactionBuilderCapacityError("session_history_builder_not_registered")
            prepared = entry.prepared
            failure = entry.preparation_failure
            latest_key = entry.latest_requested_key
            if prepared is None:
                if failure is not None and failure[0] >= latest_key:
                    _, reason, is_capacity = failure
                    if is_capacity:
                        raise CompactionBuilderCapacityError(reason)
                    raise RuntimeError(reason)
                return None
            if _prepared_order(prepared) < latest_key:
                if failure is not None and failure[0] >= latest_key:
                    _, reason, is_capacity = failure
                    if is_capacity:
                        raise CompactionBuilderCapacityError(reason)
                    raise RuntimeError(reason)
                return None
            expected_generation = entry.generation_id
            runs = tuple(
                entry.input_runs_by_timeline.get(prepared.timeline_id, ())
            )
            classification_error = entry.input_classification_errors.get(
                prepared.timeline_id
            )

        # If the worker command raced ahead of an already-published newer
        # history revision, return that coherent newer context. The worker's
        # revision CAS discards it for the stale request and waits for the
        # callback's request carrying the new revision; this is not capacity.
        del history_revision

        state_epoch_id = str(getattr(listener_snapshot, "state_epoch_id", ""))
        next_frame_id = _nonnegative_int(
            getattr(listener_snapshot, "next_frame_id", None), "listener next_frame_id"
        )
        generation_id = _nonnegative_int(
            getattr(listener_snapshot, "generation_id", None), "listener generation_id"
        )
        if generation_id != expected_generation:
            raise CompactionBuilderCapacityError("listener_generation_changed_during_compaction")

        ring = tuple(recent_q2)
        ring_ids: list[int] = []
        for item in ring:
            frame = getattr(item, "frame", None)
            if frame is None:
                raise CompactionBuilderCapacityError("recent_q2_entry_missing_frame")
            if str(getattr(frame, "session_id", "")) != session_id:
                raise CompactionBuilderCapacityError("recent_q2_contains_another_session")
            if _nonnegative_int(getattr(frame, "generation_id", None), "Q2 generation_id") != generation_id:
                raise CompactionBuilderCapacityError("recent_q2_generation_mismatch")
            if str(getattr(frame, "state_epoch_id", "")) != state_epoch_id:
                raise CompactionBuilderCapacityError("recent_q2_epoch_mismatch")
            ring_ids.append(_nonnegative_int(getattr(frame, "frame_id", None), "Q2 frame_id"))
        expected_ring_length = min(next_frame_id, MAX_RECENT_LISTENER_Q2_FRAMES)
        expected_ring_start = next_frame_id - expected_ring_length
        expected_ring_ids = list(range(expected_ring_start, next_frame_id))
        if ring_ids != expected_ring_ids:
            raise CompactionBuilderCapacityError(
                "recent_q2_ring_does_not_cover_full_listener_window"
            )

        # Keep all of the recent Q2 window. Only frames strictly before it can
        # be represented by archived CPU ASR plus explicit per-frame nonvoice.
        ring_start = expected_ring_start
        cutoff = ring_start - 1
        if cutoff >= 0:
            _validate_evicted_input_coverage(
                runs,
                classification_error=classification_error,
                asr_spans=prepared.asr_spans,
                timeline_id=prepared.timeline_id,
                cutoff_frame_id=cutoff,
            )
        eligible = [
            candidate
            for candidate in prepared.candidates
            if candidate.timeline_id == prepared.timeline_id
            and (
                candidate.history_covered_through_frame_id is None
                or candidate.history_covered_through_frame_id <= cutoff
            )
        ]
        if not eligible:
            raise CompactionBuilderCapacityError("no_history_prefix_for_input_timeline")
        candidate = max(
            eligible,
            key=lambda item: (
                -1 if item.history_covered_through_frame_id is None
                else item.history_covered_through_frame_id
            ),
        )
        replay_after_frame_id = None if cutoff < 0 else cutoff

        history_context = (
            ("history_revision", str(prepared.history_revision)),
            ("history_mutation_sequence", prepared.history_mutation_sequence),
            ("input_timeline_id", prepared.timeline_id),
            ("replay_after_frame_id", replay_after_frame_id),
            ("input_classified_through_frame_id", cutoff),
            ("included_history", candidate.included_history),
        )
        return CompactionRequestContext(
            prefix_token_ids=candidate.prefix_token_ids,
            history_context=history_context,
            history_revision=prepared.history_revision,
            replay_after_frame_id=replay_after_frame_id,
        )

    def _prepare_snapshot(self, session: Any, snapshot: Any) -> _PreparedSessionHistory:
        session_id = str(getattr(snapshot, "session_id", ""))
        timeline_id = _snapshot_timeline_id(snapshot)
        history_revision = _snapshot_revision(snapshot)
        history_sequence = _snapshot_sequence(snapshot)
        messages = tuple(_copy_history_message(item) for item in getattr(snapshot, "history", ()))
        history_item_ids = tuple(getattr(snapshot, "history_item_ids", ()))
        current_index_by_item_id = {
            item_id: index
            for index, item_id in enumerate(history_item_ids)
            if _optional_nonnegative_int(item_id) is not None
        }
        provenance = tuple(getattr(snapshot, "provenance", ()))
        records_by_index: dict[int, list[Any]] = {}
        verified: list[_VerifiedSpan] = []
        for record in provenance:
            item_id = _optional_nonnegative_int(getattr(record, "history_item_id", None))
            message_index = current_index_by_item_id.get(item_id)
            if message_index is None:
                continue
            records_by_index.setdefault(message_index, []).append(record)
            span = _verified_history_span(
                record,
                session_id=session_id,
                message_index=message_index,
                history_item_ids=history_item_ids,
                messages=messages,
                history_revision=history_revision,
                history_mutation_sequence=history_sequence,
            )
            if span is not None:
                verified.append(span)

        verified = [
            span for span in verified
            if len(records_by_index.get(span.message_index, ())) == 1
        ]

        candidates: list[_PreparedCandidate] = []
        eligible_asr_spans: list[_VerifiedSpan] = []
        timeline_spans = sorted(
            (span for span in verified if span.timeline_id == timeline_id),
            key=lambda span: (span.first_frame_id, span.last_frame_id),
        )
        covered_message_indices: set[int] = set()
        previous_last_frame_id = -1
        previous_message_index = -1
        previous_sequence = -1
        for span in timeline_spans:
            if (
                span.first_frame_id <= previous_last_frame_id
                or span.message_index <= previous_message_index
                or span.history_mutation_sequence <= previous_sequence
            ):
                break
            previous_last_frame_id = span.last_frame_id
            previous_message_index = span.message_index
            previous_sequence = span.history_mutation_sequence
            eligible_asr_spans.append(span)
            covered_message_indices.add(span.message_index)
            candidates.append(self._prepare_candidate(
                session,
                snapshot,
                messages,
                records_by_index,
                timeline_id,
                span.last_frame_id,
                set(covered_message_indices),
            ))

        # Same-timeline sources enter the prefix only when their represented
        # content covers the chosen replay cutoff.
        candidates.insert(0, self._prepare_candidate(
            session,
            snapshot,
            messages,
            records_by_index,
            timeline_id,
            None,
            set(),
        ))

        return _PreparedSessionHistory(
            session_id=session_id,
            timeline_id=timeline_id,
            history_revision=history_revision,
            history_mutation_sequence=history_sequence,
            candidates=tuple(candidates),
            asr_spans=tuple(eligible_asr_spans),
        )

    def _prepare_candidate(
        self,
        session: Any,
        snapshot: Any,
        messages: tuple[dict[str, Any], ...],
        records_by_index: Mapping[int, list[Any]],
        timeline_id: str,
        history_covered_through_frame_id: int | None,
        covered_message_indices: set[int],
    ) -> _PreparedCandidate:
        selected: list[tuple[int, dict[str, Any]]] = []
        for index, message in enumerate(messages):
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if role == "user" and index in records_by_index:
                # Coverage authorizes a cutoff only for content represented
                # in this prefix and belonging to its input timeline.
                rows = records_by_index[index]
                if len(rows) != 1:
                    continue
                record = rows[0]
                record_timeline = str(getattr(record, "timeline_id", "") or "").strip()
                if record_timeline != timeline_id:
                    # A transcript from another input timeline is confirmed
                    # conversational history, but never proves this timeline.
                    pass
                elif index not in covered_message_indices:
                    continue
            selected.append((index, {"role": role, "content": content}))

        complete_turns = _complete_turn_indices(messages)
        removed: set[int] = set()
        while True:
            filtered = [message for index, message in selected if index not in removed]
            try:
                with self._tokenizer_lock:
                    if self._unified_kv_enabled:
                        proxy = SimpleNamespace(
                            history=filtered,
                            instructions=getattr(snapshot, "instructions", None),
                            language=getattr(snapshot, "language", None),
                            listener=getattr(session, "listener", None),
                        )
                        prefix_ids = _build_unified_listener_prefix_token_ids(
                            self._engine,
                            proxy,
                            instruction_override=getattr(snapshot, "instructions", None),
                        )
                    else:
                        prefix_ids = _build_native_history_prefix_token_ids(
                            self._engine,
                            session,
                            instructions=getattr(snapshot, "instructions", None),
                            history=filtered,
                        )
                return _PreparedCandidate(
                    timeline_id=timeline_id,
                    history_covered_through_frame_id=history_covered_through_frame_id,
                    prefix_token_ids=tuple(int(value) for value in prefix_ids),
                    included_history=tuple(
                        (str(item["role"]), str(item["content"])) for item in filtered
                    ),
                )
            except RuntimeError as exc:
                if "prefix exceeds native prefix capacity" not in str(exc):
                    raise
                next_turn = next((turn for turn in complete_turns if not turn <= removed), None)
                if next_turn is None:
                    raise CompactionBuilderCapacityError(
                        "confirmed_history_and_instructions_exceed_prefix_capacity"
                    ) from exc
                removed.update(next_turn)


def _copy_history_message(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _build_native_history_prefix_token_ids(
    engine: Any,
    session: Any,
    *,
    instructions: str | None,
    history: list[dict[str, str]],
) -> tuple[int, ...]:
    """Serialize the native Control6 system/history/audio prefix without cuts."""
    tokenizer = getattr(engine, "tokenizer", None)
    processor = getattr(getattr(engine, "_thinker_wrapper", None), "processor", None)
    if tokenizer is None or not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise RuntimeError("native listener compaction requires the loaded Gemma chat template")
    if not callable(getattr(tokenizer, "encode", None)):
        raise RuntimeError("native listener compaction requires tokenizer.encode")
    marker = (
        getattr(processor, "audio_token", None)
        or getattr(tokenizer, "audio_token", None)
        or "<|audio|>"
    )
    marker = str(marker).strip()
    if not marker:
        raise RuntimeError("native listener compaction requires Gemma's audio marker")
    backend = getattr(getattr(session, "listener", None), "backend", None)
    boa_token_id = getattr(backend, "boa_token_id", None)
    if isinstance(boa_token_id, bool) or not isinstance(boa_token_id, int) or boa_token_id < 0:
        raise RuntimeError("native listener compaction requires Gemma's configured BOA token ID")
    instruction_text = str(instructions or "").strip()
    system_text = NATIVE_CONTROL6_SYSTEM_INSTRUCTION_IT
    if instruction_text:
        system_text += "\nIstruzioni della sessione: " + instruction_text
    messages = [
        {"role": "system", "content": system_text},
        *history,
        {"role": "user", "content": marker},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    token_ids = _flat_token_ids(encoded, "chat template")
    marker_ids = _flat_token_ids(
        tokenizer.encode(marker, add_special_tokens=False), "audio marker"
    )
    if len(marker_ids) != 1:
        raise RuntimeError("Gemma audio marker must map to one tokenizer token")
    marker_positions = [index for index, token_id in enumerate(token_ids) if token_id == marker_ids[0]]
    if not marker_positions:
        raise RuntimeError("Gemma chat template omitted the final listener audio marker")
    prefix_ids = token_ids[:marker_positions[-1]] + [int(boa_token_id)]
    if len(prefix_ids) > NATIVE_CONTROL6_PREFIX_MAX_TOKENS:
        raise RuntimeError(
            "native listener prefix exceeds native prefix capacity: "
            f"{len(prefix_ids)}>{NATIVE_CONTROL6_PREFIX_MAX_TOKENS}"
        )
    try:
        return validate_native_control6_prefix_ids(
            prefix_ids,
            boa_token_id=int(boa_token_id),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("native listener prefix token IDs are invalid") from exc


def _flat_token_ids(value: Any, label: str) -> list[int]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise RuntimeError(f"{label} tokenizer output has no input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise RuntimeError(f"{label} tokenizer output must be a list")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise RuntimeError(f"{label} tokenizer returned multiple conversations")
        value = value[0]
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} tokenizer output contains invalid IDs") from exc


def _snapshot_revision(snapshot: Any) -> str | int:
    value = getattr(snapshot, "history_revision", None)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("history snapshot revision must be a string or integer")
    if isinstance(value, str) and not value.strip() or isinstance(value, int) and value < 0:
        raise ValueError("history snapshot revision must be non-empty and non-negative")
    return value


def _snapshot_timeline_id(snapshot: Any) -> str:
    value = getattr(snapshot, "timeline_id", None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("history snapshot must expose a stable input timeline_id")
    return value.strip()


def _snapshot_sequence(snapshot: Any) -> int:
    return _nonnegative_int(
        getattr(snapshot, "history_mutation_sequence", None),
        "history mutation sequence",
    )


def _snapshot_order(snapshot: Any) -> tuple[int, int, int | str]:
    revision = _snapshot_revision(snapshot)
    sequence = _snapshot_sequence(snapshot)
    if isinstance(revision, int) and not isinstance(revision, bool):
        return (0, int(revision), sequence)
    return (1, int(sequence), str(revision))


def _prepared_order(prepared: _PreparedSessionHistory) -> tuple[int, int, int | str]:
    if isinstance(prepared.history_revision, int) and not isinstance(prepared.history_revision, bool):
        return (0, int(prepared.history_revision), prepared.history_mutation_sequence)
    return (
        1,
        int(prepared.history_mutation_sequence),
        str(prepared.history_revision),
    )


def _validate_evicted_input_coverage(
    runs: tuple[_InputClassificationRun, ...],
    *,
    classification_error: str | None,
    asr_spans: tuple[_VerifiedSpan, ...],
    timeline_id: str,
    cutoff_frame_id: int,
) -> None:
    """Require an explicit class for every evicted frame and ASR for all voice."""
    if classification_error is not None:
        raise CompactionBuilderCapacityError(classification_error)
    if cutoff_frame_id < 0:
        return
    if not runs:
        raise CompactionBuilderCapacityError(
            "listener_input_classification_missing_before_q2_ring"
        )

    cursor = 0
    voiced_runs: list[tuple[int, int]] = []
    for run in runs:
        if run.last_frame_id < cursor:
            continue
        if run.first_frame_id != cursor:
            raise CompactionBuilderCapacityError(
                "listener_input_classification_has_unclassified_frames"
            )
        covered_last = min(run.last_frame_id, cutoff_frame_id)
        if run.voiced:
            voiced_runs.append((cursor, covered_last))
        cursor = covered_last + 1
        if cursor > cutoff_frame_id:
            break
    if cursor <= cutoff_frame_id:
        raise CompactionBuilderCapacityError(
            "listener_input_classification_incomplete_through_q2_ring"
        )
    if not voiced_runs:
        return

    # Only full eligible spans ending before the cutoff can replace old audio.
    # A transcript whose utterance crosses into the retained ring is not split.
    spans = sorted(
        (
            span for span in asr_spans
            if span.timeline_id == timeline_id
            and span.last_frame_id <= cutoff_frame_id
        ),
        key=lambda span: (span.first_frame_id, span.last_frame_id),
    )
    span_cursor = 0
    for first, last in voiced_runs:
        cursor = first
        while cursor <= last:
            while span_cursor < len(spans) and spans[span_cursor].last_frame_id < cursor:
                span_cursor += 1
            if span_cursor >= len(spans):
                raise CompactionBuilderCapacityError(
                    "evicted_voiced_input_lacks_complete_cpu_asr_coverage"
                )
            span = spans[span_cursor]
            if span.first_frame_id > cursor:
                raise CompactionBuilderCapacityError(
                    "evicted_voiced_input_lacks_complete_cpu_asr_coverage"
                )
            cursor = min(last + 1, span.last_frame_id + 1)
            if cursor <= last and span_cursor + 1 < len(spans):
                span_cursor += 1
            elif cursor <= last:
                raise CompactionBuilderCapacityError(
                    "evicted_voiced_input_lacks_complete_cpu_asr_coverage"
                )


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CompactionBuilderCapacityError(f"invalid_{label.replace(' ', '_')}")
    return int(value)


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _verified_history_span(
    record: Any,
    *,
    session_id: str,
    message_index: int,
    history_item_ids: tuple[Any, ...],
    messages: tuple[dict[str, Any], ...],
    history_revision: str | int,
    history_mutation_sequence: int,
) -> _VerifiedSpan | None:
    archive_id = _optional_nonnegative_int(getattr(record, "archive_id", None))
    if (
        str(getattr(record, "session_id", "")) != session_id
        or _optional_nonnegative_int(getattr(record, "message_index", None)) is None
        or message_index >= len(messages)
        or _optional_nonnegative_int(getattr(record, "history_item_id", None))
        != _optional_nonnegative_int(history_item_ids[message_index])
        or messages[message_index].get("role") != "user"
        or not isinstance(messages[message_index].get("content"), str)
        or not messages[message_index].get("content", "").strip()
        or getattr(record, "coverage_complete", False) is not True
    ):
        return None
    if archive_id is None or archive_id < 1:
        return None
    try:
        if getattr(record, "compaction_eligible", False) is not True:
            return None
    except Exception:
        return None
    epoch_id = str(getattr(record, "state_epoch_id", "")).strip()
    timeline_id = str(getattr(record, "timeline_id", "")).strip()
    first = _optional_nonnegative_int(getattr(record, "first_frame_id", None))
    last = _optional_nonnegative_int(getattr(record, "last_frame_id", None))
    revision = _optional_nonnegative_int(getattr(record, "history_revision", None))
    sequence = _optional_nonnegative_int(getattr(record, "history_mutation_sequence", None))
    snapshot_revision = (
        history_revision if isinstance(history_revision, int) and not isinstance(history_revision, bool)
        else None
    )
    if (
        not epoch_id
        or not timeline_id
        or first is None
        or last is None
        or last < first
        or revision is None
        or sequence is None
        or sequence > history_mutation_sequence
        or (snapshot_revision is not None and revision > snapshot_revision)
    ):
        return None
    return _VerifiedSpan(
        message_index=message_index,
        first_frame_id=first,
        last_frame_id=last,
        timeline_id=timeline_id,
        state_epoch_id=epoch_id,
        history_revision=revision,
        history_mutation_sequence=sequence,
    )


def _complete_turn_indices(messages: tuple[dict[str, Any], ...]) -> tuple[frozenset[int], ...]:
    """Return only complete user-to-assistant history turns for safe trimming."""
    turns: list[frozenset[int]] = []
    index = 0
    while index < len(messages):
        if messages[index].get("role") != "user":
            index += 1
            continue
        next_user = next(
            (cursor for cursor in range(index + 1, len(messages))
             if messages[cursor].get("role") == "user"),
            len(messages),
        )
        assistants = [
            cursor for cursor in range(index + 1, next_user)
            if messages[cursor].get("role") == "assistant"
        ]
        if assistants:
            turns.append(frozenset(range(index, next_user)))
        index = max(index + 1, next_user)
    return tuple(turns)


__all__ = [
    "CompactionBuilderCapacityError",
    "SessionListenerCompactionAdapter",
]
