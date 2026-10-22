"""CPU-side coordination contracts for a single-owner listener microbatch worker.

The coordinator only groups complete real frames and snapshots private session
cursors.  A future worker supplies one batch processor that performs encoder,
native listener, and Control6 work; there is intentionally no per-row fallback.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Mapping
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from duplex_kernel.microbatch_contract import (
    BatchRowDescriptor,
    ListenerBatch,
    ListenerBatchManifest,
    ListenerFrame,
    ListenerSessionSnapshot,
    ListenerSessionState,
    LISTENER_MAX_SEQ_LEN,
    LISTENER_Q_PER_FRAME,
    MAX_COALESCE_WAIT_NS,
    NativeBatchCapability,
    RowStateUpdate,
    RuntimeSignature,
    snapshot_listener_state,
)


class BatchProcessor(Protocol):
    """One call consumes an exact N-row batch and returns per-row state commits."""

    def process_batch(
        self,
        batch: ListenerBatch,
        states: tuple[ListenerSessionState, ...],
    ) -> Sequence[RowStateUpdate]:
        ...


@dataclass(frozen=True, slots=True)
class RetiredSession:
    state: ListenerSessionState
    queued_frame_count: int


class ListenerMicrobatchCoordinator:
    """Group oldest ready frame per SID, with a strict maximum 2 ms coalesce.

    ``take_ready(now_ns)`` is deterministic and does not sleep, which makes the
    deadline decision directly testable.  The optional ``wait_and_take`` method
    provides the condition-variable wait used by a later single GPU-owner
    thread; it never waits beyond the earliest singleton coalesce deadline.
    """

    def __init__(
        self,
        runtime_signature: RuntimeSignature,
        capability: NativeBatchCapability,
        *,
        coalesce_window_ns: int = MAX_COALESCE_WAIT_NS,
    ) -> None:
        if not isinstance(runtime_signature, RuntimeSignature):
            raise TypeError("runtime_signature is required; baseline per-session fallback is disabled")
        if not isinstance(capability, NativeBatchCapability):
            raise TypeError("native batch capability is required; per-row fallback is disabled")
        capability.validate_for(runtime_signature)
        if capability.stream_priority != -1:
            raise ValueError(
                "the production listener dispatcher requires the listener-priority -1 stream; "
                "priority 0 is reserved for the semantic-history shadow runtime"
            )
        if isinstance(coalesce_window_ns, bool) or not isinstance(coalesce_window_ns, int):
            raise ValueError("coalesce_window_ns must be an integer")
        if not 0 <= coalesce_window_ns <= MAX_COALESCE_WAIT_NS:
            raise ValueError("coalescing must be bounded by 2 ms")
        self.runtime_signature = runtime_signature
        self.capability = capability
        self.coalesce_window_ns = coalesce_window_ns
        self._condition = threading.Condition()
        self._states: dict[str, ListenerSessionState] = {}
        self._queues: dict[str, deque[tuple[int, ListenerFrame]]] = {}
        self._next_enqueued_frame_id: dict[str, int] = {}
        self._inflight_sessions: set[str] = set()
        self._active_batch: ListenerBatch | None = None
        self._submission_order = 0
        self._failed_batches: dict[str, str] = {}
        self._gpu_owner_thread_id: int | None = None

    @property
    def ready_depth(self) -> int:
        with self._condition:
            return sum(len(queue) for queue in self._queues.values())

    @property
    def active_manifest_id(self) -> str | None:
        with self._condition:
            return None if self._active_batch is None else self._active_batch.manifest.manifest_id

    def register_session(self, state: ListenerSessionState) -> None:
        if not isinstance(state, ListenerSessionState):
            raise TypeError("state must be ListenerSessionState")
        if state.runtime_signature != self.runtime_signature:
            raise ValueError("session runtime signature differs from the shared batch runtime signature")
        if state.native_context_id is None:
            raise ValueError("session must have a native context before entering the batch worker")
        with self._condition:
            self._assert_owner_locked()
            if state.session_id in self._states:
                raise ValueError(f"session already registered: {state.session_id}")
            if state.closed or state.poisoned_reason is not None:
                raise ValueError("closed or poisoned listener state cannot be registered")
            self._assert_private_state_references(state)
            self._states[state.session_id] = state
            self._queues[state.session_id] = deque()
            self._next_enqueued_frame_id[state.session_id] = state.next_frame_id
            self._condition.notify_all()

    def _assert_private_state_references(self, candidate: ListenerSessionState) -> None:
        candidate_values = _mutable_state_values(candidate)
        for other in self._states.values():
            if candidate.native_context_id == other.native_context_id:
                raise ValueError("private state alias: native context handle is shared by sessions")
            left_context = candidate.native_context
            right_context = other.native_context
            if left_context is not None and right_context is not None:
                if _native_context_key(left_context) == _native_context_key(right_context):
                    raise ValueError("private state alias: native context is shared by sessions")
            other_values = _mutable_state_values(other)
            for left_name, left in candidate_values:
                for right_name, right in other_values:
                    if _payloads_alias(left, right):
                        raise ValueError(
                            "private state alias: "
                            f"{left_name} shares mutable storage with {right_name}"
                        )

    def enqueue(self, frame: ListenerFrame) -> None:
        if not isinstance(frame, ListenerFrame):
            raise TypeError("frame must be a complete ListenerFrame")
        with self._condition:
            state = self._states.get(frame.session_id)
            if state is None:
                raise KeyError(f"unknown listener session: {frame.session_id}")
            if state.closed:
                raise RuntimeError("listener session is closed")
            if state.poisoned_reason is not None:
                raise RuntimeError(f"listener session is poisoned: {state.poisoned_reason}")
            if (
                frame.generation_id != state.generation_id
                or frame.state_epoch_id != state.state_epoch_id
            ):
                raise ValueError("frame generation or epoch does not match listener state")
            expected = self._next_enqueued_frame_id[frame.session_id]
            if frame.frame_id != expected:
                raise ValueError(
                    f"frame id must equal next expected frame {expected}, got {frame.frame_id}"
                )
            self._queues[frame.session_id].append((self._submission_order, frame))
            self._submission_order += 1
            self._next_enqueued_frame_id[frame.session_id] = expected + 1
            self._condition.notify_all()

    def _candidate_heads_locked(self) -> list[tuple[int, ListenerFrame]]:
        heads = [
            queue[0]
            for session_id, queue in self._queues.items()
            if queue and session_id not in self._inflight_sessions
        ]
        return sorted(
            heads,
            key=lambda item: (item[1].ready_at_ns, item[0], item[1].session_id),
        )

    def _take_ready_locked(
        self,
        now_ns: int,
        *,
        before_manifest: Callable[[tuple[ListenerFrame, ...]], Mapping[str, int] | None] | None = None,
    ) -> ListenerBatch | None:
        if self._active_batch is not None:
            return None
        heads = [
            item for item in self._candidate_heads_locked()
            if item[1].ready_at_ns <= now_ns
        ]
        if not heads:
            return None
        batch_size = len(heads)
        oldest_ready_ns = heads[0][1].ready_at_ns
        if len(heads) == 1 and now_ns < oldest_ready_ns + self.coalesce_window_ns:
            return None
        if len(heads) == len(self._states):
            reason = "batch_full"
        elif len(heads) >= 2:
            reason = "multiple_ready"
        else:
            reason = "coalesce_deadline"

        chosen = heads[:batch_size]
        frames = tuple(item[1] for item in chosen)
        if before_manifest is not None:
            rebases = before_manifest(frames)
            if rebases is not None:
                if not isinstance(rebases, Mapping):
                    raise TypeError("before_manifest must return a session-to-cache-position mapping")
                ready_session_ids = {frame.session_id for frame in frames}
                if not set(rebases).issubset(ready_session_ids):
                    raise ValueError("before_manifest tried to rebase a session outside this tick")
                for session_id, cache_position in rebases.items():
                    if (
                        isinstance(cache_position, bool)
                        or not isinstance(cache_position, int)
                        or not 1 <= cache_position <= LISTENER_MAX_SEQ_LEN
                    ):
                        raise ValueError("promoted cache positions must be within native capacity")
                    state = self._states[session_id]
                    state.cache_position = cache_position
                    state.valid_history_tokens = cache_position
        rows: list[BatchRowDescriptor] = []
        rows_frames: list[ListenerFrame] = []
        for frame in frames:
            state = self._states[frame.session_id]
            if state.next_frame_id != frame.frame_id:
                raise RuntimeError(
                    "queued frame no longer matches the committed per-session frame cursor"
                )
            if state.cache_position > LISTENER_MAX_SEQ_LEN - LISTENER_Q_PER_FRAME:
                raise RuntimeError(
                    "native listener physical Q2 capacity exhausted before dispatch; "
                    f"session={state.session_id}, cache_position={state.cache_position}, "
                    f"max_seq_len={LISTENER_MAX_SEQ_LEN}. No implicit re-prefill is available."
                )
            rows.append(BatchRowDescriptor(
                session_id=state.session_id,
                session_generation=state.generation_id,
                state_epoch_id=state.state_epoch_id,
                frame_id=frame.frame_id,
                listener_state_version=state.state_version,
                sequence_position=state.sequence_position,
                cache_position=state.cache_position,
                valid_history_tokens=state.valid_history_tokens,
                native_context_id=state.native_context_id,
                input_frame_sha256=frame.pcm_sha256,
                reference_pcm_sha256=frame.reference_pcm_sha256,
                frame_ready_at_ns=frame.ready_at_ns,
                reference_frame_id=frame.reference_frame_id,
                logical_q_len=frame.logical_q_len,
                ephemeral_reason=frame.ephemeral_reason,
                queue_audio_for_next_turn=frame.queue_audio_for_next_turn,
            ))
            rows_frames.append(frame)

        manifest = ListenerBatchManifest(
            manifest_id=uuid.uuid4().hex,
            runtime_signature=self.runtime_signature,
            stream_identity=self.capability.stream_identity,
            stream_priority=self.capability.stream_priority,
            dispatch_at_ns=now_ns,
            max_coalesce_wait_ns=self.coalesce_window_ns,
            dispatch_reason=reason,
            rows=tuple(rows),
        )
        batch = ListenerBatch(manifest=manifest, frames=tuple(rows_frames))
        for frame in rows_frames:
            queue = self._queues[frame.session_id]
            queued_order, queued_frame = queue.popleft()
            if queued_frame is not frame:
                raise RuntimeError(f"session queue order changed for {frame.session_id}")
            del queued_order
            self._inflight_sessions.add(frame.session_id)
        self._active_batch = batch
        return batch

    def take_ready(
        self,
        now_ns: int | None = None,
        *,
        before_manifest: Callable[[tuple[ListenerFrame, ...]], Mapping[str, int] | None] | None = None,
    ) -> ListenerBatch | None:
        """Take one due batch without sleeping; intended for deterministic tests."""
        current_ns = time.monotonic_ns() if now_ns is None else now_ns
        if isinstance(current_ns, bool) or not isinstance(current_ns, int) or current_ns < 0:
            raise ValueError("now_ns must be a non-negative monotonic timestamp")
        with self._condition:
            self._assert_owner_locked()
            if before_manifest is not None and not callable(before_manifest):
                raise TypeError("before_manifest must be callable or None")
            batch = self._take_ready_locked(current_ns, before_manifest=before_manifest)
            if batch is not None:
                self._claim_gpu_owner_locked()
            return batch

    def next_wakeup_ns(self, now_ns: int | None = None) -> int | None:
        """Return the next coalescing deadline, not an artificial frame tick."""
        current_ns = time.monotonic_ns() if now_ns is None else now_ns
        with self._condition:
            if self._active_batch is not None:
                return None
            heads = self._candidate_heads_locked()
            if not heads:
                return None
            ready_heads = [item for item in heads if item[1].ready_at_ns <= current_ns]
            if len(ready_heads) >= 2:
                return current_ns
            if len(ready_heads) == 1:
                return ready_heads[0][1].ready_at_ns + self.coalesce_window_ns
            return heads[0][1].ready_at_ns

    def wait_and_take(self, stop_event: threading.Event | None = None) -> ListenerBatch | None:
        """Block the CPU owner until real frames are ready or stop is requested."""
        with self._condition:
            self._claim_gpu_owner_locked()
            while True:
                if stop_event is not None and stop_event.is_set():
                    return None
                now_ns = time.monotonic_ns()
                batch = self._take_ready_locked(now_ns)
                if batch is not None:
                    return batch
                heads = self._candidate_heads_locked()
                if self._active_batch is not None or not heads:
                    timeout = None
                else:
                    ready_heads = [
                        item for item in heads if item[1].ready_at_ns <= now_ns
                    ]
                    if len(ready_heads) >= 2:
                        timeout = 0.0
                    elif len(ready_heads) == 1:
                        deadline_ns = (
                            ready_heads[0][1].ready_at_ns + self.coalesce_window_ns
                        )
                        timeout = max(0.0, (deadline_ns - now_ns) / 1_000_000_000)
                    else:
                        next_ready_ns = min(item[1].ready_at_ns for item in heads)
                        timeout = max(0.0, (next_ready_ns - now_ns) / 1_000_000_000)
                if stop_event is not None:
                    # A threading.Event cannot wake this condition on its own;
                    # callers set it and call wake() during owner shutdown.
                    if timeout is None:
                        timeout = 0.1
                    else:
                        timeout = min(timeout, 0.1)
                self._condition.wait(timeout=timeout)

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _validate_active_batch_locked(self, batch: ListenerBatch) -> None:
        if self._active_batch is None or (
            self._active_batch.manifest.manifest_id != batch.manifest.manifest_id
        ):
            raise RuntimeError("batch is not the coordinator's active GPU-owner transaction")

    def complete(
        self,
        batch: ListenerBatch,
        updates: Sequence[RowStateUpdate],
        *,
        before_cursor_commit: Callable[[], None] | None = None,
    ) -> None:
        """Commit all rows after stream completion and optional staged-state commit.

        ``before_cursor_commit`` lets the single owner install another CPU-side
        staged state (for example, frontend mel/tail history) after every row
        and cursor update validates but before any coordinator cursor advances.
        If that callback fails, the whole active manifest is poisoned.
        """
        if before_cursor_commit is not None and not callable(before_cursor_commit):
            raise TypeError("before_cursor_commit must be callable or None")
        rows = tuple(updates)
        with self._condition:
            self._assert_owner_locked()
            self._validate_active_batch_locked(batch)
            try:
                expected = {row.session_id: row for row in batch.manifest.rows}
                received = {update.session_id: update for update in rows}
                if len(received) != len(rows) or set(received) != set(expected):
                    raise ValueError("batch completion must return exactly one update per manifest row")
                for session_id, update in received.items():
                    descriptor = expected[session_id]
                    state = self._states[session_id]
                    if (
                        update.frame_id != descriptor.frame_id
                        or state.generation_id != descriptor.session_generation
                        or state.state_epoch_id != descriptor.state_epoch_id
                        or state.next_frame_id != descriptor.frame_id
                        or state.state_version != descriptor.listener_state_version
                        or state.sequence_position != descriptor.sequence_position
                        or state.cache_position != descriptor.cache_position
                        or state.valid_history_tokens != descriptor.valid_history_tokens
                    ):
                        raise RuntimeError(
                            "stale batch completion; per-session state changed in flight"
                        )
                    expected_cache_position = descriptor.cache_position + descriptor.logical_q_len
                    if update.cache_position > LISTENER_MAX_SEQ_LEN:
                        raise ValueError(
                            f"updated cache_position exceeds max_seq_len={LISTENER_MAX_SEQ_LEN}"
                        )
                    if update.cache_position != expected_cache_position:
                        raise ValueError(
                            "batch update cache_position must advance by logical "
                            f"Q={descriptor.logical_q_len} (physical Q={LISTENER_Q_PER_FRAME})"
                        )
                    if update.valid_history_tokens != update.cache_position:
                        raise ValueError(
                            "valid_history_tokens must equal the updated cache_position"
                        )
                if before_cursor_commit is not None:
                    before_cursor_commit()
            except BaseException as exc:
                self._poison_active_locked(batch, exc)
                raise

            # All validation happens before any cursor changes, so a malformed
            # completion cannot partially commit a multi-session batch.
            for session_id, update in received.items():
                state = self._states[session_id]
                state.next_frame_id += 1
                state.state_version += 1
                state.sequence_position += 1
                state.cache_position = update.cache_position
                state.valid_history_tokens = update.valid_history_tokens
                self._inflight_sessions.remove(session_id)
            self._active_batch = None
            self._condition.notify_all()

    def fail(self, batch: ListenerBatch, error: BaseException) -> None:
        """Poison all rows in a failed batch; never replay/fallback implicitly."""
        with self._condition:
            self._assert_owner_locked()
            self._validate_active_batch_locked(batch)
            self._poison_active_locked(batch, error)

    def _poison_active_locked(self, batch: ListenerBatch, error: BaseException) -> None:
        reason = f"{type(error).__name__}: {error}"
        self._failed_batches[batch.manifest.manifest_id] = reason
        for row in batch.manifest.rows:
            state = self._states[row.session_id]
            state.poisoned_reason = reason
            self._inflight_sessions.discard(row.session_id)
            self._queues[row.session_id].clear()
            self._next_enqueued_frame_id[row.session_id] = state.next_frame_id
        self._active_batch = None
        self._condition.notify_all()

    def run_one(
        self,
        now_ns: int,
        processor: BatchProcessor,
    ) -> ListenerBatch | None:
        """Call the injected processor once for the entire N-row batch.

        The processor must complete audio-encoder, native Q4, and batched
        Control6 work on the owner stream before returning row cursor updates.
        Missing native batch support must fail at worker startup, not here via
        an N-call compatibility fallback.
        """
        batch = self.take_ready(now_ns)
        if batch is None:
            return None
        states = tuple(self._states[row.session_id] for row in batch.manifest.rows)
        try:
            updates = processor.process_batch(batch, states)
            self.complete(batch, updates)
        except BaseException as exc:
            if self.active_manifest_id == batch.manifest.manifest_id:
                self.fail(batch, exc)
            raise
        return batch

    def close_session(self, session_id: str) -> RetiredSession:
        """Remove an idle session; caller owns ordered cache/context teardown."""
        with self._condition:
            self._assert_owner_locked()
            if session_id in self._inflight_sessions:
                raise RuntimeError("cannot close a session while its batch row is in flight")
            state = self._states.pop(session_id, None)
            if state is None:
                raise KeyError(f"unknown listener session: {session_id}")
            queued = self._queues.pop(session_id)
            self._next_enqueued_frame_id.pop(session_id, None)
            state.closed = True
            self._condition.notify_all()
            return RetiredSession(state=state, queued_frame_count=len(queued))

    def session_state(self, session_id: str) -> ListenerSessionSnapshot:
        """Return an idle, post-commit owner snapshot.

        Once the batch GPU owner is bound, other threads cannot observe live
        state. Even the owner may snapshot only between batches; this is not a
        concurrent monitoring API or an in-flight checkpoint mechanism.
        """
        with self._condition:
            self._assert_owner_locked()
            if self._active_batch is not None:
                raise RuntimeError(
                    "session snapshots are unavailable while a batch is active"
                )
            try:
                state = self._states[session_id]
            except KeyError as exc:
                raise KeyError(f"unknown listener session: {session_id}") from exc
            return snapshot_listener_state(state)

    def rebase_cache_position(self, session_id: str, cache_position: int) -> None:
        """Replace a session's private prefix cursor without advancing its frame.

        Semantic refresh is valid only at an idle frame boundary. It changes
        the KV origin and replayed-token count, while the global frame,
        sequence, generation, and state-version cursors remain untouched.
        """
        if isinstance(cache_position, bool) or not isinstance(cache_position, int):
            raise ValueError("cache_position must be an integer")
        if not 1 <= cache_position <= LISTENER_MAX_SEQ_LEN:
            raise ValueError("cache_position is outside the native listener capacity")
        with self._condition:
            self._assert_owner_locked()
            if self._active_batch is not None:
                raise RuntimeError("semantic prefix refresh requires no active batch")
            if self._inflight_sessions:
                raise RuntimeError("semantic prefix refresh requires idle listener rows")
            state = self._states.get(session_id)
            if state is None:
                raise KeyError(f"unknown listener session: {session_id}")
            if self._queues.get(session_id):
                raise RuntimeError("semantic prefix refresh cannot cross a queued frame")
            state.cache_position = cache_position
            state.valid_history_tokens = cache_position
            self._condition.notify_all()

    def advance_cache_position(
        self,
        session_id: str,
        *,
        token_count: int,
        expected_generation_id: int,
        expected_cache_position: int,
    ) -> ListenerSessionSnapshot:
        """Commit native text tokens while preserving listener frame identity.

        This is called only by the single GPU owner after its embedded-text
        forward and output-health check have succeeded. Queued listener frames
        are allowed: they are admitted later from the new cache cursor.
        """
        if isinstance(token_count, bool) or not isinstance(token_count, int) or not 1 <= token_count <= 8:
            raise ValueError("text token_count must be one to eight")
        for name, value in (
            ("expected_generation_id", expected_generation_id),
            ("expected_cache_position", expected_cache_position),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        with self._condition:
            self._assert_owner_locked()
            if self._active_batch is not None:
                raise RuntimeError("text append requires no active listener batch")
            if session_id in self._inflight_sessions:
                raise RuntimeError("text append cannot overlap an in-flight listener row")
            state = self._states.get(session_id)
            if state is None:
                raise KeyError(f"unknown listener session: {session_id}")
            if state.closed or state.poisoned_reason is not None:
                raise RuntimeError("text append requires a live, healthy listener session")
            if (
                state.generation_id != expected_generation_id
                or state.cache_position != expected_cache_position
                or state.valid_history_tokens != expected_cache_position
            ):
                raise RuntimeError("text append lost its generation/cache-position compare-and-swap")
            physical_q = max(LISTENER_Q_PER_FRAME, token_count + (token_count & 1))
            if state.cache_position > LISTENER_MAX_SEQ_LEN - physical_q:
                raise RuntimeError(
                    "native listener physical text-Q capacity exhausted before text append; "
                    f"session={session_id}, cache_position={state.cache_position}, "
                    f"physical_q={physical_q}, max_seq_len={LISTENER_MAX_SEQ_LEN}"
                )
            state.cache_position += token_count
            state.valid_history_tokens = state.cache_position
            self._condition.notify_all()
            return snapshot_listener_state(state)

    def rewind_cache_position(
        self,
        session_id: str,
        *,
        target_cache_position: int,
        expected_generation_id: int,
        expected_cache_position: int,
    ) -> ListenerSessionSnapshot:
        """Commit an owner-ordered assistant-branch rollback without changing audio time."""
        for name, value in (
            ("target_cache_position", target_cache_position),
            ("expected_generation_id", expected_generation_id),
            ("expected_cache_position", expected_cache_position),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        with self._condition:
            self._assert_owner_locked()
            if self._active_batch is not None:
                raise RuntimeError("assistant rollback requires no active listener batch")
            if session_id in self._inflight_sessions or self._queues.get(session_id):
                raise RuntimeError("assistant rollback cannot cross queued or in-flight listener frames")
            state = self._states.get(session_id)
            if state is None:
                raise KeyError(f"unknown listener session: {session_id}")
            if state.closed or state.poisoned_reason is not None:
                raise RuntimeError("assistant rollback requires a live, healthy listener session")
            if (
                state.generation_id != expected_generation_id
                or state.cache_position != expected_cache_position
                or state.valid_history_tokens != expected_cache_position
            ):
                raise RuntimeError("assistant rollback lost its generation/cache-position compare-and-swap")
            if target_cache_position > state.cache_position:
                raise ValueError("assistant rollback target cannot extend the cache")
            state.cache_position = target_cache_position
            state.valid_history_tokens = target_cache_position
            self._condition.notify_all()
            return snapshot_listener_state(state)

    def _claim_gpu_owner_locked(self) -> None:
        current = threading.get_ident()
        if self._gpu_owner_thread_id is None:
            self._gpu_owner_thread_id = current
        elif self._gpu_owner_thread_id != current:
            raise RuntimeError("listener batch GPU state is owned by another worker thread")

    def _assert_owner_locked(self) -> None:
        if (
            self._gpu_owner_thread_id is not None
            and self._gpu_owner_thread_id != threading.get_ident()
        ):
            raise RuntimeError("listener batch GPU state is owned by another worker thread")


def _is_mutable_reference(value: Any) -> bool:
    return not isinstance(value, (str, bytes, int, float, bool, tuple, frozenset))


_PRIVATE_STATE_FIELDS = (
    "user_pcm_tail",
    "assistant_pcm_tail",
    "user_mel_history",
    "assistant_mel_history",
    "attention_mask",
    "native_kv_cache",
    "native_context",
)


def _mutable_state_values(state: ListenerSessionState) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    for field_name in _PRIVATE_STATE_FIELDS:
        value = getattr(state, field_name)
        if value is None:
            continue
        values.extend(_walk_mutable_payload(value, field_name, set()))
    return values


def _walk_mutable_payload(
    value: Any,
    path: str,
    seen: set[int],
) -> list[tuple[str, Any]]:
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return []
    identity = id(value)
    if identity in seen:
        return []
    seen.add(identity)
    result: list[tuple[str, Any]] = []
    if _is_mutable_reference(value):
        result.append((path, value))
    if isinstance(value, Mapping):
        for key, item in value.items():
            result.extend(_walk_mutable_payload(item, f"{path}[{key!r}]", seen))
    elif isinstance(value, (list, tuple, set, frozenset, deque)):
        for index, item in enumerate(value):
            result.extend(_walk_mutable_payload(item, f"{path}[{index}]", seen))
    elif not isinstance(value, np.ndarray) and _torch_storage_key(value) is None:
        try:
            attributes = vars(value)
        except TypeError:
            attributes = {}
        slots: dict[str, Any] = {}
        for cls in type(value).__mro__:
            declared = getattr(cls, "__slots__", ())
            if isinstance(declared, str):
                declared = (declared,)
            for name in declared:
                if name in {"__dict__", "__weakref__"} or name in attributes:
                    continue
                try:
                    slots[name] = getattr(value, name)
                except AttributeError:
                    continue
        attributes = {**attributes, **slots}
        for name, item in attributes.items():
            result.extend(_walk_mutable_payload(item, f"{path}.{name}", seen))
    return result


def _torch_storage_key(value: Any) -> tuple[str, int] | None:
    """Return backing storage identity without reading tensor values."""
    if not all(hasattr(value, name) for name in ("untyped_storage", "device", "numel")):
        return None
    try:
        if int(value.numel()) == 0:
            return None
        pointer = int(value.untyped_storage().data_ptr())
        if pointer == 0:
            return None
        return (str(value.device), pointer)
    except Exception as exc:
        raise ValueError("cannot verify private tensor backing storage") from exc


def _payloads_alias(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        try:
            left_root = _numpy_storage_root(left)
            right_root = _numpy_storage_root(right)
            if left_root is right_root:
                return True
            return bool(np.shares_memory(left, right))
        except Exception:
            # Memory-overlap uncertainty must fail closed for private state.
            return True
    left_storage = _torch_storage_key(left)
    return left_storage is not None and left_storage == _torch_storage_key(right)


def _numpy_storage_root(value: np.ndarray) -> Any:
    root: Any = value
    seen: set[int] = set()
    while isinstance(root, np.ndarray) and root.base is not None:
        if id(root) in seen:
            break
        seen.add(id(root))
        root = root.base
    return root


def _native_context_key(value: Any) -> tuple[str, Any]:
    pointer_value = getattr(value, "value", None)
    if isinstance(pointer_value, int):
        return ("native-context-pointer", pointer_value)
    if isinstance(value, (int, str)):
        return (type(value).__name__, value)
    return ("object-id", id(value))
