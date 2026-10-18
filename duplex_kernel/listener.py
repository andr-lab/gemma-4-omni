"""Continuous Gemma audio listener and causal assistant self-reference.

P3 deliberately stops at the native Gemma audio tower.  It does not run ASR,
text generation, V9, or the Qwen speaker.  The scheduler supplies one complete
16 kHz/80 ms frame at a time; the listener keeps a bounded causal context and
returns the last valid Gemma audio embedding for that context.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from duplex_kernel.gemma_representation import (
    AUDIO_FEATURES_PER_FRAME,
    AUDIO_TOWER_MEL_STRIDE,
    AUDIO_START_AFTER_TEXT,
    CANONICAL_LM_PREFILL_TARGET_LEN,
    TEXT_BLOCK_LEN,
    build_audio_placeholder_inputs,
    interleave_duplex_mel,
)
from duplex_kernel.conditioning_provenance import state_lineage_digest
from duplex_kernel.scheduler import DEFAULT_SAMPLE_RATE

logger = logging.getLogger("duplex_kernel.listener")

DEFAULT_FRAME_DURATION_MS = 80
DEFAULT_FRAME_SAMPLES = DEFAULT_SAMPLE_RATE * DEFAULT_FRAME_DURATION_MS // 1000
DEFAULT_REFERENCE_DELAY_FRAMES = 1
DEFAULT_CONTEXT_FRAMES = 64
MEL_PER_LISTENER_FRAME = 8
LISTENER_MEL_CHANNELS = 2
LISTENER_FEATURES_PER_FRAME = AUDIO_FEATURES_PER_FRAME
LISTENER_TEXT_BLOCK_LEN = TEXT_BLOCK_LEN
_NATIVE_LISTENER_CUDA_GRAPH_ENV = "GEMMA4_NATIVE_LISTENER_CUDA_GRAPH"


class _ListenerWaiterPriority:
    """Keep the shared listener-priority Event set until every concurrent step exits."""

    def __init__(self, event: threading.Event) -> None:
        self._event = event
        self._lock = threading.Lock()
        self._waiter_count = 0

    @property
    def waiter_count(self) -> int:
        with self._lock:
            return self._waiter_count

    @contextmanager
    def registered(self):
        with self._lock:
            self._waiter_count += 1
            self._event.set()
        try:
            yield
        finally:
            with self._lock:
                if self._waiter_count <= 0:
                    raise RuntimeError("listener waiter counter underflow")
                self._waiter_count -= 1
                if self._waiter_count == 0:
                    self._event.clear()

class FairModelLock:
    """Arbitrate listener/model quanta FIFO, optionally allowing a listener pair."""

    _PRIORITIES = frozenset({"listener", "compute"})
    _DEFAULT_HOST_TIMELINE_CAPACITY = 2048
    _NONBLOCKING_HOST_TIMELINE_EVENT_LIMIT = 64

    def __init__(
        self,
        *,
        listener_pair_lock_handoff: bool = False,
        host_timeline_enabled: bool = False,
        host_timeline_capacity: int = _DEFAULT_HOST_TIMELINE_CAPACITY,
    ) -> None:
        if int(host_timeline_capacity) <= 0:
            raise ValueError("host timeline capacity must be positive")
        self._condition = threading.Condition()
        self._queues = {"listener": deque(), "compute": deque()}
        self._next_ticket = 0
        self._owner: int | None = None
        self._depth = 0
        self._last_grant = "compute"
        self.listener_pair_lock_handoff_enabled = listener_pair_lock_handoff
        self._listener_grant_streak = 0
        self._host_timeline_capacity = int(host_timeline_capacity)
        self._host_timeline_events = (
            deque(maxlen=self._host_timeline_capacity) if host_timeline_enabled else None
        )
        self._host_timeline_initial_events = [] if host_timeline_enabled else None
        self._host_timeline_tail_dropped = 0
        self._host_timeline_next_seq = 0
        self._owner_ticket: int | None = None
        self._owner_priority: str | None = None
        self._owner_acquired_ns: int | None = None

    @property
    def host_timeline_enabled(self) -> bool:
        return self._host_timeline_events is not None

    def configure_host_timeline(self, *, enabled: bool, capacity: int = 256) -> None:
        """Enable or disable bounded host-only lock tracing while the arbiter is idle."""
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("host timeline capacity must be positive")
        if type(enabled) is not bool:
            raise TypeError("host timeline enabled must be a bool")
        with self._condition:
            already_enabled = self._host_timeline_events is not None
            if already_enabled == enabled and self._host_timeline_capacity == capacity:
                return
            if self._owner is not None or any(self._queues.values()):
                raise RuntimeError("host timeline cannot be reconfigured while the lock is active")
            if not enabled:
                self._host_timeline_capacity = capacity
                self._host_timeline_events = None
                self._host_timeline_initial_events = None
                self._host_timeline_tail_dropped = 0
                self._host_timeline_next_seq = 0
                return
            previous_events: list[dict[str, Any]] = []
            if self._host_timeline_events is not None:
                combined = list(self._host_timeline_initial_events or []) + list(
                    self._host_timeline_events
                )
                unique_events = {int(row["trace_seq"]): row for row in combined}
                previous_events = [unique_events[seq] for seq in sorted(unique_events)]
            self._host_timeline_capacity = capacity
            self._host_timeline_events = deque(maxlen=capacity)
            self._host_timeline_initial_events = previous_events[:capacity]
            self._host_timeline_events.extend(previous_events[-capacity:])
            retained_tail = min(len(previous_events), capacity)
            self._host_timeline_tail_dropped = max(
                0, self._host_timeline_next_seq - retained_tail
            )

    def _record_host_timeline(
        self,
        event: str,
        *,
        priority: str,
        ticket: int | None,
        timestamp_ns: int | None = None,
        thread_id: int | None = None,
        **fields: Any,
    ) -> int | None:
        events = self._host_timeline_events
        if events is None:
            return None
        stamp = time.monotonic_ns() if timestamp_ns is None else int(timestamp_ns)
        row = {
            "trace_seq": self._host_timeline_next_seq,
            "event": event,
            "timestamp_ns": stamp,
            "thread_id": threading.get_ident() if thread_id is None else int(thread_id),
            "priority": priority,
            "ticket": ticket,
            "pending_counts": {
                name: len(queue) for name, queue in self._queues.items()
            },
        }
        initial_events = self._host_timeline_initial_events
        if initial_events is not None and len(initial_events) < self._host_timeline_capacity:
            initial_events.append(row)
        if len(events) == self._host_timeline_capacity:
            self._host_timeline_tail_dropped += 1
        self._host_timeline_next_seq += 1
        row.update(fields)
        events.append(row)
        return stamp

    def host_timeline_snapshot(self) -> dict[str, Any]:
        """Copy the bounded host-only lock timeline for the session sidecar."""
        with self._condition:
            return self._host_timeline_snapshot_locked()

    def _host_timeline_snapshot_locked(
        self,
        *,
        max_events: int | None = None,
    ) -> dict[str, Any]:
        """Copy the timeline while the caller holds ``_condition``."""
        events = self._host_timeline_events
        if events is None:
            return {
                "enabled": False,
                "status": "not_requested",
                "scope": "shared_model_lock",
                "capacity": 0,
                "ring_capacity": 0,
                "total_event_count": 0,
                "retained_event_count": 0,
                "dropped_event_count": 0,
                "events": [],
            }
        initial_events = self._host_timeline_initial_events or []
        overlap_events = 0
        if max_events is None:
            combined = list(initial_events) + list(events)
            unique_by_seq = {int(row["trace_seq"]): row for row in combined}
            ordered_rows = [unique_by_seq[seq] for seq in sorted(unique_by_seq)]
            overlap_events = len(combined) - len(unique_by_seq)
        else:
            # The listener callback only needs recent contention context. Read
            # the two rings backwards and stop at a fixed bound; never sort or
            # copy the full history while dispatching a frame.
            selected_rows: list[dict[str, Any]] = []
            seen: set[int] = set()
            for ring in (events, initial_events):
                for row in reversed(ring):
                    seq = int(row["trace_seq"])
                    if seq in seen:
                        overlap_events += 1
                    else:
                        seen.add(seq)
                        selected_rows.append(row)
                        if len(selected_rows) >= max_events:
                            break
                if len(selected_rows) >= max_events:
                    break
            ordered_rows = list(reversed(selected_rows))

        copied = []
        for row in ordered_rows:
            snapshot_row = dict(row)
            snapshot_row["pending_counts"] = dict(row["pending_counts"])
            copied.append(snapshot_row)
        total_events = self._host_timeline_next_seq
        dropped_events = max(0, total_events - len(copied))
        snapshot = {
            "enabled": True,
            "status": "truncated" if dropped_events else "complete",
            "scope": "process_shared_model_lock",
            "clock": "time.monotonic_ns",
            "process_id": os.getpid(),
            "container_identity": os.environ.get("HOSTNAME"),
            "snapshot_monotonic_ns": time.monotonic_ns(),
            "capacity": self._host_timeline_capacity * 2,
            "ring_capacity": self._host_timeline_capacity,
            "total_event_count": total_events,
            "retained_event_count": len(copied),
            "dropped_event_count": dropped_events,
            "initial_ring_retained_count": len(initial_events),
            "tail_ring_retained_count": len(events),
            "tail_ring_overwritten_event_count": self._host_timeline_tail_dropped,
            "overlap_deduplicated_event_count": overlap_events,
            "dedupe_key": ["container_identity", "process_id", "trace_seq"],
            "join_strategy": (
                "events share time.monotonic_ns with sidecar host_tick_timings; match event "
                "timestamps to listener_step_start_ns/listener_step_end_ns intervals only "
                "when the match is unique. Keep unmatched or overlapping matches global/ambiguous. "
                "Per-session sidecars repeat this process-wide snapshot; deduplicate by the "
                "listed key and do not assign an ambiguous event to a session."
            ),
            "pending_counts": {
                name: len(queue) for name, queue in self._queues.items()
            },
            "current_owner": (
                {
                    "thread_id": self._owner,
                    "priority": self._owner_priority,
                    "ticket": self._owner_ticket,
                    "depth": self._depth,
                }
                if self._owner is not None else None
            ),
            "events": copied,
        }
        if max_events is not None:
            snapshot["snapshot_event_limit"] = max_events
            snapshot["snapshot_omitted_event_count"] = dropped_events
        return snapshot

    def host_timeline_snapshot_nonblocking(
        self,
        *,
        request_monotonic_ns: int | None = None,
    ) -> dict[str, Any]:
        """Try to copy the host timeline without delaying a listener callback."""
        if request_monotonic_ns is None:
            request_ns = time.monotonic_ns()
        elif isinstance(request_monotonic_ns, bool) or not isinstance(request_monotonic_ns, int):
            raise ValueError("request_monotonic_ns must be an integer monotonic timestamp")
        else:
            request_ns = int(request_monotonic_ns)

        if not self._condition.acquire(blocking=False):
            return {
                "enabled": self.host_timeline_enabled,
                "status": "omitted_busy",
                "omitted": True,
                "reason": "fair_model_lock_condition_busy",
                "scope": "process_shared_model_lock",
                "clock": "time.monotonic_ns",
                "request_monotonic_ns": request_ns,
                "lock_acquired_monotonic_ns": None,
                "snapshot_monotonic_ns": None,
                "age_ms": None,
                "events": [],
            }

        try:
            acquired_ns = time.monotonic_ns()
            snapshot = self._host_timeline_snapshot_locked(
                max_events=self._NONBLOCKING_HOST_TIMELINE_EVENT_LIMIT
            )
            snapshot_ns = time.monotonic_ns()
            snapshot.update(
                request_monotonic_ns=request_ns,
                lock_acquired_monotonic_ns=acquired_ns,
                snapshot_monotonic_ns=snapshot_ns,
                age_ms=round(max(0, snapshot_ns - request_ns) / 1_000_000.0, 3),
            )
            return snapshot
        finally:
            self._condition.release()

    @property
    def pending_counts(self) -> dict[str, int]:
        with self._condition:
            return {key: len(queue) for key, queue in self._queues.items()}

    def _selected_priority(self) -> str | None:
        listener_waiting = bool(self._queues["listener"])
        compute_waiting = bool(self._queues["compute"])
        if listener_waiting and compute_waiting:
            if self.listener_pair_lock_handoff_enabled:
                return "listener" if self._listener_grant_streak < 2 else "compute"
            return "listener" if self._last_grant == "compute" else "compute"
        if listener_waiting:
            return "listener"
        if compute_waiting:
            return "compute"
        return None

    def _cancel_ticket(self, priority: str, ticket: int) -> None:
        with self._condition:
            queue = self._queues[priority]
            for index, entry in enumerate(queue):
                if entry[0] == ticket:
                    del queue[index]
                    self._record_host_timeline(
                        "cancel", priority=priority, ticket=ticket, thread_id=entry[1]
                    )
                    break
            self._condition.notify_all()

    def acquire_fair(
        self,
        priority: str,
        *,
        stop_check: Any | None = None,
    ) -> bool:
        if priority not in self._PRIORITIES:
            raise ValueError(f"unknown model-lock priority: {priority}")
        owner = threading.get_ident()
        with self._condition:
            if self._owner == owner:
                self._depth += 1
                self._record_host_timeline(
                    "reentrant_acquire",
                    priority=self._owner_priority or priority,
                    ticket=self._owner_ticket,
                    depth=self._depth,
                )
                return True
            ticket = self._next_ticket
            self._next_ticket += 1
            self._queues[priority].append((ticket, owner))
            requested_ns = self._record_host_timeline(
                "request", priority=priority, ticket=ticket, thread_id=owner
            )

        try:
            while True:
                if stop_check is not None and stop_check():
                    self._cancel_ticket(priority, ticket)
                    return False
                with self._condition:
                    selected = self._selected_priority()
                    queue = self._queues[priority]
                    at_queue_head = bool(queue and queue[0][0] == ticket)
                    eligible = bool(
                        self._owner is None
                        and selected == priority
                        and at_queue_head
                    )
                    if eligible:
                        queue.popleft()
                        self._owner = owner
                        self._depth = 1
                        granted_ns = (
                            time.monotonic_ns()
                            if self._host_timeline_events is not None
                            else None
                        )
                        self._owner_ticket = ticket
                        self._owner_priority = priority
                        self._owner_acquired_ns = granted_ns
                        self._last_grant = priority
                        if self.listener_pair_lock_handoff_enabled:
                            self._listener_grant_streak = (
                                self._listener_grant_streak + 1
                                if priority == "listener" else 0
                            )
                        wait_ns = (
                            max(0, granted_ns - requested_ns)
                            if granted_ns is not None and requested_ns is not None else None
                        )
                        self._record_host_timeline(
                            "acquire",
                            priority=priority,
                            ticket=ticket,
                            timestamp_ns=granted_ns,
                            thread_id=owner,
                            wait_ns=wait_ns,
                            depth=self._depth,
                        )
                        self._condition.notify_all()
                        return True
                    self._condition.wait(timeout=0.005)
        except BaseException:
            self._cancel_ticket(priority, ticket)
            raise

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        del timeout  # The shared arbiter uses cancellation-aware fair acquisition.
        if blocking:
            return self.acquire_fair("compute")
        owner = threading.get_ident()
        with self._condition:
            ticket = None
            requested_ns = None
            if self._host_timeline_events is not None:
                ticket = self._next_ticket
                self._next_ticket += 1
                requested_ns = self._record_host_timeline(
                    "request", priority="compute", ticket=ticket, thread_id=owner,
                    blocking=False,
                )
            if self._owner is not None or any(self._queues.values()):
                self._record_host_timeline(
                    "denied", priority="compute", ticket=ticket, thread_id=owner,
                    blocking=False,
                )
                return False
            self._owner = owner
            self._depth = 1
            self._owner_ticket = ticket
            self._owner_priority = "compute"
            granted_ns = (
                time.monotonic_ns()
                if self._host_timeline_events is not None
                else None
            )
            self._owner_acquired_ns = granted_ns
            self._last_grant = "compute"
            if self.listener_pair_lock_handoff_enabled:
                self._listener_grant_streak = 0
            wait_ns = (
                max(0, granted_ns - requested_ns)
                if granted_ns is not None and requested_ns is not None else None
            )
            self._record_host_timeline(
                "acquire",
                priority="compute",
                ticket=ticket,
                timestamp_ns=granted_ns,
                thread_id=owner,
                wait_ns=wait_ns,
                blocking=False,
                depth=self._depth,
            )
            return True

    def release(self) -> None:
        owner = threading.get_ident()
        with self._condition:
            if self._owner != owner or self._depth <= 0:
                raise RuntimeError("model lock released by a non-owner")
            self._depth -= 1
            fully_released = self._depth == 0
            acquired_ns = self._owner_acquired_ns
            owner_priority = self._owner_priority or "compute"
            released_ns = (
                time.monotonic_ns()
                if self._host_timeline_events is not None
                else None
            )
            self._record_host_timeline(
                "release",
                priority=owner_priority,
                ticket=self._owner_ticket,
                timestamp_ns=released_ns,
                thread_id=owner,
                fully_released=fully_released,
                depth=self._depth,
                hold_ns=(
                    max(0, released_ns - acquired_ns)
                    if fully_released and released_ns is not None and acquired_ns is not None
                    else None
                ),
            )
            if self._depth == 0:
                self._owner = None
                self._owner_ticket = None
                self._owner_priority = None
                self._owner_acquired_ns = None
                self._condition.notify_all()


def capture_host_timeline_nonblocking(
    capture_lock: Any,
    model_lock: Any,
    *,
    batch_id: str,
    batch_size: int | None,
    batch_state: dict[str, Any],
) -> dict[str, Any]:
    """Best-effort, two-lock host timeline capture for one shared batch."""
    request_ns = time.monotonic_ns()
    common = {
        "scope": "process_shared_model_lock",
        "clock": "time.monotonic_ns",
        "batch_id": batch_id,
        "batch_size": batch_size,
        "request_monotonic_ns": request_ns,
        "capture_lock_acquired_monotonic_ns": None,
        "model_lock_acquired_monotonic_ns": None,
        "snapshot_monotonic_ns": None,
        "age_ms": None,
        "events": [],
    }

    # The dispatcher fans rows out serially. Remember the batch before either
    # nonblocking attempt so later rows never retry a contended diagnostic.
    if batch_state.get("last_attempted_batch_id") == batch_id:
        return {
            **common,
            "enabled": False,
            "status": "omitted",
            "omitted": True,
            "reason": "batch_already_attempted",
        }
    batch_state["last_attempted_batch_id"] = batch_id

    if not capture_lock.acquire(blocking=False):
        return {
            **common,
            "enabled": True,
            "status": "omitted_busy",
            "omitted": True,
            "reason": "timeline_capture_lock_busy",
        }

    try:
        capture_acquired_ns = time.monotonic_ns()
        common["capture_lock_acquired_monotonic_ns"] = capture_acquired_ns
        snapshot_fn = getattr(model_lock, "host_timeline_snapshot_nonblocking", None)
        if not callable(snapshot_fn):
            return {
                **common,
                "enabled": False,
                "status": "unavailable",
                "omitted": True,
                "reason": "nonblocking_snapshot_api_unavailable",
            }
        try:
            snapshot = snapshot_fn(request_monotonic_ns=request_ns)
        except Exception as exc:
            return {
                **common,
                "enabled": True,
                "status": "snapshot_error",
                "omitted": True,
                "reason": "snapshot_exception",
                "error_type": type(exc).__name__,
            }

        if not isinstance(snapshot, dict):
            return {
                **common,
                "enabled": False,
                "status": "invalid_snapshot",
                "omitted": True,
                "reason": "snapshot_result_not_mapping",
            }

        model_acquired_ns = snapshot.get("lock_acquired_monotonic_ns")
        common["model_lock_acquired_monotonic_ns"] = (
            int(model_acquired_ns)
            if isinstance(model_acquired_ns, int) and not isinstance(model_acquired_ns, bool)
            else None
        )
        common["snapshot_monotonic_ns"] = (
            int(snapshot["snapshot_monotonic_ns"])
            if isinstance(snapshot.get("snapshot_monotonic_ns"), int)
            and not isinstance(snapshot.get("snapshot_monotonic_ns"), bool)
            else None
        )
        common["age_ms"] = snapshot.get("age_ms")

        if snapshot.get("status") == "omitted_busy":
            return {
                **common,
                "enabled": bool(snapshot.get("enabled")),
                "status": "omitted_busy",
                "omitted": True,
                "reason": str(snapshot.get("reason", "fair_model_lock_busy")),
            }
        if not snapshot.get("enabled"):
            return {
                **common,
                "enabled": False,
                "status": str(snapshot.get("status", "not_requested")),
                "omitted": True,
                "reason": "host_timeline_disabled",
            }

        source_events = snapshot.get("events", [])
        if not isinstance(source_events, list):
            source_events = []
        # Copy at most 256 rows, and only after both diagnostic locks were held.
        events = source_events[-256:]
        omitted = max(0, len(source_events) - len(events))
        return {
            **common,
            "enabled": True,
            "status": "truncated" if omitted else str(snapshot.get("status", "complete")),
            "omitted": False,
            "reason": None,
            "capacity": 256,
            "source_total_event_count": snapshot.get("total_event_count"),
            "source_retained_event_count": snapshot.get("retained_event_count"),
            "source_dropped_event_count": snapshot.get("dropped_event_count"),
            "exported_event_count": len(events),
            "omitted_from_snapshot_event_count": omitted,
            "current_owner": snapshot.get("current_owner"),
            "events": events,
        }
    finally:
        capture_lock.release()


@contextmanager
def _timed_lock(lock: Any, timing_sink: dict[str, int], *, name: str):
    wait_started_ns = time.monotonic_ns()
    lock.acquire()
    acquired_ns = time.monotonic_ns()
    try:
        yield
    finally:
        lock.release()
        released_ns = time.monotonic_ns()
        timing_sink[f"{name}_wait_ns"] = max(0, acquired_ns - wait_started_ns)
        timing_sink[f"{name}_acquired_ns"] = acquired_ns
        timing_sink[f"{name}_released_ns"] = released_ns
        timing_sink[f"{name}_hold_ns"] = max(0, released_ns - acquired_ns)

NATIVE_LM_CUSTOM_MAX_SEQ_LEN = 4096
NATIVE_LM_EXACT_MAX_SEQ_LEN = 16384
P4_NATIVE_CERT_MAX_SEQ_LEN = NATIVE_LM_EXACT_MAX_SEQ_LEN
NATIVE_LM_PREFIX_LEN = AUDIO_START_AFTER_TEXT
NANOSECONDS = 1_000_000_000


def _native_lm_max_seq_len() -> int:
    """Select the listener KV capacity from the actual exact-backend contract."""
    attention_exact = (
        os.environ.get("GEMMA4_ATEN_SDPA_BRIDGE") == "1"
        or os.environ.get("GEMMA4_ATTENTION_BACKEND", "").strip().upper()
        in {"1", "2", "3", "ATEN_EXACT", "ATEN_REPEAT_EXACT", "ATEN_GQA_EXACT", "NATIVE_SPLITK_GQA"}
    )
    exact = (
        attention_exact
        and (
            os.environ.get("GEMMA4_RMSNORM_BACKEND") in {"ATEN_EXACT", "1", "NATIVE_EXACT", "2"}
            or os.environ.get("GEMMA4_ATEN_RMSNORM_BRIDGE") == "1"
        )
    )
    return NATIVE_LM_EXACT_MAX_SEQ_LEN if exact else NATIVE_LM_CUSTOM_MAX_SEQ_LEN


def native_frame_capacity(
    max_seq_len: int,
    *,
    prefix_len: int = AUDIO_START_AFTER_TEXT,
) -> int:
    """Return complete Q=2 audio frames fitting after the active prefix."""
    max_seq_len = int(max_seq_len)
    prefix_len = int(prefix_len)
    if prefix_len <= 0 or max_seq_len <= prefix_len:
        raise ValueError("max_seq_len deve superare la lunghezza positiva del prefisso")
    return (max_seq_len - prefix_len) // AUDIO_FEATURES_PER_FRAME


def _maybe_rollover_native_context(
    state: "GemmaListenerState",
    native: Any,
    *,
    frame_capacity: int,
    retain_frames: int,
) -> bool:
    """Compact native KV before appending beyond capacity; IDs stay absolute.

    A rebase keeps the active prompt and the most recent ``retain_frames``
    audio KV frames, while discarding older direct attention entries. Retained
    K/V states still carry their earlier contextualization, but this is not an
    exact reconstruction of the discarded full-history attention.
    """
    frame_capacity = int(frame_capacity)
    retain_frames = int(retain_frames)
    cached_frames = int(state.gemma_lm_audio_frames)
    if frame_capacity <= 0 or cached_frames < 0:
        raise ValueError("capacità/frame cache nativa non validi")
    if cached_frames > frame_capacity:
        raise RuntimeError("cache Gemma nativa ha già superato la capacità locale")
    if cached_frames < frame_capacity:
        return False
    if not 0 < retain_frames < cached_frames:
        raise ValueError("retain_frames deve essere positivo e minore della cache da compattare")
    source_ids = state.gemma_lm_source_frame_ids
    if not source_ids and cached_frames and state.gemma_lm_skipped_silence_frames == 0:
        # Compatibility with deterministic backend fixtures that predate the
        # absolute-source index; the live Gemma backend always fills this list.
        source_ids = list(
            range(state.gemma_lm_context_start_frame, state.gemma_lm_context_start_frame + cached_frames)
        )
        state.gemma_lm_source_frame_ids = source_ids
        state.gemma_lm_last_source_frame_id = source_ids[-1]
    if (
        len(source_ids) != cached_frames
        or state.gemma_lm_last_source_frame_id != (source_ids[-1] if source_ids else None)
        or (source_ids and source_ids[-1] >= state.next_frame_id)
    ):
        raise RuntimeError("provenienza dei frame compattati nella cache KV incoerente")

    dropped_frames = cached_frames - retain_frames
    reported_dropped = native.compact_audio_history(
        expected_cached_frames=cached_frames,
        retain_frames=retain_frames,
    )
    if int(reported_dropped) != dropped_frames:
        raise RuntimeError(
            "compattazione KV nativa ha riportato un numero inatteso di frame rimossi: "
            f"{reported_dropped} != {dropped_frames}"
        )
    state.gemma_lm_audio_frames = retain_frames
    state.gemma_lm_source_frame_ids = source_ids[dropped_frames:]
    state.gemma_lm_context_start_frame = state.gemma_lm_source_frame_ids[0]
    state.gemma_lm_last_source_frame_id = state.gemma_lm_source_frame_ids[-1]
    state.gemma_lm_context_rollover_count += 1
    state.gemma_lm_context_truncated_frames += dropped_frames
    return True


def _gemma_lm_cache_can_advance(state: Any) -> bool:
    """Use compact KV positions while preserving absolute source-frame IDs."""
    if (
        getattr(state, "gemma_lm_prefix_attention", None) is not None
        and getattr(state, "semantic_context_revision", None) is not None
        and int(getattr(state, "gemma_lm_audio_frames", -1)) == 0
        and not getattr(state, "gemma_lm_source_frame_ids", ())
        and getattr(state, "gemma_lm_cache", None) is not None
    ):
        # A freshly imported semantic prefix can legitimately contain no
        # replay Q2 frames (frame zero, or a gap in observed history). Its private KV
        # is already primed for the next current frame.
        return True
    last_source = state.gemma_lm_last_source_frame_id
    if last_source == int(state.next_frame_id) - 1:
        return True
    return bool(
        state.gemma_lm_skipped_silence_frames > 0
        and last_source is not None
        and int(last_source) < int(state.next_frame_id)
    )


def _commit_listener_input_cursors(
    state: "GemmaListenerState",
    *,
    user_pcm: np.ndarray,
    reference: "AssistantReferenceFrame | None",
    frame_id: int,
    frame_samples: int,
    frame_duration_ms: int,
) -> None:
    """Commit shared counters and input lineage while state._lock is held."""
    state.listener_input_frames += 1
    state.total_samples += int(frame_samples)
    state.next_frame_id += 1
    state.listener_state_version += 1
    user_digest = hashlib.sha256(
        np.ascontiguousarray(user_pcm, dtype="<f4").tobytes()
    ).hexdigest()
    reference_digest = None if reference is None else hashlib.sha256(
        np.ascontiguousarray(reference.pcm, dtype="<f4").tobytes()
    ).hexdigest()
    state.state_lineage_digest = state_lineage_digest(
        state.state_lineage_digest,
        state_epoch_id=state.state_epoch_id,
        frame_id=int(frame_id),
        state_version=state.listener_state_version,
        pcm_sha256=user_digest,
        reference_sha256=reference_digest,
    )
    state.last_audio_timestamp = state.next_frame_id * int(frame_duration_ms) * 1_000_000
    state.last_reference_frame_id = None if reference is None else int(reference.frame_id)


from duplex_kernel.reference_buffer import (
    AssistantReferenceBuffer,
    AssistantReferenceFrame,
)



@dataclass
class GemmaListenerState:
    """Mutable, session-owned state for the continuous listener."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS
    context_frames: int = DEFAULT_CONTEXT_FRAMES
    next_frame_id: int = 0
    total_samples: int = 0
    listener_state_version: int = 0
    state_epoch_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state_lineage_digest: str = ""
    listener_input_frames: int = 0
    dropped_listener_frames: int = 0
    nonfinite_hidden_count: int = 0
    empty_hidden_count: int = 0
    silence_skipped_frames: int = 0
    future_reference_leaks: int = 0
    wrong_reference_frames: int = 0
    last_audio_timestamp: int = 0
    last_reference_frame_id: int | None = None
    last_lock_wait_ms: float | None = field(default=None, repr=False, compare=False)
    last_lock_hold_ms: float | None = field(default=None, repr=False, compare=False)
    last_frontend_wall_ms: float | None = field(default=None, repr=False, compare=False)
    last_audio_tower_call_wall_ms: float | None = field(default=None, repr=False, compare=False)
    text_token_ids: tuple[int, ...] = field(default_factory=tuple, repr=False)
    # Gemma language-model KV state.  The audio tower remains the native
    # encoder; only its final multimodal LM is advanced two soft tokens per
    # listener frame.  These fields are backend-owned but session-lifetime.
    gemma_lm_cache: Any | None = field(default=None, repr=False, compare=False)
    gemma_lm_shared_kv_states: Any | None = field(default=None, repr=False, compare=False)
    gemma_lm_native: Any | None = field(default=None, repr=False, compare=False)
    # Audio frames directly represented in the current local KV position space.
    gemma_lm_audio_frames: int = 0
    # Absolute ID of the first audio frame represented by the current local KV.
    gemma_lm_context_start_frame: int = 0
    # KV positions count encoded audio frames, while these IDs preserve their
    # absolute input-frame provenance across compacted zero-PCM intervals.
    gemma_lm_source_frame_ids: list[int] = field(default_factory=list, repr=False)
    gemma_lm_last_source_frame_id: int | None = None
    gemma_lm_skipped_silence_frames: int = 0
    audio_tower_rebuild_pending: bool = False
    gemma_lm_context_rollover_count: int = 0
    # Cumulative direct KV frame entries discarded (not unique source-frame count).
    gemma_lm_context_truncated_frames: int = 0
    gemma_lm_prefix_len: int = 34
    gemma_lm_prefix_attention: Any | None = field(default=None, repr=False, compare=False)
    gemma_lm_text_context: tuple[int, ...] = field(default_factory=tuple, repr=False)
    # Real two-row audio-tower outputs retained so cache rebuilds can replay
    # each observed 80 ms transaction without re-encoding a duplex window.
    gemma_lm_audio_soft_history: deque[tuple[int, Any]] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_CONTEXT_FRAMES), repr=False, compare=False
    )
    # Bounded real Q2 soft frames for semantic-prefix refresh. Frame IDs remain
    # absolute; skipped-frame gaps are never invented during replay.
    semantic_audio_soft_history: deque[tuple[int, Any]] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_CONTEXT_FRAMES), repr=False, compare=False
    )
    semantic_context_revision: int | None = field(default=None, repr=False, compare=False)
    semantic_replay_fallback_reason: str | None = field(default=None, repr=False, compare=False)
    gemma_lm_terminal_error: str | None = field(default=None, repr=False, compare=False)
    native_last_hidden: Any | None = field(default=None, repr=False, compare=False)
    last_hidden: Any | None = field(default=None, repr=False, compare=False)
    last_hidden_source_frame_id: int | None = None
    last_hidden_at_ns: int | None = None
    last_audio_queries: Any | None = field(default=None, repr=False, compare=False)
    user_pcm_history: deque[np.ndarray] = field(default_factory=deque, repr=False)
    assistant_pcm_history: deque[np.ndarray] = field(default_factory=deque, repr=False)
    user_mel_history: deque[np.ndarray] = field(default_factory=deque, repr=False)
    assistant_mel_history: deque[np.ndarray] = field(default_factory=deque, repr=False)
    user_pcm_tail: np.ndarray | None = field(default=None, repr=False)
    assistant_pcm_tail: np.ndarray | None = field(default=None, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    _closed: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.frame_duration_ms <= 0 or self.context_frames <= 0:
            raise ValueError("configurazione GemmaListenerState non valida")
        frame_samples = self.sample_rate * self.frame_duration_ms
        if frame_samples % 1000:
            raise ValueError("frame_duration_ms deve produrre un numero intero di sample")
        self.user_pcm_history = deque(self.user_pcm_history, maxlen=int(self.context_frames))
        self.assistant_pcm_history = deque(self.assistant_pcm_history, maxlen=int(self.context_frames))
        self.user_mel_history = deque(self.user_mel_history, maxlen=int(self.context_frames))
        self.assistant_mel_history = deque(self.assistant_mel_history, maxlen=int(self.context_frames))
        self.gemma_lm_audio_soft_history = deque(
            self.gemma_lm_audio_soft_history, maxlen=int(self.context_frames)
        )

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_duration_ms // 1000

    @property
    def processed_frames(self) -> int:
        return self.listener_input_frames

    @property
    def next_input_frame_id(self) -> int:
        """ID that will be assigned to the next successful listener step."""
        with self._lock:
            return int(self.next_frame_id)

    def stats(self) -> dict[str, int | None]:
        with self._lock:
            return {
                "listener_input_frames": self.listener_input_frames,
                "dropped_listener_frames": self.dropped_listener_frames,
                "samples_received": self.total_samples,
                "samples_accounted_for": self.listener_input_frames * self.frame_samples,
                "listener_state_version": self.listener_state_version,
                "state_epoch_id": self.state_epoch_id,
                "state_lineage_digest": self.state_lineage_digest,
                "nonfinite_hidden_count": self.nonfinite_hidden_count,
                "empty_hidden_count": self.empty_hidden_count,
                "future_reference_leaks": self.future_reference_leaks,
                "wrong_reference_frames": self.wrong_reference_frames,
                "last_audio_timestamp": self.last_audio_timestamp,
                "last_reference_frame_id": self.last_reference_frame_id,
                "gemma_lm_context_start_frame": self.gemma_lm_context_start_frame,
                "gemma_lm_audio_frames": self.gemma_lm_audio_frames,
                "gemma_lm_last_source_frame_id": self.gemma_lm_last_source_frame_id,
                "gemma_lm_skipped_silence_frames": self.gemma_lm_skipped_silence_frames,
                "listener_silence_skipped_frames": self.silence_skipped_frames,
                "gemma_lm_context_rollover_count": self.gemma_lm_context_rollover_count,
                "gemma_lm_context_truncated_frames": self.gemma_lm_context_truncated_frames,
                "gemma_lm_prefix_len": self.gemma_lm_prefix_len,
            }

    def invalidate_gemma_lm_state(self) -> None:
        """Discard a partial LM transaction; history will be replayed next tick."""
        with self._lock:
            native = self.gemma_lm_native
            native_reusable = native is not None
            if native is not None:
                try:
                    native.reset()
                except BaseException:
                    # A failed stream drain means the CUDA context is not safe
                    # to reuse. Close it best-effort, then preserve the
                    # original backend exception while leaving state replayable.
                    native_reusable = False
                    logger.exception("native Gemma reset fallito; contesto scartato")
                    try:
                        native.close()
                    except BaseException:
                        logger.exception("chiusura contesto nativo fallita")
            self.gemma_lm_native = native if native_reusable else None
            self.gemma_lm_cache = None
            self.gemma_lm_shared_kv_states = None
            self.gemma_lm_audio_frames = 0
            self.gemma_lm_context_start_frame = 0
            self.gemma_lm_source_frame_ids.clear()
            self.gemma_lm_last_source_frame_id = None
            self.gemma_lm_skipped_silence_frames = 0
            self.gemma_lm_text_context = ()
            self.gemma_lm_prefix_len = 34
            self.gemma_lm_prefix_attention = None
            self.semantic_context_revision = None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.user_pcm_history.clear()
            self.assistant_pcm_history.clear()
            self.user_mel_history.clear()
            self.assistant_mel_history.clear()
            self.user_pcm_tail = None
            self.assistant_pcm_tail = None
            native = self.gemma_lm_native
            try:
                if native is not None:
                    native.close()
            finally:
                self.gemma_lm_cache = None
                self.gemma_lm_shared_kv_states = None
                self.gemma_lm_native = None
                self.gemma_lm_terminal_error = None
                self.native_last_hidden = None
                self.last_hidden = None
                self.last_audio_queries = None
                self.last_hidden_source_frame_id = None
                self.last_hidden_at_ns = None
                self.semantic_audio_soft_history.clear()
                self.gemma_lm_audio_soft_history.clear()
                self.semantic_context_revision = None
                self.semantic_replay_fallback_reason = None
            self.gemma_lm_audio_frames = 0
            self.gemma_lm_context_start_frame = 0
            self.gemma_lm_source_frame_ids.clear()
            self.gemma_lm_audio_soft_history.clear()
            self.gemma_lm_last_source_frame_id = None
            self.gemma_lm_skipped_silence_frames = 0
            self.audio_tower_rebuild_pending = False
            self.gemma_lm_context_rollover_count = 0
            self.gemma_lm_context_truncated_frames = 0
            self.gemma_lm_text_context = ()
            self.gemma_lm_prefix_len = 34
            self.gemma_lm_prefix_attention = None
            self.last_hidden = None
            self.last_audio_queries = None
            self.last_hidden_source_frame_id = None
            self.last_hidden_at_ns = None
            self.semantic_audio_soft_history.clear()
            self.gemma_lm_audio_soft_history.clear()
            self.semantic_context_revision = None
            self.semantic_replay_fallback_reason = None
            self.silence_skipped_frames = 0

    def reset(self) -> None:
        with self._lock:
            self.next_frame_id = 0
            self.total_samples = 0
            self.listener_state_version = 0
            self.state_epoch_id = uuid.uuid4().hex
            self.state_lineage_digest = ""
            self.listener_input_frames = 0
            self.dropped_listener_frames = 0
            self.nonfinite_hidden_count = 0
            self.empty_hidden_count = 0
            self.future_reference_leaks = 0
            self.wrong_reference_frames = 0
            self.last_audio_timestamp = 0
            self.last_reference_frame_id = None
            self.text_token_ids = ()
            native = self.gemma_lm_native
            try:
                if native is not None:
                    native.close()
            finally:
                self.gemma_lm_cache = None
                self.gemma_lm_shared_kv_states = None
                self.gemma_lm_native = None
                self.gemma_lm_terminal_error = None
            self.native_last_hidden = None
            self.gemma_lm_audio_frames = 0
            self.gemma_lm_context_start_frame = 0
            self.gemma_lm_source_frame_ids.clear()
            self.gemma_lm_last_source_frame_id = None
            self.gemma_lm_audio_soft_history.clear()
            self.gemma_lm_skipped_silence_frames = 0
            self.audio_tower_rebuild_pending = False
            self.gemma_lm_context_rollover_count = 0
            self.gemma_lm_context_truncated_frames = 0
            self.gemma_lm_text_context = ()
            self.gemma_lm_prefix_len = 34
            self.gemma_lm_prefix_attention = None
            self.semantic_context_revision = None
            self.semantic_replay_fallback_reason = None
            self.last_hidden = None
            self.last_audio_queries = None
            self.last_hidden_source_frame_id = None
            self.last_hidden_at_ns = None
            self.silence_skipped_frames = 0
            self.user_pcm_history.clear()
            self.assistant_pcm_history.clear()
            self.user_mel_history.clear()
            self.assistant_mel_history.clear()
            self.user_pcm_tail = None
            self.assistant_pcm_tail = None
            self._closed = False


class ListenerBackend(Protocol):
    """Backend boundary; unit tests may inject a deterministic implementation."""

    def encode(
        self,
        user_pcm: np.ndarray,
        assistant_reference: AssistantReferenceFrame | None,
        state: GemmaListenerState,
    ) -> Any:
        ...


def _native_q4_profile_step_kwargs(native: Any, state: GemmaListenerState) -> dict[str, Any]:
    """Pass sampled native Q4 frames the host frame identity already in scope."""
    if not bool(getattr(native, "q4_cuda_profile_enabled", False)):
        return {}
    host_context = getattr(state, "_native_q4_profile_host_context", None)
    frame_id = host_context.get("frame_id") if isinstance(host_context, dict) else None
    return {
        "profile_frame_id": frame_id,
        "profile_host_context": host_context,
    }


@dataclass(frozen=True)
class ListenerBackendOutput:
    """Both native audio soft tokens and the canonical multimodal LM hidden."""

    gemma_hidden_t: Any
    audio_tower_hidden: Any | None = None
    gemma_audio_queries_t: Any | None = None


@dataclass(frozen=True, slots=True)
class AudioTowerRowTiming:
    """Host submit time and async row intervals for one audio tower call."""

    graph_submit_ms: float
    copy_cuda_events: tuple[Any, Any] | None
    tower_cuda_events: tuple[Any, Any] | None
    execution_mode: str
    workspace_lock_wait_ms: float = 0.0
    workspace_lock_hold_ms: float = 0.0


def _select_audio_tower_rows(
    pooled: Any,
    valid_mask: Any,
    *,
    selected_q_start: int | None,
    all_valid_mask: bool,
    hidden_size: int,
    device: Any,
    dtype: Any,
) -> Any:
    """Snapshot valid tower rows, skipping compaction only under an all-valid contract."""
    import torch

    if (
        not isinstance(pooled, torch.Tensor)
        or pooled.ndim != 3
        or pooled.shape[0] != 1
        or pooled.shape[-1] != hidden_size
        or not isinstance(valid_mask, torch.Tensor)
        or valid_mask.ndim != 2
        or tuple(valid_mask.shape) != tuple(pooled.shape[:2])
        or valid_mask.dtype != torch.bool
    ):
        raise RuntimeError("Gemma audio tower returned incompatible hidden rows or mask")

    if selected_q_start is not None and (
        isinstance(selected_q_start, bool)
        or not isinstance(selected_q_start, int)
        or selected_q_start < 0
    ):
        raise ValueError("audio tower Q start must be a non-negative integer")

    if all_valid_mask:
        if selected_q_start is None:
            return pooled.to(device=device, dtype=dtype).clone()
        selected_q_end = selected_q_start + AUDIO_FEATURES_PER_FRAME
        if pooled.shape[1] < selected_q_end:
            raise RuntimeError("Gemma audio tower returned fewer than 2 valid Q2 rows")
        return pooled[:, selected_q_start:selected_q_end, :].to(
            device=device, dtype=dtype
        ).clone()

    valid = pooled[0][valid_mask[0].to(device=pooled.device, dtype=torch.bool)]
    if valid.numel() == 0 or valid.shape[-1] != hidden_size:
        raise RuntimeError("Gemma audio tower ha restituito hidden vuoto/incompatibile")
    if selected_q_start is None:
        return valid.unsqueeze(0).to(device=device, dtype=dtype).clone()
    selected_q_end = selected_q_start + AUDIO_FEATURES_PER_FRAME
    if valid.shape[0] < selected_q_end:
        raise RuntimeError("Gemma audio tower returned fewer than 2 valid Q2 rows")
    return valid[selected_q_start:selected_q_end].unsqueeze(0).to(
        device=device, dtype=dtype
    ).clone()


class GemmaAudioTowerBackend:
    """Gemma multimodal backend used by the continuous listener.

    The V9-facing value is *not* ``pooler_output``.  For a real Thinker it is
    the final hidden state of the multimodal language model at the first audio
    feature of the current 80 ms frame, before V9's causal convolution.  The
    direct audio-tower path remains only as a compatibility fallback for the
    small deterministic test backend.
    """

    def __init__(
        self,
        thinker: Any,
        *,
        processor: Any | None = None,
        device: str | None = None,
        text_token_ids: tuple[int, ...] | list[int] | None = None,
        use_native_lm: bool | None = None,
        runtime_profile: str = "compatibility",
        attention_backend: str | int | None = None,
        rmsnorm_backend: str | int | None = None,
        mel_window_rows: int | None = None,
    ) -> None:
        from duplex_kernel.listener_microbatch_frontend import (
            MEL_WINDOW_ROWS,
            mel_window_geometry,
        )

        geometry = mel_window_geometry(
            MEL_WINDOW_ROWS if mel_window_rows is None else mel_window_rows
        )
        self.mel_window_rows = geometry.mel_window_rows
        self.mel_context_rows = geometry.context_rows
        self.tower_output_rows = geometry.tower_output_rows
        self.current_q_start = geometry.q_start
        self.mel_history_frames = geometry.required_history_frames
        if runtime_profile not in {"compatibility", "production", "oracle"}:
            raise ValueError(
                "runtime_profile deve essere compatibility, production oppure oracle"
            )
        self.runtime_profile = runtime_profile
        if runtime_profile == "production" and use_native_lm is False:
            raise ValueError("il profilo production richiede il listener nativo")
        if runtime_profile == "production" and text_token_ids:
            raise ValueError("il profilo production non accetta text_token_ids nel listener")
        wrapper = thinker
        model = getattr(wrapper, "model", wrapper)
        gemma_model = getattr(model, "model", None)
        if gemma_model is None or not callable(getattr(gemma_model, "get_audio_features", None)):
            raise TypeError("thinker non espone Gemma4Model.get_audio_features")
        self.model = model
        self.multimodal_model = model if callable(model) else None
        self.gemma_model = gemma_model
        self.processor = processor or getattr(wrapper, "processor", None)
        self.feature_extractor = getattr(self.processor, "feature_extractor", self.processor)
        if not callable(self.feature_extractor):
            raise TypeError("processor Gemma audio non disponibile")
        from duplex_kernel.config import (
            LISTENER_PAIR_LOCK_HANDOFF_ENABLED,
            NATIVE_LISTENER_CUDA_PROFILE_ENABLED,
        )

        self._lock = FairModelLock(
            listener_pair_lock_handoff=LISTENER_PAIR_LOCK_HANDOFF_ENABLED,
            host_timeline_enabled=NATIVE_LISTENER_CUDA_PROFILE_ENABLED,
        )
        # Graph inputs and outputs are backend-shared static storage. Keep one
        # short workspace lease through replay and the per-call output snapshot.
        self._audio_tower_workspace_lock = threading.RLock()
        self.text_token_ids = tuple(int(value) for value in (text_token_ids or ()))
        config = getattr(gemma_model, "config", None)
        text_config = getattr(config, "text_config", None)
        self.hidden_size = int(getattr(text_config, "hidden_size", 1536))
        self.audio_token_id = int(getattr(config, "audio_token_id", 258881))
        self.boa_token_id = int(getattr(config, "boa_token_id", 256000))
        self.eoa_token_id = int(getattr(config, "eoa_token_id", 258883))
        self.bos_token_id = int(getattr(config, "bos_token_id", 2))
        self.pad_token_id = int(getattr(text_config, "pad_token_id", 0))
        language_model = getattr(self.gemma_model, "language_model", None)
        self._lm_has_shared_kv = any(
            bool(getattr(getattr(layer, "self_attn", None), "is_kv_shared_layer", False))
            for layer in getattr(language_model, "layers", ())
        )
        graph_selector = os.environ.get(_NATIVE_LISTENER_CUDA_GRAPH_ENV, "0").strip()
        if graph_selector not in {"0", "1"}:
            raise ValueError(f"{_NATIVE_LISTENER_CUDA_GRAPH_ENV} must be exactly '0' or '1'")
        self._native_cuda_graph_requested = graph_selector == "1"
        if self._native_cuda_graph_requested and runtime_profile != "production":
            raise RuntimeError("native listener CUDA Graph opt-in richiede runtime_profile=production")

        self._shape_stable_lm = None
        self._native_attention_backend: int | None = None
        self._native_rmsnorm_backend: int | None = None
        self._native_attention_backend_source: str | None = None
        self._canonical_prefix_cache_enabled = False
        self._canonical_prefix_cache_snapshot: Any | None = None
        self._canonical_prefix_cache_signature: tuple[Any, ...] | None = None
        self._canonical_prefix_compute_count = 0
        self._canonical_prefix_reuse_count = 0
        if runtime_profile == "production":
            from duplex_kernel.native_listener import (
                GEMMA4_ATTN_CUSTOM,
                GEMMA4_RMS_ATEN_EXACT,
                GEMMA4_RMS_NATIVE_EXACT,
                resolve_native_attention_backend,
                parse_rmsnorm_backend,
            )

            env_present = bool(os.environ.get("GEMMA4_ATTENTION_BACKEND", "").strip())
            if self._native_cuda_graph_requested:
                configured_attention = os.environ.get("GEMMA4_ATTENTION_BACKEND")
                if attention_backend is not None and resolve_native_attention_backend(attention_backend) != GEMMA4_ATTN_CUSTOM:
                    raise RuntimeError("native listener CUDA Graph richiede backend attenzione CUSTOM")
                if configured_attention is not None and configured_attention.strip():
                    if resolve_native_attention_backend(configured_attention) != GEMMA4_ATTN_CUSTOM:
                        raise RuntimeError("native listener CUDA Graph richiede GEMMA4_ATTENTION_BACKEND=CUSTOM")
                self._native_attention_backend = GEMMA4_ATTN_CUSTOM
                self._native_attention_backend_source = "native_listener_cuda_graph_opt_in"
            else:
                self._native_attention_backend = resolve_native_attention_backend(attention_backend)
                self._native_attention_backend_source = (
                    "explicit" if attention_backend is not None
                    else "environment" if env_present else "production_default"
                )
            self._native_rmsnorm_backend = (
                GEMMA4_RMS_NATIVE_EXACT
                if rmsnorm_backend is None
                else parse_rmsnorm_backend(rmsnorm_backend)
            )
            if (
                self._native_cuda_graph_requested
                and self._native_rmsnorm_backend == GEMMA4_RMS_ATEN_EXACT
            ):
                raise RuntimeError("native listener CUDA Graph non supporta GEMMA4_RMSNORM_BACKEND=ATEN_EXACT")
            self._use_native_lm = True
        elif runtime_profile == "oracle":
            from duplex_kernel.native_listener import (
                GEMMA4_ATTN_ATEN_REPEAT_EXACT,
                parse_rmsnorm_backend,
                resolve_native_attention_backend,
            )

            self._native_attention_backend = resolve_native_attention_backend(
                GEMMA4_ATTN_ATEN_REPEAT_EXACT if attention_backend is None else attention_backend
            )
            self._native_attention_backend_source = "oracle_default" if attention_backend is None else "explicit"
            self._native_rmsnorm_backend = (
                None if rmsnorm_backend is None else parse_rmsnorm_backend(rmsnorm_backend)
            )
            self._use_native_lm = True
        else:
            self._use_native_lm = (
                bool(use_native_lm)
                if use_native_lm is not None
                else os.environ.get("GEMMA4_NATIVE_LISTENER", "0") == "1"
            )
            if attention_backend is not None:
                from duplex_kernel.native_listener import resolve_native_attention_backend
                self._native_attention_backend = resolve_native_attention_backend(attention_backend)
                self._native_attention_backend_source = "explicit"
            if rmsnorm_backend is not None:
                from duplex_kernel.native_listener import parse_rmsnorm_backend
                self._native_rmsnorm_backend = parse_rmsnorm_backend(rmsnorm_backend)
        self._frame_pad_ple: Any | None = None
        self._native_runtime: Any | None = None
        if language_model is not None:
            from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM

            # This is a zero-weight-copy functional view.  Its cache/workspace
            # is listener-owned; the normal Thinker keeps its native forward.
            self._shape_stable_lm = ShapeStableGemmaListenerLM(
                language_model, target_length=CANONICAL_LM_PREFILL_TARGET_LEN
            )
        try:
            import torch
            parameter = next(self.gemma_model.audio_tower.parameters())
            self.device = torch.device(device or parameter.device)
            self.dtype = parameter.dtype
            self._cuda_graph: Any | None = None
            self._static_features: Any | None = None
            self._static_shift: Any | None = None
            self._static_new_slice: Any | None = None
            self._static_mask: Any | None = None
            self._graph_output: Any | None = None
            self._graph_stream: Any | None = None
            self._waiting_for_lm = threading.Event()
            self._listener_waiter_priority = _ListenerWaiterPriority(self._waiting_for_lm)
            self._frame_done_event = threading.Event()
            self._frame_done_event.set()
            self._tick_started_event = threading.Event()
            self._tick_started_event.set()
            self._frame_counter = 0
            self._active_streaming = False
            if self.device.type == "cuda":
                self._warmup_cuda_graph()
        except (ImportError, AttributeError, StopIteration) as exc:
            raise TypeError("audio tower Gemma non disponibile") from exc

    @staticmethod
    def _clone_canonical_prefix_snapshot(past_key_values: Any) -> Any:
        """Clone an owned read-only DynamicCache snapshot for native D2D imports.

        Transformers Cache updates mutate cache layers in place (v5.5.0 API:
        https://huggingface.co/docs/transformers/v5.5.0/en/internal/generation_utils),
        so the immutable provider snapshot must own every K/V tensor.
        """
        from transformers.cache_utils import DynamicCache

        source_layers = getattr(past_key_values, "layers", None)
        if source_layers is None:
            raise TypeError("prefisso Gemma non ha un DynamicCache clonabile")
        snapshot = DynamicCache()
        for layer_idx, layer in enumerate(source_layers):
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                raise RuntimeError(f"DynamicCache canonica incompleta al layer {layer_idx}")
            snapshot.update(keys.detach().clone(), values.detach().clone(), layer_idx)
        shared_layers = getattr(past_key_values, "shared_layers", None)
        if shared_layers:
            snapshot.shared_layers = {
                int(layer_idx): (keys.detach().clone(), values.detach().clone())
                for layer_idx, (keys, values) in shared_layers.items()
            }
        return snapshot

    def configure_canonical_prefix_cache(self, *, enabled: bool, clear: bool = False) -> None:
        """Enable backend-local canonical-prefix reuse and optionally clear its counters."""
        self._canonical_prefix_cache_enabled = bool(enabled)
        if clear:
            self._canonical_prefix_cache_snapshot = None
            self._canonical_prefix_cache_signature = None
            self._canonical_prefix_compute_count = 0
            self._canonical_prefix_reuse_count = 0

    def canonical_prefix_cache_stats(self) -> dict[str, int | bool]:
        return {
            "enabled": self._canonical_prefix_cache_enabled,
            "prefix_compute_count": self._canonical_prefix_compute_count,
            "prefix_reuse_count": self._canonical_prefix_reuse_count,
            "snapshot_cached": self._canonical_prefix_cache_snapshot is not None,
        }

    @property
    def native_max_seq_len(self) -> int:
        """Capacity selected by the actual native backend contract."""
        if self._native_cuda_graph_requested:
            # Graph opt-in is constructor-gated to CUSTOM, whose native kernel
            # has a hard 4096-token capacity. Keep legacy production unchanged.
            return NATIVE_LM_CUSTOM_MAX_SEQ_LEN
        if (
            os.environ.get("GEMMA4_NATIVE_LISTENER_ALLOW_CUSTOM_EAGER_DIAGNOSTIC") == "1"
            and os.environ.get("GEMMA4_NATIVE_LISTENER_CUDA_GRAPH", "0") == "0"
            and os.environ.get("GEMMA4_ATTENTION_BACKEND", "").strip().upper() == "CUSTOM"
            and self._native_attention_backend == 0
            and os.environ.get("GEMMA4_RMSNORM_BACKEND", "").strip().upper() == "NATIVE_EXACT"
        ):
            # The explicit eager diagnostic uses the same CUSTOM kernels and
            # therefore the same hard capacity as the graph route.
            return NATIVE_LM_CUSTOM_MAX_SEQ_LEN
        if self.runtime_profile == "production":
            return NATIVE_LM_EXACT_MAX_SEQ_LEN
        if self._native_attention_backend in {
            1, 2, 3,
        } and self._native_rmsnorm_backend in {1, 2}:
            return NATIVE_LM_EXACT_MAX_SEQ_LEN
        return _native_lm_max_seq_len()

    def _warmup_cuda_graph(self) -> None:
        if self.device.type != "cuda":
            return
        import torch
        if self._cuda_graph is None and self.multimodal_model is not None:
            self._static_features = torch.zeros(
                (1, self.mel_window_rows, 128), device=self.device, dtype=self.dtype
            )
            self._static_shift = torch.empty(
                (1, self.mel_context_rows, 128), device=self.device, dtype=self.dtype
            )
            self._static_new_slice = torch.empty((1, 8, 128), device=self.device, dtype=self.dtype)
            self._static_mask = torch.ones(
                (1, self.mel_window_rows), device=self.device, dtype=torch.bool
            )
            self._graph_stream = torch.cuda.Stream(priority=-1)
            self._graph_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._graph_stream):
                for _ in range(3):
                    with torch.inference_mode():
                        self._graph_output = self.gemma_model.get_audio_features(
                            input_features=self._static_features,
                            input_features_mask=self._static_mask,
                            return_dict=True,
                        )
            torch.cuda.current_stream().wait_stream(self._graph_stream)
            self._cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._cuda_graph, stream=self._graph_stream):
                with torch.inference_mode():
                    self._graph_output = self.gemma_model.get_audio_features(
                        input_features=self._static_features,
                        input_features_mask=self._static_mask,
                        return_dict=True,
                    )

    @contextmanager
    def _audio_tower_workspace_lease(self, timing: dict[str, float] | None = None):
        wait_started_ns = time.perf_counter_ns()
        self._audio_tower_workspace_lock.acquire()
        acquired_ns = time.perf_counter_ns()
        if timing is not None:
            timing["workspace_lock_wait_ms"] = max(0, acquired_ns - wait_started_ns) / 1_000_000.0
        try:
            yield
        finally:
            released_ns = time.perf_counter_ns()
            if timing is not None:
                timing["workspace_lock_hold_ms"] = max(0, released_ns - acquired_ns) / 1_000_000.0
            self._audio_tower_workspace_lock.release()

    def _replay_shared_audio_tower(
        self,
        update_workspace: Any,
        *,
        native_session_steady: bool,
        selected_q_start: int | None = None,
        all_valid_output_mask: bool = False,
        cuda_event_pairs: tuple[tuple[Any, Any], tuple[Any, Any]] | None = None,
        workspace_lock_timing: dict[str, float] | None = None,
    ) -> Any:
        """Replay shared graph storage and snapshot its per-call features before reuse."""
        import torch

        with self._audio_tower_workspace_lease(workspace_lock_timing):
            if self._cuda_graph is None:
                # Lazy graph creation mutates the same static workspace, so it
                # belongs to this lease as well as every later replay.
                self._warmup_cuda_graph()
            graph_stream = self._graph_stream
            if self._cuda_graph is None or graph_stream is None:
                raise RuntimeError("CUDA graph audio workspace non inizializzato")

            consumer_stream = torch.cuda.current_stream(device=self.device)
            consumer_stream.wait_stream(graph_stream)
            with torch.cuda.stream(consumer_stream):
                if cuda_event_pairs is not None:
                    cuda_event_pairs[0][0].record(consumer_stream)
                update_workspace()
                if cuda_event_pairs is not None:
                    cuda_event_pairs[0][1].record(consumer_stream)
            graph_stream.wait_stream(consumer_stream)

            audio_tower_started = time.perf_counter()
            with torch.cuda.stream(graph_stream):
                if cuda_event_pairs is not None:
                    cuda_event_pairs[1][0].record(graph_stream)
                self._cuda_graph.replay()
                pooled = getattr(self._graph_output, "pooler_output", None)
                if pooled is None or pooled.ndim != 3:
                    raise RuntimeError("Gemma audio graph non ha restituito pooler_output valido")
                if native_session_steady:
                    if pooled.shape[1] < AUDIO_FEATURES_PER_FRAME:
                        raise RuntimeError("Gemma audio tower native ha restituito shape inattesa")
                    # Clone only the Q=2 rows consumed by this listener tick.
                    snapshot = pooled[:, -AUDIO_FEATURES_PER_FRAME:, :].clone()
                else:
                    valid_mask = getattr(self._graph_output, "attention_mask", None)
                    if valid_mask is None:
                        raise RuntimeError("Gemma audio graph non ha restituito la mask encoder")
                    snapshot = _select_audio_tower_rows(
                        pooled,
                        valid_mask,
                        selected_q_start=selected_q_start,
                        all_valid_mask=all_valid_output_mask,
                        hidden_size=self.hidden_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                if cuda_event_pairs is not None:
                    cuda_event_pairs[1][1].record(graph_stream)

            # The clone is ordered after replay on graph_stream; the consumer
            # wait makes it safe for the later native input copy without a host
            # or device-wide synchronization.
            consumer_stream.wait_stream(graph_stream)
            snapshot.record_stream(consumer_stream)
            elapsed_ms = (time.perf_counter() - audio_tower_started) * 1000.0
            if cuda_event_pairs is not None:
                return snapshot, elapsed_ms, cuda_event_pairs
            return snapshot, elapsed_ms

    def run_audio_tower_q4(
        self,
        features: Any,
        q4_start: int,
    ) -> tuple[Any, AudioTowerRowTiming]:
        """Run one exact-geometry audio-tower row and clone its selected Q4.

        The frontend supplies the production fixed-width, all-valid mask
        invariant. This method owns the shared graph buffers and keeps the
        selected Q4 alive before those buffers can be reused by the next row.
        """
        import torch

        if not isinstance(features, torch.Tensor) or tuple(features.shape) != (1, 1024, 128):
            raise ValueError("audio tower row input must have exact shape [1,1024,128]")
        if features.device != self.device or features.dtype != self.dtype:
            raise ValueError("audio tower row input must match the backend device and dtype")
        if isinstance(q4_start, bool) or not isinstance(q4_start, int) or q4_start < 0:
            raise ValueError("audio tower Q4 start must be a non-negative integer")
        if self.device.type != "cuda":
            raise RuntimeError("the shared audio tower row runner requires a CUDA backend")

        from duplex_kernel.listener_microbatch_frontend import AudioTowerRowTiming

        copy_events = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        tower_events = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        graph_submit_started = time.perf_counter_ns()
        workspace_lock_timing: dict[str, float] = {}
        if (
            self._cuda_graph is not None
            and self._graph_stream is not None
            and self._static_features is not None
            and tuple(self._static_features.shape) == tuple(features.shape)
        ):
            if self._static_features is None or self._static_mask is None:
                raise RuntimeError("CUDA graph row workspace is incomplete")

            def update_workspace() -> None:
                self._static_features.copy_(features, non_blocking=True)

            selected, _wall_ms, _events = self._replay_shared_audio_tower(
                update_workspace,
                native_session_steady=False,
                selected_q_start=q4_start,
                cuda_event_pairs=(copy_events, tower_events),
                workspace_lock_timing=workspace_lock_timing,
            )
            execution_mode = "cuda_graph_n1"
        else:
            # Legacy 1024-row Q4 callers cannot reuse the configured Q2 graph.
            # Serving routes the user path through run_audio_tower_q2.
            stream = torch.cuda.current_stream(device=self.device)
            input_mask = torch.ones((1, 1024), dtype=torch.bool, device=self.device)
            tower_events[0].record(stream)
            with torch.inference_mode():
                output = self.gemma_model.get_audio_features(
                    input_features=features,
                    input_features_mask=input_mask,
                    return_dict=True,
                )
            pooled = getattr(output, "pooler_output", None)
            valid_mask = getattr(output, "attention_mask", None)
            if (
                not isinstance(pooled, torch.Tensor)
                or pooled.ndim != 3
                or not isinstance(valid_mask, torch.Tensor)
                or valid_mask.ndim != 2
                or tuple(valid_mask.shape) != tuple(pooled.shape[:2])
                or pooled.shape[0] != 1
                or pooled.shape[-1] != self.hidden_size
            ):
                raise RuntimeError("Gemma eager audio tower returned an incompatible output")
            valid = pooled[0][valid_mask[0].to(device=pooled.device, dtype=torch.bool)]
            end = q4_start + AUDIO_FEATURES_PER_FRAME
            if valid.shape[0] < end:
                raise RuntimeError("Gemma eager audio tower returned fewer rows than the Q2 selection requires")
            selected = valid[q4_start:end].unsqueeze(0).to(
                device=self.device, dtype=self.dtype
            ).clone()
            tower_events[1].record(stream)
            selected.record_stream(stream)
            execution_mode = "eager_n1_degraded"

        if tuple(selected.shape) != (1, AUDIO_FEATURES_PER_FRAME, self.hidden_size):
            raise RuntimeError("audio tower row runner did not snapshot exactly one Q2 frame")
        graph_submit_ms = max(
            0, time.perf_counter_ns() - graph_submit_started
        ) / 1_000_000.0
        graph_submit_ms = max(
            0.0,
            graph_submit_ms - workspace_lock_timing.get("workspace_lock_wait_ms", 0.0),
        )
        return selected, AudioTowerRowTiming(
            graph_submit_ms=round(graph_submit_ms, 3),
            copy_cuda_events=(copy_events if execution_mode == "cuda_graph_n1" else None),
            tower_cuda_events=tower_events,
            execution_mode=execution_mode,
            workspace_lock_wait_ms=workspace_lock_timing.get("workspace_lock_wait_ms", 0.0),
            workspace_lock_hold_ms=workspace_lock_timing.get("workspace_lock_hold_ms", 0.0),
        )

    def run_audio_tower_q2(
        self,
        features: Any,
        q_start: int,
    ) -> tuple[Any, AudioTowerRowTiming]:
        """Encode one causal user Mel prefix/window and return its current Q=2."""
        import torch

        if (
            not isinstance(features, torch.Tensor)
            or features.ndim != 3
            or features.shape[0] != 1
            or features.shape[2] != 128
            or features.shape[1] < 8
            or features.shape[1] > self.mel_window_rows
            or features.shape[1] % 4 != 0
        ):
            raise ValueError(
                f"Q=2 audio tower input must have shape [1,L,128], 8<=L<={self.mel_window_rows}, L multiple of 4"
            )
        valid_rows = int(features.shape[1])
        expected_q_start = valid_rows // AUDIO_TOWER_MEL_STRIDE - AUDIO_FEATURES_PER_FRAME
        if features.device != self.device or features.dtype != self.dtype:
            raise ValueError("audio tower Q=2 input must match the backend device and dtype")
        if q_start != expected_q_start:
            raise ValueError(
                f"Q=2 audio tower must select the final rows at q_start={expected_q_start}"
            )
        if self.device.type != "cuda":
            raise RuntimeError("the production Q=2 audio tower row runner requires CUDA")

        if valid_rows < self.mel_window_rows:
            # Startup prefixes have no left-padding: their valid Mel rows are
            # cropped before the tower. Use eager execution until the one
            # configured full-window graph is applicable.
            copy_events = None
            tower_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            graph_submit_started = time.perf_counter_ns()
            tower_events[0].record(torch.cuda.current_stream(device=self.device))
            with torch.inference_mode():
                output = self.gemma_model.get_audio_features(
                    input_features=features,
                    input_features_mask=torch.ones(
                        (1, valid_rows), device=self.device, dtype=torch.bool
                    ),
                    return_dict=True,
                )
            tower_events[1].record(torch.cuda.current_stream(device=self.device))
            graph_submit_ms = max(
                0, time.perf_counter_ns() - graph_submit_started
            ) / 1_000_000.0
            pooled = getattr(output, "pooler_output", None)
            if not isinstance(pooled, torch.Tensor) or tuple(pooled.shape) != (
                1, valid_rows // AUDIO_TOWER_MEL_STRIDE, self.hidden_size
            ):
                raise RuntimeError("startup Q=2 audio tower returned an unexpected shape")
            selected = pooled[:, q_start:q_start + AUDIO_FEATURES_PER_FRAME, :].detach().clone()
            if tuple(selected.shape) != (1, AUDIO_FEATURES_PER_FRAME, self.hidden_size):
                raise RuntimeError("startup Q=2 audio tower did not return the final two rows")
            return selected, AudioTowerRowTiming(
                graph_submit_ms=round(graph_submit_ms, 3),
                copy_cuda_events=None,
                tower_cuda_events=tower_events,
                execution_mode="eager_n1_degraded",
            )

        if self._cuda_graph is None or self._graph_stream is None:
            self._warmup_cuda_graph()
        if self._cuda_graph is None or self._graph_stream is None:
            raise RuntimeError("Q=2 CUDA graph audio workspace non inizializzato")
        copy_events = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        tower_events = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        if self._static_features is None or tuple(self._static_features.shape) != (
            1, self.mel_window_rows, 128
        ):
            raise RuntimeError("Q=2 CUDA graph has incompatible Mel window storage")

        def update_workspace() -> None:
            self._static_features.copy_(features, non_blocking=True)

        graph_submit_started = time.perf_counter_ns()
        workspace_lock_timing: dict[str, float] = {}
        selected, _wall_ms, _events = self._replay_shared_audio_tower(
            update_workspace,
            native_session_steady=False,
            selected_q_start=q_start,
            all_valid_output_mask=True,
            cuda_event_pairs=(copy_events, tower_events),
            workspace_lock_timing=workspace_lock_timing,
        )
        graph_submit_ms = max(
            0, time.perf_counter_ns() - graph_submit_started
        ) / 1_000_000.0
        graph_submit_ms = max(
            0.0,
            graph_submit_ms - workspace_lock_timing.get("workspace_lock_wait_ms", 0.0),
        )
        return selected, AudioTowerRowTiming(
            graph_submit_ms=round(graph_submit_ms, 3),
            copy_cuda_events=copy_events,
            tower_cuda_events=tower_events,
            execution_mode="cuda_graph_n1",
            workspace_lock_wait_ms=workspace_lock_timing.get("workspace_lock_wait_ms", 0.0),
            workspace_lock_hold_ms=workspace_lock_timing.get("workspace_lock_hold_ms", 0.0),
        )

    def _extract_single_frame(
        self,
        wave_1280: np.ndarray,
        past_160: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Incremental 80 ms mel extraction (8 mel rows), bit-exact to full rolling history."""
        fe = self.feature_extractor
        if not hasattr(fe, "frame_length") or not hasattr(fe, "mel_filters"):
            res = self._extract([wave_1280], rows=8)[0]
            return res, wave_1280[-160:]

        curr = np.asarray(wave_1280, dtype=np.float32)
        if past_160 is None or len(past_160) == 0:
            past = np.zeros(160, dtype=np.float32)
        else:
            past = np.asarray(past_160, dtype=np.float32)
            if len(past) < 160:
                past = np.pad(past, (160 - len(past), 0), mode="constant")
            elif len(past) > 160:
                past = past[-160:]

        padded_1441 = np.concatenate((past, curr, np.zeros(1, dtype=np.float32)))
        frames_to_process = np.lib.stride_tricks.sliding_window_view(padded_1441, window_shape=321)[::160]

        if fe.preemphasis > 0.0:
            if getattr(fe, "preemphasis_htk_flavor", False):
                first_in_frame = frames_to_process[..., :1] * (1.0 - fe.preemphasis)
                rest_in_frame = frames_to_process[..., 1:-1] - fe.preemphasis * frames_to_process[..., :-2]
                frames = np.concatenate([first_in_frame, rest_in_frame], axis=-1)
            else:
                frames = frames_to_process[..., 1:] - fe.preemphasis * frames_to_process[..., :-1]
        else:
            frames = frames_to_process[..., :-1]

        frames = frames * fe.window
        stft = np.fft.rfft(frames, n=fe.fft_length, axis=-1)
        mag = np.abs(stft)
        mel = np.matmul(mag, fe.mel_filters)
        log_mel = np.log(mel + fe.mel_floor)
        if fe.per_bin_mean is not None:
            log_mel = log_mel - fe.per_bin_mean
        if fe.per_bin_stddev is not None:
            log_mel = log_mel / fe.per_bin_stddev
        return log_mel.astype(np.float32), curr[-160:]

    def advance_silent_frame(self, state: GemmaListenerState) -> None:
        """Advance exact PCM/mel history without running the tower or LM."""
        if self.multimodal_model is None or not state.user_pcm_history:
            raise RuntimeError("silence-skip richiede il frontend Gemma canonico e una history PCM")
        user_mel, state.user_pcm_tail = self._extract_single_frame(
            state.user_pcm_history[-1], state.user_pcm_tail
        )
        assistant_mel, state.assistant_pcm_tail = self._extract_single_frame(
            state.assistant_pcm_history[-1], state.assistant_pcm_tail
        )
        state.user_mel_history.append(user_mel)
        state.assistant_mel_history.append(assistant_mel)
        # The static CUDA feature graph did not observe this frame. Rebuild its
        # rolling feature window on the next non-silent frame before reuse.
        state.audio_tower_rebuild_pending = True

    def _extract(self, waves: list[np.ndarray], *, rows: int | None = None) -> list[np.ndarray]:
        """Extract the same 100 fps mel convention used by the cache builder."""
        kwargs = {
            "padding": False,
            "truncation": False,
            "max_length": None,
            "pad_to_multiple_of": None,
            "return_tensors": "np",
        }
        prepared = []
        for wave in waves:
            # With the semicausal 160-sample left pad, one sentinel sample
            # makes an N*1280-sample history yield exactly N*8 mel rows.
            prepared.append(np.concatenate((np.asarray(wave, dtype=np.float32), np.zeros(1, dtype=np.float32))))
        try:
            features = self.feature_extractor(prepared, **kwargs)
        except TypeError:
            # Keeps compatibility with the same official extractor when an
            # older processor omits one of the optional keyword arguments.
            features = self.feature_extractor(prepared, padding=False)
        values = features["input_features"]
        result: list[np.ndarray] = []
        for value in values:
            array = np.asarray(value, dtype=np.float32)
            if array.ndim != 2 or array.shape[-1] != 128:
                raise RuntimeError(f"feature Gemma inattesa: {array.shape}")
            if rows is not None:
                array = self._pad_rows(array, rows)
            result.append(array)
        return result

    @staticmethod
    def _pad_rows(array: np.ndarray, rows: int) -> np.ndarray:
        if len(array) == rows:
            return array
        if len(array) > rows:
            return array[:rows]
        return np.pad(array, ((0, rows - len(array)), (0, 0)), mode="constant")

    def _build_multimodal_inputs(self, text_ids: tuple[int, ...], audio_feature_rows: int, *, device: Any):
        return build_audio_placeholder_inputs(
            batch_size=1,
            audio_feature_rows=audio_feature_rows,
            device=device,
            audio_token_id=self.audio_token_id,
            bos_token_id=self.bos_token_id,
            boa_token_id=self.boa_token_id,
            eoa_token_id=self.eoa_token_id,
            pad_token_id=self.pad_token_id,
            text_token_ids=text_ids,
        )

    def _forward_incremental_lm(
        self,
        *,
        embeds: Any,
        per_layer: Any,
        attention: Any,
        positions: Any,
        state: GemmaListenerState,
        input_ids: Any | None = None,
    ) -> tuple[Any, Any]:
        """Run one listener-only Gemma LM chunk and commit its KV.

        The canonical ShapeStable implementation is the default. Set
        ``GEMMA4_NATIVE_LISTENER=1`` (or pass ``use_native_lm=True``) only
        after the native BF16 parity gate has been accepted for the deployed
        checkpoint.
        """
        import torch

        lm = self.gemma_model.language_model
        if self._use_native_lm:
            if state.gemma_lm_terminal_error is not None:
                raise RuntimeError(
                    "executor Gemma nativo in stato terminale; chiamare state.reset() "
                    f"({state.gemma_lm_terminal_error})"
                )
            from duplex_kernel.native_listener import NativeGemmaListenerLM

            native = state.gemma_lm_native
            if native is None:
                try:
                    native = NativeGemmaListenerLM(
                        lm,
                        max_seq_len=self.native_max_seq_len,
                        stream_q_len=AUDIO_FEATURES_PER_FRAME,
                        enable_cuda_graph=self._native_cuda_graph_requested,
                        attention_backend=self._native_attention_backend,
                        rmsnorm_backend=self._native_rmsnorm_backend,
                        attention_backend_source=self._native_attention_backend_source,
                    )
                    if self.runtime_profile == "production":
                        from duplex_kernel.native_listener import assert_production_native_runtime
                        assert_production_native_runtime(native.native_runtime_info())
                except BaseException as exc:
                    state.gemma_lm_terminal_error = str(exc)
                    raise
                state.gemma_lm_native = native
            self._native_runtime = native
            q_len = int(embeds.shape[1])
            if q_len == AUDIO_START_AFTER_TEXT:
                if input_ids is None or attention is None:
                    raise RuntimeError("prime nativo richiede input_ids e mask canonici")
                if (
                    input_ids.ndim != 2
                    or input_ids.shape[0] != 1
                    or input_ids.shape[1] < AUDIO_START_AFTER_TEXT
                    or input_ids.dtype != torch.long
                    or input_ids.device != embeds.device
                ):
                    raise ValueError("input_ids nativi non hanno il prefisso canonico")
                prefix_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
                if int(prefix_ids[0, 0].item()) != self.bos_token_id or int(prefix_ids[0, -1].item()) != self.boa_token_id:
                    raise ValueError("prefisso nativo privo di BOS/BOA canonici")
                prefix_attention = attention[:, :AUDIO_START_AFTER_TEXT]
                expected_mask = torch.zeros_like(prefix_attention, dtype=torch.bool)
                expected_mask[:, 0] = True
                expected_mask[:, -1] = True
                if not bool(torch.equal(prefix_attention.to(dtype=torch.bool), expected_mask)):
                    raise ValueError("mask nativa non coincide con BOS + 32 slot + BOA")
                try:
                    if self._shape_stable_lm is None:
                        raise RuntimeError("import cache nativo richiede ShapeStableGemmaListenerLM")
                    native.reset()
                    signature = (
                        id(lm),
                        id(self._shape_stable_lm),
                        id(self.gemma_model.config),
                        str(embeds.device),
                        str(embeds.dtype),
                        tuple(int(value) for value in prefix_ids[0].detach().cpu().tolist()),
                        tuple(bool(value) for value in prefix_attention[0].detach().cpu().tolist()),
                        AUDIO_START_AFTER_TEXT,
                    )
                    cached_prefix = (
                        self._canonical_prefix_cache_enabled
                        and self._canonical_prefix_cache_snapshot is not None
                        and self._canonical_prefix_cache_signature == signature
                    )
                    if cached_prefix:
                        prefix_cache = self._canonical_prefix_cache_snapshot
                        self._canonical_prefix_reuse_count += 1
                    else:
                        from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

                        prefix_position_ids = torch.arange(
                            AUDIO_START_AFTER_TEXT, device=self.device, dtype=torch.long
                        ).view(1, -1)
                        prefix_masks = create_masks_for_generate(
                            self.gemma_model.config,
                            embeds,
                            prefix_attention,
                            past_key_values=None,
                            position_ids=prefix_position_ids,
                        )
                        with torch.inference_mode():
                            canonical_prefix = self._shape_stable_lm.forward(
                                inputs_embeds=embeds,
                                per_layer_inputs=per_layer,
                                attention_mask=prefix_masks,
                                position_ids=prefix_position_ids,
                                past_key_values=None,
                                shared_kv_states=None,
                            )
                        self._canonical_prefix_compute_count += 1
                        prefix_cache = canonical_prefix.past_key_values
                        if self._canonical_prefix_cache_enabled:
                            self._canonical_prefix_cache_snapshot = (
                                self._clone_canonical_prefix_snapshot(prefix_cache)
                            )
                            self._canonical_prefix_cache_signature = signature
                            prefix_cache = self._canonical_prefix_cache_snapshot
                    native.import_cache_async(
                        prefix_cache,
                        current_position=AUDIO_START_AFTER_TEXT,
                    )
                except BaseException as exc:
                    state.gemma_lm_terminal_error = str(exc)
                    raise
                state.gemma_lm_cache = native
                state.gemma_lm_shared_kv_states = None
                state.gemma_lm_prefix_len = AUDIO_START_AFTER_TEXT
                state.gemma_lm_prefix_attention = None
                return native.output_tensor
            if q_len != AUDIO_FEATURES_PER_FRAME or state.gemma_lm_cache is None:
                raise RuntimeError("stato nativo Gemma non inizializzato o query non canonica")
            try:
                profile_kwargs = _native_q4_profile_step_kwargs(native, state)
                output = native.step_async(embeds, **profile_kwargs) if profile_kwargs else native.step_async(embeds)
            except BaseException as exc:
                state.gemma_lm_terminal_error = str(exc)
                raise
            state.gemma_lm_cache = native
            state.gemma_lm_shared_kv_states = None
            return output

        if self._shape_stable_lm is None:
            raise RuntimeError("runtime LM shape-stable non disponibile")
        from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

        position_ids = positions.view(1, -1)
        # The native Gemma mask factory remains the source of truth.  The
        # listener view consumes it without changing the Thinker modules.
        mask_mapping = create_masks_for_generate(
            self.gemma_model.config,
            embeds,
            attention,
            past_key_values=state.gemma_lm_cache,
            position_ids=position_ids,
        )
        with torch.inference_mode():
            output = self._shape_stable_lm.forward(
                inputs_embeds=embeds,
                per_layer_inputs=per_layer,
                attention_mask=mask_mapping,
                position_ids=position_ids,
                past_key_values=state.gemma_lm_cache,
                shared_kv_states=state.gemma_lm_shared_kv_states,
            )
        state.gemma_lm_cache = output.past_key_values
        state.gemma_lm_shared_kv_states = getattr(output.past_key_values, "shared_layers", None)
        if self._lm_has_shared_kv and state.gemma_lm_shared_kv_states is None:
            raise RuntimeError("Gemma listener non ha costruito lo shared-KV session-owned")
        return output.last_hidden_state

    def native_runtime_info(self) -> dict[str, int | str | bool]:
        """Return live native backend facts; fail closed before context creation."""
        native = self._native_runtime
        if native is None:
            raise RuntimeError("listener native context non ancora creato")
        return native.native_runtime_info()

    def _frame_pad_per_layer_inputs(self, language_model: Any) -> Any:
        """Return immutable PLE for the four PAD identities used by every tick."""
        if self._frame_pad_ple is None:
            import torch

            pad_ids = torch.full(
                (1, AUDIO_FEATURES_PER_FRAME), self.pad_token_id,
                dtype=torch.long, device=self.device,
            )
            with torch.inference_mode():
                pad_embeds = language_model.embed_tokens(pad_ids)
                self._frame_pad_ple = language_model.get_per_layer_inputs(pad_ids, pad_embeds).detach()
        return self._frame_pad_ple

    def build_semantic_prefix_cache(
        self,
        prefix_token_ids: tuple[int, ...] | list[int],
        *,
        replay_q2: tuple[Any, ...] = (),
        replay_frame_ids: tuple[int, ...] = (),
    ) -> dict[str, Any]:
        """Build actual-length prefix KV plus distinct prior Q2 transactions.

        The shared listener owner calls this while holding the fair model lock
        and with its listener CUDA stream current. The method only builds a
        cache; the caller imports it into the target conversation's private
        native slot.
        """
        import torch
        from transformers.cache_utils import DynamicCache
        from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

        if self._shape_stable_lm is None:
            raise RuntimeError("semantic prefix requires ShapeStableGemmaListenerLM")
        token_ids = tuple(int(value) for value in prefix_token_ids)
        prefix_len = len(token_ids)
        if not 1 <= prefix_len <= 291:
            raise ValueError(f"semantic prefix length must be in [1,291], got {prefix_len}")
        replay_frame_ids = tuple(int(value) for value in replay_frame_ids)
        replay_q2 = tuple(replay_q2)
        if len(replay_q2) != len(replay_frame_ids):
            raise ValueError("semantic replay requires one Q2 tensor per frame ID")
        if any(right != left + 1 for left, right in zip(replay_frame_ids, replay_frame_ids[1:])):
            raise ValueError("semantic replay frame IDs must be contiguous")

        lm = self.gemma_model.language_model
        device = torch.device(self.device)
        prefix_started_ns = time.perf_counter_ns()
        prefix_ids = torch.tensor(token_ids, dtype=torch.long, device=device).view(1, prefix_len)
        prefix_attention = torch.ones((1, prefix_len), dtype=torch.long, device=device)
        prefix_positions = torch.arange(prefix_len, dtype=torch.long, device=device).view(1, -1)
        prefix_embeds = lm.embed_tokens(prefix_ids)
        prefix_ple = lm.get_per_layer_inputs(prefix_ids, prefix_embeds)
        prefix_masks = create_masks_for_generate(
            lm.config,
            prefix_embeds,
            prefix_attention,
            past_key_values=None,
            position_ids=prefix_positions,
        )
        # Native prefix import consumes full logical K/V tensors, while the
        # default config-backed cache physically truncates sliding layers.
        prefix_cache = DynamicCache()
        with torch.inference_mode():
            output = self._shape_stable_lm.forward(
                inputs_embeds=prefix_embeds,
                per_layer_inputs=prefix_ple,
                attention_mask=prefix_masks,
                position_ids=prefix_positions,
                past_key_values=prefix_cache,
                shared_kv_states=None,
            )
        cache = output.past_key_values
        prefix_build_ms = max(0, time.perf_counter_ns() - prefix_started_ns) / 1_000_000.0
        position = prefix_len
        pad_ple = self._frame_pad_per_layer_inputs(lm)
        replay_started_ns = time.perf_counter_ns()
        for frame_id, q2 in zip(replay_frame_ids, replay_q2, strict=True):
            if tuple(getattr(q2, "shape", ())) != (AUDIO_FEATURES_PER_FRAME, self.hidden_size):
                raise ValueError(
                    f"semantic replay frame {frame_id} must have shape [2,{self.hidden_size}]"
                )
            if getattr(q2, "device", None) != device:
                raise ValueError("semantic replay Q2 must already be on the listener device")
            if q2.is_cuda:
                q2.record_stream(torch.cuda.current_stream(device=device))
            frame_audio = q2.reshape(1, AUDIO_FEATURES_PER_FRAME, self.hidden_size).to(
                dtype=lm.embed_tokens.weight.dtype
            )
            positions = torch.arange(
                position, position + AUDIO_FEATURES_PER_FRAME, dtype=torch.long, device=device
            ).view(1, -1)
            attention = torch.ones(
                (1, position + AUDIO_FEATURES_PER_FRAME), dtype=torch.long, device=device
            )
            masks = create_masks_for_generate(
                lm.config,
                frame_audio,
                attention,
                past_key_values=cache,
                position_ids=positions,
            )
            with torch.inference_mode():
                replay_output = self._shape_stable_lm.forward(
                    inputs_embeds=frame_audio,
                    per_layer_inputs=pad_ple,
                    attention_mask=masks,
                    position_ids=positions,
                    past_key_values=cache,
                    shared_kv_states=getattr(cache, "shared_layers", None),
                )
            cache = replay_output.past_key_values
            position += AUDIO_FEATURES_PER_FRAME
        replay_ms = max(0, time.perf_counter_ns() - replay_started_ns) / 1_000_000.0

        get_seq_length = getattr(cache, "get_seq_length", None)
        if callable(get_seq_length) and int(get_seq_length(0)) != position:
            raise RuntimeError("semantic prefix KV cursor differs from its actual token/Q2 length")
        return {
            "past_key_values": cache,
            "prefix_attention": prefix_attention,
            "prefix_len": prefix_len,
            "cache_position": position,
            "replay_frame_ids": replay_frame_ids,
            "prefix_build_ms": prefix_build_ms,
            "replay_ms": replay_ms,
        }

    def prime_semantic_prefix(
        self,
        state: "GemmaListenerState",
        prefix_token_ids: tuple[int, ...] | list[int],
        *,
        prefix_len: int | None = None,
        replay_audio_soft: Any | None = None,
        replay_frame_ids: tuple[int, ...] | None = None,
    ) -> Any:
        """Prime exact variable-length context and strictly prior Q2 history."""
        token_ids = tuple(int(value) for value in prefix_token_ids)
        actual_prefix_len = len(token_ids)
        if prefix_len is not None:
            if isinstance(prefix_len, bool) or not isinstance(prefix_len, int):
                raise ValueError("semantic prefix request length must be an integer")
            if prefix_len != actual_prefix_len:
                raise ValueError(
                    "semantic prefix request length does not match its token IDs: "
                    f"{prefix_len} != {actual_prefix_len}"
                )
        if not 1 <= actual_prefix_len <= 291:
            raise ValueError(f"semantic prefix length must be in [1,291], got {actual_prefix_len}")
        replay_ids = tuple(int(value) for value in (replay_frame_ids or ()))
        replay_rows: tuple[Any, ...] = ()
        if replay_audio_soft is not None:
            if replay_audio_soft.ndim != 3 or int(replay_audio_soft.shape[0]) != 1:
                raise ValueError("semantic replay tensor must be [1,2R,1536]")
            if int(replay_audio_soft.shape[1]) != AUDIO_FEATURES_PER_FRAME * len(replay_ids):
                raise ValueError("semantic replay tensor length does not match frame IDs")
            replay_rows = tuple(
                replay_audio_soft[
                    0,
                    index * AUDIO_FEATURES_PER_FRAME:(index + 1) * AUDIO_FEATURES_PER_FRAME,
                ]
                for index in range(len(replay_ids))
            )
        if self._use_native_lm:
            # Prefix prefill and import share the backend's fair GPU lock with
            # ordinary listener transactions. The native wrapper retains the
            # imported tensors until its ordered D2D copy has completed.
            self._lock.acquire_fair("listener")
            try:
                result = self.build_semantic_prefix_cache(
                    token_ids,
                    replay_q2=replay_rows,
                    replay_frame_ids=replay_ids,
                )
                native = state.gemma_lm_native
                if native is None:
                    from duplex_kernel.native_listener import NativeGemmaListenerLM

                    native = NativeGemmaListenerLM(
                        self.gemma_model.language_model,
                        max_seq_len=self.native_max_seq_len,
                        stream_q_len=AUDIO_FEATURES_PER_FRAME,
                        enable_cuda_graph=self._native_cuda_graph_requested,
                        attention_backend=self._native_attention_backend,
                        rmsnorm_backend=self._native_rmsnorm_backend,
                        attention_backend_source=self._native_attention_backend_source,
                    )
                    if self.runtime_profile == "production":
                        from duplex_kernel.native_listener import assert_production_native_runtime

                        assert_production_native_runtime(native.native_runtime_info())
                native.reset()
                native.import_cache_async(
                    result["past_key_values"],
                    current_position=int(result["cache_position"]),
                    prefix_len=int(result["prefix_len"]),
                    prefix_mask=result["prefix_attention"],
                )
                self._native_runtime = native
            except BaseException as exc:
                state.gemma_lm_terminal_error = str(exc)
                raise
            finally:
                self._lock.release()
        else:
            self._lock.acquire_fair("listener")
            try:
                result = self.build_semantic_prefix_cache(
                    token_ids,
                    replay_q2=replay_rows,
                    replay_frame_ids=replay_ids,
                )
            finally:
                self._lock.release()
        with state._lock:
            state.gemma_lm_cache = native if self._use_native_lm else result["past_key_values"]
            state.gemma_lm_native = native if self._use_native_lm else None
            state.gemma_lm_shared_kv_states = (
                None if self._use_native_lm
                else getattr(result["past_key_values"], "shared_layers", None)
            )
            state.gemma_lm_prefix_len = int(result["prefix_len"])
            state.gemma_lm_prefix_attention = result["prefix_attention"].detach()
            state.gemma_lm_audio_frames = len(replay_ids)
            state.gemma_lm_source_frame_ids = list(replay_ids)
            state.gemma_lm_context_start_frame = replay_ids[0] if replay_ids else state.next_frame_id
            state.gemma_lm_last_source_frame_id = replay_ids[-1] if replay_ids else None
            state.gemma_lm_text_context = ()
            state.gemma_lm_terminal_error = None
            state.gemma_lm_audio_soft_history.clear()
            state.gemma_lm_audio_soft_history.extend(
                (int(frame_id), rows.detach().clone())
                for frame_id, rows in zip(replay_ids, replay_rows, strict=True)
            )
        return result

    def prime_steady_state(self, state: GemmaListenerState, text_token_ids: tuple[int, ...] = ()) -> None:
        """Prime the listener state to steady-state so frame 0 executes in <=80ms without cold replay."""
        import torch
        text_ids = tuple(int(x) for x in (text_token_ids or self.text_token_ids))
        if self.runtime_profile == "production" and text_ids:
            raise ValueError("il profilo production non accetta un piano testuale nel listener")
        lm = self.gemma_model.language_model
        input_ids, attention_mask = self._build_multimodal_inputs(text_ids, 1024, device=self.device)
        stable_attention = attention_mask[:, :AUDIO_START_AFTER_TEXT].to(device=self.device, dtype=torch.long)
        stable_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
        stable_embeds = lm.embed_tokens(stable_ids)
        stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)
        state.gemma_lm_cache = None
        state.gemma_lm_shared_kv_states = None
        state.gemma_lm_audio_frames = 0
        state.gemma_lm_context_rollover_count = 0
        state.gemma_lm_context_truncated_frames = 0
        self._forward_incremental_lm(
            embeds=stable_embeds,
            per_layer=stable_ple,
            attention=stable_attention,
            positions=torch.arange(AUDIO_START_AFTER_TEXT, device=self.device, dtype=torch.long),
            state=state,
            input_ids=input_ids,
        )
        state.gemma_lm_text_context = text_ids
        state.gemma_lm_terminal_error = None
        state.gemma_lm_context_start_frame = 0
        state.next_frame_id = 0

    def _run_incremental_lm(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        audio_soft: Any,
        state: GemmaListenerState,
        text_ids: tuple[int, ...],
    ) -> tuple[Any, Any]:
        """Advance Gemma's final multimodal LM by one two-token frame.

        Gemma's native audio tower is causal in time (right context zero), but
        its output is a complete fixed context. The steady LM stage consumes
        only the newest two audio soft tokens. At native KV capacity, a rebase
        retains the active prefix and most recent 64 audio frames, then
        shifts RoPE positions; older direct full-attention entries are truncated.
        Retained K/V remains context-conditioned, but this is not exact
        full-history continuity.
        """
        import torch

        lm = self.gemma_model.language_model
        lm_dtype = lm.embed_tokens.weight.dtype
        frame_id = int(state.next_frame_id)
        prior_soft_history = tuple(state.gemma_lm_audio_soft_history)
        if any(int(source_id) >= frame_id for source_id, _ in prior_soft_history):
            raise RuntimeError("Q=2 audio replay history includes the current or a future frame")
        replay_items = (*prior_soft_history, (frame_id, audio_soft[0]))
        replay_items = replay_items[-min(state.context_frames, DEFAULT_CONTEXT_FRAMES):]
        if not replay_items or int(replay_items[-1][0]) != frame_id:
            raise RuntimeError("Q=2 audio replay history does not end at the current frame")
        context_start = int(replay_items[0][0])
        context_changed = (
            state.gemma_lm_cache is None
            or (self._use_native_lm and state.gemma_lm_native is None)
            or state.gemma_lm_text_context != text_ids
            or not _gemma_lm_cache_can_advance(state)
        )
        if self._use_native_lm and not context_changed:
            _maybe_rollover_native_context(
                state,
                state.gemma_lm_native,
                frame_capacity=native_frame_capacity(
                    self.native_max_seq_len,
                    prefix_len=state.gemma_lm_prefix_len,
                ),
                retain_frames=min(state.context_frames, DEFAULT_CONTEXT_FRAMES),
            )
        stable_attention = (
            (
                state.gemma_lm_prefix_attention.to(device=self.device, dtype=torch.long)
                if state.gemma_lm_prefix_attention is not None
                else attention_mask[:, :AUDIO_START_AFTER_TEXT].to(device=self.device, dtype=torch.long)
            )
            if not (self._use_native_lm and not context_changed)
            else None
        )
        frame_rows = AUDIO_FEATURES_PER_FRAME

        if context_changed:
            stable_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
            stable_embeds = lm.embed_tokens(stable_ids)
            stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)
            # Recreate the prefix, then replay every observed audio frame as a
            # distinct Q=2 call. Joining prefix+first-frame changes the BF16 GEMM shape
            # and is therefore not representation-equivalent.
            state.gemma_lm_cache = None
            state.gemma_lm_shared_kv_states = None
            state.gemma_lm_audio_frames = 0
            self._forward_incremental_lm(
                embeds=stable_embeds,
                per_layer=stable_ple,
                attention=stable_attention,
                positions=torch.arange(AUDIO_START_AFTER_TEXT, device=self.device, dtype=torch.long),
                state=state,
                input_ids=input_ids,
            )
            frame_ple = self._frame_pad_per_layer_inputs(lm)
            for cached_frame, (_source_frame_id, soft_rows) in enumerate(replay_items):
                frame_audio = soft_rows.unsqueeze(0).to(dtype=lm_dtype)
                if tuple(frame_audio.shape) != (1, frame_rows, self.hidden_size):
                    raise RuntimeError(
                        f"Q=2 audio replay row has unexpected shape {tuple(frame_audio.shape)}"
                    )
                start = cached_frame * frame_rows
                start_position = AUDIO_START_AFTER_TEXT + start
                positions = torch.arange(
                    start_position, start_position + frame_rows,
                    device=self.device, dtype=torch.long,
                )
                frame_attention = torch.ones(
                    (1, start_position + frame_rows), device=self.device, dtype=torch.long
                )
                frame_attention[:, :AUDIO_START_AFTER_TEXT] = stable_attention
                hidden = self._forward_incremental_lm(
                    embeds=frame_audio,
                    per_layer=frame_ple,
                    attention=frame_attention,
                    positions=positions,
                    state=state,
                )
                current_hidden = hidden[:, 0, :]
                current_queries = hidden
                state.gemma_lm_audio_frames = cached_frame + 1
            state.gemma_lm_context_start_frame = context_start
            state.gemma_lm_source_frame_ids = [int(source_id) for source_id, _ in replay_items]
            state.gemma_lm_last_source_frame_id = frame_id
            state.gemma_lm_skipped_silence_frames = 0
            state.gemma_lm_text_context = text_ids
            return current_hidden, current_queries

        frame_audio = audio_soft
        frame_audio = frame_audio.to(dtype=lm_dtype)
        if tuple(frame_audio.shape) != (1, frame_rows, self.hidden_size):
            raise RuntimeError(f"frame audio nativo inatteso: {tuple(frame_audio.shape)}")
        frame_ple = self._frame_pad_per_layer_inputs(lm)
        if self._use_native_lm:
            # The C executor owns absolute position and persistent mask state.
            positions = attention = None
        else:
            start_position = state.gemma_lm_prefix_len + state.gemma_lm_audio_frames * frame_rows
            positions = torch.arange(
                start_position, start_position + frame_rows,
                device=self.device, dtype=torch.long,
            )
            attention = torch.ones(
                (1, start_position + frame_rows), device=self.device, dtype=torch.long
            )
            attention[:, :state.gemma_lm_prefix_len] = stable_attention
        hidden = self._forward_incremental_lm(
            embeds=frame_audio,
            per_layer=frame_ple,
            attention=attention,
            positions=positions,
            state=state,
        )
        state.gemma_lm_audio_frames += 1
        state.gemma_lm_source_frame_ids.append(int(state.next_frame_id))
        state.gemma_lm_last_source_frame_id = int(state.next_frame_id)
        state.gemma_lm_skipped_silence_frames = 0
        state.gemma_lm_text_context = text_ids
        return hidden[:, 0, :], hidden

    def _encode_causal_user_q2(
        self,
        state: GemmaListenerState,
        *,
        text_ids: tuple[int, ...],
    ) -> ListenerBackendOutput:
        """Extract the configured causal user Mel window and commit one Q=2 tick."""
        import torch

        if not state.user_pcm_history:
            raise RuntimeError("Q=2 listener requires the current user PCM frame")
        frontend_started = time.perf_counter()
        last_user_pcm = state.user_pcm_history[-1]
        last_assistant_pcm = state.assistant_pcm_history[-1]
        required_mel_frames = max(
            self.mel_history_frames,
            int(state.user_mel_history.maxlen or 0),
        )
        if state.user_mel_history.maxlen != required_mel_frames:
            state.user_mel_history = deque(
                state.user_mel_history, maxlen=required_mel_frames
            )
        if state.assistant_mel_history.maxlen != required_mel_frames:
            state.assistant_mel_history = deque(
                state.assistant_mel_history, maxlen=required_mel_frames
            )
        previous_user_mels = tuple(state.user_mel_history)
        user_mel, state.user_pcm_tail = self._extract_single_frame(
            last_user_pcm, state.user_pcm_tail
        )
        assistant_mel, state.assistant_pcm_tail = self._extract_single_frame(
            last_assistant_pcm, state.assistant_pcm_tail
        )
        if user_mel.shape != (8, 128) or assistant_mel.shape != (8, 128):
            raise RuntimeError("incremental Gemma Mel extractor must return [8,128] rows")
        state.user_mel_history.append(user_mel)
        state.assistant_mel_history.append(assistant_mel)

        prior = (
            np.concatenate(previous_user_mels, axis=0)[-self.mel_context_rows:]
            if previous_user_mels and self.mel_context_rows
            else np.empty((0, 128), dtype=np.float32)
        )
        window = np.concatenate((prior, user_mel), axis=0)
        valid_rows = int(window.shape[0])
        features = torch.from_numpy(window).unsqueeze(0).to(
            device=self.device, dtype=self.dtype
        )
        feature_mask = torch.ones((1, valid_rows), dtype=torch.bool, device=self.device)
        state.last_frontend_wall_ms = (time.perf_counter() - frontend_started) * 1000.0

        tower_started = time.perf_counter()
        if self.device.type == "cuda":
            q_start = valid_rows // AUDIO_TOWER_MEL_STRIDE - AUDIO_FEATURES_PER_FRAME
            audio_soft, timing = self.run_audio_tower_q2(features, q_start)
            if tuple(audio_soft.shape) != (1, 2, self.hidden_size):
                raise RuntimeError("Q=2 audio tower must return exact [1,2,1536] tokens")
            expected_mode = (
                "cuda_graph_n1" if valid_rows == self.mel_window_rows else "eager_n1_degraded"
            )
            if timing.execution_mode != expected_mode:
                raise RuntimeError("Q=2 audio tower execution mode differs from its window regime")
        else:
            with torch.inference_mode():
                output = self.gemma_model.get_audio_features(
                    input_features=features,
                    input_features_mask=feature_mask,
                    return_dict=True,
                )
            pooled = getattr(output, "pooler_output", None)
            valid_mask = getattr(output, "attention_mask", None)
            if (
                not isinstance(pooled, torch.Tensor)
                or tuple(pooled.shape) != (
                    1, valid_rows // AUDIO_TOWER_MEL_STRIDE, self.hidden_size
                )
                or not isinstance(valid_mask, torch.Tensor)
                or tuple(valid_mask.shape) != (1, valid_rows // AUDIO_TOWER_MEL_STRIDE)
                or not bool(valid_mask.all())
            ):
                raise RuntimeError("CPU Q=2 audio tower must return valid rows for its Mel prefix")
            q_start = valid_rows // AUDIO_TOWER_MEL_STRIDE - AUDIO_FEATURES_PER_FRAME
            audio_soft = pooled[:, q_start:q_start + AUDIO_FEATURES_PER_FRAME, :].to(
                device=self.device, dtype=self.dtype
            )
        state.last_audio_tower_call_wall_ms = (time.perf_counter() - tower_started) * 1000.0
        state.audio_tower_rebuild_pending = False

        native_session_steady = bool(
            self._use_native_lm
            and state.gemma_lm_native is not None
            and state.gemma_lm_cache is not None
            and state.gemma_lm_text_context == text_ids
            and _gemma_lm_cache_can_advance(state)
        )
        input_ids, attention = self._build_multimodal_inputs(
            text_ids, self.mel_window_rows, device=self.device
        )
        if native_session_steady:
            input_ids = attention = None

        t_before_lock = time.perf_counter()
        self._lock.acquire_fair("listener")
        state.last_lock_wait_ms = (time.perf_counter() - t_before_lock) * 1000.0
        lock_acquired = time.perf_counter()
        try:
            hidden, audio_queries = self._run_incremental_lm(
                input_ids=input_ids,
                attention_mask=attention,
                audio_soft=audio_soft,
                state=state,
                text_ids=text_ids,
            )
            hidden = hidden.detach()
            if audio_queries is not None:
                audio_queries = audio_queries.detach()
        finally:
            state.last_lock_hold_ms = (time.perf_counter() - lock_acquired) * 1000.0
            self._lock.release()
        if hidden.shape[-1] != self.hidden_size:
            raise RuntimeError(f"hidden Gemma multimodale inatteso: {tuple(hidden.shape)}")
        state.gemma_lm_audio_soft_history.append(
            (int(state.next_frame_id), audio_soft[0].detach().clone())
        )
        return ListenerBackendOutput(
            gemma_hidden_t=hidden,
            audio_tower_hidden=audio_soft[0].detach(),
            gemma_audio_queries_t=audio_queries,
        )

    def encode(
        self,
        user_pcm: np.ndarray,
        assistant_reference: AssistantReferenceFrame | None,
        state: GemmaListenerState,
    ) -> Any:
        del user_pcm, assistant_reference
        canonical = self.multimodal_model is not None
        if canonical and state.context_frames > DEFAULT_CONTEXT_FRAMES:
            raise ValueError(
                f"contesto listener Gemma oltre il massimo attestato di {DEFAULT_CONTEXT_FRAMES} frame"
            )
        target_rows = state.context_frames * MEL_PER_LISTENER_FRAME if canonical else None
        import torch

        if canonical:
            text_ids = tuple(getattr(state, "text_token_ids", ())) or self.text_token_ids
            return self._encode_causal_user_q2(state, text_ids=text_ids)

        frontend_started = time.perf_counter()
        text_ids = tuple(getattr(state, "text_token_ids", ())) or self.text_token_ids
        native_session_steady = bool(
            self._use_native_lm
            and state.gemma_lm_native is not None
            and state.gemma_lm_cache is not None
            and state.gemma_lm_text_context == text_ids
            and _gemma_lm_cache_can_advance(state)
        )
        graph_audio_soft = None
        audio_output = None

        # 1. Incremental mel extraction (outside the language-model lock)
        is_steady_state = (
            canonical
            and self.device.type == "cuda"
            and hasattr(state, "user_mel_history")
            and len(state.user_mel_history) >= 64
            and self._cuda_graph is not None
            and not state.audio_tower_rebuild_pending
        )

        if is_steady_state:
            last_u_pcm = state.user_pcm_history[-1]
            last_a_pcm = state.assistant_pcm_history[-1]
            u_mel_8, state.user_pcm_tail = self._extract_single_frame(last_u_pcm, state.user_pcm_tail)
            a_mel_8, state.assistant_pcm_tail = self._extract_single_frame(last_a_pcm, state.assistant_pcm_tail)
            state.user_mel_history.append(u_mel_8)
            state.assistant_mel_history.append(a_mel_8)
            interleaved_8 = interleave_duplex_mel(u_mel_8, a_mel_8)
            state.last_frontend_wall_ms = (time.perf_counter() - frontend_started) * 1000.0
            if self._static_shift is None or self._static_new_slice is None or self._graph_stream is None:
                raise RuntimeError("CUDA graph feature workspace non inizializzato")

            def update_steady_workspace() -> None:
                self._static_shift.copy_(self._static_features[:, 16:, :])
                self._static_features[:, :-16, :].copy_(self._static_shift)
                new_slice = torch.from_numpy(interleaved_8).unsqueeze(0)
                self._static_new_slice.copy_(new_slice, non_blocking=True)
                self._static_features[:, -16:, :].copy_(self._static_new_slice)

            graph_audio_soft, state.last_audio_tower_call_wall_ms = (
                self._replay_shared_audio_tower(
                    update_steady_workspace,
                    native_session_steady=native_session_steady,
                )
            )
            interleaved_rows = 1024
        elif canonical and hasattr(state, "user_mel_history") and len(state.user_pcm_history) > 0:
            last_u_pcm = state.user_pcm_history[-1]
            last_a_pcm = state.assistant_pcm_history[-1]
            u_mel_8, state.user_pcm_tail = self._extract_single_frame(last_u_pcm, state.user_pcm_tail)
            a_mel_8, state.assistant_pcm_tail = self._extract_single_frame(last_a_pcm, state.assistant_pcm_tail)
            state.user_mel_history.append(u_mel_8)
            state.assistant_mel_history.append(a_mel_8)
            u_mel_all = np.concatenate(list(state.user_mel_history), axis=0)
            a_mel_all = np.concatenate(list(state.assistant_mel_history), axis=0)
            rows = target_rows or len(u_mel_all)
            user_mel = self._pad_rows(u_mel_all, rows)
            assistant_mel = self._pad_rows(a_mel_all, rows)
            interleaved = interleave_duplex_mel(user_mel, assistant_mel)
            features = torch.from_numpy(interleaved).unsqueeze(0).to(self.device, dtype=self.dtype)
            feature_mask = torch.ones((1, interleaved.shape[0]), dtype=torch.bool, device=self.device)
            state.last_frontend_wall_ms = (time.perf_counter() - frontend_started) * 1000.0
            if interleaved.shape[0] == 1024 and self.device.type == "cuda":
                def update_full_workspace() -> None:
                    self._static_features.copy_(features)

                graph_audio_soft, state.last_audio_tower_call_wall_ms = (
                    self._replay_shared_audio_tower(
                        update_full_workspace,
                        native_session_steady=native_session_steady,
                    )
                )
            else:
                audio_tower_started = time.perf_counter()
                with torch.inference_mode():
                    audio_output = self.gemma_model.get_audio_features(
                        input_features=features,
                        input_features_mask=feature_mask,
                        return_dict=True,
                    )
                state.last_audio_tower_call_wall_ms = (time.perf_counter() - audio_tower_started) * 1000.0
            interleaved_rows = interleaved.shape[0]
        else:
            user_wave = np.concatenate(tuple(state.user_pcm_history), axis=0)
            assistant_wave = np.concatenate(tuple(state.assistant_pcm_history), axis=0)
            user_mel, assistant_mel = self._extract([user_wave, assistant_wave], rows=target_rows)
            rows = target_rows or max(len(user_mel), len(assistant_mel))
            if rows <= 0:
                raise RuntimeError("Gemma audio tower non ha prodotto feature")
            user_mel = self._pad_rows(user_mel, rows)
            assistant_mel = self._pad_rows(assistant_mel, rows)
            interleaved = np.stack((user_mel, assistant_mel), axis=1).reshape(rows * 2, 128)
            features = torch.from_numpy(interleaved).unsqueeze(0).to(self.device, dtype=self.dtype)
            feature_mask = torch.ones((1, interleaved.shape[0]), dtype=torch.bool, device=self.device)
            state.last_frontend_wall_ms = (time.perf_counter() - frontend_started) * 1000.0
            audio_tower_started = time.perf_counter()
            with torch.inference_mode():
                output = self.gemma_model.get_audio_features(
                    input_features=features,
                    input_features_mask=feature_mask,
                    return_dict=True,
                )
            state.last_audio_tower_call_wall_ms = (time.perf_counter() - audio_tower_started) * 1000.0
            pooled = getattr(output, "pooler_output", None)
            valid_mask = getattr(output, "attention_mask", None)
            if pooled is None or valid_mask is None:
                raise RuntimeError("Gemma audio tower non ha restituito pooler_output/mask")
            valid = pooled[0][valid_mask[0].to(pooled.device)]
            if valid.numel() == 0:
                raise RuntimeError("Gemma audio tower ha restituito hidden vuoto")
            hidden = valid[-1:].detach()
            return ListenerBackendOutput(gemma_hidden_t=hidden, audio_tower_hidden=hidden)

        state.audio_tower_rebuild_pending = False
        if native_session_steady:
            # The legacy incremental path consumes only the newest Q=2 rows;
            # avoid rebuilding full GPU multimodal IDs/masks and boolean-index
            # temporaries once the persistent LM transaction is established.
            ids = attention = None
        else:
            ids, attention = self._build_multimodal_inputs(
                text_ids, interleaved_rows, device=self.device
            )
        if graph_audio_soft is not None:
            audio_soft = graph_audio_soft
        else:
            pooled = getattr(audio_output, "pooler_output", None)
            valid_mask = getattr(audio_output, "attention_mask", None)
            if pooled is None or valid_mask is None:
                raise RuntimeError("Gemma audio tower non ha restituito soft token validi")
            if native_session_steady:
                if pooled.ndim != 3 or pooled.shape[1] < AUDIO_FEATURES_PER_FRAME:
                    raise RuntimeError("Gemma audio tower native ha restituito shape inattesa")
                audio_soft = pooled[:, -AUDIO_FEATURES_PER_FRAME:, :].to(
                    device=self.device, dtype=self.dtype
                )
            else:
                valid = pooled[0][valid_mask[0].to(pooled.device)]
                if valid.numel() == 0 or valid.shape[-1] != self.hidden_size:
                    raise RuntimeError("Gemma audio tower ha restituito hidden vuoto/incompatibile")
                audio_soft = valid.unsqueeze(0).to(device=self.device, dtype=self.dtype)

        # 3. Multimodal LM forward (inside lock with dedicated GPU)
        t_before_lock = time.perf_counter()
        self._lock.acquire_fair("listener")
        state.last_lock_wait_ms = (time.perf_counter() - t_before_lock) * 1000.0
        lock_acquired = time.perf_counter()
        try:
            hidden, audio_queries = self._run_incremental_lm(
                input_ids=ids,
                attention_mask=attention,
                audio_soft=audio_soft,
                state=state,
                text_ids=text_ids,
            )
            hidden = hidden.detach()
            if audio_queries is not None:
                audio_queries = audio_queries.detach()
        finally:
            state.last_lock_hold_ms = (time.perf_counter() - lock_acquired) * 1000.0
            self._lock.release()
        if hidden.shape[-1] != self.hidden_size:
            raise RuntimeError(f"hidden Gemma multimodale inatteso: {tuple(hidden.shape)}")
        return ListenerBackendOutput(
            gemma_hidden_t=hidden,
            audio_tower_hidden=audio_soft[0].detach(),
            gemma_audio_queries_t=audio_queries,
        )


@dataclass(frozen=True)
class ListenerOutput:
    """Result of one committed listener tick.

    ``gemma_hidden_t`` is the only canonical V9 input. ``hidden_t`` remains a
    read-only compatibility alias for older control-plane tests.
    """

    gemma_hidden_t: Any
    audio_timestamp: int
    listener_state_version: int
    finite: bool
    frame_id: int
    reference_frame_id: int | None = None
    audio_tower_hidden: Any | None = None
    gemma_audio_queries_t: Any | None = None
    source_frame_id: int | None = None
    source_available_at_ns: int | None = None
    fresh: bool = True
    silence_skipped: bool = False
    native_output_health: Any | None = None
    native_control6_raw_scores: Any | None = None
    listener_generation_id: int | None = None
    listener_state_epoch_id: str | None = None
    listener_sequence_position: int | None = None
    listener_cache_position: int | None = None
    logical_q_len: int | None = None
    ephemeral_reason: str | None = None
    queue_audio_for_next_turn: bool = False
    kv_rollback_tokens: int | None = None
    kv_rollback_enqueue_ms: float | None = None
    listener_valid_history_tokens: int | None = None
    native_cache_current_position_after: int | None = None
    native_cache_valid_kv_len_after: int | None = None

    @property
    def hidden_t(self) -> Any:
        return self.gemma_hidden_t


class GemmaListener:
    """Frame-locked causal listener with a replaceable model backend."""

    def __init__(
        self,
        backend: ListenerBackend | None = None,
        *,
        thinker: Any | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
        reference_delay_frames: int = DEFAULT_REFERENCE_DELAY_FRAMES,
        text_token_ids: tuple[int, ...] | list[int] | None = None,
        use_native_lm: bool | None = None,
        runtime_profile: str = "compatibility",
        attention_backend: str | int | None = None,
        rmsnorm_backend: str | int | None = None,
        mel_window_rows: int | None = None,
        enable_silence_skip: bool = False,
    ) -> None:
        if mel_window_rows is not None:
            from duplex_kernel.listener_microbatch_frontend import validate_mel_window_rows

            validate_mel_window_rows(mel_window_rows)
        if runtime_profile not in {"compatibility", "production", "oracle"}:
            raise ValueError(
                "runtime_profile deve essere compatibility, production oppure oracle"
            )
        if runtime_profile == "production" and backend is not None:
            raise ValueError("il profilo production richiede GemmaAudioTowerBackend nativo")
        if runtime_profile == "production" and text_token_ids:
            raise ValueError("il profilo production non accetta text_token_ids nel listener")
        self.runtime_profile = runtime_profile
        if backend is None:
            if thinker is None:
                raise ValueError("GemmaListener richiede backend o thinker")
            backend = GemmaAudioTowerBackend(
                thinker,
                text_token_ids=text_token_ids,
                use_native_lm=use_native_lm,
                runtime_profile=runtime_profile,
                attention_backend=attention_backend,
                rmsnorm_backend=rmsnorm_backend,
                mel_window_rows=mel_window_rows,
            )
        elif mel_window_rows is not None and getattr(backend, "mel_window_rows", None) != mel_window_rows:
            raise ValueError("GemmaListener mel_window_rows must match its audio tower backend")
        if sample_rate <= 0 or frame_duration_ms <= 0:
            raise ValueError("sample_rate e frame_duration_ms devono essere positivi")
        frame_samples = sample_rate * frame_duration_ms
        if frame_samples % 1000:
            raise ValueError("frame_duration_ms deve produrre un numero intero di sample")
        if reference_delay_frames < 0:
            raise ValueError("reference_delay_frames non può essere negativo")
        self.backend = backend
        self.mel_window_rows = getattr(backend, "mel_window_rows", mel_window_rows)
        self.sample_rate = int(sample_rate)
        self.frame_duration_ms = int(frame_duration_ms)
        self.frame_samples = frame_samples // 1000
        self.reference_delay_frames = int(reference_delay_frames)
        self.text_token_ids = tuple(int(value) for value in (text_token_ids or ()))
        self.enable_silence_skip = bool(enable_silence_skip)

    @classmethod
    def from_thinker(cls, thinker: Any, **kwargs: Any) -> "GemmaListener":
        return cls(thinker=thinker, **kwargs)

    def set_text_context(self, state: GemmaListenerState, token_ids: list[int] | tuple[int, ...]) -> None:
        """Set diagnostic text context; production keeps the plan separate."""
        if not isinstance(state, GemmaListenerState):
            raise TypeError("state deve essere GemmaListenerState")
        values = tuple(int(value) for value in token_ids)
        if self.runtime_profile == "production" and values:
            raise ValueError(
                "il listener production è audio-causale: il piano testuale resta separato"
            )
        with state._lock:
            state.text_token_ids = values

    @staticmethod
    def _validate_user_pcm(pcm: np.ndarray, expected_samples: int) -> np.ndarray:
        array = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if array.size != expected_samples:
            raise ValueError(f"un listener frame deve contenere esattamente {expected_samples} sample")
        if not np.isfinite(array).all():
            raise ValueError("listener PCM deve contenere solo sample finite")
        return np.array(array, dtype=np.float32, copy=True)

    def _coerce_reference(
        self,
        reference: AssistantReferenceFrame | np.ndarray | None,
        *,
        frame_id: int,
    ) -> AssistantReferenceFrame | None:
        allowed = frame_id - self.reference_delay_frames
        if reference is None:
            return None
        if isinstance(reference, AssistantReferenceFrame):
            source_id = reference.frame_id
            if source_id > allowed:
                raise ValueError(
                    f"assistant reference futura: frame {source_id} visibile al listener {frame_id} "
                    f"(massimo causale {allowed})"
                )
            if source_id != allowed:
                raise ValueError(
                    f"assistant reference non allineata: frame {source_id}, atteso {allowed}"
                )
            if reference.sample_rate != self.sample_rate or reference.pcm.size != self.frame_samples:
                raise ValueError("assistant reference non normalizzata al frame listener")
            return reference
        array = np.asarray(reference, dtype=np.float32).reshape(-1)
        if array.size != self.frame_samples:
            raise ValueError(f"assistant reference deve contenere esattamente {self.frame_samples} sample")
        if not np.isfinite(array).all():
            raise ValueError("assistant reference deve contenere solo sample finite")
        if allowed < 0:
            raise ValueError("assistant reference fornita prima del primo frame causale")
        return AssistantReferenceFrame(
            frame_id=allowed,
            pcm=array,
            emitted_at_ns=(allowed + 1) * self.frame_duration_ms * 1_000_000,
            sample_rate=self.sample_rate,
        )

    @staticmethod
    def _hidden_health(hidden: Any) -> tuple[bool, bool]:
        try:
            numel = int(hidden.numel()) if hasattr(hidden, "numel") else int(np.asarray(hidden).size)
        except Exception:
            numel = 0
        if numel <= 0:
            return False, False
        try:
            if hasattr(hidden, "is_floating_point"):
                import torch
                finite = bool(torch.isfinite(hidden.sum()).item())
            else:
                finite = bool(np.isfinite(np.asarray(hidden)).all())
        except Exception:
            finite = False
        return finite, True

    def step(
        self,
        user_pcm_t: np.ndarray,
        assistant_reference_t: AssistantReferenceFrame | np.ndarray | None,
        state: GemmaListenerState,
        *,
        allow_silence_skip: bool = False,
    ) -> ListenerOutput:
        waiter_priority = getattr(self.backend, "_listener_waiter_priority", None)
        if waiter_priority is not None:
            with waiter_priority.registered():
                return self._step_registered(
                    user_pcm_t, assistant_reference_t, state,
                    allow_silence_skip=allow_silence_skip,
                )
        waiting = getattr(self.backend, "_waiting_for_lm", None)
        if waiting is not None:
            waiting.set()
        try:
            return self._step_registered(
                user_pcm_t, assistant_reference_t, state,
                allow_silence_skip=allow_silence_skip,
            )
        finally:
            if waiting is not None:
                waiting.clear()

    def _step_registered(
        self,
        user_pcm_t: np.ndarray,
        assistant_reference_t: AssistantReferenceFrame | np.ndarray | None,
        state: GemmaListenerState,
        *,
        allow_silence_skip: bool = False,
    ) -> ListenerOutput:
        """Consume exactly one 80 ms user frame and return its causal hidden."""
        if not isinstance(state, GemmaListenerState):
            raise TypeError("state deve essere GemmaListenerState")
        if hasattr(self.backend, "_tick_started_event") and self.backend._tick_started_event is not None:
            self.backend._tick_started_event.set()
        state.last_frontend_wall_ms = None
        state.last_audio_tower_call_wall_ms = None
        state.last_lock_wait_ms = None
        state.last_lock_hold_ms = None
        try:
            with state._lock:
                if state._closed:
                    raise RuntimeError("GemmaListenerState chiuso")
                if state.sample_rate != self.sample_rate or state.frame_duration_ms != self.frame_duration_ms:
                    raise ValueError("state e listener hanno sample-rate/frame-duration diversi")
                frame_id = state.next_frame_id
                if self.runtime_profile == "production" and (
                    self.text_token_ids
                    or getattr(state, "text_token_ids", ())
                    or getattr(state, "gemma_lm_text_context", ())
                ):
                    raise ValueError("il profilo production ha ricevuto conditioning testuale nel listener")
                user = self._validate_user_pcm(user_pcm_t, self.frame_samples)
                try:
                    reference = self._coerce_reference(assistant_reference_t, frame_id=frame_id)
                except ValueError as exc:
                    message = str(exc)
                    if "futura" in message or "causale" in message and "prima" not in message:
                        state.future_reference_leaks += 1
                    else:
                        state.wrong_reference_frames += 1
                    raise

                can_skip_exact_silence = bool(
                    self.enable_silence_skip
                    and allow_silence_skip
                    and not np.count_nonzero(user)
                    and callable(getattr(self.backend, "advance_silent_frame", None))
                    and state.last_hidden is not None
                    and state.last_hidden_source_frame_id is not None
                    and state.last_audio_queries is not None
                )
                if can_skip_exact_silence:
                    user_pcm_before = tuple(state.user_pcm_history)
                    assistant_pcm_before = tuple(state.assistant_pcm_history)
                    user_mel_before = tuple(state.user_mel_history)
                    assistant_mel_before = tuple(state.assistant_mel_history)
                    user_tail_before = None if state.user_pcm_tail is None else state.user_pcm_tail.copy()
                    assistant_tail_before = None if state.assistant_pcm_tail is None else state.assistant_pcm_tail.copy()
                    graph_rebuild_before = bool(state.audio_tower_rebuild_pending)
                    state.user_pcm_history.append(user)
                    state.assistant_pcm_history.append(
                        np.zeros(self.frame_samples, dtype=np.float32)
                        if reference is None else np.array(reference.pcm, dtype=np.float32, copy=True)
                    )
                    try:
                        self.backend.advance_silent_frame(state)
                    except BaseException:
                        state.user_pcm_history.clear()
                        state.user_pcm_history.extend(user_pcm_before)
                        state.assistant_pcm_history.clear()
                        state.assistant_pcm_history.extend(assistant_pcm_before)
                        state.user_mel_history.clear()
                        state.user_mel_history.extend(user_mel_before)
                        state.assistant_mel_history.clear()
                        state.assistant_mel_history.extend(assistant_mel_before)
                        state.user_pcm_tail = user_tail_before
                        state.assistant_pcm_tail = assistant_tail_before
                        state.audio_tower_rebuild_pending = graph_rebuild_before
                        logger.exception("silence-skip frontend fallito; elaboro il frame con il percorso completo")
                    else:
                        _commit_listener_input_cursors(
                            state,
                            user_pcm=user,
                            reference=reference,
                            frame_id=frame_id,
                            frame_samples=self.frame_samples,
                            frame_duration_ms=self.frame_duration_ms,
                        )
                        state.gemma_lm_skipped_silence_frames += 1
                        state.silence_skipped_frames += 1
                        return ListenerOutput(
                            gemma_hidden_t=state.last_hidden,
                            audio_timestamp=state.last_audio_timestamp,
                            listener_state_version=state.listener_state_version,
                            finite=True,
                            frame_id=frame_id,
                            reference_frame_id=state.last_reference_frame_id,
                            gemma_audio_queries_t=state.last_audio_queries,
                            source_frame_id=state.last_hidden_source_frame_id,
                            source_available_at_ns=state.last_hidden_at_ns,
                            fresh=False,
                            silence_skipped=True,
                        )

                # Snapshot all rolling histories because encode() advances them
                # before the LM transaction is known to succeed. This also
                # restores a frame evicted by a full deque on failure.
                user_pcm_before = tuple(state.user_pcm_history)
                assistant_pcm_before = tuple(state.assistant_pcm_history)
                user_mel_before = tuple(state.user_mel_history)
                assistant_mel_before = tuple(state.assistant_mel_history)
                user_tail_before = None if state.user_pcm_tail is None else state.user_pcm_tail.copy()
                assistant_tail_before = None if state.assistant_pcm_tail is None else state.assistant_pcm_tail.copy()
                graph_rebuild_before = bool(state.audio_tower_rebuild_pending)

                def restore_feature_state() -> None:
                    state.user_pcm_history.clear()
                    state.user_pcm_history.extend(user_pcm_before)
                    state.assistant_pcm_history.clear()
                    state.assistant_pcm_history.extend(assistant_pcm_before)
                    state.user_mel_history.clear()
                    state.user_mel_history.extend(user_mel_before)
                    state.assistant_mel_history.clear()
                    state.assistant_mel_history.extend(assistant_mel_before)
                    state.user_pcm_tail = user_tail_before
                    state.assistant_pcm_tail = assistant_tail_before
                    state.audio_tower_rebuild_pending = graph_rebuild_before

                # Commit the causal input context before calling the backend.  The
                # state is locked, so the backend sees a coherent aligned pair.
                state.user_pcm_history.append(user)
                state.assistant_pcm_history.append(
                    np.zeros(self.frame_samples, dtype=np.float32)
                    if reference is None else np.array(reference.pcm, dtype=np.float32, copy=True)
                )
                try:
                    encoded = self.backend.encode(user, reference, state)
                    if isinstance(encoded, ListenerBackendOutput):
                        hidden = encoded.gemma_hidden_t
                        audio_tower_hidden = encoded.audio_tower_hidden
                        audio_queries = encoded.gemma_audio_queries_t
                    else:
                        hidden = encoded
                        audio_tower_hidden = None
                        audio_queries = None
                except BaseException:
                    restore_feature_state()
                    # The functional LM may have appended part of the failing
                    # chunk before an exception.  Never reuse that partial cache:
                    # the next valid tick will deterministically replay history.
                    state.invalidate_gemma_lm_state()
                    state.dropped_listener_frames += 1
                    raise
                native_fast_path = bool(
                    getattr(self.backend, "_use_native_lm", False)
                    and state.gemma_lm_native is not None
                )
                if native_fast_path:
                    # Native step_async has a persistent output tensor. Avoid a
                    # scalar GPU reduction/readback in the frame hot path;
                    # native launch/lifecycle errors remain fail-closed.
                    finite = True
                    nonempty = int(hidden.numel()) > 0
                else:
                    finite, nonempty = self._hidden_health(hidden)
                if not nonempty:
                    state.empty_hidden_count += 1
                    restore_feature_state()
                    state.invalidate_gemma_lm_state()
                    state.dropped_listener_frames += 1
                    raise RuntimeError("listener hidden vuoto")
                if not finite:
                    state.nonfinite_hidden_count += 1
                    restore_feature_state()
                    state.invalidate_gemma_lm_state()
                    state.dropped_listener_frames += 1
                    raise RuntimeError("listener hidden non finito")

                if state.gemma_lm_audio_frames:
                    cached_frames = int(state.gemma_lm_audio_frames)
                    source_ids = state.gemma_lm_source_frame_ids
                    if len(source_ids) == cached_frames - 1 and (
                        state.gemma_lm_last_source_frame_id is None
                        or state.gemma_lm_last_source_frame_id < frame_id
                    ):
                        # Keep compatibility test backends that expose the
                        # cache count but do not yet publish source IDs.
                        source_ids.append(int(frame_id))
                        state.gemma_lm_last_source_frame_id = int(frame_id)
                    elif len(source_ids) != cached_frames:
                        raise RuntimeError("indice sorgente della cache KV non allineato ai frame incorporati")

                _commit_listener_input_cursors(
                    state,
                    user_pcm=user,
                    reference=reference,
                    frame_id=frame_id,
                    frame_samples=self.frame_samples,
                    frame_duration_ms=self.frame_duration_ms,
                )
                if hasattr(hidden, "detach"):
                    detached_hidden = hidden.detach()
                    if native_fast_path:
                        import torch
                        if (
                            state.native_last_hidden is None
                            or state.native_last_hidden.shape != detached_hidden.shape
                            or state.native_last_hidden.device != detached_hidden.device
                            or state.native_last_hidden.dtype != detached_hidden.dtype
                        ):
                            state.native_last_hidden = torch.empty_like(detached_hidden)
                        state.native_last_hidden.copy_(detached_hidden, non_blocking=True)
                        state.last_hidden = state.native_last_hidden
                    else:
                        state.last_hidden = detached_hidden.clone()
                else:
                    state.last_hidden = np.array(hidden, copy=True)
                state.last_hidden_source_frame_id = int(frame_id)
                state.last_hidden_at_ns = time.monotonic_ns()
                if audio_queries is None:
                    state.last_audio_queries = None
                elif callable(getattr(audio_queries, "detach", None)):
                    state.last_audio_queries = audio_queries.detach().clone()
                else:
                    state.last_audio_queries = np.array(audio_queries, copy=True)
                return ListenerOutput(
                    gemma_hidden_t=hidden,
                    audio_timestamp=state.last_audio_timestamp,
                    listener_state_version=state.listener_state_version,
                    finite=finite,
                    frame_id=frame_id,
                    reference_frame_id=state.last_reference_frame_id,
                    audio_tower_hidden=audio_tower_hidden,
                    gemma_audio_queries_t=audio_queries,
                    source_frame_id=int(frame_id),
                    source_available_at_ns=state.last_hidden_at_ns,
                    fresh=True,
                    silence_skipped=False,
                )
        finally:
            if hasattr(self.backend, "_frame_done_event") and self.backend._frame_done_event is not None:
                self.backend._frame_done_event.set()
            if hasattr(self.backend, "_frame_counter") and self.backend._frame_counter is not None:
                self.backend._frame_counter += 1

    def commit_microbatch_output(
        self,
        user_pcm_t: np.ndarray,
        assistant_reference_t: AssistantReferenceFrame | np.ndarray | None,
        state: GemmaListenerState,
        row_output: Any,
        *,
        source_available_at_ns: int,
        timing_sink: dict[str, int] | None = None,
    ) -> ListenerOutput:
        """Commit a completed native batch row through the ordinary state contract.

        The worker owns frontend/native cursors. This bridge installs their
        committed mel/tail snapshot and applies the same frame counters,
        lineage, and output history as ``_step_registered`` before the normal
        per-session policy/V9 callback continues.
        """
        if not isinstance(state, GemmaListenerState):
            raise TypeError("state deve essere GemmaListenerState")
        if isinstance(source_available_at_ns, bool) or not isinstance(source_available_at_ns, int):
            raise ValueError("source_available_at_ns must be an integer monotonic timestamp")
        if hasattr(self.backend, "_tick_started_event") and self.backend._tick_started_event is not None:
            self.backend._tick_started_event.set()
        state.last_frontend_wall_ms = None
        state.last_audio_tower_call_wall_ms = None
        state_lock_context = (
            _timed_lock(state._lock, timing_sink, name="state_lock")
            if timing_sink is not None else state._lock
        )
        with state_lock_context:
            if state._closed:
                raise RuntimeError("GemmaListenerState chiuso")
            if state.sample_rate != self.sample_rate or state.frame_duration_ms != self.frame_duration_ms:
                raise ValueError("state e listener hanno sample-rate/frame-duration diversi")
            frame_id = int(state.next_frame_id)
            if int(getattr(row_output, "frame_id", -1)) != frame_id:
                raise RuntimeError("microbatch row frame does not match the listener cursor")
            if str(getattr(row_output, "state_epoch_id", "")) != state.state_epoch_id:
                raise RuntimeError("microbatch row state epoch does not match the listener")
            expected_version = int(state.listener_state_version) + 1
            if int(getattr(row_output, "listener_state_version", -1)) != expected_version:
                raise RuntimeError("microbatch row state version does not follow the listener cursor")
            user = self._validate_user_pcm(user_pcm_t, self.frame_samples)
            reference = self._coerce_reference(assistant_reference_t, frame_id=frame_id)
            hidden = getattr(row_output, "gemma_hidden_t", None)
            queries = getattr(row_output, "gemma_audio_queries_t", None)
            tower = getattr(row_output, "audio_tower_hidden", None)
            hidden_size = int(getattr(self.backend, "hidden_size", 1536))
            if tuple(getattr(hidden, "shape", ())) != (1, hidden_size):
                raise RuntimeError("microbatch row hidden must have shape [1,1536]")
            if tuple(getattr(queries, "shape", ())) != (AUDIO_FEATURES_PER_FRAME, hidden_size):
                raise RuntimeError("microbatch row Q2 queries must have shape [2,1536]")
            if tuple(getattr(tower, "shape", ())) != (AUDIO_FEATURES_PER_FRAME, hidden_size):
                raise RuntimeError("microbatch row audio tower output must have shape [2,1536]")

            state.user_pcm_history.append(np.array(user, dtype=np.float32, copy=True))
            state.assistant_pcm_history.append(
                np.zeros(self.frame_samples, dtype=np.float32)
                if reference is None else np.array(reference.pcm, dtype=np.float32, copy=True)
            )
            snapshot = getattr(row_output, "state_snapshot", None)
            if snapshot is None:
                raise RuntimeError("microbatch row lacks its committed frontend-state snapshot")
            for name in ("user_mel_history", "assistant_mel_history"):
                values = getattr(snapshot, name, None)
                if values is None:
                    raise RuntimeError(f"microbatch state snapshot lacks {name}")
                history = getattr(state, name)
                history.clear()
                history.extend(np.array(value, dtype=np.float32, copy=True) for value in values)
            state.user_pcm_tail = (
                None if snapshot.user_pcm_tail is None
                else np.array(snapshot.user_pcm_tail, dtype=np.float32, copy=True)
            )
            state.assistant_pcm_tail = (
                None if snapshot.assistant_pcm_tail is None
                else np.array(snapshot.assistant_pcm_tail, dtype=np.float32, copy=True)
            )

            _commit_listener_input_cursors(
                state,
                user_pcm=user,
                reference=reference,
                frame_id=frame_id,
                frame_samples=self.frame_samples,
                frame_duration_ms=self.frame_duration_ms,
            )
            if state.listener_state_version != expected_version:
                raise RuntimeError("listener cursor commit changed an unexpected state version")
            state.native_last_hidden = hidden.detach()
            state.last_hidden = state.native_last_hidden
            state.last_audio_queries = queries.detach()
            state.last_lock_wait_ms = getattr(
                row_output, "listener_model_lock_wait_ms", None
            )
            state.last_lock_hold_ms = getattr(
                row_output, "listener_model_lock_hold_ms", None
            )
            state.last_hidden_source_frame_id = frame_id
            state.last_hidden_at_ns = int(source_available_at_ns)
            native_state = getattr(row_output, "native_state", None)
            current_position = getattr(native_state, "current_position", None)
            prefix_len = getattr(native_state, "prefix_len", None)
            if isinstance(current_position, int) and isinstance(prefix_len, int):
                state.gemma_lm_audio_frames = max(
                    0, (int(current_position) - int(prefix_len)) // AUDIO_FEATURES_PER_FRAME
                )
            else:
                state.gemma_lm_audio_frames += 1
            state.gemma_lm_source_frame_ids.append(frame_id)
            retained = max(1, int(state.gemma_lm_audio_frames))
            if len(state.gemma_lm_source_frame_ids) > retained:
                del state.gemma_lm_source_frame_ids[:-retained]
            if state.gemma_lm_source_frame_ids:
                state.gemma_lm_context_start_frame = state.gemma_lm_source_frame_ids[0]
                state.gemma_lm_last_source_frame_id = state.gemma_lm_source_frame_ids[-1]
            state.gemma_lm_text_context = ()
            state.gemma_lm_cache = None
            state.gemma_lm_shared_kv_states = None
            state.gemma_lm_native = None
            state.gemma_lm_terminal_error = None
            row_prefix_revision = getattr(row_output, "semantic_context_revision", None)
            if row_prefix_revision is not None:
                state.semantic_context_revision = int(row_prefix_revision)
                state.semantic_replay_fallback_reason = getattr(
                    row_output, "semantic_replay_fallback_reason", None
                )
                prefix_length = getattr(row_output, "semantic_prefix_len", None)
                if isinstance(prefix_length, int) and not isinstance(prefix_length, bool):
                    state.gemma_lm_prefix_len = prefix_length

            if hasattr(self.backend, "_frame_done_event") and self.backend._frame_done_event is not None:
                self.backend._frame_done_event.set()
            if hasattr(self.backend, "_frame_counter") and self.backend._frame_counter is not None:
                self.backend._frame_counter += 1

            return ListenerOutput(
                gemma_hidden_t=hidden,
                audio_timestamp=state.last_audio_timestamp,
                listener_state_version=state.listener_state_version,
                finite=True,
                frame_id=frame_id,
                reference_frame_id=state.last_reference_frame_id,
                audio_tower_hidden=tower,
                gemma_audio_queries_t=queries,
                source_frame_id=frame_id,
                source_available_at_ns=int(source_available_at_ns),
                fresh=True,
                silence_skipped=False,
                native_output_health=getattr(row_output, "native_output_health", None),
                native_control6_raw_scores=getattr(
                    row_output, "native_control6_raw_scores", None
                ),
                listener_generation_id=int(getattr(row_output, "generation_id")),
                listener_state_epoch_id=str(getattr(row_output, "state_epoch_id")),
                listener_sequence_position=int(getattr(row_output, "sequence_position")),
                listener_cache_position=int(getattr(row_output, "cache_position")),
                logical_q_len=int(getattr(row_output, "logical_q_len")),
                ephemeral_reason=getattr(row_output, "ephemeral_reason", None),
                queue_audio_for_next_turn=bool(
                    getattr(row_output, "queue_audio_for_next_turn", False)
                ),
                kv_rollback_tokens=int(getattr(row_output, "kv_rollback_tokens", 0)),
                kv_rollback_enqueue_ms=float(
                    getattr(row_output, "kv_rollback_enqueue_ms", 0.0)
                ),
                listener_valid_history_tokens=int(
                    getattr(row_output, "valid_history_tokens")
                ),
                native_cache_current_position_after=(
                    int(getattr(getattr(row_output, "native_state", None), "current_position"))
                    if getattr(getattr(row_output, "native_state", None), "current_position", None)
                    is not None
                    else None
                ),
                native_cache_valid_kv_len_after=(
                    int(getattr(getattr(row_output, "native_state", None), "valid_kv_len"))
                    if getattr(getattr(row_output, "native_state", None), "valid_kv_len", None)
                    is not None
                    else None
                ),
            )

    def prime_steady_state(self, state: GemmaListenerState) -> None:
        if hasattr(self.backend, "prime_steady_state"):
            self.backend.prime_steady_state(state)

    def stats(self, state: GemmaListenerState) -> dict[str, int | None]:
        return state.stats()


__all__ = [
    "AssistantReferenceBuffer",
    "AssistantReferenceFrame",
    "GemmaAudioTowerBackend",
    "ListenerBackendOutput",
    "GemmaListener",
    "GemmaListenerState",
    "ListenerBackend",
    "ListenerOutput",
    "capture_host_timeline_nonblocking",
]
