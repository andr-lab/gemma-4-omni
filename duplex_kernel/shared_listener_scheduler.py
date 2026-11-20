"""Fixed-grid and input-ready ingress gather for one shared listener owner.

The dispatcher snapshots at most one complete frame per registered session
per batch, using the absolute 80 ms grid as a nominal deadline and waking early
when ingress has a complete frame. It sends each row set through one batch call
and only then fans results back to the per-session scheduler. The dispatcher
never creates silence rows or falls back to per-session inference.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import faulthandler
import logging
import math
import os
import queue
import sys
import threading
import time
from typing import Any

from duplex_kernel.scheduler import AudioFrame, DuplexTick


DEFAULT_SHARED_LISTENER_PERIOD_NS = 80_000_000
_LOGGER = logging.getLogger(__name__)
_IRRECOVERABLE_GPU_ERROR_MARKERS = (
    "illegal memory access",
    "device-side assert triggered",
    "unspecified launch failure",
    "misaligned address",
    "device has been lost",
    "context is destroyed",
)


def _is_irrecoverable_gpu_error(error: BaseException) -> bool:
    """Recognize explicit process-wide CUDA failures; ordinary row errors recover."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "process_fatal_gpu", False) is True:
            return True
        message = str(current).lower()
        if any(marker in message for marker in _IRRECOVERABLE_GPU_ERROR_MARKERS):
            return True
        next_error = current.__cause__ or current.__context__
        current = next_error if isinstance(next_error, BaseException) else None
    return False


@dataclass(frozen=True, slots=True)
class SharedListenerInput:
    """One frame removed from a session ingress queue for a common tick."""

    session_id: str
    frame: AudioFrame
    tick: DuplexTick
    listener_generation_id: int
    state_epoch_id: str | None


@dataclass(frozen=True, slots=True)
class SharedListenerTickResult:
    """Timing and row identity for one common listener dispatch."""

    tick_index: int
    deadline_ns: int
    observed_ns: int
    batch_started_ns: int | None
    batch_finished_ns: int | None
    rows: tuple[SharedListenerInput, ...]
    error: str | None = None
    dispatch_id: int = 0

    @property
    def batch_size(self) -> int:
        return len(self.rows)

    @property
    def batch_queue_wait_ms(self) -> float | None:
        if not self.rows or self.batch_started_ns is None:
            return None
        oldest_enqueue_ns = min(int(row.frame.enqueue_ns) for row in self.rows)
        return round(max(0, self.batch_started_ns - oldest_enqueue_ns) / 1_000_000.0, 3)

    @property
    def batch_service_ms(self) -> float | None:
        """Batch-callback-to-result wall time, including worker queue and validation."""
        if self.batch_started_ns is None or self.batch_finished_ns is None:
            return None
        return round(max(0, self.batch_finished_ns - self.batch_started_ns) / 1_000_000.0, 3)

    @property
    def batch_compute_ms(self) -> float | None:
        """Legacy alias for batch service wall time; not GPU compute duration."""
        return self.batch_service_ms


BatchListener = Callable[
    [tuple[SharedListenerInput, ...]], Mapping[str, Any] | Future[Mapping[str, Any]]
]
ResultCallback = Callable[[SharedListenerInput, Any, SharedListenerTickResult], None]
ErrorCallback = Callable[[SharedListenerInput, BaseException, SharedListenerTickResult], None]


class _ListenerComputePermit:
    """One admitted compute quantum that publishes its stream completion."""

    def __init__(self, dispatcher: "SharedListenerTickDispatcher", permit_id: int) -> None:
        self._dispatcher = dispatcher
        self._permit_id = permit_id

    def record_cuda_completion(self, event: Any, stream: Any) -> None:
        self._dispatcher._record_compute_completion(self._permit_id, event, stream)

    def fail(self, error: BaseException) -> None:
        self._dispatcher._fail_compute_permit(self._permit_id, error)


class SharedListenerTickDispatcher:
    """Gather ready sessions on one shared grid with input-ready wakeups.

    batch_listener must return one output keyed by every input's session ID. It
    may return a Future from the GPU-owner worker; only this dispatcher thread
    waits for that Future. Results are fanned out in row order after the whole
    batch completes, preserving the existing per-session policy path.
    """

    def __init__(
        self,
        batch_listener: BatchListener,
        *,
        on_result: ResultCallback | None = None,
        on_error: ErrorCallback | None = None,
        batch_context: Callable[[tuple[SharedListenerInput, ...]], Any] | None = None,
        period_ns: int = DEFAULT_SHARED_LISTENER_PERIOD_NS,
        now_ns: Callable[[], int] = time.monotonic_ns,
        wait_until: Callable[[int], None] | None = None,
        stack_watchdog_ms: int | None = None,
    ) -> None:
        if not callable(batch_listener):
            raise TypeError("batch_listener must be callable")
        if on_result is not None and not callable(on_result):
            raise TypeError("on_result must be callable or None")
        if on_error is not None and not callable(on_error):
            raise TypeError("on_error must be callable or None")
        if batch_context is not None and not callable(batch_context):
            raise TypeError("batch_context must be callable or None")
        if isinstance(period_ns, bool) or not isinstance(period_ns, int) or period_ns <= 0:
            raise ValueError("period_ns must be a positive integer")
        if not callable(now_ns):
            raise TypeError("now_ns must be callable")
        if wait_until is not None and not callable(wait_until):
            raise TypeError("wait_until must be callable or None")
        if stack_watchdog_ms is None:
            try:
                stack_watchdog_ms = int(
                    os.environ.get("DUPLEX_LISTENER_STACK_WATCHDOG_MS", "0")
                )
            except ValueError as exc:
                raise ValueError(
                    "DUPLEX_LISTENER_STACK_WATCHDOG_MS must be a non-negative integer"
                ) from exc
        if (
            isinstance(stack_watchdog_ms, bool)
            or not isinstance(stack_watchdog_ms, int)
            or stack_watchdog_ms < 0
        ):
            raise ValueError("stack_watchdog_ms must be a non-negative integer")

        self._batch_listener = batch_listener
        self._on_result = on_result
        self._on_error = on_error
        self._batch_context = batch_context
        self._period_ns = period_ns
        self._now_ns = now_ns
        self._external_wait_until = wait_until
        self._stack_watchdog_ms = stack_watchdog_ms
        self._stack_watchdog_fired = False
        self._stack_watchdog_state_lock = threading.Lock()
        self._registrations: dict[str, tuple[Any, int, str | None]] = {}
        self._deferred: dict[str, AudioFrame] = {}
        self._paused_sessions: set[str] = set()
        self._ingress_publishers: dict[str, int] = {}
        self._lock = threading.RLock()
        self._dispatch_idle = threading.Condition(self._lock)
        self._inflight_sessions: set[str] = set()
        self._compute_permits: set[int] = set()
        self._published_compute_permits: set[int] = set()
        self._next_compute_permit_id = 0
        self._pending_compute_events: dict[tuple[int, int], Any] = {}
        self._active_compute_dependencies: tuple[tuple[tuple[int, int], Any], ...] = ()
        self._lifecycle_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._fatal_error: BaseException | None = None
        self._anchor_ns: int | None = None
        self._tick_index = 0
        self._dispatch_index = 0

    @property
    def fatal_error(self) -> BaseException | None:
        return self._fatal_error

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def active_session_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._registrations)

    @property
    def stack_watchdog_ms(self) -> int:
        return self._stack_watchdog_ms

    def has_pending_work(self, session_id: str) -> bool:
        """Report dispatcher-owned/deferred work for one conversation.

        Ingress queue depth alone is insufficient for an initial-prefix CAS:
        the dispatcher may already have removed a frame into its deferred or
        in-flight set.
        """
        with self._lock:
            return session_id in self._deferred or session_id in self._inflight_sessions

    def wait_for_listener_admission(
        self,
        session_id: str,
        ingress: Any,
        generation_is_current: Callable[[], bool],
        *,
        queue_depth_limit: int = 0,
        cancellation_check_interval_s: float = 0.05,
    ) -> bool:
        """Wait for dispatcher-owned work to clear before a Talker quantum.

        The service-wide ingress depth and dispatcher ownership are inspected
        while holding the same condition used by dequeue and in-flight
        publication. A frame therefore cannot disappear from both views during
        selection. Any ready ingress frame or selected/in-flight batch blocks
        admission; Talker proceeds when the shared listener has no queued work.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        if not callable(getattr(ingress, "qsize", None)) and not isinstance(
            getattr(ingress, "qsize", None), int
        ):
            raise TypeError("ingress must expose qsize as a property or method")
        if not callable(generation_is_current):
            raise TypeError("generation_is_current must be callable")
        if (
            isinstance(queue_depth_limit, bool)
            or not isinstance(queue_depth_limit, int)
            or queue_depth_limit < 0
        ):
            raise ValueError("queue_depth_limit must be a non-negative integer")
        if (
            isinstance(cancellation_check_interval_s, bool)
            or not isinstance(cancellation_check_interval_s, (int, float))
            or not math.isfinite(float(cancellation_check_interval_s))
            or cancellation_check_interval_s <= 0
        ):
            raise ValueError("cancellation_check_interval_s must be positive and finite")

        while True:
            if not generation_is_current() or self._stop_event.is_set():
                return False
            with self._dispatch_idle:
                if self._fatal_error is not None:
                    return False
                if session_id not in self._registrations:
                    return False
                queue_depth = 0
                for registered_id, (registered_ingress, _, _) in self._registrations.items():
                    if registered_id in self._paused_sessions:
                        continue
                    depth = getattr(registered_ingress, "qsize", 0)
                    depth = depth() if callable(depth) else depth
                    queue_depth += max(0, int(depth))
                pending = bool(self._inflight_sessions) or any(
                    deferred_id not in self._paused_sessions
                    for deferred_id in self._deferred
                ) or any(
                    publisher_id in self._registrations
                    and publisher_id not in self._paused_sessions
                    and count > 0
                    for publisher_id, count in self._ingress_publishers.items()
                )
                if not pending and queue_depth <= queue_depth_limit:
                    return True
                self._dispatch_idle.wait(timeout=float(cancellation_check_interval_s))

    @contextmanager
    def compute_quantum(
        self,
        generation_is_current: Callable[[], bool],
        *,
        cancellation_check_interval_s: float = 0.05,
    ):
        """Atomically admit one GPU quantum only while listener demand is absent.

        The returned permit remains owned through the host launch and records a
        completion event before release. A listener arriving after acquisition
        may follow that one quantum, then blocks every subsequent acquisition.
        """
        if not callable(generation_is_current):
            raise TypeError("generation_is_current must be callable")
        if (
            isinstance(cancellation_check_interval_s, bool)
            or not isinstance(cancellation_check_interval_s, (int, float))
            or not math.isfinite(float(cancellation_check_interval_s))
            or cancellation_check_interval_s <= 0
        ):
            raise ValueError("cancellation_check_interval_s must be positive and finite")
        permit_id = self._acquire_compute_permit(
            generation_is_current,
            cancellation_check_interval_s=float(cancellation_check_interval_s),
        )
        if permit_id is None:
            yield False
            return
        if not generation_is_current():
            self._release_compute_permit(permit_id)
            yield False
            return
        permit = _ListenerComputePermit(self, permit_id)
        try:
            yield permit
        finally:
            self._release_compute_permit(permit_id)

    @contextmanager
    def ingress_publication(self, session_id: str):
        """Publish a short ingress claim before a producer mutates its queue.

        The producer holds this claim only across the queue append. Since the
        claim and compute permit share ``_dispatch_idle``, an enqueue cannot
        slip between the demand check and permit ownership.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        claimed = False
        with self._dispatch_idle:
            if session_id in self._registrations:
                self._ingress_publishers[session_id] = (
                    self._ingress_publishers.get(session_id, 0) + 1
                )
                claimed = True
                self._dispatch_idle.notify_all()
        try:
            yield
        finally:
            if claimed:
                with self._dispatch_idle:
                    remaining = self._ingress_publishers.get(session_id, 0) - 1
                    if remaining > 0:
                        self._ingress_publishers[session_id] = remaining
                    else:
                        self._ingress_publishers.pop(session_id, None)
                    self._dispatch_idle.notify_all()

    def _listener_demand_unlocked(self) -> bool:
        if self._inflight_sessions or any(
            deferred_id not in self._paused_sessions for deferred_id in self._deferred
        ) or any(
            publisher_id in self._registrations
            and publisher_id not in self._paused_sessions
            and count > 0
            for publisher_id, count in self._ingress_publishers.items()
        ):
            return True
        for session_id, (ingress, _, _) in self._registrations.items():
            if session_id in self._paused_sessions:
                continue
            depth = getattr(ingress, "qsize", 0)
            depth = depth() if callable(depth) else depth
            if int(depth) > 0:
                return True
        return False

    def _acquire_compute_permit(
        self,
        generation_is_current: Callable[[], bool],
        *,
        cancellation_check_interval_s: float,
    ) -> int | None:
        while True:
            # Generation checks can acquire session locks. Keep them outside
            # the coordinator condition to avoid reversing lock order with
            # teardown and ingress callbacks.
            if self._stop_event.is_set() or not generation_is_current():
                return None
            with self._dispatch_idle:
                if self._stop_event.is_set() or self._fatal_error is not None:
                    return None
                if not self._compute_permits and not self._listener_demand_unlocked():
                    permit_id = self._next_compute_permit_id
                    self._next_compute_permit_id += 1
                    self._compute_permits.add(permit_id)
                    return permit_id
                self._dispatch_idle.wait(timeout=cancellation_check_interval_s)

    def _record_compute_completion(self, permit_id: int, event: Any, stream: Any) -> None:
        if event is None:
            raise ValueError("CUDA completion event must not be None")
        stream_handle = getattr(stream, "cuda_stream", None)
        if isinstance(stream_handle, bool) or not isinstance(stream_handle, int) or stream_handle < 0:
            raise ValueError("compute completion stream must have a valid CUDA handle")
        device = getattr(stream, "device", None)
        device_index = getattr(device, "index", None)
        if device_index is None:
            device_index = 0
        if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
            raise ValueError("compute completion stream must name a CUDA device")
        with self._dispatch_idle:
            if permit_id not in self._compute_permits:
                raise RuntimeError("compute completion was published by an inactive permit")
            if permit_id in self._published_compute_permits:
                raise RuntimeError("compute completion was already published for this permit")
            self._published_compute_permits.add(permit_id)
            self._pending_compute_events[(device_index, stream_handle)] = event
            self._dispatch_idle.notify_all()

    def _fail_compute_permit(self, permit_id: int, error: BaseException) -> None:
        with self._dispatch_idle:
            if permit_id not in self._compute_permits:
                raise RuntimeError("compute failure was reported by an inactive permit")
            self._fatal_error = error
            self._dispatch_idle.notify_all()

    def _release_compute_permit(self, permit_id: int) -> None:
        with self._dispatch_idle:
            if permit_id not in self._compute_permits:
                raise RuntimeError("compute permit was already released or is stale")
            self._compute_permits.remove(permit_id)
            self._published_compute_permits.discard(permit_id)
            self._dispatch_idle.notify_all()

    def wait_for_compute_quanta(self) -> None:
        """Wait without model/state locks until every admitted launch published its event."""
        with self._dispatch_idle:
            self._dispatch_idle.wait_for(lambda: not self._compute_permits)

    def active_compute_dependencies(self) -> tuple[Any, ...]:
        with self._dispatch_idle:
            return tuple(event for _, event in self._active_compute_dependencies)

    def _snapshot_compute_dependencies(self) -> None:
        with self._dispatch_idle:
            self._active_compute_dependencies = tuple(self._pending_compute_events.items())

    def _finish_compute_dependencies(self, *, completed: bool) -> None:
        with self._dispatch_idle:
            if completed:
                for key, event in self._active_compute_dependencies:
                    if self._pending_compute_events.get(key) is event:
                        self._pending_compute_events.pop(key, None)
            self._active_compute_dependencies = ()
            self._dispatch_idle.notify_all()

    def notify_input_ready(self, session_id: str) -> None:
        """Wake the dispatcher after a session enqueues a complete PCM frame."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        # The ingress callback runs without the ingress lock. Wake both the
        # scheduler and admission waiters without coupling either to polling.
        with self._dispatch_idle:
            self._dispatch_idle.notify_all()
        self._wake_event.set()

    def pause_session(
        self,
        session_id: str,
        *,
        wait_for_inflight: bool = True,
        timeout: float = 5.0,
    ) -> bool:
        """Pause dispatch without consuming queued or deferred audio frames.

        The pause is installed before waiting, so a tick not yet snapshotted
        cannot race into the batch. An already snapshotted row is allowed to
        finish and is included in the bounded wait.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        if not isinstance(wait_for_inflight, bool):
            raise TypeError("wait_for_inflight must be a bool")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or timeout < 0:
            raise ValueError("timeout must be a finite non-negative number")
        with self._dispatch_idle:
            if session_id not in self._registrations:
                return False
            self._paused_sessions.add(session_id)
            if wait_for_inflight and not self._dispatch_idle.wait_for(
                lambda: session_id not in self._inflight_sessions,
                timeout=float(timeout),
            ):
                raise TimeoutError(f"shared listener row is still in flight for {session_id}")
            return True

    def resume_session(self, session_id: str) -> bool:
        """Resume a paused registration; ingress/deferred FIFO stays intact."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        with self._dispatch_idle:
            if session_id not in self._registrations:
                return False
            was_paused = session_id in self._paused_sessions
            self._paused_sessions.discard(session_id)
            self._dispatch_idle.notify_all()
        if was_paused:
            self._wake_event.set()
        return was_paused

    def register_session(
        self,
        session_id: str,
        ingress: Any,
        *,
        listener_generation_id: int = 0,
        state_epoch_id: str | None = None,
    ) -> None:
        """Register one conversation; storage grows with active sessions."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        if not callable(getattr(ingress, "get_nowait", None)):
            raise TypeError("ingress must expose get_nowait()")
        if (
            isinstance(listener_generation_id, bool)
            or not isinstance(listener_generation_id, int)
            or listener_generation_id < 0
        ):
            raise ValueError("listener_generation_id must be a non-negative integer")
        if state_epoch_id is not None and (
            not isinstance(state_epoch_id, str) or not state_epoch_id.strip()
        ):
            raise ValueError("state_epoch_id must be non-empty or None")
        with self._lock:
            if session_id in self._registrations:
                raise ValueError(f"session already registered: {session_id}")
            self._registrations[session_id] = (
                ingress,
                listener_generation_id,
                state_epoch_id,
            )

    def unregister_session(
        self,
        session_id: str,
        *,
        wait_for_inflight: bool = False,
        timeout: float = 5.0,
    ) -> bool:
        """Remove a session and optionally wait for its snapshotted row to finish."""
        if not isinstance(wait_for_inflight, bool):
            raise TypeError("wait_for_inflight must be a bool")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or timeout < 0:
            raise ValueError("timeout must be a finite non-negative number")
        with self._dispatch_idle:
            removed = self._registrations.pop(session_id, None) is not None
            self._deferred.pop(session_id, None)
            self._paused_sessions.discard(session_id)
            self._dispatch_idle.notify_all()
            if wait_for_inflight and not self._dispatch_idle.wait_for(
                lambda: (
                    session_id not in self._inflight_sessions
                    and self._ingress_publishers.get(session_id, 0) == 0
                ),
                timeout=float(timeout),
            ):
                raise TimeoutError(f"shared listener row is still in flight for {session_id}")
            return removed

    def start(self, *, name: str = "shared-listener-tick-dispatcher") -> None:
        """Start the process-shared absolute 80 ms clock loop."""
        with self._lifecycle_lock:
            if self._thread is not None:
                raise RuntimeError("shared listener dispatcher was already started")
            self._anchor_ns = int(self._now_ns())
            self._tick_index = 0
            self._dispatch_index = 0
            with self._stack_watchdog_state_lock:
                self._stack_watchdog_fired = False
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name=name, daemon=True)
            self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Stop the clock thread and join any in-flight batch/fanout."""
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or timeout < 0:
            raise ValueError("timeout must be a finite non-negative number")
        with self._lifecycle_lock:
            thread = self._thread
            self._stop_event.set()
            self._wake_event.set()
        with self._dispatch_idle:
            self._dispatch_idle.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=float(timeout))
        if thread is None or not thread.is_alive():
            with self._lifecycle_lock:
                self._thread = None

    def dispatch_next_tick(self) -> SharedListenerTickResult:
        """Wait for a ready frame or the next deadline, then batch and fan out."""
        if self._fatal_error is not None:
            raise RuntimeError(f"shared listener dispatcher is fail-closed: {self._fatal_error}")
        started_ns = time.monotonic_ns()
        watchdog_context: dict[str, Any] = {
            "tick_index": None,
            "rows": (),
            "phase": "setup",
            "phase_started_ns": started_ns,
        }
        watchdog_stop = threading.Event()
        watchdog_thread: threading.Thread | None = None
        if self._stack_watchdog_ms > 0:
            with self._stack_watchdog_state_lock:
                if not self._stack_watchdog_fired:
                    watchdog_thread = threading.Thread(
                        target=self._watch_dispatch_tick,
                        args=(watchdog_stop, started_ns, watchdog_context),
                        name="listener-dispatch-stack-watchdog",
                        daemon=True,
                    )
                    watchdog_thread.start()
        try:
            return self._dispatch_next_tick(watchdog_context)
        finally:
            with self._stack_watchdog_state_lock:
                watchdog_stop.set()
            if watchdog_thread is not None:
                watchdog_thread.join()

    def _watch_dispatch_tick(
        self,
        stop_event: threading.Event,
        started_ns: int,
        context: dict[str, Any],
    ) -> None:
        if stop_event.wait(self._stack_watchdog_ms / 1000.0):
            return
        with self._stack_watchdog_state_lock:
            if stop_event.is_set() or self._stack_watchdog_fired:
                return
            self._stack_watchdog_fired = True
        now_ns = time.monotonic_ns()
        phase_started_ns = int(context.get("phase_started_ns", started_ns))
        rows = context.get("rows", ())
        row_ids = ",".join(
            f"{row.session_id}:{row.frame.frame_id}" for row in rows
        ) or "none"
        _LOGGER.error(
            "listener dispatcher tick exceeded watchdog: threshold_ms=%d elapsed_ms=%.1f "
            "phase=%s phase_elapsed_ms=%.1f tick=%s rows=%s; "
            "dumping all Python thread stacks",
            self._stack_watchdog_ms,
            max(0, now_ns - started_ns) / 1_000_000.0,
            context.get("phase", "unknown"),
            max(0, now_ns - phase_started_ns) / 1_000_000.0,
            context.get("tick_index"),
            row_ids,
        )
        try:
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        except BaseException:
            _LOGGER.exception("listener dispatcher stack watchdog dump failed")

    def _has_backlog_unlocked(self, now_ns: int) -> bool:
        if any(int(frame.enqueue_ns) <= now_ns for frame in self._deferred.values()):
            return True
        for session_id, (ingress, _, _) in self._registrations.items():
            if session_id in self._paused_sessions:
                continue
            deque_obj = getattr(getattr(ingress, "_queue", None), "queue", None)
            if deque_obj is None:
                deque_obj = getattr(ingress, "queue", None)
            if deque_obj is None and hasattr(ingress, "__len__") and hasattr(ingress, "__getitem__"):
                deque_obj = ingress
            if deque_obj is not None:
                try:
                    if len(deque_obj) > 0:
                        first_frame = deque_obj[0]
                        if int(getattr(first_frame, "enqueue_ns", 0)) <= now_ns:
                            return True
                except (TypeError, AttributeError, IndexError):
                    pass
        return False

    def _dispatch_next_tick(
        self,
        watchdog_context: dict[str, Any],
    ) -> SharedListenerTickResult:
        with self._lock:
            if self._anchor_ns is None:
                self._anchor_ns = int(self._now_ns())
                self._tick_index = 0
            now = int(self._now_ns())
            nominal_deadline_ns = self._anchor_ns + (self._tick_index + 1) * self._period_ns
            if now > nominal_deadline_ns + self._period_ns:
                missed_ticks = (now - nominal_deadline_ns) // self._period_ns
                self._tick_index += int(missed_ticks)
                nominal_deadline_ns = self._anchor_ns + (self._tick_index + 1) * self._period_ns

            has_backlog = self._has_backlog_unlocked(now)
            # Keep the absolute grid deadline as the latency reference even
            # when an input-ready notification dispatches before that deadline.
            deadline_ns = nominal_deadline_ns
            tick_index = self._tick_index
            dispatch_index = self._dispatch_index
        watchdog_context.update(
            tick_index=tick_index,
            phase="wait_deadline",
            phase_started_ns=time.monotonic_ns(),
        )

        if self._external_wait_until is not None:
            self._external_wait_until(deadline_ns)
        elif not has_backlog:
            self._wait_until(deadline_ns)
        observed_ns = int(self._now_ns())
        watchdog_context.update(phase="snapshot", phase_started_ns=time.monotonic_ns())
        rows = self._snapshot_ready_rows(tick_index, deadline_ns, observed_ns)
        watchdog_context.update(
            rows=rows,
            phase="batch_setup",
            phase_started_ns=time.monotonic_ns(),
        )
        batch_context_entered = False
        result_error: BaseException | None = None
        outputs: Mapping[str, Any] = {}
        if rows:
            try:
                # Marked in-flight at snapshot time, so no later Talker quantum
                # can enter while an already admitted one drains. Do this before
                # registering listener lock priority: an admitted Talker must
                # still be able to acquire its model lock and finish.
                self.wait_for_compute_quanta()
                if self._fatal_error is not None:
                    raise self._fatal_error
                self._snapshot_compute_dependencies()
            except BaseException as exc:
                result_error = exc
                self._fatal_error = exc

        if rows and result_error is None:
            try:
                batch_context = (
                    self._batch_context(rows)
                    if self._batch_context is not None
                    else nullcontext()
                )
                batch_context.__enter__()
                batch_context_entered = True
            except BaseException as exc:
                result_error = exc
                self._fatal_error = exc
        batch_started_ns: int | None = None
        batch_finished_ns: int | None = None
        dependencies_completed = False

        if rows and batch_context_entered and result_error is None:
            batch_started_ns = int(self._now_ns())
            for row in rows:
                row.tick.listener_start_ns = batch_started_ns
                row.tick.listener_batch_started_ns = batch_started_ns
                row.tick.listener_batch_id = f"shared-listener-{dispatch_index}"
                row.tick.listener_batch_size = len(rows)
                row.tick.microbatch_queue_wait_ms = round(
                    max(0, batch_started_ns - int(row.frame.enqueue_ns)) / 1_000_000.0, 3
                )
            try:
                watchdog_context.update(
                    phase="batch_listener",
                    phase_started_ns=time.monotonic_ns(),
                )
                outputs = self._batch_listener(rows)
                if isinstance(outputs, Future):
                    watchdog_context.update(
                        phase="batch_future_wait",
                        phase_started_ns=time.monotonic_ns(),
                    )
                    outputs = outputs.result()
                self._finish_compute_dependencies(completed=True)
                dependencies_completed = True
                watchdog_context.update(
                    phase="output_validation",
                    phase_started_ns=time.monotonic_ns(),
                )
                try:
                    outputs = self._validate_outputs(rows, outputs)
                except BaseException as exc:
                    # Rows have already been removed from ingress. If the
                    # batch output cannot be matched exactly, continuing could
                    # silently lose those frames or commit stale state.
                    self._fatal_error = exc
                    raise
            except BaseException as exc:
                result_error = exc
                outputs = {}
                if _is_irrecoverable_gpu_error(exc):
                    self._fatal_error = exc
                if not dependencies_completed:
                    self._finish_compute_dependencies(completed=False)
            batch_finished_ns = int(self._now_ns())
            for row in rows:
                row.tick.listener_end_ns = batch_finished_ns
                row.tick.listener_batch_finished_ns = batch_finished_ns

        # The worker Future and output validation are the listener's service
        # boundary. Release GPU priority and publish idle before host fanout so
        # ACK/state callbacks cannot hold compute admission closed.
        if rows and result_error is not None and not dependencies_completed:
            self._finish_compute_dependencies(completed=False)
        if batch_context_entered:
            try:
                batch_context.__exit__(None, None, None)
            except BaseException as exc:
                self._fatal_error = exc
                if result_error is None:
                    result_error = exc
                    outputs = {}
        if rows:
            with self._dispatch_idle:
                self._inflight_sessions.difference_update(row.session_id for row in rows)
                self._dispatch_idle.notify_all()

        receipt = SharedListenerTickResult(
            tick_index=tick_index,
            deadline_ns=deadline_ns,
            observed_ns=observed_ns,
            batch_started_ns=batch_started_ns,
            batch_finished_ns=batch_finished_ns,
            rows=rows,
            error=None if result_error is None else f"{type(result_error).__name__}: {result_error}",
            dispatch_id=dispatch_index,
        )
        for row in rows:
            row.tick.microbatch_error = receipt.error
            row.tick.microbatch_service_ms = receipt.batch_service_ms
            row.tick.microbatch_compute_ms = receipt.batch_compute_ms

        watchdog_context.update(
            phase="fanout",
            phase_started_ns=time.monotonic_ns(),
        )
        if result_error is not None:
            if _is_irrecoverable_gpu_error(result_error):
                self._fatal_error = result_error
            target_session_id = getattr(result_error, "session_id", None)
            if target_session_id is not None:
                matched_rows = tuple(r for r in rows if r.session_id == target_session_id)
                if matched_rows:
                    self._fanout_errors(matched_rows, result_error, receipt)
                else:
                    self._fanout_errors(rows, result_error, receipt)
            else:
                self._fanout_errors(rows, result_error, receipt)
        elif rows:
            self._fanout_results(rows, outputs, receipt)

        with self._lock:
            self._dispatch_index += 1
            if self._anchor_ns is not None:
                elapsed_grid_index = max(
                    0,
                    (int(self._now_ns()) - self._anchor_ns) // self._period_ns,
                )
                self._tick_index = max(self._tick_index, int(elapsed_grid_index))
        return receipt

    def _snapshot_ready_rows(
        self,
        tick_index: int,
        deadline_ns: int,
        observed_ns: int,
    ) -> tuple[SharedListenerInput, ...]:
        del tick_index
        with self._dispatch_idle:
            registrations = tuple(self._registrations.items())
            rows: list[SharedListenerInput] = []
            for session_id, (ingress, listener_generation_id, state_epoch_id) in registrations:
                if session_id in self._paused_sessions:
                    continue
                frame = self._deferred.pop(session_id, None)
                if frame is None:
                    try:
                        frame = ingress.get_nowait()
                    except queue.Empty:
                        continue
                if not isinstance(frame, AudioFrame):
                    error = TypeError(f"ingress for {session_id} returned a non-AudioFrame value")
                    self._fatal_error = error
                    raise error
                cutoff_ns = max(deadline_ns, observed_ns)
                if int(frame.enqueue_ns) > cutoff_ns:
                    # Keep a frame consumed during the snapshot but enqueued after
                    # the shared deadline for the next tick. Ingress is FIFO, so
                    # no earlier frame for this session can be hidden behind it.
                    if session_id in self._registrations:
                        if session_id in self._deferred:
                            raise RuntimeError(f"multiple deferred frames for {session_id}")
                        self._deferred[session_id] = frame
                    continue
                tick = DuplexTick(
                    frame_id=int(frame.frame_id),
                    capture_start_ns=int(frame.capture_start_ns),
                    capture_end_ns=int(frame.capture_end_ns),
                    enqueue_ns=int(frame.enqueue_ns),
                    deadline_ns=deadline_ns,
                    observed_ns=observed_ns,
                    dispatch_mode="shared_fixed_80ms_microbatch",
                )
                rows.append(SharedListenerInput(
                    session_id,
                    frame,
                    tick,
                    listener_generation_id,
                    state_epoch_id,
                ))
            self._inflight_sessions.update(row.session_id for row in rows)
            return tuple(rows)

    @staticmethod
    def _validate_outputs(
        rows: tuple[SharedListenerInput, ...],
        outputs: Any,
    ) -> Mapping[str, Any]:
        if not isinstance(outputs, Mapping):
            raise TypeError("batch listener must return an output mapping keyed by session ID")
        expected = {row.session_id for row in rows}
        actual = set(outputs)
        if actual != expected:
            raise RuntimeError(
                "batch listener outputs must exactly cover the ready rows "
                f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
            )
        for row in rows:
            output = outputs[row.session_id]
            output_frame_id = getattr(output, "frame_id", None)
            if output_frame_id is not None and int(output_frame_id) != int(row.frame.frame_id):
                raise RuntimeError(
                    f"batch output frame identity mismatch for {row.session_id}: "
                    f"{output_frame_id} != {row.frame.frame_id}"
                )
        return outputs

    def _fanout_results(
        self,
        rows: tuple[SharedListenerInput, ...],
        outputs: Mapping[str, Any],
        receipt: SharedListenerTickResult,
    ) -> None:
        if self._on_result is None:
            return
        for index, row in enumerate(rows):
            try:
                self._on_result(row, outputs[row.session_id], receipt)
            except BaseException as exc:
                try:
                    self._notify_error(row, exc, receipt)
                except BaseException:
                    pass
                if not _is_irrecoverable_gpu_error(exc):
                    continue
                self._fatal_error = exc
                self._fanout_errors(rows[index + 1:], exc, receipt)
                return

    def _fanout_errors(
        self,
        rows: tuple[SharedListenerInput, ...],
        error: BaseException,
        receipt: SharedListenerTickResult,
    ) -> None:
        for row in rows:
            try:
                self._notify_error(row, error, receipt)
            except BaseException:
                pass

    def _notify_error(
        self,
        row: SharedListenerInput,
        error: BaseException,
        receipt: SharedListenerTickResult,
    ) -> None:
        if self._on_error is not None:
            self._on_error(row, error, receipt)

    def _wait_until(self, deadline_ns: int) -> None:
        while not self._stop_event.is_set():
            if self._wake_event.is_set():
                self._wake_event.clear()
                if self._stop_event.is_set():
                    break
                return
            remaining_ns = int(deadline_ns) - int(self._now_ns())
            if remaining_ns <= 0:
                return
            if self._wake_event.wait(remaining_ns / 1_000_000_000.0):
                self._wake_event.clear()
                if self._stop_event.is_set():
                    break
                return
        raise RuntimeError("shared listener dispatcher stopped before the next deadline")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.dispatch_next_tick()
            except BaseException as exc:
                if self._stop_event.is_set():
                    return
                if self._fatal_error is None:
                    self._fatal_error = exc
                self._stop_event.set()
                return
