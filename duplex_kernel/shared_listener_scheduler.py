"""Fixed-grid ingress gather for one shared 80 ms listener owner.

The dispatcher snapshots at most one complete frame per registered session at
each absolute deadline, sends every frame present at that deadline through one
batch call, and only then fans results back to the per-session scheduler. A
frame that was enqueued after the deadline is retained for the next tick. The
dispatcher never creates silence rows or falls back to per-session inference.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future
from contextlib import nullcontext
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

    @property
    def batch_size(self) -> int:
        return len(self.rows)

    @property
    def batch_queue_wait_ms(self) -> float | None:
        if not self.rows:
            return None
        oldest_enqueue_ns = min(int(row.frame.enqueue_ns) for row in self.rows)
        return round(max(0, self.deadline_ns - oldest_enqueue_ns) / 1_000_000.0, 3)

    @property
    def batch_compute_ms(self) -> float | None:
        if self.batch_started_ns is None or self.batch_finished_ns is None:
            return None
        return round(max(0, self.batch_finished_ns - self.batch_started_ns) / 1_000_000.0, 3)


BatchListener = Callable[
    [tuple[SharedListenerInput, ...]], Mapping[str, Any] | Future[Mapping[str, Any]]
]
ResultCallback = Callable[[SharedListenerInput, Any, SharedListenerTickResult], None]
ErrorCallback = Callable[[SharedListenerInput, BaseException, SharedListenerTickResult], None]


class SharedListenerTickDispatcher:
    """Gather every ready session on one absolute 80 ms cadence.

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
        self._lock = threading.RLock()
        self._dispatch_idle = threading.Condition(self._lock)
        self._inflight_sessions: set[str] = set()
        self._lifecycle_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._fatal_error: BaseException | None = None
        self._anchor_ns: int | None = None
        self._tick_index = 0

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
            if wait_for_inflight and not self._dispatch_idle.wait_for(
                lambda: session_id not in self._inflight_sessions,
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
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=float(timeout))
        if thread is None or not thread.is_alive():
            with self._lifecycle_lock:
                self._thread = None

    def dispatch_next_tick(self) -> SharedListenerTickResult:
        """Wait for the next shared deadline, gather once, batch, then fan out."""
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

    def _dispatch_next_tick(
        self,
        watchdog_context: dict[str, Any],
    ) -> SharedListenerTickResult:
        with self._lock:
            if self._anchor_ns is None:
                self._anchor_ns = int(self._now_ns())
                self._tick_index = 0
            tick_index = self._tick_index
            deadline_ns = self._anchor_ns + (tick_index + 1) * self._period_ns
        watchdog_context.update(
            tick_index=tick_index,
            phase="wait_deadline",
            phase_started_ns=time.monotonic_ns(),
        )

        if self._external_wait_until is not None:
            self._external_wait_until(deadline_ns)
        else:
            self._wait_until(deadline_ns)
        observed_ns = int(self._now_ns())
        watchdog_context.update(phase="snapshot", phase_started_ns=time.monotonic_ns())
        rows = self._snapshot_ready_rows(tick_index, deadline_ns, observed_ns)
        watchdog_context.update(
            rows=rows,
            phase="batch_setup",
            phase_started_ns=time.monotonic_ns(),
        )
        batch_context = (
            self._batch_context(rows)
            if rows and self._batch_context is not None
            else nullcontext()
        )
        try:
            batch_context.__enter__()
        except BaseException:
            if rows:
                with self._dispatch_idle:
                    self._inflight_sessions.difference_update(row.session_id for row in rows)
                    self._dispatch_idle.notify_all()
            raise
        batch_started_ns: int | None = None
        batch_finished_ns: int | None = None
        result_error: BaseException | None = None

        if rows:
            batch_started_ns = int(self._now_ns())
            for row in rows:
                row.tick.listener_start_ns = batch_started_ns
                row.tick.listener_batch_started_ns = batch_started_ns
                row.tick.listener_batch_id = f"shared-listener-{tick_index}"
                row.tick.listener_batch_size = len(rows)
                row.tick.microbatch_queue_wait_ms = round(
                    max(0, deadline_ns - int(row.frame.enqueue_ns)) / 1_000_000.0, 3
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
                watchdog_context.update(
                    phase="output_validation",
                    phase_started_ns=time.monotonic_ns(),
                )
                outputs = self._validate_outputs(rows, outputs)
            except BaseException as exc:
                result_error = exc
                outputs = {}
            batch_finished_ns = int(self._now_ns())
            for row in rows:
                row.tick.listener_end_ns = batch_finished_ns
                row.tick.listener_batch_finished_ns = batch_finished_ns

        receipt = SharedListenerTickResult(
            tick_index=tick_index,
            deadline_ns=deadline_ns,
            observed_ns=observed_ns,
            batch_started_ns=batch_started_ns,
            batch_finished_ns=batch_finished_ns,
            rows=rows,
            error=None if result_error is None else f"{type(result_error).__name__}: {result_error}",
        )
        for row in rows:
            row.tick.microbatch_error = receipt.error
            row.tick.microbatch_compute_ms = receipt.batch_compute_ms

        try:
            watchdog_context.update(
                phase="fanout",
                phase_started_ns=time.monotonic_ns(),
            )
            if result_error is not None:
                self._fatal_error = result_error
                self._fanout_errors(rows, result_error, receipt)
            elif rows:
                self._fanout_results(rows, outputs, receipt)
        finally:
            try:
                batch_context.__exit__(None, None, None)
            finally:
                if rows:
                    with self._dispatch_idle:
                        self._inflight_sessions.difference_update(row.session_id for row in rows)
                        self._dispatch_idle.notify_all()

        with self._lock:
            self._tick_index += 1
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
                if int(frame.enqueue_ns) > deadline_ns:
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
                self._fatal_error = exc
                for pending in rows[index:]:
                    try:
                        self._notify_error(pending, exc, receipt)
                    except BaseException:
                        pass
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
            remaining_ns = int(deadline_ns) - int(self._now_ns())
            if remaining_ns <= 0:
                return
            self._stop_event.wait(min(remaining_ns / 1_000_000_000.0, 0.01))
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

