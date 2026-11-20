from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager
import asyncio
import faulthandler
import threading
import queue
import time

import numpy as np
import pytest

from duplex_kernel.scheduler import AudioFrame, AudioIngressQueue, DuplexScheduler, DuplexTick
from duplex_kernel.shared_listener_scheduler import (
    SharedListenerTickDispatcher,
)
from duplex_kernel.server_state import _SchedulerEventBuffer


class _FakeIngress:
    def __init__(self, *frames: AudioFrame) -> None:
        self.frames = deque(frames)

    @property
    def qsize(self) -> int:
        return len(self.frames)

    def get_nowait(self) -> AudioFrame:
        if not self.frames:
            raise queue.Empty
        return self.frames.popleft()


class _Clock:
    def __init__(self, now_ns: int = 0) -> None:
        self.now = now_ns
        self.waited: list[int] = []

    def now_ns(self) -> int:
        return self.now

    def wait_until(self, deadline_ns: int) -> None:
        self.waited.append(deadline_ns)
        self.now = max(self.now, deadline_ns)


def _frame(frame_id: int, enqueue_ns: int) -> AudioFrame:
    return AudioFrame(
        frame_id=frame_id,
        pcm=np.zeros(1280, dtype=np.float32),
        capture_start_ns=max(0, enqueue_ns - 80_000_000),
        capture_end_ns=enqueue_ns,
        enqueue_ns=enqueue_ns,
    )


def test_fixed_80ms_ticks_batch_all_ready_sessions_in_one_call_and_keep_row_identity() -> None:
    clock = _Clock()
    batches: list[tuple[tuple[str, int], ...]] = []
    delivered: list[tuple[str, int, str, int]] = []

    def run_batch(rows):
        batches.append(tuple((row.session_id, row.frame.frame_id) for row in rows))
        return {
            row.session_id: (row.session_id, row.frame.frame_id)
            for row in rows
        }

    dispatcher = SharedListenerTickDispatcher(
        run_batch,
        on_result=lambda row, output, receipt: delivered.append((
            row.session_id,
            row.frame.frame_id,
            str(output[0]),
            receipt.tick_index,
        )),
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
    )
    dispatcher.register_session("a", _FakeIngress(_frame(3, 70_000_000)))
    dispatcher.register_session("b", _FakeIngress(_frame(8, 79_000_000)))

    receipt = dispatcher.dispatch_next_tick()

    assert clock.waited == [80_000_000]
    assert batches == [(('a', 3), ('b', 8))]
    assert receipt.tick_index == 0
    assert receipt.deadline_ns == 80_000_000
    assert receipt.batch_size == 2
    assert receipt.batch_compute_ms == 0.0
    assert receipt.batch_service_ms == receipt.batch_compute_ms
    assert [item[:3] for item in delivered] == [
        ("a", 3, "a"),
        ("b", 8, "b"),
    ]
    assert all(item[3] == 0 for item in delivered)
    assert all(row.tick.dispatch_mode == "shared_fixed_80ms_microbatch" for row in receipt.rows)
    assert all(row.tick.listener_batch_size == 2 for row in receipt.rows)


def test_input_ready_dispatch_measures_queue_wait_from_actual_batch_start() -> None:
    clock = _Clock()
    dispatcher = SharedListenerTickDispatcher(
        lambda rows: {row.session_id: row.frame.frame_id for row in rows},
        now_ns=clock.now_ns,
        wait_until=lambda _deadline: None,
    )
    dispatcher.register_session("early", _FakeIngress(_frame(1, 0)))

    receipt = dispatcher.dispatch_next_tick()

    assert receipt.deadline_ns == 80_000_000
    assert receipt.batch_started_ns == 0
    assert receipt.batch_queue_wait_ms == 0.0
    assert receipt.rows[0].tick.microbatch_queue_wait_ms == 0.0


def test_batch_context_enter_failure_reports_selected_rows_without_losing_them() -> None:
    clock = _Clock()
    errors: list[tuple[int, str]] = []

    class _FailingContext:
        def __enter__(self):
            raise RuntimeError("priority context unavailable")

        def __exit__(self, *_args):
            pytest.fail("__exit__ must not run when __enter__ fails")

    dispatcher = SharedListenerTickDispatcher(
        lambda _rows: pytest.fail("batch listener must not run without priority context"),
        on_error=lambda row, error, _receipt: errors.append((row.frame.frame_id, str(error))),
        batch_context=lambda _rows: _FailingContext(),
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
    )
    ingress = _FakeIngress(_frame(9, 70_000_000))
    dispatcher.register_session("s", ingress)

    receipt = dispatcher.dispatch_next_tick()

    assert receipt.batch_size == 1
    assert receipt.batch_started_ns is None
    assert receipt.batch_finished_ns is None
    assert receipt.error == "RuntimeError: priority context unavailable"
    assert errors == [(9, "priority context unavailable")]
    assert not dispatcher.has_pending_work("s")


def test_fixed_80ms_tick_keeps_three_ready_sessions_in_one_batch() -> None:
    clock = _Clock()
    batches: list[tuple[str, ...]] = []
    delivered: list[tuple[str, int]] = []

    def run_batch(rows):
        batches.append(tuple(row.session_id for row in rows))
        return {row.session_id: row.frame.frame_id for row in rows}

    dispatcher = SharedListenerTickDispatcher(
        run_batch,
        on_result=lambda row, output, _receipt: delivered.append((row.session_id, output)),
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
    )
    for index in range(3):
        dispatcher.register_session(
            f"s{index}", _FakeIngress(_frame(index + 10, 79_000_000))
        )

    receipt = dispatcher.dispatch_next_tick()

    assert batches == [("s0", "s1", "s2")]
    assert receipt.batch_size == 3
    assert all(row.tick.listener_batch_size == 3 for row in receipt.rows)
    assert delivered == [("s0", 10), ("s1", 11), ("s2", 12)]


def test_uneven_readiness_batches_only_deadline_ready_rows_then_late_session() -> None:
    clock = _Clock()
    batches: list[tuple[str, ...]] = []
    dispatcher = SharedListenerTickDispatcher(
        lambda rows: batches.append(tuple(row.session_id for row in rows))
        or {row.session_id: row.frame.frame_id for row in rows},
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
    )
    dispatcher.register_session("a", _FakeIngress(_frame(1, 79_000_000)))
    dispatcher.register_session("b", _FakeIngress(_frame(2, 80_000_000)))
    dispatcher.register_session("late", _FakeIngress(_frame(3, 81_000_000)))

    first = dispatcher.dispatch_next_tick()
    second = dispatcher.dispatch_next_tick()

    assert first.batch_size == 2
    assert second.batch_size == 1
    assert batches == [("a", "b"), ("late",)]


def test_no_ready_rows_do_not_call_batcher_or_fabricate_listener_output() -> None:
    clock = _Clock()
    batch_calls: list[object] = []
    results: list[object] = []
    dispatcher = SharedListenerTickDispatcher(
        lambda rows: batch_calls.append(rows) or {},
        on_result=lambda *args: results.append(args),
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
    )
    dispatcher.register_session("quiet", _FakeIngress())

    receipt = dispatcher.dispatch_next_tick()

    assert receipt.batch_size == 0
    assert receipt.rows == ()
    assert batch_calls == []
    assert results == []
    assert clock.waited == [80_000_000]


def test_frame_arriving_after_shared_deadline_is_deferred_without_reordering() -> None:
    clock = _Clock()
    batch_frames: list[tuple[int, ...]] = []
    ingress = _FakeIngress(_frame(1, 81_000_000), _frame(2, 90_000_000))
    dispatcher = SharedListenerTickDispatcher(
        lambda rows: batch_frames.append(tuple(row.frame.frame_id for row in rows))
        or {row.session_id: row.frame.frame_id for row in rows},
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
    )
    dispatcher.register_session("late", ingress)

    first = dispatcher.dispatch_next_tick()
    second = dispatcher.dispatch_next_tick()

    assert first.deadline_ns == 80_000_000
    assert first.batch_size == 0
    assert second.deadline_ns == 160_000_000
    assert second.batch_size == 1
    assert batch_frames == [(1,)]
    assert list(ingress.frames)[0].frame_id == 2


def test_initial_prefix_guard_can_see_deferred_and_inflight_dispatcher_frames() -> None:
    clock = _Clock()
    during_batch: list[bool] = []
    dispatcher_holder = {}
    dispatcher = SharedListenerTickDispatcher(
        lambda rows: during_batch.append(
            dispatcher_holder["dispatcher"].has_pending_work(rows[0].session_id)
        ) or {row.session_id: row.frame.frame_id for row in rows},
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
    )
    dispatcher_holder["dispatcher"] = dispatcher
    dispatcher.register_session("deferred", _FakeIngress(_frame(1, 81_000_000)))

    first = dispatcher.dispatch_next_tick()
    assert first.batch_size == 0
    assert dispatcher.has_pending_work("deferred")
    second = dispatcher.dispatch_next_tick()

    assert second.batch_size == 1
    assert during_batch == [True]
    assert not dispatcher.has_pending_work("deferred")


def test_shared_dispatcher_accepts_sixteen_active_conversations() -> None:
    dispatcher = SharedListenerTickDispatcher(lambda rows: {})
    for index in range(16):
        dispatcher.register_session(f"s{index}", _FakeIngress())

    assert len(dispatcher.active_session_ids) == 16


def test_dispatcher_stack_watchdog_covers_blocking_fanout_once(monkeypatch, caplog) -> None:
    clock = _Clock()
    dump_calls: list[dict[str, object]] = []
    dump_finished = threading.Event()
    fanout_entered = threading.Event()
    release_fanout = threading.Event()
    monkeypatch.setattr(
        faulthandler,
        "dump_traceback",
        lambda **kwargs: (dump_calls.append(kwargs), dump_finished.set()),
    )

    def run_batch(rows):
        return {row.session_id: "ok" for row in rows}

    def on_result(_row, _output, _receipt):
        fanout_entered.set()
        assert release_fanout.wait(2.0)

    dispatcher = SharedListenerTickDispatcher(
        run_batch,
        on_result=on_result,
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
        stack_watchdog_ms=20,
    )
    dispatcher.register_session("s", _FakeIngress(_frame(0, 70_000_000)))
    receipts = []
    dispatch_thread = threading.Thread(
        target=lambda: receipts.append(dispatcher.dispatch_next_tick()),
        name="test-listener-dispatch",
    )
    dispatch_thread.start()
    try:
        assert fanout_entered.wait(1.0)
        assert dump_finished.wait(1.0)
    finally:
        release_fanout.set()
        dispatch_thread.join(2.0)

    assert not dispatch_thread.is_alive()
    assert len(receipts) == 1 and receipts[0].error is None
    assert len(dump_calls) == 1
    assert dump_calls[0]["all_threads"] is True
    assert "phase=fanout" in caplog.text
    assert "tick=0" in caplog.text
    assert "s:0" in caplog.text


def test_dispatcher_stack_watchdog_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("DUPLEX_LISTENER_STACK_WATCHDOG_MS", raising=False)
    dispatcher = SharedListenerTickDispatcher(lambda rows: {})

    assert dispatcher.stack_watchdog_ms == 0


@pytest.mark.parametrize("row_count", (3, 5, 8, 12))
def test_shared_tick_batches_arbitrary_ready_conversation_counts(row_count: int) -> None:
    clock = _Clock()
    seen: list[tuple[str, ...]] = []
    dispatcher = SharedListenerTickDispatcher(
        lambda rows: seen.append(tuple(row.session_id for row in rows))
        or {row.session_id: row.frame.frame_id for row in rows},
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
    )
    for index in range(row_count):
        dispatcher.register_session(
            f"s{index}", _FakeIngress(_frame(index, 79_000_000))
        )

    receipt = dispatcher.dispatch_next_tick()

    assert receipt.batch_size == row_count
    assert seen == [tuple(f"s{index}" for index in range(row_count))]
    assert all(row.tick.listener_batch_size == row_count for row in receipt.rows)


def test_batch_result_must_cover_exact_rows_and_never_falls_back_per_session() -> None:
    clock = _Clock()
    per_row_calls: list[object] = []
    errors: list[tuple[str, str]] = []
    dispatcher = SharedListenerTickDispatcher(
        lambda rows: {rows[0].session_id: "only-first"},
        on_result=lambda *args: per_row_calls.append(args),
        on_error=lambda row, error, receipt: errors.append((row.session_id, str(error))),
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
    )
    dispatcher.register_session("a", _FakeIngress(_frame(0, 1)))
    dispatcher.register_session("b", _FakeIngress(_frame(0, 1)))

    receipt = dispatcher.dispatch_next_tick()

    assert receipt.batch_size == 2
    assert per_row_calls == []
    assert {session_id for session_id, _ in errors} == {"a", "b"}
    assert all("exactly cover" in message for _, message in errors)
    assert dispatcher.fatal_error is not None


def test_future_batch_result_is_resolved_before_fanout() -> None:
    clock = _Clock()
    delivered: list[object] = []
    result_future: Future[dict[str, int]] = Future()
    result_future.set_result({"s": 7})
    dispatcher = SharedListenerTickDispatcher(
        lambda _rows: result_future,
        on_result=lambda row, output, _receipt: delivered.append((row.frame.frame_id, output)),
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
    )
    dispatcher.register_session("s", _FakeIngress(_frame(7, 1)))

    dispatcher.dispatch_next_tick()

    assert delivered == [(7, 7)]


def test_listener_admission_sees_service_batch_from_another_session() -> None:
    batch_entered = threading.Event()
    release_batch = threading.Event()
    waiter_started = threading.Event()
    waiter_finished = threading.Event()
    ingress = AudioIngressQueue()
    other_ingress = AudioIngressQueue()
    other_ingress.put(np.zeros(1280, dtype=np.float32), frame_id=0)

    def run_batch(rows):
        batch_entered.set()
        assert release_batch.wait(2.0)
        return {row.session_id: row.frame.frame_id for row in rows}

    dispatcher = SharedListenerTickDispatcher(run_batch)
    dispatcher.register_session("s", ingress)
    dispatcher.register_session("other", other_ingress)
    dispatch_thread = threading.Thread(target=dispatcher.dispatch_next_tick, daemon=True)
    dispatch_thread.start()
    assert batch_entered.wait(1.0)
    assert ingress.qsize == 0
    assert other_ingress.qsize == 0

    admission: list[bool] = []

    def wait_for_admission():
        waiter_started.set()
        admission.append(dispatcher.wait_for_listener_admission(
            "s",
            ingress,
            lambda: True,
        ))
        waiter_finished.set()

    waiter_thread = threading.Thread(target=wait_for_admission, daemon=True)
    waiter_thread.start()
    assert waiter_started.wait(1.0)
    assert not waiter_finished.wait(0.05)

    release_batch.set()
    assert waiter_finished.wait(1.0)
    dispatch_thread.join(1.0)
    waiter_thread.join(1.0)
    assert admission == [True]
    assert not dispatch_thread.is_alive()
    assert not waiter_thread.is_alive()


def test_listener_admission_waits_for_queued_input_before_dispatch_selection() -> None:
    batch_entered = threading.Event()
    release_batch = threading.Event()
    waiter_started = threading.Event()
    waiter_finished = threading.Event()
    ingress = AudioIngressQueue()

    def run_batch(rows):
        batch_entered.set()
        assert release_batch.wait(2.0)
        return {row.session_id: row.frame.frame_id for row in rows}

    dispatcher = SharedListenerTickDispatcher(run_batch)
    dispatcher.register_session("s", ingress)
    ingress.set_on_frame_ready(lambda: dispatcher.notify_input_ready("s"))
    ingress.put(np.zeros(1280, dtype=np.float32), frame_id=0)

    admission: list[bool] = []

    def wait_for_admission():
        waiter_started.set()
        admission.append(dispatcher.wait_for_listener_admission(
            "s",
            ingress,
            lambda: True,
        ))
        waiter_finished.set()

    waiter_thread = threading.Thread(target=wait_for_admission, daemon=True)
    waiter_thread.start()
    assert waiter_started.wait(1.0)
    assert not waiter_finished.wait(0.05)

    dispatch_thread = threading.Thread(target=dispatcher.dispatch_next_tick, daemon=True)
    dispatch_thread.start()
    assert batch_entered.wait(1.0)
    assert ingress.qsize == 0
    assert not waiter_finished.wait(0.05)

    release_batch.set()
    assert waiter_finished.wait(1.0)
    dispatch_thread.join(1.0)
    waiter_thread.join(1.0)
    assert admission == [True]
    assert not dispatch_thread.is_alive()
    assert not waiter_thread.is_alive()


def test_compute_permit_publishes_stream_event_before_listener_batch_and_blocks_next_quantum() -> None:
    batch_future: Future[dict[str, int]] = Future()
    ingress = _FakeIngress()
    dispatcher_holder: dict[str, SharedListenerTickDispatcher] = {}
    first_permit_entered = threading.Event()
    release_first_permit = threading.Event()
    batch_priority_entered = threading.Event()
    batch_callback_entered = threading.Event()
    second_permit_started = threading.Event()
    event = object()
    stream = type("_Stream", (), {
        "device": type("_Device", (), {"index": 0})(),
        "cuda_stream": 42,
    })()
    observed_dependencies: list[tuple[object, ...]] = []

    def run_batch(rows):
        observed_dependencies.append(
            dispatcher_holder["dispatcher"].active_compute_dependencies()
        )
        batch_callback_entered.set()
        return batch_future

    @contextmanager
    def listener_priority(_rows):
        batch_priority_entered.set()
        yield

    dispatcher = SharedListenerTickDispatcher(
        run_batch,
        batch_context=listener_priority,
        now_ns=lambda: 0,
        wait_until=lambda _deadline: None,
    )
    dispatcher_holder["dispatcher"] = dispatcher
    dispatcher.register_session("s", ingress)

    def hold_first_quantum() -> None:
        with dispatcher.compute_quantum(lambda: True) as permit:
            assert permit is not False
            first_permit_entered.set()
            assert release_first_permit.wait(2.0)
            permit.record_cuda_completion(event, stream)

    first_thread = threading.Thread(target=hold_first_quantum, daemon=True)
    first_thread.start()
    assert first_permit_entered.wait(1.0)
    ingress.frames.append(_frame(1, 1))

    dispatch_thread = threading.Thread(target=dispatcher.dispatch_next_tick, daemon=True)
    dispatch_thread.start()
    assert dispatcher._dispatch_idle.wait_for(  # synchronize with the selected-row ticket
        lambda: dispatcher.has_pending_work("s"), timeout=1.0
    )

    def acquire_second_quantum() -> None:
        with dispatcher.compute_quantum(lambda: True) as permit:
            assert permit is not False
            second_permit_started.set()

    second_thread = threading.Thread(target=acquire_second_quantum, daemon=True)
    second_thread.start()
    assert not second_permit_started.wait(0.05)
    assert not batch_priority_entered.is_set()

    release_first_permit.set()
    assert batch_priority_entered.wait(1.0)
    assert batch_callback_entered.wait(1.0)
    assert observed_dependencies == [(event,)]
    assert not second_permit_started.wait(0.05)

    batch_future.set_result({"s": 1})
    assert second_permit_started.wait(1.0)
    first_thread.join(1.0)
    dispatch_thread.join(1.0)
    second_thread.join(1.0)
    assert not first_thread.is_alive()
    assert not dispatch_thread.is_alive()
    assert not second_thread.is_alive()
    assert dispatcher.active_compute_dependencies() == ()


def test_ingress_publication_claim_closes_empty_queue_permit_race() -> None:
    ingress = _FakeIngress()
    dispatcher = SharedListenerTickDispatcher(
        lambda _rows: {},
        now_ns=lambda: 0,
        wait_until=lambda _deadline: None,
    )
    dispatcher.register_session("s", ingress)
    admission_checked = threading.Event()
    permit_acquired = threading.Event()

    def generation_is_current() -> bool:
        admission_checked.set()
        return True

    with dispatcher.ingress_publication("s"):
        assert ingress.qsize == 0

        def acquire_compute_quantum() -> None:
            with dispatcher.compute_quantum(generation_is_current) as permit:
                assert permit is not False
                permit_acquired.set()

        thread = threading.Thread(target=acquire_compute_quantum, daemon=True)
        thread.start()
        assert admission_checked.wait(1.0)
        assert not permit_acquired.wait(0.05)

    assert permit_acquired.wait(1.0)
    thread.join(1.0)
    assert not thread.is_alive()


def test_compute_permit_aborts_when_generation_stales_during_wait() -> None:
    dispatcher = SharedListenerTickDispatcher(lambda _rows: {})
    dispatcher.register_session("s", _FakeIngress())
    waiting = threading.Event()
    stale = threading.Event()
    results: list[bool] = []

    def generation_is_current() -> bool:
        waiting.set()
        return not stale.is_set()

    def wait_for_permit() -> None:
        with dispatcher.compute_quantum(generation_is_current) as permit:
            results.append(permit is False)

    with dispatcher.ingress_publication("s"):
        thread = threading.Thread(target=wait_for_permit, daemon=True)
        thread.start()
        assert waiting.wait(1.0)
        stale.set()

    thread.join(1.0)
    assert not thread.is_alive()
    assert results == [True]
    assert dispatcher._compute_permits == set()


def test_compute_completion_failure_keeps_listener_batch_fail_closed() -> None:
    ingress = _FakeIngress()
    permit_entered = threading.Event()
    fail_permit = threading.Event()
    dispatch_finished = threading.Event()
    batch_calls: list[int] = []
    row_errors: list[str] = []

    def run_batch(rows):
        batch_calls.append(len(rows))
        return {row.session_id: row.frame.frame_id for row in rows}

    dispatcher = SharedListenerTickDispatcher(
        run_batch,
        on_error=lambda _row, error, _receipt: row_errors.append(str(error)),
        now_ns=lambda: 0,
        wait_until=lambda _deadline: None,
    )
    dispatcher.register_session("s", ingress)

    def hold_failing_quantum() -> None:
        with dispatcher.compute_quantum(lambda: True) as permit:
            assert permit is not False
            permit_entered.set()
            assert fail_permit.wait(2.0)
            permit.fail(RuntimeError("stream event recording failed"))

    permit_thread = threading.Thread(target=hold_failing_quantum, daemon=True)
    permit_thread.start()
    assert permit_entered.wait(1.0)
    ingress.frames.append(_frame(3, 1))

    receipt_holder = []

    def dispatch() -> None:
        receipt_holder.append(dispatcher.dispatch_next_tick())
        dispatch_finished.set()

    dispatch_thread = threading.Thread(target=dispatch, daemon=True)
    dispatch_thread.start()
    assert dispatcher._dispatch_idle.wait_for(
        lambda: dispatcher.has_pending_work("s"), timeout=1.0
    )
    fail_permit.set()

    assert dispatch_finished.wait(1.0)
    permit_thread.join(1.0)
    dispatch_thread.join(1.0)
    assert not permit_thread.is_alive()
    assert not dispatch_thread.is_alive()
    assert batch_calls == []
    assert "stream event recording failed" in row_errors[0]
    assert receipt_holder[0].error is not None


def test_listener_priority_and_demand_release_before_result_fanout() -> None:
    result_future: Future[dict[str, int]] = Future()
    fanout_started = threading.Event()
    release_fanout = threading.Event()
    waiter_started = threading.Event()
    waiter_finished = threading.Event()
    context_active = threading.Event()
    ingress = AudioIngressQueue()
    ingress.put(np.zeros(1280, dtype=np.float32), frame_id=0)

    class _PriorityContext:
        def __enter__(self):
            context_active.set()

        def __exit__(self, *_args):
            context_active.clear()

    def on_result(_row, _output, _receipt):
        fanout_started.set()
        assert release_fanout.wait(2.0)

    dispatcher = SharedListenerTickDispatcher(
        lambda _rows: result_future,
        on_result=on_result,
        batch_context=lambda _rows: _PriorityContext(),
    )
    dispatcher.register_session("s", ingress)
    dispatch_thread = threading.Thread(target=dispatcher.dispatch_next_tick, daemon=True)
    dispatch_thread.start()

    waiter_result: list[bool] = []

    def wait_for_admission():
        waiter_started.set()
        waiter_result.append(dispatcher.wait_for_listener_admission(
            "s", ingress, lambda: True,
        ))
        waiter_finished.set()

    waiter_thread = threading.Thread(target=wait_for_admission, daemon=True)
    waiter_thread.start()
    assert waiter_started.wait(1.0)
    assert not waiter_finished.wait(0.05)

    result_future.set_result({"s": 0})
    assert fanout_started.wait(1.0)
    try:
        assert not context_active.is_set(), "listener priority context stayed active through result fanout"
        assert waiter_finished.wait(0.5), "listener demand stayed active through host result fanout"
        assert waiter_result == [True]
    finally:
        release_fanout.set()
    dispatch_thread.join(1.0)
    waiter_thread.join(1.0)
    assert not dispatch_thread.is_alive()
    assert not waiter_thread.is_alive()


def test_listener_admission_aborts_when_generation_turns_stale() -> None:
    batch_entered = threading.Event()
    release_batch = threading.Event()
    waiter_started = threading.Event()
    waiter_finished = threading.Event()
    generation_current = threading.Event()
    generation_current.set()
    ingress = AudioIngressQueue()
    ingress.put(np.zeros(1280, dtype=np.float32), frame_id=0)

    def run_batch(rows):
        batch_entered.set()
        assert release_batch.wait(2.0)
        return {row.session_id: row.frame.frame_id for row in rows}

    dispatcher = SharedListenerTickDispatcher(run_batch)
    dispatcher.register_session("s", ingress)
    dispatch_thread = threading.Thread(target=dispatcher.dispatch_next_tick, daemon=True)
    dispatch_thread.start()
    assert batch_entered.wait(1.0)

    admission: list[bool] = []

    def wait_for_admission():
        waiter_started.set()
        admission.append(dispatcher.wait_for_listener_admission(
            "s",
            ingress,
            generation_current.is_set,
        ))
        waiter_finished.set()

    waiter_thread = threading.Thread(target=wait_for_admission, daemon=True)
    waiter_thread.start()
    assert waiter_started.wait(1.0)
    generation_current.clear()
    assert waiter_finished.wait(0.5)

    release_batch.set()
    dispatch_thread.join(1.0)
    waiter_thread.join(1.0)
    assert admission == [False]
    assert not dispatch_thread.is_alive()
    assert not waiter_thread.is_alive()


def test_pause_session_waits_for_inflight_and_preserves_ingress_fifo() -> None:
    clock = _Clock()
    batch_entered = threading.Event()
    release_batch = threading.Event()
    pause_finished = threading.Event()
    ingress = _FakeIngress(
        _frame(0, 70_000_000),
        _frame(1, 71_000_000),
    )

    def run_batch(rows):
        batch_entered.set()
        assert release_batch.wait(2.0)
        return {row.session_id: row.frame.frame_id for row in rows}

    dispatcher = SharedListenerTickDispatcher(
        run_batch,
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
    )
    dispatcher.register_session("s", ingress)
    receipts = []
    dispatch_thread = threading.Thread(
        target=lambda: receipts.append(dispatcher.dispatch_next_tick()),
        daemon=True,
    )
    dispatch_thread.start()
    assert batch_entered.wait(1.0)

    pause_result = []

    def pause():
        pause_result.append(dispatcher.pause_session("s", timeout=2.0))
        pause_finished.set()

    pause_thread = threading.Thread(target=pause, daemon=True)
    pause_thread.start()
    assert not pause_finished.wait(0.05)
    release_batch.set()
    dispatch_thread.join(2.0)
    pause_thread.join(2.0)

    assert not dispatch_thread.is_alive()
    assert not pause_thread.is_alive()
    assert pause_result == [True]
    assert len(receipts) == 1 and receipts[0].rows[0].frame.frame_id == 0
    paused = dispatcher.dispatch_next_tick()
    assert paused.rows == ()
    assert [frame.frame_id for frame in ingress.frames] == [1]

    assert dispatcher.resume_session("s") is True
    resumed = dispatcher.dispatch_next_tick()
    assert [row.frame.frame_id for row in resumed.rows] == [1]
    assert list(ingress.frames) == []


def test_shared_context_covers_batch_and_releases_before_all_row_fanout() -> None:
    clock = _Clock()
    phases: list[str] = []

    @contextmanager
    def context(_rows):
        phases.append("enter")
        yield
        phases.append("exit")

    def run_batch(rows):
        phases.append("batch")
        return {row.session_id: row.frame.frame_id for row in rows}

    dispatcher = SharedListenerTickDispatcher(
        run_batch,
        on_result=lambda *_: phases.append("fanout"),
        batch_context=context,
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
    )
    dispatcher.register_session("a", _FakeIngress(_frame(1, 1)))
    dispatcher.register_session("b", _FakeIngress(_frame(2, 1)))

    dispatcher.dispatch_next_tick()

    assert phases == ["enter", "batch", "exit", "fanout", "fanout"]


def test_row_fanout_failure_isolated_to_each_session() -> None:
    clock = _Clock()
    delivered: list[str] = []
    errors: list[str] = []

    def on_result(row, _output, _receipt):
        delivered.append(row.session_id)
        raise RuntimeError("row commit failed")

    dispatcher = SharedListenerTickDispatcher(
        lambda rows: {row.session_id: row.frame.frame_id for row in rows},
        on_result=on_result,
        on_error=lambda row, _error, _receipt: errors.append(row.session_id),
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
    )
    dispatcher.register_session("a", _FakeIngress(_frame(1, 1)))
    dispatcher.register_session("b", _FakeIngress(_frame(2, 1)))

    dispatcher.dispatch_next_tick()

    assert delivered == ["a", "b"]
    assert errors == ["a", "b"]
    assert dispatcher.fatal_error is None


def test_session_scheduler_fans_out_a_batched_frame_without_dequeuing_it_again() -> None:
    ingress = AudioIngressQueue(max_frames=2)
    seen: list[str] = []
    scheduler = DuplexScheduler(
        ingress,
        on_listener=lambda _frame, _tick: seen.append("listener"),
        on_policy=lambda _frame, _tick: seen.append("policy"),
        on_speaker=lambda _frame, _tick: seen.append("speaker"),
    )
    frame = _frame(12, 81_000_000)
    tick = DuplexTick(
        frame_id=frame.frame_id,
        capture_start_ns=frame.capture_start_ns,
        capture_end_ns=frame.capture_end_ns,
        enqueue_ns=frame.enqueue_ns,
        deadline_ns=160_000_000,
        observed_ns=160_000_000,
        listener_start_ns=160_000_000,
        listener_end_ns=165_000_000,
        dispatch_mode="shared_fixed_80ms_microbatch",
    )

    returned = scheduler.dispatch_shared_frame(frame, tick)

    assert returned is tick
    assert seen == ["listener", "policy", "speaker"]
    assert scheduler.stats()["ticks"] == 1
    assert scheduler.stats()["missing_ticks"] == 0
    assert scheduler.history == (tick,)
    assert ingress.qsize == 0
    assert tick.listener_start_ns == 160_000_000
    assert tick.listener_end_ns == 165_000_000
    assert tick.policy_ns is not None
    assert tick.speaker_start_ns is not None and tick.speaker_end_ns is not None


def test_shared_listener_dispatcher_with_audio_ingress_queue_backlog() -> None:
    now = 1_000_000_000
    clock = _Clock(now_ns=now)
    batches = []
    dispatcher = SharedListenerTickDispatcher(
        lambda rows: {r.session_id: r.frame.frame_id for r in rows},
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
    )
    ingress = AudioIngressQueue()
    frame = _frame(0, enqueue_ns=now + 40_000_000)
    ingress._queue.put(frame)
    dispatcher.register_session("sess1", ingress)

    receipt = dispatcher.dispatch_next_tick()

    assert receipt.batch_size == 1
    assert receipt.rows[0].session_id == "sess1"


def test_audio_ingress_notifies_after_complete_frame_is_queued_and_outside_lock() -> None:
    ingress = AudioIngressQueue()
    notifications: list[int] = []

    def on_frame_ready() -> None:
        assert not ingress._lock._is_owned()
        notifications.append(ingress.qsize)

    ingress.set_on_frame_ready(on_frame_ready)
    ingress.append(np.zeros(1279, dtype=np.float32))
    assert notifications == []

    ingress.append(np.zeros(1, dtype=np.float32))
    assert notifications == [1]
    ingress.get_nowait()

    ingress.put(np.zeros(1280, dtype=np.float32))
    assert notifications == [1, 1]


def test_ready_frame_wakes_dispatcher_after_empty_snapshot_without_next_grid_wait() -> None:
    ingress = AudioIngressQueue()
    empty_snapshot = threading.Event()
    frame_enqueued = threading.Event()
    delivered = threading.Event()
    receipts = []
    dispatcher = SharedListenerTickDispatcher(
        lambda rows: {row.session_id: row.frame.frame_id for row in rows},
        on_result=lambda _row, _output, receipt: (receipts.append(receipt), delivered.set()),
        period_ns=250_000_000,
    )
    original_snapshot = dispatcher._snapshot_ready_rows
    snapshot_count = 0

    def snapshot(tick_index: int, deadline_ns: int, observed_ns: int):
        nonlocal snapshot_count
        rows = original_snapshot(tick_index, deadline_ns, observed_ns)
        if snapshot_count == 0 and not rows:
            empty_snapshot.set()
        snapshot_count += 1
        return rows

    dispatcher._snapshot_ready_rows = snapshot
    ingress.set_on_frame_ready(lambda: dispatcher.notify_input_ready("late"))
    dispatcher.register_session("late", ingress)

    def enqueue_just_after_snapshot() -> None:
        assert empty_snapshot.wait(1.5)
        time.sleep(0.0015)
        ingress.put(np.zeros(1280, dtype=np.float32))
        frame_enqueued.set()

    producer = threading.Thread(target=enqueue_just_after_snapshot, daemon=True)
    dispatcher.start()
    producer.start()
    try:
        assert frame_enqueued.wait(1.5)
        assert delivered.wait(0.15)
    finally:
        dispatcher.stop(timeout=1.0)
        producer.join(1.0)

    assert not producer.is_alive()
    assert len(receipts) == 1
    assert receipts[0].rows[0].frame.frame_id == 0
    assert receipts[0].deadline_ns - receipts[0].observed_ns > 50_000_000


def test_ready_backlog_keeps_nominal_grid_deadline_across_dispatches() -> None:
    ingress = AudioIngressQueue()
    delivered = threading.Event()
    dispatches = []
    dispatcher = SharedListenerTickDispatcher(
        lambda rows: {row.session_id: row.frame.frame_id for row in rows},
        on_result=lambda row, _output, receipt: (
            dispatches.append((row.frame.frame_id, row.tick.listener_batch_id, receipt)),
            delivered.set() if len(dispatches) == 2 else None,
        ),
        period_ns=250_000_000,
    )
    ingress.set_on_frame_ready(lambda: dispatcher.notify_input_ready("backlog"))
    dispatcher.register_session("backlog", ingress)
    ingress.put(np.zeros(1280, dtype=np.float32))
    ingress.put(np.zeros(1280, dtype=np.float32))

    dispatcher.start()
    try:
        assert delivered.wait(0.2)
    finally:
        dispatcher.stop(timeout=1.0)

    assert [frame_id for frame_id, _batch_id, _receipt in dispatches] == [0, 1]
    assert dispatches[0][2].tick_index == dispatches[1][2].tick_index
    assert dispatches[0][2].deadline_ns == dispatches[1][2].deadline_ns
    assert dispatches[0][2].dispatch_id != dispatches[1][2].dispatch_id
    assert dispatches[0][1] != dispatches[1][1]


def test_blocked_policy_fanout_holds_later_acks_but_not_sender_or_event_loop() -> None:
    loop = asyncio.new_event_loop()
    loop_started = threading.Event()
    loop_thread = threading.Thread(
        target=lambda: (asyncio.set_event_loop(loop), loop_started.set(), loop.run_forever()),
        name="test-ack-event-loop",
        daemon=True,
    )
    loop_thread.start()
    assert loop_started.wait(1.0)
    events = _SchedulerEventBuffer(loop, maxsize=32)
    sent: list[dict[str, object]] = []
    first_ack_sent = threading.Event()
    all_acks_sent = threading.Event()
    policy_entered = threading.Event()
    release_policy = threading.Event()

    async def sender() -> None:
        while True:
            event = await events.get()
            if event is None or event is events.OVERFLOW:
                return
            sent.append(event)
            if len(sent) == 1:
                first_ack_sent.set()
            if len(sent) == 17:
                all_acks_sent.set()

    sender_future = asyncio.run_coroutine_threadsafe(sender(), loop)
    ingress = AudioIngressQueue(max_frames=16)

    def fanout(row, _output, _receipt) -> None:
        events.put_nowait({
            "type": "input_audio_buffer.consumed",
            "frame_id": row.frame.frame_id,
        })
        if row.frame.frame_id == 0:
            # The listener has already queued its ACK; emulate the following
            # synchronous policy callback holding the serialized dispatcher.
            policy_entered.set()
            assert release_policy.wait(2.0)

    dispatcher = SharedListenerTickDispatcher(
        lambda rows: {row.session_id: row.frame.frame_id for row in rows},
        on_result=fanout,
        period_ns=5_000_000,
    )
    ingress.set_on_frame_ready(lambda: dispatcher.notify_input_ready("session"))
    dispatcher.register_session("session", ingress)

    dispatch_thread = threading.Thread(
        target=dispatcher.dispatch_next_tick,
        name="test-stalled-listener-fanout",
        daemon=True,
    )
    try:
        ingress.put(np.zeros(ingress.frame_samples, dtype=np.float32))
        dispatch_thread.start()
        assert policy_entered.wait(1.0)
        assert first_ack_sent.wait(1.0)

        for _ in range(16):
            ingress.put(np.zeros(ingress.frame_samples, dtype=np.float32))

        health_probe = asyncio.run_coroutine_threadsafe(
            asyncio.sleep(0, result={"status": "ok"}), loop
        )
        assert health_probe.result(timeout=0.2) == {"status": "ok"}
        assert ingress.qsize == 16
        assert [event["frame_id"] for event in sent] == [0]

        release_policy.set()
        dispatch_thread.join(1.0)
        assert not dispatch_thread.is_alive()
        for _ in range(16):
            dispatcher.dispatch_next_tick()
        assert all_acks_sent.wait(1.0)
        assert [event["frame_id"] for event in sent] == list(range(17))
        assert ingress.qsize == 0
    finally:
        release_policy.set()
        dispatch_thread.join(1.0)
        events.close()
        sender_future.result(timeout=1.0)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(1.0)
        loop.close()
