from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager
import faulthandler
import threading
import queue

import numpy as np
import pytest

from duplex_kernel.scheduler import AudioFrame, AudioIngressQueue, DuplexScheduler, DuplexTick
from duplex_kernel.shared_listener_scheduler import (
    SharedListenerTickDispatcher,
)


class _FakeIngress:
    def __init__(self, *frames: AudioFrame) -> None:
        self.frames = deque(frames)

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
    assert [item[:3] for item in delivered] == [
        ("a", 3, "a"),
        ("b", 8, "b"),
    ]
    assert all(item[3] == 0 for item in delivered)
    assert all(row.tick.dispatch_mode == "shared_fixed_80ms_microbatch" for row in receipt.rows)
    assert all(row.tick.listener_batch_size == 2 for row in receipt.rows)


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


def test_shared_context_spans_batch_future_and_all_row_fanout() -> None:
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

    assert phases == ["enter", "batch", "fanout", "fanout", "exit"]


def test_row_fanout_failure_poison_stops_remaining_commits() -> None:
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

    assert delivered == ["a"]
    assert errors == ["a", "b"]
    assert isinstance(dispatcher.fatal_error, RuntimeError)


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
