from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from duplex_kernel.scheduler import AudioIngressQueue, DuplexScheduler, MonotonicFrameClock, VirtualFrameClock

FRAME_SAMPLES = 1280


def test_first_input_anchor_waits_for_complete_frame_and_anchors_owned_cadence() -> None:
    ingress = AudioIngressQueue()
    clock = VirtualFrameClock(start_ns=0)
    callbacks: list[int] = []
    consumed = threading.Event()

    def listener(frame, _tick) -> None:
        callbacks.append(frame.frame_id)
        consumed.set()

    scheduler = DuplexScheduler(
        ingress,
        clock=clock,
        on_listener=listener,
        anchor_first_input=True,
    )
    result: list[object] = []
    worker = threading.Thread(target=lambda: result.extend(scheduler.run_ticks(1)), daemon=True)
    worker.start()
    ingress.append(np.zeros(FRAME_SAMPLES - 1, dtype=np.float32), capture_start_ns=10)
    assert not consumed.wait(timeout=0.03)
    assert callbacks == []

    ingress.clear()  # An input reset discards the incomplete frame and wakes the waiter.
    ingress.append(np.zeros(FRAME_SAMPLES - 1, dtype=np.float32), capture_start_ns=20)
    assert not consumed.wait(timeout=0.03)
    completed = ingress.append(
        np.zeros(1, dtype=np.float32),
        capture_start_ns=20 + ingress._sample_time_ns(FRAME_SAMPLES - 1),
    )
    assert len(completed) == 1
    assert consumed.wait(timeout=1.0)
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert callbacks == [completed[0].frame_id]
    assert len(result) == 1
    tick = result[0]
    assert tick.enqueue_ns == completed[0].enqueue_ns
    assert tick.deadline_ns == completed[0].enqueue_ns
    assert clock.start_ns == completed[0].enqueue_ns


def test_first_input_anchor_uses_common_peer_phase_and_one_frame_per_80ms_deadline() -> None:
    first_enqueue_ns = 5_000_000_000
    schedulers = []
    for _ in range(2):
        ingress = AudioIngressQueue()
        for index in range(3):
            ingress.put(
                np.zeros(FRAME_SAMPLES, dtype=np.float32),
                frame_id=index,
                capture_start_ns=index * 80_000_000,
                enqueue_ns=first_enqueue_ns,
            )
        scheduler = DuplexScheduler(
            ingress,
            clock=VirtualFrameClock(start_ns=1),
            anchor_first_input=True,
        )
        schedulers.append(scheduler)

    ticks_by_peer = [scheduler.run_ticks(3) for scheduler in schedulers]
    expected_deadlines = [
        first_enqueue_ns,
        first_enqueue_ns + 80_000_000,
        first_enqueue_ns + 160_000_000,
    ]
    for ticks in ticks_by_peer:
        assert [tick.frame_id for tick in ticks] == [0, 1, 2]
        assert [tick.deadline_ns for tick in ticks] == expected_deadlines
    assert ticks_by_peer[0][0].deadline_ns == ticks_by_peer[1][0].deadline_ns


def test_first_input_anchor_waits_for_real_frame_instead_of_returning_empty_tick() -> None:
    ingress = AudioIngressQueue()
    clock = VirtualFrameClock(start_ns=0)
    ingress.put(
        np.zeros(FRAME_SAMPLES, dtype=np.float32),
        frame_id=0,
        capture_start_ns=0,
        enqueue_ns=0,
    )
    callback_frames: list[int] = []

    def listener(frame, _tick) -> None:
        callback_frames.append(frame.frame_id)
        if frame.frame_id == 0:
            clock._now_ns = 160_000_000

    scheduler = DuplexScheduler(
        ingress,
        clock=clock,
        on_listener=listener,
        anchor_first_input=True,
    )
    first = scheduler.tick()
    assert first is not None and first.frame_id == 0

    finished = threading.Event()
    result: list[object] = []

    def run_next_tick() -> None:
        result.append(scheduler.tick())
        finished.set()

    worker = threading.Thread(target=run_next_tick, daemon=True)
    worker.start()
    if not finished.wait(timeout=0.03):
        clock._now_ns = 200_000_000
        next_frame = ingress.put(
            np.zeros(FRAME_SAMPLES, dtype=np.float32),
            frame_id=1,
            capture_start_ns=80_000_000,
            enqueue_ns=200_000_000,
        )
        assert finished.wait(timeout=1.0)
        assert next_frame.frame_id == 1
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert len(result) == 1
    tick = result[0]
    assert tick is not None and tick.frame_id == 1
    assert tick.enqueue_ns == 200_000_000
    assert tick.deadline_ns == 80_000_000
    assert tick.observed_ns == 200_000_000
    assert callback_frames == [0, 1]
    assert scheduler.stats()["missing_ticks"] == 0
    assert scheduler.stats()["skipped_overdue_deadlines"] == 0


@pytest.mark.parametrize("wake_kind", ["stop", "close"])
def test_first_input_anchor_wait_exits_without_empty_callbacks(wake_kind: str) -> None:
    ingress = AudioIngressQueue()
    calls: list[int] = []
    scheduler = DuplexScheduler(
        ingress,
        clock=MonotonicFrameClock(),
        on_listener=lambda frame, _tick: calls.append(frame.frame_id),
        anchor_first_input=True,
    )
    scheduler.start(name=f"test-first-input-anchor-{wake_kind}")
    time.sleep(0.02)
    if wake_kind == "stop":
        scheduler.stop(timeout=1.0)
    else:
        ingress.close()
        deadline = time.monotonic() + 1.0
        while scheduler.is_running and time.monotonic() < deadline:
            time.sleep(0.005)

    assert not scheduler.is_running
    assert scheduler.error is None
    assert calls == []
    assert scheduler.stats()["ticks"] == 0


def test_scheduler_trace_metadata_uses_host_monotonic_residual_components() -> None:
    from duplex_kernel.server import scheduler_tick_trace_metadata

    tick = type("Tick", (), {
        "enqueue_ns": 100_000_000,
        "deadline_ns": 125_000_000,
        "observed_ns": 135_000_000,
        "listener_start_ns": 140_000_000,
        "listener_end_ns": 160_000_000,
    })()
    metadata = scheduler_tick_trace_metadata(tick)

    assert metadata == {
        "scheduler_clock": "time.monotonic_ns",
        "scheduler_enqueue_ns": 100_000_000,
        "scheduler_deadline_ns": 125_000_000,
        "scheduler_observed_ns": 135_000_000,
        "scheduler_listener_start_ns": 140_000_000,
        "scheduler_listener_end_ns": 160_000_000,
        "scheduler_enqueue_to_deadline_ms": 25.0,
        "scheduler_deadline_lateness_ms": 10.0,
        "scheduler_observed_to_listener_start_ms": 5.0,
        "scheduler_listener_callback_ms": 20.0,
    }


def test_scheduler_trace_metadata_marks_listener_end_unmeasured_until_callback_returns() -> None:
    from duplex_kernel.server import scheduler_tick_trace_metadata

    tick = type("Tick", (), {
        "enqueue_ns": None,
        "deadline_ns": 100,
        "observed_ns": 110,
        "listener_start_ns": None,
        "listener_end_ns": None,
    })()
    metadata = scheduler_tick_trace_metadata(tick)

    assert metadata["scheduler_enqueue_ns"] is None
    assert metadata["scheduler_enqueue_to_deadline_ms"] is None
    assert metadata["scheduler_observed_to_listener_start_ms"] is None
    assert metadata["scheduler_listener_callback_ms"] is None


def test_first_input_scheduler_anchor_configuration_is_strict_and_default_off(monkeypatch) -> None:
    from duplex_kernel.config import (
        FIRST_INPUT_SCHEDULER_ANCHOR_ENV,
        _strict_env_flag,
    )

    monkeypatch.delenv(FIRST_INPUT_SCHEDULER_ANCHOR_ENV, raising=False)
    assert _strict_env_flag(FIRST_INPUT_SCHEDULER_ANCHOR_ENV) is False
    monkeypatch.setenv(FIRST_INPUT_SCHEDULER_ANCHOR_ENV, "1")
    assert _strict_env_flag(FIRST_INPUT_SCHEDULER_ANCHOR_ENV) is True
    monkeypatch.setenv(FIRST_INPUT_SCHEDULER_ANCHOR_ENV, "true")
    with pytest.raises(RuntimeError, match="exactly '0' or '1'"):
        _strict_env_flag(FIRST_INPUT_SCHEDULER_ANCHOR_ENV)


def test_first_input_anchor_catches_up_fifo_on_original_grid_then_recovers_backlog() -> None:
    period_ns = 80_000_000
    ingress = AudioIngressQueue(max_frames=32)
    clock = VirtualFrameClock(start_ns=0)
    ingress.put(
        np.zeros(FRAME_SAMPLES, dtype=np.float32),
        frame_id=0,
        capture_start_ns=0,
        enqueue_ns=0,
    )
    next_arrival_id = 1
    next_arrival_ns = period_ns
    callback_durations_ns = [257_000_000, 211_000_000, 211_000_000]
    consumed: list[int] = []
    deadlines: list[int] = []
    observed: list[int] = []

    def listener(frame, tick) -> None:
        nonlocal next_arrival_id, next_arrival_ns
        consumed.append(frame.frame_id)
        deadlines.append(tick.deadline_ns)
        observed.append(tick.observed_ns)
        duration = callback_durations_ns.pop(0) if callback_durations_ns else 40_000_000
        clock._now_ns += duration
        while next_arrival_ns <= clock.now_ns():
            ingress.put(
                np.zeros(FRAME_SAMPLES, dtype=np.float32),
                frame_id=next_arrival_id,
                capture_start_ns=next_arrival_ns,
                enqueue_ns=next_arrival_ns,
            )
            next_arrival_id += 1
            next_arrival_ns += period_ns

    scheduler = DuplexScheduler(
        ingress,
        clock=clock,
        on_listener=listener,
        anchor_first_input=True,
    )

    slow_ticks = [scheduler.tick() for _ in range(3)]

    assert all(tick is not None for tick in slow_ticks)
    assert consumed == [0, 1, 2]
    assert deadlines == [0, period_ns, 2 * period_ns]
    assert observed == [0, 257_000_000, 468_000_000]
    assert [tick.observed_ns - tick.deadline_ns for tick in slow_ticks if tick is not None] == [
        0,
        177_000_000,
        308_000_000,
    ]
    # Slow service is reported as backlog, not hidden by skipping clock slots.
    assert ingress.qsize == 6
    assert scheduler.stats()["queue_depth_frames"] == 6
    assert scheduler.stats()["skipped_overdue_deadlines"] == 0

    # With service below the 80 ms arrival rate, one callback consumes each
    # queued frame and the backlog eventually drains without changing cadence.
    while ingress.qsize:
        tick = scheduler.tick()
        assert tick is not None
    assert consumed == list(range(next_arrival_id - ingress.qsize))
    assert deadlines == [index * period_ns for index in range(len(deadlines))]
    assert scheduler.stats()["skipped_overdue_deadlines"] == 0

    # Empty ingress waits for an actual frame; it does not create idle ticks.
    next_frame_id = next_arrival_id
    next_deadline_ns = len(deadlines) * period_ns
    finished = threading.Event()
    result: list[object] = []
    worker = threading.Thread(
        target=lambda: (result.append(scheduler.tick()), finished.set()), daemon=True
    )
    worker.start()
    assert not finished.wait(timeout=0.03)
    enqueue_ns = clock.now_ns()
    next_frame_start_ns = next_arrival_ns
    next_arrival_id += 1
    next_arrival_ns += period_ns
    ingress.put(
        np.zeros(FRAME_SAMPLES, dtype=np.float32),
        frame_id=next_frame_id,
        capture_start_ns=next_frame_start_ns,
        enqueue_ns=enqueue_ns,
    )
    assert finished.wait(timeout=1.0)
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert len(result) == 1
    next_tick = result[0]
    assert next_tick is not None
    assert next_tick.frame_id == next_frame_id
    assert next_tick.deadline_ns == next_deadline_ns
    assert next_tick.observed_ns >= enqueue_ns
    assert consumed[-1] == next_frame_id


def test_first_input_anchor_stop_wins_before_dequeue_admission() -> None:
    ingress = AudioIngressQueue()
    ingress.put(np.zeros(FRAME_SAMPLES, dtype=np.float32), enqueue_ns=time.monotonic_ns())
    original_wait = ingress.wait_for_frame_or_stop
    wait_returned = threading.Event()
    release_wait = threading.Event()
    calls: list[int] = []

    def gated_wait(stop_event: threading.Event) -> bool:
        ready = original_wait(stop_event)
        wait_returned.set()
        assert release_wait.wait(timeout=1.0)
        return ready

    ingress.wait_for_frame_or_stop = gated_wait  # type: ignore[method-assign]
    scheduler = DuplexScheduler(
        ingress,
        clock=MonotonicFrameClock(),
        on_listener=lambda frame, _tick: calls.append(frame.frame_id),
        anchor_first_input=True,
    )
    scheduler.start(name="test-first-input-anchor-stop-race")
    assert wait_returned.wait(timeout=1.0)
    scheduler.stop(timeout=0.0)
    release_wait.set()
    deadline = time.monotonic() + 1.0
    while scheduler.is_running and time.monotonic() < deadline:
        time.sleep(0.005)

    assert not scheduler.is_running
    assert calls == []
    assert scheduler.stats()["ticks"] == 0
    assert ingress.qsize == 1


def test_first_input_anchor_restart_resets_virtual_tick_index_only_when_opted_in() -> None:
    ingress = AudioIngressQueue()
    clock = VirtualFrameClock(start_ns=0)
    second_restart_frame = threading.Event()
    observed: list[tuple[int, int, int]] = []
    scheduler: DuplexScheduler

    def listener(frame, tick) -> None:
        observed.append((frame.frame_id, tick.deadline_ns, tick.observed_ns))
        if frame.frame_id == 2:
            second_restart_frame.set()
            scheduler.stop(timeout=0.0)

    scheduler = DuplexScheduler(
        ingress,
        clock=clock,
        on_listener=listener,
        anchor_first_input=True,
    )
    ingress.put(np.zeros(FRAME_SAMPLES, dtype=np.float32), enqueue_ns=100_000_000)
    first = scheduler.run_ticks(1)
    assert first[0].deadline_ns == 100_000_000
    assert scheduler._next_tick_index == 1

    # Restart has an old logical index, but the first complete frame starts a
    # new owned phase. Prequeue both frames so the second remains on the future
    # 80 ms grid instead of being drained at the first dispatch.
    ingress.put(np.zeros(FRAME_SAMPLES, dtype=np.float32), enqueue_ns=200_000_000)
    ingress.put(np.zeros(FRAME_SAMPLES, dtype=np.float32), enqueue_ns=280_000_000)
    scheduler.start(name="test-first-input-anchor-virtual-restart")
    assert second_restart_frame.wait(timeout=1.0)
    scheduler.stop(timeout=1.0)

    assert not scheduler.is_running
    assert observed == [
        (0, 100_000_000, 100_000_000),
        (1, 200_000_000, 200_000_000),
        (2, 280_000_000, 280_000_000),
    ]


def test_first_input_anchor_stop_during_later_deadline_wait_does_not_admit_next_frame() -> None:
    wait_entered = threading.Event()
    release_wait = threading.Event()

    class BlockingVirtualFrameClock(VirtualFrameClock):
        def wait_until(self, deadline_ns: int) -> None:
            wait_entered.set()
            assert release_wait.wait(timeout=1.0)
            super().wait_until(deadline_ns)

    ingress = AudioIngressQueue()
    clock = BlockingVirtualFrameClock()
    first_callback = threading.Event()
    calls: list[int] = []

    def listener(frame, _tick) -> None:
        calls.append(frame.frame_id)
        first_callback.set()

    scheduler = DuplexScheduler(
        ingress,
        clock=clock,
        on_listener=listener,
        anchor_first_input=True,
    )
    scheduler.start(name="test-first-input-anchor-stop-later-wait")
    ingress.put(np.zeros(FRAME_SAMPLES, dtype=np.float32), enqueue_ns=10_000_000)
    assert first_callback.wait(timeout=1.0)
    queued = ingress.put(
        np.zeros(FRAME_SAMPLES, dtype=np.float32),
        frame_id=1,
        enqueue_ns=90_000_000,
    )
    assert wait_entered.wait(timeout=1.0)

    scheduler.stop(timeout=0.0)
    release_wait.set()
    scheduler.stop(timeout=1.0)

    assert not scheduler.is_running
    assert calls == [0]
    assert scheduler.stats()["ticks"] == 1
    assert ingress.qsize == 1
    assert ingress.get_nowait().frame_id == queued.frame_id
