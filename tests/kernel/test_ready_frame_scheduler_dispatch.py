from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from duplex_kernel.scheduler import (
    AudioIngressQueue,
    DuplexScheduler,
    MonotonicFrameClock,
    VirtualFrameClock,
)


FRAME_SAMPLES = 1280
FRAME_NS = 80_000_000


def _put_frames(ingress: AudioIngressQueue, count: int) -> list[object]:
    return [
        ingress.put(
            np.zeros(FRAME_SAMPLES, dtype=np.float32),
            frame_id=index,
            capture_start_ns=index * FRAME_NS,
            enqueue_ns=index * FRAME_NS,
        )
        for index in range(count)
    ]


def test_ready_frame_dispatch_runs_ready_frame_without_waiting_for_next_grid_slot() -> None:
    ingress = AudioIngressQueue()
    clock = VirtualFrameClock(start_ns=0)
    clock._now_ns = 78_000_000
    frame = ingress.put(
        np.zeros(FRAME_SAMPLES, dtype=np.float32),
        frame_id=9,
        capture_start_ns=0,
        enqueue_ns=78_000_000,
    )
    callback_observed: list[int] = []

    scheduler = DuplexScheduler(
        ingress,
        clock=clock,
        on_listener=lambda _frame, tick: callback_observed.append(tick.listener_start_ns),
        ready_frame_dispatch=True,
    )

    tick = scheduler.tick()

    assert tick is not None
    assert tick.frame_id == frame.frame_id
    assert tick.deadline_ns == frame.capture_end_ns
    assert tick.observed_ns == 78_000_000
    assert tick.listener_start_ns == 78_000_000
    assert callback_observed == [78_000_000]
    from duplex_kernel.server import scheduler_tick_trace_metadata
    timing = scheduler_tick_trace_metadata(tick)
    assert timing["scheduler_deadline_lateness_ms"] == -2.0
    assert timing["scheduler_dispatch_mode"] == "ready_frame_no_fixed_grid"
    tick.listener_batch_id = "batch-metrics"
    tick.microbatch_mel_prepare_ms = 1.25
    tick.microbatch_frontend_copy_ms = 0.75
    tick.microbatch_frontend_copy_cuda_ms = 0.5
    tick.microbatch_audio_tower_lock_wait_ms = 0.1
    tick.microbatch_audio_tower_lock_hold_ms = 2.0
    tick.microbatch_audio_tower_workspace_lock_wait_ms = 0.2
    tick.microbatch_audio_tower_workspace_lock_hold_ms = 0.9
    tick.microbatch_audio_tower_row_copy_cuda_ms = 0.25
    tick.microbatch_audio_tower_graph_submit_ms = 0.75
    tick.microbatch_audio_tower_cuda_ms = 1.5
    tick.microbatch_audio_tower_batch_cuda_span_ms = 2.5
    tick.microbatch_audio_tower_execution_mode = "cuda_graph_n1"
    tick.microbatch_native_cuda_ms = 3.5
    tick.microbatch_semantic_prefix_request_ms = 0.2
    tick.microbatch_semantic_prefix_build_ms = 1.1
    tick.microbatch_semantic_prefix_replay_ms = 2.2
    tick.microbatch_semantic_prefix_import_ms = 3.3
    tick.microbatch_semantic_prefix_refresh_event = True
    tick.microbatch_semantic_prefix_refresh_event_id = "manifest:session:epoch:9:9"
    tick.microbatch_semantic_prefix_refresh_revision = 12
    tick.microbatch_semantic_prefix_len = 141
    tick.microbatch_native_control6_raw_score_ms = 4.4
    tick.microbatch_native_output_health_ms = 0.6
    tick.microbatch_native_output_health_checked = True
    tick.microbatch_native_output_health_status = "healthy"
    tick.microbatch_native_output_health_nonfinite = False
    tick.microbatch_native_output_health_generation = 7
    tick.microbatch_native_output_health_row_frame_id = 9
    tick.microbatch_native_output_health_row_state_version = 11
    timing = scheduler_tick_trace_metadata(tick)
    assert timing["listener_batch_mel_prepare_ms"] == 1.25
    assert timing["listener_batch_frontend_copy_ms"] == 0.75
    assert timing["listener_batch_frontend_copy_cuda_ms"] == 0.5
    assert timing["listener_batch_audio_tower_lock_wait_ms"] == 0.1
    assert timing["listener_batch_audio_tower_lock_hold_ms"] == 2.0
    assert timing["listener_batch_audio_tower_workspace_lock_wait_ms"] == 0.2
    assert timing["listener_batch_audio_tower_workspace_lock_hold_ms"] == 0.9
    assert timing["listener_batch_audio_tower_row_copy_cuda_ms"] == 0.25
    assert timing["listener_batch_audio_tower_graph_submit_ms"] == 0.75
    assert timing["listener_batch_audio_tower_cuda_ms"] == 1.5
    assert timing["listener_batch_audio_tower_batch_cuda_span_ms"] == 2.5
    assert timing["listener_batch_audio_tower_execution_mode"] == "cuda_graph_n1"
    assert timing["listener_batch_native_cuda_ms"] == 3.5
    assert timing["listener_batch_semantic_prefix_request_ms"] == 0.2
    assert timing["listener_batch_semantic_prefix_build_ms"] == 1.1
    assert timing["listener_batch_semantic_prefix_replay_ms"] == 2.2
    assert timing["listener_batch_semantic_prefix_import_ms"] == 3.3
    assert timing["listener_batch_semantic_prefix_refresh_event"] is True
    assert timing["listener_batch_semantic_prefix_refresh_event_id"] == "manifest:session:epoch:9:9"
    assert timing["listener_batch_semantic_prefix_refresh_revision"] == 12
    assert timing["listener_batch_semantic_prefix_len"] == 141
    assert timing["listener_batch_native_control6_raw_score_ms"] == 4.4
    assert timing["listener_batch_native_output_health_ms"] == 0.6
    assert timing["listener_batch_native_output_health_checked"] is True
    assert timing["listener_batch_native_output_health_status"] == "healthy"
    assert timing["listener_batch_native_output_health_nonfinite"] is False
    assert timing["listener_batch_native_output_health_generation"] == 7
    assert timing["listener_batch_native_output_health_row_frame_id"] == 9
    assert timing["listener_batch_native_output_health_row_state_version"] == 11
    assert scheduler.stats()["ticks"] == 1
    assert scheduler.stats()["missing_ticks"] == 0


def test_scheduler_trace_projects_one_bounded_host_timeline_without_cuda_profile() -> None:
    from types import SimpleNamespace

    from duplex_kernel.server import scheduler_tick_trace_metadata

    timeline = {
        "enabled": True,
        "status": "complete",
        "clock": "time.monotonic_ns",
        "batch_id": "batch-host-lock-1",
        "capacity": 256,
        "events": [{"trace_seq": 19, "event": "release", "hold_ns": 239_000_000}],
    }
    tick = SimpleNamespace(
        deadline_ns=1_000_000,
        listener_batch_id="batch-host-lock-1",
        listener_batch_size=2,
        microbatch_host_lock_timeline=timeline,
        microbatch_row_callback_entry_ns=1_100_000,
        microbatch_timeline_request_ns=1_200_000,
        microbatch_timeline_capture_lock_acquired_ns=1_210_000,
        microbatch_timeline_model_lock_acquired_ns=1_220_000,
        microbatch_timeline_snapshot_ns=1_230_000,
        microbatch_timeline_age_ms=0.03,
        microbatch_timeline_status="complete",
        microbatch_timeline_reason=None,
        scheduler_dispatch_start_ns=1_300_000,
        scheduler_dispatch_end_ns=1_400_000,
        microbatch_commit_start_ns=1_310_000,
        microbatch_commit_end_ns=1_350_000,
        microbatch_commit_state_lock_acquired_ns=1_320_000,
        microbatch_commit_state_lock_released_ns=1_340_000,
        microbatch_commit_state_lock_wait_ms=0.01,
        microbatch_commit_state_lock_hold_ms=0.02,
        policy_decision_at_ns=2_000_000,
    )

    # The host timeline is an explicit batch trace field and does not depend on
    # the global CUDA profiler switch.
    ordinary = scheduler_tick_trace_metadata(tick)
    with_timeline = scheduler_tick_trace_metadata(tick, include_host_lock_timeline=True)

    assert "listener_batch_host_lock_timeline" not in ordinary
    assert with_timeline["listener_batch_host_lock_timeline"] == timeline
    assert with_timeline["scheduler_row_callback_entry_ns"] == 1_100_000
    assert with_timeline["scheduler_timeline_request_ns"] == 1_200_000
    assert with_timeline["scheduler_timeline_capture_lock_acquired_ns"] == 1_210_000
    assert with_timeline["scheduler_timeline_model_lock_acquired_ns"] == 1_220_000
    assert with_timeline["scheduler_timeline_snapshot_ns"] == 1_230_000
    assert with_timeline["scheduler_timeline_age_ms"] == 0.03
    assert with_timeline["scheduler_timeline_status"] == "complete"
    assert with_timeline["scheduler_dispatch_start_ns"] == 1_300_000
    assert with_timeline["scheduler_dispatch_end_ns"] == 1_400_000
    assert with_timeline["scheduler_commit_microbatch_output_start_ns"] == 1_310_000
    assert with_timeline["scheduler_commit_microbatch_output_end_ns"] == 1_350_000
    assert with_timeline["scheduler_commit_state_lock_wait_ms"] == 0.01
    assert with_timeline["scheduler_commit_state_lock_hold_ms"] == 0.02
    assert with_timeline["scheduler_policy_decision_at_ns"] == 2_000_000
    assert with_timeline["scheduler_deadline_to_policy_decision_ms"] == 1.0


def test_ready_dispatch_waits_for_a_complete_frame_and_dispatches_fifo_without_empty_ticks() -> None:
    ingress = AudioIngressQueue(max_frames=8)
    calls: list[int] = []
    scheduler = DuplexScheduler(
        ingress,
        clock=MonotonicFrameClock(),
        on_listener=lambda frame, _tick: calls.append(frame.frame_id),
        ready_frame_dispatch=True,
    )
    scheduler.start(name="ready-dispatch-partial-frame")
    try:
        ingress.append(np.zeros(FRAME_SAMPLES - 1, dtype=np.float32), capture_start_ns=100)
        time.sleep(0.03)
        assert calls == []
        assert scheduler.stats()["ticks"] == 0

        completed = ingress.append(
            np.zeros(1, dtype=np.float32),
            capture_start_ns=100 + ingress._sample_time_ns(FRAME_SAMPLES - 1),
        )
        assert len(completed) == 1
        ingress.put(
            np.zeros(FRAME_SAMPLES, dtype=np.float32),
            frame_id=completed[0].frame_id + 1,
            capture_start_ns=completed[0].capture_end_ns,
        )
        deadline = time.monotonic() + 1.0
        while len(calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)

        assert calls == [completed[0].frame_id, completed[0].frame_id + 1]
        assert scheduler.stats()["ticks"] == 2
        assert scheduler.stats()["missing_ticks"] == 0
        assert ingress.qsize == 0
    finally:
        scheduler.stop(timeout=1.0)


def test_ready_dispatch_delivers_backlog_without_grid_wait_and_reports_actual_lateness() -> None:
    ingress = AudioIngressQueue(max_frames=8)
    _put_frames(ingress, 4)
    clock = VirtualFrameClock(start_ns=0)
    callback_observed: list[int] = []

    def slow_listener(_frame, _tick) -> None:
        callback_observed.append(clock.now_ns())
        clock._now_ns += 257_000_000 if len(callback_observed) == 1 else 211_000_000

    scheduler = DuplexScheduler(
        ingress,
        clock=clock,
        on_listener=slow_listener,
        ready_frame_dispatch=True,
    )
    ticks = scheduler.run_ticks(4)

    assert [tick.frame_id for tick in ticks] == [0, 1, 2, 3]
    assert callback_observed == [0, 257_000_000, 468_000_000, 679_000_000]
    assert [tick.deadline_ns for tick in ticks] == [80_000_000, 160_000_000, 240_000_000, 320_000_000]
    assert [tick.observed_ns for tick in ticks] == callback_observed
    assert [
        int(tick.observed_ns) - int(tick.enqueue_ns)
        for tick in ticks
    ] == [0, 177_000_000, 308_000_000, 439_000_000]
    assert all(tick.dispatch_mode == "ready_frame_no_fixed_grid" for tick in ticks)
    assert ingress.qsize == 0
    assert scheduler.stats()["missing_ticks"] == 0
    assert scheduler.stats()["input_frames_received"] == 4


@pytest.mark.parametrize("wake_kind", ["stop", "close"])
def test_ready_dispatch_empty_wait_exits_without_callbacks(wake_kind: str) -> None:
    ingress = AudioIngressQueue()
    calls: list[int] = []
    scheduler = DuplexScheduler(
        ingress,
        clock=MonotonicFrameClock(),
        on_listener=lambda frame, _tick: calls.append(frame.frame_id),
        ready_frame_dispatch=True,
    )
    scheduler.start(name=f"ready-dispatch-empty-{wake_kind}")
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
    assert scheduler.stats()["missing_ticks"] == 0


def test_ready_dispatch_stop_during_later_empty_wait_does_not_make_an_idle_tick() -> None:
    ingress = AudioIngressQueue()
    calls: list[int] = []
    first_call = threading.Event()
    scheduler = DuplexScheduler(
        ingress,
        clock=MonotonicFrameClock(),
        on_listener=lambda frame, _tick: (calls.append(frame.frame_id), first_call.set()),
        ready_frame_dispatch=True,
    )
    scheduler.start(name="ready-dispatch-later-empty-wait")
    ingress.put(np.zeros(FRAME_SAMPLES, dtype=np.float32), frame_id=4)
    assert first_call.wait(timeout=1.0)
    scheduler.stop(timeout=1.0)

    assert calls == [4]
    assert scheduler.stats()["ticks"] == 1
    assert scheduler.stats()["missing_ticks"] == 0
    assert scheduler.error is None


def test_ready_dispatch_stop_wins_before_dequeue_and_leaves_ready_frame_queued() -> None:
    ingress = AudioIngressQueue()
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
        ready_frame_dispatch=True,
    )
    scheduler.start(name="ready-dispatch-stop-race")
    try:
        ingress.put(np.zeros(FRAME_SAMPLES, dtype=np.float32), frame_id=23)
        assert wait_returned.wait(timeout=1.0)
        scheduler.stop(timeout=0.0)
    finally:
        release_wait.set()
        scheduler.stop(timeout=1.0)

    assert not scheduler.is_running
    assert calls == []
    assert scheduler.stats()["ticks"] == 0
    assert ingress.qsize == 1
    assert ingress.get_nowait().frame_id == 23


def test_ready_dispatch_restart_resets_logical_tick_index_without_retrograde_clock() -> None:
    ingress = AudioIngressQueue()
    clock = VirtualFrameClock(start_ns=0)
    scheduler = DuplexScheduler(
        ingress,
        clock=clock,
        ready_frame_dispatch=True,
    )
    ingress.put(np.zeros(FRAME_SAMPLES, dtype=np.float32), frame_id=0, capture_start_ns=0)
    assert scheduler.run_ticks(1)[0].deadline_ns == FRAME_NS
    old_now = clock.now_ns()

    scheduler.start(name="ready-dispatch-virtual-restart")
    ingress.put(
        np.zeros(FRAME_SAMPLES, dtype=np.float32),
        frame_id=1,
        capture_start_ns=FRAME_NS,
        enqueue_ns=FRAME_NS,
    )
    deadline = time.monotonic() + 1.0
    while scheduler.stats()["ticks"] < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    scheduler.stop(timeout=1.0)

    assert scheduler.error is None
    assert scheduler.stats()["ticks"] == 2
    assert scheduler._next_tick_index == 1
    assert clock.now_ns() >= old_now


def test_ready_dispatch_restarts_cleanly_with_monotonic_clock() -> None:
    ingress = AudioIngressQueue()
    calls: list[int] = []
    called = threading.Event()
    scheduler = DuplexScheduler(
        ingress,
        clock=MonotonicFrameClock(),
        on_listener=lambda frame, _tick: (calls.append(frame.frame_id), called.set()),
        ready_frame_dispatch=True,
    )
    for frame_id in (5, 6):
        scheduler.start(name=f"ready-dispatch-monotonic-restart-{frame_id}")
        ingress.put(np.zeros(FRAME_SAMPLES, dtype=np.float32), frame_id=frame_id)
        assert called.wait(timeout=1.0)
        scheduler.stop(timeout=1.0)
        called.clear()

    assert calls == [5, 6]
    assert scheduler.stats()["ticks"] == 2
    assert scheduler.stats()["missing_ticks"] == 0
    assert scheduler.error is None


def test_ready_dispatch_is_mutually_exclusive_with_first_input_anchor() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        DuplexScheduler(
            AudioIngressQueue(),
            anchor_first_input=True,
            ready_frame_dispatch=True,
        )


def test_ready_dispatch_environment_is_strict_and_mutually_exclusive() -> None:
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["GEMMA4_OMNI_FIRST_INPUT_SCHEDULER_ANCHOR"] = "1"
    env["GEMMA4_OMNI_READY_FRAME_SCHEDULER_DISPATCH"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", "import duplex_kernel.config"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


def test_ready_dispatch_environment_flag_uses_strict_default_off_parser(monkeypatch) -> None:
    from duplex_kernel.config import (
        READY_FRAME_SCHEDULER_DISPATCH_ENV,
        _strict_env_flag,
    )

    monkeypatch.delenv(READY_FRAME_SCHEDULER_DISPATCH_ENV, raising=False)
    assert _strict_env_flag(READY_FRAME_SCHEDULER_DISPATCH_ENV) is False
    monkeypatch.setenv(READY_FRAME_SCHEDULER_DISPATCH_ENV, "1")
    assert _strict_env_flag(READY_FRAME_SCHEDULER_DISPATCH_ENV) is True
    monkeypatch.setenv(READY_FRAME_SCHEDULER_DISPATCH_ENV, "true")
    with pytest.raises(RuntimeError, match="exactly '0' or '1'"):
        _strict_env_flag(READY_FRAME_SCHEDULER_DISPATCH_ENV)
