from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from duplex_kernel.listener import FairModelLock, capture_host_timeline_nonblocking


def test_default_model_lock_timeline_is_off_without_clock_reads(monkeypatch) -> None:
    lock = FairModelLock()
    monkeypatch.setattr(
        "duplex_kernel.listener.time.monotonic_ns",
        lambda: (_ for _ in ()).throw(AssertionError("timeline clock read while disabled")),
    )

    assert lock.acquire_fair("compute")
    lock.release()
    assert lock.acquire(blocking=False)
    lock.release()
    snapshot = lock.host_timeline_snapshot()
    assert snapshot["enabled"] is False
    assert snapshot["status"] == "not_requested"
    assert snapshot["events"] == []


def test_enabled_timeline_records_fair_request_grant_and_nested_release(monkeypatch) -> None:
    stamps = iter((100, 200, 300, 400, 500, 600))
    monkeypatch.setattr("duplex_kernel.listener.time.monotonic_ns", lambda: next(stamps))
    lock = FairModelLock(host_timeline_enabled=True)

    assert lock.acquire_fair("listener")
    assert lock.acquire_fair("listener")
    lock.release()
    lock.release()

    snapshot = lock.host_timeline_snapshot()
    assert snapshot["enabled"] is True
    assert snapshot["status"] == "complete"
    assert snapshot["scope"] == "process_shared_model_lock"
    assert snapshot["clock"] == "time.monotonic_ns"
    assert snapshot["process_id"] > 0
    assert snapshot["snapshot_monotonic_ns"] == 600
    assert snapshot["dedupe_key"] == ["container_identity", "process_id", "trace_seq"]
    assert "Keep unmatched or overlapping matches global/ambiguous" in snapshot["join_strategy"]
    assert snapshot["capacity"] == 4096
    assert snapshot["ring_capacity"] == 2048
    assert snapshot["retained_event_count"] == 5
    assert snapshot["dropped_event_count"] == 0
    assert [row["event"] for row in snapshot["events"]] == [
        "request", "acquire", "reentrant_acquire", "release", "release",
    ]
    request, acquired, nested, nested_release, final_release = snapshot["events"]
    assert request["timestamp_ns"] < acquired["timestamp_ns"]
    assert acquired["wait_ns"] == 100
    assert nested["ticket"] == acquired["ticket"] == request["ticket"]
    assert nested["priority"] == "listener"
    assert nested_release["fully_released"] is False
    assert final_release["fully_released"] is True
    assert all(row["thread_id"] == threading.get_ident() for row in snapshot["events"])
    assert [row["trace_seq"] for row in snapshot["events"]] == list(range(5))
    assert all("pending_counts" in row for row in snapshot["events"])


def test_enabled_timeline_bounded_ring_reports_dropped_events_truthfully(monkeypatch) -> None:
    tick = iter(range(100, 200))
    monkeypatch.setattr("duplex_kernel.listener.time.monotonic_ns", lambda: next(tick))
    lock = FairModelLock(host_timeline_enabled=True, host_timeline_capacity=4)

    for _ in range(5):
        assert lock.acquire_fair("compute")
        lock.release()

    snapshot = lock.host_timeline_snapshot()
    assert snapshot["status"] == "truncated"
    assert snapshot["capacity"] == 8
    assert snapshot["ring_capacity"] == 4
    assert snapshot["total_event_count"] == 15
    assert snapshot["retained_event_count"] == 8
    assert snapshot["dropped_event_count"] == 7
    assert snapshot["initial_ring_retained_count"] == 4
    assert snapshot["tail_ring_retained_count"] == 4
    assert snapshot["tail_ring_overwritten_event_count"] == 11
    assert snapshot["overlap_deduplicated_event_count"] == 0
    assert [row["trace_seq"] for row in snapshot["events"]] == [0, 1, 2, 3, 11, 12, 13, 14]
    assert [row["event"] for row in snapshot["events"]] == [
        "request", "acquire", "release", "request",
        "release", "request", "acquire", "release",
    ]


def test_timeline_two_rings_deduplicate_overlap_and_keep_all_events_until_gap(monkeypatch) -> None:
    tick = iter(range(100, 200))
    monkeypatch.setattr("duplex_kernel.listener.time.monotonic_ns", lambda: next(tick))
    lock = FairModelLock(host_timeline_enabled=True, host_timeline_capacity=4)

    for _ in range(2):
        assert lock.acquire_fair("compute")
        lock.release()

    snapshot = lock.host_timeline_snapshot()
    assert snapshot["status"] == "complete"
    assert snapshot["total_event_count"] == 6
    assert snapshot["retained_event_count"] == 6
    assert snapshot["dropped_event_count"] == 0
    assert snapshot["overlap_deduplicated_event_count"] == 2
    assert [row["trace_seq"] for row in snapshot["events"]] == list(range(6))


def test_enabled_timeline_joins_concurrent_waiter_ticket_to_grant() -> None:
    lock = FairModelLock(host_timeline_enabled=True)
    assert lock.acquire_fair("compute")
    acquired = threading.Event()

    def listener_waiter() -> None:
        assert lock.acquire_fair("listener")
        acquired.set()
        lock.release()

    worker = threading.Thread(target=listener_waiter)
    worker.start()
    deadline = __import__("time").monotonic() + 2
    while lock.pending_counts["listener"] != 1 and __import__("time").monotonic() < deadline:
        __import__("time").sleep(0.001)
    assert lock.pending_counts["listener"] == 1
    lock.release()
    assert acquired.wait(2)
    worker.join(2)
    assert not worker.is_alive()

    events = lock.host_timeline_snapshot()["events"]
    requests = [row for row in events if row["event"] == "request"]
    grants = [row for row in events if row["event"] == "acquire"]
    listener_request = next(row for row in requests if row["priority"] == "listener")
    listener_grant = next(row for row in grants if row["priority"] == "listener")
    assert listener_request["ticket"] == listener_grant["ticket"]
    assert listener_request["thread_id"] == listener_grant["thread_id"]
    assert listener_grant["wait_ns"] >= 0


def test_timeline_rejects_nonpositive_capacity() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        FairModelLock(host_timeline_enabled=True, host_timeline_capacity=0)


def test_timeline_can_be_enabled_for_shared_batch_on_idle_lock() -> None:
    lock = FairModelLock()
    lock.configure_host_timeline(enabled=True, capacity=4)

    assert lock.host_timeline_enabled is True
    assert lock.acquire_fair("listener")
    lock.release()
    snapshot = lock.host_timeline_snapshot()
    assert snapshot["ring_capacity"] == 4
    assert snapshot["total_event_count"] == 3
    assert [event["event"] for event in snapshot["events"]] == [
        "request", "acquire", "release",
    ]

    lock.configure_host_timeline(enabled=False, capacity=4)
    assert lock.host_timeline_enabled is False
    assert lock.host_timeline_snapshot()["status"] == "not_requested"


def test_timeline_cannot_be_reconfigured_while_lock_has_an_owner() -> None:
    lock = FairModelLock()
    assert lock.acquire_fair("compute")
    try:
        with pytest.raises(RuntimeError, match="while the lock is active"):
            lock.configure_host_timeline(enabled=True, capacity=4)
    finally:
        lock.release()
    assert lock.host_timeline_enabled is False


def test_same_timeline_configuration_is_idempotent_while_lock_has_an_owner() -> None:
    lock = FairModelLock(host_timeline_enabled=True, host_timeline_capacity=4)
    assert lock.acquire_fair("listener")
    try:
        lock.configure_host_timeline(enabled=True, capacity=4)
        assert lock.host_timeline_enabled is True
    finally:
        lock.release()


def test_nonblocking_host_timeline_omits_busy_locks_and_callback_continues_promptly() -> None:
    capture_lock = threading.Lock()
    model_lock = FairModelLock(host_timeline_enabled=True, host_timeline_capacity=8)
    assert model_lock.acquire_fair("compute")
    model_lock.release()

    locks_held = threading.Event()
    release_capture = threading.Event()
    capture_released = threading.Event()
    release_condition = threading.Event()

    def hold_both_diagnostic_locks() -> None:
        capture_lock.acquire()
        model_lock._condition.acquire()
        locks_held.set()
        release_capture.wait(2)
        capture_lock.release()
        capture_released.set()
        release_condition.wait(2)
        model_lock._condition.release()

    holder = threading.Thread(target=hold_both_diagnostic_locks)
    holder.start()
    assert locks_held.wait(1)
    batch_state: dict[str, object] = {}

    started = time.perf_counter()
    capture_busy = capture_host_timeline_nonblocking(
        capture_lock,
        model_lock,
        batch_id="batch-capture-busy",
        batch_size=2,
        batch_state=batch_state,
    )
    callback_continued_after_capture_busy = True
    assert time.perf_counter() - started < 0.1
    assert callback_continued_after_capture_busy is True
    assert capture_busy["status"] == "omitted_busy"
    assert capture_busy["reason"] == "timeline_capture_lock_busy"
    assert isinstance(capture_busy["request_monotonic_ns"], int)
    assert capture_busy["snapshot_monotonic_ns"] is None
    assert capture_busy["events"] == []

    release_capture.set()
    assert capture_released.wait(1)

    started = time.perf_counter()
    model_lock_busy = capture_host_timeline_nonblocking(
        capture_lock,
        model_lock,
        batch_id="batch-model-lock-busy",
        batch_size=2,
        batch_state=batch_state,
    )
    callback_continued_after_model_busy = True
    assert time.perf_counter() - started < 0.1
    assert callback_continued_after_model_busy is True
    assert model_lock_busy["status"] == "omitted_busy"
    assert model_lock_busy["reason"] == "fair_model_lock_condition_busy"
    assert isinstance(model_lock_busy["request_monotonic_ns"], int)
    assert model_lock_busy["capture_lock_acquired_monotonic_ns"] is not None
    assert model_lock_busy["model_lock_acquired_monotonic_ns"] is None
    assert model_lock_busy["snapshot_monotonic_ns"] is None
    assert model_lock_busy["events"] == []
    assert "current_owner" not in model_lock_busy

    release_condition.set()
    holder.join(1)
    assert not holder.is_alive()

    unlocked = capture_host_timeline_nonblocking(
        capture_lock,
        model_lock,
        batch_id="batch-unlocked",
        batch_size=2,
        batch_state=batch_state,
    )
    assert unlocked["status"] == "complete"
    assert unlocked["request_monotonic_ns"] <= unlocked["snapshot_monotonic_ns"]
    assert unlocked["capture_lock_acquired_monotonic_ns"] is not None
    assert unlocked["model_lock_acquired_monotonic_ns"] is not None
    assert unlocked["age_ms"] >= 0
    assert unlocked["events"]


def test_nonblocking_host_timeline_copies_only_bounded_tail(monkeypatch) -> None:
    lock = FairModelLock(host_timeline_enabled=True, host_timeline_capacity=256)
    for _ in range(100):
        assert lock.acquire_fair("listener")
        lock.release()

    # The listener callback must not enter the full-history snapshot path.
    monkeypatch.setattr(
        lock,
        "host_timeline_snapshot",
        lambda: pytest.fail("full host timeline snapshot entered from callback"),
    )
    snapshot = lock.host_timeline_snapshot_nonblocking()

    assert snapshot["status"] == "truncated"
    assert snapshot["snapshot_event_limit"] == 64
    assert snapshot["retained_event_count"] == 64
    assert snapshot["snapshot_omitted_event_count"] == 236
    assert [row["trace_seq"] for row in snapshot["events"]] == list(range(236, 300))


def test_audio_tower_lock_timeline_follows_existing_cuda_profile_flag(monkeypatch) -> None:
    import torch

    from duplex_kernel import config
    from duplex_kernel.listener import GemmaAudioTowerBackend

    class AudioTower(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class Gemma:
        def __init__(self) -> None:
            self.audio_tower = AudioTower()
            self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))

        def get_audio_features(self, **_kwargs):
            raise NotImplementedError

    wrapper = SimpleNamespace(
        model=SimpleNamespace(model=Gemma()),
        processor=lambda **_kwargs: None,
    )

    monkeypatch.setattr(config, "NATIVE_LISTENER_CUDA_PROFILE_ENABLED", False)
    off = GemmaAudioTowerBackend(wrapper, use_native_lm=False)
    assert off._lock.host_timeline_enabled is False
    monkeypatch.setattr(config, "NATIVE_LISTENER_CUDA_PROFILE_ENABLED", True)
    on = GemmaAudioTowerBackend(wrapper, use_native_lm=False)
    assert on._lock.host_timeline_enabled is True
