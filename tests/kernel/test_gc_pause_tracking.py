from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from duplex_kernel import server
from duplex_kernel.listener_microbatch_worker import gc_pause_overlap


def test_gc_pause_tracker_records_bounded_sample() -> None:
    before = server.gc_pause_snapshot()
    before_sequence = before[-1].sequence if before else 0

    server.gc_pause_callback("start", {"generation": 2})
    server.gc_pause_callback(
        "stop", {"generation": 2, "collected": 3, "uncollectable": 1}
    )

    samples = server.gc_pause_snapshot()
    sample = next(
        sample
        for sample in reversed(samples)
        if sample.sequence > before_sequence
        and sample.generation == 2
        and sample.collected == 3
        and sample.uncollectable == 1
    )
    assert sample.generation == 2
    assert sample.end_ns >= sample.start_ns
    assert sample.collected == 3
    assert sample.uncollectable == 1
    assert len(server.gc_pause_ring) <= 256
    assert sum(callback is server.gc_pause_callback for callback in server.gc.callbacks) == 1


def test_duplex_gc_control_is_idempotent_and_restores_prior_state(monkeypatch) -> None:
    enabled = {"value": True}
    calls: list[str] = []

    monkeypatch.setattr(server.gc, "isenabled", lambda: enabled["value"])

    def disable() -> None:
        calls.append("disable")
        enabled["value"] = False

    def collect() -> int:
        calls.append("collect")
        return 0

    def enable() -> None:
        calls.append("enable")
        enabled["value"] = True

    monkeypatch.setattr(server.gc, "disable", disable)
    monkeypatch.setattr(server.gc, "collect", collect)
    monkeypatch.setattr(server.gc, "enable", enable)

    first = f"gc-test-{uuid4()}"
    second = f"gc-test-{uuid4()}"
    server.activate_duplex_session_gc(first)
    server.activate_duplex_session_gc(first)
    server.activate_duplex_session_gc(second)
    assert first in server.active_duplex_sessions
    assert second in server.active_duplex_sessions
    assert calls == ["disable"]

    server.finish_duplex_session_gc(first)
    assert calls == ["disable"]
    server.finish_duplex_session_gc(second)
    server.finish_duplex_session_gc(second)
    assert calls == ["disable", "collect", "enable"]
    assert enabled["value"] is True
    assert not server.active_duplex_sessions


def test_duplex_gc_control_keeps_gc_disabled_if_it_was_disabled(monkeypatch) -> None:
    enabled = {"value": False}
    calls: list[str] = []
    monkeypatch.setattr(server.gc, "isenabled", lambda: enabled["value"])
    monkeypatch.setattr(server.gc, "disable", lambda: calls.append("disable"))
    monkeypatch.setattr(server.gc, "collect", lambda: calls.append("collect") or 0)
    monkeypatch.setattr(
        server.gc,
        "enable",
        lambda: (calls.append("enable"), enabled.__setitem__("value", True)),
    )

    session_id = f"gc-test-{uuid4()}"
    server.activate_duplex_session_gc(session_id)
    server.finish_duplex_session_gc(session_id)
    assert calls == ["collect"]
    assert enabled["value"] is False


def test_gc_pause_overlap_measures_only_intersecting_nanoseconds() -> None:
    samples = (
        SimpleNamespace(start_ns=50, end_ns=125),
        SimpleNamespace(start_ns=120, end_ns=190),
        SimpleNamespace(start_ns=220, end_ns=250),
    )
    elapsed_ms, overlapped = gc_pause_overlap(samples, 100, 200)
    assert elapsed_ms == pytest.approx(0.000095)
    assert overlapped is True

    elapsed_ms, overlapped = gc_pause_overlap(samples, 200, 220)
    assert elapsed_ms == 0.0
    assert overlapped is False
