from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import duplex_kernel.server as server_module
import tools.acceptance_telemetry as telemetry_module
from duplex_kernel.server import (
    TerminalResponseArchive,
    _attach_terminal_response_finalizer,
    create_app,
)
from tools.acceptance_telemetry import RuntimeTelemetrySampler, compact_sample


def _record(response_id: str, **overrides):
    return {
        "response_id": response_id,
        "generation_id": 3,
        "speaker_segment_id": "segment-1",
        "finalized": True,
        "ended_response": True,
        "finalizer_task_status": "completed",
        "playback_cancelled": False,
        "playback_drained": True,
        "playback_drain_failure": None,
        "playback_pending": False,
        "client_consumed_pcm_sequence": 8,
        "client_consumed_segments": ["segment-1"],
        "spoken_prefix_exclusive_cursor": 42,
        "kv_cache_position": 42,
        "kv_cache_source": "session.kv_cache.current_position",
        "playback_segments": [
            {
                "speaker_segment_id": "segment-1",
                "start_sample": 0,
                "end_sample": 3200,
                "emitted_end_sample": 3200,
                "done_emitted": True,
            }
        ],
        "allocator": {
            "allocated_bytes": 100,
            "reserved_bytes": 200,
            "peak_allocated_bytes": 300,
            "peak_reserved_bytes": 400,
        },
        "finalized_at_utc": "2026-11-02T12:00:00.000Z",
        **overrides,
    }


def _telemetry_session(session_id: str, response_id: str = "current-response"):
    return SimpleNamespace(
        session_id=session_id,
        _generation_lock=threading.RLock(),
        response_id=response_id,
        session_generation_id=3,
        speaker_segment_id="segment-1",
        active_response=False,
        response_cancel_notified=False,
        kv_cache=SimpleNamespace(current_position=42),
        _playback_ack_branches_lock=threading.RLock(),
        _playback_ack_branches={},
    )


class _Sessions:
    def __init__(self, sessions=()):
        self._sessions = list(sessions)

    def all(self):
        return list(self._sessions)


def _app(sessions):
    return create_app(
        SimpleNamespace(loaded=False, enforce_production_listener=False),
        sessions,
    )


def test_finalizer_completion_publishes_one_lightweight_idempotent_record(monkeypatch):
    archive = TerminalResponseArchive()
    session = SimpleNamespace(session_id="session-a", kv_cache=SimpleNamespace(current_position=42))
    branch = {
        **_record("response-a"),
        "session_generation_id": 3,
        "spoken_prefix_lock": threading.RLock(),
        "deferred_playback_finalizer": object(),
        "pcm": np.zeros(64, dtype=np.float32),
    }
    monkeypatch.setattr(server_module.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(server_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(server_module.torch.cuda, "memory_allocated", lambda _device: 101)
    monkeypatch.setattr(server_module.torch.cuda, "memory_reserved", lambda _device: 202)
    monkeypatch.setattr(server_module.torch.cuda, "max_memory_allocated", lambda _device: 303)
    monkeypatch.setattr(server_module.torch.cuda, "max_memory_reserved", lambda _device: 404)

    async def run_finalizer():
        await asyncio.sleep(0)

    async def publish():
        task = asyncio.create_task(run_finalizer())
        _attach_terminal_response_finalizer(archive, session, "response-a", branch, task)
        await task
        await asyncio.sleep(0)

    asyncio.run(publish())

    records = archive.get_for_session("session-a")
    assert len(records) == 1
    assert records[0]["finalizer_task_status"] == "completed"
    assert records[0]["finalized"] is True
    assert records[0]["allocator"] == {
        "status": "available",
        "device": "cuda:0",
        "allocated_bytes": 101,
        "reserved_bytes": 202,
        "peak_allocated_bytes": 303,
        "peak_reserved_bytes": 404,
    }
    assert "pcm" not in records[0]
    assert all(not isinstance(value, np.ndarray) for value in records[0].values())

    archive.record_finalized("session-a", "response-a", _record("response-a"))
    records = archive.get_for_session("session-a")
    assert len(records) == 1


def test_archive_is_bounded_fifo_without_mutating_operational_ack_branches():
    archive = TerminalResponseArchive(capacity=64)
    operational = {"live": {"playback_ack_event": object(), "playback_pending": True}}
    original = operational["live"]

    for index in range(65):
        archive.record_finalized("session-a", f"response-{index}", _record(f"response-{index}"))

    records = archive.get_for_session("session-a")
    assert len(records) == 64
    assert records[0]["response_id"] == "response-1"
    assert records[-1]["response_id"] == "response-64"
    assert archive.eviction_count == 1
    assert list(operational) == ["live"]
    assert operational["live"] is original


def test_archive_concurrent_reads_and_writes_remain_bounded_and_keyed_by_session():
    archive = TerminalResponseArchive(capacity=64)

    def publish(index):
        session_id = f"session-{index % 4}"
        response_id = f"response-{index}"
        archive.record_finalized(session_id, response_id, _record(response_id))
        for _ in range(3):
            assert len(archive.get_all()) <= archive.capacity
            archive.get_for_session(session_id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(publish, range(80)))

    assert len(archive.get_all()) == 64
    assert archive.eviction_count == 16
    assert archive.get_for_session("session-0")
    assert all(record["session_id"] == "session-0" for record in archive.get_for_session("session-0"))


def test_archive_idempotency_is_scoped_to_session_and_response_pair():
    archive = TerminalResponseArchive()
    archive.record_finalized("session-a", "shared-response", _record("shared-response", generation_id=1))
    archive.record_finalized("session-b", "shared-response", _record("shared-response", generation_id=2))
    archive.record_finalized("session-a", "shared-response", _record("shared-response", generation_id=3))

    assert len(archive.get_all()) == 2
    assert archive.get_for_session("session-a")[0]["generation_id"] == 3
    assert archive.get_for_session("session-b")[0]["generation_id"] == 2


def test_runtime_telemetry_merges_archived_branch_after_operational_prune(monkeypatch):
    monkeypatch.setenv("DUPLEX_RUNTIME_TELEMETRY", "1")
    monkeypatch.setattr(server_module.torch.cuda, "is_initialized", lambda: False)
    app = _app(_Sessions([_telemetry_session("session-a")]))
    app.state.terminal_response_archive.record_finalized(
        "session-a", "response-a", _record("response-a")
    )

    with TestClient(app) as client:
        response = client.get(
            "/debug/runtime-telemetry?session_id=session-a&response_id=response-a"
        )

    assert response.status_code == 200
    sessions = response.json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["branches"] == [
        {**_record("response-a"), "session_id": "session-a"}
    ]


def test_runtime_telemetry_updates_and_deduplicates_matching_active_branch(monkeypatch):
    monkeypatch.setenv("DUPLEX_RUNTIME_TELEMETRY", "1")
    monkeypatch.setattr(server_module.torch.cuda, "is_initialized", lambda: False)
    session = _telemetry_session("session-a", response_id="response-a")
    session._playback_ack_branches["response-a"] = {
        "response_id": "response-a",
        "session_generation_id": 3,
        "speaker_segment_id": "segment-1",
        "playback_segments": {},
        "playback_cancelled": False,
        "finalized": False,
        "ended_response": False,
        "client_consumed_segments": (),
        "spoken_prefix_exclusive_cursor": 8,
        "spoken_prefix_lock": threading.RLock(),
    }
    app = _app(_Sessions([session]))
    app.state.terminal_response_archive.record_finalized(
        "session-a", "response-a", _record("response-a")
    )

    with TestClient(app) as client:
        response = client.get(
            "/debug/runtime-telemetry?session_id=session-a&response_id=response-a"
        )

    branches = response.json()["sessions"][0]["branches"]
    assert len(branches) == 1
    assert branches[0]["finalized"] is True
    assert branches[0]["spoken_prefix_exclusive_cursor"] == 42
    assert branches[0]["allocator"]["allocated_bytes"] == 100


def test_runtime_telemetry_keeps_closed_session_records_and_applies_filters(monkeypatch):
    monkeypatch.setenv("DUPLEX_RUNTIME_TELEMETRY", "1")
    monkeypatch.setattr(server_module.torch.cuda, "is_initialized", lambda: False)
    app = _app(_Sessions())
    archive = app.state.terminal_response_archive
    archive.record_finalized("closed-a", "response-a", _record("response-a"))
    archive.record_finalized("closed-a", "response-b", _record("response-b"))
    archive.record_finalized("closed-b", "response-c", _record("response-c"))

    with TestClient(app) as client:
        response = client.get(
            "/debug/runtime-telemetry?session_id=closed-a&response_id=response-b"
        )
        absent = client.get("/debug/runtime-telemetry?session_id=closed-b&response_id=response-b")

    assert response.status_code == 200
    assert len(response.json()["sessions"]) == 1
    snapshot = response.json()["sessions"][0]
    assert snapshot["session_id"] == "closed-a"
    assert [branch["response_id"] for branch in snapshot["branches"]] == ["response-b"]
    assert absent.json()["sessions"] == []


def test_compact_sample_and_observe_use_only_completed_terminal_allocator(tmp_path):
    sampler = RuntimeTelemetrySampler("http://unused", output_dir=tmp_path)
    payload = {
        "cuda_allocator": {
            "status": "available",
            "device": "cuda:0",
            "allocated_bytes": 900,
            "reserved_bytes": 1200,
            "peak_allocated_bytes": 1400,
            "peak_reserved_bytes": 1600,
        },
        "sessions": [
            {
                "session_id": "session-a",
                "branches": [
                    _record(
                        "cancelled-pending",
                        finalizer_task_status="pending",
                        playback_cancelled=True,
                    ),
                    _record(
                        "response-a",
                        allocator={
                            "allocated_bytes": 2 * 1024 * 1024,
                            "reserved_bytes": 3 * 1024 * 1024,
                            "peak_allocated_bytes": 4 * 1024 * 1024,
                            "peak_reserved_bytes": 5 * 1024 * 1024,
                        },
                    ),
                ],
            }
        ],
    }
    row = compact_sample(payload, "2026-11-02T12:01:00.000Z")

    sampler._observe(row)

    assert row["sessions"][0]["branches"][1]["allocator"]["allocated_bytes"] == 2 * 1024 * 1024
    assert row["sessions"][0]["branches"][1]["finalized_at_utc"] == "2026-11-02T12:00:00.000Z"
    assert set(sampler.post_drain) == {"response-a"}
    assert sampler.post_drain["response-a"]["allocated_mib"] == 2.0
    assert sampler.post_drain["response-a"]["reserved_mib"] == 3.0
    assert sampler.post_drain["response-a"]["sampled_at_utc"] == "2026-11-02T12:00:00.000Z"


def test_stop_observes_the_final_postrun_sample(monkeypatch, tmp_path):
    payload = {
        "cuda_allocator": {
            "status": "available",
            "device": "cuda:0",
            "allocated_bytes": 900,
            "reserved_bytes": 1200,
            "peak_allocated_bytes": 1400,
            "peak_reserved_bytes": 1600,
        },
        "sessions": [
            {"session_id": "closed-a", "branches": [_record("response-a")]}
        ],
    }
    sampler = RuntimeTelemetrySampler("http://unused", output_dir=tmp_path)
    monkeypatch.setattr(telemetry_module, "http_json", lambda _endpoint: payload)

    sampler.stop()

    assert sampler.postrun_raw is payload
    assert sampler.post_drain["response-a"]["session_id"] == "closed-a"
    assert sampler.post_drain["response-a"]["allocated_mib"] == 100 / (1024 * 1024)

