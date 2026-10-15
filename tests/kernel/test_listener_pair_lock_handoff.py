from __future__ import annotations

import threading
import time
import json
import subprocess
from unittest.mock import patch

import pytest

from duplex_kernel.config import (
    LISTENER_PAIR_LOCK_HANDOFF_ENV,
    listener_pair_lock_handoff_flag,
)
from duplex_kernel.listener import FairModelLock
from tools import run_pair_duplex_proof_3090 as runner


def test_default_model_lock_retains_listener_compute_alternation() -> None:
    lock = FairModelLock()
    with lock._condition:
        lock._last_grant = "listener"
        lock._queues["listener"].append((1, 101))
        lock._queues["compute"].append((2, 202))

    assert lock._selected_priority() == "compute"


def test_pair_handoff_admits_second_listener_before_waiting_compute() -> None:
    lock = FairModelLock(listener_pair_lock_handoff=True)
    with lock._condition:
        # After one successful listener grant, a second waiting listener gets
        # the remaining listener quantum before a queued compute waiter.
        lock._listener_grant_streak = 1
        lock._last_grant = "listener"
        lock._queues["listener"].append((1, 101))
        lock._queues["compute"].append((2, 202))

    assert lock._selected_priority() == "listener"


def _await_pending(lock: FairModelLock, listener: int, compute: int) -> None:
    deadline = time.monotonic() + 3.0
    with lock._condition:
        while time.monotonic() < deadline:
            counts = {
                key: len(queue) for key, queue in lock._queues.items()
            }
            if counts == {"listener": listener, "compute": compute}:
                return
            lock._condition.wait(timeout=max(0.0, deadline - time.monotonic()))
    raise AssertionError(
        f"waiter queues did not reach listener={listener}, compute={compute}: {lock.pending_counts}"
    )


def _run_two_waiter_handoff(lock: FairModelLock) -> list[str]:
    assert lock.acquire_fair("compute")
    order: list[str] = []
    first_acquired = threading.Event()
    release_first = threading.Event()

    def first_listener() -> None:
        assert lock.acquire_fair("listener")
        order.append("L_A")
        first_acquired.set()
        assert release_first.wait(3.0)
        lock.release()

    def queued_waiter(priority: str, label: str) -> None:
        assert lock.acquire_fair(priority)
        order.append(label)
        lock.release()

    listener_a = threading.Thread(target=first_listener)
    compute = threading.Thread(target=queued_waiter, args=("compute", "C"))
    listener_b = threading.Thread(target=queued_waiter, args=("listener", "L_B"))
    listener_a.start()
    compute.start()
    _await_pending(lock, listener=1, compute=1)
    lock.release()
    assert first_acquired.wait(3.0)

    listener_b.start()
    _await_pending(lock, listener=1, compute=1)
    release_first.set()
    for thread in (listener_a, listener_b, compute):
        thread.join(3.0)
        assert not thread.is_alive()
    return order


def test_pair_handoff_thread_order_is_listener_pair_then_compute() -> None:
    lock = FairModelLock(listener_pair_lock_handoff=True)
    assert _run_two_waiter_handoff(lock) == ["L_A", "L_B", "C"]


def test_default_thread_order_keeps_single_listener_quantum() -> None:
    assert _run_two_waiter_handoff(FairModelLock()) == ["L_A", "C", "L_B"]


def test_pair_handoff_bounds_listener_streak_and_keeps_fifo_per_class() -> None:
    lock = FairModelLock(listener_pair_lock_handoff=True)
    assert lock.acquire_fair("compute")
    order: list[str] = []
    first_acquired = threading.Event()
    release_first = threading.Event()

    def first_listener() -> None:
        assert lock.acquire_fair("listener")
        order.append("L_A")
        first_acquired.set()
        assert release_first.wait(3.0)
        lock.release()

    def queued_waiter(priority: str, label: str) -> None:
        assert lock.acquire_fair(priority)
        order.append(label)
        lock.release()

    threads = [
        threading.Thread(target=first_listener),
        threading.Thread(target=queued_waiter, args=("compute", "C_1")),
    ]
    for thread in threads:
        thread.start()
    _await_pending(lock, listener=1, compute=1)
    lock.release()
    assert first_acquired.wait(3.0)

    threads.extend([
        threading.Thread(target=queued_waiter, args=("listener", "L_B1")),
        threading.Thread(target=queued_waiter, args=("listener", "L_B2")),
        threading.Thread(target=queued_waiter, args=("listener", "L_B3")),
        threading.Thread(target=queued_waiter, args=("compute", "C_2")),
    ])
    for thread in threads[2:]:
        thread.start()
    _await_pending(lock, listener=3, compute=2)
    release_first.set()
    for thread in threads:
        thread.join(3.0)
        assert not thread.is_alive()

    assert order == ["L_A", "L_B1", "C_1", "L_B2", "L_B3", "C_2"]


def test_reentry_cancellation_and_invalid_release_do_not_count_as_grants() -> None:
    lock = FairModelLock(listener_pair_lock_handoff=True)
    assert lock.acquire_fair("listener")
    assert lock._listener_grant_streak == 1
    assert lock.acquire_fair("listener")
    assert lock._listener_grant_streak == 1
    lock.release()
    assert lock._listener_grant_streak == 1
    lock.release()
    assert lock._listener_grant_streak == 1

    with pytest.raises(RuntimeError, match="non-owner"):
        lock.release()
    assert lock._listener_grant_streak == 1

    assert lock.acquire_fair("listener", stop_check=lambda: True) is False
    assert lock._listener_grant_streak == 1
    with lock._condition:
        lock._queues["listener"].append((100, 101))
        lock._queues["compute"].append((101, 202))
        assert lock._selected_priority() == "listener"


def test_pair_handoff_flag_is_strict_and_default_off(monkeypatch) -> None:
    monkeypatch.delenv(LISTENER_PAIR_LOCK_HANDOFF_ENV, raising=False)
    assert listener_pair_lock_handoff_flag() is False
    assert listener_pair_lock_handoff_flag("1") is True
    assert listener_pair_lock_handoff_flag("0") is False
    with pytest.raises(RuntimeError, match="exactly '0' or '1'"):
        listener_pair_lock_handoff_flag("true")


def test_runner_wires_pair_handoff_to_probe_and_server_argv() -> None:
    flag = runner.LISTENER_PAIR_LOCK_HANDOFF_ENV
    assert runner.build_parser().parse_args([]).listener_pair_lock_handoff is False
    assert runner.build_parser().parse_args(["--listener-pair-lock-handoff"]).listener_pair_lock_handoff is True
    server_argv = runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle")
    preflight = runner.build_runtime_probe_argv(
        "/repo", "gemma4-omni:latest", listener_pair_lock_handoff=True,
    )
    default_preflight = runner.build_runtime_probe_argv("/repo", "gemma4-omni:latest")
    server = runner.build_docker_run_argv(
        repository_root="/repo", container_name="pair-lock", image="gemma4-omni:latest",
        server_argv=server_argv, listener_pair_lock_handoff=True,
    )
    default_server = runner.build_docker_run_argv(
        repository_root="/repo", container_name="pair-lock-default", image="gemma4-omni:latest",
        server_argv=server_argv,
    )
    assert f"{flag}=1" in preflight
    assert f"{flag}=1" in server
    assert f"{flag}=0" in default_preflight
    assert f"{flag}=0" in default_server
    assert "listener_pair_lock_handoff_setting" in runner.runtime_version_probe_code()
    assert "listener_pair_lock_handoff_enabled" in runner.runtime_version_probe_code()


def test_pair_handoff_health_is_required_and_must_match_request() -> None:
    assert runner.listener_pair_lock_handoff_record(
        {
            "listener_pair_lock_handoff_setting": "1",
            "listener_pair_lock_handoff_enabled": True,
        }, requested=True,
    ) == {"requested": True, "actual": True, "status": "PASS"}
    with pytest.raises(RuntimeError, match="health differs"):
        runner.listener_pair_lock_handoff_record({}, requested=True)
    with pytest.raises(RuntimeError, match="health differs"):
        runner.listener_pair_lock_handoff_record(
            {
                "listener_pair_lock_handoff_setting": "1",
                "listener_pair_lock_handoff_enabled": False,
            }, requested=True,
        )
    with pytest.raises(RuntimeError, match="health differs"):
        runner.listener_pair_lock_handoff_record(
            {"listener_pair_lock_handoff_enabled": True}, requested=True,
        )
    with pytest.raises(RuntimeError, match="health differs"):
        runner.listener_pair_lock_handoff_record(
            {
                "listener_pair_lock_handoff_setting": "0",
                "listener_pair_lock_handoff_enabled": True,
            }, requested=True,
        )
    assert runner.listener_pair_lock_handoff_record(
        {
            "listener_pair_lock_handoff_setting": "0",
            "listener_pair_lock_handoff_enabled": False,
        }, requested=False,
    ) == {"requested": False, "actual": False, "status": "PASS"}
    with pytest.raises(RuntimeError, match="health differs"):
        runner.listener_pair_lock_handoff_record(
            {"listener_pair_lock_handoff_enabled": False}, requested=False,
        )


def test_pair_handoff_preflight_requires_the_requested_effective_flag() -> None:
    talker = {
        "path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
        "configured_path": runner.NATIVE_TALKER_LIBRARY_CONTAINER_PATH,
        "size_bytes": 123,
        "sha256": runner.EXPECTED_NATIVE_TALKER_LIBRARY_SHA256,
        "native_abi_version": runner.EXPECTED_NATIVE_TALKER_ABI,
        "trace_abi_version": runner.EXPECTED_NATIVE_TALKER_TRACE_ABI,
    }
    evidence = {
        "python": "3.11.9",
        "packages": {"torch": "2.12.1", "transformers": "5.5.0"},
        "native_talker": talker,
        "talker_native_q1_setting": "1",
        "talker_native_q1_graph_setting": "1",
        "native_control6_trace_setting": "0",
        "native_listener_cuda_profile_setting": "0",
        "native_listener_cuda_profile_directory_setting": None,
        "native_listener_cuda_profile_enabled": False,
        "native_control6_fused_readback_setting": "0",
        "native_control6_fused_readback_enabled": False,
        "first_input_scheduler_anchor_setting": "0",
        "first_input_scheduler_anchor_enabled": False,
        "ready_frame_scheduler_dispatch_setting": "0",
        "ready_frame_scheduler_dispatch_enabled": False,
        "listener_pair_lock_handoff_setting": "1",
        "listener_pair_lock_handoff_enabled": True,
    }
    argv = runner.build_runtime_probe_argv(
        "/repo", "gemma4-omni:latest", listener_pair_lock_handoff=True,
    )
    with patch.object(
        runner,
        "_command",
        return_value=subprocess.CompletedProcess(argv, 0, json.dumps(evidence), ""),
    ):
        result = runner.probe_container_runtime(
            "/repo", "gemma4-omni:latest", listener_pair_lock_handoff=True,
        )
    assert result["listener_pair_lock_handoff_enabled"] is True

    for key, wrong_value in (
        ("listener_pair_lock_handoff_setting", "0"),
        ("listener_pair_lock_handoff_enabled", False),
    ):
        mismatch = dict(evidence)
        mismatch[key] = wrong_value
        with patch.object(
            runner,
            "_command",
            return_value=subprocess.CompletedProcess(argv, 0, json.dumps(mismatch), ""),
        ):
            with pytest.raises(RuntimeError, match="listener pair lock handoff CPU preflight"):
                runner.probe_container_runtime(
                    "/repo", "gemma4-omni:latest", listener_pair_lock_handoff=True,
                )
