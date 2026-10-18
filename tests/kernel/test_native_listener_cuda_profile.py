from __future__ import annotations

import ctypes
import json
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from duplex_kernel import native_listener
from duplex_kernel.listener import _native_q4_profile_step_kwargs
from duplex_kernel.config import (
    native_control6_fused_readback_flag,
    native_listener_cuda_profile_flag,
)
from tools import run_pair_duplex_proof_3090 as runner


class _FakeEvent:
    def __init__(self, calls: list[tuple[object, ...]], *, query_result: bool = True) -> None:
        self.calls = calls
        self.query_result = query_result
        self.recorded_stream = None

    def record(self, stream) -> None:
        self.recorded_stream = stream
        self.calls.append(("event_record", self, stream))

    def query(self) -> bool:
        self.calls.append(("event_query", self))
        return self.query_result

    def elapsed_time(self, end) -> float:
        self.calls.append(("elapsed_time", self, end))
        assert end.query_result is True
        return 12.5


class _FakeTensor:
    device = "cuda:0"
    shape = (1, 4, native_listener.GEMMA4_HIDDEN_SIZE)

    def __init__(self, calls: list[tuple[object, ...]], name: str) -> None:
        self.calls = calls
        self.name = name

    def record_stream(self, stream) -> None:
        self.calls.append((f"{self.name}_record_stream", stream))

    def copy_(self, value, *, non_blocking: bool) -> None:
        self.calls.append(("copy", value, non_blocking))

    def data_ptr(self) -> int:
        return 0x7000


def _host_context(session_id: str, frame_id: int) -> dict[str, object]:
    return {
        "session_id": session_id,
        "frame_id": frame_id,
        "capture_start_ns": 1000 + frame_id,
        "capture_end_ns": 2000 + frame_id,
        "enqueue_ns": 2100 + frame_id,
        "deadline_ns": 3000 + frame_id,
        "observed_ns": 3100 + frame_id,
        "listener_step_start_ns": 3200 + frame_id,
        "listener_step_end_ns": 3300 + frame_id,
        "listener_callback_end_ns": 3400 + frame_id,
        "shared_lm_lock_wait_ms": None,
        "shared_lm_lock_host_hold_ms": None,
        "clock": "time.monotonic_ns",
    }


def _initialize_existing_runtime_timing_fixture(native: object) -> None:
    # The active Linux source snapshot also has an existing, default-off
    # Torch-side sampler. These are constructor-established fields; tests that
    # allocate the object with __new__ must model them explicitly.
    native._cuda_timing_enabled = False
    native._cuda_timing_step_count = 0
    native._last_output_ready_event = SimpleNamespace(record=lambda _stream: None)


def test_native_q4_sampler_wraps_actual_ctypes_call_on_exact_torch_stream() -> None:
    calls: list[tuple[object, ...]] = []
    stream = SimpleNamespace(cuda_stream=0xCAFE, device="cuda:0")
    event_factory = lambda **_kwargs: _FakeEvent(calls)
    profiler = native_listener._NativeQ4CudaProfiler(True, event_factory=event_factory)
    native = native_listener.NativeGemmaListenerLM.__new__(native_listener.NativeGemmaListenerLM)
    _initialize_existing_runtime_timing_fixture(native)
    native._q4_cuda_profiler = profiler
    native.enable_cuda_graph = False
    native.stream_q_len = 2
    native._ctx = ctypes.c_void_p(123)
    native._frame_meta_ptr = ctypes.c_void_p(456)
    native._frame_input = _FakeTensor(calls, "frame")
    native.output_tensor = _FakeTensor(calls, "output")
    native._lib = SimpleNamespace(gemma4_listener_step_async=lambda *_args: calls.append(("native_q4", *_args)) or 0)
    native._cuda_stream = lambda requested: requested or stream
    native._resolve_stream = lambda requested, _producer, **_kwargs: requested or stream
    native._require_cuda_tensor = lambda *_args, **_kwargs: None
    native._last_execution_stream = None
    input_tensor = _FakeTensor(calls, "input")

    with patch.object(native_listener.torch.cuda, "stream", return_value=nullcontext()):
        result = native.step_async(
            input_tensor,
            profile_frame_id=17,
            profile_host_context=_host_context("peer-a", 17),
        )

    event_records = [call for call in calls if call[0] == "event_record"]
    native_call = next(call for call in calls if call[0] == "native_q4")
    assert len(event_records) == 2
    assert event_records[0][2] is stream
    assert event_records[1][2] is stream
    assert native_call[-1].value == stream.cuda_stream
    assert calls.index(event_records[0]) < calls.index(native_call) < calls.index(event_records[1])
    assert result is native.output_tensor
    assert native._last_execution_stream is stream


def test_native_q4_profile_disabled_creates_no_events_or_cuda_timing_calls() -> None:
    calls: list[tuple[object, ...]] = []
    stream = SimpleNamespace(cuda_stream=0xCAFE, device="cuda:0")
    native = native_listener.NativeGemmaListenerLM.__new__(native_listener.NativeGemmaListenerLM)
    _initialize_existing_runtime_timing_fixture(native)
    native._q4_cuda_profiler = None
    native.enable_cuda_graph = False
    native.stream_q_len = 2
    native._ctx = ctypes.c_void_p(123)
    native._frame_meta_ptr = ctypes.c_void_p(456)
    native._frame_input = _FakeTensor(calls, "frame")
    native.output_tensor = _FakeTensor(calls, "output")
    native._lib = SimpleNamespace(gemma4_listener_step_async=lambda *_args: calls.append(("native_q4", *_args)) or 0)
    native._cuda_stream = lambda requested: requested or stream
    native._resolve_stream = lambda requested, _producer, **_kwargs: requested or stream
    native._require_cuda_tensor = lambda *_args, **_kwargs: None
    native._last_execution_stream = None

    with patch.object(native_listener.torch.cuda, "stream", return_value=nullcontext()), patch.object(
        native_listener.torch.cuda, "Event", side_effect=AssertionError("event allocation while disabled")
    ):
        native.step_async(_FakeTensor(calls, "input"), profile_frame_id=17)
    assert not any(call[0] == "event_record" for call in calls)


def test_requested_native_profile_fails_closed_before_dso_when_event_unavailable() -> None:
    calls: list[tuple[object, ...]] = []
    stream = SimpleNamespace(cuda_stream=0xCAFE, device="cuda:0")
    profiler = native_listener._NativeQ4CudaProfiler(
        True, event_factory=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("event unavailable"))
    )
    native = native_listener.NativeGemmaListenerLM.__new__(native_listener.NativeGemmaListenerLM)
    _initialize_existing_runtime_timing_fixture(native)
    native._q4_cuda_profiler = profiler
    native.enable_cuda_graph = False
    native.stream_q_len = 2
    native._ctx = ctypes.c_void_p(123)
    native._frame_meta_ptr = ctypes.c_void_p(456)
    native._frame_input = _FakeTensor(calls, "frame")
    native.output_tensor = _FakeTensor(calls, "output")
    native._lib = SimpleNamespace(
        gemma4_listener_step_async=lambda *_args: calls.append(("native_q4", *_args)) or 0
    )
    native._cuda_stream = lambda requested: requested or stream
    native._resolve_stream = lambda requested, _producer, **_kwargs: requested or stream
    native._require_cuda_tensor = lambda *_args, **_kwargs: None
    native._last_execution_stream = None

    with patch.object(native_listener.torch.cuda, "stream", return_value=nullcontext()):
        with pytest.raises(RuntimeError, match="CUDA profile failed"):
            native.step_async(
                _FakeTensor(calls, "input"),
                profile_frame_id=17,
                profile_host_context=_host_context("peer-a", 17),
            )
    assert not any(call[0] == "native_q4" for call in calls)
    summary = profiler.summary(session_id="peer-a", runtime_info={})
    assert summary["status"] == "failed"
    assert summary["failure"]["stage"] == "event_create_or_start_record"


def test_native_q4_profile_reports_pending_without_elapsed_time_or_sync() -> None:
    calls: list[tuple[object, ...]] = []
    stream = SimpleNamespace(cuda_stream=0xA0, device="cuda:0")
    profiler = native_listener._NativeQ4CudaProfiler(
        True, event_factory=lambda **_kwargs: _FakeEvent(calls, query_result=False)
    )
    sample = profiler.begin_step(stream, frame_id=31, host_context=_host_context("peer-a", 31))
    assert sample is not None
    profiler.finish_step(sample, stream)
    summary = profiler.summary(
        session_id="peer-a",
        runtime_info={"cuda_graph_enabled": True, "attention_backend": "CUSTOM"},
    )
    assert summary["status"] == "incomplete"
    assert summary["completed_sample_count"] == 0
    assert summary["pending_sample_count"] == 1
    assert summary["pending_frame_ids"] == [31]
    assert summary["pending_samples"][0]["host_context"]["enqueue_ns"] == 2131
    assert "bounded listener.tick_timings history has rolled over" in summary["host_join_contract"]
    assert not any(call[0] == "elapsed_time" for call in calls)
    assert not any("synchronize" in str(call[0]).lower() for call in calls)


def test_native_q4_profile_keeps_host_context_live_until_sidecar_snapshot() -> None:
    from duplex_kernel.server import (
        attach_native_q4_lock_metrics,
        native_q4_cuda_profile_host_context,
        write_native_listener_cuda_profile_sidecar,
    )

    calls: list[tuple[object, ...]] = []
    stream = SimpleNamespace(cuda_stream=0xB0, device="cuda:0")
    profiler = native_listener._NativeQ4CudaProfiler(
        True, event_factory=lambda **_kwargs: _FakeEvent(calls, query_result=True)
    )
    native = native_listener.NativeGemmaListenerLM.__new__(native_listener.NativeGemmaListenerLM)
    _initialize_existing_runtime_timing_fixture(native)
    native._q4_cuda_profiler = profiler
    native.enable_cuda_graph = False
    native.stream_q_len = 2
    native._ctx = ctypes.c_void_p(123)
    native._frame_meta_ptr = ctypes.c_void_p(456)
    native._frame_input = _FakeTensor(calls, "frame")
    native.output_tensor = _FakeTensor(calls, "output")
    native._lib = SimpleNamespace(
        gemma4_listener_step_async=lambda *_args: calls.append(("native_q4", *_args)) or 0
    )
    native._cuda_stream = lambda requested: requested or stream
    native._resolve_stream = lambda requested, _producer, **_kwargs: requested or stream
    native._require_cuda_tensor = lambda *_args, **_kwargs: None
    native._last_execution_stream = None

    frame = SimpleNamespace(
        frame_id=17,
        capture_start_ns=1000,
        capture_end_ns=2000,
        enqueue_ns=2100,
    )
    tick = SimpleNamespace(
        deadline_ns=3000,
        observed_ns=3100,
        listener_start_ns=3150,
        listener_end_ns=None,
    )
    host_context = native_q4_cuda_profile_host_context("peer-a", frame, tick)
    host_context["listener_step_start_ns"] = 3200

    with patch.object(native_listener.torch.cuda, "stream", return_value=nullcontext()):
        native.step_async(
            _FakeTensor(calls, "input"),
            profile_frame_id=17,
            profile_host_context=host_context,
        )

    # Model the real ordering: the event has completed and is harvested before
    # scheduler_listener's finally block fills its host timing dictionary.
    profiler._poll_completed(raise_on_error=True)
    host_context["listener_step_end_ns"] = 3300
    host_context["listener_callback_end_ns"] = 3400
    attach_native_q4_lock_metrics(
        host_context,
        SimpleNamespace(last_lock_wait_ms=2.25, last_lock_hold_ms=21.5),
    )

    # The sampled frame has rolled out of the ordinary bounded 128-row history.
    recent_tick_history = [
        {"frame_id": frame_id, "enqueue_ns": 10_000 + frame_id}
        for frame_id in range(400, 528)
    ]
    summary = profiler.summary(session_id="peer-a", runtime_info={})
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp_dir:
        sidecar_path = write_native_listener_cuda_profile_sidecar(
            session_id="peer-a",
            profile=summary,
            host_tick_timings=recent_tick_history,
            directory=temp_dir,
        )
        assert sidecar_path is not None
        sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))

    sampled_row = sidecar["sampled_host_tick_timings"][0]
    assert sampled_row["frame_id"] == 17
    assert sampled_row["listener_step_end_ns"] == 3300
    assert sampled_row["listener_callback_end_ns"] == 3400
    assert sampled_row["shared_lm_lock_wait_ms"] == 2.25
    assert sampled_row["shared_lm_lock_host_hold_ms"] == 21.5
    assert sampled_row["scheduler_listener_end_ns"] is None
    assert all(row["frame_id"] != 17 for row in sidecar["host_tick_timings"])


def test_native_q4_profile_samples_every_eight_caps_at_64_per_context_without_aliasing() -> None:
    calls: list[tuple[object, ...]] = []
    stream = SimpleNamespace(cuda_stream=0xA0, device="cuda:0")
    factory = lambda **_kwargs: _FakeEvent(calls, query_result=True)
    profilers = [native_listener._NativeQ4CudaProfiler(True, event_factory=factory) for _ in range(2)]
    summaries = []
    for sid, profiler in zip(("peer-a", "peer-b"), profilers, strict=True):
        for frame_id in range(520):
            sample = profiler.begin_step(
                stream,
                frame_id=frame_id,
                host_context=_host_context(sid, frame_id),
            )
            if sample is not None:
                profiler.finish_step(sample, stream)
        summaries.append(profiler.summary(session_id=sid, runtime_info={"cuda_graph_enabled": True}))
    first, second = summaries
    assert first["sampling_frequency_frames"] == 8
    assert first["sample_cap"] == 64
    assert first["sampled_event_pair_count"] == 64
    assert first["completed_sample_count"] == 64
    assert first["pending_sample_count"] == 0
    assert first["samples"][0]["frame_id"] == 0
    assert first["samples"][1]["frame_id"] == 8
    assert first["samples"][0]["host_context"]["session_id"] == "peer-a"
    assert first["samples"] is not second["samples"]
    assert first["session_id"] == "peer-a" and second["session_id"] == "peer-b"
    assert sum(call[0] == "event_record" for call in calls) == 2 * 64 * 2


def test_profile_flag_is_strict_and_default_off() -> None:
    assert native_listener_cuda_profile_flag(None) is False
    assert native_listener_cuda_profile_flag("0") is False
    assert native_listener_cuda_profile_flag("1") is True
    with pytest.raises(RuntimeError, match="must be exactly"):
        native_listener_cuda_profile_flag("true")


def test_native_control6_fused_readback_flag_is_strict_and_default_off(monkeypatch) -> None:
    from duplex_kernel.config import NATIVE_CONTROL6_FUSED_READBACK_ENV

    monkeypatch.delenv(NATIVE_CONTROL6_FUSED_READBACK_ENV, raising=False)
    assert native_control6_fused_readback_flag() is False
    assert native_control6_fused_readback_flag("0") is False
    assert native_control6_fused_readback_flag("1") is True
    with pytest.raises(RuntimeError, match="must be exactly"):
        native_control6_fused_readback_flag("true")


def test_listener_only_passes_sampled_frame_metadata_when_sampler_is_enabled() -> None:
    state = SimpleNamespace(_native_q4_profile_host_context=_host_context("peer-a", 23))
    assert _native_q4_profile_step_kwargs(SimpleNamespace(q4_cuda_profile_enabled=False), state) == {}
    assert _native_q4_profile_step_kwargs(SimpleNamespace(q4_cuda_profile_enabled=True), state) == {
        "profile_frame_id": 23,
        "profile_host_context": state._native_q4_profile_host_context,
    }
    assert _native_q4_profile_step_kwargs(
        SimpleNamespace(q4_cuda_profile_enabled=True), SimpleNamespace()
    ) == {"profile_frame_id": None, "profile_host_context": None}


def test_pair_runner_profile_flag_is_default_off_and_passed_to_preflight_and_server() -> None:
    flag = runner.NATIVE_LISTENER_CUDA_PROFILE_ENV
    directory = "/workspace/artifacts/profile-run/native_listener_cuda_profile"
    assert runner.build_parser().parse_args([]).native_listener_cuda_profile is False
    parsed = runner.build_parser().parse_args(["--native-listener-cuda-profile"])
    assert parsed.native_listener_cuda_profile is True
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp_dir:
        root = Path(temp_dir)
        preflight = runner.build_runtime_probe_argv(
            root, "gemma4-omni:latest", native_listener_cuda_profile=True,
            native_listener_cuda_profile_directory=directory,
        )
        server = runner.build_docker_run_argv(
            repository_root=root,
            container_name="profile-run",
            image="gemma4-omni:latest",
            server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
            native_listener_cuda_profile=True,
            native_listener_cuda_profile_directory=directory,
        )
    for argv in (preflight, server):
        assert f"{flag}=1" in argv
        assert f"{runner.NATIVE_LISTENER_CUDA_PROFILE_DIRECTORY_ENV}={directory}" in argv
    assert "native_listener_cuda_profile_setting" in runner.runtime_version_probe_code()


def test_pair_runner_fused_readback_flag_is_default_off_and_passed_to_preflight_and_server() -> None:
    flag = runner.NATIVE_CONTROL6_FUSED_READBACK_ENV
    assert runner.build_parser().parse_args([]).native_control6_fused_readback is False
    parsed = runner.build_parser().parse_args(["--native-control6-fused-readback"])
    assert parsed.native_control6_fused_readback is True
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp_dir:
        root = Path(temp_dir)
        preflight = runner.build_runtime_probe_argv(
            root, "gemma4-omni:latest", native_control6_fused_readback=True,
        )
        server = runner.build_docker_run_argv(
            repository_root=root,
            container_name="fused-run",
            image="gemma4-omni:latest",
            server_argv=runner.build_server_argv(bundle_path="/workspace/checkpoints/bundle"),
            native_control6_fused_readback=True,
        )
    for argv in (preflight, server):
        assert f"{flag}=1" in argv
    assert "native_control6_fused_readback_setting" in runner.runtime_version_probe_code()
    assert runner.native_control6_fused_readback_record({
        "native_control6_fused_readback_enabled": True,
        "native_control6_fused_readback_setting": "1",
    }, requested=True)["status"] == "PASS"
    with pytest.raises(RuntimeError, match="health differs"):
        runner.native_control6_fused_readback_record(
            {"native_control6_fused_readback_enabled": True,
             "native_control6_fused_readback_setting": "1"},
            requested=False,
        )
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
        "first_input_scheduler_anchor_setting": "0",
        "first_input_scheduler_anchor_enabled": False,
        "ready_frame_scheduler_dispatch_setting": "0",
        "ready_frame_scheduler_dispatch_enabled": False,
        "listener_pair_lock_handoff_setting": "0",
        "listener_pair_lock_handoff_enabled": False,
        "native_control6_fused_readback_setting": "1",
        "native_control6_fused_readback_enabled": True,
    }
    argv = runner.build_runtime_probe_argv(
        "/repo", "gemma4-omni:latest", native_control6_fused_readback=True,
    )
    with patch.object(
        runner,
        "_command",
        return_value=subprocess.CompletedProcess(argv, 0, json.dumps(evidence), ""),
    ):
        result = runner.probe_container_runtime(
            "/repo", "gemma4-omni:latest", native_control6_fused_readback=True,
        )
    assert result["native_control6_fused_readback_enabled"] is True


def test_profile_preflight_attests_exact_requested_flag_and_directory_without_gpu() -> None:
    directory = "/workspace/artifacts/profile-run/native_listener_cuda_profile"
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
        "native_listener_cuda_profile_setting": "1",
        "native_listener_cuda_profile_directory_setting": directory,
        "native_listener_cuda_profile_enabled": True,
        "native_control6_fused_readback_setting": "0",
        "native_control6_fused_readback_enabled": False,
        "first_input_scheduler_anchor_setting": "0",
        "first_input_scheduler_anchor_enabled": False,
        "ready_frame_scheduler_dispatch_setting": "0",
        "ready_frame_scheduler_dispatch_enabled": False,
        "listener_pair_lock_handoff_setting": "0",
        "listener_pair_lock_handoff_enabled": False,
    }
    argv = runner.build_runtime_probe_argv(
        "/repo", "gemma4-omni:latest", native_listener_cuda_profile=True,
        native_listener_cuda_profile_directory=directory,
    )
    assert "--gpus" not in argv
    with patch.object(
        runner,
        "_command",
        return_value=subprocess.CompletedProcess(argv, 0, json.dumps(evidence), ""),
    ):
        result = runner.probe_container_runtime(
            "/repo", "gemma4-omni:latest", native_listener_cuda_profile=True,
            native_listener_cuda_profile_directory=directory,
        )
    assert result["native_listener_cuda_profile_enabled"] is True
    assert runner.native_listener_cuda_profile_record(
        {
            "native_listener_cuda_profile_enabled": True,
            "listener_model_lock_host_timeline_enabled": True,
            "native_listener_cuda_profile_directory": directory,
        }, requested=True, directory=directory,
    )["status"] == "PASS"
    with pytest.raises(RuntimeError, match="health differs"):
        runner.native_listener_cuda_profile_record({}, requested=True, directory=directory)
    with pytest.raises(RuntimeError, match="host timeline health differs"):
        runner.native_listener_cuda_profile_record(
            {
                "native_listener_cuda_profile_enabled": True,
                "listener_model_lock_host_timeline_enabled": False,
                "native_listener_cuda_profile_directory": directory,
            }, requested=True, directory=directory,
        )
    with pytest.raises(RuntimeError, match="directory differs"):
        runner.native_listener_cuda_profile_record(
            {
                "native_listener_cuda_profile_enabled": True,
                "listener_model_lock_host_timeline_enabled": True,
            }, requested=True, directory=directory,
        )
    with pytest.raises(RuntimeError, match="directory differs"):
        runner.native_listener_cuda_profile_record(
            {
                "native_listener_cuda_profile_enabled": True,
                "listener_model_lock_host_timeline_enabled": True,
                "native_listener_cuda_profile_directory": "/workspace/wrong",
            },
            requested=True,
            directory=directory,
        )


def test_native_q4_profile_sidecar_is_atomic_and_keeps_host_join_rows() -> None:
    from duplex_kernel.server import write_native_listener_cuda_profile_sidecar

    profile = {
        "enabled": True,
        "status": "incomplete",
        "session_id": "peer/a",
        "samples": [{"frame_id": 8, "elapsed_ms": 11.25, "host_context": _host_context("peer/a", 8)}],
        "pending_samples": [{
            "frame_id": 9,
            "host_context": _host_context("peer/a", 9),
        }],
    }
    host_rows = [{"frame_id": 999, "enqueue_ns": 101, "listener_start_ns": 202}]
    lock_timeline = {
        "enabled": True,
        "status": "complete",
        "scope": "process_shared_model_lock",
        "clock": "time.monotonic_ns",
        "process_id": 567,
        "container_identity": "proof-container",
        "snapshot_monotonic_ns": 456789,
        "capacity": 2048,
        "retained_event_count": 1,
        "dropped_event_count": 0,
        "dedupe_key": ["container_identity", "process_id", "trace_seq"],
        "join_strategy": "unique interval or ambiguous global",
        "events": [{"event": "acquire", "priority": "listener", "ticket": 7}],
    }
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp_dir:
        result = write_native_listener_cuda_profile_sidecar(
            session_id="peer/a", profile=profile, host_tick_timings=host_rows,
            model_lock_timeline=lock_timeline, directory=temp_dir,
        )
        assert result is not None
        path = Path(result)
        assert path.is_file()
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["session_id"] == "peer/a"
        assert value["profile"]["samples"][0]["frame_id"] == 8
        assert value["host_tick_timings"] == host_rows
        assert value["model_lock_timeline"] == lock_timeline
        assert value["sampled_host_tick_timings"] == [
            {"session_id": "peer/a", **_host_context("peer/a", 8)},
            {"session_id": "peer/a", **_host_context("peer/a", 9)},
        ]
        assert list(Path(temp_dir).iterdir()) == [path]


def test_profile_sidecars_mark_repeated_lock_timeline_as_process_global() -> None:
    from duplex_kernel.server import write_native_listener_cuda_profile_sidecar

    timeline = {
        "enabled": True,
        "status": "complete",
        "scope": "process_shared_model_lock",
        "clock": "time.monotonic_ns",
        "process_id": 567,
        "container_identity": "proof-container",
        "snapshot_monotonic_ns": 456789,
        "capacity": 8,
        "retained_event_count": 1,
        "dropped_event_count": 0,
        "dedupe_key": ["container_identity", "process_id", "trace_seq"],
        "join_strategy": (
            "unique listener-step interval only; otherwise global or ambiguous"
        ),
        "events": [{"trace_seq": 4, "event": "acquire", "priority": "listener"}],
    }
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp_dir:
        paths = []
        for sid in ("peer/a", "peer/b"):
            result = write_native_listener_cuda_profile_sidecar(
                session_id=sid,
                profile={"enabled": True, "samples": [], "pending_samples": []},
                host_tick_timings=[],
                model_lock_timeline=timeline,
                directory=temp_dir,
            )
            assert result is not None
            paths.append(Path(result))
        assert paths[0] != paths[1]
        first = json.loads(paths[0].read_text(encoding="utf-8"))
        second = json.loads(paths[1].read_text(encoding="utf-8"))

    assert first["session_id"] == "peer/a"
    assert second["session_id"] == "peer/b"
    assert first["model_lock_timeline"] == second["model_lock_timeline"] == timeline
    assert first["model_lock_timeline"]["scope"] == "process_shared_model_lock"
    assert first["model_lock_timeline"]["dedupe_key"] == [
        "container_identity", "process_id", "trace_seq",
    ]


def test_native_q4_profile_effective_directory_resolves_symlink_target() -> None:
    from duplex_kernel.server import (
        native_listener_cuda_profile_effective_directory,
        write_native_listener_cuda_profile_sidecar,
    )

    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp_dir:
        root = Path(temp_dir)
        actual = root / "actual"
        actual.mkdir()
        replacement = root / "replacement"
        replacement.mkdir()
        alias = root / "alias"
        try:
            alias.symlink_to(actual, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlinks unavailable: {exc}")

        pinned_directory = native_listener_cuda_profile_effective_directory(alias)
        assert pinned_directory == str(actual.resolve())
        alias.unlink()
        alias.symlink_to(replacement, target_is_directory=True)
        sidecar_path = write_native_listener_cuda_profile_sidecar(
            session_id="peer/a",
            profile={"enabled": True, "samples": [], "pending_samples": []},
            host_tick_timings=[],
            directory=pinned_directory,
        )
        assert sidecar_path is not None
        assert Path(sidecar_path).parent == actual.resolve()
        assert list(replacement.iterdir()) == []


def test_native_q4_profile_effective_directory_is_canonical() -> None:
    from duplex_kernel.server import native_listener_cuda_profile_effective_directory

    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp_dir:
        root = Path(temp_dir)
        requested = root / "nested" / ".." / "profile"
        assert native_listener_cuda_profile_effective_directory(requested) == str(
            (root / "profile").resolve()
        )


def _profile_close_test_app(monkeypatch, tmp_path, *, release_delay_s: float):
    import asyncio
    import threading

    import numpy as np
    from duplex_kernel.listener import GemmaListener
    from duplex_kernel.server import create_app
    from duplex_kernel.session import SessionManager
    from duplex_kernel.turn_policy import NoOpPolicy
    import duplex_kernel.server as server_module

    loop_holder: list[asyncio.AbstractEventLoop] = []
    release_callback = threading.Event()
    callback_started = threading.Event()
    callback_finished = threading.Event()
    captured_contexts: list[dict[str, object]] = []
    profile_queries = {"runtime_info": 0, "native_profile": 0, "audio_tower_timing": 0}

    class _NativeSummary:
        def native_runtime_info(self):
            profile_queries["runtime_info"] += 1
            return {"available": True, "synthetic_lifecycle_test": True}

        def native_q4_cuda_profile_summary(self, *, session_id, runtime_info):
            profile_queries["native_profile"] += 1
            del runtime_info
            context = captured_contexts[-1]
            complete = context.get("listener_step_end_ns") is not None
            return {
                "schema_version": 1,
                "enabled": True,
                "status": "complete" if complete else "incomplete",
                "session_id": session_id,
                "sampled_event_pair_count": 1,
                "completed_sample_count": int(complete),
                "pending_sample_count": int(not complete),
                "pending_frame_ids": [] if complete else [context["frame_id"]],
                "pending_samples": [] if complete else [{
                    "frame_id": context["frame_id"], "host_context": context,
                }],
                "samples": [{"frame_id": context["frame_id"], "host_context": context}]
                if complete else [],
            }

        def close(self):
            return None

    class _Backend:
        hidden_size = 4

        def encode(self, _pcm, _reference, state):
            context = getattr(state, "_native_q4_profile_host_context", None)
            if isinstance(context, dict):
                captured_contexts.append(context)

            def audio_timing_summary():
                profile_queries["audio_tower_timing"] += 1
                return {"available": True, "synthetic_lifecycle_test": True}

            state.audio_tower_cuda_timing_summary = audio_timing_summary
            state.gemma_lm_native = _NativeSummary()
            callback_started.set()

            async def release_later():
                await asyncio.sleep(release_delay_s)
                release_callback.set()

            loop = loop_holder[0]
            loop.call_soon_threadsafe(lambda: loop.create_task(release_later()))
            if not release_callback.wait(timeout=4.0):
                raise TimeoutError("profile close test callback was not released")
            callback_finished.set()
            return np.ones((1, self.hidden_size), dtype=np.float32)

    class _Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return messages[-1]["content"]

        def __call__(self, _text, **_kwargs):
            return {"input_ids": np.array([[1]], dtype=np.int64)}

    class _Engine:
        loaded = True
        enforce_production_listener = False
        tokenizer = _Tokenizer()
        default_turn_policy_factory = NoOpPolicy

        def __init__(self):
            self.listener = GemmaListener(backend=_Backend())

    def capture_loop(_engine, **_kwargs):
        loop_holder.append(asyncio.get_running_loop())
        return None

    monkeypatch.setattr(server_module, "NATIVE_LISTENER_CUDA_PROFILE_ENABLED", True)
    monkeypatch.setenv(
        server_module.NATIVE_LISTENER_CUDA_PROFILE_DIRECTORY_ENV,
        str(tmp_path),
    )
    monkeypatch.setattr(server_module, "_resolve_p7_session_voice", capture_loop)
    sessions = SessionManager()
    app = create_app(_Engine(), sessions)
    return app, sessions, callback_started, callback_finished, release_callback, profile_queries


def _run_profile_close_websocket(app, callback_started) -> list[dict[str, object]]:
    """Drive the real ASGI websocket route without a Windows portal thread."""
    import asyncio
    import base64

    import numpy as np

    async def drive() -> list[dict[str, object]]:
        incoming: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        outgoing: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        await incoming.put({"type": "websocket.connect"})

        async def receive():
            return await incoming.get()

        async def send(message):
            await outgoing.put(message)

        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "scheme": "ws",
            "path": "/v1/realtime",
            "raw_path": b"/v1/realtime",
            "query_string": b"duplex=1",
            "root_path": "",
            "headers": [(b"host", b"testserver")],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "subprotocols": [],
            "state": {},
        }
        app_task = asyncio.create_task(app(scope, receive, send))
        try:
            pcm = np.full(1280, 500, dtype=np.int16).tobytes()
            await incoming.put({
                "type": "websocket.receive",
                "text": json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }),
            })
            assert await asyncio.to_thread(callback_started.wait, 2.0)
            await incoming.put({
                "type": "websocket.receive",
                "text": json.dumps({"type": "session.close"}),
            })
            await asyncio.wait_for(app_task, timeout=8.0)
        finally:
            if not app_task.done():
                await incoming.put({"type": "websocket.disconnect", "code": 1000})
                await asyncio.gather(app_task, return_exceptions=True)
        messages = []
        while not outgoing.empty():
            messages.append(outgoing.get_nowait())
        return messages

    return asyncio.run(drive())


def test_profile_close_quiesces_real_websocket_scheduler_without_blocking_loop(
    monkeypatch, tmp_path,
) -> None:
    app, _sessions, started, finished, release, profile_queries = _profile_close_test_app(
        monkeypatch, tmp_path, release_delay_s=0.1,
    )
    _run_profile_close_websocket(app, started)

    assert finished.is_set()
    sidecars = list(tmp_path.glob("native_q4_cuda_profile_*.json"))
    assert len(sidecars) == 1
    sidecar = json.loads(sidecars[0].read_text(encoding="utf-8"))
    profile = sidecar["profile"]
    assert profile["status"] == "complete"
    assert profile["host_finalization"]["status"] == "complete"
    context = profile["samples"][0]["host_context"]
    assert context["listener_step_end_ns"] is not None
    assert context["listener_callback_end_ns"] is not None
    assert sidecar["sampled_host_tick_timings"][0]["listener_callback_end_ns"] is not None
    assert profile_queries == {"runtime_info": 1, "native_profile": 1, "audio_tower_timing": 1}
    assert release.is_set()


def test_profile_close_timeout_marks_callback_fields_unknown_and_releases_resources_later(
    monkeypatch, tmp_path,
) -> None:
    import duplex_kernel.server as server_module

    monkeypatch.setattr(server_module, "NATIVE_LISTENER_PROFILE_STOP_TIMEOUT_S", 0.01)
    app, _sessions, started, finished, release, profile_queries = _profile_close_test_app(
        monkeypatch, tmp_path, release_delay_s=2.0,
    )
    try:
        _run_profile_close_websocket(app, started)
        sidecars = list(tmp_path.glob("native_q4_cuda_profile_*.json"))
        assert len(sidecars) == 1
        sidecar = json.loads(sidecars[0].read_text(encoding="utf-8"))
        profile = sidecar["profile"]
        assert profile["status"] == "incomplete"
        assert profile["native_event_snapshot_status"] == "not_taken_while_scheduler_active"
        assert profile["host_finalization"]["status"] == "incomplete_callbacks_active"
        context = profile["active_callback_host_context"]
        assert context["status"] == "unknown_while_scheduler_active"
        assert context["listener_step_end_ns"] is None
        assert context["listener_callback_end_ns"] is None
        assert context["shared_lm_lock_wait_ms"] is None
        assert profile_queries == {"runtime_info": 0, "native_profile": 0, "audio_tower_timing": 0}
        assert not finished.is_set()
    finally:
        # The server's existing deferred cleanup owns the still-running session.
        # Releasing the synthetic callback lets that cleanup finish safely.
        release.set()
    assert finished.wait(timeout=2.0)
