from __future__ import annotations

import ast
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable
from types import SimpleNamespace

import numpy as np

from tools import run_pair_duplex_proof_3090 as pair_runner


def _load_archive_start_from_server() -> tuple[Callable[..., bool], dict[str, Any]]:
    server_path = Path(__file__).resolve().parents[1] / "duplex_kernel" / "server.py"
    tree = ast.parse(server_path.read_text(encoding="utf-8"), filename=str(server_path))
    wanted = {"_submit_or_defer_audio_history_archive", "start_audio_history_archive"}
    nodes = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    }
    assert "start_audio_history_archive" in nodes
    namespace: dict[str, Any] = {
        "Any": Any,
        "Callable": Callable,
        "np": np,
        "time": time,
    }
    module = ast.Module(
        body=[nodes[name] for name in sorted(wanted) if name in nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(server_path), "exec"), namespace)
    return namespace["start_audio_history_archive"], namespace


def _load_response_runtime_metric_projection() -> tuple[Callable[..., None], Callable[..., dict[str, Any]], dict[str, Any]]:
    server_path = Path(__file__).resolve().parents[1] / "duplex_kernel" / "server.py"
    tree = ast.parse(server_path.read_text(encoding="utf-8"), filename=str(server_path))
    names = {"update_response_runtime_metrics", "runtime_metrics_event"}
    nodes = {
        item.name: item for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name in names
    }
    assert set(nodes) == names
    from duplex_kernel import server

    namespace: dict[str, Any] = dict(vars(server))
    namespace.update({
        "response_runtime_metrics": {},
        "response_runtime_metrics_lock": threading.Lock(),
        "sess": SimpleNamespace(scheduler=SimpleNamespace(stats=lambda: {"source": "test"})),
    })
    module = ast.Module(body=[nodes[name] for name in sorted(names)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(server_path), "exec"), namespace)
    return (
        namespace["update_response_runtime_metrics"],
        namespace["runtime_metrics_event"],
        namespace,
    )


class _Future:
    def add_done_callback(self, callback: Callable[..., Any]) -> None:
        self.callback = callback


class _Session:
    session_id = "session-proof"

    def __init__(self, snapshot: dict[str, Any]) -> None:
        self._snapshot = snapshot
        self._generation_lock = threading.RLock()
        self.submitted_archive_ids: list[int] = []
        self.failures: list[tuple[int, str]] = []

    def get_audio_history_archive(self, archive_id: int) -> dict[str, Any] | None:
        return self._snapshot if archive_id == self._snapshot["archive_id"] else None

    def mark_audio_history_archive_submitted(self, archive_id: int, **_values: Any) -> bool:
        self.submitted_archive_ids.append(archive_id)
        self._snapshot["status"] = "submitted"
        return True

    def fail_audio_history_archive(self, archive_id: int, reason: str) -> bool:
        self.failures.append((archive_id, reason))
        return True


def _exercise_actual_archive_start(*, defer_offline: bool) -> dict[str, Any]:
    start_archive, namespace = _load_archive_start_from_server()
    pcm = np.linspace(-0.2, 0.2, 1280, dtype=np.float32)
    snapshot = {
        "archive_id": 7,
        "pcm": pcm.copy(),
        "first_frame_id": 12,
        "last_frame_id": 12,
        "source": "post_response_asr",
        "response_created_at_monotonic_ns": 987654321,
        "reserved_at_monotonic_ns": 987654400,
        "created_at_monotonic_ns": 987654400,
        "sample_rate": 16000,
        "transcript": None,
        "assistant_text": "",
        "status": "pending",
        "submitted_at_monotonic_ns": None,
        "completed_at_monotonic_ns": None,
    }
    session = _Session(snapshot)
    submitted: list[np.ndarray] = []
    metrics: list[tuple[str, str, dict[str, Any]]] = []
    futures: dict[int, _Future] = {}
    namespace.update({
        "sess": session,
        "audio_archive_lock": threading.Lock(),
        "audio_archive_futures": futures,
        "CPU_AUDIO_ARCHIVE_WORKER": type("Worker", (), {
            "submit": staticmethod(lambda value: (submitted.append(value.copy()), _Future())[1])
        })(),
        "response_runtime_metrics_lock": threading.Lock(),
        "response_runtime_metrics": {},
        "audio_history_archive_defer_offline": defer_offline,
        "update_response_runtime_metrics": lambda response_id, phase, values: metrics.append(
            (response_id, phase, values)
        ),
    })
    result = start_archive(7, "resp-proof")
    return {
        "result": result,
        "submitted": submitted,
        "metrics": metrics,
        "session": session,
        "snapshot": snapshot,
        "pcm": pcm,
    }


def test_proof_defer_skips_actual_archive_worker_and_preserves_pending_pcm() -> None:
    result = _exercise_actual_archive_start(defer_offline=True)

    assert result["submitted"] == []
    assert result["result"] is False
    assert result["session"].submitted_archive_ids == []
    assert result["session"].failures == []
    assert np.array_equal(result["snapshot"]["pcm"], result["pcm"])
    assert result["snapshot"]["status"] == "pending"
    assert result["snapshot"]["transcript"] is None
    assert result["snapshot"]["assistant_text"] == ""
    assert result["metrics"] == [(
        "resp-proof",
        "audio_history_archive",
        {
            "status": "deferred_offline",
            "archive_id": 7,
            "session_id": "session-proof",
            "source": "post_response_asr",
            "response_created_at_monotonic_ns": 987654321,
            "reserved_at_monotonic_ns": 987654400,
            "deferred_at_monotonic_ns": result["metrics"][0][2]["deferred_at_monotonic_ns"],
            "input_first_frame_id": 12,
            "input_last_frame_id": 12,
            "input_audio_samples": 1280,
            "sample_rate": 16000,
            "transcript_committed": False,
        },
    )]
    assert type(result["metrics"][0][2]["deferred_at_monotonic_ns"]) is int
    assert result["metrics"][0][2]["deferred_at_monotonic_ns"] > 987654400


def test_default_actual_archive_path_still_submits_worker() -> None:
    result = _exercise_actual_archive_start(defer_offline=False)

    assert result["result"] is True
    assert len(result["submitted"]) == 1
    assert np.array_equal(result["submitted"][0], result["pcm"])
    assert result["session"].submitted_archive_ids == [7]
    assert result["metrics"][-1][2]["status"] == "submitted"


def test_deferred_archive_transcript_flag_survives_projection_into_postrun_gate() -> None:
    result = _exercise_actual_archive_start(defer_offline=True)
    update_metrics, runtime_metrics_event, _namespace = _load_response_runtime_metric_projection()
    emitted = result["metrics"][0][2]
    update_metrics("speak-0", "audio_history_archive", emitted)
    stored = runtime_metrics_event("speak-0")["audio_history_archive"]
    assert stored["transcript_committed"] is False

    responses = [
        {"response_id": f"speak-{index}", "runtime_metrics": {"audio_history_archive": dict(stored)}}
        for index in range(3)
    ]
    responses.extend([
        {"response_id": f"backchannel-{index}", "runtime_metrics": {"audio_history_archive": {
            "source": "heard_action_say_v2",
            "status": "skipped_backpressure",
        }}}
        for index in range(2)
    ])
    report = {
        "participants": [
            {"speaker": speaker, "session_id": f"session-{speaker}", "session_updated": {
                "audio_history_archive_defer_offline": True,
                "audio_history_archive_cpu_enabled": False,
            }}
            for speaker in ("A", "B")
        ],
        "responses": responses,
    }
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp_dir:
        report_path = Path(temp_dir) / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        gate = pair_runner.archive_defer_postrun_attestation(report_path)
        assert gate["status"] == "PASS", gate
        assert gate["deferred_archive_count"] == 3

        missing = dict(stored)
        missing.pop("transcript_committed")
        responses[0]["runtime_metrics"]["audio_history_archive"] = missing
        report_path.write_text(json.dumps(report), encoding="utf-8")
        failed = pair_runner.archive_defer_postrun_attestation(report_path)
        assert failed["status"] == "FAIL"
