import json
import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from duplex_kernel.microbatch.native_listener_capacity_probe_contract import TOLERANCES
from duplex_kernel.microbatch.native_listener_capacity_managed_runner import (
    CONTAINER_PYTHON,
    build_parser,
    run_managed_probe,
)


SERVICE = "gemma4-speech-runtime-query-film-v2-20261009"
SERVICE_ID = "3b5880d65c5d521f365e02953706cbf275275677594288328730c5410d2ee5f6"


def _repo_fixture(root: Path) -> tuple[Path, Path, Path]:
    (root / "tools" / "kernel_gates").mkdir(parents=True)
    probe = root / "tools" / "kernel_gates" / "probe.py"
    probe.write_text("# fixture\n", encoding="utf-8")
    paths = {
        "baseline_dso": root / "native" / "baseline.so",
        "baseline_sha": root / "artifacts" / "baseline.sha256",
        "baseline_root": root / ".tmp" / "baseline" / "native",
        "baseline_override_root": root / ".tmp" / "baseline" / "overrides",
        "baseline_override": root / ".tmp" / "baseline" / "overrides" / "file.cu",
        "candidate_dso": root / "native" / "candidate.so",
        "candidate_sha": root / "artifacts" / "candidate.sha256",
        "candidate_root": root / ".tmp" / "candidate" / "native",
    }
    for key, path in paths.items():
        if key.endswith("_root"):
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
    baseline_dso_sha = hashlib.sha256(paths["baseline_dso"].read_bytes()).hexdigest()
    candidate_dso_sha = hashlib.sha256(paths["candidate_dso"].read_bytes()).hexdigest()
    manifest = {
        "schema": "gemma4-native-listener-capacity-manifest/v1",
        "runtime": {
            "abi_version": 3,
            "attention_backend_id": 3,
            "rmsnorm_backend_id": 2,
            "max_seq_len": 16384,
            "stream_q_len": 4,
            "enable_cuda_graph": False,
        },
        "tolerances": TOLERANCES,
        "libraries": {
            "baseline": {
                "path": "/workspace/native/baseline.so",
                "sha256": baseline_dso_sha,
                "source_manifest_path": "/workspace/artifacts/baseline.sha256",
                "source_manifest_sha256": "b" * 64,
                "source_root": "/workspace/.tmp/baseline/native",
                "source_override_root": "/workspace/.tmp/baseline/overrides",
                "source_overrides": {"./file.cu": "/workspace/.tmp/baseline/overrides/file.cu"},
            },
            "candidate": {
                "path": "/workspace/native/candidate.so",
                "sha256": candidate_dso_sha,
                "source_manifest_path": "/workspace/artifacts/candidate.sha256",
                "source_manifest_sha256": "d" * 64,
                "source_root": "/workspace/.tmp/candidate/native",
            },
        },
    }
    manifest_path = root / "artifacts" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, probe, root / "artifacts" / "run"


class _FakeRunner:
    DEFAULT_MAINTENANCE_CONTAINER = SERVICE
    OWNERSHIP_LABEL = "org.codex.task"
    OWNERSHIP_VALUE = "pair-duplex-proof-3090"

    def __init__(self, root: Path, events: list[str]) -> None:
        self.root = root
        self.events = events
        self.snapshot = {
            "id": SERVICE_ID,
            "name": SERVICE,
            "was_running": True,
            "inspect": {
                "Id": SERVICE_ID,
                "Image": "sha256:" + "e" * 64,
                "Config": {"Image": "managed-image", "User": "1000:1000"},
                "HostConfig": {"IpcMode": "host", "NetworkMode": "bridge"},
                "Mounts": [{"Source": str(root), "Destination": "/workspace", "RW": True}],
            },
        }

    def _health(self, url: str, timeout_s: float = 3.0) -> dict[str, Any]:
        self.events.append("health")
        return {
            "status": "ok",
            "engine_loaded": True,
            "active_realtime_sessions": 0,
            "listener_backlog_frames": 0,
            "active_responses": 0,
            "listener_callbacks_active": False,
            "thinker_audio_forward_active": False,
            "controller_guided_talk_context_pool": {"active": 0},
        }

    def snapshot_maintenance_container(self, container_name: str, repository_root: Path) -> dict[str, Any]:
        self.events.append("snapshot")
        return self.snapshot

    def pause_maintenance_container(self, snapshot: dict[str, Any], repository_root: Path) -> dict[str, Any]:
        self.events.append("pause")
        snapshot["test_pause_mutation"] = "same-snapshot"
        return {"compute_apps": [], "gpus": [{"index": 0, "memory_free_mib": 20480}]}

    def wait_for_gpu_vram_release(self, snapshot: dict[str, Any], repository_root: Path) -> dict[str, Any]:
        self.events.append("barrier")
        assert snapshot.get("test_pause_mutation") == "same-snapshot"
        return {"released": True, "observations": []}

    def cleanup_owned_container(self, container_id: str | None, container_name: str, repository_root: Path) -> None:
        self.events.append("cleanup")

    def restore_maintenance_container(self, snapshot: dict[str, Any], timeout_s: float = 300.0) -> None:
        self.events.append("restore")

    def _command(self, argv: list[str], cwd: Path) -> SimpleNamespace:
        self.events.append("inspect_restored")
        return SimpleNamespace(stdout=json.dumps([{"Id": SERVICE_ID, "State": {"Running": True}}]))


def _fake_run_command(events: list[str], *, preflight_payload: dict[str, Any], worker_exit: int = 0):
    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if argv[:2] == ["docker", "exec"]:
            if "--help" in argv:
                events.append("cli_help")
                return subprocess.CompletedProcess(argv, 0, stdout="usage: probe\n", stderr="")
            if "-c" in argv:
                events.append("manifest_preflight")
                return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(preflight_payload) + "\n", stderr="")
        if argv[:2] == ["docker", "run"]:
            events.append("worker")
            return subprocess.CompletedProcess(argv, worker_exit, stdout="worker stdout\n", stderr="worker stderr\n")
        raise AssertionError(f"unexpected command: {argv!r}")

    return run


def _call(root: Path, *, events: list[str], worker_exit: int = 0, execute: bool = True) -> dict[str, Any]:
    manifest_path, probe_path, output_dir = _repo_fixture(root)
    runner = _FakeRunner(root, events)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = {
        role: {
            "sha256": record["sha256"],
            "source_manifest_sha256": record["source_manifest_sha256"],
            "verified_source_files": 3,
        }
        for role, record in manifest["libraries"].items()
    }
    preflight_payload = {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "roles": roles,
        "torch_cuda_initialized": False,
    }
    return run_managed_probe(
        repository_root=root,
        service_name=SERVICE,
        expected_service_id=SERVICE_ID,
        manifest_path=manifest_path,
        probe_path=probe_path,
        output_dir=output_dir,
        runner_helpers=runner,
        run_command=_fake_run_command(events, preflight_payload=preflight_payload, worker_exit=worker_exit),
        sleep=lambda _seconds: None,
        execute=execute,
    )


def _call_worker_mode(root: Path, *, events: list[str], worker_mode: str, execute: bool = False) -> dict[str, Any]:
    manifest_path, probe_path, output_dir = _repo_fixture(root)
    runner = _FakeRunner(root, events)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = {
        role: {
            "sha256": record["sha256"],
            "source_manifest_sha256": record["source_manifest_sha256"],
            "verified_source_files": 3,
        }
        for role, record in manifest["libraries"].items()
    }
    preflight_payload = {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "roles": roles,
        "torch_cuda_initialized": False,
    }
    return run_managed_probe(
        repository_root=root,
        service_name=SERVICE,
        expected_service_id=SERVICE_ID,
        manifest_path=manifest_path,
        probe_path=probe_path,
        output_dir=output_dir,
        runner_helpers=runner,
        run_command=_fake_run_command(events, preflight_payload=preflight_payload),
        sleep=lambda _seconds: None,
        execute=execute,
        worker_mode=worker_mode,
    )


def test_full_host_wrapper_dryrun_covers_preflight_pause_barrier_worker_and_restore(tmp_path: Path) -> None:
    events: list[str] = []
    state = _call(tmp_path / "repo", events=events)

    assert state["status"] == "worker_complete"
    assert state["primary_error"] is None
    assert state["cpu_preflight"]["same_image_exact_entrypoint_help"]["exit_code"] == 0
    assert state["cpu_preflight"]["manifest_result"]["torch_cuda_initialized"] is False
    assert state["maintenance"]["gpu_release_barrier"]["released"] is True
    assert state["maintenance"]["restored_same_id"] is True
    assert state["worker_container_paths"] == [
        "-B",
        "/workspace/tools/kernel_gates/probe.py",
        "run",
        "--manifest",
        "/workspace/artifacts/manifest.json",
        "--output-dir",
        "/workspace/artifacts/run",
    ]
    assert events.index("cli_help") < events.index("manifest_preflight") < events.index("pause")
    assert events.index("pause") < events.index("barrier") < events.index("worker")
    assert events.index("worker") < events.index("cleanup") < events.index("restore")
    assert CONTAINER_PYTHON == "/workspace/.venv_e2e_runtime/bin/python"


def test_worker_failure_still_cleans_owned_container_and_restores_same_service(tmp_path: Path) -> None:
    events: list[str] = []
    state = _call(tmp_path / "repo", events=events, worker_exit=1)

    assert state["status"] == "worker_failed"
    assert "exited 1" in state["primary_error"]
    assert events.index("worker") < events.index("cleanup") < events.index("restore")
    assert state["maintenance"]["restored_same_id"] is True


def test_nonexecute_dryrun_builds_container_argv_without_pause_or_worker(tmp_path: Path) -> None:
    events: list[str] = []
    state = _call(tmp_path / "repo", events=events, execute=False)

    assert state["status"] == "dry_run_ready"
    assert "pause" not in events and "barrier" not in events and "worker" not in events
    assert state["maintenance"]["restored_same_id"] is True


def test_command_line_requires_explicit_dry_run_or_execute_mode() -> None:
    parser = build_parser()
    base = ["--expected-service-id", SERVICE_ID, "--manifest", "manifest.json", "--output-dir", "out"]
    dry = parser.parse_args([*base, "--dry-run"])
    execute = parser.parse_args([*base, "--execute"])
    assert dry.dry_run is True and dry.execute is False
    assert execute.execute is True and execute.dry_run is False


def test_diagnostic_mode_is_cpu_preflighted_and_maps_unique_artifacts(tmp_path: Path) -> None:
    events: list[str] = []
    state = _call_worker_mode(
        tmp_path / "repo", events=events, worker_mode="diagnose-baseline", execute=False,
    )

    assert state["status"] == "dry_run_ready"
    assert state["cpu_preflight"]["same_image_exact_entrypoint_help"]["argv"][-2:] == [
        "diagnose-baseline", "--help",
    ]
    assert state["worker_container_paths"] == [
        "-B", "/workspace/tools/kernel_gates/probe.py", "diagnose-baseline",
        "--manifest", "/workspace/artifacts/manifest.json",
        "--report", "/workspace/artifacts/run/baseline_reset_report.json",
        "--arrays", "/workspace/artifacts/run/baseline_reset_tensors.npz",
    ]
    assert state["cpu_preflight"]["manifest_result"]["torch_cuda_initialized"] is False


def test_production_prefix_mode_is_cpu_preflighted_and_invokes_two_role_runner(tmp_path: Path) -> None:
    events: list[str] = []
    state = _call_worker_mode(
        tmp_path / "repo", events=events,
        worker_mode="diagnose-production-prefix", execute=False,
    )

    assert state["status"] == "dry_run_ready"
    assert state["cpu_preflight"]["same_image_exact_entrypoint_help"]["argv"][-2:] == [
        "diagnose-production-prefix", "--help",
    ]
    assert state["worker_container_paths"] == [
        "-B", "/workspace/tools/kernel_gates/probe.py", "diagnose-production-prefix",
        "--manifest", "/workspace/artifacts/manifest.json",
        "--output-dir", "/workspace/artifacts/run",
    ]
    assert "pause" not in events and "worker" not in events
    assert state["cpu_preflight"]["manifest_result"]["torch_cuda_initialized"] is False


def test_command_line_worker_mode_defaults_to_full_parity_and_accepts_diagnostics() -> None:
    parser = build_parser()
    base = ["--expected-service-id", SERVICE_ID, "--manifest", "manifest.json", "--output-dir", "out"]
    assert parser.parse_args([*base, "--dry-run"]).worker_mode == "run"
    assert parser.parse_args([*base, "--worker-mode", "diagnose-baseline", "--dry-run"]).worker_mode == "diagnose-baseline"
    assert parser.parse_args([*base, "--worker-mode", "diagnose-production-prefix", "--dry-run"]).worker_mode == "diagnose-production-prefix"


def test_managed_runner_rejects_unreviewed_worker_mode_before_filesystem_mutation(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    manifest_path, probe_path, output_dir = _repo_fixture(root)
    with pytest.raises(ValueError, match="unsupported listener capacity worker mode"):
        run_managed_probe(
            repository_root=root,
            service_name=SERVICE,
            expected_service_id=SERVICE_ID,
            manifest_path=manifest_path,
            probe_path=probe_path,
            output_dir=output_dir,
            worker_mode="worker",
        )
    assert not output_dir.exists()
