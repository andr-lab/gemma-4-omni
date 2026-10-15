"""Run the listener-capacity probe under the managed-service GPU lease.

The explicit ``--dry-run`` mode performs the same-image, CUDA-hidden preflight
and prints the worker argv without stopping the serving container. Passing
``--execute`` enables the reviewed stop/barrier/worker/restore lifecycle.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import uuid
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping, Sequence

# Direct script invocation places tools/kernel_gates on sys.path, not the
# repository root needed for the sibling tools package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from duplex_kernel.microbatch.native_listener_capacity_managed_launch import (
    CONTAINER_PYTHON,
    WORKER_MODES,
    build_probe_docker_argv,
    container_workspace_path,
    validate_workspace_manifest_paths,
)
from duplex_kernel.microbatch.probe_native_listener_capacity_parity import load_manifest


HEALTH_URL = "http://127.0.0.1:8000/health"
HEALTH_SNAPSHOTS = 2
CPU_PREFLIGHT_TIMEOUT_S = 120
WORKER_TIMEOUT_S = 1800


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    options = {"text": True, "capture_output": True, "check": False}
    options.update(kwargs)
    return subprocess.run(list(argv), **options)


def _validate_helper_contract(helpers: Any) -> None:
    """Fail before service inspection if a runner API drifted or is absent."""
    required = {
        "DEFAULT_MAINTENANCE_CONTAINER": None,
        "OWNERSHIP_LABEL": None,
        "OWNERSHIP_VALUE": None,
        "snapshot_maintenance_container": ("container_name", "repository_root"),
        "pause_maintenance_container": ("snapshot", "repository_root"),
        "wait_for_gpu_vram_release": ("maintenance_snapshot", "repository_root"),
        "cleanup_owned_container": ("container_id", "container_name", "repository_root"),
        "restore_maintenance_container": ("snapshot", "timeout_s"),
        "_health": ("url", "timeout_s"),
        "_command": ("argv", "cwd"),
    }
    missing = [name for name in required if not hasattr(helpers, name)]
    if missing:
        raise RuntimeError(f"runner helper contract is missing attributes: {missing!r}")
    for name, parameter_names in required.items():
        if parameter_names is None:
            continue
        function = getattr(helpers, name)
        if not callable(function):
            raise RuntimeError(f"runner helper {name} is not callable")
        signature = inspect.signature(function)
        try:
            signature.bind(*([object()] * len(parameter_names)))
        except TypeError:
            # Keyword-only helper parameters (restore/_health) are part of the
            # real runner API, so validate the equivalent named invocation.
            keyword_values = {parameter: object() for parameter in parameter_names}
            try:
                signature.bind(**keyword_values)
            except TypeError as exc:
                raise RuntimeError(
                    f"runner helper {name} has an incompatible signature {signature}"
                ) from exc
    if CONTAINER_PYTHON != "/workspace/.venv_e2e_runtime/bin/python":
        raise RuntimeError(f"unexpected pinned managed runtime path: {CONTAINER_PYTHON!r}")


def _health_is_idle(health: Any, label: str) -> dict[str, Any]:
    if not isinstance(health, Mapping) or health.get("status") != "ok" or health.get("engine_loaded") is not True:
        raise RuntimeError(f"{label} health snapshot is unavailable or engine is not loaded: {health!r}")
    required_zero = ("active_realtime_sessions", "listener_backlog_frames", "active_responses")
    for key in required_zero:
        if key not in health or health[key] != 0:
            raise RuntimeError(f"{label} health is not idle: {key}={health.get(key)!r}")
    for key in ("listener_callbacks_active", "thinker_audio_forward_active"):
        if key not in health or health[key] is not False:
            raise RuntimeError(f"{label} health is not idle: {key}={health.get(key)!r}")
    pool = health.get("controller_guided_talk_context_pool")
    if not isinstance(pool, Mapping) or pool.get("active") != 0:
        raise RuntimeError(f"{label} health has active Talker contexts: {pool!r}")
    return dict(health)


def _container_identity(snapshot: Mapping[str, Any], service_name: str, expected_id: str, root: Path) -> str:
    container_id = snapshot.get("id")
    inspect = snapshot.get("inspect")
    if container_id != expected_id or not isinstance(inspect, Mapping) or inspect.get("Id") != expected_id:
        raise RuntimeError("managed service immutable container ID differs from the pinned ID")
    if str(inspect.get("Name", "")).lstrip("/") not in (service_name, ""):
        raise RuntimeError("managed service snapshot name differs from the requested service")
    image_id = inspect.get("Image")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise RuntimeError("managed service image identity is not an immutable sha256 ID")
    config = inspect.get("Config")
    if not isinstance(config, Mapping) or config.get("User") != "1000:1000":
        raise RuntimeError("managed service is not running as the required nonroot UID:GID 1000:1000")
    host_config = inspect.get("HostConfig")
    if not isinstance(host_config, Mapping) or host_config.get("IpcMode") != "host":
        raise RuntimeError("managed service does not use the required host IPC namespace")
    mounts = inspect.get("Mounts")
    if not isinstance(mounts, list):
        raise RuntimeError("managed service mount snapshot is missing")
    mounted_root = root.resolve()
    workspace_mount = [mount for mount in mounts if isinstance(mount, Mapping) and mount.get("Destination") == "/workspace"]
    if len(workspace_mount) != 1:
        raise RuntimeError("managed service must have exactly one /workspace mount")
    try:
        if Path(str(workspace_mount[0].get("Source", ""))).resolve() != mounted_root:
            raise RuntimeError("managed service /workspace source differs from the selected repository")
    except OSError as exc:
        raise RuntimeError("unable to resolve managed service /workspace source") from exc
    return image_id


def _manifest_preflight_code(manifest_path: str) -> str:
    return "\n".join((
        "import json, sys, torch",
        "from pathlib import Path",
        "sys.path.insert(0, '/workspace')",
        "from duplex_kernel.microbatch.probe_native_listener_capacity_parity import load_manifest, _verify_source_manifest, sha256_file",
        "from duplex_kernel.microbatch.native_listener_capacity_managed_launch import validate_workspace_manifest_paths",
        f"manifest_path = Path({manifest_path!r})",
        "manifest = load_manifest(manifest_path)",
        "validate_workspace_manifest_paths(manifest, '/workspace')",
        "roles = {}",
        "for role, record in manifest['libraries'].items():",
        "    actual_dso = sha256_file(Path(record['path']))",
        "    if actual_dso.lower() != record['sha256'].lower(): raise RuntimeError(f'{role} DSO hash mismatch')",
        "    source = _verify_source_manifest(record)",
        "    roles[role] = {'sha256': actual_dso, 'source_manifest_sha256': source['sha256'], 'verified_source_files': source['verified_file_count']}",
        "print(json.dumps({'manifest_sha256': sha256_file(manifest_path), 'roles': roles, 'torch_cuda_initialized': bool(torch.cuda.is_initialized())}, sort_keys=True))",
    ))


def _run_checked(
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    result = run_command(argv, cwd=str(cwd), timeout=timeout, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command exited {result.returncode}: {list(argv)!r}; "
            f"stdout={(result.stdout or '')[-4000:]!r}; stderr={(result.stderr or '')[-4000:]!r}"
        )
    return result


def _cpu_preflight(
    *,
    helpers: Any,
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    repository_root: Path,
    service_id: str,
    image_id: str,
    manifest_path: Path,
    probe_path: Path,
    manifest: Mapping[str, Any],
    worker_mode: str,
) -> dict[str, Any]:
    if worker_mode not in WORKER_MODES:
        raise ValueError(f"unsupported listener capacity worker mode: {worker_mode!r}")
    probe_container = container_workspace_path(probe_path, repository_root)
    manifest_container = container_workspace_path(manifest_path, repository_root)
    cli = [
        "docker", "exec", "--user", "1000:1000", "--env", "CUDA_VISIBLE_DEVICES=", service_id,
        CONTAINER_PYTHON, "-B", probe_container, worker_mode, "--help",
    ]
    # Assert this is exactly the image running the managed service. Docker exec
    # inherits that immutable container image; image_id is recorded for audit.
    help_result = _run_checked(run_command, cli, cwd=repository_root, timeout=CPU_PREFLIGHT_TIMEOUT_S)
    code = _manifest_preflight_code(manifest_container)
    manifest_argv = [
        "docker", "exec", "--user", "1000:1000", "--env", "CUDA_VISIBLE_DEVICES=", service_id,
        CONTAINER_PYTHON, "-B", "-c", code,
    ]
    result = _run_checked(run_command, manifest_argv, cwd=repository_root, timeout=CPU_PREFLIGHT_TIMEOUT_S)
    try:
        payload = json.loads((result.stdout or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError("same-image manifest preflight did not return valid JSON") from exc
    expected_manifest_sha = _sha256_file(manifest_path)
    if payload.get("manifest_sha256", "").lower() != expected_manifest_sha.lower():
        raise RuntimeError("same-image manifest preflight hash differs from the pinned local manifest")
    if payload.get("torch_cuda_initialized") is not False:
        raise RuntimeError("CUDA became initialized during the required CPU-only same-image preflight")
    roles = payload.get("roles")
    if not isinstance(roles, Mapping):
        raise RuntimeError("same-image preflight omitted verified library roles")
    for role in ("baseline", "candidate"):
        record = manifest["libraries"][role]
        role_result = roles.get(role)
        if not isinstance(role_result, Mapping):
            raise RuntimeError(f"same-image preflight omitted {role} library verification")
        if role_result.get("sha256", "").lower() != record["sha256"].lower():
            raise RuntimeError(f"same-image {role} DSO differs from the pinned manifest")
        if role_result.get("source_manifest_sha256", "").lower() != record["source_manifest_sha256"].lower():
            raise RuntimeError(f"same-image {role} source manifest differs from the pinned manifest")
        if not isinstance(role_result.get("verified_source_files"), int) or role_result["verified_source_files"] <= 0:
            raise RuntimeError(f"same-image {role} source manifest did not verify any files")
    return {
        "managed_service_image_id": image_id,
        "same_image_exact_entrypoint_help": {
            "argv": cli,
            "exit_code": help_result.returncode,
            "stdout_tail": (help_result.stdout or "")[-2000:],
        },
        "manifest_result": payload,
    }


def _inspect_same_running_service(helpers: Any, repository_root: Path, service_id: str) -> bool:
    result = helpers._command(["docker", "inspect", service_id], cwd=repository_root)
    try:
        records = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError("restored managed service inspect returned invalid JSON") from exc
    return bool(
        isinstance(records, list) and records and records[0].get("Id") == service_id
        and records[0].get("State", {}).get("Running") is True
    )


def run_managed_probe(
    *,
    repository_root: str | Path,
    service_name: str,
    expected_service_id: str,
    manifest_path: str | Path,
    probe_path: str | Path,
    output_dir: str | Path,
    runner_helpers: Any | None = None,
    run_command: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    sleep: Callable[[float], None] | None = None,
    execute: bool = False,
    worker_mode: str = "run",
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=False)
    manifest_file = Path(manifest_path).resolve(strict=False)
    probe_file = Path(probe_path).resolve(strict=False)
    if not root.is_dir():
        raise ValueError(f"repository root does not exist: {root}")
    if worker_mode not in WORKER_MODES:
        raise ValueError(f"unsupported listener capacity worker mode: {worker_mode!r}")
    if not manifest_file.is_file() or not probe_file.is_file():
        raise ValueError("manifest and probe paths must point to existing files")
    destination = Path(output_dir).resolve(strict=False)
    container_workspace_path(manifest_file, root)
    container_workspace_path(probe_file, root)
    container_workspace_path(destination, root)
    destination.mkdir(parents=True, exist_ok=False)

    if runner_helpers is None:
        sys.path.insert(0, str(root))
        import tools.run_pair_duplex_proof_3090 as runner_helpers  # type: ignore[no-redef]
    helpers = runner_helpers
    command = run_command or _run_command
    sleeper = sleep or __import__("time").sleep
    state: dict[str, Any] = {
        "schema": "gemma4-native-listener-capacity-managed-launch/v1",
        "status": "preflight_failed",
        "execute_requested": bool(execute),
        "worker_mode": worker_mode,
        "primary_error": None,
        "maintenance": {},
        "cpu_preflight": {},
        "worker_container_paths": [],
        "worker": None,
    }
    snapshot: dict[str, Any] | None = None
    worker_name = f"native-listener-capacity-{uuid.uuid4().hex[:12]}"
    worker_started = False
    cleanup_error: str | None = None
    restore_error: str | None = None

    def persist() -> None:
        path = destination / "managed_launch_state.json"
        path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        _validate_helper_contract(helpers)
        if service_name != helpers.DEFAULT_MAINTENANCE_CONTAINER:
            raise RuntimeError("service name differs from the runner's pinned managed container")
        manifest = load_manifest(manifest_file)
        validate_workspace_manifest_paths(manifest, root)
        for role, record in manifest["libraries"].items():
            dso_relative = PurePosixPath(record["path"]).relative_to("/workspace")
            dso_path = (root.joinpath(*dso_relative.parts)).resolve()
            if not dso_path.is_file() or _sha256_file(dso_path).lower() != record["sha256"].lower():
                raise RuntimeError(f"host {role} DSO differs from the pinned manifest")
        health_snapshots: list[dict[str, Any]] = []
        for index in range(HEALTH_SNAPSHOTS):
            health = helpers._health(HEALTH_URL, timeout_s=3.0)
            health_snapshots.append(_health_is_idle(health, f"health snapshot {index + 1}"))
            if index + 1 < HEALTH_SNAPSHOTS:
                sleeper(1.0)
        state["maintenance"]["health_snapshots_before"] = health_snapshots

        snapshot_value = helpers.snapshot_maintenance_container(service_name, root)
        if not isinstance(snapshot_value, dict) or snapshot_value.get("was_running") is not True:
            raise RuntimeError("managed maintenance container was not running at snapshot")
        snapshot = snapshot_value
        image_id = _container_identity(snapshot, service_name, expected_service_id, root)
        state["maintenance"].update({
            "container_id": expected_service_id,
            "container_name": service_name,
            "image_id": image_id,
            "uid": snapshot["inspect"]["Config"]["User"],
            "ipc_mode": snapshot["inspect"]["HostConfig"]["IpcMode"],
            "workspace_mount": next(mount for mount in snapshot["inspect"]["Mounts"] if mount.get("Destination") == "/workspace"),
        })
        state["cpu_preflight"] = _cpu_preflight(
            helpers=helpers,
            run_command=command,
            repository_root=root,
            service_id=expected_service_id,
            image_id=image_id,
            manifest_path=manifest_file,
            probe_path=probe_file,
            manifest=manifest,
            worker_mode=worker_mode,
        )
        argv = build_probe_docker_argv(
            repository_root=root,
            image_id=image_id,
            worker_name=worker_name,
            manifest_path=manifest_file,
            output_dir=destination,
            probe_path=probe_file,
            manifest=manifest,
            ownership_label=helpers.OWNERSHIP_LABEL,
            ownership_value=helpers.OWNERSHIP_VALUE,
            worker_mode=worker_mode,
        )
        image_index = argv.index(image_id)
        state["worker_container_paths"] = argv[image_index + 1 :]
        state["worker_argv"] = argv
        if not execute:
            state["status"] = "dry_run_ready"
            return state

        state["status"] = "maintenance_starting"
        paused_snapshot = helpers.pause_maintenance_container(snapshot, root)
        state["maintenance"]["pause_snapshot"] = paused_snapshot
        release = helpers.wait_for_gpu_vram_release(snapshot, root)
        if not isinstance(release, Mapping) or release.get("released") is not True:
            raise RuntimeError(f"GPU ownership release barrier did not pass: {release!r}")
        state["maintenance"]["gpu_release_barrier"] = dict(release)
        worker_started = True
        worker_result = command(argv, cwd=str(root), timeout=WORKER_TIMEOUT_S, text=True, capture_output=True, check=False)
        state["worker"] = {
            "argv": argv,
            "exit_code": worker_result.returncode,
            "stdout": worker_result.stdout or "",
            "stderr": worker_result.stderr or "",
        }
        if worker_result.returncode != 0:
            raise RuntimeError(f"GPU parity probe worker exited {worker_result.returncode}")
        state["status"] = "worker_complete"
    except Exception as exc:
        state["primary_error"] = f"{type(exc).__name__}: {exc}"
        if state["status"] != "worker_complete":
            state["status"] = "worker_failed" if worker_started else "preflight_failed"
    finally:
        if execute and snapshot is not None:
            try:
                # Cleanup is ownership-label guarded in the pinned runner helper.
                helpers.cleanup_owned_container(None, worker_name, root)
                state["maintenance"]["worker_cleanup"] = "complete"
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
                state["maintenance"]["worker_cleanup_error"] = cleanup_error
            try:
                helpers.restore_maintenance_container(snapshot, timeout_s=300.0)
                state["maintenance"]["restored_same_id"] = _inspect_same_running_service(
                    helpers, root, expected_service_id,
                )
                if not state["maintenance"]["restored_same_id"]:
                    raise RuntimeError("managed service was not restored with the exact immutable ID")
                _health_is_idle(helpers._health(HEALTH_URL, timeout_s=3.0), "restored service")
                state["maintenance"]["restored_healthy"] = True
            except Exception as exc:
                restore_error = f"{type(exc).__name__}: {exc}"
                state["maintenance"]["restore_error"] = restore_error
        elif snapshot is not None:
            try:
                state["maintenance"]["restored_same_id"] = _inspect_same_running_service(
                    helpers, root, expected_service_id,
                )
                _health_is_idle(helpers._health(HEALTH_URL, timeout_s=3.0), "unchanged service")
                state["maintenance"]["restored_healthy"] = True
            except Exception as exc:
                restore_error = f"{type(exc).__name__}: {exc}"
                state["maintenance"]["restore_error"] = restore_error
        if cleanup_error or restore_error:
            if state["primary_error"] is None:
                state["primary_error"] = cleanup_error or restore_error
            state["status"] = "restoration_failed"
        elif execute and worker_started and state["primary_error"] is None:
            state["status"] = "worker_complete"
        try:
            persist()
        except Exception as exc:
            state["state_write_error"] = f"{type(exc).__name__}: {exc}"

    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/home/administrator/gemma-4-omni"))
    parser.add_argument("--service-name", default="gemma4-speech-runtime-query-film-v2-20261009")
    parser.add_argument("--expected-service-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--worker-mode", choices=WORKER_MODES, default="run",
        help="select the existing parity run, the baseline reset diagnostic, or the production-prefix capacity probe",
    )
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--execute", action="store_true", help="pause managed service, run one GPU worker, and restore it")
    operation.add_argument("--dry-run", action="store_true", help="run CPU-only same-image preflight without pausing service")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    default_probe = Path(
        "tools/kernel_gates/probe_native_listener_capacity_production.py"
        if args.worker_mode == "diagnose-production-prefix"
        else "tools/kernel_gates/probe_native_listener_capacity_parity.py"
    )
    probe_value = args.probe or default_probe
    probe = probe_value if probe_value.is_absolute() else root / probe_value
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    result = run_managed_probe(
        repository_root=root,
        service_name=args.service_name,
        expected_service_id=args.expected_service_id,
        manifest_path=manifest,
        probe_path=probe,
        output_dir=output,
        execute=args.execute,
        worker_mode=args.worker_mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["status"] in {"dry_run_ready", "worker_complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
