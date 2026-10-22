"""Build and validate the isolated Docker launch for the listener parity probe."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Mapping


CONTAINER_WORKSPACE = PurePosixPath("/workspace")
CONTAINER_PYTHON = "/workspace/.venv_e2e_runtime/bin/python"
_LIBRARY_ROLES = ("baseline", "candidate")
_REQUIRED_LIBRARY_PATH_FIELDS = ("path", "source_manifest_path", "source_root")
WORKER_MODES = ("run", "diagnose-baseline", "diagnose-production-prefix")


def container_workspace_path(host_path: str | Path, repository_root: str | Path) -> str:
    """Map a host path under the mounted repository to its container path."""
    root = Path(repository_root).resolve()
    path = Path(host_path).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must be inside the mounted repository: {path}") from exc
    if not relative.parts:
        return str(CONTAINER_WORKSPACE)
    return str(CONTAINER_WORKSPACE.joinpath(*relative.parts))


def _require_container_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty container path under /workspace")
    path = PurePosixPath(value)
    if not path.is_absolute() or (path != CONTAINER_WORKSPACE and CONTAINER_WORKSPACE not in path.parents):
        raise ValueError(f"{label} must be container paths under /workspace: {value!r}")
    if ".." in path.parts or "\\" in value:
        raise ValueError(f"{label} must be normalized container paths under /workspace: {value!r}")
    return value


def _resolve_manifest_path(value: str, label: str, repository_root: Path) -> Path:
    container_path = PurePosixPath(_require_container_path(value, label))
    relative = container_path.relative_to(CONTAINER_WORKSPACE)
    host_path = repository_root.joinpath(*relative.parts)
    if not host_path.exists():
        raise ValueError(f"manifest path does not exist: {value!r}")
    resolved = host_path.resolve(strict=False)
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(
            f"manifest path resolves outside the mounted repository: {value!r} -> {resolved}"
        ) from exc
    return resolved


def validate_workspace_manifest_paths(
    manifest: Mapping[str, Any], repository_root: str | Path
) -> None:
    """Require existing manifest targets to resolve inside the mounted repository."""
    root = Path(repository_root).resolve(strict=False)
    if not root.is_dir():
        raise ValueError(f"mounted repository root does not exist: {repository_root}")
    libraries = manifest.get("libraries")
    if not isinstance(libraries, Mapping):
        raise ValueError("manifest libraries must be a mapping")
    for role in _LIBRARY_ROLES:
        record = libraries.get(role)
        if not isinstance(record, Mapping):
            raise ValueError(f"manifest library record missing for {role}")
        for field in _REQUIRED_LIBRARY_PATH_FIELDS:
            value = _require_container_path(record.get(field), f"{role}.{field}")
            _resolve_manifest_path(value, f"{role}.{field}", root)
        for field in ("source_override_root",):
            value = record.get(field)
            if value is not None:
                container_value = _require_container_path(value, f"{role}.{field}")
                _resolve_manifest_path(container_value, f"{role}.{field}", root)
        overrides = record.get("source_overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError(f"{role}.source_overrides must be a mapping")
        for relative, value in overrides.items():
            if not isinstance(relative, str) or not relative.startswith("./"):
                raise ValueError(f"{role}.source_overrides keys must be relative source paths")
            container_value = _require_container_path(value, f"{role}.source_overrides[{relative!r}]")
            _resolve_manifest_path(container_value, f"{role}.source_overrides[{relative!r}]", root)


def build_probe_docker_argv(
    *,
    repository_root: str | Path,
    image_id: str,
    worker_name: str,
    manifest_path: str | Path,
    output_dir: str | Path,
    probe_path: str | Path,
    manifest: Mapping[str, Any],
    ownership_label: str,
    ownership_value: str,
    worker_mode: str = "run",
) -> list[str]:
    """Construct the offline, nonroot worker invocation with container paths."""
    root = Path(repository_root).resolve()
    if not image_id.startswith("sha256:"):
        raise ValueError("worker image must be pinned by immutable sha256 image ID")
    if not worker_name or not ownership_label or not ownership_value:
        raise ValueError("worker name and ownership label/value must be non-empty")
    if worker_mode not in WORKER_MODES:
        raise ValueError(f"unsupported listener capacity worker mode: {worker_mode!r}")
    validate_workspace_manifest_paths(manifest, root)
    container_probe = container_workspace_path(probe_path, root)
    container_manifest = container_workspace_path(manifest_path, root)
    container_output = container_workspace_path(output_dir, root)
    if worker_mode == "run":
        probe_arguments = [
            "run", "--manifest", container_manifest, "--output-dir", container_output,
        ]
    elif worker_mode == "diagnose-production-prefix":
        probe_arguments = [
            "diagnose-production-prefix", "--manifest", container_manifest,
            "--output-dir", container_output,
        ]
    else:
        probe_arguments = [
            "diagnose-baseline", "--manifest", container_manifest,
            "--report", f"{container_output}/baseline_reset_report.json",
            "--arrays", f"{container_output}/baseline_reset_tensors.npz",
        ]
    argv = [
        "docker", "run", "--name", worker_name,
        "--label", f"{ownership_label}={ownership_value}",
        "--user", "1000:1000", "--ipc", "host", "--gpus", "all", "--network", "none",
        "--env", "CUDA_VISIBLE_DEVICES=0",
        "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
        "--env", "HF_HUB_DISABLE_TELEMETRY=1",
        "--volume", f"{root.as_posix()}:/workspace", "--workdir", "/workspace", "--pull=never",
        "--entrypoint", CONTAINER_PYTHON, image_id,
        "-B", container_probe, *probe_arguments,
    ]
    image_index = argv.index(image_id)
    container_args = argv[image_index + 1 :]
    if str(root) in container_args or any(arg.startswith("/home/") for arg in container_args):
        raise ValueError("worker command contains a host path after the immutable image argument")
    return argv
