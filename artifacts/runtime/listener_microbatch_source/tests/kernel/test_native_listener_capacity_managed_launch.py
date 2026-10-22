from pathlib import Path

import pytest

from duplex_kernel.microbatch.native_listener_capacity_managed_launch import (
    build_probe_docker_argv,
    container_workspace_path,
    validate_workspace_manifest_paths,
)


def _manifest(repo: Path) -> dict[str, object]:
    paths = {
        "baseline_dso": repo / "native" / "baseline.so",
        "baseline_sha": repo / "artifacts" / "baseline.sha256",
        "baseline_root": repo / ".tmp" / "baseline" / "native",
        "baseline_override_root": repo / ".tmp" / "baseline" / "overrides",
        "baseline_override": repo / ".tmp" / "baseline" / "overrides" / "file.cu",
        "candidate_dso": repo / "native" / "candidate.so",
        "candidate_sha": repo / "artifacts" / "candidate.sha256",
        "candidate_root": repo / ".tmp" / "candidate" / "native",
    }
    for key, path in paths.items():
        if key.endswith("_root"):
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
    return {
        "libraries": {
            "baseline": {
                "path": "/workspace/native/baseline.so",
                "source_manifest_path": "/workspace/artifacts/baseline.sha256",
                "source_root": "/workspace/.tmp/baseline/native",
                "source_override_root": "/workspace/.tmp/baseline/overrides",
                "source_overrides": {"./file.cu": "/workspace/.tmp/baseline/overrides/file.cu"},
            },
            "candidate": {
                "path": "/workspace/native/candidate.so",
                "source_manifest_path": "/workspace/artifacts/candidate.sha256",
                "source_root": "/workspace/.tmp/candidate/native",
            },
        }
    }


def test_container_workspace_path_maps_host_paths_without_leaking_host_prefix(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    output = repo / "artifacts" / "run one"

    assert container_workspace_path(repo / "tools" / "probe.py", repo) == "/workspace/tools/probe.py"
    assert container_workspace_path(output, repo) == "/workspace/artifacts/run one"
    with pytest.raises(ValueError, match="inside the mounted repository"):
        container_workspace_path(tmp_path / "outside.txt", repo)


def test_probe_docker_argv_uses_container_paths_and_required_isolation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = _manifest(repo)
    validate_workspace_manifest_paths(manifest, repo)
    argv = build_probe_docker_argv(
        repository_root=repo,
        image_id="sha256:0123456789abcdef",
        worker_name="native-listener-capacity-test",
        manifest_path=repo / "artifacts" / "manifest.json",
        output_dir=repo / "artifacts" / "run",
        probe_path=repo / "tools" / "probe.py",
        manifest=manifest,
        ownership_label="org.codex.task",
        ownership_value="pair-duplex-proof-3090",
    )

    image_index = argv.index("sha256:0123456789abcdef")
    container_command = argv[image_index + 1 :]
    assert container_command == [
        "-B",
        "/workspace/tools/probe.py",
        "run",
        "--manifest",
        "/workspace/artifacts/manifest.json",
        "--output-dir",
        "/workspace/artifacts/run",
    ]
    assert str(repo) not in container_command
    assert "--user" in argv and argv[argv.index("--user") + 1] == "1000:1000"
    assert "--ipc" in argv and argv[argv.index("--ipc") + 1] == "host"
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    assert "--pull=never" in argv
    assert "--gpus" in argv and argv[argv.index("--gpus") + 1] == "all"


def test_diagnostic_worker_argv_uses_bound_report_and_array_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = _manifest(repo)
    argv = build_probe_docker_argv(
        repository_root=repo,
        image_id="sha256:0123456789abcdef",
        worker_name="native-listener-capacity-test",
        manifest_path=repo / "artifacts" / "manifest.json",
        output_dir=repo / "artifacts" / "run",
        probe_path=repo / "tools" / "probe.py",
        manifest=manifest,
        ownership_label="org.codex.task",
        ownership_value="pair-duplex-proof-3090",
        worker_mode="diagnose-baseline",
    )

    image_index = argv.index("sha256:0123456789abcdef")
    assert argv[image_index + 1 :] == [
        "-B",
        "/workspace/tools/probe.py",
        "diagnose-baseline",
        "--manifest",
        "/workspace/artifacts/manifest.json",
        "--report",
        "/workspace/artifacts/run/baseline_reset_report.json",
        "--arrays",
        "/workspace/artifacts/run/baseline_reset_tensors.npz",
    ]


def test_production_prefix_worker_argv_runs_the_new_two_role_diagnostic(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = _manifest(repo)
    argv = build_probe_docker_argv(
        repository_root=repo,
        image_id="sha256:0123456789abcdef",
        worker_name="native-listener-production-prefix-test",
        manifest_path=repo / "artifacts" / "manifest.json",
        output_dir=repo / "artifacts" / "run",
        probe_path=repo / "tools" / "probe_native_listener_capacity_production.py",
        manifest=manifest,
        ownership_label="org.codex.task",
        ownership_value="pair-duplex-proof-3090",
        worker_mode="diagnose-production-prefix",
    )

    image_index = argv.index("sha256:0123456789abcdef")
    assert argv[image_index + 1 :] == [
        "-B",
        "/workspace/tools/probe_native_listener_capacity_production.py",
        "diagnose-production-prefix",
        "--manifest",
        "/workspace/artifacts/manifest.json",
        "--output-dir",
        "/workspace/artifacts/run",
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "/home/administrator/gemma-4-omni/native/lib.so"),
        ("source_manifest_path", "/home/administrator/gemma-4-omni/source.sha256"),
        ("source_root", "/home/administrator/gemma-4-omni/native"),
        ("source_override_root", "/home/administrator/gemma-4-omni/overrides"),
    ],
)
def test_workspace_manifest_rejects_host_paths(tmp_path: Path, field: str, value: str) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = _manifest(repo)
    manifest["libraries"]["baseline"][field] = value

    with pytest.raises(ValueError, match="must be container paths under /workspace"):
        validate_workspace_manifest_paths(manifest, repo)


def test_workspace_manifest_rejects_host_override_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = _manifest(repo)
    manifest["libraries"]["baseline"]["source_overrides"]["./file.cu"] = (
        "/home/administrator/gemma-4-omni/file.cu"
    )

    with pytest.raises(ValueError, match="must be container paths under /workspace"):
        validate_workspace_manifest_paths(manifest, repo)


def test_workspace_manifest_rejects_symlink_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = _manifest(repo)
    outside = tmp_path / "outside.so"
    outside.write_text("outside fixture\n", encoding="utf-8")
    original_resolve = Path.resolve

    def resolve_with_escaped_dso(path: Path, strict: bool = False) -> Path:
        if path.name == "baseline.so":
            return original_resolve(outside, strict=strict)
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve_with_escaped_dso)
    with pytest.raises(ValueError, match="resolves outside the mounted repository"):
        validate_workspace_manifest_paths(manifest, repo)


def test_workspace_manifest_requires_existing_target(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = _manifest(repo)
    (repo / "native" / "baseline.so").unlink()

    with pytest.raises(ValueError, match="manifest path does not exist"):
        validate_workspace_manifest_paths(manifest, repo)
