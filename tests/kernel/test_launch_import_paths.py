from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from duplex_kernel.launch import _log_runtime_source_files, _prioritize_explicit_pythonpath


def test_canonical_checkout_stays_ahead_of_explicit_pythonpath_and_fallbacks(monkeypatch):
    overlay = "source-overlay"
    runtime = "runtime-packages"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((overlay, runtime)))
    monkeypatch.setattr(sys, "path", ["script-directory", "workspace", runtime])

    _prioritize_explicit_pythonpath("workspace", "qwen3tts-src")

    assert Path(sys.path[0]).resolve() == Path(__file__).resolve().parents[2]
    assert sys.path[1:4] == [overlay, runtime, "script-directory"]
    assert sys.path.count("workspace") == 1
    assert sys.path[-1] == "qwen3tts-src"


def test_workspace_is_only_appended_when_not_already_explicit(monkeypatch):
    workspace = "workspace"
    monkeypatch.setenv("PYTHONPATH", workspace)
    monkeypatch.setattr(sys, "path", ["script-directory", workspace])

    _prioritize_explicit_pythonpath(workspace, "qwen3tts-src")

    assert Path(sys.path[0]).resolve() == Path(__file__).resolve().parents[2]
    assert sys.path[1:] == [workspace, "script-directory", "qwen3tts-src"]


def test_launch_entrypoint_imports_canonical_workspace_source():
    workspace = Path(__file__).resolve().parents[2]
    snapshot = workspace / "artifacts" / "runtime" / "listener_microbatch_source"
    launch = workspace / "duplex_kernel" / "launch.py"
    code = (
        "import runpy, sys; "
        f"sys.path[:0] = [{str(snapshot)!r}, {str(workspace)!r}]; "
        f"runpy.run_path({str(launch)!r}); "
        "import duplex_kernel.listener as listener; "
        "print(listener.__file__)"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(snapshot), str(workspace)))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    imported = Path(result.stdout.strip()).resolve()
    expected = (workspace / "duplex_kernel" / "listener.py").resolve()
    assert imported == expected


def test_runtime_source_log_reports_imported_module_files(caplog):
    modules = [SimpleNamespace(__file__=f"/overlay/{name}.py") for name in (
        "server", "listener", "frontend", "worker", "session", "native"
    )]

    with caplog.at_level("INFO", logger="duplex_kernel.launch"):
        _log_runtime_source_files(*modules)

    assert '"server":' in caplog.text
    assert "server.py" in caplog.text
    assert '"native_binding":' in caplog.text
    assert "native.py" in caplog.text


def test_runtime_source_diagnostics_resolve_paths_and_read_versions_without_framework_imports(
    monkeypatch,
    tmp_path,
):
    import builtins
    from importlib import metadata

    from duplex_kernel.runtime_source_diagnostics import (
        collect_runtime_source_diagnostics,
    )

    names = ("server", "listener", "frontend", "worker", "session", "native_binding")
    modules = {
        name: SimpleNamespace(__file__=str(tmp_path / f"{name}.py"))
        for name in names
    }
    versions = {"torch": "2.12.1+cu130", "transformers": "5.5.0", "tokenizers": "0.22.2"}
    monkeypatch.setattr(metadata, "version", lambda name: versions[name])
    for name, version in versions.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__version__=version))
    original_import = builtins.__import__

    def forbid_framework_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "transformers", "tokenizers"}:
            raise AssertionError(f"diagnostics imported framework package {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_framework_import)
    diagnostics = collect_runtime_source_diagnostics(modules)

    for name in names:
        assert diagnostics[name] == str((tmp_path / f"{name}.py").resolve())
    assert diagnostics["python_executable"] == str(Path(sys.executable).resolve())
    assert diagnostics["versions"] == versions


def test_runtime_source_diagnostics_preserve_imported_torch_build_tag(monkeypatch):
    import sys
    from importlib import metadata

    from duplex_kernel.runtime_source_diagnostics import _distribution_version

    monkeypatch.setattr(metadata, "version", lambda _name: "2.12.1")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.12.1+cu130"))

    assert _distribution_version("torch") == "2.12.1+cu130"


def test_canonical_workspace_source_precedes_artifact_package_on_launch(tmp_path):
    workspace = Path(__file__).resolve().parents[2]
    snapshot = tmp_path / "artifacts" / "runtime" / "old_snapshot"
    shadow_package = snapshot / "duplex_kernel"
    shadow_package.mkdir(parents=True)
    (shadow_package / "__init__.py").write_text("# stale snapshot\n", encoding="utf-8")
    (shadow_package / "listener.py").write_text("SOURCE = 'snapshot'\n", encoding="utf-8")
    vendor = snapshot / "runtimepackages_original"
    vendor.mkdir()

    launch = workspace / "duplex_kernel" / "launch.py"
    code = (
        "import runpy, sys; "
        f"sys.path[:0] = [{str(snapshot)!r}, {str(vendor)!r}, {str(workspace)!r}]; "
        f"runpy.run_path({str(launch)!r}); "
        "import duplex_kernel.listener as listener; "
        "print(listener.__file__)"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(snapshot), str(vendor), str(workspace)))
    env.pop("DUPLEX_LISTENER_MICROBATCH_SOURCE_ROOT", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    imported = Path(result.stdout.strip()).resolve()
    assert imported == (workspace / "duplex_kernel" / "listener.py").resolve()


def test_speech_launcher_uses_canonical_source_and_keeps_container_isolation():
    launcher = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "start_speech_service_3090.sh"
    ).read_text(encoding="utf-8")

    assert 'readonly LISTENER_PYTHONPATH="/workspace:$LISTENER_VENDOR_PACKAGES_CONTAINER"' in launcher
    assert '--workdir /workspace \\' in launcher
    assert '"$PYTHON" /workspace/duplex_kernel/launch.py' in launcher
    assert '[[ "$workdir" == "/workspace" ]]' in launcher
    assert "/workspace/duplex_kernel/launch.py" in launcher
    assert "DUPLEX_LISTENER_MICROBATCH_SOURCE_ROOT" not in launcher
    assert "LISTENER_SOURCE_ROOT_RELATIVE" not in launcher
    assert "required persistent listener source module" not in launcher
    assert "replace" in launcher.lower()
    assert "DUPLEX_SERVICE_CONTAINER to a new name" in launcher
    assert "DUPLEX_RESPONSE_TALK_CONTEXT_CAPACITY" in launcher
    assert "--response-talk-context-capacity" in launcher
    assert "RESPONSE_TALK_CONTEXT_CAPACITY" in launcher
    assert 'readonly GPU_ADMISSION_CAPACITY="${DUPLEX_GPU_ADMISSION_CAPACITY:-1}"' in launcher
    assert "DUPLEX_GPU_ADMISSION_CAPACITY" in launcher
    assert 'readonly READY_FRAME_SCHEDULER_DISPATCH="${GEMMA4_OMNI_READY_FRAME_SCHEDULER_DISPATCH:-1}"' in launcher
    assert 'GEMMA4_OMNI_FIRST_INPUT_SCHEDULER_ANCHOR=0' in launcher
    assert 'GEMMA4_OMNI_READY_FRAME_SCHEDULER_DISPATCH="$READY_FRAME_SCHEDULER_DISPATCH"' in launcher
    assert 'check_env GEMMA4_OMNI_READY_FRAME_SCHEDULER_DISPATCH "$READY_FRAME_SCHEDULER_DISPATCH"' in launcher

    assert "--user 1000:1000" in launcher
    assert "--ipc=host" in launcher
    assert "--publish 8000:8000" in launcher
    assert '--volume "$REPO_ROOT:/workspace"' in launcher
    assert "--privileged" not in launcher
    assert "--network host" not in launcher


def test_launch_exposes_default_one_gpu_admission_capacity():
    launch = (Path(__file__).resolve().parents[2] / "duplex_kernel" / "launch.py").read_text(
        encoding="utf-8"
    )

    assert 'gpu_admission_capacity_default = os.environ.get("DUPLEX_GPU_ADMISSION_CAPACITY", "1")' in launch
    assert '"--gpu-admission-capacity"' in launch
    assert 'if args.gpu_admission_capacity < 1:' in launch
