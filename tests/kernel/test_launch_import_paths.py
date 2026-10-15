from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from duplex_kernel.launch import _log_runtime_source_files, _prioritize_explicit_pythonpath


def test_explicit_pythonpath_stays_ahead_of_script_and_workspace_fallbacks(monkeypatch):
    overlay = "source-overlay"
    runtime = "runtime-packages"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((overlay, runtime)))
    monkeypatch.setattr(sys, "path", ["script-directory", "workspace", runtime])

    _prioritize_explicit_pythonpath("workspace", "qwen3tts-src")

    assert sys.path[:3] == [overlay, runtime, "script-directory"]
    assert sys.path.count("workspace") == 1
    assert sys.path[-1] == "qwen3tts-src"


def test_workspace_is_only_appended_when_not_already_explicit(monkeypatch):
    workspace = "workspace"
    monkeypatch.setenv("PYTHONPATH", workspace)
    monkeypatch.setattr(sys, "path", ["script-directory", workspace])

    _prioritize_explicit_pythonpath(workspace, "qwen3tts-src")

    assert sys.path == [workspace, "script-directory", "qwen3tts-src"]


def test_launch_entrypoint_imports_explicit_source_ahead_of_workspace():
    source_root = Path(os.environ.get(
        "DUPLEX_LISTENER_MICROBATCH_SOURCE_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )).resolve()
    workspace = Path(os.environ.get("GEMMA4_OMNI_WORKSPACE", str(source_root.parent))).resolve()
    launch = source_root / "duplex_kernel" / "launch.py"
    code = (
        "import runpy, sys; "
        f"sys.path[:0] = [{str(workspace)!r}, {str(source_root)!r}]; "
        f"runpy.run_path({str(launch)!r}); "
        "import duplex_kernel.listener as listener; "
        "print(listener.__file__)"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(source_root), str(workspace)))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    imported = Path(result.stdout.strip()).resolve()
    expected = (source_root / "duplex_kernel" / "listener.py").resolve()
    assert imported == expected


def test_runtime_source_log_reports_imported_module_files(caplog):
    modules = [SimpleNamespace(__file__=f"/overlay/{name}.py") for name in (
        "server", "listener", "frontend", "worker", "session", "native"
    )]

    with caplog.at_level("INFO", logger="duplex_kernel.launch"):
        _log_runtime_source_files(*modules)

    assert "server=/overlay/server.py" in caplog.text
    assert "native=/overlay/native.py" in caplog.text
