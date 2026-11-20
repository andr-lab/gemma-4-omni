#!/usr/bin/env python3
"""Run three acceptance sessions and collect the evidence used by the simulator.

Run this on Mondaini from the repository checkout. By default it uses
``docker exec`` against the existing attempt9 container; when run inside that
container (or with ``--execution direct``), it invokes the simulator with the
current Python interpreter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


DEFAULT_CONTAINER = "gemma4-listener-attempt9-implementation-fixes-20261024"
DEFAULT_MANIFEST = "/workspace/work/gpu_forensics/fixtures_16k_final/fixture_manifest_verified.json"
DEFAULT_CA_CERT = "/workspace/work/gpu_forensics/fixtures_16k_final/mondaini-server.crt"
DEFAULT_BUDGET_CONFIG = "/workspace/work/gpu_forensics/fixtures_16k_final/acceptance_budget_config.json"
DEFAULT_URL = "wss://172.17.0.1:8443/v1/realtime?duplex=1"
DEFAULT_TELEMETRY_URL = "http://127.0.0.1:8000/debug/runtime-telemetry"
DEFAULT_HEALTH_URL = "http://127.0.0.1:8000/health"
DEFAULT_PYTHON = "/workspace/.venv_e2e_runtime/bin/python"
PYTHONPATH_IN_CONTAINER = (
    "/workspace:/workspace/tools:"
    "/workspace/artifacts/runtime/system_convergence_final_20261023_rerun6/source"
)
INSTRUCTIONS = (
    "Rispondi in italiano in modo naturale e coerente. Per la richiesta sulla "
    "backpressure, spiega il flusso con l esempio numerico fornito, distingui "
    "il ritardo accumulato dalla perdita dei pacchetti e proponi misure pratiche."
)
try:
    from .acceptance_telemetry import (
        SAMPLE_INTERVAL_SECONDS,
        RuntimeTelemetrySampler,
        http_json,
        read_json,
        utc_now,
        write_json,
    )
except ImportError:
    from acceptance_telemetry import (
        SAMPLE_INTERVAL_SECONDS,
        RuntimeTelemetrySampler,
        http_json,
        read_json,
        utc_now,
        write_json,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _in_container() -> bool:
    return Path("/.dockerenv").exists() or bool(os.environ.get("container"))


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _container_running(container: str) -> bool:
    if not _docker_available():
        return False
    result = subprocess.run(
        ["docker", "inspect", "--format={{.State.Running}}", container],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def _choose_execution(requested: str, container: str) -> str:
    if requested != "auto":
        if requested == "docker" and not _container_running(container):
            raise RuntimeError(f"Docker container is unavailable or stopped: {container}")
        return requested
    if _in_container():
        return "direct"
    if _container_running(container):
        return "docker"
    return "direct"


def _host_path(value: str, repo_root: Path) -> Path:
    if value.startswith("/workspace/"):
        return repo_root / Path(value.removeprefix("/workspace/"))
    return Path(value).expanduser().resolve()


def _container_path(path: Path, repo_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Docker output/input paths must be under the repository: {path}") from exc
    return "/workspace/" + relative.as_posix()


def _container_argument(value: str, repo_root: Path) -> str:
    if value.startswith("/workspace/"):
        return value
    return _container_path(Path(value).expanduser().resolve(), repo_root)


def _load_budgets(config_value: str, execution: str, repo_root: Path) -> dict[str, float]:
    path = _host_path(config_value, repo_root)
    if not path.is_file():
        if execution == "direct":
            raise FileNotFoundError(f"acceptance VRAM budget config was not found: {path}")
        raise FileNotFoundError(f"acceptance VRAM budget config was not found: {config_value}")
    payload = read_json(path)
    budgets = payload.get("cli") if isinstance(payload, Mapping) else None
    if not isinstance(budgets, Mapping):
        raise ValueError(f"budget config has no cli object: {path}")
    names = (
        "vram_peak_over_baseline_budget_mib",
        "vram_post_drain_growth_budget_mib",
        "vram_reserved_peak_growth_budget_mib",
        "vram_reserved_post_drain_growth_budget_mib",
    )
    values = {name: float(budgets[name]) for name in names}
    if any(value < 0 for value in values.values()):
        raise ValueError(f"VRAM budgets must be non-negative: {path}")
    return values


def _simulator_command(
    *,
    execution: str,
    container: str,
    python_bin: str,
    repo_root: Path,
    fixture_manifest: str,
    ca_cert: str,
    output_dir: Path,
    budgets: Mapping[str, float],
) -> list[str]:
    if execution == "docker":
        root_arg = _container_path(output_dir, repo_root)
        simulator_path = "/workspace/tools/real_dialogue_simulator.py"
        python_command = python_bin
        prefix = [
            "docker", "exec", "--user", "1000:1000", "--workdir", "/workspace",
            "--env", f"PYTHONPATH={PYTHONPATH_IN_CONTAINER}", container,
        ]
        manifest_arg = _container_argument(fixture_manifest, repo_root)
        ca_arg = _container_argument(ca_cert, repo_root)
    else:
        root_arg = str(output_dir.resolve())
        simulator_path = str(repo_root / "tools" / "real_dialogue_simulator.py")
        python_command = sys.executable
        prefix = []
        manifest_arg = str(_host_path(fixture_manifest, repo_root))
        ca_arg = str(_host_path(ca_cert, repo_root))

    command = prefix + [
        python_command,
        simulator_path,
        "--fixture-manifest", manifest_arg,
        "--mode", "acceptance",
        "--response-profile", "long",
        "--output-dir", root_arg,
        "--server-log", root_arg + "/server.log",
        "--server-log-window", root_arg + "/server-log-window.json",
        "--runtime-telemetry", root_arg + "/runtime_telemetry.json",
        "--url", DEFAULT_URL,
        "--ca-cert", ca_arg,
        "--tls-server-name", "mondaini",
        "--noise-rms", "0.002",
        "--seed", "407",
        "--playback-buffer-seconds", "4.0",
        "--pacing-tolerance-ms", "60.0",
        "--handshake-timeout-s", "60.0",
        "--response-timeout-s", "300.0",
        "--playback-timeout-s", "300.0",
        "--cancellation-timeout-s", "45.0",
        "--cancellation-budget-ms", "250.0",
        "--instructions", INSTRUCTIONS,
    ]
    for name, value in budgets.items():
        command.extend((f"--{name.replace('_', '-')}", str(value)))
    return command


def _parse_log_timestamp(line: str) -> datetime | None:
    parts = line.split(maxsplit=2)
    if not parts:
        return None
    token = parts[0]
    if len(token) == 10 and len(parts) > 1 and token[4:5] == "-":
        token += "T" + parts[1]
    if len(token) < 19 or token[4:5] != "-" or token[10:11] != "T":
        return None
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _capture_from_source(source: Path, destination: Path, since: datetime, until: datetime) -> dict[str, Any]:
    if not source.is_file():
        destination.write_bytes(b"")
        return {"source": str(source), "exit_code": 1, "timestamped_records": 0}
    included: list[str] = []
    timestamped = 0
    current_in_window = False
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = _parse_log_timestamp(line)
        if parsed is not None:
            timestamped += 1
            current_in_window = since <= parsed <= until
        if current_in_window:
            included.append(line)
    destination.write_text("\n".join(included) + ("\n" if included else ""), encoding="utf-8")
    return {"source": str(source), "exit_code": 0, "timestamped_records": timestamped}


def _capture_server_log(
    *,
    destination: Path,
    container: str,
    since_text: str,
    until_text: str,
    source_path: str | None,
) -> dict[str, Any]:
    since = _parse_log_timestamp(since_text)
    until = _parse_log_timestamp(until_text)
    if since is None or until is None:
        destination.write_bytes(b"")
        return {"source": "unavailable", "exit_code": 1, "error": "invalid run-window timestamp"}
    if _docker_available():
        with destination.open("wb") as output:
            result = subprocess.run(
                ["docker", "logs", "--timestamps", "--since", since_text, "--until", until_text, container],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode == 0:
            return {"source": "docker logs --timestamps --since/--until", "exit_code": 0}
        docker_error = result.returncode
    else:
        docker_error = None
    fallback = source_path or os.environ.get("GEMMA_SERVER_LOG")
    if fallback:
        result = _capture_from_source(Path(fallback).expanduser().resolve(), destination, since, until)
        result["docker_logs_exit_code"] = docker_error
        return result
    destination.write_bytes(b"")
    return {
        "source": "unavailable",
        "exit_code": docker_error if docker_error is not None else 1,
        "error": "Docker logs failed or Docker CLI is unavailable and no --server-log-source was supplied",
    }


def _report_session_id(report: Mapping[str, Any]) -> str | None:
    connection = report.get("connection")
    if isinstance(connection, Mapping) and isinstance(connection.get("session_id"), str):
        return connection["session_id"]
    return None


def _run_one(
    *,
    index: int,
    run_dir: Path,
    args: argparse.Namespace,
    execution: str,
    repo_root: Path,
    budgets: Mapping[str, float],
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    sampler = RuntimeTelemetrySampler(args.telemetry_url, run_dir)
    health = http_json(args.health_url)
    write_json(run_dir / "health_pre_run.json", health)
    sampler.prepare()
    command = _simulator_command(
        execution=execution,
        container=args.container,
        python_bin=args.python_bin,
        repo_root=repo_root,
        fixture_manifest=args.fixture_manifest,
        ca_cert=args.ca_cert,
        output_dir=run_dir,
        budgets=budgets,
    )
    collector_started = utc_now()
    sampler.start()
    write_json(run_dir / "runner_invocation.json", {
        "run_index": index,
        "execution": execution,
        "container": args.container if execution == "docker" else None,
        "command": command,
        "collector_started_at_utc": collector_started,
        "telemetry_endpoint": args.telemetry_url,
        "sample_interval_ms": int(SAMPLE_INTERVAL_SECONDS * 1000),
    })
    simulator_stdout = run_dir / "simulator_stdout.log"
    try:
        with simulator_stdout.open("wb") as output:
            completed = subprocess.run(
                command,
                cwd=repo_root if execution == "direct" else None,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        simulator_exit_code = completed.returncode
    except Exception as exc:
        simulator_exit_code = 127
        with simulator_stdout.open("ab") as output:
            output.write(f"\nrunner launch failure: {type(exc).__name__}: {exc}\n".encode("utf-8", errors="replace"))
    finally:
        sampler.stop()
    collector_finished = utc_now()

    report_path = run_dir / "report.json"
    report: dict[str, Any] = {}
    if report_path.is_file():
        try:
            loaded = read_json(report_path)
            if isinstance(loaded, dict):
                report = loaded
        except (OSError, json.JSONDecodeError):
            pass
    write_json(run_dir / "runtime_telemetry.json", sampler.sidecar(report))

    run_window = report.get("run_window_utc")
    run_window = run_window if isinstance(run_window, Mapping) else {}
    since_text = str(run_window.get("started_at") or collector_started)
    until_text = str(run_window.get("finished_at") or collector_finished)
    log_capture = _capture_server_log(
        destination=run_dir / "server.log",
        container=args.container,
        since_text=since_text,
        until_text=until_text,
        source_path=args.server_log_source,
    )
    write_json(run_dir / "server-log-window.json", {
        "schema": "duplex_server_log_window_v1",
        "session_id": _report_session_id(report),
        "since_utc": since_text,
        "until_utc": until_text,
        **log_capture,
    })

    finalizer_exit_code: int | None = None
    final_report_path = run_dir / "report.finalized.json"
    if report_path.is_file():
        if execution == "docker":
            finalizer_cmd = [
                "docker", "exec", "--user", "1000:1000", "--workdir", "/workspace",
                "--env", f"PYTHONPATH={PYTHONPATH_IN_CONTAINER}", args.container,
                args.python_bin,
                "/workspace/tools/real_dialogue_simulator.py",
                "--finalize-report", _container_path(run_dir, repo_root),
            ]
        else:
            finalizer_cmd = [
                sys.executable,
                str(repo_root / "tools" / "real_dialogue_simulator.py"),
                "--finalize-report", str(run_dir.resolve()),
            ]
        with (run_dir / "finalizer_stdout.log").open("wb") as output:
            finalized = subprocess.run(
                finalizer_cmd,
                cwd=repo_root if execution == "direct" else None,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        finalizer_exit_code = finalized.returncode

    final_report: dict[str, Any] = {}
    if final_report_path.is_file():
        try:
            loaded = read_json(final_report_path)
            if isinstance(loaded, dict):
                final_report = loaded
        except (OSError, json.JSONDecodeError):
            pass
    summary = {
        "run_index": index,
        "run_dir": str(run_dir.resolve()),
        "execution": execution,
        "simulator_exit_code": simulator_exit_code,
        "finalizer_exit_code": finalizer_exit_code,
        "server_log_capture_exit_code": log_capture.get("exit_code"),
        "server_log_bytes": (run_dir / "server.log").stat().st_size,
        "session_id": _report_session_id(report),
        "report_status": report.get("overall_status"),
        "final_status": final_report.get("overall_status"),
        "telemetry_samples": len(sampler.samples),
        "telemetry_sample_errors": len(sampler.errors),
        "post_drain_snapshots": len(sampler.post_drain),
        "post_drain_response_ids": sorted(sampler.post_drain),
        "run_window_utc": {"started_at": since_text, "finished_at": until_text},
    }
    write_json(run_dir / "runner_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="number of consecutive acceptance runs (default: 3)")
    parser.add_argument("--output-root", default=None, help="new output directory; defaults under artifacts/acceptance_suite")
    parser.add_argument("--execution", choices=("auto", "docker", "direct"), default="auto")
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON, help="Python executable inside the Docker container")
    parser.add_argument("--fixture-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--ca-cert", default=DEFAULT_CA_CERT)
    parser.add_argument("--budget-config", default=DEFAULT_BUDGET_CONFIG)
    parser.add_argument("--telemetry-url", default=DEFAULT_TELEMETRY_URL)
    parser.add_argument("--health-url", default=DEFAULT_HEALTH_URL)
    parser.add_argument("--server-log-source", default=None, help="timestamped server log file when docker logs is unavailable")
    parser.add_argument("--repo-root", default=str(repo_root))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.runs <= 0:
        parser.error("--runs must be a positive integer")
    repo_root = Path(args.repo_root).expanduser().resolve()
    execution = _choose_execution(args.execution, args.container)
    budgets = _load_budgets(args.budget_config, execution, repo_root)
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_root = repo_root / "artifacts" / f"acceptance_suite_{stamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    summaries: list[dict[str, Any]] = []
    for index in range(1, args.runs + 1):
        run_dir = output_root / f"run_{index:02d}"
        summary = _run_one(
            index=index,
            run_dir=run_dir,
            args=args,
            execution=execution,
            repo_root=repo_root,
            budgets=budgets,
        )
        summaries.append(summary)
        write_json(output_root / "suite_progress.json", {"runs": summaries})

    overall_pass = len(summaries) == args.runs and all(
        row.get("simulator_exit_code") in (0, 2)
        and row.get("report_status") in ("PASS", "INCOMPLETE")
        and row.get("finalizer_exit_code") == 0
        and row.get("server_log_capture_exit_code") == 0
        and row.get("final_status") == "PASS"
        and row.get("telemetry_samples", 0) > 0
        and row.get("telemetry_sample_errors", 0) <= 1
        for row in summaries
    )
    suite = {
        "schema": "duplex_acceptance_suite_v1",
        "execution": execution,
        "container": args.container if execution == "docker" else None,
        "run_count": args.runs,
        "overall_status": "PASS" if overall_pass else "INCOMPLETE",
        "output_root": str(output_root),
        "runs": summaries,
    }
    write_json(output_root / "suite_summary.json", suite)
    print(json.dumps(suite, ensure_ascii=False, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
