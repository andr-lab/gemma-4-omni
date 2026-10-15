"""OpenAI Codex CLI Orchestration Tool.

Exposes CLI orchestration for Codex CLI running gpt-6-luna with xhigh reasoning,
supporting synchronous and background execution, streaming logs, and session management.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

CODEX_BIN = Path(r"C:\Users\andre\AppData\Local\Programs\OpenAI\Codex\bin\codex.exe")
RUNS_DIR = Path("C:/Users/andre/Documents/gemma-4-omni/.tmp/codex_runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
PID_FILE = RUNS_DIR / "active_pids.json"


def _is_pid_alive(pid: int) -> bool:
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return exit_code.value == 259  # STILL_ACTIVE
    except Exception:
        return False


def _load_active_pids() -> Dict[str, Dict[str, Any]]:
    if not PID_FILE.exists():
        return {}
    try:
        data = json.loads(PID_FILE.read_text(encoding="utf-8"))
        alive = {}
        for pid_str, info in data.items():
            pid = int(pid_str)
            if _is_pid_alive(pid):
                alive[pid_str] = info
        if len(alive) != len(data):
            _save_active_pids(alive)
        return alive
    except Exception:
        return {}


def _save_active_pids(pids: Dict[str, Dict[str, Any]]) -> None:
    try:
        PID_FILE.write_text(json.dumps(pids, indent=2), encoding="utf-8")
    except Exception as exc:
        sys.stderr.write(f"[codex_tool] Warning: failed to save PID file: {exc}\n")


def run_codex(
    prompt: str,
    *,
    cwd: Optional[str] = None,
    timeout_s: Optional[int] = None,
    task_name: Optional[str] = None,
) -> Dict[str, Any]:
    if prompt.startswith("@") and Path(prompt[1:]).exists():
        prompt = Path(prompt[1:]).read_text(encoding="utf-8")

    work_dir = Path(cwd or "C:/Users/andre/Documents/gemma-4-omni").resolve()
    run_id = f"codex_{int(time.time())}_{os.getpid()}"
    log_file = RUNS_DIR / f"{run_id}.log"
    last_msg_file = RUNS_DIR / f"{run_id}_last.md"

    user_instruction = f"Esegui il piano dettagliato fornito in input stdin ({task_name or 'autonomous_task'}). Procedi autonomamente passo per passo fino alla fine."
    cmd = [
        str(CODEX_BIN),
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-m",
        "gpt-6-luna",
        "-c",
        'model_reasoning_effort="xhigh"',
        "-C",
        str(work_dir),
        "-o",
        str(last_msg_file),
        user_instruction,
    ]

    sys.stdout.write(f"[codex_tool] Avvio Codex CLI ({run_id})\n")
    sys.stdout.write(f"[codex_tool] Modello: gpt-6-luna (xhigh reasoning)\n")
    sys.stdout.write(f"[codex_tool] Task: {task_name or prompt[:80]}...\n")
    sys.stdout.write(f"[codex_tool] Log: {log_file}\n")
    sys.stdout.flush()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(work_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if proc.stdin:
            try:
                proc.stdin.write(prompt)
                proc.stdin.close()
            except Exception as write_exc:
                sys.stderr.write(f"[codex_tool] Warning: error writing prompt to stdin: {write_exc}\n")


        active = _load_active_pids()
        active[str(proc.pid)] = {
            "run_id": run_id,
            "task_name": task_name or prompt[:80],
            "started_at": time.time(),
            "log_file": str(log_file),
        }
        _save_active_pids(active)

        full_output = []
        try:
            start_t = time.time()
            for line in proc.stdout:  # type: ignore
                lf.write(line)
                lf.flush()
                full_output.append(line)
                sys.stdout.write(line)
                sys.stdout.flush()
                if timeout_s and (time.time() - start_t) > timeout_s:
                    proc.kill()
                    raise TimeoutError(f"Codex execution exceeded {timeout_s}s timeout")
            proc.wait()
        finally:
            active = _load_active_pids()
            active.pop(str(proc.pid), None)
            _save_active_pids(active)

    sys.stdout.write(f"\n[codex_tool] Codex terminato con exit code {proc.returncode}\n")
    last_msg = last_msg_file.read_text(encoding="utf-8") if last_msg_file.exists() else ""
    return {
        "run_id": run_id,
        "pid": proc.pid,
        "exit_code": proc.returncode,
        "output": "".join(full_output),
        "last_message": last_msg,
        "log_file": str(log_file),
    }


def status() -> None:
    active = _load_active_pids()
    sys.stdout.write(f"=== Codex CLI Orchestrator Status ===\n")
    sys.stdout.write(f"Istanze Codex attive: {len(active)}\n")
    for pid, info in active.items():
        elapsed = time.time() - info.get("started_at", time.time())
        sys.stdout.write(
            f"- PID {pid} [{info.get('run_id')}]: {info.get('task_name')} "
            f"({elapsed:.1f}s) -> Log: {info.get('log_file')}\n"
        )


def kill_all() -> None:
    active = _load_active_pids()
    for pid_str in list(active.keys()):
        pid = int(pid_str)
        try:
            os.kill(pid, 9)
            sys.stdout.write(f"Ucciso processo Codex PID {pid}\n")
        except Exception as exc:
            sys.stdout.write(f"Errore terminazione PID {pid}: {exc}\n")
    _save_active_pids({})


def main() -> None:
    parser = argparse.ArgumentParser(description="Codex CLI Orchestration Tool")
    sub = parser.add_subparsers(dest="subcmd")

    run_p = sub.add_parser("run", help="Run a task via Codex CLI")
    run_p.add_argument("prompt", help="Prompt / task instruction (or @path/to/prompt.txt)")
    run_p.add_argument("--cwd", default=None, help="Working directory")
    run_p.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    run_p.add_argument("--name", default=None, help="Task name")

    sub.add_parser("status", help="Show active Codex instances")
    sub.add_parser("kill-all", help="Kill all active Codex instances")

    args = parser.parse_args()
    if args.subcmd == "status":
        status()
    elif args.subcmd == "kill-all":
        kill_all()
    elif args.subcmd == "run":
        run_codex(
            args.prompt,
            cwd=args.cwd,
            timeout_s=args.timeout,
            task_name=args.name,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
