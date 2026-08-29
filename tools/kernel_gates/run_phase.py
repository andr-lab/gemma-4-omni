#!/usr/bin/env python3
"""Execute a kernel gate phase and emit raw trace plus a verifiable report.

P0 is intentionally model-backed: no FakeEngine is accepted by this runner.
Later phases add their own adapters but keep the same report envelope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import platform
import socket
import statistics
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return "unknown"


def set_global_seed(seed: int) -> None:
    """Imposta esplicitamente tutte le sorgenti RNG usate dal benchmark."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metadata(seed: int, manifest_path: Path, git_sha_override: str | None = None, git_dirty_override: bool | None = None) -> dict[str, Any]:
    import torch
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unknown"
    gpu = command("nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader")
    gpu_name, gpu_uuid, driver = (gpu.split(", ", 2) + ["unknown"] * 3)[:3] if gpu != "unknown" else ("unknown", "unknown", "unknown")
    sha = git_sha_override or command("git", "rev-parse", "HEAD")
    dirty = git_dirty_override if git_dirty_override is not None else bool(command("git", "status", "--porcelain"))
    return {
        "schema_version": 1,
        "phase": "P0",
        "git_sha": sha,
        "git_dirty": dirty,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu_name": gpu_name.strip(),
        "gpu_uuid": gpu_uuid.strip(),
        "cuda_version": torch.version.cuda or "none",
        "driver_version": driver.strip(),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "python_version": platform.python_version(),
        "seed": seed,
        "rng_seed_applied": True,
        "input_manifest_sha256": sha256(manifest_path),
    }


def percentile(values: list[float], q: float) -> float | None:
    return round(float(__import__("numpy").percentile(values, q)), 3) if values else None


def run_cpu_contract() -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", "tests/test_duplex_kernel_contract.py", "-q"]
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return {
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def run_p0(manifest_path: Path, raw_path: Path, workspace: Path, seed: int) -> dict[str, Any]:
    sys.path.insert(0, str(workspace))
    sys.path.insert(0, str(workspace / "qwen3tts_src"))
    import numpy as np
    import torch
    from duplex_kernel.engine import PipelineEngine

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = manifest["inputs"]
    engine = PipelineEngine()
    events: list[dict[str, Any]] = []
    try:
        load_started = time.perf_counter()
        loaded = bool(engine.load(workspace=str(workspace)))
        load_s = round(time.perf_counter() - load_started, 3)
        if not loaded:
            raise RuntimeError("PipelineEngine.load() ha restituito False")
        for index, item in enumerate(inputs):
            set_global_seed(seed + index)
            event: dict[str, Any] = {
                "event": "p0_request",
                "index": index,
                "input_id": item["id"],
                "prompt": item["prompt"],
                "success": False,
                "exceptions": [],
                "nan_count": 0,
                "inf_count": 0,
                "empty_audio": False,
                "rng_seed": seed + index,
            }
            started = time.perf_counter()
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                chat = engine.tokenizer.apply_chat_template(
                    [{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True
                )
                ids = engine.tokenizer(chat, return_tensors="pt")["input_ids"][0].tolist()
                t0 = time.perf_counter()
                think = engine.think(ids, max_new_tokens=48, temperature=0.0, top_p=1.0)
                t1 = time.perf_counter()
                h_pfx = engine.v9_prefix(think["hidden"], think["prompt_len"])
                t2 = time.perf_counter()
                first_audio: list[float] = []
                def on_audio_chunk(chunk: np.ndarray) -> None:
                    if not first_audio:
                        first_audio.append((time.perf_counter() - started) * 1000.0)
                    array = np.asarray(chunk)
                    event["nan_count"] += int(np.isnan(array).sum())
                    event["inf_count"] += int(np.isinf(array).sum())
                text = str(think["text"])
                talk_metrics: dict[str, Any] = {}
                wav = engine.talk(
                    text, h_pfx, n_words=max(1, len(text.split())), on_audio_chunk=on_audio_chunk,
                    metrics=talk_metrics,
                )
                t3 = time.perf_counter()
                wav = np.asarray(wav, dtype=np.float32)
                event["nan_count"] += int(np.isnan(wav).sum())
                event["inf_count"] += int(np.isinf(wav).sum())
                event.update({
                    "success": True,
                    "text": text,
                    "think_time_ms": (t1 - t0) * 1000.0,
                    "v9_time_ms": (t2 - t1) * 1000.0,
                    "talker_time_ms": (t3 - t2) * 1000.0,
                    "total_time_ms": (t3 - started) * 1000.0,
                    "ttfa_ms": first_audio[0] if first_audio else None,
                    "generated_cb0_frames": int(talk_metrics.get("generated_cb0_frames", 0)),
                    "generated_pcm_samples": int(len(wav)),
                    "audio_duration_ms": len(wav) / 24.0,
                    "e2e_rtf": ((t3 - started) * 1000.0) / max(len(wav) / 24.0, 1.0),
                    "speaker_rtf": ((t3 - t2) * 1000.0) / max(len(wav) / 24.0, 1.0),
                    "empty_audio": len(wav) == 0,
                })
            except Exception as exc:
                event["exceptions"].append(f"{type(exc).__name__}: {exc}")
                event["total_time_ms"] = (time.perf_counter() - started) * 1000.0
            if torch.cuda.is_available():
                event["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                event["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            events.append(event)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            print(f"[{index + 1}/{len(inputs)}] {item['id']} success={event['success']} total_ms={event['total_time_ms']:.0f}", flush=True)
    finally:
        try:
            engine.unload()
        except Exception:
            pass

    def vals(key: str) -> list[float]:
        return [float(e[key]) for e in events if e.get("success") and e.get(key) is not None and math.isfinite(float(e[key]))]
    success = [e for e in events if e.get("success")]
    failures = [e for e in events if not e.get("success")]
    return {
        "requests": len(events),
        "successful_requests": len(success),
        "success_rate": len(success) / len(events) if events else 0.0,
        "nan_count": sum(int(e.get("nan_count", 0)) for e in events),
        "inf_count": sum(int(e.get("inf_count", 0)) for e in events),
        "empty_audio_responses": sum(bool(e.get("empty_audio")) for e in events),
        "uncaught_exceptions": sum(len(e.get("exceptions", [])) for e in events),
        "failed_input_ids": [e["input_id"] for e in failures],
        "think_time_ms": {"p50": percentile(vals("think_time_ms"), 50), "p95": percentile(vals("think_time_ms"), 95)},
        "v9_time_ms": {"p50": percentile(vals("v9_time_ms"), 50), "p95": percentile(vals("v9_time_ms"), 95)},
        "talker_time_ms": {"p50": percentile(vals("talker_time_ms"), 50), "p95": percentile(vals("talker_time_ms"), 95)},
        "ttfa_ms": {"p50": percentile(vals("ttfa_ms"), 50), "p95": percentile(vals("ttfa_ms"), 95)},
        "e2e_rtf": {"p50": percentile(vals("e2e_rtf"), 50), "p95": percentile(vals("e2e_rtf"), 95)},
        "speaker_rtf": {"p50": percentile(vals("speaker_rtf"), 50), "p95": percentile(vals("speaker_rtf"), 95)},
        "generated_cb0_frames": sum(int(e.get("generated_cb0_frames", 0)) for e in events),
        "generated_pcm_samples": sum(int(e.get("generated_pcm_samples", 0)) for e in events),
        "cuda_peak_allocated_bytes": max((int(e.get("cuda_peak_allocated_bytes", 0)) for e in events), default=0),
        "cuda_peak_reserved_bytes": max((int(e.get("cuda_peak_reserved_bytes", 0)) for e in events), default=0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["P0"], required=True)
    ap.add_argument("--workspace", type=Path, default=ROOT)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=ROOT / "artifacts" / "kernel_gates")
    ap.add_argument("--run-id", default="run1", help="identificatore della ripetizione, ad esempio run1/run2")
    ap.add_argument("--git-sha", default=None, help="SHA del checkout host quando il container non contiene .git")
    ap.add_argument("--git-dirty", choices=["true", "false"], default=None)
    args = ap.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    seed = int(json.loads(manifest_path.read_text(encoding="utf-8")).get("seed", 1234))
    set_global_seed(seed)
    dirty_override = None if args.git_dirty is None else args.git_dirty == "true"
    meta = metadata(seed, manifest_path, args.git_sha, dirty_override)
    artifact_dir = args.out_root / meta["git_sha"] / args.phase / args.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_path = artifact_dir / "raw_trace.jsonl"
    artifact_manifest = artifact_dir / "input_manifest.json"
    shutil.copyfile(manifest_path, artifact_manifest)
    if raw_path.exists():
        raw_path.unlink()
    cpu = run_cpu_contract()
    print("CPU contract exit_code=", cpu["exit_code"], flush=True)
    with raw_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"event": "phase_metadata", "seed": seed, "rng_seed_applied": True, "cpu_contract": cpu}, ensure_ascii=False, sort_keys=True) + "\n")
    metrics = run_p0(manifest_path, raw_path, args.workspace, seed)
    metrics["cpu_contract"] = cpu
    thresholds = {
        "cpu_contract_exit_code": 0,
        "success_rate": 1.0,
        "nan_count": 0,
        "inf_count": 0,
        "empty_audio_responses": 0,
        "uncaught_exceptions": 0,
    }
    passed = (
        cpu["exit_code"] == 0 and metrics["success_rate"] >= 1.0 and
        metrics["nan_count"] == 0 and metrics["inf_count"] == 0 and
        metrics["empty_audio_responses"] == 0 and metrics["uncaught_exceptions"] == 0
    )
    report = {**meta, "run_id": args.run_id, "input_manifest": artifact_manifest.name, "raw_trace": raw_path.name, "raw_trace_sha256": sha256(raw_path),
              "metrics": metrics, "thresholds": thresholds, "passed": passed}
    report_path = artifact_dir / "P0_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report={report_path} passed={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
