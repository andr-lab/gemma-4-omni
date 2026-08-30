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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def metadata(phase: str, seed: int, manifest_path: Path, git_sha_override: str | None = None, git_dirty_override: bool | None = None) -> dict[str, Any]:
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
        "phase": phase,
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


def run_cpu_contract(test_path: str | list[str]) -> dict[str, Any]:
    test_paths = [test_path] if isinstance(test_path, str) else test_path
    cmd = [sys.executable, "-m", "pytest", *test_paths, "-q"]
    started = time.perf_counter()
    env = os.environ.copy()
    # I contract CPU non devono ereditare il contesto CUDA inizializzato dal
    # runner GPU: alcuni test V9 esercitano torch CPU ma l'import di torch
    # eredita comunque la visibilità della GPU del processo padre.
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, env=env)
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
            with raw_path.open("ab") as f:
                f.write((json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
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


def run_p1(raw_path: Path, workspace: Path, seed: int, source_binding: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from tools.kernel_gates.p1_checks import run_checks
    events: list[dict[str, Any]] = [
        {"event": "phase_metadata", "seed": seed, "rng_seed_applied": True},
        {"event": "source_workspace_binding", **source_binding},
        *({"event": "source_workspace_file", **item} for item in source_binding["files"]),
    ]
    checks = run_checks(workspace, seed=seed, sequences=10_000, races=1_000, trace=events)
    cpu = run_cpu_contract(["tests/kernel/test_p1_control_plane.py", "tests/kernel/test_gemma_pcm_mel.py"])
    events.append({"event": "cpu_contract_result", **cpu})
    metrics = {**checks, "cpu_contract": cpu}
    thresholds = {
        "cpu_contract_exit_code": 0,
        "direct_turnhead_imports": 0,
        "direct_barge_in_server_branches": 0,
        "illegal_state_transitions": 0,
        "uncaught_exceptions": 0,
        "stale_audio_events_emitted": 0,
        "stale_text_events_emitted": 0,
        "stale_history_writes": 0,
        "stale_kv_writes": 0,
        "stale_session_mutations": 0,
        "history_non_emitted_accepted": 0,
    }
    metrics["passed"] = bool(checks["passed"] and cpu["exit_code"] == 0)
    raw_path.write_bytes("".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in events).encode("utf-8"))
    return metrics, thresholds


def run_p3(raw_path: Path, workspace: Path, seed: int, source_binding: dict[str, Any],
            cpu_contract: dict[str, Any], listener_frames: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the real Gemma audio-tower listener replay on the GPU host."""
    import torch
    from duplex_kernel.listener import GemmaListener, GemmaListenerState
    from duplex_kernel.models.thinker import load_thinker
    from tools.kernel_gates.p3_checks import run_checks

    events: list[dict[str, Any]] = [
        {
            "event": "phase_metadata",
            "seed": seed,
            "rng_seed_applied": True,
            "sample_rate": 16_000,
            "frame_duration_ms": 80,
            "reference_delay_frames": 1,
            "real_model_required": True,
        },
        {"event": "source_workspace_binding", **source_binding},
        *({"event": "source_workspace_file", **item} for item in source_binding["files"]),
    ]
    thinker = None
    try:
        thinker = load_thinker(str(workspace), device="cuda", dtype=torch.bfloat16)
        listener = GemmaListener.from_thinker(thinker)
        state = GemmaListenerState(context_frames=8)
        metrics, thresholds = run_checks(
            workspace,
            listener=listener,
            state=state,
            seed=seed,
            total_frames=listener_frames,
            trace=events,
        )
    finally:
        if thinker is not None:
            del thinker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics["cpu_contract"] = cpu_contract
    thresholds["cpu_contract_exit_code"] = 0
    metrics["passed"] = bool(
        cpu_contract["exit_code"] == 0
        and metrics["samples_received"] == metrics["samples_accounted_for"]
        and metrics["listener_input_frames"] >= thresholds["listener_input_frames_min"]
        and metrics["replay_frames"] >= thresholds["replay_frames_min"]
        and metrics["replay_seconds"] >= thresholds["replay_seconds_min"]
        and metrics["listener_coverage_while_speaking"] >= thresholds["listener_coverage_while_speaking"]
        and metrics["dropped_listener_frames"] == thresholds["dropped_listener_frames"]
        and metrics["nonfinite_hidden_count"] == thresholds["nonfinite_hidden_count"]
        and metrics["empty_hidden_count"] == thresholds["empty_hidden_count"]
        and metrics["future_reference_leaks"] == thresholds["future_reference_leaks"]
        and metrics["wrong_reference_frames"] == thresholds["wrong_reference_frames"]
        and (metrics["listener_step_gpu_ms"]["p95"] is not None
             and metrics["listener_step_gpu_ms"]["p95"] <= thresholds["listener_step_gpu_ms_p95_max"])
        and metrics["listener_errors"] == thresholds["listener_errors"]
        and metrics["assistant_reference_delay_frames"] == thresholds["assistant_reference_delay_frames"]
    )
    events.append({"event": "cpu_contract_result", **cpu_contract})
    raw_path.write_bytes("".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in events).encode("utf-8"))
    return metrics, thresholds


def run_p4(
    raw_path: Path,
    workspace: Path,
    seed: int,
    source_binding: dict[str, Any],
    cpu_contract: dict[str, Any],
    cache_path: Path,
    codes_path: Path | None,
    codes_metadata_path: Path | None,
    v9_ckpt: str | None,
    heads_ckpt: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run P4 on the real V9 and Qwen weights; no fake backend is accepted."""
    import torch
    from duplex_kernel.models.talker import load_talker
    from duplex_kernel.models.v9 import IncrementalV9Projector, load_v9_conditioning
    from tools.kernel_gates.p4_checks import run_checks

    events: list[dict[str, Any]] = [
        {
            "event": "phase_metadata",
            "seed": seed,
            "rng_seed_applied": True,
            "real_model_required": True,
            "dtype": "bfloat16",
            "max_seq_len": 16_384,
        },
        {"event": "source_workspace_binding", **source_binding},
        *({"event": "source_workspace_file", **item} for item in source_binding["files"]),
    ]
    projector = causal_conv = fp32_projector = fp32_causal_conv = talker = None
    try:
        projector, causal_conv = load_v9_conditioning(
            v9_ckpt=v9_ckpt, heads_ckpt=heads_ckpt, device="cuda", dtype=torch.bfloat16
        )
        stream = IncrementalV9Projector(projector, causal_conv, max_seq_len=16_384)
        fp32_projector, fp32_causal_conv = load_v9_conditioning(
            v9_ckpt=v9_ckpt, heads_ckpt=heads_ckpt, device="cuda", dtype=torch.float32
        )
        fp32_stream = IncrementalV9Projector(fp32_projector, fp32_causal_conv, max_seq_len=16_384)
        events.append({
            "event": "p4_model_metadata",
            "real_model": True,
            "backend": "IncrementalV9Projector",
            "v9_parameters": sum(parameter.numel() for parameter in projector.parameters()),
            "raw_dim": stream.raw_dim,
            "hidden_dim": stream.hidden_dim,
            "layers": stream.num_layers,
            "max_seq_len": stream.max_seq_len,
            "future_prefix_dtype": "float32",
        })
        if codes_path is not None and codes_path.is_file():
            try:
                talker = load_talker(str(workspace), device="cuda")
                events.append({"event": "p4_qwen_model_metadata", "real_model": True, "backend": "Qwen3TTSTalkerModel"})
            except Exception as exc:
                events.append({"event": "p4_qwen_load_error", "error": f"{type(exc).__name__}: {exc}"})
        metrics, thresholds = run_checks(
            workspace,
            cache_path=cache_path,
            codes_path=codes_path,
            codes_metadata_path=codes_metadata_path,
            stream=stream,
            future_stream=fp32_stream,
            talker=talker,
            trace=events,
        )
    finally:
        del talker, projector, causal_conv, fp32_projector, fp32_causal_conv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics["cpu_contract"] = cpu_contract
    thresholds["cpu_contract_exit_code"] = 0
    metrics["passed"] = bool(metrics["passed"] and cpu_contract["exit_code"] == 0)
    events.append({"event": "cpu_contract_result", **cpu_contract})
    raw_path.write_bytes("".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in events).encode("utf-8"))
    return metrics, thresholds


def run_p2(raw_path: Path, workspace: Path, seed: int, source_binding: dict[str, Any],
            real_duration_s: float, cpu_contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from tools.kernel_gates.p2_checks import run_checks

    events: list[dict[str, Any]] = []
    metrics, thresholds = run_checks(
        workspace,
        seed=seed,
        real_duration_s=real_duration_s,
        cpu_contract=cpu_contract,
        trace=events,
    )
    binding_index = 1
    events.insert(binding_index, {"event": "source_workspace_binding", **source_binding})
    events[binding_index + 1:binding_index + 1] = [
        {"event": "source_workspace_file", **item} for item in source_binding["files"]
    ]
    raw_path.write_bytes("".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in events).encode("utf-8"))
    return metrics, thresholds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["P0", "P1", "P2", "P3", "P4"], required=True)
    ap.add_argument("--workspace", type=Path, default=ROOT)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=ROOT / "artifacts" / "kernel_gates")
    ap.add_argument("--run-id", default="run1", help="identificatore della ripetizione, ad esempio run1/run2")
    ap.add_argument("--git-sha", default=None, help="SHA del checkout host quando il container non contiene .git")
    ap.add_argument("--git-dirty", choices=["true", "false"], default=None)
    ap.add_argument("--source-attestation", type=Path, required=True,
                    help="attestation content-addressed del checkout Git pulito")
    ap.add_argument("--real-duration-s", type=float, default=60.0,
                    help="durata real-clock P2; il gate completo richiede almeno 60 s")
    ap.add_argument("--listener-frames", type=int, default=10_000,
                    help="listener updates P3; il gate completo richiede almeno 10.000")
    ap.add_argument("--v9-cache", type=Path, default=Path("/workspace/cache_ars_full_v2.pt"))
    ap.add_argument("--codes", type=Path, default=Path("/workspace/data/train_tokens_cv26_it_40.8h_31916s_codes.parquet"))
    ap.add_argument("--codes-metadata", type=Path, default=Path("/workspace/data/train_tokens_cv26_it_40.8h_31916s_hyperparams.json"))
    ap.add_argument("--v9-ckpt", default=None)
    ap.add_argument("--heads-ckpt", default=None)
    args = ap.parse_args()
    manifest_path = args.manifest or (ROOT / "tools" / "kernel_gates" / "manifests" / "p0_inputs.json")
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    seed = int(json.loads(manifest_path.read_text(encoding="utf-8")).get("seed", 1234))
    set_global_seed(seed)
    dirty_override = None if args.git_dirty is None else args.git_dirty == "true"
    from tools.kernel_gates.source_attestation import verify_workspace_attestation
    try:
        source_binding = verify_workspace_attestation(args.source_attestation, args.workspace)
    except Exception as exc:
        print(f"source workspace binding FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    meta = metadata(args.phase, seed, manifest_path, args.git_sha, dirty_override)
    artifact_dir = args.out_root / meta["git_sha"] / args.phase / args.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_path = artifact_dir / "raw_trace.jsonl"
    artifact_manifest = artifact_dir / "input_manifest.json"
    shutil.copyfile(manifest_path, artifact_manifest)
    source_attestation = artifact_dir / "source_attestation.json"
    shutil.copyfile(args.source_attestation, source_attestation)
    if raw_path.exists():
        raw_path.unlink()
    if args.phase == "P1":
        metrics, thresholds = run_p1(raw_path, args.workspace, seed, source_binding)
        passed = bool(metrics["passed"])
    elif args.phase == "P3":
        cpu = run_cpu_contract([
            "tests/kernel/test_p3_listener.py",
            "tests/kernel/test_p2_scheduler.py",
            "tests/test_duplex_kernel_contract.py",
        ])
        print("CPU contract exit_code=", cpu["exit_code"], flush=True)
        metrics, thresholds = run_p3(
            raw_path, args.workspace, seed, source_binding, cpu, args.listener_frames,
        )
        passed = bool(metrics["passed"])
    elif args.phase == "P2":
        cpu = run_cpu_contract(["tests/kernel/test_p2_scheduler.py", "tests/test_duplex_kernel_contract.py"])
        print("CPU contract exit_code=", cpu["exit_code"], flush=True)
        metrics, thresholds = run_p2(
            raw_path, args.workspace, seed, source_binding, args.real_duration_s, cpu,
        )
        passed = bool(metrics["passed"])
    elif args.phase == "P4":
        cpu = run_cpu_contract([
            "tests/kernel/test_p4_v9_incremental.py",
            "tests/kernel/test_p3_listener.py",
            "tests/kernel/test_p2_scheduler.py",
            "tests/test_duplex_kernel_contract.py",
        ])
        print("CPU contract exit_code=", cpu["exit_code"], flush=True)
        metrics, thresholds = run_p4(
            raw_path,
            args.workspace,
            seed,
            source_binding,
            cpu,
            args.v9_cache,
            args.codes,
            args.codes_metadata,
            args.v9_ckpt,
            args.heads_ckpt,
        )
        passed = bool(metrics["passed"])
    else:
        cpu = run_cpu_contract("tests/test_duplex_kernel_contract.py")
        print("CPU contract exit_code=", cpu["exit_code"], flush=True)
        with raw_path.open("wb") as f:
            initial_events = [
                {"event": "phase_metadata", "seed": seed, "rng_seed_applied": True, "cpu_contract": cpu},
                {"event": "source_workspace_binding", **source_binding},
                *({"event": "source_workspace_file", **item} for item in source_binding["files"]),
            ]
            f.write("".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in initial_events).encode("utf-8"))
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
    report = {**meta, "run_id": args.run_id, "input_manifest": artifact_manifest.name, "raw_trace": raw_path.name,
              "raw_trace_sha256": sha256(raw_path), "source_attestation": source_attestation.name,
              "source_workspace_binding": source_binding,
              "source_attestation_sha256": sha256(source_attestation), "metrics": metrics,
              "thresholds": thresholds, "passed": passed}
    report_path = artifact_dir / f"{args.phase}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report={report_path} passed={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
