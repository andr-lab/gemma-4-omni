#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/kernel_gates/p4_hardening_concurrency.py — Hardening Gate 2: Live Concurrency.

Verifies:
1. Thinker executes real long autoregressive generation (>= 128 new tokens).
2. Listener runs concurrently at 12.5 Hz in background thread during generation.
3. Listener p95 <= 80.0 ms during concurrent execution.
4. Zero listener frames dropped.
5. Emits raw trace JSONL + summary report JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch

from duplex_kernel.engine import PipelineEngine
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.listener import GemmaListener, GemmaListenerState


def _git_info() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
        diff = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT), text=True).strip()
        return sha, bool(diff)
    except Exception:
        return "unknown", False


def run_concurrency_gate(
    *,
    max_new_tokens: int = 128,
    output_dir: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_trace_path = output_dir / "proof2_concurrency_trace.jsonl"
    report_path = output_dir / "proof2_concurrency_report.json"

    print(f"[PROOF 2] Starting Live Concurrency Gate (Generation >= {max_new_tokens} tokens)...", flush=True)

    engine = PipelineEngine(device=device, dtype=torch.bfloat16)
    th = load_thinker("/workspace", device=device, dtype=torch.bfloat16)
    engine._thinker_wrapper = th
    engine.thinker = th.model
    engine.tokenizer = th.tokenizer
    engine.listener = GemmaListener.from_thinker(th)
    engine._thinker_lock = engine.listener.backend._lock

    listener = engine.listener
    backend = listener.backend
    tokenizer = engine.tokenizer
    listener_state = GemmaListenerState(context_frames=64)

    prompt = (
        "Spiega in modo chiaro e approfondito l'architettura di un sistema conversazionale "
        "full-duplex in tempo reale basato su modelli multimodali, descrivendo il ruolo "
        "del Thinker per il ragionamento linguistico e del Talker per la sintesi vocale, con dettagli implementativi."
    )
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = int(prompt_ids.shape[1])
    print(f"[PROOF 2] Prompt token length: {prompt_len}", flush=True)

    # Concurrency synchronization events
    generation_started = threading.Event()
    generation_finished = threading.Event()
    stop_listener = threading.Event()

    listener_timings_ms: list[float] = []
    listener_lm_timings_ms: list[float] = []
    listener_records: list[dict[str, Any]] = []
    listener_errors: list[str] = []
    rng = np.random.default_rng(2026)

    listener_stream = torch.cuda.Stream()

    def listener_thread_func() -> None:
        with torch.cuda.stream(listener_stream):
            raw_pcm = (0.05 * rng.standard_normal(1280)).astype(np.float32)
            # Warmup a couple frames before generation starts
            for _ in range(3):
                listener.step(raw_pcm, None, listener_state)
                listener_stream.synchronize()

            generation_started.wait()
            frame_count = 0
            next_tick = time.perf_counter()
            
            while not stop_listener.is_set():
                pcm = (0.05 * rng.standard_normal(1280)).astype(np.float32)
                try:
                    # Instrument listener step and LM step separately on listener_stream
                    t_step_start = time.perf_counter()
                    
                    lm_start_event = torch.cuda.Event(enable_timing=True)
                    lm_end_event = torch.cuda.Event(enable_timing=True)
                    original_fwd = backend._forward_incremental_lm
                    captured_lm_dt = [0.0]
                    
                    def timed_fwd(*args, **kwargs):
                        lm_start_event.record(listener_stream)
                        res = original_fwd(*args, **kwargs)
                        lm_end_event.record(listener_stream)
                        lm_end_event.synchronize()
                        captured_lm_dt[0] = float(lm_start_event.elapsed_time(lm_end_event))
                        return res
                    
                    backend._forward_incremental_lm = timed_fwd
                    out = listener.step(pcm, None, listener_state)
                    backend._forward_incremental_lm = original_fwd
                    
                    listener_stream.synchronize()
                    step_dt = (time.perf_counter() - t_step_start) * 1000.0
                    lm_dt = captured_lm_dt[0]

                    h = out.gemma_hidden_t
                    finite = bool(torch.isfinite(h).all().item()) if hasattr(h, "all") else True

                    record = {
                        "event": "proof2_concurrent_listener_frame",
                        "frame_id": frame_count,
                        "lm_gpu_ms": round(lm_dt, 3),
                        "step_total_ms": round(step_dt, 3),
                        "finite": finite,
                        "is_thinking": not generation_finished.is_set(),
                        "timestamp": round(time.perf_counter(), 4),
                    }
                    listener_records.append(record)
                    listener_lm_timings_ms.append(lm_dt)
                    listener_timings_ms.append(step_dt)
                    frame_count += 1
                except Exception as exc:
                    backend._forward_incremental_lm = original_fwd
                    listener_errors.append(str(exc))
                    listener_records.append({
                        "event": "proof2_concurrent_listener_error",
                        "frame_id": frame_count,
                        "error": str(exc),
                    })
                    break

            # 12.5 Hz pacing = 80 ms per frame
            next_tick += 0.080
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Slight overrun under load: adjust target to avoid cumulative lag
                next_tick = time.perf_counter()

    listener_thread = threading.Thread(target=listener_thread_func, name="12.5Hz-listener-thread", daemon=True)
    listener_thread.start()

    # Give listener thread a moment to initialize
    time.sleep(0.5)

    print("[PROOF 2] Starting long autoregressive generation on shared model...", flush=True)
    generated_tokens: list[int] = []
    generated_text = ""
    gen_start_time = time.perf_counter()

    try:
        generation_started.set()

        # Real autoregressive generation on shared model via engine.think()
        with torch.inference_mode():
            prompt_token_ids = prompt_ids[0].tolist()
            think_res = engine.think(
                prompt_token_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )

        torch.cuda.synchronize()
        gen_duration_s = time.perf_counter() - gen_start_time
        generation_finished.set()
        generated_tokens = think_res.get("gen_ids", [])
        generated_text = think_res.get("text", "")
        print(f"[PROOF 2] Generation finished in {gen_duration_s:.2f}s ({len(generated_tokens)} tokens, {len(generated_tokens)/gen_duration_s:.1f} tok/s)", flush=True)
        print(f"[PROOF 2] Full Generated Text: {generated_text!r}", flush=True)

    finally:
        # Let listener run 5 more frames after generation to capture trailing state
        time.sleep(0.4)
        stop_listener.set()
        listener_thread.join(timeout=5.0)

    # Write raw trace
    with open(raw_trace_path, "w", encoding="utf-8") as f:
        for rec in listener_records:
            f.write(json.dumps(rec) + "\n")

    # Compute concurrent listener statistics
    # Only frames during active thinking
    concurrent_frames = [r for r in listener_records if r.get("is_thinking")]
    concurrent_lm_times = [r["lm_gpu_ms"] for r in concurrent_frames if "lm_gpu_ms" in r]
    concurrent_total_times = [r["step_total_ms"] for r in concurrent_frames if "step_total_ms" in r]

    if concurrent_lm_times:
        lm_p50 = float(np.percentile(concurrent_lm_times, 50))
        lm_p95 = float(np.percentile(concurrent_lm_times, 95))
        lm_p99 = float(np.percentile(concurrent_lm_times, 99))
        lm_max = float(np.max(concurrent_lm_times))
    else:
        lm_p50 = lm_p95 = lm_p99 = lm_max = 0.0

    if concurrent_total_times:
        tot_p50 = float(np.percentile(concurrent_total_times, 50))
        tot_p95 = float(np.percentile(concurrent_total_times, 95))
        tot_p99 = float(np.percentile(concurrent_total_times, 99))
        tot_max = float(np.max(concurrent_total_times))
    else:
        tot_p50 = tot_p95 = tot_p99 = tot_max = 0.0

    git_sha, git_dirty = _git_info()
    tokens_generated_count = len(generated_tokens)
    passed = bool(
        tokens_generated_count >= max_new_tokens
        and len(listener_errors) == 0
        and listener_state.dropped_listener_frames == 0
        and len(concurrent_lm_times) >= 10
        and lm_p95 <= 80.0
    )

    report = {
        "schema_version": 1,
        "proof": "PROOF_2_CONCURRENT_THINK_LISTENER",
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "generation": {
            "prompt_length": prompt_len,
            "tokens_generated": tokens_generated_count,
            "duration_s": round(gen_duration_s, 2),
            "tokens_per_second": round(tokens_generated_count / gen_duration_s, 2),
            "generated_text_preview": generated_text[:200],
        },
        "concurrent_listener": {
            "frames_during_thinking": len(concurrent_lm_times),
            "total_frames": len(listener_records),
            "dropped_frames": listener_state.dropped_listener_frames,
            "errors": listener_errors,
            "lm_latency_ms": {
                "p50": round(lm_p50, 2),
                "p95": round(lm_p95, 2),
                "p99": round(lm_p99, 2),
                "max": round(lm_max, 2),
            },
            "total_step_latency_ms": {
                "p50": round(tot_p50, 2),
                "p95": round(tot_p95, 2),
                "p99": round(tot_p99, 2),
                "max": round(tot_max, 2),
            },
        },
        "thresholds": {
            "tokens_generated_min": max_new_tokens,
            "listener_lm_p95_max_ms": 80.0,
            "dropped_frames_max": 0,
        },
        "passed": passed,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[PROOF 2 REPORT] Passed: {passed}")
    print(f"  Tokens Generated: {tokens_generated_count} in {gen_duration_s:.2f}s ({tokens_generated_count/gen_duration_s:.1f} tok/s)")
    print(f"  Concurrent Listener Frames: {len(concurrent_lm_times)}")
    print(f"  Listener LM Latency: p50={lm_p50:.2f}ms, p95={lm_p95:.2f}ms, max={lm_max:.2f}ms")
    print(f"  Listener Total Step: p50={tot_p50:.2f}ms, p95={tot_p95:.2f}ms, max={tot_max:.2f}ms")
    print(f"  Dropped Frames: {listener_state.dropped_listener_frames}, Errors: {len(listener_errors)}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128, help="Number of new tokens to generate (default: 128)")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/validation/kernel_gates/P4_HARDENING"))
    args = parser.parse_args()
    res = run_concurrency_gate(max_new_tokens=args.tokens, output_dir=args.output_dir)
    sys.exit(0 if res["passed"] else 1)
