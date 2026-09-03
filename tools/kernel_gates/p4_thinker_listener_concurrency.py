#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/kernel_gates/p4_thinker_listener_concurrency.py — Production DuplexScheduler & AudioIngressQueue Concurrency Gate.

Implements Sections 2, 3, 4, 5, 6 of P4 Final Specification:
- True production DuplexScheduler + MonotonicFrameClock (80 ms period, no clock rebasing)
- Real AudioIngressQueue fed at 12.5 Hz (1,280 samples/frame @ 16 kHz)
- Listener active during BOTH Thinker prompt prefill and autoregressive decode (>=128 tokens)
- Real GemmaListener (shared weights) + IncrementalV9Projector (L=0)
- Nanosecond telemetry per frame and strict deadline miss rate gating (<= 1.0%)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch

from duplex_kernel.engine import PipelineEngine
from duplex_kernel.listener import GemmaListener, GemmaListenerState
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.models.v9 import IncrementalV9Projector, load_v9_conditioning
from duplex_kernel.scheduler import AudioFrame, AudioIngressQueue, DuplexScheduler, DuplexTick, MonotonicFrameClock


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_thinker_listener_concurrency(
    tokens: int = 128,
    output_dir: Path | None = None,
) -> bool:
    print("=" * 70)
    print(f"[PROOF 2 FINAL] Production DuplexScheduler Hardening (tokens={tokens})...")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    workspace = Path("/workspace") if Path("/workspace").exists() else ROOT

    # 1. Load Models (Shared Gemma + V9 Projector)
    print("[PROOF 2 FINAL] Loading Thinker, Listener, and V9 Projector...", flush=True)
    th = load_thinker(workspace, device=device, dtype=torch.bfloat16)
    engine = PipelineEngine(device=device, dtype=torch.bfloat16)
    engine.thinker = th.model
    engine.tokenizer = th.tokenizer
    engine.listener = GemmaListener.from_thinker(th)
    engine._thinker_lock = engine.listener.backend._lock

    listener = engine.listener
    backend = listener.backend
    listener_state = GemmaListenerState(context_frames=64)

    # Load V9 Projector and initialize incremental state
    v9, causal_conv = load_v9_conditioning(device=device, dtype=torch.bfloat16)
    v9_stream = IncrementalV9Projector(v9, causal_conv, max_seq_len=16_384)
    v9_state = v9_stream.new_state(batch_size=1, max_seq_len=4096, device=device, dtype=torch.bfloat16)
    v9_anchor = torch.zeros((1, 1536), device=device, dtype=torch.bfloat16)

    # Prompt preparation
    prompt = (
        "Spiega in modo chiaro e approfondito l'architettura di un sistema conversazionale "
        "full-duplex in tempo reale basato su modelli multimodali, descrivendo il ruolo "
        "del Thinker per il ragionamento linguistico e del Talker per la sintesi vocale, con dettagli implementativi."
    )
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = engine.tokenizer(chat_prompt, return_tensors="pt").input_ids.to(device)
    prompt_token_ids = prompt_ids[0].tolist()

    # 1. Warmup Thinker with persistent StaticCache and torch.compile
    print("[PROOF 2 FINAL] Compiling and warming up Thinker for 13ms/tok decode...", flush=True)
    engine._compiled_thinker = torch.compile(engine.thinker, mode="default")
    # Run full engine.think warmup (4 tokens) to compile step 0, step 1, and steady-state loop
    engine.think(prompt_token_ids, max_new_tokens=4, min_new_tokens=4)
    torch.cuda.synchronize()
    print("[PROOF 2 FINAL] Thinker warmup complete.", flush=True)

    # 2. Warmup Listener + V9 to steady state (64 frames) on dedicated stream
    listener_stream = torch.cuda.Stream()
    print("[PROOF 2 FINAL] Warming up Listener + V9 to steady state (64 frames)...", flush=True)
    with torch.cuda.stream(listener_stream):
        warmup_pcm = (0.05 * np.random.default_rng(42).standard_normal(1280)).astype(np.float32)
        for _ in range(64):
            w_out = listener.step(warmup_pcm, None, listener_state)
            w_h = w_out.gemma_hidden_t
            if w_h.dim() == 1:
                w_h = w_h.unsqueeze(0)
            v9_stream.step(w_h, v9_anchor, v9_state)
        listener.prime_steady_state(listener_state)
        listener_stream.synchronize()
    print("[PROOF 2 FINAL] Steady-state CUDA Graph active. Systems primed.", flush=True)

    # 3. Setup Production Ingress & Scheduler
    clock = MonotonicFrameClock(period_ms=80)
    ingress = AudioIngressQueue(sample_rate=16000, frame_duration_ms=80)

    # Telemetry storage
    frame_records: list[dict] = []
    latencies_ms: list[float] = []
    deadline_latenesses_ms: list[float] = []
    lm_gpu_latencies_ms: list[float] = []

    # Synchronization flags
    thinker_prefill_started = threading.Event()
    thinker_prefill_done = threading.Event()
    thinker_done = threading.Event()
    stop_scheduler = threading.Event()

    prefill_frames_count = 0
    decode_frames_count = 0
    prefill_duration_ms = 0.0
    decode_duration_ms = 0.0

    # Listener callback dispatched by DuplexScheduler
    def on_listener_tick(frame: AudioFrame | None, tick: DuplexTick) -> None:
        nonlocal prefill_frames_count, decode_frames_count
        if frame is None:
            return

        queue_depth_before = ingress._queue.qsize()

        # Signal waiting priority
        if hasattr(backend, "_waiting_for_lm") and backend._waiting_for_lm is not None:
            backend._waiting_for_lm.set()

        t_start_ns = time.monotonic_ns()
        try:
            with torch.cuda.stream(listener_stream):
                # 1. Gemma Listener Step
                out = listener.step(frame.pcm, None, listener_state)
                # 2. V9 Incremental Step
                h_gemma = out.gemma_hidden_t
                if h_gemma.dim() == 1:
                    h_gemma = h_gemma.unsqueeze(0)
                cond_frame = v9_stream.step(h_gemma, v9_anchor, v9_state)
                listener_stream.synchronize()
            t_cond_ready_ns = time.monotonic_ns()

            step_latency_ms = (t_cond_ready_ns - t_start_ns) / 1_000_000.0
            completion_deadline_ns = tick.deadline_ns + clock.period_ns
            lateness_ms = (t_cond_ready_ns - completion_deadline_ns) / 1_000_000.0

            queue_depth_after = ingress._queue.qsize()

            is_prefill = (frame.frame_id == 0 and thinker_prefill_started.is_set()) or (thinker_prefill_started.is_set() and not thinker_prefill_done.is_set())
            is_decode = thinker_prefill_done.is_set() and not thinker_done.is_set() and not is_prefill

            if is_prefill:
                prefill_frames_count += 1
            elif is_decode:
                decode_frames_count += 1

            record = {
                "frame_id": frame.frame_id,
                "capture_start_ns": frame.capture_start_ns,
                "capture_end_ns": frame.capture_end_ns,
                "enqueue_ns": frame.enqueue_ns,
                "scheduler_deadline_ns": completion_deadline_ns,
                "scheduler_observed_ns": tick.observed_ns,
                "listener_start_ns": t_start_ns,
                "listener_end_ns": tick.listener_end_ns or t_cond_ready_ns,
                "conditioning_ready_ns": t_cond_ready_ns,
                "step_latency_ms": round(step_latency_ms, 3),
                "deadline_lateness_ms": round(lateness_ms, 3),
                "queue_depth_before": queue_depth_before,
                "queue_depth_after": queue_depth_after,
                "is_prefill": is_prefill,
                "is_decode": is_decode,
            }
            frame_records.append(record)
            latencies_ms.append(step_latency_ms)
            deadline_latenesses_ms.append(lateness_ms)
        finally:
            if hasattr(backend, "_waiting_for_lm") and backend._waiting_for_lm is not None:
                backend._waiting_for_lm.clear()

    scheduler = DuplexScheduler(
        ingress=ingress,
        clock=clock,
        on_listener=on_listener_tick,
        deadline_jitter_budget_ms=0.0,  # Strict zero-tolerance monotonic check
    )

    # Microphone Producer Thread (real monotonic 80 ms generator)
    rng = np.random.default_rng(12345)
    def mic_producer() -> None:
        start_ns = clock.start_ns
        frame_idx = 0
        while not stop_scheduler.is_set():
            t_sched = start_ns + (frame_idx + 1) * clock.period_ns
            now = time.monotonic_ns()
            rem = t_sched - now
            if rem > 0:
                time.sleep(rem / 1_000_000_000.0)
            cap_start = start_ns + frame_idx * clock.period_ns
            cap_end = t_sched
            pcm = (0.05 * rng.standard_normal(1280)).astype(np.float32)
            ingress.append(pcm, capture_start_ns=cap_start)
            frame_idx += 1

    producer_thread = threading.Thread(target=mic_producer, name="mic-producer", daemon=True)

    # Thinker Execution Thread
    generated_tokens: list[int] = []
    generated_text = ""
    def thinker_thread_func() -> None:
        nonlocal generated_tokens, generated_text, prefill_duration_ms, decode_duration_ms
        backend._active_streaming = True
        # 1. Start Prefill concurrently with first listener frame
        print("[PROOF 2 FINAL] Starting Thinker Prompt Prefill (concurrent with listener)...", flush=True)
        thinker_prefill_started.set()
        t_pref0 = time.perf_counter()

        def on_decode_begin():
            nonlocal prefill_duration_ms
            prefill_duration_ms = (time.perf_counter() - t_pref0) * 1000.0
            thinker_prefill_done.set()
            print(f"[PROOF 2 FINAL] Prefill complete in {prefill_duration_ms:.2f} ms. Starting Autoregressive Decode...", flush=True)

        t_dec0 = time.perf_counter()
        think_res = engine.think(
            prompt_token_ids,
            max_new_tokens=tokens,
            min_new_tokens=tokens,
            temperature=0.7,
            top_p=0.9,
            on_generation_start=on_decode_begin,
        )
        decode_duration_ms = (time.perf_counter() - t_dec0) * 1000.0
        thinker_done.set()
        backend._active_streaming = False
        generated_tokens = think_res.get("gen_ids", [])
        generated_text = think_res.get("text", "")
        print(f"[PROOF 2 FINAL] Decode complete in {decode_duration_ms/1000.0:.2f} s ({len(generated_tokens)} tokens, {len(generated_tokens)/(decode_duration_ms/1000.0):.1f} tok/s)", flush=True)

    thinker_worker = threading.Thread(target=thinker_thread_func, name="thinker-worker", daemon=True)

    # 3. Start System
    t_test_start_ns = time.monotonic_ns()
    producer_thread.start()
    scheduler.start()
    thinker_worker.start()

    # Wait for Thinker to finish
    thinker_worker.join(timeout=300.0)
    if thinker_worker.is_alive():
        raise RuntimeError("Thinker worker timed out after 300 seconds")

    # Stop producer first so no more frames are generated
    stop_scheduler.set()
    producer_thread.join(timeout=1.0)
    # Drain any remaining frames in ingress queue
    drain_deadline = time.time() + 1.0
    while ingress.qsize > 0 and time.time() < drain_deadline:
        time.sleep(0.010)
    scheduler.stop()
    t_test_end_ns = time.monotonic_ns()
    total_test_duration_s = (t_test_end_ns - t_test_start_ns) / 1_000_000_000.0

    # 4. Analyze Results
    expected_frames = int(round(total_test_duration_s / 0.080))
    actual_frames = len(frame_records)
    missing_frames = max(0, expected_frames - actual_frames)

    # Backlog analysis
    final_backlog = ingress._queue.qsize()
    max_backlog = max([r["queue_depth_after"] for r in frame_records]) if frame_records else 0

    # Latency percentiles (Complete Listener + V9 Step)
    if latencies_ms:
        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
        p99 = float(np.percentile(latencies_ms, 99))
        max_lat = float(np.max(latencies_ms))
    else:
        p50 = p95 = p99 = max_lat = 0.0

    # Deadline Miss Rate (Frames that finished after deadline)
    deadline_misses = sum(1 for lat in deadline_latenesses_ms if lat > 0.0)
    deadline_miss_rate = (deadline_misses / len(deadline_latenesses_ms)) * 100.0 if deadline_latenesses_ms else 0.0
    max_lateness_ms = float(np.max(deadline_latenesses_ms)) if deadline_latenesses_ms else 0.0

    # Scheduler Ingress stats
    ingress_stats = ingress.stats()
    dropped_frames = int(ingress_stats.get("dropped_input_frames", 0))
    duplicate_frames = int(ingress_stats.get("duplicate_input_frames", 0))
    out_of_order_frames = int(ingress_stats.get("out_of_order_input_frames", 0))

    passed = bool(
        len(generated_tokens) >= tokens
        and p95 <= 80.0
        and deadline_miss_rate <= 1.0
        and final_backlog == 0
        and dropped_frames == 0
        and duplicate_frames == 0
        and out_of_order_frames == 0
        and prefill_frames_count >= 1
        and decode_frames_count >= 10
    )

    print("\n" + "=" * 70)
    print(f"[PROOF 2 FINAL REPORT] Result: {'PASS' if passed else 'FAIL'}")
    print(f"  • Tokens Generated: {len(generated_tokens)} in {decode_duration_ms/1000.0:.2f}s")
    print(f"  • Prefill Duration: {prefill_duration_ms:.2f} ms ({prefill_frames_count} frames during prefill)")
    print(f"  • Decode Frames: {decode_frames_count} frames during decode")
    print(f"  • Total Listener Frames: {actual_frames} (Expected: ~{expected_frames})")
    print(f"  • Complete Step Latency (Listener + V9): p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms, max={max_lat:.2f}ms")
    print(f"  • Deadline Misses: {deadline_misses} / {actual_frames} ({deadline_miss_rate:.2f}%, Gate <= 1.0%)")
    print(f"  • Max Lateness: {max_lateness_ms:.2f} ms")
    print(f"  • Backlog: max={max_backlog}, final={final_backlog}")
    print(f"  • Health: dropped={dropped_frames}, duplicates={duplicate_frames}, out_of_order={out_of_order_frames}")
    print("=" * 70)

    # Save Artifacts
    if output_dir is None:
        output_dir = ROOT / "validation" / "kernel_gates" / "P4_FINAL"
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = output_dir / "P4_THINKER_LISTENER_CONCURRENCY_TRACE.jsonl"
    report_path = output_dir / "P4_THINKER_LISTENER_CONCURRENCY.json"

    with open(trace_path, "w", encoding="utf-8") as f:
        for rec in frame_records:
            f.write(json.dumps(rec) + "\n")

    report = {
        "schema_version": 1,
        "gate": "P4_THINKER_LISTENER_CONCURRENCY",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "thinker": {
            "tokens_generated": len(generated_tokens),
            "prefill_duration_ms": prefill_duration_ms,
            "decode_duration_ms": decode_duration_ms,
            "generated_text_preview": generated_text[:200],
        },
        "listener_scheduler": {
            "expected_frames": expected_frames,
            "actual_frames": actual_frames,
            "missing_frames": missing_frames,
            "frames_during_prefill": prefill_frames_count,
            "frames_during_decode": decode_frames_count,
            "complete_step_latency_ms": {
                "p50": round(p50, 2),
                "p95": round(p95, 2),
                "p99": round(p99, 2),
                "max": round(max_lat, 2),
            },
            "deadline_misses": deadline_misses,
            "deadline_miss_rate": round(deadline_miss_rate, 2),
            "max_lateness_ms": round(max_lateness_ms, 2),
            "input_backlog_max": max_backlog,
            "input_backlog_final": final_backlog,
            "dropped_frames": dropped_frames,
            "duplicate_frames": duplicate_frames,
            "out_of_order_frames": out_of_order_frames,
        },
        "thresholds": {
            "tokens_generated_min": tokens,
            "complete_listener_p95_max_ms": 80.0,
            "deadline_miss_rate_max_pct": 1.0,
            "final_backlog_max": 0,
            "dropped_frames_max": 0,
            "duplicate_frames_max": 0,
            "out_of_order_frames_max": 0,
        },
        "trace_sha256": sha256_file(trace_path),
        "passed": passed,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Report: {report_path}")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    ok = run_thinker_listener_concurrency(tokens=args.tokens, output_dir=args.output_dir)
    sys.exit(0 if ok else 1)
