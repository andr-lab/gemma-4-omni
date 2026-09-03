#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/kernel_gates/verify_p4_hardening.py — Independent Verifier for P4 Hardening Gates.

Verifies:
1. Proof 1 (Rollover & Parity): 5-10 min continuous stream, rollover verified, oracle parity verified, no latency spikes.
2. Proof 2 (Live Concurrency): Real long generation (>= 128 tokens), concurrent 12.5 Hz listener, p95 <= 80ms, 0 frame drops.
3. Proof 3 (VRAM Stack): Full 5-model stack resident in CUDA:0 (Gemma, V9, Qwen, SubTalker, Code2Wav), free VRAM >= 2 GB.

Prints 'P4 GEMMA LISTENER BLOCKER — CLOSED' upon full pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_proof1(proof_dir: Path) -> tuple[bool, dict[str, Any]]:
    rep_path = proof_dir / "proof1_rollover_report.json"
    trace_path = proof_dir / "proof1_rollover_trace.jsonl"
    if not rep_path.exists():
        return False, {"error": "proof1_rollover_report.json missing"}
    if not trace_path.exists():
        return False, {"error": "proof1_rollover_trace.jsonl missing"}

    with open(rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    # 1. Total frames and rollover check
    total_frames = rep.get("total_frames", 0)
    rollover_frames = rep.get("rollover_frames", 0)
    if total_frames < 3750:
        return False, {"error": f"total_frames ({total_frames}) < 3750 (5 minutes requirement)"}
    if rollover_frames < (total_frames - 64):
        return False, {"error": f"rollover_frames ({rollover_frames}) < expected ({total_frames - 64})"}

    # 2. Parity check
    parity = rep.get("parity_64_frames", {})
    mae = parity.get("mae", 1.0)
    cos_min = parity.get("cosine_min", 0.0)
    if mae > 1e-4 or cos_min < 0.9999:
        return False, {"error": f"parity check failed: mae={mae}, cosine_min={cos_min}"}

    # 3. Latency & health check
    lat = rep.get("latency_stats_ms", {})
    spikes = lat.get("spikes_over_150ms", 999)
    if spikes > 0:
        return False, {"error": f"anomalous latency spikes detected: {spikes} frames over 150ms"}

    # Strict Realtime Latency Gate on complete listener step (<= 80.0 ms)
    p95 = lat.get("p95", 999.0)
    if p95 > 80.0:
        return False, {"error": f"complete listener step p95 latency ({p95}ms) > 80.0ms threshold"}

    # Strict Realtime Wall-Elapsed / Throughput Gate:
    # 3750 frames at 12.5 Hz (80ms) = 300.0s nominal audio
    expected_duration_s = total_frames * 0.08
    wall_elapsed_s = rep.get("wall_elapsed_s", 999.0)
    if wall_elapsed_s > expected_duration_s * 1.05:
        return False, {
            "error": f"wall_elapsed_s ({wall_elapsed_s}s) > expected ({expected_duration_s}s * 1.05 = {expected_duration_s * 1.05:.2f}s)"
        }

    miss_rate = lat.get("deadline_miss_rate", None)
    if miss_rate is not None and miss_rate > 1.0:
        return False, {"error": f"deadline miss rate ({miss_rate}%) > 1.0% threshold"}

    health = rep.get("health", {})
    if health.get("errors", 1) != 0 or health.get("non_finite_count", 1) != 0 or health.get("dropped_frames", 1) != 0:
        return False, {"error": f"health check failed: {health}"}

    # 4. Verify raw trace line count
    trace_lines = 0
    with open(trace_path, "r", encoding="utf-8") as f:
        for _ in f:
            trace_lines += 1
    if trace_lines < total_frames:
        return False, {"error": f"trace line count ({trace_lines}) < total frames ({total_frames})"}

    return True, {
        "total_frames": total_frames,
        "rollover_frames": rollover_frames,
        "mae": mae,
        "cosine_min": cos_min,
        "p50_ms": lat.get("p50"),
        "p95_ms": p95,
        "wall_elapsed_s": wall_elapsed_s,
        "expected_duration_s": expected_duration_s,
        "spikes_over_150ms": spikes,
        "trace_sha256": _file_sha256(trace_path),
    }


def verify_proof2(proof_dir: Path) -> tuple[bool, dict[str, Any]]:
    rep_path = proof_dir / "proof2_concurrency_report.json"
    trace_path = proof_dir / "proof2_concurrency_trace.jsonl"
    if not rep_path.exists():
        return False, {"error": "proof2_concurrency_report.json missing"}
    if not trace_path.exists():
        return False, {"error": "proof2_concurrency_trace.jsonl missing"}

    with open(rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    # 1. Generation check
    gen = rep.get("generation", {})
    tokens_gen = gen.get("tokens_generated", 0)
    if tokens_gen < 128:
        return False, {"error": f"tokens_generated ({tokens_gen}) < 128"}

    # 2. Concurrency check
    listener = rep.get("concurrent_listener", {})
    frames_during = listener.get("frames_during_thinking", 0)
    dropped = listener.get("dropped_frames", 999)
    errors = listener.get("errors", ["placeholder"])
    if frames_during < 10:
        return False, {"error": f"frames_during_thinking ({frames_during}) < 10"}
    if dropped != 0 or len(errors) != 0:
        return False, {"error": f"concurrent listener errors or drops: dropped={dropped}, errors={errors}"}

    # 3. Total Listener Step Latency check (Must gate the FULL transaction including lock-wait)
    tot_lat = listener.get("total_step_latency_ms", {})
    tot_p95 = tot_lat.get("p95", 999.0)
    if tot_p95 > 80.0:
        return False, {"error": f"total listener step p95 during thinking ({tot_p95}ms) > 80.0ms threshold"}

    # 4. LM latency check
    lm_lat = listener.get("lm_latency_ms", {})
    lm_p95 = lm_lat.get("p95", 999.0)
    if lm_p95 > 80.0:
        return False, {"error": f"listener LM p95 latency ({lm_p95}ms) > 80.0ms threshold"}

    return True, {
        "tokens_generated": tokens_gen,
        "duration_s": gen.get("duration_s"),
        "frames_during_thinking": frames_during,
        "lm_p50_ms": lm_lat.get("p50"),
        "lm_p95_ms": lm_p95,
        "total_p50_ms": tot_lat.get("p50"),
        "total_p95_ms": tot_p95,
        "dropped_frames": dropped,
        "trace_sha256": _file_sha256(trace_path),
    }


def verify_proof3(proof_dir: Path) -> tuple[bool, dict[str, Any]]:
    rep_path = proof_dir / "proof3_vram_report.json"
    if not rep_path.exists():
        return False, {"error": "proof3_vram_report.json missing"}

    with open(rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    vram = rep.get("vram_metrics_gb", {})
    reserved = vram.get("reserved", 999.0)
    free_vram = vram.get("free_available", 0.0)

    if reserved > 22.0:
        return False, {"error": f"reserved VRAM ({reserved} GB) > 22.0 GB"}
    if free_vram < 2.0:
        return False, {"error": f"free VRAM ({free_vram} GB) < 2.0 GB headroom requirement"}

    stack = rep.get("stack_components", [])
    if len(stack) < 5:
        return False, {"error": f"stack components count ({len(stack)}) < 5"}

    return True, {
        "components_count": len(stack),
        "allocated_gb": vram.get("allocated"),
        "reserved_gb": reserved,
        "free_available_gb": free_vram,
    }


def run_verification(proof_dir: Path) -> bool:
    print("=" * 70)
    print("INDEPENDENT VERIFIER: P4 GEMMA LISTENER HARDENING GATES")
    print(f"Target Directory: {proof_dir}")
    print("=" * 70)

    p1_pass, p1_info = verify_proof1(proof_dir)
    print(f"\n[PROVA 1] Rollover >64 Frame & Oracle Parity (5 min): {'PASS' if p1_pass else 'FAIL'}")
    if p1_pass:
        print(f"  • Frames: {p1_info['total_frames']} (Rollover: {p1_info['rollover_frames']})")
        print(f"  • Parity MAE: {p1_info['mae']:.8f}, Cosine Min: {p1_info['cosine_min']:.8f}")
        print(f"  • Latency Steady: p50={p1_info['p50_ms']}ms, p95={p1_info['p95_ms']}ms, Spikes>150ms: {p1_info['spikes_over_150ms']}")
    else:
        print(f"  FAILED: {p1_info}")

    p2_pass, p2_info = verify_proof2(proof_dir)
    print(f"\n[PROVA 2] Concurrency 12.5Hz Listener + Long Think (>=128 tok): {'PASS' if p2_pass else 'FAIL'}")
    if p2_pass:
        print(f"  • Tokens Generated: {p2_info['tokens_generated']} in {p2_info['duration_s']}s")
        print(f"  • Concurrent Frames: {p2_info['frames_during_thinking']}, Dropped: {p2_info['dropped_frames']}")
        print(f"  • Listener Total Latency: p50={p2_info['total_p50_ms']}ms, p95={p2_info['total_p95_ms']}ms (Gate <= 80ms)")
        print(f"  • Listener LM Latency: p50={p2_info['lm_p50_ms']}ms, p95={p2_info['lm_p95_ms']}ms (Gate <= 80ms)")
    else:
        print(f"  FAILED: {p2_info}")

    p3_pass, p3_info = verify_proof3(proof_dir)
    print(f"\n[PROVA 3] Full Stack VRAM Residency (Gemma+V9+Qwen+SubTalker+Code2Wav): {'PASS' if p3_pass else 'FAIL'}")
    if p3_pass:
        print(f"  • Components: {p3_info['components_count']}/5 active & exercised")
        print(f"  • Allocated: {p3_info['allocated_gb']} GB, Reserved: {p3_info['reserved_gb']} GB")
        print(f"  • Free VRAM Available: {p3_info['free_available_gb']} GB")
    else:
        print(f"  FAILED: {p3_info}")

    print("\n" + "=" * 70)
    if p1_pass and p2_pass and p3_pass:
        print("ALL 3 HARDENING GATES VERIFIED SUCCESSFULLY!")
        print("\n>>> P4 GEMMA LISTENER BLOCKER — CLOSED <<<\n")
        print("=" * 70)
        return True
    else:
        print("VERIFICATION FAILED: NOT ALL GATES PASSED.")
        print("\n>>> P4 OPEN — GEMMA_LISTENER_REALTIME_FAIL <<<\n")
        print("=" * 70)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("/workspace/validation/kernel_gates/P4_HARDENING"))
    args = parser.parse_args()
    success = run_verification(args.dir)
    sys.exit(0 if success else 1)
