#!/usr/bin/env python3
"""Independent Master Verifier for P4 Final Single-GPU Full-Duplex Gate."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * (p / 100.0)
    floor_idx = math.floor(idx)
    ceil_idx = math.ceil(idx)
    if floor_idx == ceil_idx:
        return float(s[int(idx)])
    d0 = s[floor_idx] * (ceil_idx - idx)
    d1 = s[ceil_idx] * (idx - floor_idx)
    return float(d0 + d1)


def verify_report(report_path: Path) -> bool:
    if not report_path.exists():
        print(f"[FAIL] Report file does not exist: {report_path}", file=sys.stderr)
        return False

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 70)
    print(f"VERIFYING P4 FINAL MASTER GATE: {report_path.name}")
    print("=" * 70)

    # 1. Gate Name & Schema
    gate_name = data.get("gate")
    if gate_name != "P4_SPEAKER_LISTENER_CONCURRENCY":
        print(f"[FAIL] Unexpected gate name: {gate_name}")
        return False

    # 2. Extract Raw Frame Records (Do NOT trust precomputed summary booleans)
    listener_records = data.get("listener_records", [])
    speaker_records = data.get("speaker_records", [])
    duration_s = float(data.get("duration_s", 0.0))

    if not listener_records:
        print("[FAIL] Empty listener records!")
        return False

    # Recompute Listener Latency Metrics
    latencies = [float(r["latency_ms"]) for r in listener_records]
    latenesses = [float(r.get("lateness_ms", 0.0)) for r in listener_records]
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)

    misses = sum(1 for late in latenesses if late > 0.0)
    miss_rate = (misses / len(latenesses)) * 100.0

    # Recompute Speaker Throughput
    speaker_count = len(speaker_records)
    speaker_fps = speaker_count / duration_s if duration_s > 0 else 0.0

    # Playout & Reference Verification
    valid_refs = sum(1 for r in listener_records if r.get("ref_frame_id") is not None)
    expected_min_refs = max(0, len(listener_records) - 2) # first frame has no prior reference

    # Backlog & Dropped
    final_backlog = int(data.get("final_ingress_backlog", -1))
    dropped_frames = int(data.get("dropped_input_frames", -1))
    vram_growth_mb = float(data.get("vram_growth_mb", 999.0))

    # Detailed Printout
    print(f"Total Duration:          {duration_s:.2f} s")
    print(f"Listener Frames:         {len(listener_records)}")
    print(f"Listener Latency p50:    {p50:.2f} ms")
    print(f"Listener Latency p95:    {p95:.2f} ms (Gate: <= 80.0 ms)")
    print(f"Listener Latency p99:    {p99:.2f} ms")
    print(f"Listener Deadline Miss:  {miss_rate:.2f}% (Gate: <= 1.0%)")
    print(f"Speaker Frames:          {speaker_count}")
    print(f"Speaker Throughput:      {speaker_fps:.2f} fps (Gate: >= 12.5 fps)")
    print(f"Valid Echo Ref Frames:   {valid_refs}/{len(listener_records)}")
    print(f"Final Ingress Backlog:   {final_backlog} (Gate: == 0)")
    print(f"Dropped Frames:          {dropped_frames} (Gate: == 0)")
    print(f"VRAM Growth:             {vram_growth_mb:.2f} MB (Gate: < 50.0 MB)")

    # CUDA / Wall Breakdown Reconciliation check
    cuda_breakdown = data.get("cuda_breakdown_ms", {})
    if cuda_breakdown:
        print("\nReconciled Sub-Step CUDA Breakdown:")
        l_gpu = cuda_breakdown.get("listener_mean_ms", 0.0)
        v9_gpu = cuda_breakdown.get("v9_mean_ms", 0.0)
        s_gpu = cuda_breakdown.get("speaker_mean_ms", 0.0)
        total_gpu = cuda_breakdown.get("total_tick_mean_ms", 0.0)
        wall_mean = cuda_breakdown.get("wall_tick_mean_ms", 0.0)
        print(f"  Listener GPU:  {l_gpu:.2f} ms")
        print(f"  V9 GPU:        {v9_gpu:.2f} ms")
        print(f"  Speaker GPU:   {s_gpu:.2f} ms")
        print(f"  Total GPU:     {total_gpu:.2f} ms")
        print(f"  Wall Tick:     {wall_mean:.2f} ms")
        recon_diff = abs(total_gpu - wall_mean)
        print(f"  Reconciliation |GPU - Wall|: {recon_diff:.2f} ms (Target: <= 1.5 ms)")

    # Evaluate Pass/Fail
    checks = {
        "listener_p95_le_80": p95 <= 80.0,
        "listener_miss_rate_le_1": miss_rate <= 1.0,
        "speaker_fps_ge_12_5": speaker_fps >= 12.5,
        "ingress_backlog_zero": final_backlog == 0,
        "dropped_frames_zero": dropped_frames == 0,
        "vram_leak_safe": vram_growth_mb < 50.0,
        "reference_frames_valid": valid_refs >= expected_min_refs,
    }

    print("\nIndividual Gate Assertions:")
    all_passed = True
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {check_name:25s}: {status}")

    print("=" * 70)
    if all_passed:
        print("FINAL VERDICT: PASS — P4 SINGLE-GPU FULL-DUPLEX CERTIFIED")
        return True
    else:
        print("FINAL VERDICT: FAIL — P4 GATES NOT MET")
        return False


def main() -> None:
    report_file = ROOT / "validation" / "kernel_gates" / "P4_FINAL" / "P4_SPEAKER_LISTENER_CONCURRENCY.json"
    if len(sys.argv) > 1:
        report_file = Path(sys.argv[1])

    success = verify_report(report_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
