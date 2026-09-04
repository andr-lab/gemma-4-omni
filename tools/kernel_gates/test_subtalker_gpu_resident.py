#!/usr/bin/env python3
"""Benchmark and verify GPU-resident SubTalker (zero host synchronization)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.talker import load_talker
from duplex_kernel.subtalker_graph import SubtalkerGraph


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    print("=" * 70)
    print("[TEST] Benchmarking SubTalker: Host-Sync vs GPU-Resident")
    print("=" * 70, flush=True)

    talker = load_talker("/workspace", device=device, load_code2wav=False)
    talker.talker.to(dtype=dtype)
    cp = talker.talker.code_predictor
    embed = talker.talker.get_input_embeddings()

    subtalker = SubtalkerGraph(cp, talker_embed=embed).capture()

    last_h = torch.randn(1, 1, 2048, device=device, dtype=dtype)
    c0 = 100
    c0_embed = embed(torch.tensor([[c0]], device=device))

    # --- Warmup ---
    for _ in range(10):
        _ = subtalker.generate(last_h, c0, c0_embed=c0_embed, do_sample=False)
        _ = subtalker.generate(last_h, c0, c0_embed=c0_embed, do_sample=True)
    torch.cuda.synchronize()

    # --- Benchmark Greedy Mode ---
    print("\n[BENCHMARK] Greedy Mode (Deterministic Parity)...", flush=True)
    times_greedy = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens = subtalker.generate(last_h, c0, c0_embed=c0_embed, do_sample=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_greedy.append((t1 - t0) * 1000.0)

    p50_g = float(np.percentile(times_greedy, 50))
    p95_g = float(np.percentile(times_greedy, 95))
    p99_g = float(np.percentile(times_greedy, 99))
    print(f"  • Greedy:   p50={p50_g:.2f}ms, p95={p95_g:.2f}ms, p99={p99_g:.2f}ms, tokens={len(tokens)}")

    # --- Benchmark Sampling Mode ---
    print("\n[BENCHMARK] Sampling Mode (Production)...", flush=True)
    times_sample = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens = subtalker.generate(last_h, c0, c0_embed=c0_embed, do_sample=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_sample.append((t1 - t0) * 1000.0)

    p50_s = float(np.percentile(times_sample, 50))
    p95_s = float(np.percentile(times_sample, 95))
    p99_s = float(np.percentile(times_sample, 99))
    print(f"  • Sampling: p50={p50_s:.2f}ms, p95={p95_s:.2f}ms, p99={p99_s:.2f}ms, tokens={len(tokens)}")

    print("\n" + "=" * 70)
    print(f"[SUMMARY] SubTalker Latencies:")
    print(f"  Greedy:   p50={p50_g:.2f}ms, p95={p95_g:.2f}ms")
    print(f"  Sampling: p50={p50_s:.2f}ms, p95={p95_s:.2f}ms")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
