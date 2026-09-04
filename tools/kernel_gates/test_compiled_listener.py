#!/usr/bin/env python3
"""Test torch.compile on ShapeStableGemmaListenerLM on RTX 3090."""
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

from duplex_kernel.listener import GemmaListener, GemmaListenerState
from duplex_kernel.models.thinker import load_thinker


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    print("=" * 70)
    print("[TEST] Testing torch.compile on Gemma Listener LM")
    print("=" * 70, flush=True)

    th = load_thinker("/workspace", device=device, dtype=dtype)
    listener = GemmaListener.from_thinker(th)
    state = GemmaListenerState(context_frames=64)
    listener.prime_steady_state(state)

    rng = np.random.default_rng(2026)
    dummy_pcm = (0.05 * rng.standard_normal(1280)).astype(np.float32)

    # 1. Warmup eager for 5 frames
    print("[TEST] Running 5 frames eager...", flush=True)
    for _ in range(5):
        listener.step(dummy_pcm, None, state)
    torch.cuda.synchronize()

    # 2. Compile _shape_stable_lm.forward
    print("[TEST] Compiling ShapeStableGemmaListenerLM.forward with torch.compile...", flush=True)
    lm_view = listener.backend._shape_stable_lm
    lm_view.forward = torch.compile(lm_view.forward, mode="default")

    # 3. Warmup compiled forward (1 frame to compile)
    print("[TEST] Running warmup step to trigger compilation...", flush=True)
    t0 = time.perf_counter()
    out0 = listener.step(dummy_pcm, None, state)
    torch.cuda.synchronize()
    t_compile = time.perf_counter() - t0
    print(f"[TEST] First step (compilation included): {t_compile:.2f}s", flush=True)

    # 4. Benchmark 30 frames
    print("[TEST] Benchmarking 30 compiled frames...", flush=True)
    times = []
    for _ in range(30):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        listener.step(dummy_pcm, None, state)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    t = np.array(times)
    print(f"\nListener Step Latency with torch.compile (30 frames):")
    print(f"  p50:  {np.percentile(t, 50):.2f} ms")
    print(f"  p95:  {np.percentile(t, 95):.2f} ms")
    print(f"  mean: {np.mean(t):.2f} ms")
    print(f"  min:  {np.min(t):.2f} ms")
    print(f"  max:  {np.max(t):.2f} ms")


if __name__ == "__main__":
    main()
