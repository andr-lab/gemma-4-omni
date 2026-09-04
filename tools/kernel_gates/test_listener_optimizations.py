#!/usr/bin/env python3
"""Measure breakdown of Gemma Listener components on RTX 3090."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.listener import GemmaListener, GemmaListenerState
from duplex_kernel.models.thinker import load_thinker


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    print("=" * 70)
    print("[BREAKDOWN] Gemma Listener Component Profiling on RTX 3090")
    print("=" * 70, flush=True)

    th = load_thinker("/workspace", device=device, dtype=dtype)
    listener = GemmaListener.from_thinker(th)

    state = GemmaListenerState(context_frames=64)
    listener.prime_steady_state(state)

    dummy_pcm = (0.05 * np.random.default_rng(42).standard_normal(1280)).astype(np.float32)

    # Warmup
    for _ in range(16):
        listener.step(dummy_pcm, None, state)
    torch.cuda.synchronize()

    # Monkey patch or instrument backend methods to measure exact breakdown
    backend = listener.backend
    orig_extract = backend._extract_single_frame
    orig_lm = backend._forward_incremental_lm

    t_mel_list = []
    t_tower_list = []
    t_lm_list = []

    def timed_extract(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_extract(*args, **kwargs)
        t_mel_list.append((time.perf_counter() - t0) * 1000.0)
        return res

    def timed_lm(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = orig_lm(*args, **kwargs)
        torch.cuda.synchronize()
        t_lm_list.append((time.perf_counter() - t0) * 1000.0)
        return res

    backend._extract_single_frame = timed_extract
    backend._forward_incremental_lm = timed_lm

    print("[BREAKDOWN] Running 50 frames with instrumentation...", flush=True)
    for _ in range(50):
        listener.step(dummy_pcm, None, state)
    torch.cuda.synchronize()

    t_mel = np.array(t_mel_list)
    t_lm = np.array(t_lm_list)

    print(f"Incremental Mel Extraction (NumPy):")
    print(f"  p50: {np.percentile(t_mel, 50):.3f} ms | p95: {np.percentile(t_mel, 95):.3f} ms")

    print(f"ShapeStableLM Forward (PyTorch 28-layer):")
    print(f"  p50: {np.percentile(t_lm, 50):.2f} ms | p95: {np.percentile(t_lm, 95):.2f} ms")

    # Measure CUDA Graph replay for Audio Tower alone
    print("\n[BREAKDOWN] Measuring Audio Tower CUDA Graph Replay alone...", flush=True)
    tower_times = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        backend._cuda_graph.replay()
        torch.cuda.synchronize()
        tower_times.append((time.perf_counter() - t0) * 1000.0)
    
    t_tower = np.array(tower_times)
    print(f"Audio Tower CUDA Graph Replay:")
    print(f"  p50: {np.percentile(t_tower, 50):.2f} ms | p95: {np.percentile(t_tower, 95):.2f} ms")


if __name__ == "__main__":
    main()
