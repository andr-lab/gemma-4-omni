#!/usr/bin/env python3
"""Benchmark F.rms_norm and fast path in ShapeStableGemmaListenerLM on RTX 3090."""
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
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM


class FastShapeStableGemmaListenerLM(ShapeStableGemmaListenerLM):
    """Optimized shape-stable listener view with fused RMSNorm and direct linear dispatch."""

    def _padded_linear(self, linear, value):
        # Fast path when length matches workspace stream length
        axis = self._token_axis(value)
        if value.shape[axis] == self._workspace.target_length:
            return F.linear(value, linear.weight, linear.bias)
        return super()._padded_linear(linear, value)

    def _padded_rmsnorm(self, norm, value):
        axis = self._token_axis(value)
        weight = getattr(norm, "weight", None)
        eps = float(norm.eps)
        norm_shape = (value.shape[-1],)
        if value.shape[axis] == self._workspace.target_length:
            return F.rms_norm(value, norm_shape, weight, eps)
        # Fallback to padded if shape doesn't match
        padded = self._workspace.padded(value, axis=axis)
        out = F.rms_norm(padded, norm_shape, weight, eps)
        if padded is value:
            return out
        original_length = int(value.shape[axis])
        slices = [slice(None)] * out.ndim
        slices[axis] = slice(0, original_length)
        return out[tuple(slices)]


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    print("=" * 70)
    print("[BENCHMARK & PARITY] Testing FastShapeStableGemmaListenerLM")
    print("=" * 70, flush=True)

    th = load_thinker("/workspace", device=device, dtype=dtype)
    
    # 1. Parity test: run 50 frames with baseline listener vs fast listener
    listener_orig = GemmaListener.from_thinker(th)
    state_orig = GemmaListenerState(context_frames=64)
    listener_orig.prime_steady_state(state_orig)

    listener_fast = GemmaListener.from_thinker(th)
    # Replace with fast ShapeStable LM
    listener_fast.backend._shape_stable_lm = FastShapeStableGemmaListenerLM(
        th.model.model.language_model, target_length=291
    )
    state_fast = GemmaListenerState(context_frames=64)
    listener_fast.prime_steady_state(state_fast)

    rng = np.random.default_rng(2026)
    pcm_frames = [(0.05 * rng.standard_normal(1280)).astype(np.float32) for _ in range(50)]

    print("[PARITY] Comparing 50 frames between baseline and fast LM...", flush=True)
    maes = []
    cosines = []
    max_diffs = []

    for i, pcm in enumerate(pcm_frames):
        out_orig = listener_orig.step(pcm, None, state_orig)
        out_fast = listener_fast.step(pcm, None, state_fast)
        
        h_orig = out_orig.gemma_hidden_t.float()
        h_fast = out_fast.gemma_hidden_t.float()
        
        diff = (h_orig - h_fast).abs()
        mae = diff.mean().item()
        m_diff = diff.max().item()
        cos = F.cosine_similarity(h_orig.view(1, -1), h_fast.view(1, -1)).item()
        
        maes.append(mae)
        cosines.append(cos)
        max_diffs.append(m_diff)

    print(f"Parity over 50 frames:")
    print(f"  Mean MAE:     {np.mean(maes):.8e}")
    print(f"  Max Diff:     {np.max(max_diffs):.8e}")
    print(f"  Min Cosine:   {np.min(cosines):.8f}")

    # 2. Timing benchmark of 100 frames
    print("\n[LATENCY] Benchmarking 100 frames of fast listener...", flush=True)
    times_fast = []
    for _ in range(100):
        pcm = (0.05 * rng.standard_normal(1280)).astype(np.float32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        listener_fast.step(pcm, None, state_fast)
        torch.cuda.synchronize()
        times_fast.append((time.perf_counter() - t0) * 1000.0)

    t_fast = np.array(times_fast)
    print(f"Fast Listener Complete Step Latency (100 frames):")
    print(f"  p50:  {np.percentile(t_fast, 50):.2f} ms")
    print(f"  p95:  {np.percentile(t_fast, 95):.2f} ms")
    print(f"  mean: {np.mean(t_fast):.2f} ms")
    print(f"  min:  {np.min(t_fast):.2f} ms")
    print(f"  max:  {np.max(t_fast):.2f} ms")


if __name__ == "__main__":
    main()
