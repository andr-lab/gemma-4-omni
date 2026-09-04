#!/usr/bin/env python3
"""Verify full parity and benchmark Triton Gemma RMSNorm in ShapeStableGemmaListenerLM."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.listener import GemmaListener, GemmaListenerState
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM


@triton.jit
def gemma_rmsnorm_kernel(
    X, W, Y,
    stride_x_row,
    stride_y_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    x_ptr = X + row_idx * stride_x_row + col_offsets
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / N + eps
    normed = x * tl.rsqrt(mean_sq)

    if HAS_WEIGHT:
        w_ptr = W + col_offsets
        w = tl.load(w_ptr, mask=mask, other=0.0).to(tl.float32)
        normed = normed * w

    y_ptr = Y + row_idx * stride_y_row + col_offsets
    tl.store(y_ptr, normed.to(tl.bfloat16), mask=mask)


def gemma_fast_rmsnorm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    shape = x.shape
    x_2d = x.reshape(-1, shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(N)
    gemma_rmsnorm_kernel[(M,)](
        x_2d, weight, y,
        x_2d.stride(0), y.stride(0),
        N=N, eps=eps, HAS_WEIGHT=(weight is not None),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y.reshape(shape)


class TritonShapeStableLM(ShapeStableGemmaListenerLM):
    def _padded_rmsnorm(self, norm, value):
        weight = getattr(norm, "weight", None)
        eps = float(norm.eps)
        # Fast Triton path without intermediate allocations or padding loops
        return gemma_fast_rmsnorm(value, weight, eps)


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    print("=" * 70)
    print("[TRITON RMSNORM] Parity & Latency Benchmark on RTX 3090")
    print("=" * 70, flush=True)

    th = load_thinker("/workspace", device=device, dtype=dtype)
    lm = th.model.model.language_model

    listener_orig = GemmaListener.from_thinker(th)
    state_orig = GemmaListenerState(context_frames=64)
    listener_orig.prime_steady_state(state_orig)

    listener_tri = GemmaListener.from_thinker(th)
    listener_tri.backend._shape_stable_lm = TritonShapeStableLM(lm, target_length=291)
    state_tri = GemmaListenerState(context_frames=64)
    listener_tri.prime_steady_state(state_tri)

    rng = np.random.default_rng(2026)
    pcm_frames = [(0.05 * rng.standard_normal(1280)).astype(np.float32) for _ in range(50)]

    print("[PARITY] Comparing 50 frames of real audio processing...", flush=True)
    maes = []
    max_diffs = []
    cosines = []

    for i, pcm in enumerate(pcm_frames):
        out_orig = listener_orig.step(pcm, None, state_orig)
        out_tri = listener_tri.step(pcm, None, state_tri)

        h_orig = out_orig.gemma_hidden_t.float()
        h_tri = out_tri.gemma_hidden_t.float()

        diff = (h_orig - h_tri).abs()
        mae = diff.mean().item()
        m_diff = diff.max().item()
        cos = F.cosine_similarity(h_orig.view(1, -1), h_tri.view(1, -1)).item()

        maes.append(mae)
        max_diffs.append(m_diff)
        cosines.append(cos)

        if i < 5 or i % 10 == 0:
            print(f"  Frame {i:2d}: MAE={mae:.8e} | MaxDiff={m_diff:.8e} | Cosine={cos:.8f}")

    print(f"\n[PARITY RESULT over 50 frames]:")
    print(f"  Mean MAE:   {np.mean(maes):.8e}")
    print(f"  Max Diff:   {np.max(max_diffs):.8e}")
    print(f"  Min Cosine: {np.min(cosines):.8f}")

    # Latency benchmark
    print("\n[LATENCY] Benchmarking 50 frames with Triton RMSNorm...", flush=True)
    times = []
    for _ in range(50):
        pcm = (0.05 * rng.standard_normal(1280)).astype(np.float32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        listener_tri.step(pcm, None, state_tri)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    t = np.array(times)
    print(f"Listener Latency with Triton RMSNorm (50 frames):")
    print(f"  p50:  {np.percentile(t, 50):.2f} ms")
    print(f"  p95:  {np.percentile(t, 95):.2f} ms")
    print(f"  mean: {np.mean(t):.2f} ms")
    print(f"  min:  {np.min(t):.2f} ms")
    print(f"  max:  {np.max(t):.2f} ms")


if __name__ == "__main__":
    main()
