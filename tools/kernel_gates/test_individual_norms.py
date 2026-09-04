#!/usr/bin/env python3
"""Pinpoint exact divergence between baseline and Triton in Layer 0."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

th = load_thinker("/workspace", device="cuda:0", dtype=torch.bfloat16)
lm = th.model.model.language_model

from duplex_kernel.listener import GemmaListener, GemmaListenerState
l_base = GemmaListener.from_thinker(th)
s_base = GemmaListenerState(context_frames=64)
l_base.prime_steady_state(s_base)

# Test single frame inputs
rng = np.random.default_rng(2026)
pcm = (0.05 * rng.standard_normal(1280)).astype(np.float32)

# Get inputs to _forward_incremental_lm
backend = l_base.backend
frame_rows = 4
soft = torch.randn((1, 4, 1536), device="cuda:0", dtype=torch.bfloat16)
ple = backend._frame_pad_per_layer_inputs(lm)

# Baseline RMSNorm vs Triton RMSNorm on ple and soft
norm0 = lm.layers[0].input_layernorm
with torch.inference_mode():
    ref_norm = l_base.backend._shape_stable_lm._padded_rmsnorm(norm0, soft)
    tri_norm = gemma_fast_rmsnorm(soft, norm0.weight, float(norm0.eps))
    print("input_layernorm diff:", (ref_norm.float() - tri_norm.float()).abs().max().item())

    # Test attention q_norm, k_norm, v_norm
    q = torch.randn((1, 4, 16, 256), device="cuda:0", dtype=torch.bfloat16)
    ref_q = l_base.backend._shape_stable_lm._padded_rmsnorm(lm.layers[0].self_attn.q_norm, q)
    tri_q = gemma_fast_rmsnorm(q, lm.layers[0].self_attn.q_norm.weight, float(lm.layers[0].self_attn.q_norm.eps))
    print("q_norm diff:", (ref_q.float() - tri_q.float()).abs().max().item())

    ref_v = l_base.backend._shape_stable_lm._padded_rmsnorm(lm.layers[0].self_attn.v_norm, q)
    tri_v = gemma_fast_rmsnorm(q, None, float(lm.layers[0].self_attn.v_norm.eps))
    print("v_norm (weight=None) diff:", (ref_v.float() - tri_v.float()).abs().max().item())

    # Test post_per_layer_input_norm
    ref_ple_norm = l_base.backend._shape_stable_lm._padded_rmsnorm(lm.per_layer_projection_norm, torch.randn(1, 4, 35, 256, device="cuda:0", dtype=torch.bfloat16))
    tri_ple_norm = gemma_fast_rmsnorm(torch.randn(1, 4, 35, 256, device="cuda:0", dtype=torch.bfloat16), lm.per_layer_projection_norm.weight, float(lm.per_layer_projection_norm.eps))
    print("All individual RMSNorm diffs checked!")
