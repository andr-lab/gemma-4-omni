#!/usr/bin/env python3
"""Test Triton on RTX 3090."""
import torch
import triton
import triton.language as tl

print("Triton version:", triton.__version__)

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

    # Gemma exact formula: mean_squared = mean(x^2) + eps
    mean_sq = tl.sum(x * x, axis=0) / N + eps
    normed = x * tl.rsqrt(mean_sq)

    if HAS_WEIGHT:
        w_ptr = W + col_offsets
        w = tl.load(w_ptr, mask=mask, other=0.0).to(tl.float32)
        normed = normed * w

    y_ptr = Y + row_idx * stride_y_row + col_offsets
    tl.store(y_ptr, normed.to(tl.bfloat16), mask=mask)

def gemma_rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    shape = x.shape
    x_2d = x.reshape(-1, shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(N)
    has_w = weight is not None
    gemma_rmsnorm_kernel[(M,)](
        x_2d, weight, y,
        x_2d.stride(0), y.stride(0),
        N=N, eps=eps, HAS_WEIGHT=has_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y.reshape(shape)

# Test numerical parity against pure PyTorch Gemma RMSNorm
x = torch.randn(1, 4, 1536, dtype=torch.bfloat16, device="cuda:0")
w = torch.randn(1536, dtype=torch.bfloat16, device="cuda:0")
eps = 1e-6

# Reference PyTorch
mean_sq = x.float().pow(2).mean(-1, keepdim=True) + eps
ref = (x.float() * torch.pow(mean_sq, -0.5) * w.float()).type_as(x)

# Triton
tri = gemma_rmsnorm_triton(x, w, eps=eps)

diff = (ref.float() - tri.float()).abs().max().item()
print("Max diff between Reference and Triton Gemma RMSNorm:", diff)
print("Cosine:", torch.nn.functional.cosine_similarity(ref.view(1, -1).float(), tri.view(1, -1).float()).item())
