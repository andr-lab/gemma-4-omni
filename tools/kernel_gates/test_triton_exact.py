#!/usr/bin/env python3
"""Test tl.sqrt vs tl.rsqrt in Triton for bit-exact RMSNorm."""
import torch
import triton
import triton.language as tl

@triton.jit
def test_kernel(X, Y_rsqrt, Y_exact, N: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / N + eps
    
    # 1. rsqrt approx
    out_rsqrt = x * tl.rsqrt(mean_sq)
    tl.store(Y_rsqrt + offsets, out_rsqrt.to(tl.bfloat16), mask=mask)
    
    # 2. exact sqrt
    out_exact = x / tl.sqrt(mean_sq)
    tl.store(Y_exact + offsets, out_exact.to(tl.bfloat16), mask=mask)

x = torch.randn(1, 1536, dtype=torch.bfloat16, device="cuda:0")
eps = 1e-6

mean_sq = x.float().pow(2).mean(-1, keepdim=True) + eps
ref = (x.float() * torch.pow(mean_sq, -0.5)).type_as(x)

y_rsqrt = torch.empty_like(x)
y_exact = torch.empty_like(x)

test_kernel[(1,)](x, y_rsqrt, y_exact, N=1536, eps=eps, BLOCK_SIZE=2048)

diff_rsqrt = (ref.float() - y_rsqrt.float()).abs().max().item()
diff_exact = (ref.float() - y_exact.float()).abs().max().item()

print(f"diff_rsqrt: {diff_rsqrt:.8e}")
print(f"diff_exact: {diff_exact:.8e}")
