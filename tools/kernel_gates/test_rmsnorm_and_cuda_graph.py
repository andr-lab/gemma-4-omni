#!/usr/bin/env python3
"""Test F.rms_norm and CUDA Graph opportunities for ShapeStableGemmaListenerLM."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

print("PyTorch version:", torch.__version__)
print("Has F.rms_norm:", hasattr(F, "rms_norm"))

# Let's test F.rms_norm vs manual Gemma RMSNorm numerical parity
x = torch.randn(1, 4, 1536, dtype=torch.bfloat16, device="cuda:0")
w = torch.ones(1536, dtype=torch.bfloat16, device="cuda:0")
eps = 1e-6

# Manual Gemma RMSNorm
mean_sq = x.float().pow(2).mean(-1, keepdim=True) + eps
out_manual = (x.float() * torch.pow(mean_sq, -0.5) * w.float()).type_as(x)

if hasattr(F, "rms_norm"):
    out_fused = F.rms_norm(x, (1536,), weight=w, eps=eps)
    diff = (out_manual.float() - out_fused.float()).abs().max().item()
    print("F.rms_norm vs Manual max diff:", diff)
else:
    print("F.rms_norm not in torch.nn.functional")
