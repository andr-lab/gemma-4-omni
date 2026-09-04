#!/usr/bin/env python3
import torch
import torch.nn.functional as F

x = torch.randn(1, 4, 1536, dtype=torch.bfloat16, device="cuda:0")
weight = torch.randn(1536, dtype=torch.bfloat16, device="cuda:0")
eps = 1e-6

# Method 1: ShapeStable formula
mean_sq = x.float().pow(2).mean(-1, keepdim=True) + eps
out1 = (x.float() * torch.pow(mean_sq, -0.5) * weight.float()).type_as(x)

# Method 2: F.rms_norm
out2 = F.rms_norm(x, (1536,), weight, eps)

diff = (out1.float() - out2.float()).abs().max().item()
print("diff between out1 and out2:", diff)

# What if weight is None?
out1_none = (x.float() * torch.pow(mean_sq, -0.5)).type_as(x)
out2_none = F.rms_norm(x, (1536,), None, eps)
print("diff None weight:", (out1_none.float() - out2_none.float()).abs().max().item())
