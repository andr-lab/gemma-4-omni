#!/usr/bin/env python3
import torch
from pathlib import Path

v9_path = Path("/workspace/checkpoints/v9_causal_l0_unknown.pt")
heads_path = Path("/workspace/_archive_ckpts/gemma4_duplex_sota/custom_duplex_heads.pt")

sd_v9 = torch.load(v9_path, map_location="cpu", weights_only=True)
sd_heads = torch.load(heads_path, map_location="cpu", weights_only=True)

cc_v9 = sd_v9["causal_conv"]
cc_heads = sd_heads["causal_conv"]

for k in ["bias", "weight"]:
    t_v9 = cc_v9[k]
    t_h = cc_heads[k]
    print(f"=== {k} ===")
    print(f"V9:    dtype={t_v9.dtype}, shape={t_v9.shape}")
    print(f"Heads: dtype={t_h.dtype}, shape={t_h.shape}")
    diff = (t_v9.float() - t_h.float()).abs()
    print(f"Max diff: {diff.max().item():.8e}, Mean diff: {diff.mean().item():.8e}")
    mismatch_count = (t_v9 != t_h).sum().item()
    total_count = t_v9.numel()
    print(f"Mismatch elements: {mismatch_count} / {total_count} ({mismatch_count/total_count*100:.2f}%)")
    if t_v9.dtype != t_h.dtype:
        diff_cast = (t_v9.float() - t_h.to(t_v9.dtype).float()).abs()
        print(f"Diff after cast: {diff_cast.max().item():.8e}")
