#!/usr/bin/env python3
"""Verifica formale della provenienza di causal_conv:
Dimostra che la causal_conv salvata nel checkpoint causale v9_causal_l0_final.pt (o v9_causal_l0_unknown.pt)
e' identica bit-per-bit alla causal_conv caricata da custom_duplex_heads.pt.
"""
import sys
from pathlib import Path
import torch

v9_path = Path("/workspace/checkpoints/v9_causal_l0_final.pt")
if not v9_path.exists():
    v9_path = Path("/workspace/checkpoints/v9_causal_l0_unknown.pt")

heads_path = Path("/workspace/_archive_ckpts/gemma4_duplex_sota/custom_duplex_heads.pt")

print(f"Loading V9 causal from: {v9_path}")
print(f"Loading Heads from:     {heads_path}")

sd_v9 = torch.load(v9_path, map_location="cpu", weights_only=True)
sd_heads = torch.load(heads_path, map_location="cpu", weights_only=True)

if "causal_conv" not in sd_v9:
    print("FAIL: 'causal_conv' non trovata nel checkpoint causale V9!")
    sys.exit(1)

if "causal_conv" not in sd_heads:
    print("FAIL: 'causal_conv' non trovata in custom_duplex_heads.pt!")
    sys.exit(1)

cc_v9 = sd_v9["causal_conv"]
cc_heads = sd_heads["causal_conv"]

keys_v9 = sorted(list(cc_v9.keys()))
keys_heads = sorted(list(cc_heads.keys()))

print(f"Keys in v9:    {keys_v9}")
print(f"Keys in heads: {keys_heads}")

if keys_v9 != keys_heads:
    print(f"FAIL: Chiavi disallineate! {keys_v9} != {keys_heads}")
    sys.exit(1)

all_equal = True
for k in keys_v9:
    t_v9 = cc_v9[k]
    t_heads = cc_heads[k]
    is_eq = torch.equal(t_v9, t_heads)
    max_d = (t_v9.float() - t_heads.float()).abs().max().item()
    print(f"Tensor '{k}': shape={tuple(t_v9.shape)}, dtype={t_v9.dtype}, torch.equal={is_eq}, max_diff={max_d}")
    if not is_eq:
        all_equal = False

if all_equal:
    print("\nSUCCESS: causal_conv nel checkpoint causale e' 100% identica bit-per-bit a custom_duplex_heads.pt!")
    sys.exit(0)
else:
    print("\nFAIL: causal_conv differisce tra i due checkpoint!")
    sys.exit(1)
