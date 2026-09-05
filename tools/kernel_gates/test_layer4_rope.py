#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import torch
from duplex_kernel.models.thinker import load_thinker

th = load_thinker("/workspace", device="cpu", dtype=torch.bfloat16)
lm = th.model.model.language_model

print("Config layer_types:", lm.config.layer_types[:10])
print("unique_layer_types:", lm.unique_layer_types)

rotary = lm.rotary_emb
print("rotary_emb:", type(rotary))

for lt in lm.unique_layer_types:
    print(f"\n--- Layer type: {lt} ---")
    pos_ids = torch.arange(0, 35).unsqueeze(0)
    dummy = torch.zeros(1, 35, 1536, dtype=torch.bfloat16)
    cos, sin = lm.rotary_emb(dummy, pos_ids, lt)
    print(f"cos shape: {cos.shape}, sin shape: {sin.shape}")
    print(f"cos[0, :3, :6]:\n{cos[0, :3, :6]}")
    print(f"sin[0, :3, :6]:\n{sin[0, :3, :6]}")
    print(f"cos[0, :3, 250:256]:\n{cos[0, :3, 250:256]}")
    print(f"sin[0, :3, 250:256]:\n{sin[0, :3, 250:256]}")

# Inspect buffers of rotary
print("\n--- Rotary buffers ---")
for k, v in rotary._buffers.items():
    if v is not None:
        print(f"Buffer {k}: shape {v.shape}, dtype {v.dtype}")
        print(f"  first 5: {v.flatten()[:5]}")
        print(f"  last 5: {v.flatten()[-5:]}")

# Now compare with Native precomputed inv_freq
print("\n--- Checking inv_freq formula ---")
# Sliding (D=256, half=128, base=10000)
inv_s = 1.0 / (10000.0 ** (torch.arange(0, 128, dtype=torch.float32) * 2 / 256.0))
# Full (D=512, half=256, base=1000000)
inv_f = 1.0 / (1000000.0 ** (torch.arange(0, 256, dtype=torch.float32) * 2 / 512.0))

print(f"inv_s: first 3 = {inv_s[:3]}, last 3 = {inv_s[-3:]}")
print(f"inv_f: first 3 = {inv_f[:3]}, last 3 = {inv_f[-3:]}")

# Check apply_rotary_pos_emb on full attention
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
print("apply_rotary_pos_emb signature and test:")
q_test = torch.ones(1, 4, 8, 512, dtype=torch.float32)
cos_full, sin_full = lm.rotary_emb(dummy[:, :4], torch.arange(0, 4).unsqueeze(0), "full_attention")
print(f"cos_full shape: {cos_full.shape}")
q_rot = apply_rotary_pos_emb(q_test, cos_full, sin_full, unsqueeze_dim=2)
print(f"q_rot shape: {q_rot.shape}")
print(f"q_rot[0, 0, 0, :6]: {q_rot[0, 0, 0, :6]}")
