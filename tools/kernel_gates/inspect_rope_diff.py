#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import torch
from duplex_kernel.models.thinker import load_thinker

th = load_thinker("/workspace", device="cpu", dtype=torch.bfloat16)
lm = th.model.model.language_model
rotary = lm.rotary_emb

buf_f = rotary._buffers["full_attention_inv_freq"]

# Correct proportional formula:
rope_angles = int(0.25 * 512 // 2) # 64
inv_rotated = 1.0 / (1000000.0 ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / 512.0))
inv_correct = torch.cat([inv_rotated, torch.zeros(256 - rope_angles, dtype=torch.float32)])

max_diff = (buf_f - inv_correct).abs().max().item()
print(f"Max diff between buf_f and inv_correct: {max_diff:.8e}")
assert max_diff == 0.0, f"Mismatch: {max_diff}"
print("PROPORTIONAL ROPE FORMULA MATCHES 100% BIT-EXACT!")
