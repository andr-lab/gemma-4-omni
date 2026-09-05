#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")
import torch
from duplex_kernel.models.thinker import load_thinker

t = load_thinker("/workspace")
lm = t.model.model.language_model
rot = lm.rotary_emb
print("rotary_emb:", type(rot))
print("rotary_emb attrs:", [a for a in dir(rot) if not a.startswith("_")])
pos = torch.arange(0, 4, device="cuda").unsqueeze(0)
x = torch.zeros((1, 4, 1536), device="cuda", dtype=torch.bfloat16)
for lt in ["sliding_attention", "full_attention"]:
    cos, sin = rot(x, pos, lt)
    print(lt, "cos shape:", cos.shape, "sin shape:", sin.shape, "dtype:", cos.dtype)
    print(lt, "cos[0, 0, :8]:", cos[0, 0, :8].float().cpu().tolist())
    print(lt, "sin[0, 0, :8]:", sin[0, 0, :8].float().cpu().tolist())
    print(lt, "cos[0, 1, :8]:", cos[0, 1, :8].float().cpu().tolist())
    print(lt, "sin[0, 1, :8]:", sin[0, 1, :8].float().cpu().tolist())

print("sliding inv_freq shape:", rot.sliding_attention_inv_freq.shape)
print("full inv_freq shape:", rot.full_attention_inv_freq.shape)
print("sliding attn scaling:", getattr(rot, "sliding_attention_attention_scaling", None))
print("full attn scaling:", getattr(rot, "full_attention_attention_scaling", None))
print("sliding inv_freq[:8]:", rot.sliding_attention_inv_freq[:8].float().tolist())
print("full inv_freq[:8]:", rot.full_attention_inv_freq[:8].float().tolist())
