#!/usr/bin/env python3
import sys
sys.path.insert(0, "/workspace")
import time
import numpy as np
import torch
import torch.nn.functional as F
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from duplex_kernel.listener import GemmaListener, GemmaListenerState

class UnpaddedRMSNormLM(ShapeStableGemmaListenerLM):
    def _padded_rmsnorm(self, norm, value):
        mean_squared = value.float().pow(2).mean(-1, keepdim=True) + float(norm.eps)
        output = value.float() * torch.pow(mean_squared, -0.5)
        weight = getattr(norm, "weight", None)
        if weight is not None:
            output = output * weight.float()
        return output.type_as(value)

device = "cuda:0"
dtype = torch.bfloat16

th = load_thinker("/workspace", device=device, dtype=dtype)
lm = th.model.model.language_model

l_base = GemmaListener.from_thinker(th)
s_base = GemmaListenerState(context_frames=64)
l_base.prime_steady_state(s_base)

l_unpad = GemmaListener.from_thinker(th)
l_unpad.backend._shape_stable_lm = UnpaddedRMSNormLM(lm, target_length=291, stream_length=4)
s_unpad = GemmaListenerState(context_frames=64)
l_unpad.prime_steady_state(s_unpad)

print("Checking KV cache after prime_steady_state:")
for l, (lb, lu) in enumerate(zip(s_base.gemma_lm_cache.layers, s_unpad.gemma_lm_cache.layers)):
    kb, vb = lb.keys, lb.values
    ku, vu = lu.keys, lu.values
    k_diff = (kb.float() - ku.float()).abs().max().item()
    v_diff = (vb.float() - vu.float()).abs().max().item()
    if k_diff > 0 or v_diff > 0:
        print(f"  Layer {l} KV cache diff after prime! K diff={k_diff:.8e}, V diff={v_diff:.8e}")
        break
else:
    print("  ALL KV cache tensors are bit-exact after prime_steady_state!")

rng = np.random.default_rng(2026)
pcm_frames = [(0.05 * rng.standard_normal(1280)).astype(np.float32) for _ in range(50)]

print("\nComparing 50 frames between baseline and UnpaddedRMSNormLM...")
maes = []
max_diffs = []
cosines = []

for i, pcm in enumerate(pcm_frames):
    out_b = l_base.step(pcm, None, s_base)
    out_u = l_unpad.step(pcm, None, s_unpad)
    
    hb = out_b.gemma_hidden_t.float()
    hu = out_u.gemma_hidden_t.float()
    
    diff = (hb - hu).abs()
    mae = diff.mean().item()
    m_diff = diff.max().item()
    cos = F.cosine_similarity(hb.view(1, -1), hu.view(1, -1)).item()
    
    maes.append(mae)
    max_diffs.append(m_diff)
    cosines.append(cos)
    if i < 5 or i % 10 == 0:
        print(f"  Frame {i:2d}: MAE={mae:.8e} | MaxDiff={m_diff:.8e} | Cosine={cos:.8f}")

print(f"\n[PARITY RESULTS over 50 frames]:")
print(f"  Mean MAE:   {np.mean(maes):.8e}")
print(f"  Max Diff:   {np.max(max_diffs):.8e}")
print(f"  Min Cosine: {np.min(cosines):.8f}")
