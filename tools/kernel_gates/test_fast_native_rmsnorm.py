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

class FastRMSNormLM(ShapeStableGemmaListenerLM):
    def _padded_rmsnorm(self, norm, value):
        weight = getattr(norm, "weight", None)
        eps = float(norm.eps)
        return F.rms_norm(value, (value.shape[-1],), weight, eps)

device = "cuda:0"
dtype = torch.bfloat16

th = load_thinker("/workspace", device=device, dtype=dtype)
lm = th.model.model.language_model

l_base = GemmaListener.from_thinker(th)
s_base = GemmaListenerState(context_frames=64)
l_base.prime_steady_state(s_base)

l_fast = GemmaListener.from_thinker(th)
l_fast.backend._shape_stable_lm = FastRMSNormLM(lm, target_length=291, stream_length=4)
s_fast = GemmaListenerState(context_frames=64)
l_fast.prime_steady_state(s_fast)

rng = np.random.default_rng(2026)
pcm_frames = [(0.05 * rng.standard_normal(1280)).astype(np.float32) for _ in range(50)]

print("Comparing 50 frames between baseline and FastRMSNormLM...")
maes = []
max_diffs = []
cosines = []

for i, pcm in enumerate(pcm_frames):
    out_b = l_base.step(pcm, None, s_base)
    out_f = l_fast.step(pcm, None, s_fast)
    
    hb = out_b.gemma_hidden_t.float()
    hf = out_f.gemma_hidden_t.float()
    
    diff = (hb - hf).abs()
    mae = diff.mean().item()
    m_diff = diff.max().item()
    cos = F.cosine_similarity(hb.view(1, -1), hf.view(1, -1)).item()
    
    maes.append(mae)
    max_diffs.append(m_diff)
    cosines.append(cos)
    if i < 5 or i % 10 == 0:
        print(f"  Frame {i:2d}: MAE={mae:.8e} | MaxDiff={m_diff:.8e} | Cosine={cos:.8f}")

print(f"\n[PARITY RESULTS over 50 frames]:")
print(f"  Mean MAE:   {np.mean(maes):.8e}")
print(f"  Max Diff:   {np.max(max_diffs):.8e}")
print(f"  Min Cosine: {np.min(cosines):.8f}")

# Measure latency
times_b = []
times_f = []
for pcm in pcm_frames:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    l_base.step(pcm, None, s_base)
    torch.cuda.synchronize()
    times_b.append((time.perf_counter() - t0) * 1000)
    
    t0 = time.perf_counter()
    l_fast.step(pcm, None, s_fast)
    torch.cuda.synchronize()
    times_f.append((time.perf_counter() - t0) * 1000)

tb = np.array(times_b)
tf = np.array(times_f)
print(f"\n[LATENCY COMPARISON (50 frames)]:")
print(f"  Baseline: p50={np.percentile(tb, 50):.2f} ms, p95={np.percentile(tb, 95):.2f} ms")
print(f"  Fast RMS: p50={np.percentile(tf, 50):.2f} ms, p95={np.percentile(tf, 95):.2f} ms")
print(f"  SAVINGS:  {np.percentile(tb, 50) - np.percentile(tf, 50):.2f} ms!")
