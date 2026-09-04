#!/usr/bin/env python3
import sys
sys.path.insert(0, "/workspace")
import time
import numpy as np
import torch
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

l_unpad = GemmaListener.from_thinker(th)
l_unpad.backend._shape_stable_lm = UnpaddedRMSNormLM(lm, target_length=291, stream_length=4)
s_unpad = GemmaListenerState(context_frames=64)
l_unpad.prime_steady_state(s_unpad)

rng = np.random.default_rng(2026)
pcm_frames = [(0.05 * rng.standard_normal(1280)).astype(np.float32) for _ in range(50)]

# Warmup
for pcm in pcm_frames[:5]:
    l_unpad.step(pcm, None, s_unpad)
torch.cuda.synchronize()

times = []
for pcm in pcm_frames:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    l_unpad.step(pcm, None, s_unpad)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000.0)

t = np.array(times)
print(f"UnpaddedRMSNormLM Step Latency (50 frames):")
print(f"  p50:  {np.percentile(t, 50):.2f} ms")
print(f"  p95:  {np.percentile(t, 95):.2f} ms")
print(f"  mean: {np.mean(t):.2f} ms")
print(f"  min:  {np.min(t):.2f} ms")
print(f"  max:  {np.max(t):.2f} ms")
