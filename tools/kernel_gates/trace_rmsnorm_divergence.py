#!/usr/bin/env python3
import sys
sys.path.insert(0, "/workspace")
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

# Check steady state KV cache equality after prime_steady_state!
print("Comparing KV cache after prime_steady_state:")
print("Comparing KV cache after prime_steady_state:")
print("DynamicCache attributes:", [a for a in dir(s_base.gemma_lm_cache) if not a.startswith("__")])
cache_b = s_base.gemma_lm_cache
cache_f = s_fast.gemma_lm_cache
print("layer_b attributes:", [a for a in dir(cache_b.layers[0]) if not a.startswith("__")])
for l, (layer_b, layer_f) in enumerate(zip(cache_b.layers, cache_f.layers)):
    kb, vb = layer_b.keys, layer_b.values
    kf, vf = layer_f.keys, layer_f.values
    k_diff = (kb.float() - kf.float()).abs().max().item()
    v_diff = (vb.float() - vf.float()).abs().max().item()
    if k_diff > 0 or v_diff > 0:
        print(f"  Layer {l} KV cache diff after prime! K diff={k_diff:.8e}, V diff={v_diff:.8e}")
        break
else:
    print("  ALL KV cache tensors are bit-exact after prime_steady_state!")
