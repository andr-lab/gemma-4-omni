#!/usr/bin/env python3
"""Check prime_steady_state cache difference."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.listener import GemmaListener, GemmaListenerState
from tools.kernel_gates.test_triton_listener_parity import TritonShapeStableLM

th = load_thinker("/workspace", device="cuda:0", dtype=torch.bfloat16)

l_base = GemmaListener.from_thinker(th)
s_base = GemmaListenerState(context_frames=64)
l_base.prime_steady_state(s_base)

l_tri = GemmaListener.from_thinker(th)
l_tri.backend._shape_stable_lm = TritonShapeStableLM(th.model.model.language_model, target_length=291)
s_tri = GemmaListenerState(context_frames=64)
l_tri.prime_steady_state(s_tri)

# Compare key/values in s_base vs s_tri
max_diff = 0.0
for layer_idx in range(len(s_base.gemma_lm_cache.layers)):
    kb = s_base.gemma_lm_cache.layers[layer_idx].keys
    kt = s_tri.gemma_lm_cache.layers[layer_idx].keys
    d = (kb.float() - kt.float()).abs().max().item()
    if d > max_diff:
        max_diff = d

print(f"Max diff in KV cache after prime_steady_state: {max_diff:.8e}")
