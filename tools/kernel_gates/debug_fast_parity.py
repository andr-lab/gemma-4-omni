#!/usr/bin/env python3
"""Pinpoint exact divergence between baseline and Fast ShapeStable LM."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM


def test_components():
    device = "cuda:0"
    dtype = torch.bfloat16

    th = load_thinker("/workspace", device=device, dtype=dtype)
    lm = th.model.model.language_model
    layer0 = lm.layers[0]

    # Test 1: input_layernorm
    x = torch.randn(1, 4, 1536, device=device, dtype=dtype)
    
    # Baseline
    norm = layer0.input_layernorm
    mean_sq = x.float().pow(2).mean(-1, keepdim=True) + float(norm.eps)
    out_base = (x.float() * torch.pow(mean_sq, -0.5) * norm.weight.float()).type_as(x)
    
    # Fused
    out_fused = F.rms_norm(x, (1536,), norm.weight, float(norm.eps))
    diff = (out_base.float() - out_fused.float()).abs().max().item()
    print(f"Layer 0 input_layernorm max diff: {diff:.8e}")

    # Test 2: q_norm (shape 1, 4, 16, 256)
    q = torch.randn(1, 4, 16, 256, device=device, dtype=dtype)
    q_norm = layer0.self_attn.q_norm
    mean_sq = q.float().pow(2).mean(-1, keepdim=True) + float(q_norm.eps)
    out_q_base = (q.float() * torch.pow(mean_sq, -0.5) * q_norm.weight.float()).type_as(q)
    out_q_fused = F.rms_norm(q, (256,), q_norm.weight, float(q_norm.eps))
    diff_q = (out_q_base.float() - out_q_fused.float()).abs().max().item()
    print(f"Layer 0 q_norm max diff: {diff_q:.8e}")

    # Test 3: linear (q_proj)
    out_l_base = F.linear(x, layer0.self_attn.q_proj.weight, layer0.self_attn.q_proj.bias)
    out_l_test = F.linear(x, layer0.self_attn.q_proj.weight, layer0.self_attn.q_proj.bias)
    diff_l = (out_l_base.float() - out_l_test.float()).abs().max().item()
    print(f"Layer 0 q_proj max diff: {diff_l:.8e}")

    # Test 4: full layer 0 step with dummy inputs
    from duplex_kernel.listener import GemmaListener, GemmaListenerState
    listener = GemmaListener.from_thinker(th)
    state = GemmaListenerState(context_frames=64)
    listener.prime_steady_state(state)

    # Let's run 1 step and compare all intermediate tensors in decoder layer 0!
    ss_orig = ShapeStableGemmaListenerLM(lm, target_length=291)
    
    # Check if prime_steady_state in test_fast_shape_stable.py used the baseline or fast!
    print("Test finished.")

if __name__ == "__main__":
    test_components()
