#!/usr/bin/env python3
"""Inspect all norms in Gemma 4 to see weights and shapes."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.thinker import load_thinker

th = load_thinker("/workspace", device="cuda:0", dtype=torch.bfloat16)
lm = th.model.model.language_model

print("lm.norm:", type(lm.norm), getattr(lm.norm, "weight", None).shape if getattr(lm.norm, "weight", None) is not None else None)
print("lm.per_layer_projection_norm:", type(lm.per_layer_projection_norm), getattr(lm.per_layer_projection_norm, "weight", None).shape if getattr(lm.per_layer_projection_norm, "weight", None) is not None else None)

l0 = lm.layers[0]
for name in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"]:
    m = getattr(l0, name, None)
    w = getattr(m, "weight", None)
    print(f"l0.{name}:", type(m), w.shape if w is not None else None)

for name in ["q_norm", "k_norm", "v_norm"]:
    m = getattr(l0.self_attn, name, None)
    w = getattr(m, "weight", None)
    print(f"l0.self_attn.{name}:", type(m), w.shape if w is not None else None)
