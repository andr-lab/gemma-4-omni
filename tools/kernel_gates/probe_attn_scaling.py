#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")
import torch
from duplex_kernel.models.thinker import load_thinker

th = load_thinker("/workspace")
lm = th.model.model.language_model

for idx in [0, 1, 4, 13, 14, 15]:
    attn = lm.layers[idx].self_attn
    print(f"Layer {idx} ({attn.layer_type}): head_dim={attn.head_dim}, scaling={attn.scaling}, sliding_window={attn.sliding_window}")
