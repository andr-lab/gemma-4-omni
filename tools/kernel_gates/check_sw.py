#!/usr/bin/env python3
"""Test sliding window handling in StaticCache."""
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
from transformers import StaticCache, DynamicCache
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

th = load_thinker("/workspace", device="cuda:0", dtype=torch.bfloat16)
lm = th.model.model.language_model

for i, layer in enumerate(lm.layers):
    sw = getattr(layer.self_attn, "sliding_window", None)
    if sw is not None:
        print(f"Layer {i}: sliding_window = {sw}")
