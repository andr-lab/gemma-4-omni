#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")
import torch
import inspect
from duplex_kernel.models.thinker import load_thinker

th = load_thinker("/workspace")
lm = th.model.model.language_model
print("=== ROTARY EMB FORWARD SOURCE ===")
print(inspect.getsource(lm.rotary_emb.forward))
print("=== ROTARY EMB INIT / DEFAULT PARAMS ===")
print(inspect.getsource(lm.rotary_emb.compute_default_rope_parameters))
