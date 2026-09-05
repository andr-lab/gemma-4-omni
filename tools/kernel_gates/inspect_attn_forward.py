#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import inspect
from duplex_kernel.models.thinker import load_thinker

th = load_thinker("/workspace")
lm = th.model.model.language_model
attn = lm.layers[0].self_attn
print("attn type:", type(attn))
print("--- attn.forward source ---")
print(inspect.getsource(attn.forward))
