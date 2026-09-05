#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")
import torch
import inspect
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb, rotate_half

print("=== ROTATE HALF SOURCE ===")
print(inspect.getsource(rotate_half))

print("=== APPLY ROTARY POS EMB SOURCE ===")
print(inspect.getsource(apply_rotary_pos_emb))
