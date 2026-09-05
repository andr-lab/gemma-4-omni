#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import inspect
from transformers.models.gemma4 import modeling_gemma4

mapping = getattr(modeling_gemma4, "LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING", None)
print("mapping:", mapping)
print("--- create_causal_mask ---")
print(inspect.getsource(modeling_gemma4.create_causal_mask))
print("--- create_sliding_window_causal_mask ---")
print(inspect.getsource(modeling_gemma4.create_sliding_window_causal_mask))
