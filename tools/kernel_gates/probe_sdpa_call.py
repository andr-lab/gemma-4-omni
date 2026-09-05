#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")
import torch
import inspect
from transformers.integrations.sdpa_attention import sdpa_attention_forward

print("=== SDPA ATTENTION FORWARD SOURCE ===")
print(inspect.getsource(sdpa_attention_forward))
