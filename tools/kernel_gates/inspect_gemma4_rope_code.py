#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import inspect
import transformers.models.gemma4.modeling_gemma4 as mg

# Print class definition of Gemma4TextRotaryEmbedding
print("Source of Gemma4TextRotaryEmbedding:")
print(inspect.getsource(mg.Gemma4TextRotaryEmbedding))

# Print source of apply_rotary_pos_emb
print("\nSource of apply_rotary_pos_emb:")
print(inspect.getsource(mg.apply_rotary_pos_emb))
