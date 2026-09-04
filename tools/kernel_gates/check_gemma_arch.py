#!/usr/bin/env python3
import sys
sys.path.insert(0, "/workspace")
from duplex_kernel.models.thinker import load_thinker

th = load_thinker("/workspace", device="cpu")
cfg = th.model.model.language_model.config
print("layer_types:", cfg.layer_types)
print("num_layers:", len(cfg.layer_types))
print("sliding_window:", getattr(cfg, "sliding_window", None))
print("unique_layer_types:", getattr(th.model.model.language_model, "unique_layer_types", None))
for i, l in enumerate(th.model.model.language_model.layers[:8]):
    attn = l.self_attn
    print(f"Layer {i}: shared_kv={getattr(attn, 'is_kv_shared_layer', False)}, shared_idx={getattr(attn, 'kv_shared_layer_index', None)}, full_kv={getattr(attn, 'store_full_length_kv', False)}")
