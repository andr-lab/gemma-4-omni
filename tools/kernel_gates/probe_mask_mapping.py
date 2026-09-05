#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")
import torch
from duplex_kernel.models.thinker import load_thinker
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
import inspect
print(inspect.getsource(create_masks_for_generate))

th = load_thinker("/workspace", device=device, dtype=dtype)
lm = th.model.model.language_model

# 1. Prefix (35 tokens)
prefix_ids = torch.full((1, 35), 0, device=device, dtype=torch.long)
prefix_embeds = lm.embed_tokens(prefix_ids)
prefix_pos = torch.arange(0, 35, device=device, dtype=torch.long).unsqueeze(0)
prefix_attn = torch.ones((1, 35), device=device, dtype=torch.long)

from transformers.cache_utils import DynamicCache
cache = DynamicCache(config=lm.config)

mask_prefix = create_masks_for_generate(
    lm.config,
    prefix_embeds,
    prefix_attn,
    past_key_values=cache,
    position_ids=prefix_pos,
)
print("=== PREFIX MASK ===")
for k, v in mask_prefix.items():
    if v is not None:
        print(f"Key {k}: shape={v.shape}, dtype={v.dtype}, min={v.min().item()}, max={v.max().item()}")
    else:
        print(f"Key {k}: None")

# Simulate cache having 35 tokens
k_dummy = torch.zeros((1, 1, 35, 256), device=device, dtype=dtype)
v_dummy = torch.zeros((1, 1, 35, 256), device=device, dtype=dtype)
cache.update(k_dummy, v_dummy, 0)

# 2. Step (4 tokens)
step_embeds = torch.zeros((1, 4, 1536), device=device, dtype=dtype)
step_attn = torch.ones((1, 39), device=device, dtype=torch.long) # total seq len
step_pos = torch.arange(35, 39, device=device, dtype=torch.long).unsqueeze(0)

mask_step = create_masks_for_generate(
    lm.config,
    step_embeds,
    step_attn,
    past_key_values=cache,
    position_ids=step_pos,
)
print("=== STEP MASK ===")
for k, v in mask_step.items():
    if v is not None:
        print(f"Key {k}: shape={v.shape}, dtype={v.dtype}, min={v.min().item()}, max={v.max().item()}")
        if v.shape[-2] == 4:
            print(f"Mask matrix:\n{v[0, 0, :, -6:]}") # last 6 keys
    else:
        print(f"Key {k}: None")
