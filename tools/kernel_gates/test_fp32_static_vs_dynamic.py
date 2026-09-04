#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.thinker import load_thinker
from transformers import StaticCache, DynamicCache
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

device = "cuda:0"
# Test in float32
dtype = torch.float32

th = load_thinker("/workspace", device=device, dtype=dtype)
lm = th.model.model.language_model

sc = StaticCache(config=lm.config, max_batch_size=1, max_cache_len=384, device=device, dtype=dtype)
dc = DynamicCache(config=lm.config)

rng = np.random.default_rng(2026)
stable_embeds = torch.tensor(0.05 * rng.standard_normal((1, 35, 1536)), device=device, dtype=dtype)
pos_ids = torch.arange(35, device=device, dtype=torch.long).view(1, -1)
att = torch.ones((1, 35), device=device, dtype=torch.long)

m_dyn = create_masks_for_generate(lm.config, stable_embeds, att, past_key_values=dc, position_ids=pos_ids)
m_stat = create_masks_for_generate(lm.config, stable_embeds, att, past_key_values=sc, position_ids=pos_ids)
stable_ple = lm.get_per_layer_inputs(torch.full((1, 35), 2, device=device, dtype=torch.long), stable_embeds)

with torch.inference_mode():
    out_d = lm(
        inputs_embeds=stable_embeds,
        per_layer_inputs=stable_ple,
        attention_mask=m_dyn,
        position_ids=pos_ids,
        past_key_values=dc,
        use_cache=True,
        output_hidden_states=True,
    )
    out_s = lm(
        inputs_embeds=stable_embeds,
        per_layer_inputs=stable_ple,
        attention_mask=m_stat,
        position_ids=pos_ids,
        past_key_values=sc,
        use_cache=True,
        output_hidden_states=True,
    )

print("=== FP32 LAYER-BY-LAYER DIFF ON PREFILL ===")
for i in range(len(out_d.hidden_states)):
    hd = out_d.hidden_states[i]
    hs = out_s.hidden_states[i]
    diff = (hd - hs).abs().max().item()
    cos = F.cosine_similarity(hd.flatten(), hs.flatten(), dim=0).item()
    layer_name = "Embedding" if i == 0 else f"Layer {i-1:2d}"
    extra = ""
    if i > 0:
        lt = lm.config.layer_types[i-1]
        shared = lm.layers[i-1].self_attn.is_kv_shared_layer
        extra = f"({lt:18s} | shared={shared})"
    print(f"{layer_name:10s} {extra}: diff={diff:.6e}, cos={cos:.7f}")
