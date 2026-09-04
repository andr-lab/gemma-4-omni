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
dtype = torch.bfloat16

th = load_thinker("/workspace", device=device, dtype=dtype)
lm = th.model.model.language_model

sc = StaticCache(config=lm.config, max_batch_size=1, max_cache_len=384, device=device, dtype=dtype)
dc = DynamicCache(config=lm.config)

# Prefill: 35 tokens
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

diff_prefill = (out_d.last_hidden_state.float() - out_s.last_hidden_state.float()).abs().max().item()
cos_prefill = F.cosine_similarity(out_d.last_hidden_state.float().flatten(), out_s.last_hidden_state.float().flatten(), dim=0).item()
print(f"Full LM Prefill (35 tokens): max_diff={diff_prefill:.8e}, cos={cos_prefill:.7f}")

from duplex_kernel.listener import GemmaListener
backend = GemmaListener.from_thinker(th).backend
ple_frame = backend._frame_pad_per_layer_inputs(lm)

# Run frames 0 to 28
for f in range(29):
    frame_embeds = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype)
    start_pos = 35 + f * 4
    cur_pos = torch.arange(start_pos, start_pos + 4, device=device, dtype=torch.long).view(1, -1)
    cur_att = torch.ones((1, start_pos + 4), device=device, dtype=torch.long)

    m_d = create_masks_for_generate(lm.config, frame_embeds, cur_att, past_key_values=dc, position_ids=cur_pos)
    m_s = create_masks_for_generate(lm.config, frame_embeds, cur_att, past_key_values=sc, position_ids=cur_pos)

    with torch.inference_mode():
        od = lm(
            inputs_embeds=frame_embeds,
            per_layer_inputs=ple_frame,
            attention_mask=m_d,
            position_ids=cur_pos,
            past_key_values=dc,
            use_cache=True,
            output_hidden_states=True,
        )
        os = lm(
            inputs_embeds=frame_embeds,
            per_layer_inputs=ple_frame,
            attention_mask=m_s,
            position_ids=cur_pos,
            past_key_values=sc,
            use_cache=True,
            output_hidden_states=True,
        )

    diff = (od.last_hidden_state.float() - os.last_hidden_state.float()).abs().max().item()
    cos = F.cosine_similarity(od.last_hidden_state.float().flatten(), os.last_hidden_state.float().flatten(), dim=0).item()
    print(f"Frame {f:2d} (pos {start_pos:3d}..{start_pos+3:3d}): max_diff={diff:.8e}, cos={cos:.7f}")
    if diff > 1e-4 or cos < 0.9999:
        print(f"  --> Divergence at frame {f}!")
        break
