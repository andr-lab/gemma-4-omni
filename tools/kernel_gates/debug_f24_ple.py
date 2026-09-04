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
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from transformers import StaticCache, DynamicCache
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

th = load_thinker("/workspace", device="cuda:0", dtype=torch.bfloat16)
lm = th.model.model.language_model

ss_dyn = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
cache_dyn = DynamicCache(config=lm.config)
ss_stat = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
cache_stat = StaticCache(config=lm.config, max_batch_size=1, max_cache_len=384, device="cuda:0", dtype=torch.bfloat16)

from duplex_kernel.listener import GemmaListener
backend = GemmaListener.from_thinker(th).backend
ple_frame = backend._frame_pad_per_layer_inputs(lm)

# Prefill
stable_embeds = torch.randn((1, 35, 1536), device="cuda:0", dtype=torch.bfloat16)
pos_ids = torch.arange(35, device="cuda:0", dtype=torch.long).view(1, -1)
att = torch.ones((1, 35), device="cuda:0", dtype=torch.long)
m_dyn = create_masks_for_generate(lm.config, stable_embeds, att, past_key_values=cache_dyn, position_ids=pos_ids)
m_stat = create_masks_for_generate(lm.config, stable_embeds, att, past_key_values=cache_stat, position_ids=pos_ids)
stable_ple = lm.get_per_layer_inputs(torch.full((1, 35), 2, device="cuda:0", dtype=torch.long), stable_embeds)

with torch.inference_mode():
    out_d = ss_dyn.forward(inputs_embeds=stable_embeds, per_layer_inputs=stable_ple, attention_mask=m_dyn, position_ids=pos_ids, past_key_values=cache_dyn, shared_kv_states=None)
    out_s = ss_stat.forward(inputs_embeds=stable_embeds, per_layer_inputs=stable_ple, attention_mask=m_stat, position_ids=pos_ids, past_key_values=cache_stat, shared_kv_states=None)

rng = np.random.default_rng(2026)
for f in range(25):
    frame_embeds = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device="cuda:0", dtype=torch.bfloat16)
    start_pos = 35 + f * 4
    cur_pos = torch.arange(start_pos, start_pos + 4, device="cuda:0", dtype=torch.long).view(1, -1)
    cur_att = torch.ones((1, start_pos + 4), device="cuda:0", dtype=torch.long)
    
    m_d = create_masks_for_generate(lm.config, frame_embeds, cur_att, past_key_values=cache_dyn, position_ids=cur_pos)
    m_s = create_masks_for_generate(lm.config, frame_embeds, cur_att, past_key_values=cache_stat, position_ids=cur_pos)
    
    with torch.inference_mode():
        od = ss_dyn.forward(inputs_embeds=frame_embeds, per_layer_inputs=ple_frame, attention_mask=m_d, position_ids=cur_pos, past_key_values=cache_dyn, shared_kv_states=getattr(out_d.past_key_values, "shared_layers", None))
        os = ss_stat.forward(inputs_embeds=frame_embeds, per_layer_inputs=ple_frame, attention_mask=m_s, position_ids=cur_pos, past_key_values=cache_stat, shared_kv_states=getattr(out_s.past_key_values, "shared_layers", None))
    
    diff = (od.last_hidden_state.float() - os.last_hidden_state.float()).abs().max().item()
    if f >= 22:
        print(f"Frame {f}: diff={diff:.8e}")
        for k in m_d:
            if isinstance(m_d[k], torch.Tensor) and isinstance(m_s[k], torch.Tensor):
                print(f"  mask {k}: dyn {m_d[k].shape}, stat {m_s[k].shape}")
