#!/usr/bin/env python3
"""Inspect layer 0 attention difference between DynamicCache and StaticCache."""
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
cache_stat = StaticCache(config=lm.config, max_batch_size=1, max_cache_len=291, device="cuda:0", dtype=torch.bfloat16)

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

    # Let's inspect layer 0 self_attn
    # In ss_dyn._attention vs ss_stat._attention:
    l0 = lm.layers[0]
    pe = {lt: lm.rotary_emb(frame_embeds, cur_pos, lt) for lt in lm.unique_layer_types}
    
    with torch.inference_mode():
        # norm
        hn_d = ss_dyn._padded_rmsnorm(l0.input_layernorm, frame_embeds)
        hn_s = ss_stat._padded_rmsnorm(l0.input_layernorm, frame_embeds)
        
        # attn
        att_d = ss_dyn._attention(l0.self_attn, hn_d, pe['sliding_attention'], m_d['sliding_attention'], cur_pos, {}, cache_dyn)
        att_s = ss_stat._attention(l0.self_attn, hn_s, pe['sliding_attention'], m_s['sliding_attention'], cur_pos, {}, cache_stat)
        
        diff_att = (att_d.float() - att_s.float()).abs().max().item()
        
        od = ss_dyn.forward(inputs_embeds=frame_embeds, per_layer_inputs=ple_frame, attention_mask=m_d, position_ids=cur_pos, past_key_values=cache_dyn, shared_kv_states=getattr(out_d.past_key_values, "shared_layers", None))
        os = ss_stat.forward(inputs_embeds=frame_embeds, per_layer_inputs=ple_frame, attention_mask=m_s, position_ids=cur_pos, past_key_values=cache_stat, shared_kv_states=getattr(out_s.past_key_values, "shared_layers", None))
        diff_final = (od.last_hidden_state.float() - os.last_hidden_state.float()).abs().max().item()

    if diff_att > 1e-5 or f >= 20:
        print(f"Frame {f:2d}: diff_att={diff_att:.8e} | diff_final={diff_final:.8e}")
        if diff_att > 1e-5:
            # Check key_states in cache_dyn vs cache_stat
            k_d = cache_dyn.layers[0].keys
            k_s = cache_stat.layers[0].keys[:, :, :k_d.shape[-2], :]
            k_diff = (k_d.float() - k_s.float()).abs().max().item()
            print(f"  k_diff in cache: {k_diff:.8e}")
            break
