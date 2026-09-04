#!/usr/bin/env python3
"""Compare m_dyn and m_stat at Frame 24."""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.thinker import load_thinker
from transformers import StaticCache, DynamicCache
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

th = load_thinker("/workspace", device="cuda:0", dtype=torch.bfloat16)
lm = th.model.model.language_model

cache_dyn = DynamicCache(config=lm.config)
cache_stat = StaticCache(config=lm.config, max_batch_size=1, max_cache_len=291, device="cuda:0", dtype=torch.bfloat16)

frame_embeds = torch.zeros((1, 4, 1536), device="cuda:0", dtype=torch.bfloat16)
start_pos = 35 + 24 * 4  # 131
cur_pos = torch.arange(start_pos, start_pos + 4, device="cuda:0", dtype=torch.long).view(1, -1)
cur_att = torch.ones((1, start_pos + 4), device="cuda:0", dtype=torch.long)

m_d = create_masks_for_generate(lm.config, frame_embeds, cur_att, past_key_values=cache_dyn, position_ids=cur_pos)
m_s = create_masks_for_generate(lm.config, frame_embeds, cur_att, past_key_values=cache_stat, position_ids=cur_pos)

for k in m_d:
    print(f"Layer type {k}:")
    print("  dyn shape:", m_d[k].shape)
    print("  stat shape:", m_s[k].shape)
    # compare values in the common prefix [..., :135]
    diff = (m_d[k].float() - m_s[k][..., :m_d[k].shape[-1]].float()).abs().max().item()
    print("  diff on valid positions:", diff)
    print("  stat trailing values min/max:", m_s[k][..., m_d[k].shape[-1]:].min().item(), m_s[k][..., m_d[k].shape[-1]:].max().item())
