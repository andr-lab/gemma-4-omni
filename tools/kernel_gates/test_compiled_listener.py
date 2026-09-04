#!/usr/bin/env python3
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.thinker import load_thinker
from transformers import StaticCache
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

device = "cuda:0"
dtype = torch.bfloat16
workspace = Path("/workspace")

th = load_thinker(workspace, device=device, dtype=dtype)
lm = th.model.model.language_model

from duplex_kernel.listener import GemmaListener
backend = GemmaListener.from_thinker(th).backend
ple_frame = backend._frame_pad_per_layer_inputs(lm)

# Prefill with 35 tokens
pos_prefill = torch.arange(35, device=device, dtype=torch.long).view(1, -1)
embeds_prefill = torch.randn((1, 35, 1536), device=device, dtype=dtype)
att_prefill = torch.ones((1, 35), device=device, dtype=torch.long)
ple_prefill = lm.get_per_layer_inputs(torch.full((1, 35), 2, device=device, dtype=torch.long), embeds_prefill)

cache = StaticCache(config=lm.config, max_batch_size=1, max_cache_len=384, device=device, dtype=dtype)
m_prefill = create_masks_for_generate(lm.config, embeds_prefill, att_prefill, past_key_values=cache, position_ids=pos_prefill)

print("--- RUNNING PREFILL ---")
with torch.inference_mode():
    out_pre = lm(
        inputs_embeds=embeds_prefill,
        per_layer_inputs=ple_prefill,
        attention_mask=m_prefill,
        position_ids=pos_prefill,
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=True,
    )

print(f"Prefill done. Cache seq length: {cache.get_seq_length()}")

# Now compile lm for step with q_len=4
compiled_lm = torch.compile(lm, mode="default")

frame_embeds = torch.randn((1, 4, 1536), device=device, dtype=dtype)

print("--- WARMUP COMPILED LM (q_len=4) ---")
with torch.inference_mode():
    for f in range(6):
        start_pos = 35 + f * 4
        cur_pos = torch.arange(start_pos, start_pos + 4, device=device, dtype=torch.long).view(1, -1)
        cur_att = torch.ones((1, start_pos + 4), device=device, dtype=torch.long)
        m_s = create_masks_for_generate(lm.config, frame_embeds, cur_att, past_key_values=cache, position_ids=cur_pos)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = compiled_lm(
            inputs_embeds=frame_embeds,
            per_layer_inputs=ple_frame,
            attention_mask=m_s,
            position_ids=cur_pos,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"Step {f} (pos {start_pos}..{start_pos+3}): {dt:.2f} ms")

print("--- MEASURING STEADY-STATE COMPILED LATENCY (frames 6 to 20) ---")
times = []
with torch.inference_mode():
    for f in range(6, 20):
        start_pos = 35 + f * 4
        cur_pos = torch.arange(start_pos, start_pos + 4, device=device, dtype=torch.long).view(1, -1)
        cur_att = torch.ones((1, start_pos + 4), device=device, dtype=torch.long)
        m_s = create_masks_for_generate(lm.config, frame_embeds, cur_att, past_key_values=cache, position_ids=cur_pos)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = compiled_lm(
            inputs_embeds=frame_embeds,
            per_layer_inputs=ple_frame,
            attention_mask=m_s,
            position_ids=cur_pos,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)
        print(f"Frame {f:2d}: {dt:.2f} ms")

t = np.array(times)
print(f"\nCompiled LM Steady Latency (q_len=4): p50={np.percentile(t, 50):.2f} ms, p95={np.percentile(t, 95):.2f} ms")
