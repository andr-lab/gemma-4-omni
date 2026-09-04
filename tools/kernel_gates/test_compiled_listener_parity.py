#!/usr/bin/env python3
"""Verify numerical parity of compiled Gemma Listener LM with StaticCache against Baseline Oracle."""
from __future__ import annotations

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
from duplex_kernel.listener import GemmaListener, GemmaListenerState
from transformers import StaticCache
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

device = "cuda:0"
dtype = torch.bfloat16
workspace = Path("/workspace")

th = load_thinker(workspace, device=device, dtype=dtype)
lm = th.model.model.language_model

# 1. Setup hook on Baseline Oracle
listener_base = GemmaListener.from_thinker(th)
backend = listener_base.backend

intercepted_calls = []
orig_forward = backend._forward_incremental_lm

def hook_forward(*, embeds, per_layer, attention, positions, state):
    intercepted_calls.append({
        "embeds": embeds.detach().clone(),
        "per_layer": per_layer.detach().clone() if hasattr(per_layer, "clone") else per_layer,
        "attention": attention.detach().clone(),
        "positions": positions.detach().clone(),
    })
    return orig_forward(embeds=embeds, per_layer=per_layer, attention=attention, positions=positions, state=state)

backend._forward_incremental_lm = hook_forward

state_base = GemmaListenerState(context_frames=64)
listener_base.prime_steady_state(state_base)

# The prime_steady_state produces the prefill call
prefill_call = intercepted_calls.pop(0)
print(f"Prefill call captured: embeds shape {prefill_call['embeds'].shape}, positions {prefill_call['positions'].shape}", flush=True)

# Generate 64 test PCM frames (80 ms each, 1280 samples)
rng = np.random.default_rng(2026)
pcm_frames = [(0.05 * rng.standard_normal(1280)).astype(np.float32) for _ in range(64)]

print("--- RUNNING 64 FRAMES ON BASELINE ORACLE ---", flush=True)
oracle_hiddens = []
with torch.inference_mode():
    for f, pcm in enumerate(pcm_frames):
        out = listener_base.step(pcm, None, state_base)
        oracle_hiddens.append(out.gemma_hidden_t.float().clone())

print(f"Baseline oracle run complete. {len(oracle_hiddens)} frames collected. {len(intercepted_calls)} calls intercepted.", flush=True)

# 2. Setup candidate: compiled LM with StaticCache
cache = StaticCache(config=lm.config, max_batch_size=1, max_cache_len=384, device=device, dtype=dtype)
ple_frame = backend._frame_pad_per_layer_inputs(lm)

# Compile LM
print("--- COMPILING LM ---", flush=True)
compiled_lm = torch.compile(lm, mode="default")

# Warmup compile with dummy step inputs (shape 1, 4, 1536)
print("--- WARMING UP COMPILED LM ---", flush=True)
dummy_embeds = torch.randn((1, 4, 1536), device=device, dtype=dtype)
dummy_pos = torch.arange(35, 39, device=device, dtype=torch.long).view(1, -1)
dummy_att = torch.ones((1, 39), device=device, dtype=torch.long)
dummy_mask = create_masks_for_generate(lm.config, dummy_embeds, dummy_att, past_key_values=cache, position_ids=dummy_pos)

with torch.inference_mode():
    for _ in range(4):
        _ = compiled_lm(
            inputs_embeds=dummy_embeds,
            per_layer_inputs=ple_frame,
            attention_mask=dummy_mask,
            position_ids=dummy_pos,
            past_key_values=cache,
            use_cache=True,
        )

print("Warmup complete. Instantiating fresh cache and running candidate...", flush=True)
cache = StaticCache(config=lm.config, max_batch_size=1, max_cache_len=384, device=device, dtype=dtype)

# Candidate prefill
pre_embeds = prefill_call["embeds"]
pre_pos = prefill_call["positions"].view(1, -1)
pre_att = prefill_call["attention"]
pre_mask = create_masks_for_generate(lm.config, pre_embeds, pre_att, past_key_values=cache, position_ids=pre_pos)
pre_ple = prefill_call["per_layer"]

with torch.inference_mode():
    _ = lm(
        inputs_embeds=pre_embeds,
        per_layer_inputs=pre_ple,
        attention_mask=pre_mask,
        position_ids=pre_pos,
        past_key_values=cache,
        use_cache=True,
    )

print(f"Candidate prefill done. Cache seq length: {cache.get_seq_length()}", flush=True)

# Candidate step
cand_hiddens = []
with torch.inference_mode():
    for f in range(64):
        call = intercepted_calls[f]
        f_embeds = call["embeds"]
        f_pos = call["positions"].view(1, -1)
        f_att = call["attention"]
        f_ple = call["per_layer"]
        f_mask = create_masks_for_generate(lm.config, f_embeds, f_att, past_key_values=cache, position_ids=f_pos)
        
        out = compiled_lm(
            inputs_embeds=f_embeds,
            per_layer_inputs=f_ple,
            attention_mask=f_mask,
            position_ids=f_pos,
            past_key_values=cache,
            use_cache=True,
        )
        h_cand = out.last_hidden_state[:, 0, :].float()
        cand_hiddens.append(h_cand.clone())

# Parity analysis
diffs = []
cosines = []
for f in range(64):
    h_o = oracle_hiddens[f].flatten()
    h_c = cand_hiddens[f].flatten()
    diff = (h_o - h_c).abs().max().item()
    cos = F.cosine_similarity(h_o, h_c, dim=0).item()
    diffs.append(diff)
    cosines.append(cos)
    if f < 5 or f % 10 == 0 or f >= 60:
        print(f"Frame {f:2d}: max_diff={diff:.6e}, cosine={cos:.7f}", flush=True)

max_d = max(diffs)
mean_d = sum(diffs) / len(diffs)
min_c = min(cosines)

print("\n" + "=" * 70, flush=True)
print("PARITY SUMMARY OVER 64 FRAMES:", flush=True)
print(f"  Max Diff:    {max_d:.6e} (Gate: <= 1e-4)", flush=True)
print(f"  Mean Diff:   {mean_d:.6e}", flush=True)
print(f"  Min Cosine:  {min_c:.7f} (Gate: >= 0.9999)", flush=True)
print("=" * 70, flush=True)

if max_d <= 1e-4 and min_c >= 0.9999:
    print("VERDICT: PASS — COMPILED LISTENER BIT-PARITY ACHIEVED", flush=True)
    sys.exit(0)
else:
    print(f"VERDICT: FAIL — DIVERGENCE FROM ORACLE (MaxDiff={max_d:.4e}, MinCosine={min_c:.6f})", flush=True)
    sys.exit(1)
