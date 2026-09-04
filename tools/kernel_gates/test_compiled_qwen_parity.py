#!/usr/bin/env python3
"""Verify numerical parity of compiled Qwen talker backbone with FuturePrefixQwen slot-KV against Eager."""
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

from duplex_kernel.models.talker import load_talker
from duplex_kernel.future_prefix import FuturePrefixQwen

device = "cuda:0"
dtype = torch.bfloat16
workspace = "/workspace"

# 1. Load Eager Talker
print("--- LOADING EAGER TALKER ---", flush=True)
talker = load_talker(workspace, device=device, load_code2wav=False)
talker.talker.to(dtype=dtype)

speaker_eager = FuturePrefixQwen(
    talker,
    lookahead_frames=0,
    max_prefix_frames=64,
    device=device,
    dtype=dtype,
    max_audio_frames=128,
)

# Deterministic conditioning frames
rng = np.random.default_rng(2026)
cond_frames = [
    torch.tensor(0.05 * rng.standard_normal((1, 2048)), device=device, dtype=dtype)
    for _ in range(64)
]
fixed_codes = [
    torch.tensor(rng.integers(0, 2048, size=(16,)), device=device, dtype=torch.long)
    for _ in range(64)
]

prompt_text = "Continuous conversational speech synthesis test prompt."

print("--- RUNNING 64 FRAMES ON EAGER SPEAKER ---", flush=True)
speaker_eager.begin(prompt_text)
eager_logits = []
eager_hiddens = []

with torch.inference_mode():
    for t in range(64):
        speaker_eager.push_v9_frame(t, cond_frames[t])
        step_out = speaker_eager.step_frame()
        eager_logits.append(step_out.logits.float().clone())
        eager_hiddens.append(step_out.last_hidden.float().clone())
        speaker_eager.commit_codes(fixed_codes[t])

print("Eager run complete. 64 frames collected.", flush=True)

# 2. Compile Qwen backbone
print("--- COMPILING QWEN BACKBONE WITH TORCH.COMPILE ---", flush=True)
talker.talker.model = torch.compile(talker.talker.model, mode="default")

speaker_compiled = FuturePrefixQwen(
    talker,
    lookahead_frames=0,
    max_prefix_frames=64,
    device=device,
    dtype=dtype,
    max_audio_frames=128,
)

# Warmup compile
print("--- WARMING UP COMPILED SPEAKER (triggering Dynamo compilation) ---", flush=True)
speaker_compiled.begin(prompt_text)
dummy_cond = torch.randn((1, 2048), device=device, dtype=dtype)
dummy_code = torch.randint(0, 2048, (16,), device=device, dtype=torch.long)
with torch.inference_mode():
    for t in range(6):
        t0 = time.perf_counter()
        speaker_compiled.push_v9_frame(t, dummy_cond)
        _ = speaker_compiled.step_frame()
        speaker_compiled.commit_codes(dummy_code)
        torch.cuda.synchronize()
        print(f"  Warmup step {t}: {time.perf_counter() - t0:.2f}s", flush=True)

print("Warmup complete. Resetting for parity test...", flush=True)

# Run compiled test
speaker_compiled.begin(prompt_text)
comp_logits = []
comp_hiddens = []

with torch.inference_mode():
    for t in range(64):
        speaker_compiled.push_v9_frame(t, cond_frames[t])
        step_out = speaker_compiled.step_frame()
        comp_logits.append(step_out.logits.float().clone())
        comp_hiddens.append(step_out.last_hidden.float().clone())
        speaker_compiled.commit_codes(fixed_codes[t])

print("Compiled run complete. Computing parity metrics...", flush=True)

# Compare metrics
top1_matches = 0
total_tokens = 0
mae_logits = []
cos_hiddens = []

for t in range(64):
    lg_e = eager_logits[t]
    lg_c = comp_logits[t]
    
    # Top-1 agreement
    k0_e = lg_e[:, :1, :].argmax(dim=-1).item()
    k0_c = lg_c[:, :1, :].argmax(dim=-1).item()
    if k0_e == k0_c:
        top1_matches += 1
    total_tokens += 1
    
    mae = (lg_e - lg_c).abs().mean().item()
    mae_logits.append(mae)
    
    h_e = eager_hiddens[t].flatten()
    h_c = comp_hiddens[t].flatten()
    cos = F.cosine_similarity(h_e, h_c, dim=0).item()
    cos_hiddens.append(cos)
    
    if t < 5 or t % 10 == 0 or t >= 60:
        print(f"Frame {t:2d}: Top-1 ({k0_e} vs {k0_c}), Logit MAE={mae:.6e}, Hidden Cosine={cos:.7f}", flush=True)

top1_acc = (top1_matches / total_tokens) * 100.0
mean_mae = sum(mae_logits) / len(mae_logits)
min_cos = min(cos_hiddens)

print("\n" + "=" * 70, flush=True)
print("QWEN SLOT-KV PARITY SUMMARY OVER 64 FRAMES:", flush=True)
print(f"  Top-1 Agreement: {top1_acc:.2f}% (Gate: 100.0%)", flush=True)
print(f"  Mean Logit MAE:  {mean_mae:.6e} (Gate: <= 0.02)", flush=True)
print(f"  Min Hidden Cos:  {min_cos:.7f} (Gate: >= 0.999)", flush=True)
print("=" * 70, flush=True)

if top1_acc == 100.0 and mean_mae <= 0.02 and min_cos >= 0.999:
    print("VERDICT: PASS — COMPILED QWEN SLOT-KV PARITY CERTIFIED", flush=True)
    sys.exit(0)
else:
    print("VERDICT: FAIL — COMPILED QWEN PARITY GATE NOT MET", flush=True)
    sys.exit(1)
