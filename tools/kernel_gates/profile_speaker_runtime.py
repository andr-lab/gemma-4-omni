#!/usr/bin/env python3
"""Profile exact breakdown of SpeakerRuntime and Code2Wav."""
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.talker import load_talker
from duplex_kernel.future_prefix import FuturePrefixQwen
from duplex_kernel.subtalker_graph import SubtalkerGraph
from duplex_kernel.speaker_runtime import SpeakerRuntime

device = "cuda:0"
dtype = torch.bfloat16
workspace = "/workspace"

talker = load_talker(workspace, device=device, load_code2wav=True)
talker.talker.to(dtype=dtype)
talker.talker.model = torch.compile(talker.talker.model, mode="default")

cp = talker.talker.code_predictor
subtalker_graph = SubtalkerGraph(cp, talker_embed=talker.talker.get_input_embeddings()).capture()

speaker = FuturePrefixQwen(talker, lookahead_frames=0, max_prefix_frames=64, device=device, dtype=dtype, max_audio_frames=128)
runtime = SpeakerRuntime(speaker, subtalker_graph, talker, device=device)

# Warmup
prompt = "Continuous conversational speech synthesis test prompt."
runtime.begin(prompt)
dummy_cond = torch.randn((1, 2048), device=device, dtype=dtype)
for t in range(5):
    _ = runtime.step(dummy_cond, frame_id=t, record_event=False, do_sample=False)
torch.cuda.synchronize()

print("--- PROFILING 32 STEPS OF SPEAKERRUNTIME ---")
qwen_times = []
cb0_times = []
subtalker_times = []
c2w_times = []

ev0 = torch.cuda.Event(enable_timing=True)
ev1 = torch.cuda.Event(enable_timing=True)
ev2 = torch.cuda.Event(enable_timing=True)
ev3 = torch.cuda.Event(enable_timing=True)
ev4 = torch.cuda.Event(enable_timing=True)

runtime.begin(prompt)
for t in range(32):
    ev0.record()
    
    # 1. Qwen
    runtime.speaker.push_v9_frame(t, dummy_cond)
    step_out = runtime.speaker.step_frame()
    ev1.record()
    
    # 2. CB0
    c0_tensor = step_out.logits[:, :1, :].argmax(dim=-1).squeeze().view(1)
    c0_embed = runtime.talker.talker.get_input_embeddings()(c0_tensor.view(1, 1))
    ev2.record()
    
    # 3. SubTalker
    sub_codes = runtime.subtalker_graph.generate(step_out.last_hidden, c0_token=c0_tensor, c0_embed=c0_embed, do_sample=False, return_tensor=True)
    codes_16 = torch.cat([c0_tensor.view(1), sub_codes.view(-1)], dim=0)
    runtime.speaker.commit_codes(codes_16)
    ev3.record()
    
    # 4. Code2Wav 1-frame decode
    with torch.inference_mode():
        _ = runtime.talker.decode(codes_16.view(1, 1, 16))
    ev4.record()
    
    torch.cuda.synchronize()
    
    q_ms = ev0.elapsed_time(ev1)
    cb0_ms = ev1.elapsed_time(ev2)
    sub_ms = ev2.elapsed_time(ev3)
    c2w_ms = ev3.elapsed_time(ev4)
    
    qwen_times.append(q_ms)
    cb0_times.append(cb0_ms)
    subtalker_times.append(sub_ms)
    c2w_times.append(c2w_ms)
    
    if t < 5 or t % 10 == 0:
        print(f"Step {t:2d}: Qwen={q_ms:.2f}ms, CB0={cb0_ms:.3f}ms, SubTalker={sub_ms:.2f}ms, Code2Wav={c2w_ms:.2f}ms, Total={q_ms+cb0_ms+sub_ms+c2w_ms:.2f}ms")

print("\n=== SUMMARY OVER 32 STEPS ===")
print(f"  Qwen mean:      {np.mean(qwen_times):.2f} ms")
print(f"  CB0 mean:       {np.mean(cb0_times):.3f} ms")
print(f"  SubTalker mean: {np.mean(subtalker_times):.2f} ms")
print(f"  Code2Wav mean:  {np.mean(c2w_times):.2f} ms")
print(f"  TOTAL SPEAKER:  {np.mean(qwen_times)+np.mean(subtalker_times)+np.mean(c2w_times):.2f} ms")
