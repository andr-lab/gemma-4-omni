#!/usr/bin/env python3
"""Reproduce Code2Wav assert with CUDA_LAUNCH_BLOCKING=1."""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import sys
sys.path.insert(0, "/workspace")
import time
import numpy as np
import torch
from duplex_kernel.models.talker import load_talker
from duplex_kernel.subtalker_graph import SubtalkerGraph
from duplex_kernel.future_prefix import FuturePrefixQwen
from duplex_kernel.speaker_runtime import SpeakerRuntime

device = "cuda:0"
dtype = torch.bfloat16

print("Loading models...", flush=True)
talker = load_talker("/workspace", device=device, load_code2wav=True)
talker.talker.to(dtype=dtype)
cp = talker.talker.code_predictor
subtalker_graph = SubtalkerGraph(cp, talker_embed=talker.talker.get_input_embeddings()).capture()
speaker = FuturePrefixQwen(talker, lookahead_frames=0, max_prefix_frames=64, device=device, dtype=dtype, max_audio_frames=64)
runtime = SpeakerRuntime(speaker, subtalker_graph, talker, device=device)

runtime.begin("Continuous conversational speech.")
dummy_cond = torch.randn((1, 2048), device=device, dtype=dtype)

print("Running 500 steps...", flush=True)
for f in range(500):
    t_turn = f % 64
    if t_turn == 0 and f > 0:
        runtime.begin("Continuous conversational speech.")
    
    res = runtime.step(dummy_cond, frame_id=t_turn, record_event=False)
    # Check bounds of codes
    if res.rvq16_tensor.min().item() < 0 or res.rvq16_tensor.max().item() >= 2048:
        print(f"OUT OF BOUNDS at frame {f}: min={res.rvq16_tensor.min().item()}, max={res.rvq16_tensor.max().item()}")
    
    # Decode frame
    audio = runtime.decode_frame(res.rvq16_tensor)
    if f % 50 == 0:
        print(f"Frame {f} OK, audio shape {audio.shape}")

print("ALL 500 STEPS PASSED SUCCESSFULLY!")
