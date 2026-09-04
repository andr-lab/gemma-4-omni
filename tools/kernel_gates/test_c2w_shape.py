#!/usr/bin/env python3
import sys
sys.path.insert(0, "/workspace")
import torch
from duplex_kernel.models.talker import load_talker

talker = load_talker("/workspace", device="cuda:0", load_code2wav=True)
codes = torch.randint(0, 2048, (1, 2, 16), device="cuda:0", dtype=torch.long)
with torch.no_grad():
    out = talker.code2wav.decode(codes)
print("type of audio_values:", type(out.audio_values), len(out.audio_values))
wav0 = out.audio_values[0]
print("wav0 shape:", wav0.shape, "dtype:", wav0.dtype)
talker_decoded = talker.decode(codes)
print("talker.decode(codes) shape:", talker_decoded.shape, "dtype:", talker_decoded.dtype)
