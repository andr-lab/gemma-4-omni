#!/usr/bin/env python3
"""Validate Code2Wav 1-frame decoding, boundary continuity, and sample counts."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.talker import load_talker
from duplex_kernel.reference_buffer import AssistantReferenceBuffer

device = "cuda:0"
workspace = "/workspace"

print("--- LOADING TALKER WITH CODE2WAV ---", flush=True)
talker = load_talker(workspace, device=device, load_code2wav=True)
code2wav = talker.code2wav

# Generate 8 random RVQ frames (16 quantizers each)
rng = np.random.default_rng(2026)
raw_codes = rng.integers(0, 2048, size=(1, 8, 16))
codes_tensor = torch.tensor(raw_codes, device=device, dtype=torch.long)

print("\n--- 1. SAMPLE COUNT & SHAPE VERIFICATION ---", flush=True)
with torch.inference_mode():
    single_out = code2wav.decode(codes_tensor[:, :1, :])
    audio_single = single_out.audio_values[0].detach().cpu().numpy().astype(np.float32)

print(f"Single frame decode audio shape: {audio_single.shape}")
expected_24k_samples = 1920  # 80 ms @ 24 kHz = 1920 samples
if audio_single.size != expected_24k_samples:
    print(f"WARNING: audio sample count {audio_single.size} != expected {expected_24k_samples}")
else:
    print(f"✅ Exact sample count verified: {audio_single.size} samples @ 24 kHz (80 ms)")

# Test resampling to 16 kHz in AssistantReferenceBuffer
ref_buf = AssistantReferenceBuffer(delay_frames=1)
ref_frame = ref_buf.append(audio_single, frame_id=0, sample_rate=24000)
print(f"Resampled reference frame PCM shape: {ref_frame.pcm.shape} (expected 1280 @ 16 kHz)")
assert ref_frame.pcm.size == 1280, f"Expected 1280 samples, got {ref_frame.pcm.size}"
print("✅ Reference buffer 16 kHz resampling exact: 1280 samples (80 ms)")

print("\n--- 2. SINGLE-FRAME VS MULTI-FRAME CONTINUITY ---", flush=True)
with torch.inference_mode():
    # Multi-frame joint decode of 8 frames
    multi_out = code2wav.decode(codes_tensor)
    audio_multi = multi_out.audio_values[0].detach().cpu().numpy().astype(np.float32)

    # Frame-by-frame sequential decode
    single_frames = []
    for f in range(8):
        sf_out = code2wav.decode(codes_tensor[:, f:f+1, :])
        sf_pcm = sf_out.audio_values[0].detach().cpu().numpy().astype(np.float32)
        single_frames.append(sf_pcm)
    audio_concatenated = np.concatenate(single_frames)

print(f"Multi-frame audio length: {audio_multi.size}")
print(f"Concatenated single-frame audio length: {audio_concatenated.size}")

# Boundary click analysis at frame boundaries (every 1920 samples)
discontinuities = []
for f in range(1, 8):
    boundary_idx = f * 1920
    # Step change at boundary in concatenated single-frames:
    step_single = abs(audio_concatenated[boundary_idx] - audio_concatenated[boundary_idx - 1])
    # Step change at boundary in multi-frame:
    step_multi = abs(audio_multi[boundary_idx] - audio_multi[boundary_idx - 1])
    discontinuities.append((step_single, step_multi))
    print(f"  Boundary frame {f-1}->{f} (sample {boundary_idx}): single-step={step_single:.5f}, multi-step={step_multi:.5f}")

max_step_single = max(d[0] for d in discontinuities)
max_step_multi = max(d[1] for d in discontinuities)
print(f"\nMax boundary step: single={max_step_single:.5f}, multi={max_step_multi:.5f}")

print("\n--- 3. PASS/FAIL VERDICT ---")
if audio_single.size == 1920 and ref_frame.pcm.size == 1280 and max_step_single < 0.5:
    print("VERDICT: PASS — CODE2WAV 1-FRAME VALIDATED")
    sys.exit(0)
else:
    print("VERDICT: FAIL — CODE2WAV VALIDATION FAILED")
    sys.exit(1)
