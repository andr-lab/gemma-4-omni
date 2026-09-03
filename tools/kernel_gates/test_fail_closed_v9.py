#!/usr/bin/env python3
import os
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.v9 import load_v9_conditioning
from duplex_kernel.config import V9_CKPT, EXPECTED_CAUSAL_V9_SHA256

print(f"V9_CKPT resolved to: {V9_CKPT}")
proj, cc = load_v9_conditioning(device="cuda")
print(f"Projector loaded: {type(proj)}")
print(f"CausalConv loaded: {type(cc)}")
print(f"CausalConv conv weight shape: {tuple(cc.conv.weight.shape)}")
print("TEST 1 (Normal production load) PASS!")
