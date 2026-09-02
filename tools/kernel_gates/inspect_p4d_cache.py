#!/usr/bin/env python3
import sys
from pathlib import Path
import torch

for name in [
    "cache_ars_runtime_aligned_v1.pt",
    "cache_ars_preconv_v1.pt",
    "cache_ars_stateful_v3.pt",
]:
    p = Path("/workspace") / name
    if not p.exists():
        continue
    c = torch.load(p, map_location="cpu", weights_only=False)
    print("=" * 60)
    print(f"CACHE: {name}")
    print(f"  Keys: {list(c.keys())}")
    print(f"  v9_raw_stage: {c.get('v9_raw_stage')}")
    print(f"  validation_start_frame: {c.get('validation_start_frame')}")
    va_raw = c.get('va_raw', [])
    print(f"  va_raw count: {len(va_raw)}")
    if va_raw:
        print(f"  va_raw[0] shape: {va_raw[0].shape}, dtype: {va_raw[0].dtype}")
    va_htx = c.get('va_htx', [])
    if va_htx:
        print(f"  va_htx[0] shape: {va_htx[0].shape}, dtype: {va_htx[0].dtype}")
    va_c = c.get('va_c', [])
    if va_c:
        print(f"  va_c[0] shape: {va_c[0].shape}, dtype: {va_c[0].dtype}")
