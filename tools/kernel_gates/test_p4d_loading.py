#!/usr/bin/env python3
import sys
from pathlib import Path
import torch

ROOT = Path("/workspace")
sys.path.insert(0, str(ROOT))

from tools.kernel_gates.p4_checks import _load_codes, _stack_cache

cache_path = Path("/workspace/cache_ars_runtime_aligned_v1.pt")
codes_path = Path("/workspace/data/train_tokens_cv26_it_40.8h_31916s_codes.parquet")
codes_metadata_path = Path("/workspace/data/train_tokens_cv26_it_40.8h_31916s_hyperparams.json")

print("[TEST] Stacking cache...", flush=True)
raw, anchors, cache_codes, raw_stage = _stack_cache(cache_path)
print(f"[TEST] Stacked cache: raw={raw.shape}, anchors={anchors.shape}, cache_codes={cache_codes.shape}, stage={raw_stage}")

print("[TEST] Loading codes...", flush=True)
cache_metadata = torch.load(cache_path, map_location="cpu", weights_only=False)
codes, codes_start = _load_codes(
    codes_path,
    codes_metadata_path,
    cache_codes,
    validation_start_frame=cache_metadata.get("validation_start_frame"),
)
print(f"[TEST] Loaded codes: codes={codes.shape}, start={codes_start}")
print("[TEST] SUCCESS! All inputs perfectly aligned.")
