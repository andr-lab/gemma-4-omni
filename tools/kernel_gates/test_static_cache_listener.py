#!/usr/bin/env python3
"""Test StaticCache with ShapeStableGemmaListenerLM on RTX 3090."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from transformers import StaticCache


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    print("=" * 70)
    print("[TEST] Testing StaticCache compatibility with Gemma Listener")
    print("=" * 70, flush=True)

    th = load_thinker("/workspace", device=device, dtype=dtype)
    lm = th.model.model.language_model

    sc = StaticCache(
        config=lm.config,
        max_batch_size=1,
        max_cache_len=384,
        device=device,
        dtype=dtype,
    )
    print("StaticCache initialized successfully!")
    print("StaticCache key layers:", len(sc.key_cache))
    print("StaticCache key shape[0]:", sc.key_cache[0].shape)


if __name__ == "__main__":
    main()
