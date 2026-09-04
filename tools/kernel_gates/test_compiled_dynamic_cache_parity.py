#!/usr/bin/env python3
"""Test torch.compile on Gemma Listener LM with DynamicCache (no StaticCache) for parity and latency."""
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

from duplex_kernel.listener import GemmaListener, GemmaListenerState
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM


def main() -> None:
    device = "cuda:0"
    dtype = torch.bfloat16
    print("=" * 70)
    print("[TEST] Gemma Listener LM with torch.compile + DynamicCache Parity")
    print("=" * 70, flush=True)

    th = load_thinker("/workspace", device=device, dtype=dtype)
    lm = th.model.model.language_model

    # 1. Baseline Oracle (Eager ShapeStableGemmaListenerLM with DynamicCache)
    listener_base = GemmaListener.from_thinker(th)
    state_base = GemmaListenerState(context_frames=64)
    listener_base.prime_steady_state(state_base)

    # 2. Test candidate: ShapeStable LM
    print("\n--- 1. Testing ShapeStable LM ---", flush=True)
    fast_lm = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
    
    listener_fast = GemmaListener.from_thinker(th)
    listener_fast.backend._shape_stable_lm = fast_lm
    state_fast = GemmaListenerState(context_frames=64)
    listener_fast.prime_steady_state(state_fast)

    rng = np.random.default_rng(2026)
    pcm_test = [(0.05 * rng.standard_normal(1280)).astype(np.float32) for _ in range(10)]

    print("Checking eager baseline vs eager fast_lm...")
    max_diff_eager = 0.0
    for idx, pcm in enumerate(pcm_test[:5]):
        out_b = listener_base.step(pcm, None, state_base)
        out_f = listener_fast.step(pcm, None, state_fast)
        diff = (out_b.gemma_hidden_t.float() - out_f.gemma_hidden_t.float()).abs().max().item()
        cos = F.cosine_similarity(
            out_b.gemma_hidden_t.float().view(1, -1),
            out_f.gemma_hidden_t.float().view(1, -1),
        ).item()
        print(f"  Frame {idx}: diff={diff:.6e}, cos={cos:.7f}")
        max_diff_eager = max(max_diff_eager, diff)

    print(f"Eager baseline match: max_diff = {max_diff_eager:.6e}")

    # 3. Test torch.compile on language_model with DynamicCache
    print("\n--- 2. Testing torch.compile(lm, dynamic=True) with DynamicCache ---", flush=True)
    try:
        compiled_lm = torch.compile(lm, mode="default", dynamic=True)
        # Wrap in a new listener
        listener_compiled = GemmaListener.from_thinker(th)
        # Point the language model to compiled_lm
        listener_compiled.backend.language_model = compiled_lm
        state_comp = GemmaListenerState(context_frames=64)
        print("Priming compiled listener...")
        listener_compiled.prime_steady_state(state_comp)
        
        print("Running 10 frames on compiled listener...")
        diffs = []
        cosines = []
        latencies = []
        for idx, pcm in enumerate(pcm_test):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_c = listener_compiled.step(pcm, None, state_comp)
            torch.cuda.synchronize()
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)
            
            # Compare with oracle
            state_base_fresh = GemmaListenerState(context_frames=64)
            listener_base.prime_steady_state(state_base_fresh)
            # replay up to idx
            for p in pcm_test[:idx+1]:
                out_b = listener_base.step(p, None, state_base_fresh)
                
            diff = (out_b.gemma_hidden_t.float() - out_c.gemma_hidden_t.float()).abs().max().item()
            cos = F.cosine_similarity(
                out_b.gemma_hidden_t.float().view(1, -1),
                out_c.gemma_hidden_t.float().view(1, -1),
            ).item()
            diffs.append(diff)
            cosines.append(cos)
            print(f"  Frame {idx}: diff={diff:.6e}, cos={cos:.7f}, lat={lat:.2f}ms")
            
        print(f"\nResult: MaxDiff={max(diffs):.6e}, MinCos={min(cosines):.7f}")
        print(f"Latency p50={np.percentile(latencies[2:], 50):.2f}ms, p95={np.percentile(latencies[2:], 95):.2f}ms")
    except Exception as e:
        print(f"Compiled DynamicCache failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
