#!/usr/bin/env python3
"""Test CUDA Graph capture on ShapeStableGemmaListenerLM on RTX 3090."""
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
from transformers import StaticCache


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    print("=" * 70)
    print("[CUDA GRAPH] Testing CUDA Graph Capture on Gemma Listener LM")
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

    ss_lm = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
    backend = GemmaListener.from_thinker(th).backend

    # Static inputs for capture
    static_embeds = torch.zeros((1, 4, 1536), device=device, dtype=dtype)
    static_ple = backend._frame_pad_per_layer_inputs(lm)
    static_pos = torch.tensor([[100, 101, 102, 103]], device=device, dtype=torch.long)
    static_mask = {
        lt: torch.zeros((1, 1, 4, 384), device=device, dtype=dtype)
        for lt in lm.unique_layer_types
    }

    print("[CUDA GRAPH] Testing warmup runs in dedicated stream...", flush=True)
    graph_stream = torch.cuda.Stream()
    graph_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(graph_stream):
        for _ in range(3):
            with torch.inference_mode():
                out = ss_lm.forward(
                    inputs_embeds=static_embeds,
                    per_layer_inputs=static_ple,
                    attention_mask=static_mask,
                    position_ids=static_pos,
                    past_key_values=sc,
                    shared_kv_states=None,
                )
    torch.cuda.current_stream().wait_stream(graph_stream)
    print("Warmup successful!")

    # Now attempt CUDA Graph capture!
    print("[CUDA GRAPH] Attempting torch.cuda.graph capture...", flush=True)
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, stream=graph_stream):
            with torch.inference_mode():
                out = ss_lm.forward(
                    inputs_embeds=static_embeds,
                    per_layer_inputs=static_ple,
                    attention_mask=static_mask,
                    position_ids=static_pos,
                    past_key_values=sc,
                    shared_kv_states=None,
                )
        print("CUDA GRAPH CAPTURE SUCCESSFUL! Node count:", g.pool())
    except Exception as e:
        print("CUDA GRAPH CAPTURE FAILED:", e)
        import traceback
        traceback.print_exc()
        return

    # Benchmark graph replay!
    print("[CUDA GRAPH] Benchmarking 100 graph replays...", flush=True)
    times = []
    for _ in range(100):
        # Reset StaticCache cumulative_length before replay to prevent offset overflow
        with torch.inference_mode():
            for _l in sc.layers:
                _l.cumulative_length.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    t = np.array(times)
    print(f"\nCUDA Graph Replay Latency (100 replays):")
    print(f"  p50:  {np.percentile(t, 50):.2f} ms")
    print(f"  p95:  {np.percentile(t, 95):.2f} ms")
    print(f"  mean: {np.mean(t):.2f} ms")
    print(f"  min:  {np.min(t):.2f} ms")
    print(f"  max:  {np.max(t):.2f} ms")


if __name__ == "__main__":
    main()
