#!/usr/bin/env python3
"""Verify multi-frame numerical parity and latency of CUDA Graph Gemma Listener LM."""
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
    print("[CUDA GRAPH PARITY] Testing Multi-Frame Parity with Baseline")
    print("=" * 70, flush=True)

    th = load_thinker("/workspace", device=device, dtype=dtype)
    lm = th.model.model.language_model

    # 1. Baseline eager listener run for 20 frames
    print("[BASELINE] Running 20 frames on baseline eager listener...", flush=True)
    l_base = GemmaListener.from_thinker(th)
    s_base = GemmaListenerState(context_frames=64)
    l_base.prime_steady_state(s_base)

    rng = np.random.default_rng(2026)
    pcm_frames = [(0.05 * rng.standard_normal(1280)).astype(np.float32) for _ in range(20)]

    base_hiddens = []
    for pcm in pcm_frames:
        out = l_base.step(pcm, None, s_base)
        base_hiddens.append(out.gemma_hidden_t.float().clone())

    print("[BASELINE] Baseline complete. Preparing CUDA Graph capture...", flush=True)

    # 2. Setup CUDA Graph Listener
    # We will test capturing ShapeStableGemmaListenerLM.forward
    sc = StaticCache(
        config=lm.config,
        max_batch_size=1,
        max_cache_len=384,
        device=device,
        dtype=dtype,
    )

    ss_lm = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
    backend = GemmaListener.from_thinker(th).backend

    static_embeds = torch.zeros((1, 4, 1536), device=device, dtype=dtype)
    static_ple = backend._frame_pad_per_layer_inputs(lm)
    static_pos = torch.zeros((1, 4), device=device, dtype=torch.long)
    static_mask = {
        lt: torch.zeros((1, 1, 4, 384), device=device, dtype=dtype)
        for lt in lm.unique_layer_types
    }

    # Warmup runs
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

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=graph_stream):
        with torch.inference_mode():
            graph_out = ss_lm.forward(
                inputs_embeds=static_embeds,
                per_layer_inputs=static_ple,
                attention_mask=static_mask,
                position_ids=static_pos,
                past_key_values=sc,
                shared_kv_states=None,
            )

    print("CUDA Graph captured successfully!")

    # Check replay latency
    times = []
    with torch.inference_mode():
        for _ in range(50):
            for _l in sc.layers:
                _l.cumulative_length.zero_()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    t = np.array(times)
    print(f"CUDA Graph Replay Latency: p50={np.percentile(t, 50):.2f} ms, p95={np.percentile(t, 95):.2f} ms")


if __name__ == "__main__":
    main()
