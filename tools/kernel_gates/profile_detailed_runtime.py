#!/usr/bin/env python3
"""Detailed CUDA-event profiler per il realtime path di Gemma-4-Omni:
Scompone la latenza per-frame con risoluzione al microsecondo:
- V9 incremental step
- Qwen push prefix slot
- Qwen input embedding prep (aggregate committed codes)
- Qwen position_ids & cache_position tensor setup
- Qwen _slot_mask creation
- Qwen 28 Transformer layers forward
- Qwen codec_head (CB0 projection)
- SubTalker (15 CUDA graphs: prefill + 14 steps + sampling)
- Allocazioni dinamiche per frame (torch.cuda.memory_stats)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.future_prefix import FuturePrefixQwen, _aggregate_committed_codes
from duplex_kernel.models.talker import load_talker
from duplex_kernel.models.v9 import (
    DeepCausalTransformerProjectorV9,
    CausalConv1D,
    IncrementalV9Projector,
    load_v9_conditioning,
)
from duplex_kernel.subtalker_graph import SubtalkerGraph


def p_stats(arr: list[float]) -> dict[str, float]:
    a = np.array(arr)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(np.mean(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def run_detailed_profile(num_frames: int = 100, turn_len: int = 64) -> dict[str, Any]:
    device = "cuda"
    dtype = torch.bfloat16

    print("[PROFILER] Caricamento componenti in bfloat16...", flush=True)
    talker = load_talker("/workspace", device=device, load_code2wav=False)
    talker.talker.to(dtype=dtype)
    projector, causal_conv = load_v9_conditioning(device=device, dtype=dtype)

    cp = talker.talker.code_predictor
    subtalker_graph = SubtalkerGraph(cp, talker_embed=talker.talker.get_input_embeddings()).capture()

    inc_v9 = IncrementalV9Projector(projector, causal_conv)
    v9_state = inc_v9.new_state(batch_size=1, dtype=dtype, max_seq_len=turn_len)

    speaker = FuturePrefixQwen(
        talker,
        lookahead_frames=0,
        max_prefix_frames=turn_len,
        device=device,
        dtype=dtype,
        max_audio_frames=turn_len,
    )
    speaker.begin("Verifica causale del kernel.")

    dummy_raw = torch.randn((1, 1536), device=device, dtype=dtype)
    dummy_anchor = torch.randn((1, 1536), device=device, dtype=dtype)
    dummy_codes = torch.randint(0, 2048, (16,), device=device, dtype=torch.long)

    # Warmup
    print("[PROFILER] Warmup 10 frame...", flush=True)
    for t in range(10):
        h_pfx = inc_v9.step(dummy_raw, dummy_anchor, v9_state)
        speaker.push_v9_frame(t, h_pfx)
        out = speaker.step_frame()
        c0 = int(out.logits[:, :1, :].argmax(-1).item())
        c0_embed = talker.talker.get_input_embeddings()(torch.tensor([[c0]], device=device))
        subtalker_graph.generate(out.last_hidden, c0, c0_embed=c0_embed)
        speaker.commit_codes(dummy_codes)

    speaker.begin("Verifica causale del kernel.")
    v9_state = inc_v9.new_state(batch_size=1, dtype=dtype, max_seq_len=turn_len)

    # CUDA Events per micro-profiling
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_v9 = torch.cuda.Event(enable_timing=True)
    ev_push = torch.cuda.Event(enable_timing=True)
    ev_embed_prep = torch.cuda.Event(enable_timing=True)
    ev_pos_setup = torch.cuda.Event(enable_timing=True)
    ev_mask_setup = torch.cuda.Event(enable_timing=True)
    ev_layers = torch.cuda.Event(enable_timing=True)
    ev_head = torch.cuda.Event(enable_timing=True)
    ev_subtalker = torch.cuda.Event(enable_timing=True)

    metrics: dict[str, list[float]] = {
        "v9_step": [],
        "push_slot": [],
        "qwen_embed_prep": [],
        "qwen_pos_setup": [],
        "qwen_mask_setup": [],
        "qwen_layers": [],
        "qwen_codec_head": [],
        "qwen_total_step": [],
        "subtalker_total": [],
        "combined_frame": [],
    }

    codec_head = getattr(talker.talker, "codec_head", None)
    if codec_head is None:
        codec_head = talker.talker.model.codec_head
    model = getattr(getattr(talker.talker, "talker", talker.talker), "model", talker.talker)

    print(f"[PROFILER] Esecuzione micro-profiling su {num_frames} frame...", flush=True)

    mem_before = torch.cuda.memory_allocated()
    allocations_count = 0

    for i in range(num_frames):
        t = i % turn_len
        if t == 0 and i > 0:
            speaker.begin("Verifica causale del kernel.")
            v9_state = inc_v9.new_state(batch_size=1, dtype=dtype, max_seq_len=turn_len)

        ev_start.record()

        # 1. V9 incremental step
        h_pfx = inc_v9.step(dummy_raw, dummy_anchor, v9_state)
        ev_v9.record()

        # 2. Push V9 frame nello slot
        speaker.push_v9_frame(t, h_pfx)
        ev_push.record()

        # 3. Qwen step decomposto:
        # 3a. Input embedding preparation
        if t == 0:
            embedding = speaker._bos
        else:
            embedding = _aggregate_committed_codes(
                getattr(talker.talker, "talker", talker.talker),
                speaker._committed_codes[-1:],
                device=device,
                dtype=dtype,
            )
        ev_embed_prep.record()

        # 3b. Position setup
        pos_int = speaker._audio_start + t
        position = torch.tensor([pos_int], device=device, dtype=torch.long)
        pos_ids = speaker._position_ids(position, 1)
        ev_pos_setup.record()

        # 3c. Slot mask setup
        mask = speaker._slot_mask(position, visible_prefix_end=t, q_pos_int=pos_int)
        ev_mask_setup.record()

        # 3d. Qwen 28 Transformer layers forward
        with torch.inference_mode():
            out = model(
                inputs_embeds=embedding,
                attention_mask=mask,
                position_ids=pos_ids,
                cache_position=position,
                past_key_values=speaker._slot_cache,
                use_cache=True,
            )
            hidden = out.last_hidden_state
        ev_layers.record()

        # 3e. Codec head projection (CB0)
        with torch.inference_mode():
            logits = codec_head(hidden)
        ev_head.record()

        # 4. SubTalker (15 CUDA graphs)
        c0 = int(logits[:, :1, :].argmax(-1).item())
        c0_embed = talker.talker.get_input_embeddings()(torch.tensor([[c0]], device=device))
        resid = subtalker_graph.generate(hidden, c0, c0_embed=c0_embed)
        ev_subtalker.record()

        speaker.commit_codes(dummy_codes)

        torch.cuda.synchronize()

        t_v9 = ev_start.elapsed_time(ev_v9)
        t_push = ev_v9.elapsed_time(ev_push)
        t_embed = ev_push.elapsed_time(ev_embed_prep)
        t_pos = ev_embed_prep.elapsed_time(ev_pos_setup)
        t_mask = ev_pos_setup.elapsed_time(ev_mask_setup)
        t_layers = ev_mask_setup.elapsed_time(ev_layers)
        t_head = ev_layers.elapsed_time(ev_head)
        t_qwen_total = ev_push.elapsed_time(ev_head)
        t_sub = ev_head.elapsed_time(ev_subtalker)
        t_comb = ev_start.elapsed_time(ev_subtalker)

        metrics["v9_step"].append(t_v9)
        metrics["push_slot"].append(t_push)
        metrics["qwen_embed_prep"].append(t_embed)
        metrics["qwen_pos_setup"].append(t_pos)
        metrics["qwen_mask_setup"].append(t_mask)
        metrics["qwen_layers"].append(t_layers)
        metrics["qwen_codec_head"].append(t_head)
        metrics["qwen_total_step"].append(t_qwen_total)
        metrics["subtalker_total"].append(t_sub)
        metrics["combined_frame"].append(t_comb)

    mem_after = torch.cuda.memory_allocated()
    print("\n" + "=" * 70)
    print("DETAILED REALTIME BREAKDOWN (CUDA Events, bfloat16, RTX 3090):")
    print("=" * 70)
    summary: dict[str, Any] = {}
    for name, arr in metrics.items():
        st = p_stats(arr)
        summary[name] = st
        print(f"  {name:<20}: p50={st['p50']:6.2f}ms | p95={st['p95']:6.2f}ms | p99={st['p99']:6.2f}ms | mean={st['mean']:6.2f}ms")

    print("-" * 70)
    comb_p95 = summary["combined_frame"]["p95"]
    comb_p50 = summary["combined_frame"]["p50"]
    jitter = summary["combined_frame"]["p99"] - comb_p50
    print(f"  COMBINED TOTAL FRAME: p50={comb_p50:.2f}ms | p95={comb_p95:.2f}ms | jitter={jitter:.2f}ms")
    print(f"  Memory delta: {mem_after - mem_before} B")
    print("=" * 70 + "\n")

    out_path = ROOT / "validation" / "kernel_gates" / "P4D_R" / "P4D_DETAILED_BREAKDOWN.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    run_detailed_profile(num_frames=120)
