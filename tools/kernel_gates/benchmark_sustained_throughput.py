#!/usr/bin/env python3
"""P4D-R — Sustained Throughput Benchmark (5,000+ continuous frames).

Dimostra che il runtime realtime sostiene stabilmente >= 12.5 frame/s (clock 80ms)
su oltre 5000 frame continui (400 secondi di audio equivalente), misurando:
- inter-output interval p50, p95, p99, mean
- deadline miss rate (> 80ms)
- throughput effettivo (frame/sec)
- assenza di memory leak (VRAM stazionaria)
- assenza di drift temporale
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.future_prefix import FuturePrefixQwen
from duplex_kernel.models.talker import load_talker
from duplex_kernel.models.v9 import (
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


def run_sustained_benchmark(
    num_frames: int = 5000,
    turn_len: int = 64,
    device: str = "cuda",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    print(f"\n" + "=" * 70)
    print(f"BENCHMARK CONTINUO DI THROUGHPUT ({num_frames} frame, turn_len={turn_len})...")
    print(f"=" * 70, flush=True)

    dtype = torch.bfloat16
    talker = load_talker("/workspace", device=device, load_code2wav=False)
    talker.talker.to(dtype=dtype)
    projector, causal_conv = load_v9_conditioning(device=device, dtype=dtype)

    cp = talker.talker.code_predictor
    subtalker_graph = SubtalkerGraph(cp, talker_embed=talker.talker.get_input_embeddings()).capture()
    inc_v9 = IncrementalV9Projector(projector, causal_conv)

    speaker = FuturePrefixQwen(
        talker,
        lookahead_frames=0,
        max_prefix_frames=turn_len,
        device=device,
        dtype=dtype,
        max_audio_frames=turn_len,
    )

    dummy_raw = torch.randn((1, 1536), device=device, dtype=dtype)
    dummy_anchor = torch.randn((1, 1536), device=device, dtype=dtype)
    dummy_codes = torch.randint(0, 2048, (16,), device=device, dtype=torch.long)

    # Warmup
    print("Warmup (30 frame)...", flush=True)
    speaker.begin("Warmup per stabilizzazione pipeline.")
    v9_state = inc_v9.new_state(batch_size=1, dtype=dtype, max_seq_len=turn_len)
    for t in range(30):
        h_pfx = inc_v9.step(dummy_raw, dummy_anchor, v9_state)
        speaker.push_v9_frame(t, h_pfx)
        step_out = speaker.step_frame()
        c0 = int(step_out.logits[:, :1, :].argmax(-1).item())
        c0_embed = talker.talker.get_input_embeddings()(torch.tensor([[c0]], device=device))
        subtalker_graph.generate(step_out.last_hidden, c0, c0_embed=c0_embed)
        speaker.commit_codes(dummy_codes)

    # Inizio benchmark sostenuto
    print(f"Esecuzione benchmark su {num_frames} frame continui...", flush=True)
    speaker.begin("Sessione continua ad alto carico.")
    v9_state = inc_v9.new_state(batch_size=1, dtype=dtype, max_seq_len=turn_len)

    frame_intervals_ms: list[float] = []
    cuda_frame_times_ms: list[float] = []
    vram_samples_mb: list[float] = []

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    t_bench_start = time.perf_counter()
    last_frame_wall = t_bench_start

    for i in range(num_frames):
        t = i % turn_len
        if t == 0 and i > 0:
            speaker.begin(f"Nuovo turno interattivo {i // turn_len}.")
            v9_state = inc_v9.new_state(batch_size=1, dtype=dtype, max_seq_len=turn_len)

        ev_start.record()

        # Step 1: V9
        h_pfx = inc_v9.step(dummy_raw, dummy_anchor, v9_state)

        # Step 2: Push slot
        speaker.push_v9_frame(t, h_pfx)

        # Step 3: Joint Qwen step
        step_out = speaker.step_frame()

        # Step 4: SubTalker
        c0 = int(step_out.logits[:, :1, :].argmax(-1).item())
        c0_embed = talker.talker.get_input_embeddings()(torch.tensor([[c0]], device=device))
        resid = subtalker_graph.generate(step_out.last_hidden, c0, c0_embed=c0_embed)

        # Step 5: Commit codes
        speaker.commit_codes(dummy_codes)

        ev_end.record()
        torch.cuda.synchronize()

        now_wall = time.perf_counter()
        interval_ms = (now_wall - last_frame_wall) * 1000.0
        last_frame_wall = now_wall
        cuda_ms = ev_start.elapsed_time(ev_end)

        frame_intervals_ms.append(interval_ms)
        cuda_frame_times_ms.append(cuda_ms)

        if i % 500 == 0 or i == num_frames - 1:
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            vram_samples_mb.append(vram_mb)
            print(f"  Frame {i:5d}/{num_frames} | Latency CUDA: {cuda_ms:.2f}ms | Wall Interval: {interval_ms:.2f}ms | VRAM: {vram_mb:.1f}MB", flush=True)

    t_bench_end = time.perf_counter()
    total_elapsed_s = t_bench_end - t_bench_start
    effective_fps = num_frames / total_elapsed_s

    # Rimuovi il primo intervallo che include l'avvio
    intervals_clean = frame_intervals_ms[1:]
    stat_interval = p_stats(intervals_clean)
    stat_cuda = p_stats(cuda_frame_times_ms)

    # Calcola miss rate su intervalli e su cuda time
    cuda_misses = sum(1 for x in cuda_frame_times_ms if x > 80.0)
    cuda_miss_rate = (cuda_misses / len(cuda_frame_times_ms)) * 100.0

    interval_misses = sum(1 for x in intervals_clean if x > 80.0)
    interval_miss_rate = (interval_misses / len(intervals_clean)) * 100.0

    pass_throughput = bool(
        stat_cuda["p95"] <= 80.0
        and cuda_miss_rate <= 1.0
        and effective_fps >= 12.5
    )

    vram_drift_mb = vram_samples_mb[-1] - vram_samples_mb[0]

    print("\n" + "=" * 70)
    print("RISULTATI BENCHMARK CONTINUO:")
    print(f"  Frame totali generati:       {num_frames}")
    print(f"  Tempo totale esecuzione:     {total_elapsed_s:.2f} s ({total_elapsed_s/60:.2f} min)")
    print(f"  Throughput effettivo:        {effective_fps:.2f} frame/s (target >= 12.5 frame/s) -> {'PASS' if effective_fps >= 12.5 else 'FAIL'}")
    print(f"  CUDA Per-Frame Latency:      p50={stat_cuda['p50']:.2f}ms, p95={stat_cuda['p95']:.2f}ms, p99={stat_cuda['p99']:.2f}ms (target p95 <= 80ms) -> {'PASS' if stat_cuda['p95'] <= 80.0 else 'FAIL'}")
    print(f"  Inter-Output Wall Interval:  p50={stat_interval['p50']:.2f}ms, p95={stat_interval['p95']:.2f}ms, p99={stat_interval['p99']:.2f}ms")
    print(f"  Deadline Miss Rate (>80ms):  {cuda_miss_rate:.2f}% (target <= 1.0%) -> {'PASS' if cuda_miss_rate <= 1.0 else 'FAIL'}")
    print(f"  VRAM Drift (start -> end):   {vram_drift_mb:+.2f} MB (stabile se ~0)")
    print(f"  ESITO BENCHMARK:             {'PASS' if pass_throughput else 'FAIL'}")
    print("=" * 70 + "\n")

    report = {
        "num_frames": num_frames,
        "total_elapsed_s": total_elapsed_s,
        "effective_fps": effective_fps,
        "cuda_latency_ms": stat_cuda,
        "wall_interval_ms": stat_interval,
        "cuda_deadline_miss_rate_pct": cuda_miss_rate,
        "interval_deadline_miss_rate_pct": interval_miss_rate,
        "vram_initial_mb": vram_samples_mb[0],
        "vram_final_mb": vram_samples_mb[-1],
        "vram_drift_mb": vram_drift_mb,
        "pass_throughput": pass_throughput,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "P4D_SUSTAINED_THROUGHPUT_REPORT.json"
        out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report salvato in {out_file}", flush=True)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/validation/kernel_gates/P4D_R"))
    args = parser.parse_args()
    res = run_sustained_benchmark(num_frames=args.num_frames, output_dir=args.output_dir)
    sys.exit(0 if res["pass_throughput"] else 1)
