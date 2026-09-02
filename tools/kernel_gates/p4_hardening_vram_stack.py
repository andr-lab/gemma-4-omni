#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/kernel_gates/p4_hardening_vram_stack.py — Hardening Gate 3: Full Stack VRAM Residency.

Concurrently loads all 5 models into CUDA:0 on NVIDIA RTX 3090 (24GB):
1. Gemma 4 E2B (Thinker + Listener, shared BF16)
2. V9 Causal Transformer Projector + Causal Conv
3. Qwen3-TTS 12Hz 1.7B Base Talker (frozen)
4. SubTalker RVQ Code Predictor (frozen, with CUDA graph)
5. Code2Wav 24kHz Vocoder (TokenizerV2, frozen)

Runs execution verification on each model to ensure all static workspace allocations,
KV caches, and CUDA kernels are resident, then measures allocated, reserved, and free VRAM.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch

from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.listener import GemmaAudioTowerBackend, GemmaListener, GemmaListenerState
from duplex_kernel.models.v9 import load_v9_conditioning
from duplex_kernel.models.talker import load_talker
from duplex_kernel.subtalker_graph import SubtalkerGraph


def _git_info() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
        diff = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT), text=True).strip()
        return sha, bool(diff)
    except Exception:
        return "unknown", False


def run_vram_stack_gate(
    *,
    output_dir: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "proof3_vram_report.json"

    print("[PROOF 3] Starting Full Stack VRAM Residency Gate...", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # 1. Load Gemma 4 E2B (Thinker + Listener shared)
    print("  Loading 1/5: Gemma 4 E2B Thinker + Listener...", flush=True)
    mem_0 = torch.cuda.memory_allocated(device) / (1024**3)
    thinker = load_thinker("/workspace", device=device, dtype=torch.bfloat16)
    backend = GemmaAudioTowerBackend(thinker)
    listener = GemmaListener(backend=backend)
    mem_thinker = torch.cuda.memory_allocated(device) / (1024**3)
    print(f"    Gemma 4 E2B loaded: {mem_thinker - mem_0:.2f} GB allocated", flush=True)

    # 2. Load V9 Conditioning
    print("  Loading 2/5: V9 Causal Transformer Projector + Causal Conv...", flush=True)
    v9, causal_conv = load_v9_conditioning(device=device, dtype=torch.bfloat16)
    mem_v9 = torch.cuda.memory_allocated(device) / (1024**3)
    print(f"    V9 loaded: {mem_v9 - mem_thinker:.2f} GB allocated", flush=True)

    # 3. Load Qwen3-TTS Talker + SubTalker + Code2Wav
    print("  Loading 3/5, 4/5, 5/5: Qwen3-TTS Talker, SubTalker, Code2Wav...", flush=True)
    talker_model = load_talker("/workspace", device=device, load_code2wav=True)
    mem_talker = torch.cuda.memory_allocated(device) / (1024**3)
    print(f"    Talker + Code2Wav loaded: {mem_talker - mem_v9:.2f} GB allocated", flush=True)

    # 4. Initialize SubTalker Graph
    print("  Initializing SubTalker CUDA Graph...", flush=True)
    cp = talker_model.talker.code_predictor
    subtalker_graph = SubtalkerGraph(cp, talker_embed=talker_model.talker.get_input_embeddings()).capture()
    print(f"    SubTalker Graph captured: {subtalker_graph.captured}", flush=True)

    # 5. Execution verification of each component to ensure full activation workspaces are allocated
    print("  Running activation exercise across all 5 models...", flush=True)
    
    # 5a. Thinker forward (1 token)
    with torch.inference_mode():
        dummy_ids = torch.tensor([[2, 100, 200]], device=device, dtype=torch.long)
        th_out = thinker.model(input_ids=dummy_ids, use_cache=True, return_dict=True)
    
    # 5b. Listener step (1 frame)
    listener_state = GemmaListenerState(context_frames=64)
    dummy_pcm = np.zeros(1280, dtype=np.float32)
    with torch.inference_mode():
        list_out = listener.step(dummy_pcm, None, listener_state)
    
    # 5c. V9 forward
    with torch.inference_mode():
        dummy_style = torch.randn(1, 16, 1536, device=device, dtype=torch.bfloat16)
        dummy_anchor = torch.randn(1, 1536, device=device, dtype=torch.bfloat16)
        v9_out = v9(dummy_style, dummy_anchor)
    
    # 5d. SubTalker step
    with torch.inference_mode():
        dummy_c0 = 100
        c0_embed = talker_model.talker.get_input_embeddings()(torch.tensor([[dummy_c0]], device=device))
        dummy_ht = torch.randn(1, 1, 2048, device=device, dtype=torch.bfloat16)
        codes_15 = subtalker_graph.generate(dummy_ht, dummy_c0, c0_embed=c0_embed)
    
    # 5e. Code2Wav decode
    with torch.inference_mode():
        dummy_codes = torch.zeros(1, 12, 16, device=device, dtype=torch.long)
        wav = talker_model.decode(dummy_codes)

    torch.cuda.synchronize()

    # 6. Measure final memory stats
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    alloc_bytes = torch.cuda.memory_allocated(device)
    res_bytes = torch.cuda.memory_reserved(device)
    peak_alloc_bytes = torch.cuda.max_memory_allocated(device)
    peak_res_bytes = torch.cuda.max_memory_reserved(device)

    total_gb = total_bytes / (1024**3)
    free_gb = free_bytes / (1024**3)
    alloc_gb = alloc_bytes / (1024**3)
    res_gb = res_bytes / (1024**3)
    peak_alloc_gb = peak_alloc_bytes / (1024**3)
    peak_res_gb = peak_res_bytes / (1024**3)

    print(f"\n[PROOF 3 RESULT] Full Stack VRAM Residency:")
    print(f"  Total VRAM:           {total_gb:.2f} GB")
    print(f"  Allocated VRAM:       {alloc_gb:.2f} GB")
    print(f"  Reserved VRAM:        {res_gb:.2f} GB")
    print(f"  Peak Allocated:       {peak_alloc_gb:.2f} GB")
    print(f"  Peak Reserved:        {peak_res_gb:.2f} GB")
    print(f"  FREE VRAM Available:  {free_gb:.2f} GB")

    git_sha, git_dirty = _git_info()
    passed = bool(
        subtalker_graph.captured
        and th_out.logits.numel() > 0
        and list_out.gemma_hidden_t.numel() > 0
        and v9_out.numel() > 0
        and len(codes_15) == 15
        and wav.numel() > 0
        and res_gb <= 22.0  # Must fit comfortably inside 24GB RTX 3090 with >= 2GB headroom
        and free_gb >= 2.0
    )

    report = {
        "schema_version": 1,
        "proof": "PROOF_3_FULL_STACK_VRAM",
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "stack_components": [
            {"name": "Gemma 4 E2B Thinker + Listener", "dtype": "bfloat16", "status": "loaded & exercised"},
            {"name": "V9 Projector + Causal Conv", "dtype": "bfloat16", "status": "loaded & exercised"},
            {"name": "Qwen3-TTS 12Hz 1.7B Base Talker", "dtype": "bfloat16", "status": "loaded & exercised"},
            {"name": "SubTalker RVQ Code Predictor", "dtype": "bfloat16", "status": f"loaded & exercised (cuda_graph={subtalker_graph.captured})"},
            {"name": "Code2Wav 24kHz Vocoder", "dtype": "float32", "status": "loaded & exercised"},
        ],
        "vram_metrics_gb": {
            "total": round(total_gb, 3),
            "allocated": round(alloc_gb, 3),
            "reserved": round(res_gb, 3),
            "peak_allocated": round(peak_alloc_gb, 3),
            "peak_reserved": round(peak_res_gb, 3),
            "free_available": round(free_gb, 3),
        },
        "thresholds": {
            "max_reserved_gb": 22.0,
            "min_free_gb": 2.0,
            "all_components_exercised": True,
        },
        "passed": passed,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/validation/kernel_gates/P4_HARDENING"))
    args = parser.parse_args()
    res = run_vram_stack_gate(output_dir=args.output_dir)
    sys.exit(0 if res["passed"] else 1)
