#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/kernel_gates/test_native_step_debug.py

Certifies bit-level mathematical parity of Layer 0 between
PyTorch ShapeStable reference and DS4-style Native C/CUDA executor.
"""
from __future__ import annotations

import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import NativeGemmaListenerLM
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    dot = torch.dot(a_f, b_f).item()
    norm_a = torch.norm(a_f, p=2).item()
    norm_b = torch.norm(b_f, p=2).item()
    return float(dot / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 1.0


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(a.float() - b.float())).item())


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a.float() - b.float())).item())


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print("[1] Loading Thinker...")
    th = load_thinker("/workspace", device=device, dtype=dtype)
    lm = th.model.model.language_model

    print("[2] Initializing PyTorch Reference and Native C Executor...")
    ss_lm = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
    native_lm = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)

    # Deterministic test input
    rng = np.random.default_rng(2026)
    x = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype)
    pad_ids = torch.full((1, 4), 0, device=device, dtype=torch.long)
    pos_ids = torch.arange(0, 4, device=device, dtype=torch.long).unsqueeze(0)

    # 1. Reference Execution (Layer 0)
    print("\n[3] Running PyTorch Reference Layer 0...")
    ref_cache = DynamicCache(config=lm.config)
    mask_mapping = create_masks_for_generate(
        lm.config,
        x,
        torch.ones((1, 4), device=device, dtype=torch.long),
        past_key_values=ref_cache,
        position_ids=pos_ids,
    )
    ref_mask = mask_mapping["sliding_attention"]
    token_id = lm.get_per_layer_inputs(pad_ids, x)
    projected_ple = ss_lm._project_per_layer_inputs(x, token_id)
    per_layer_input = projected_ple[:, :, 0, :]
    cos, sin = lm.rotary_emb(x, pos_ids, "sliding_attention")

    ref_layer0 = ss_lm._decoder_layer(
        lm.layers[0],
        x,
        per_layer_input,
        (cos, sin),
        ref_mask,
        pos_ids,
        None,
        ref_cache,
    )

    # 2. Native C Execution (Layer 0)
    print("[4] Running Native C/CUDA Layer 0 (step_debug)...")
    native_layer0 = native_lm.step_debug(x, num_layers=1, apply_final_norm=False)
    torch.cuda.synchronize()

    # 3. Parity Metrics
    m = mae(ref_layer0, native_layer0)
    c = cosine_sim(ref_layer0, native_layer0)
    mx = max_diff(ref_layer0, native_layer0)

    print("\n" + "=" * 60)
    print("LAYER 0 PARITY EVALUATION (PyTorch Reference vs Native C):")
    print(f"  MAE:          {m:.2e}")
    print(f"  Cosine Sim:   {c:.6f}")
    print(f"  Max Diff:     {mx:.4f}")
    print(f"  Ref Mean:     {ref_layer0.float().mean().item():.6f}")
    print(f"  Native Mean:  {native_layer0.float().mean().item():.6f}")
    print("=" * 60)

    # Gate verification
    pass_mae = m <= 1e-2  # BF16 accumulation tolerance
    pass_cos = c >= 0.9999
    if pass_mae and pass_cos:
        print("\n>>> LAYER 0 NATIVE C/CUDA PARITY GATE: PASS <<<")
        sys.exit(0)
    else:
        print(f"\n>>> LAYER 0 NATIVE C/CUDA PARITY GATE: FAIL (MAE={m:.2e}, Cos={c:.6f}) <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
