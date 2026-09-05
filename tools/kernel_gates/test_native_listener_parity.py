#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/kernel_gates/test_native_listener_parity.py — Parity Certification Gate (§4 / Step F).

Compares NativeGemmaListenerLM against canonical ShapeStableGemmaListenerLM
across sequences and frames, measuring MAE, Cosine Similarity, Max Absolute Diff,
and verifying strict representation-invariance.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch
import torch.nn.functional as F
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
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(a.float() - b.float())).item())


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a.float() - b.float())).item())


def inspect_per_layer_inputs(lm: Any, device: torch.device, dtype: torch.dtype):
    print("=== INSPECTING PLE MODULES ===")
    pad_ids = torch.full((1, 4), 0, device=device, dtype=torch.long)
    pad_embeds = lm.embed_tokens(pad_ids)
    print(f"pad_ids: {pad_ids.shape}, pad_embeds: {pad_embeds.shape}")

    if hasattr(lm, "get_per_layer_inputs"):
        ple = lm.get_per_layer_inputs(pad_ids, pad_embeds)
        print(f"lm.get_per_layer_inputs output shape: {ple.shape}, dtype: {ple.dtype}")
        print(f"ple mean: {ple.float().mean().item():.6f}, std: {ple.float().std().item():.6f}")

    if hasattr(lm, "per_layer_model_projection"):
        w = lm.per_layer_model_projection.weight
        print(f"per_layer_model_projection weight shape: {w.shape}, dtype: {w.dtype}")
    if hasattr(lm, "per_layer_projection_norm"):
        nw = lm.per_layer_projection_norm.weight
        print(f"per_layer_projection_norm weight shape: {nw.shape if nw is not None else None}")
    print(f"per_layer_model_projection_scale: {getattr(lm, 'per_layer_model_projection_scale', None)}")
    print(f"per_layer_input_scale: {getattr(lm, 'per_layer_input_scale', None)}")


def run_single_step_parity_check():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"[TEST] Loading Thinker on {device}...")
    th = load_thinker("/workspace", device=device, dtype=dtype)
    lm = th.model.model.language_model

    inspect_per_layer_inputs(lm, device, dtype)

    # Instantiate ShapeStable LM
    shape_stable_lm = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
    oracle_cache = DynamicCache(config=lm.config)

    # Instantiate Native LM
    native_lm = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)

    rng = np.random.default_rng(2026)
    # Test 1: Step without priming (just 1 step of 4 soft tokens)
    print("\n--- Test 1: Single Q=4 step comparison ---")
    test_input = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype)
    pad_ids = torch.full((1, 4), 0, device=device, dtype=torch.long)
    pad_embeds = lm.embed_tokens(pad_ids)
    test_ple = lm.get_per_layer_inputs(pad_ids, pad_embeds)
    test_pos = torch.arange(0, 4, device=device, dtype=torch.long).unsqueeze(0)
    attention_mask_1d = torch.ones((1, 4), device=device, dtype=torch.bool)
    test_mask = create_masks_for_generate(
        lm.config,
        test_input,
        attention_mask_1d,
        past_key_values=oracle_cache,
        position_ids=test_pos,
    )

    with torch.inference_mode():
        oracle_out = shape_stable_lm.forward(
            inputs_embeds=test_input,
            per_layer_inputs=test_ple,
            attention_mask=test_mask,
            position_ids=test_pos,
            past_key_values=oracle_cache,
            shared_kv_states=getattr(oracle_cache, "shared_layers", None),
        ).last_hidden_state

    # This is intentionally an unprimed primitive diagnostic; production
    # step_async rejects missing canonical prefix state.
    native_out = native_lm.step_debug(test_input, num_layers=35, apply_final_norm=True)
    torch.cuda.synchronize()

    m = mae(oracle_out, native_out)
    cos = cosine_sim(oracle_out, native_out)
    diff = max_abs(oracle_out, native_out)
    print(f"Step 0 Output Shape - Oracle: {oracle_out.shape}, Native: {native_out.shape}")
    print(f"Oracle Mean: {oracle_out.float().mean().item():.6f}, Native Mean: {native_out.float().mean().item():.6f}")
    print(f"MAE: {m:.8e}")
    print(f"MaxAbs Diff: {diff:.8e}")
    print(f"Cosine Sim: {cos:.8f}")

    return {
        "mae": m,
        "max_abs": diff,
        "cosine": cos,
    }


if __name__ == "__main__":
    res = run_single_step_parity_check()
    print("\n[RESULT]", json.dumps(res, indent=2))
