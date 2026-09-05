#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/kernel_gates/test_progressive_layers.py

Progressive layer-by-layer validation of Native C/CUDA vs PyTorch reference.
Validates:
- Sliding window layers (0..3, head_dim=256)
- Full attention layer (4, head_dim=512)
- All independent KV layers (0..14)
- Shared-KV layers (15..34)
- Full model output (Layer 34 + Final RMSNorm)
"""
from __future__ import annotations

import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

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


def run_pytorch_reference_layers(
    lm: Any,
    ss_lm: ShapeStableGemmaListenerLM,
    x: torch.Tensor,
    num_layers: int,
    apply_final_norm: bool = False,
) -> torch.Tensor:
    device = x.device
    dtype = x.dtype
    q_len = x.shape[1]
    pos_ids = torch.arange(0, q_len, device=device, dtype=torch.long).unsqueeze(0)

    # 1. PLE projection
    pad_ids = torch.full((1, q_len), 0, device=device, dtype=torch.long)
    token_id = lm.get_per_layer_inputs(pad_ids, x)
    projected_ple = ss_lm._project_per_layer_inputs(x, token_id)

    # 2. Rotary embeddings per attention type
    cos_sin_sliding = lm.rotary_emb(x, pos_ids, "sliding_attention")
    cos_sin_full = lm.rotary_emb(x, pos_ids, "full_attention")

    # 3. Cache for reference
    ref_cache = DynamicCache(config=lm.config)
    all_k: dict[int, torch.Tensor] = {}
    all_v: dict[int, torch.Tensor] = {}
    causal_mask = torch.tril(torch.ones(q_len, q_len, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    hidden = x.clone()


    for l_idx in range(num_layers):
        layer = lm.layers[l_idx]
        attn = layer.self_attn
        mlp = layer.mlp
        is_full = (lm.config.layer_types[l_idx] == "full_attention")
        D = getattr(attn, "head_dim", 512 if is_full else 256)
        num_heads = lm.config.num_attention_heads
        cos, sin = cos_sin_full if is_full else cos_sin_sliding
        per_layer_in = projected_ple[:, :, l_idx, :]



        # 1. Input Norm
        mean_sq = hidden.float().pow(2).mean(-1, keepdim=True) + layer.input_layernorm.eps
        norm_x = (hidden.float() * torch.pow(mean_sq, -0.5) * layer.input_layernorm.weight.float()).to(dtype)

        # 2. Q/K/V Proj + Norm + RoPE
        hidden_shape = (1, q_len, -1, D)
        q = F.linear(norm_x, attn.q_proj.weight).view(hidden_shape)
        q_mean_sq = q.float().pow(2).mean(-1, keepdim=True) + attn.q_norm.eps
        q = (q.float() * torch.pow(q_mean_sq, -0.5) * attn.q_norm.weight.float()).to(dtype)

        half_d = D // 2
        cos_half = cos[:, :, :half_d].unsqueeze(2)
        sin_half = sin[:, :, :half_d].unsqueeze(2)
        q1, q2 = q[:, :, :, :half_d], q[:, :, :, half_d:]
        q_rot = torch.cat([q1 * cos_half - q2 * sin_half, q2 * cos_half + q1 * sin_half], dim=-1).to(dtype).transpose(1, 2)

        if not attn.is_kv_shared_layer:
            k = F.linear(norm_x, attn.k_proj.weight).view(hidden_shape)
            k_mean_sq = k.float().pow(2).mean(-1, keepdim=True) + attn.k_norm.eps
            k = (k.float() * torch.pow(k_mean_sq, -0.5) * attn.k_norm.weight.float()).to(dtype)
            k1, k2 = k[:, :, :, :half_d], k[:, :, :, half_d:]
            k_rot = torch.cat([k1 * cos_half - k2 * sin_half, k2 * cos_half + k1 * sin_half], dim=-1).to(dtype).transpose(1, 2)

            v = F.linear(norm_x, attn.v_proj.weight).view(hidden_shape)
            v_mean_sq = v.float().pow(2).mean(-1, keepdim=True) + attn.v_norm.eps
            v_w = getattr(attn.v_norm, "weight", None)
            v = (v.float() * torch.pow(v_mean_sq, -0.5) * (v_w.float() if v_w is not None else 1.0)).to(dtype).transpose(1, 2)

            ref_cache.update(k_rot, v, l_idx)
            all_k[l_idx] = k_rot
            all_v[l_idx] = v
            effective_k, effective_v = k_rot, v
        else:
            src_idx = getattr(attn, "kv_shared_layer_index", 13 if not is_full else 14)
            effective_k = all_k[src_idx]
            effective_v = all_v[src_idx]



        # 3. SDPA
        scores = torch.matmul(q_rot.float(), effective_k.transpose(-1, -2).float()) * attn.scaling
        scores = scores.masked_fill(~causal_mask, -1e20)
        attn_weights = F.softmax(scores, dim=-1).to(dtype)
        sdpa_out = torch.matmul(attn_weights, effective_v).transpose(1, 2).reshape(1, q_len, -1)

        # 4. O Proj + Post-Attn Norm + Residual Add
        o_proj = F.linear(sdpa_out, attn.o_proj.weight)
        o_mean_sq = o_proj.float().pow(2).mean(-1, keepdim=True) + layer.post_attention_layernorm.eps
        o_norm = (o_proj.float() * torch.pow(o_mean_sq, -0.5) * layer.post_attention_layernorm.weight.float()).to(dtype)
        hidden = hidden + o_norm

        # 5. Pre-FF Norm + MLP + Post-FF Norm + Residual Add
        ff_mean_sq = hidden.float().pow(2).mean(-1, keepdim=True) + layer.pre_feedforward_layernorm.eps
        ff_norm = (hidden.float() * torch.pow(ff_mean_sq, -0.5) * layer.pre_feedforward_layernorm.weight.float()).to(dtype)
        gate = F.linear(ff_norm, mlp.gate_proj.weight)
        up = F.linear(ff_norm, mlp.up_proj.weight)
        act = mlp.act_fn(gate) * up
        mlp_down = F.linear(act, mlp.down_proj.weight)
        down_mean_sq = mlp_down.float().pow(2).mean(-1, keepdim=True) + layer.post_feedforward_layernorm.eps
        down_norm = (mlp_down.float() * torch.pow(down_mean_sq, -0.5) * layer.post_feedforward_layernorm.weight.float()).to(dtype)
        hidden = hidden + down_norm

        # 6. PLE Block
        ple_gate = F.linear(hidden, layer.per_layer_input_gate.weight)
        ple_act = layer.act_fn(ple_gate) * per_layer_in
        ple_proj = F.linear(ple_act, layer.per_layer_projection.weight)
        ple_mean_sq = ple_proj.float().pow(2).mean(-1, keepdim=True) + layer.post_per_layer_input_norm.eps
        ple_norm = (ple_proj.float() * torch.pow(ple_mean_sq, -0.5) * layer.post_per_layer_input_norm.weight.float()).to(dtype)
        hidden = (hidden + ple_norm) * layer.layer_scalar

    if apply_final_norm:
        final_mean_sq = hidden.float().pow(2).mean(-1, keepdim=True) + 1e-6
        hidden = (hidden.float() * torch.pow(final_mean_sq, -0.5) * lm.norm.weight.float()).to(dtype)

    return hidden


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print("[1] Loading Thinker...")
    th = load_thinker("/workspace", device=device, dtype=dtype)
    lm = th.model.model.language_model

    print("[2] Initializing PyTorch Reference and Native C Executor...")
    ss_lm = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
    native_lm = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)

    test_stages = [
        (1, False, "Layer 0 (Sliding, First Layer)"),
        (2, False, "Layers 0..1 (Sliding)"),
        (4, False, "Layers 0..3 (All First 4 Sliding Layers)"),
        (5, False, "Layers 0..4 (Includes Layer 4 Full-Attn D=512)"),
        (10, False, "Layers 0..9 (Mixed Sliding & Full Attn)"),
        (15, False, "Layers 0..14 (All Independent KV Layers)"),
        (16, False, "Layers 0..15 (Includes First Shared-KV Layer)"),
        (25, False, "Layers 0..24 (Deep Shared-KV Layers)"),
        (35, False, "All 35 Layers (Pre-Final Norm)"),
        (35, True,  "Full Model Output (35 Layers + Final RMSNorm)"),
    ]

    rng = np.random.default_rng(2026)
    x = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype)

    print("\n" + "=" * 78)
    print(f"{'STAGE':<42} | {'MAE':<10} | {'COSINE':<10} | {'MAX DIFF':<10} | STATUS")
    print("=" * 78)

    all_passed = True
    for num_layers, final_norm, label in test_stages:
        native_lm.reset()
        ref_out = run_pytorch_reference_layers(lm, ss_lm, x, num_layers=num_layers, apply_final_norm=final_norm)
        native_out = native_lm.step_debug(x, num_layers=num_layers, apply_final_norm=final_norm)
        torch.cuda.synchronize()

        m = mae(ref_out, native_out)
        c = cosine_sim(ref_out, native_out)
        mx = max_diff(ref_out, native_out)

        # In BF16, 35 layers accumulate minor ULP roundings, requiring cos >= 0.999
        pass_gate = (c >= 0.999) and (m <= 0.05)
        status = "PASS" if pass_gate else "FAIL"
        if not pass_gate:
            all_passed = False

        print(f"{label:<42} | {m:<10.2e} | {c:<10.6f} | {mx:<10.4f} | {status}")

    print("=" * 78)
    if all_passed:
        print("\n>>> ALL PROGRESSIVE LAYER STAGES: 100% PASS <<<")
        sys.exit(0)
    else:
        print("\n>>> ONE OR MORE STAGES FAILED <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
