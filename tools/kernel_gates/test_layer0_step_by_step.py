#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/kernel_gates/test_layer0_step_by_step.py

Step-by-step mathematical dissection of Layer 0 to certify exact parity.
"""
from __future__ import annotations

import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

from duplex_kernel.models.thinker import load_thinker
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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print("[1] Loading Thinker...")
    th = load_thinker("/workspace", device=device, dtype=dtype)
    lm = th.model.model.language_model

    layer = lm.layers[0]
    attn = layer.self_attn
    mlp = layer.mlp

    # Inputs
    rng = np.random.default_rng(2026)
    x = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype)
    pad_ids = torch.full((1, 4), 0, device=device, dtype=torch.long)
    token_id = lm.get_per_layer_inputs(pad_ids, x)
    pos_ids = torch.arange(0, 4, device=device, dtype=torch.long).unsqueeze(0)

    ss_lm = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
    ref_cache = DynamicCache(config=lm.config)
    ref_mask = {"sliding_attention": None, "full_attention": None}

    projected_ple = ss_lm._project_per_layer_inputs(x, token_id)
    per_layer_input = projected_ple[:, :, 0, :]
    cos, sin = lm.rotary_emb(x, pos_ids, "sliding_attention")

    # --- Step 1: Input RMSNorm ---
    ref_x_norm = ss_lm._padded_rmsnorm(layer.input_layernorm, x)
    mean_sq = x.float().pow(2).mean(-1, keepdim=True) + layer.input_layernorm.eps
    cand_x_norm = (x.float() * torch.pow(mean_sq, -0.5) * layer.input_layernorm.weight.float()).to(dtype)
    print(f"1. Input RMSNorm: MAE={mae(ref_x_norm, cand_x_norm):.2e}, Cos={cosine_sim(ref_x_norm, cand_x_norm):.6f}")

    # --- Step 2: Attention Q/K/V linear & norm ---
    hidden_shape = (1, 4, -1, 256)
    ref_q = ss_lm._padded_rmsnorm(attn.q_norm, ss_lm._padded_linear(attn.q_proj, ref_x_norm).view(hidden_shape))
    cand_q = F.linear(cand_x_norm, attn.q_proj.weight).view(hidden_shape)
    q_mean_sq = cand_q.float().pow(2).mean(-1, keepdim=True) + attn.q_norm.eps
    q_w = getattr(attn.q_norm, "weight", None)
    cand_q = (cand_q.float() * torch.pow(q_mean_sq, -0.5) * (q_w.float() if q_w is not None else 1.0)).to(dtype)
    print(f"2. Q Norm:        MAE={mae(ref_q, cand_q):.2e}, Cos={cosine_sim(ref_q, cand_q):.6f}")

    ref_k = ss_lm._padded_rmsnorm(attn.k_norm, ss_lm._padded_linear(attn.k_proj, ref_x_norm).view(hidden_shape))
    cand_k = F.linear(cand_x_norm, attn.k_proj.weight).view(hidden_shape)
    k_mean_sq = cand_k.float().pow(2).mean(-1, keepdim=True) + attn.k_norm.eps
    k_w = getattr(attn.k_norm, "weight", None)
    cand_k = (cand_k.float() * torch.pow(k_mean_sq, -0.5) * (k_w.float() if k_w is not None else 1.0)).to(dtype)
    print(f"3. K Norm:        MAE={mae(ref_k, cand_k):.2e}, Cos={cosine_sim(ref_k, cand_k):.6f}")

    ref_v = ss_lm._padded_rmsnorm(attn.v_norm, ss_lm._padded_linear(attn.v_proj, ref_x_norm).view(hidden_shape))
    cand_v = F.linear(cand_x_norm, attn.v_proj.weight).view(hidden_shape)
    v_mean_sq = cand_v.float().pow(2).mean(-1, keepdim=True) + attn.v_norm.eps
    v_w = getattr(attn.v_norm, "weight", None)
    cand_v = (cand_v.float() * torch.pow(v_mean_sq, -0.5) * (v_w.float() if v_w is not None else 1.0)).to(dtype)
    print(f"4. V Norm:        MAE={mae(ref_v, cand_v):.2e}, Cos={cosine_sim(ref_v, cand_v):.6f}")

    # --- Step 3: RoPE ---
    ref_q_rot = apply_rotary_pos_emb(ref_q, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    # Manual rotation
    half_d = 128
    cos_half = cos[:, :, :half_d].unsqueeze(2)
    sin_half = sin[:, :, :half_d].unsqueeze(2)
    q1, q2 = cand_q[:, :, :, :half_d], cand_q[:, :, :, half_d:]
    cand_q_rot = torch.cat([q1 * cos_half - q2 * sin_half, q2 * cos_half + q1 * sin_half], dim=-1).to(dtype).transpose(1, 2)
    print(f"5. Q RoPE:        MAE={mae(ref_q_rot, cand_q_rot):.2e}, Cos={cosine_sim(ref_q_rot, cand_q_rot):.6f}")

    ref_k_rot = apply_rotary_pos_emb(ref_k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    k1, k2 = cand_k[:, :, :, :half_d], cand_k[:, :, :, half_d:]
    cand_k_rot = torch.cat([k1 * cos_half - k2 * sin_half, k2 * cos_half + k1 * sin_half], dim=-1).to(dtype).transpose(1, 2)
    print(f"6. K RoPE:        MAE={mae(ref_k_rot, cand_k_rot):.2e}, Cos={cosine_sim(ref_k_rot, cand_k_rot):.6f}")

    # --- Step 4: SDPA ---
    ref_attn_out = ss_lm._attention(attn, ref_x_norm, (cos, sin), None, pos_ids, getattr(ref_cache, "shared_layers", None), ref_cache)

    # Candidate manual SDPA
    scores = torch.matmul(cand_q_rot.float(), cand_k_rot.transpose(-1, -2).float()) * attn.scaling
    # Note: ref_attention received attention_mask=None with is_causal=False
    attn_weights = F.softmax(scores, dim=-1).to(dtype)
    cand_v_t = cand_v.transpose(1, 2)
    cand_sdpa = torch.matmul(attn_weights, cand_v_t).transpose(1, 2).reshape(1, 4, -1)
    cand_attn_out = F.linear(cand_sdpa, attn.o_proj.weight)
    print(f"7. Attn O Proj:   MAE={mae(ref_attn_out, cand_attn_out):.2e}, Cos={cosine_sim(ref_attn_out, cand_attn_out):.6f}")

    # --- Step 5: Post-Attn Norm & Add ---
    ref_post_attn = x + ss_lm._padded_rmsnorm(layer.post_attention_layernorm, ref_attn_out)
    o_mean_sq = cand_attn_out.float().pow(2).mean(-1, keepdim=True) + layer.post_attention_layernorm.eps
    cand_o_norm = (cand_attn_out.float() * torch.pow(o_mean_sq, -0.5) * layer.post_attention_layernorm.weight.float()).to(dtype)
    cand_post_attn = x + cand_o_norm
    print(f"8. Post-Attn Add: MAE={mae(ref_post_attn, cand_post_attn):.2e}, Cos={cosine_sim(ref_post_attn, cand_post_attn):.6f}")

    # --- Step 6: MLP ---
    ref_pre_ff = ss_lm._padded_rmsnorm(layer.pre_feedforward_layernorm, ref_post_attn)
    ref_mlp_out = ss_lm._mlp(mlp, ref_pre_ff)
    ff_mean_sq = cand_post_attn.float().pow(2).mean(-1, keepdim=True) + layer.pre_feedforward_layernorm.eps
    cand_ff_norm = (cand_post_attn.float() * torch.pow(ff_mean_sq, -0.5) * layer.pre_feedforward_layernorm.weight.float()).to(dtype)
    gate = F.linear(cand_ff_norm, mlp.gate_proj.weight)
    up = F.linear(cand_ff_norm, mlp.up_proj.weight)
    act = mlp.act_fn(gate) * up
    cand_mlp_out = F.linear(act, mlp.down_proj.weight)
    print(f"9. MLP: ref_mean={ref_mlp_out.float().mean().item():.4f}, cand_mean={cand_mlp_out.float().mean().item():.4f}")
    print(f"   gate_mean={gate.float().mean().item():.4f}, up_mean={up.float().mean().item():.4f}, act_mean={act.float().mean().item():.4f}")
    print(f"   MLP Output:    MAE={mae(ref_mlp_out, cand_mlp_out):.2e}, Cos={cosine_sim(ref_mlp_out, cand_mlp_out):.6f}")

    ref_post_ff = ref_post_attn + ss_lm._padded_rmsnorm(layer.post_feedforward_layernorm, ref_mlp_out)
    down_mean_sq = cand_mlp_out.float().pow(2).mean(-1, keepdim=True) + layer.post_feedforward_layernorm.eps
    cand_down_norm = (cand_mlp_out.float() * torch.pow(down_mean_sq, -0.5) * layer.post_feedforward_layernorm.weight.float()).to(dtype)
    cand_post_ff = cand_post_attn + cand_down_norm
    print(f"10. Post-FF Add:  MAE={mae(ref_post_ff, cand_post_ff):.2e}, Cos={cosine_sim(ref_post_ff, cand_post_ff):.6f}")

    # --- Step 7: PLE ---
    ple_gate = F.linear(cand_post_ff, layer.per_layer_input_gate.weight)
    ple_act = layer.act_fn(ple_gate) * per_layer_input
    ple_proj = F.linear(ple_act, layer.per_layer_projection.weight)
    ple_mean_sq = ple_proj.float().pow(2).mean(-1, keepdim=True) + layer.post_per_layer_input_norm.eps
    ple_norm = (ple_proj.float() * torch.pow(ple_mean_sq, -0.5) * layer.post_per_layer_input_norm.weight.float()).to(dtype)
    cand_final = (cand_post_ff + ple_norm) * layer.layer_scalar

    ref_final = ss_lm._decoder_layer(layer, x, per_layer_input, (cos, sin), None, pos_ids, getattr(ref_cache, "shared_layers", None), ref_cache)
    print(f"11. Layer 0 Final:MAE={mae(ref_final, cand_final):.2e}, Cos={cosine_sim(ref_final, cand_final):.6f}")


if __name__ == "__main__":
    main()
