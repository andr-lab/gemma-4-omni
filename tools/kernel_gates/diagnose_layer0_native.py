#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/kernel_gates/diagnose_layer0_native.py

Step-by-step mathematical dissection of Native C/CUDA vs PyTorch ShapeStable
to isolate the exact sub-block where divergence occurs.
"""
from __future__ import annotations

import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb, create_masks_for_generate

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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print("[1] Loading Thinker...")
    th = load_thinker("/workspace", device=device, dtype=dtype)
    lm = th.model.model.language_model

    layer = lm.layers[0]
    attn = layer.self_attn
    mlp = layer.mlp

    # Deterministic Inputs
    rng = np.random.default_rng(2026)
    x = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype)
    pad_ids = torch.full((1, 4), 0, device=device, dtype=torch.long)
    pos_ids = torch.arange(0, 4, device=device, dtype=torch.long).unsqueeze(0)

    print("[2] Initializing Native C Executor...")
    native_lm = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)
    ss_lm = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)

    # Run Native C Layer 0
    print("[3] Running Native C Layer 0 (step_debug)...")
    native_out = native_lm.step_debug(x, num_layers=1, apply_final_norm=False)
    torch.cuda.synchronize()

    # PyTorch Step-by-Step Components
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

    # 1. Input RMSNorm
    mean_sq = x.float().pow(2).mean(-1, keepdim=True) + layer.input_layernorm.eps
    pt_norm_buf = (x.float() * torch.pow(mean_sq, -0.5) * layer.input_layernorm.weight.float()).to(dtype)
    c_norm_buf = native_lm.copy_debug_tensor("norm_buf", (1, 4, 1536))
    print(f"1. Norm Buf (Input RMSNorm): MAE={mae(pt_norm_buf, c_norm_buf):.2e}, Cos={cosine_sim(pt_norm_buf, c_norm_buf):.6f}")

    # 2. Q Projection & RoPE
    hidden_shape = (1, 4, -1, 256)
    cand_q = F.linear(pt_norm_buf, attn.q_proj.weight).view(hidden_shape)
    q_mean_sq = cand_q.float().pow(2).mean(-1, keepdim=True) + attn.q_norm.eps
    cand_q = (cand_q.float() * torch.pow(q_mean_sq, -0.5) * attn.q_norm.weight.float()).to(dtype)
    half_d = 128
    cos_half = cos[:, :, :half_d].unsqueeze(2)
    sin_half = sin[:, :, :half_d].unsqueeze(2)
    q1, q2 = cand_q[:, :, :, :half_d], cand_q[:, :, :, half_d:]
    pt_q_rope = torch.cat([q1 * cos_half - q2 * sin_half, q2 * cos_half + q1 * sin_half], dim=-1).to(dtype).reshape(1, 4, 2048)
    c_q_buf = native_lm.copy_debug_tensor("q_buf", (1, 4, 2048))
    print(f"2. Q Buf (Q Proj+Norm+RoPE): MAE={mae(pt_q_rope, c_q_buf):.2e}, Cos={cosine_sim(pt_q_rope, c_q_buf):.6f}")

    # 3. K Projection & RoPE
    cand_k = F.linear(pt_norm_buf, attn.k_proj.weight).view(hidden_shape)
    k_mean_sq = cand_k.float().pow(2).mean(-1, keepdim=True) + attn.k_norm.eps
    cand_k = (cand_k.float() * torch.pow(k_mean_sq, -0.5) * attn.k_norm.weight.float()).to(dtype)
    k1, k2 = cand_k[:, :, :, :half_d], cand_k[:, :, :, half_d:]
    pt_k_rope = torch.cat([k1 * cos_half - k2 * sin_half, k2 * cos_half + k1 * sin_half], dim=-1).to(dtype).reshape(1, 4, 256)
    c_k_buf = native_lm.copy_debug_tensor("k_buf", (1, 4, 256))
    print(f"3. K Buf (K Proj+Norm+RoPE): MAE={mae(pt_k_rope, c_k_buf):.2e}, Cos={cosine_sim(pt_k_rope, c_k_buf):.6f}")

    # 4. SDPA Output
    c_attn_out = native_lm.copy_debug_tensor("attn_out", (1, 4, 2048))
    # PT manual SDPA with causal mask
    pt_q_heads = pt_q_rope.reshape(1, 4, 8, 256).transpose(1, 2) # [1, 8, 4, 256]
    pt_k_heads = pt_k_rope.reshape(1, 4, 1, 256).transpose(1, 2) # [1, 1, 4, 256]
    cand_v = F.linear(pt_norm_buf, attn.v_proj.weight).view(hidden_shape)
    v_mean_sq = cand_v.float().pow(2).mean(-1, keepdim=True) + attn.v_norm.eps
    v_w = getattr(attn.v_norm, "weight", None)
    cand_v = (cand_v.float() * torch.pow(v_mean_sq, -0.5) * (v_w.float() if v_w is not None else 1.0)).to(dtype)
    pt_v_heads = cand_v.transpose(1, 2) # [1, 1, 4, 256]


    scores = torch.matmul(pt_q_heads.float(), pt_k_heads.transpose(-1, -2).float()) * attn.scaling
    # Causal lower-triangular mask
    causal_mask = torch.tril(torch.ones(4, 4, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(~causal_mask, -1e20)
    attn_weights = F.softmax(scores, dim=-1).to(dtype)
    pt_sdpa = torch.matmul(attn_weights, pt_v_heads).transpose(1, 2).reshape(1, 4, 2048)
    print(f"4. SDPA Attn Out:            MAE={mae(pt_sdpa, c_attn_out):.2e}, Cos={cosine_sim(pt_sdpa, c_attn_out):.6f}")

    ref_attn_out = ss_lm._attention(attn, pt_norm_buf, (cos, sin), ref_mask, pos_ids, None, ref_cache)
    # 5. O Projection + Post-Attn Norm
    pt_o_proj = F.linear(pt_sdpa, attn.o_proj.weight)
    print(f"   ShapeStable _attention vs Manual pt_o_proj: MAE={mae(ref_attn_out, pt_o_proj):.2e}, Cos={cosine_sim(ref_attn_out, pt_o_proj):.6f}")


    o_mean_sq = pt_o_proj.float().pow(2).mean(-1, keepdim=True) + layer.post_attention_layernorm.eps
    pt_post_attn_norm = (pt_o_proj.float() * torch.pow(o_mean_sq, -0.5) * layer.post_attention_layernorm.weight.float()).to(dtype)
    c_hidden_b = native_lm.copy_debug_tensor("hidden_b", (1, 4, 1536))
    print(f"5. O Proj + Post-Attn Norm:  MAE={mae(pt_post_attn_norm, c_hidden_b):.2e}, Cos={cosine_sim(pt_post_attn_norm, c_hidden_b):.6f}")

    # 6. Post-Attn Residual Add
    pt_post_attn = x + pt_post_attn_norm

    # 7. Pre-FF Norm & MLP
    ff_mean_sq = pt_post_attn.float().pow(2).mean(-1, keepdim=True) + layer.pre_feedforward_layernorm.eps
    pt_ff_norm = (pt_post_attn.float() * torch.pow(ff_mean_sq, -0.5) * layer.pre_feedforward_layernorm.weight.float()).to(dtype)
    gate = F.linear(pt_ff_norm, mlp.gate_proj.weight)
    up = F.linear(pt_ff_norm, mlp.up_proj.weight)
    act = mlp.act_fn(gate) * up
    c_mlp_act = native_lm.copy_debug_tensor("mlp_act_buf", (1, 4, 6144))
    print(f"6. MLP GeGLU Activation:     MAE={mae(act, c_mlp_act):.2e}, Cos={cosine_sim(act, c_mlp_act):.6f}")

    pt_mlp_down = F.linear(act, mlp.down_proj.weight)
    down_mean_sq = pt_mlp_down.float().pow(2).mean(-1, keepdim=True) + layer.post_feedforward_layernorm.eps
    pt_down_norm = (pt_mlp_down.float() * torch.pow(down_mean_sq, -0.5) * layer.post_feedforward_layernorm.weight.float()).to(dtype)
    c_hidden_b_ff = native_lm.copy_debug_tensor("hidden_b", (1, 4, 1536))
    # hidden_b at the end of FF has down_norm
    print(f"7. MLP Down + Post-FF Norm:  MAE={mae(pt_down_norm, c_hidden_b_ff):.2e}, Cos={cosine_sim(pt_down_norm, c_hidden_b_ff):.6f}")

    pt_post_ff = pt_post_attn + pt_down_norm

    # 8. PLE Projection & Norm
    c_proj_ple = native_lm.copy_debug_tensor("projected_ple", (1, 4, 35, 256))
    print(f"8. Projected PLE Tensor:     MAE={mae(projected_ple, c_proj_ple):.2e}, Cos={cosine_sim(projected_ple, c_proj_ple):.6f}")

    ple_gate = F.linear(pt_post_ff, layer.per_layer_input_gate.weight)
    ple_act = layer.act_fn(ple_gate) * per_layer_input
    ple_proj = F.linear(ple_act, layer.per_layer_projection.weight)
    ple_mean_sq = ple_proj.float().pow(2).mean(-1, keepdim=True) + layer.post_per_layer_input_norm.eps
    ple_norm = (ple_proj.float() * torch.pow(ple_mean_sq, -0.5) * layer.post_per_layer_input_norm.weight.float()).to(dtype)
    c_ple_proj_out = native_lm.copy_debug_tensor("ple_proj_out", (1, 4, 1536))
    print(f"9. PLE Proj + Post-PLE Norm: MAE={mae(ple_norm, c_ple_proj_out):.2e}, Cos={cosine_sim(ple_norm, c_ple_proj_out):.6f}")

    # 10. Final Layer 0 Output
    pt_final = (pt_post_ff + ple_norm) * layer.layer_scalar
    c_final = native_lm.copy_debug_tensor("hidden_a", (1, 4, 1536))
    print(f"10. Final Layer 0 Output:    MAE={mae(pt_final, c_final):.2e}, Cos={cosine_sim(pt_final, c_final):.6f}")
    print(f"    Native step_debug out:   MAE={mae(pt_final, native_out):.2e}, Cos={cosine_sim(pt_final, native_out):.6f}")


if __name__ == "__main__":
    main()
