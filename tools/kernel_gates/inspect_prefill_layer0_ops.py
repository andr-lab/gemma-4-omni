#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import torch
import numpy as np
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from duplex_kernel.listener import GemmaAudioTowerBackend
from duplex_kernel.native_listener import NativeGemmaListenerLM
from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT
from test_progressive_layers import cosine_sim, mae, max_diff

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

th = load_thinker("/workspace", device=device, dtype=dtype)
lm = th.model.model.language_model
backend = GemmaAudioTowerBackend(th, device=str(device))
native_lm = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)
ss_lm = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)

input_ids, attention_mask = backend._build_multimodal_inputs((), 1024, device=device)
stable_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
stable_attention = attention_mask[:, :AUDIO_START_AFTER_TEXT].to(dtype=torch.int64).contiguous()
with torch.inference_mode():
    stable_embeds = lm.embed_tokens(stable_ids)
    stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)

q_len = AUDIO_START_AFTER_TEXT # 34
pos_ids = torch.arange(0, q_len, device=device, dtype=torch.long).unsqueeze(0)
layer = lm.layers[0]
attn = layer.self_attn
mlp = layer.mlp
D = 256
hidden_shape = (1, q_len, -1, D)

# PyTorch Reference Step by Step:
# 1. Input Norm
mean_sq = stable_embeds.float().pow(2).mean(-1, keepdim=True) + layer.input_layernorm.eps
norm_py = (stable_embeds.float() * torch.pow(mean_sq, -0.5) * layer.input_layernorm.weight.float()).to(dtype)

# 2. Q Proj + Norm + RoPE
import torch.nn.functional as F
q = F.linear(norm_py, attn.q_proj.weight).view(hidden_shape)
q_mean_sq = q.float().pow(2).mean(-1, keepdim=True) + attn.q_norm.eps
q = (q.float() * torch.pow(q_mean_sq, -0.5) * attn.q_norm.weight.float()).to(dtype)

cos, sin = lm.rotary_emb(stable_embeds, pos_ids, "sliding_attention")
half_d = D // 2
cos_half = cos[:, :, :half_d].unsqueeze(2)
sin_half = sin[:, :, :half_d].unsqueeze(2)
q1, q2 = q[:, :, :, :half_d], q[:, :, :, half_d:]
q_rot_py = torch.cat([q1 * cos_half - q2 * sin_half, q2 * cos_half + q1 * sin_half], dim=-1).to(dtype).transpose(1, 2)

# 3. K Proj + Norm + RoPE
k = F.linear(norm_py, attn.k_proj.weight).view(hidden_shape)
k_mean_sq = k.float().pow(2).mean(-1, keepdim=True) + attn.k_norm.eps
k = (k.float() * torch.pow(k_mean_sq, -0.5) * attn.k_norm.weight.float()).to(dtype)
k1, k2 = k[:, :, :, :half_d], k[:, :, :, half_d:]
k_rot_py = torch.cat([k1 * cos_half - k2 * sin_half, k2 * cos_half + k1 * sin_half], dim=-1).to(dtype).transpose(1, 2)

v = F.linear(norm_py, attn.v_proj.weight).view(hidden_shape)
v_mean_sq = v.float().pow(2).mean(-1, keepdim=True) + attn.v_norm.eps
v_w = getattr(attn.v_norm, "weight", None)
v_py = (v.float() * torch.pow(v_mean_sq, -0.5) * (v_w.float() if v_w is not None else 1.0)).to(dtype).transpose(1, 2)

# 4. Causal SDPA in PyTorch
causal_mask = torch.tril(torch.ones(q_len, q_len, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
causal_mask = causal_mask & stable_attention.to(torch.bool).view(1, 1, 1, q_len)
scores = torch.matmul(q_rot_py.float(), k_rot_py.transpose(-1, -2).float()) * attn.scaling
scores = scores.masked_fill(~causal_mask, -1e20)
attn_weights = F.softmax(scores, dim=-1).to(dtype)
sdpa_py = torch.matmul(attn_weights, v_py).transpose(1, 2).reshape(1, q_len, -1)

# 5. O Proj + Post-Attn Norm + Residual Add
o_proj = F.linear(sdpa_py, attn.o_proj.weight)
o_mean_sq = o_proj.float().pow(2).mean(-1, keepdim=True) + layer.post_attention_layernorm.eps
o_norm_py = (o_proj.float() * torch.pow(o_mean_sq, -0.5) * layer.post_attention_layernorm.weight.float()).to(dtype)
hidden_after_attn = stable_embeds + o_norm_py

# 6. Pre-FF Norm + MLP + Post-FF Norm + Residual Add
ff_mean_sq = hidden_after_attn.float().pow(2).mean(-1, keepdim=True) + layer.pre_feedforward_layernorm.eps
ff_norm_py = (hidden_after_attn.float() * torch.pow(ff_mean_sq, -0.5) * layer.pre_feedforward_layernorm.weight.float()).to(dtype)
gate_py = F.linear(ff_norm_py, mlp.gate_proj.weight)
up_py = F.linear(ff_norm_py, mlp.up_proj.weight)
act_py = mlp.act_fn(gate_py) * up_py
mlp_down = F.linear(act_py, mlp.down_proj.weight)
down_mean_sq = mlp_down.float().pow(2).mean(-1, keepdim=True) + layer.post_feedforward_layernorm.eps
down_norm_py = (mlp_down.float() * torch.pow(down_mean_sq, -0.5) * layer.post_feedforward_layernorm.weight.float()).to(dtype)
hidden_after_ff = hidden_after_attn + down_norm_py

# 7. PLE Block
projected_ple = ss_lm._project_per_layer_inputs(stable_embeds, stable_ple)
per_layer_in = projected_ple[:, :, 0, :]
ple_gate_py = F.linear(hidden_after_ff, layer.per_layer_input_gate.weight)
ple_act_py = layer.act_fn(ple_gate_py) * per_layer_in
ple_proj = F.linear(ple_act_py, layer.per_layer_projection.weight)
ple_mean_sq = ple_proj.float().pow(2).mean(-1, keepdim=True) + layer.post_per_layer_input_norm.eps
ple_norm_py = (ple_proj.float() * torch.pow(ple_mean_sq, -0.5) * layer.post_per_layer_input_norm.weight.float()).to(dtype)
layer0_out_py = (hidden_after_ff + ple_norm_py) * layer.layer_scalar

# Now check Native:
native_lm.reset()
native_lm.prime_debug(
    stable_embeds, stable_ple, stable_attention, num_layers=1, apply_final_norm=False
)
torch.cuda.synchronize()

# Read Native Buffers:
nat_norm = native_lm.copy_debug_tensor("norm_buf", (1, q_len, 1536))
nat_q = native_lm.copy_debug_tensor("q_buf", (1, q_len, 8, D))
nat_sdpa = native_lm.copy_debug_tensor("attn_out", (1, q_len, 8 * D))
nat_hidden_a = native_lm.copy_debug_tensor("hidden_a", (1, q_len, 1536))
nat_gate = native_lm.copy_debug_tensor("gate_buf", (1, q_len, mlp.gate_proj.weight.shape[0]))
nat_act = native_lm.copy_debug_tensor("mlp_act_buf", (1, q_len, mlp.gate_proj.weight.shape[0]))
nat_ple_gate = native_lm.copy_debug_tensor("ple_gate_buf", (1, q_len, 256))
nat_ple_act = native_lm.copy_debug_tensor("ple_act_buf", (1, q_len, 256))
nat_ple_proj = native_lm.copy_debug_tensor("ple_proj_out", (1, q_len, 1536))

print("=== Layer 0 Prefill (Q=34) Validation ===")
q_rot_py_t = q_rot_py.transpose(1, 2)
print(f"1. Q RoPE:      MAE = {mae(q_rot_py_t, nat_q):.4e}, Cosine = {cosine_sim(q_rot_py_t, nat_q):.6f}")
print(f"2. SDPA Out:    MAE = {mae(sdpa_py, nat_sdpa):.4e}, Cosine = {cosine_sim(sdpa_py, nat_sdpa):.6f}")
print(f"3. Pre-FF Norm: MAE = {mae(ff_norm_py, nat_norm):.4e}, Cosine = {cosine_sim(ff_norm_py, nat_norm):.6f}")
print(f"4. Gate Proj:   MAE = {mae(gate_py, nat_gate):.4e}, Cosine = {cosine_sim(gate_py, nat_gate):.6f}")
print(f"5. GeGLU Act:   MAE = {mae(act_py, nat_act):.4e}, Cosine = {cosine_sim(act_py, nat_act):.6f}")
print(f"6. PLE Gate:    MAE = {mae(ple_gate_py, nat_ple_gate):.4e}, Cosine = {cosine_sim(ple_gate_py, nat_ple_gate):.6f}")
print(f"7. PLE Act:     MAE = {mae(ple_act_py, nat_ple_act):.4e}, Cosine = {cosine_sim(ple_act_py, nat_ple_act):.6f}")
print(f"8. PLE Norm:    MAE = {mae(ple_norm_py, nat_ple_proj):.4e}, Cosine = {cosine_sim(ple_norm_py, nat_ple_proj):.6f}")
print(f"9. Layer 0 Out: MAE = {mae(layer0_out_py, nat_hidden_a):.4e}, Cosine = {cosine_sim(layer0_out_py, nat_hidden_a):.6f}")
