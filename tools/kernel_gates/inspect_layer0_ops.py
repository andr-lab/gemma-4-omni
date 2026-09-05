#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import torch
import numpy as np
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from duplex_kernel.listener import GemmaAudioTowerBackend
from test_progressive_layers import cosine_sim, mae, max_diff

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

th = load_thinker("/workspace", device=device, dtype=dtype)
lm = th.model.model.language_model
backend = GemmaAudioTowerBackend(th, device=str(device))
ss_lm = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)

rng = np.random.default_rng(2026)
audio_soft = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype)
pos_ids = torch.arange(0, 4, device=device, dtype=torch.long).unsqueeze(0)

# Setup Layer 0
layer = lm.layers[0]
attn = layer.self_attn
mlp = layer.mlp

# 1. Input Norm
norm_ref = (audio_soft.float() * torch.pow(audio_soft.float().pow(2).mean(-1, keepdim=True) + layer.input_layernorm.eps, -0.5) * layer.input_layernorm.weight.float()).to(dtype)
norm_ss = ss_lm._padded_rmsnorm(layer.input_layernorm, audio_soft)
print(f"1. Input Norm: MAE={mae(norm_ref, norm_ss):.4e}, Cos={cosine_sim(norm_ref, norm_ss):.6f}")

# 2. Q projection and norm
hidden_shape = (1, 4, -1, 256)
import torch.nn.functional as F
q_ref = F.linear(norm_ref, attn.q_proj.weight).view(hidden_shape)
q_ref = (q_ref.float() * torch.pow(q_ref.float().pow(2).mean(-1, keepdim=True) + attn.q_norm.eps, -0.5) * attn.q_norm.weight.float()).to(dtype)

q_ss = ss_lm._padded_linear(attn.q_proj, norm_ss).view(hidden_shape)
q_ss = ss_lm._padded_rmsnorm(attn.q_norm, q_ss)
print(f"2. Q Proj+Norm: MAE={mae(q_ref, q_ss):.4e}, Cos={cosine_sim(q_ref, q_ss):.6f}")

# 3. RoPE
cos_sin_sliding = lm.rotary_emb(audio_soft, pos_ids, "sliding_attention")
cos, sin = cos_sin_sliding
half_d = 128
cos_half = cos[:, :, :half_d].unsqueeze(2)
sin_half = sin[:, :, :half_d].unsqueeze(2)
q1, q2 = q_ref[:, :, :, :half_d], q_ref[:, :, :, half_d:]
q_rot_ref = torch.cat([q1 * cos_half - q2 * sin_half, q2 * cos_half + q1 * sin_half], dim=-1).to(dtype).transpose(1, 2)

from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
q_rot_ss = apply_rotary_pos_emb(q_ss, cos, sin, unsqueeze_dim=2).transpose(1, 2)
print(f"3. Q RoPE: MAE={mae(q_rot_ref, q_rot_ss):.4e}, Cos={cosine_sim(q_rot_ref, q_rot_ss):.6f}")

# 4. K and V
k_ref = F.linear(norm_ref, attn.k_proj.weight).view(hidden_shape)
k_ref = (k_ref.float() * torch.pow(k_ref.float().pow(2).mean(-1, keepdim=True) + attn.k_norm.eps, -0.5) * attn.k_norm.weight.float()).to(dtype)
k1, k2 = k_ref[:, :, :, :half_d], k_ref[:, :, :, half_d:]
k_rot_ref = torch.cat([k1 * cos_half - k2 * sin_half, k2 * cos_half + k1 * sin_half], dim=-1).to(dtype).transpose(1, 2)

v_ref = F.linear(norm_ref, attn.v_proj.weight).view(hidden_shape)
v_w = getattr(attn.v_norm, "weight", None)
v_ref = (v_ref.float() * torch.pow(v_ref.float().pow(2).mean(-1, keepdim=True) + attn.v_norm.eps, -0.5) * (v_w.float() if v_w is not None else 1.0)).to(dtype).transpose(1, 2)

k_ss = ss_lm._padded_linear(attn.k_proj, norm_ss).view(hidden_shape)
k_ss = ss_lm._padded_rmsnorm(attn.k_norm, k_ss)
k_rot_ss = apply_rotary_pos_emb(k_ss, cos, sin, unsqueeze_dim=2).transpose(1, 2)
v_ss = ss_lm._padded_linear(attn.v_proj, norm_ss).view(hidden_shape)
v_ss = ss_lm._padded_rmsnorm(attn.v_norm, v_ss).transpose(1, 2)

print(f"4. K RoPE: MAE={mae(k_rot_ref, k_rot_ss):.4e}, Cos={cosine_sim(k_rot_ref, k_rot_ss):.6f}")
print(f"4. V Norm: MAE={mae(v_ref, v_ss):.4e}, Cos={cosine_sim(v_ref, v_ss):.6f}")

# 5. Attention SDPA
causal_mask = torch.tril(torch.ones(4, 4, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
scores_ref = torch.matmul(q_rot_ref.float(), k_rot_ref.transpose(-1, -2).float()) * attn.scaling
scores_ref = scores_ref.masked_fill(~causal_mask, -1e20)
attn_weights_ref = F.softmax(scores_ref, dim=-1).to(dtype)
sdpa_ref = torch.matmul(attn_weights_ref, v_ref).transpose(1, 2).reshape(1, 4, -1)

from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate
mask_mapping = create_masks_for_generate(
    lm.config,
    audio_soft,
    torch.ones((1, 4), device=device, dtype=torch.long),
    past_key_values=None,
    position_ids=pos_ids,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward
ss_mask = mask_mapping["sliding_attention"]
key_padded = ss_lm._workspace.padded(k_rot_ss, axis=-2, slot="attention_key")
value_padded = ss_lm._workspace.padded(v_ss, axis=-2, slot="attention_value")
mask_padded = ss_lm._workspace.padded_mask(ss_mask, key_length=4) if isinstance(ss_mask, torch.Tensor) else ss_mask

sdpa_ss, _ = sdpa_attention_forward(
    attn,
    q_rot_ss,
    key_padded,
    value_padded,
    mask_padded,
    dropout=0.0,
    scaling=attn.scaling,
    sliding_window=attn.sliding_window,
    position_ids=pos_ids,
)
sdpa_ss = sdpa_ss.reshape(1, 4, -1).contiguous()

print(f"5. SDPA: MAE={mae(sdpa_ref, sdpa_ss):.4e}, Cos={cosine_sim(sdpa_ref, sdpa_ss):.6f}")
print("ss_mask:", ss_mask)
print("mask_padded shape:", mask_padded.shape if isinstance(mask_padded, torch.Tensor) else type(mask_padded))
