#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import torch
import torch.nn.functional as F
import numpy as np
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import NativeGemmaListenerLM
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from duplex_kernel.listener import GemmaAudioTowerBackend
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

# Prime both
native_lm.reset()
native_lm.prime_async(stable_embeds, stable_ple, stable_attention)
torch.cuda.synchronize()

from collections import UserDict
from transformers.cache_utils import DynamicCache
shared = UserDict()
past_key_values = DynamicCache(config=lm.config)
pos_ids_prefill = torch.arange(0, AUDIO_START_AFTER_TEXT, device=device, dtype=torch.long).unsqueeze(0)
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate
mask_mapping_prefill = create_masks_for_generate(
    lm.config,
    stable_embeds,
    stable_attention,
    past_key_values=None,
    position_ids=pos_ids_prefill,
)
projected_ple_prefill = ss_lm._project_per_layer_inputs(stable_embeds, stable_ple)
pos_embs_prefill = {
    lt: lm.rotary_emb(stable_embeds, pos_ids_prefill, lt)
    for lt in lm.unique_layer_types
}

py_h = stable_embeds.clone()
for l in range(35):
    py_h = ss_lm._decoder_layer(
        lm.layers[l],
        py_h,
        projected_ple_prefill[:, :, l, :],
        pos_embs_prefill[lm.config.layer_types[l]],
        mask_mapping_prefill[lm.config.layer_types[l]],
        pos_ids_prefill,
        shared,
        past_key_values,
    )

# Step 1
rng = np.random.default_rng(2026)
audio_soft = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype)
pos_ids_s1 = torch.arange(AUDIO_START_AFTER_TEXT, AUDIO_START_AFTER_TEXT + 4, device=device, dtype=torch.long).unsqueeze(0)
step_attention = torch.ones((1, AUDIO_START_AFTER_TEXT + 4), device=device, dtype=torch.long)
step_attention[:, :AUDIO_START_AFTER_TEXT] = stable_attention
mask_mapping_s1 = create_masks_for_generate(
    lm.config,
    audio_soft,
    step_attention,
    past_key_values=past_key_values,
    position_ids=pos_ids_s1,
)
pad_ids = torch.full((1, 4), 0, device=device, dtype=torch.long)
frame_pad_ple = lm.get_per_layer_inputs(pad_ids, audio_soft)
projected_ple_s1 = ss_lm._project_per_layer_inputs(audio_soft, frame_pad_ple)
pos_embs_s1 = {
    lt: lm.rotary_emb(audio_soft, pos_ids_s1, lt)
    for lt in lm.unique_layer_types
}

# Run layers 0..3 PyTorch
h = audio_soft.clone()
for l in range(4):
    h = ss_lm._decoder_layer(
        lm.layers[l],
        h,
        projected_ple_s1[:, :, l, :],
        pos_embs_s1[lm.config.layer_types[l]],
        mask_mapping_s1[lm.config.layer_types[l]],
        pos_ids_s1,
        shared,
        past_key_values,
    )

# Now inspect Layer 4 in PyTorch step-by-step:
layer4 = lm.layers[4]
attn4 = layer4.self_attn
mlp4 = layer4.mlp

# Input to layer 4 is h
# In Native, run step_debug(num_layers=4) to get input to layer 4
nat_h3 = native_lm.step_debug(audio_soft, num_layers=4, apply_final_norm=False)
torch.cuda.synchronize()
print(f"Layer 4 Input: MAE={mae(h, nat_h3):.4e}, Cos={cosine_sim(h, nat_h3):.6f}")

# Compare PyTorch Layer 4 output vs Native step_debug(num_layers=5)
py_h4 = ss_lm._decoder_layer(
    layer4,
    h,
    projected_ple_s1[:, :, 4, :],
    pos_embs_s1[lm.config.layer_types[4]],
    mask_mapping_s1[lm.config.layer_types[4]],
    pos_ids_s1,
    shared,
    past_key_values,
)

native_lm.reset()
native_lm.prime_async(stable_embeds, stable_ple, stable_attention)
torch.cuda.synchronize()
nat_h4 = native_lm.step_debug(audio_soft, num_layers=5, apply_final_norm=False)
torch.cuda.synchronize()

print(f"Layer 4 Output: MAE={mae(py_h4, nat_h4):.4e}, Cos={cosine_sim(py_h4, nat_h4):.6f}, MaxDiff={max_diff(py_h4, nat_h4):.4f}")

# Now inspect inside Layer 4:
nat_norm = native_lm.copy_debug_tensor("norm_buf", (1, 4, 1536))
nat_q = native_lm.copy_debug_tensor("q_buf", (1, 4, 8, 512))
nat_sdpa = native_lm.copy_debug_tensor("attn_out", (1, 4, 8 * 512))

# PyTorch Q for Layer 4:
norm_py = ss_lm._padded_rmsnorm(layer4.input_layernorm, h)
q_py = ss_lm._padded_rmsnorm(attn4.q_norm, F.linear(norm_py, attn4.q_proj.weight).view(1, 4, -1, 512))
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
cos4, sin4 = pos_embs_s1["full_attention"]
q_rot_py = apply_rotary_pos_emb(q_py, cos4, sin4, unsqueeze_dim=2)

print(f"Layer 4 Q RoPE: MAE={mae(q_rot_py.transpose(1, 2), nat_q):.4e}, Cos={cosine_sim(q_rot_py.transpose(1, 2), nat_q):.6f}")

# PyTorch SDPA for Layer 4:
k_cache_l4 = past_key_values.layers[4].keys
v_cache_l4 = past_key_values.layers[4].values
k_nat_l4, v_nat_l4 = native_lm.get_cached_kv(4)
print(f"Layer 4 K Cache (38 tokens): MAE={mae(k_cache_l4, k_nat_l4):.4e}, Cos={cosine_sim(k_cache_l4, k_nat_l4):.6f}")
print(f"Layer 4 V Cache (38 tokens): MAE={mae(v_cache_l4, v_nat_l4):.4e}, Cos={cosine_sim(v_cache_l4, v_nat_l4):.6f}")

# SDPA calculation
# In PyTorch:
scores_py = torch.matmul(q_rot_py.transpose(1, 2).float(), k_cache_l4.transpose(-1, -2).float()) * attn4.scaling
causal_mask = torch.tril(torch.ones(4, 38, device=device, dtype=torch.bool)) # positions 34..37 vs 0..37
# wait! query positions are 34, 35, 36, 37!
# key positions are 0..37!
# Query 34 attends to 0..34 (35 keys)!
# Query 35 attends to 0..35 (36 keys)!
# Query 36 attends to 0..36 (37 keys)!
# Query 37 attends to 0..37 (38 keys)!
mask_py = torch.zeros(1, 1, 4, 38, device=device, dtype=torch.bool)
for qi in range(4):
    mask_py[0, 0, qi, :(34 + qi + 1)] = True
scores_py = scores_py.masked_fill(~mask_py, -1e20)
weights_py = F.softmax(scores_py, dim=-1).to(dtype)
sdpa_py = torch.matmul(weights_py, v_cache_l4).transpose(1, 2).reshape(1, 4, -1)

print(f"Layer 4 SDPA: MAE={mae(sdpa_py, nat_sdpa):.4e}, Cos={cosine_sim(sdpa_py, nat_sdpa):.6f}")
