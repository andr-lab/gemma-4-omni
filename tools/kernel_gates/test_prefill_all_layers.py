#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import torch
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
stable_attention = attention_mask[:, :AUDIO_START_AFTER_TEXT].to(device=device, dtype=torch.long).contiguous()
with torch.inference_mode():
    stable_embeds = lm.embed_tokens(stable_ids)
    stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)

# 1. Native prime all 35 layers:
native_lm.reset()
native_lm.prime_async(stable_embeds, stable_ple, stable_attention)
torch.cuda.synchronize()

# 2. PyTorch ShapeStable prime all 35 layers:
pos_ids_prefill = torch.arange(0, AUDIO_START_AFTER_TEXT, device=device, dtype=torch.long).unsqueeze(0)
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate
mask_mapping_prefill = create_masks_for_generate(
    lm.config,
    stable_embeds,
    stable_attention,
    past_key_values=None,
    position_ids=pos_ids_prefill,
)

from collections import UserDict
from transformers.cache_utils import DynamicCache
shared = UserDict()
past_key_values = DynamicCache(config=lm.config)
projected_ple_prefill = ss_lm._project_per_layer_inputs(stable_embeds, stable_ple)
pos_embs_prefill = {
    lt: lm.rotary_emb(stable_embeds, pos_ids_prefill, lt)
    for lt in lm.unique_layer_types
}

py_hidden_prefill = stable_embeds.clone()
for l_idx in range(35):
    py_hidden_prefill = ss_lm._decoder_layer(
        lm.layers[l_idx],
        py_hidden_prefill,
        projected_ple_prefill[:, :, l_idx, :],
        pos_embs_prefill[lm.config.layer_types[l_idx]],
        mask_mapping_prefill[lm.config.layer_types[l_idx]],
        pos_ids_prefill,
        shared,
        past_key_values,
    )

print(f"Prefill done. Cache len at L0: {past_key_values.get_seq_length(0)}, Native pos: {native_lm.current_position}")

# Check cache parity for all layers:
print("\n" + "=" * 65)
print(f"{'LAYER':<10} | {'K MAE':<12} | {'K COS':<12} | {'V MAE':<12} | {'V COS':<12}")
print("=" * 65)
for l in [0, 1, 2, 3, 4, 10, 13, 14, 15, 25, 34]:
    if not lm.layers[l].self_attn.is_kv_shared_layer:
        k_py = past_key_values.layers[l].keys
        v_py = past_key_values.layers[l].values
    else:
        src = lm.layers[l].self_attn.kv_shared_layer_index
        k_py, v_py = getattr(past_key_values, "shared_layers", {})[src]
    k_nat, v_nat = native_lm.get_cached_kv(l)
    print(f"Layer {l:<4d} | {mae(k_py, k_nat):<12.4e} | {cosine_sim(k_py, k_nat):<12.6f} | {mae(v_py, v_nat):<12.4e} | {cosine_sim(v_py, v_nat):<12.6f}")

# NOW STEP 1 (Audio frame Q=4)
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

# Run PyTorch Step 1 across all 35 layers:
py_hiddens_s1 = []
curr_h = audio_soft.clone()
for l_idx in range(35):
    curr_h = ss_lm._decoder_layer(
        lm.layers[l_idx],
        curr_h,
        projected_ple_s1[:, :, l_idx, :],
        pos_embs_s1[lm.config.layer_types[l_idx]],
        mask_mapping_s1[lm.config.layer_types[l_idx]],
        pos_ids_s1,
        shared,
        past_key_values,
    )
    py_hiddens_s1.append(curr_h.clone())

# Final RMSNorm for PyTorch
py_final = ss_lm._padded_rmsnorm(lm.norm, curr_h)

# Now check Native step_debug for each layer:
print("\n" + "=" * 65)
print(f"{'STAGE':<25} | {'MAE':<12} | {'COSINE':<12} | {'MAX DIFF':<12}")
print("=" * 65)

for l in [1, 2, 3, 4, 5, 10, 14, 15, 16, 25, 35]:
    # Reset native and prime fresh
    native_lm.reset()
    native_lm.prime_async(stable_embeds, stable_ple, stable_attention)
    torch.cuda.synchronize()

    nat_out = native_lm.step_debug(audio_soft, num_layers=l, apply_final_norm=False)
    torch.cuda.synchronize()
    m = mae(py_hiddens_s1[l-1], nat_out)
    c = cosine_sim(py_hiddens_s1[l-1], nat_out)
    mx = max_diff(py_hiddens_s1[l-1], nat_out)
    print(f"Step 1 Layer 0..{l-1:<2d}       | {m:<12.4e} | {c:<12.6f} | {mx:<12.4f}")
    if l == 35:
        norm_of_native = ss_lm._padded_rmsnorm(lm.norm, nat_out)
        print(f"Step 1 Native hidden -> PyTorch norm | MAE = {mae(py_final, norm_of_native):.4e} | COS = {cosine_sim(py_final, norm_of_native):.6f} | MAX = {max_diff(py_final, norm_of_native):.4f}")

# Finally, check full step with final norm:
native_lm.reset()
native_lm.prime_async(stable_embeds, stable_ple, stable_attention)
torch.cuda.synchronize()
nat_final = native_lm.step_async(audio_soft)
torch.cuda.synchronize()
m_f = mae(py_final, nat_final)
c_f = cosine_sim(py_final, nat_final)
mx_f = max_diff(py_final, nat_final)
print(f"{'Step 1 Full (with norm)':<25} | {m_f:<12.4e} | {c_f:<12.6f} | {mx_f:<12.4f}")
