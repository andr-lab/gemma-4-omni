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
stable_attention = attention_mask[:, :AUDIO_START_AFTER_TEXT].to(dtype=torch.int64).contiguous()
with torch.inference_mode():
    stable_embeds = lm.embed_tokens(stable_ids)
    stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)

# Native prime all 35 layers
native_lm.reset()
native_lm.prime_async(stable_embeds, stable_ple, stable_attention)
torch.cuda.synchronize()

# PyTorch prefill
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

# Run PyTorch Step 1 all 35 layers:
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

# Run Native step_debug without final norm:
nat_h34 = native_lm.step_debug(audio_soft, num_layers=35, apply_final_norm=False)
torch.cuda.synchronize()

print(f"Layer 34 pre-norm: MAE={mae(curr_h, nat_h34):.4e}, Cos={cosine_sim(curr_h, nat_h34):.6f}, MaxDiff={max_diff(curr_h, nat_h34):.4f}")

# Apply PyTorch norm to both:
py_norm_on_py = lm.norm(curr_h)
py_norm_on_nat = lm.norm(nat_h34)
print(f"PyTorch norm on both: MAE={mae(py_norm_on_py, py_norm_on_nat):.4e}, Cos={cosine_sim(py_norm_on_py, py_norm_on_nat):.6f}, MaxDiff={max_diff(py_norm_on_py, py_norm_on_nat):.4f}")

# Run Native with final norm:
native_lm.reset()
native_lm.prime_async(stable_embeds, stable_ple, stable_attention)
torch.cuda.synchronize()
nat_final = native_lm.step_async(audio_soft)
torch.cuda.synchronize()

print(f"PyTorch norm on Py vs Native final: MAE={mae(py_norm_on_py, nat_final):.4e}, Cos={cosine_sim(py_norm_on_py, nat_final):.6f}, MaxDiff={max_diff(py_norm_on_py, nat_final):.4f}")
print(f"PyTorch norm on Nat vs Native final: MAE={mae(py_norm_on_nat, nat_final):.4e}, Cos={cosine_sim(py_norm_on_nat, nat_final):.6f}, MaxDiff={max_diff(py_norm_on_nat, nat_final):.4f}")
