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

# 1. Native prime 2 layers:
native_lm.reset()
native_lm.prime_debug(
    stable_embeds, stable_ple, stable_attention, num_layers=2, apply_final_norm=False
)
torch.cuda.synchronize()

# 2. PyTorch ShapeStable prime 2 layers:
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

py_l0_prefill = ss_lm._decoder_layer(
    lm.layers[0],
    stable_embeds,
    projected_ple_prefill[:, :, 0, :],
    pos_embs_prefill[lm.config.layer_types[0]],
    mask_mapping_prefill[lm.config.layer_types[0]],
    pos_ids_prefill,
    shared,
    past_key_values,
)

py_l1_prefill = ss_lm._decoder_layer(
    lm.layers[1],
    py_l0_prefill,
    projected_ple_prefill[:, :, 1, :],
    pos_embs_prefill[lm.config.layer_types[1]],
    mask_mapping_prefill[lm.config.layer_types[1]],
    pos_ids_prefill,
    shared,
    past_key_values,
)

print(f"Prefill L0 cache len: {past_key_values.get_seq_length(0)}, L1 cache len: {past_key_values.get_seq_length(1)}")
print(f"Native current_position: {native_lm.current_position}")

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

# PyTorch Step 1 Layer 0
py_s1_l0 = ss_lm._decoder_layer(
    lm.layers[0],
    audio_soft,
    projected_ple_s1[:, :, 0, :],
    pos_embs_s1[lm.config.layer_types[0]],
    mask_mapping_s1[lm.config.layer_types[0]],
    pos_ids_s1,
    shared,
    past_key_values,
)

# PyTorch Step 1 Layer 1
py_s1_l1 = ss_lm._decoder_layer(
    lm.layers[1],
    py_s1_l0,
    projected_ple_s1[:, :, 1, :],
    pos_embs_s1[lm.config.layer_types[1]],
    mask_mapping_s1[lm.config.layer_types[1]],
    pos_ids_s1,
    shared,
    past_key_values,
)

# Native Step 1 Layer 0
nat_s1_l0 = native_lm.step_debug(audio_soft, num_layers=1, apply_final_norm=False)
torch.cuda.synchronize()
print(f"Step 1 Layer 0: MAE = {mae(py_s1_l0, nat_s1_l0):.4e}, Cosine = {cosine_sim(py_s1_l0, nat_s1_l0):.6f}")

# Reset Native and Step 1 Layer 1 (2 layers)
native_lm.reset()
native_lm.prime_debug(
    stable_embeds, stable_ple, stable_attention, num_layers=2, apply_final_norm=False
)
torch.cuda.synchronize()

nat_s1_l1 = native_lm.step_debug(audio_soft, num_layers=2, apply_final_norm=False)
torch.cuda.synchronize()
print(f"Step 1 Layer 1: MAE = {mae(py_s1_l1, nat_s1_l1):.4e}, Cosine = {cosine_sim(py_s1_l1, nat_s1_l1):.6f}")
