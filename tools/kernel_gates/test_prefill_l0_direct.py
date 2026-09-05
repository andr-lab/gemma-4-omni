#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import torch
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import NativeGemmaListenerLM
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from duplex_kernel.listener import GemmaAudioTowerBackend, GemmaListenerState
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

# 1. Native prime Layer 0 only:
native_lm.reset()
native_lm.prime_debug(
    stable_embeds, stable_ple, stable_attention, num_layers=1, apply_final_norm=False
)
torch.cuda.synchronize()
nat_h0 = native_lm.copy_debug_tensor("hidden_a", (1, AUDIO_START_AFTER_TEXT, 1536))

# 2. PyTorch ShapeStable Layer 0:
pos_ids = torch.arange(0, AUDIO_START_AFTER_TEXT, device=device, dtype=torch.long).unsqueeze(0)
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate
mask_mapping = create_masks_for_generate(
    lm.config,
    stable_embeds,
    stable_attention,
    past_key_values=None,
    position_ids=pos_ids,
)

from collections import UserDict
from transformers.cache_utils import DynamicCache
shared = UserDict()
past_key_values = DynamicCache(config=lm.config)
projected_ple = ss_lm._project_per_layer_inputs(stable_embeds, stable_ple)
pos_embs = {
    lt: lm.rotary_emb(stable_embeds, pos_ids, lt)
    for lt in lm.unique_layer_types
}

py_l0 = ss_lm._decoder_layer(
    lm.layers[0],
    stable_embeds,
    projected_ple[:, :, 0, :],
    pos_embs[lm.config.layer_types[0]],
    mask_mapping[lm.config.layer_types[0]],
    pos_ids,
    shared,
    past_key_values,
)

print(f"Direct Compare Layer 0 Prefill Hidden State:")
print(f"MAE = {mae(py_l0, nat_h0):.4e}, Cosine = {cosine_sim(py_l0, nat_h0):.6f}, MaxDiff = {max_diff(py_l0, nat_h0):.4f}")

# Check what mask_mapping[sliding_attention] is:
mask_val = mask_mapping[lm.config.layer_types[0]]
print(f"mask_mapping type: {type(mask_val)}")
if isinstance(mask_val, torch.Tensor):
    print(f"mask shape: {mask_val.shape}, dtype: {mask_val.dtype}, min: {mask_val.min().item()}, max: {mask_val.max().item()}")
