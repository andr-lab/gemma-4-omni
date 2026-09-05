#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import torch
import numpy as np
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

# PyTorch ShapeStable prefill
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

# Run Layer 0 PyTorch
layer0 = lm.layers[0]
py_l0 = ss_lm._decoder_layer(
    layer0,
    stable_embeds,
    projected_ple[:, :, 0, :],
    pos_embs[lm.config.layer_types[0]],
    mask_mapping[lm.config.layer_types[0]],
    pos_ids,
    shared,
    past_key_values,
)

# Now check Native: let's run prefill layer 0 in Native
# In native_lm, let's reset and prime
native_lm.reset()
native_lm.prime_debug(
    stable_embeds,
    stable_ple,
    stable_attention,
    num_layers=1,
    apply_final_norm=False,
)
torch.cuda.synchronize()

layer_cache = past_key_values.layers[0]
print("DynamicSlidingWindowLayer attributes:", [a for a in dir(layer_cache) if not a.startswith("_")])
k_py = getattr(layer_cache, "keys", getattr(layer_cache, "key_states", getattr(layer_cache, "key", None)))
v_py = getattr(layer_cache, "values", getattr(layer_cache, "value_states", getattr(layer_cache, "value", None)))
print("PyTorch Layer 0 K cache shape:", k_py.shape if k_py is not None else None, "V cache shape:", v_py.shape if v_py is not None else None)

# Check Native cache K and V for layer 0:
k_nat, v_nat = native_lm.get_cached_kv(0)
print("Native Layer 0 K cache shape:", k_nat.shape, "V cache shape:", v_nat.shape)

print(f"Layer 0 K: MAE = {mae(k_py, k_nat):.4e}, Cosine = {cosine_sim(k_py, k_nat):.6f}, MaxDiff = {max_diff(k_py, k_nat):.4f}")
print(f"Layer 0 V: MAE = {mae(v_py, v_nat):.4e}, Cosine = {cosine_sim(v_py, v_nat):.6f}, MaxDiff = {max_diff(v_py, v_nat):.4f}")

# Check hidden state after Layer 0
# Copy native hidden_a from native executor after layer 0
nat_h0 = native_lm.copy_debug_tensor("hidden_a", (1, AUDIO_START_AFTER_TEXT, 1536))
nat_ple = native_lm.copy_debug_tensor(
    "projected_ple", (1, AUDIO_START_AFTER_TEXT, 35, 256)
)
print(f"Layer 0 Hidden Output: MAE = {mae(py_l0, nat_h0):.4e}, Cosine = {cosine_sim(py_l0, nat_h0):.6f}, MaxDiff = {max_diff(py_l0, nat_h0):.4f}")
print(f"PLE Projection: MAE = {mae(projected_ple, nat_ple):.4e}, Cosine = {cosine_sim(projected_ple, nat_ple):.6f}, MaxDiff = {max_diff(projected_ple, nat_ple):.4f}")
