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

state_py = GemmaListenerState()
backend.prime_steady_state(state_py)
native_lm.reset()

input_ids, attention_mask = backend._build_multimodal_inputs((), 1024, device=device)
stable_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
stable_attention = attention_mask[:, :AUDIO_START_AFTER_TEXT].to(dtype=torch.int64).contiguous()
with torch.inference_mode():
    stable_embeds = lm.embed_tokens(stable_ids)
    stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)

native_lm.prime_async(stable_embeds, stable_ple, stable_attention)
torch.cuda.synchronize()

print("\n=== PREFILL CACHE PARITY CHECKS ===")
for l in [0, 1, 2, 3, 4]:
    k_py = state_py.gemma_lm_cache.layers[l].keys
    v_py = state_py.gemma_lm_cache.layers[l].values
    k_nat, v_nat = native_lm.get_cached_kv(l)
    print(f"Layer {l} Prefill K: MAE = {mae(k_py, k_nat):.4e}, Cosine = {cosine_sim(k_py, k_nat):.6f}")
    print(f"Layer {l} Prefill V: MAE = {mae(v_py, v_nat):.4e}, Cosine = {cosine_sim(v_py, v_nat):.6f}")

rng = np.random.default_rng(2026)
audio_soft = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype)

pos_ids_step1 = torch.arange(AUDIO_START_AFTER_TEXT, AUDIO_START_AFTER_TEXT + 4, device=device, dtype=torch.long).unsqueeze(0)
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate
step_attention = torch.ones((1, AUDIO_START_AFTER_TEXT + 4), device=device, dtype=torch.long)
step_attention[:, :AUDIO_START_AFTER_TEXT] = stable_attention
mask_mapping = create_masks_for_generate(
    lm.config,
    audio_soft,
    step_attention,
    past_key_values=state_py.gemma_lm_cache,
    position_ids=pos_ids_step1,
)

pad_ids = torch.full((1, 4), 0, device=device, dtype=torch.long)
frame_pad_ple = lm.get_per_layer_inputs(pad_ids, audio_soft)
projected_ple = ss_lm._project_per_layer_inputs(audio_soft, frame_pad_ple)
pos_embs = {
    lt: lm.rotary_emb(audio_soft, pos_ids_step1, lt)
    for lt in lm.unique_layer_types
}

# Run Layer 0 PyTorch
py_l0 = ss_lm._decoder_layer(
    lm.layers[0],
    audio_soft,
    projected_ple[:, :, 0, :],
    pos_embs[lm.config.layer_types[0]],
    mask_mapping[lm.config.layer_types[0]],
    pos_ids_step1,
    state_py.gemma_lm_shared_kv_states,
    state_py.gemma_lm_cache,
)

# Run Layer 1 PyTorch
py_l1 = ss_lm._decoder_layer(
    lm.layers[1],
    py_l0,
    projected_ple[:, :, 1, :],
    pos_embs[lm.config.layer_types[1]],
    mask_mapping[lm.config.layer_types[1]],
    pos_ids_step1,
    state_py.gemma_lm_shared_kv_states,
    state_py.gemma_lm_cache,
)

# Run Native 2 layers
nat_l1 = native_lm.step_debug(audio_soft, num_layers=2, apply_final_norm=False)
torch.cuda.synchronize()

print("\n=== STEP 1 - LAYER 1 CHECKS ===")
print(f"Layer 1 Output: MAE = {mae(py_l1, nat_l1):.4e}, Cosine = {cosine_sim(py_l1, nat_l1):.6f}, MaxDiff = {max_diff(py_l1, nat_l1):.4f}")
