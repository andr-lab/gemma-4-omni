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

print("[1] Loading Thinker...")
th = load_thinker("/workspace", device=device, dtype=dtype)
lm = th.model.model.language_model

print("[2] Initializing Backends...")
backend = GemmaAudioTowerBackend(th, device=str(device))
native_lm = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)
ss_lm = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)

input_ids, attention_mask = backend._build_multimodal_inputs((), 1024, device=device)
stable_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
stable_attention = attention_mask[:, :AUDIO_START_AFTER_TEXT].to(device=device, dtype=torch.long).contiguous()
with torch.inference_mode():
    stable_embeds = lm.embed_tokens(stable_ids)
    stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)

print(f"Prefix tokens: {stable_ids.shape[1]}")

# 1. Run PyTorch ShapeStable prefill
state_py = GemmaListenerState()
backend.prime_steady_state(state_py)
py_cache = state_py.gemma_lm_cache
print("PyTorch cache length:", py_cache.get_seq_length())

# 2. Run Native prime
native_lm.reset()
native_lm.prime_async(stable_embeds, stable_ple, stable_attention)
torch.cuda.synchronize()
print("Native current pos:", native_lm.current_position)

# Inspect Native Cache vs PyTorch Cache for Layer 0
# We can inspect the internal buffers of native_lm using debug pointers or by inspecting the C context
print("\n--- Testing Single Audio Step ---")
rng = np.random.default_rng(2026)
audio_soft = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype)

# Run 1 layer in PyTorch vs Native
# First let's inspect what PyTorch produces for step 1
step_attention = torch.ones((1, AUDIO_START_AFTER_TEXT + 4), device=device, dtype=torch.long)
step_attention[:, :AUDIO_START_AFTER_TEXT] = stable_attention
py_step1 = backend._forward_incremental_lm(
    embeds=audio_soft,
    per_layer=backend._frame_pad_per_layer_inputs(lm),
    attention=step_attention,
    positions=torch.arange(AUDIO_START_AFTER_TEXT, AUDIO_START_AFTER_TEXT + 4, device=device, dtype=torch.long),
    state=state_py,
)

nat_step1 = native_lm.step_async(audio_soft)
torch.cuda.synchronize()

print(f"Step 1 - PyTorch: {py_step1.shape}, Native: {nat_step1.shape}")
print(f"MAE: {mae(py_step1, nat_step1):.4e}")
print(f"Cosine: {cosine_sim(py_step1, nat_step1):.6f}")
print(f"Max Diff: {max_diff(py_step1, nat_step1):.4f}")

# Now let's test WITHOUT priming: what does step 1 look like when both start at position 0?
print("\n--- Testing Unprimed Step (base_pos=0) ---")
state_py_unprimed = GemmaListenerState()
py_unprimed = backend._forward_incremental_lm(
    embeds=audio_soft,
    per_layer=backend._frame_pad_per_layer_inputs(lm),
    attention=torch.ones((1, 4), device=device, dtype=torch.long),
    positions=torch.arange(0, 4, device=device, dtype=torch.long),
    state=state_py_unprimed,
)

native_lm.reset()
nat_unprimed = native_lm.step_debug(audio_soft)
torch.cuda.synchronize()

print(f"Unprimed Step - PyTorch: {py_unprimed.shape}, Native: {nat_unprimed.shape}")
print(f"MAE: {mae(py_unprimed, nat_unprimed):.4e}")
print(f"Cosine: {cosine_sim(py_unprimed, nat_unprimed):.6f}")
print(f"Max Diff: {max_diff(py_unprimed, nat_unprimed):.4f}")
