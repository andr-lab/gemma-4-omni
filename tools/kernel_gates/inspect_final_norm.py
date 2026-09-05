#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import torch
import numpy as np
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import NativeGemmaListenerLM

th = load_thinker("/workspace")
lm = th.model.model.language_model

import inspect


print("type(lm):", type(lm))
norm = getattr(lm, "norm", None)
print("lm.norm:", norm)
if norm is not None:
    print("norm.weight shape:", getattr(norm, "weight", None).shape if hasattr(norm, "weight") else None)
    print("norm.weight dtype:", getattr(norm, "weight", None).dtype if hasattr(norm, "weight") else None)
    print("norm.eps:", getattr(norm, "eps", None))
    print("norm class:", type(norm))
    # Test norm on a dummy tensor
    x = torch.randn(1, 4, 1536, dtype=torch.bfloat16, device="cuda:0")
    y_mod = norm(x)

    # Check manual formula
    final_mean_sq = x.float().pow(2).mean(-1, keepdim=True) + norm.eps
    y_manual = (x.float() * torch.pow(final_mean_sq, -0.5) * norm.weight.float()).to(torch.bfloat16)
native_lm = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)

from test_progressive_layers import run_pytorch_reference_layers, cosine_sim, mae, max_diff
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM

from duplex_kernel.listener import GemmaAudioTowerBackend, GemmaListenerState

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
print("\n--- Testing Primed Full Listener Pipeline ---")
backend = GemmaAudioTowerBackend(th, device=str(device))
state_py = GemmaListenerState()

backend.prime_steady_state(state_py)
native_lm.reset()

from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT

input_ids, attention_mask = backend._build_multimodal_inputs((), 1024, device=device)
stable_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
stable_attention = attention_mask[:, :AUDIO_START_AFTER_TEXT].to(dtype=torch.int64).contiguous()
with torch.inference_mode():
    stable_embeds = lm.embed_tokens(stable_ids)
    stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)

native_lm.prime_async(stable_embeds, stable_ple, stable_attention)
torch.cuda.synchronize()

print(f"Native position after prime: {native_lm.current_position}")
print(f"PyTorch cache len after prime: {state_py.gemma_lm_cache.get_seq_length()}")

# Test with realistic soft audio embeds
rng = np.random.default_rng(2026)
audio_soft = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype)
print(f"audio_soft shape: {audio_soft.shape}")

# 1. PyTorch ShapeStable step
step_attention = torch.ones((1, AUDIO_START_AFTER_TEXT + 4), device=device, dtype=torch.long)
step_attention[:, :AUDIO_START_AFTER_TEXT] = stable_attention
py_out = backend._forward_incremental_lm(
    embeds=audio_soft,
    per_layer=backend._frame_pad_per_layer_inputs(lm),
    attention=step_attention,
    positions=torch.arange(AUDIO_START_AFTER_TEXT, AUDIO_START_AFTER_TEXT + 4, device=device, dtype=torch.long),
    state=state_py,
)

# 2. Native C step
nat_out = native_lm.step_async(audio_soft)
torch.cuda.synchronize()

print(f"Primed Step 1 - PyTorch: {py_out.shape}, Native: {nat_out.shape}")
print(f"MAE: {mae(py_out, nat_out):.4e}")
print(f"Cosine: {cosine_sim(py_out, nat_out):.6f}")
print(f"Max Diff: {max_diff(py_out, nat_out):.4f}")
