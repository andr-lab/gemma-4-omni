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

input_ids, attention_mask = backend._build_multimodal_inputs((), 1024, device=device)
stable_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
stable_attention = attention_mask[:, :AUDIO_START_AFTER_TEXT].to(dtype=torch.int64).contiguous()
with torch.inference_mode():
    stable_embeds = lm.embed_tokens(stable_ids)
    stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)

# 1. PyTorch prefill
state_py = GemmaListenerState()
backend.prime_steady_state(state_py)
py_cache = state_py.gemma_lm_cache

# 2. Native prime
native_lm.reset()
native_lm.prime_async(stable_embeds, stable_ple, stable_attention)
torch.cuda.synchronize()

print(f"{'Layer':<6} | {'K MAE':<10} | {'K Cos':<10} | {'V MAE':<10} | {'V Cos':<10}")
print("-" * 55)

# Extract native cache ptrs for each layer
# In NativeGemmaListenerLM, let's expose or read the cache pointers
# The cache struct is at ctx->cache.
# Let's inspect layers 0, 1, 4, 13, 14
import ctypes

class LayerCache(ctypes.Structure):
    _fields_ = [
        ("k_data", ctypes.c_void_p),
        ("v_data", ctypes.c_void_p),
        ("head_dim", ctypes.c_uint32),
        ("is_shared", ctypes.c_bool),
        ("shared_src", ctypes.c_int32),
    ]

class CacheStruct(ctypes.Structure):
    _fields_ = [
        ("max_seq_len", ctypes.c_uint32),
        ("current_pos", ctypes.c_uint32),
        ("valid_kv_len", ctypes.c_uint32),
        ("layers", LayerCache * 35),
    ]

# ctx pointer is native_lm._ctx. The first member of gemma4_listener_ctx is:
# Let's check struct gemma4_listener_ctx in gemma4_listener.cu
