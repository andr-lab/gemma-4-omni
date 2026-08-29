#!/usr/bin/env python3
import sys, types, importlib.machinery
try:
    import torchaudio
except Exception:
    _m = types.ModuleType('torchaudio'); _m.__spec__ = importlib.machinery.ModuleSpec('torchaudio', None)
    sys.modules['torchaudio'] = _m
    sys.modules['torchaudio._extension'] = types.ModuleType('torchaudio._extension')
    sys.modules['torchaudio.compliance'] = types.ModuleType('torchaudio.compliance')
    sys.modules['torchaudio.compliance.kaldi'] = types.ModuleType('torchaudio.compliance.kaldi')
sys.modules['sox'] = types.ModuleType('sox')

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/qwen3tts_src')
import os, glob, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers.modeling_rope_utils as _ropeu
def _default_rope_init(config, device=None, seq_len=None, **kwargs):
    base = getattr(config, "rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    _head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    _dim = int(_head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, _dim, 2, dtype=torch.int64).float().to(device) / _dim))
    return inv_freq, 1.0
_ropeu.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
_ropeu.ROPE_INIT_FUNCTIONS["interleaved"] = _default_rope_init

import qwen_tts.core.models as _M
import safetensors.torch as st

snap = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*'))[-1]
_cfg = _M.Qwen3TTSConfig.from_pretrained(snap)
_tc = _cfg.talker_config
if not hasattr(_tc, 'pad_token_id'):
    object.__setattr__(_tc, 'pad_token_id', None)

teacher = _M.Qwen3TTSForConditionalGeneration(_cfg).to(torch.bfloat16)
_sd = st.load_file(snap + '/model.safetensors', device='cpu')
teacher.load_state_dict(_sd, strict=False)
del _sd; torch.cuda.empty_cache()
teacher = teacher.to(device='cuda', dtype=torch.bfloat16).eval()
talker = teacher.talker

L = 10
torch.manual_seed(0)
x_prefill = torch.randn(1, L, 2048, device='cuda', dtype=torch.bfloat16)
x_step1 = torch.randn(1, 1, 2048, device='cuda', dtype=torch.bfloat16)
x_full = torch.cat([x_prefill, x_step1], dim=1) # Length L+1

layer0 = talker.model.layers[0]

# --- 1. SEQUENTIAL ---
out_pref = talker.model(inputs_embeds=x_prefill, use_cache=True)
kv = out_pref.past_key_values

# Step 1
out_s1 = talker.model(inputs_embeds=x_step1, use_cache=True, past_key_values=kv)
h_seq_step1 = out_s1.last_hidden_state

# --- 2. PARALLEL ---
pos_3d = torch.arange(L+1, device='cuda').view(1, 1, -1).expand(3, 1, -1)
out_par = talker.model(inputs_embeds=x_full, position_ids=pos_3d, use_cache=False)
h_par_step1 = out_par.last_hidden_state[:, L:L+1, :]

print(f"=== LAYER 0 TO 27 ISOLATION ===")
print(f"h_seq_step1 shape: {h_seq_step1.shape} | h_par_step1 shape: {h_par_step1.shape}")
print(f"Max diff: {(h_seq_step1 - h_par_step1).abs().max().item():.6f}")
print(f"Cosine  : {F.cosine_similarity(h_seq_step1.float(), h_par_step1.float(), dim=-1).item():.6f}")
