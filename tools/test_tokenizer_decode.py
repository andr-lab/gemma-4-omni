#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, glob, torch, numpy as np

# Mock torchaudio and sox
import types, importlib.machinery
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
sys.path.append('/workspace/.extra_pkgs')

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

import qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 as _tok_v2
_tok_v2.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
_tok_v2.ROPE_INIT_FUNCTIONS["interleaved"] = _default_rope_init

from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
import kaggle_pipeline as KP

def test():
    snap = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*'))[-1]
    tok = Qwen3TTSTokenizerV2Model.from_pretrained(snap).cuda().eval()
    _, _, agent_codes, _ = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')
    sample_codes = np.array(agent_codes[:, :64], copy=True)
    c_t_trans = torch.from_numpy(sample_codes).long().cuda().transpose(0, 1).unsqueeze(0)

    with torch.no_grad():
        out = tok.decode(c_t_trans)
        print("out attributes:", dir(out))
        print("audio_values len:", len(out.audio_values))
        item0 = out.audio_values[0]
        print("item0 type:", type(item0))
        if isinstance(item0, torch.Tensor):
            print("item0 shape:", item0.shape, "min:", item0.min().item(), "max:", item0.max().item(), "NaN:", torch.isnan(item0).any().item())
        elif isinstance(item0, np.ndarray):
            print("item0 shape:", item0.shape, "min:", item0.min(), "max:", item0.max(), "NaN:", np.isnan(item0).any())
        else:
            print("item0 value:", item0)

if __name__ == '__main__':
    test()
