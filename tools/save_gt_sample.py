#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Salva l'audio Ground Truth del Sample 00 per confronto uditivo e spettrale
"""
import os, sys, glob, json
import numpy as np
import torch
import soundfile as sf
from faster_whisper import WhisperModel

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

def save_gt():
    snap_tok = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*'))[-1]
    tok_decoder = Qwen3TTSTokenizerV2Model.from_pretrained(snap_tok).cuda().eval()

    _, _, agent_codes, tbw = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')
    val_start = int(0.90 * agent_codes.shape[1])
    seg_ends = json.load(open('/workspace/data/train_tokens_cv26_it_40.8h_31916s_hyperparams.json')).get('SEGMENT_ENDS', [])
    for e in seg_ends:
        if e >= val_start: val_start = e; break

    codes_np = np.asarray(agent_codes[:, val_start:val_start+64], dtype=np.int64)
    codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1).unsqueeze(0) # (1, 64, 16)
    
    with torch.no_grad():
        out = tok_decoder.decode(codes)
        wav = out.audio_values[0].cpu().numpy().astype(np.float32)
        p_peak = float(np.percentile(np.abs(wav), 99.7))
        if p_peak > 1e-6:
            wav = np.clip(wav / (p_peak * 1.05), -1.0, 1.0)
        sf.write('/workspace/samples_v7/gt_sample_00.wav', wav, 24000)

    asr = WhisperModel('small', device='cuda', compute_type='float16')
    segs, _ = asr.transcribe('/workspace/samples_v7/gt_sample_00.wav', language='it')
    print("GT Decoded ASR:", ' '.join(s.text.strip() for s in segs))

if __name__ == '__main__':
    save_gt()
