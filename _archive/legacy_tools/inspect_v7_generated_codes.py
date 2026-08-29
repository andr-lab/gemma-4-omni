#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ispezione dettagliata dei codici RVQ generati (CB0 e residui) per ciascun campione
"""
import os, sys, glob, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

from transformers import AutoTokenizer
import qwen_tts.core.models as _M
from qwen_tts.core.models import Qwen3TTSProcessor
from tools.test_v7_audio_synthesis import TextAnchoredProjectionV7
import safetensors.torch as st
import kaggle_pipeline as KP
import qwen_subtalker as QS

def inspect():
    snap = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*'))[-1]
    _cfg = _M.Qwen3TTSConfig.from_pretrained(snap)
    if not hasattr(_cfg.talker_config, 'pad_token_id'):
        object.__setattr__(_cfg.talker_config, 'pad_token_id', None)
    teacher = _M.Qwen3TTSForConditionalGeneration(_cfg).to(torch.bfloat16)
    teacher.load_state_dict(st.load_file(snap + '/model.safetensors', device='cpu'), strict=False)
    teacher = teacher.to(device='cuda', dtype=torch.bfloat16).eval()
    talker = teacher.talker
    tok_qwen = AutoTokenizer.from_pretrained(snap, use_fast=True)
    proc_qwen = Qwen3TTSProcessor(tokenizer=tok_qwen)

    gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    tok_gemma = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True) if len(gemma_dirs) > 0 else None

    sub, _ = QS.load_qwen_subtalker('/workspace/qwen_subtalker', device='cuda', dtype=torch.bfloat16)

    ckpt_path = '/workspace/checkpoints/proj_v7_final.pt'
    proj = TextAnchoredProjectionV7().to(device='cuda', dtype=torch.bfloat16)
    sd = torch.load(ckpt_path, map_location='cpu')
    proj.load_state_dict(sd.get('proj', sd), strict=False)
    proj.eval()

    _, _, agent_codes, tbw = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')
    ck = torch.load('/workspace/cache_ars_full_v2.pt', map_location='cpu', weights_only=False, mmap=True)
    va_raw = ck['va_raw']
    va_htx = ck['va_htx']
    val_start = int(0.90 * agent_codes.shape[1])
    seg_ends = json.load(open('/workspace/data/train_tokens_cv26_it_40.8h_31916s_hyperparams.json')).get('SEGMENT_ENDS', [])
    for e in seg_ends:
        if e >= val_start: val_start = e; break

    CFG = teacher.config
    TCFG = CFG.talker_config
    CBOS = TCFG.codec_bos_id

    for i in range(3):
        print(f"\n==================== SAMPLE {i} ====================")
        c0 = val_start + i * 64
        codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
        codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
        n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
        codes = codes[:n_real]

        gi = (val_start // 64) + i
        t_item = tbw[gi] if gi < len(tbw) else None
        text_str = tok_gemma.decode(t_item, skip_special_tokens=True).strip() if (t_item and isinstance(t_item, list)) else 'Buongiorno.'
        print(f"Ground Truth: \"{text_str}\"")
        print(f"GT CB0 (primi 30 frame): {codes[:30, 0].tolist()}")

        h_raw = torch.from_numpy(np.asarray(va_raw[i][:n_real], dtype=np.float32)).to('cuda')
        h_tx = torch.from_numpy(np.asarray(va_htx[i], dtype=np.float32)).to('cuda')

        with torch.no_grad():
            h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))

            assistant = f"<|im_start|>assistant\n{text_str}<|im_end|>\n<|im_start|>assistant\n"
            tids = proc_qwen(text=assistant, return_tensors='pt')['input_ids'][:, :-5].to('cuda')
            text_emb = talker.text_projection(talker.get_text_embeddings()(tids))
            tts_eos = talker.text_projection(talker.get_text_embeddings()(torch.tensor([[CFG.tts_eos_token_id]], device='cuda', dtype=tids.dtype)))
            c_pad = talker.get_input_embeddings()(torch.tensor([[TCFG.codec_pad_id]], device='cuda', dtype=tids.dtype))
            t_pad = talker.text_projection(talker.get_text_embeddings()(torch.tensor([[CFG.tts_pad_token_id]], device='cuda', dtype=tids.dtype)))
            cbos_emb = talker.get_input_embeddings()(torch.tensor([[CBOS]], device='cuda', dtype=tids.dtype))
            icl = torch.cat([text_emb, tts_eos], dim=1) + c_pad.expand(1, text_emb.shape[1]+1, -1)
            icl = torch.cat([icl, cbos_emb + t_pad], dim=1)

            out = talker.model(inputs_embeds=icl, attention_mask=torch.ones(1, icl.shape[1], device='cuda'), use_cache=True)
            kv = out.past_key_values
            past_h = out.last_hidden_state[:, -1:, :]

            cb0_preds = []
            cb0_pred = talker.codec_head(past_h).argmax(dim=-1)
            cb0_preds.append(cb0_pred[0, 0].item())

            # Autoregressive generation vs Teacher-forced generation
            # Let's test what happens when feeding GENERATED codes vs feeding GT codes!
            for t in range(1, n_real):
                # Mode A: Autoregressive with predicted code
                cp = codes[t-1].to('cuda') # Note: in our eval loop we fed codes[t-1] (Teacher Forcing!)
                agg = sum(talker.get_input_embeddings()(cp[g:g+1].reshape(1,1)) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](cp[g:g+1].reshape(1,1)) for g in range(16))
                in_t = agg + h_cond[:, t:t+1, :]
                out = talker.model(inputs_embeds=in_t, attention_mask=torch.ones(1, 1, device='cuda'), use_cache=True, past_key_values=kv)
                kv = out.past_key_values
                past_h = out.last_hidden_state[:, -1:, :]
                cb0_pred = talker.codec_head(past_h).argmax(dim=-1)
                cb0_preds.append(cb0_pred[0, 0].item())

            print(f"Pred CB0 (primi 30 frame): {cb0_preds[:30]}")
            n_unique_pred = len(set(cb0_preds))
            print(f"Unique CB0 in 64 frames: {n_unique_pred} / {len(cb0_preds)}")
            
            # Check repetition / mode collapse
            diffs = np.diff(cb0_preds)
            n_repeats = (diffs == 0).sum()
            print(f"Repeated consecutive tokens: {n_repeats} / {len(cb0_preds)-1} ({n_repeats/(len(cb0_preds)-1)*100:.1f}%)")

if __name__ == '__main__':
    inspect()
