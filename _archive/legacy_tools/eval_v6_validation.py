#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, glob, json, time
import numpy as np

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

import torch
import torch.nn as nn
import soundfile as sf
from faster_whisper import WhisperModel
import kaggle_pipeline as KP
import qwen_subtalker as QS

# ---- shim compat transformers 5.x (ADR-011) ----
import transformers.modeling_rope_utils as _ropeu
def _default_rope_init(config, device=None, seq_len=None, **kwargs):
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    _head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    _dim = int(_head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, _dim, 2, dtype=torch.int64).float().to(device) / _dim))
    return inv_freq, 1.0
_ropeu.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
_ropeu.ROPE_INIT_FUNCTIONS["interleaved"] = _default_rope_init

from transformers import AutoTokenizer
import qwen_tts.core.models as _M
from qwen_tts.core.models import Qwen3TTSProcessor
import safetensors.torch as st
class TextAnchoredProjection(nn.Module):
    def __init__(self, in_dim=1536, out_dim=2048, hidden_dim=3072):
        super().__init__()
        self.audio_skip = nn.Linear(in_dim, out_dim)
        self.audio_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.audio_norm = nn.LayerNorm(out_dim)
        self.text_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.text_norm = nn.LayerNorm(out_dim)
        nn.init.zeros_(self.text_mlp[2].weight)
        nn.init.zeros_(self.text_mlp[2].bias)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, h_raw, h_tx):
        if h_tx.dim() == 2:
            h_tx = h_tx.unsqueeze(1)
        h_a = self.audio_norm(self.audio_mlp(h_raw) + self.audio_skip(h_raw))
        h_t = self.text_norm(self.text_mlp(h_tx))
        return self.out_norm(h_a + h_t)

print("[V6 EVAL] Caricamento modelli e dati...", flush=True)
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
tok = AutoTokenizer.from_pretrained(snap, use_fast=True)
proc = Qwen3TTSProcessor(tokenizer=tok)
CFG = teacher.config
TCFG = CFG.talker_config
CBOS = TCFG.codec_bos_id
sub, _ = QS.load_qwen_subtalker('/workspace/qwen_subtalker', device='cuda', dtype=torch.bfloat16)
decoder = KP.Code2WavDecoder()
asr = WhisperModel('small', device='cuda', compute_type='float16')

# Checkpoint
ckpt_path = '/workspace/checkpoints/proj_v6_final.pt'
ck_data = torch.load('/workspace/cache_ars_full_v2.pt', map_location='cpu', weights_only=False, mmap=True)
user_mel, agent_mel, agent_codes, tbw = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')

proj = TextAnchoredProjection().to('cuda', dtype=torch.bfloat16).eval()
if os.path.exists(ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu')
    proj.load_state_dict(sd['proj'])
    print(f"✅ Loaded checkpoint {ckpt_path} (step {sd.get('step', '?')})")

val_start = int(0.90 * agent_codes.shape[1])
seg_ends = json.load(open('/workspace/data/train_tokens_cv26_it_40.8h_31916s_hyperparams.json')).get('SEGMENT_ENDS', [])
for e in seg_ends:
    if e >= val_start:
        val_start = e
        break

def build_prefill(text):
    assistant = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = proc(text=assistant, return_tensors='pt')['input_ids']
    return text_ids[:, :-5].to('cuda')

tdecay = [round(0.8 * (0.75 ** (g - 1)), 6) for g in range(1, 16)]
os.makedirs('/workspace/samples_v6', exist_ok=True)

print("=== INIZIO EVAL SU 5 FINESTRE DI VALIDAZIONE ===")
with torch.inference_mode():
    for i in range(5):
        c0 = val_start + i * 64
        codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
        codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1) # (64, 16)
        n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
        if n_real < 16:
            continue
        codes = codes[:n_real]
        T = n_real
        
        h_raw = torch.from_numpy(np.asarray(ck_data['va_raw'][i], dtype=np.float32)[:n_real]).to('cuda')
        h_tx = torch.from_numpy(np.asarray(ck_data['va_htx'][i], dtype=np.float32)).to('cuda')
        
        # Ground Truth Audio decode & ASR
        wav_gt = decoder.decode(codes)
        f_gt = f'/workspace/samples_v6/val_{i:02d}_gt.wav'
        sf.write(f_gt, wav_gt, 24000)
        segs_gt, _ = asr.transcribe(f_gt, language='it', beam_size=5)
        txt_gt = ' '.join(s.text.strip() for s in segs_gt)
        
        # Model generation with TextAnchoredProjection
        h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))
        
        tids = build_prefill(' ')
        text_embed = talker.text_projection(talker.get_text_embeddings()(tids))
        tts_eos_embed = talker.text_projection(talker.get_text_embeddings()(
            torch.tensor([[CFG.tts_eos_token_id]], device='cuda', dtype=tids.dtype)))
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)
        codec_pad_embed = talker.get_input_embeddings()(
            torch.tensor([[TCFG.codec_pad_id]], device='cuda', dtype=tids.dtype))
        tts_pad_embed = talker.text_projection(talker.get_text_embeddings()(
            torch.tensor([[CFG.tts_pad_token_id]], device='cuda', dtype=tids.dtype)))
        icl = text_embed + codec_pad_embed.expand(text_embed.shape[0], text_embed.shape[1], -1)
        cbos_emb = talker.get_input_embeddings()(torch.tensor([[CBOS]], device='cuda', dtype=tids.dtype))
        icl = torch.cat([icl, cbos_emb + tts_pad_embed], dim=1)
        
        attn_pre = torch.ones(1, icl.shape[1], dtype=torch.long, device='cuda')
        attn1 = torch.ones(1, 1, dtype=torch.long, device='cuda')
        
        out = talker.model(inputs_embeds=icl, attention_mask=attn_pre, use_cache=True)
        past_kv = out.past_key_values
        past_hidden = out.last_hidden_state[:, -1:, :]
        
        resid_all, cb0_all = [], []
        cb0 = codes[:, 0]
        resid_t = sub.sample_acoustic_codes(past_hidden.to(torch.bfloat16),
                                            cb0[0:1].reshape(1, 1).to('cuda'),
                                            top_k=50, top_p=1.0, temperatures_per_level=tdecay)
        resid_all.append(resid_t[0, 0].cpu().long()); cb0_all.append(cb0[0].item())
        
        for t in range(1, T):
            agg = torch.zeros(1, 1, 2048, dtype=torch.bfloat16, device='cuda')
            cp = codes[t-1].to('cuda')
            for g in range(16):
                if g == 0:
                    e = talker.get_input_embeddings()(cp[g:g+1].reshape(1, 1))
                else:
                    e = talker.code_predictor.get_input_embeddings()[g-1](cp[g:g+1].reshape(1, 1))
                agg = agg + e
            
            inputs = agg + h_cond[:, t:t+1, :]
            out = talker.model(inputs_embeds=inputs, attention_mask=attn1, use_cache=True,
                               past_key_values=past_kv)
            past_kv = out.past_key_values
            past_hidden = out.last_hidden_state[:, -1:, :]
            resid_t = sub.sample_acoustic_codes(past_hidden.to(torch.bfloat16),
                                                cb0[t:t+1].reshape(1, 1).to('cuda'),
                                                top_k=50, top_p=1.0, temperatures_per_level=tdecay)
            resid_all.append(resid_t[0, 0].cpu().long()); cb0_all.append(cb0[t].item())
            
        R = torch.stack(resid_all, dim=0)
        C = torch.tensor(cb0_all).unsqueeze(-1)
        full = torch.cat([C, R], dim=-1)
        
        wav_gen = decoder.decode(full)
        f_gen = f'/workspace/samples_v6/val_{i:02d}_gen.wav'
        sf.write(f_gen, wav_gen, 24000)
        segs_gen, _ = asr.transcribe(f_gen, language='it', beam_size=5)
        txt_gen = ' '.join(s.text.strip() for s in segs_gen)
        
        rms_gt = float(np.sqrt(np.mean(np.square(wav_gt))))
        rms_gen = float(np.sqrt(np.mean(np.square(wav_gen))))
        print(f"\n--- Val Window {i} ({n_real} frames, {n_real*0.08:.2f}s) ---")
        print(f"  GT  ASR (RMS={rms_gt:.4f}): {repr(txt_gt)}")
        print(f"  GEN ASR (RMS={rms_gen:.4f}): {repr(txt_gen)}")
