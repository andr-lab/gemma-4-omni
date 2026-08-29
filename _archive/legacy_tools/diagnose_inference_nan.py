#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostica approfondita della pipeline di inferenza per isolare l'origine dei NaN.
"""
import os, sys, glob, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mock torchaudio and sox for venv_phase1 compatibility
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

import kaggle_pipeline as KP
import qwen_subtalker as QS

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
import soundfile as sf

class TextAnchoredProjectionV7(nn.Module):
    def __init__(self, in_dim=1536, out_dim=2048, hidden_dim=3072, initial_gamma=0.25):
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
        self.out_norm = nn.LayerNorm(out_dim)
        self.gamma = nn.Parameter(torch.tensor(initial_gamma, dtype=torch.float32))

    def forward(self, h_raw, h_tx):
        if h_tx.dim() == 2:
            h_tx = h_tx.unsqueeze(1)
        h_a = self.audio_norm(self.audio_mlp(h_raw) + self.audio_skip(h_raw))
        h_t = self.text_norm(self.text_mlp(h_tx))
        return (self.gamma.to(h_a.dtype)) * self.out_norm(h_a + h_t)

def diagnose():
    print("=" * 70, flush=True)
    print("=== DIAGNOSTICA STEP-BY-STEP INFERENZA & AUDIO SYNTHESIS ===", flush=True)
    print("=" * 70, flush=True)

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
    tok_qwen = AutoTokenizer.from_pretrained(snap, use_fast=True)
    proc_qwen = Qwen3TTSProcessor(tokenizer=tok_qwen)

    gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    tok_gemma = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True) if len(gemma_dirs) > 0 else None

    sub, _ = QS.load_qwen_subtalker('/workspace/qwen_subtalker', device='cuda', dtype=torch.bfloat16)
    decoder = KP.Code2WavDecoder()

    # Carica Adapter V7 finale
    ckpt_path = '/workspace/checkpoints/proj_v7_final.pt'
    proj = TextAnchoredProjectionV7().to(device='cuda', dtype=torch.bfloat16)
    sd = torch.load(ckpt_path, map_location='cpu')
    proj.load_state_dict(sd.get('proj', sd), strict=False)
    proj.eval()
    print(f"Modello caricato da {ckpt_path} (gamma={proj.gamma.item():.4f})", flush=True)

    # Dati di validazione
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

    # Test su 2 campioni
    for i in range(2):
        print(f"\n--- [TEST SAMPLE {i}] ---", flush=True)
        c0 = val_start + i * 64
        codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
        codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
        n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
        codes = codes[:n_real]

        gi = (val_start // 64) + i
        t_item = tbw[gi] if gi < len(tbw) else None
        text_str = tok_gemma.decode(t_item, skip_special_tokens=True).strip() if (t_item and isinstance(t_item, list)) else 'Buongiorno.'
        print(f"  Testo: \"{text_str}\"", flush=True)

        h_raw = torch.from_numpy(np.asarray(va_raw[i][:n_real], dtype=np.float32)).to('cuda')
        h_tx = torch.from_numpy(np.asarray(va_htx[i], dtype=np.float32)).to('cuda')

        print(f"  h_raw: shape={h_raw.shape}, NaN={torch.isnan(h_raw).any().item()}, Norm={h_raw.norm(dim=-1).mean().item():.2f}", flush=True)
        print(f"  h_tx : shape={h_tx.shape}, NaN={torch.isnan(h_tx).any().item()}, Norm={h_tx.norm(dim=-1).mean().item():.2f}", flush=True)

        with torch.no_grad():
            h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))
            print(f"  h_cond: shape={h_cond.shape}, NaN={torch.isnan(h_cond).any().item()}, Norm={h_cond.norm(dim=-1).mean().item():.2f}", flush=True)

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
            print(f"  prefill past_h: NaN={torch.isnan(past_h).any().item()}, Norm={past_h.norm(dim=-1).mean().item():.2f}", flush=True)

            tdecay = [0.95 ** j for j in range(15)]
            resid_all, cb0_all = [], []

            cb0_pred = talker.codec_head(past_h).argmax(dim=-1)
            print(f"  frame 0 cb0_pred: {cb0_pred.tolist()}, logits NaN={torch.isnan(talker.codec_head(past_h)).any().item()}", flush=True)

            resid_t = sub.sample_acoustic_codes(past_h.to(torch.bfloat16), cb0_pred.reshape(1, 1).to('cuda'), top_k=50, top_p=1.0, temperatures_per_level=tdecay)
            print(f"  frame 0 resid_t: shape={resid_t.shape}, min={resid_t.min().item()}, max={resid_t.max().item()}, NaN={torch.isnan(resid_t.float()).any().item()}", flush=True)

            resid_all.append(resid_t[0, 0].cpu().long())
            cb0_all.append(cb0_pred[0, 0].item())

            # Loop sui frame successivi
            has_nan_loop = False
            for t in range(1, n_real):
                cp = codes[t-1].to('cuda')
                agg = sum(talker.get_input_embeddings()(cp[g:g+1].reshape(1,1)) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](cp[g:g+1].reshape(1,1)) for g in range(16))
                in_t = agg + h_cond[:, t:t+1, :]
                out = talker.model(inputs_embeds=in_t, attention_mask=torch.ones(1, 1, device='cuda'), use_cache=True, past_key_values=kv)
                kv = out.past_key_values
                past_h = out.last_hidden_state[:, -1:, :]
                if torch.isnan(past_h).any().item():
                    print(f"  🚨 NaN rilevato a frame {t} in past_h!", flush=True)
                    has_nan_loop = True
                    break
                cb0_pred = talker.codec_head(past_h).argmax(dim=-1)
                resid_t = sub.sample_acoustic_codes(past_h.to(torch.bfloat16), cb0_pred.reshape(1, 1).to('cuda'), top_k=50, top_p=1.0, temperatures_per_level=tdecay)
                resid_all.append(resid_t[0, 0].cpu().long())
                cb0_all.append(cb0_pred[0, 0].item())

            if not has_nan_loop:
                print(f"  Tutti i {n_real} frame completati SENZA NaN in past_h.", flush=True)

            R = torch.stack(resid_all, dim=0) # (T, 15)
            C = torch.tensor(cb0_all).unsqueeze(-1) # (T, 1)
            full_codes = torch.cat([C, R], dim=-1) # (T, 16)
            print(f"  full_codes: shape={full_codes.shape}, min={full_codes.min().item()}, max={full_codes.max().item()}, dtype={full_codes.dtype}", flush=True)

            # Test decode su Ground Truth codes prima
            print("  Decodifica Ground Truth codes con Code2Wav...", flush=True)
            gt_wav = decoder.decode(codes.cpu())
            print(f"    * GT wav: min={np.min(gt_wav):.4f}, max={np.max(gt_wav):.4f}, NaN={np.isnan(gt_wav).any()}, RMS={np.sqrt(np.mean(gt_wav**2)):.4f}", flush=True)

            # Test decode su full_codes generati
            print("  Decodifica Generated full_codes con Code2Wav...", flush=True)
            gen_wav = decoder.decode(full_codes)
            print(f"    * Gen wav: min={np.min(gen_wav):.4f}, max={np.max(gen_wav):.4f}, NaN={np.isnan(gen_wav).any()}, RMS={np.sqrt(np.mean(gen_wav**2)):.4f}", flush=True)

if __name__ == '__main__':
    diagnose()
