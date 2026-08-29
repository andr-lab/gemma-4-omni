#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sintesi audio con Prefill Ufficiale Completo (con Language Tag 'italian' e Role Embeddings)
"""
import os, sys, glob, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from faster_whisper import WhisperModel

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
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
import safetensors.torch as st
import kaggle_pipeline as KP
import qwen_subtalker as QS
from tools.test_v7_audio_synthesis import TextAnchoredProjectionV7, sample_cb0_with_penalty

def test_synthesis():
    print("=" * 70)
    print("=== TEST SINTESI CON PREFILL UFFICIALE QWEN3-TTS (ITALIAN TAG) ===")
    print("=" * 70)

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

    snap_tok = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*'))[-1]
    tok_decoder = Qwen3TTSTokenizerV2Model.from_pretrained(snap_tok).cuda().eval()

    gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    tok_gemma = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True) if len(gemma_dirs) > 0 else None

    sub, _ = QS.load_qwen_subtalker('/workspace/qwen_subtalker', device='cuda', dtype=torch.bfloat16)

    ckpt_path = '/workspace/checkpoints/proj_v7_final.pt'
    proj = TextAnchoredProjectionV7().to(device='cuda', dtype=torch.bfloat16)
    sd = torch.load(ckpt_path, map_location='cpu')
    proj.load_state_dict(sd.get('proj', sd), strict=False)
    proj.eval()
    print(f"Modello V7 caricato da {ckpt_path} (gamma={proj.gamma.item():.4f})")

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
    lang_map = getattr(TCFG, 'codec_language_id', {})
    print("Codec languages supportati:", lang_map)
    lang_id = lang_map.get('italian', lang_map.get('auto', None))

    os.makedirs('/workspace/samples_v7', exist_ok=True)
    asr = WhisperModel('small', device='cuda', compute_type='float16')
    tdecay = [0.95 ** j for j in range(15)]

    for i in range(3):
        c0 = val_start + i * 64
        codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
        codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
        n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
        codes = codes[:n_real]

        gi = (val_start // 64) + i
        t_item = tbw[gi] if gi < len(tbw) else None
        text_str = tok_gemma.decode(t_item, skip_special_tokens=True).strip() if (t_item and isinstance(t_item, list)) else 'Buongiorno.'
        print(f"\n--- [SAMPLE {i}] ---")
        print(f"Ground Truth: \"{text_str}\"")

        h_raw = torch.from_numpy(np.asarray(va_raw[i][:n_real], dtype=np.float32)).to('cuda')
        h_tx = torch.from_numpy(np.asarray(va_htx[i], dtype=np.float32)).to('cuda')

        with torch.no_grad():
            h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))

            # COSTRUZIONE PREFILL UFFICIALE
            raw_input_ids = proc_qwen(text=text_str, return_tensors='pt')['input_ids'].to('cuda')
            
            tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
                talker.get_text_embeddings()(
                    torch.tensor([[CFG.tts_bos_token_id, CFG.tts_eos_token_id, CFG.tts_pad_token_id]], device='cuda', dtype=raw_input_ids.dtype)
                )
            ).chunk(3, dim=1)

            if lang_id is None:
                codec_prefill = [[TCFG.codec_nothink_id, TCFG.codec_think_bos_id, TCFG.codec_think_eos_id]]
            else:
                codec_prefill = [[TCFG.codec_think_id, TCFG.codec_think_bos_id, lang_id, TCFG.codec_think_eos_id]]

            c_emb_0 = talker.get_input_embeddings()(torch.tensor(codec_prefill, device='cuda', dtype=raw_input_ids.dtype))
            c_emb_1 = talker.get_input_embeddings()(torch.tensor([[TCFG.codec_pad_id, TCFG.codec_bos_id]], device='cuda', dtype=raw_input_ids.dtype))
            codec_input_emebdding = torch.cat([c_emb_0, c_emb_1], dim=1)

            # Role embedding: <|im_start|>assistant\n
            role_ids = proc_qwen(text="<|im_start|>assistant\n", return_tensors='pt')['input_ids'].to('cuda')
            role_embed = talker.text_projection(talker.get_text_embeddings()(role_ids))

            _talker_input_embed = torch.cat((tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1), tts_bos_embed), dim=1) + codec_input_emebdding[:, :-1]
            talker_input_embed = torch.cat((role_embed, _talker_input_embed), dim=1)

            # Text content + eos
            text_proj_emb = talker.text_projection(talker.get_text_embeddings()(raw_input_ids))
            c_pad_full = talker.get_input_embeddings()(torch.tensor([[TCFG.codec_pad_id] * (raw_input_ids.shape[1] + 1)], device='cuda', dtype=raw_input_ids.dtype))
            text_part = torch.cat([text_proj_emb, tts_eos_embed], dim=1) + c_pad_full
            bos_part = tts_pad_embed + talker.get_input_embeddings()(torch.tensor([[TCFG.codec_bos_id]], device='cuda', dtype=raw_input_ids.dtype))

            full_prefill = torch.cat([talker_input_embed, text_part, bos_part], dim=1)

            # Esecuzione Prefill
            out = talker.model(inputs_embeds=full_prefill, attention_mask=torch.ones(1, full_prefill.shape[1], device='cuda'), use_cache=True)
            kv = out.past_key_values
            past_h = out.last_hidden_state[:, -1:, :]

            # Loop di generazione
            cb0_all, resid_all = [], []
            lg0 = talker.codec_head(past_h)
            cb0_val = sample_cb0_with_penalty(lg0, cb0_all, temperature=0.8, top_k=50, top_p=0.92, rep_penalty=1.25)
            resid_t = sub.sample_acoustic_codes(past_h.to(torch.bfloat16), torch.tensor([[cb0_val]], device='cuda'), top_k=50, top_p=0.95, temperatures_per_level=tdecay)
            resid_all.append(resid_t[0, 0].cpu().long())
            cb0_all.append(cb0_val)

            # Loop autoregressivo: testiamo sia Teacher Forcing che Free Running!
            # Test 1: Autoregressivo puro (passa il codice generato full_codes[t-1])
            curr_code = torch.cat([torch.tensor([[cb0_val]], device='cuda'), resid_t[0, 0:1]], dim=-1) # (1, 16)

            for t in range(1, n_real):
                cp = curr_code[0] # Autoregressivo puro!
                agg = sum(talker.get_input_embeddings()(cp[g:g+1].reshape(1,1)) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](cp[g:g+1].reshape(1,1)) for g in range(16))
                in_t = agg + h_cond[:, t:t+1, :]
                out = talker.model(inputs_embeds=in_t, attention_mask=torch.ones(1, 1, device='cuda'), use_cache=True, past_key_values=kv)
                kv = out.past_key_values
                past_h = out.last_hidden_state[:, -1:, :]

                lg = talker.codec_head(past_h)
                cb0_val = sample_cb0_with_penalty(lg, cb0_all, temperature=0.8, top_k=50, top_p=0.92, rep_penalty=1.25)
                resid_t = sub.sample_acoustic_codes(past_h.to(torch.bfloat16), torch.tensor([[cb0_val]], device='cuda'), top_k=50, top_p=0.95, temperatures_per_level=tdecay)
                resid_all.append(resid_t[0, 0].cpu().long())
                cb0_all.append(cb0_val)
                curr_code = torch.cat([torch.tensor([[cb0_val]], device='cuda'), resid_t[0, 0:1]], dim=-1)

            R = torch.stack(resid_all, dim=0) # (T, 15)
            C = torch.tensor(cb0_all).unsqueeze(-1) # (T, 1)
            full_codes = torch.cat([C, R], dim=-1) # (T, 16)

            c_input = full_codes.unsqueeze(0).cuda()
            out_dec = tok_decoder.decode(c_input)
            wav_arr = out_dec.audio_values[0].cpu().numpy().astype(np.float32)

            p_peak = float(np.percentile(np.abs(wav_arr), 99.7))
            if p_peak > 1e-6:
                wav_norm = np.clip(wav_arr / (p_peak * 1.05), -1.0, 1.0)
            else:
                wav_norm = wav_arr

            wav_path = f'/workspace/samples_v7/v7_official_prefill_{i:02d}.wav'
            sf.write(wav_path, wav_norm, 24000)
            rms = float(np.sqrt(np.mean(wav_norm**2)))

            segs, _ = asr.transcribe(wav_path, language='it', beam_size=5)
            transcript = ' '.join(s.text.strip() for s in segs)

            print(f"  • RMS: {rms:.4f} | Token Unici: {len(set(cb0_all))}/64")
            print(f"  • ASR Trascrizione: \"{transcript}\"")

if __name__ == '__main__':
    test_synthesis()
