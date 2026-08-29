#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test V8 Free-Running (Autoregressive Open-Loop) Audio Generation & ASR Evaluation
Generates speech without ground-truth past codes using Nucleus Sampling and Repetition Penalty,
then evaluates transcriptions locally via faster-whisper-small.
"""

import os, sys, glob, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import safetensors.torch as st
from transformers import AutoTokenizer
import soundfile as sf

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
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
from qwen_tts.core.models import Qwen3TTSProcessor

def build_official_prefill_func(tok_qwen, teacher):
    def build_official_prefill(text_prompt: str) -> torch.Tensor:
        tc = teacher.talker.config
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "repeat after me: " + text_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": text_prompt}]}
        ]
        text = tok_qwen.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tok_out = tok_qwen(text, return_tensors="pt")
        input_ids = tok_out.input_ids.cuda()

        if hasattr(tc, 'codec_think_id') and tc.codec_think_id is not None:
            prefill_part1 = input_ids[:, :-5]
            prefill_part2 = input_ids[:, -5:]
            language_id = 2070 # Italian
            codec_think = torch.tensor([[tc.codec_think_id, language_id]], dtype=torch.long, device='cuda')
            full_ids = torch.cat([prefill_part1, codec_think, prefill_part2], dim=1)
        else:
            full_ids = input_ids

        embed_layer = teacher.talker.text_embedding
        icl_embeds = embed_layer(full_ids)
        return icl_embeds
    return build_official_prefill

class TextAnchoredProjectionV8(nn.Module):
    def __init__(self, raw_dim=1536, text_dim=1536, hidden_dim=2048, out_dim=2048, initial_gamma=0.25):
        super().__init__()
        self.audio_mlp = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.audio_skip = nn.Linear(raw_dim, out_dim, bias=False)
        self.audio_norm = nn.LayerNorm(out_dim)

        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
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

def sample_top_p_top_k(logits, top_p=0.92, top_k=50, temperature=0.8, rep_penalty=1.25, past_tokens=None):
    logits = logits / temperature
    if past_tokens is not None and len(past_tokens) > 0 and rep_penalty != 1.0:
        for t in set(past_tokens[-16:]):
            if logits[0, t] > 0:
                logits[0, t] /= rep_penalty
            else:
                logits[0, t] *= rep_penalty

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = -float('Inf')

    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    return token

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='/workspace/checkpoints/proj_v8_final.pt')
    parser.add_argument('--num-samples', type=int, default=3)
    parser.add_argument('--out-dir', type=str, default='/workspace/samples_v8_free')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== TEST GENERAZIONE LIBERA AUTOREGRESSIVA (OPEN-LOOP) ===")
    print(f"  Checkpoint: {args.ckpt}")

    snap = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*'))[-1]
    _cfg = _M.Qwen3TTSConfig.from_pretrained(snap)
    if not hasattr(_cfg.talker_config, 'pad_token_id'):
        object.__setattr__(_cfg.talker_config, 'pad_token_id', None)

    teacher = _M.Qwen3TTSForConditionalGeneration(_cfg).to(torch.bfloat16)
    _sd = st.load_file(snap + '/model.safetensors', device='cpu')
    teacher.load_state_dict(_sd, strict=False)
    del _sd; torch.cuda.empty_cache()
    teacher = teacher.to(device='cuda', dtype=torch.bfloat16).eval()
    talker = teacher.talker

    tok_qwen = AutoTokenizer.from_pretrained(snap, use_fast=True)
    build_prefill = build_official_prefill_func(tok_qwen, teacher)

    snap_tok = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*'))[-1]
    tok_decoder = Qwen3TTSTokenizerV2Model.from_pretrained(snap_tok).cuda().eval()

    proj = TextAnchoredProjectionV8().to('cuda', dtype=torch.bfloat16)
    if os.path.exists(args.ckpt):
        st_data = torch.load(args.ckpt, map_location='cuda')
        proj.load_state_dict(st_data['proj'])
        print(f"  Proiettore caricato con successo (gamma={proj.gamma.item():.4f}).")
    proj.eval()

    # Carica cache v2 e token
    import kaggle_pipeline as KP
    _, _, agent_codes, tbw = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')
    ck = torch.load('/workspace/cache_ars_full_v2.pt', map_location='cpu', weights_only=False, mmap=True)
    va_raw = ck['va_raw']
    va_htx = ck['va_htx']

    N_all = agent_codes.shape[1]
    val_start = int(0.90 * N_all)
    seg_ends = json.load(open('/workspace/data/train_tokens_cv26_it_40.8h_31916s_hyperparams.json')).get('SEGMENT_ENDS', [])
    for e in seg_ends:
        if e >= val_start: val_start = e; break

    gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    tok_gemma = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True) if len(gemma_dirs) > 0 else None

    def get_italian_text_prompt(tbw_item):
        if tbw_item is None: return "Buongiorno."
        if isinstance(tbw_item, (list, np.ndarray)):
            if tok_gemma is not None:
                return tok_gemma.decode(tbw_item, skip_special_tokens=True).strip()
            return "Frase in lingua italiana."
        return str(tbw_item).strip()

    from faster_whisper import WhisperModel
    whisper = WhisperModel('small', device='cuda', compute_type='float16')

    for s_idx in range(args.num_samples):
        c0 = val_start + s_idx * 64
        codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
        n_real = int((codes_np[0] != 4096).sum())
        if n_real < 16: n_real = 64
        
        gi = (val_start // 64) + s_idx
        target_text = get_italian_text_prompt(tbw[gi] if gi < len(tbw) else None)
        h_raw = torch.from_numpy(np.asarray(va_raw[s_idx][:n_real], dtype=np.float32)).to('cuda')
        h_tx = torch.from_numpy(np.asarray(va_htx[s_idx], dtype=np.float32)).to('cuda')

        T = n_real
        with torch.no_grad():
            h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))
            icl = build_prefill(target_text)
            L_prefill = icl.shape[1]

            # Autoregressive Open-Loop Generation
            gen_codes = []
            past_cb0 = []
            
            # 1. Prefill
            out = talker.model(inputs_embeds=icl, use_cache=True)
            past_kv = out.past_key_values
            last_h = out.last_hidden_state[:, -1:, :]

            for t in range(T):
                # Sample CB0
                cb0_lg = talker.codec_head(last_h)[:, -1, :]
                cb0_tok = sample_top_p_top_k(cb0_lg, top_p=0.92, top_k=50, temperature=0.8, rep_penalty=1.25, past_tokens=past_cb0)
                cb0_val = int(cb0_tok.item())
                past_cb0.append(cb0_val)

                # Sample CB1..15 via Sub-Talker
                sub_lg, _ = talker.forward_sub_talker_finetune(torch.tensor([[cb0_val] + [0]*15], device='cuda'), last_h[0, 0])
                res_toks = [cb0_val]
                for g in range(1, 16):
                    rg_lg = sub_lg[0, g-1:g, :]
                    rg_tok = int(rg_lg.argmax(dim=-1).item())
                    res_toks.append(rg_tok)
                
                gen_codes.append(res_toks)

                if t < T - 1:
                    # Construct next input embedding: agg_audio + h_cond[t+1]
                    cur_codes_t = torch.tensor(res_toks, device='cuda')
                    agg_t = talker.get_input_embeddings()(cur_codes_t[0:1].unsqueeze(0))
                    for g in range(1, 16):
                        agg_t = agg_t + talker.code_predictor.get_input_embeddings()[g-1](cur_codes_t[g:g+1].unsqueeze(0))
                    
                    next_embed = agg_t.transpose(0, 1) + h_cond[:, t+1:t+2, :] # (1, 1, 2048)
                    out = talker.model(inputs_embeds=next_embed, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                    last_h = out.last_hidden_state[:, -1:, :]

            # Decode Audio
            gen_matrix = torch.tensor(gen_codes, device='cuda').transpose(0, 1).unsqueeze(0) # (1, 16, T)
            wav_t = tok_decoder.decode(gen_matrix)
            wav_np = wav_t.squeeze().float().cpu().numpy()
            
            wav_path = os.path.join(args.out_dir, f"sample_free_{s_idx}.wav")
            sf.write(wav_path, wav_np, 24000)

            # Transcribe with Whisper
            segments, _ = whisper.transcribe(wav_path, language="it")
            asr_text = " ".join([seg.text for seg in segments]).strip()

            print(f"\n[SAMPLE {s_idx} - OPEN LOOP]")
            print(f"  Target Text: \"{target_text}\"")
            print(f"  ASR Whisper: \"{asr_text}\"")
            print(f"  WAV Saved  : {wav_path}")

if __name__ == '__main__':
    main()
