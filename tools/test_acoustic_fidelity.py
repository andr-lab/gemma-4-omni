#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acoustic Fidelity & Noise Diagnostic (TDD / SDD)
Tests:
1. Generation with rep_penalty=1.0 vs 1.25 (Vowel stability test)
2. SubTalker sampling: Official Qwen3 CodePredictor vs ad-hoc temperature decay
3. Ground-truth acoustic residual injection vs open-loop residual drift (Exposure bias test)
"""

import sys, os, glob, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from transformers import AutoTokenizer
import qwen_tts.core.models as _M
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
import safetensors.torch as st
import kaggle_pipeline as KP
from faster_whisper import WhisperModel

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

def sample_top_p_top_k(logits, top_p=0.92, top_k=50, temperature=0.8):
    logits = logits / temperature
    # Suppress special tokens >= 2048
    logits[0, 2048:] = -float('Inf')
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
    return torch.multinomial(probs, num_samples=1)

def main():
    out_dir = "/workspace/samples_diagnostic"
    os.makedirs(out_dir, exist_ok=True)

    print("=== [DIAGNOSTIC ACOUSTIC NOISE] Caricamento Modelli ===", flush=True)
    snap = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*'))[-1]
    _cfg = _M.Qwen3TTSConfig.from_pretrained(snap)
    _tc = _cfg.talker_config
    if not hasattr(_tc, 'pad_token_id'): object.__setattr__(_tc, 'pad_token_id', None)

    teacher = _M.Qwen3TTSForConditionalGeneration(_cfg).to(torch.bfloat16)
    _sd = st.load_file(snap + '/model.safetensors', device='cpu')
    teacher.load_state_dict(_sd, strict=False)
    del _sd; torch.cuda.empty_cache()
    teacher = teacher.to(device='cuda', dtype=torch.bfloat16).eval()
    talker = teacher.talker

    tok_qwen = AutoTokenizer.from_pretrained(snap, use_fast=True)
    proc_qwen = _M.Qwen3TTSProcessor(tokenizer=tok_qwen)

    snap_tok = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*'))[-1]
    tok_decoder = Qwen3TTSTokenizerV2Model.from_pretrained(snap_tok).cuda().eval()

    gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    tok_gemma = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True) if len(gemma_dirs) > 0 else None

    CFG = teacher.config
    TCFG = CFG.talker_config
    lang_map = getattr(TCFG, 'codec_language_id', {})
    lang_id = lang_map.get('italian', 2070)

    proj = TextAnchoredProjectionV8().to(device='cuda', dtype=torch.bfloat16)
    ckpt_path = '/workspace/checkpoints/proj_v8_final.pt'
    sd = torch.load(ckpt_path, map_location='cpu')
    proj.load_state_dict(sd.get('proj', sd), strict=False)
    proj.eval()

    _, _, agent_codes, tbw = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')
    ck = torch.load('/workspace/cache_ars_full_v2.pt', map_location='cpu', weights_only=False, mmap=True)
    va_raw = ck['va_raw']
    va_htx = ck['va_htx']

    N_all = agent_codes.shape[1]
    val_start = int(0.90 * N_all)
    seg_ends = json.load(open('/workspace/data/train_tokens_cv26_it_40.8h_31916s_hyperparams.json')).get('SEGMENT_ENDS', [])
    for e in seg_ends:
        if e >= val_start: val_start = e; break

    whisper_dirs = glob.glob('/workspace/.hf_cache/models--Systran--faster-whisper-small/snapshots/*')
    whisper_model_path = whisper_dirs[0] if len(whisper_dirs) > 0 else "small"
    whisper_model = WhisperModel(whisper_model_path, device="cuda", compute_type="float16", download_root="/workspace/.hf_cache")

    def build_official_prefill(text_str):
        raw_ids = proc_qwen(text=text_str, return_tensors='pt')['input_ids'].to('cuda')
        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            talker.get_text_embeddings()(
                torch.tensor([[CFG.tts_bos_token_id, CFG.tts_eos_token_id, CFG.tts_pad_token_id]], device='cuda', dtype=raw_ids.dtype)
            )
        ).chunk(3, dim=1)

        codec_prefill = [[TCFG.codec_think_id, TCFG.codec_think_bos_id, lang_id, TCFG.codec_think_eos_id]]
        c_emb_0 = talker.get_input_embeddings()(torch.tensor(codec_prefill, device='cuda', dtype=raw_ids.dtype))
        c_emb_1 = talker.get_input_embeddings()(torch.tensor([[TCFG.codec_pad_id, TCFG.codec_bos_id]], device='cuda', dtype=raw_ids.dtype))
        codec_input_emebdding = torch.cat([c_emb_0, c_emb_1], dim=1)

        role_ids = proc_qwen(text="<|im_start|>assistant\n", return_tensors='pt')['input_ids'].to('cuda')
        role_embed = talker.text_projection(talker.get_text_embeddings()(role_ids))
        _talker_input_embed = torch.cat((tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1), tts_bos_embed), dim=1) + codec_input_emebdding[:, :-1]
        talker_input_embed = torch.cat((role_embed, _talker_input_embed), dim=1)

        text_proj_emb = talker.text_projection(talker.get_text_embeddings()(raw_ids))
        c_pad_full = talker.get_input_embeddings()(torch.tensor([[TCFG.codec_pad_id] * (raw_ids.shape[1] + 1)], device='cuda', dtype=raw_ids.dtype))
        text_part = torch.cat([text_proj_emb, tts_eos_embed], dim=1) + c_pad_full
        bos_part = tts_pad_embed + talker.get_input_embeddings()(torch.tensor([[TCFG.codec_bos_id]], device='cuda', dtype=raw_ids.dtype))

        return torch.cat([talker_input_embed, text_part, bos_part], dim=1)

    # Test on Sample 1
    sample_idx = 1
    c0 = val_start + sample_idx * 64
    codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
    codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
    n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
    if n_real < 16: n_real = 64
    codes = codes[:n_real]
    T = n_real

    gi = (val_start // 64) + sample_idx
    text_str = tok_gemma.decode(tbw[gi], skip_special_tokens=True).strip() if (gi < len(tbw) and isinstance(tbw[gi], list)) else "Si tratta del secondo singolo."
    h_raw = torch.from_numpy(np.asarray(va_raw[sample_idx][:n_real], dtype=np.float32)).to('cuda')
    h_tx = torch.from_numpy(np.asarray(va_htx[sample_idx], dtype=np.float32)).to('cuda')

    print(f"\n[TEST SU CAMPIONE 1]: \"{text_str}\" (T={T} frames)")

    with torch.no_grad():
        h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))
        icl = build_official_prefill(text_str)

        # -------------------------------------------------------------
        # EXPERIMENT A: Official CodePredictor Sampling (No Rep Penalty)
        # -------------------------------------------------------------
        out_free = talker.model(inputs_embeds=icl, use_cache=True)
        past_kv = out_free.past_key_values
        last_h = out_free.last_hidden_state[:, -1:, :]

        codes_exp_a = []
        for t in range(T):
            cb0_lg = talker.codec_head(last_h)[:, -1, :]
            cb0_tok = sample_top_p_top_k(cb0_lg, top_p=0.92, top_k=50, temperature=0.8)
            c0_val = int(cb0_tok.item())

            # Official CodePredictor generate
            last_id_h = talker.get_input_embeddings()(torch.tensor([[c0_val]], device='cuda'))
            pred_res = talker.code_predictor.generate(
                inputs_embeds=torch.cat((last_h, last_id_h), dim=1),
                max_new_tokens=15,
                do_sample=True,
                top_p=1.0,
                top_k=50,
                temperature=0.9,
                return_dict_in_generate=True,
            )
            res_codes = pred_res.sequences[0].cpu().tolist()
            full_t = [c0_val] + res_codes
            codes_exp_a.append(full_t)

            if t < T - 1:
                cur_codes_t = torch.tensor(full_t, device='cuda')
                agg_t = talker.get_input_embeddings()(cur_codes_t[0:1].unsqueeze(0))
                for g in range(1, 16):
                    agg_t = agg_t + talker.code_predictor.get_input_embeddings()[g-1](cur_codes_t[g:g+1].unsqueeze(0))
                next_embed = agg_t.transpose(0, 1) + h_cond[:, t+1:t+2, :]
                out_step = talker.model(inputs_embeds=next_embed, past_key_values=past_kv, use_cache=True)
                past_kv = out_step.past_key_values
                last_h = out_step.last_hidden_state[:, -1:, :]

        t_exp_a = torch.clamp(torch.tensor(codes_exp_a, device='cuda', dtype=torch.long).unsqueeze(0), 0, 2047)
        wav_a = tok_decoder.decode(t_exp_a).audio_values[0].cpu().numpy().astype(np.float32)
        p_peak = float(np.percentile(np.abs(wav_a), 99.7))
        wav_a = np.clip(wav_a / (p_peak * 1.05), -1.0, 1.0) if p_peak > 1e-6 else wav_a
        path_a = os.path.join(out_dir, "exp_a_official_codepredictor.wav")
        sf.write(path_a, wav_a, 24000)
        segs_a, _ = whisper_model.transcribe(path_a, language="it", temperature=0.0)
        text_a = " ".join([s.text for s in segs_a]).strip()
        print(f"  [EXP A - Official CodePredictor]: \"{text_a}\" -> {path_a}")

        # -------------------------------------------------------------
        # EXPERIMENT B: Ground Truth Acoustic Residuals (Isolates CB0 vs Residuals)
        # -------------------------------------------------------------
        codes_exp_b = []
        for t in range(T):
            c0_val = codes_exp_a[t][0] # take the same CB0 from Exp A
            gt_residuals = codes[t, 1:].cpu().tolist() # take GT acoustic residuals
            codes_exp_b.append([c0_val] + gt_residuals)

        t_exp_b = torch.clamp(torch.tensor(codes_exp_b, device='cuda', dtype=torch.long).unsqueeze(0), 0, 2047)
        wav_b = tok_decoder.decode(t_exp_b).audio_values[0].cpu().numpy().astype(np.float32)
        p_peak_b = float(np.percentile(np.abs(wav_b), 99.7))
        wav_b = np.clip(wav_b / (p_peak_b * 1.05), -1.0, 1.0) if p_peak_b > 1e-6 else wav_b
        path_b = os.path.join(out_dir, "exp_b_clean_gt_residuals.wav")
        sf.write(path_b, wav_b, 24000)
        segs_b, _ = whisper_model.transcribe(path_b, language="it", temperature=0.0)
        text_b = " ".join([s.text for s in segs_b]).strip()
        print(f"  [EXP B - Clean GT Residuals]    : \"{text_b}\" -> {path_b}")

if __name__ == '__main__':
    main()
