#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Codebook Cutoff & Greedy vs Sampled SubTalker (TDD Test Suite)
Investigates how many codebooks and what sampling strategy eliminates white noise distortion.
"""

import sys, os, glob, json
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

def main():
    out_dir = "/workspace/samples_cutoff_test"
    os.makedirs(out_dir, exist_ok=True)

    print("=== [TDD TEST] CARICAMENTO MODELLI ===", flush=True)
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

    sample_idx = 1
    c0 = val_start + sample_idx * 64
    codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
    codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
    codes = torch.clamp(codes, 0, 2047)
    n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
    if n_real < 16: n_real = 64
    codes = codes[:n_real]
    T = n_real

    text_str = "Si tratta del secondo singolo promozionale nella storia della band."
    h_raw = torch.from_numpy(np.asarray(va_raw[sample_idx][:n_real], dtype=np.float32)).to('cuda')
    h_tx = torch.from_numpy(np.asarray(va_htx[sample_idx], dtype=np.float32)).to('cuda')

    with torch.no_grad():
        h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))
        icl = build_official_prefill(text_str)

        # 1. Generate Open Loop CB0 tokens
        out_free = talker.model(inputs_embeds=icl, use_cache=True)
        past_kv = out_free.past_key_values
        last_h = out_free.last_hidden_state[:, -1:, :]

        cb0_tokens = []
        hidden_states_list = []
        for t in range(T):
            cb0_lg = talker.codec_head(last_h)[:, -1, :]
            cb0_lg[0, 2048:] = -float('Inf')
            c0_val = int(cb0_lg.argmax(dim=-1).item()) # Greedy CB0
            cb0_tokens.append(c0_val)
            hidden_states_list.append(last_h)

            if t < T - 1:
                # Use greedy feedback for next step
                last_id_h = talker.get_input_embeddings()(torch.tensor([[c0_val]], device='cuda'))
                next_embed = last_id_h + h_cond[:, t+1:t+2, :]
                out_step = talker.model(inputs_embeds=next_embed, past_key_values=past_kv, use_cache=True)
                past_kv = out_step.past_key_values
                last_h = out_step.last_hidden_state[:, -1:, :]

        # TEST 1: Greedy SubTalker Argmax (Zero Temperature - deterministic mode)
        codes_greedy = []
        for t in range(T):
            c0_val = cb0_tokens[t]
            h_t = hidden_states_list[t]
            last_id_h = talker.get_input_embeddings()(torch.tensor([[c0_val]], device='cuda'))
            pred_res = talker.code_predictor.generate(
                inputs_embeds=torch.cat((h_t, last_id_h), dim=1),
                max_new_tokens=15,
                do_sample=False, # GREEDY ARGMAX!
                return_dict_in_generate=True,
            )
            codes_greedy.append([c0_val] + pred_res.sequences[0].cpu().tolist())

        t_greedy = torch.clamp(torch.tensor(codes_greedy, device='cuda', dtype=torch.long).unsqueeze(0), 0, 2047)
        wav_greedy = tok_decoder.decode(t_greedy).audio_values[0].cpu().numpy().astype(np.float32)
        p_peak = float(np.percentile(np.abs(wav_greedy), 99.7))
        wav_greedy = np.clip(wav_greedy / (p_peak * 1.05), -1.0, 1.0) if p_peak > 1e-6 else wav_greedy
        path_greedy = os.path.join(out_dir, "test1_greedy_subtalker.wav")
        sf.write(path_greedy, wav_greedy, 24000)
        segs_g, _ = whisper_model.transcribe(path_greedy, language="it", temperature=0.0)
        text_g = " ".join([s.text for s in segs_g]).strip()
        print(f"\n[TEST 1 - GREEDY ARGMAX SUBTALKER]: \"{text_g}\" -> {path_greedy}")

        # TEST 2: Greedy CB0 + GT Res (Baseline clean)
        codes_gt_res = []
        for t in range(T):
            codes_gt_res.append([cb0_tokens[t]] + codes[t, 1:].cpu().tolist())
        t_gt_res = torch.clamp(torch.tensor(codes_gt_res, device='cuda', dtype=torch.long).unsqueeze(0), 0, 2047)
        wav_gt_res = tok_decoder.decode(t_gt_res).audio_values[0].cpu().numpy().astype(np.float32)
        p_peak = float(np.percentile(np.abs(wav_gt_res), 99.7))
        wav_gt_res = np.clip(wav_gt_res / (p_peak * 1.05), -1.0, 1.0) if p_peak > 1e-6 else wav_gt_res
        path_gt_res = os.path.join(out_dir, "test2_gt_residuals.wav")
        sf.write(path_gt_res, wav_gt_res, 24000)
        print(f"[TEST 2 - GT RESIDUALS]: -> {path_gt_res}")

        # TEST 3: Cutoff to 4 codebooks (CB0..CB3 greedy, CB4..15 set to 0)
        codes_4cb = []
        for t in range(T):
            c_full = codes_greedy[t]
            c_cut = c_full[:4] + [0] * 12
            codes_4cb.append(c_cut)
        t_4cb = torch.clamp(torch.tensor(codes_4cb, device='cuda', dtype=torch.long).unsqueeze(0), 0, 2047)
        wav_4cb = tok_decoder.decode(t_4cb).audio_values[0].cpu().numpy().astype(np.float32)
        p_peak = float(np.percentile(np.abs(wav_4cb), 99.7))
        wav_4cb = np.clip(wav_4cb / (p_peak * 1.05), -1.0, 1.0) if p_peak > 1e-6 else wav_4cb
        path_4cb = os.path.join(out_dir, "test3_cutoff_4cb.wav")
        sf.write(path_4cb, wav_4cb, 24000)
        segs_4, _ = whisper_model.transcribe(path_4cb, language="it", temperature=0.0)
        text_4 = " ".join([s.text for s in segs_4]).strip()
        print(f"[TEST 3 - CUTOFF 4 CODEBOOKS]: \"{text_4}\" -> {path_4cb}")

if __name__ == '__main__':
    main()
