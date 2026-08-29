#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test V8 MLP 25000 Steps with RVQ Codebook Cutoff (N7, N3, N1 vs Full N15 vs GT)
Evaluates acoustic quality, crest factor, spectral distribution, and Whisper ASR.
"""

import sys, os, glob, json, math, time
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
sys.modules['soxr'] = types.ModuleType('soxr')
_l = types.ModuleType('librosa'); _l.__spec__ = importlib.machinery.ModuleSpec('librosa', None)
_l.filters = types.ModuleType('librosa.filters')
_l.filters.mel = lambda *args, **kwargs: None
sys.modules['librosa'] = _l
sys.modules['librosa.filters'] = _l.filters

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

def compute_audio_metrics(wav, sr=24000):
    rms = float(np.sqrt(np.mean(wav**2)))
    peak = float(np.max(np.abs(wav)))
    crest = float(peak / (rms + 1e-9))
    zcr = float(np.mean(np.abs(np.diff(np.sign(wav)))) / 2.0)
    
    fft = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), 1.0/sr)
    e_sub = float(np.sum(fft[freqs < 100]**2))
    e_f0 = float(np.sum(fft[(freqs >= 100) & (freqs < 300)]**2))
    e_form = float(np.sum(fft[(freqs >= 300) & (freqs < 3000)]**2))
    e_high = float(np.sum(fft[freqs >= 3000]**2))
    total_e = e_sub + e_f0 + e_form + e_high + 1e-12
    
    return {
        'rms': rms,
        'peak': peak,
        'crest': crest,
        'zcr': zcr,
        'sub_pct': (e_sub / total_e) * 100.0,
        'f0_pct': (e_f0 / total_e) * 100.0,
        'form_pct': (e_form / total_e) * 100.0,
        'high_pct': (e_high / total_e) * 100.0
    }

def main():
    out_dir = "/workspace/samples_cutoff_n7_eval"
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = "/workspace/checkpoints/proj_v8_step25000.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "/workspace/checkpoints/proj_v8_final.pt"

    print("==================================================================", flush=True)
    print(f"  TEST CODEBOOK CUTOFF N7 SU CHECKPOINT V8 (MLP 25000 STEP)", flush=True)
    print(f"  Checkpoint: {ckpt_path}", flush=True)
    print(f"  Output Dir: {out_dir}", flush=True)
    print("==================================================================", flush=True)

    # 1. Carica Qwen3-TTS
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

    # 2. Carica Proiettore MLP V8
    proj = TextAnchoredProjectionV8().to(device='cuda', dtype=torch.bfloat16)
    sd = torch.load(ckpt_path, map_location='cpu')
    proj.load_state_dict(sd.get('proj', sd), strict=False)
    proj.eval()
    print(f"  [PROIETTORE CARICATO] Gamma={proj.gamma.item():.4f}", flush=True)

    # 3. Dati
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

    results = []

    # Testiamo 5 campioni di validazione
    for sample_idx in range(5):
        c0 = val_start + sample_idx * 64
        codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
        codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
        n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
        if n_real < 16: n_real = 64
        codes = codes[:n_real]
        T = n_real

        gi = (val_start // 64) + sample_idx
        tbw_entry = tbw[gi] if gi < len(tbw) else None
        if isinstance(tbw_entry, (list, np.ndarray)) and tok_gemma is not None:
            text_str = tok_gemma.decode(tbw_entry, skip_special_tokens=True).strip()
        else:
            text_str = str(tbw_entry).strip() if tbw_entry is not None else f"Campione di validazione {sample_idx}."

        h_raw = torch.from_numpy(np.asarray(va_raw[sample_idx][:n_real], dtype=np.float32)).to('cuda')
        h_tx = torch.from_numpy(np.asarray(va_htx[sample_idx], dtype=np.float32)).to('cuda')

        print(f"\n------------------------------------------------------------------", flush=True)
        print(f"  [CAMPIONE {sample_idx}]: \"{text_str}\" (T={T} frames)", flush=True)
        print(f"------------------------------------------------------------------", flush=True)

        with torch.no_grad():
            h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))
            icl = build_official_prefill(text_str)

            # Generazione Autoregressiva Open-Loop CB0
            out_free = talker.model(inputs_embeds=icl, use_cache=True)
            past_kv = out_free.past_key_values
            last_h = out_free.last_hidden_state[:, -1:, :]

            codes_full = []
            for t in range(T):
                cb0_lg = talker.codec_head(last_h)[:, -1, :]
                cb0_tok = sample_top_p_top_k(cb0_lg, top_p=0.92, top_k=50, temperature=0.8)
                c0_val = int(cb0_tok.item())

                # Generazione dei 15 residui tramite CodePredictor ufficiale
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
                codes_full.append(full_t)

                if t < T - 1:
                    cur_codes_t = torch.tensor(full_t, device='cuda')
                    agg_t = talker.get_input_embeddings()(cur_codes_t[0:1].unsqueeze(0))
                    for g in range(1, 16):
                        agg_t = agg_t + talker.code_predictor.get_input_embeddings()[g-1](cur_codes_t[g:g+1].unsqueeze(0))
                    next_embed = agg_t.transpose(0, 1) + h_cond[:, t+1:t+2, :]
                    out_step = talker.model(inputs_embeds=next_embed, past_key_values=past_kv, use_cache=True)
                    past_kv = out_step.past_key_values
                    last_h = out_step.last_hidden_state[:, -1:, :]

            # Config 1: Full N15 (Tutti i 16 quantizzatori)
            t_n15 = torch.clamp(torch.tensor(codes_full, device='cuda', dtype=torch.long).unsqueeze(0), 0, 2047)
            wav_n15 = tok_decoder.decode(t_n15).audio_values[0].cpu().numpy().astype(np.float32)
            p_peak = float(np.percentile(np.abs(wav_n15), 99.7))
            wav_n15 = np.clip(wav_n15 / (p_peak * 1.05), -1.0, 1.0) if p_peak > 1e-6 else wav_n15
            path_n15 = os.path.join(out_dir, f"sample_{sample_idx}_full_N15.wav")
            sf.write(path_n15, wav_n15, 24000)
            segs_n15, _ = whisper_model.transcribe(path_n15, language="it", temperature=0.0)
            text_n15 = " ".join([s.text for s in segs_n15]).strip()
            m_n15 = compute_audio_metrics(wav_n15)

            # Config 2: Cutoff N7 (cb0..cb7 mantenuti, cb8..15 azzerati)
            codes_n7 = [[c[i] if i <= 7 else 0 for i in range(16)] for c in codes_full]
            t_n7 = torch.clamp(torch.tensor(codes_n7, device='cuda', dtype=torch.long).unsqueeze(0), 0, 2047)
            wav_n7 = tok_decoder.decode(t_n7).audio_values[0].cpu().numpy().astype(np.float32)
            p_peak = float(np.percentile(np.abs(wav_n7), 99.7))
            wav_n7 = np.clip(wav_n7 / (p_peak * 1.05), -1.0, 1.0) if p_peak > 1e-6 else wav_n7
            path_n7 = os.path.join(out_dir, f"sample_{sample_idx}_cutoff_N7.wav")
            sf.write(path_n7, wav_n7, 24000)
            segs_n7, _ = whisper_model.transcribe(path_n7, language="it", temperature=0.0)
            text_n7 = " ".join([s.text for s in segs_n7]).strip()
            m_n7 = compute_audio_metrics(wav_n7)

            # Config 3: Cutoff N3 (cb0..cb3 mantenuti, cb4..15 azzerati)
            codes_n3 = [[c[i] if i <= 3 else 0 for i in range(16)] for c in codes_full]
            t_n3 = torch.clamp(torch.tensor(codes_n3, device='cuda', dtype=torch.long).unsqueeze(0), 0, 2047)
            wav_n3 = tok_decoder.decode(t_n3).audio_values[0].cpu().numpy().astype(np.float32)
            p_peak = float(np.percentile(np.abs(wav_n3), 99.7))
            wav_n3 = np.clip(wav_n3 / (p_peak * 1.05), -1.0, 1.0) if p_peak > 1e-6 else wav_n3
            path_n3 = os.path.join(out_dir, f"sample_{sample_idx}_cutoff_N3.wav")
            sf.write(path_n3, wav_n3, 24000)
            segs_n3, _ = whisper_model.transcribe(path_n3, language="it", temperature=0.0)
            text_n3 = " ".join([s.text for s in segs_n3]).strip()
            m_n3 = compute_audio_metrics(wav_n3)

            # Config 4: Ground Truth Reference
            t_gt = torch.clamp(codes.unsqueeze(0), 0, 2047)
            wav_gt = tok_decoder.decode(t_gt).audio_values[0].cpu().numpy().astype(np.float32)
            p_peak = float(np.percentile(np.abs(wav_gt), 99.7))
            wav_gt = np.clip(wav_gt / (p_peak * 1.05), -1.0, 1.0) if p_peak > 1e-6 else wav_gt
            path_gt = os.path.join(out_dir, f"sample_{sample_idx}_gt_ref.wav")
            sf.write(path_gt, wav_gt, 24000)
            segs_gt, _ = whisper_model.transcribe(path_gt, language="it", temperature=0.0)
            text_gt = " ".join([s.text for s in segs_gt]).strip()
            m_gt = compute_audio_metrics(wav_gt)

            print(f"  [FULL N15]   RMS={m_n15['rms']:.4f} Crest={m_n15['crest']:.2f} Sub={m_n15['sub_pct']:.1f}% Form={m_n15['form_pct']:.1f}% | ASR: \"{text_n15}\"", flush=True)
            print(f"  [CUTOFF N7]  RMS={m_n7['rms']:.4f} Crest={m_n7['crest']:.2f} Sub={m_n7['sub_pct']:.1f}% Form={m_n7['form_pct']:.1f}% | ASR: \"{text_n7}\"", flush=True)
            print(f"  [CUTOFF N3]  RMS={m_n3['rms']:.4f} Crest={m_n3['crest']:.2f} Sub={m_n3['sub_pct']:.1f}% Form={m_n3['form_pct']:.1f}% | ASR: \"{text_n3}\"", flush=True)
            print(f"  [GT REF]     RMS={m_gt['rms']:.4f} Crest={m_gt['crest']:.2f} Sub={m_gt['sub_pct']:.1f}% Form={m_gt['form_pct']:.1f}% | ASR: \"{text_gt}\"", flush=True)

            results.append({
                'sample_idx': sample_idx,
                'target_text': text_str,
                'n15': {'asr': text_n15, 'metrics': m_n15, 'wav': path_n15},
                'n7': {'asr': text_n7, 'metrics': m_n7, 'wav': path_n7},
                'n3': {'asr': text_n3, 'metrics': m_n3, 'wav': path_n3},
                'gt': {'asr': text_gt, 'metrics': m_gt, 'wav': path_gt}
            })

    report_path = os.path.join(out_dir, "cutoff_n7_evaluation_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[COMPLETATO] Report salvato in {report_path}", flush=True)

if __name__ == '__main__':
    main()
