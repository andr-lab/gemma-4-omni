#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation & Audio Generation for Gemma4-Omni V8.5 (Step 25000)
- Free-running (Open-Loop) Generation
- Teacher Forcing Comparison (Omni vs Native Base Qwen)
- Whisper ASR Transcription & WER / CER computation
"""

import os, sys, glob, json, math, time
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
import qwen_subtalker as QS
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

def compute_wer(reference: str, hypothesis: str) -> float:
    r = reference.lower().split()
    h = hypothesis.lower().split()
    if not r: return 0.0 if not h else 1.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    return float(d[len(r)][len(h)]) / float(len(r))

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
    return torch.multinomial(probs, num_samples=1)

def main():
    out_dir = "/workspace/samples_v8_eval"
    os.makedirs(out_dir, exist_ok=True)

    print("=== [EVAL V8.5] CARICAMENTO MODELLI E PESI ===", flush=True)
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
    proc_qwen = _M.Qwen3TTSProcessor(tokenizer=tok_qwen)

    snap_tok = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*'))[-1]
    tok_decoder = Qwen3TTSTokenizerV2Model.from_pretrained(snap_tok).cuda().eval()
    sub, _ = QS.load_qwen_subtalker('/workspace/qwen_subtalker', device='cuda', dtype=torch.bfloat16)

    gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    tok_gemma = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True) if len(gemma_dirs) > 0 else None

    CFG = teacher.config
    TCFG = CFG.talker_config
    lang_map = getattr(TCFG, 'codec_language_id', {})
    lang_id = lang_map.get('italian', 2070)

    # Carica Projector finale (Step 25000)
    proj = TextAnchoredProjectionV8().to(device='cuda', dtype=torch.bfloat16)
    ckpt_path = '/workspace/checkpoints/proj_v8_final.pt'
    sd = torch.load(ckpt_path, map_location='cpu')
    proj.load_state_dict(sd.get('proj', sd), strict=False)
    proj.eval()
    print(f"  Proiettore V8.5 finale caricato da {ckpt_path} (gamma={proj.gamma.item():.4f})", flush=True)

    # Carica Dataset e Cache
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

    eval_results = []
    print("\n=== AVVIO VALUTAZIONE ACUSTICA & TRASCRIZIONE SU 5 CAMPIONI ===", flush=True)

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
            text_str = str(tbw_entry).strip() if tbw_entry is not None else "Buongiorno."

        h_raw = torch.from_numpy(np.asarray(va_raw[sample_idx][:n_real], dtype=np.float32)).to('cuda')
        h_tx = torch.from_numpy(np.asarray(va_htx[sample_idx], dtype=np.float32)).to('cuda')

        with torch.no_grad():
            h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))
            icl = build_official_prefill(text_str)
            L_prefill = icl.shape[1]

            codes_prev = codes[:-1]
            agg_audio = sum(talker.get_input_embeddings()(codes_prev[:, 0:1]) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](codes_prev[:, g:g+1]) for g in range(16)).transpose(0, 1)

            # ----------------------------------------------------
            # 1. Native Talker Base (Teacher Forcing, a=0.0)
            # ----------------------------------------------------
            inp_nat = torch.cat([icl, agg_audio], dim=1)
            out_nat = talker.model(inputs_embeds=inp_nat, use_cache=False)
            h_nat = out_nat.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]
            cb0_lg_nat = talker.codec_head(h_nat)
            pred_cb0_nat = cb0_lg_nat.argmax(dim=-1)[0]
            loss_cb0_nat = float(F.cross_entropy(cb0_lg_nat.reshape(-1, cb0_lg_nat.shape[-1]), codes[:, 0].reshape(-1)).item())
            acc_nat = float((pred_cb0_nat == codes[:, 0]).float().mean().item())

            tdecay = [0.8 * (0.92 ** k) for k in range(15)]
            pred_codes_nat = []
            for t in range(T):
                c0_tok = int(pred_cb0_nat[t].item())
                resid_t = sub.sample_acoustic_codes(h_nat[0:1, t:t+1], torch.tensor([[c0_tok]], device='cuda'), top_k=50, top_p=0.95, temperatures_per_level=tdecay)
                pred_codes_nat.append([c0_tok] + resid_t[0, 0].cpu().tolist())

            full_codes_nat = torch.clamp(torch.tensor(pred_codes_nat, device='cuda', dtype=torch.long).unsqueeze(0), 0, 2047)
            out_dec_nat = tok_decoder.decode(full_codes_nat)
            wav_nat = out_dec_nat.audio_values[0].cpu().numpy().astype(np.float32)
            p_peak_nat = float(np.percentile(np.abs(wav_nat), 99.7))
            wav_norm_nat = np.clip(wav_nat / (p_peak_nat * 1.05), -1.0, 1.0) if p_peak_nat > 1e-6 else wav_nat
            wav_path_nat = os.path.join(out_dir, f"sample_{sample_idx}_tf_native.wav")
            sf.write(wav_path_nat, wav_norm_nat, 24000)
            segs_nat, _ = whisper_model.transcribe(wav_path_nat, language="it", temperature=0.0)
            whisper_nat = " ".join([s.text for s in segs_nat]).strip()
            wer_nat = compute_wer(text_str, whisper_nat)

            # ----------------------------------------------------
            # 2. Omni Projector (Teacher Forcing, a=1.0)
            # ----------------------------------------------------
            audio_embeds_omni = agg_audio + h_cond[:, 1:T, :]
            inp_omni = torch.cat([icl, audio_embeds_omni], dim=1)
            out_omni = talker.model(inputs_embeds=inp_omni, use_cache=False)
            h_omni = out_omni.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]
            cb0_lg_omni = talker.codec_head(h_omni)
            pred_cb0_omni = cb0_lg_omni.argmax(dim=-1)[0]
            loss_cb0_omni = float(F.cross_entropy(cb0_lg_omni.reshape(-1, cb0_lg_omni.shape[-1]), codes[:, 0].reshape(-1)).item())
            acc_omni = float((pred_cb0_omni == codes[:, 0]).float().mean().item())

            pred_codes_omni = []
            for t in range(T):
                c0_tok = int(pred_cb0_omni[t].item())
                resid_t = sub.sample_acoustic_codes(h_omni[0:1, t:t+1], torch.tensor([[c0_tok]], device='cuda'), top_k=50, top_p=0.95, temperatures_per_level=tdecay)
                pred_codes_omni.append([c0_tok] + resid_t[0, 0].cpu().tolist())

            full_codes_omni = torch.clamp(torch.tensor(pred_codes_omni, device='cuda', dtype=torch.long).unsqueeze(0), 0, 2047)
            out_dec_omni = tok_decoder.decode(full_codes_omni)
            wav_omni = out_dec_omni.audio_values[0].cpu().numpy().astype(np.float32)
            p_peak_omni = float(np.percentile(np.abs(wav_omni), 99.7))
            wav_norm_omni = np.clip(wav_omni / (p_peak_omni * 1.05), -1.0, 1.0) if p_peak_omni > 1e-6 else wav_omni
            wav_path_omni = os.path.join(out_dir, f"sample_{sample_idx}_tf_omni.wav")
            sf.write(wav_path_omni, wav_norm_omni, 24000)
            segs_omni, _ = whisper_model.transcribe(wav_path_omni, language="it", temperature=0.0)
            whisper_omni = " ".join([s.text for s in segs_omni]).strip()
            wer_omni = compute_wer(text_str, whisper_omni)

            # ----------------------------------------------------
            # 3. Omni Projector (Free-Running Open-Loop Generation)
            # ----------------------------------------------------
            out_free_prefill = talker.model(inputs_embeds=icl, use_cache=True)
            past_kv = out_free_prefill.past_key_values
            last_h = out_free_prefill.last_hidden_state[:, -1:, :]

            gen_free_codes = []
            past_cb0 = []
            for t in range(T):
                cb0_lg_free = talker.codec_head(last_h)[:, -1, :]
                cb0_tok_free = sample_top_p_top_k(cb0_lg_free, top_p=0.92, top_k=50, temperature=0.8, rep_penalty=1.25, past_tokens=past_cb0)
                c0_f_val = int(cb0_tok_free.item())
                past_cb0.append(c0_f_val)

                resid_t_f = sub.sample_acoustic_codes(last_h, torch.tensor([[c0_f_val]], device='cuda'), top_k=50, top_p=0.95, temperatures_per_level=tdecay)
                res_toks_f = [c0_f_val] + resid_t_f[0, 0].cpu().tolist()
                gen_free_codes.append(res_toks_f)

                if t < T - 1:
                    cur_codes_t = torch.tensor(res_toks_f, device='cuda')
                    agg_t = talker.get_input_embeddings()(cur_codes_t[0:1].unsqueeze(0))
                    for g in range(1, 16):
                        agg_t = agg_t + talker.code_predictor.get_input_embeddings()[g-1](cur_codes_t[g:g+1].unsqueeze(0))
                    next_embed = agg_t.transpose(0, 1) + h_cond[:, t+1:t+2, :]
                    out_step = talker.model(inputs_embeds=next_embed, past_key_values=past_kv, use_cache=True)
                    past_kv = out_step.past_key_values
                    last_h = out_step.last_hidden_state[:, -1:, :]

            full_codes_free = torch.clamp(torch.tensor(gen_free_codes, device='cuda', dtype=torch.long).unsqueeze(0), 0, 2047)
            out_dec_free = tok_decoder.decode(full_codes_free)
            wav_free = out_dec_free.audio_values[0].cpu().numpy().astype(np.float32)
            p_peak_free = float(np.percentile(np.abs(wav_free), 99.7))
            wav_norm_free = np.clip(wav_free / (p_peak_free * 1.05), -1.0, 1.0) if p_peak_free > 1e-6 else wav_free
            wav_path_free = os.path.join(out_dir, f"sample_{sample_idx}_free_omni.wav")
            sf.write(wav_path_free, wav_norm_free, 24000)
            segs_free, _ = whisper_model.transcribe(wav_path_free, language="it", temperature=0.0)
            whisper_free = " ".join([s.text for s in segs_free]).strip()
            wer_free = compute_wer(text_str, whisper_free)

            cos_sim = float(F.cosine_similarity(h_omni.float(), h_nat.float(), dim=-1).mean().item())
            norm_ratio = float((h_omni.norm(dim=-1).mean() / (h_nat.norm(dim=-1).mean() + 1e-6)).item())

            result_item = {
                'sample_idx': sample_idx,
                'target_text': text_str,
                'whisper_nat': whisper_nat,
                'wer_nat': wer_nat,
                'loss_cb0_nat': loss_cb0_nat,
                'acc_nat': acc_nat,
                'whisper_omni_tf': whisper_omni,
                'wer_omni_tf': wer_omni,
                'loss_cb0_omni': loss_cb0_omni,
                'acc_omni': acc_omni,
                'whisper_omni_free': whisper_free,
                'wer_omni_free': wer_free,
                'cos_sim': cos_sim,
                'norm_ratio': norm_ratio,
                'delta_loss': loss_cb0_omni - loss_cb0_nat,
                'wav_nat': wav_path_nat,
                'wav_omni_tf': wav_path_omni,
                'wav_omni_free': wav_path_free
            }
            eval_results.append(result_item)

            print(f"\n==================== [CAMPIONE {sample_idx}] ====================")
            print(f"  TARGET TESTUALE              : \"{text_str}\"")
            print(f"  ASR TALKER NATIVO (TF a=0.0) : \"{whisper_nat}\" [WER={wer_nat*100:.1f}%, Loss={loss_cb0_nat:.3f}, Acc={acc_nat*100:.1f}%]")
            print(f"  ASR OMNI V8.5 (TF a=1.0)     : \"{whisper_omni}\" [WER={wer_omni*100:.1f}%, Loss={loss_cb0_omni:.3f}, Acc={acc_omni*100:.1f}%]")
            print(f"  ASR OMNI V8.5 (FREE OPENLOOP): \"{whisper_free}\" [WER={wer_free*100:.1f}%]")
            print(f"  DELTA LOSS OMNI vs NATIVO    : {loss_cb0_omni - loss_cb0_nat:+.3f} (Cos={cos_sim:.4f}, Norm={norm_ratio:.4f})")
            print(f"  WAV SALVATI                  : {wav_path_free} | {wav_path_omni}")

    # Salva report JSON
    rep_path = os.path.join(out_dir, "v8_final_evaluation_report.json")
    with open(rep_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"\n=== REPORT COMPLETO SALVATO IN {rep_path} ===", flush=True)

if __name__ == '__main__':
    main()
