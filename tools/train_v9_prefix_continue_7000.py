#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma4-Omni V9: Continue Training for 7000 Steps (from Step 3000 to 10000)
Warmstarted from /workspace/checkpoints/v9_prefix_final.pt.
Architecture: 211.83M Deep Causal Transformer Projector with Prefix KV-Cache, MTP, Scheduled Sampling & Moshi 100:1 Rebalancing.
"""

import sys, os, glob, json, math, argparse, time, random
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

# -------------------------------------------------------------
# 1. ARCHITETTURA V9: DEEP CAUSAL TRANSFORMER PROJECTOR (211.83M)
# -------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len):
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim=2048, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim=2048, ffn_dim=5632):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim=2048, num_heads=16, ffn_dim=5632):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.attn = CausalSelfAttention(hidden_dim, num_heads)
        self.norm2 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.ffn = SwiGLUFFN(hidden_dim, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class DeepCausalTransformerProjectorV9(nn.Module):
    def __init__(self, raw_dim=1536, text_dim=1536, hidden_dim=2048, num_layers=4, num_heads=16, ffn_dim=5632):
        super().__init__()
        self.audio_in = nn.Linear(raw_dim, hidden_dim, bias=False)
        self.text_in = nn.Linear(text_dim, hidden_dim, bias=False)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ffn_dim) for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(hidden_dim, eps=1e-6)

    def forward(self, h_raw, h_tx):
        if h_tx.dim() == 2:
            h_tx = h_tx.unsqueeze(1)
        x = self.audio_in(h_raw) + self.text_in(h_tx)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

# -------------------------------------------------------------
# 2. METRICHE ACUSTICHE E FISICHE (ADR-021)
# -------------------------------------------------------------
def compute_acoustic_physical_metrics(wav, sr=24000):
    if len(wav) == 0 or np.all(wav == 0):
        return {'crest_factor': 0.0, 'hnr_db': -50.0, 'thd_n_pct': 100.0, 'spectral_flatness': 1.0}

    peak = float(np.max(np.abs(wav)))
    rms = float(np.sqrt(np.mean(wav ** 2))) + 1e-9
    crest_factor = peak / rms

    frame_len = int(0.040 * sr)
    hop_len = int(0.020 * sr)
    hnr_list = []

    for start in range(0, len(wav) - frame_len, hop_len):
        frame = wav[start : start + frame_len]
        f_rms = np.sqrt(np.mean(frame ** 2))
        if f_rms < 0.01: continue

        frame_win = frame * np.hanning(len(frame))
        r = np.correlate(frame_win, frame_win, mode='full')
        r = r[len(r) // 2 :]

        min_lag = int(sr / 400)
        max_lag = int(sr / 60)
        if max_lag >= len(r): continue
        peak_idx = min_lag + np.argmax(r[min_lag:max_lag])
        r0 = r[0] + 1e-9
        r_max = r[peak_idx]

        if r_max > 0 and r0 > r_max:
            hnr_val = 10.0 * np.log10(r_max / (r0 - r_max + 1e-9))
            hnr_list.append(hnr_val)

    mean_hnr = float(np.mean(hnr_list)) if len(hnr_list) > 0 else 0.0
    thd_n = float(100.0 / (1.0 + 10.0 ** (mean_hnr / 10.0))) if mean_hnr > -30.0 else 99.0

    fft_mag = np.abs(np.fft.rfft(wav)) + 1e-9
    geom_mean = np.exp(np.mean(np.log(fft_mag)))
    arith_mean = np.mean(fft_mag)
    spectral_flatness = float(geom_mean / arith_mean)

    return {
        'crest_factor': round(crest_factor, 2),
        'hnr_db': round(mean_hnr, 2),
        'thd_n_pct': round(thd_n, 2),
        'spectral_flatness': round(spectral_flatness, 4),
    }

# -------------------------------------------------------------
# 3. TRAINING ENGINE V9 (CONTINUE 7000 STEPS)
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-step', type=int, default=3000)
    parser.add_argument('--steps', type=int, default=7000)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lambda-r', type=float, default=0.01) # Moshi 100:1
    parser.add_argument('--p-scheduled', type=float, default=0.15)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--out', type=str, default='/workspace/checkpoints')
    parser.add_argument('--samples-out', type=str, default='/workspace/samples_v9_prefix')
    parser.add_argument('--warmstart', type=str, default='/workspace/checkpoints/v9_prefix_final.pt')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.samples_out, exist_ok=True)

    print("==================================================================", flush=True)
    print("  GEMMA-4 OMNI: V9 TRANSFORMER (CONTINUE 7000 STEPS: 3000 -> 10000)", flush=True)
    print(f"  start={args.start_step} steps={args.steps} lr={args.lr} lambda_r={args.lambda_r}", flush=True)
    print(f"  warmstart={args.warmstart}", flush=True)
    print("==================================================================", flush=True)

    # 1. Caricamento Qwen3-TTS (100% Frozen)
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

    for p in talker.parameters(): p.requires_grad = False
    for p in talker.code_predictor.parameters(): p.requires_grad = False

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

    # 2. Inizializzazione Deep Causal Transformer V9 con Warmstart
    proj = DeepCausalTransformerProjectorV9().to(device='cuda', dtype=torch.bfloat16)
    if os.path.exists(args.warmstart):
        sd = torch.load(args.warmstart, map_location='cpu')
        proj.load_state_dict(sd.get('proj', sd), strict=False)
        print(f"  [WARMSTART] Pesi Step 3000 caricati con successo da {args.warmstart}", flush=True)
    else:
        print(f"  [ATTENZIONE] Checkpoint {args.warmstart} non trovato, inizializzazione da zero.", flush=True)

    total_steps = args.start_step + args.steps
    optimizer = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    # 3. Dati
    _, _, agent_codes, tbw = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')
    ck = torch.load('/workspace/cache_ars_full_v2.pt', map_location='cpu', weights_only=False, mmap=True)
    tr_raw = ck['tr_raw']; tr_htx = ck['tr_htx']
    va_raw = ck['va_raw']; va_htx = ck['va_htx']

    N_all = agent_codes.shape[1]
    val_start = int(0.90 * N_all)
    seg_ends = json.load(open('/workspace/data/train_tokens_cv26_it_40.8h_31916s_hyperparams.json')).get('SEGMENT_ENDS', [])
    for e in seg_ends:
        if e >= val_start: val_start = e; break

    N_train = val_start // 64
    N_val = min(100, (N_all - val_start) // 64)

    whisper_dirs = glob.glob('/workspace/.hf_cache/models--Systran--faster-whisper-small/snapshots/*')
    whisper_path = whisper_dirs[0] if len(whisper_dirs) > 0 else "small"
    whisper_model = WhisperModel(whisper_path, device="cuda", compute_type="float16", download_root="/workspace/.hf_cache")

    def build_official_prefill_embeddings(text_str):
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

        return torch.cat([talker_input_embed, text_part], dim=1), bos_part

    # 4. In-Loop Validation Split
    def run_validation_eval(val_samples=10):
        proj.eval()
        v_cb0_loss, v_cb0_acc = 0.0, 0.0
        v_res_loss, v_res_acc = 0.0, 0.0

        with torch.no_grad():
            for v_i in range(val_samples):
                c0 = val_start + v_i * 64
                codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
                codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
                codes = torch.clamp(codes, 0, 2047)
                n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
                if n_real < 16: continue
                codes = codes[:n_real]
                T = n_real

                gi = (val_start // 64) + v_i
                tbw_entry = tbw[gi] if gi < len(tbw) else None
                if isinstance(tbw_entry, (list, np.ndarray)) and tok_gemma is not None:
                    text_str = tok_gemma.decode(tbw_entry, skip_special_tokens=True).strip()
                else:
                    text_str = str(tbw_entry).strip() if tbw_entry is not None else "Testo di validazione."

                h_raw_s = torch.from_numpy(np.asarray(va_raw[v_i][:n_real], dtype=np.float32)).to('cuda')
                h_tx_s = torch.from_numpy(np.asarray(va_htx[v_i], dtype=np.float32)).to('cuda')

                icl_part, bos_part = build_official_prefill_embeddings(text_str)
                h_pfx = proj(h_raw_s.unsqueeze(0).to(torch.bfloat16), h_tx_s.unsqueeze(0).to(torch.bfloat16))

                pfx_embeds = torch.cat([icl_part, h_pfx, bos_part], dim=1)
                L_pfx = pfx_embeds.shape[1]

                codes_in = codes[:-1]
                agg_audio_in = sum(talker.get_input_embeddings()(codes_in[:, 0:1]) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](codes_in[:, g:g+1]) for g in range(16)).transpose(0, 1)

                full_inputs = torch.cat([pfx_embeds, agg_audio_in], dim=1)
                out_talker = talker.model(inputs_embeds=full_inputs, use_cache=False)
                h_omni = out_talker.last_hidden_state[:, L_pfx - 1 : L_pfx - 1 + T, :]

                cb0_lg = talker.codec_head(h_omni)
                loss_cb0 = F.cross_entropy(cb0_lg.reshape(-1, cb0_lg.shape[-1]), codes[:, 0].reshape(-1))
                cb0_acc = (cb0_lg.argmax(dim=-1) == codes[:, 0]).float().mean().item() * 100

                sub_lg, loss_res = talker.forward_sub_talker_finetune(codes, h_omni[0])
                res_acc = (sub_lg.argmax(dim=-1) == codes[:, 1:]).float().mean().item() * 100

                v_cb0_loss += loss_cb0.item()
                v_cb0_acc += cb0_acc
                v_res_loss += loss_res.item()
                v_res_acc += res_acc

        proj.train()
        return {
            'cb0_loss': v_cb0_loss / val_samples,
            'cb0_acc': v_cb0_acc / val_samples,
            'res_loss': v_res_loss / val_samples,
            'res_acc': v_res_acc / val_samples,
        }

    # 5. Generazione & Validazione Acustica Fisica Completa con EOS/VAD Trimming
    def run_acoustic_physical_eval(step_num):
        print(f"\n--- [VALUTAZIONE ACUSTICA & FISICA V9 @ STEP {step_num}] ---", flush=True)
        proj.eval()

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
        h_raw_sample = torch.from_numpy(np.asarray(va_raw[sample_idx][:n_real], dtype=np.float32)).to('cuda')
        h_tx_sample = torch.from_numpy(np.asarray(va_htx[sample_idx], dtype=np.float32)).to('cuda')

        with torch.no_grad():
            icl_part, bos_part = build_official_prefill_embeddings(text_str)
            h_pfx = proj(h_raw_sample.unsqueeze(0).to(torch.bfloat16), h_tx_sample.unsqueeze(0).to(torch.bfloat16))

            pfx_embeds = torch.cat([icl_part, h_pfx, bos_part], dim=1)
            out_pfx = talker.model(inputs_embeds=pfx_embeds, use_cache=True)
            past_kv = out_pfx.past_key_values
            last_h = out_pfx.last_hidden_state[:, -1:, :]

            codes_gen = []
            seen_tokens = []
            for t in range(T):
                cb0_lg = talker.codec_head(last_h)[:, -1, :]
                cb0_lg = cb0_lg / 0.75 # Calibrated temperature
                cb0_lg[0, 2048:] = -float('Inf')

                # Repetition penalty
                if len(seen_tokens) > 0:
                    for tok in set(seen_tokens[-8:]):
                        if cb0_lg[0, tok] > 0:
                            cb0_lg[0, tok] /= 1.15
                        else:
                            cb0_lg[0, tok] *= 1.15

                probs_0 = F.softmax(cb0_lg, dim=-1)
                c0_val = int(torch.multinomial(probs_0, 1).item())
                seen_tokens.append(c0_val)

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
                codes_gen.append(full_t)

                if t < T - 1:
                    cur_codes_t = torch.tensor(full_t, device='cuda')
                    agg_t = talker.get_input_embeddings()(cur_codes_t[0:1].unsqueeze(0))
                    for g in range(1, 16):
                        agg_t = agg_t + talker.code_predictor.get_input_embeddings()[g-1](cur_codes_t[g:g+1].unsqueeze(0))

                    next_embed = agg_t.transpose(0, 1)
                    out_step = talker.model(inputs_embeds=next_embed, past_key_values=past_kv, use_cache=True)
                    past_kv = out_step.past_key_values
                    last_h = out_step.last_hidden_state[:, -1:, :]

            t_gen = torch.clamp(torch.tensor(codes_gen, device='cuda', dtype=torch.long).unsqueeze(0), 0, 2047)
            wav = tok_decoder.decode(t_gen).audio_values[0].detach().cpu().numpy().astype(np.float32)

            # Clean EOS / VAD Trimming (Stop trailing burst)
            chunk_size = int(0.080 * 24000)
            energies = [np.sqrt(np.mean(wav[i:i+chunk_size]**2)) for i in range(0, len(wav), chunk_size)]
            voice_end_idx = len(energies)
            for idx in range(len(energies)-1, 10, -1):
                if energies[idx] > 0.015:
                    voice_end_idx = min(len(energies), idx + 4)
                    break
            trim_sample = min(len(wav), voice_end_idx * chunk_size)
            clean_wav = wav[:trim_sample].copy()
            if len(clean_wav) > 1200:
                fade_len = int(0.040 * 24000)
                clean_wav[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

            p_peak = float(np.percentile(np.abs(clean_wav), 99.7))
            clean_wav = np.clip(clean_wav / (p_peak * 1.05), -1.0, 1.0) if p_peak > 1e-6 else clean_wav
            wav_path = os.path.join(args.samples_out, f"sample_v9_prefix_step{step_num}.wav")
            sf.write(wav_path, clean_wav, 24000)

            phys_metrics = compute_acoustic_physical_metrics(clean_wav, 24000)
            segs, _ = whisper_model.transcribe(wav_path, language="it", temperature=0.0)
            asr_text = " ".join([s.text for s in segs]).strip()

            print(f"  [AUDIO V9 STEP {step_num}]: \"{asr_text}\" -> {wav_path}", flush=True)
            print(f"  [FISICA ACUSTICA]: Crest Factor: {phys_metrics['crest_factor']} (Target >5.0) | "
                  f"HNR: {phys_metrics['hnr_db']} dB (Target >3.0) | THD+N: {phys_metrics['thd_n_pct']}% (Target <30%) | "
                  f"Flatness: {phys_metrics['spectral_flatness']}", flush=True)

            report = {
                'step': step_num,
                'target_text': text_str,
                'asr_transcription': asr_text,
                'acoustic_metrics': phys_metrics,
                'wav_path': wav_path,
            }
            json_path = os.path.join(args.samples_out, f"report_v9_prefix_step{step_num}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        proj.train()
        return report

    # 6. Training Loop V9
    print(f"\n[TRAINING] Avvio Training V9 Transformer Prefix KV (Step {args.start_step+1} a {total_steps})...", flush=True)
    t_start = time.time()
    proj.train()

    curr_step = args.start_step

    for local_step in range(1, args.steps + 1):
        curr_step += 1
        optimizer.zero_grad()
        loss_batch = 0.0
        cb0_batch = 0.0
        res_batch = 0.0
        cb0_acc_batch = 0.0
        res_acc_batch = 0.0

        for b in range(args.batch):
            idx = (curr_step * args.batch + b) % N_train
            c0 = idx * 64
            codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
            codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
            codes = torch.clamp(codes, 0, 2047)
            n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
            if n_real < 16: continue
            codes = codes[:n_real]
            T = n_real

            tbw_entry = tbw[idx] if idx < len(tbw) else None
            if isinstance(tbw_entry, (list, np.ndarray)) and tok_gemma is not None:
                text_str = tok_gemma.decode(tbw_entry, skip_special_tokens=True).strip()
            else:
                text_str = str(tbw_entry).strip() if tbw_entry is not None else "Buongiorno."

            h_raw_sample = torch.from_numpy(np.asarray(tr_raw[idx][:n_real], dtype=np.float32)).to('cuda')
            h_tx_sample = torch.from_numpy(np.asarray(tr_htx[idx], dtype=np.float32)).to('cuda')

            icl_part, bos_part = build_official_prefill_embeddings(text_str)
            h_pfx = proj(h_raw_sample.unsqueeze(0).to(torch.bfloat16), h_tx_sample.unsqueeze(0).to(torch.bfloat16))

            pfx_embeds = torch.cat([icl_part, h_pfx, bos_part], dim=1)
            L_pfx = pfx_embeds.shape[1]

            # SCHEDULED SAMPLING / AUTOREGRESSIVE ROLLOUT
            codes_in = codes[:-1].clone()
            if random.random() < args.p_scheduled:
                with torch.no_grad():
                    out_probe = talker.model(inputs_embeds=pfx_embeds, use_cache=True)
                    last_h_probe = out_probe.last_hidden_state[:, -1:, :]
                    c0_probe = talker.codec_head(last_h_probe).argmax(dim=-1).squeeze()
                    if c0_probe.numel() == 1:
                        codes_in[0, 0] = torch.clamp(c0_probe, 0, 2047)

            agg_audio_in = sum(talker.get_input_embeddings()(codes_in[:, 0:1]) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](codes_in[:, g:g+1]) for g in range(16)).transpose(0, 1)

            full_inputs = torch.cat([pfx_embeds, agg_audio_in], dim=1)
            out_talker = talker.model(inputs_embeds=full_inputs, use_cache=False)
            h_omni = out_talker.last_hidden_state[:, L_pfx - 1 : L_pfx - 1 + T, :]

            cb0_lg = talker.codec_head(h_omni)
            loss_cb0 = F.cross_entropy(cb0_lg.reshape(-1, cb0_lg.shape[-1]), codes[:, 0].reshape(-1))

            sub_lg, loss_res = talker.forward_sub_talker_finetune(codes, h_omni[0])

            # MOSHI 100:1 REBALANCING (lambda_r = 0.01)
            loss_total = (1.0 * loss_cb0 + args.lambda_r * loss_res) / args.batch
            loss_total.backward()

            with torch.no_grad():
                cb0_acc = (cb0_lg.argmax(dim=-1) == codes[:, 0]).float().mean().item() * 100
                res_acc = (sub_lg.argmax(dim=-1) == codes[:, 1:]).float().mean().item() * 100

            loss_batch += loss_total.item() * args.batch
            cb0_batch += loss_cb0.item()
            res_batch += loss_res.item()
            cb0_acc_batch += cb0_acc
            res_acc_batch += res_acc

        torch.nn.utils.clip_grad_norm_(proj.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t_start

        if local_step % 10 == 0 or local_step == 1:
            loss_avg = loss_batch / args.batch
            cb0_avg = cb0_batch / args.batch
            res_avg = res_batch / args.batch
            cb0_acc_avg = cb0_acc_batch / args.batch
            res_acc_avg = res_acc_batch / args.batch
            print(f"[V9-TRANSFORMER {curr_step:05d}/{total_steps:05d}] Loss: {loss_avg:.3f} | CB0: {cb0_avg:.3f} (Acc: {cb0_acc_avg:.1f}%) | "
                  f"Res: {res_avg:.3f} (Acc: {res_acc_avg:.1f}%) ({elapsed/local_step:.2f}s/step)", flush=True)

        if local_step % args.eval_every == 0:
            val_metrics = run_validation_eval(val_samples=10)
            print(f"\n>>> [EVAL V9 @ STEP {curr_step:05d}] Val CB0 Loss: {val_metrics['cb0_loss']:.3f} | Val CB0 Acc: {val_metrics['cb0_acc']:.1f}% | "
                  f"Val Res Loss: {val_metrics['res_loss']:.3f} | Val Res Acc: {val_metrics['res_acc']:.1f}% <<<\n", flush=True)

        if local_step % args.save_every == 0 or local_step == args.steps:
            ckpt_path = os.path.join(args.out, f"v9_prefix_step{curr_step}.pt")
            torch.save({
                'step': curr_step,
                'proj': proj.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(args)
            }, ckpt_path)
            print(f"  [CHECKPOINT] Salvato con successo in {ckpt_path}", flush=True)
            run_acoustic_physical_eval(curr_step)

    final_ckpt = os.path.join(args.out, "v9_prefix_final.pt")
    torch.save({'step': curr_step, 'proj': proj.state_dict()}, final_ckpt)
    print(f"\n[COMPLETATO] Training V9 Transformer 10000 Step terminato! Checkpoint: {final_ckpt}", flush=True)

if __name__ == '__main__':
    main()
