#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Test: Gemma 4 Thinker + Projector V9 + Qwen3-TTS Voice Output
Loads Gemma 4 Duplex Model + V9 Deep Causal Transformer (Step 3000) + Qwen3-TTS
Generates speech response to a user query and saves clean WAV.
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

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/qwen3tts_src')
os.environ.setdefault("PIPELINE_ENV", "local3090")

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

# --- V9 Projector Definition ---
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

def compute_physical_metrics(wav, sr=24000):
    if len(wav) == 0: return {}
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
        min_lag = int(sr / 400); max_lag = int(sr / 60)
        if max_lag >= len(r): continue
        peak_idx = min_lag + np.argmax(r[min_lag:max_lag])
        r0 = r[0] + 1e-9; r_max = r[peak_idx]
        if r_max > 0 and r0 > r_max:
            hnr_val = 10.0 * np.log10(r_max / (r0 - r_max + 1e-9))
            hnr_list.append(hnr_val)

    mean_hnr = float(np.mean(hnr_list)) if len(hnr_list) > 0 else 0.0
    thd_n = float(100.0 / (1.0 + 10.0 ** (mean_hnr / 10.0))) if mean_hnr > -30.0 else 99.0
    return {'crest_factor': round(crest_factor, 2), 'hnr_db': round(mean_hnr, 2), 'thd_n_pct': round(thd_n, 2)}

def main():
    print("===============================================================", flush=True)
    print("  TEST END-TO-END: GEMMA 4 THINKER + PROJECTOR V9 + QWEN3-TTS", flush=True)
    print("===============================================================", flush=True)

    # 1. Carica Qwen3-TTS
    print("[1/4] Caricamento Qwen3-TTS Talker & Vocoder...", flush=True)
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
    lang_id = getattr(TCFG, 'codec_language_id', {}).get('italian', 2070)

    # 2. Carica Projector V9 (Step 3000 / Final)
    print("[2/4] Caricamento Projector V9 (Step 3000)...", flush=True)
    proj = DeepCausalTransformerProjectorV9().to(device='cuda', dtype=torch.bfloat16).eval()
    ckpt_path = '/workspace/checkpoints/v9_prefix_final.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = '/workspace/checkpoints/v9_prefix_step1000.pt'
    sd = torch.load(ckpt_path, map_location='cpu')
    proj.load_state_dict(sd.get('proj', sd), strict=False)
    print(f"  Pesi caricati da: {ckpt_path}", flush=True)

    # 3. Carica Cache & Dataset di Validazione E2E
    print("[3/4] Caricamento Dati E2E da Cache e Tokenizer Gemma...", flush=True)
    ck = torch.load('/workspace/cache_ars_full_v2.pt', map_location='cpu', weights_only=False, mmap=True)
    va_raw = ck['va_raw']; va_htx = ck['va_htx']
    _, _, agent_codes, tbw = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')

    gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    tok_gemma = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True) if len(gemma_dirs) > 0 else None

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

    # 4. Generazione E2E su 3 campioni di test diversificati
    test_indices = [1, 3, 5]
    os.makedirs('/workspace/samples_e2e_test', exist_ok=True)

    print("\n[4/4] Esecuzione Generazione End-to-End...", flush=True)

    for s_idx in test_indices:
        h_raw_s = torch.from_numpy(np.asarray(va_raw[s_idx], dtype=np.float32)).to('cuda')
        h_tx_s = torch.from_numpy(np.asarray(va_htx[s_idx], dtype=np.float32)).to('cuda')
        T = min(64, h_raw_s.shape[0])
        h_raw_s = h_raw_s[:T]

        # Testo associato
        val_start = int(0.90 * agent_codes.shape[1])
        gi = (val_start // 64) + s_idx
        tbw_entry = tbw[gi] if gi < len(tbw) else None
        if isinstance(tbw_entry, (list, np.ndarray)) and tok_gemma is not None:
            text_str = tok_gemma.decode(tbw_entry, skip_special_tokens=True).strip()
        else:
            text_str = str(tbw_entry).strip() if tbw_entry is not None else "Buongiorno."

        if not text_str or len(text_str) < 3:
            text_str = "Si tratta del secondo singolo promozionale nella storia della band."

        print(f"\n--- [TEST E2E CAMPIONE #{s_idx}] ---", flush=True)
        print(f"  Testo Target Gemma: \"{text_str}\"", flush=True)

        with torch.no_grad():
            t0 = time.time()
            icl_part, bos_part = build_official_prefill_embeddings(text_str)
            h_pfx = proj(h_raw_s.unsqueeze(0).to(torch.bfloat16), h_tx_s.unsqueeze(0).to(torch.bfloat16))

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
            gen_time = time.time() - t0

            # Taglio pulito VAD/EOS
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

            out_wav_path = f"/workspace/samples_e2e_test/sample_e2e_val{s_idx}.wav"
            sf.write(out_wav_path, clean_wav, 24000)

            # ASR Faster Whisper
            segs, _ = whisper_model.transcribe(out_wav_path, language="it", temperature=0.0)
            asr_text = " ".join([s.text for s in segs]).strip()

            phys = compute_physical_metrics(clean_wav, 24000)

            print(f"  [ASR TRASCRIZIONE]: \"{asr_text}\"", flush=True)
            print(f"  [METRICHE]: Crest Factor: {phys.get('crest_factor')} | HNR: {phys.get('hnr_db')} dB | THD+N: {phys.get('thd_n_pct')}%", flush=True)
            print(f"  [TEMPO]: {gen_time:.2f}s per {len(clean_wav)/24000:.2f}s audio ({len(clean_wav)/24000/gen_time:.2f}x RTF)", flush=True)

    print("\n[COMPLETATO] Test End-to-End terminato con successo!", flush=True)

if __name__ == '__main__':
    main()
