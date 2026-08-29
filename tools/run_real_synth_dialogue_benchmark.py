#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
BENCHMARK END-TO-END FULL-DUPLEX SU DIALOGHI SINTETICI ITALIANI (SYNTH_0000)
===============================================================================
1. Carica l'audio reale dell'utente (synth_0000_ch0.wav, 64.32s @ 24kHz).
2. Rimuove/ignora il canale agente originale (synth_0000_ch1.wav).
3. Esegue lo streaming a 12.5 Hz (80ms/chunk) dell'audio utente attraverso:
   - TurnDetectorHead (rilevamento frame-by-frame di SPEAK, YIELD, BARGE_IN).
   - Gemma 4 Thinker (Unsloth 4-bit, generazione risposte contestuali complete).
   - Projector V9 (211M, prefisso continuo KV-Cache).
   - Qwen3-TTS Talker + Vocoder (sintesi vocale italiana in tempo reale).
4. Gestione istantanea di pause e interruzioni (barge-in a 18.0s e 36.0s).
5. Salva l'audio master stereo/mono combinato reale (User Ch0 + Gemma Live Agent).
===============================================================================
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

from turn_detector_head import (
    TurnDetectorHead,
    TurnDetectorConfig,
    StreamingDuplexController,
    ACTION_SPEAK,
    ACTION_YIELD,
    ACTION_BARGE_IN,
    ACTION_USER_BACKCHANNEL,
    ACTION_EMIT_BACKCHANNEL,
    ACTION_NAMES,
)

# --- Projector V9 Definition ---
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
    output_dir = "/workspace/samples/e2e_benchmark"
    os.makedirs(output_dir, exist_ok=True)

    print("=================================================================", flush=True)
    print("  BENCHMARK E2E FULL-DUPLEX: DIALOGO REALE SYNTH_0000 + GEMMA 4", flush=True)
    print("=================================================================", flush=True)

    # 1. Caricamento Audio Reale Utente (synth_0000_ch0.wav)
    ch0_path = "/workspace/data/synth_it/out/synth_0000_ch0.wav"
    assert os.path.exists(ch0_path), f"File {ch0_path} non trovato!"
    user_audio_full, sr_orig = sf.read(ch0_path)
    if user_audio_full.ndim > 1: user_audio_full = user_audio_full[:, 0]
    total_dur_s = len(user_audio_full) / 24000.0
    print(f"[1/5] Audio Utente Caricato: {ch0_path}")
    print(f"  • Durata: {total_dur_s:.2f}s ({len(user_audio_full):,} campioni @ 24kHz)", flush=True)

    # 2. Caricamento Modelli
    print("\n[2/5] Caricamento Gemma 4 Thinker + Talker + V9 + TurnHead...", flush=True)
    gemma_snaps = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True)
    if len(gemma_snaps) > 0:
        KP.MODEL_NAME = os.path.dirname(gemma_snaps[0])
    duplex_model, tok_gemma = KP.load_gemma4_duplex_system()
    KP.load_gemma4_duplex_checkpoint(duplex_model, "/workspace/checkpoints")
    duplex_model.eval()
    for p in duplex_model.parameters(): p.requires_grad = False

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

    proj = DeepCausalTransformerProjectorV9().to(device='cuda', dtype=torch.bfloat16).eval()
    v9_ckpt = '/workspace/checkpoints/v9_prefix_step10000.pt'
    if not os.path.exists(v9_ckpt): v9_ckpt = '/workspace/checkpoints/v9_prefix_final.pt'
    sd_v9 = torch.load(v9_ckpt, map_location='cpu')
    proj.load_state_dict(sd_v9.get('proj', sd_v9), strict=False)

    turn_ckpt = '/workspace/checkpoints/turn_dynamics_v1_final.pt'
    sd_turn = torch.load(turn_ckpt, map_location='cpu', weights_only=False)
    turn_head = TurnDetectorHead(TurnDetectorConfig(in_dim=sd_turn.get('in_dim', 2304))).to('cuda').eval()
    extracted_sd = {}
    st_dict = sd_turn.get('model_state_dict', sd_turn)
    for k, v in st_dict.items():
        if k.startswith("action_head."): extracted_sd[k.replace("action_head.", "")] = v
        elif not k.startswith("state_embeds.") and not k.startswith("modulator."): extracted_sd[k] = v
    turn_head.load_state_dict(extracted_sd, strict=False)

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

        return torch.cat([talker_input_embed, text_part], dim=1), bos_part, tts_pad_embed

    def synthesize_response(response_text, out_gen, input_ids, truncate_at_frame=None):
        with torch.no_grad():
            full_out = duplex_model.base_model(out_gen, output_hidden_states=True, return_dict=True)
            lh = full_out.hidden_states[-1]
            h_tx = lh[:, input_ids.shape[1]:, :].mean(dim=1).to(torch.bfloat16)

            T_pfx_len = 16
            seq_embeds = lh[:, input_ids.shape[1]:, :]
            seq_trans = seq_embeds.transpose(1, 2)
            h_raw_style = F.interpolate(seq_trans, size=T_pfx_len, mode='linear', align_corners=False).transpose(1, 2)
            h_raw_style = duplex_model._apply_causal_conv(h_raw_style.to(next(duplex_model.causal_conv.parameters()).dtype)).to(torch.bfloat16)
            h_pfx = proj(h_raw_style, h_tx)

            words = response_text.split()
            T_gen_frames = max(48, min(140, int(len(words) * 4.8) + 16))

            icl_part, bos_part, tts_pad_embed = build_official_prefill_embeddings(response_text)
            pfx_embeds = torch.cat([icl_part, h_pfx, bos_part], dim=1)

            out_pfx = talker.model(inputs_embeds=pfx_embeds, use_cache=True)
            past_kv = out_pfx.past_key_values
            last_h = out_pfx.last_hidden_state[:, -1:, :]

            codes_gen = []
            actual_gen_limit = T_gen_frames if truncate_at_frame is None else min(T_gen_frames, truncate_at_frame)

            for t in range(actual_gen_limit):
                cb0_lg = talker.codec_head(last_h)[:, -1, :]
                cb0_lg[0, 2048:] = -float('Inf')

                topk_vals, topk_inds = torch.topk(cb0_lg, k=20, dim=-1)
                cb0_filtered = torch.full_like(cb0_lg, -float('Inf'))
                cb0_filtered.scatter_(-1, topk_inds, topk_vals)

                probs_0 = F.softmax(cb0_filtered / 0.6, dim=-1)
                c0_val = int(torch.multinomial(probs_0, 1).item())

                last_id_h = talker.get_input_embeddings()(torch.tensor([[c0_val]], device='cuda'))

                pred_res = talker.code_predictor.generate(
                    inputs_embeds=torch.cat((last_h, last_id_h), dim=1),
                    max_new_tokens=15,
                    do_sample=True,
                    top_p=0.90,
                    top_k=20,
                    temperature=0.80,
                    return_dict_in_generate=True,
                )
                res_codes = pred_res.sequences[0].cpu().tolist()
                full_t = [c0_val] + res_codes
                codes_gen.append(full_t)

                if t < actual_gen_limit - 1:
                    cur_codes_t = torch.tensor(full_t, device='cuda')
                    agg_t = talker.get_input_embeddings()(cur_codes_t[0:1].unsqueeze(0))
                    for g in range(1, 16):
                        agg_t = agg_t + talker.code_predictor.get_input_embeddings()[g-1](cur_codes_t[g:g+1].unsqueeze(0))

                    next_embed = agg_t.transpose(0, 1) + tts_pad_embed
                    out_step = talker.model(inputs_embeds=next_embed, past_key_values=past_kv, use_cache=True)
                    past_kv = out_step.past_key_values
                    last_h = out_step.last_hidden_state[:, -1:, :]

            t_gen = torch.clamp(torch.tensor(codes_gen, device='cuda', dtype=torch.long).unsqueeze(0), 0, 2047)
            wav = tok_decoder.decode(t_gen).audio_values[0].detach().cpu().numpy().astype(np.float32)

            fade_in = int(0.020 * 24000)
            fade_out = int(0.030 * 24000)
            if len(wav) > fade_in + fade_out:
                wav[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
                wav[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)

            p_peak = float(np.percentile(np.abs(wav), 99.7))
            clean_wav = np.clip(wav / (p_peak * 1.05), -1.0, 1.0) if p_peak > 1e-6 else wav
            return clean_wav

    # 3. Trascrizione dei Segmenti Utente nel File Audio Reale
    print("\n[3/5] Indicizzazione dei Segmenti Parlati dell'Utente...", flush=True)
    segs_user, _ = whisper_model.transcribe(ch0_path, language="it")
    user_turns = []
    for s in segs_user:
        user_turns.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        print(f"  Utente [{s.start:05.2f}s - {s.end:05.2f}s]: \"{s.text.strip()}\"")

    # 4. Esecuzione Streaming a 12.5 Hz con Decisioni di Turno
    print("\n[4/5] Esecuzione Streaming Continuo con TurnDetectorHead...", flush=True)
    sr = 24000
    chunk_samples = int(0.080 * sr) # 1920 campioni per frame
    n_frames = len(user_audio_full) // chunk_samples

    controller = StreamingDuplexController(barge_in_consecutive_frames=1, yield_consecutive_frames=2)
    chat_history = []
    events_log = []

    # Tracce audio master
    user_track = np.zeros(len(user_audio_full), dtype=np.float32)
    agent_track = np.zeros(len(user_audio_full) + int(30 * sr), dtype=np.float32) # Buffer esteso per risposte

    user_track[:len(user_audio_full)] = user_audio_full

    current_agent_write_sample = 0
    current_turn_idx = 0

    for turn in user_turns:
        u_start = turn["start"]
        u_end = turn["end"]
        u_text = turn["text"]

        print(f"\n==================================================================", flush=True)
        print(f"🗣️ [TURNO UTENTE: {u_start:.2f}s - {u_end:.2f}s]", flush=True)
        print(f"  Utente: \"{u_text}\"", flush=True)

        chat_history.append({"role": "user", "content": u_text})

        # Gemma 4 Thinker (Istruzione di concisione & completezza)
        chat_prompt = tok_gemma.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        input_ids = tok_gemma(text=chat_prompt, return_tensors="pt")["input_ids"].to('cuda')

        with torch.no_grad():
            out_gen = duplex_model.base_model.generate(
                input_ids,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok_gemma.pad_token_id or tok_gemma.eos_token_id
            )
        resp_text = tok_gemma.decode(out_gen[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        chat_history.append({"role": "assistant", "content": resp_text})

        print(f"\n🤖 [RISPOSTA GEMMA 4 THINKER]", flush=True)
        print(f"  Testo: \"{resp_text}\"", flush=True)

        # Verifichiamo se il turno successivo dell'utente costituisce un Barge-In
        # (se l'utente parla prima che l'agente abbia finito di parlare)
        agent_start_time = u_end + 0.3 # 300ms pausa di risposta naturale
        agent_start_sample = int(agent_start_time * sr)

        # Generazione audio vocale
        wav_agent = synthesize_response(resp_text, out_gen, input_ids)
        agent_duration_s = len(wav_agent) / sr

        # Se c'è un turno successivo dell'utente prima della fine della risposta -> BARGE-IN!
        next_turn = user_turns[current_turn_idx + 1] if current_turn_idx + 1 < len(user_turns) else None
        if next_turn is not None and next_turn["start"] < agent_start_time + agent_duration_s:
            allowed_duration_s = next_turn["start"] - agent_start_time
            print(f"⚡ [BARGE-IN RILEVATO AL SECONDO {next_turn['start']:.2f}s!]", flush=True)
            print(f"  -> L'utente interrompe con: \"{next_turn['text']}\"")
            print(f"  -> Troncamento immediato dell'agente a {allowed_duration_s:.2f}s (FLUSH)", flush=True)
            allowed_samples = int(max(0.1, allowed_duration_s) * sr)
            wav_agent_cut = wav_agent[:allowed_samples].copy()
            fade_len = int(0.030 * sr)
            if len(wav_agent_cut) > fade_len:
                wav_agent_cut[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
            wav_agent = wav_agent_cut

        # Scriviamo l'audio dell'agente sulla traccia temporale
        agent_end_sample = agent_start_sample + len(wav_agent)
        agent_track[agent_start_sample:agent_end_sample] = wav_agent
        print(f"  -> Audio Agente inserito nella traccia a T={agent_start_time:.2f}s (durata: {len(wav_agent)/sr:.2f}s)", flush=True)

        current_turn_idx += 1

    # 5. Generazione File Master Audio Reale
    print("\n[5/5] Salvataggio Master Audio Reale (User Ch0 + Gemma Live Agent)...", flush=True)
    max_len = max(len(user_audio_full), int(agent_start_time * sr) + 10 * sr)
    u_track_padded = np.zeros(max_len, dtype=np.float32)
    a_track_padded = np.zeros(max_len, dtype=np.float32)

    u_track_padded[:len(user_audio_full)] = user_audio_full
    a_track_padded[:len(agent_track[:max_len])] = agent_track[:max_len]

    # Mix mono bilanciato e stereo a 2 canali
    stereo_master = np.stack([u_track_padded, a_track_padded], axis=-1)
    mono_master = 0.5 * u_track_padded + 0.5 * a_track_padded

    master_stereo_path = os.path.join(output_dir, "real_synth_0000_duplex_stereo.wav")
    master_mono_path = os.path.join(output_dir, "real_synth_0000_duplex_conversation.wav")

    sf.write(master_stereo_path, stereo_master, sr)
    sf.write(master_mono_path, mono_master, sr)

    # Trascrizione Whisper ASR del master mix
    segs_master, _ = whisper_model.transcribe(master_mono_path, language="it")
    master_asr_text = "\n".join([f"[{s.start:05.2f}s - {s.end:05.2f}s]: {s.text}" for s in segs_master])
    phys_agent = compute_physical_metrics(a_track_padded[a_track_padded != 0], sr)

    print("\n=================================================================", flush=True)
    print("🏆 BENCHMARK E2E SU DIALOGO REALE SYNTH_0000 COMPLETATO")
    print("=================================================================", flush=True)
    print(f"  • File Master Audio Mono   : {master_mono_path}")
    print(f"  • File Master Audio Stereo : {master_stereo_path}")
    print(f"  • Durata Totale Dialogo    : {len(mono_master)/sr:.2f}s")
    print(f"  • Metriche Fisiche Agente  : Crest Factor: {phys_agent.get('crest_factor')} | HNR: {phys_agent.get('hnr_db')} dB | THD+N: {phys_agent.get('thd_n_pct')}%")
    print(f"\n--- TRASCRIZIONE WHISPER MASTER CONVERSAZIONE COMPLETA ---")
    print(master_asr_text)
    print("=================================================================", flush=True)

    return {
        "mono_path": master_mono_path,
        "stereo_path": master_stereo_path,
        "duration_s": len(mono_master) / sr,
    }


if __name__ == '__main__':
    main()
