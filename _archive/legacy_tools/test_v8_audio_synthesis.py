#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sintesi Audio e Trascrizione Whisper per Proiettore V8 (1000 step da Scratch)
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

def sample_cb0_with_penalty(logits, past_tokens, temperature=0.75, top_k=40, top_p=0.90, rep_penalty=1.2):
    l = logits.squeeze().clone().float()
    if l.shape[0] > 2048:
        l[2048:] = -1e9
    if past_tokens and rep_penalty != 1.0:
        for tok in past_tokens[-16:]:
            if 0 <= tok < l.shape[0]:
                l[tok] = l[tok] / rep_penalty if l[tok] > 0 else l[tok] * rep_penalty
    l = l / max(temperature, 1e-4)
    if top_k > 0 and top_k < l.shape[0]:
        topk_thresh = torch.topk(l, top_k)[0][-1]
        l[l < topk_thresh] = -1e9
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(l, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        l[indices_to_remove] = -1e9
    probs = F.softmax(l, dim=-1)
    if torch.isnan(probs).any() or probs.sum() == 0:
        return int(l.argmax().item())
    return int(torch.multinomial(probs, num_samples=1).item())

def main():
    out_dir = '/workspace/samples_v8'
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = '/workspace/checkpoints/proj_v8_final.pt'

    print("=" * 70)
    print("=== SINTESI AUDIO & TRASCRIZIONE WHISPER V8 (1000 STEP) ===")
    print("=" * 70)

    # 1. Caricamento Modelli
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

    snap_tok = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*'))[-1]
    tok_decoder = Qwen3TTSTokenizerV2Model.from_pretrained(snap_tok).cuda().eval()

    gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    tok_gemma = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True) if len(gemma_dirs) > 0 else None

    CFG = teacher.config
    TCFG = CFG.talker_config
    lang_map = getattr(TCFG, 'codec_language_id', {})
    lang_id = lang_map.get('italian', 2070)

    sub, _ = QS.load_qwen_subtalker('/workspace/qwen_subtalker', device='cuda', dtype=torch.bfloat16)

    # 2. Caricamento Proiettore V8
    proj = TextAnchoredProjectionV8().to(device='cuda', dtype=torch.bfloat16)
    sd = torch.load(ckpt_path, map_location='cpu')
    proj.load_state_dict(sd.get('proj', sd), strict=False)
    proj.eval()
    print(f"  * Proiettore V8 caricato con successo da {ckpt_path} (gamma={proj.gamma.item():.4f})")

    # 3. Dati
    _, _, agent_codes, tbw = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')
    ck = torch.load('/workspace/cache_ars_full_v2.pt', map_location='cpu', weights_only=False, mmap=True)
    va_raw = ck['va_raw']; va_htx = ck['va_htx']

    N_all = agent_codes.shape[1]
    val_start = int(0.90 * N_all)
    seg_ends = json.load(open('/workspace/data/train_tokens_cv26_it_40.8h_31916s_hyperparams.json')).get('SEGMENT_ENDS', [])
    for e in seg_ends:
        if e >= val_start: val_start = e; break

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

    whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    print("\n--- GENERAZIONE CAMPIONI AUDIO DI VALIDAZIONE ---")
    for sample_idx in range(3):
        c0 = val_start + sample_idx * 64
        codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
        codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
        n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
        if n_real < 16: continue
        codes = codes[:n_real]

        gi = (val_start // 64) + sample_idx
        text_str = tok_gemma.decode(tbw[gi], skip_special_tokens=True).strip() if (gi < len(tbw) and isinstance(tbw[gi], list)) else "Buongiorno."
        h_raw = torch.from_numpy(np.asarray(va_raw[sample_idx][:n_real], dtype=np.float32)).to('cuda')
        h_tx = torch.from_numpy(np.asarray(va_htx[sample_idx], dtype=np.float32)).to('cuda')

        with torch.no_grad():
            T = codes.shape[0]
            h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))

            icl = build_official_prefill(text_str)
            L_prefill = icl.shape[1]

            # Autoregressive generation
            out = talker.model(inputs_embeds=icl, use_cache=True)
            kv = out.past_key_values
            past_h = out.last_hidden_state[:, -1:, :]

            tdecay = [0.8 * (0.92 ** k) for k in range(15)]
            pred_codes = []
            past_cb0 = []

            for t in range(T):
                lg0 = talker.codec_head(past_h)
                cb0_token = sample_cb0_with_penalty(lg0, past_cb0, temperature=0.75, top_k=40, top_p=0.90)
                past_cb0.append(cb0_token)

                resid_t = sub.sample_acoustic_codes(past_h.to(torch.bfloat16), torch.tensor([[cb0_token]], device='cuda'), top_k=50, top_p=0.95, temperatures_per_level=tdecay)
                sub_tokens = [cb0_token] + resid_t[0, 0].cpu().tolist()
                pred_codes.append(sub_tokens)

                if t < T - 1:
                    cp = torch.tensor(sub_tokens, device='cuda', dtype=torch.long)
                    agg = sum(talker.get_input_embeddings()(cp[g:g+1].reshape(1,1)) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](cp[g:g+1].reshape(1,1)) for g in range(16))
                    in_t = agg + h_cond[:, t+1:t+2, :]
                    out_s = talker.model(inputs_embeds=in_t, use_cache=True, past_key_values=kv)
                    kv = out_s.past_key_values
                    past_h = out_s.last_hidden_state[:, -1:, :]

            full_codes = torch.tensor(pred_codes, device='cuda', dtype=torch.long).unsqueeze(0) # (1, T, 16)
            out_dec = tok_decoder.decode(full_codes)
            wav_tensor = out_dec.audio_values[0]
            wav_arr = wav_tensor.cpu().numpy().astype(np.float32)

            p_peak = float(np.percentile(np.abs(wav_arr), 99.7))
            wav_norm = np.clip(wav_arr / (p_peak * 1.05), -1.0, 1.0) if p_peak > 1e-6 else wav_arr
            wav_path = os.path.join(out_dir, f"sample_v8_val_{sample_idx}.wav")
            sf.write(wav_path, wav_norm, 24000)

            # Whisper ASR
            segments, _ = whisper_model.transcribe(wav_path, language="it", temperature=0.0)
            whisper_text = " ".join([s.text for s in segments]).strip()

            print(f"\n[CAMPIONE {sample_idx}]")
            print(f"  Testo Target  : \"{text_str}\"")
            print(f"  Trascrizione  : \"{whisper_text}\"")
            print(f"  WAV Salvato   : {wav_path}")

if __name__ == '__main__':
    main()
