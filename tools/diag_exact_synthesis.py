#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic script to test exact synthesis and inspect shapes, codes, and raw wav output.
"""
import sys, os, glob
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

# --- V9 Projector ---
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

def main():
    print("Loading models for diagnostic...", flush=True)
    gemma_snaps = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True)
    if len(gemma_snaps) > 0:
        KP.MODEL_NAME = os.path.dirname(gemma_snaps[0])
    duplex_model, tok_gemma = KP.load_gemma4_duplex_system()
    KP.load_gemma4_duplex_checkpoint(duplex_model, "/workspace/checkpoints")
    duplex_model.eval()

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
    v9_ckpt = '/workspace/checkpoints/v9_prefix_final.pt'
    sd_v9 = torch.load(v9_ckpt, map_location='cpu')
    proj.load_state_dict(sd_v9.get('proj', sd_v9), strict=False)

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

    test_prompt = "Ciao, sono Gemma 4. Posso aiutarti a scrivere il tuo racconto noir."
    print(f"\nSynthesizing test prompt: '{test_prompt}'", flush=True)

    with torch.no_grad():
        input_ids = tok_gemma(text="<start_of_turn>user\nCiao<end_of_turn>\n<start_of_turn>model\n", return_tensors="pt")["input_ids"].to('cuda')
        out_gen = tok_gemma(text=test_prompt, return_tensors="pt")["input_ids"].to('cuda')
        full_out = duplex_model.base_model(out_gen, output_hidden_states=True, return_dict=True)
        lh = full_out.hidden_states[-1]
        h_tx = lh.mean(dim=1).to(torch.bfloat16)

        T_pfx_len = 16
        seq_embeds = lh
        seq_trans = seq_embeds.transpose(1, 2)
        h_raw_style = F.interpolate(seq_trans, size=T_pfx_len, mode='linear', align_corners=False).transpose(1, 2)
        h_raw_style = duplex_model._apply_causal_conv(h_raw_style.to(next(duplex_model.causal_conv.parameters()).dtype)).to(torch.bfloat16)
        h_pfx = proj(h_raw_style, h_tx)

        T_gen_frames = 50
        icl_part, bos_part, tts_pad_embed = build_official_prefill_embeddings(test_prompt)
        pfx_embeds = torch.cat([icl_part, h_pfx, bos_part], dim=1)

        out_pfx = talker.model(inputs_embeds=pfx_embeds, use_cache=True)
        past_kv = out_pfx.past_key_values
        last_h = out_pfx.last_hidden_state[:, -1:, :]

        codes_gen = []
        for t in range(T_gen_frames):
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

            if t < T_gen_frames - 1:
                cur_codes_t = torch.tensor(full_t, device='cuda')
                agg_t = talker.get_input_embeddings()(cur_codes_t[0:1].unsqueeze(0))
                for g in range(1, 16):
                    agg_t = agg_t + talker.code_predictor.get_input_embeddings()[g-1](cur_codes_t[g:g+1].unsqueeze(0))
                next_embed = agg_t.transpose(0, 1) + tts_pad_embed
                out_step = talker.model(inputs_embeds=next_embed, past_key_values=past_kv, use_cache=True)
                past_kv = out_step.past_key_values
                last_h = out_step.last_hidden_state[:, -1:, :]

        codes_tensor = torch.tensor(codes_gen, device='cuda', dtype=torch.long)
        print(f"codes_tensor shape: {codes_tensor.shape}")
        print(f"Sample codes at t=0: {codes_gen[0]}")
        print(f"Sample codes at t=10: {codes_gen[10]}")

        t_gen = torch.clamp(codes_tensor.unsqueeze(0), 0, 2047)
        print(f"t_gen shape for tok_decoder.decode: {t_gen.shape}")

        dec_out = tok_decoder.decode(t_gen)
        print(f"dec_out type: {type(dec_out)}")
        if hasattr(dec_out, 'audio_values'):
            raw_wav = dec_out.audio_values[0].detach().cpu().numpy().astype(np.float32)
        else:
            raw_wav = dec_out[0].detach().cpu().numpy().astype(np.float32)

        print(f"raw_wav shape: {raw_wav.shape}, min: {np.min(raw_wav)}, max: {np.max(raw_wav)}, mean: {np.mean(raw_wav)}, std: {np.std(raw_wav)}")
        sf.write("/workspace/samples/e2e_benchmark/diag_raw_wav.wav", raw_wav, 24000)

        # Trimming check
        chunk_size = int(0.080 * 24000)
        energies = [np.sqrt(np.mean(raw_wav[i:i+chunk_size]**2)) for i in range(0, len(raw_wav), chunk_size)]
        print(f"energies (first 10): {energies[:10]}")
        print(f"energies (last 10): {energies[-10:]}")
        voice_end_idx = len(energies)
        for idx in range(len(energies)-1, 10, -1):
            if energies[idx] > 0.015:
                voice_end_idx = min(len(energies), idx + 4)
                break
        print(f"voice_end_idx: {voice_end_idx} / {len(energies)}")
        trim_sample = min(len(raw_wav), voice_end_idx * chunk_size)
        clean_wav = raw_wav[:trim_sample].copy()
        if len(clean_wav) > 1200:
            fade_len = int(0.040 * 24000)
            clean_wav[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

        p_peak = float(np.percentile(np.abs(clean_wav), 99.7))
        print(f"p_peak: {p_peak}")
        norm_wav = np.clip(clean_wav / (p_peak * 1.05), -1.0, 1.0) if p_peak > 1e-6 else clean_wav
        print(f"norm_wav shape: {norm_wav.shape}, min: {np.min(norm_wav)}, max: {np.max(norm_wav)}, mean: {np.mean(norm_wav)}, std: {np.std(norm_wav)}")
        sf.write("/workspace/samples/e2e_benchmark/diag_norm_wav.wav", norm_wav, 24000)
        print("Wrote /workspace/samples/e2e_benchmark/diag_norm_wav.wav successfully!")

if __name__ == '__main__':
    main()
