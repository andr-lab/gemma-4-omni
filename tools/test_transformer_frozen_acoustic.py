#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empirical Test: Deep Causal Transformer Projector with 100% FROZEN Acoustic Block
(Both Talker 1.7B and SubTalker code_predictor are completely frozen).
Tests if the sequence-level causal attention can optimize both semantic and acoustic codes.
"""

import sys, os, glob, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
import safetensors.torch as st
import kaggle_pipeline as KP

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0).to(x.dtype), emb.sin().unsqueeze(0).to(x.dtype)

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim=2048, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim=2048, ffn_dim=4096):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class CausalTransformerBlock(nn.Module):
    def __init__(self, hidden_dim=2048, num_heads=8, ffn_dim=4096):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.attn = CausalSelfAttention(hidden_dim, num_heads)
        self.norm2 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.ffn = SwiGLUFFN(hidden_dim, ffn_dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x

class DeepCausalTransformerProjectorV9(nn.Module):
    def __init__(self, raw_dim=1536, text_dim=1536, hidden_dim=2048, num_layers=4, num_heads=8, ffn_dim=4096, initial_gamma=0.25):
        super().__init__()
        self.audio_in = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.text_in = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        nn.init.zeros_(self.text_in[-1].weight)
        nn.init.zeros_(self.text_in[-1].bias)

        self.rope = RotaryEmbedding(dim=hidden_dim // num_heads)
        self.layers = nn.ModuleList([
            CausalTransformerBlock(hidden_dim=hidden_dim, num_heads=num_heads, ffn_dim=ffn_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.gamma = nn.Parameter(torch.tensor(initial_gamma, dtype=torch.float32))

    def forward(self, h_raw, h_tx):
        if h_tx.dim() == 2:
            h_tx = h_tx.unsqueeze(1)
        x = self.audio_in(h_raw) + self.text_in(h_tx)
        cos, sin = self.rope(x)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return (self.gamma.to(x.dtype)) * self.final_norm(x)

def main():
    print("=== TEST EMPIRICO: CAUSAL TRANSFORMER CON BLOCCO ACUSTICO 100% CONGELATO ===", flush=True)
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

    # BLOCCO ACUSTICO COMPLETO 100% CONGELATO
    for p in talker.parameters():
        p.requires_grad = False
    for p in talker.code_predictor.parameters():
        p.requires_grad = False

    tok_qwen = AutoTokenizer.from_pretrained(snap, use_fast=True)
    proc_qwen = _M.Qwen3TTSProcessor(tokenizer=tok_qwen)

    gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    tok_gemma = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True) if len(gemma_dirs) > 0 else None

    CFG = teacher.config
    TCFG = CFG.talker_config
    lang_map = getattr(TCFG, 'codec_language_id', {})
    lang_id = lang_map.get('italian', 2070)

    # Inizializza Proiettore V9 Causal Transformer (65.4M)
    proj = DeepCausalTransformerProjectorV9().to(device='cuda', dtype=torch.bfloat16)
    proj_params = sum(p.numel() for p in proj.parameters() if p.requires_grad)
    print(f"  Proiettore DeepCausalTransformerProjectorV9: {proj_params/1e6:.2f}M parametri addestrabili", flush=True)

    optimizer = torch.optim.AdamW(proj.parameters(), lr=1.5e-4, weight_decay=0.01)

    # Dati
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

    print("\n--- ESECUZIONE 50 STEP TRAINING TRANSFORMER (BLOCCO ACUSTICO FROZEN) ---", flush=True)
    proj.train()

    for step in range(1, 51):
        optimizer.zero_grad()
        loss_batch = 0.0
        cb0_batch = 0.0
        res_batch = 0.0
        cos_batch = 0.0
        cb0_acc_batch = 0.0

        batch_size = 4
        for b in range(batch_size):
            idx = (step * batch_size + b) % N_train
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

            h_cond = proj(h_raw_sample.unsqueeze(0).to(torch.bfloat16), h_tx_sample.unsqueeze(0).to(torch.bfloat16))
            icl = build_official_prefill(text_str)
            L_prefill = icl.shape[1]

            codes_prev = codes[:-1]
            agg_audio = sum(talker.get_input_embeddings()(codes_prev[:, 0:1]) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](codes_prev[:, g:g+1]) for g in range(16)).transpose(0, 1)

            audio_embeds_omni = agg_audio + h_cond[:, 1:T, :]
            inp_omni = torch.cat([icl, audio_embeds_omni], dim=1)
            out_omni = talker.model(inputs_embeds=inp_omni, use_cache=False)
            h_omni = out_omni.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]

            with torch.no_grad():
                inp_nat = torch.cat([icl, agg_audio], dim=1)
                out_nat = talker.model(inputs_embeds=inp_nat, use_cache=False)
                h_nat = out_nat.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]

            # CB0 Loss
            cb0_lg = talker.codec_head(h_omni)
            loss_cb0 = F.cross_entropy(cb0_lg.reshape(-1, cb0_lg.shape[-1]), codes[:, 0].reshape(-1))

            # Acoustic Residual Loss (Frozen SubTalker)
            sub_lg, loss_res = talker.forward_sub_talker_finetune(codes, h_omni[0])

            # Manifold Loss
            cos_sim = F.cosine_similarity(h_omni.float(), h_nat.float().detach(), dim=-1).mean()
            loss_manifold = 1.0 - cos_sim
            norm_ratio = h_omni.norm(dim=-1).mean() / (h_nat.norm(dim=-1).mean() + 1e-6)
            loss_norm = (norm_ratio - 1.0) ** 2

            loss_total = (loss_cb0 + 0.50 * loss_res + 0.20 * loss_manifold + 0.10 * loss_norm) / batch_size
            loss_total.backward()

            loss_batch += loss_total.item() * batch_size
            cb0_batch += loss_cb0.item()
            res_batch += loss_res.item()
            cos_batch += cos_sim.item()

            with torch.no_grad():
                cb0_acc = (cb0_lg.argmax(dim=-1) == codes[:, 0]).float().mean().item() * 100
                cb0_acc_batch += cb0_acc

        torch.nn.utils.clip_grad_norm_(proj.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 10 == 0 or step == 50:
            print(f"Step {step:02d}/50 | CB0 Loss: {cb0_batch/batch_size:.3f} | Res Loss: {res_batch/batch_size:.3f} | Cos Sim: {cos_batch/batch_size:.4f} | CB0 Acc: {cb0_acc_batch/batch_size:.1f}% | gamma={proj.gamma.item():.4f}", flush=True)

if __name__ == '__main__':
    main()
