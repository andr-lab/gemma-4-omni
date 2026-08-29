#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# DIAGNOSTICS MANIFOLD & ORACLE PROBE — V7 ADAPTED
# Source-Driven & Doubt-Driven Development Verification Tool
#
# References:
#   - Qwen3-TTS Technical Report (arXiv:2601.15621) - Dual-Track LM & 12Hz 16-RVQ
#   - ADR-013: Post-Train Manifold & Oracle Diagnostics
# ============================================================================
import os, sys, json, glob, math, argparse
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
sys.path.append('/workspace/.extra_pkgs')

import kaggle_pipeline as KP
import qwen_subtalker as QS

# ---- shim compat transformers 5.x (ADR-011) ----
import transformers.modeling_rope_utils as _ropeu
def _default_rope_init(config, device=None, seq_len=None, **kwargs):
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    _head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    _dim = int(_head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, _dim, 2, dtype=torch.int64).float().to(device) / _dim))
    return inv_freq, 1.0
_ropeu.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
_ropeu.ROPE_INIT_FUNCTIONS["interleaved"] = _default_rope_init

from transformers import AutoTokenizer
import qwen_tts.core.models as _M
from qwen_tts.core.models import Qwen3TTSProcessor
import safetensors.torch as st

class TextAnchoredProjectionV7(nn.Module):
    def __init__(self, in_dim=1536, out_dim=2048, hidden_dim=2048, initial_gamma=0.25):
        super().__init__()
        self.audio_skip = nn.Linear(in_dim, out_dim, bias=False)
        self.audio_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.audio_norm = nn.LayerNorm(out_dim)

        self.text_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
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

def run_diagnostics(ckpt_path, prefix='/workspace/data/train_tokens_cv26_it_40.8h_31916s',
                    cache_path='/workspace/cache_ars_full_v2.pt', n_val=6):
    print(f"\n======================================================================")
    print(f"=== ESECUZIONE DIAGNOSTICA MANIFOLD SU: {os.path.basename(ckpt_path)} ===")
    print(f"======================================================================\n")

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

    gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    tok_gemma = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True) if len(gemma_dirs) > 0 else None

    # Carica Adapter
    proj = TextAnchoredProjectionV7().to(device='cuda', dtype=torch.bfloat16)
    sd = torch.load(ckpt_path, map_location='cpu')
    sd_proj = sd.get('proj', sd)
    proj.load_state_dict(sd_proj, strict=False)
    gamma_val = float(proj.gamma.item()) if hasattr(proj, 'gamma') else 1.0
    proj.eval()
    print(f"  * Modello caricato: {ckpt_path} (gamma={gamma_val:.4f})")

    # Dati
    _, _, agent_codes, tbw = KP.load_tokens_from_parquet(prefix + '.parquet')
    ck = torch.load(cache_path, map_location='cpu', weights_only=False, mmap=True)
    va_raw = ck['va_raw']
    va_htx = ck['va_htx']
    val_start = int(0.90 * agent_codes.shape[1])
    seg_ends = json.load(open(prefix + '_hyperparams.json')).get('SEGMENT_ENDS', [])
    for e in seg_ends:
        if e >= val_start: val_start = e; break

    CFG = teacher.config
    TCFG = CFG.talker_config
    CBOS = TCFG.codec_bos_id

    def build_prefill(text_str):
        assistant = f"<|im_start|>assistant\n{text_str}<|im_end|>\n<|im_start|>assistant\n"
        text_ids = proc_qwen(text=assistant, return_tensors='pt')['input_ids']
        tids = text_ids[:, :-5].to('cuda')
        text_emb = talker.text_projection(talker.get_text_embeddings()(tids))
        tts_eos = talker.text_projection(talker.get_text_embeddings()(torch.tensor([[CFG.tts_eos_token_id]], device='cuda', dtype=tids.dtype)))
        c_pad = talker.get_input_embeddings()(torch.tensor([[TCFG.codec_pad_id]], device='cuda', dtype=tids.dtype))
        t_pad = talker.text_projection(talker.get_text_embeddings()(torch.tensor([[CFG.tts_pad_token_id]], device='cuda', dtype=tids.dtype)))
        cbos_emb = talker.get_input_embeddings()(torch.tensor([[CBOS]], device='cuda', dtype=tids.dtype))
        icl = torch.cat([text_emb, tts_eos], dim=1) + c_pad.expand(1, text_emb.shape[1]+1, -1)
        return torch.cat([icl, cbos_emb + t_pad], dim=1)

    cos_sims, norm_ratios, ranks = [], [], []
    cb0_losses_1, cb0_losses_0, res_losses_1, res_losses_0, form_losses_1, form_losses_0 = [], [], [], [], [], []

    with torch.no_grad():
        for i in range(min(n_val, len(va_raw))):
            c0 = val_start + i * 64
            codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
            codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
            n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
            if n_real < 16: continue
            codes = codes[:n_real]

            gi = (val_start // 64) + i
            t_item = tbw[gi] if gi < len(tbw) else None
            text_str = tok_gemma.decode(t_item, skip_special_tokens=True).strip() if (t_item and isinstance(t_item, list)) else 'Buongiorno.'
            h_raw = torch.from_numpy(np.asarray(va_raw[i][:n_real], dtype=np.float32)).to('cuda')
            h_tx = torch.from_numpy(np.asarray(va_htx[i], dtype=np.float32)).to('cuda')
            h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))

            icl = build_prefill(text_str)

            # Passaggio Condizionato (alpha=1.0)
            out_1 = talker.model(inputs_embeds=icl, attention_mask=torch.ones(1, icl.shape[1], device='cuda'), use_cache=True)
            kv_1 = out_1.past_key_values
            ph_1_list = [out_1.last_hidden_state[:, -1:, :]]

            # Passaggio Nativo / Disaccoppiato (alpha=0.0)
            out_0 = talker.model(inputs_embeds=icl, attention_mask=torch.ones(1, icl.shape[1], device='cuda'), use_cache=True)
            kv_0 = out_0.past_key_values
            ph_0_list = [out_0.last_hidden_state[:, -1:, :]]

            for t in range(1, n_real):
                cp = codes[t-1].to('cuda')
                agg = sum(talker.get_input_embeddings()(cp[g:g+1].reshape(1,1)) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](cp[g:g+1].reshape(1,1)) for g in range(16))

                # Fwd 1
                o1 = talker.model(inputs_embeds=agg + h_cond[:, t:t+1, :], attention_mask=torch.ones(1, 1, device='cuda'), use_cache=True, past_key_values=kv_1)
                kv_1 = o1.past_key_values
                ph_1_list.append(o1.last_hidden_state[:, -1:, :])

                # Fwd 0
                o0 = talker.model(inputs_embeds=agg, attention_mask=torch.ones(1, 1, device='cuda'), use_cache=True, past_key_values=kv_0)
                kv_0 = o0.past_key_values
                ph_0_list.append(o0.last_hidden_state[:, -1:, :])

            ph_1 = torch.cat(ph_1_list, dim=1) # (1, T, 2048)
            ph_0 = torch.cat(ph_0_list, dim=1)

            # Metriche Geometriche tra i vettori di Talker (Omni vs Nativo)
            cos = F.cosine_similarity(ph_1, ph_0, dim=-1).mean().item()
            n_ratio = (ph_1.norm(dim=-1) / (ph_0.norm(dim=-1) + 1e-6)).mean().item()
            u, s, v = torch.svd(ph_1[0].float())
            eff_rank = float((s.sum()**2 / ((s**2).sum() + 1e-6)).item())

            cos_sims.append(cos)
            norm_ratios.append(n_ratio)
            ranks.append(eff_rank)

            # Cross-Entropy Losses
            lg_1 = talker.codec_head(ph_1)
            l_cb0_1 = F.cross_entropy(lg_1.reshape(-1, lg_1.shape[-1]), codes[:, 0]).item()
            slg_1, l_res_1 = talker.forward_sub_talker_finetune(codes, ph_1[0])
            l_form_1 = F.cross_entropy(slg_1[:, :3].reshape(-1, slg_1.shape[-1]), codes[:, 1:4].reshape(-1)).item()

            lg_0 = talker.codec_head(ph_0)
            l_cb0_0 = F.cross_entropy(lg_0.reshape(-1, lg_0.shape[-1]), codes[:, 0]).item()
            slg_0, l_res_0 = talker.forward_sub_talker_finetune(codes, ph_0[0])
            l_form_0 = F.cross_entropy(slg_0[:, :3].reshape(-1, slg_0.shape[-1]), codes[:, 1:4].reshape(-1)).item()

            cb0_losses_1.append(l_cb0_1); cb0_losses_0.append(l_cb0_0)
            res_losses_1.append(l_res_1.item()); res_losses_0.append(l_res_0.item())
            form_losses_1.append(l_form_1); form_losses_0.append(l_form_0)

    avg_cos = float(np.mean(cos_sims))
    avg_nratio = float(np.mean(norm_ratios))
    avg_rank = float(np.mean(ranks))

    print(f"📊 [PROBE GEOMETRICO SUL MANIFOLD]")
    print(f"  • Cosine Similarity (Omni past_hidden vs Nativo): {avg_cos:.4f}")
    print(f"  • L2 Norm Ratio (Omni / Nativo)                 : {avg_nratio:.4f} (target ~1.00)")
    print(f"  • SVD Effective Rank                            : {avg_rank:.1f} / 2048")
    print(f"\n📉 [METRICHE CROSS-ENTROPY]")
    print(f"  • CB0 Loss        : Omni(alpha=1.0) = {np.mean(cb0_losses_1):.3f} | Nativo(alpha=0.0) = {np.mean(cb0_losses_0):.3f} | Delta = {np.mean(cb0_losses_1)-np.mean(cb0_losses_0):+.3f}")
    print(f"  • Formanti CB1..3 : Omni(alpha=1.0) = {np.mean(form_losses_1):.3f} | Nativo(alpha=0.0) = {np.mean(form_losses_0):.3f} | Delta = {np.mean(form_losses_1)-np.mean(form_losses_0):+.3f}")
    print(f"  • Residui Totali  : Omni(alpha=1.0) = {np.mean(res_losses_1):.3f} | Nativo(alpha=0.0) = {np.mean(res_losses_0):.3f} | Delta = {np.mean(res_losses_1)-np.mean(res_losses_0):+.3f}")
    print(f"======================================================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='/workspace/checkpoints/proj_v7_step100.pt')
    args = parser.parse_args()
    run_diagnostics(args.checkpoint)
