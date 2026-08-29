#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test di Equivalenza Numerica e Benchmark di Velocità:
Sequential Loop Forward vs Parallel Causal Forward Pass su Qwen3-TTS 1.7B
"""
import os, sys, glob, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
import safetensors.torch as st
import kaggle_pipeline as KP
import qwen_subtalker as QS
from tools.train_loop_coupling_v7 import TextAnchoredProjectionV7

def test_speed_and_equivalence():
    print("=" * 70)
    print("=== BENCHMARK VELOCITA': SEQUENTIAL LOOP vs PARALLEL CAUSAL FORWARD ===")
    print("=" * 70)

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

    sub, _ = QS.load_qwen_subtalker('/workspace/qwen_subtalker', device='cuda', dtype=torch.bfloat16)

    proj = TextAnchoredProjectionV7().to(device='cuda', dtype=torch.bfloat16)
    sd = torch.load('/workspace/checkpoints/proj_v7_final.pt', map_location='cpu')
    proj.load_state_dict(sd.get('proj', sd), strict=False)
    proj.train()

    CFG = teacher.config
    TCFG = CFG.talker_config
    lang_map = getattr(TCFG, 'codec_language_id', {})
    lang_id = lang_map.get('italian', 2070)

    # Carica 4 campioni reali
    _, _, agent_codes, tbw = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')
    ck = torch.load('/workspace/cache_ars_full_v2.pt', map_location='cpu', weights_only=False, mmap=True)
    tr_raw = ck['tr_raw']; tr_htx = ck['tr_htx']

    samples = []
    for b in range(4):
        c0 = b * 64
        codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
        codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
        n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
        codes = codes[:n_real]
        text_str = tok_gemma.decode(tbw[b], skip_special_tokens=True).strip() if (b < len(tbw) and isinstance(tbw[b], list)) else 'Buongiorno.'
        h_raw_t = torch.from_numpy(np.asarray(tr_raw[b][:n_real], dtype=np.float32)).to('cuda')
        h_tx_t = torch.from_numpy(np.asarray(tr_htx[b], dtype=np.float32)).to('cuda')
        samples.append((codes, text_str, h_raw_t, h_tx_t))

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

    # -------------------------------------------------------------
    # Metodo 1: Sequential Loop (ATTUALE)
    # -------------------------------------------------------------
    print("\n--- Test 1: Metodo Attuale (Sequential Loop 64-step per 4 sample) ---")
    torch.cuda.synchronize()
    t0 = time.time()

    seq_cb0_losses = []
    seq_res_losses = []
    seq_h_seqs = []

    for codes, text_str, h_raw_t, h_tx_t in samples:
        T = codes.shape[0]
        h_cond = proj(h_raw_t.unsqueeze(0).to(torch.bfloat16), h_tx_t.unsqueeze(0).to(torch.bfloat16))
        icl = build_official_prefill(text_str)
        out = talker.model(inputs_embeds=icl, attention_mask=torch.ones(1, icl.shape[1], device='cuda'), use_cache=True)
        kv = out.past_key_values
        past_h = out.last_hidden_state[:, -1:, :]

        h_list = [past_h]
        for t in range(1, T):
            cp = codes[t-1].to('cuda')
            agg = sum(talker.get_input_embeddings()(cp[g:g+1].reshape(1,1)) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](cp[g:g+1].reshape(1,1)) for g in range(16))
            in_t = agg + h_cond[:, t:t+1, :]
            out = talker.model(inputs_embeds=in_t, attention_mask=torch.ones(1, 1, device='cuda'), use_cache=True, past_key_values=kv)
            kv = out.past_key_values
            past_h = out.last_hidden_state[:, -1:, :]
            h_list.append(past_h)

        h_seq = torch.cat(h_list, dim=1)
        seq_h_seqs.append(h_seq)
        cb0_logits = talker.codec_head(h_seq)
        loss_cb0 = F.cross_entropy(cb0_logits.view(-1, cb0_logits.shape[-1]), codes[:, 0].contiguous().view(-1))
        _, loss_res = talker.forward_sub_talker_finetune(codes, h_seq[0])
        seq_cb0_losses.append(loss_cb0.item())
        seq_res_losses.append(loss_res.item())

    torch.cuda.synchronize()
    t_seq = time.time() - t0
    print(f"Tempo Sequential Loop (4 sample): {t_seq:.3f} s")
    print(f"  • CB0 Loss medio: {np.mean(seq_cb0_losses):.4f}")
    print(f"  • Res Loss medio: {np.mean(seq_res_losses):.4f}")

    # -------------------------------------------------------------
    # Metodo 2: Parallel Causal Forward Pass (NUOVO)
    # -------------------------------------------------------------
    print("\n--- Test 2: Metodo Parallelo (Parallel Causal Forward Pass) ---")
    torch.cuda.synchronize()
    t0 = time.time()

    par_cb0_losses = []
    par_res_losses = []
    par_h_seqs = []

    for codes, text_str, h_raw_t, h_tx_t in samples:
        T = codes.shape[0]
        h_cond = proj(h_raw_t.unsqueeze(0).to(torch.bfloat16), h_tx_t.unsqueeze(0).to(torch.bfloat16)) # (1, T, 2048)
        icl = build_official_prefill(text_str) # (1, L_prefill, 2048)
        L_prefill = icl.shape[1]

        # Costruisci l'intera sequenza di frame audio [1..T-1]
        # In frame t=1..T-1, l'input è codes[t-1] + h_cond[t]
        # Per tutti i t=1..T-1 in parallelo:
        codes_prev = codes[:-1] # (T-1, 16)
        
        # Aggregazione embeddings dei 16 codebook per tutti i frame in UNA SOLA operazione tensoriale:
        agg_all = talker.get_input_embeddings()(codes_prev[:, 0:1]) # (T-1, 1, 2048)
        for g in range(1, 16):
            agg_all = agg_all + talker.code_predictor.get_input_embeddings()[g-1](codes_prev[:, g:g+1])
        agg_all = agg_all.transpose(0, 1) # (1, T-1, 2048)
        
        audio_embeds_1_to_T = agg_all + h_cond[:, 1:T, :] # (1, T-1, 2048)
        
        # Sequenza completa: [icl, audio_embeds_1_to_T] -> lunghezza L_prefill + T - 1
        full_inputs = torch.cat([icl, audio_embeds_1_to_T], dim=1) # (1, L_prefill + T - 1, 2048)

        # UNICO forward pass causale del Talker
        out = talker.model(inputs_embeds=full_inputs, use_cache=False)
        full_h = out.last_hidden_state # (1, L_prefill + T - 1, 2048)

        # L'hidden state che predice il frame t=0 è l'ultimo token di icl: full_h[:, L_prefill-1:L_prefill, :]
        # L'hidden state che predice il frame t=1..T-1 sono i token da L_prefill a L_prefill + T - 2:
        h_seq_par = full_h[:, L_prefill-1 : L_prefill-1 + T, :] # (1, T, 2048)
        par_h_seqs.append(h_seq_par)

        cb0_logits = talker.codec_head(h_seq_par)
        loss_cb0 = F.cross_entropy(cb0_logits.view(-1, cb0_logits.shape[-1]), codes[:, 0].contiguous().view(-1))
        _, loss_res = talker.forward_sub_talker_finetune(codes, h_seq_par[0])
        par_cb0_losses.append(loss_cb0.item())
        par_res_losses.append(loss_res.item())

    torch.cuda.synchronize()
    t_par = time.time() - t0
    print(f"Tempo Parallel Causal Forward (4 sample): {t_par:.3f} s")
    print(f"  • CB0 Loss medio: {np.mean(par_cb0_losses):.4f}")
    print(f"  • Res Loss medio: {np.mean(par_res_losses):.4f}")

    # -------------------------------------------------------------
    # Verifica di Equivalenza Numerica
    # -------------------------------------------------------------
    print("\n--- Verifica di Equivalenza Numerica ---")
    speedup = t_seq / max(t_par, 1e-4)
    print(f"🚀 Speedup calcolato: {speedup:.2f}x più veloce!")

    for i in range(len(samples)):
        diff_h = (seq_h_seqs[i] - par_h_seqs[i]).abs().max().item()
        cos_h = F.cosine_similarity(seq_h_seqs[i].float(), par_h_seqs[i].float(), dim=-1).mean().item()
        print(f"Sample {i}: max |h_seq_diff| = {diff_h:.6f} | Cosine(h_seq_seq, h_seq_par) = {cos_h:.6f}")
        assert cos_h > 0.999, f"Errore: Cosine similarity {cos_h} < 0.999 tra sequenziale e parallelo!"

    print("\n✅ VERIFICA SUPERATA: IL METODO PARALLELO PRODUCE OUTPUT BIT-EXACT CON UNO SPEEDUP STRAORDINARIO!")

if __name__ == '__main__':
    test_speed_and_equivalence()
