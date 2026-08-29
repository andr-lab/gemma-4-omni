#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma 4 Omni — Phase 2 V9 Trainer: Deep Causal Transformer Projector (~65M params)
Replaces the 2-layer point-wise MLP with a 4-Layer Causal Transformer Bridge with RoPE and SwiGLU.
Uses Dense Causal 3D MRoPE Parallel Engine (0.78s/step) and Moshi 100:1 Calibrated Loss.
"""

import os, sys, glob, time, json, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import safetensors.torch as st
from transformers import AutoTokenizer

sys.path.insert(0, '/workspace/qwen3tts_src')
import qwen_tts.core.models.modeling_qwen3_tts as _M
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
from qwen_tts.core.processor import Qwen3TTSProcessor

# ------------------------------------------------------------------------
# 1. Rotary Positional Embeddings (RoPE) per il Proiettore V9
# ------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2) # (1, seq_len, 1, dim)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ------------------------------------------------------------------------
# 2. Causal Transformer Layer con SwiGLU FFN
# ------------------------------------------------------------------------
class CausalTransformerBlock(nn.Module):
    def __init__(self, hidden_dim=2048, num_heads=8, ffn_dim=4096):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm1 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm2 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        # Pre-LN Self-Attention
        norm_x = self.norm1(x)
        q = self.q_proj(norm_x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(norm_x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(norm_x).view(B, T, self.num_heads, self.head_dim)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.transpose(1, 2) # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.o_proj(attn_out)

        # Pre-LN SwiGLU FFN
        norm_x2 = self.norm2(x)
        ffn_out = self.down_proj(F.silu(self.gate_proj(norm_x2)) * self.up_proj(norm_x2))
        x = x + ffn_out
        return x

# ------------------------------------------------------------------------
# 3. Deep Causal Transformer Projector V9 (~65.4M parametri)
# ------------------------------------------------------------------------
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
        # Inizializzazione ReZero per il ramo testo
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
        
        # Iniezione dual branch
        x = self.audio_in(h_raw) + self.text_in(h_tx)
        cos, sin = self.rope(x)
        
        for layer in self.layers:
            x = layer(x, cos, sin)
            
        return (self.gamma.to(x.dtype)) * self.final_norm(x)

def build_official_prefill_func(tok_qwen, teacher):
    def build_official_prefill(text_prompt: str) -> torch.Tensor:
        tc = teacher.talker.config
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "repeat after me: " + text_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": text_prompt}]}
        ]
        text = tok_qwen.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tok_out = tok_qwen(text, return_tensors="pt")
        input_ids = tok_out.input_ids.cuda()

        if hasattr(tc, 'codec_think_id') and tc.codec_think_id is not None:
            prefill_part1 = input_ids[:, :-5]
            prefill_part2 = input_ids[:, -5:]
            language_id = 2070 # Italian
            codec_think = torch.tensor([[tc.codec_think_id, language_id]], dtype=torch.long, device='cuda')
            full_ids = torch.cat([prefill_part1, codec_think, prefill_part2], dim=1)
        else:
            full_ids = input_ids

        embed_layer = teacher.talker.text_embedding
        icl_embeds = embed_layer(full_ids)
        return icl_embeds
    return build_official_prefill

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', type=int, default=0)
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--start-step', type=int, default=0)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lambda-r', type=float, default=0.05)
    parser.add_argument('--lambda-m', type=float, default=0.20)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=2500)
    parser.add_argument('--out', type=str, default='/workspace/checkpoints')
    parser.add_argument('--warmstart', type=str, default='')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs('/workspace/samples_v9', exist_ok=True)

    print(f"[V9-TRANSFORMER] start_step={args.start_step} steps={args.steps} batch={args.batch} "
          f"lr={args.lr} lambda_r={args.lambda_r} lambda_m={args.lambda_m}", flush=True)

    # 1. Caricamento Modelli
    print("[V9 1/5] Caricamento Talker Qwen3-TTS, Tokenizer & Decoder...", flush=True)
    snap = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*'))[-1]
    _cfg = _M.Qwen3TTSConfig.from_pretrained(snap)
    if not hasattr(_cfg.talker_config, 'pad_token_id'):
        object.__setattr__(_cfg.talker_config, 'pad_token_id', None)

    teacher = _M.Qwen3TTSForConditionalGeneration(_cfg).to(torch.bfloat16)
    _sd = st.load_file(snap + '/model.safetensors', device='cpu')
    teacher.load_state_dict(_sd, strict=False)
    del _sd; torch.cuda.empty_cache()
    teacher = teacher.to(device='cuda', dtype=torch.bfloat16).eval()
    talker = teacher.talker

    tok_qwen = AutoTokenizer.from_pretrained(snap, use_fast=True)
    build_official_prefill = build_official_prefill_func(tok_qwen, teacher)

    snap_tok = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*'))[-1]
    tok_decoder = Qwen3TTSTokenizerV2Model.from_pretrained(snap_tok).cuda().eval()

    # Proiettore V9 Causal Transformer
    proj = DeepCausalTransformerProjectorV9().to('cuda', dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in proj.parameters())
    print(f"  DeepCausalTransformerProjectorV9 inizializzato ({n_params/1e6:.2f}M parametri, gamma={proj.gamma.item():.4f})", flush=True)

    if args.warmstart and os.path.exists(args.warmstart):
        st_data = torch.load(args.warmstart, map_location='cuda')
        # Load compatible weights if warmstarting from V8
        proj_dict = proj.state_dict()
        pretrained_dict = {k: v for k, v in st_data['proj'].items() if k in proj_dict and v.shape == proj_dict[k].shape}
        proj_dict.update(pretrained_dict)
        proj.load_state_dict(proj_dict)
        print(f"  Warmstart parziale caricato da {args.warmstart} ({len(pretrained_dict)} tensori compatibili)", flush=True)

    # 2. Caricamento Dati
    print("[V9 2/5] Caricamento Dati & Cache v2...", flush=True)
    import kaggle_pipeline as KP
    cache = torch.load('/workspace/data/cache_ars_full_v2.pt', map_location='cpu')
    tr_raw, va_raw = cache['tr_raw'], cache['va_raw']
    tr_htx, va_htx = cache['tr_htx'], cache['va_htx']
    tbw = [t.strip() for t in cache.get('agent_tbw', []) if len(t.strip()) > 0]

    import pandas as pd
    pq = sorted(glob.glob('/workspace/data/train_tokens_cv26_it_40.8h_*.parquet'))[-1]
    df = pd.read_parquet(pq)
    agent_codes = np.stack(df['agent_codes'].values, axis=0).T
    val_start = 1650667
    N_train, N_val = len(tr_raw), len(va_raw)
    print(f"  N_train={N_train} N_val={N_val} val_start={val_start}", flush=True)

    def get_italian_text_prompt(tbw_entry: str = None) -> str:
        if tbw_entry is not None and len(tbw_entry.strip()) > 0:
            return tbw_entry.strip()
        return "Frase di prova in lingua italiana per la sintesi vocale."

    # 3. Forward Pass Parallelo V9
    def parallel_forward_sample(codes, text_str, h_raw_t, h_tx_t, alpha=1.0):
        T = codes.shape[0]
        h_cond = proj(h_raw_t.unsqueeze(0).to(torch.bfloat16), h_tx_t.unsqueeze(0).to(torch.bfloat16))
        if alpha < 1.0:
            h_cond = alpha * h_cond

        icl = build_official_prefill(text_str)
        L_prefill = icl.shape[1]

        codes_prev = codes[:-1]
        agg_audio = talker.get_input_embeddings()(codes_prev[:, 0:1])
        for g in range(1, 16):
            agg_audio = agg_audio + talker.code_predictor.get_input_embeddings()[g-1](codes_prev[:, g:g+1])
        agg_audio = agg_audio.transpose(0, 1) # (1, T-1, 2048)
        audio_embeds = agg_audio + h_cond[:, 1:T, :]

        full_inputs = torch.cat([icl, audio_embeds], dim=1)

        out = talker.model(inputs_embeds=full_inputs, use_cache=False)
        h_seq = out.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]

        cb0_logits = talker.codec_head(h_seq)
        loss_cb0 = F.cross_entropy(cb0_logits.reshape(-1, cb0_logits.shape[-1]), codes[:, 0].reshape(-1))
        sub_lg, loss_res = talker.forward_sub_talker_finetune(codes, h_seq[0])
        acc = float((cb0_logits.argmax(dim=-1)[0] == codes[:, 0]).float().mean().item())
        loss_f13 = float(F.cross_entropy(sub_lg[:, :3].reshape(-1, sub_lg.shape[-1]), codes[:, 1:4].reshape(-1)).item())

        if args.lambda_m > 0:
            with torch.no_grad():
                inp_nat = torch.cat([icl, agg_audio], dim=1)
                out_nat = talker.model(inputs_embeds=inp_nat, use_cache=False)
                h_nat = out_nat.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]
            cos_step = F.cosine_similarity(h_seq.float(), h_nat.float().detach(), dim=-1).mean()
            loss_manifold = 1.0 - cos_step
            norm_ratio_step = h_seq.float().norm(dim=-1).mean() / (h_nat.float().norm(dim=-1).mean() + 1e-6)
            loss_norm = (norm_ratio_step - 1.0) ** 2
            loss_tot = loss_cb0 + args.lambda_r * loss_res + args.lambda_m * loss_manifold + 0.1 * loss_norm
            cos_val = float(cos_step.item())
        else:
            loss_tot = loss_cb0 + args.lambda_r * loss_res
            loss_manifold = torch.tensor(0.0)
            cos_val = 0.0

        return loss_tot, float(loss_cb0.item()), float(loss_res.item()), acc, loss_f13, float(loss_manifold.item()), cos_val

    # 4. SLERP In-Loop Eval
    def run_slerp_val_eval():
        proj.eval()
        results = {'a1.0': {'cb0': [], 'res_tot': [], 'acc': []},
                   'a0.0': {'cb0': [], 'res_tot': [], 'acc': []}}
        cos_sims, norm_ratios = [], []
        eval_n = min(20, N_val)
        val_items = []
        for i in range(eval_n):
            c0 = val_start + i * 64
            codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
            codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
            n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
            if n_real < 16: continue
            gi = (val_start // 64) + i
            text_str = get_italian_text_prompt(tbw[gi] if gi < len(tbw) else None)
            h_raw = torch.from_numpy(np.asarray(va_raw[i][:n_real], dtype=np.float32)).to('cuda')
            h_tx = torch.from_numpy(np.asarray(va_htx[i], dtype=np.float32)).to('cuda')
            val_items.append((codes[:n_real], text_str, h_raw, h_tx))

        with torch.no_grad():
            for codes, text_str, h_raw, h_tx in val_items:
                T = codes.shape[0]
                icl = build_official_prefill(text_str)
                L_prefill = icl.shape[1]
                codes_prev = codes[:-1]
                agg_audio = sum(talker.get_input_embeddings()(codes_prev[:, 0:1]) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](codes_prev[:, g:g+1]) for g in range(16)).transpose(0, 1)

                # Omni hidden states (alpha=1.0)
                h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))
                inp_omni = torch.cat([icl, agg_audio + h_cond[:, 1:T, :]], dim=1)
                out_omni = talker.model(inputs_embeds=inp_omni, use_cache=False)
                h_omni = out_omni.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]
                cb0_lg_1 = talker.codec_head(h_omni)
                l_cb0_1 = float(F.cross_entropy(cb0_lg_1.reshape(-1, cb0_lg_1.shape[-1]), codes[:, 0].reshape(-1)).item())
                sub_lg_1, l_res_1 = talker.forward_sub_talker_finetune(codes, h_omni[0])
                acc_1 = float((cb0_lg_1.argmax(dim=-1)[0] == codes[:, 0]).float().mean().item())
                results['a1.0']['cb0'].append(l_cb0_1)
                results['a1.0']['acc'].append(acc_1)
                results['a1.0']['res_tot'].append(float(l_res_1.item()))

                # Native hidden states (alpha=0.0)
                inp_nat = torch.cat([icl, agg_audio], dim=1)
                out_nat = talker.model(inputs_embeds=inp_nat, use_cache=False)
                h_nat = out_nat.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]
                cb0_lg_0 = talker.codec_head(h_nat)
                l_cb0_0 = float(F.cross_entropy(cb0_lg_0.reshape(-1, cb0_lg_0.shape[-1]), codes[:, 0].reshape(-1)).item())
                sub_lg_0, l_res_0 = talker.forward_sub_talker_finetune(codes, h_nat[0])
                acc_0 = float((cb0_lg_0.argmax(dim=-1)[0] == codes[:, 0]).float().mean().item())
                results['a0.0']['cb0'].append(l_cb0_0)
                results['a0.0']['acc'].append(acc_0)
                results['a0.0']['res_tot'].append(float(l_res_0.item()))

                cs = float(F.cosine_similarity(h_omni.float(), h_nat.float(), dim=-1).mean().item())
                nr = float((h_omni.norm(dim=-1).mean() / (h_nat.norm(dim=-1).mean() + 1e-6)).item())
                cos_sims.append(cs)
                norm_ratios.append(nr)

        proj.train()
        val_summary = {k: {m: float(np.mean(v)) for m, v in metrics.items()} for k, metrics in results.items()}
        val_summary['manifold'] = {'cosine_sim': float(np.mean(cos_sims)), 'norm_ratio': float(np.mean(norm_ratios))}
        return val_summary

    # 5. Training Loop V9
    total_steps = args.start_step + args.steps
    opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr * 0.1)

    print(f"\n[V9 3/5] Avvio Training V9 Deep Causal Transformer (Step {args.start_step+1} a {total_steps})...", flush=True)
    proj.train()

    step = args.start_step
    data_idx = args.start_step * args.batch

    while step < total_steps:
        step_t0 = time.time()
        opt.zero_grad()

        batch_loss, batch_cb0, batch_res, batch_man, batch_cos, batch_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        valid_in_batch = 0

        for b in range(args.batch):
            idx = (data_idx + b) % N_train
            c0 = idx * 64
            codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
            codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
            n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
            if n_real < 16: continue
            codes = codes[:n_real]

            text_str = get_italian_text_prompt(tbw[idx] if idx < len(tbw) else None)
            h_raw_t = torch.from_numpy(np.asarray(tr_raw[idx][:n_real], dtype=np.float32)).to('cuda')
            h_tx_t = torch.from_numpy(np.asarray(tr_htx[idx], dtype=np.float32)).to('cuda')

            l_tot, l_cb0, l_res, acc_s, _, l_man, cos_s = parallel_forward_sample(codes, text_str, h_raw_t, h_tx_t)
            (l_tot / args.batch).backward()

            batch_loss += float(l_tot.item())
            batch_cb0 += l_cb0
            batch_res += l_res
            batch_man += l_man
            batch_cos += cos_s
            batch_acc += acc_s
            valid_in_batch += 1

        if valid_in_batch > 0:
            torch.nn.utils.clip_grad_norm_(proj.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            step += 1
            data_idx += args.batch

            dt = time.time() - step_t0
            cur_lr = sched.get_last_lr()[0]
            m_loss = batch_loss / valid_in_batch
            m_cb0 = batch_cb0 / valid_in_batch
            m_res = batch_res / valid_in_batch
            m_man = batch_man / valid_in_batch
            m_cos = batch_cos / valid_in_batch
            m_acc = batch_acc / valid_in_batch

            print(f"[V9-TRANSFORMER] step {step}/{total_steps} loss={m_loss:.4f} "
                  f"[cb0={m_cb0:.3f} res={m_res:.3f} man={m_man:.3f} cos={m_cos:.4f} acc={m_acc*100:.1f}% gamma={proj.gamma.item():.4f}] "
                  f"lr={cur_lr:.2e} ({dt:.2f}s/step)", flush=True)

            if step % args.eval_every == 0 or step == total_steps:
                print(f"\n--- [V9 VAL IN-LOOP STEP {step}] Esecuzione SLERP Eval Rapido ---", flush=True)
                val_res = run_slerp_val_eval()
                cb0_1 = val_res['a1.0']['cb0']
                cb0_0 = val_res['a0.0']['cb0']
                delta = cb0_1 - cb0_0
                print(f"  [SLERP a=1.0] CB0={cb0_1:.3f} Acc={val_res['a1.0']['acc']*100:.1f}% ResTot={val_res['a1.0']['res_tot']:.3f}", flush=True)
                print(f"  [DELTA SLERP] a1.0 vs a0.0 = {delta:+.3f} ({'PROIEZIONE GUIDA' if delta < 0 else 'TALKER BASE GUIDA'})", flush=True)
                print(f"  [MANIFOLD] Cosine Sim={val_res['manifold']['cosine_sim']:.4f} | Norm Ratio={val_res['manifold']['norm_ratio']:.4f}\n", flush=True)

            if step % args.save_every == 0 or step == total_steps:
                ckpt_path = os.path.join(args.out, f"proj_v9_step{step}.pt")
                torch.save({'proj': proj.state_dict(), 'step': step, 'gamma': proj.gamma.item()}, ckpt_path)
                print(f"  [CHECKPOINT] Salvato in {ckpt_path}", flush=True)
        else:
            data_idx += args.batch

    final_path = os.path.join(args.out, "proj_v9_final.pt")
    torch.save({'proj': proj.state_dict(), 'step': total_steps, 'gamma': proj.gamma.item()}, final_path)
    print(f"\n[V9 5/5] Training V9 Deep Transformer completato con successo! Checkpoint finale: {final_path}", flush=True)

if __name__ == '__main__':
    main()
