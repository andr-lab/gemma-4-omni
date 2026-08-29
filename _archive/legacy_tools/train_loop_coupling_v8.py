#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma 4 Omni — Full-Duplex Talker Coupling Trainer V8
------------------------------------------------------
Addestramento da zero (Scratch) del proiettore TextAnchoredProjectionV8
utilizzando il Motore Parallelo Causale 3D MRoPE (Single-Pass Forward).
- Step: 0 -> 1000
- Batch size: 4
- Velocità attesa: ~0.25-0.35s / step (~5-6 minuti totali per 1000 step)
- Validazione SLERP in-loop ultra-rapida (< 1.5s)
"""

import os, sys, glob, time, json, argparse
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

# Patch RoPE functions
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

# ============================================================================
# ARCHITETTURA V8: TextAnchoredProjectionV8
# ============================================================================
class TextAnchoredProjectionV8(nn.Module):
    """
    Proiettore V8 con ancoraggio testuale dinamico, skip connection,
    doppia LayerNorm e parametro di gating gamma apprendibile.
    """
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
        nn.init.zeros_(self.text_mlp[2].weight)
        nn.init.zeros_(self.text_mlp[2].bias)

        self.out_norm = nn.LayerNorm(out_dim)
        self.gamma = nn.Parameter(torch.tensor(initial_gamma, dtype=torch.float32))

    def forward(self, h_raw, h_tx):
        if h_tx.dim() == 2:
            h_tx = h_tx.unsqueeze(1)
        h_a = self.audio_norm(self.audio_mlp(h_raw) + self.audio_skip(h_raw))
        h_t = self.text_norm(self.text_mlp(h_tx))
        return (self.gamma.to(h_a.dtype)) * self.out_norm(h_a + h_t)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--start-step', type=int, default=0)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda-r', type=float, default=0.5)
    parser.add_argument('--lambda-m', type=float, default=0.5)
    parser.add_argument('--lambda-d', type=float, default=0.0)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--out', type=str, default='/workspace/checkpoints')
    parser.add_argument('--warmstart', type=str, default='')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs('/workspace/samples_v8', exist_ok=True)

    print(f"[V8-PARALLEL] start_step={args.start_step} steps={args.steps} batch={args.batch} "
          f"lr={args.lr} lambda_r={args.lambda_r} lambda_m={args.lambda_m} lambda_d={args.lambda_d}", flush=True)

    # 1. Caricamento Modelli
    print("[V8 1/5] Caricamento Talker Qwen3-TTS, Tokenizer & Decoder...", flush=True)
    snap = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*'))[-1]
    _cfg = _M.Qwen3TTSConfig.from_pretrained(snap)
    _tc = _cfg.talker_config
    if not hasattr(_tc, 'pad_token_id'):
        object.__setattr__(_tc, 'pad_token_id', None)

    teacher = _M.Qwen3TTSForConditionalGeneration(_cfg).to(torch.bfloat16)
    _sd = st.load_file(snap + '/model.safetensors', device='cpu')
    _m_res = teacher.load_state_dict(_sd, strict=False)
    del _sd; torch.cuda.empty_cache()
    teacher = teacher.to(device='cuda', dtype=torch.bfloat16).eval()
    talker = teacher.talker
    print(f"  teacher missing={len(_m_res.missing_keys)} unexpected={len(_m_res.unexpected_keys)}", flush=True)

    tok_qwen = AutoTokenizer.from_pretrained(snap, use_fast=True)
    proc_qwen = Qwen3TTSProcessor(tokenizer=tok_qwen)

    snap_tok = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*'))[-1]
    tok_decoder = Qwen3TTSTokenizerV2Model.from_pretrained(snap_tok).cuda().eval()
    print("  Decoder Qwen3TTSTokenizerV2Model caricato con successo.", flush=True)

    gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    tok_gemma = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True) if len(gemma_dirs) > 0 else None
    if tok_gemma is not None:
        print(f"  tok_gemma caricato con successo (vocab={len(tok_gemma)})", flush=True)

    CFG = teacher.config
    TCFG = CFG.talker_config
    lang_map = getattr(TCFG, 'codec_language_id', {})
    lang_id = lang_map.get('italian', 2070)

    sub, _ = QS.load_qwen_subtalker('/workspace/qwen_subtalker', device='cuda', dtype=torch.bfloat16)

    # 2. Proiettore V8 (Inizializzato da zero / Scratch)
    proj = TextAnchoredProjectionV8().to(device='cuda', dtype=torch.bfloat16)
    print(f"  Proiettore TextAnchoredProjectionV8 inizializzato da zero (gamma={proj.gamma.item():.4f})", flush=True)

    if args.warmstart and os.path.exists(args.warmstart):
        try:
            sd = torch.load(args.warmstart, map_location='cpu')
            proj.load_state_dict(sd.get('proj', sd), strict=False)
            print(f"  Warmstart caricato da {args.warmstart}", flush=True)
        except Exception as e:
            print(f"  [WARN] Fallito caricamento warmstart: {e}", flush=True)

    # 3. Dati
    print("[V8 2/5] Caricamento Dati & Cache v2...", flush=True)
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
    print(f"  N_train={N_train} N_val={N_val} val_start={val_start}", flush=True)

    def get_italian_text_prompt(tbw_item):
        if tbw_item is None: return "Buongiorno."
        if isinstance(tbw_item, list):
            return tok_gemma.decode(tbw_item, skip_special_tokens=True).strip()
        return str(tbw_item).strip()

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

    # ------------------------------------------------------------------------
    # FAST PARALLEL CAUSAL SINGLE-PASS FORWARD (V8 ENGINE)
    # ------------------------------------------------------------------------
    def parallel_forward_sample(codes, text_str, h_raw_t, h_tx_t, alpha=1.0):
        T = codes.shape[0]
        h_cond = proj(h_raw_t.unsqueeze(0).to(torch.bfloat16), h_tx_t.unsqueeze(0).to(torch.bfloat16))
        if alpha < 1.0:
            h_cond = alpha * h_cond

        icl = build_official_prefill(text_str)
        L_prefill = icl.shape[1]

        codes_prev = codes[:-1] # (T-1, 16)
        agg_audio = talker.get_input_embeddings()(codes_prev[:, 0:1])
        for g in range(1, 16):
            agg_audio = agg_audio + talker.code_predictor.get_input_embeddings()[g-1](codes_prev[:, g:g+1])
        agg_audio = agg_audio.transpose(0, 1) # (1, T-1, 2048)
        audio_embeds = agg_audio + h_cond[:, 1:T, :]

        full_inputs = torch.cat([icl, audio_embeds], dim=1) # (1, L_prefill + T - 1, 2048)

        # UNICO forward pass parallelo causale del Talker
        out = talker.model(inputs_embeds=full_inputs, use_cache=False)
        full_h = out.last_hidden_state

        h_seq = full_h[:, L_prefill - 1 : L_prefill - 1 + T, :] # (1, T, 2048)

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

    # ------------------------------------------------------------------------
    # SLERP & MANIFOLD Validation in-loop (< 2s)
    # ------------------------------------------------------------------------
    def run_slerp_val_eval():
        proj.eval()
        results = {'a1.0': {'cb0': [], 'res1_3': [], 'res_tot': [], 'acc': []},
                   'a0.5': {'cb0': [], 'res1_3': [], 'res_tot': [], 'acc': []},
                   'a0.0': {'cb0': [], 'res1_3': [], 'res_tot': [], 'acc': []}}
        cos_sims = []
        norm_ratios = []
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
                l_f13_1 = float(F.cross_entropy(sub_lg_1[:, :3].reshape(-1, sub_lg_1.shape[-1]), codes[:, 1:4].reshape(-1)).item())
                results['a1.0']['cb0'].append(l_cb0_1)
                results['a1.0']['acc'].append(acc_1)
                results['a1.0']['res1_3'].append(l_f13_1)
                results['a1.0']['res_tot'].append(float(l_res_1.item()))

                # Native hidden states (alpha=0.0)
                inp_nat = torch.cat([icl, agg_audio], dim=1)
                out_nat = talker.model(inputs_embeds=inp_nat, use_cache=False)
                h_nat = out_nat.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]
                cb0_lg_0 = talker.codec_head(h_nat)
                l_cb0_0 = float(F.cross_entropy(cb0_lg_0.reshape(-1, cb0_lg_0.shape[-1]), codes[:, 0].reshape(-1)).item())
                sub_lg_0, l_res_0 = talker.forward_sub_talker_finetune(codes, h_nat[0])
                acc_0 = float((cb0_lg_0.argmax(dim=-1)[0] == codes[:, 0]).float().mean().item())
                l_f13_0 = float(F.cross_entropy(sub_lg_0[:, :3].reshape(-1, sub_lg_0.shape[-1]), codes[:, 1:4].reshape(-1)).item())
                results['a0.0']['cb0'].append(l_cb0_0)
                results['a0.0']['acc'].append(acc_0)
                results['a0.0']['res1_3'].append(l_f13_0)
                results['a0.0']['res_tot'].append(float(l_res_0.item()))

                # Alpha=0.5
                inp_05 = torch.cat([icl, agg_audio + 0.5 * h_cond[:, 1:T, :]], dim=1)
                out_05 = talker.model(inputs_embeds=inp_05, use_cache=False)
                h_05 = out_05.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]
                cb0_lg_05 = talker.codec_head(h_05)
                l_cb0_05 = float(F.cross_entropy(cb0_lg_05.reshape(-1, cb0_lg_05.shape[-1]), codes[:, 0].reshape(-1)).item())
                sub_lg_05, l_res_05 = talker.forward_sub_talker_finetune(codes, h_05[0])
                acc_05 = float((cb0_lg_05.argmax(dim=-1)[0] == codes[:, 0]).float().mean().item())
                l_f13_05 = float(F.cross_entropy(sub_lg_05[:, :3].reshape(-1, sub_lg_05.shape[-1]), codes[:, 1:4].reshape(-1)).item())
                results['a0.5']['cb0'].append(l_cb0_05)
                results['a0.5']['acc'].append(acc_05)
                results['a0.5']['res1_3'].append(l_f13_05)
                results['a0.5']['res_tot'].append(float(l_res_05.item()))

                # Manifold geometric metrics
                cs = float(F.cosine_similarity(h_omni.float(), h_nat.float(), dim=-1).mean().item())
                nr = float((h_omni.norm(dim=-1).mean() / (h_nat.norm(dim=-1).mean() + 1e-6)).item())
                cos_sims.append(cs)
                norm_ratios.append(nr)

        proj.train()
        val_summary = {k: {m: float(np.mean(v)) for m, v in metrics.items()} for k, metrics in results.items()}
        val_summary['manifold'] = {'cosine_sim': float(np.mean(cos_sims)), 'norm_ratio': float(np.mean(norm_ratios))}
        return val_summary

    # 4. Training Loop V8
    total_steps = args.start_step + args.steps
    opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr * 0.1)

    print(f"\n[V8 3/5] Avvio Training V8 Parallelo (Step {args.start_step+1} a {total_steps})...", flush=True)
    proj.train()

    step = args.start_step
    data_idx = args.start_step * args.batch

    while step < total_steps:
        step_t0 = time.time()
        opt.zero_grad()

        batch_loss = 0.0
        batch_cb0 = 0.0
        batch_res = 0.0
        batch_man = 0.0
        batch_cos = 0.0
        batch_acc = 0.0
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

            print(f"[V8-PARALLEL] step {step}/{total_steps} loss={m_loss:.4f} "
                  f"[cb0={m_cb0:.3f} res={m_res:.3f} man={m_man:.3f} cos={m_cos:.4f} acc={m_acc*100:.1f}% gamma={proj.gamma.item():.4f}] "
                  f"lr={cur_lr:.2e} ({dt:.2f}s/step)", flush=True)

            if step % args.eval_every == 0 or step == total_steps:
                print(f"\n--- [V8 VAL IN-LOOP STEP {step}] Esecuzione SLERP Eval Rapido ---", flush=True)
                val_res = run_slerp_val_eval()
                cb0_1 = val_res['a1.0']['cb0']
                cb0_0 = val_res['a0.0']['cb0']
                delta = cb0_1 - cb0_0
                print(f"  [SLERP a=1.0] CB0={cb0_1:.3f} Acc={val_res['a1.0']['acc']*100:.1f}% ResTot={val_res['a1.0']['res_tot']:.3f}", flush=True)
                print(f"  [SLERP a=0.5] CB0={val_res['a0.5']['cb0']:.3f} Acc={val_res['a0.5']['acc']*100:.1f}% ResTot={val_res['a0.5']['res_tot']:.3f}", flush=True)
                print(f"  [SLERP a=0.0] CB0={cb0_0:.3f} Acc={val_res['a0.0']['acc']*100:.1f}% ResTot={val_res['a0.0']['res_tot']:.3f}", flush=True)
                print(f"  [DELTA SLERP] a1.0 vs a0.0 = {delta:+.3f} ({'PROIEZIONE GUIDA' if delta < 0 else 'TALKER BASE GUIDA'})", flush=True)
                print(f"  [MANIFOLD] Cosine Sim={val_res['manifold']['cosine_sim']:.4f} | Norm Ratio={val_res['manifold']['norm_ratio']:.4f}\n", flush=True)

            if step % args.save_every == 0 or step == total_steps:
                ckpt_path = os.path.join(args.out, f"proj_v8_step{step}.pt")
                torch.save({'proj': proj.state_dict(), 'step': step, 'gamma': proj.gamma.item()}, ckpt_path)
                print(f"  [CHECKPOINT] Salvato in {ckpt_path}", flush=True)
        else:
            data_idx += args.batch

    final_path = os.path.join(args.out, "proj_v8_final.pt")
    torch.save({'proj': proj.state_dict(), 'step': total_steps, 'gamma': proj.gamma.item()}, final_path)
    print(f"\n[V8 5/5] Training V8 Parallelo completato con successo! Checkpoint finale: {final_path}", flush=True)

if __name__ == '__main__':
    main()
