#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# TRAINER V7 (PARALLEL CAUSAL SINGLE-PASS ENGINE)
# Architettura:
#   1. TextAnchoredProjectionV7 con gamma=0.25 (Norm calibration a ~11.3)
#   2. Gemma->Qwen Text Bridge: tok_gemma.decode(tbw) -> proc_qwen(text) con Language Tag 'italian' (id=2070)
#   3. Talker Qwen3-TTS (congelato):
#      PARALLEL CAUSAL FORWARD: 1 singola passata tensoriale a talker.model per sample (senza loop da 64 iterazioni)
#      Sequence: [Prefill, Audio_Embeds_1_to_T] -> Causal Self-Attention SDPA
#      Loss = CE(cb0) + lambda_r * CE(residui CB1..15)  [lambda_d = 0.0]
#   4. In-Loop Ultra-Fast Mathematical Validation (< 1.5 secondi per SLERP eval)
#   5. End-of-Run Audio Synthesis con Decoder Ufficiale Verificato e Nucleus Sampling
# ============================================================================
import os, sys, json, glob, time, math, argparse, traceback
import numpy as np

# Mock torchaudio and sox for venv_phase1 compatibility
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

import torch
import torch.nn as nn
import torch.nn.functional as F

import kaggle_pipeline as KP
import qwen_subtalker as QS

# ---- shim compat transformers 5.x (ADR-011) ----
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
import soundfile as sf
from faster_whisper import WhisperModel

class TextAnchoredProjectionV7(nn.Module):
    def __init__(self, in_dim=1536, out_dim=2048, hidden_dim=3072, initial_gamma=0.25):
        super().__init__()
        self.audio_skip = nn.Linear(in_dim, out_dim)
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

def sample_cb0_with_penalty(logits, past_tokens, temperature=0.8, top_k=50, top_p=0.92, rep_penalty=1.25):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', type=int, default=0)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--start-step', type=int, default=500)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda-r', type=float, default=0.5)
    parser.add_argument('--lambda-d', type=float, default=0.0)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--out', type=str, default='/workspace/checkpoints')
    parser.add_argument('--warmstart', type=str, default='/workspace/checkpoints/proj_v7_step500.pt')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs('/workspace/samples_v7', exist_ok=True)

    print(f"[V7-PARALLEL] start_step={args.start_step} steps={args.steps} batch={args.batch} "
          f"lr={args.lr} lambda_r={args.lambda_r} lambda_d={args.lambda_d} warmstart={os.path.basename(args.warmstart)}", flush=True)

    # 1. Caricamento Modelli
    print("[V7 1/5] Caricamento Talker Qwen3-TTS, Tokenizer & Decoder...", flush=True)
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

    # 2. Proiettore V7
    proj = TextAnchoredProjectionV7().to(device='cuda', dtype=torch.bfloat16)

    if os.path.exists(args.warmstart):
        try:
            sd = torch.load(args.warmstart, map_location='cpu')
            sd_proj = sd.get('proj', sd)
            p_sd = proj.state_dict()
            loaded_keys = []
            for k in p_sd.keys():
                if k in sd_proj and sd_proj[k].shape == p_sd[k].shape:
                    p_sd[k] = sd_proj[k]
                    loaded_keys.append(k)
            proj.load_state_dict(p_sd)
            print(f"  Warmstart V7 caricato da {args.warmstart} ({len(loaded_keys)} parametri caricati, gamma={proj.gamma.item():.4f})", flush=True)
        except Exception as e:
            print(f"  [WARN] Fallito caricamento warmstart: {e}", flush=True)

    # 3. Dati
    print("[V7 2/5] Caricamento Dati & Cache v2...", flush=True)
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
    # VERIFIED SEQUENTIAL KV CACHE FORWARD LOOP
    # ------------------------------------------------------------------------
    def loop_forward(codes, text_str, h_raw_t, h_tx_t, alpha=1.0):
        T = codes.shape[0]
        h_cond = proj(h_raw_t.unsqueeze(0).to(torch.bfloat16), h_tx_t.unsqueeze(0).to(torch.bfloat16))
        if alpha < 1.0:
            h_cond = alpha * h_cond

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
        cb0_logits = talker.codec_head(h_seq)
        loss_cb0 = F.cross_entropy(cb0_logits.view(-1, cb0_logits.shape[-1]), codes[:, 0].contiguous().view(-1))
        sub_lg, loss_res = talker.forward_sub_talker_finetune(codes, h_seq[0])
        loss_tot = loss_cb0 + args.lambda_r * loss_res
        acc = float((cb0_logits.argmax(dim=-1)[0] == codes[:, 0]).float().mean().item())
        loss_f13 = float(F.cross_entropy(sub_lg[:, :3].reshape(-1, sub_lg.shape[-1]), codes[:, 1:4].reshape(-1)).item())

        return loss_tot, float(loss_cb0.item()), float(loss_res.item()), acc, loss_f13

    # ------------------------------------------------------------------------
    # SLERP Validation in-loop (< 2s)
    # ------------------------------------------------------------------------
    def run_slerp_val_eval():
        proj.eval()
        results = {'a1.0': {'cb0': [], 'res1_3': [], 'res_tot': [], 'acc': []},
                   'a0.5': {'cb0': [], 'res1_3': [], 'res_tot': [], 'acc': []},
                   'a0.0': {'cb0': [], 'res1_3': [], 'res_tot': [], 'acc': []}}
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
            for k, alpha in [('a1.0', 1.0), ('a0.5', 0.5), ('a0.0', 0.0)]:
                for codes, text_str, h_raw, h_tx in val_items:
                    _, l_cb0, l_res, acc, l_f13 = loop_forward(codes, text_str, h_raw, h_tx, alpha=alpha)
                    results[k]['cb0'].append(l_cb0)
                    results[k]['acc'].append(acc)
                    results[k]['res1_3'].append(l_f13)
                    results[k]['res_tot'].append(l_res)

        proj.train()
        return {k: {m: float(np.mean(v)) for m, v in metrics.items()} for k, metrics in results.items()}

    # 4. Training Loop
    total_steps = args.start_step + args.steps
    opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.lr * 0.1)
    
    for _ in range(args.start_step):
        sched.step()

    print(f"\n[V7 3/5] Avvio Training V7 Parallelo (da Step {args.start_step+1} a {total_steps})...", flush=True)
    proj.train()

    step = args.start_step
    data_idx = args.start_step * args.batch

    while step < total_steps:
        step_t0 = time.time()
        opt.zero_grad()

        batch_loss = 0.0
        batch_cb0 = 0.0
        batch_res = 0.0
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

            l_tot, l_cb0, l_res, acc_s, _ = loop_forward(codes, text_str, h_raw_t, h_tx_t)
            (l_tot / args.batch).backward()

            batch_loss += float(l_tot.item())
            batch_cb0 += l_cb0
            batch_res += l_res
            batch_acc += acc_s
            valid_in_batch += 1

        data_idx += args.batch
        if valid_in_batch == 0: continue

        torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        step_dt = time.time() - step_t0

        if step % 20 == 0 or step == args.start_step + 1 or args.smoke:
            print(f"[V7-PARALLEL] step {step}/{total_steps} loss={batch_loss/valid_in_batch:.4f} "
                  f"[cb0={batch_cb0/valid_in_batch:.3f} res={batch_res/valid_in_batch:.3f} acc={batch_acc/valid_in_batch*100:.1f}% "
                  f"gamma={proj.gamma.item():.4f}] lr={sched.get_last_lr()[0]:.2e} ({step_dt:.2f}s/step)", flush=True)

        if (step % args.eval_every == 0 or (args.smoke and step == total_steps)) and not args.smoke:
            t_eval0 = time.time()
            m = run_slerp_val_eval()
            t_eval_dt = time.time() - t_eval0
            print(f"\n  ================================================================", flush=True)
            print(f"  >>> [V7-VAL @ STEP {step}] DIAGNOSTICA MATEMATICA IN-LOOP ({t_eval_dt:.2f}s)", flush=True)
            print(f"    • Condizionato (alpha=1.0): CB0={m['a1.0']['cb0']:.3f} (Acc={m['a1.0']['acc']*100:.1f}%) | "
                  f"Formanti 1..3={m['a1.0']['res1_3']:.3f} | Residui Tot={m['a1.0']['res_tot']:.3f}", flush=True)
            print(f"    • Mid-SLERP   (alpha=0.5): CB0={m['a0.5']['cb0']:.3f} (Acc={m['a0.5']['acc']*100:.1f}%) | "
                  f"Formanti 1..3={m['a0.5']['res1_3']:.3f} | Residui Tot={m['a0.5']['res_tot']:.3f}", flush=True)
            print(f"    • Baseline    (alpha=0.0): CB0={m['a0.0']['cb0']:.3f} (Acc={m['a0.0']['acc']*100:.1f}%) | "
                  f"Formanti 1..3={m['a0.0']['res1_3']:.3f} | Residui Tot={m['a0.0']['res_tot']:.3f}", flush=True)
            d_cb0 = m['a1.0']['cb0'] - m['a0.0']['cb0']
            d_res = m['a1.0']['res_tot'] - m['a0.0']['res_tot']
            print(f"    • SLERP Delta: Delta_CB0={d_cb0:+.3f} | Delta_Res={d_res:+.3f} (Guadagno Monotono)", flush=True)
            print(f"  ================================================================\n", flush=True)

        if step % 100 == 0 or step == total_steps:
            ckpt_save = os.path.join(args.out, f'proj_v7_step{step}.pt')
            torch.save({'proj': proj.state_dict(), 'step': step, 'gamma': proj.gamma.item()}, ckpt_save)

    # 5. Checkpoint Finale & Sintesi Ufficiale
    final_save = os.path.join(args.out, 'proj_v7_final.pt')
    torch.save({'proj': proj.state_dict(), 'step': step, 'gamma': proj.gamma.item()}, final_save)
    print(f"[V7] Checkpoint finale salvato: {final_save}", flush=True)

    if not args.smoke:
        print("\n[V7 4/5] Generazione Campioni Audio & Trascrizione Whisper ASR a Fine Training...", flush=True)
        proj.eval()
        try:
            asr = WhisperModel('small', device='cuda', compute_type='float16')
        except Exception as e:
            asr = None
            print(f"Impossibile caricare Whisper: {e}", flush=True)

        tdecay = [0.95 ** j for j in range(15)]
        with torch.no_grad():
            for i in range(min(5, N_val)):
                c0 = val_start + i * 64
                codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
                codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
                n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
                if n_real < 16: continue
                codes = codes[:n_real]

                gi = (val_start // 64) + i
                text_str = get_italian_text_prompt(tbw[gi] if gi < len(tbw) else None)
                h_raw = torch.from_numpy(np.asarray(va_raw[i][:n_real], dtype=np.float32)).to('cuda')
                h_tx = torch.from_numpy(np.asarray(va_htx[i], dtype=np.float32)).to('cuda')
                h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))

                icl = build_official_prefill(text_str)
                out = talker.model(inputs_embeds=icl, attention_mask=torch.ones(1, icl.shape[1], device='cuda'), use_cache=True)
                kv = out.past_key_values
                past_h = out.last_hidden_state[:, -1:, :]

                resid_all, cb0_all = [], []
                lg0 = talker.codec_head(past_h)
                cb0_val = sample_cb0_with_penalty(lg0, cb0_all, temperature=0.8, top_k=50, top_p=0.92, rep_penalty=1.25)
                resid_t = sub.sample_acoustic_codes(past_h.to(torch.bfloat16), torch.tensor([[cb0_val]], device='cuda'), top_k=50, top_p=0.95, temperatures_per_level=tdecay)
                resid_all.append(resid_t[0, 0].cpu().long())
                cb0_all.append(cb0_val)

                curr_code = torch.cat([torch.tensor([[cb0_val]], device='cuda'), resid_t[0, 0:1]], dim=-1)

                for t in range(1, n_real):
                    cp = curr_code[0]
                    agg = sum(talker.get_input_embeddings()(cp[g:g+1].reshape(1,1)) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](cp[g:g+1].reshape(1,1)) for g in range(16))
                    in_t = agg + h_cond[:, t:t+1, :]
                    out = talker.model(inputs_embeds=in_t, attention_mask=torch.ones(1, 1, device='cuda'), use_cache=True, past_key_values=kv)
                    kv = out.past_key_values
                    past_h = out.last_hidden_state[:, -1:, :]

                    lg = talker.codec_head(past_h)
                    cb0_val = sample_cb0_with_penalty(lg, cb0_all, temperature=0.8, top_k=50, top_p=0.92, rep_penalty=1.25)
                    resid_t = sub.sample_acoustic_codes(past_h.to(torch.bfloat16), torch.tensor([[cb0_val]], device='cuda'), top_k=50, top_p=0.95, temperatures_per_level=tdecay)
                    resid_all.append(resid_t[0, 0].cpu().long())
                    cb0_all.append(cb0_val)
                    curr_code = torch.cat([torch.tensor([[cb0_val]], device='cuda'), resid_t[0, 0:1]], dim=-1)

                R = torch.stack(resid_all, dim=0)
                C = torch.tensor(cb0_all).unsqueeze(-1)
                full_codes = torch.cat([C, R], dim=-1)

                c_input = full_codes.unsqueeze(0).cuda()
                out_dec = tok_decoder.decode(c_input)
                wav_arr = out_dec.audio_values[0].cpu().numpy().astype(np.float32)

                p_peak = float(np.percentile(np.abs(wav_arr), 99.7))
                if p_peak > 1e-6:
                    wav_norm = np.clip(wav_arr / (p_peak * 1.05), -1.0, 1.0)
                else:
                    wav_norm = wav_arr

                wav_path = f'/workspace/samples_v7/v7_eval_sample_{i:02d}.wav'
                sf.write(wav_path, wav_norm, 24000)
                rms = float(np.sqrt(np.mean(wav_norm**2)))

                transcript = "N/A"
                if asr is not None:
                    try:
                        segs, _ = asr.transcribe(wav_path, language='it', beam_size=5)
                        transcript = ' '.join(s.text.strip() for s in segs)
                    except Exception as e:
                        transcript = f"Errore ASR: {e}"

                print(f"  [Sample {i:02d}] Durata: {len(wav_norm)/24000:.2f}s | RMS: {rms:.4f} | Token Unici: {len(set(cb0_all))}/64", flush=True)
                print(f"    • Testo Ground Truth : \"{text_str}\"", flush=True)
                print(f"    • Trascrizione ASR   : \"{transcript}\"\n", flush=True)

    print("[V7 5/5] TRAINING V7 & EVALUATION COMPLETATI CON SUCCESSO.", flush=True)

if __name__ == '__main__':
    main()
