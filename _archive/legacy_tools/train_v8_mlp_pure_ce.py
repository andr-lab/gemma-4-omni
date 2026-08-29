#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma4-Omni V8.6: Pure Cross-Entropy Training of TextAnchoredProjectionV8 (MLP)
REMOVES Cosine Similarity Loss (lambda_m = 0) and Norm Loss (lambda_norm = 0)
Eliminates gradient conflict / interference to test if MLP can freely optimize acoustic residuals.

Key Features:
- Model: TextAnchoredProjectionV8 (25.19M params).
- Warmstart: /workspace/checkpoints/proj_v8_final.pt (Step 25000).
- Qwen3-TTS: 100% FROZEN (Talker 1.7B + SubTalker code_predictor 175M).
- Loss: L_tot = 1.0 * L_cb0 + 0.50 * L_res (Pure Cross-Entropy, Literature Standard).
- Metrics logged: CB0 Loss, CB0 Acc, Res Loss, Res Acc (CB1..15), Cosine Sim (passive), Norm (passive).
- Cadence: Eval every 100 steps, Checkpoints every 500 steps.
- Post-Eval: Comparative audio synthesis (Omni MLP Pure-CE vs Native).
"""

import sys, os, glob, json, math, argparse, time
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lambda-r', type=float, default=0.50)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--out', type=str, default='/workspace/checkpoints')
    parser.add_argument('--samples-out', type=str, default='/workspace/samples_v8_pure_ce')
    parser.add_argument('--warmstart', type=str, default='/workspace/checkpoints/proj_v8_final.pt')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.samples_out, exist_ok=True)

    print("==================================================================", flush=True)
    print("  GEMMA-4 OMNI: V8.6 PURE CROSS-ENTROPY TRAINING (MLP 25M)", flush=True)
    print("  [FREE FROM GEOMETRIC LOCK] lambda_m = 0.0, lambda_norm = 0.0", flush=True)
    print(f"  steps={args.steps} batch={args.batch} lr={args.lr} lambda_r={args.lambda_r}", flush=True)
    print(f"  warmstart={args.warmstart}", flush=True)
    print("==================================================================", flush=True)

    # 1. Caricamento Qwen3-TTS (100% FROZEN)
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

    # VINCOLO INDEROGABILE: TUTTO QWEN3-TTS E' 100% CONGELATO
    for p in talker.parameters():
        p.requires_grad = False
    for p in talker.code_predictor.parameters():
        p.requires_grad = False

    tok_qwen = AutoTokenizer.from_pretrained(snap, use_fast=True)
    proc_qwen = _M.Qwen3TTSProcessor(tokenizer=tok_qwen)

    snap_tok = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*'))[-1]
    tok_decoder = Qwen3TTSTokenizerV2Model.from_pretrained(snap_tok).cuda().eval()

    gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    tok_gemma = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True) if len(gemma_dirs) > 0 else None

    CFG = teacher.config
    TCFG = CFG.talker_config
    lang_map = getattr(TCFG, 'codec_language_id', {})
    lang_id = lang_map.get('italian', 2070)

    # 2. Inizializza Proiettore MLP V8 con Warmstart Step 25000
    proj = TextAnchoredProjectionV8().to(device='cuda', dtype=torch.bfloat16)
    if os.path.exists(args.warmstart):
        sd = torch.load(args.warmstart, map_location='cpu')
        proj.load_state_dict(sd.get('proj', sd), strict=False)
        print(f"  [WARMSTART] Pesi caricati con successo da {args.warmstart} (gamma={proj.gamma.item():.4f})", flush=True)
    else:
        print(f"  [WARN] Checkpoint warmstart non trovato in {args.warmstart}, inizio da zero.", flush=True)

    optimizer = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    # 3. Dati
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
    print(f"  Dataset: {N_train} finestre train, {N_val} finestre validation.", flush=True)

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

    # 4. Validation Function
    def run_validation_eval(val_samples=10):
        proj.eval()
        v_cb0_loss, v_cb0_acc = 0.0, 0.0
        v_res_loss, v_res_acc = 0.0, 0.0
        v_cos, v_norm = 0.0, 0.0

        with torch.no_grad():
            for v_i in range(val_samples):
                c0 = val_start + v_i * 64
                codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
                codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
                codes = torch.clamp(codes, 0, 2047)
                n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
                if n_real < 16: continue
                codes = codes[:n_real]
                T = n_real

                gi = (val_start // 64) + v_i
                tbw_entry = tbw[gi] if gi < len(tbw) else None
                if isinstance(tbw_entry, (list, np.ndarray)) and tok_gemma is not None:
                    text_str = tok_gemma.decode(tbw_entry, skip_special_tokens=True).strip()
                else:
                    text_str = str(tbw_entry).strip() if tbw_entry is not None else "Testo di validazione."

                h_raw_sample = torch.from_numpy(np.asarray(va_raw[v_i][:n_real], dtype=np.float32)).to('cuda')
                h_tx_sample = torch.from_numpy(np.asarray(va_htx[v_i], dtype=np.float32)).to('cuda')

                h_cond = proj(h_raw_sample.unsqueeze(0).to(torch.bfloat16), h_tx_sample.unsqueeze(0).to(torch.bfloat16))
                icl = build_official_prefill(text_str)
                L_prefill = icl.shape[1]

                codes_prev = codes[:-1]
                agg_audio = sum(talker.get_input_embeddings()(codes_prev[:, 0:1]) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](codes_prev[:, g:g+1]) for g in range(16)).transpose(0, 1)

                audio_embeds_omni = agg_audio + h_cond[:, 1:T, :]
                inp_omni = torch.cat([icl, audio_embeds_omni], dim=1)
                out_omni = talker.model(inputs_embeds=inp_omni, use_cache=False)
                h_omni = out_omni.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]

                inp_nat = torch.cat([icl, agg_audio], dim=1)
                out_nat = talker.model(inputs_embeds=inp_nat, use_cache=False)
                h_nat = out_nat.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]

                cb0_lg = talker.codec_head(h_omni)
                loss_cb0 = F.cross_entropy(cb0_lg.reshape(-1, cb0_lg.shape[-1]), codes[:, 0].reshape(-1))
                cb0_acc = (cb0_lg.argmax(dim=-1) == codes[:, 0]).float().mean().item() * 100

                sub_lg, loss_res = talker.forward_sub_talker_finetune(codes, h_omni[0])
                res_acc = (sub_lg.argmax(dim=-1) == codes[:, 1:]).float().mean().item() * 100

                cos_sim = F.cosine_similarity(h_omni.float(), h_nat.float(), dim=-1).mean().item()
                norm_r = (h_omni.norm(dim=-1).mean() / (h_nat.norm(dim=-1).mean() + 1e-6)).item()

                v_cb0_loss += loss_cb0.item()
                v_cb0_acc += cb0_acc
                v_res_loss += loss_res.item()
                v_res_acc += res_acc
                v_cos += cos_sim
                v_norm += norm_r

        proj.train()
        return {
            'cb0_loss': v_cb0_loss / val_samples,
            'cb0_acc': v_cb0_acc / val_samples,
            'res_loss': v_res_loss / val_samples,
            'res_acc': v_res_acc / val_samples,
            'cos_sim': v_cos / val_samples,
            'norm_ratio': v_norm / val_samples,
        }

    # 5. Training Loop Pure-CE
    print("\n[TRAINING] Avvio 1000 step di addestramento Pure Cross-Entropy su RTX 3090...", flush=True)
    t_start = time.time()
    proj.train()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        loss_batch = 0.0
        cb0_batch = 0.0
        res_batch = 0.0
        cos_batch = 0.0
        cb0_acc_batch = 0.0
        res_acc_batch = 0.0
        norm_batch = 0.0

        for b in range(args.batch):
            idx = (step * args.batch + b) % N_train
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

            # PURE CROSS-ENTROPY LOSSES (Zero Cosine Loss, Zero Norm Loss)
            cb0_lg = talker.codec_head(h_omni)
            loss_cb0 = F.cross_entropy(cb0_lg.reshape(-1, cb0_lg.shape[-1]), codes[:, 0].reshape(-1))

            sub_lg, loss_res = talker.forward_sub_talker_finetune(codes, h_omni[0])

            # Loss Pura senza vincoli angolari artificiali
            loss_total = (1.0 * loss_cb0 + args.lambda_r * loss_res) / args.batch
            loss_total.backward()

            # Metriche passive di monitoraggio (senza gradiente)
            with torch.no_grad():
                inp_nat = torch.cat([icl, agg_audio], dim=1)
                out_nat = talker.model(inputs_embeds=inp_nat, use_cache=False)
                h_nat = out_nat.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]
                cos_sim = F.cosine_similarity(h_omni.float(), h_nat.float(), dim=-1).mean()
                norm_ratio = h_omni.norm(dim=-1).mean() / (h_nat.norm(dim=-1).mean() + 1e-6)

                cb0_acc = (cb0_lg.argmax(dim=-1) == codes[:, 0]).float().mean().item() * 100
                res_acc = (sub_lg.argmax(dim=-1) == codes[:, 1:]).float().mean().item() * 100

            loss_batch += loss_total.item() * args.batch
            cb0_batch += loss_cb0.item()
            res_batch += loss_res.item()
            cos_batch += cos_sim.item()
            norm_batch += norm_ratio.item()
            cb0_acc_batch += cb0_acc
            res_acc_batch += res_acc

        torch.nn.utils.clip_grad_norm_(proj.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t_start

        # Logging ogni 10 step (media per sample)
        if step % 10 == 0 or step == 1:
            loss_avg = loss_batch / args.batch
            cb0_avg = cb0_batch / args.batch
            res_avg = res_batch / args.batch
            cos_avg = cos_batch / args.batch
            cb0_acc_avg = cb0_acc_batch / args.batch
            res_acc_avg = res_acc_batch / args.batch
            norm_avg = norm_batch / args.batch
            lr_curr = scheduler.get_last_lr()[0]
            print(f"[TRAIN-MLP {step:04d}/{args.steps:04d}] Sample Loss: {loss_avg:.3f} | CB0: {cb0_avg:.3f} (Acc: {cb0_acc_avg:.1f}%) | "
                  f"Res: {res_avg:.3f} (Acc: {res_acc_avg:.1f}%) | Cos (passivo): {cos_avg:.4f} | Norm: {norm_avg:.4f} | gamma={proj.gamma.item():.4f} "
                  f"({elapsed/step:.2f}s/step)", flush=True)

        # EVAL ogni 100 step
        if step % args.eval_every == 0:
            val_metrics = run_validation_eval(val_samples=10)
            print(f"\n>>> [EVAL-MLP @ STEP {step:04d}] Val CB0 Loss: {val_metrics['cb0_loss']:.3f} | Val CB0 Acc: {val_metrics['cb0_acc']:.1f}% | "
                  f"Val Res Loss: {val_metrics['res_loss']:.3f} | Val Res Acc: {val_metrics['res_acc']:.1f}% | "
                  f"Val Cos (passivo): {val_metrics['cos_sim']:.4f} | Val Norm: {val_metrics['norm_ratio']:.4f} <<<\n", flush=True)

        # CHECKPOINT ogni 500 step
        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = os.path.join(args.out, f"v8_pure_ce_step{step}.pt")
            torch.save({
                'step': step,
                'proj': proj.state_dict(),
                'optimizer': optimizer.state_dict(),
                'gamma': proj.gamma.item(),
                'args': vars(args)
            }, ckpt_path)
            print(f"  [CHECKPOINT] Salvato con successo in {ckpt_path}", flush=True)

    # Checkpoint Finale
    final_ckpt = os.path.join(args.out, "v8_pure_ce_final.pt")
    torch.save({
        'step': args.steps,
        'proj': proj.state_dict(),
        'gamma': proj.gamma.item(),
    }, final_ckpt)
    print(f"\n[COMPLETATO] Training MLP Pure-CE 1000 step terminato. Checkpoint: {final_ckpt}", flush=True)

    # -------------------------------------------------------------
    # SINTESI COMPARATIVA AUDIO
    # -------------------------------------------------------------
    print("\n=== SINTESI COMPARATIVA AUDIO (MLP PURE-CE) ===", flush=True)
    proj.eval()

    sample_idx = 1
    c0 = val_start + sample_idx * 64
    codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
    codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
    codes = torch.clamp(codes, 0, 2047)
    n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
    if n_real < 16: n_real = 64
    codes = codes[:n_real]
    T = n_real

    gi = (val_start // 64) + sample_idx
    tbw_entry = tbw[gi] if gi < len(tbw) else None
    if isinstance(tbw_entry, (list, np.ndarray)) and tok_gemma is not None:
        text_str = tok_gemma.decode(tbw_entry, skip_special_tokens=True).strip()
    else:
        text_str = str(tbw_entry).strip() if tbw_entry is not None else "Si tratta del secondo singolo promozionale nella storia della band."

    h_raw = torch.from_numpy(np.asarray(va_raw[sample_idx][:n_real], dtype=np.float32)).to('cuda')
    h_tx = torch.from_numpy(np.asarray(va_htx[sample_idx], dtype=np.float32)).to('cuda')

    whisper_dirs = glob.glob('/workspace/.hf_cache/models--Systran--faster-whisper-small/snapshots/*')
    whisper_path = whisper_dirs[0] if len(whisper_dirs) > 0 else "small"
    whisper_model = WhisperModel(whisper_path, device="cuda", compute_type="float16", download_root="/workspace/.hf_cache")

    with torch.no_grad():
        h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))
        icl = build_official_prefill(text_str)

        # Generazione OMNI MLP Pure-CE
        out_free = talker.model(inputs_embeds=icl, use_cache=True)
        past_kv = out_free.past_key_values
        last_h = out_free.last_hidden_state[:, -1:, :]

        codes_omni_mlp = []
        for t in range(T):
            cb0_lg = talker.codec_head(last_h)[:, -1, :]
            cb0_lg[0, 2048:] = -float('Inf')
            c0_val = int(cb0_lg.argmax(dim=-1).item())

            last_id_h = talker.get_input_embeddings()(torch.tensor([[c0_val]], device='cuda'))
            pred_res = talker.code_predictor.generate(
                inputs_embeds=torch.cat((last_h, last_id_h), dim=1),
                max_new_tokens=15,
                do_sample=False,
                return_dict_in_generate=True,
            )
            res_codes = pred_res.sequences[0].cpu().tolist()
            full_t = [c0_val] + res_codes
            codes_omni_mlp.append(full_t)

            if t < T - 1:
                cur_codes_t = torch.tensor(full_t, device='cuda')
                agg_t = talker.get_input_embeddings()(cur_codes_t[0:1].unsqueeze(0))
                for g in range(1, 16):
                    agg_t = agg_t + talker.code_predictor.get_input_embeddings()[g-1](cur_codes_t[g:g+1].unsqueeze(0))
                next_embed = agg_t.transpose(0, 1) + h_cond[:, t+1:t+2, :]
                out_step = talker.model(inputs_embeds=next_embed, past_key_values=past_kv, use_cache=True)
                past_kv = out_step.past_key_values
                last_h = out_step.last_hidden_state[:, -1:, :]

        t_omni_mlp = torch.clamp(torch.tensor(codes_omni_mlp, device='cuda', dtype=torch.long).unsqueeze(0), 0, 2047)
        wav_omni = tok_decoder.decode(t_omni_mlp).audio_values[0].cpu().numpy().astype(np.float32)
        p_peak_o = float(np.percentile(np.abs(wav_omni), 99.7))
        wav_omni = np.clip(wav_omni / (p_peak_o * 1.05), -1.0, 1.0) if p_peak_o > 1e-6 else wav_omni
        path_omni = os.path.join(args.samples_out, "sample_1_omni_mlp_pure_ce.wav")
        sf.write(path_omni, wav_omni, 24000)

        segs_o, _ = whisper_model.transcribe(path_omni, language="it", temperature=0.0)
        text_omni = " ".join([s.text for s in segs_o]).strip()
        print(f"  [AUDIO OMNI MLP PURE-CE]: \"{text_omni}\" -> {path_omni}", flush=True)

        report = {
            'step': args.steps,
            'target_text': text_str,
            'omni_mlp_asr': text_omni,
            'omni_wav_path': path_omni,
        }
        json_path = os.path.join(args.samples_out, "v8_pure_ce_report.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  Report salvato in {json_path}", flush=True)

if __name__ == '__main__':
    main()
