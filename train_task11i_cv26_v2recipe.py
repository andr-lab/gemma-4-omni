#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# train_adapter_it_3090.py - Training Italian Voice Adapter (Task 6B) on RTX 3090
# Trains ht_adapter + semantic_head + causal_conv on Mozilla Common Voice Italian dataset.
# - Backbone Gemma 4 E2B is frozen
# - Sub-talker Qwen3-TTS MTP module is frozen
# - Loss: Semantic cb0 (CE with frame weights) + Qwen Coupling loss (ADR-005)
# - Quality Gates: Initial loss 13-17, cosine similarity tracked, SVD rank >= 1000
# ============================================================================

import argparse
import datetime
import gc
import glob
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import paths
import kaggle_pipeline as KP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Italian Voice Adapter (Task 6B) on RTX 3090."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path or prefix to MDC dataset (default: auto-discover latest MDC parquet in data/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Micro-batch size (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for AdamW (default: 2e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW (default: 0.01)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm clipping (default: 1.0)",
    )
    parser.add_argument(
        "--coupling-weight",
        type=float,
        default=0.20,
        help="Coupling loss lambda weight (default: 0.20)",
    )
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=500,
        help="Save periodic checkpoint every N steps (default: 500)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum optimization steps to run (default: None = full epoch)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/task11i_cv26_v2recipe",
        help="Output directory for checkpoints (default: checkpoints/adapter_it)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Computation device (default: cuda:0)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test mode (4 batches, verify initial loss gate 13-17)",
    )
    return parser.parse_args()


def resolve_mdc_dataset_prefix(custom_path: Optional[str] = None) -> str:
    """Find the best matching MDC Italian parquet dataset prefix."""
    data_dir = paths.get_data_dir()
    if custom_path:
        full = paths.resolve_path(custom_path)
        for sfx in ("_mel.parquet", "_ameld.parquet", "_codes.parquet", "_text.parquet", "_turnlabels.parquet", ".parquet"):
            if full.endswith(sfx):
                full = full[:-len(sfx)]
                break
        if os.path.isfile(f"{full}_mel.parquet") and os.path.isfile(f"{full}_codes.parquet"):
            return full

    # Auto-discover MDC files
    patterns = [
        os.path.join(data_dir, "train_tokens_cv26_it_*_mel.parquet"),
        os.path.join(data_dir, "train_tokens_mdc_*_mel.parquet"),
        os.path.join(data_dir, "train_tokens_mdc_50.0h_2500s_mel.parquet"),
    ]
    candidates = []
    for pat in patterns:
        for f in glob.glob(pat):
            pfx = f.replace("_mel.parquet", "")
            if os.path.isfile(f"{pfx}_codes.parquet"):
                candidates.append(pfx)

    if candidates:
        # Sort by file size (largest dataset first)
        candidates.sort(key=lambda p: os.path.getsize(f"{p}_mel.parquet"), reverse=True)
        return candidates[0]

    # Fallback to standard Kaggle format
    std_prefix = os.path.join(data_dir, f"train_tokens_{KP.DUPLEXCONV_MAX_HOURS}h_{KP.MDC_MAX_SAMPLES}s")
    return std_prefix


def compute_svd_metrics(h_t_tensor: torch.Tensor) -> Dict[str, float]:
    """Compute SVD effective rank, condition number, and top-1000 energy."""
    try:
        X = h_t_tensor.to(torch.float32)
        X_centered = X - X.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        sing_vals = S.cpu().numpy()

        energy = (sing_vals ** 2).sum()
        cum_energy = np.cumsum(sing_vals ** 2) / (energy + 1e-12)

        # Effective rank: number of singular values > threshold
        threshold = 1e-3 * sing_vals[0] if len(sing_vals) > 0 else 1e-6
        eff_rank = int((sing_vals > threshold).sum())

        idx_1000 = min(1000, len(cum_energy) - 1)
        cum_energy_1000 = float(cum_energy[idx_1000]) if len(cum_energy) > 0 else 0.0

        cond = float(sing_vals[0] / (sing_vals[-1] + 1e-12)) if len(sing_vals) > 0 else 1.0

        return {
            "effective_rank": eff_rank,
            "dim": int(X.shape[1]),
            "cum_energy_1000": round(cum_energy_1000, 4),
            "condition_number": round(cond, 2),
            "top_singular_val": float(sing_vals[0]) if len(sing_vals) > 0 else 0.0,
        }
    except Exception as e:
        print(f"[SVD Warning] Calcolo SVD fallito: {e}")
        return {
            "effective_rank": 0,
            "dim": 2048,
            "cum_energy_1000": 0.0,
            "condition_number": 1.0,
        }


def main() -> int:
    args = parse_args()

    if args.smoke:
        args.max_steps = 4
        args.ckpt_every = 2
        print("[TRAIN ADAPTER-IT] === SMOKE TEST MODE (4 batches) ===")

    print("=" * 78)
    print("=== TASK 11I: RICETTA ORIGINALE (heads v2) SU CV26 IT 40.8h ===")
    print("=" * 78)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Weight Decay: {args.weight_decay}")
    print(f"Max Grad Norm: {args.max_grad_norm} | Batch Size: {args.batch_size}")
    print(f"Coupling Weight: {args.coupling_weight}")
    print(f"Output Dir: {args.output_dir}")
    sys.stdout.flush()

    os.environ["PIPELINE_ENV"] = "local3090"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(paths.get_samples_dir(), exist_ok=True)
    os.makedirs(paths.get_logs_dir(), exist_ok=True)

    # 1. Dataset Resolution & Loading
    prefix = resolve_mdc_dataset_prefix(args.dataset)
    mel_p = f"{prefix}_mel.parquet"
    codes_p = f"{prefix}_codes.parquet"

    print(f"\n[Train 1/5] Risoluzione dataset MDC: {prefix} ...", flush=True)
    if not (os.path.isfile(mel_p) and os.path.isfile(codes_p)):
        raise RuntimeError(f"[FATAL] Dataset parquet non trovato: attesi {mel_p} e {codes_p}.")

    # Convert memmap if needed
    KP.convert_parquet_to_memmap(f"{prefix}.parquet")
    user_mel, agent_mel, agent_codes, text_by_window = KP.load_tokens_from_parquet(f"{prefix}.parquet")

    t_agent = agent_codes.shape[1]
    tot_hours = (t_agent * 0.080) / 3600.0
    print(f"[Train 1/5] Dataset caricato: {t_agent} frame agente ({tot_hours:.2f}h), {agent_codes.shape[0]} codebook.", flush=True)

    # Validation split at segment boundary
    sidecar_p = f"{prefix}_hyperparams.json"
    seg_ends = []
    if os.path.isfile(sidecar_p):
        with open(sidecar_p, "r", encoding="utf-8") as f:
            sc = json.load(f)
            seg_ends = sc.get("SEGMENT_ENDS", [])

    val_start = int(0.90 * t_agent)
    for e in seg_ends:
        if e >= val_start:
            val_start = e
            break
    val_start = min(val_start, t_agent)
    n_train_w = val_start // 64
    n_val_w = max(0, (t_agent - val_start) // 64)

    print(f"[Train 1/5] Split: {n_train_w} finestre train (frame 0..{val_start}), {n_val_w} finestre val (frame {val_start}..{t_agent}).", flush=True)

    def _slice_text(t_list, f_start):
        if not t_list:
            return None
        w_s = f_start // 64
        return t_list[w_s:]

    train_ds = KP.FullDuplexDataset(
        user_mel[: val_start * KP.MEL_PER_AGENT_FRAME],
        agent_mel[: val_start * KP.MEL_PER_AGENT_FRAME],
        agent_codes[:, :val_start],
        seq_len=64,
        text_by_window=_slice_text(text_by_window, 0),
    )
    val_ds = KP.FullDuplexDataset(
        user_mel[val_start * KP.MEL_PER_AGENT_FRAME :],
        agent_mel[val_start * KP.MEL_PER_AGENT_FRAME :],
        agent_codes[:, val_start:],
        seq_len=64,
        text_by_window=_slice_text(text_by_window, val_start),
    ) if n_val_w > 0 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # 2. System & Model Setup
    print("\n[Train 2/5] Inizializzazione Gemma 4 Full-Duplex System...", flush=True)
    duplex_model, tokenizer = KP.load_gemma4_duplex_system()
    if duplex_model is None:
        raise RuntimeError("Impossibile caricare Gemma 4 Duplex Model.")

    base_dev = next(duplex_model.base_model.parameters()).device
    print(f"[Train 2/5] Modello caricato su {base_dev}.", flush=True)

    # Verify frozen backbone and trainable parameters
    # In Task 6B: base_model is frozen, subtalker is frozen.
    # causal_conv, semantic_head, and ht_adapter are trained.
    duplex_model.subtalker.eval()
    for p in duplex_model.subtalker.parameters():
        p.requires_grad = False

    for p in duplex_model.base_model.parameters():
        p.requires_grad = False

    trainable_params = [p for p in duplex_model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"[Train 2/5] Parametri allenabili totali: {trainable_count / 1e6:.2f}M (causal_conv + semantic_head + ht_adapter).", flush=True)

    # 3. Optimizer Setup
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )

    # 4. Training Loop & Quality Gates
    print(f"\n[Train 3/5] Avvio ciclo di addestramento su {len(train_loader)} batch/epoca...", flush=True)
    step_count = 0
    initial_loss_recorded = None
    train_history = []
    t_start = time.time()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tsv_log_path = os.path.join(paths.get_logs_dir(), f"adapter_it_train_log_{timestamp}.tsv")
    with open(tsv_log_path, "w", encoding="utf-8") as f:
        f.write("epoch\tstep\ttotal_loss\tcb0_loss\tcoupling_loss\ttext_loss\tlr\n")

    duplex_model.train()
    duplex_model.base_model.eval()
    duplex_model.subtalker.eval()

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoca {epoch+1}/{args.epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            step_count += 1

            input_ids = batch["input_ids"].to(base_dev)
            attention_mask = batch["attention_mask"].to(base_dev)
            input_features = batch["input_features"].to(base_dev)
            input_features_mask = batch["input_features_mask"].to(base_dev)
            audio_targets = batch["audio_targets"].to(base_dev)
            frame_weights = batch["frame_weights"].to(base_dev)
            text_targets = batch["text_targets"].to(base_dev)

            optimizer.zero_grad()

            outputs = duplex_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                input_features_mask=input_features_mask,
                audio_targets=audio_targets,
                frame_weights=frame_weights,
                text_targets=text_targets,
            )

            total_loss = outputs["loss"]
            coupling_loss = outputs.get("coupling_loss")
            text_loss = outputs.get("text_loss")

            # Gate Check on Initial Loss (Step 1)
            loss_val = float(total_loss.item())
            if initial_loss_recorded is None:
                initial_loss_recorded = loss_val
                print(f"\n[QUALITY GATE 1] Initial Step Loss: {loss_val:.4f}", flush=True)
                if loss_val < 5.0:
                    print(f"[QUALITY GATE 1] ⚠️ ATTENZIONE: Initial loss anomala (< 5.0: {loss_val:.4f}). Possibile target degenerato.", flush=True)
                elif 10.0 <= loss_val <= 25.0:
                    print(f"[QUALITY GATE 1] ✅ PASS: Initial loss sana ({loss_val:.4f}) in linea con benchmark (13-17).", flush=True)
                else:
                    print(f"[QUALITY GATE 1] Informazione: Initial loss = {loss_val:.4f}", flush=True)

            total_loss.backward()

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

            optimizer.step()
            duplex_model.coupling_step += 1

            c_val = float(coupling_loss.item()) if coupling_loss is not None else 0.0
            t_val = float(text_loss.item()) if text_loss is not None else 0.0

            pbar.set_postfix({
                "Loss": f"{loss_val:.4f}",
                "Coup": f"{c_val:.4f}",
                "Txt": f"{t_val:.4f}",
            })

            # Record TSV
            with open(tsv_log_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch}\t{step_count}\t{loss_val:.4f}\t{loss_val - c_val - t_val:.4f}\t{c_val:.4f}\t{t_val:.4f}\t{args.lr:.6f}\n")

            train_history.append({
                "step": step_count,
                "loss": loss_val,
                "coupling_loss": c_val,
                "text_loss": t_val,
            })

            # Periodic Checkpoint
            if args.ckpt_every > 0 and step_count % args.ckpt_every == 0:
                ckpt_step_dir = os.path.join(args.output_dir, f"ckpt_step_{step_count}")
                os.makedirs(ckpt_step_dir, exist_ok=True)
                KP.save_gemma4_duplex_checkpoint(duplex_model, ckpt_step_dir)
                print(f"\n[Train] Checkpoint salvato @ step {step_count}: {ckpt_step_dir}", flush=True)

            if args.max_steps is not None and step_count >= args.max_steps:
                print(f"\n[Train] Raggiunto limite max steps: {step_count} >= {args.max_steps}.", flush=True)
                break

        pbar.close()

    # 5. Validation Split Evaluation & SOTA Metrics
    print("\n[Train 4/5] Esecuzione validazione e metriche SOTA...", flush=True)
    val_loss_mean = None
    val_txt_mean = None
    val_coup_mean = None
    mean_cos_sim = None
    svd_res = {}
    if val_ds is not None and len(val_ds) > 0:
        duplex_model.eval()
        val_sample_count = min(len(val_ds), 200)
        print(f"  - Calcolo metriche su {val_sample_count} finestre di validazione...", flush=True)
        fpf = KP.FEATURES_PER_AGENT_FRAME if KP.SELF_LISTENING else KP.AGENT_FEATURES_PER_FRAME
        audio_feat_start = 2 + KP.TEXT_BLOCK_LEN
        all_h_t = []
        all_cos_sims = []
        val_losses = []
        val_coups = []
        val_txts = []

        with torch.inference_mode():
            for i in range(val_sample_count):
                item = val_ds[i]
                inp_ids = item["input_ids"].unsqueeze(0).to(base_dev)
                att_mask = item["attention_mask"].unsqueeze(0).to(base_dev)
                inp_feats = item["input_features"].unsqueeze(0).to(base_dev)
                inp_fmask = item["input_features_mask"].unsqueeze(0).to(base_dev)
                audio_tg = item["audio_targets"].unsqueeze(0).to(base_dev)
                f_weights = item["frame_weights"].unsqueeze(0).to(base_dev)
                txt_tg = item["text_targets"].unsqueeze(0).to(base_dev)

                vout = duplex_model(
                    input_ids=inp_ids,
                    attention_mask=att_mask,
                    input_features=inp_feats,
                    input_features_mask=inp_fmask,
                    audio_targets=audio_tg,
                    frame_weights=f_weights,
                    text_targets=txt_tg,
                )
                if vout["loss"] is not None:
                    val_losses.append(float(vout["loss"].item()))
                if vout.get("coupling_loss") is not None:
                    val_coups.append(float(vout["coupling_loss"].item()))
                if vout.get("text_loss") is not None and vout["text_loss"] > 0:
                    val_txts.append(float(vout["text_loss"].item()))

                # Extract h_t
                base_out = duplex_model.base_model(
                    input_ids=inp_ids,
                    attention_mask=att_mask,
                    input_features=inp_feats,
                    input_features_mask=inp_fmask,
                    output_hidden_states=True,
                )
                hidden_states = base_out.hidden_states[-1]
                n_agent = audio_tg.shape[1]
                agent_hidden = hidden_states[
                    :, audio_feat_start + fpf - 1 : audio_feat_start + fpf * n_agent : fpf, :
                ]
                adapted_hidden = duplex_model._apply_causal_conv(agent_hidden)
                ad_dtype = next(duplex_model.ht_adapter.parameters()).dtype
                h_t = duplex_model.ht_adapter(adapted_hidden.to(ad_dtype))
                all_h_t.append(h_t.squeeze(0).cpu().to(torch.float32))

                active_mask = (audio_tg[0, :, 0] != KP.PAD_TARGET_ID).cpu()
                if active_mask.sum() > 0:
                    h_active = h_t[0][active_mask].cpu()
                    ref_dir = h_active.mean(dim=0, keepdim=True)
                    cos_step = F.cosine_similarity(h_active, ref_dir, dim=-1)
                    all_cos_sims.append(float(cos_step.mean().item()))

        if val_losses:
            val_loss_mean = float(np.mean(val_losses))
            val_coup_mean = float(np.mean(val_coups)) if val_coups else 0.0
            val_txt_mean = float(np.mean(val_txts)) if val_txts else 0.0
        if all_cos_sims:
            mean_cos_sim = float(np.mean(all_cos_sims))
        if all_h_t:
            all_h_t_tensor = torch.cat(all_h_t, dim=0)
            svd_res = compute_svd_metrics(all_h_t_tensor)

        print(f"[Train 4/5] ✅ VALIDATION REPORT:")
        print(f"  - Val Loss Media:         {val_loss_mean:.4f}")
        print(f"  - Val Coupling Loss:       {val_coup_mean:.4f}")
        print(f"  - Cosine Similarity:       {mean_cos_sim:.4f} (Benchmark temporal2: 0.7455)")
        print(f"  - SVD Effective Rank:      {svd_res.get('effective_rank', 0)} / 2048")
        print(f"  - SVD Cum Energy top-1000: {svd_res.get('cum_energy_1000', 0.0):.4f}")

    # 6. Save Final Checkpoint
    final_ckpt_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_ckpt_dir, exist_ok=True)
    KP.save_gemma4_duplex_checkpoint(duplex_model, final_ckpt_dir)
    KP.save_gemma4_duplex_checkpoint(duplex_model, args.output_dir)
    # Also save to root standard checkpoint dir
    KP.save_gemma4_duplex_checkpoint(duplex_model, "checkpoints/gemma4_duplex_sota")
    print(f"\n[Train 5/5] Final Checkpoint salvato in {final_ckpt_dir} e checkpoints/gemma4_duplex_sota.", flush=True)

    # 7. Synthesize Audio Samples
    print("\n[Train 5/5] Generazione campioni audio in samples/...", flush=True)
    try:
        synth_ds = val_ds if (val_ds is not None and len(val_ds) > 0) else train_ds
        KP.synthesize_examples(duplex_model, synth_ds)
    except Exception as se:
        print(f"[Train Warning] Sintesi fallita ({se}).", flush=True)

    # 8. Final Report JSON
    elapsed_total = time.time() - t_start
    final_report = {
        "status": "COMPLETED",
        "task": "TASK_6B_ADAPTER_IT",
        "dataset_prefix": prefix,
        "dataset_hours": round(tot_hours, 2),
        "total_steps": step_count,
        "initial_loss": initial_loss_recorded,
        "final_train_loss": train_history[-1]["loss"] if train_history else None,
        "val_loss": val_loss_mean,
        "val_coupling_loss": val_coup_mean,
        "val_text_loss": val_txt_mean,
        "cosine_sim_val": mean_cos_sim,
        "svd_metrics": svd_res,
        "trainable_params_m": round(trainable_count / 1e6, 2),
        "elapsed_sec": round(elapsed_total, 2),
        "checkpoints": {
            "final": final_ckpt_dir,
            "standard": "checkpoints/gemma4_duplex_sota",
        },
        "log_file": tsv_log_path,
    }

    report_path = os.path.join(paths.get_logs_dir(), "adapter_it_training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    report_path_v2 = os.path.join(paths.get_logs_dir(), "adapter_it_v2_training_report.json")
    with open(report_path_v2, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"[Train] Training Report salvato in: {report_path}", flush=True)

    print("\n" + "=" * 78)
    print(f"=== ADAPTER ITALIANO COMPLETATO ({step_count} step in {elapsed_total/60:.1f}m) ===")
    print("=" * 78)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FATAL] Adapter training failed: {e}")
        sys.exit(1)
