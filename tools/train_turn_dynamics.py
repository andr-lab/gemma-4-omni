#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script per TurnDynamicsModule su RTX 3090 (20h Dataset, 836k frame).
Caratteristiche:
  1. Caricamento memory-mapped ad altissima velocità da turnhead_20h_z_feats.npy (Dim=2304)
  2. Partizionamento Train / Validation (85% / 15% = ~710k train / ~126k val)
  3. CalibratedTurnFocalLoss (ASPIRin & DuplexCascade) con pesi bilanciati
  4. Ottimizzatore AdamW + Cosine Annealing LR Scheduler
  5. Checkpoint su Macro F1 & Balanced Accuracy (zero collasso su SPEAK)
  6. Salvataggio checkpoint finale in checkpoints/turn_dynamics_v1_final.pt
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from turn_dynamics_module import (
    TurnDynamicsModule,
    TurnActionHead,
    LearnableStateEmbeddings,
    CalibratedTurnFocalLoss,
    ACTION_NAMES,
    NUM_ACTIONS,
)

class TurnFeaturesDataset(Dataset):
    def __init__(self, z_feats_mmap: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.z_feats = z_feats_mmap
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        feat = torch.from_numpy(self.z_feats[real_idx].copy()).float()
        label = int(self.labels[real_idx])
        return feat, label


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for feats, targets in loader:
            feats = feats.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model.action_head(feats)
            loss = criterion(logits, targets)

            total_loss += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calcolo metriche per classe
    unique_classes = np.unique(all_targets)
    recalls = {}
    precisions = {}
    f1s = {}

    for c in range(NUM_ACTIONS):
        mask_true = (all_targets == c)
        mask_pred = (all_preds == c)
        tp = np.logical_and(mask_true, mask_pred).sum()
        fp = np.logical_and(~mask_true, mask_pred).sum()
        fn = np.logical_and(mask_true, ~mask_pred).sum()

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        recalls[c] = float(rec)
        precisions[c] = float(prec)
        f1s[c] = float(f1)

    overall_acc = float((all_preds == all_targets).mean())
    active_classes = [c for c in unique_classes]
    balanced_acc = float(np.mean([recalls[c] for c in active_classes]))
    macro_f1 = float(np.mean([f1s[c] for c in active_classes]))

    # Matrice di confusione
    conf_matrix = np.zeros((NUM_ACTIONS, NUM_ACTIONS), dtype=int)
    for t, p in zip(all_targets, all_preds):
        if 0 <= t < NUM_ACTIONS and 0 <= p < NUM_ACTIONS:
            conf_matrix[t, p] += 1

    return {
        "val_loss": total_loss / max(1, num_batches),
        "accuracy": overall_acc,
        "balanced_acc": balanced_acc,
        "macro_f1": macro_f1,
        "recalls": recalls,
        "precisions": precisions,
        "f1s": f1s,
        "confusion_matrix": conf_matrix.tolist(),
    }


def train_turn_dynamics(
    data_dir: str = "/workspace/data",
    epochs: int = 15,
    batch_size: int = 4096,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_ckpt: str = "/workspace/checkpoints/turn_dynamics_v1_final.pt",
):
    print(f"=== [START] Training TurnDynamicsModule su {device} ===", flush=True)
    feats_path = os.path.join(data_dir, "turnhead_20h_z_feats.npy")
    labels_path = os.path.join(data_dir, "turnhead_20h_labels.npy")

    if not os.path.exists(feats_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"File dati mancanti: {feats_path} o {labels_path}")

    # 1. Caricamento dati mmap
    z_feats = np.load(feats_path, mmap_mode="r")
    labels = np.load(labels_path)
    N, D = z_feats.shape
    print(f"  * Dataset Caricato: N={N:,} frames | Dim={D} (~{N*0.08/3600:.2f} ore)", flush=True)

    # 2. Split Train / Val (85% / 15%)
    np.random.seed(42)
    indices = np.random.permutation(N)
    split = int(0.85 * N)
    train_indices = indices[:split]
    val_indices = indices[split:]
    print(f"  * Split: Train={len(train_indices):,} | Val={len(val_indices):,}", flush=True)

    train_ds = TurnFeaturesDataset(z_feats, labels, train_indices)
    val_ds = TurnFeaturesDataset(z_feats, labels, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=0, pin_memory=True)

    # 3. Inizializzazione Modello e Criterio
    model = TurnDynamicsModule(
        in_dim=D,
        out_dim=2048,
        hidden_dim=512,
        mid_dim=128,
        gamma=0.25,
    ).to(device)

    # Pesi per classe calibrati su sbilanciamento reale (SPEAK=1.0, YIELD=4.0, BARGE_IN=1.5, EMIT_BC=2.5)
    class_weights = [1.0, 4.0, 1.5, 3.0, 2.5]
    criterion = CalibratedTurnFocalLoss(class_weights=class_weights, gamma=1.2, label_smoothing=0.04).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_score = 0.0
    best_metrics = {}
    os.makedirs(os.path.dirname(output_ckpt), exist_ok=True)

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for feats, targets in train_loader:
            feats = feats.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model.action_head(feats)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        scheduler.step()
        train_loss_avg = train_loss / max(1, n_train_batches)
        dt = time.time() - t0

        # Validazione a fine epoca
        metrics = evaluate(model, val_loader, criterion, device)
        comp_score = 0.5 * metrics["balanced_acc"] + 0.5 * metrics["macro_f1"]

        # Log sintetico
        rec_str = " | ".join([f"{ACTION_NAMES[c]}: {metrics['recalls'][c]*100:.1f}%" for c in range(NUM_ACTIONS)])
        print(
            f"[EPOCA {epoch:02d}/{epochs:02d}] ({dt:.1f}s) "
            f"Loss: Tr={train_loss_avg:.4f} Val={metrics['val_loss']:.4f} | "
            f"Acc={metrics['accuracy']*100:.2f}% BalAcc={metrics['balanced_acc']*100:.2f}% "
            f"MacroF1={metrics['macro_f1']*100:.2f}% | Score={comp_score*100:.2f}%\n"
            f"    -> Recall: {rec_str}",
            flush=True
        )

        if comp_score > best_score:
            best_score = comp_score
            best_metrics = metrics
            best_metrics["epoch"] = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "in_dim": D,
                "out_dim": 2048,
                "best_score": best_score,
                "metrics": best_metrics,
            }, output_ckpt)
            print(f"    * Nuovo Checkpoint Salvato -> {output_ckpt} (Score: {best_score*100:.2f}%)", flush=True)

    print("\n=== [COMPLETATO] Training TurnDynamicsModule Terminato con Successo! ===", flush=True)
    print(f"  * Miglior Epoca: {best_metrics.get('epoch', epochs)}")
    print(f"  * Overall Accuracy  : {best_metrics.get('accuracy', 0)*100:.2f}%")
    print(f"  * Balanced Accuracy : {best_metrics.get('balanced_acc', 0)*100:.2f}%")
    print(f"  * Macro F1 Score    : {best_metrics.get('macro_f1', 0)*100:.2f}%")
    print(f"  * Checkpoint Finale : {output_ckpt}", flush=True)

    # Salvataggio report JSON
    report_path = output_ckpt.replace(".pt", "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"  * Report JSON salvato in: {report_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/workspace/data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output_ckpt", type=str, default="/workspace/checkpoints/turn_dynamics_v1_final.pt")
    args = parser.parse_args()

    train_turn_dynamics(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_ckpt=args.output_ckpt,
    )
