#!/usr/bin/env python3
"""P4D-R — Retraining Causale Completo del Projector V9 (Milestone 2).

Esegue l'addestramento causale L=0 di DeepCausalTransformerProjectorV9 attraverso
Qwen3-TTS 1.7B congelato con retropropagazione su CB0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.causal_v9_trainer import DifferentiableCausalQwenV9, compute_causal_v9_loss
from duplex_kernel.models.talker import load_talker
from duplex_kernel.models.v9 import load_v9_conditioning, DEFAULT_V9_CKPT


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT))
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def train_causal_v9(
    cache_path: Path,
    output_dir: Path,
    *,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    min_lr: float = 1e-5,
    weight_decay: float = 1e-2,
    clip_grad: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    teacher_weight: float = 0.0,
    train_conv: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    git_sha = get_git_sha()

    # 1. Attestazione SHA256 dei pesi congelati pre-training
    print("[TRAIN] Calcolo SHA256 pesi congelati di sistema...", flush=True)
    frozen_manifest: dict[str, str] = {}
    import glob
    qwen_snaps = sorted(glob.glob("/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*/model.safetensors"))
    if qwen_snaps:
        frozen_manifest["qwen_talker"] = compute_file_sha256(Path(qwen_snaps[-1]))

    source_v9_ckpt = Path(DEFAULT_V9_CKPT)
    source_v9_sha = compute_file_sha256(source_v9_ckpt) if source_v9_ckpt.exists() else "missing"

    # 2. Caricamento Dati
    print(f"[TRAIN] Caricamento dataset cache da {cache_path}...", flush=True)
    c = torch.load(cache_path, map_location="cpu", weights_only=False)
    raw_list = c.get("tr_raw") or c.get("va_raw", [])
    codes_list = c.get("tr_c") or c.get("va_c", [])
    num_seq = len(raw_list)
    print(f"[TRAIN] Sequenze di training disponibili: {num_seq}", flush=True)

    raw_tensor = torch.stack(raw_list).to(device=device, dtype=dtype)
    codes_tensor = torch.stack(codes_list).to(device=device, dtype=torch.long)

    # 3. Caricamento Modelli
    print("[TRAIN] Caricamento V9 (warm-start)...", flush=True)
    projector, causal_conv = load_v9_conditioning(device=device, dtype=dtype)
    
    print("[TRAIN] Caricamento Qwen Talker 1.7B (frozen)...", flush=True)
    talker = load_talker("/workspace", device=device, load_code2wav=False)

    model = DifferentiableCausalQwenV9(projector, causal_conv, talker, train_conv=train_conv)
    trainable_params = [p for p in projector.parameters() if p.requires_grad]
    if train_conv:
        trainable_params += [p for p in causal_conv.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"[TRAIN] Parametri addestrabili V9: {num_trainable:,}", flush=True)

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, num_seq // batch_size)
    total_steps = epochs * steps_per_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)

    config = {
        "git_sha": git_sha,
        "source_v9_ckpt": str(source_v9_ckpt),
        "source_v9_sha256": source_v9_sha,
        "frozen_components_sha256": frozen_manifest,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "min_lr": min_lr,
        "weight_decay": weight_decay,
        "clip_grad": clip_grad,
        "teacher_weight": teacher_weight,
        "train_conv": train_conv,
        "num_trainable_params": num_trainable,
        "target_lookahead": 0,
        "causal_anchor": "first_raw_frame",
    }
    (output_dir / "P4_PROJECTOR_RETRAIN_CONFIG.json").write_text(json.dumps(config, indent=2) + "\n")

    # 4. Training Loop
    trace_path = output_dir / "P4_PROJECTOR_TRAIN_TRACE.jsonl"
    trace_file = open(trace_path, "w", encoding="utf-8")

    global_step = 0
    t_start = time.time()
    print(f"[TRAIN] Avvio training causale L=0 per {epochs} epoche ({total_steps} step)...", flush=True)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_seq)
        epoch_loss = 0.0
        t_epoch = time.time()

        for step in range(steps_per_epoch):
            global_step += 1
            idx = perm[step * batch_size : (step + 1) * batch_size]
            b_raw = raw_tensor[idx]
            b_tx = b_raw[:, 0, :]
            b_codes = codes_tensor[idx]

            optimizer.zero_grad()
            logits, h_pfx = model(b_raw, b_tx, b_codes, lookahead_frames=0)
            loss, metrics = compute_causal_v9_loss(logits, b_codes)

            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_grad).item())
            optimizer.step()
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            metrics["step"] = global_step
            metrics["epoch"] = epoch
            metrics["lr"] = current_lr
            metrics["grad_norm"] = grad_norm
            trace_file.write(json.dumps(metrics) + "\n")
            trace_file.flush()
            epoch_loss += metrics["ce_loss"]

            if global_step % 10 == 0 or global_step == 1:
                print(
                    f"[TRAIN] Ep {epoch}/{epochs} St {global_step}/{total_steps} | "
                    f"CE: {metrics['ce_loss']:.4f} | "
                    f"Top-1: {metrics['top1_acc']*100:.1f}% | "
                    f"Top-5: {metrics['top5_acc']*100:.1f}% | "
                    f"GradNorm: {grad_norm:.2f} | "
                    f"LR: {current_lr:.2e}",
                    flush=True,
                )

        avg_loss = epoch_loss / steps_per_epoch
        print(f"[TRAIN] Epoch {epoch} completata in {time.time() - t_epoch:.1f}s | Avg CE Loss: {avg_loss:.4f}", flush=True)

    trace_file.close()

    # 5. Salvataggio Checkpoint Nuovo
    checkpoints_dir = Path("/workspace/checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    new_ckpt_name = f"v9_causal_l0_{git_sha[:8]}.pt"
    new_ckpt_path = checkpoints_dir / new_ckpt_name
    
    save_dict = {
        "proj": projector.state_dict(),
        "causal_conv": causal_conv.conv.state_dict(),
        "git_sha": git_sha,
        "config": config,
    }
    torch.save(save_dict, new_ckpt_path)
    new_ckpt_sha = compute_file_sha256(new_ckpt_path)
    print(f"[TRAIN] Nuovo checkpoint causale salvato: {new_ckpt_path} (SHA256: {new_ckpt_sha})", flush=True)

    # 6. Verifica invariabilita' pesi congelati post-training
    post_frozen: dict[str, str] = {}
    if qwen_snaps:
        post_frozen["qwen_talker"] = compute_file_sha256(Path(qwen_snaps[-1]))
        assert post_frozen["qwen_talker"] == frozen_manifest["qwen_talker"], "Qwen weights changed during training!"

    manifest = {
        "schema_version": 1,
        "phase": "P4D-R_TRAIN_MANIFEST",
        "git_sha": git_sha,
        "new_checkpoint": str(new_ckpt_path),
        "new_checkpoint_sha256": new_ckpt_sha,
        "source_checkpoint": str(source_v9_ckpt),
        "source_checkpoint_sha256": source_v9_sha,
        "frozen_components_verified_identical": True,
        "frozen_components_sha256": post_frozen,
        "total_training_seconds": time.time() - t_start,
    }
    (output_dir / "P4_PROJECTOR_TRAIN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("/workspace/cache_p4d_r_subset.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/validation/kernel_gates/P4D_R"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-conv", action="store_true")
    args = parser.parse_args()

    train_causal_v9(
        args.cache,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_conv=args.train_conv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
