#!/usr/bin/env python3
"""Milestone 1 — Overfit Diagnostico Causal V9 su piccolo subset (P4D-R §16).

Verifica che il grafo differenziabile (CrossEntropy -> Qwen frozen -> h_pfx -> V9 trainable)
sia in grado di overfittare rapidamente un subset di 32-64 finestre causali a L=0.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.causal_v9_trainer import DifferentiableCausalQwenV9, compute_causal_v9_loss
from duplex_kernel.models.talker import load_talker
from duplex_kernel.models.v9 import load_v9_conditioning


def run_overfit_diagnostic(
    cache_path: Path,
    *,
    steps: int = 60,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    output_dir: Path | None = None,
) -> dict[str, object]:
    print(f"[OVERFIT] Caricamento cache {cache_path}...", flush=True)
    c = torch.load(cache_path, map_location="cpu", weights_only=False)
    
    # Prende tr_raw se disponibile, altrimenti va_raw
    raw_list = c.get("tr_raw") or c.get("va_raw", [])
    codes_list = c.get("tr_c") or c.get("va_c", [])
    
    if not raw_list:
        raise ValueError(f"Cache non contiene dati grezzi: {list(c.keys())}")
        
    num_seq = min(len(raw_list), 64)
    print(f"[OVERFIT] Utilizzo di {num_seq} sequenze per l'overfit diagnostico", flush=True)
    
    raw_tensor = torch.stack(raw_list[:num_seq]).to(device=device, dtype=dtype)  # (N, 64, 1536)
    codes_tensor = torch.stack(codes_list[:num_seq]).to(device=device, dtype=torch.long)  # (N, 64, 16)
    
    print("[OVERFIT] Caricamento V9 conditioning warm-start...", flush=True)
    projector, causal_conv = load_v9_conditioning(device=device, dtype=dtype)
    
    print("[OVERFIT] Caricamento Qwen Talker 1.7B...", flush=True)
    talker = load_talker("/workspace", device=device, load_code2wav=False)
    
    model = DifferentiableCausalQwenV9(projector, causal_conv, talker, train_conv=False)
    optimizer = AdamW(projector.parameters(), lr=lr, weight_decay=1e-2)

    history: list[dict[str, float]] = []
    initial_loss: float = 0.0
    final_loss: float = 0.0

    print(f"[OVERFIT] Avvio ciclo di {steps} step di ottimizzazione causale L=0...", flush=True)
    t0 = time.time()
    
    for step in range(1, steps + 1):
        # Mini-batch casuale o ciclico dal subset
        indices = torch.randint(0, num_seq, (batch_size,))
        b_raw = raw_tensor[indices]
        b_tx = b_raw[:, 0, :]  # Causal anchor (first_raw_frame)
        b_codes = codes_tensor[indices]

        optimizer.zero_grad()
        logits, h_pfx = model(b_raw, b_tx, b_codes, lookahead_frames=0)
        loss, metrics = compute_causal_v9_loss(logits, b_codes)

        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0).item())
        optimizer.step()

        metrics["step"] = step
        metrics["grad_norm"] = grad_norm
        history.append(metrics)

        if step == 1:
            initial_loss = metrics["ce_loss"]

        if step % 5 == 0 or step == 1 or step == steps:
            print(
                f"[OVERFIT] Step {step:03d}/{steps} | "
                f"CE Loss: {metrics['ce_loss']:.4f} | "
                f"Top-1: {metrics['top1_acc']*100:.1f}% | "
                f"Top-5: {metrics['top5_acc']*100:.1f}% | "
                f"Grad Norm: {grad_norm:.2f}",
                flush=True,
            )

    final_loss = history[-1]["ce_loss"]
    elapsed = time.time() - t0
    
    # Criterio di successo overfit: riduzione significativa della loss (almeno -20%)
    relative_drop = (initial_loss - final_loss) / max(initial_loss, 1e-4)
    passed = bool(relative_drop >= 0.20 or final_loss < 2.5)

    print("=" * 60, flush=True)
    print(f"[OVERFIT] Loss Iniziale: {initial_loss:.4f} -> Finale: {final_loss:.4f}", flush=True)
    print(f"[OVERFIT] Riduzione Relativa: {relative_drop * 100:.1f}% (gate >= 20.0%)", flush=True)
    print(f"[OVERFIT] Tempo totale: {elapsed:.1f}s ({elapsed/steps:.2f}s/step)", flush=True)
    print(f"[OVERFIT] ESITO MILESTONE 1: {'PASS' if passed else 'FAIL'}", flush=True)
    print("=" * 60, flush=True)

    report = {
        "schema_version": 1,
        "phase": "P4D-R_MILESTONE_1_OVERFIT",
        "num_sequences": num_seq,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "initial_ce_loss": initial_loss,
        "final_ce_loss": final_loss,
        "relative_drop": relative_drop,
        "final_top1_acc": history[-1]["top1_acc"],
        "final_top5_acc": history[-1]["top5_acc"],
        "elapsed_seconds": elapsed,
        "passed": passed,
        "history": history,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "P4D_R_OVERFIT_REPORT.json"
        out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[OVERFIT] Report salvato in {out_file}", flush=True)

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("/workspace/cache_p4d_r_subset.pt"))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/validation/kernel_gates/P4D_R"))
    args = parser.parse_args()

    report = run_overfit_diagnostic(
        args.cache,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
