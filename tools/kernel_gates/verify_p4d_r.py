#!/usr/bin/env python3
"""P4D-R — Verifier Indipendente Post-Retraining Causal V9.

Verifica:
1. T1: Incremental equivalence (batch vs V9.step) — MAE <= 0.01, CosMin >= 0.999
2. T2: Causalita' — future perturbation max diff <= 1e-5 FP32
3. T3: Performance — V9.step p95 <= 10 ms
4. P4D Sweep L=0..4 con FuturePrefixQwen su 100 sequenze di validazione
5. Attestazione crittografica frozen weights
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.v9 import (
    DeepCausalTransformerProjectorV9,
    CausalConv1D,
    IncrementalV9Projector,
)
from duplex_kernel.models.talker import load_talker
from duplex_kernel.future_prefix import FuturePrefixQwen
from tools.kernel_gates.p4_checks import _stack_cache, _qwen_logits


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_candidate_v9(ckpt_path: Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    proj = DeepCausalTransformerProjectorV9()
    proj.load_state_dict(sd.get("proj", sd), strict=False)
    proj = proj.to(device=device, dtype=dtype).eval()

    causal_conv = CausalConv1D(hidden_size=1536, kernel_size=3)
    if "causal_conv" in sd:
        causal_conv.conv.load_state_dict(sd["causal_conv"], strict=False)
    else:
        # Fallback to default heads
        h_sd = torch.load("/workspace/_archive_ckpts/gemma4_duplex_sota/custom_duplex_heads.pt", map_location="cpu", weights_only=True)
        causal_conv.conv.load_state_dict(h_sd["causal_conv"], strict=False)
    causal_conv = causal_conv.to(device=device, dtype=dtype).eval()

    for p in proj.parameters():
        p.requires_grad = False
    for p in causal_conv.parameters():
        p.requires_grad = False

    return proj, causal_conv


def verify_t1_t2_t3(
    projector: DeepCausalTransformerProjectorV9,
    causal_conv: CausalConv1D,
    cache_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    print("[VERIFY] Esecuzione gate T1, T2, T3 per nuovo V9...", flush=True)
    raw, _, _, _ = _stack_cache(cache_path)
    seqs = raw[:100].to(device=device, dtype=dtype)
    num_seq = seqs.shape[0]

    # --- T1: Incremental Equivalence ---
    maes: list[float] = []
    cos_mins: list[float] = []
    latencies_ms: list[float] = []

    for i in range(num_seq):
        seq = seqs[i:i+1]
        h_tx = seq[:, 0, :]
        with torch.no_grad():
            batch_conv = causal_conv(seq)
            batch_pfx = projector(batch_conv, h_tx)

        # Stateful incremental
        inc = IncrementalV9Projector(projector, causal_conv)
        state = inc.new_state(batch_size=1, dtype=seq.dtype, max_seq_len=seq.shape[1])
        inc_frames: list[torch.Tensor] = []
        for t in range(seq.shape[1]):
            t_start = time.perf_counter()
            f = inc.step(seq[0, t, :], h_tx[0], state)
            latencies_ms.append((time.perf_counter() - t_start) * 1000.0)
            inc_frames.append(f)
        inc_pfx = torch.stack(inc_frames, dim=1)

        diff = (batch_pfx - inc_pfx).abs()
        mae = float(diff.mean().item())
        maes.append(mae)

        b_norm = F.normalize(batch_pfx.float(), p=2, dim=-1)
        i_norm = F.normalize(inc_pfx.float(), p=2, dim=-1)
        cos = (b_norm * i_norm).sum(dim=-1).min().item()
        cos_mins.append(cos)

    max_mae = max(maes)
    min_cos = min(cos_mins)
    t1_pass = bool(max_mae <= 0.01 and min_cos >= 0.999)
    print(f"  T1: Max MAE = {max_mae:.6f} (<= 0.01), Min Cos = {min_cos:.6f} (>= 0.999) -> {'PASS' if t1_pass else 'FAIL'}", flush=True)

    # --- T2: Causality Perturbation ---
    pert_diffs: list[float] = []
    for i in range(num_seq):
        seq = seqs[i:i+1].clone()
        t = int(torch.randint(1, seq.shape[1] - 1, (1,)).item())
        h_tx = seq[:, 0, :]
        with torch.no_grad():
            ref_pfx = projector(causal_conv(seq), h_tx)

        seq_pert = seq.clone()
        seq_pert[:, t+1:, :] += 50.0 * torch.randn_like(seq_pert[:, t+1:, :])
        h_tx_pert = seq_pert[:, 0, :]
        with torch.no_grad():
            pert_pfx = projector(causal_conv(seq_pert), h_tx_pert)

        diff_t = float((ref_pfx[:, t, :] - pert_pfx[:, t, :]).abs().max().item())
        pert_diffs.append(diff_t)

    max_pert_diff = max(pert_diffs)
    t2_pass = bool(max_pert_diff <= 1e-5)
    print(f"  T2: Max Perturbation Diff = {max_pert_diff:.2e} (<= 1e-5) -> {'PASS' if t2_pass else 'FAIL'}", flush=True)

    # --- T3: Performance ---
    lat_tensor = torch.tensor(latencies_ms)
    p50 = float(lat_tensor.median().item())
    p95 = float(lat_tensor.quantile(0.95).item())
    p99 = float(lat_tensor.quantile(0.99).item())
    t3_pass = bool(p95 <= 10.0)
    print(f"  T3: Latency p50={p50:.2f}ms, p95={p95:.2f}ms (<= 10.0ms), p99={p99:.2f}ms -> {'PASS' if t3_pass else 'FAIL'}", flush=True)

    return {
        "t1_max_mae": max_mae,
        "t1_min_cos": min_cos,
        "t1_pass": t1_pass,
        "t2_max_diff": max_pert_diff,
        "t2_pass": t2_pass,
        "t3_p50_ms": p50,
        "t3_p95_ms": p95,
        "t3_p99_ms": p99,
        "t3_pass": t3_pass,
        "all_pass": bool(t1_pass and t2_pass and t3_pass),
    }


def verify_p4d_sweep(
    projector: DeepCausalTransformerProjectorV9,
    causal_conv: CausalConv1D,
    talker: Any,
    cache_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    print("[VERIFY] Esecuzione P4D Causal Sweep (L=0..4) con nuovo V9...", flush=True)
    raw, _, codes, _ = _stack_cache(cache_path)
    seqs = raw[:100].to(device=device, dtype=dtype)
    code_seqs = codes[:100].to(device=device)
    num_seq = seqs.shape[0]

    # Calcolo prefissi per tutte le sequenze
    prefixes: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(num_seq):
            s = seqs[i:i+1]
            tx = s[:, 0, :]
            c = causal_conv(s)
            p = projector(c, tx)
            prefixes.append(p)
    all_pfx = torch.cat(prefixes, dim=0)

    def batch_qwen_logits(talker, pfx, codes, causal_prefix=False, prefix_lookahead=0, chunk_size=20):
        logits_list = []
        for i in range(0, pfx.shape[0], chunk_size):
            chunk_pfx = pfx[i:i+chunk_size]
            chunk_codes = codes[i:i+chunk_size]
            l = _qwen_logits(talker, chunk_pfx, chunk_codes, causal_prefix=causal_prefix, prefix_lookahead=prefix_lookahead)
            logits_list.append(l)
        return torch.cat(logits_list, dim=0)


    # 1. Full Sequence Reference (non causale)
    print("  Calcolo reference full-sequence...", flush=True)
    with torch.no_grad():
        full_logits = batch_qwen_logits(talker, all_pfx, code_seqs, causal_prefix=False)
        target_cb0 = (code_seqs[:, :, 0] if code_seqs.dim() == 3 else code_seqs).contiguous()
        full_ce = float(F.cross_entropy(full_logits.view(-1, full_logits.shape[-1]), target_cb0.view(-1)).item())
        full_top1 = float((full_logits.argmax(-1) == target_cb0).float().mean().item())
        full_top5 = float(full_logits.topk(5, -1).indices.eq(target_cb0.unsqueeze(-1)).any(-1).float().mean().item())

    print(f"  Reference: CE={full_ce:.4f}, Top-1={full_top1*100:.2f}%, Top-5={full_top5*100:.2f}%", flush=True)

    # 2. Sweep causale L=0..4
    sweep_results: dict[str, Any] = {}
    min_passing_l: int | None = None

    for L in [0, 1, 2, 3, 4]:
        print(f"  Valutazione L={L}...", flush=True)
        with torch.no_grad():
            causal_logits = batch_qwen_logits(talker, all_pfx, code_seqs, causal_prefix=True, prefix_lookahead=L)
            causal_ce = float(F.cross_entropy(causal_logits.view(-1, causal_logits.shape[-1]), target_cb0.view(-1)).item())
            causal_top1 = float((causal_logits.argmax(-1) == target_cb0).float().mean().item())
            causal_top5 = float(causal_logits.topk(5, -1).indices.eq(target_cb0.unsqueeze(-1)).any(-1).float().mean().item())

        ce_ratio = causal_ce / max(full_ce, 1e-4)
        top1_drop = full_top1 - causal_top1
        top5_drop = full_top5 - causal_top5

        passed = bool(ce_ratio <= 1.10 and top1_drop <= 0.05 and top5_drop <= 0.03)
        if passed and min_passing_l is None:
            min_passing_l = L

        sweep_results[f"L{L}"] = {
            "lookahead_frames": L,
            "playout_delay_ms": L * 80,
            "causal_ce": causal_ce,
            "ce_ratio": ce_ratio,
            "top1_acc": causal_top1,
            "top1_drop": top1_drop,
            "top5_acc": causal_top5,
            "top5_drop": top5_drop,
            "passed": passed,
        }
        print(f"    L={L}: CE ratio={ce_ratio:.4f} (gate <= 1.10), top1_drop={top1_drop*100:.2f}pp (<= 5pp), top5_drop={top5_drop*100:.2f}pp (<= 3pp) -> {'PASS' if passed else 'FAIL'}", flush=True)

    return {
        "full_reference": {
            "ce": full_ce,
            "top1_acc": full_top1,
            "top5_acc": full_top5,
        },
        "sweep": sweep_results,
        "min_passing_lookahead": min_passing_l,
        "l0_passed": bool(sweep_results["L0"]["passed"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="Path del nuovo checkpoint causale V9")
    parser.add_argument("--cache", type=Path, default=Path("/workspace/cache_ars_runtime_aligned_v1.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/validation/kernel_gates/P4D_R"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    dtype = torch.bfloat16

    print(f"[VERIFY] Avvio verifica completa per {args.ckpt}...", flush=True)
    projector, causal_conv = load_candidate_v9(args.ckpt, device=device, dtype=dtype)
    talker = load_talker("/workspace", device=device, load_code2wav=False)

    # 1. T1, T2, T3
    t_report = verify_t1_t2_t3(projector, causal_conv, args.cache, device=device, dtype=dtype)
    (args.output_dir / "P4_V9_T1_T2_T3.json").write_text(json.dumps(t_report, indent=2) + "\n")

    # 2. P4D Sweep L=0..4
    sweep_report = verify_p4d_sweep(projector, causal_conv, talker, args.cache, device=device, dtype=dtype)
    (args.output_dir / "P4D_CAUSAL_PROJECTOR_SWEEP.json").write_text(json.dumps(sweep_report, indent=2) + "\n")

    # 3. Verdetto Ufficiale
    if sweep_report["l0_passed"] and t_report["all_pass"]:
        verdict = "P4D CAUSAL PROJECTOR PASS — L=0"
    elif sweep_report["min_passing_lookahead"] is not None and t_report["all_pass"]:
        verdict = f"P4D CAUSAL PROJECTOR PASS — L={sweep_report['min_passing_lookahead']}"
    else:
        verdict = "P4D CAUSAL PROJECTOR FAIL — BLOCKERS REMAIN"

    final_report = {
        "schema_version": 1,
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": sha256_file(args.ckpt),
        "t1_t2_t3": t_report,
        "p4d_sweep": sweep_report,
        "verdict": verdict,
    }
    (args.output_dir / "P4D_CAUSAL_PROJECTOR_FINAL.json").write_text(json.dumps(final_report, indent=2) + "\n")

    print("=" * 60, flush=True)
    print(f"VERDETTO FINALE: {verdict}", flush=True)
    print("=" * 60, flush=True)

    return 0 if "PASS" in verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
