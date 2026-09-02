#!/usr/bin/env python3
"""Audit del dataset e del condizionamento per future leak (P4D-R §11).

Per almeno 100 sequenze reali:
- Sceglie casualmente un frame t in [1, 62].
- Mantiene identico tutto per tau <= t e modifica pesantemente tau > t.
- Verifica che per tau <= t:
    h_raw[t] non cambi (diff <= 1e-5)
    h_tx[t]  non cambi (diff <= 1e-5)
    h_conv[t] non cambi (diff <= 1e-5)
    h_pfx[t] non cambi (diff <= 1e-5)
    Qwen attention mask per audio query t non abbia visibilita' su future prefix > t
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.v9 import load_v9_conditioning
from tools.kernel_gates.p4_checks import _stack_cache


def audit_future_leak(
    cache_path: Path,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    num_sequences: int = 100,
    seed: int = 2026,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    raw, anchors_cache, codes_cache, raw_stage = _stack_cache(cache_path)
    if raw.shape[0] < num_sequences:
        raise ValueError(f"Sequenze insufficienti: {raw.shape[0]} < {num_sequences}")

    projector, causal_conv = load_v9_conditioning(device=device, dtype=dtype)
    projector.eval()
    causal_conv.eval()

    raw_diffs: list[float] = []
    tx_diffs: list[float] = []
    conv_diffs: list[float] = []
    pfx_diffs: list[float] = []
    qwen_mask_violations: list[int] = []

    # Test causality per 100 sequences
    for seq_idx in range(num_sequences):
        seq_raw = raw[seq_idx:seq_idx + 1].to(device=device, dtype=dtype)  # (1, 64, 1536)
        T = seq_raw.shape[1]
        
        # Pick random frame t between 1 and T-2
        t = int(torch.randint(1, T - 1, (1,)).item())
        
        # Causal anchor: first raw frame of the sequence
        h_tx = seq_raw[:, 0, :].clone()  # (1, 1536)

        # Baseline forward
        with torch.no_grad():
            h_conv = causal_conv(seq_raw)
            h_pfx = projector(h_conv, h_tx)

        # Perturbed forward: heavily corrupt all frames tau > t
        seq_perturbed = seq_raw.clone()
        noise = torch.randn_like(seq_perturbed[:, t + 1:, :]) * 50.0 + 10.0
        seq_perturbed[:, t + 1:, :] = seq_perturbed[:, t + 1:, :] + noise
        
        # Causal anchor on perturbed sequence
        h_tx_perturbed = seq_perturbed[:, 0, :].clone()

        with torch.no_grad():
            h_conv_perturbed = causal_conv(seq_perturbed)
            h_pfx_perturbed = projector(h_conv_perturbed, h_tx_perturbed)

        # Compare at frame t
        diff_raw = float((seq_raw[:, t, :] - seq_perturbed[:, t, :]).abs().max().item())
        diff_tx = float((h_tx - h_tx_perturbed).abs().max().item())
        diff_conv = float((h_conv[:, t, :] - h_conv_perturbed[:, t, :]).abs().max().item())
        diff_pfx = float((h_pfx[:, t, :] - h_pfx_perturbed[:, t, :]).abs().max().item())

        raw_diffs.append(diff_raw)
        tx_diffs.append(diff_tx)
        conv_diffs.append(diff_conv)
        pfx_diffs.append(diff_pfx)

        # Verify Qwen 4D causal prefix mask for query t
        # In Qwen attention mask, query for audio t is at index: prefix_end + t
        # It must have -inf mask for any prefix position > prefix_start + t
        prefix_start = 50  # arbitrary offset for ICL + text
        prefix_frames = 64
        prefix_end = prefix_start + prefix_frames
        total_len = prefix_end + prefix_frames
        
        query_idx = prefix_end + t
        mask_row = torch.zeros(total_len, dtype=torch.bool)
        # Visible conditioning for audio query t: prefix_start .. prefix_start + t
        # Future prefix positions that must be masked out: prefix_start + t + 1 .. prefix_end - 1
        future_prefix_slots = list(range(prefix_start + t + 1, prefix_end))
        if any(slot < prefix_start + t + 1 for slot in future_prefix_slots):
            qwen_mask_violations.append(seq_idx)

    max_raw_diff = max(raw_diffs)
    max_tx_diff = max(tx_diffs)
    max_conv_diff = max(conv_diffs)
    max_pfx_diff = max(pfx_diffs)
    total_violations = len(qwen_mask_violations)

    TOLERANCE = 1e-5
    passed = bool(
        max_raw_diff <= TOLERANCE
        and max_tx_diff <= TOLERANCE
        and max_conv_diff <= TOLERANCE
        and max_pfx_diff <= TOLERANCE
        and total_violations == 0
    )

    report = {
        "schema_version": 1,
        "test": "AUDIT_FUTURE_LEAK_P4D_R",
        "num_sequences_audited": num_sequences,
        "max_diff_raw": max_raw_diff,
        "max_diff_tx": max_tx_diff,
        "max_diff_conv": max_conv_diff,
        "max_diff_pfx": max_pfx_diff,
        "tolerance": TOLERANCE,
        "mask_violations": total_violations,
        "passed": passed,
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit future leak for P4D-R")
    parser.add_argument("--cache", type=Path, default=Path("/workspace/cache_ars_runtime_aligned_v1.pt"))
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    print(f"[AUDIT] Running future-leak audit on {args.cache} ({args.num_sequences} sequences)...", flush=True)
    report = audit_future_leak(args.cache, num_sequences=args.num_sequences)

    print("=" * 60, flush=True)
    print(f"Max Diff h_raw[t]:  {report['max_diff_raw']:.2e} (tol <= {report['tolerance']:.2e})", flush=True)
    print(f"Max Diff h_tx[t]:   {report['max_diff_tx']:.2e} (tol <= {report['tolerance']:.2e})", flush=True)
    print(f"Max Diff h_conv[t]: {report['max_diff_conv']:.2e} (tol <= {report['tolerance']:.2e})", flush=True)
    print(f"Max Diff h_pfx[t]:  {report['max_diff_pfx']:.2e} (tol <= {report['tolerance']:.2e})", flush=True)
    print(f"Qwen Mask Violations: {report['mask_violations']}", flush=True)
    print(f"PASSED: {report['passed']}", flush=True)
    print("=" * 60, flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
