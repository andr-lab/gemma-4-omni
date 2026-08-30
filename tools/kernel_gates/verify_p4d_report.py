#!/usr/bin/env python3
"""Independent verifier for P4D frozen-Qwen experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

CE_RATIO_MAX = 1.10
TOP1_DROP_MAX = 0.05
TOP5_DROP_MAX = 0.03
MIN_SEQUENCES = 100
MIN_FRAMES = 5_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8)


def _recompute(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("event") != "p4_qwen_ablation_frame":
            continue
        experiment_id = str(event.get("experiment_id", ""))
        if not experiment_id:
            raise ValueError("evento P4D senza experiment_id")
        grouped.setdefault(experiment_id, []).append(event)
    results: dict[str, dict[str, Any]] = {}
    for experiment_id, rows in grouped.items():
        sequences = {int(row["sequence_id"]) for row in rows}
        frames = len(rows)
        full_ce = [float(row["full_ce"]) for row in rows]
        causal_ce = [float(row["causal_ce"]) for row in rows]
        full_top1 = [bool(row["full_top1"]) for row in rows]
        causal_top1 = [bool(row["causal_top1"]) for row in rows]
        full_top5 = [bool(row["full_top5"]) for row in rows]
        causal_top5 = [bool(row["causal_top5"]) for row in rows]
        kl = [float(row["kl_full_to_causal"]) for row in rows]
        deltas = [float(row["max_logit_delta"]) for row in rows]
        finite = all(
            _finite(value)
            for values in (full_ce, causal_ce, kl, deltas)
            for value in values
        )
        full_mean = sum(full_ce) / max(len(full_ce), 1)
        causal_mean = sum(causal_ce) / max(len(causal_ce), 1)
        full_acc1 = sum(full_top1) / max(len(full_top1), 1)
        causal_acc1 = sum(causal_top1) / max(len(causal_top1), 1)
        full_acc5 = sum(full_top5) / max(len(full_top5), 1)
        causal_acc5 = sum(causal_top5) / max(len(causal_top5), 1)
        ratio = causal_mean / max(full_mean, 1e-12)
        top1_drop = full_acc1 - causal_acc1
        top5_drop = full_acc5 - causal_acc5
        results[experiment_id] = {
            "sequences": len(sequences),
            "frames": frames,
            "full_ce": full_mean,
            "causal_ce": causal_mean,
            "full_top1": full_acc1,
            "causal_top1": causal_acc1,
            "full_top5": full_acc5,
            "causal_top5": causal_acc5,
            "kl_full_to_causal": sum(kl) / max(len(kl), 1),
            "max_logit_delta": max(deltas, default=0.0),
            "finite": finite,
            "mask_future_prefix": all(bool(row.get("anchor_causal")) for row in rows),
            "ce_ratio": ratio,
            "top1_drop": top1_drop,
            "top5_drop": top5_drop,
            "passed": bool(
                len(sequences) >= MIN_SEQUENCES
                and frames >= MIN_FRAMES
                and finite
                and all(bool(row.get("anchor_causal")) for row in rows)
                and max(deltas, default=0.0) > 1e-6
                and ratio <= CE_RATIO_MAX
                and top1_drop <= TOP1_DROP_MAX
                and top5_drop <= TOP5_DROP_MAX
            ),
        }
    return results


def verify(report_path: Path) -> list[str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if report.get("phase") != "P4D":
        errors.append("phase non è P4D")
    qwen = report.get("qwen", {})
    if qwen.get("frozen") is not True or qwen.get("weights_modified") is not False:
        errors.append("Qwen non attestato frozen/immutato")
    if qwen.get("trainable_parameter_names") != []:
        errors.append("Qwen contiene parametri trainable")
    hashes = [item.get("sha256") for item in qwen.get("checkpoint_files", [])]
    if not hashes or any(not isinstance(value, str) or len(value) != 64 for value in hashes):
        errors.append("hash checkpoint Qwen mancante o invalido")
    raw_path = report_path.parent / str(report.get("raw_trace", ""))
    if not raw_path.is_file():
        errors.append("raw trace P4D mancante")
        return errors
    if report.get("raw_trace_sha256") != _sha256(raw_path):
        errors.append("sha256 raw trace non corrispondente")
    events = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    recomputed = _recompute(events)
    variants = report.get("variants", {})
    if set(recomputed) != set(variants):
        errors.append("insieme varianti report/raw non corrispondente")
    for experiment_id, actual in recomputed.items():
        expected = variants.get(experiment_id, {})
        for key in ("sequences", "frames", "full_ce", "causal_ce", "full_top1", "causal_top1", "full_top5", "causal_top5", "kl_full_to_causal", "max_logit_delta", "ce_ratio", "top1_drop", "top5_drop", "finite", "passed"):
            if key not in expected:
                errors.append(f"{experiment_id}: campo {key} mancante")
            elif isinstance(actual[key], float):
                if not _close(actual[key], float(expected[key])):
                    errors.append(f"{experiment_id}: {key} non ricomputabile")
            elif actual[key] != expected[key]:
                errors.append(f"{experiment_id}: {key} non ricomputabile")
    if report.get("passed") is not any(value["passed"] for value in recomputed.values()):
        errors.append("passed globale non coerente con le varianti")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    errors = verify(args.report)
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
