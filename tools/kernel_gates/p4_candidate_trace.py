#!/usr/bin/env python3
"""Trace one incremental precision/backend candidate against native full BF16."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.thinker import load_thinker  # noqa: E402
from tools.kernel_gates.p4_numeric_bisect import (  # noqa: E402
    Capture,
    _compare,
    _fixture,
    _incremental,
    _native_full,
    _record_sort_key,
    _is_significant,
)
from tools.kernel_gates.p4_precision_ablations import _candidate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-bf16-reduced-reduction", action="store_true")
    parser.add_argument("--padding-length", type=int, default=None)
    args = parser.parse_args()
    if args.no_bf16_reduced_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    thinker = load_thinker("/workspace", device="cuda", dtype=torch.bfloat16)
    model = thinker.model.model
    try:
        ids, attention, features, feature_mask, audio_soft, audio_start = _fixture(model, args.frames, args.seed)
        positions = (ids[0] == int(model.config.audio_token_id)).nonzero(as_tuple=False).flatten().tolist()
        model.language_model._p4_padding_length = int(args.padding_length or ids.shape[1])
        full_capture = Capture(set(positions))
        _native_full(
            model,
            mode_name="full_bf16",
            input_ids=ids,
            attention_mask=attention,
            input_features=features,
            input_features_mask=feature_mask,
            capture=full_capture,
        )
        candidate_capture = Capture(set(positions))
        with _candidate(model.language_model, args.candidate) as applicable:
            if not applicable:
                print(json.dumps({"candidate": args.candidate, "applicable": False}, indent=2))
                return 0
            _incremental(
                model,
                mode_name=args.candidate,
                input_ids=ids,
                attention_mask=attention,
                audio_soft=audio_soft,
                audio_start=audio_start,
                capture=candidate_capture,
            )
        groups: list[dict[str, Any]] = []
        for key in sorted(set(full_capture.records) | set(candidate_capture.records), key=_record_sort_key):
            layer, component = key
            if layer < 0:
                continue
            values = []
            for position in sorted(set(full_capture.records.get(key, {})) | set(candidate_capture.records.get(key, {}))):
                a = full_capture.records.get(key, {}).get(position)
                b = candidate_capture.records.get(key, {}).get(position)
                comparison = _compare(a, b)
                if comparison.get("available") and not comparison.get("shape_mismatch"):
                    values.append(comparison)
            if not values:
                continue
            maes = [float(v["mae"]) for v in values if v.get("mae") is not None]
            maxima = [float(v["max_abs"]) for v in values if v.get("max_abs") is not None]
            cosines = [float(v["cosine"]) for v in values if v.get("cosine") is not None]
            mismatches = [float(v["mismatch_rate"]) for v in values if v.get("mismatch_rate") is not None]
            groups.append({
                "layer": layer,
                "component": component,
                "positions": len(values),
                "mae_mean": sum(maes) / len(maes) if maes else None,
                "mae_max": max(maes) if maes else None,
                "max_abs": max(maxima) if maxima else None,
                "cosine_min": min(cosines) if cosines else None,
                "mismatch_rate": sum(mismatches) / len(mismatches) if mismatches else None,
            })
        first = next((item for item in groups if _is_significant(item)), None)
        print(json.dumps({
            "candidate": args.candidate,
            "applicable": True,
            "first_significant": first,
            "groups": groups,
        }, indent=2))
        return 0
    finally:
        del thinker, model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
