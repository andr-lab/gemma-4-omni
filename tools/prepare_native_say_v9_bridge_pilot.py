#!/usr/bin/env python3
"""Prepare a v5-only native SAY/V9/Qwen bridge pilot manifest on CPU."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from duplex_kernel.native_say_v9_bridge import (
    DEFAULT_TRAIN_QUOTA,
    DEFAULT_VALIDATION_QUOTA,
    prepare_native_say_bridge_pilot,
    read_jsonl,
    write_native_say_bridge_pilot,
)


def _quotas(value: str, *, name: str) -> dict[str, int]:
    try:
        fields = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} expects speak,backchannel,replan counts") from exc
    if len(fields) != 3 or any(item < 0 for item in fields):
        raise argparse.ArgumentTypeError(f"{name} expects three non-negative counts")
    return {"SPEAK": fields[0], "BACKCHANNEL": fields[1], "REPLAN": fields[2]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("/workspace/data/audio_response_protocol_v2_v5_20261002"),
    )
    parser.add_argument(
        "--raw-manifest",
        type=Path,
        default=Path("/workspace/data/synthetic_it_t6_audio_grounded_v5_candidate_seed260928/manifest_audio_grounded_v5.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/data/native_say_v9_bridge_pilot_20261005"),
    )
    parser.add_argument("--seed", type=int, default=260928)
    parser.add_argument("--train-quota", type=lambda value: _quotas(value, name="train-quota"), default=None)
    parser.add_argument("--validation-quota", type=lambda value: _quotas(value, name="validation-quota"), default=None)
    args = parser.parse_args()

    prepared = {
        split: read_jsonl(args.prepared_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    raw = read_jsonl(args.raw_manifest)
    result = prepare_native_say_bridge_pilot(
        prepared,
        raw,
        train_action_quota=args.train_quota or DEFAULT_TRAIN_QUOTA,
        validation_action_quota=args.validation_quota or DEFAULT_VALIDATION_QUOTA,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_native_say_bridge_pilot(
        result,
        manifest_path=args.output_dir / "pilot_manifest.jsonl",
        report_path=args.output_dir / "preparation_report.json",
    )
    print(f"prepared={len(result['records'])} output={args.output_dir}")
    print(result["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
