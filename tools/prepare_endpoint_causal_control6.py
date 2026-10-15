#!/usr/bin/env python3
"""Prepare a separate source-group-preserving endpoint-causal Control6 set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.duplex_qlora.endpoint_causal_data import prepare_endpoint_causal_control6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-cap-per-action", type=int, default=256)
    parser.add_argument("--validation-cap-per-action", type=int, default=64)
    parser.add_argument("--test-cap-per-action", type=int, default=64)
    args = parser.parse_args()
    report = prepare_endpoint_causal_control6(
        args.input_dir,
        args.output_dir,
        train_cap_per_action=args.train_cap_per_action,
        validation_cap_per_action=args.validation_cap_per_action,
        test_cap_per_action=args.test_cap_per_action,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "splits": report["splits"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
