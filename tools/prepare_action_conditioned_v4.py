#!/usr/bin/env python3
"""Prepare a non-overwriting v4 action-conditioned assistant-text dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.duplex_qlora.action_conditioned_data import prepare_action_conditioned_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_action_conditioned_dataset(args.prepared_dir, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "splits": result["splits"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
