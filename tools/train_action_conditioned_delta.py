#!/usr/bin/env python3
"""Train an action-conditioned Gemma LoRA delta from the current adapter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.duplex_qlora.action_delta import train_action_conditioned_delta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--initial-adapter-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--smoke", action="store_true", help="run four optimizer updates and verify adapter save/load")
    args = parser.parse_args()
    result = train_action_conditioned_delta(
        train_jsonl=args.train_jsonl,
        validation_jsonl=args.validation_jsonl,
        initial_adapter_dir=args.initial_adapter_dir,
        model_path=args.model_path,
        adapter_dir=args.adapter_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        smoke=args.smoke,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
