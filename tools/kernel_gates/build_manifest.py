#!/usr/bin/env python3
"""Create a canonical, versioned input manifest for kernel gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="JSON manifest source")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--phase", default=None)
    args = ap.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("inputs"), list):
        raise SystemExit("manifest deve contenere un oggetto con inputs[]")
    inputs = data["inputs"]
    if len({item.get("id") for item in inputs}) != len(inputs):
        raise SystemExit("gli id del manifest devono essere univoci")
    if any(not isinstance(item.get("prompt"), str) or not item["prompt"].strip() for item in inputs):
        raise SystemExit("ogni input deve avere un prompt non vuoto")
    if args.phase:
        data["phase"] = args.phase
    data["inputs"] = sorted(inputs, key=lambda item: str(item["id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"manifest={args.output} inputs={len(inputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
