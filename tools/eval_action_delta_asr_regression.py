#!/usr/bin/env python3
"""Compare Stage1-400 and an action-conditioned delta on the same 32 held-out ASR rows."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_duplex_qlora_stage1 import _aggregate, _evenly_spaced, _score_one
from training.duplex_qlora.prepare import read_jsonl
from training.duplex_qlora.stage1 import _training_records


def _load_adapter_model(model_path: str, adapter_dir: Path) -> tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    return model, processor


def evaluate_asr_regression(
    *,
    validation_jsonl: Path,
    model_path: str,
    initial_adapter_dir: Path,
    delta_adapter_dir: Path,
    output_path: Path,
    sample_count: int = 32,
) -> dict[str, Any]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("ASR regression probe richiede la GPU CUDA")
    conversations = read_jsonl(validation_jsonl)
    records = _training_records(conversations, conditioning_mode="runtime_asr_text")
    asr = sorted(
        [row for row in records if row.get("task_kind") == "asr" and not row.get("synthetic")],
        key=lambda row: (row.get("conversation_id", ""), row.get("audio_start_ms") or 0),
    )
    selected = _evenly_spaced(asr, sample_count)
    if len(selected) != sample_count:
        raise RuntimeError(f"heldout ASR insufficienti: richiesti {sample_count}, disponibili {len(asr)}")
    selection = [
        {
            "conversation_id": str(row.get("conversation_id", "")),
            "audio_start_ms": row.get("audio_start_ms"),
        }
        for row in selected
    ]
    selection_sha = hashlib.sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metrics: dict[str, Any] = {}
    device = torch.device("cuda:0")
    for label, adapter_dir in (
        ("stage1_adapter_400", initial_adapter_dir),
        ("action_conditioned_delta", delta_adapter_dir),
    ):
        model, processor = _load_adapter_model(model_path, adapter_dir)
        rows = [
            _score_one(model, processor, record, device=device, enabled=True)
            for record in selected
        ]
        metrics[label] = _aggregate(rows)
        del model, processor, rows
        gc.collect()
        torch.cuda.empty_cache()
    initial_ce = float(metrics["stage1_adapter_400"]["token_mean_ce"])
    delta_ce = float(metrics["action_conditioned_delta"]["token_mean_ce"])
    report = {
        "schema_version": 1,
        "purpose": "small ASR regression check for action-conditioned text LoRA delta",
        "validation_jsonl": str(validation_jsonl),
        "model_path": model_path,
        "adapter_dirs": {
            "stage1_adapter_400": str(initial_adapter_dir),
            "action_conditioned_delta": str(delta_adapter_dir),
        },
        "selection": {
            "examples": len(selected),
            "policy": "same deterministic selector as eval_duplex_qlora_stage1: sorted non-synthetic ASR by conversation_id/audio_start_ms then evenly spaced",
            "selection_sha256": selection_sha,
            "rows": selection,
        },
        "metrics": metrics,
        "delta_minus_stage1_token_mean_ce": delta_ce - initial_ce,
        "delta_relative_token_mean_ce_change": (delta_ce - initial_ce) / initial_ce if initial_ce else None,
        "transcripts_in_report": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--initial-adapter-dir", type=Path, required=True)
    parser.add_argument("--delta-adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=32)
    args = parser.parse_args()
    result = evaluate_asr_regression(
        validation_jsonl=args.validation_jsonl,
        model_path=args.model_path,
        initial_adapter_dir=args.initial_adapter_dir,
        delta_adapter_dir=args.delta_adapter_dir,
        output_path=args.output,
        sample_count=args.sample_count,
    )
    print(json.dumps({"selection": result["selection"], "metrics": result["metrics"], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
