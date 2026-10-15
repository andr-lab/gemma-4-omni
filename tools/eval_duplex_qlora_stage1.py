#!/usr/bin/env python3
"""Held-out Stage1 evaluation with one BF16 base+PEFT load.

Scores synthetic Italian conversation responses and a bounded, deterministic
sample of role-free ASR targets with the adapter enabled and disabled. The
report omits reference transcripts; optional short generations are retained
for interaction review.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.duplex_qlora.prepare import read_jsonl
from training.duplex_qlora.stage1 import (
    _format_chat,
    _move_batch_to_device,
    _training_records,
    build_training_inputs,
)


def _evenly_spaced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    indexes = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    return [rows[index] for index in indexes]


def _score_one(
    model: Any,
    processor: Any,
    record: Mapping[str, Any],
    *,
    device: torch.device,
    enabled: bool,
) -> dict[str, Any]:
    batch = build_training_inputs(processor, record)
    batch, _status = _move_batch_to_device(
        batch, device=device, compute_dtype=torch.bfloat16
    )
    with torch.inference_mode():
        if enabled:
            output = model(**batch, use_cache=False, return_dict=True)
        else:
            with model.disable_adapter():
                output = model(**batch, use_cache=False, return_dict=True)
    labels = batch["labels"][:, 1:]
    logits = output.logits[:, :-1].float()
    supervised = labels.ne(-100)
    count = int(supervised.sum().item())
    if count <= 0:
        raise ValueError("held-out row has no causal target labels")
    token_loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    token_loss = float(token_loss_sum.cpu())
    model_loss = float(output.loss.detach().float().cpu())
    manual_mean = token_loss / count
    if not math.isfinite(manual_mean) or not math.isfinite(model_loss):
        raise RuntimeError("held-out CE is not finite")
    return {
        "source": str(record.get("source", "unknown")),
        "task_kind": str(record.get("task_kind", "unknown")),
        "conversation_id": str(record.get("conversation_id", "")),
        "token_count": count,
        "token_loss_sum": token_loss,
        "token_mean_ce": manual_mean,
        "model_loss": model_loss,
        "model_manual_ce_abs_diff": abs(model_loss - manual_mean),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"examples": len(rows)}
    if not rows:
        result.update({"target_tokens": 0, "token_mean_ce": None, "example_mean_ce": None})
        return result
    token_count = sum(int(row["token_count"]) for row in rows)
    result.update({
        "target_tokens": token_count,
        "token_mean_ce": sum(float(row["token_loss_sum"]) for row in rows) / token_count,
        "example_mean_ce": statistics.mean(float(row["token_mean_ce"]) for row in rows),
        "max_model_manual_ce_abs_diff": max(float(row["model_manual_ce_abs_diff"]) for row in rows),
    })
    return result


def _generate_one(
    model: Any,
    processor: Any,
    record: Mapping[str, Any],
    *,
    device: torch.device,
    enabled: bool,
    max_new_tokens: int,
) -> str:
    prompt = _format_chat(processor, record, add_target=False)
    if "<|audio|>" in prompt or "<|soft_audio|>" in prompt:
        raise RuntimeError("runtime_asr_text generation prompt contains audio markers")
    tokenizer = getattr(processor, "tokenizer", processor)
    encoded = processor(text=[prompt], return_tensors="pt", padding=True)
    if encoded.get("input_features") is not None:
        raise RuntimeError("runtime_asr_text generation prompt unexpectedly produced audio features")
    encoded = {
        key: value.to(device) for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    kwargs: dict[str, Any] = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
    }
    if getattr(tokenizer, "pad_token_id", None) is not None:
        kwargs["pad_token_id"] = tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    with torch.inference_mode():
        if enabled:
            generated = model.generate(**kwargs)
        else:
            with model.disable_adapter():
                generated = model.generate(**kwargs)
    prompt_length = int(encoded["input_ids"].shape[1])
    return tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()


def evaluate(
    *,
    validation_jsonl: Path,
    model_path: str,
    adapter_dir: Path,
    output_path: Path,
    asr_count: int = 32,
    generation_count: int = 8,
    max_new_tokens: int = 72,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("held-out evaluation requires the 3090 CUDA runtime")
    torch.cuda.set_device(0)
    started = time.time()
    conversations = read_jsonl(validation_jsonl)
    records = _training_records(conversations, conditioning_mode="runtime_asr_text")
    gold = sorted(
        [
            row for row in records
            if row.get("task_kind") == "conversation_response"
            and row.get("synthetic") is True
            and str(row.get("language", "")).lower() in {"it", "ita", "italian"}
        ],
        key=lambda row: (row.get("conversation_id", ""), row.get("audio_start_ms") or 0, row.get("target_text", "")),
    )
    asr = sorted(
        [row for row in records if row.get("task_kind") == "asr" and not row.get("synthetic")],
        key=lambda row: (row.get("conversation_id", ""), row.get("audio_start_ms") or 0),
    )
    if not gold:
        raise RuntimeError("held-out validation has no synthetic Italian response targets")
    selected_asr = _evenly_spaced(asr, asr_count)

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    device = torch.device("cuda:0")

    scores: dict[str, dict[str, list[dict[str, Any]]]] = {
        "gold_italian": {"adapter_on": [], "adapter_off": []},
        "asr_aux": {"adapter_on": [], "adapter_off": []},
    }
    for category, rows in (("gold_italian", gold), ("asr_aux", selected_asr)):
        for row in rows:
            for enabled, key in ((True, "adapter_on"), (False, "adapter_off")):
                scores[category][key].append(
                    _score_one(model, processor, row, device=device, enabled=enabled)
                )

    generations = []
    for row in _evenly_spaced(gold, generation_count):
        generations.append({
            "conversation_id": str(row.get("conversation_id", "")),
            "adapter_off": _generate_one(
                model, processor, row, device=device, enabled=False,
                max_new_tokens=max_new_tokens,
            ),
            "adapter_on": _generate_one(
                model, processor, row, device=device, enabled=True,
                max_new_tokens=max_new_tokens,
            ),
        })

    report = {
        "schema_version": 1,
        "conditioning_mode": "runtime_asr_text",
        "validation_jsonl": str(validation_jsonl),
        "model_path": model_path,
        "adapter_dir": str(adapter_dir),
        "adapter_comparison": "same BF16 base load; PEFT disable_adapter context for base baseline",
        "selection": {
            "gold_italian_available": len(gold),
            "asr_aux_available": len(asr),
            "asr_aux_selected": len(selected_asr),
            "asr_selection": "deterministic evenly spaced by sorted conversation_id and utterance start",
        },
        "metrics": {
            category: {
                key: _aggregate(rows)
                for key, rows in mode_rows.items()
            }
            for category, mode_rows in scores.items()
        },
        "generations": generations,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del model, processor
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asr-count", type=int, default=32)
    parser.add_argument("--generation-count", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=72)
    args = parser.parse_args()
    report = evaluate(
        validation_jsonl=args.validation_jsonl,
        model_path=args.model_path,
        adapter_dir=args.adapter_dir,
        output_path=args.output,
        asr_count=args.asr_count,
        generation_count=args.generation_count,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps({
        "selection": report["selection"],
        "metrics": report["metrics"],
        "elapsed_seconds": report["elapsed_seconds"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
