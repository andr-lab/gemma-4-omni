#!/usr/bin/env python3
"""Compare canonical vs empty internal user-event prompts on saved heldout contexts."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.action_conditioning import action_conditioned_messages, action_conditioning_instruction
from training.duplex_qlora.action_delta import _GENERATION_BUDGET_BY_ACTION, _read_records
from training.duplex_qlora.stage1 import _load_qlora, _response_language_instruction


def _select_saved_contexts(report_path: Path, validation_jsonl: Path) -> list[dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = _read_records(validation_jsonl, split="validation")
    by_target = {
        (str(row.get("conversation_id")), str(row.get("target_utterance_id"))): row
        for row in records
    }
    selected = []
    for pair in report["training"]["generation_comparison"]["paired_examples"]:
        row = pair["delta"]
        key = (str(row["conversation_id"]), str(row["target_utterance_id"]))
        record = by_target.get(key)
        if record is None or str(record.get("action")) != str(row.get("action")):
            raise RuntimeError(f"contesto heldout non trovato nel dataset v3: {key}")
        selected.append(record)
    if len(selected) != 6 or len({(row["conversation_id"], row["target_utterance_id"]) for row in selected}) != 6:
        raise RuntimeError(f"ablation richiede gli stessi sei contesti unici, trovati {len(selected)}")
    return selected


def _eos_ids(model: Any, tokenizer: Any) -> set[int]:
    values = {getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "eot_id", None)}
    model_values = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(model_values, (int,)):
        values.add(model_values)
    elif isinstance(model_values, (list, tuple, set)):
        values.update(model_values)
    result = {int(value) for value in values if isinstance(value, int)}
    if not result:
        raise RuntimeError("nessun EOS/EOT configurato")
    return result


def _generate_one(
    model: Any,
    processor: Any,
    record: Mapping[str, Any],
    *,
    device: torch.device,
    adapter_stage: str,
) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", processor)
    action = str(record["action"])
    messages = action_conditioned_messages(
        _response_language_instruction(record.get("language")),
        action,
        record.get("history_messages") or [],
    )
    # This is the approved inference ablation: preserve the action system
    # instruction and user role, but remove only the event contents.
    messages[-1]["content"] = ""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[prompt], return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    token_budget = _GENERATION_BUDGET_BY_ACTION[action]
    eos_ids = _eos_ids(model, tokenizer)
    generation_eos: int | list[int] = sorted(eos_ids)
    if len(eos_ids) == 1:
        generation_eos = next(iter(eos_ids))
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = next(iter(eos_ids))
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=token_budget,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=int(pad_id),
            eos_token_id=generation_eos,
        )
    token_ids = [int(value) for value in output[0, input_ids.shape[1]:].detach().cpu().tolist()]
    terminal_index = next((index for index, value in enumerate(token_ids) if value in eos_ids), None)
    content_ids = token_ids[:terminal_index] if terminal_index is not None else token_ids
    text = tokenizer.decode(content_ids, skip_special_tokens=True).strip()
    return {
        "generation_pair_id": "|".join((str(record["source_group_id"]), str(record["target_utterance_id"]))),
        "adapter_stage": adapter_stage,
        "action": action,
        "gold_action_label": action,
        "conversation_id": record.get("conversation_id"),
        "target_utterance_id": record.get("target_utterance_id"),
        "prompt_roles": [message["role"] for message in messages],
        "system_action_instruction": action_conditioning_instruction(action),
        "internal_control_event_content": "",
        "input_token_ids": [int(value) for value in input_ids[0].detach().cpu().tolist()],
        "generated_token_ids": token_ids,
        "generated_text": text,
        "generated_word_count": len(text.split()),
        "token_budget": token_budget,
        "eos_terminated": terminal_index is not None,
        "eos_token_id": token_ids[terminal_index] if terminal_index is not None else None,
        "reached_budget_without_eos": terminal_index is None and len(token_ids) >= token_budget,
        "completed_user_context": next(
            (str(item.get("content", "")) for item in reversed(record.get("history_messages", [])) if item.get("role") == "user"),
            "",
        ),
        "reference_text": record.get("target_text"),
    }


def evaluate_event_ablation(
    *,
    training_report: Path,
    validation_jsonl: Path,
    model_path: str,
    initial_adapter_dir: Path,
    delta_adapter_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("event ablation richiede CUDA")
    contexts = _select_saved_contexts(training_report, validation_jsonl)
    event_pairs = json.loads(training_report.read_text(encoding="utf-8"))["training"]["generation_comparison"]["paired_examples"]
    event_by_pair = {
        pair["generation_pair_id"]: {stage: pair[stage]["generated_text"] for stage in ("baseline", "delta")}
        for pair in event_pairs
    }
    generated: dict[str, list[dict[str, Any]]] = {}
    device = torch.device("cuda:0")
    started = time.time()
    for stage, adapter_dir in (
        ("stage1_adapter_400", initial_adapter_dir),
        ("action_conditioned_delta_200", delta_adapter_dir),
    ):
        model, processor, _trainable, _modules, _quantization = _load_qlora(
            model_path,
            dtype=torch.bfloat16,
            resume_adapter_dir=adapter_dir,
        )
        model.eval()
        generated[stage] = [
            _generate_one(model, processor, record, device=device, adapter_stage=stage)
            for record in contexts
        ]
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    delta_key = "action_conditioned_delta_200"
    baseline_key = "stage1_adapter_400"
    ablation_by_pair = {
        stage: {row["generation_pair_id"]: row for row in rows}
        for stage, rows in generated.items()
    }
    pairs = []
    for record in contexts:
        key = "|".join((str(record["source_group_id"]), str(record["target_utterance_id"])))
        pairs.append({
            "generation_pair_id": key,
            "event_prompt_generation": event_by_pair[key],
            "empty_user_event_ablation": {
                stage: ablation_by_pair[stage][key]
                for stage in (baseline_key, delta_key)
            },
        })
    report = {
        "schema_version": 1,
        "purpose": "diagnostic prompt ablation: same six contexts and gold action, keep system action instruction, make final internal user-role control-event content empty",
        "training_run_id": json.loads(training_report.read_text(encoding="utf-8"))["run_id"],
        "training_report": str(training_report),
        "validation_jsonl": str(validation_jsonl),
        "model_path": model_path,
        "canonical_event_marker_removed_only_for_this_probe": True,
        "training_run_or_adapter_changed": False,
        "paired_examples": pairs,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--initial-adapter-dir", type=Path, required=True)
    parser.add_argument("--delta-adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_event_ablation(
        training_report=args.training_report,
        validation_jsonl=args.validation_jsonl,
        model_path=args.model_path,
        initial_adapter_dir=args.initial_adapter_dir,
        delta_adapter_dir=args.delta_adapter_dir,
        output_path=args.output,
    )
    print(json.dumps({"training_run_id": result["training_run_id"], "examples": len(result["paired_examples"]), "elapsed_seconds": result["elapsed_seconds"], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
