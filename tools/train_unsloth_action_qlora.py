"""Train an action-conditioned Gemma 4 text adapter with masked prompt loss."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from build_native_action_dialogue_dataset import (
    ACTION_TO_TOKEN_ID,
    TOKEN_ID_TO_STRING,
    is_conversational_speak,
    word_count,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "action_dialogue_v1"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "gemma4_action_qlora_v1"
DEFAULT_VALIDATION_REPORT = ROOT / "reports" / "gemma4_action_qlora_v1_validation.json"
MODEL_NAME = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH = 512
ACTIVE_ACTIONS = ("SPEAK", "BACKCHANNEL", "REPLAN")
CONTROL_TOKENS = [TOKEN_ID_TO_STRING[ACTION_TO_TOKEN_ID[action]] for action in ACTION_TO_TOKEN_ID]
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def normalize_for_gemma4(text: str) -> str:
    """Translate the requested row format to Gemma 4's native turn tokens."""
    text = text.replace("<start_of_turn>user\n", "<|turn>user\n")
    text = text.replace("<start_of_turn>assistant\n", "<|turn>model\n")
    return text.replace("<end_of_turn>", "<turn|>")


def get_text_tokenizer(processor_or_tokenizer: Any) -> Any:
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    if not hasattr(tokenizer, "add_special_tokens") or not hasattr(tokenizer, "encode"):
        raise TypeError("Gemma 4 loader result does not expose a compatible text tokenizer")
    return tokenizer


def encode_record(
    row: Mapping[str, Any], tokenizer: Any, *, max_seq_length: int = MAX_SEQ_LENGTH
) -> dict[str, Any]:
    """Tokenize prompt+target and mask the complete prompt through model prefix."""
    prompt = row.get("prompt")
    target = row.get("target")
    if not isinstance(prompt, str) or not isinstance(target, str):
        raise ValueError("each dataset row needs string prompt and target fields")
    action_token = row.get("action_token")
    token_id = row.get("token_id")
    if not isinstance(action_token, str) or not isinstance(token_id, int):
        raise ValueError("each dataset row needs action_token and integer token_id fields")
    action_ids = tokenizer.encode(action_token, add_special_tokens=False)
    if action_ids != [token_id]:
        raise ValueError(
            f"control token {action_token!r} must encode as its single native id {token_id}; got {action_ids}"
        )

    prompt_ids = list(tokenizer.encode(normalize_for_gemma4(prompt), add_special_tokens=False))
    target_ids = list(tokenizer.encode(normalize_for_gemma4(target), add_special_tokens=False))
    joint_ids = list(tokenizer.encode(normalize_for_gemma4(prompt + target), add_special_tokens=False))
    if joint_ids != prompt_ids + target_ids:
        raise ValueError("prompt and target tokenization did not preserve the masked boundary")
    if not target_ids or target_ids[0] != token_id:
        raise ValueError("target must begin with the row's action control token")
    end_id = tokenizer.convert_tokens_to_ids("<turn|>")
    if target_ids[-1] != end_id:
        raise ValueError("target must end with Gemma 4's native turn-end token")

    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and (not prompt_ids or prompt_ids[0] != bos_id):
        prompt_ids.insert(0, bos_id)
    if len(target_ids) >= max_seq_length:
        raise ValueError("target alone exceeds max_seq_length")
    prompt_budget = max_seq_length - len(target_ids)
    if len(prompt_ids) > prompt_budget:
        if bos_id is not None and prompt_budget > 0:
            prompt_ids = [bos_id] + (prompt_ids[-(prompt_budget - 1) :] if prompt_budget > 1 else [])
        else:
            prompt_ids = prompt_ids[-prompt_budget:]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt_length": len(prompt_ids),
        "action": row.get("action"),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def collate_features(features: Sequence[Mapping[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch

    max_length = max(len(item["input_ids"]) for item in features)
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    labels: list[list[int]] = []
    for item in features:
        padding = max_length - len(item["input_ids"])
        input_ids.append(list(item["input_ids"]) + [pad_token_id] * padding)
        attention_masks.append(list(item["attention_mask"]) + [0] * padding)
        labels.append(list(item["labels"]) + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_dataset(allow_experimental_derived_data: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary_path = DATA_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dataset_status") == "EXPERIMENTAL_DERIVED_LABELS_NOT_HUMAN_ADJUDICATED" and not allow_experimental_derived_data:
        raise ValueError("dataset contains heuristic unadjudicated labels; pass --allow-experimental-derived-data to run an explicitly experimental fine-tune")
    train_all = read_jsonl(DATA_DIR / "train.jsonl")
    validation_all = read_jsonl(DATA_DIR / "validation.jsonl")
    train = [row for row in train_all if row.get("action") in ACTIVE_ACTIONS]
    validation = [row for row in validation_all if row.get("action") in ACTIVE_ACTIONS]
    if not train or not validation:
        raise ValueError("active-action train and validation partitions must both be non-empty")
    train_counts = Counter(row["action"] for row in train)
    total_counts = Counter(row["action"] for row in train + validation)
    return train, validation, {
        "dataset_status": summary.get("dataset_status", "UNMARKED"),
        "total_examples_by_action": dict(sorted(total_counts.items())),
        "train_examples_by_action": dict(sorted(train_counts.items())),
        "validation_examples_by_action": dict(sorted(Counter(row["action"] for row in validation).items())),
    }


def _evaluate_loss(model: Any, features: Sequence[Mapping[str, Any]], pad_id: int, torch: Any) -> float:
    if not features:
        return math.nan
    model.eval()
    losses: list[float] = []
    device = torch.device("cuda")
    with torch.inference_mode():
        for start in range(0, len(features), 4):
            batch = collate_features(features[start : start + 4], pad_id)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**batch, use_cache=False)
            losses.append(float(output.loss.detach().float().item()))
    return sum(losses) / len(losses)


def _select_eval_rows(rows: Sequence[Mapping[str, Any]], action: str, limit: int = 100) -> list[Mapping[str, Any]]:
    eligible = [row for row in rows if row.get("action") == action]
    eligible.sort(key=lambda row: str(row.get("example_id", "")))
    if len(eligible) <= limit:
        return eligible
    stride = len(eligible) / limit
    return [eligible[min(len(eligible) - 1, int(index * stride))] for index in range(limit)]


def _generate_action(
    model: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_new_tokens: int,
    end_token_id: int,
    torch: Any,
) -> dict[str, Any]:
    prompt = normalize_for_gemma4(str(row["prompt"])) + str(row["action_token"])
    input_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and (not input_ids or input_ids[0] != bos_id):
        input_ids.insert(0, bos_id)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_tensor)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model.generate(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=end_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    generated_ids = output[0, len(input_ids) :].tolist()
    end_position = next((index for index, token in enumerate(generated_ids) if token == end_token_id), None)
    body_ids = generated_ids if end_position is None else generated_ids[:end_position]
    generated_text = tokenizer.decode(body_ids, skip_special_tokens=False).strip()
    words = word_count(generated_text)
    return {
        "example_id": row.get("example_id"),
        "action": row["action"],
        "generated_text": generated_text,
        "generated_token_count_before_end": end_position if end_position is not None else len(generated_ids),
        "word_count": words,
        "emitted_native_end_token": end_position is not None,
        "native_end_token": "<turn|>",
        "ends_within_limit": end_position is not None and end_position + 1 <= max_new_tokens,
        "short_1_to_3_words": 1 <= words <= 3,
        "short_1_to_4_words": 1 <= words <= 4,
        "complete_speak_answer": is_conversational_speak(generated_text),
    }


def _validate_generation(
    model: Any,
    tokenizer: Any,
    validation_rows: Sequence[Mapping[str, Any]],
    torch: Any,
    *,
    sample_limit: int = 100,
) -> dict[str, Any]:
    end_token_id = tokenizer.convert_tokens_to_ids("<turn|>")
    output: dict[str, Any] = {
        "native_turn_end": "<turn|>",
        "requested_row_alias": "<end_of_turn>",
        "backchannel_max_new_tokens": 10,
        "backchannel_samples": [],
        "speak_max_new_tokens": 160,
        "speak_samples": [],
    }
    for action, limit, key in (("BACKCHANNEL", 10, "backchannel_samples"), ("SPEAK", 160, "speak_samples")):
        selected = _select_eval_rows(validation_rows, action, limit=sample_limit)
        if not selected:
            raise ValueError(f"validation has no examples for {action}")
        output[key] = [
            _generate_action(
                model,
                tokenizer,
                row,
                max_new_tokens=limit,
                end_token_id=end_token_id,
                torch=torch,
            )
            for row in selected
        ]
    backchannels = output["backchannel_samples"]
    speaks = output["speak_samples"]
    output["backchannel_metrics"] = {
        "count": len(backchannels),
        "eos_within_10_tokens_rate": sum(item["ends_within_limit"] for item in backchannels) / len(backchannels),
        "one_to_three_words_rate": sum(item["short_1_to_3_words"] for item in backchannels) / len(backchannels),
        "one_to_four_words_rate": sum(item["short_1_to_4_words"] for item in backchannels) / len(backchannels),
        "strict_success_rate": sum(item["ends_within_limit"] and item["short_1_to_3_words"] for item in backchannels) / len(backchannels),
    }
    output["speak_metrics"] = {
        "count": len(speaks),
        "eos_rate": sum(item["emitted_native_end_token"] for item in speaks) / len(speaks),
        "complete_8_to_80_word_answer_rate": sum(item["complete_speak_answer"] for item in speaks) / len(speaks),
        "strict_success_rate": sum(item["emitted_native_end_token"] and item["complete_speak_answer"] for item in speaks) / len(speaks),
    }
    return output


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from unsloth import FastLanguageModel
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested RTX 3090 run")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support BF16")
    train_rows, validation_rows, dataset_report = _load_dataset(args.allow_experimental_derived_data)
    total_counts = dataset_report["total_examples_by_action"]
    if not args.smoke and any(total_counts.get(action, 0) < 1000 for action in ACTIVE_ACTIONS):
        raise RuntimeError(f"full training requires at least 1000 total examples for each active action: {total_counts}")
    steps = 2 if args.smoke else args.max_steps
    batch_size = 1 if args.smoke else args.batch_size
    grad_accum = 1 if args.smoke else args.gradient_accumulation_steps
    if not args.smoke and not 400 <= steps <= 600:
        raise ValueError("--max-steps must be between 400 and 600 outside --smoke")

    model, processor_or_tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    tokenizer = get_text_tokenizer(processor_or_tokenizer)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": CONTROL_TOKENS},
        replace_extra_special_tokens=False,
    )
    for action in ACTION_TO_TOKEN_ID:
        token = TOKEN_ID_TO_STRING[ACTION_TO_TOKEN_ID[action]]
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [ACTION_TO_TOKEN_ID[action]]:
            raise RuntimeError(f"Gemma action token {token} did not retain its native id")
    embedding_rows = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_rows:
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for config in (model.config, getattr(model.config, "text_config", None)):
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=args.seed,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    train_features = [encode_record(row, tokenizer) for row in train_rows]
    validation_features = [encode_record(row, tokenizer) for row in validation_rows]
    pad_id = int(tokenizer.pad_token_id)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    loader = DataLoader(
        train_features,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda features: collate_features(features, pad_id),
    )
    iterator = iter(loader)
    train_probe = [train_features[index] for index in range(min(batch_size, len(train_features)))]
    initial_train_loss = _evaluate_loss(model, train_probe, pad_id, torch)
    eval_features: list[dict[str, Any]] = []
    for action in ACTIVE_ACTIONS:
        selected_rows = _select_eval_rows(validation_rows, action, limit=16)
        eval_features.extend(encode_record(row, tokenizer) for row in selected_rows)
    initial_validation_loss = _evaluate_loss(model, eval_features, pad_id, torch)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=steps,
    )
    history: list[dict[str, float]] = []
    initial_step_loss: float | None = None
    final_step_loss: float = math.nan
    started = time.time()
    device = torch.device("cuda")
    model.train()
    for optimizer_step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _micro_step in range(grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**batch, use_cache=False)
                loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at optimizer step {optimizer_step + 1}")
            if initial_step_loss is None:
                initial_step_loss = float(loss.detach().float().item())
            accumulated_loss += float(loss.detach().float().item()) / grad_accum
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()
        final_step_loss = accumulated_loss
        history.append({"step": optimizer_step + 1, "loss": accumulated_loss, "learning_rate": scheduler.get_last_lr()[0]})
        if (optimizer_step + 1) % 20 == 0 or optimizer_step + 1 == steps:
            print(json.dumps(history[-1], sort_keys=True), flush=True)

    final_train_loss = _evaluate_loss(model, train_probe, pad_id, torch)
    final_validation_loss = _evaluate_loss(model, eval_features, pad_id, torch)
    checkpoint = Path(args.output_dir)
    if args.smoke:
        checkpoint = ROOT / ".tmp" / "gemma4_action_qlora_v1_smoke"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    FastLanguageModel.for_inference(model)
    generation = _validate_generation(
        model,
        tokenizer,
        validation_rows,
        torch,
        sample_limit=5 if args.smoke else 100,
    )
    report = {
        "schema_version": 1,
        "status": "SMOKE_PASS" if args.smoke else "TRAINED_EXPERIMENTAL",
        "model": MODEL_NAME,
        "container_framework_versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "unsloth": __import__("importlib.metadata", fromlist=["version"]).version("unsloth"),
        },
        "dataset": dataset_report,
        "training": {
            "optimizer_steps_completed": steps,
            "per_device_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "learning_rate": args.learning_rate,
            "scheduler": "cosine",
            "warmup_steps": args.warmup_steps,
            "precision": "bfloat16 autocast",
            "lora_rank": 16,
            "lora_alpha": 32,
            "target_modules": LORA_TARGET_MODULES,
            "initial_train_loss": initial_train_loss,
            "first_optimizer_step_loss": initial_step_loss,
            "last_optimizer_step_loss": final_step_loss,
            "final_train_loss_same_probe": final_train_loss,
            "initial_validation_loss_sample": initial_validation_loss,
            "final_validation_loss_sample": final_validation_loss,
            "elapsed_seconds": time.time() - started,
            "history": history,
        },
        "checkpoint": str(checkpoint),
        "generation_validation": generation,
        "limitations": [
            "DuplexConv source labels are unadjudicated hypotheses; derived actions are experimental supervision.",
            "The action-generation check measures format, length, and native turn-end emission, not semantic correctness.",
            "The prompt JSONL uses requested aliases; training maps them to Gemma 4 native turn markers.",
        ],
    }
    if not args.smoke:
        _write_json(DEFAULT_VALIDATION_REPORT, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-experimental-derived-data", action="store_true")
    return parser.parse_args()


def main() -> int:
    report = train(parse_args())
    summary = {
        "status": report["status"],
        "checkpoint": report["checkpoint"],
        "optimizer_steps_completed": report["training"]["optimizer_steps_completed"],
        "initial_train_loss": report["training"]["initial_train_loss"],
        "final_train_loss_same_probe": report["training"]["final_train_loss_same_probe"],
        "backchannel_metrics": report["generation_validation"]["backchannel_metrics"],
        "speak_metrics": report["generation_validation"]["speak_metrics"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
