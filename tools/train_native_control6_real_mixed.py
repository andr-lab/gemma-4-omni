#!/usr/bin/env python3
"""Prepare real endpoint targets and train a 50/50 native Control6 candidate.

The model is only the existing six-row Gemma output-head LoRA. Real audio
features are extracted separately by train_native_control6_lora.py; this file
never loads an audio model and never changes response-generation weights.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_native_control6_lora import (
    TARGET_SCHEMA,
    TARGET_SCHEMA_VERSION,
    _read_jsonl,
    _sequence_windows,
    _validate_real_endpoint_rows,
    _write_jsonl,
    load_sequences,
)
from training.duplex_qlora.native_control6 import (
    ACTIONS,
    ACTION_TO_TOKEN_ID,
    BASE_WEIGHT_SOURCE,
    HIDDEN_CONTRACT,
    HIDDEN_NORM,
    HIDDEN_SOURCE,
    HIDDEN_WIDTH,
    MODEL_ID,
    NativeControl6LoRA,
    TICK_MS,
    TOKEN_ID_TO_STRING,
    load_base_rows,
    load_native_control6_adapter,
)


def prepare_real_targets(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    source_rows = _read_jsonl(manifest_path)
    audit = _validate_real_endpoint_rows(source_rows)
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for source in source_rows:
        conversation_id = str(source["conversation_id"])
        source_group = str(source["source_group_id"])
        split = str(source["split"])
        for frame in source["frames"]:
            # Keep the complete frame provenance (evidence IDs, source label,
            # masked reason, t-1 audio reference) alongside the model target.
            row = {
                **frame,
                "schema_version": TARGET_SCHEMA_VERSION,
                "schema": TARGET_SCHEMA,
                "conversation_id": conversation_id,
                "template_family_id": source_group,
                "source_group_id": source_group,
                "source": source["source"],
                "split": split,
            }
            by_split[split].append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"schema": "native_control6_real_endpoint_targets_v1", "audit": audit}
    for split, rows in by_split.items():
        _write_jsonl(output_dir / f"{split}_targets.jsonl", rows)
        counts = Counter(row["target_action"] for row in rows if row["action_mask"])
        summary[split] = {
            "rows": len(rows),
            "masked": sum(not row["action_mask"] for row in rows),
            "action_counts": {action: int(counts[action]) for action in ACTIONS},
            "source_groups": sorted({row["source_group_id"] for row in rows}),
        }
    temp = output_dir / "summary.json.tmp"
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(output_dir / "summary.json")
    return summary


def _weights_from_counts(counts: Mapping[str, int]) -> torch.Tensor:
    present = [action for action in ACTIONS if int(counts.get(action, 0)) > 0]
    if not present:
        raise ValueError("training source has no explicitly labeled actions")
    total = sum(int(counts[action]) for action in present)
    weights = [
        total / (len(present) * int(counts[action])) if action in present else 0.0
        for action in ACTIONS
    ]
    return torch.tensor(weights, dtype=torch.float32)


def _weighted_window_loss(
    model: NativeControl6LoRA,
    window: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    hidden, targets, mask = window
    selected_targets = targets[mask]
    if selected_targets.numel() == 0:
        raise ValueError("sampled training window has no labeled targets")
    logits = model(hidden)
    per_item = F.cross_entropy(logits[mask], selected_targets, reduction="none")
    selected_weights = weights.index_select(0, selected_targets)
    return (per_item * selected_weights).sum() / selected_weights.sum().clamp_min(1e-12)


def _evaluate(
    model: NativeControl6LoRA,
    sequences: Sequence[Mapping[str, Any]],
    *,
    batch_frames: int = 256,
) -> dict[str, Any]:
    confusion = torch.zeros((len(ACTIONS), len(ACTIONS)), dtype=torch.int64)
    model.eval()
    with torch.inference_mode():
        for sequence in sequences:
            n = int(sequence["frame_ids"].numel())
            for start in range(0, n, batch_frames):
                end = min(n, start + batch_frames)
                mask = sequence["loss_mask"][start:end]
                if not bool(mask.any()):
                    continue
                hidden = sequence["hidden_states"][start:end]
                targets = sequence["target_ids"][start:end]
                predicted = model(hidden).argmax(dim=-1)
                truth = targets[mask]
                guess = predicted[mask]
                bins = truth * len(ACTIONS) + guess
                confusion += torch.bincount(
                    bins, minlength=len(ACTIONS) ** 2
                ).reshape(len(ACTIONS), len(ACTIONS))
    support = confusion.sum(dim=1)
    predicted_support = confusion.sum(dim=0)
    tp = confusion.diag()
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for i, action in enumerate(ACTIONS):
        precision = float(tp[i] / predicted_support[i]) if int(predicted_support[i]) else 0.0
        recall = float(tp[i] / support[i]) if int(support[i]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if int(support[i]):
            f1_values.append(f1)
        per_class[action] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support[i]),
        }
    examples = int(support.sum())
    return {
        "examples": examples,
        "accuracy": float(tp.sum() / max(1, examples)),
        "macro_f1_supported_classes": sum(f1_values) / max(1, len(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "confusion_order": list(ACTIONS),
    }


def _validate_group_splits(
    t6_train: Sequence[Mapping[str, Any]],
    t6_validation: Sequence[Mapping[str, Any]],
    real_train: Sequence[Mapping[str, Any]],
    real_validation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups = {
        "t6_train": {str(row["template_family_id"]) for row in t6_train},
        "t6_validation": {str(row["template_family_id"]) for row in t6_validation},
        "real_train": {str(row["template_family_id"]) for row in real_train},
        "real_validation": {str(row["template_family_id"]) for row in real_validation},
    }
    for left, right in (("t6_train", "t6_validation"), ("real_train", "real_validation")):
        overlap = groups[left] & groups[right]
        if overlap:
            raise ValueError(f"source groups leak across {left}/{right}: {sorted(overlap)}")
    return {key: sorted(value) for key, value in groups.items()}


def train_mixed(
    *,
    real_train_targets: Path,
    real_validation_targets: Path,
    real_hidden_cache: Path,
    t6_train_targets: Path,
    t6_validation_targets: Path,
    t6_hidden_cache: Path,
    real_base_rows: Path,
    t6_base_rows: Path,
    baseline_adapter: Path,
    output_adapter: Path,
    report_path: Path,
    steps: int = 600,
    batch_frames: int = 128,
    eval_every: int = 100,
    learning_rate: float = 1e-3,
    seed: int = 260008,
) -> dict[str, Any]:
    if steps <= 0 or batch_frames <= 0 or eval_every <= 0:
        raise ValueError("steps, batch_frames, and eval_every must be positive")
    torch.set_num_threads(2)
    rng = random.Random(seed)

    real_train, real_train_info = load_sequences(real_train_targets, real_hidden_cache, "train")
    real_validation, real_validation_info = load_sequences(real_validation_targets, real_hidden_cache, "validation")
    t6_train, t6_train_info = load_sequences(t6_train_targets, t6_hidden_cache, "train")
    t6_validation, t6_validation_info = load_sequences(t6_validation_targets, t6_hidden_cache, "validation")
    group_splits = _validate_group_splits(t6_train, t6_validation, real_train, real_validation)
    real_metadata = real_train_info["cache_metadata"]
    t6_metadata = t6_train_info["cache_metadata"]
    for field in ("base_model_path", "response_adapter", "native_library_path", "native_max_seq_len", "native_rebase_retain_frames"):
        if real_metadata.get(field) != t6_metadata.get(field):
            raise ValueError(f"real/T6 hidden cache differs at native feature contract field {field}")

    real_rows, real_base_info = load_base_rows(real_base_rows)
    t6_rows, t6_base_info = load_base_rows(t6_base_rows)
    if real_base_info.get("base_weight_source") != BASE_WEIGHT_SOURCE or t6_base_info.get("base_weight_source") != BASE_WEIGHT_SOURCE:
        raise ValueError("both frozen six-row bundles must come from Gemma's tied native lm_head")
    if not torch.equal(real_rows.to(t6_rows.dtype), t6_rows):
        raise ValueError("real and T6 caches do not share identical frozen native Gemma rows")

    # This remains the unchanged six-row synthetic baseline artifact; initialize
    # a separate candidate from it and never overwrite it.
    model, baseline_report = load_native_control6_adapter(baseline_adapter, real_rows, device="cpu")
    if model.rank != 16 or model.alpha != 16.0:
        raise ValueError("baseline adapter rank/alpha differs from the approved rank-16 output-row contract")
    baseline_real = _evaluate(model, real_validation)
    baseline_t6 = _evaluate(model, t6_validation)
    best_score = 0.5 * (
        baseline_real["macro_f1_supported_classes"] + baseline_t6["macro_f1_supported_classes"]
    )
    best_step = 0
    best_state = {key: value.clone() for key, value in model.adapter_state_dict().items()}
    eval_history: list[dict[str, Any]] = [{
        "step": 0,
        "blended_macro_f1": best_score,
        "real_validation": baseline_real,
        "t6_validation": baseline_t6,
        "kind": "baseline600",
    }]

    real_counts = real_train_info["action_counts"]
    t6_counts = t6_train_info["action_counts"]
    real_weights = _weights_from_counts(real_counts)
    t6_weights = _weights_from_counts(t6_counts)
    real_windows = _sequence_windows(real_train, batch_frames=batch_frames, generator=None)
    t6_windows = _sequence_windows(t6_train, batch_frames=batch_frames, generator=None)
    if not real_windows or not t6_windows:
        raise ValueError("both real and synthetic train splits must have labeled chronological windows")
    optimizer = torch.optim.AdamW((model.lora_A, model.lora_B), lr=learning_rate)
    training_losses: list[dict[str, float]] = []
    no_improvement = 0
    stop_reason = "max_steps"

    for step in range(1, steps + 1):
        model.train()
        real_window = rng.choice(real_windows)
        t6_window = rng.choice(t6_windows)
        optimizer.zero_grad(set_to_none=True)
        real_loss = _weighted_window_loss(model, real_window, real_weights)
        t6_loss = _weighted_window_loss(model, t6_window, t6_weights)
        loss = 0.5 * real_loss + 0.5 * t6_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite mixed loss at update {step}")
        loss.backward()
        if model.lora_A.grad is None or model.lora_B.grad is None:
            raise RuntimeError("both native six-row LoRA factors must receive gradient")
        if model.base_rows.requires_grad:
            raise RuntimeError("native base output rows must remain frozen")
        optimizer.step()
        training_losses.append({
            "real_ce": float(real_loss.detach()),
            "t6_ce": float(t6_loss.detach()),
            "mixed_ce": float(loss.detach()),
        })

        if step % eval_every != 0 and step != steps:
            continue
        real_metrics = _evaluate(model, real_validation)
        t6_metrics = _evaluate(model, t6_validation)
        score = 0.5 * (real_metrics["macro_f1_supported_classes"] + t6_metrics["macro_f1_supported_classes"])
        eval_history.append({
            "step": step,
            "blended_macro_f1": score,
            "real_validation": real_metrics,
            "t6_validation": t6_metrics,
            "kind": "mixed_candidate",
        })
        if score > best_score + 1e-9:
            best_score = score
            best_step = step
            best_state = {key: value.clone() for key, value in model.adapter_state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= 3:
                stop_reason = "three_evaluations_without_blended_macro_f1_improvement"
                break

    model.load_adapter_state_dict(best_state)
    final_real = _evaluate(model, real_validation)
    final_t6 = _evaluate(model, t6_validation)
    report = {
        "schema_version": 1,
        "architecture": "native_gemma_selected_output_rows_lora",
        "model_id": MODEL_ID,
        "base_weight_source": BASE_WEIGHT_SOURCE,
        "actions": list(ACTIONS),
        "action_to_token_id": dict(ACTION_TO_TOKEN_ID),
        "token_id_to_string": {str(k): v for k, v in TOKEN_ID_TO_STRING.items()},
        "hidden_contract": HIDDEN_CONTRACT,
        "hidden_source": HIDDEN_SOURCE,
        "hidden_normalization": HIDDEN_NORM,
        "hidden_width": HIDDEN_WIDTH,
        "tick_ms": TICK_MS,
        "rank": model.rank,
        "alpha": model.alpha,
        "starting_checkpoint": str(baseline_adapter),
        "baseline_report_reference": baseline_report.get("validation_metrics"),
        "steps_completed": len(training_losses),
        "best_step": best_step,
        "stop_reason": stop_reason,
        "batch_frames_per_source": batch_frames,
        "source_mixture_per_update": {"real_endpoint": 0.5, "synthetic_t6": 0.5},
        "class_balance_weights": {
            "real_endpoint": {action: float(real_weights[i]) for i, action in enumerate(ACTIONS)},
            "synthetic_t6": {action: float(t6_weights[i]) for i, action in enumerate(ACTIONS)},
        },
        "learning_rate": learning_rate,
        "seed": seed,
        "response_adapter": "OFF",
        "ordinary_vocabulary_rows_modified": False,
        "base_gemma_weights_modified": False,
        "source_groups": group_splits,
        "train": {
            "real_endpoint": {key: value for key, value in real_train_info.items() if key != "cache_metadata"},
            "synthetic_t6": {key: value for key, value in t6_train_info.items() if key != "cache_metadata"},
        },
        "validation": {
            "baseline600": {"real_endpoint": baseline_real, "synthetic_t6": baseline_t6},
            "best_candidate": {"real_endpoint": final_real, "synthetic_t6": final_t6},
        },
        "evaluation_selection": "best mean(real supported-class macro-F1, synthetic six-class macro-F1); CE does not select checkpoints",
        "evaluation_history": eval_history,
        "mean_training_ce": {
            source: sum(row[key] for row in training_losses) / max(1, len(training_losses))
            for source, key in (("real_endpoint", "real_ce"), ("synthetic_t6", "t6_ce"), ("mixed", "mixed_ce"))
        },
        "last_50_training_ce": {
            source: sum(row[key] for row in training_losses[-50:]) / max(1, min(50, len(training_losses)))
            for source, key in (("real_endpoint", "real_ce"), ("synthetic_t6", "t6_ce"), ("mixed", "mixed_ce"))
        },
    }
    artifact = {
        "schema_version": 1,
        "architecture": report["architecture"],
        "model_id": MODEL_ID,
        "action_to_token_id": dict(ACTION_TO_TOKEN_ID),
        "token_id_to_string": {str(k): v for k, v in TOKEN_ID_TO_STRING.items()},
        "actions": list(ACTIONS),
        "hidden_contract": HIDDEN_CONTRACT,
        "rank": model.rank,
        "alpha": model.alpha,
        "base_weight_source": BASE_WEIGHT_SOURCE,
        "base_rows": real_rows.detach().cpu(),
        "response_adapter": "OFF",
        "ordinary_vocabulary_rows_modified": False,
        "base_gemma_weights_modified": False,
        "adapter_state_dict": best_state,
        "report": report,
    }
    output_adapter.parent.mkdir(parents=True, exist_ok=True)
    temp_artifact = output_adapter.with_suffix(output_adapter.suffix + ".tmp")
    torch.save(artifact, temp_artifact)
    temp_artifact.replace(output_adapter)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temp_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_report.replace(report_path)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-targets")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    train = sub.add_parser("train")
    for option in (
        "real-train-targets", "real-validation-targets", "real-hidden-cache",
        "t6-train-targets", "t6-validation-targets", "t6-hidden-cache",
        "real-base-rows", "t6-base-rows", "baseline-adapter", "output-adapter", "report",
    ):
        train.add_argument("--" + option, type=Path, required=True)
    train.add_argument("--steps", type=int, default=600)
    train.add_argument("--batch-frames", type=int, default=128)
    train.add_argument("--eval-every", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=260008)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare-targets":
        result = prepare_real_targets(args.manifest, args.output_dir)
    else:
        result = train_mixed(
            real_train_targets=args.real_train_targets,
            real_validation_targets=args.real_validation_targets,
            real_hidden_cache=args.real_hidden_cache,
            t6_train_targets=args.t6_train_targets,
            t6_validation_targets=args.t6_validation_targets,
            t6_hidden_cache=args.t6_hidden_cache,
            real_base_rows=args.real_base_rows,
            t6_base_rows=args.t6_base_rows,
            baseline_adapter=args.baseline_adapter,
            output_adapter=args.output_adapter,
            report_path=args.report,
            steps=args.steps,
            batch_frames=args.batch_frames,
            eval_every=args.eval_every,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
