"""Continuation QLoRA for action-conditioned assistant text generation."""
from __future__ import annotations

import gc
import json
import math
import random
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from duplex_kernel.action_conditioning import action_control_event
from .checkpointing import save_training_checkpoint
from .stage1 import (
    _format_chat,
    _load_qlora,
    _move_batch_to_device,
    _save_final_adapter,
    _training_manifest_path,
    _validate_probe_target,
    _write_training_manifest,
    build_training_inputs,
    run_text_conditioning_probe,
)

_ACTIONS = ("BACKCHANNEL", "SPEAK", "REPLAN")
_GENERATION_BUDGET_BY_ACTION = {"BACKCHANNEL": 16, "SPEAK": 192, "REPLAN": 128}


def _generation_observation(
    *,
    action: str,
    token_ids: list[int],
    eos_token_ids: set[int],
    token_budget: int,
    generated_text: str,
) -> dict[str, Any]:
    """Classify actual generated IDs without confusing EOS with budget truncation."""
    if action not in _ACTIONS or token_budget <= 0:
        raise ValueError("azione o budget generativo non valido")
    terminal_index = next((index for index, token_id in enumerate(token_ids) if token_id in eos_token_ids), None)
    content_ids = token_ids[:terminal_index] if terminal_index is not None else token_ids
    words = len(generated_text.split())
    eos_terminated = terminal_index is not None
    reached_budget_without_eos = not eos_terminated and len(token_ids) >= token_budget
    return {
        "generated_text": generated_text,
        "generated_token_ids": token_ids,
        "generated_content_token_ids": content_ids,
        "generated_token_count_before_eos": len(content_ids),
        "generated_word_count": words,
        "token_budget": token_budget,
        "eos_terminated": eos_terminated,
        "eos_token_id": token_ids[terminal_index] if terminal_index is not None else None,
        "reached_budget_without_eos": reached_budget_without_eos,
        "empty_generation": not generated_text.strip(),
        "backchannel_1_to_4_word_candidate": (
            action == "BACKCHANNEL" and 1 <= words <= 4 and eos_terminated and not reached_budget_without_eos
        ),
        "backchannel_full_answer_length_risk": action == "BACKCHANNEL" and words > 4,
        "qualitative_review_required": action in _ACTIONS,
    }


def _summarize_generations(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for action in _ACTIONS:
        selected = [row for row in rows if row.get("action") == action]
        summary[action] = {
            "examples": len(selected),
            "eos_terminated": sum(bool(row.get("eos_terminated")) for row in selected),
            "reached_budget_without_eos": sum(bool(row.get("reached_budget_without_eos")) for row in selected),
            "empty_generation": sum(bool(row.get("empty_generation")) for row in selected),
            "backchannel_1_to_4_word_candidates": sum(bool(row.get("backchannel_1_to_4_word_candidate")) for row in selected),
            "backchannel_full_answer_length_risk": sum(bool(row.get("backchannel_full_answer_length_risk")) for row in selected),
            "mean_generated_words": (
                sum(int(row.get("generated_word_count", 0)) for row in selected) / len(selected) if selected else None
            ),
        }
    return summary


def _read_records(path: str | Path, *, split: str) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
    if not records:
        raise ValueError(f"dataset action-conditioned vuoto: {path}")
    for record in records:
        if record.get("split") != split or record.get("conditioning_mode") != "action_conditioned_text":
            raise ValueError(f"{path}: record non action-conditioned o split errato")
        if record.get("task_kind") != "conversation_response" or record.get("audio_mask") is not False:
            raise ValueError(f"{path}: il delta deve supervisionare solo testo assistant senza feature audio")
        if record.get("action") not in _ACTIONS or not str(record.get("target_text", "")).strip():
            raise ValueError(f"{path}: target/action mancante")
        if not isinstance(record.get("history_messages"), list):
            raise ValueError(f"{path}: history_messages non è una lista")
    return records


def _assert_group_disjoint(train_records: list[Mapping[str, Any]], validation_records: list[Mapping[str, Any]]) -> None:
    train_groups = {str(record.get("source_group_id")) for record in train_records}
    validation_groups = {str(record.get("source_group_id")) for record in validation_records}
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError(f"source_group_id attraversa train/validation: {sorted(overlap)[:5]}")


def _balanced_schedule(
    records: list[Mapping[str, Any]], *, steps: int, seed: int
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = {
        action: [record for record in records if record.get("action") == action]
        for action in _ACTIONS
    }
    empty = [action for action, values in buckets.items() if not values]
    if empty:
        raise ValueError(f"il train split non contiene tutte le modalità richieste: {empty}")
    rng = random.Random(int(seed))
    # The continuation delta allocates half its updates to substantive speech
    # while preserving equal quarter shares for acknowledgment and repair.
    action_slots = ["BACKCHANNEL", "SPEAK", "SPEAK", "REPLAN"]
    schedule_actions = [action_slots[index % len(action_slots)] for index in range(int(steps))]
    rng.shuffle(schedule_actions)
    for values in buckets.values():
        rng.shuffle(values)
    positions = {action: 0 for action in _ACTIONS}
    schedule: list[Mapping[str, Any]] = []
    for action in schedule_actions:
        bucket = buckets[action]
        if positions[action] >= len(bucket):
            rng.shuffle(bucket)
            positions[action] = 0
        schedule.append(bucket[positions[action]])
        positions[action] += 1
    counts = Counter(str(record["action"]) for record in schedule)
    return schedule, {
        "sampling": "50_percent_speak_25_percent_backchannel_25_percent_replan_with_replacement_after_class_exhaustion",
        "scheduled_update_shares": {action: counts[action] / int(steps) for action in _ACTIONS},
        "dataset_records_by_action": {action: len(buckets[action]) for action in _ACTIONS},
        "scheduled_updates_by_action": {action: counts[action] for action in _ACTIONS},
        "unique_train_target_texts_by_action": {
            action: len({str(record.get("target_text_sha256")) for record in buckets[action]})
            for action in _ACTIONS
        },
    }


def _evaluate_by_action(
    model: Any,
    processor: Any,
    records: list[Mapping[str, Any]],
    *,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    from .stage1 import build_training_inputs

    device = next(model.parameters()).device
    was_training = bool(getattr(model, "training", True))
    model.eval()
    losses: dict[str, list[float]] = defaultdict(list)
    token_counts: Counter[str] = Counter()
    try:
        with torch.no_grad():
            for record in records:
                inputs = build_training_inputs(processor, record)
                if "input_features" in inputs or "input_features_mask" in inputs:
                    raise RuntimeError("action-conditioned text validation deve restare senza audio")
                moved, batch_status = _move_batch_to_device(
                    inputs, device=device, compute_dtype=compute_dtype
                )
                output = model(**moved, use_cache=False, return_dict=True)
                loss = getattr(output, "loss", None)
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise RuntimeError("loss validation action-conditioned non finita")
                action = str(record["action"])
                losses[action].append(float(loss.detach().float().cpu()))
                token_counts[action] += int(batch_status["non_ignored_label_tokens"])
    finally:
        model.train(was_training)
    return {
        "examples_by_action": {action: len(losses[action]) for action in _ACTIONS},
        "mean_causal_lm_loss_by_action": {
            action: (sum(losses[action]) / len(losses[action]) if losses[action] else None)
            for action in _ACTIONS
        },
        "median_causal_lm_loss_by_action": {
            action: (float(np.median(losses[action])) if losses[action] else None)
            for action in _ACTIONS
        },
        "assistant_supervised_tokens_by_action": {action: token_counts[action] for action in _ACTIONS},
    }


def _generate_validation_examples(
    model: Any,
    processor: Any,
    records: list[Mapping[str, Any]],
    *,
    limit_per_action: int = 2,
    adapter_stage: str,
) -> list[dict[str, Any]]:
    tokenizer = getattr(processor, "tokenizer", processor)
    device = next(model.parameters()).device
    selected: list[Mapping[str, Any]] = []
    for action in _ACTIONS:
        by_group: dict[str, Mapping[str, Any]] = {}
        for record in records:
            if record.get("action") != action:
                continue
            by_group.setdefault(str(record.get("source_group_id")), record)
        selected.extend(list(by_group.values())[:limit_per_action])
    generated: list[dict[str, Any]] = []
    was_training = bool(getattr(model, "training", True))
    model.eval()
    try:
        with torch.no_grad():
            for record in selected:
                prompt = _format_chat(processor, record, add_target=False)
                encoded = processor(text=[prompt], return_tensors="pt", padding=True)
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                action = str(record["action"])
                token_budget = _GENERATION_BUDGET_BY_ACTION[action]
                eos_token_ids = {
                    int(value)
                    for value in (
                        getattr(tokenizer, "eos_token_id", None),
                        getattr(tokenizer, "eot_id", None),
                        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
                    )
                    if isinstance(value, int)
                }
                # Gemma chat turn ends may be declared as more than one EOS ID.
                model_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
                if isinstance(model_eos, (list, tuple, set)):
                    eos_token_ids.update(int(value) for value in model_eos)
                if not eos_token_ids:
                    raise RuntimeError("nessun EOS/EOT configurato per la generazione di validazione")
                generation_eos: int | list[int] = sorted(eos_token_ids)
                if len(eos_token_ids) == 1:
                    generation_eos = next(iter(eos_token_ids))
                pad_token_id = getattr(tokenizer, "pad_token_id", None)
                if pad_token_id is None:
                    pad_token_id = next(iter(eos_token_ids))
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=token_budget,
                    do_sample=False,
                    pad_token_id=int(pad_token_id),
                    eos_token_id=generation_eos,
                )
                new_tokens = [int(value) for value in output[0, input_ids.shape[1]:].detach().cpu().tolist()]
                terminal_index = next((index for index, value in enumerate(new_tokens) if value in eos_token_ids), None)
                content_tokens = new_tokens[:terminal_index] if terminal_index is not None else new_tokens
                text = tokenizer.decode(content_tokens, skip_special_tokens=True).strip()
                user_context = next(
                    (str(item.get("content", "")) for item in reversed(record.get("history_messages", [])) if item.get("role") == "user"),
                    "",
                )
                observation = _generation_observation(
                    action=action,
                    token_ids=new_tokens,
                    eos_token_ids=eos_token_ids,
                    token_budget=token_budget,
                    generated_text=text,
                )
                generated.append({
                    "generation_pair_id": "|".join((str(record.get("source_group_id")), str(record.get("target_utterance_id")))),
                    "adapter_stage": adapter_stage,
                    "action": action,
                    "source_group_id": record.get("source_group_id"),
                    "conversation_id": record.get("conversation_id"),
                    "target_utterance_id": record.get("target_utterance_id"),
                    "prompt_roles": [
                        "system",
                        *(str(item["role"]) for item in record.get("history_messages", [])),
                        "user",
                    ],
                    "input_token_ids": [int(value) for value in input_ids[0].detach().cpu().tolist()],
                    "action_conditioning_source": "ordinary_text_system_instruction_and_empty_ephemeral_user_turn",
                    "action_conditioning_is_atomic_token": False,
                    "control_action_label": action,
                    "internal_control_event_content": action_control_event(action),
                    "completed_user_context": user_context,
                    "reference_text": record.get("target_text"),
                    **observation,
                })
    finally:
        model.train(was_training)
    return generated


def train_action_conditioned_delta(
    *,
    train_jsonl: str | Path,
    validation_jsonl: str | Path,
    initial_adapter_dir: str | Path,
    model_path: str,
    adapter_dir: str | Path,
    max_steps: int = 200,
    learning_rate: float = 2e-5,
    seed: int = 20261001,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train an action-balanced, fresh-optimizer delta from an existing adapter."""
    if max_steps <= 0 or learning_rate <= 0:
        raise ValueError("max_steps e learning_rate devono essere positivi")
    steps = 4 if smoke else int(max_steps)
    initial_adapter = Path(initial_adapter_dir).expanduser().resolve()
    output_adapter = Path(adapter_dir).expanduser().resolve()
    if not initial_adapter.is_dir():
        raise FileNotFoundError(f"adapter iniziale v4 non trovato: {initial_adapter}")
    if output_adapter.exists():
        raise FileExistsError(f"adapter delta esistente, non sovrascritto: {output_adapter}")
    run_root = output_adapter.parent
    old_steps = sorted(run_root.glob("step_[0-9][0-9][0-9][0-9][0-9][0-9]")) if run_root.exists() else []
    if old_steps:
        raise FileExistsError(f"checkpoint delta già presenti, non sovrascritti: {old_steps[-1]}")
    train_records = _read_records(train_jsonl, split="train")
    validation_records = _read_records(validation_jsonl, split="validation")
    _assert_group_disjoint(train_records, validation_records)
    schedule, balance_report = _balanced_schedule(train_records, steps=steps, seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    compute_dtype = torch.bfloat16
    model, processor, trainable, matched_lora_modules, quantization_report = _load_qlora(
        model_path, dtype=compute_dtype, resume_adapter_dir=initial_adapter
    )
    trainable_model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_parameters = [parameter for _, parameter in trainable]
    if {id(parameter) for parameter in optimizer_parameters} != {id(parameter) for parameter in trainable_model_parameters}:
        raise RuntimeError("delta optimizer deve contenere esattamente i parametri LoRA trainabili")
    starting_adapter_snapshot = [parameter.detach().float().clone() for parameter in optimizer_parameters]
    tokenizer = getattr(processor, "tokenizer", processor)
    run_id = uuid.uuid4().hex
    initial_manifest_path = initial_adapter.parent / "training_manifest.json"
    initial_manifest = json.loads(initial_manifest_path.read_text(encoding="utf-8")) if initial_manifest_path.is_file() else {}
    training_config = {
        "train_jsonl": str(Path(train_jsonl).expanduser().resolve()),
        "validation_jsonl": str(Path(validation_jsonl).expanduser().resolve()),
        "model_path": model_path,
        "initial_adapter_dir": str(initial_adapter),
        "initial_adapter_run_id": initial_manifest.get("run_id"),
        "adapter_dir": str(output_adapter),
        "max_steps": steps,
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "checkpoint_interval": 100,
        "conditioning_mode": "action_conditioned_text",
        "action_balance": balance_report["sampling"],
        "optimizer_state": "fresh_adamw",
        "objective": "action_conditioned_text_assistant_causal_lm",
    }
    probe_record = schedule[0]
    probe_inputs = build_training_inputs(processor, probe_record)
    probe_target = _validate_probe_target(processor, probe_record, probe_inputs)
    conditioning_probe = run_text_conditioning_probe(
        model,
        processor,
        probe_record,
        probe_inputs,
        device=next(model.parameters()).device,
        compute_dtype=compute_dtype,
    )
    conditioning_probe["action"] = str(probe_record["action"])
    conditioning_probe["action_conditioning_instruction"] = __import__(
        "duplex_kernel.action_conditioning", fromlist=["action_conditioning_instruction"]
    ).action_conditioning_instruction(str(probe_record["action"]))
    validation_before = None if smoke else _evaluate_by_action(
        model, processor, validation_records, compute_dtype=compute_dtype
    )
    baseline_generations = [] if smoke else _generate_validation_examples(
        model,
        processor,
        validation_records,
        limit_per_action=2,
        adapter_stage="initial_adapter",
    )
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=float(learning_rate))
    adapter_root = run_root
    adapter_root.mkdir(parents=True, exist_ok=True)
    losses: list[float] = []
    step_runtime: list[dict[str, Any]] = []
    model.train()
    for index, record in enumerate(schedule):
        inputs = build_training_inputs(processor, record)
        moved, batch_status = _move_batch_to_device(
            inputs, device=next(model.parameters()).device, compute_dtype=compute_dtype
        )
        labels = moved.get("labels")
        if not isinstance(labels, torch.Tensor) or not bool(labels.ne(-100).any()):
            raise RuntimeError("delta action-conditioned senza label assistant dopo la mask prompt")
        optimizer.zero_grad(set_to_none=True)
        output = model(**moved, use_cache=False, return_dict=True)
        loss = getattr(output, "loss", None)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise RuntimeError("loss action-conditioned non finita")
        loss.backward()
        gradients = [parameter.grad for parameter in optimizer_parameters if parameter.grad is not None]
        if not gradients or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError("gradienti LoRA assenti o non finiti nel delta action-conditioned")
        if not any(bool(gradient.ne(0).any()) for gradient in gradients):
            raise RuntimeError("tutti i gradienti LoRA action-conditioned sono zero")
        optimizer.step()
        step = index + 1
        loss_value = float(loss.detach().float().cpu())
        losses.append(loss_value)
        step_runtime.append({
            "step": step,
            "action": str(record["action"]),
            "loss": loss_value,
            "non_ignored_label_tokens": int(batch_status["non_ignored_label_tokens"]),
            "lora_gradient_count": len(gradients),
            "lora_gradients_finite": True,
            "lora_has_nonzero_gradient": True,
        })
        if step % 100 == 0 or step == steps:
            checkpoint_path = save_training_checkpoint(
                checkpoint_root=adapter_root,
                completed_steps=step,
                optimizer=optimizer,
                training_config=training_config,
                losses=losses,
                step_runtime=step_runtime,
                run_metadata={
                    "run_id": run_id,
                    "initial_adapter_dir": str(initial_adapter),
                    "initial_adapter_run_id": initial_manifest.get("run_id"),
                    "conditioning_probe": conditioning_probe,
                    "action_balance": balance_report,
                    "validation_before": validation_before,
                },
                save_adapter=lambda path: model.save_pretrained(path, save_embedding_layers=False),
            )
            print(f"Action delta checkpoint step={step} path={checkpoint_path}")
    max_delta = max(
        float((parameter.detach().float() - before).abs().max().cpu())
        for parameter, before in zip(optimizer_parameters, starting_adapter_snapshot)
    )
    if not max_delta > 0:
        raise RuntimeError("nessun parametro LoRA è cambiato nel delta action-conditioned")
    validation_after = None if smoke else _evaluate_by_action(
        model, processor, validation_records, compute_dtype=compute_dtype
    )
    heldout_generations = [] if smoke else _generate_validation_examples(
        model,
        processor,
        validation_records,
        limit_per_action=2,
        adapter_stage="action_delta_after_training",
    )
    baseline_by_pair = {row["generation_pair_id"]: row for row in baseline_generations}
    delta_by_pair = {row["generation_pair_id"]: row for row in heldout_generations}
    if not smoke and baseline_by_pair.keys() != delta_by_pair.keys():
        raise RuntimeError("generazioni baseline e delta non usano gli stessi contesti heldout")
    generation_comparison = {
        "initial_adapter_by_action": _summarize_generations(baseline_generations),
        "delta_adapter_by_action": _summarize_generations(heldout_generations),
        "same_context_pair_count": len(baseline_by_pair.keys() & delta_by_pair.keys()),
        "paired_examples": [
            {"generation_pair_id": key, "baseline": baseline_by_pair[key], "delta": delta_by_pair[key]}
            for key in sorted(baseline_by_pair.keys() & delta_by_pair.keys())
        ],
        "qualitative_review_contract": {
            "BACKCHANNEL": "Valutare come breve riscontro 1-4 parole, terminato da EOS, senza sviluppare una risposta completa; nessun testo/frasario fisso è richiesto.",
            "SPEAK": "Valutare pertinenza rispetto al contesto utente completo e alla reference dello stesso target.",
            "REPLAN": "Valutare se la risposta incorpora la correzione/preferenza utente completa più recente.",
            "automated_semantic_judgment": False,
        },
        "generation_inputs_and_control_labels_are_reported_separately": True,
    }
    smoke_reference_loss = None
    if smoke:
        model.eval()
        reference_inputs = build_training_inputs(processor, probe_record)
        reference_moved, _ = _move_batch_to_device(
            reference_inputs, device=next(model.parameters()).device, compute_dtype=compute_dtype
        )
        with torch.no_grad():
            reference_output = model(**reference_moved, use_cache=False, return_dict=True)
        smoke_reference_loss = float(reference_output.loss.detach().float().cpu())
        if not math.isfinite(smoke_reference_loss):
            raise RuntimeError("loss smoke pre-save non finita")
    _save_final_adapter(model, processor, output_adapter)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    save_load_check = None
    if smoke:
        reloaded, reloaded_processor, _, _, _ = _load_qlora(
            model_path, dtype=compute_dtype, resume_adapter_dir=output_adapter
        )
        reloaded.eval()
        reload_inputs = build_training_inputs(reloaded_processor, probe_record)
        reload_moved, _ = _move_batch_to_device(
            reload_inputs, device=next(reloaded.parameters()).device, compute_dtype=compute_dtype
        )
        with torch.no_grad():
            reload_output = reloaded(**reload_moved, use_cache=False, return_dict=True)
        reload_loss = float(reload_output.loss.detach().float().cpu())
        loss_delta = abs(reload_loss - float(smoke_reference_loss))
        save_load_check = {
            "loaded_adapter_dir": str(output_adapter),
            "finite_loss_after_reload": math.isfinite(reload_loss),
            "reload_probe_loss": reload_loss,
            "pre_save_eval_probe_loss": smoke_reference_loss,
            "absolute_loss_difference_after_reload": loss_delta,
        }
        if not save_load_check["finite_loss_after_reload"] or loss_delta > 5e-4:
            raise RuntimeError(
                "adapter smoke salvato ma il forward dopo reload non è finito/equivalente: "
                f"{save_load_check}"
            )
        del reloaded, reload_output
        gc.collect()
        torch.cuda.empty_cache()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "base_model_id": model_path,
        "base_model_path": str(Path(model_path).expanduser().resolve()) if Path(model_path).exists() else model_path,
        "checkpoint_root": str(adapter_root.resolve()),
        "adapter_dir": str(output_adapter),
        "initialized_from_adapter_dir": str(initial_adapter),
        "initialized_from_run_id": initial_manifest.get("run_id"),
        "control_head_path": str((adapter_root / "control_head.pt").resolve()),
        "control_head": {"input_dim": 1536, "pooling": "last_audio_query", "tick_ms": 80, "actions": ["LISTEN", "SPEAK", "CONTINUE", "YIELD", "BACKCHANNEL", "REPLAN"]},
        "action_conditioning": {
            "system_instruction_module": "duplex_kernel.action_conditioning",
            "actions": list(_ACTIONS),
            "atomic_action_tokens_added": False,
        },
        "training": {
            "objective": "action_conditioned_text_assistant_causal_lm",
            "optimizer": "fresh_adamw",
            "steps": steps,
            "learning_rate": float(learning_rate),
            "trainable_lora_parameter_count": sum(parameter.numel() for _, parameter in trainable),
            "lora_target_modules": matched_lora_modules,
            "lora_max_abs_parameter_delta": max_delta,
            "lora_max_abs_parameter_delta_reference": "initialized_from_adapter",
            "quantization_runtime": quantization_report,
            "loss_mean": sum(losses) / len(losses),
            "action_balance": balance_report,
            "conditioning_probe": conditioning_probe,
            "probe_target_contract": probe_target,
            "validation_before_by_action": validation_before,
            "validation_after_by_action": validation_after,
            "generation_comparison": generation_comparison,
            "heldout_generations": heldout_generations,
            "step_runtime": step_runtime,
            "smoke_save_load_check": save_load_check,
        },
        "data": {
            "train_jsonl": str(Path(train_jsonl).expanduser().resolve()),
            "validation_jsonl": str(Path(validation_jsonl).expanduser().resolve()),
            "train_records_by_action": balance_report["dataset_records_by_action"],
            "validation_records_by_action": {
                action: sum(record.get("action") == action for record in validation_records)
                for action in _ACTIONS
            },
            "validation_source_groups": sorted({str(record.get("source_group_id")) for record in validation_records}),
            "train_validation_group_disjoint": True,
            "whisper_quality_evidence": "not_measured_for_this_v4_target_set",
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
        },
        "smoke": bool(smoke),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = _training_manifest_path(output_adapter)
    _write_training_manifest(manifest_path, manifest)
    report_path = adapter_root / "action_delta_report.json"
    report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"run_id": run_id, "adapter_dir": str(output_adapter), "training_manifest": str(manifest_path), "report": str(report_path), "training": manifest["training"]}


__all__ = ["train_action_conditioned_delta"]
