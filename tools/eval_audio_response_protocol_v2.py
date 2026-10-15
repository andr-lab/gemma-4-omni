#!/usr/bin/env python3
"""Unconstrained evaluation of v2 audio responses and paired native-audio ASR."""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.audio_response_protocol import AudioResponseV2Parser
from training.duplex_qlora.prepare import read_jsonl
from training.duplex_qlora.stage1 import (
    _format_chat,
    _load_qlora,
    _read_audio_for_record,
    _validate_audio_asr_mix_records,
)


def _record_key(record: Mapping[str, Any]) -> str:
    conversation = str(record.get("conversation_id") or "")
    utterance = str(
        record.get("target_utterance_id")
        or record.get("segment_id")
        or record.get("target_start_ms")
        or record.get("audio_start_ms")
        or ""
    )
    if not conversation or not utterance:
        raise ValueError("evaluation record lacks a stable conversation/utterance key")
    return f"{conversation}::{utterance}"


def _audio_duration_ms(record: Mapping[str, Any]) -> float:
    duration = record.get("input_audio_duration_ms")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
        return float(duration)
    start, end = record.get("audio_start_ms"), record.get("audio_end_ms")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end <= start:
        raise ValueError(f"record {_record_key(record)} has no valid audio duration")
    return float(end) - float(start)


def _counterfactual_pairs(
    records: list[Mapping[str, Any]], *, per_action: int = 8, max_duration_ratio: float = 0.20,
) -> list[dict[str, Any]]:
    """Match distinct conversations within action and duration, without reusing targets/audio donors."""
    if per_action <= 0 or not 0 < max_duration_ratio < 1:
        raise ValueError("counterfactual pair limits must be positive and duration ratio below one")
    pairs: list[dict[str, Any]] = []
    for action in ("SPEAK", "BACKCHANNEL", "REPLAN"):
        candidates = sorted(
            (row for row in records if row.get("action") == action), key=_record_key,
        )
        donor_edges: dict[str, list[tuple[float, str]]] = {}
        by_key = {_record_key(row): row for row in candidates}
        for target in candidates:
            target_key = _record_key(target)
            target_duration = _audio_duration_ms(target)
            edges: list[tuple[float, str]] = []
            for donor in candidates:
                donor_key = _record_key(donor)
                if donor_key == target_key or donor.get("conversation_id") == target.get("conversation_id"):
                    continue
                donor_duration = _audio_duration_ms(donor)
                ratio = abs(target_duration - donor_duration) / max(target_duration, donor_duration)
                if ratio <= max_duration_ratio:
                    edges.append((ratio, donor_key))
            donor_edges[target_key] = sorted(edges)

        # Kuhn augmentation finds a maximum-cardinality one-to-one assignment.
        donor_to_target: dict[str, str] = {}

        def assign(target_key: str, seen: set[str]) -> bool:
            for _ratio, donor_key in donor_edges[target_key]:
                if donor_key in seen:
                    continue
                seen.add(donor_key)
                previous_target = donor_to_target.get(donor_key)
                if previous_target is None or assign(previous_target, seen):
                    donor_to_target[donor_key] = target_key
                    return True
            return False

        target_order = sorted(donor_edges, key=lambda key: (len(donor_edges[key]), key))
        for target_key in target_order:
            assign(target_key, set())
        matched = []
        for donor_key, target_key in donor_to_target.items():
            target, donor = by_key[target_key], by_key[donor_key]
            target_duration, donor_duration = _audio_duration_ms(target), _audio_duration_ms(donor)
            matched.append({
                "action": action,
                "target": target,
                "donor": donor,
                "duration_ratio": abs(target_duration - donor_duration) / max(target_duration, donor_duration),
                "target_duration_ms": target_duration,
                "donor_duration_ms": donor_duration,
            })
        pairs.extend(sorted(
            matched,
            key=lambda pair: (pair["duration_ratio"], _record_key(pair["target"]), _record_key(pair["donor"])),
        )[:per_action])
    return pairs


def _replace_counterfactual_audio(
    target: Mapping[str, Any], donor: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy only the physical audio source/span while preserving the target prompt and labels."""
    record = dict(target)
    for field in (
        "audio_path", "audio_start_ms", "audio_end_ms", "audio_channel_index",
        "sample_rate_hz", "source_audio_sha256",
    ):
        record[field] = donor[field]
    return record


def _text_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"\w+(?:['’]\w+)*", normalized, flags=re.UNICODE)


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return Levenshtein word errors and reference words for WER aggregation."""
    ref = _text_tokens(reference)
    hyp = _text_tokens(hypothesis)
    previous = list(range(len(hyp) + 1))
    for ref_index, ref_word in enumerate(ref, start=1):
        current = [ref_index]
        for hyp_index, hyp_word in enumerate(hyp, start=1):
            current.append(min(
                current[-1] + 1,
                previous[hyp_index] + 1,
                previous[hyp_index - 1] + int(ref_word != hyp_word),
            ))
        previous = current
    return previous[-1], len(ref)


def _eos_ids(model: Any, tokenizer: Any) -> set[int]:
    values = [
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "eot_id", None),
        getattr(tokenizer, "turn_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ]
    expanded: set[int] = set()
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            expanded.add(int(value))
        elif isinstance(value, (tuple, list, set)):
            expanded.update(int(item) for item in value if isinstance(item, int) and not isinstance(item, bool))
    if not expanded:
        raise RuntimeError("Gemma tokenizer/model has no EOS or EOT token configured")
    return expanded


def _encode_prompt(processor: Any, record: Mapping[str, Any], *, device: torch.device) -> dict[str, torch.Tensor]:
    prompt = _format_chat(processor, record, add_target=False)
    audio = _read_audio_for_record(record)
    encoded = processor(
        text=[prompt],
        audio=[audio],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    if "input_features" not in encoded or not isinstance(encoded["input_features"], torch.Tensor):
        raise RuntimeError("native audio was not converted into Gemma input_features")
    moved: dict[str, torch.Tensor] = {}
    for name, value in encoded.items():
        if not isinstance(value, torch.Tensor):
            continue
        dtype = torch.bfloat16 if name == "input_features" else value.dtype
        moved[name] = value.to(device=device, dtype=dtype)
    if not isinstance(moved.get("input_ids"), torch.Tensor) or moved["input_ids"].ndim != 2:
        raise RuntimeError("processor did not return prompt input_ids [1, sequence]")
    return moved


def _generate_one(
    model: Any,
    processor: Any,
    record: Mapping[str, Any],
    *,
    device: torch.device,
    max_new_tokens: int,
    use_cache: bool = True,
) -> dict[str, Any]:
    encoded = _encode_prompt(processor, record, device=device)
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_ids = _eos_ids(model, tokenizer)
    generation_config: dict[str, Any] = {
        **encoded,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "num_beams": 1,
        "use_cache": bool(use_cache),
        "eos_token_id": sorted(eos_ids),
    }
    pad_id = getattr(tokenizer, "pad_token_id", None)
    generation_config["pad_token_id"] = int(pad_id if pad_id is not None else min(eos_ids))
    prompt_length = int(encoded["input_ids"].shape[1])
    with torch.inference_mode():
        output = model.generate(**generation_config)
    sequences = output.sequences if hasattr(output, "sequences") else output
    generated_ids = [int(value) for value in sequences[0, prompt_length:].detach().cpu().tolist()]
    eos_index = next((i for i, token_id in enumerate(generated_ids) if token_id in eos_ids), None)
    content_ids = generated_ids[:eos_index] if eos_index is not None else generated_ids
    raw_text = tokenizer.decode(content_ids, skip_special_tokens=True).strip()
    result: dict[str, Any] = {
        "record_id": _record_key(record),
        "conversation_id": str(record.get("conversation_id") or ""),
        "target_utterance_id": str(record.get("target_utterance_id") or record.get("segment_id") or ""),
        "source": str(record.get("source") or "unknown"),
        "gold_action": str(record.get("action") or "") if record.get("task_kind") == "conversation_response" else None,
        "gold_text": str(record.get("target_text") or ""),
        "raw_generated_text": raw_text,
        "generated_token_count": len(generated_ids),
        "eos_terminated": eos_index is not None,
        "generation_budget": int(max_new_tokens),
        "use_cache": bool(use_cache),
    }
    if record.get("task_kind") == "conversation_response":
        result["gold_heard"] = str(record.get("target_heard_text") or "")
        result["gold_say"] = str(record.get("target_say_text") or "")
        parser = AudioResponseV2Parser()
        try:
            parser.feed(raw_text)
            parsed = parser.finish()
        except ValueError as exc:
            result.update({"protocol_valid": False, "protocol_error": getattr(exc, "reason", "invalid_json")})
        else:
            say_words = len(_text_tokens(parsed["say"]))
            heard_errors, heard_words = word_error_counts(result["gold_heard"], parsed["heard"])
            result.update({
                "protocol_valid": True,
                "parsed": parsed,
                "action_correct": parsed["action"] == result["gold_action"],
                "heard_exact": _text_tokens(parsed["heard"]) == _text_tokens(result["gold_heard"]),
                "heard_word_errors": heard_errors,
                "heard_reference_words": heard_words,
                "say_word_count": say_words,
                "backchannel_at_most_four_words": (
                    say_words <= 4 if result["gold_action"] == "BACKCHANNEL" else None
                ),
            })
    else:
        result["asr_word_errors"], result["asr_reference_words"] = word_error_counts(
            result["gold_text"], raw_text,
        )
    return result


def _atomic_save(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _empty_report(*, model_path: str, candidate_adapter: Path, baseline_adapter: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "heard_action_say_v2",
        "evaluation_policy": "greedy unconstrained generation; no expected action or transcript is added to model input",
        "model_path": model_path,
        "candidate_adapter": str(candidate_adapter),
        "baseline_v1_adapter": str(baseline_adapter),
        "decode_contract": None,
        "status": "in_progress",
        "v2_direct": {},
        "v2_counterfactual": {},
        "v2_asr": {},
        "v1_asr": {},
        "updated_utc_epoch": time.time(),
    }


def _load_report(path: Path, *, model_path: str, candidate_adapter: Path, baseline_adapter: Path) -> dict[str, Any]:
    if not path.is_file():
        return _empty_report(
            model_path=model_path, candidate_adapter=candidate_adapter, baseline_adapter=baseline_adapter,
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = (model_path, str(candidate_adapter), str(baseline_adapter))
    actual = (report.get("model_path"), report.get("candidate_adapter"), report.get("baseline_v1_adapter"))
    if report.get("protocol") != "heard_action_say_v2" or actual != expected:
        raise ValueError("existing evaluation report belongs to a different protocol/model/adapter")
    for key in ("v2_direct", "v2_counterfactual", "v2_asr", "v1_asr"):
        if not isinstance(report.get(key), dict):
            raise ValueError(f"existing evaluation report field {key} is invalid")
    return report


def _aggregate_direct(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows.values())
    result: dict[str, Any] = {"examples": len(values)}
    result["protocol_valid"] = sum(bool(row.get("protocol_valid")) for row in values)
    result["action_accuracy"] = (
        sum(bool(row.get("action_correct")) for row in values) / len(values) if values else None
    )
    by_action: dict[str, dict[str, Any]] = {}
    for action in ("SPEAK", "BACKCHANNEL", "REPLAN"):
        selected = [row for row in values if row.get("gold_action") == action]
        valid = [row for row in selected if row.get("protocol_valid") is True]
        errors = sum(int(row.get("heard_word_errors", 0)) for row in valid if action in {"SPEAK", "REPLAN"})
        words = sum(int(row.get("heard_reference_words", 0)) for row in valid if action in {"SPEAK", "REPLAN"})
        say_lengths = [int(row.get("say_word_count", 0)) for row in valid]
        by_action[action] = {
            "examples": len(selected),
            "protocol_valid": len(valid),
            "action_accuracy": (
                sum(bool(row.get("action_correct")) for row in selected) / len(selected) if selected else None
            ),
            "heard_wer": errors / words if words else (0.0 if action == "BACKCHANNEL" and valid else None),
            "heard_reference_words": words,
            "heard_empty_correct": (
                sum(bool(row.get("heard_exact")) for row in valid) if action == "BACKCHANNEL" else None
            ),
            "mean_say_words": statistics.mean(say_lengths) if say_lengths else None,
            "backchannels_at_most_four_words": (
                sum(bool(row.get("backchannel_at_most_four_words")) for row in valid)
                if action == "BACKCHANNEL" else None
            ),
        }
    result["by_action"] = by_action
    return result


def _aggregate_asr(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows.values())
    errors = sum(int(row.get("asr_word_errors", 0)) for row in values)
    words = sum(int(row.get("asr_reference_words", 0)) for row in values)
    return {
        "examples": len(values),
        "reference_words": words,
        "word_errors": errors,
        "wer": errors / words if words else None,
    }


def _aggregate_counterfactual(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows.values())
    result: dict[str, Any] = {"examples": len(values)}
    result["protocol_valid"] = sum(row.get("protocol_valid") is True for row in values)
    result["duration_ratio_mean"] = (
        statistics.mean(float(row["duration_ratio"]) for row in values) if values else None
    )
    result["duration_ratio_max"] = (
        max(float(row["duration_ratio"]) for row in values) if values else None
    )
    by_action: dict[str, dict[str, Any]] = {}
    for action in ("SPEAK", "BACKCHANNEL", "REPLAN"):
        selected = [row for row in values if row.get("gold_action") == action]
        valid = [row for row in selected if row.get("protocol_valid") is True]
        errors = sum(int(row.get("heard_word_errors", 0)) for row in valid if action != "BACKCHANNEL")
        words = sum(int(row.get("heard_reference_words", 0)) for row in valid if action != "BACKCHANNEL")
        say_lengths = [int(row.get("say_word_count", 0)) for row in valid]
        by_action[action] = {
            "examples": len(selected),
            "protocol_valid": len(valid),
            "action_accuracy": (
                sum(bool(row.get("action_correct")) for row in selected) / len(selected)
                if selected else None
            ),
            "heard_wer": errors / words if words else (
                0.0 if action == "BACKCHANNEL" and valid else None
            ),
            "heard_reference_words": words,
            "heard_empty_correct": (
                sum(bool(row.get("heard_exact")) for row in valid) if action == "BACKCHANNEL" else None
            ),
            "mean_say_words": statistics.mean(say_lengths) if say_lengths else None,
            "backchannels_at_most_four_words": (
                sum(bool(row.get("backchannel_at_most_four_words")) for row in valid)
                if action == "BACKCHANNEL" else None
            ),
            "mean_duration_ratio": (
                statistics.mean(float(row["duration_ratio"]) for row in selected) if selected else None
            ),
        }
    result["by_action"] = by_action
    return result


def _refresh_metrics(report: dict[str, Any]) -> None:
    report["metrics"] = {
        "v2_direct": _aggregate_direct(report["v2_direct"]),
        "v2_counterfactual": _aggregate_counterfactual(report["v2_counterfactual"]),
        "v2_asr": _aggregate_asr(report["v2_asr"]),
        "v1_asr": _aggregate_asr(report["v1_asr"]),
    }
    v2 = report["v2_asr"]
    v1 = report["v1_asr"]
    common = sorted(set(v2) & set(v1))
    if common:
        errors = sum(int(v2[key].get("asr_word_errors", 0)) for key in common)
        words = sum(int(v2[key].get("asr_reference_words", 0)) for key in common)
        baseline_errors = sum(int(v1[key].get("asr_word_errors", 0)) for key in common)
        baseline_words = sum(int(v1[key].get("asr_reference_words", 0)) for key in common)
        report["metrics"]["paired_asr"] = {
            "matched_examples": len(common),
            "v2_wer": errors / words if words else None,
            "v1_wer": baseline_errors / baseline_words if baseline_words else None,
            "wer_delta_v2_minus_v1": (
                errors / words - baseline_errors / baseline_words if words and baseline_words else None
            ),
        }
    else:
        report["metrics"]["paired_asr"] = {"matched_examples": 0, "v2_wer": None, "v1_wer": None, "wer_delta_v2_minus_v1": None}
    report["updated_utc_epoch"] = time.time()


def _run_stage(
    *,
    model_path: str,
    adapter_dir: Path,
    stage: str,
    records: list[dict[str, Any]],
    report: dict[str, Any],
    output_path: Path,
    max_new_tokens: int,
    use_cache: bool,
) -> None:
    pending = [row for row in records if _record_key(row) not in report[stage]]
    if not pending:
        return
    model, processor, _trainable, _modules, _quantization = _load_qlora(
        model_path,
        dtype=torch.bfloat16,
        resume_adapter_dir=adapter_dir,
    )
    model.eval()
    device = torch.device("cuda:0")
    try:
        for record in pending:
            result = _generate_one(
                model,
                processor,
                record,
                device=device,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
            )
            report[stage][result["record_id"]] = result
            _refresh_metrics(report)
            _atomic_save(output_path, report)
    finally:
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_counterfactual_stage(
    *,
    model_path: str,
    adapter_dir: Path,
    pairs: list[dict[str, Any]],
    report: dict[str, Any],
    output_path: Path,
    max_new_tokens: int,
    use_cache: bool,
) -> None:
    pending = [
        pair for pair in pairs
        if f"{_record_key(pair['target'])}<-{_record_key(pair['donor'])}" not in report["v2_counterfactual"]
    ]
    if not pending:
        return
    model, processor, _trainable, _modules, _quantization = _load_qlora(
        model_path,
        dtype=torch.bfloat16,
        resume_adapter_dir=adapter_dir,
    )
    model.eval()
    device = torch.device("cuda:0")
    try:
        for pair in pending:
            target, donor = pair["target"], pair["donor"]
            target_key, donor_key = _record_key(target), _record_key(donor)
            pair_key = f"{target_key}<-{donor_key}"
            changed_audio = _replace_counterfactual_audio(target, donor)
            result = _generate_one(
                model,
                processor,
                changed_audio,
                device=device,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
            )
            result.pop("gold_text", None)
            result.pop("gold_say", None)
            donor_heard = str(donor.get("target_heard_text") or "")
            result.update({
                "record_id": pair_key,
                "target_record_id": target_key,
                "donor_record_id": donor_key,
                "donor_source": str(donor.get("source") or "unknown"),
                "gold_action": str(donor.get("action") or ""),
                "gold_heard": donor_heard,
                "duration_ratio": float(pair["duration_ratio"]),
                "target_audio_duration_ms": float(pair["target_duration_ms"]),
                "donor_audio_duration_ms": float(pair["donor_duration_ms"]),
                "counterfactual_audio_only": True,
            })
            if result.get("protocol_valid") is True:
                parsed = result["parsed"]
                heard_errors, heard_words = word_error_counts(donor_heard, parsed["heard"])
                result.update({
                    "heard_exact": _text_tokens(parsed["heard"]) == _text_tokens(donor_heard),
                    "heard_word_errors": heard_errors,
                    "heard_reference_words": heard_words,
                    "action_correct": parsed["action"] == str(donor.get("action") or ""),
                })
            report["v2_counterfactual"][pair_key] = result
            _refresh_metrics(report)
            _atomic_save(output_path, report)
    finally:
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate(
    *,
    test_jsonl: Path,
    validation_jsonl: Path,
    model_path: str,
    candidate_adapter_dir: Path,
    baseline_v1_adapter_dir: Path,
    output_path: Path,
    max_direct_new_tokens: int = 384,
    max_asr_new_tokens: int = 160,
    use_cache: bool = True,
    direct_only: bool = False,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("audio response evaluation requires the 3090 CUDA runtime")
    torch.cuda.set_device(0)
    _test_rows, direct_by_task = _validate_audio_asr_mix_records(
        read_jsonl(test_jsonl), split="test", response_protocol="v2", require_asr=False,
    )
    direct_test = direct_by_task["conversation_response"]
    if len(direct_test) != 32 or {row["action"] for row in direct_test} != {"SPEAK", "BACKCHANNEL", "REPLAN"}:
        raise ValueError(f"held-out v5 test must contain its unique 32 response rows; found {len(direct_test)}")
    _, validation_by_task = _validate_audio_asr_mix_records(
        read_jsonl(validation_jsonl), split="validation", response_protocol="v2",
    )
    asr_rows = sorted(validation_by_task["asr"], key=_record_key)
    if len(asr_rows) != 32 or len({_record_key(row) for row in asr_rows}) != 32:
        raise ValueError(f"paired ASR retention evaluation requires 32 unique utterances; found {len(asr_rows)}")
    counterfactual_pairs = _counterfactual_pairs(direct_test, per_action=8, max_duration_ratio=0.20)
    counterfactual_counts = {
        action: sum(pair["action"] == action for pair in counterfactual_pairs)
        for action in ("SPEAK", "BACKCHANNEL", "REPLAN")
    }
    report = _load_report(
        output_path,
        model_path=model_path,
        candidate_adapter=candidate_adapter_dir,
        baseline_adapter=baseline_v1_adapter_dir,
    )
    decode_contract = {"use_cache": bool(use_cache), "direct_only": bool(direct_only)}
    if report.get("decode_contract") not in (None, decode_contract):
        raise ValueError("existing evaluation report was produced with a different cache/task contract")
    report["decode_contract"] = decode_contract
    report["evaluation_inputs"] = {
        "test_jsonl": str(test_jsonl),
        "validation_jsonl": str(validation_jsonl),
        "direct_test_examples": len(direct_test),
        "direct_test_actions": {
            action: sum(row["action"] == action for row in direct_test)
            for action in ("SPEAK", "BACKCHANNEL", "REPLAN")
        },
        "asr_validation_examples": len(asr_rows),
        "asr_validation_conversation_ids": len({str(row.get("conversation_id")) for row in asr_rows}),
        "counterfactual_pair_examples_by_action": counterfactual_counts,
        "counterfactual_max_duration_ratio": 0.20,
        "decode": (
            "greedy; no expected class, transcript, trie, or forced JSON tokens supplied; "
            f"use_cache={bool(use_cache)}; direct_only={bool(direct_only)}"
        ),
    }
    _atomic_save(output_path, report)
    _run_stage(
        model_path=model_path,
        adapter_dir=candidate_adapter_dir,
        stage="v2_direct",
        records=direct_test,
        report=report,
        output_path=output_path,
        max_new_tokens=max_direct_new_tokens,
        use_cache=use_cache,
    )
    _run_counterfactual_stage(
        model_path=model_path,
        adapter_dir=candidate_adapter_dir,
        pairs=counterfactual_pairs,
        report=report,
        output_path=output_path,
        max_new_tokens=max_direct_new_tokens,
        use_cache=use_cache,
    )
    if not direct_only:
        _run_stage(
            model_path=model_path,
            adapter_dir=candidate_adapter_dir,
            stage="v2_asr",
            records=asr_rows,
            report=report,
            output_path=output_path,
            max_new_tokens=max_asr_new_tokens,
            use_cache=use_cache,
        )
        _run_stage(
            model_path=model_path,
            adapter_dir=baseline_v1_adapter_dir,
            stage="v1_asr",
            records=asr_rows,
            report=report,
            output_path=output_path,
            max_new_tokens=max_asr_new_tokens,
            use_cache=use_cache,
        )
    report["status"] = "complete"
    _refresh_metrics(report)
    _atomic_save(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--candidate-adapter-dir", type=Path, required=True)
    parser.add_argument("--baseline-v1-adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-direct-new-tokens", type=int, default=384)
    parser.add_argument("--max-asr-new-tokens", type=int, default=160)
    parser.add_argument("--use-cache", choices=("true", "false"), default="true")
    parser.add_argument("--direct-only", action="store_true", help="evaluate only the direct-audio test split")
    args = parser.parse_args()
    report = evaluate(
        test_jsonl=args.test_jsonl,
        validation_jsonl=args.validation_jsonl,
        model_path=args.model_path,
        candidate_adapter_dir=args.candidate_adapter_dir,
        baseline_v1_adapter_dir=args.baseline_v1_adapter_dir,
        output_path=args.output,
        max_direct_new_tokens=args.max_direct_new_tokens,
        max_asr_new_tokens=args.max_asr_new_tokens,
        use_cache=args.use_cache == "true",
        direct_only=args.direct_only,
    )
    print(json.dumps({"status": report["status"], "metrics": report.get("metrics"), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
