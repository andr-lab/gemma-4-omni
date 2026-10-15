"""Build reviewable, train-only short response targets from V5 gold slots.

The builder changes only ``target_text``/``target_say_text`` and provenance
metadata on new candidate rows. It reuses the source audio, current-turn
``heard`` label, history, split, and conversation/group identifiers verbatim.
It deliberately emits no BACKCHANNEL, held-out, ASR, or legacy rows.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

_PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "duplex_kernel" / "audio_response_protocol.py"
_PROTOCOL_SPEC = importlib.util.spec_from_file_location("_duplex_audio_response_protocol", _PROTOCOL_PATH)
if _PROTOCOL_SPEC is None or _PROTOCOL_SPEC.loader is None:
    raise ImportError(f"cannot load shared audio response protocol: {_PROTOCOL_PATH}")
_PROTOCOL_MODULE = importlib.util.module_from_spec(_PROTOCOL_SPEC)
sys.modules[_PROTOCOL_SPEC.name] = _PROTOCOL_MODULE
_PROTOCOL_SPEC.loader.exec_module(_PROTOCOL_MODULE)
AUDIO_RESPONSE_PROTOCOL_V2 = _PROTOCOL_MODULE.AUDIO_RESPONSE_PROTOCOL_V2
format_audio_response_v2_target = _PROTOCOL_MODULE.format_audio_response_v2_target


_V5_PLACES = (
    "nella sala piccola", "all'ingresso", "nel cortile",
    "al tavolo vicino alla finestra", "nel punto di ritrovo",
)
_SPEAK_FOCUS_PATTERNS = (
    re.compile(r"^Partirei da (?P<focus>[^:;.!?]+):", re.IGNORECASE),
    re.compile(r"^Una prima possibilità è (?P<focus>[^;:.!?]+)[;:.!?]", re.IGNORECASE),
    re.compile(
        r"^Allora proporrei (?P<focus>.+?) (?:nella sala piccola|all'ingresso|nel cortile|"
        r"al tavolo vicino alla finestra|nel punto di ritrovo)[;:.!?]",
        re.IGNORECASE,
    ),
    re.compile(r"^Metterei (?P<focus>.+?) al primo posto\b", re.IGNORECASE),
)
_REPLAN_CORRECTION_PATTERNS = (
    re.compile(
        r"dobbiamo considerare (?P<positive>.+?),\s*non (?P<negative>.+?)[.!?]*$",
        re.IGNORECASE,
    ),
    re.compile(r"è cambiato (?P<positive>[^;:.!?]+)", re.IGNORECASE),
)


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFKC", " ".join(value.casefold().split()))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _extract_speak_focus(say: str) -> str | None:
    for pattern in _SPEAK_FOCUS_PATTERNS:
        match = pattern.search(say.strip())
        if match:
            focus = match.group("focus").strip(" \t\r\n,;:.!?—–-\"")
            if len(focus.split()) >= 2:
                return focus
    return None


def _extract_replan_slots(heard: str) -> tuple[str, str | None] | None:
    for pattern in _REPLAN_CORRECTION_PATTERNS:
        match = pattern.search(heard.strip())
        if not match:
            continue
        positive = match.group("positive").strip(" \t\r\n,;:.!?—–-\"")
        negative = match.groupdict().get("negative")
        if len(positive.split()) >= 2:
            return positive, negative.strip(" \t\r\n,;:.!?—–-\"") if negative else None
    return None


def _variants_for_row(row: Mapping[str, Any]) -> tuple[list[tuple[str, str]], dict[str, Any]] | None:
    action = str(row.get("action", "")).upper()
    heard = row.get("target_heard_text")
    say = row.get("target_say_text")
    if not isinstance(heard, str) or not isinstance(say, str) or not say.strip():
        return None

    if action == "SPEAK":
        focus = _extract_speak_focus(say)
        if focus is None:
            return None
        variants = [
            ("focus_concise_1", f"Come primo punto, consideriamo {focus}."),
            ("focus_concise_2", f"Per iniziare, teniamo {focus} come riferimento."),
        ]
        if not all(focus in candidate for _, candidate in variants):
            return None
        return variants, {"focus": focus, "has_explicit_negated_slot": False}

    if action == "REPLAN":
        slots = _extract_replan_slots(heard)
        if slots is None:
            return None
        positive, negative = slots
        if _normalized(positive) not in _normalized(say):
            return None
        variants = [
            ("correction_concise_1", f"Ricevuto, tengo presente {positive}."),
            ("correction_concise_2", f"Aggiorno il piano: il dato da seguire è {positive}."),
        ]
        normalized_negative = _normalized(negative) if negative else ""
        if normalized_negative and any(normalized_negative in _normalized(candidate) for _, candidate in variants):
            return None
        if not all(positive in candidate for _, candidate in variants):
            return None
        return variants, {
            "positive_correction": positive,
            "explicitly_negated_slot": negative,
            "has_explicit_negated_slot": bool(negative),
        }
    return None


def build_train_repair_candidates(
    records: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create two concise candidates for each eligible V5 train S/R row."""
    candidates: list[dict[str, Any]] = []
    eligible_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    sample_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    slot_checks: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for index, source in enumerate(records):
        action = str(source.get("action", "")).upper()
        eligible = (
            source.get("split") == "train"
            and source.get("profile") == "audio_grounded_v5"
            and source.get("source") == "authored_synthetic_audio_candidate"
            and source.get("task_kind") == "conversation_response"
            and source.get("audio_response_protocol") == AUDIO_RESPONSE_PROTOCOL_V2
            and action in {"SPEAK", "REPLAN"}
        )
        if not eligible:
            skipped["noneligible_row"] += 1
            continue
        eligible_counts[action] += 1
        generated = _variants_for_row(source)
        if generated is None:
            skipped[f"unsupported_or_unparsed_{action.lower()}"] += 1
            continue
        variants, evidence = generated
        heard = str(source["target_heard_text"])
        source_target = str(source["target_say_text"])
        source_target_sha = _sha256(_normalized(source_target))
        for variant_id, say in variants:
            row = dict(source)
            row.update(
                {
                    "target_say_text": say,
                    "target_text": format_audio_response_v2_target(heard=heard, action=action, say=say),
                    "target_text_sha256": _sha256(_normalized(format_audio_response_v2_target(
                        heard=heard, action=action, say=say,
                    ))),
                    "augmentation_only": True,
                    "target_variant_id": variant_id,
                    "target_variant_source_sha256": source_target_sha,
                    "target_variant_method": "source_slot_concise_v1",
                }
            )
            if row["target_heard_text"] != source["target_heard_text"]:
                raise AssertionError("builder modified current-turn heard supervision")
            if row.get("history_messages") != source.get("history_messages"):
                raise AssertionError("builder modified prompt history")
            if row.get("audio_path") != source.get("audio_path"):
                raise AssertionError("builder modified the physical audio source")
            if action == "SPEAK":
                slot_checks["speak_focus_exact_match_denominator"] += 1
                slot_checks["speak_focus_exact_match_numerator"] += int(evidence["focus"] in say)
            else:
                slot_checks["replan_positive_exact_match_denominator"] += 1
                slot_checks["replan_positive_exact_match_numerator"] += int(evidence["positive_correction"] in say)
                if evidence["has_explicit_negated_slot"]:
                    slot_checks["replan_negative_absence_denominator"] += 1
                    slot_checks["replan_negative_absence_numerator"] += int(
                        _normalized(evidence["explicitly_negated_slot"]) not in _normalized(say)
                    )
            candidates.append(row)
            variant_counts[action] += 1
            if sample_counts[action] < 2:
                samples.append(
                    {
                        "conversation_id": source.get("conversation_id"),
                        "target_utterance_id": source.get("target_utterance_id"),
                        "action": action,
                        "variant_id": variant_id,
                        "source_gold_say": source_target,
                        "source_heard": heard,
                        "candidate_say": say,
                        **evidence,
                    }
                )
                sample_counts[action] += 1

    # Stable order makes the candidate file and its review samples reproducible.
    candidates.sort(key=lambda row: (
        str(row.get("conversation_id", "")),
        str(row.get("target_utterance_id", "")),
        str(row.get("target_variant_id", "")),
    ))
    by_action = {action: int(variant_counts[action]) for action in ("SPEAK", "REPLAN")}
    source_by_action = {action: int(eligible_counts[action]) for action in ("SPEAK", "REPLAN")}
    summary = {
        "schema_version": 1,
        "purpose": "train_only_audio_response_target_candidates",
        "method": "source_slot_concise_v1",
        "protocol": AUDIO_RESPONSE_PROTOCOL_V2,
        "source_rows_by_action": source_by_action,
        "variants_by_action": by_action,
        "candidate_rows": len(candidates),
        "variants_per_eligible_source": 2,
        "excluded_noneligible_rows": int(skipped["noneligible_row"]),
        "skipped_rows": dict(sorted(skipped.items())),
        "slot_metrics": {
            "interpretation": "exact source-phrase retention/exclusion only; not semantic correctness",
            "speak_focus_exact_match": {
                "matched": int(slot_checks["speak_focus_exact_match_numerator"]),
                "total": int(slot_checks["speak_focus_exact_match_denominator"]),
            },
            "replan_positive_correction_exact_match": {
                "matched": int(slot_checks["replan_positive_exact_match_numerator"]),
                "total": int(slot_checks["replan_positive_exact_match_denominator"]),
            },
            "explicitly_negated_replan_slot_absent": {
                "matched": int(slot_checks["replan_negative_absence_numerator"]),
                "total": int(slot_checks["replan_negative_absence_denominator"]),
            },
        },
        "input_contract": {
            "only_v5_train_speak_replan": True,
            "validation_test_unchanged": True,
            "backchannel_targets_unchanged": True,
            "asr_targets_unchanged": True,
            "audio_and_current_heard_unchanged": True,
            "history_and_prompt_inputs_unchanged": True,
            "speak_variants_use_only_focus_span_from_source_gold": True,
            "replan_variants_use_only_positive_correction_from_current_heard": True,
            "explicitly_negated_correction_slot_excluded_from_replan_say": True,
            "speak_variants_add_no_time_or_schedule_detail": True,
            "weekday_inside_source_focus_is_preserved": True,
        },
        "review_samples": samples,
    }
    return candidates, summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: JSONL record must be an object")
                records.append(value)
    return records


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("conversation_id") or ""),
        str(record.get("target_utterance_id") or ""),
        str(record.get("action") or ""),
    )


def _training_example_id(record: Mapping[str, Any], ordinal: int) -> str:
    payload = json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"train-{ordinal:05d}-{_sha256(payload)[:16]}"


def _schedule_entry(record: Mapping[str, Any], step: int) -> dict[str, Any]:
    return {
        "step": step,
        **{
            key: record.get(key)
            for key in (
                "training_example_id", "task_mix_contract", "split", "task_kind", "action",
                "source", "source_group_id", "conversation_id", "target_variant_id",
            )
        },
    }


def build_repair_schedule(
    records: list[Mapping[str, Any]], *, updates: int = 200, seed: int = 20261002,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Create a deterministic 40/30/15/15 ordered schedule without replacement.

    SPEAK and REPLAN each cycle through two distinct repair variants followed
    by one original V5 target. All response examples come from the V5 train
    set; ASR examples are unique draws from the clean-ASR train pool.
    """
    if updates <= 0 or updates % 20:
        raise ValueError("repair schedule updates must be a positive multiple of 20")
    asr_count = updates * 2 // 5
    speak_count = updates * 3 // 10
    backchannel_count = updates * 3 // 20
    replan_count = updates - asr_count - speak_count - backchannel_count
    if speak_count % 3 or replan_count % 3:
        raise ValueError("SPEAK and REPLAN update counts must support an exact 2-variant:1-original cycle")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        example_id = str(record.get("training_example_id") or "").strip()
        if not example_id:
            raise ValueError(f"training record {index}: training_example_id mancante")
        if example_id in by_id:
            raise ValueError(f"duplicate training_example_id: {example_id}")
        by_id[example_id] = record
    asr_pool = [row for row in records if row.get("task_kind") == "asr"]
    original_by_action: dict[str, dict[tuple[str, str, str], Mapping[str, Any]]] = {
        action: {} for action in ("SPEAK", "BACKCHANNEL", "REPLAN")
    }
    variants_by_action: dict[str, dict[tuple[str, str, str], list[Mapping[str, Any]]]] = {
        action: {} for action in ("SPEAK", "REPLAN")
    }
    for row in records:
        if row.get("task_kind") != "conversation_response":
            continue
        action = str(row.get("action") or "")
        if row.get("source") != "authored_synthetic_audio_candidate" or row.get("profile") != "audio_grounded_v5":
            raise ValueError("repair schedule accepts only V5 direct-audio response rows")
        key = _record_key(row)
        if action in {"SPEAK", "BACKCHANNEL", "REPLAN"}:
            if row.get("augmentation_only") is True:
                if action == "BACKCHANNEL":
                    raise ValueError("BACKCHANNEL repair variants are not part of this schedule")
                variants_by_action[action].setdefault(key, []).append(row)
            elif key in original_by_action[action]:
                raise ValueError(f"duplicate original V5 response record: {key}")
            else:
                original_by_action[action][key] = row
    rng = random.Random(int(seed))
    if len(asr_pool) < asr_count:
        raise ValueError(f"clean-ASR pool too small: need {asr_count}, have {len(asr_pool)}")
    asr_queue = rng.sample(asr_pool, asr_count)
    rng.shuffle(asr_queue)
    if len(original_by_action["BACKCHANNEL"]) < backchannel_count:
        raise ValueError(f"V5 BACKCHANNEL pool too small: need {backchannel_count}")
    backchannel_queue = rng.sample(list(original_by_action["BACKCHANNEL"].values()), backchannel_count)
    speak_queue: list[Mapping[str, Any]] = []
    replan_queue: list[Mapping[str, Any]] = []
    for action, action_count, output_queue in (
        ("SPEAK", speak_count, speak_queue),
        ("REPLAN", replan_count, replan_queue),
    ):
        cycles = action_count // 3
        original_pool = original_by_action[action]
        variants_pool = variants_by_action[action]
        variant_keys = sorted(key for key, items in variants_pool.items() if len(items) >= 2)
        original_keys = sorted(original_pool)
        selected_variant_keys = rng.sample(variant_keys, cycles) if len(variant_keys) >= cycles else []
        selected_variant_set = set(selected_variant_keys)
        remaining_original_keys = [key for key in original_keys if key not in selected_variant_set]
        if len(selected_variant_keys) != cycles or len(remaining_original_keys) < cycles:
            raise ValueError(
                f"{action} pool cannot provide {cycles} disjoint variant/original cycles"
            )
        selected_original_keys = rng.sample(remaining_original_keys, cycles)
        cycle_rows: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
        for variant_key, original_key in zip(selected_variant_keys, selected_original_keys):
            pair = rng.sample(variants_pool[variant_key], 2)
            cycle_rows.append((pair[0], pair[1], original_pool[original_key]))
        rng.shuffle(cycle_rows)
        for first_variant, second_variant, original in cycle_rows:
            output_queue.extend((first_variant, second_variant, original))

    task_queues: dict[str, list[Mapping[str, Any]]] = {
        "asr": asr_queue,
        "SPEAK": speak_queue,
        "BACKCHANNEL": backchannel_queue,
        "REPLAN": replan_queue,
    }
    if any(len(task_queues[key]) != count for key, count in (
        ("asr", asr_count), ("SPEAK", speak_count),
        ("BACKCHANNEL", backchannel_count), ("REPLAN", replan_count),
    )):
        raise AssertionError("repair schedule queue count mismatch")
    first_four = ("asr", "SPEAK", "BACKCHANNEL", "REPLAN")
    schedule_records: list[Mapping[str, Any]] = [task_queues[name].pop(0) for name in first_four]
    remaining_labels = [name for name, queue in task_queues.items() for _ in queue]
    rng.shuffle(remaining_labels)
    for task_name in remaining_labels:
        schedule_records.append(task_queues[task_name].pop(0))
    if any(task_queues.values()) or len(schedule_records) != updates:
        raise AssertionError("ordered repair schedule did not consume all planned examples")
    identifiers = [str(row["training_example_id"]) for row in schedule_records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("repair schedule contains duplicate training IDs")

    task_counts = Counter(
        "asr" if row.get("task_kind") == "asr" else str(row.get("action"))
        for row in schedule_records
    )
    variants_count = Counter(
        str(row["action"])
        for row in schedule_records
        if row.get("augmentation_only") is True
    )
    summary = {
        "updates": updates,
        "seed": int(seed),
        "sampling": "without replacement; fixed first four updates cover ASR/SPEAK/BACKCHANNEL/REPLAN",
        "task_mix": {name: int(task_counts[name]) for name in ("asr", "SPEAK", "BACKCHANNEL", "REPLAN")},
        "variant_rows_by_action": {name: int(variants_count[name]) for name in ("SPEAK", "REPLAN")},
        "original_v5_rows_by_action": {
            action: int(sum(
                row.get("action") == action and row.get("augmentation_only") is not True
                for row in schedule_records
                if row.get("task_kind") == "conversation_response"
            ))
            for action in ("SPEAK", "BACKCHANNEL", "REPLAN")
        },
        "unique_training_example_ids": len(set(identifiers)),
        "schedule_rows": [_schedule_entry(row, step) for step, row in enumerate(schedule_records, 1)],
    }
    return schedule_records, summary


def prepare_repair_training_data(
    train_jsonl: Path, variants_jsonl: Path, output_dir: Path, *,
    schedule_updates: int = 200, seed: int = 20261002,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    source_rows = _read_jsonl(train_jsonl)
    variant_rows = _read_jsonl(variants_jsonl)
    repair_base_rows = [
        row for row in source_rows
        if row.get("split") == "train"
        and row.get("task_kind") == "conversation_response"
        and row.get("source") == "authored_synthetic_audio_candidate"
        and row.get("profile") == "audio_grounded_v5"
        and row.get("action") in {"SPEAK", "REPLAN"}
    ]
    base_by_key = {_record_key(row): row for row in repair_base_rows}
    if len(base_by_key) != len(repair_base_rows):
        raise ValueError("base V5 train JSONL contains duplicate conversation/target/action keys")
    seen_variants: set[tuple[tuple[str, str, str], str]] = set()
    for index, variant in enumerate(variant_rows):
        key = _record_key(variant)
        base = base_by_key.get(key)
        if base is None:
            raise ValueError(f"repair variant {index} has no original train target: {key}")
        if (
            variant.get("augmentation_only") is not True
            or variant.get("split") != "train"
            or variant.get("source") != "authored_synthetic_audio_candidate"
            or variant.get("profile") != "audio_grounded_v5"
            or variant.get("target_variant_source_sha256") != _sha256(_normalized(str(base.get("target_say_text", ""))))
        ):
            raise ValueError(f"repair variant {index} is not derived from its V5 train source target")
        variant_key = (key, str(variant.get("target_variant_id") or ""))
        if not variant_key[1] or variant_key in seen_variants:
            raise ValueError(f"repair variant {index} has a missing or duplicate variant ID")
        seen_variants.add(variant_key)
        for field in ("audio_path", "audio_start_ms", "audio_end_ms", "target_heard_text", "history_messages", "source_group_id"):
            if variant.get(field) != base.get(field):
                raise ValueError(f"repair variant {index} changed source input/provenance field: {field}")

    selected: list[dict[str, Any]] = []
    excluded = Counter()
    for row in source_rows:
        if row.get("split") not in {None, "train"}:
            raise ValueError("repair input train JSONL contains a non-train split row")
        if row.get("task_kind") == "asr" and row.get("source") == "synthetic_v3_clean_asr":
            selected_asr = dict(row)
            selected_asr["split"] = "train"
            selected.append(selected_asr)
        elif (
            row.get("task_kind") == "conversation_response"
            and row.get("source") == "authored_synthetic_audio_candidate"
            and row.get("profile") == "audio_grounded_v5"
            and row.get("audio_response_protocol") == AUDIO_RESPONSE_PROTOCOL_V2
            and row.get("action") in {"SPEAK", "BACKCHANNEL", "REPLAN"}
        ):
            selected.append(dict(row))
        else:
            excluded[str(row.get("task_kind") or "unknown") + ":" + str(row.get("source") or "unknown")] += 1
    selected.extend(dict(row) for row in variant_rows)
    if not selected:
        raise ValueError("repair train mix contains no eligible rows")
    for index, row in enumerate(selected):
        if row.get("task_mix_contract") != "audio_asr_mix_v2":
            raise ValueError(f"repair train row {index} lacks audio_asr_mix_v2 contract")
        row["training_example_id"] = _training_example_id(row, index)
    schedule, schedule_summary = build_repair_schedule(selected, updates=schedule_updates, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    schedule_path = output_dir / "ordered_schedule.jsonl"
    _write_jsonl(train_path, selected)
    _write_jsonl(schedule_path, schedule_summary["schedule_rows"])
    mix_task_counts = Counter(
        "asr" if row.get("task_kind") == "asr" else str(row.get("action"))
        for row in selected
    )
    summary = {
        "schema_version": 1,
        "purpose": "audio_response_v2_train_only_repair_mix",
        "train_jsonl": str(train_path),
        "ordered_schedule_jsonl": str(schedule_path),
        "source_train_jsonl": str(train_jsonl),
        "source_variant_jsonl": str(variants_jsonl),
        "source_rows": len(source_rows),
        "mixed_train_rows": len(selected),
        "mixed_train_rows_by_task": {name: int(mix_task_counts[name]) for name in ("asr", "SPEAK", "BACKCHANNEL", "REPLAN")},
        "excluded_rows_by_source_task": dict(sorted(excluded.items())),
        "repair_variant_rows": len(variant_rows),
        "schedule": {key: value for key, value in schedule_summary.items() if key != "schedule_rows"},
        "validation_jsonl_unchanged": str(train_jsonl.parent / "validation.jsonl"),
        "test_jsonl_unchanged": str(train_jsonl.parent / "test.jsonl"),
    }
    summary_path = output_dir / "manifest.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return summary


def prepare_candidates(train_jsonl: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates, summary = build_train_repair_candidates(_read_jsonl(train_jsonl))
    _write_jsonl(output_dir / "train_target_variants.jsonl", candidates)
    summary["input_train_jsonl"] = str(train_jsonl)
    summary["candidate_jsonl"] = str(output_dir / "train_target_variants.jsonl")
    summary["candidate_audio_paths_reused"] = len({str(row.get("audio_path", "")) for row in candidates})
    summary_path = output_dir / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants-jsonl", type=Path, default=None,
                        help="combine reviewed repair candidates with clean-ASR and original V5 train rows")
    parser.add_argument("--schedule-updates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args(argv)
    summary = (
        prepare_repair_training_data(
            args.train_jsonl, args.variants_jsonl, args.output_dir,
            schedule_updates=args.schedule_updates, seed=args.seed,
        )
        if args.variants_jsonl is not None
        else prepare_candidates(args.train_jsonl, args.output_dir)
    )
    print(json.dumps({key: summary[key] for key in (
        "candidate_rows", "source_rows_by_action", "variants_by_action", "skipped_rows",
        "candidate_jsonl", "candidate_audio_paths_reused", "mixed_train_rows",
        "mixed_train_rows_by_task", "repair_variant_rows", "ordered_schedule_jsonl",
    ) if key in summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
