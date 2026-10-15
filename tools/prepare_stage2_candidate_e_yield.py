#!/usr/bin/env python3
"""Prepare Candidate E feature targets and a one-target native-listener probe."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


ACTIONS = ("LISTEN", "SPEAK", "CONTINUE", "YIELD", "BACKCHANNEL", "REPLAN")
FRAME_SAMPLES = 1280
YIELD_DELAYS_MS = {320, 400, 480}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _target_key(row: Mapping[str, Any]) -> str:
    if "frame_targets" in row:
        return "frame_targets"
    if "frames" in row:
        return "frames"
    raise ValueError(f"{row.get('conversation_id', 'unknown')}: missing frames/frame_targets")


def _input_frame_id(frame: Mapping[str, Any], *, is_v5: bool) -> int:
    key = "input_frame_id" if is_v5 else "frame_id"
    value = frame.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {key}: {value!r}")
    if is_v5:
        if frame.get("frame_id", value) != value or frame.get("target_frame_id") != value + 1:
            raise ValueError("V5 input frame must supervise target input+1")
    return value


def _is_masked(frame: Mapping[str, Any], *, is_v5: bool) -> bool:
    value = frame.get("target_action_mask" if is_v5 else "action_mask")
    return value is not True


def _set_mask(frame: dict[str, Any], *, is_v5: bool, selected: bool, reason: str | None = None) -> None:
    frame["target_action_mask" if is_v5 else "action_mask"] = bool(selected)
    if reason is not None:
        frame["action_masked_reason"] = reason
    elif selected:
        frame.pop("action_masked_reason", None)


def _index_records(rows: Sequence[Mapping[str, Any]], *, split: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        conversation_id = str(row.get("conversation_id", ""))
        if not conversation_id or conversation_id in result:
            raise ValueError(f"{split}: missing or duplicate conversation_id {conversation_id!r}")
        result[conversation_id] = row
    return result


def _candidate_c_selection(rows: Sequence[Mapping[str, Any]], *, split: str) -> dict[str, dict[int, dict[str, Any]]]:
    selected: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        conversation_id = str(row.get("conversation_id", ""))
        input_id = row.get("input_frame_id", row.get("frame_id"))
        if not conversation_id or isinstance(input_id, bool) or not isinstance(input_id, int) or input_id < 0:
            raise ValueError(f"Candidate C {split}: invalid conversation/frame identity")
        if type(row.get("action_mask")) is not bool:
            raise ValueError(f"Candidate C {split}: action_mask must be boolean")
        if input_id in selected[conversation_id]:
            raise ValueError(f"Candidate C {split}: duplicate target {conversation_id}:{input_id}")
        target_id = row.get("target_frame_id", input_id)
        expected_target = input_id + 1 if row.get("profile") == "audio_grounded_v5" else input_id
        if target_id != expected_target:
            raise ValueError(f"Candidate C {split}: target-frame alignment mismatch at {conversation_id}:{input_id}")
        if row["action_mask"] and row.get("action") not in ACTIONS:
            raise ValueError(f"Candidate C {split}: supervised action is outside six-class ABI")
        selected[conversation_id][input_id] = dict(row)
    return dict(selected)


def _apply_candidate_c_selection(
    raw: Mapping[str, Any],
    selected: Mapping[int, Mapping[str, Any]],
    *,
    split: str,
) -> tuple[dict[str, Any], list[int]]:
    row = json.loads(json.dumps(raw))
    is_v5 = _target_key(row) == "frame_targets"
    key = _target_key(row)
    input_ids: set[int] = set()
    selected_ids: list[int] = []
    for frame in row[key]:
        input_id = _input_frame_id(frame, is_v5=is_v5)
        if input_id in input_ids:
            raise ValueError(f"{split} {row['conversation_id']}: duplicate input frame {input_id}")
        input_ids.add(input_id)
        c_row = selected.get(input_id)
        if c_row is None:
            _set_mask(frame, is_v5=is_v5, selected=False, reason="candidate_e_not_in_candidate_c_target_pool")
            continue
        c_action = c_row.get("action")
        source_action = frame.get("action") if is_v5 else frame.get("target_action")
        if c_row.get("action_mask") is True:
            if source_action != c_action:
                raise ValueError(
                    f"Candidate C/source action mismatch at {row['conversation_id']}:{input_id}"
                )
            _set_mask(frame, is_v5=is_v5, selected=True)
            selected_ids.append(input_id)
        else:
            _set_mask(
                frame,
                is_v5=is_v5,
                selected=False,
                reason=str(c_row.get("action_masked_reason") or "candidate_c_masked_target"),
            )
    missing = set(selected) - input_ids
    if missing:
        raise ValueError(f"Candidate C selected input frames absent from source {row['conversation_id']}: {sorted(missing)[:5]}")
    if not selected_ids:
        raise ValueError(f"Candidate C source row has no supervised feature targets: {row['conversation_id']}")
    if str(row.get("split")) != split:
        raise ValueError(f"Candidate C source split mismatch for {row['conversation_id']}")
    return row, selected_ids


def _yield_variant(
    raw: Mapping[str, Any],
    *,
    source_group_id: str,
) -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    row = json.loads(json.dumps(raw))
    if row.get("split") not in {"train", "validation"}:
        raise ValueError("test-set YIELD variants are not allowed in Candidate E")
    if _target_key(row) != "frame_targets":
        raise ValueError("YIELD remix must use causal V5 frame_targets")
    delay = row.get("yield_cut_delay_ms")
    if isinstance(delay, bool) or not isinstance(delay, int) or delay not in YIELD_DELAYS_MS:
        raise ValueError(f"unsupported YIELD remix delay {delay!r}")
    events = [event for event in row.get("event_schedule", ()) if event.get("action") == "YIELD"]
    if len(events) != 1:
        raise ValueError(f"{row['conversation_id']}: expected one YIELD event, found {len(events)}")
    event = events[0]
    cut = event.get("assistant_cut_sample")
    correction = event.get("correction_onset_sample")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (cut, correction)):
        raise ValueError(f"{row['conversation_id']}: missing sample-accurate correction/cut")
    if cut % FRAME_SAMPLES or correction < 0 or cut <= correction:
        raise ValueError(f"{row['conversation_id']}: correction/cut is not causal or frame-aligned")
    if (cut - correction) * 1000 != delay * 16000:
        raise ValueError(f"{row['conversation_id']}: YIELD cut delay disagrees with metadata")
    cut_tick = cut // FRAME_SAMPLES
    input_ids = (cut_tick - 1, cut_tick + 1, cut_tick + 3)
    expected_actions = ("YIELD", "LISTEN", "LISTEN")
    frames = row["frame_targets"]
    by_input = {int(frame["input_frame_id"]): frame for frame in frames}
    if len(by_input) != len(frames):
        raise ValueError(f"{row['conversation_id']}: duplicate V5 input frame")
    selected: list[int] = []
    for input_id, expected_action in zip(input_ids, expected_actions):
        frame = by_input.get(input_id)
        if frame is None:
            raise ValueError(f"{row['conversation_id']}: missing selected input frame {input_id}")
        if frame.get("target_frame_id") != input_id + 1 or frame.get("action") != expected_action:
            raise ValueError(f"{row['conversation_id']}:{input_id}: expected t->t+1 {expected_action}")
        if frame.get("target_action_mask") is not True:
            raise ValueError(f"{row['conversation_id']}:{input_id}: selected target is masked")
        if frame.get("input_text_visible") is not False or frame.get("future_audio_excluded") is not True:
            raise ValueError(f"{row['conversation_id']}:{input_id}: target violates audio-only causal contract")
        if frame.get("input_audio_end_sample") != (input_id + 1) * FRAME_SAMPLES:
            raise ValueError(f"{row['conversation_id']}:{input_id}: input cutoff is not causal")
        if frame.get("target_action_start_sample") != (input_id + 1) * FRAME_SAMPLES:
            raise ValueError(f"{row['conversation_id']}:{input_id}: target interval is not t+1")
        if frame.get("target_action_end_sample") != (input_id + 2) * FRAME_SAMPLES:
            raise ValueError(f"{row['conversation_id']}:{input_id}: target interval end is not one frame")
        if expected_action == "YIELD" and frame["target_action_start_sample"] != cut:
            raise ValueError(f"{row['conversation_id']}: YIELD target must start at the physical assistant cut")
        if expected_action == "LISTEN" and frame["target_action_start_sample"] <= cut:
            raise ValueError(f"{row['conversation_id']}:{input_id}: LISTEN target must follow assistant cut")
        selected.append(input_id)

    # Every target interval that includes any correction speech before the cut
    # is ambiguous for listener action. Keep its source action but mask it.
    for frame in frames:
        start = int(frame["target_action_start_sample"])
        end = int(frame["target_action_end_sample"])
        if start < cut and end > correction and int(frame["input_frame_id"]) not in selected:
            if frame.get("target_action_mask") is True:
                raise ValueError(
                    f"{row['conversation_id']}:{frame['input_frame_id']}: preemption window has an unmasked target"
                )

    for frame in frames:
        input_id = int(frame["input_frame_id"])
        if input_id in selected:
            frame["target_action_mask"] = True
            frame.pop("action_masked_reason", None)
        else:
            frame["target_action_mask"] = False
            if frame.get("action_masked_reason") is None:
                frame["action_masked_reason"] = "candidate_e_yield_variant_target_not_selected"
    row["source_group_id"] = source_group_id
    row["stage2_candidate_e_parent_conversation_id"] = str(row.get("parent_conversation_id", ""))
    row["stage2_candidate_e_target_input_ids"] = selected
    info = {
        "conversation_id": str(row["conversation_id"]),
        "parent_conversation_id": str(row.get("parent_conversation_id", "")),
        "source_group_id": source_group_id,
        "split": str(row["split"]),
        "yield_cut_delay_ms": delay,
        "cut_input_frame_id": selected[0],
        "selected_input_frame_ids": selected,
    }
    return row, selected, info


def _split_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    actions: Counter[str] = Counter()
    supervised = 0
    input_steps = 0
    max_frames = 0
    sources: Counter[str] = Counter()
    groups: set[str] = set()
    for row in rows:
        key = _target_key(row)
        is_v5 = key == "frame_targets"
        chosen = []
        for frame in row[key]:
            if _is_masked(frame, is_v5=is_v5):
                continue
            action = frame.get("action") if is_v5 else frame.get("target_action")
            if action not in ACTIONS:
                raise ValueError(f"{row['conversation_id']}: supervised action outside six-class ABI")
            chosen.append(_input_frame_id(frame, is_v5=is_v5))
            actions[str(action)] += 1
        if not chosen:
            raise ValueError(f"{row['conversation_id']}: no supervised Candidate E targets")
        replay_frames = max(chosen) + 1
        input_steps += replay_frames
        max_frames = max(max_frames, replay_frames)
        supervised += len(chosen)
        sources[str(row.get("source", "unknown"))] += len(chosen)
        groups.add(str(row.get("source_group_id") or row["conversation_id"]))
    return {
        "conversation_rows": len(rows),
        "source_groups": len(groups),
        "supervised_feature_targets": supervised,
        "supervised_targets_by_action": {action: actions[action] for action in ACTIONS},
        "targets_by_source": dict(sorted(sources.items())),
        "conversation_replays": len(rows),
        "causal_input_frame_steps_from_frame_zero": input_steps,
        "max_replay_input_frames": max_frames,
    }


def _prepare_smoke_row(row: Mapping[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(row))
    key = _target_key(clone)
    is_v5 = key == "frame_targets"
    supervised = [frame for frame in clone[key] if not _is_masked(frame, is_v5=is_v5)]
    chosen = min(supervised, key=lambda frame: _input_frame_id(frame, is_v5=is_v5))
    keep_id = _input_frame_id(chosen, is_v5=is_v5)
    for frame in clone[key]:
        frame_id = _input_frame_id(frame, is_v5=is_v5)
        if frame_id != keep_id:
            _set_mask(frame, is_v5=is_v5, selected=False, reason="candidate_e_bounded_probe_target_not_selected")
    clone["stage2_candidate_e_probe_target_input_id"] = keep_id
    return clone


def prepare_candidate_e(
    *,
    base_prepared_dir: Path,
    traincap_prepared_dir: Path,
    candidate_c_feature_dir: Path,
    yield_manifest: Path,
    output_dir: Path,
    smoke_output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    smoke_output_dir = smoke_output_dir.expanduser().resolve()
    if output_dir.exists() or smoke_output_dir.exists():
        raise FileExistsError("Candidate E output directory already exists; choose fresh paths")
    if output_dir == smoke_output_dir or output_dir in smoke_output_dir.parents or smoke_output_dir in output_dir.parents:
        raise ValueError("full Candidate E and smoke output directories must be separate")

    output_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    candidate_counts: dict[str, Any] = {}
    original_group_sets: dict[str, set[str]] = {"train": set(), "validation": set()}
    for split in ("train", "validation"):
        base_rows = _read_jsonl(base_prepared_dir / f"{split}.jsonl")
        v5_rows = _read_jsonl(traincap_prepared_dir / f"{split}.jsonl")
        natural = [row for row in base_rows if row.get("source") in {"SmoothConv", "DuplexConv"}]
        indexed = _index_records(natural, split=f"{split} natural")
        for row in v5_rows:
            if row.get("source") != "authored_synthetic_audio_candidate" or _target_key(row) != "frame_targets":
                raise ValueError(f"{split}: train-cap source must be causal V5 audio")
            conversation_id = str(row["conversation_id"])
            if conversation_id in indexed:
                raise ValueError(f"{split}: duplicate natural/V5 conversation_id {conversation_id}")
            indexed[conversation_id] = dict(row)

        feature_rows = json.loads((candidate_c_feature_dir / f"{split}.json").read_text(encoding="utf-8"))
        if not isinstance(feature_rows, list) or not feature_rows:
            raise ValueError(f"Candidate C {split} feature metadata must be a non-empty list")
        selected_by_conversation = _candidate_c_selection(feature_rows, split=split)
        missing = set(selected_by_conversation) - set(indexed)
        if missing:
            raise ValueError(f"Candidate C {split}: source conversations missing: {sorted(missing)[:5]}")
        masked_only_conversations = 0
        for conversation_id, selected in selected_by_conversation.items():
            source = indexed[conversation_id]
            source_is_v5 = _target_key(source) == "frame_targets"
            if source_is_v5 != (source.get("profile") == "audio_grounded_v5"):
                raise ValueError(f"{split} {conversation_id}: target format/profile mismatch")
            source_group = str(source.get("source_group_id") or conversation_id)
            feature_group = {
                str(row.get("source_group_id") or row.get("conversation_id"))
                for row in feature_rows
                if str(row.get("conversation_id")) == conversation_id
            }
            if feature_group != {source_group}:
                raise ValueError(f"{split} {conversation_id}: Candidate C source-group mismatch")
            if not any(row.get("action_mask") is True for row in selected.values()):
                masked_only_conversations += 1
                continue
            prepared, selected_ids = _apply_candidate_c_selection(source, selected, split=split)
            output_rows[split].append(prepared)
            original_group_sets[split].add(source_group)
        candidate_counts[split] = {
            "metadata_rows": len(feature_rows),
            "supervised_rows_kept": sum(bool(row.get("action_mask")) for row in feature_rows),
            "masked_rows_not_embedded": sum(not bool(row.get("action_mask")) for row in feature_rows),
            "selected_conversations": len(selected_by_conversation) - masked_only_conversations,
            "masked_only_conversations_not_embedded": masked_only_conversations,
        }

    remix_rows = _read_jsonl(yield_manifest)
    remix_counts = Counter(str(row.get("split")) for row in remix_rows)
    if set(remix_counts) - {"train", "validation"}:
        raise ValueError("YIELD remix manifest must not contain test variants")
    if remix_counts != Counter({"train": 96, "validation": 24}):
        raise ValueError(f"unexpected YIELD remix split counts: {dict(remix_counts)}")

    parent_rows = _index_records(
        [row for split in ("train", "validation") for row in _read_jsonl(traincap_prepared_dir / f"{split}.jsonl")],
        split="V5 parent",
    )
    variant_infos: list[dict[str, Any]] = []
    seen_ids = {str(row["conversation_id"]) for rows in output_rows.values() for row in rows}
    for raw in remix_rows:
        parent_id = str(raw.get("parent_conversation_id", ""))
        parent = parent_rows.get(parent_id)
        if parent is None:
            raise ValueError(f"YIELD variant parent is absent from canonical V5 data: {parent_id}")
        if raw.get("split") != parent.get("split"):
            raise ValueError(f"YIELD variant split differs from parent {parent_id}")
        conversation_id = str(raw.get("conversation_id", ""))
        if not conversation_id or conversation_id in seen_ids:
            raise ValueError(f"duplicate Candidate E conversation_id: {conversation_id!r}")
        seen_ids.add(conversation_id)
        source_group = str(parent.get("source_group_id") or parent.get("topic_group_id") or parent_id)
        variant, selected_ids, info = _yield_variant(raw, source_group_id=source_group)
        output_rows[str(raw["split"])].append(variant)
        variant_infos.append(info)

    group_overlap = original_group_sets["train"] & original_group_sets["validation"]
    if group_overlap:
        raise ValueError(f"source groups leak between train and validation: {sorted(group_overlap)[:5]}")
    for split in ("train", "validation"):
        split_rows = output_rows[split]
        split_groups = {str(row.get("source_group_id") or row["conversation_id"]) for row in split_rows}
        if split == "train":
            original_group_sets[split] = split_groups
        elif split_groups & original_group_sets["train"]:
            raise ValueError("Candidate E train/validation source-group leakage after adding YIELD variants")
        if any(str(row.get("split")) != split for row in split_rows):
            raise ValueError(f"Candidate E {split}: mixed split rows")

    train_ids = {str(row["conversation_id"]) for row in output_rows["train"]}
    val_ids = {str(row["conversation_id"]) for row in output_rows["validation"]}
    if train_ids & val_ids:
        raise ValueError("Candidate E conversation IDs leak across splits")
    report = {
        "schema_version": 1,
        "purpose": "candidate_e_full_prefix_native_listener_feature_extraction",
        "candidate_c_feature_metadata": str(candidate_c_feature_dir.resolve()),
        "canonical_prepared_dir": str(base_prepared_dir.resolve()),
        "v5_prepared_dir": str(traincap_prepared_dir.resolve()),
        "yield_remix_manifest": str(yield_manifest.resolve()),
        "masked_rows": "preserved in source metadata but not embedded; all intermediate input frames are still replayed causally",
        "target_alignment": {
            "natural": "same-index legacy source target",
            "audio_grounded_v5": "input_frame_id=t predicts target_frame_id=t+1",
            "yield_variant_targets": "one YIELD at input cut_tick-1 and LISTEN at cut_tick+1 and cut_tick+3",
            "replay": "one production listener state per conversation from absolute frame 0 through last selected input",
            "front_end_window_frames": 64,
            "native_kv_capacity": "derived from the loaded production listener at extraction time",
        },
        "candidate_c_selection": candidate_counts,
        "yield_variants": {
            "variant_count": len(variant_infos),
            "variant_counts_by_split": dict(sorted(remix_counts.items())),
            "target_counts_by_action": {"YIELD": len(variant_infos), "LISTEN": 2 * len(variant_infos)},
            "delay_counts_by_split_and_ms": {
                split: dict(sorted(Counter(int(row["yield_cut_delay_ms"]) for row in variant_infos if row["split"] == split).items()))
                for split in ("train", "validation")
            },
            "source_groups_reused_from_parent": True,
            "test_variants_included": False,
        },
        "splits": {split: _split_summary(output_rows[split]) for split in ("train", "validation")},
        "group_disjoint": True,
        "conversation_ids_disjoint": True,
        "limitations": [
            "Generated synthetic YIELD audio provides candidate supervision, not human-validated behavior evidence.",
            "Validation YIELD variants reuse the eight existing held-out V5 conversation groups and are not independent groups.",
        ],
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    smoke_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    smoke_staging = Path(tempfile.mkdtemp(prefix=f".{smoke_output_dir.name}.tmp-", dir=smoke_output_dir.parent))
    try:
        for split in ("train", "validation"):
            _write_jsonl(staging / f"{split}.jsonl", output_rows[split])
        (staging / "candidate_e_manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        first_real = next(
            row for row in output_rows["train"]
            if row.get("source") in {"SmoothConv", "DuplexConv"} and not bool(row.get("synthetic"))
        )
        smoke_row = _prepare_smoke_row(first_real)
        _write_jsonl(smoke_staging / "train.jsonl", [smoke_row])
        smoke_report = {
            "schema_version": 1,
            "purpose": "one_conversation_one_supervised_target_production_listener_probe",
            "conversation_id": str(smoke_row["conversation_id"]),
            "source_group_id": str(smoke_row.get("source_group_id") or smoke_row["conversation_id"]),
            "target_input_frame_id": int(smoke_row["stage2_candidate_e_probe_target_input_id"]),
            "target_action": next(
                frame.get("target_action")
                for frame in smoke_row["frames"]
                if frame.get("frame_id") == smoke_row["stage2_candidate_e_probe_target_input_id"]
            ),
            "actual_input_frames_to_replay": int(smoke_row["stage2_candidate_e_probe_target_input_id"]) + 1,
            "separate_from_full_candidate_e": True,
        }
        (smoke_staging / "bounded_probe_manifest.json").write_text(
            json.dumps(smoke_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
        os.replace(smoke_staging, smoke_output_dir)
    except Exception:
        # Keep staging artifacts for diagnosis; never remove existing user data.
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prepared-dir", type=Path, required=True)
    parser.add_argument("--traincap-prepared-dir", type=Path, required=True)
    parser.add_argument("--candidate-c-feature-dir", type=Path, required=True)
    parser.add_argument("--yield-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_candidate_e(
        base_prepared_dir=args.base_prepared_dir,
        traincap_prepared_dir=args.traincap_prepared_dir,
        candidate_c_feature_dir=args.candidate_c_feature_dir,
        yield_manifest=args.yield_manifest,
        output_dir=args.output_dir,
        smoke_output_dir=args.smoke_output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
