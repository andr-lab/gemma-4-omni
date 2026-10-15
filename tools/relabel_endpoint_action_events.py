"""Copy stage-2 features and retain event labels only at causal onset ticks."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.duplex_qlora.event_label_correction import (
    EVENT_ACTIONS,
    assistant_active_in_previous_tick,
    event_run_offset,
    relabel_event_action,
    verified_assistant_segments,
)
from training.duplex_qlora.schema import ACTIONS


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _frame_segment_ids(row: dict[str, Any]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for frame in row.get("frames", ()):
        action = frame.get("target_action")
        if action not in EVENT_ACTIONS or frame.get("action_mask") is not True:
            continue
        start = _number(frame.get("start_ms"))
        end = _number(frame.get("end_ms"))
        if start is None or end is None or end <= start:
            continue
        segments, unsafe = verified_assistant_segments(
            row.get("segments", ()),
            row.get("channel_map", {}),
            interval_start_ms=start,
            interval_end_ms=end,
        )
        if unsafe:
            continue
        candidates = [
            segment
            for segment in segments
            if float(segment["start_ms"]) < end and float(segment["end_ms"]) > start
        ]
        if action == "BACKCHANNEL" and not bool(row.get("synthetic")):
            candidates = [
                segment
                for segment in candidates
                if segment.get("native_turn_state") == "backchannel"
            ]
        if not candidates:
            continue
        segment = min(
            candidates,
            key=lambda item: (
                float(item["end_ms"]) - float(item["start_ms"]),
                abs(float(item["start_ms"]) - start),
            ),
        )
        identity = segment.get("utterance_id") or segment.get("segment_id")
        if identity is not None:
            mapping[int(frame["frame_id"])] = str(identity)
    return mapping


def _read_prepared_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            conversation_id = row.get("conversation_id")
            if not isinstance(conversation_id, str) or not conversation_id:
                raise ValueError(f"prepared row privo di conversation_id in {path}")
            if conversation_id in rows:
                raise ValueError(f"conversation_id duplicato nel prepared split: {conversation_id}")
            rows[conversation_id] = row
    return rows


def _correct_split(
    *,
    split: str,
    output: Path,
    prepared_path: Path,
    actions: tuple[str, ...] = ACTIONS,
) -> dict[str, Any]:
    metadata_path = output / f"{split}.json"
    feature_path = output / f"{split}.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prepared_rows = _read_prepared_rows(prepared_path)
    with np.load(feature_path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    labels = arrays["labels"].astype(np.int64, copy=True)
    frame_ids = arrays["frame_ids"]
    if len(metadata) != len(labels) or frame_ids.shape != labels.shape:
        raise ValueError(f"{split}: metadata, labels e frame IDs non allineati")

    before = Counter()
    after = Counter()
    correction_counts: Counter[str] = Counter()
    offsets: dict[str, Counter[int]] = defaultdict(Counter)
    reasons: Counter[str] = Counter()
    changed = 0

    for index, item in enumerate(metadata):
        old_action = item.get("action") if item.get("action_mask") else None
        old_label = int(labels[index])
        if old_action in actions and old_label != actions.index(old_action):
            raise ValueError(f"{split}: feature action e label NPZ divergono all'indice {index}")
        if old_action in actions:
            before[old_action] += 1
        if old_action not in EVENT_ACTIONS:
            if old_action in actions:
                after[old_action] += 1
            continue

        conversation = prepared_rows.get(str(item.get("conversation_id")))
        if conversation is None:
            offset = None
            active = None
        else:
            frame_id = int(frame_ids[index])
            frame_by_id = {
                int(frame["frame_id"]): frame
                for frame in conversation.get("frames", ())
                if frame.get("frame_id") is not None
            }
            frame = frame_by_id.get(frame_id)
            segments = conversation.get("segments", ())
            channel_map = conversation.get("channel_map", {})
            segment_ids = _frame_segment_ids(conversation)
            offset = event_run_offset(
                conversation.get("frames", ()),
                frame_id=frame_id,
                action=str(old_action),
                segment_ids=segment_ids,
            )
            start_ms = _number(frame.get("start_ms")) if frame is not None else None
            tick_ms = _number(conversation.get("tick_ms"))
            if start_ms is None or tick_ms is None:
                active = None
            else:
                active = assistant_active_in_previous_tick(
                    frame_start_ms=start_ms,
                    tick_ms=tick_ms,
                    segments=segments,
                    channel_map=channel_map,
                )

        if offset is not None:
            offsets[str(old_action)][offset] += 1
        new_action, new_mask, reason = relabel_event_action(
            str(old_action), offset_ticks=offset, assistant_prev_active=active
        )
        reasons[reason] += 1
        item["event_label_correction"] = {
            "from_action": old_action,
            "to_action": new_action,
            "event_offset_ticks": offset,
            "assistant_previous_tick_active": active,
            "reason": reason,
        }
        item["action"] = new_action
        item["action_mask"] = bool(new_mask)
        new_label = actions.index(new_action) if new_mask and new_action in actions else -100
        labels[index] = new_label
        if new_mask and new_action in actions:
            after[new_action] += 1
        if new_action != old_action or not new_mask:
            changed += 1

    arrays["labels"] = labels.astype(arrays["labels"].dtype, copy=False)
    np.savez_compressed(feature_path, **arrays)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "examples": len(metadata),
        "labeled_before": int(sum(before.values())),
        "labeled_after": int(sum(after.values())),
        "masked_after": int(len(metadata) - sum(after.values())),
        "action_support_before": {action: int(before.get(action, 0)) for action in actions},
        "action_support_after": {action: int(after.get(action, 0)) for action in actions},
        "event_offsets_by_action": {
            action: {str(offset): int(count) for offset, count in sorted(offsets.get(action, {}).items())}
            for action in sorted(EVENT_ACTIONS)
        },
        "changes_by_reason": dict(sorted(reasons.items())),
        "changed_targets": changed,
    }


def relabel_feature_root(
    *,
    source_feature_dir: Path,
    prepared_dir: Path,
    output_feature_dir: Path,
) -> dict[str, Any]:
    """Copy feature artifacts to a fresh directory and correct only labels."""
    source_feature_dir = source_feature_dir.resolve()
    prepared_dir = prepared_dir.resolve()
    output_feature_dir = output_feature_dir.resolve()
    if output_feature_dir.exists():
        raise FileExistsError(f"il nuovo feature root esiste già: {output_feature_dir}")
    if not (source_feature_dir / "feature_manifest.json").is_file():
        raise FileNotFoundError(source_feature_dir / "feature_manifest.json")
    temp_output = output_feature_dir.with_name(output_feature_dir.name + ".tmp")
    if temp_output.exists():
        raise FileExistsError(f"copia temporanea già esistente: {temp_output}")
    shutil.copytree(source_feature_dir, temp_output)
    try:
        split_reports = {
            split: _correct_split(
                split=split,
                output=temp_output,
                prepared_path=prepared_dir / f"{split}.jsonl",
            )
            for split in ("train", "validation", "test")
        }
        manifest_path = temp_output / "feature_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("feature_contract", {}).get("input_dim") != 1536:
            raise ValueError("feature contract width inattesa")
        for split, summary in split_reports.items():
            manifest["counts"][split]["labeled_frames"] = summary["labeled_after"]
            manifest["counts"][split]["action_support_after_correction"] = summary[
                "action_support_after"
            ]
        manifest["event_label_correction"] = {
            "schema_version": 1,
            "correction_id": "endpoint_event_onsets_causal_v1",
            "source_feature_dir": str(source_feature_dir),
            "prepared_timing_dir": str(prepared_dir),
            "native_query_features_modified": False,
            "original_feature_root_preserved": True,
            "rules": {
                "BACKCHANNEL_REPLAN": "onset only; interior becomes CONTINUE iff verified assistant audio overlaps the preceding tick, otherwise masked",
                "YIELD": "onset only; interior becomes LISTEN iff verified assistant audio is inactive in the preceding tick, otherwise masked",
                "assistant_activity_window": "[target_tick_start - tick_ms, target_tick_start); channel ownership and time_mask must be verified",
                "event_onset": "first contiguous action-labeled frame in the same role-owned assistant utterance segment",
                "unknown_source_ownership_or_timing": "mask target",
            },
            "splits": split_reports,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temp_output.rename(output_feature_dir)
    except Exception:
        # Keep the intermediate folder for inspection after a failed correction.
        raise
    return {
        "status": "completed",
        "feature_root": str(output_feature_dir),
        "correction": split_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-feature-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-feature-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            relabel_feature_root(
                source_feature_dir=args.source_feature_dir,
                prepared_dir=args.prepared_dir,
                output_feature_dir=args.output_feature_dir,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
