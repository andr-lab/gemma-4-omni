#!/usr/bin/env python3
"""Validate rendered audio-grounded records and materialize Stage2 split JSONL."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.duplex_qlora.audio_grounded_v5 import validate_audio_grounded_v5_record
from training.duplex_qlora.schema import ACTIONS


def prepare(
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest = Path(manifest_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"rendered audio manifest is missing: {manifest}")
    if output.exists():
        raise FileExistsError(f"Stage2 v5 output already exists, refusing overwrite: {output}")
    records: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{manifest}:{line_number}: expected a JSON object")
            audio_value = raw.get("audio_path")
            audio_path = Path(audio_value).expanduser() if isinstance(audio_value, str) else None
            audio_base = manifest.parent
            if audio_path is not None and not audio_path.is_absolute():
                audio_base = next(
                    (
                        base
                        for base in (manifest.parent, Path.cwd())
                        if (base / audio_path).is_file()
                    ),
                    manifest.parent,
                )
            records.append(validate_audio_grounded_v5_record(raw, base_dir=audio_base))
    if not records:
        raise ValueError(f"rendered audio manifest is empty: {manifest}")

    seen_conversations: set[str] = set()
    groups_by_split: dict[str, set[str]] = {split: set() for split in ("train", "validation", "test")}
    audio_by_split: dict[str, set[str]] = {split: set() for split in ("train", "validation", "test")}
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in groups_by_split}
    for row in records:
        conversation_id = row["conversation_id"]
        if conversation_id in seen_conversations:
            raise ValueError(f"duplicate conversation_id in audio manifest: {conversation_id}")
        seen_conversations.add(conversation_id)
        split = row["split"]
        groups_by_split[split].add(row["source_group_id"])
        audio_by_split[split].add(row["audio_sha256"])
        rows_by_split[split].append(row)
    split_names = list(rows_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                raise ValueError(f"source groups leak across {left}/{right}: {sorted(overlap)[:5]}")
            audio_overlap = audio_by_split[left] & audio_by_split[right]
            if audio_overlap:
                raise ValueError(f"rendered audio leaks across {left}/{right}: {sorted(audio_overlap)[:3]}")

    counts = {split: len(rows) for split, rows in rows_by_split.items()}
    class_support: dict[str, dict[str, int]] = {}
    absent_by_split: dict[str, list[str]] = {}
    for split, rows in rows_by_split.items():
        support = Counter(
            str(frame["action"])
            for row in rows
            for frame in row["frame_targets"]
            if frame["target_action_mask"]
        )
        class_support[split] = {action: int(support[action]) for action in ACTIONS}
        absent_by_split[split] = [action for action in ACTIONS if not support[action]]

    output.mkdir(parents=True)
    split_report: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        path = output / f"{split}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        split_report[split] = {
            "rows": len(rows),
            "source_groups": len(groups_by_split[split]),
            "supervised_frames_by_action": class_support[split],
        }
    report = {
        "schema_version": 1,
        "status": "audio_grounded_v5_listener_candidate_ready",
        "source_manifest": str(manifest),
        "source_version_filter": None,
        "functional_contract": [
            "readable stereo 16 kHz PCM with user channel 0 and assistant channel 1",
            "80 ms input frame t supervises target frame t+1 on exact sample boundaries",
            "audio-only input cutoff, with future target PCM excluded",
            "ambiguous targets remain masked from loss",
            "event_schedule references declared utterance IDs",
            "conversation, source-group, and audio hashes are disjoint across splits",
        ],
        "split_counts": counts,
        "splits": split_report,
        "absent_classes_by_split": absent_by_split,
        "feature_abi": {
            "listener": "native Gemma production listener",
            "queries_per_tick": 4,
            "selected_query_index": 3,
            "hidden_size": 1536,
        },
        "warning": "Candidate ingest only; generated audio labels are not human-validated behavior evidence.",
    }
    (output / "listener_v5_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.manifest, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
