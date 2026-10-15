#!/usr/bin/env python3
"""Combine native conversational views with synthetic audio-grounded targets.

Each source's split/group assignment is preserved. Synthetic input eligibility
comes from readable audio, causal frame targets, and explicit loss masks; source
and profile strings remain provenance metadata.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import copy
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.duplex_qlora.schema import ACTIONS


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def _frame_rows(row: dict[str, Any]):
    return row.get("frame_targets", ()) or row.get("frames", ()) or ()


def _frame_action(frame: dict[str, Any]) -> str | None:
    return frame.get("action") if "target_action_mask" in frame else frame.get("target_action")


def _frame_mask(frame: dict[str, Any]) -> bool:
    return frame.get("target_action_mask") is True if "target_action_mask" in frame else frame.get("action_mask") is True


def _actions(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for frame in _frame_rows(row):
            action = _frame_action(frame)
            if _frame_mask(frame) and action in ACTIONS:
                counts[str(action)] += 1
    return counts


def _cap_labeled_frames(
    rows: list[dict[str, Any]],
    *,
    cap_per_class: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select frames class-first, source-balanced, then round-robin by group."""
    candidates: dict[str, dict[str, dict[str, list[tuple[int, int]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row_index, row in enumerate(rows):
        source = str(row.get("source", "unknown"))
        group = str(row.get("source_group_id") or row.get("conversation_id"))
        for frame_index, frame in enumerate(_frame_rows(row)):
            action = _frame_action(frame)
            if _frame_mask(frame) and action in ACTIONS:
                candidates[str(action)][source][group].append((row_index, frame_index))

    selected: set[tuple[int, int]] = set()
    report: dict[str, Any] = {
        "cap_per_class": cap_per_class,
        "candidate_frames_by_action": {},
        "selected_frames_by_action": {},
        "selected_frames_by_source_action": {},
        "selected_groups_by_source_action": {},
    }
    for action in ACTIONS:
        source_groups = candidates.get(action, {})
        candidate_count = sum(
            len(frame_refs)
            for groups in source_groups.values()
            for frame_refs in groups.values()
        )
        report["candidate_frames_by_action"][action] = candidate_count
        if cap_per_class <= 0 or candidate_count <= cap_per_class:
            chosen = [
                ref
                for groups in source_groups.values()
                for frame_refs in groups.values()
                for ref in frame_refs
            ]
        else:
            source_queues: dict[str, list[tuple[int, int]]] = {}
            for source, groups in sorted(source_groups.items()):
                ordered_groups = []
                for group_id, refs in sorted(groups.items()):
                    ordered_groups.append(
                        sorted(
                            refs,
                            key=lambda ref: (
                                str(rows[ref[0]].get("conversation_id", "")),
                                int(_frame_rows(rows[ref[0]])[ref[1]].get(
                                    "input_frame_id",
                                    _frame_rows(rows[ref[0]])[ref[1]].get("frame_id", -1),
                                )),
                            ),
                        )
                    )
                queue: list[tuple[int, int]] = []
                for offset in range(max((len(group) for group in ordered_groups), default=0)):
                    for group_refs in ordered_groups:
                        if offset < len(group_refs):
                            queue.append(group_refs[offset])
                source_queues[source] = queue
            chosen = []
            source_order = sorted(source_queues)
            while len(chosen) < cap_per_class and source_order:
                remaining_sources = []
                for source in source_order:
                    queue = source_queues[source]
                    if queue and len(chosen) < cap_per_class:
                        chosen.append(queue.pop(0))
                    if queue:
                        remaining_sources.append(source)
                source_order = remaining_sources
        selected.update(chosen)
        report["selected_frames_by_action"][action] = len(chosen)
        source_counts: Counter[str] = Counter(str(rows[row_index].get("source", "unknown")) for row_index, _ in chosen)
        groups_by_source: dict[str, set[str]] = defaultdict(set)
        for row_index, _ in chosen:
            source = str(rows[row_index].get("source", "unknown"))
            groups_by_source[source].add(
                str(rows[row_index].get("source_group_id") or rows[row_index].get("conversation_id"))
            )
        report["selected_frames_by_source_action"][action] = dict(source_counts)
        report["selected_groups_by_source_action"][action] = {
            source: len(groups) for source, groups in sorted(groups_by_source.items())
        }

    filtered_rows = []
    for row_index, row in enumerate(rows):
        clone = copy.deepcopy(row)
        frame_key = "frame_targets" if "frame_targets" in row else "frames"
        clone[frame_key] = [
            frame
            for frame_index, frame in enumerate(_frame_rows(row))
            if (row_index, frame_index) in selected
            or (frame_key == "frame_targets" and not _frame_mask(frame))
        ]
        filtered_rows.append(clone)
    return filtered_rows, report


def assemble(
    *,
    real_prepared_dir: Path,
    synthetic_prepared_dir: Path,
    output_dir: Path,
    train_cap_per_class: int = 128,
    validation_cap_per_class: int = 32,
    test_cap_per_class: int = 0,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output Stage2 esistente, non sovrascritto: {output_dir}")
    output_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "native Control6 features; real conversational dynamics plus synthetic Italian YIELD/REPLAN",
        "real_prepared_dir": str(real_prepared_dir),
        "synthetic_prepared_dir": str(synthetic_prepared_dir),
        "synthetic_filter": "synthetic=true with functional audio/timeline/mask contract; source/version tags are provenance only",
        "splits": {},
    }
    group_split: dict[str, str] = {}
    group_sources: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "validation", "test"):
        real_rows = _read_rows(real_prepared_dir / f"{split}.jsonl")
        synthetic_rows = [
            row for row in _read_rows(synthetic_prepared_dir / f"{split}.jsonl")
            if row.get("synthetic") is True
        ]
        all_rows = real_rows + synthetic_rows
        for row in all_rows:
            conversation_id = str(row.get("conversation_id") or "")
            group_id = str(row.get("source_group_id") or conversation_id)
            if not conversation_id or not group_id:
                raise ValueError(f"{split}: record senza conversation_id/source_group_id")
            previous = group_split.setdefault(group_id, split)
            if previous != split:
                raise ValueError(f"source group {group_id} compare in split {previous} e {split}")
            group_sources[group_id].add(str(row.get("source", "unknown")))
            if not row.get("audio_path"):
                raise ValueError(f"{conversation_id}: audio_path mancante")
            if not any(_frame_mask(frame) and _frame_action(frame) in ACTIONS for frame in _frame_rows(row)):
                raise ValueError(f"{conversation_id}: nessuna label Control6 utilizzabile")
        cap = {
            "train": train_cap_per_class,
            "validation": validation_cap_per_class,
            "test": test_cap_per_class,
        }[split]
        all_rows, selection = _cap_labeled_frames(all_rows, cap_per_class=cap)
        group_ids = [str(row.get("source_group_id") or row["conversation_id"]) for row in all_rows]
        # Multiple rows per source_group_id are intentional: paired real views
        # and synthetic episodes linked to the same scenario must stay together
        # under their original physical/group split.
        output_path = output_dir / f"{split}.jsonl"
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in all_rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        source_rows = Counter(str(row.get("source", "unknown")) for row in all_rows)
        report["splits"][split] = {
            "rows": len(all_rows),
            "real_rows": len(real_rows),
            "synthetic_italian_rows": len(synthetic_rows),
            "unique_source_groups": len(set(group_ids)),
            "source_rows": dict(source_rows),
            "supervised_frames_by_action": dict(sorted(_actions(all_rows).items())),
            "supervised_frames": sum(_actions(all_rows).values()),
            "frame_selection": selection,
        }
    cross_source_groups = {
        group: sorted(sources) for group, sources in group_sources.items() if len(sources) > 1
    }
    if cross_source_groups:
        raise ValueError(f"source_group_id collisions across corpora: {list(cross_source_groups)[:5]}")
    report["source_group_count"] = len(group_split)
    report["group_split_count"] = dict(sorted(Counter(group_split.values()).items()))
    manifest_path = output_dir / "stage2_input_manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-prepared-dir", type=Path, required=True)
    parser.add_argument("--synthetic-prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-cap-per-class", type=int, default=128)
    parser.add_argument("--validation-cap-per-class", type=int, default=32)
    parser.add_argument("--test-cap-per-class", type=int, default=0)
    args = parser.parse_args()
    report = assemble(
        real_prepared_dir=args.real_prepared_dir,
        synthetic_prepared_dir=args.synthetic_prepared_dir,
        output_dir=args.output_dir,
        train_cap_per_class=args.train_cap_per_class,
        validation_cap_per_class=args.validation_cap_per_class,
        test_cap_per_class=args.test_cap_per_class,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "source_group_count": report["source_group_count"],
        "splits": report["splits"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
