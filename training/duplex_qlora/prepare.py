"""Create deterministic split JSONL files for the two training stages."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .schema import sha256_file, split_conversations


def _synthetic_target_language_issue(
    *, template_family: str, script_role: str, text: str
) -> str | None:
    """Identify the known v4 template agreement errors at utterance granularity."""
    normalized = " ".join(text.casefold().split())
    if template_family == "lesson_delivery" and script_role == "assistant_replan":
        if re.search(r"\bda\s+(?:il|lo|la|i|gli|le)\b", normalized):
            return "uncontracted_preposition_article"
    return None


def _mask_known_synthetic_target_errors(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Mask only the affected assistant utterance targets; preserve context text/audio."""
    masked: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for conversation in rows:
        row = dict(conversation)
        segments = row.get("segments", ())
        copied_segments = []
        for original in segments if isinstance(segments, (list, tuple)) else ():
            if not isinstance(original, dict):
                copied_segments.append(original)
                continue
            segment = dict(original)
            if row.get("synthetic") is True and segment.get("role") == "assistant":
                text = segment.get("text", segment.get("script_text", segment.get("transcript", "")))
                if isinstance(text, str):
                    reason = _synthetic_target_language_issue(
                        template_family=str(row.get("template_family_id", "")),
                        script_role=str(segment.get("script_role", "")),
                        text=text,
                    )
                    if reason:
                        segment["target_mask"] = False
                        segment["target_mask_reason"] = reason
                        masked[reason] += 1
                    else:
                        segment.setdefault("target_mask", True)
            copied_segments.append(segment)
        row["segments"] = copied_segments
        output.append(row)
    return output, masked


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_splits(
    conversations: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    seed: int = 20260930,
) -> dict[str, Any]:
    """Assign connected conversations once and emit auditable counts."""
    rows, target_masks = _mask_known_synthetic_target_errors(list(conversations))
    assignments = split_conversations(rows, seed=seed)
    ids = [str(row["conversation_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("conversation_id duplicati prima del prepare")
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    owner: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        keys = [
            ("source_group", str(row.get("source_group_id") or row["conversation_id"])),
            ("conversation", str(row["conversation_id"])),
        ]
        digest = row.get("audio_sha256")
        audio_path = row.get("audio_path")
        if digest:
            keys.append(("audio_sha256", str(digest).lower()))
        elif audio_path and Path(str(audio_path)).is_file():
            keys.append(("audio_sha256", sha256_file(str(audio_path))))
        elif audio_path:
            keys.append(("audio_path", str(Path(str(audio_path)).resolve()).lower()))
        for key in keys:
            previous = owner.setdefault(key, index)
            union(index, previous)
    components: dict[int, list[int]] = {}
    for index in range(len(rows)):
        components.setdefault(find(index), []).append(index)

    by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "validation", "test")}
    quarantined: list[dict[str, Any]] = []
    preserved_rows = 0
    quarantined_groups = 0
    for component in components.values():
        declared = {
            str(rows[index].get("split"))
            for index in component
            if rows[index].get("split") in {"train", "validation", "test"}
        }
        if len(declared) > 1:
            quarantined_groups += 1
            conflicts = sorted(declared)
            for index in component:
                quarantined.append({**rows[index], "split": "quarantine", "split_conflict": conflicts})
            continue
        split = next(iter(declared), assignments[str(rows[component[0]]["conversation_id"])])
        if declared:
            preserved_rows += sum(rows[index].get("split") in {"train", "validation", "test"} for index in component)
        for index in component:
            by_split[split].append({**rows[index], "split": split})

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in by_split.items():
        _write_jsonl(output / f"{name}.jsonl", rows)
    _write_jsonl(output / "quarantine.jsonl", quarantined)
    prepared_rows = [row for split_rows in by_split.values() for row in split_rows]
    counts = {
        "schema_version": 1,
        "seed": int(seed),
        "split_conversation_counts": {name: len(rows) for name, rows in by_split.items()},
        "preserved_source_splits": int(preserved_rows),
        "quarantined_group_count": int(quarantined_groups),
        "quarantined_conversation_count": len(quarantined),
        "synthetic_target_text_masks": dict(sorted(target_masks.items())),
        "source_split_conversation_counts": {
            source: {name: sum(row["source"] == source for row in rows) for name, rows in by_split.items()}
            for source in sorted({str(row["source"]) for row in prepared_rows})
        },
        "split_action_frame_counts": {
            name: dict(Counter(
                frame["target_action"]
                for row in rows
                for frame in row.get("frames", ())
                if frame.get("action_mask") and frame.get("target_action")
            ))
            for name, rows in by_split.items()
        },
        "files": {**{name: f"{name}.jsonl" for name in by_split}, "quarantine": "quarantine.jsonl"},
    }
    (output / "preparation_manifest.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return counts


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row deve essere un object")
            rows.append(value)
    return rows


__all__ = ["prepare_splits", "read_jsonl"]
