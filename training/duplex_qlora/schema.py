"""Canonical conversation/frame schema and leak-resistant data splitting."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

ACTIONS = ("LISTEN", "SPEAK", "CONTINUE", "YIELD", "BACKCHANNEL", "REPLAN")
SCHEMA_VERSION = 1
DEFAULT_TICK_MS = 80


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} deve essere una stringa non vuota")
    return value.strip()


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and abs(value) != float("inf")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_conversation(
    row: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    tick_ms: int = DEFAULT_TICK_MS,
) -> dict[str, Any]:
    """Normalize explicit annotations while retaining unknowns and raw times.

    Action labels are never inferred from speaker changes, role alternation, or
    overlap. Invalid event intervals keep their source timestamps and receive
    a false loss mask instead of being clipped to the media duration.
    """
    if not isinstance(row, Mapping):
        raise TypeError("conversation deve essere un object")
    conversation_id = _required_text(row.get("conversation_id"), "conversation_id")
    source = _required_text(row.get("source", "unknown"), "source")
    duration_ms = row.get("duration_ms")
    if duration_ms is None and _finite_number(row.get("duration_s")):
        duration_ms = float(row["duration_s"]) * 1000.0
    if not _finite_number(duration_ms) or duration_ms <= 0:
        raise ValueError(f"{conversation_id}: duration_ms deve essere positivo")
    if isinstance(tick_ms, bool) or int(tick_ms) <= 0:
        raise ValueError("tick_ms deve essere positivo")

    raw_audio_path = row.get("audio_path", row.get("media_path", row.get("wave_path")))
    audio_path = None
    if raw_audio_path is not None:
        path = Path(str(raw_audio_path))
        if not path.is_absolute() and base_dir is not None:
            if not path.exists():
                path = Path(base_dir) / path
        audio_path = str(path.resolve())

    sample_rate = int(row.get("sample_rate_hz", 16000))
    if sample_rate <= 0:
        raise ValueError(f"{conversation_id}: sample_rate_hz deve essere positivo")

    segments: list[dict[str, Any]] = []
    source_segments = row.get("segments", row.get("transcript_segments", row.get("utterances", ()))) or ()
    for index, source_segment in enumerate(source_segments):
        if not isinstance(source_segment, Mapping):
            raise ValueError(f"{conversation_id}: segments[{index}] non è un object")
        segment = dict(source_segment)
        start = segment.get("start_ms", segment.get("start"))
        end = segment.get("end_ms", segment.get("end"))
        if start is None and _finite_number(segment.get("start_sample")):
            start = int(segment["start_sample"]) * 1000.0 / sample_rate
        if start is None and _finite_number(segment.get("start_s")):
            start = float(segment["start_s"]) * 1000.0
        if end is None and _finite_number(segment.get("end_sample")):
            end = int(segment["end_sample"]) * 1000.0 / sample_rate
        if end is None and _finite_number(segment.get("end_s")):
            end = float(segment["end_s"]) * 1000.0
        segment["start_ms"] = start
        segment["end_ms"] = end
        valid_time = bool(
            _finite_number(start)
            and _finite_number(end)
            and 0 <= start < end <= duration_ms
        )
        segment["time_mask"] = bool(segment.get("time_mask", True) and valid_time)
        text_value = segment.get("text", segment.get("transcript", segment.get("utterance", "")))
        has_text = isinstance(text_value, str) and bool(text_value.strip())
        segment["text_mask"] = bool(segment.get("text_mask", True) and has_text and segment["time_mask"])
        segments.append(segment)

    events: list[dict[str, Any]] = []
    source_events = row.get("events", row.get("labeled_events", row.get("annotations", ()))) or ()
    if isinstance(source_events, Mapping):
        source_events = [source_events]
    for index, source_event in enumerate(source_events):
        if not isinstance(source_event, Mapping):
            raise ValueError(f"{conversation_id}: events[{index}] non è un object")
        event = dict(source_event)
        action = event.get("action", event.get("target_action", event.get("adjudicated_action")))
        start = event.get("start_ms", event.get("start"))
        end = event.get("end_ms", event.get("end"))
        if start is None and _finite_number(event.get("start_sample")):
            start = int(event["start_sample"]) * 1000.0 / sample_rate
        if start is None and _finite_number(event.get("start_s")):
            start = float(event["start_s"]) * 1000.0
        if end is None and _finite_number(event.get("end_sample")):
            end = int(event["end_sample"]) * 1000.0 / sample_rate
        if end is None and _finite_number(event.get("end_s")):
            end = float(event["end_s"]) * 1000.0
        # Retain original numeric values; do not repair malformed/out-of-range data.
        if start is not None:
            event["start_ms"] = start
        if end is not None:
            event["end_ms"] = end
        event["action"] = action
        valid_time = bool(
            _finite_number(start)
            and _finite_number(end)
            and 0 <= start < end <= duration_ms
        )
        event["time_mask"] = bool(event.get("time_mask", True) and valid_time)
        event["action_mask"] = bool(event.get("action_mask", True) and event["time_mask"] and action in ACTIONS)
        events.append(event)

    frames: list[dict[str, Any]] = []
    explicit_frames = "frames" in row
    source_frames = row.get("frames", ()) or ()
    if not explicit_frames:
        generated: list[dict[str, Any]] = []
        frame_count = int((float(duration_ms) + int(tick_ms) - 1) // int(tick_ms))
        for frame_id in range(frame_count):
            start = frame_id * int(tick_ms)
            end = min(float(duration_ms), start + int(tick_ms))
            active_actions = {
                event["action"] for event in events
                if event.get("action_mask")
                and float(event["start_ms"]) < end
                and float(event["end_ms"]) > start
            }
            text_parts = [
                str(segment.get("text", segment.get("transcript", segment.get("utterance", "")))).strip()
                for segment in segments
                if segment.get("time_mask")
                and float(segment["start_ms"]) < end
                and float(segment["end_ms"]) > start
                and isinstance(segment.get("text", segment.get("transcript", segment.get("utterance", ""))), str)
            ]
            generated.append(
                {
                    "frame_id": frame_id,
                    "start_ms": start,
                    "end_ms": end,
                    "valid_ms": max(0.0, end - start),
                    "valid_samples": int(round(max(0.0, end - start) * sample_rate / 1000.0)),
                    "target_action": next(iter(active_actions)) if len(active_actions) == 1 else None,
                    "action_mask": len(active_actions) == 1,
                    "text": " ".join(part for part in text_parts if part),
                }
            )
        source_frames = generated
    for index, source_frame in enumerate(source_frames):
        if not isinstance(source_frame, Mapping):
            raise ValueError(f"{conversation_id}: frames[{index}] non è un object")
        frame = dict(source_frame)
        frame_id = frame.get("frame_id", frame.get("frame_index", index))
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError(f"{conversation_id}: frame_id non valido")
        start = frame.get("start_ms", frame.get("timestamp_ms", frame_id * tick_ms))
        end = frame.get("end_ms", start + tick_ms if _finite_number(start) else None)
        raw_action = frame.get("target_action", frame.get("action"))
        requested_mask = frame.get("action_mask", True)
        valid_action = raw_action in ACTIONS
        frame["frame_id"] = frame_id
        frame["start_ms"] = start
        frame["end_ms"] = end
        frame["raw_target_action"] = raw_action
        frame["target_action"] = raw_action if valid_action else None
        valid_interval = bool(
            _finite_number(start)
            and _finite_number(end)
            and 0 <= start < end
            and start < duration_ms
        )
        if "valid_ms" not in frame:
            frame["valid_ms"] = (
                max(0.0, min(float(end), float(duration_ms)) - float(start))
                if _finite_number(start) and _finite_number(end)
                else 0.0
            )
        valid_ms = frame.get("valid_ms", 0.0)
        valid_ms_ok = bool(
            _finite_number(valid_ms)
            and 0 < valid_ms <= int(tick_ms)
            and valid_interval
        )
        frame.setdefault("valid_samples", int(round(float(valid_ms) * sample_rate / 1000.0)) if _finite_number(valid_ms) else 0)
        frame["time_mask"] = bool(frame.get("time_mask", True) and valid_ms_ok)
        frame["action_mask"] = bool(
            requested_mask
            and valid_action
            and frame["time_mask"]
        )
        frame["text_mask"] = bool(
            frame.get("text_mask", True)
            and any(isinstance(frame.get(key), str) and frame[key].strip() for key in ("text", "user_text", "assistant_text"))
        )
        frame["audio_mask"] = bool(frame.get("audio_mask", audio_path is not None))
        frame["events"] = [dict(event) for event in (frame.get("events", ()) or ())]
        frames.append(frame)

    result = dict(row)
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "conversation_id": conversation_id,
            "source_group_id": _required_text(
                row.get("source_group_id", row.get("cross_dataset_source_group_id", conversation_id)),
                "source_group_id",
            ),
            "source": source,
            "duration_ms": float(duration_ms),
            "tick_ms": int(tick_ms),
            "audio_path": audio_path,
            "sample_rate_hz": int(row.get("sample_rate_hz", 16000)),
            "audio_sha256": row.get("audio_sha256"),
            "segments": segments,
            "events": events,
            "frames": frames,
            "split": row.get("split"),
        }
    )
    return result


def split_conversations(
    conversations: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20260930,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, str]:
    """Assign whole connected source/audio groups to deterministic splits."""
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions devono essere non negative e sommare a meno di 1")
    rows = list(conversations)
    ids = [_required_text(row.get("conversation_id"), "conversation_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("conversation_id duplicati prima dello split")
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

    owner_by_key: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        keys = [
            ("source_group", str(row.get("source_group_id") or row.get("conversation_id"))),
            ("conversation", str(row.get("conversation_id"))),
        ]
        digest = row.get("audio_sha256")
        if digest:
            keys.append(("audio_sha256", str(digest).lower()))
        audio_path = row.get("audio_path", row.get("media_path"))
        if not digest and audio_path and Path(str(audio_path)).is_file():
            keys.append(("audio_sha256", sha256_file(str(audio_path))))
        for key in keys:
            previous = owner_by_key.setdefault(key, index)
            union(index, previous)

    groups: dict[int, list[int]] = {}
    for index in range(len(ids)):
        groups.setdefault(find(index), []).append(index)
    explicit_by_root: dict[int, str] = {}
    for root, members in groups.items():
        explicit = {
            str(rows[index].get("split")).strip().lower()
            for index in members
            if rows[index].get("split") not in (None, "", "unassigned")
        }
        aliases = {"val": "validation", "valid": "validation", "dev": "validation", "training": "train"}
        normalized = {aliases.get(value, value) for value in explicit}
        if not normalized <= {"train", "validation", "test"}:
            raise ValueError(f"split esplicito non riconosciuto nel gruppo {ids[members[0]]}: {sorted(normalized)}")
        if len(normalized) > 1:
            # The preparation layer quarantines this component with its full
            # conflict metadata. At this lower level keep the group together
            # and avoid choosing either conflicting declaration.
            continue
        if normalized:
            explicit_by_root[root] = next(iter(normalized))

    group_keys = sorted(groups, key=lambda root: tuple(sorted(ids[index] for index in groups[root])))
    unassigned = [root for root in group_keys if root not in explicit_by_root]
    random.Random(int(seed)).shuffle(unassigned)
    group_count = len(unassigned)
    n_validation = int(round(group_count * validation_fraction))
    n_test = int(round(group_count * test_fraction))
    if group_count == 2 and validation_fraction > 0:
        n_validation, n_test = 1, 0
    if group_count >= 3:
        if validation_fraction > 0:
            n_validation = max(1, n_validation)
        if test_fraction > 0:
            n_test = max(1, n_test)
        while n_validation + n_test >= group_count:
            if n_test >= n_validation and n_test:
                n_test -= 1
            elif n_validation:
                n_validation -= 1
            else:
                break
    random_assignments: dict[int, str] = {}
    for position, root in enumerate(unassigned):
        random_assignments[root] = "validation" if position < n_validation else (
            "test" if position < n_validation + n_test else "train"
        )
    assignments: dict[str, str] = {}
    for root, members in groups.items():
        split = explicit_by_root[root] if root in explicit_by_root else random_assignments[root]
        for index in members:
            assignments[ids[index]] = split
    return assignments


__all__ = ["ACTIONS", "SCHEMA_VERSION", "canonicalize_conversation", "sha256_file", "split_conversations"]
