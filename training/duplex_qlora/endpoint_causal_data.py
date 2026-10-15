"""Build causal endpoint labels for the bounded native Control6 listener set."""
from __future__ import annotations

import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .schema import ACTIONS


_SPOKEN_ACTIONS = {"LISTEN", "SPEAK", "CONTINUE", "BACKCHANNEL"}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _role(row: Mapping[str, Any], segment: Mapping[str, Any]) -> str | None:
    role = str(segment.get("perspective_role") or segment.get("role") or "").strip().lower()
    channel = segment.get("channel_index")
    channel_map = row.get("channel_map") or {}
    mapped = channel_map.get(str(channel), channel_map.get(channel)) if channel is not None else None
    if role not in {"user", "assistant"} or str(mapped or "").strip().lower() != role:
        return None
    if segment.get("audio_channel_verified") is not True or segment.get("time_mask", True) is not True:
        return None
    start, end = segment.get("start_ms"), segment.get("end_ms")
    duration = row.get("duration_ms")
    if not (_finite(start) and _finite(end) and _finite(duration)):
        return None
    if float(start) < 0 or float(start) >= float(end) or float(end) > float(duration):
        return None
    return role


def _state(segment: Mapping[str, Any]) -> str:
    return str(segment.get("native_turn_state") or segment.get("native_turn_state_raw") or "").strip().lower()


def _overlaps(segment: Mapping[str, Any], start_ms: float, end_ms: float) -> bool:
    return float(segment["start_ms"]) < end_ms and float(segment["end_ms"]) > start_ms


def _active_at(segments: list[Mapping[str, Any]], boundary_ms: float) -> bool:
    return any(float(segment["start_ms"]) < boundary_ms < float(segment["end_ms"]) for segment in segments)


def _frame_bounds(frame: Mapping[str, Any], tick_ms: int) -> tuple[int, float, float]:
    frame_id = int(frame.get("frame_id", -1))
    if frame_id < 0:
        raise ValueError("frame_id endpoint causale deve essere non negativo")
    start_ms = float(frame.get("start_ms", frame_id * tick_ms))
    end_ms = float(frame.get("end_ms", (frame_id + 1) * tick_ms))
    if not math.isfinite(start_ms) or not math.isfinite(end_ms) or not start_ms < end_ms:
        raise ValueError(f"frame {frame_id}: intervallo temporale non valido")
    return frame_id, start_ms, end_ms


def _set_label(
    frame: dict[str, Any],
    action: str | None,
    *,
    source: str | None = None,
    assistant_reference_frame_id: int | None = None,
) -> None:
    frame["target_action"] = action
    frame["action_mask"] = action in ACTIONS
    if source is not None:
        frame["target_action_source"] = source
    else:
        frame.pop("target_action_source", None)
    if assistant_reference_frame_id is not None:
        frame["assistant_reference_frame_id"] = int(assistant_reference_frame_id)
    else:
        frame.pop("assistant_reference_frame_id", None)


def derive_native_endpoint_labels(
    conversation: Mapping[str, Any],
    *,
    pause_ms: float = 160.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Relabel a native row using completed-user readiness and directed actions.

    A SPEAK target is the first 80 ms tick whose *end* observes at least
    ``pause_ms`` after a verified native COMPLETE user segment. Its head input
    remains causal: the native extractor supplies assistant PCM from t-1 only.
    The dataset rejects a candidate when another user is active at the target
    boundary or assistant PCM is present in t-1. It never reads future
    assistant onset to create a target.
    """
    if pause_ms < 0 or not math.isfinite(float(pause_ms)):
        raise ValueError("pause_ms deve essere finito e non negativo")
    result = copy.deepcopy(dict(conversation))
    if result.get("synthetic") is True:
        return result, {
            "synthetic_row_preserved": True,
            "speak_endpoint_candidates": 0,
            "speak_endpoints_created": 0,
            "speak_endpoint_rule": "first_tick_end_at_or_after_user_end_plus_160ms",
        }

    tick_ms = int(result.get("tick_ms", 80))
    if tick_ms != 80:
        raise ValueError(f"{result.get('conversation_id')}: endpoint Control6 richiede tick 80 ms")
    duration_ms = float(result.get("duration_ms", math.nan))
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError(f"{result.get('conversation_id')}: duration_ms non valido")
    channel_map = result.get("channel_map") or {}
    if "assistant" not in {str(value).strip().lower() for value in channel_map.values()}:
        raise ValueError(f"{result.get('conversation_id')}: manca il canale assistant per il riferimento t-1")

    role_segments: dict[str, list[Mapping[str, Any]]] = {"user": [], "assistant": []}
    unverified = 0
    for segment in result.get("segments", ()) or ():
        role = _role(result, segment)
        if role is None:
            unverified += 1
            continue
        role_segments[role].append(segment)
    users = role_segments["user"]
    assistants = role_segments["assistant"]

    frame_by_id: dict[int, dict[str, Any]] = {}
    for original in result.get("frames", ()) or ():
        frame = dict(original)
        frame_id, _, _ = _frame_bounds(frame, tick_ms)
        if frame_id in frame_by_id:
            raise ValueError(f"{result.get('conversation_id')}: frame_id duplicato {frame_id}")
        frame_by_id[frame_id] = frame

    linked_actions: dict[int, str] = {}
    stats: Counter[str] = Counter()
    for frame_id, frame in frame_by_id.items():
        _frame_id, start_ms, end_ms = _frame_bounds(frame, tick_ms)
        original_action = frame.get("target_action") if frame.get("action_mask") is True else None
        if original_action == "BACKCHANNEL":
            linked = any(
                _state(segment) == "backchannel" and _overlaps(segment, start_ms, end_ms)
                for segment in assistants
            )
            if linked:
                linked_actions[frame_id] = "BACKCHANNEL"
                stats["backchannel_kept"] += 1
            else:
                stats["backchannel_unlinked_removed"] += 1
        elif original_action == "CONTINUE":
            linked = any(_overlaps(segment, start_ms, end_ms) for segment in assistants)
            if linked:
                linked_actions[frame_id] = "CONTINUE"
                stats["continue_kept"] += 1
            else:
                stats["continue_unlinked_removed"] += 1
        elif original_action == "SPEAK":
            stats["old_assistant_onset_speak_removed"] += 1
        elif original_action in {"YIELD", "REPLAN"}:
            stats["native_external_action_ignored"] += 1
        _set_label(frame, None)

    endpoint_by_id: dict[int, list[str]] = defaultdict(list)
    tick_total = int(math.ceil(duration_ms / tick_ms))
    for segment in users:
        if _state(segment) != "complete":
            continue
        stats["verified_complete_user_segments"] += 1
        required_end = float(segment["end_ms"]) + float(pause_ms)
        frame_id = max(0, int(math.ceil(required_end / tick_ms - 1e-12)) - 1)
        if frame_id >= tick_total:
            stats["endpoint_rejected_beyond_audio"] += 1
            continue
        start_ms = frame_id * tick_ms
        end_ms = min((frame_id + 1) * tick_ms, duration_ms)
        if end_ms < required_end:
            stats["endpoint_rejected_beyond_audio"] += 1
            continue
        if _active_at(users, end_ms):
            stats["endpoint_rejected_user_active"] += 1
            continue
        previous_start = max(0.0, start_ms - tick_ms)
        if frame_id > 0 and any(_overlaps(segment, previous_start, start_ms) for segment in assistants):
            stats["endpoint_rejected_assistant_t_minus_1_active"] += 1
            continue
        endpoint_by_id[frame_id].append(str(segment.get("segment_id") or "unknown"))

    for frame_id, segment_ids in endpoint_by_id.items():
        if frame_id in linked_actions:
            stats["endpoint_rejected_directed_action_conflict"] += len(segment_ids)
            continue
        if frame_id not in frame_by_id:
            start_ms = frame_id * tick_ms
            end_ms = min((frame_id + 1) * tick_ms, duration_ms)
            sample_rate = int(result.get("sample_rate_hz", 16000))
            valid_ms = max(0.0, end_ms - start_ms)
            valid_samples = max(0, int(round(valid_ms * sample_rate / 1000.0)))
            frame_by_id[frame_id] = {
                "frame_id": frame_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "valid_ms": valid_ms,
                "valid_samples": valid_samples,
                "time_mask": valid_ms > 0,
                "audio_mask": False,
                "events": [],
                "target_action": None,
                "action_mask": False,
            }
            stats["endpoint_sparse_frame_added"] += 1
        frame = frame_by_id[frame_id]
        _set_label(
            frame,
            "SPEAK",
            source="verified_user_complete_end_plus_pause_160ms",
            assistant_reference_frame_id=frame_id - 1 if frame_id > 0 else -1,
        )
        frame["endpoint_user_segment_ids"] = sorted(set(segment_ids))
        stats["speak_endpoints_created"] += 1

    for frame_id, frame in frame_by_id.items():
        if frame.get("action_mask") is True:
            continue
        _frame_id, start_ms, end_ms = _frame_bounds(frame, tick_ms)
        if frame_id in endpoint_by_id and frame.get("target_action") == "SPEAK":
            continue
        if frame_id in linked_actions:
            _set_label(frame, linked_actions[frame_id], source="verified_native_assistant_action")
            continue

        previous_start = max(0.0, start_ms - tick_ms)
        assistant_previous_active = frame_id > 0 and any(
            _overlaps(segment, previous_start, start_ms) for segment in assistants
        )
        if assistant_previous_active:
            continue
        if _active_at(users, end_ms):
            _set_label(frame, "LISTEN", source="verified_remote_user_active")
            stats["user_active_listen_frames"] += 1
            continue

        prior_users = [segment for segment in users if float(segment["start_ms"]) < end_ms]
        if not prior_users:
            continue
        latest_user = max(prior_users, key=lambda segment: (float(segment["start_ms"]), float(segment["end_ms"])))
        latest_state = _state(latest_user)
        if latest_state in {"incomplete", "backchannel"}:
            _set_label(frame, "LISTEN", source="verified_user_turn_incomplete_or_nonfinal")
            stats["incomplete_user_listen_frames"] += 1
            continue
        if latest_state == "complete" and end_ms < float(latest_user["end_ms"]) + float(pause_ms):
            _set_label(frame, "LISTEN", source="complete_user_pause_below_speak_endpoint")
            stats["pre_endpoint_listen_frames"] += 1

    result["frames"] = [frame_by_id[frame_id] for frame_id in sorted(frame_by_id)]
    stats["speak_endpoint_candidates"] = sum(len(ids) for ids in endpoint_by_id.values())
    stats["verified_user_segments"] = len(users)
    stats["verified_assistant_segments"] = len(assistants)
    stats["unverified_or_unmapped_segments"] = unverified
    report = dict(stats)
    report["speak_endpoint_rule"] = "first_tick_end_at_or_after_user_end_plus_160ms"
    report["tick_ms"] = tick_ms
    report["pause_ms"] = float(pause_ms)
    report["assistant_input_reference"] = "t_minus_1_only"
    report["future_assistant_onset_used_for_speak_creation"] = False
    return result, report


def _action_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(frame.get("action") if "target_action_mask" in frame else frame.get("target_action"))
        for row in rows
        for frame in (row.get("frame_targets", ()) or row.get("frames", ()) or ())
        if (frame.get("target_action_mask") is True if "target_action_mask" in frame else frame.get("action_mask") is True)
        and (frame.get("action") if "target_action_mask" in frame else frame.get("target_action")) in ACTIONS
    )
    return {action: int(counts[action]) for action in ACTIONS}


def prepare_endpoint_causal_control6(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    train_cap_per_action: int = 256,
    validation_cap_per_action: int = 64,
    test_cap_per_action: int = 64,
) -> dict[str, Any]:
    """Relabel a preserved combined source set and write a separate capped candidate."""
    from tools.assemble_duplex_qlora_stage2_dataset import _cap_labeled_frames, _read_rows

    input_root = Path(input_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"endpoint dataset già presente, non sovrascritto: {output_root}")
    output_root.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "causal native SPEAK readiness endpoint; preserves source groups/splits and synthetic external actions",
        "source_dataset_dir": str(input_root),
        "label_contract": {
            "speak_trigger": "verified remote/user native COMPLETE end_ms plus at least 160 ms pause observed by target tick end",
            "target_tick": "first 80 ms frame whose end_ms >= user_end_ms + 160 ms",
            "assistant_input": "only physical assistant channel frame t-1; no future assistant onset is used to create SPEAK",
            "endpoint_exclusions": ["another user segment active at target boundary", "assistant segment overlaps t-1 input frame", "endpoint beyond available audio", "conflict with verified directed BACKCHANNEL/CONTINUE"],
            "natural_listen": "verified active user, incomplete/nonfinal user pause, or complete-user pause before endpoint, when assistant t-1 is quiet",
            "natural_backchannel_continue": "preserved only when linked to verified native assistant segment timing/state",
            "natural_yield_replan": "not inferred; synthetic source labels preserved unchanged",
        },
        "caps_per_action": {
            "train": int(train_cap_per_action),
            "validation": int(validation_cap_per_action),
            "test": int(test_cap_per_action),
        },
        "tick_ms": 80,
        "endpoint_pause_ms": 160,
        "splits": {},
    }
    group_split: dict[str, str] = {}
    all_group_ids: dict[str, set[str]] = defaultdict(set)
    for split, cap in (
        ("train", train_cap_per_action),
        ("validation", validation_cap_per_action),
        ("test", test_cap_per_action),
    ):
        rows = _read_rows(input_root / f"{split}.jsonl")
        if not rows:
            raise ValueError(f"source split {split} vuoto o assente")
        transformed: list[dict[str, Any]] = []
        row_stats: Counter[str] = Counter()
        by_source: dict[str, Counter[str]] = defaultdict(Counter)
        source_groups_by_action: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for row in rows:
            conversation_id = str(row.get("conversation_id") or "")
            group_id = str(row.get("source_group_id") or conversation_id)
            if not conversation_id or not group_id:
                raise ValueError(f"{split}: conversation_id/source_group_id mancante")
            previous = group_split.setdefault(group_id, split)
            if previous != split:
                raise ValueError(f"source group {group_id} attraversa {previous}/{split}")
            all_group_ids[split].add(group_id)
            if row.get("synthetic") is True:
                clone = copy.deepcopy(row)
                stats = {"synthetic_row_preserved": True}
            else:
                clone, stats = derive_native_endpoint_labels(row)
            transformed.append(clone)
            for key, value in stats.items():
                if type(value) is int and key != "tick_ms":
                    row_stats[key] += value
                elif value is True:
                    row_stats["synthetic_rows_preserved"] += 1
        capped, selection = _cap_labeled_frames(transformed, cap_per_class=int(cap))
        for row in capped:
            source = str(row.get("source", "unknown"))
            group_id = str(row.get("source_group_id") or row.get("conversation_id"))
            for frame in (row.get("frame_targets", ()) or row.get("frames", ()) or ()):
                action = frame.get("action") if "target_action_mask" in frame else frame.get("target_action")
                action_mask = frame.get("target_action_mask") is True if "target_action_mask" in frame else frame.get("action_mask") is True
                if action_mask and action in ACTIONS:
                    by_source[source][str(action)] += 1
                    source_groups_by_action[str(action)][source].add(group_id)
        output_path = output_root / f"{split}.jsonl"
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in capped:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        report["splits"][split] = {
            "source_rows": len(rows),
            "output_rows": len(capped),
            "unique_source_groups": len(all_group_ids[split]),
            "source_rows_by_corpus": dict(sorted(Counter(str(row.get("source", "unknown")) for row in rows).items())),
            "candidate_frames_by_action": _action_counts(transformed),
            "selected_frames_by_action": _action_counts(capped),
            "selected_frames_by_source_action": {
                source: {action: int(counts[action]) for action in ACTIONS}
                for source, counts in sorted(by_source.items())
            },
            "selected_groups_by_source_action": {
                action: {source: len(groups) for source, groups in sorted(sources.items())}
                for action, sources in sorted(source_groups_by_action.items())
            },
            "frame_selection": selection,
            "label_derivation_counts": dict(sorted(row_stats.items())),
            "absent_or_unsupported_actions": [
                action for action, count in _action_counts(capped).items() if count == 0
            ],
        }
    result_groups = [all_group_ids[split] for split in ("train", "validation", "test")]
    for left_index, left in enumerate(result_groups):
        for right_index in range(left_index + 1, len(result_groups)):
            overlap = sorted(left & result_groups[right_index])
            if overlap:
                raise ValueError(f"source_group_id attraversa split: {overlap[:5]}")
    manifest_path = output_root / "endpoint_causal_manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["derive_native_endpoint_labels", "prepare_endpoint_causal_control6"]
