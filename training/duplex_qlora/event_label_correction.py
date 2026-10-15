"""Causal correction of repeated event labels in precomputed control features.

This module changes only action supervision. Native audio/query features remain
untouched. Timing is read from the prepared conversation's role-owned segments,
and unavailable ownership or timing is handled as an ignored target.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


EVENT_ACTIONS = frozenset({"BACKCHANNEL", "REPLAN", "YIELD"})
CONTINUE_ON_INTERIOR_ACTIONS = frozenset({"BACKCHANNEL", "REPLAN"})


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _channel_owner(channel_map: Mapping[Any, Any], channel_index: Any) -> str | None:
    if channel_index is None:
        return None
    return channel_map.get(str(channel_index), channel_map.get(channel_index))


def verified_assistant_segments(
    segments: Sequence[Mapping[str, Any]],
    channel_map: Mapping[Any, Any],
    *,
    interval_start_ms: float | None = None,
    interval_end_ms: float | None = None,
) -> tuple[list[Mapping[str, Any]], bool]:
    """Return role-owned, timed assistant segments and whether timing is unsafe.

    A conflicting or incomplete role/channel/timing record is reported as
    unsafe instead of silently treating the interval as inactive.
    """
    verified: list[Mapping[str, Any]] = []
    unsafe = False
    query_start = _finite_number(interval_start_ms)
    query_end = _finite_number(interval_end_ms)
    query_is_valid = (
        query_start is not None
        and query_end is not None
        and query_end > query_start
    )
    for segment in segments:
        if segment.get("role") != "assistant":
            continue
        # A false time mask explicitly says this segment is not observed as
        # physical assistant audio, so it cannot make another tick active.
        if segment.get("time_mask") is False:
            continue
        owner = _channel_owner(channel_map, segment.get("channel_index"))
        start = _finite_number(segment.get("start_ms"))
        end = _finite_number(segment.get("end_ms"))
        if owner != "assistant" or segment.get("time_mask") is not True:
            if not query_is_valid or start is None or end is None or (
                start < query_end and end > query_start
            ):
                unsafe = True
            continue
        if start is None or end is None or end <= start:
            unsafe = True
            continue
        verified.append(segment)
    return verified, unsafe


def assistant_active_in_previous_tick(
    *,
    frame_start_ms: float,
    tick_ms: float,
    segments: Sequence[Mapping[str, Any]],
    channel_map: Mapping[Any, Any],
) -> bool | None:
    """Whether verified assistant audio overlaps the tick immediately before.

    ``None`` means source ownership/timing is incomplete and callers must mask
    an event interior. The queried window ends at ``frame_start_ms`` so no
    current or future audio can affect this predicate.
    """
    frame_start = _finite_number(frame_start_ms)
    tick = _finite_number(tick_ms)
    if frame_start is None or tick is None or tick <= 0:
        return None
    previous_start = frame_start - tick
    previous_end = frame_start
    assistant, unsafe = verified_assistant_segments(
        segments,
        channel_map,
        interval_start_ms=previous_start,
        interval_end_ms=previous_end,
    )
    if unsafe:
        return None
    assistant_rows = [segment for segment in segments if segment.get("role") == "assistant"]
    if not assistant_rows:
        return None
    if not assistant:
        explicit_inactive = all(
            segment.get("time_mask") is False
            and _channel_owner(channel_map, segment.get("channel_index")) == "assistant"
            for segment in assistant_rows
        )
        return False if explicit_inactive else None
    return any(
        float(segment["start_ms"]) < previous_end
        and float(segment["end_ms"]) > previous_start
        for segment in assistant
    )


def event_run_offset(
    frames: Sequence[Mapping[str, Any]],
    *,
    frame_id: int,
    action: str,
    segment_ids: Mapping[int, str] | None = None,
) -> int | None:
    """Return the causal tick offset in a contiguous same-event label run.

    If ``segment_ids`` is supplied, adjacent labels count as one event only
    while both frames map to the same verified utterance segment.
    """
    by_id = {
        int(frame["frame_id"]): frame
        for frame in frames
        if frame.get("frame_id") is not None
    }
    target = by_id.get(int(frame_id))
    if (
        target is None
        or target.get("target_action") != action
        or target.get("action_mask") is not True
    ):
        return None
    if segment_ids is not None and int(frame_id) not in segment_ids:
        return None
    offset = 0
    cursor = int(frame_id)
    while True:
        previous_id = cursor - 1
        previous = by_id.get(previous_id)
        if (
            previous is None
            or previous.get("target_action") != action
            or previous.get("action_mask") is not True
        ):
            break
        if segment_ids is not None and (
            previous_id not in segment_ids
            or segment_ids[previous_id] != segment_ids[cursor]
        ):
            break
        offset += 1
        cursor = previous_id
    return offset


def relabel_event_action(
    action: str,
    *,
    offset_ticks: int | None,
    assistant_prev_active: bool | None,
) -> tuple[str | None, bool, str]:
    """Keep only an event onset and map/mask its repeated interior frames."""
    if action not in EVENT_ACTIONS:
        raise ValueError(f"azione non-evento non supportata: {action!r}")
    if offset_ticks is None or offset_ticks < 0:
        return None, False, "missing_event_or_assistant_timing"
    if offset_ticks == 0:
        return action, True, "event_onset"
    if assistant_prev_active is None:
        return None, False, "missing_event_or_assistant_timing"
    if action in CONTINUE_ON_INTERIOR_ACTIONS:
        if assistant_prev_active:
            return "CONTINUE", True, "interior_assistant_active"
        return None, False, "interior_without_prior_assistant_audio"
    if action == "YIELD":
        if not assistant_prev_active:
            return "LISTEN", True, "interior_assistant_inactive"
        return None, False, "interior_assistant_active"
    raise AssertionError(f"evento non gestito: {action}")


__all__ = [
    "CONTINUE_ON_INTERIOR_ACTIONS",
    "EVENT_ACTIONS",
    "assistant_active_in_previous_tick",
    "event_run_offset",
    "relabel_event_action",
    "verified_assistant_segments",
]
