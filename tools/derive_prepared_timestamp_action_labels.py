#!/usr/bin/env python3
"""Derive timestamp-based action targets in a copied Candidate E train/val set."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# This offline data utility only needs the schema/source modules. Avoid the
# package's eager model-feature imports, which are not part of conversion and
# can bind an unrelated serving source overlay in the runtime image.
_DUPLEX_PACKAGE = "training.duplex_qlora"
if _DUPLEX_PACKAGE not in sys.modules:
    _package = types.ModuleType(_DUPLEX_PACKAGE)
    _package.__path__ = [str(Path(__file__).resolve().parents[1] / "training" / "duplex_qlora")]
    _package.__package__ = _DUPLEX_PACKAGE
    sys.modules[_DUPLEX_PACKAGE] = _package

from training.duplex_qlora.prepare import read_jsonl
from training.duplex_qlora.sources import relabel_prepared_timestamp_actions


_TICK_MS = 80
_RMS_FLOOR = 0.001
_SPEAK_SEED_SOURCES = {
    "timestamp_derived_assistant_onset_after_peer_termination",
    "timestamp_derived_last_vocal_complete_user_turn",
}
_YIELD_SEED_SOURCE = "timestamp_derived_peer_onset_confirmed_local_stop"


def _target_fields(row: Mapping[str, Any]) -> tuple[str, str, str, str, bool]:
    is_v5 = isinstance(row.get("frame_targets"), list)
    if is_v5:
        return "frame_targets", "action", "target_action_mask", "input_frame_id", True
    return "frames", "target_action", "action_mask", "frame_id", False


def _frame_action(frame: Mapping[str, Any], *, action_key: str) -> str | None:
    action = frame.get(action_key, frame.get("target_action"))
    return str(action).upper() if isinstance(action, str) else None


def _frame_id(frame: Mapping[str, Any], *, id_key: str) -> int | None:
    value = frame.get(id_key, frame.get("frame_id"))
    return value if type(value) is int and value >= 0 else None


def _manual_review(frame: Mapping[str, Any]) -> bool:
    return frame.get("target_action_source") == "explicit_source_action_reviewed"


def _source_segments(row: Mapping[str, Any], *, is_v5: bool) -> list[dict[str, Any]]:
    values = row.get("utterances", ()) if is_v5 else row.get("segments", ())
    sample_rate = int(row.get("sample_rate_hz", row.get("sample_rate", 16000)))
    result: list[dict[str, Any]] = []
    for raw in values or ():
        if not isinstance(raw, Mapping):
            continue
        if is_v5:
            # In V5, speaker is the physical/timeline role. `role` is an event
            # kind such as assistant_start_interruptible or fresh_correction.
            speaker = str(raw.get("speaker") or "").strip().lower()
            start_sample, end_sample = raw.get("start_sample"), raw.get("end_sample")
            if (speaker not in {"user", "assistant"} or type(start_sample) is not int
                    or type(end_sample) is not int or end_sample <= start_sample or sample_rate <= 0):
                continue
            start_ms = start_sample * 1000.0 / sample_rate
            end_ms = end_sample * 1000.0 / sample_rate
            state = "complete"
            segment_id = raw.get("utterance_id") or raw.get("segment_id")
        else:
            role = str(raw.get("perspective_role") or raw.get("role") or "").strip().lower()
            if role not in {"user", "assistant"} or raw.get("audio_channel_verified") is not True:
                continue
            if raw.get("time_mask", True) is not True:
                continue
            start_ms, end_ms = raw.get("start_ms"), raw.get("end_ms")
            if (isinstance(start_ms, bool) or isinstance(end_ms, bool)
                    or not isinstance(start_ms, (int, float)) or not isinstance(end_ms, (int, float))
                    or not math.isfinite(float(start_ms)) or not math.isfinite(float(end_ms))
                    or float(end_ms) <= float(start_ms)):
                continue
            speaker = role
            start_ms, end_ms = float(start_ms), float(end_ms)
            state = str(raw.get("native_turn_state") or raw.get("annotation_state") or "").strip().lower()
            state = state.strip("<>|")
            segment_id = raw.get("segment_id") or raw.get("source_event_id")
        result.append({
            "id": str(segment_id or ""),
            "speaker": speaker,
            "start_ms": float(start_ms),
            "end_ms": float(end_ms),
            "state": state,
            "raw": raw,
        })
    return sorted(result, key=lambda item: (item["start_ms"], item["end_ms"], item["id"]))


def _raw_action_fields(frame: Mapping[str, Any], *, action_key: str, mask_key: str) -> dict[str, Any]:
    return {
        "action": frame.get(action_key, frame.get("target_action")),
        "action_mask": frame.get(mask_key),
        "target_action_source": frame.get("target_action_source"),
        "target_action_evidence": copy.deepcopy(frame.get("target_action_evidence")),
        "action_masked_reason": frame.get("action_masked_reason"),
    }


def _set_temporal_target(
    frame: dict[str, Any], *, action_key: str, mask_key: str, action: str,
    source: str, evidence: Mapping[str, Any], reason: str | None = None,
) -> None:
    if "native1540_temporal_view_original" not in frame:
        frame["native1540_temporal_view_original"] = _raw_action_fields(
            frame, action_key=action_key, mask_key=mask_key
        )
    frame[action_key] = action
    frame[mask_key] = True
    frame["target_action_source"] = source
    frame["target_action_evidence"] = dict(evidence)
    frame["action_masked_reason"] = reason
    if action_key == "target_action":
        frame["action"] = action
    if action_key == "action":
        frame["target_action"] = action
        frame["target_action_mask"] = True


def _mask_temporal_target(
    frame: dict[str, Any], *, action_key: str, mask_key: str, reason: str, source: str,
) -> None:
    if "native1540_temporal_view_original" not in frame:
        frame["native1540_temporal_view_original"] = _raw_action_fields(
            frame, action_key=action_key, mask_key=mask_key
        )
    frame[mask_key] = False
    frame["action_masked_reason"] = reason
    frame["target_action_source"] = source
    if action_key == "target_action":
        frame["action_mask"] = False
    if action_key == "action":
        frame["target_action_mask"] = False


def _natural_frame(
    row: Mapping[str, Any], frame_id: int, *, segments: Sequence[Mapping[str, Any]],
    rms: Sequence[float], duration_ms: float,
) -> dict[str, Any]:
    start_ms = frame_id * _TICK_MS
    end_ms = min(duration_ms, start_ms + _TICK_MS)
    sample_rate = int(row.get("sample_rate_hz", 16000))
    assistant_active = any(
        segment["speaker"] == "assistant" and segment["start_ms"] <= start_ms < segment["end_ms"]
        for segment in segments
    )
    user_active = any(
        segment["speaker"] == "user" and segment["start_ms"] < end_ms and segment["end_ms"] > start_ms
        for segment in segments
    )
    return {
        "frame_id": frame_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "valid_ms": max(0.0, end_ms - start_ms),
        "valid_samples": int(round(max(0.0, end_ms - start_ms) * sample_rate / 1000.0)),
        "target_action": None,
        "action_mask": False,
        "target_action_source": None,
        "target_action_evidence": None,
        "action_masked_reason": "native1540_temporal_target_context",
        "event_onset": False,
        "time_mask": end_ms > start_ms,
        "audio_mask": bool(row.get("audio_path")),
        "speaker_active_at_frame_start": assistant_active,
        "peer_active_during_frame": user_active,
        "context_provenance": "verified_role_intervals_and_user_pcm_rms",
        "native1540_user_rms": float(rms[frame_id]) if frame_id < len(rms) else 0.0,
    }


def apply_native1540_pcm_temporal_targets(
    conversation: Mapping[str, Any], user_rms: Sequence[float], *, rms_floor: float = _RMS_FLOOR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a TRAIN/VAL-only copy whose labels follow measured 80 ms PCM edges.

    Existing source frames and evidence are copied verbatim into per-frame
    `native1540_temporal_view_original` fields before a target is changed.
    """
    if not math.isfinite(float(rms_floor)) or rms_floor <= 0:
        raise ValueError("rms_floor must be finite and positive")
    row = copy.deepcopy(dict(conversation))
    split = row.get("split")
    if split not in {"train", "validation"}:
        raise ValueError("native1540 temporal view accepts train/validation rows only")
    key, action_key, mask_key, id_key, is_v5 = _target_fields(row)
    frames = row.get(key)
    if not isinstance(frames, list):
        raise ValueError(f"{row.get('conversation_id')}: {key} must be a list")
    segments = _source_segments(row, is_v5=is_v5)
    by_id: dict[int, dict[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError(f"{row.get('conversation_id')}: frame targets must be objects")
        frame_id = _frame_id(frame, id_key=id_key)
        if frame_id is None or frame_id in by_id:
            raise ValueError(f"{row.get('conversation_id')}: duplicate or invalid feature frame id")
        by_id[frame_id] = frame
    rms = np.asarray(user_rms, dtype=np.float64).reshape(-1)
    if not np.isfinite(rms).all() or np.any(rms < 0):
        raise ValueError("user_rms must contain finite non-negative values")
    duration_ms = float(row.get("duration_ms", len(rms) * _TICK_MS))
    report: dict[str, Any] = {
        "conversation_id": str(row.get("conversation_id") or ""),
        "split": split,
        "source": str(row.get("source") or "unknown"),
        "v5_speaker_field_used": is_v5,
        "voiced_frames": int(np.count_nonzero(rms >= rms_floor)),
        "speak_events_qualified": 0,
        "speak_events_without_voiced_tick": 0,
        "speak_target_frames_added": 0,
        "speak_late_source_frames_masked": 0,
        "held_speak_frames_as_continue": 0,
        "yield_events_qualified": 0,
        "yield_events_without_voiced_overlap": 0,
        "yield_target_frames_added": 0,
        "reviewed_conflicts_protected": 0,
        "unverified_role_segments": 0,
    }

    def ensure_frame(frame_id: int) -> dict[str, Any] | None:
        if frame_id < 0 or frame_id >= len(rms) or frame_id * _TICK_MS >= duration_ms:
            return None
        frame = by_id.get(frame_id)
        if frame is None:
            if is_v5:
                # V5 is a dense input timeline; adding a row would violate the
                # input t -> target t+1 contract and is deliberately forbidden.
                return None
            frame = _natural_frame(row, frame_id, segments=segments, rms=rms, duration_ms=duration_ms)
            by_id[frame_id] = frame
            frames.append(frame)
        return frame

    def protected(frame: Mapping[str, Any], wanted: str) -> bool:
        current = _frame_action(frame, action_key=action_key)
        return bool(
            _manual_review(frame)
            and frame.get(mask_key) is True
            and current != wanted
        )

    def own_active(frame_id: int) -> bool:
        tick_start = frame_id * _TICK_MS
        return any(
            segment["speaker"] == "assistant"
            and segment["start_ms"] <= tick_start < segment["end_ms"]
            for segment in segments
        )

    def user_segment_by_id(segment_id: Any) -> Mapping[str, Any] | None:
        if not isinstance(segment_id, str) or not segment_id:
            return None
        return next((segment for segment in segments if segment["id"] == segment_id and segment["speaker"] == "user"), None)

    def assistant_segment_by_id(segment_id: Any) -> Mapping[str, Any] | None:
        if not isinstance(segment_id, str) or not segment_id:
            return None
        return next((segment for segment in segments if segment["id"] == segment_id and segment["speaker"] == "assistant"), None)

    speak_events: dict[tuple[str, str], tuple[Mapping[str, Any], Mapping[str, Any], str]] = {}
    if is_v5:
        utterance_by_id = {segment["id"]: segment for segment in segments if segment["id"]}
        for event in row.get("event_schedule", ()) or ():
            if not isinstance(event, Mapping) or str(event.get("action", "")).upper() != "SPEAK":
                continue
            peer = utterance_by_id.get(str(event.get("context_utterance_id") or ""))
            local = utterance_by_id.get(str(event.get("response_utterance_id") or ""))
            if (peer and local and peer["speaker"] == "user" and local["speaker"] == "assistant"
                    and peer["end_ms"] <= local["start_ms"]):
                speak_events[(peer["id"], local["id"])] = (peer, local, "v5_measured_speak_event")
    else:
        for frame in frames:
            if (_frame_action(frame, action_key=action_key) != "SPEAK"
                    or frame.get(mask_key) is not True
                    or frame.get("target_action_source") not in _SPEAK_SEED_SOURCES):
                continue
            evidence = frame.get("target_action_evidence")
            evidence = evidence if isinstance(evidence, Mapping) else {}
            local_id = evidence.get("assistant_segment_id") or evidence.get("local_segment_id")
            local = assistant_segment_by_id(local_id)
            peer_id = evidence.get("peer_segment_id") or evidence.get("previous_peer_segment_id")
            peer = user_segment_by_id(peer_id)
            if local is None:
                continue
            if peer is None:
                prior = [segment for segment in segments if segment["speaker"] == "user" and segment["end_ms"] <= local["start_ms"]]
                peer = max(prior, key=lambda segment: segment["end_ms"], default=None)
            if (peer and peer["state"] == "complete" and peer["end_ms"] <= local["start_ms"]
                    and not any(
                        segment["speaker"] == "user"
                        and peer["start_ms"] < segment["start_ms"] < local["start_ms"]
                        for segment in segments
                    )
                    and not any(
                        segment["speaker"] == "assistant"
                        and segment["id"] != local["id"]
                        and segment["start_ms"] < local["start_ms"]
                        and segment["end_ms"] > peer["end_ms"]
                        for segment in segments
                    )):
                speak_events[(peer["id"], local["id"])] = (peer, local, str(frame.get("target_action_source")))

    allowed_speak_frames: set[int] = set()
    for (peer_id, local_id), (peer, local, provenance) in speak_events.items():
        peer_start, peer_end, own_start = peer["start_ms"], peer["end_ms"], local["start_ms"]
        next_user_start = min((segment["start_ms"] for segment in segments
                               if segment["speaker"] == "user" and segment["start_ms"] > peer_start),
                              default=math.inf)
        limit_ms = min(peer_end + 160.0, next_user_start, own_start, duration_ms)
        first = max(0, int(peer_start // _TICK_MS))
        stop = min(len(rms), int(math.ceil(limit_ms / _TICK_MS)))
        voiced = [fid for fid in range(first, stop) if rms[fid] >= rms_floor]
        report["speak_events_qualified"] += 1
        if not voiced:
            report["speak_events_without_voiced_tick"] += 1
            continue
        last_voice = voiced[-1]
        candidate_ids = [last_voice]
        quiet_after = last_voice + 1
        if quiet_after < stop and rms[quiet_after] < rms_floor:
            candidate_ids.append(quiet_after)
        for fid in candidate_ids:
            if own_active(fid):
                continue
            frame = ensure_frame(fid)
            if frame is None:
                continue
            if protected(frame, "SPEAK"):
                report["reviewed_conflicts_protected"] += 1
                continue
            _set_temporal_target(
                frame, action_key=action_key, mask_key=mask_key, action="SPEAK",
                source="native1540_pcm_rms_last_vocal_endpoint",
                evidence={"peer_segment_id": peer_id, "assistant_segment_id": local_id,
                          "last_voiced_frame_id": last_voice, "tick_after_last_voice": fid == quiet_after,
                          "rms_floor": rms_floor, "allowed_until_ms": limit_ms,
                          "source_qualification": provenance, "pcm_rms_threshold": float(rms[fid])},
            )
            allowed_speak_frames.add(fid)
            report["speak_target_frames_added"] += 1

    # Raw event spans may repeat SPEAK after its initial edge. Keep those ticks
    # as CONTINUE once the assistant is already speaking; mask any late idle S.
    for frame_id, frame in list(by_id.items()):
        if (_frame_action(frame, action_key=action_key) != "SPEAK"
                or frame.get(mask_key) is not True or _manual_review(frame)):
            continue
        source = frame.get("target_action_source")
        evidence = frame.get("target_action_evidence")
        evidence = evidence if isinstance(evidence, Mapping) else {}
        local_id = evidence.get("assistant_segment_id") or evidence.get("local_segment_id")
        peer_id = evidence.get("peer_segment_id") or evidence.get("previous_peer_segment_id")
        v5_source = is_v5 and any(
            peer["start_ms"] <= frame_id * _TICK_MS < local["end_ms"]
            for peer, local, _provenance in speak_events.values()
        )
        natural_pairs_for_local = [pair for pair in speak_events if pair[1] == local_id]
        qualified_natural_source = (
            not is_v5 and source in _SPEAK_SEED_SOURCES and bool(natural_pairs_for_local)
            and (peer_id in {pair[0] for pair in natural_pairs_for_local}
                 if peer_id else len(natural_pairs_for_local) == 1)
        )
        if not (qualified_natural_source or v5_source):
            continue
        if frame_id in allowed_speak_frames:
            continue
        if own_active(frame_id):
            _set_temporal_target(
                frame, action_key=action_key, mask_key=mask_key, action="CONTINUE",
                source="native1540_temporal_held_speak_to_continue",
                evidence={"raw_source": source or "v5_measured_speak_event", "frame_id": frame_id},
            )
            report["held_speak_frames_as_continue"] += 1
        else:
            _mask_temporal_target(
                frame, action_key=action_key, mask_key=mask_key,
                reason="raw_speak_outside_pcm_endpoint_window",
                source="native1540_temporal_late_speak_mask",
            )
            report["speak_late_source_frames_masked"] += 1

    yield_events: dict[tuple[str, str], tuple[Mapping[str, Any], Mapping[str, Any], float, float, str]] = {}
    if is_v5:
        utterance_by_id = {segment["id"]: segment for segment in segments if segment["id"]}
        for event in row.get("event_schedule", ()) or ():
            if not isinstance(event, Mapping) or str(event.get("action", "")).upper() != "YIELD":
                continue
            local = utterance_by_id.get(str(event.get("context_utterance_id") or ""))
            peer = utterance_by_id.get(str(event.get("trigger_utterance_id") or ""))
            if local is None or peer is None or local["speaker"] != "assistant" or peer["speaker"] != "user":
                continue
            own_stop = float(event.get("assistant_cut_sample", 0)) * 1000.0 / int(row.get("sample_rate_hz", 16000))
            onset = float(event.get("correction_onset_sample", 0)) * 1000.0 / int(row.get("sample_rate_hz", 16000))
            if (local["start_ms"] + 80.0 <= onset < own_stop
                    and own_stop - onset <= 320.0 and peer["end_ms"] - own_stop >= 160.0):
                yield_events[(local["id"], peer["id"])] = (local, peer, onset, own_stop, "v5_measured_yield_event")
    else:
        for frame in frames:
            if (_frame_action(frame, action_key=action_key) != "YIELD"
                    or frame.get("target_action_source") != _YIELD_SEED_SOURCE):
                continue
            evidence = frame.get("target_action_evidence")
            if not isinstance(evidence, Mapping):
                continue
            local = assistant_segment_by_id(evidence.get("local_segment_id"))
            peer = user_segment_by_id(evidence.get("peer_segment_id"))
            if local is None or peer is None:
                continue
            onset = float(evidence.get("peer_onset_ms", peer["start_ms"]))
            own_stop = float(evidence.get("local_stop_ms", local["end_ms"]))
            if (peer["end_ms"] - own_stop >= 160.0 and own_stop - onset <= 320.0
                    and onset - local["start_ms"] >= 80.0):
                yield_events[(local["id"], peer["id"])] = (local, peer, onset, own_stop, _YIELD_SEED_SOURCE)

    allowed_yield_frames: set[int] = set()
    for (local_id, peer_id), (local, peer, onset, own_stop, provenance) in yield_events.items():
        first = max(0, int(onset // _TICK_MS))
        stop = min(len(rms), int(math.ceil(own_stop / _TICK_MS)))
        voiced = [fid for fid in range(first, stop) if rms[fid] >= rms_floor and own_active(fid)]
        report["yield_events_qualified"] += 1
        if not voiced:
            report["yield_events_without_voiced_overlap"] += 1
            continue
        onset_frame = voiced[0]
        for fid in range(onset_frame, stop):
            tick_start, tick_end = fid * _TICK_MS, (fid + 1) * _TICK_MS
            if not (local["start_ms"] < tick_end and local["end_ms"] > tick_start):
                continue
            frame = ensure_frame(fid)
            if frame is None:
                continue
            if protected(frame, "YIELD"):
                report["reviewed_conflicts_protected"] += 1
                continue
            _set_temporal_target(
                frame, action_key=action_key, mask_key=mask_key, action="YIELD",
                source="native1540_pcm_rms_yield_onset_until_own_stop",
                evidence={"local_segment_id": local_id, "peer_segment_id": peer_id,
                          "first_voiced_frame_id": onset_frame, "local_stop_ms": own_stop,
                          "source_qualification": provenance, "rms_floor": rms_floor,
                          "includes_partial_final_own_active_tick": tick_end > own_stop},
            )
            allowed_yield_frames.add(fid)
            report["yield_target_frames_added"] += 1

    # The old annotation-timestamp YIELD edge can precede actual PCM onset.
    for frame_id, frame in list(by_id.items()):
        if (_frame_action(frame, action_key=action_key) == "YIELD"
                and frame.get("target_action_source") == _YIELD_SEED_SOURCE
                and frame_id not in allowed_yield_frames and not _manual_review(frame)):
            if own_active(frame_id):
                _set_temporal_target(
                    frame, action_key=action_key, mask_key=mask_key, action="CONTINUE",
                    source="native1540_temporal_yield_before_voiced_onset_continue",
                    evidence={"frame_id": frame_id, "rms_floor": rms_floor},
                )
            else:
                _mask_temporal_target(
                    frame, action_key=action_key, mask_key=mask_key,
                    reason="yield_outside_own_active_pcm_overlap",
                    source="native1540_temporal_yield_mask",
                )

    row[key] = sorted(frames, key=lambda frame: int(frame[id_key]))
    row["native1540_temporal_supervision_view"] = {
        "schema": "native1540_pcm_temporal_supervision_v1",
        "tick_ms": _TICK_MS,
        "user_pcm_rms_floor": float(rms_floor),
        "rms_input": "mono physical user channel, float32 decoded once, causal 16 kHz resample, fixed 80 ms ticks with zero-padded partial final tick",
        "target_causality_preserved": True,
        "source_rows_and_raw_target_values_preserved": True,
    }
    report["supervised_frames_after"] = sum(
        frame.get(mask_key) is True and _frame_action(frame, action_key=action_key) is not None
        for frame in frames
    )
    report["frames_after"] = len(frames)
    return row, report


def _load_all_channel_frame_rms(row: Mapping[str, Any]) -> np.ndarray:
    import soundfile as sf

    audio_path = row.get("audio_path")
    if not isinstance(audio_path, str) or not Path(audio_path).is_file():
        raise FileNotFoundError(f"prepared audio is missing: {audio_path}")
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    sample_rate = int(sample_rate)
    if sample_rate <= 0 or audio.ndim != 2 or audio.shape[1] <= 0:
        raise ValueError(f"invalid PCM input: {audio_path}")
    if sample_rate != 16000:
        from math import gcd
        from scipy.signal import resample_poly
        common = gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // common, sample_rate // common, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(audio).all():
        raise ValueError(f"non-finite PCM input: {audio_path}")
    frame_samples = 1280
    frame_count = max(1, (len(audio) + frame_samples - 1) // frame_samples)
    rms = np.zeros((frame_count, audio.shape[1]), dtype=np.float64)
    for frame_id in range(frame_count):
        start = frame_id * frame_samples
        chunk = np.zeros((frame_samples, audio.shape[1]), dtype=np.float32)
        valid = audio[start:min(len(audio), start + frame_samples)]
        if len(valid):
            chunk[:len(valid)] = valid
        rms[frame_id] = np.sqrt(np.mean(chunk.astype(np.float64) ** 2, axis=0))
    return rms


def _user_channel(row: Mapping[str, Any], channel_count: int) -> int:
    channel_map = row.get("channel_map")
    if not isinstance(channel_map, Mapping):
        raise ValueError(f"{row.get('conversation_id')}: explicit channel_map is required")
    matches = [
        int(key) for key, value in channel_map.items()
        if str(key).isdigit() and str(value).strip().lower() == "user"
    ]
    if len(matches) != 1 or not 0 <= matches[0] < channel_count:
        raise ValueError(f"{row.get('conversation_id')}: expected one explicit user PCM channel")
    return matches[0]


def _supervised_action_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        key, action_key, mask_key, _id_key, _is_v5 = _target_fields(row)
        source = str(row.get("source") or "unknown")
        bucket = counts.setdefault(source, {})
        for frame in row.get(key, ()) or ():
            if not isinstance(frame, Mapping) or frame.get(mask_key) is not True:
                continue
            action = _frame_action(frame, action_key=action_key)
            if action:
                bucket[action] = bucket.get(action, 0) + 1
    return counts


def _feature_rows_missing_exactly(
    rows: Sequence[Mapping[str, Any]], existing_keys: set[tuple[str, int]],
) -> dict[str, Any]:
    missing_by_source_action: dict[str, dict[str, int]] = {}
    all_supervised = 0
    missing = 0
    for row in rows:
        key, action_key, mask_key, id_key, _is_v5 = _target_fields(row)
        conversation_id = str(row.get("conversation_id") or "")
        for frame in row.get(key, ()) or ():
            if not isinstance(frame, Mapping) or frame.get(mask_key) is not True:
                continue
            frame_id = _frame_id(frame, id_key=id_key)
            if frame_id is None:
                continue
            all_supervised += 1
            if (conversation_id, frame_id) in existing_keys:
                continue
            missing += 1
            source = str(row.get("source") or "unknown")
            action = _frame_action(frame, action_key=action_key) or "unknown"
            bucket = missing_by_source_action.setdefault(source, {})
            bucket[action] = bucket.get(action, 0) + 1
    return {
        "new_supervised_frame_keys": all_supervised,
        "feature_rows_missing_exactly": missing,
        "missing_by_source_action": missing_by_source_action,
    }


def _derive_native1540_rows(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    feature_dir: Path,
    *,
    rms_floor: float = _RMS_FLOOR,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    output: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    cache: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {
        "schema": "native1540_pcm_temporal_supervision_view_v1",
        "tick_ms": _TICK_MS,
        "user_pcm_rms_floor": float(rms_floor),
        "feature_values_changed": False,
        "feature_schema_changed": False,
        "raw_prepared_source_rows_modified": False,
        "test_split_read": False,
        "source_action_counts_before_by_split": {},
        "source_action_counts_after_by_split": {},
        "exact_feature_key_coverage_by_split": {},
        "events_by_split_source": {},
        "audio_files_decoded_by_split_source": {},
        "role_and_audio_channel_checks_by_split_source": {},
    }
    for split in ("train", "validation"):
        rows = list(rows_by_split[split])
        existing_path = feature_dir / f"{split}.json"
        if not existing_path.is_file():
            raise FileNotFoundError(f"current feature metadata is missing: {existing_path}")
        metadata = json.loads(existing_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, list):
            raise ValueError(f"{existing_path}: expected an array of frame metadata")
        existing_keys: set[tuple[str, int]] = set()
        for feature_row in metadata:
            if not isinstance(feature_row, Mapping):
                raise ValueError(f"{existing_path}: feature metadata rows must be objects")
            conversation_id, frame_id = feature_row.get("conversation_id"), feature_row.get("frame_id")
            if not isinstance(conversation_id, str) or type(frame_id) is not int or frame_id < 0:
                raise ValueError(f"{existing_path}: row lacks exact conversation_id/frame_id")
            existing_keys.add((conversation_id, frame_id))

        report["source_action_counts_before_by_split"][split] = _supervised_action_counts(rows)
        frame_reports: list[dict[str, Any]] = []
        audio_counts: dict[str, int] = {}
        role_counts: dict[str, dict[str, int]] = {}
        for source_row in rows:
            source = str(source_row.get("source") or "unknown")
            audio_key = str(Path(str(source_row.get("audio_path") or "")).resolve())
            all_channel_rms = cache.get(audio_key)
            if all_channel_rms is None:
                all_channel_rms = _load_all_channel_frame_rms(source_row)
                cache[audio_key] = all_channel_rms
                audio_counts[source] = audio_counts.get(source, 0) + 1
            user_channel = _user_channel(source_row, all_channel_rms.shape[1])
            key, _action_key, _mask_key, _id_key, is_v5 = _target_fields(source_row)
            if is_v5:
                segments = source_row.get("utterances", ()) or ()
                role_valid = bool(segments) and all(
                    isinstance(segment, Mapping)
                    and str(segment.get("speaker") or "").strip().lower() in {"user", "assistant"}
                    for segment in segments
                )
            else:
                segments = source_row.get("segments", ()) or ()
                role_valid = bool(segments) and all(
                    isinstance(segment, Mapping)
                    and str(segment.get("perspective_role") or segment.get("role") or "").strip().lower()
                    in {"user", "assistant"}
                    for segment in segments
                    if segment.get("audio_channel_verified") is True and segment.get("time_mask", True) is True
                )
            role_bucket = role_counts.setdefault(source, {"rows": 0, "verified_speaker_roles": 0, "explicit_user_channel": 0})
            role_bucket["rows"] += 1
            role_bucket["verified_speaker_roles"] += int(role_valid)
            role_bucket["explicit_user_channel"] += 1
            derived, frame_report = apply_native1540_pcm_temporal_targets(
                source_row, all_channel_rms[:, user_channel], rms_floor=rms_floor
            )
            output[split].append(derived)
            frame_reports.append(frame_report)
        # Deduplicate the read cache by audio path while retaining per-split
        # counts of prepared rows whose source audio decoded successfully.
        report["audio_files_decoded_by_split_source"][split] = audio_counts
        report["role_and_audio_channel_checks_by_split_source"][split] = role_counts
        events: dict[str, dict[str, int]] = {}
        for item in frame_reports:
            bucket = events.setdefault(item["source"], {})
            for name, value in item.items():
                if name in {"conversation_id", "split", "source", "v5_speaker_field_used"}:
                    continue
                if isinstance(value, int):
                    bucket[name] = bucket.get(name, 0) + value
        report["events_by_split_source"][split] = events
        report["source_action_counts_after_by_split"][split] = _supervised_action_counts(output[split])
        report["exact_feature_key_coverage_by_split"][split] = _feature_rows_missing_exactly(
            output[split], existing_keys
        )
    report["source_prepared_rows"] = {split: len(rows_by_split[split]) for split in ("train", "validation")}
    report["derived_rows"] = {split: len(output[split]) for split in ("train", "validation")}
    report["audio_integrity"] = "all source WAVs decoded as finite PCM; no source digest recomputed or replaced"
    report["feature_cache_used_for_exact_key_comparison"] = str(feature_dir.resolve())
    return output, report


def derive_native1540_missing_feature_supplement(
    temporal_view_dir: Path,
    output_dir: Path,
    existing_feature_dir: Path,
) -> dict[str, Any]:
    """Write affected TRAIN/VAL rows with extraction masks limited to missing keys.

    Each selected conversation keeps its complete temporal/audio context and
    every target record. Only the copied action masks change, so Stage2
    extraction advances from frame zero but emits only exact missing targets.
    """
    temporal_view_dir = temporal_view_dir.resolve()
    output_dir = output_dir.resolve()
    existing_feature_dir = existing_feature_dir.resolve()
    if output_dir in {temporal_view_dir, existing_feature_dir}:
        raise ValueError("supplement output must be distinct from temporal view and feature cache")
    if output_dir.exists():
        raise FileExistsError(f"missing-feature supplement output must be fresh: {output_dir}")

    report: dict[str, Any] = {
        "schema": "native1540_missing_feature_extraction_supplement_v1",
        "temporal_view_dir": str(temporal_view_dir),
        "existing_feature_dir": str(existing_feature_dir),
        "output_dir": str(output_dir),
        "test_split_read": False,
        "source_view_rows_modified": False,
        "feature_values_changed": False,
        "feature_schema_changed": False,
        "splits": {},
    }
    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    view_manifest_path = temporal_view_dir / "preparation_manifest.json"
    if not view_manifest_path.is_file():
        raise FileNotFoundError(f"temporal supervision view manifest missing: {view_manifest_path}")
    view_manifest = json.loads(view_manifest_path.read_text(encoding="utf-8"))
    if view_manifest.get("schema") != "native1540_pcm_temporal_supervision_view_v1":
        raise ValueError("missing-feature supplement requires native1540 PCM temporal supervision view")
    for split in ("train", "validation"):
        prepared_path = temporal_view_dir / f"{split}.jsonl"
        feature_metadata_path = existing_feature_dir / f"{split}.json"
        if not prepared_path.is_file() or not feature_metadata_path.is_file():
            raise FileNotFoundError(f"missing {split} temporal view or feature metadata")
        source_rows = read_jsonl(prepared_path)
        cache_rows = json.loads(feature_metadata_path.read_text(encoding="utf-8"))
        if not isinstance(cache_rows, list):
            raise ValueError(f"{feature_metadata_path}: expected an array of feature rows")
        cache_keys: set[tuple[str, int]] = set()
        for item in cache_rows:
            if not isinstance(item, Mapping):
                raise ValueError(f"{feature_metadata_path}: feature rows must be objects")
            conversation_id, frame_id = item.get("conversation_id"), item.get("frame_id")
            if not isinstance(conversation_id, str) or type(frame_id) is not int or frame_id < 0:
                raise ValueError(f"{feature_metadata_path}: invalid exact feature key")
            key = (conversation_id, frame_id)
            if key in cache_keys:
                raise ValueError(f"{feature_metadata_path}: duplicate exact feature key {key}")
            cache_keys.add(key)

        selected_keys: set[tuple[str, int]] = set()
        view_keys: set[tuple[str, int]] = set()
        missing_by_source_action: dict[str, dict[str, int]] = {}
        causal_steps = 0
        affected_groups: set[str] = set()
        for source_row in source_rows:
            if source_row.get("split") != split:
                raise ValueError(f"{prepared_path}: row split differs from file split")
            row_view = source_row.get("native1540_temporal_supervision_view")
            if not isinstance(row_view, Mapping) or row_view.get("schema") != "native1540_pcm_temporal_supervision_v1":
                raise ValueError(f"{source_row.get('conversation_id')}: missing native1540 temporal supervision provenance")
            conversation_id = str(source_row.get("conversation_id") or "")
            if not conversation_id:
                raise ValueError(f"{prepared_path}: row lacks conversation_id")
            key_name, action_key, mask_key, id_key, _is_v5 = _target_fields(source_row)
            targets = source_row.get(key_name, ()) or ()
            row_targets: list[tuple[Mapping[str, Any], tuple[str, int], str]] = []
            for frame in targets:
                if not isinstance(frame, Mapping):
                    raise ValueError(f"{conversation_id}: target record must be an object")
                frame_id = _frame_id(frame, id_key=id_key)
                if frame_id is None:
                    raise ValueError(f"{conversation_id}: invalid target frame id")
                key = (conversation_id, frame_id)
                action = _frame_action(frame, action_key=action_key)
                if frame.get(mask_key) is True and action is not None:
                    if key in view_keys:
                        raise ValueError(f"{split}: duplicate supervised temporal-view key {key}")
                    view_keys.add(key)
                    row_targets.append((frame, key, action))
            missing = [(frame, key, action) for frame, key, action in row_targets if key not in cache_keys]
            if not missing:
                continue
            row_missing_keys = {key for _, key, _ in missing}
            copied = copy.deepcopy(dict(source_row))
            copied_targets = copied[key_name]
            for target in copied_targets:
                frame_id = _frame_id(target, id_key=id_key)
                key = (conversation_id, frame_id) if frame_id is not None else None
                was_supervised = bool(target.get(mask_key) is True and _frame_action(target, action_key=action_key))
                keep = key in row_missing_keys
                if was_supervised and not keep:
                    target.setdefault("native1540_supplement_original_action_mask", target.get(mask_key))
                    target[mask_key] = False
                    if key_name == "frame_targets" or "target_action_mask" in target:
                        target["target_action_mask"] = False
                    target["action_mask"] = False
            copied["native1540_missing_feature_supplement"] = {
                "schema": "native1540_missing_feature_extraction_supplement_v1",
                "selected_missing_frame_ids": sorted(key[1] for key in row_missing_keys),
                "full_source_timeline_retained": True,
            }
            outputs[split].append(copied)
            missing_ids = [key[1] for _, key, _ in missing]
            causal_steps += max(missing_ids) + 1
            affected_groups.add(str(source_row.get("source_group_id") or conversation_id))
            selected_keys.update(key for _, key, _ in missing)
            for _, _, action in missing:
                source = str(source_row.get("source") or "unknown")
                action_counts = missing_by_source_action.setdefault(source, {})
                action_counts[action] = action_counts.get(action, 0) + 1

        selected_target_keys = view_keys
        report["splits"][split] = {
            "source_conversations": len(source_rows),
            "affected_conversations": len(outputs[split]),
            "affected_source_groups": len(affected_groups),
            "temporal_view_supervised_keys": len(selected_target_keys),
            "cached_keys": len(cache_keys),
            "reusable_cached_target_keys": len(selected_target_keys & cache_keys),
            "missing_exact_keys_selected": len(selected_keys),
            "missing_by_source_action": missing_by_source_action,
            "obsolete_cached_keys_excluded_from_new_view": len(cache_keys - selected_target_keys),
            "final_selected_keys_after_merge": len(selected_target_keys),
            "causal_input_frame_steps": causal_steps,
            "full_timeline_rows_retained_for_affected_conversations": len(outputs[split]),
        }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(output_dir.name + ".staging")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        for split in ("train", "validation"):
            _write_jsonl(staging / f"{split}.jsonl", outputs[split])
        report["affected_conversations"] = {
            split: len(outputs[split]) for split in ("train", "validation")
        }
        report["missing_exact_keys"] = sum(
            int(report["splits"][split]["missing_exact_keys_selected"])
            for split in ("train", "validation")
        )
        report["causal_input_frame_steps"] = sum(
            int(report["splits"][split]["causal_input_frame_steps"])
            for split in ("train", "validation")
        )
        (staging / "preparation_manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    except BaseException:
        raise
    return report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def derive_prepared_dataset(
    prepared_dir: Path,
    output_dir: Path,
    *,
    supplement_output_dir: Path | None = None,
    existing_feature_dir: Path | None = None,
    native1540_temporal_view: bool = False,
    native1540_missing_feature_supplement: bool = False,
    rms_floor: float = _RMS_FLOOR,
) -> dict:
    prepared_dir = prepared_dir.resolve()
    output_dir = output_dir.resolve()
    if prepared_dir == output_dir:
        raise ValueError("output-dir must differ from the historical prepared dataset")
    if native1540_temporal_view and native1540_missing_feature_supplement:
        raise ValueError("native1540 temporal view and missing-feature supplement modes are mutually exclusive")
    if output_dir.exists():
        raise FileExistsError(f"derived prepared output must be fresh: {output_dir}")
    if supplement_output_dir is not None:
        if native1540_temporal_view:
            raise ValueError("native1540 temporal view cannot write the legacy endpoint supplement")
        supplement_output_dir = supplement_output_dir.resolve()
        if supplement_output_dir in {prepared_dir, output_dir}:
            raise ValueError("supplement output must be a distinct directory")
        if supplement_output_dir.exists():
            raise FileExistsError(f"supplement prepared output must be fresh: {supplement_output_dir}")
    if (native1540_temporal_view or native1540_missing_feature_supplement) and existing_feature_dir is None:
        raise ValueError("native1540 view/supplement requires --existing-feature-dir for exact frame coverage")
    existing_keys: set[tuple[str, int]] = set()
    if existing_feature_dir is not None and not native1540_temporal_view:
        feature_root = existing_feature_dir.resolve()
        for split in ("train", "validation"):
            metadata_path = feature_root / f"{split}.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"existing feature metadata is missing: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, list):
                raise ValueError(f"{metadata_path}: expected a JSON array")
            for row in metadata:
                if not isinstance(row, dict):
                    raise ValueError(f"{metadata_path}: feature rows must be objects")
                conversation_id = row.get("conversation_id")
                frame_id = row.get("frame_id")
                if not isinstance(conversation_id, str) or not conversation_id or type(frame_id) is not int or frame_id < 0:
                    raise ValueError(f"{metadata_path}: feature row lacks an exact conversation/frame key")
                existing_keys.add((conversation_id, frame_id))
    source_rows: list[dict] = []
    for split in ("train", "validation"):
        path = prepared_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Candidate E split is missing: {path}")
        rows = read_jsonl(path)
        if any(row.get("split") != split for row in rows):
            raise ValueError(f"{path}: row split differs from file split")
        source_rows.extend(rows)
    if native1540_missing_feature_supplement:
        return derive_native1540_missing_feature_supplement(
            prepared_dir,
            output_dir,
            existing_feature_dir.resolve(),
        )
    if native1540_temporal_view:
        rows_by_split = {
            split: [row for row in source_rows if row.get("split") == split]
            for split in ("train", "validation")
        }
        by_split, report = _derive_native1540_rows(
            rows_by_split,
            existing_feature_dir.resolve(),
            rms_floor=rms_floor,
        )
    else:
        transformed, report = relabel_prepared_timestamp_actions(source_rows)
        by_split = {
            split: [row for row in transformed if row.get("split") == split]
            for split in ("train", "validation")
        }
    if sum(map(len, by_split.values())) != len(source_rows):
        raise RuntimeError("prepared timestamp conversion lost or duplicated source rows")
    endpoint_rows = {} if native1540_temporal_view else {
        split: [
            (row, [
                frame for frame in (row.get("frames", ()) or ())
                if frame.get("target_action_source") == "timestamp_derived_last_vocal_complete_user_turn"
                and frame.get("action_mask") is True
            ])
            for row in by_split[split]
        ]
        for split in ("train", "validation")
    }
    supplement_by_split: dict[str, list[dict]] = {"train": [], "validation": []}
    endpoint_counts: dict[str, dict[str, int]] = {}
    for split, conversations in endpoint_rows.items():
        total = 0
        missing = 0
        for row, endpoint_frames in conversations:
            total += len(endpoint_frames)
            absent = [
                frame for frame in endpoint_frames
                if (str(row["conversation_id"]), int(frame["frame_id"])) not in existing_keys
            ]
            if not absent:
                continue
            missing += len(absent)
            supplement_row = dict(row)
            supplement_row["frames"] = [dict(frame) for frame in absent]
            supplement_row["timestamp_speak_endpoint_supplement_only"] = True
            supplement_by_split[split].append(supplement_row)
        endpoint_counts[split] = {
            "endpoint_targets": total,
            "feature_rows_missing_exactly": missing,
            "supplement_conversations": len(supplement_by_split[split]),
        }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(output_dir.name + ".staging")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        for split, rows in by_split.items():
            _write_jsonl(staging / f"{split}.jsonl", rows)
        report["source_prepared_dir"] = str(prepared_dir)
        report["output_prepared_dir"] = str(output_dir)
        if not native1540_temporal_view:
            report["timestamp_speak_endpoint_targets_by_split"] = endpoint_counts
        report["existing_feature_dir_used_for_exact_key_comparison"] = (
            str(existing_feature_dir.resolve()) if existing_feature_dir is not None else None
        )
        report["split_counts"] = {
            split: {
                "rows": len(rows),
                "source_groups": len({str(row.get("source_group_id") or row["conversation_id"]) for row in rows}),
                "source_counts": {
                    source: sum(row.get("source") == source for row in rows)
                    for source in sorted({str(row.get("source", "unknown")) for row in rows})
                },
            }
            for split, rows in by_split.items()
        }
        (staging / "preparation_manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
        if supplement_output_dir is not None:
            supplement_staging = supplement_output_dir.with_name(supplement_output_dir.name + ".staging")
            supplement_staging.mkdir(parents=False, exist_ok=False)
            for split, rows in supplement_by_split.items():
                _write_jsonl(supplement_staging / f"{split}.jsonl", rows)
            supplement_report = {
                "schema": "native_control6_timestamp_speak_endpoint_supplement_v1",
                "source_prepared_dir": str(prepared_dir),
                "derived_prepared_dir": str(output_dir),
                "existing_feature_dir": str(existing_feature_dir.resolve()) if existing_feature_dir else None,
                "exact_conversation_frame_targets_by_split": {
                    split: [
                        {"conversation_id": row["conversation_id"], "source_group_id": row.get("source_group_id"),
                         "frame_id": int(frame["frame_id"]), "source": row.get("source")}
                        for row in supplement_by_split[split]
                        for frame in row["frames"]
                    ]
                    for split in ("train", "validation")
                },
                "target_counts_by_split": {
                    split: {
                        "conversations": len(supplement_by_split[split]),
                        "feature_rows": sum(len(row["frames"]) for row in supplement_by_split[split]),
                        "source_groups": len({str(row.get("source_group_id") or row["conversation_id"]) for row in supplement_by_split[split]}),
                    }
                    for split in ("train", "validation")
                },
                "preserves_source_audio_and_segments": True,
                "training_targets_only": True,
            }
            (supplement_staging / "preparation_manifest.json").write_text(
                json.dumps(supplement_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(supplement_staging, supplement_output_dir)
    except BaseException:
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--supplement-output-dir", type=Path)
    parser.add_argument("--existing-feature-dir", type=Path)
    parser.add_argument(
        "--native1540-temporal-view",
        action="store_true",
        help="derive an immutable PCM-RMS action-target view for train/validation without extracting features or training",
    )
    parser.add_argument(
        "--native1540-missing-feature-supplement",
        action="store_true",
        help="copy affected conversation timelines while masking extraction targets to exact missing feature keys",
    )
    parser.add_argument("--rms-floor", type=float, default=_RMS_FLOOR)
    args = parser.parse_args()
    print(json.dumps(
        derive_prepared_dataset(
            args.prepared_dir,
            args.output_dir,
            supplement_output_dir=args.supplement_output_dir,
            existing_feature_dir=args.existing_feature_dir,
            native1540_temporal_view=args.native1540_temporal_view,
            native1540_missing_feature_supplement=args.native1540_missing_feature_supplement,
            rms_floor=args.rms_floor,
        ),
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
