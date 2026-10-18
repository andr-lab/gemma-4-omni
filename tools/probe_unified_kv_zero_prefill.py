"""Real pair-mode probe for listener Q0 silence and Q2 user-audio commits.

Run this inside the existing Mondaini runtime while its pair service is already
up. The probe uses ``client/duplex_stream_e2e.py --mode pair`` in-process and
does not start or restart a server. It prepends exactly 60 seconds of digital
silence to the repository's existing spoken ``pair_seed.wav`` fixture, then
preserves the unified-KV telemetry fields that the normal client log projection
would otherwise omit.

Example (from the repository root on Mondaini):
    python tools/probe_unified_kv_zero_prefill.py \
      --ws-url ws://127.0.0.1:8092/v1/realtime?duplex=1 \
      --output-dir /tmp/unified-kv-zero-prefill

Use ``--self-test`` for the CPU-only parser checks, or ``--analyze-events`` to
analyze an existing pair client's ``events.jsonl`` without connecting to a
server.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import wave
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "artifacts" / "conversation_quality_20261014" / "inputs" / "pair_seed.wav"
DEFAULT_WS_URL = "ws://127.0.0.1:8092/v1/realtime?duplex=1"
FRAME_MS = 80
INPUT_RATE = 16_000
FRAME_SAMPLES = INPUT_RATE * FRAME_MS // 1000
LOGICAL_Q_CAPACITY = 2
VOICE_ACTIVITY_RMS_THRESHOLD = 0.002

_TELEMETRY_FIELDS = (
    "logical_q_len",
    "ephemeral_reason",
    "queue_audio_for_next_turn",
    "kv_rollback_tokens",
    "kv_rollback_enqueue_ms",
    "listener_session_id",
    "listener_frame_id",
    "listener_output_frame_id",
    "listener_cache_position_before",
    "listener_cache_position_after",
    "listener_valid_history_tokens_before",
    "listener_valid_history_tokens_after",
    "native_cache_current_position_after",
    "native_cache_valid_kv_len_after",
    "policy_decision_at_ns",
    "thinker_first_logits_ready_at_ns",
    "first_thinker_token_at_ns",
    "intent_source_frame_id",
    "intent_endpoint_frame_id",
    "input_first_frame_id",
    "input_last_frame_id",
    "action",
    "effective_action",
    "wake_action",
    "response_action",
    "response_created_at_monotonic_ns",
    "server_listener_step_start_ns",
    "server_listener_step_end_ns",
    "unified_native_profile",
    "thinker_runtime_metrics",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _int_field(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _number_field(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct * len(ordered)) - 1))
    return round(ordered[index], 3)


def _frame_timing(sent: list[dict[str, Any]]) -> dict[str, Any]:
    stamps = [_int_field(row, "sent_at_ns") for row in sent]
    valid = [stamp for stamp in stamps if stamp is not None]
    gaps = [((right - left) / 1_000_000.0) for left, right in zip(valid, valid[1:])]
    return {
        "frames": len(sent),
        "interval_ms_p50": round(statistics.median(gaps), 3) if gaps else None,
        "interval_ms_p95": _percentile(gaps, 0.95),
        "intervals_60_to_100ms": sum(60.0 <= gap <= 100.0 for gap in gaps),
        "interval_count": len(gaps),
    }


def _cursor_and_q_check(
    rows: list[dict[str, Any]], expected_q: int, *, allow_assistant_q0: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not rows:
        return {"rows": 0, "status": "not_measurable"}, ["no consumed ACK rows"]
    required = (
        "logical_q_len",
        "queue_audio_for_next_turn",
        "kv_rollback_tokens",
        "kv_rollback_enqueue_ms",
        "listener_cache_position_before",
        "listener_cache_position_after",
        "listener_valid_history_tokens_before",
        "listener_valid_history_tokens_after",
        "native_cache_current_position_after",
        "native_cache_valid_kv_len_after",
    )
    missing = sorted({field for row in rows for field in required if field not in row})
    if missing:
        return {
            "rows": len(rows),
            "status": "not_measurable",
            "missing_fields": missing,
        }, [f"telemetry missing: {', '.join(missing)}"]

    previous_cache_position: int | None = None
    previous_valid_history: int | None = None
    previous_phase: str | None = None
    q_counts: dict[str, int] = defaultdict(int)
    phase_counts: dict[str, int] = defaultdict(int)
    unresolved_phase_rows: list[Any] = []
    rollback_values: list[int] = []
    rollback_enqueue_ms: list[float] = []
    for row in rows:
        frame_id = row.get("frame_id", row.get("client_frame_id"))
        q_len = _int_field(row, "logical_q_len")
        before = _int_field(row, "listener_cache_position_before")
        after = _int_field(row, "listener_cache_position_after")
        valid_before = _int_field(row, "listener_valid_history_tokens_before")
        valid_after = _int_field(row, "listener_valid_history_tokens_after")
        native_after = _int_field(row, "native_cache_current_position_after")
        native_valid_after = _int_field(row, "native_cache_valid_kv_len_after")
        rolled_back = _int_field(row, "kv_rollback_tokens")
        rollback_enqueue = _number_field(row, "kv_rollback_enqueue_ms")
        ephemeral_reason = row.get("ephemeral_reason")
        reason_missing = "ephemeral_reason" not in row or ephemeral_reason is None
        queue_for_next_turn = row.get("queue_audio_for_next_turn")
        if None in (q_len, before, after, valid_before, valid_after, native_after, native_valid_after, rolled_back, rollback_enqueue):
            errors.append(f"frame {frame_id}: malformed logical-Q/cursor telemetry")
            continue
        if not isinstance(queue_for_next_turn, bool):
            errors.append(f"frame {frame_id}: malformed queue_audio_for_next_turn telemetry")
            continue
        if ephemeral_reason is not None and not isinstance(ephemeral_reason, str):
            errors.append(f"frame {frame_id}: malformed ephemeral_reason telemetry")
            continue

        if q_len == 0 and reason_missing:
            phase = "q0_phase_unresolved"
            unresolved_phase_rows.append(frame_id)
        elif q_len == 0 and ephemeral_reason == "prespeech_silence" and not queue_for_next_turn:
            phase = "prespeech_q0"
        elif q_len == 0 and ephemeral_reason == "assistant_turn_open" and queue_for_next_turn:
            phase = "assistant_q0"
        elif q_len == LOGICAL_Q_CAPACITY and reason_missing and not queue_for_next_turn:
            phase = "user_q2"
        else:
            phase = "unknown"
            errors.append(
                f"frame {frame_id}: unsupported phase/Q contract "
                f"(logical_q_len={q_len}, ephemeral_reason={ephemeral_reason!r}, "
                f"queue_audio_for_next_turn={queue_for_next_turn!r})"
            )
        q_counts[str(q_len)] += 1
        phase_counts[phase] += 1
        rollback_values.append(rolled_back)
        rollback_enqueue_ms.append(rollback_enqueue)
        assistant_exception = allow_assistant_q0 and phase == "assistant_q0"
        unresolved_q0 = phase == "q0_phase_unresolved"
        phase_is_valid_for_check = (
            phase in {"prespeech_q0", "q0_phase_unresolved"} if expected_q == 0
            else phase == "user_q2" or assistant_exception or unresolved_q0
        )
        if q_len != expected_q and not assistant_exception and not (
            unresolved_q0 and allow_assistant_q0 and q_len == 0
        ):
            errors.append(f"frame {frame_id}: logical_q_len={q_len}, expected {expected_q}")
        expected_rollback = (
            LOGICAL_Q_CAPACITY - q_len
            if phase_is_valid_for_check else LOGICAL_Q_CAPACITY - expected_q
        )
        if rolled_back != expected_rollback:
            errors.append(
                f"frame {frame_id}: kv_rollback_tokens={rolled_back}, "
                f"expected {expected_rollback}"
            )
        expected_row_q = q_len if assistant_exception or unresolved_q0 else expected_q
        if after != before + expected_row_q:
            errors.append(
                f"frame {frame_id}: listener cache cursor {before}->{after} "
                f"is not Q={expected_row_q}"
            )
        if valid_after != valid_before + expected_row_q:
            errors.append(
                f"frame {frame_id}: valid history {valid_before}->{valid_after} "
                f"is not Q={expected_row_q}"
            )
        if native_after != after or native_valid_after != valid_after:
            errors.append(
                f"frame {frame_id}: native cursor/valid ({native_after},{native_valid_after}) "
                f"does not match listener ({after},{valid_after})"
            )
        continuity_phase = "prespeech_q0" if expected_q == 0 else "user_q2"
        unresolved_prespeech = (
            expected_q == 0 and unresolved_q0 and not queue_for_next_turn
        )
        if phase == continuity_phase or unresolved_prespeech:
            if previous_phase == continuity_phase:
                if previous_cache_position is not None and before != previous_cache_position:
                    errors.append(
                        f"frame {frame_id}: cache cursor discontinuity "
                        f"{previous_cache_position}->{before}"
                    )
                if previous_valid_history is not None and valid_before != previous_valid_history:
                    errors.append(
                        f"frame {frame_id}: valid-history discontinuity "
                        f"{previous_valid_history}->{valid_before}"
                    )
            previous_cache_position = after
            previous_valid_history = valid_after
            previous_phase = continuity_phase
        else:
            # Assistant Q0 is an ephemeral policy observation. Thinker text
            # commits may move the cache between these rows, and branch close
            # or cancellation may rewind it before later user Q2 rows.
            previous_cache_position = None
            previous_valid_history = None
            previous_phase = phase
    return {
        "rows": len(rows),
        "status": "failed" if errors else "not_measurable" if unresolved_phase_rows else "verified",
        "unresolved_phase_frame_ids": unresolved_phase_rows,
        "logical_q_counts": dict(q_counts),
        "phase_counts": dict(phase_counts),
        "rollback_tokens_min": min(rollback_values) if rollback_values else None,
        "rollback_tokens_max": max(rollback_values) if rollback_values else None,
        "rollback_enqueue_ms_p50": round(statistics.median(rollback_enqueue_ms), 3)
        if rollback_enqueue_ms else None,
        "rollback_enqueue_ms_p95": _percentile(rollback_enqueue_ms, 0.95),
        "rollback_enqueue_ms_max": round(max(rollback_enqueue_ms), 3)
        if rollback_enqueue_ms else None,
        "final_cache_position": previous_cache_position,
        "final_valid_history_tokens": previous_valid_history,
    }, errors


def _silence_response_check(
    events: list[dict[str, Any]], *, silence_start_at_ms: float | None,
    speech_start_at_ms: float | None,
) -> tuple[dict[str, Any], list[str]]:
    """Check for any response/Thinker activity during the input-silence window.

    ``at_ms`` is stamped by the pair client for both sent frames and received
    events, so this comparison stays entirely within one client monotonic clock.
    Server monotonic timestamps are intentionally not compared with it.
    """
    if (silence_start_at_ms is None or speech_start_at_ms is None
            or speech_start_at_ms <= silence_start_at_ms):
        return {
            "status": "not_measurable",
            "reason": "could not establish client-clock silence and speech boundaries",
            "silence_start_at_ms": silence_start_at_ms,
            "speech_start_at_ms": speech_start_at_ms,
            "responses_in_silence": [],
            "clock_scope": "pair-client at_ms only",
        }, ["missing or invalid client-clock silence boundary"]

    direct_response_types = {
        "response.created",
        "response.thinker_metrics",
        "response.output_text.delta",
        "response.output_text.done",
        "response.audio.delta",
        "response.audio.done",
        "response.audio_transcript.delta",
        "response.audio_transcript.done",
        "response.done",
    }
    activity: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in events:
        kind = row.get("type")
        if kind == "response.action_selected":
            actions = {
                str(row.get(key)).strip().upper()
                for key in ("action", "effective_action", "response_action")
                if isinstance(row.get(key), str)
            }
            is_response_activity = "SPEAK" in actions
        else:
            is_response_activity = (
                kind in direct_response_types
                or any(key in row for key in (
                    "thinker_first_logits_ready_at_ns",
                    "first_thinker_token_at_ns",
                ))
            )
        if not is_response_activity:
            continue
        at_ms = _number_field(row, "at_ms")
        if at_ms is None:
            errors.append(
                f"{kind or 'response event'} for speaker {row.get('speaker')!r} "
                "has no client at_ms timestamp"
            )
            continue
        if silence_start_at_ms <= at_ms < speech_start_at_ms:
            activity.append({
                "type": kind,
                "speaker": row.get("speaker"),
                "response_id": row.get("response_id"),
                "at_ms": at_ms,
                "action": row.get("effective_action", row.get("response_action", row.get("action"))),
            })

    if activity:
        errors.extend(
            f"{item['type']} from speaker {item['speaker']!r} at {item['at_ms']} ms "
            "occurred during input silence"
            for item in activity
        )
    status = "failed" if activity else "not_measurable" if errors else "verified"
    return {
        "status": status,
        "reason": None if not errors else errors[0],
        "silence_start_at_ms": silence_start_at_ms,
        "speech_start_at_ms": speech_start_at_ms,
        "responses_in_silence": activity,
        "clock_scope": "pair-client at_ms only; no client/server clock subtraction",
    }, errors


def _first_int(rows: list[dict[str, Any]], key: str) -> int | None:
    values = [value for row in rows if (value := _int_field(row, key)) is not None]
    return min(values) if values else None


def _response_timings(
    events: list[dict[str, Any]], spoken_turn_first_frame: int, *, speaker: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        if row.get("speaker") != speaker:
            continue
        response_id = row.get("response_id")
        if isinstance(response_id, str) and response_id:
            grouped[response_id].append(row)

    measurements: list[dict[str, Any]] = []
    for response_id, rows in grouped.items():
        actions: set[str] = set()
        for row in rows:
            # wake_action describes the policy/wake side of a response and can
            # differ from the action actually applied by the response protocol.
            # Only applied/response action fields may establish a real SPEAK.
            for key in ("action", "effective_action", "response_action"):
                value = row.get(key)
                if isinstance(value, str):
                    actions.add(value.strip().upper())
        if "SPEAK" not in actions:
            continue
        decision = _first_int(rows, "policy_decision_at_ns")
        logits = _first_int(rows, "thinker_first_logits_ready_at_ns")
        first_token = _first_int(rows, "first_thinker_token_at_ns")
        endpoint_frame = _first_int(rows, "intent_endpoint_frame_id")
        input_last_frame = _first_int(rows, "input_last_frame_id")
        utterance_end = endpoint_frame if endpoint_frame is not None else input_last_frame
        relative_to_spoken_turn = (
            utterance_end >= spoken_turn_first_frame if utterance_end is not None else None
        )
        backwards = (
            (decision is not None and logits is not None and logits < decision)
            or (logits is not None and first_token is not None and first_token < logits)
            or (decision is not None and first_token is not None and first_token < decision)
        )
        decision_to_logits = (
            round((logits - decision) / 1_000_000.0, 3)
            if decision is not None and logits is not None and not backwards else None
        )
        decision_to_first_token = (
            round((first_token - decision) / 1_000_000.0, 3)
            if decision is not None and first_token is not None and not backwards else None
        )
        logits_to_first_token = (
            round((first_token - logits) / 1_000_000.0, 3)
            if logits is not None and first_token is not None and not backwards else None
        )
        measurements.append({
            "response_id": response_id,
            "speaker": next((row.get("speaker") for row in rows if row.get("speaker")), None),
            "intent_endpoint_frame_id": endpoint_frame,
            "input_last_frame_id": input_last_frame,
            "relative_to_spoken_turn": relative_to_spoken_turn,
            "policy_decision_at_ns": decision,
            "thinker_first_logits_ready_at_ns": logits,
            "first_thinker_token_at_ns": first_token,
            "timestamps_monotonic": not backwards,
            "decision_to_logits_ms": decision_to_logits,
            "decision_to_first_token_ms": decision_to_first_token,
            "logits_to_first_token_ms": logits_to_first_token,
        })

    relevant = [row for row in measurements if row["relative_to_spoken_turn"] is True]
    if not relevant:
        if not measurements:
            reason = "no SPEAK action was emitted"
            outcome = "no_speak_or_stall"
        elif any(row["relative_to_spoken_turn"] is None for row in measurements):
            reason = "SPEAK was emitted but could not be tied to the spoken turn by endpoint frame ID"
            outcome = "speak_without_turn_attribution"
        else:
            reason = "SPEAK was emitted only before the spoken user turn"
            outcome = "premature_speak_before_user_audio"
        return {
            "status": "not_measurable",
            "reason": reason,
            "outcome": outcome,
            "responses": measurements,
            "target_decision_to_first_token_ms_lt_15": None,
            "clock_scope": "server monotonic_ns only; client timestamps are never subtracted from server timestamps",
        }
    measured = next(
        (row for row in relevant if row["decision_to_first_token_ms"] is not None),
        None,
    )
    if measured is None:
        return {
            "status": "not_measurable",
            "reason": "SPEAK was selected but policy/first-token timestamps were incomplete",
            "outcome": "speak_without_complete_timing",
            "responses": measurements,
            "target_decision_to_first_token_ms_lt_15": None,
            "clock_scope": "server monotonic_ns only; client timestamps are never subtracted from server timestamps",
        }
    return {
        "status": "measured",
        "reason": None,
        "outcome": "speak_emitted",
        "responses": measurements,
        "decision_to_logits_ms": measured["decision_to_logits_ms"],
        "decision_to_first_token_ms": measured["decision_to_first_token_ms"],
        "logits_to_first_token_ms": measured["logits_to_first_token_ms"],
        "clock_scope": "server monotonic_ns only; client timestamps are never subtracted from server timestamps",
        "target_decision_to_first_token_ms_lt_15": (
            measured["decision_to_first_token_ms"] < 15.0
        ),
    }


def analyze_events(
    events: list[dict[str, Any]],
    *,
    silence_frames: int,
    speech_frames: int,
    seed_speaker: str = "A",
    fixture_duration_s: float | None = None,
) -> dict[str, Any]:
    sent_by_frame: dict[int, dict[str, Any]] = {}
    ack_by_client_frame: dict[int, dict[str, Any]] = {}
    consumed_by_ack_frame: dict[int, dict[str, Any]] = {}
    for row in events:
        if row.get("speaker") != seed_speaker:
            continue
        if row.get("type") == "probe.input_frame_sent" and row.get("source_response_id") == "seed":
            frame_id = _int_field(row, "frame_id")
            if frame_id is not None:
                sent_by_frame[frame_id] = row
        elif row.get("type") == "probe.input_frame_ack":
            frame_id = _int_field(row, "client_frame_id")
            if frame_id is not None:
                ack_by_client_frame[frame_id] = row
        elif row.get("type") == "input_audio_buffer.consumed":
            frame_id = _int_field(row, "frame_id")
            if frame_id is not None:
                consumed_by_ack_frame[frame_id] = row

    seed_ids = sorted(sent_by_frame)
    explicit_silence_ids = seed_ids[:silence_frames]
    fixture_ids = seed_ids[silence_frames:silence_frames + speech_frames]
    fixture_sent = [sent_by_frame[index] for index in fixture_ids]
    first_active_offset = next((
        index for index, row in enumerate(fixture_sent)
        if (_number_field(row, "rms") or 0.0) >= VOICE_ACTIVITY_RMS_THRESHOLD
    ), None)
    leading_fixture_silence = (
        len(fixture_sent) if first_active_offset is None else first_active_offset
    )
    preonset_ids = explicit_silence_ids + fixture_ids[:leading_fixture_silence]
    spoken_ids = fixture_ids[leading_fixture_silence:]
    silence_sent = [sent_by_frame[index] for index in preonset_ids]
    speech_sent = [sent_by_frame[index] for index in spoken_ids]
    silence_start_at_ms = (
        _number_field(silence_sent[0], "at_ms") if silence_sent else None
    )
    speech_start_at_ms = (
        _number_field(speech_sent[0], "at_ms") if speech_sent else None
    )

    def consumed_for(client_frame_ids: list[int]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for client_frame_id in client_frame_ids:
            ack = ack_by_client_frame.get(client_frame_id)
            server_frame_id = _int_field(ack, "ack_frame_id") if ack is not None else None
            if server_frame_id is None:
                continue
            consumed = consumed_by_ack_frame.get(server_frame_id)
            if consumed is not None:
                rows.append(consumed)
        return rows

    silence_acks = consumed_for(preonset_ids)
    speech_acks = consumed_for(spoken_ids)

    silence_check, silence_errors = _cursor_and_q_check(silence_acks, 0)
    speech_check, speech_errors = _cursor_and_q_check(
        speech_acks, LOGICAL_Q_CAPACITY, allow_assistant_q0=True,
    )
    silence_response_check, silence_response_errors = _silence_response_check(
        events,
        silence_start_at_ms=silence_start_at_ms,
        speech_start_at_ms=speech_start_at_ms,
    )
    if len(explicit_silence_ids) != silence_frames:
        silence_errors.append(
            f"only {len(explicit_silence_ids)}/{silence_frames} explicit-silence seed frames were sent"
        )
    if len(fixture_ids) != speech_frames:
        speech_errors.append(f"only {len(fixture_ids)}/{speech_frames} fixture frames were sent")
    if len(silence_acks) != len(preonset_ids):
        silence_errors.append(
            f"only {len(silence_acks)}/{len(preonset_ids)} pre-onset consumed events were logged"
        )
    if len(speech_acks) != len(spoken_ids):
        speech_errors.append(
            f"only {len(speech_acks)}/{len(spoken_ids)} post-onset consumed events were logged"
        )
    first_post_onset = speech_acks[0] if speech_acks else None
    first_post_onset_is_user_q2 = (
        first_post_onset is not None
        and _int_field(first_post_onset, "logical_q_len") == LOGICAL_Q_CAPACITY
        and first_post_onset.get("ephemeral_reason") is None
        and first_post_onset.get("queue_audio_for_next_turn") is False
    )
    if silence_acks and first_post_onset_is_user_q2:
        last_silence_position = _int_field(
            silence_acks[-1], "listener_cache_position_after"
        )
        first_speech_position = _int_field(
            first_post_onset, "listener_cache_position_before"
        )
        last_silence_valid = _int_field(
            silence_acks[-1], "listener_valid_history_tokens_after"
        )
        first_speech_valid = _int_field(
            first_post_onset, "listener_valid_history_tokens_before"
        )
        if last_silence_position != first_speech_position:
            speech_errors.append(
                "cache cursor does not continue from pre-onset Q0 into first user Q2: "
                f"{last_silence_position}->{first_speech_position}"
            )
        if last_silence_valid != first_speech_valid:
            speech_errors.append(
                "valid-history cursor does not continue from pre-onset Q0 into first user Q2: "
                f"{last_silence_valid}->{first_speech_valid}"
            )
    if silence_check["status"] != "not_measurable" and silence_errors:
        silence_check["status"] = "failed"
    if speech_check["status"] != "not_measurable" and speech_errors:
        speech_check["status"] = "failed"

    cache_status = (
        "failed" if "failed" in {silence_check["status"], speech_check["status"]}
        else "verified" if silence_check["status"] == speech_check["status"] == "verified"
        else "not_measurable"
    )
    return {
        "scenario": {
            "silence_seconds": silence_frames * FRAME_MS / 1000.0,
            "silence_frames_expected": silence_frames,
        "fixture_frames_expected": speech_frames,
        "fixture_original_duration_s": round(fixture_duration_s, 4)
        if fixture_duration_s is not None else None,
        "fixture_streamed_duration_s": round(speech_frames * FRAME_MS / 1000.0, 3),
        "fixture_leading_preonset_frames": leading_fixture_silence,
        "fixture_voiced_frames_over_rms_threshold": sum(
            (_number_field(row, "rms") or 0.0) >= VOICE_ACTIVITY_RMS_THRESHOLD
            for row in fixture_sent
        ),
        "fixture_voiced_duration_s_at_80ms_frames": round(
            sum((_number_field(row, "rms") or 0.0) >= VOICE_ACTIVITY_RMS_THRESHOLD
                for row in fixture_sent) * FRAME_MS / 1000.0,
            3,
        ),
        "fixture_voice_activity_rms_threshold": VOICE_ACTIVITY_RMS_THRESHOLD,
        "post_onset_user_frames_expected": len(spoken_ids),
            "seed_speaker": seed_speaker,
            "frame_ms": FRAME_MS,
        },
        "input_timing": {
            "preonset_silence": _frame_timing(silence_sent),
            "speech": _frame_timing(speech_sent),
        },
        "cache": {
            "status": cache_status,
            "silence": silence_check,
            "silence_errors": silence_errors,
            "speech": speech_check,
            "speech_errors": speech_errors,
        },
        "silence_response": {
            **silence_response_check,
            "errors": silence_response_errors,
        },
        "ttft": _response_timings(
            events,
            silence_frames + leading_fixture_silence,
            speaker=seed_speaker,
        ),
    }


def _load_fixture_16k(path: Path):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from client.duplex_stream_e2e import read_audio_16k

    samples = read_audio_16k(path)
    if samples.ndim != 1 or not samples.size:
        raise ValueError(f"spoken fixture is empty or not mono: {path}")
    duration_s = samples.size / INPUT_RATE
    peak = float(abs(samples).max())
    rms = float((samples.astype("float64") ** 2).mean() ** 0.5)
    if not 6.0 <= duration_s <= 10.0:
        raise ValueError(f"expected an approximately 8-second spoken fixture, got {duration_s:.3f}s")
    if peak < 0.02 or rms < 0.001:
        raise ValueError(f"spoken fixture is silent or too quiet: peak={peak:.5f}, rms={rms:.5f}")
    return samples, {"duration_s": round(duration_s, 4), "peak": round(peak, 6), "rms": round(rms, 6)}


def compose_probe_wav(fixture: Path, output: Path, silence_seconds: float) -> tuple[int, int, dict[str, Any]]:
    if silence_seconds <= 0 or not math.isclose(silence_seconds * 1000 / FRAME_MS,
                                                 round(silence_seconds * 1000 / FRAME_MS),
                                                 abs_tol=1e-9):
        raise ValueError("silence duration must be a positive multiple of the 80 ms input frame")
    silence_frames = int(round(silence_seconds * 1000 / FRAME_MS))
    speech, speech_metrics = _load_fixture_16k(fixture)
    speech_frames = math.ceil(speech.size / FRAME_SAMPLES)
    import numpy as np

    padded_speech = np.zeros(speech_frames * FRAME_SAMPLES, dtype=np.float32)
    padded_speech[:speech.size] = speech
    silence = np.zeros(silence_frames * FRAME_SAMPLES, dtype=np.float32)
    pcm = np.concatenate((silence, padded_speech))
    pcm16 = np.rint(np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(INPUT_RATE)
        wav.writeframes(pcm16.tobytes())
    return silence_frames, speech_frames, speech_metrics


def _preserve_unified_kv_telemetry(client_module) -> tuple[Any, Any]:
    old_project = client_module._project_server_event_for_log
    old_ack_project = client_module._project_input_ack_fields

    def project(event: dict[str, Any]) -> dict[str, Any]:
        result = old_project(event)
        for field in _TELEMETRY_FIELDS:
            if field in event:
                result[field] = event[field]
        return result

    def project_ack(event: dict[str, Any]) -> dict[str, Any]:
        result = old_ack_project(event)
        for field in _TELEMETRY_FIELDS:
            if field in event:
                result[field] = event[field]
        return result

    client_module._project_server_event_for_log = project
    client_module._project_input_ack_fields = project_ack
    return old_project, old_ack_project


def _run_pair_client(
    *, ws_url: str, audio: Path, output_dir: Path, max_duration_s: float,
    max_turns: int, max_created_responses: int,
) -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from client import duplex_stream_e2e as client

    old_project, old_ack_project = _preserve_unified_kv_telemetry(client)
    old_argv = sys.argv
    sys.argv = [
        str(ROOT / "client" / "duplex_stream_e2e.py"),
        "--mode", "pair",
        "--ws-url", ws_url,
        "--audio", str(audio),
        "--output-dir", str(output_dir),
        "--turn-policy", "gemma_native_control6",
        "--max-turns", str(max_turns),
        "--max-created-responses", str(max_created_responses),
        "--max-duration-s", str(max_duration_s),
    ]
    try:
        return int(client.main())
    finally:
        sys.argv = old_argv
        client._project_server_event_for_log = old_project
        client._project_input_ack_fields = old_ack_project


def _self_test() -> None:
    events: list[dict[str, Any]] = []
    cursor = 120
    valid = 120
    for frame_id, q_len in enumerate((0, 0, 2, 2)):
        sent_at = 1_000_000_000 + frame_id * FRAME_MS * 1_000_000
        events.append({
            "type": "probe.input_frame_sent", "speaker": "A", "frame_id": frame_id,
            "source_response_id": "seed", "sent_at_ns": sent_at,
            "at_ms": frame_id * FRAME_MS,
            "rms": 0.0 if frame_id < 2 else 0.1,
        })
        consumed = {
            "type": "input_audio_buffer.consumed", "speaker": "A", "frame_id": frame_id,
            "logical_q_len": q_len, "kv_rollback_tokens": 2 - q_len,
            "ephemeral_reason": "prespeech_silence" if q_len == 0 else None,
            "queue_audio_for_next_turn": False,
            "kv_rollback_enqueue_ms": 0.05,
            "listener_cache_position_before": cursor,
            "listener_cache_position_after": cursor + q_len,
            "listener_valid_history_tokens_before": valid,
            "listener_valid_history_tokens_after": valid + q_len,
            "native_cache_current_position_after": cursor + q_len,
            "native_cache_valid_kv_len_after": valid + q_len,
        }
        events.append(consumed)
        events.append({
            "type": "probe.input_frame_ack", "speaker": "A", "client_frame_id": frame_id,
            "ack_frame_id": frame_id,
        })
        cursor += q_len
        valid += q_len
    events.extend([
        {
            "type": "response.action_selected", "speaker": "A", "response_id": "r1",
            "response_action": "SPEAK", "policy_decision_at_ns": 2_000_000_000,
            "intent_endpoint_frame_id": 3, "at_ms": 300.0,
        },
        {
            "type": "response.thinker_metrics", "speaker": "A", "response_id": "r1",
            "thinker_first_logits_ready_at_ns": 2_005_000_000,
            "first_thinker_token_at_ns": 2_012_000_000, "at_ms": 301.0,
        },
    ])
    result = analyze_events(events, silence_frames=2, speech_frames=2)
    assert result["cache"]["status"] == "verified", result
    assert result["silence_response"]["status"] == "verified", result
    assert result["ttft"]["status"] == "measured", result
    assert result["ttft"]["decision_to_first_token_ms"] == 12.0, result
    assert result["ttft"]["target_decision_to_first_token_ms_lt_15"] is True, result

    no_speak = analyze_events(events[:8], silence_frames=2, speech_frames=2)
    assert no_speak["ttft"]["status"] == "not_measurable", no_speak
    assert no_speak["ttft"]["outcome"] == "no_speak_or_stall", no_speak

    wake_only_speak = analyze_events([
        {
            "type": "response.action_selected", "speaker": "A", "response_id": "r2",
            "wake_action": "SPEAK", "response_action": "LISTEN",
            "intent_endpoint_frame_id": 3, "at_ms": 300.0,
        },
    ], silence_frames=2, speech_frames=2)
    assert wake_only_speak["ttft"]["outcome"] == "no_speak_or_stall", wake_only_speak

    other_peer_speak = analyze_events(events[:8] + [
        {
            "type": "response.action_selected", "speaker": "B", "response_id": "peer-r1",
            "response_action": "SPEAK", "policy_decision_at_ns": 2_000_000_000,
            "intent_endpoint_frame_id": 3, "at_ms": 50.0,
        },
        {
            "type": "response.thinker_metrics", "speaker": "B", "response_id": "peer-r1",
            "thinker_first_logits_ready_at_ns": 2_005_000_000,
            "first_thinker_token_at_ns": 2_012_000_000, "at_ms": 51.0,
        },
    ], silence_frames=2, speech_frames=2)
    assert other_peer_speak["ttft"]["outcome"] == "no_speak_or_stall", other_peer_speak
    assert other_peer_speak["silence_response"]["status"] == "failed", other_peer_speak

    leading_pause: list[dict[str, Any]] = []
    cursor = valid = 40
    q_values = (0, 0, 0, 2, 2, 2)
    rms_values = (0.0, 0.0, 0.0, 0.1, 0.0, 0.1)
    for frame_id, (q_len, rms) in enumerate(zip(q_values, rms_values, strict=True)):
        sent_at = 3_000_000_000 + frame_id * FRAME_MS * 1_000_000
        leading_pause.append({
            "type": "probe.input_frame_sent", "speaker": "A", "frame_id": frame_id,
            "source_response_id": "seed", "sent_at_ns": sent_at,
            "at_ms": frame_id * FRAME_MS, "rms": rms,
        })
        leading_pause.append({
            "type": "input_audio_buffer.consumed", "speaker": "A", "frame_id": frame_id,
            "logical_q_len": q_len, "kv_rollback_tokens": 2 - q_len,
            "ephemeral_reason": "prespeech_silence" if q_len == 0 else None,
            "queue_audio_for_next_turn": False,
            "kv_rollback_enqueue_ms": 0.05,
            "listener_cache_position_before": cursor,
            "listener_cache_position_after": cursor + q_len,
            "listener_valid_history_tokens_before": valid,
            "listener_valid_history_tokens_after": valid + q_len,
            "native_cache_current_position_after": cursor + q_len,
            "native_cache_valid_kv_len_after": valid + q_len,
        })
        leading_pause.append({
            "type": "probe.input_frame_ack", "speaker": "A", "client_frame_id": frame_id,
            "ack_frame_id": frame_id,
        })
        cursor += q_len
        valid += q_len
    leading_result = analyze_events(leading_pause, silence_frames=2, speech_frames=4)
    assert leading_result["cache"]["status"] == "verified", leading_result
    assert leading_result["scenario"]["fixture_leading_preonset_frames"] == 1, leading_result
    assert leading_result["scenario"]["fixture_voiced_frames_over_rms_threshold"] == 2, leading_result
    assert leading_result["scenario"]["post_onset_user_frames_expected"] == 3, leading_result

    # Small copied cursor values from the real smoke2 rows: initial Q0 190->190,
    # user Q2 190->192, assistant Q0 314->314, and post-cancel Q2 332->334.
    # The assistant-phase observations are separated by committed Thinker text.
    mixed_rows = [
        {
            "frame_id": 4, "logical_q_len": 2, "ephemeral_reason": None,
            "queue_audio_for_next_turn": False, "kv_rollback_tokens": 0,
            "kv_rollback_enqueue_ms": 0.05,
            "listener_cache_position_before": 190, "listener_cache_position_after": 192,
            "listener_valid_history_tokens_before": 190, "listener_valid_history_tokens_after": 192,
            "native_cache_current_position_after": 192, "native_cache_valid_kv_len_after": 192,
        },
        {
            "frame_id": 65, "logical_q_len": 0, "ephemeral_reason": "assistant_turn_open",
            "queue_audio_for_next_turn": True, "kv_rollback_tokens": 2,
            "kv_rollback_enqueue_ms": 0.05,
            "listener_cache_position_before": 314, "listener_cache_position_after": 314,
            "listener_valid_history_tokens_before": 314, "listener_valid_history_tokens_after": 314,
            "native_cache_current_position_after": 314, "native_cache_valid_kv_len_after": 314,
        },
        {
            "frame_id": 66, "logical_q_len": 0, "ephemeral_reason": "assistant_turn_open",
            "queue_audio_for_next_turn": True, "kv_rollback_tokens": 2,
            "kv_rollback_enqueue_ms": 0.05,
            "listener_cache_position_before": 320, "listener_cache_position_after": 320,
            "listener_valid_history_tokens_before": 320, "listener_valid_history_tokens_after": 320,
            "native_cache_current_position_after": 320, "native_cache_valid_kv_len_after": 320,
        },
        {
            "frame_id": 75, "logical_q_len": 2, "ephemeral_reason": None,
            "queue_audio_for_next_turn": False, "kv_rollback_tokens": 0,
            "kv_rollback_enqueue_ms": 0.05,
            "listener_cache_position_before": 332, "listener_cache_position_after": 334,
            "listener_valid_history_tokens_before": 332, "listener_valid_history_tokens_after": 334,
            "native_cache_current_position_after": 334, "native_cache_valid_kv_len_after": 334,
        },
    ]
    mixed_check, mixed_errors = _cursor_and_q_check(
        mixed_rows, LOGICAL_Q_CAPACITY, allow_assistant_q0=True,
    )
    assert mixed_check["status"] == "verified", (mixed_check, mixed_errors)

    broken_ephemeral = [dict(mixed_rows[1], listener_cache_position_after=316)]
    broken_check, broken_errors = _cursor_and_q_check(
        broken_ephemeral, LOGICAL_Q_CAPACITY, allow_assistant_q0=True,
    )
    assert broken_check["status"] == "failed" and broken_errors, (broken_check, broken_errors)

    missing_phase = dict(mixed_rows[1])
    missing_phase.pop("ephemeral_reason")
    unknown_check, unknown_errors = _cursor_and_q_check(
        [missing_phase], LOGICAL_Q_CAPACITY, allow_assistant_q0=True,
    )
    assert unknown_check["status"] == "not_measurable", (unknown_check, unknown_errors)
    assert unknown_check["unresolved_phase_frame_ids"] == [65], (unknown_check, unknown_errors)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run CPU-only parser and invariant checks")
    parser.add_argument("--analyze-events", type=Path, help="analyze an existing pair events.jsonl and exit")
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--audio", type=Path, default=DEFAULT_FIXTURE,
                        help="existing spoken WAV fixture; defaults to artifacts/.../pair_seed.wav")
    parser.add_argument("--output-dir", type=Path,
                        help="pair run output directory; required for a live probe")
    parser.add_argument("--silence-seconds", type=float, default=60.0)
    parser.add_argument("--max-duration-s", type=float, default=110.0,
                        help="pair wall deadline, including the 60 s silence and response drain")
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--max-created-responses", type=int, default=4)
    parser.add_argument("--speech-frames", type=int,
                        help="override the spoken frame count inferred from --audio during analysis")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.self_test:
        _self_test()
        print("probe self-test passed")
        return 0
    if args.analyze_events is not None:
        rows = _read_jsonl(args.analyze_events)
        silence_frames = int(round(args.silence_seconds * 1000 / FRAME_MS))
        speech_samples, fixture_metrics = _load_fixture_16k(args.audio.expanduser().resolve())
        speech_frames = args.speech_frames or math.ceil(speech_samples.size / FRAME_SAMPLES)
        print(json.dumps(analyze_events(rows, silence_frames=silence_frames,
                                        speech_frames=speech_frames,
                                        fixture_duration_s=fixture_metrics["duration_s"]),
                          ensure_ascii=False, indent=2))
        return 0
    if args.output_dir is None:
        _parser().error("--output-dir is required unless --self-test or --analyze-events is used")
    if args.max_duration_s <= args.silence_seconds + 8.0:
        _parser().error("--max-duration-s must leave time after the silence + spoken fixture for response/drain")
    if args.max_turns < 1 or args.max_created_responses < 1:
        _parser().error("response budgets must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_audio = output_dir / "unified_kv_zero_prefill_input.wav"
    silence_frames, speech_frames, fixture_metrics = compose_probe_wav(
        args.audio.expanduser().resolve(), probe_audio, args.silence_seconds,
    )
    print(json.dumps({
        "probe": "unified_kv_zero_prefill",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "server_must_already_be_running": True,
        "ws_url": args.ws_url,
        "turn_policy": "gemma_native_control6",
        "forced_response_create": False,
        "fixture": str(args.audio.expanduser().resolve()),
        "fixture_metrics": fixture_metrics,
        "probe_audio": str(probe_audio),
        "silence_frames": silence_frames,
        "speech_frames": speech_frames,
        "pair_client": "client/duplex_stream_e2e.py --mode pair",
    }, ensure_ascii=False))
    client_rc = _run_pair_client(
        ws_url=args.ws_url,
        audio=probe_audio,
        output_dir=output_dir,
        max_duration_s=args.max_duration_s,
        max_turns=args.max_turns,
        max_created_responses=args.max_created_responses,
    )
    events_path = output_dir / "events.jsonl"
    if not events_path.is_file():
        print(json.dumps({
            "status": "not_measurable",
            "reason": "pair client produced no events.jsonl",
            "client_exit_code": client_rc,
        }, ensure_ascii=False, indent=2))
        return max(1, client_rc)
    events = _read_jsonl(events_path)
    result = analyze_events(
        events,
        silence_frames=silence_frames,
        speech_frames=speech_frames,
        fixture_duration_s=fixture_metrics["duration_s"],
    )
    result["client_exit_code"] = client_rc
    result["events_jsonl"] = str(events_path)
    result["probe_audio"] = str(probe_audio)
    result["pair_report"] = str(output_dir / "report.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return client_rc


if __name__ == "__main__":
    raise SystemExit(main())
