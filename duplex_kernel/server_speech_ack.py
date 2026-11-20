"""Speech ACK, packet sequence, and client-consumed PCM helpers."""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from duplex_kernel.audio_stream import AudioPacket, CODEC_SAMPLE_RATE
from duplex_kernel.engine import ListenerBacklogAdmission
from duplex_kernel.live_say import LiveQwenSayStream
from duplex_kernel.playout import PlayoutFrame
from duplex_kernel.turn_policy import ControlAction

from duplex_kernel.server_pcm_tracker import (
    _canonical_client_consumed_pcm_ack,
    _client_consumed_pcm_ack_duplicate_status,
    _commit_client_consumed_segment,
    _complete_client_pcm_segment,
    _record_client_consumed_pcm,
    _record_emitted_pcm_packet,
    _require_closed_tts_codec_eos,
    _turn_local_vocalized_prefix_snapshot,
    _wait_for_listener_backlog_admission,
)
from duplex_kernel.server_speech_ack_streams import (
    _ClosedTTSSegment,
    _ack_emitted_audio_packet,
    _ack_response_speech_packet,
    _make_finish_audio_segment_callback,
    _register_response_speech_ack_streams,
    _response_speech_ack_lock,
    _response_speech_ack_resources,
    _retire_response_speech_ack_streams,
)

_REALTIME_THINKER_METRICS_MAX_BYTES = 8 * 1024
_REALTIME_THINKER_METRICS_MAX_FIELDS = 32
_REALTIME_THINKER_METRICS_MAX_STRING_CHARS = 128


def _realtime_metric_scalar(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if len(str(value)) <= 32 else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:_REALTIME_THINKER_METRICS_MAX_STRING_CHARS]
    return None


def _project_realtime_metric_summary(
    values: Mapping[str, Any],
    *,
    depth: int = 0,
) -> dict[str, Any]:
    """Keep bounded summary fields while omitting per-token collections."""
    if depth >= 3:
        return {}
    projected: dict[str, Any] = {}
    for raw_key, value in values.items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key[:96]
        normalized_key = key.lower()
        if (
            "per_token" in normalized_key
            or "token_timings" in normalized_key
            or "token_latencies" in normalized_key
            or "token_metrics" in normalized_key
            or "token_steps" in normalized_key
            or normalized_key.endswith("_steps")
        ):
            continue
        if isinstance(value, Mapping):
            nested = _project_realtime_metric_summary(value, depth=depth + 1)
            if nested:
                projected[key] = nested
        else:
            scalar = _realtime_metric_scalar(value)
            if scalar is not None or value is None:
                projected[key] = scalar
        if len(projected) >= _REALTIME_THINKER_METRICS_MAX_FIELDS:
            break
    return projected


def _project_realtime_native_profile(value: Any) -> dict[str, Any]:
    profile = value if isinstance(value, Mapping) else {}
    projected: dict[str, Any] = {}
    for raw_key, field_value in profile.items():
        if not isinstance(raw_key, str) or raw_key in {"append_steps", "sampling_steps"}:
            continue
        if isinstance(field_value, Mapping):
            summary = _project_realtime_metric_summary(field_value)
            if summary:
                projected[raw_key[:96]] = summary
        else:
            scalar = _realtime_metric_scalar(field_value)
            if scalar is not None or field_value is None:
                projected[raw_key[:96]] = scalar
    append_steps = profile.get("append_steps")
    if isinstance(append_steps, (list, tuple)):
        projected["append_steps_count"] = len(append_steps)
    sampling_steps = profile.get("sampling_steps")
    if isinstance(sampling_steps, (list, tuple)):
        projected["sampling_steps_count"] = len(sampling_steps)
    return projected


def _compact_realtime_event_size(event: Mapping[str, Any]) -> int:
    return len(
        json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )


def _project_realtime_thinker_metrics(
    metrics: Mapping[str, Any],
    *,
    response_id: str,
) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    if "policy_decision_at_ns" in metrics:
        projected["policy_decision_at_ns"] = _realtime_metric_scalar(
            metrics["policy_decision_at_ns"]
        )
    native_profile = metrics.get("unified_native_profile")
    if native_profile is not None:
        projected["unified_native_profile"] = _project_realtime_native_profile(
            native_profile
        )
    runtime_metrics = metrics.get("thinker_runtime_metrics")
    if isinstance(runtime_metrics, Mapping):
        projected["thinker_runtime_metrics"] = _project_realtime_metric_summary(
            runtime_metrics
        )

    for raw_key, value in metrics.items():
        if not isinstance(raw_key, str):
            continue
        if raw_key not in {
            "policy_decision_at_ns",
            "unified_native_profile",
            "thinker_runtime_metrics",
            "type",
            "response_id",
        }:
            scalar = _realtime_metric_scalar(value)
            if scalar is not None or value is None:
                projected[raw_key[:96]] = scalar
        if len(projected) >= _REALTIME_THINKER_METRICS_MAX_FIELDS:
            break

    event: dict[str, Any] = {
        "type": "response.thinker_metrics",
        "response_id": response_id[:128],
        **projected,
    }
    event["type"] = "response.thinker_metrics"
    event["response_id"] = response_id[:128]
    if _compact_realtime_event_size(event) < _REALTIME_THINKER_METRICS_MAX_BYTES:
        return event

    # Oversized metadata is trimmed only after the normal bounded projection.
    protected_top_level = {
        "type",
        "response_id",
        "policy_decision_at_ns",
        "unified_native_profile",
    }
    for key in reversed(tuple(event)):
        if key in protected_top_level or key.endswith("_ns") or key.endswith("_at_ns"):
            continue
        event.pop(key, None)
        if _compact_realtime_event_size(event) < _REALTIME_THINKER_METRICS_MAX_BYTES:
            return event

    profile = event.get("unified_native_profile")
    if isinstance(profile, dict):
        protected_profile = {
            key for key in profile
            if key in {"append_steps_count", "sampling_steps_count", "policy_decision_at_ns"}
            or key.startswith("token0")
            or key.endswith("_ns")
            or key.endswith("_at_ns")
        }
        for key in reversed(tuple(profile)):
            if key in protected_profile:
                continue
            profile.pop(key, None)
            if _compact_realtime_event_size(event) < _REALTIME_THINKER_METRICS_MAX_BYTES:
                return event

    # Generated response IDs are short; this final fallback also bounds every
    # retained field if unexpected scalar metadata exceeds the wire budget.
    fallback: dict[str, Any] = {
        "type": "response.thinker_metrics",
        "response_id": response_id[:128],
    }
    policy_timestamp = projected.get("policy_decision_at_ns")
    if policy_timestamp is not None:
        fallback["policy_decision_at_ns"] = policy_timestamp
    fallback_profile: dict[str, Any] = {}
    if isinstance(profile, Mapping):
        for key, field_value in profile.items():
            if (
                key in {"append_steps_count", "sampling_steps_count", "policy_decision_at_ns"}
                or (
                    key.startswith("token0")
                    and key.endswith("_ns")
                    and isinstance(field_value, (int, float))
                )
            ):
                fallback_profile[key] = field_value
    if fallback_profile:
        fallback["unified_native_profile"] = fallback_profile
    if _compact_realtime_event_size(fallback) >= _REALTIME_THINKER_METRICS_MAX_BYTES:
        essential_profile = {}
        if isinstance(profile, Mapping):
            for key in (
                "policy_decision_at_ns",
                "token0_forward_submit_ns",
                "token0_forward_return_ns",
                "token0_forward_owner_future_wait_ns",
                "token0_forward_owner_queue_wait_ns",
                "append_steps_count",
                "sampling_steps_count",
            ):
                value = profile.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    essential_profile[key] = value
        fallback.pop("unified_native_profile", None)
        if essential_profile:
            fallback["unified_native_profile"] = essential_profile
    return fallback


def _record_server_ack_lock_wait_ms(
    event: dict[str, Any],
    *,
    lock_requested_ns: int,
    lock_acquired_ns: int,
) -> None:
    if event.get("type") != "input_audio_buffer.consumed":
        return
    event["server_ack_lock_wait_ms"] = round(
        max(0, int(lock_acquired_ns) - int(lock_requested_ns)) / 1_000_000.0,
        3,
    )


def _validate_unified_say_packet_frontier(
    branch: Mapping[str, Any],
    token_index: int,
    exclusive_kv_cursor: int,
    *,
    kv_committed: bool,
) -> int:
    """Validate an early or committed packet against its native KV frontier."""
    if isinstance(token_index, bool) or not isinstance(token_index, int) or token_index < 0:
        raise ValueError("unified SAY packet requires a nonnegative token index")
    if (
        isinstance(exclusive_kv_cursor, bool)
        or not isinstance(exclusive_kv_cursor, int)
        or exclusive_kv_cursor < 0
    ):
        raise ValueError("unified SAY packet requires a nonnegative exclusive KV cursor")
    base_cursor = branch.get("spoken_prefix_base_cursor")
    actual_frontier = branch.get("cache_position")
    transition_ids = tuple(branch.get("transition_token_ids", ()))
    transition_cursor = branch.get("spoken_prefix_transition_cursor")
    if (
        isinstance(base_cursor, bool)
        or not isinstance(base_cursor, int)
        or isinstance(actual_frontier, bool)
        or not isinstance(actual_frontier, int)
    ):
        raise RuntimeError("unified SAY branch has no authoritative transition/frontier cursor")
    if transition_ids and transition_cursor != base_cursor:
        raise RuntimeError(
            "unified SAY packet arrived before the native transition suffix was anchored: "
            f"transition_token_count={len(transition_ids)}, "
            f"transition_cursor={transition_cursor}, base_cursor={base_cursor}, "
            f"actual_frontier={actual_frontier}"
        )

    expected_cursor = int(base_cursor) + token_index + 1
    expected_frontier = expected_cursor if kv_committed else expected_cursor - 1
    if int(exclusive_kv_cursor) != expected_cursor or int(actual_frontier) != expected_frontier:
        raise RuntimeError(
            "unified V1 SPEAK packet cursor mismatch: "
            f"base_cursor={base_cursor}, "
            f"transition_token_count={len(transition_ids)}, "
            f"token_index={token_index}, "
            f"packet_cursor={exclusive_kv_cursor}, "
            f"expected_cursor={expected_cursor}, "
            f"expected_frontier={expected_frontier}, "
            f"actual_frontier={actual_frontier}, "
            f"kv_committed={bool(kv_committed)}, "
            f"session_generation_id={branch.get('session_generation_id')}, "
            f"listener_generation_id={branch.get('listener_generation_id')}"
        )
    return expected_cursor

def _publish_thinker_metrics_event(
    values: Mapping[str, Any],
    *,
    response_id: str,
    policy_decision_at_ns: int | None,
    session: Any,
    token: Any,
    cancellation_event: threading.Event | None,
    update_runtime_metrics: Callable[[str, str, Any], None],
    send_event: Callable[[dict[str, Any]], bool],
) -> bool:
    """Keep local Thinker metrics, but drop their WS event after cancellation."""
    metrics = dict(values)
    if policy_decision_at_ns is not None:
        metrics["policy_decision_at_ns"] = int(policy_decision_at_ns)
    update_runtime_metrics(response_id, "thinker_metrics", metrics)

    cancelled = bool(cancellation_event is not None and cancellation_event.is_set())
    if cancelled or not session.is_generation_current(token):
        return False

    event = _project_realtime_thinker_metrics(metrics, response_id=response_id)
    sent = bool(send_event(event))
    if sent:
        return True

    cancelled = bool(cancellation_event is not None and cancellation_event.is_set())
    if cancelled or not session.is_generation_current(token):
        return False
    raise RuntimeError("thinker timing event lost its response generation")

def _record_unified_native_profile_timestamp(
    branch: Any,
    field: str,
    timestamp_ns: int | None = None,
) -> None:
    if not isinstance(branch, dict):
        return
    profile = branch.get("unified_native_profile")
    if not isinstance(profile, dict):
        return
    timestamp = time.monotonic_ns() if timestamp_ns is None else int(timestamp_ns)
    lock = branch.get("unified_native_profile_lock")
    if lock is None:
        profile.setdefault(field, timestamp)
        return
    with lock:
        profile.setdefault(field, timestamp)

def _wait_for_qwen_dynamic_prefill_model_entry(
    entry_event: threading.Event,
    result: Mapping[str, Any],
    *,
    cancelled: Callable[[], bool],
    stopped: Callable[[], bool],
    poll_interval_s: float = 0.02,
) -> int:
    """Wait for actual dynamic ``t.model`` entry, propagating pre-entry failures."""
    while not entry_event.wait(poll_interval_s):
        if cancelled():
            raise RuntimeError("Qwen token-0 prefill was cancelled before entry")
        if stopped():
            raise RuntimeError("Qwen token-0 prefill lost its response generation")
    error = result.get("error")
    if isinstance(error, BaseException):
        raise error
    entry_ns = result.get("qwen_dynamic_prefill_model_entry_at_ns")
    if isinstance(entry_ns, int):
        return int(entry_ns)
    if cancelled():
        raise RuntimeError("Qwen token-0 prefill was cancelled before entry")
    if stopped():
        raise RuntimeError("Qwen token-0 prefill lost its response generation")
    raise RuntimeError("Qwen token-0 prefill signal arrived without dynamic t.model entry")

def _mark_unified_say_packet_committed(
    branch: dict[str, Any],
    live_stream: Any,
    token_index: int,
    exclusive_kv_cursor: int,
    *,
    generation_id: int,
    segment_id: int,
) -> Any:
    """Commit the matching early V1 packet, then release its queued PCM ACKs."""
    lock = branch.get("spoken_prefix_lock")
    if lock is None:
        raise RuntimeError("unified SAY branch has no spoken-prefix lock")
    if (
        isinstance(token_index, bool)
        or not isinstance(token_index, int)
        or token_index < 0
        or isinstance(exclusive_kv_cursor, bool)
        or not isinstance(exclusive_kv_cursor, int)
        or exclusive_kv_cursor < 0
    ):
        raise ValueError("unified SAY commit requires a nonnegative token index and KV cursor")
    with lock:
        try:
            stream_generation_id = int(live_stream.generation_id)
            stream_segment_id = int(live_stream.segment_id)
        except (AttributeError, TypeError, ValueError):
            return None
        if (
            int(branch.get("session_generation_id", -1)) != int(generation_id)
            or stream_generation_id != int(generation_id)
            or stream_segment_id != int(segment_id)
        ):
            return None
        actual_frontier = branch.get("cache_position")
        if (
            isinstance(actual_frontier, bool)
            or not isinstance(actual_frontier, int)
            or int(exclusive_kv_cursor) > int(actual_frontier)
        ):
            raise RuntimeError(
                "native token commit arrived before its KV cursor was committed: "
                f"token_index={token_index}, packet_cursor={exclusive_kv_cursor}, "
                f"actual_frontier={actual_frontier}"
            )
        packet_records = branch.get("say_token_packets", ())
        segment_records = branch.get("segment_token_packets", ())
        newly_tts_consumed: list[dict[str, Any]] = []
        if token_index >= len(packet_records) or token_index >= len(segment_records):
            raise RuntimeError("native token commit has no matching early SAY packet")
        packet = packet_records[token_index]
        record = segment_records[token_index]
        expected_cursor = int(branch["spoken_prefix_base_cursor"]) + token_index + 1
        if (
            int(packet.exclusive_kv_cursor) != expected_cursor
            or int(record.get("token_index", -1)) != token_index
            or int(record.get("exclusive_kv_cursor", -1)) != expected_cursor
            or expected_cursor != exclusive_kv_cursor
            or bool(packet.kv_committed)
            or bool(record.get("kv_committed"))
        ):
            raise RuntimeError("native token commit does not match its pending SAY packet")
        packet_records[token_index] = replace(packet, kv_committed=True)
        record["kv_committed"] = True
    return live_stream.mark_packet_committed(
        exclusive_kv_cursor,
        generation_id=int(generation_id),
        segment_id=int(segment_id),
    )

def _feed_unified_v1_say_packet(
    live_stream: LiveQwenSayStream,
    token_text: str,
    stream_hidden: Any,
    *,
    token_index: int,
    exclusive_kv_cursor: int,
    producer_event: Any | None,
    kv_committed: bool,
    normalized_action: str,
) -> bool:
    """Feed one native V1 packet and seal only the first SPEAK fragment."""
    return live_stream.feed_piece(
        token_text,
        stream_hidden,
        producer_event=producer_event,
        token_index=int(token_index),
        exclusive_kv_cursor=int(exclusive_kv_cursor),
        kv_committed=bool(kv_committed),
        seal_initial_segment=normalized_action == ControlAction.SPEAK.value,
    )

def _publish_acknowledged_speech_prefix_legacy(
    branch: dict[str, Any] | None,
    generation_id: int,
    exclusive_kv_cursor: int,
    cumulative_vocalized_text: str,
) -> bool:
    """Advance canonical V1 speech state at a committed, per-frame ACK frontier."""
    if not isinstance(branch, dict):
        return False
    lock = branch.get("spoken_prefix_lock")
    if lock is None:
        return False
    if (
        isinstance(exclusive_kv_cursor, bool)
        or not isinstance(exclusive_kv_cursor, int)
        or exclusive_kv_cursor < 0
        or not isinstance(cumulative_vocalized_text, str)
    ):
        raise ValueError("acknowledged V1 speech prefix has an invalid cursor or text")
    with lock:
        if int(branch.get("session_generation_id", -1)) != int(generation_id):
            return False
        base_cursor = int(branch.get("spoken_prefix_base_cursor", 0))
        previous_cursor = int(branch.get("spoken_prefix_exclusive_cursor", base_cursor))
        if exclusive_kv_cursor < previous_cursor:
            raise RuntimeError("acknowledged V1 speech cursor moved backwards")
        if exclusive_kv_cursor == previous_cursor:
            if str(branch.get("spoken_prefix_text", "")) != cumulative_vocalized_text:
                raise RuntimeError("same V1 speech cursor was reported with different text")
            return False
        if exclusive_kv_cursor <= base_cursor:
            raise RuntimeError("acknowledged V1 speech cursor did not advance past its transition")

        packet_records = branch.get("say_token_packets", ())
        segment_records = branch.get("segment_token_packets", ())
        token_ids = list(int(value) for value in branch.get("spoken_prefix_token_ids", ()))
        text_parts = list(branch.get("spoken_prefix_text_parts", ()))
        newly_tts_consumed: list[dict[str, Any]] = []
        newly_source_records: list[Any] = []
        cursor = previous_cursor
        for token_index, record in enumerate(segment_records):
            record_cursor = int(record.get("exclusive_kv_cursor", -1))
            if record_cursor <= previous_cursor:
                continue
            if record_cursor > exclusive_kv_cursor:
                break
            if record_cursor != cursor + 1:
                raise RuntimeError("acknowledged V1 speech packets are not contiguous")
            if token_index >= len(packet_records):
                raise RuntimeError("acknowledged V1 speech packet record has no source packet")
            packet = packet_records[token_index]
            if (
                not bool(record.get("kv_committed"))
                or not bool(packet.kv_committed)
                or int(packet.exclusive_kv_cursor) != record_cursor
                or int(packet.token_id) != int(record.get("token_id", -1))
            ):
                raise RuntimeError("V1 speech ACK crossed an uncommitted Gemma packet")
            token_ids.append(int(packet.token_id))
            text_parts.append(str(packet.text_piece))
            source_seq = record.get("source_token_seq")
            source_row = {
                "token_id": int(packet.token_id),
                "text_piece": str(packet.text_piece),
                "exclusive_kv_cursor": int(record_cursor),
                "revision_id": record.get("revision_id"),
                "cache_epoch_id": record.get("cache_epoch_id"),
            }
            if source_seq is not None:
                source_row["source_token_seq"] = int(source_seq)
            newly_tts_consumed.append(source_row)
            newly_source_records.append(record)
            cursor = record_cursor
        if cursor != exclusive_kv_cursor:
            raise RuntimeError("V1 speech ACK cursor has no contiguous source Gemma packet")

        branch["spoken_prefix_token_ids"] = tuple(token_ids)
        branch["spoken_prefix_text_parts"] = tuple(text_parts)
        branch["spoken_prefix_exclusive_cursor"] = int(exclusive_kv_cursor)
        branch["spoken_prefix_kv_position"] = int(exclusive_kv_cursor)
        branch["spoken_prefix_text"] = cumulative_vocalized_text
        branch["spoken_prefix_ack_provenance"] = "v1_live_qwen_frame_ack"
        ledger = branch.get("response_event_ledger")
        record_generated = getattr(ledger, "record_generated_tokens", None)
        record_tts = getattr(ledger, "record_tts_consumed", None)
        if newly_tts_consumed and callable(record_generated):
            sequences = record_generated(newly_tts_consumed)
            for source_row, source_record, source_seq in zip(
                newly_tts_consumed, newly_source_records, sequences, strict=True
            ):
                source_row["source_token_seq"] = int(source_seq)
                if isinstance(source_record, dict):
                    source_record["source_token_seq"] = int(source_seq)
            if sequences and callable(record_tts):
                record_tts(max(sequences))
        return True


_ACK_METADATA_UNSET = object()


def _same_cache_epoch_id(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if (
        isinstance(left, bool)
        or not isinstance(left, (int, str))
        or isinstance(right, bool)
        or not isinstance(right, (int, str))
    ):
        return False
    return str(left) == str(right)


def _publish_acknowledged_speech_prefix(
    branch: dict[str, Any] | None,
    generation_id: int,
    exclusive_kv_cursor: int,
    cumulative_vocalized_text: str,
    *,
    source_token_seq: int | None = None,
    revision_id: Any = _ACK_METADATA_UNSET,
    cache_epoch_id: Any = _ACK_METADATA_UNSET,
) -> bool:
    """Advance the played V1 source prefix by logical identity.

    The physical cursor is only provenance for locating the callback's packet
    in one revision. It is deliberately not compared with the previous
    physical cursor: a cache rebase can reuse or move those positions. The
    logical source sequence is monotone across revisions and remains the
    prefix identity.
    """
    if not isinstance(branch, dict):
        return False
    ledger = branch.get("response_event_ledger")
    if source_token_seq is None:
        if ledger is None:
            return _publish_acknowledged_speech_prefix_legacy(
                branch,
                generation_id,
                exclusive_kv_cursor,
                cumulative_vocalized_text,
            )
        raise RuntimeError("V1 PCM ACK lacks its preassigned logical source sequence")
    if (
        isinstance(source_token_seq, bool)
        or not isinstance(source_token_seq, int)
        or source_token_seq < 0
        or isinstance(exclusive_kv_cursor, bool)
        or not isinstance(exclusive_kv_cursor, int)
        or exclusive_kv_cursor < 0
        or not isinstance(cumulative_vocalized_text, str)
    ):
        raise ValueError("acknowledged V1 speech prefix has invalid logical identity or text")

    branch_lock = branch.get("spoken_prefix_lock")
    if branch_lock is None:
        return False
    ledger_lock = getattr(ledger, "_lock", None)
    source_tokens = getattr(ledger, "_source_tokens", None)
    if ledger is None or ledger_lock is None or not isinstance(source_tokens, dict):
        raise RuntimeError("logical V1 PCM ACK requires its response event ledger")

    with branch_lock:
        if int(branch.get("session_generation_id", -1)) != int(generation_id):
            return False

        packet_records = branch.get("say_token_packets", ())
        segment_records = branch.get("segment_token_packets", ())
        if not isinstance(packet_records, (tuple, list)) or not isinstance(
            segment_records, (tuple, list)
        ):
            raise RuntimeError("V1 speech ACK has no adopted source packet records")
        if len(packet_records) != len(segment_records):
            raise RuntimeError("V1 source token records and SAY packets have different lengths")

        by_sequence: dict[int, tuple[Mapping[str, Any], Any]] = {}
        previous_record_sequence = -1
        for record, packet in zip(segment_records, packet_records, strict=True):
            if not isinstance(record, Mapping):
                raise RuntimeError("V1 speech source token record is not a mapping")
            record_sequence = record.get("source_token_seq")
            if (
                isinstance(record_sequence, bool)
                or not isinstance(record_sequence, int)
                or record_sequence < 0
                or record_sequence <= previous_record_sequence
            ):
                raise RuntimeError("V1 logical source token records are not strictly ordered")
            token_id = record.get("token_id")
            text_piece = record.get("text_piece")
            record_cursor = record.get("exclusive_kv_cursor")
            packet_token_id = getattr(packet, "token_id", None)
            packet_text_piece = getattr(packet, "text_piece", None)
            packet_cursor = getattr(packet, "exclusive_kv_cursor", None)
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or not isinstance(text_piece, str)
                or isinstance(record_cursor, bool)
                or not isinstance(record_cursor, int)
                or record_cursor < 0
                or packet_token_id != token_id
                or packet_text_piece != text_piece
                or packet_cursor != record_cursor
            ):
                raise RuntimeError("V1 logical source token does not match its SAY packet")
            by_sequence[int(record_sequence)] = (record, packet)
            previous_record_sequence = int(record_sequence)

        target = by_sequence.get(int(source_token_seq))
        if target is None:
            raise RuntimeError("V1 PCM ACK references an absent adopted source token")
        target_record, target_packet = target

        if revision_id is _ACK_METADATA_UNSET:
            effective_revision = target_record.get("revision_id", branch.get("revision_id"))
        else:
            effective_revision = revision_id
        if not isinstance(effective_revision, str) or not effective_revision:
            raise RuntimeError("V1 PCM ACK has no revision identity")
        if cache_epoch_id is _ACK_METADATA_UNSET:
            effective_epoch = target_record.get(
                "cache_epoch_id", branch.get("cache_epoch_id")
            )
        else:
            effective_epoch = cache_epoch_id
        if effective_epoch is not None and (
            isinstance(effective_epoch, bool)
            or not isinstance(effective_epoch, (int, str))
        ):
            raise RuntimeError("V1 PCM ACK has an invalid cache epoch identity")

        provenance = target_record.get("physical_kv_provenance")
        if not isinstance(provenance, (tuple, list)):
            raise RuntimeError("V1 PCM ACK token has no per-revision physical provenance")
        matching_provenance = any(
            isinstance(item, Mapping)
            and item.get("revision_id") == effective_revision
            and _same_cache_epoch_id(item.get("cache_epoch_id"), effective_epoch)
            and item.get("physical_kv_cursor") == int(exclusive_kv_cursor)
            for item in provenance
        )
        if (
            not matching_provenance
            or int(target_record["exclusive_kv_cursor"]) != int(exclusive_kv_cursor)
            or getattr(target_packet, "kv_committed", False) is not True
            or target_record.get("kv_committed") is not True
        ):
            raise RuntimeError("V1 PCM ACK does not identify an exact committed revision packet")

        previous_source_sequence = branch.get("spoken_prefix_source_seq", -1)
        if (
            isinstance(previous_source_sequence, bool)
            or not isinstance(previous_source_sequence, int)
            or previous_source_sequence < -1
        ):
            raise RuntimeError("V1 spoken prefix has an invalid logical frontier")
        previous_text = str(branch.get("spoken_prefix_text", ""))
        if int(source_token_seq) < previous_source_sequence:
            return False
        if int(source_token_seq) == previous_source_sequence:
            if previous_text != cumulative_vocalized_text:
                raise RuntimeError("same V1 logical source frontier was reported with different text")
            return False

        token_ids = list(int(value) for value in branch.get("spoken_prefix_token_ids", ()))
        text_parts = list(str(value) for value in branch.get("spoken_prefix_text_parts", ()))
        if "".join(text_parts) != previous_text:
            raise RuntimeError("V1 spoken prefix text parts disagree with their published text")
        newly_selected: list[tuple[int, Mapping[str, Any], Any]] = []
        for sequence, (record, packet) in by_sequence.items():
            if previous_source_sequence < sequence <= int(source_token_seq):
                if (
                    record.get("kv_committed") is not True
                    or getattr(packet, "kv_committed", False) is not True
                ):
                    raise RuntimeError("V1 speech ACK crossed an uncommitted Gemma packet")
                newly_selected.append((sequence, record, packet))
        if not newly_selected or newly_selected[-1][0] != int(source_token_seq):
            raise RuntimeError("V1 PCM ACK skipped its logical source prefix")

        # The event ledger is read under its lock so a concurrent suffix CAS
        # cannot retire or replace any identity between validation and publish.
        with ledger_lock:
            if int(getattr(ledger, "generation_id", -1)) != int(generation_id):
                return False
            current_sources = getattr(ledger, "_source_tokens", None)
            if not isinstance(current_sources, dict):
                raise RuntimeError("response ledger lost its source token registry")
            expected_sequences = {
                sequence
                for sequence in current_sources
                if previous_source_sequence < int(sequence) <= int(source_token_seq)
            }
            selected_sequences = {sequence for sequence, _record, _packet in newly_selected}
            if selected_sequences != expected_sequences:
                raise RuntimeError("V1 PCM ACK omitted a live logical source token")
            for sequence, record, _packet in newly_selected:
                source_row = current_sources.get(sequence)
                if (
                    not isinstance(source_row, Mapping)
                    or source_row.get("token_id") != record.get("token_id")
                    or source_row.get("text_piece") != record.get("text_piece")
                ):
                    raise RuntimeError("V1 PCM ACK source identity is absent or changed in the ledger")
                if sequence == int(source_token_seq):
                    ledger_provenance = source_row.get("physical_kv_provenance")
                    if not isinstance(ledger_provenance, (tuple, list)) or not any(
                        isinstance(item, Mapping)
                        and item.get("revision_id") == effective_revision
                        and _same_cache_epoch_id(item.get("cache_epoch_id"), effective_epoch)
                        and item.get("physical_kv_cursor") == int(exclusive_kv_cursor)
                        for item in ledger_provenance
                    ):
                        raise RuntimeError("V1 PCM ACK provenance is absent from the response ledger")
            candidate_token_ids = token_ids + [
                int(record["token_id"]) for _, record, _packet in newly_selected
            ]
            candidate_text_parts = text_parts + [
                str(record["text_piece"]) for _, record, _packet in newly_selected
            ]
            if "".join(candidate_text_parts) != cumulative_vocalized_text:
                raise RuntimeError("V1 PCM ACK text does not match the adopted logical source prefix")

            branch["spoken_prefix_token_ids"] = tuple(candidate_token_ids)
            branch["spoken_prefix_text_parts"] = tuple(candidate_text_parts)
            branch["spoken_prefix_source_seq"] = int(source_token_seq)
            branch["spoken_prefix_exclusive_cursor"] = int(exclusive_kv_cursor)
            branch["spoken_prefix_kv_position"] = int(exclusive_kv_cursor)
            branch["spoken_prefix_revision_id"] = effective_revision
            branch["spoken_prefix_cache_epoch_id"] = effective_epoch
            branch["spoken_prefix_text"] = cumulative_vocalized_text
            branch["spoken_prefix_ack_provenance"] = "v1_live_qwen_frame_ack"
        return True


def _client_consumed_v1_token_packets_legacy(
    branch: Mapping[str, Any] | None,
    *,
    termination_reason: str,
) -> tuple[dict[str, Any], ...]:
    """Return committed V1 source tokens covered by the vocalized PCM frontier."""
    if not isinstance(branch, Mapping) or branch.get("spoken_prefix_ack_provenance") != (
        "v1_live_qwen_frame_ack"
    ):
        return ()
    frontier = branch.get("spoken_prefix_exclusive_cursor")
    if isinstance(frontier, bool) or not isinstance(frontier, int):
        raise RuntimeError("V1 vocalized PCM frontier has no exact token cursor")
    records = branch.get("segment_token_packets")
    packets = branch.get("say_token_packets")
    if not isinstance(records, (tuple, list)) or not isinstance(packets, (tuple, list)):
        raise RuntimeError("V1 vocalized PCM frontier has no source token records")
    if len(records) != len(packets):
        raise RuntimeError("V1 source token records and SAY packets have different lengths")
    result: list[dict[str, Any]] = []
    for index, (record, packet) in enumerate(zip(records, packets, strict=True)):
        if not isinstance(record, Mapping):
            raise RuntimeError("V1 source token record is not a mapping")
        cursor = record.get("exclusive_kv_cursor")
        if isinstance(cursor, bool) or not isinstance(cursor, int):
            raise RuntimeError("V1 source token record has no exact KV cursor")
        if cursor > frontier:
            break
        token_id = record.get("token_id")
        text_piece = record.get("text_piece")
        packet_cursor = getattr(packet, "exclusive_kv_cursor", None)
        packet_token_id = getattr(packet, "token_id", None)
        packet_text = getattr(packet, "text_piece", None)
        if (
            record.get("kv_committed") is not True
            or getattr(packet, "kv_committed", False) is not True
            or isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or packet_token_id != token_id
            or packet_cursor != cursor
            or packet_text != text_piece
            or not isinstance(text_piece, str)
        ):
            raise RuntimeError("V1 vocalized PCM token is not an exact committed source packet")
        result.append({
            "token_id": int(token_id),
            "exclusive_kv_cursor": int(cursor),
            "text_piece": text_piece,
            "source_token_index": int(index),
            "termination_reason": str(termination_reason),
        })
    if result and int(result[-1]["exclusive_kv_cursor"]) != frontier:
        raise RuntimeError("V1 vocalized PCM frontier skipped an original source token")
    return tuple(result)


def _client_consumed_v1_token_packets(
    branch: Mapping[str, Any] | None,
    *,
    termination_reason: str,
) -> tuple[dict[str, Any], ...]:
    """Return exact committed V1 tokens through the logical played frontier."""
    if not isinstance(branch, Mapping) or branch.get("spoken_prefix_ack_provenance") != (
        "v1_live_qwen_frame_ack"
    ):
        return ()
    logical_frontier = branch.get("spoken_prefix_source_seq")
    ledger = branch.get("response_event_ledger")
    if logical_frontier is None:
        if ledger is not None:
            raise RuntimeError("V1 played prefix has no logical source frontier")
        return _client_consumed_v1_token_packets_legacy(
            branch,
            termination_reason=termination_reason,
        )
    if (
        isinstance(logical_frontier, bool)
        or not isinstance(logical_frontier, int)
        or logical_frontier < 0
    ):
        raise RuntimeError("V1 played prefix has an invalid logical source frontier")
    records = branch.get("segment_token_packets")
    packets = branch.get("say_token_packets")
    if not isinstance(records, (tuple, list)) or not isinstance(packets, (tuple, list)):
        raise RuntimeError("V1 played prefix has no source token records")
    if len(records) != len(packets):
        raise RuntimeError("V1 source token records and SAY packets have different lengths")
    ledger_lock = getattr(ledger, "_lock", None)
    source_tokens = getattr(ledger, "_source_tokens", None)
    if ledger is None or ledger_lock is None or not isinstance(source_tokens, dict):
        raise RuntimeError("V1 logical played prefix requires its response event ledger")

    result: list[dict[str, Any]] = []
    previous_sequence = -1
    reached_frontier = False
    with ledger_lock:
        for index, (record, packet) in enumerate(zip(records, packets, strict=True)):
            if not isinstance(record, Mapping):
                raise RuntimeError("V1 source token record is not a mapping")
            sequence = record.get("source_token_seq")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
                or sequence <= previous_sequence
            ):
                raise RuntimeError("V1 logical source token records are not strictly ordered")
            previous_sequence = int(sequence)
            if sequence > logical_frontier:
                break

            token_id = record.get("token_id")
            text_piece = record.get("text_piece")
            cursor = record.get("exclusive_kv_cursor")
            revision = record.get("revision_id")
            epoch = record.get("cache_epoch_id")
            provenance = record.get("physical_kv_provenance")
            packet_cursor = getattr(packet, "exclusive_kv_cursor", None)
            packet_token_id = getattr(packet, "token_id", None)
            packet_text = getattr(packet, "text_piece", None)
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or not isinstance(text_piece, str)
                or isinstance(cursor, bool)
                or not isinstance(cursor, int)
                or cursor < 0
                or not isinstance(revision, str)
                or not revision
                or (
                    epoch is not None
                    and (
                        isinstance(epoch, bool)
                        or not isinstance(epoch, (int, str))
                    )
                )
                or record.get("kv_committed") is not True
                or getattr(packet, "kv_committed", False) is not True
                or packet_token_id != token_id
                or packet_cursor != cursor
                or packet_text != text_piece
                or not isinstance(provenance, (tuple, list))
                or not any(
                    isinstance(item, Mapping)
                    and item.get("revision_id") == revision
                    and _same_cache_epoch_id(item.get("cache_epoch_id"), epoch)
                    and item.get("physical_kv_cursor") == cursor
                    for item in provenance
                )
            ):
                raise RuntimeError("V1 played token has no exact committed physical provenance")
            ledger_row = source_tokens.get(int(sequence))
            ledger_provenance = (
                ledger_row.get("physical_kv_provenance")
                if isinstance(ledger_row, Mapping) else None
            )
            if (
                not isinstance(ledger_row, Mapping)
                or ledger_row.get("token_id") != token_id
                or ledger_row.get("text_piece") != text_piece
                or not isinstance(ledger_provenance, (tuple, list))
                or not any(
                    isinstance(item, Mapping)
                    and item.get("revision_id") == revision
                    and _same_cache_epoch_id(item.get("cache_epoch_id"), epoch)
                    and item.get("physical_kv_cursor") == cursor
                    for item in ledger_provenance
                )
            ):
                raise RuntimeError("V1 played token identity is absent from the response ledger")
            raw_index = record.get("token_index", index)
            token_index = (
                int(raw_index)
                if isinstance(raw_index, int) and not isinstance(raw_index, bool)
                else int(index)
            )
            result.append({
                "token_id": int(token_id),
                "source_token_seq": int(sequence),
                "exclusive_kv_cursor": int(cursor),
                "revision_id": revision,
                "cache_epoch_id": epoch,
                "physical_kv_provenance": tuple(dict(item) for item in provenance),
                "text_piece": text_piece,
                "source_token_index": token_index,
                "termination_reason": str(termination_reason),
            })
            reached_frontier = int(sequence) == int(logical_frontier)
            if reached_frontier:
                break
    if not result or not reached_frontier:
        raise RuntimeError("V1 played prefix skipped its logical source frontier")
    return tuple(result)
