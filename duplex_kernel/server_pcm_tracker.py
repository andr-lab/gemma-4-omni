"""Client-consumed PCM tracking and listener admission helpers."""
from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from duplex_kernel.audio_stream import CODEC_SAMPLE_RATE
from duplex_kernel.engine import ListenerBacklogAdmission
import logging
logger = logging.getLogger(__name__)
MAX_RETAINED_EMITTED_PCM_SAMPLES = 24_000 * 30
_CLIENT_CONSUMED_PCM_ACK_FIELDS = (
    "type",
    "ack_version",
    "response_id",
    "generation_id",
    "speaker_segment_id",
    "start_sample",
    "end_sample",
    "sample_rate",
    "sequence",
    "consumption_semantics",
    "final",
)


def _canonical_client_consumed_pcm_ack(acknowledgement: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return immutable wire metadata used for response-branch deduplication."""
    return tuple(
        (field, field in acknowledgement, acknowledgement.get(field))
        for field in _CLIENT_CONSUMED_PCM_ACK_FIELDS
    )


def _client_consumed_pcm_ack_duplicate_status(
    branch: dict[str, Any] | None,
    acknowledgement: dict[str, Any],
) -> bool | None:
    """Return exact-match status for a retained branch ACK, else ``None``.

    The response branch is authoritative. A connection-level cache may be
    evicted while the response remains active, so its retention window cannot
    define whether an old branch ACK is a duplicate.
    """
    sequence = acknowledgement.get("sequence")
    ack_version = acknowledgement.get("ack_version", 1)
    if (
        not isinstance(branch, dict)
        or not isinstance(acknowledgement, dict)
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or isinstance(ack_version, bool)
        or ack_version != 2
    ):
        return None
    seen = branch.get("client_consumed_pcm_ack_by_sequence")
    if not isinstance(seen, dict) or sequence not in seen:
        return None
    previous = seen[sequence]
    if isinstance(previous, Mapping):
        previous = _canonical_client_consumed_pcm_ack(previous)
    return previous == _canonical_client_consumed_pcm_ack(acknowledgement)


def _retained_emitted_pcm_samples(branch: Mapping[str, Any]) -> int:
    segments = branch.get("playback_segments")
    if not isinstance(segments, Mapping):
        return 0
    return sum(
        int(np.asarray(packet.get("pcm", ()), dtype=np.float32).size)
        for segment in segments.values()
        if isinstance(segment, Mapping)
        for packet in segment.get("emitted_pcm_packets", ())
        if isinstance(packet, Mapping)
    )


def _enforce_emitted_pcm_retention(branch: dict[str, Any]) -> None:
    """Bound unconsumed PCM evidence while preserving the newest ranges."""
    segments = branch.get("playback_segments")
    if not isinstance(segments, dict):
        return
    maximum = branch.get("max_retained_emitted_pcm_samples", MAX_RETAINED_EMITTED_PCM_SAMPLES)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        maximum = MAX_RETAINED_EMITTED_PCM_SAMPLES
    retained = _retained_emitted_pcm_samples(branch)
    for segment in segments.values():
        if retained <= maximum or not isinstance(segment, dict):
            continue
        packets = segment.get("emitted_pcm_packets")
        if not isinstance(packets, list):
            continue
        while packets and retained > maximum:
            packet = packets[0]
            pcm = np.asarray(packet.get("pcm", ()), dtype=np.float32).reshape(-1)
            excess = retained - maximum
            if excess >= pcm.size:
                packets.pop(0)
                retained -= int(pcm.size)
                segment["emitted_pcm_truncated"] = True
                continue
            packet["pcm"] = np.array(pcm[excess:], dtype=np.float32, copy=True)
            packet["start_sample"] = int(packet["start_sample"]) + int(excess)
            retained -= int(excess)
            segment["emitted_pcm_truncated"] = True
        if retained <= maximum:
            break
    branch["emitted_pcm_retained_samples"] = int(retained)


def _trim_emitted_pcm_after_ack(
    branch: dict[str, Any], segment: dict[str, Any], *, consumed_end: int, final: bool
) -> None:
    packets = segment.get("emitted_pcm_packets")
    if not isinstance(packets, list):
        return
    if final:
        packets.clear()
        branch["emitted_pcm_retained_samples"] = _retained_emitted_pcm_samples(branch)
        return
    trimmed: list[dict[str, Any]] = []
    for packet in packets:
        packet_end = int(packet["end_sample"])
        if packet_end <= consumed_end:
            continue
        packet_start = int(packet["start_sample"])
        pcm = np.asarray(packet["pcm"], dtype=np.float32).reshape(-1)
        if packet_start < consumed_end:
            offset = consumed_end - packet_start
            packet = dict(packet)
            packet["start_sample"] = int(consumed_end)
            packet["pcm"] = np.array(pcm[offset:], dtype=np.float32, copy=True)
        trimmed.append(packet)
    segment["emitted_pcm_packets"] = trimmed
    branch["emitted_pcm_retained_samples"] = _retained_emitted_pcm_samples(branch)

def _record_client_consumed_pcm(
    branch: dict[str, Any] | None,
    acknowledgement: dict[str, Any],
) -> bool:
    """Record one cumulative client-device watermark against emitted PCM.

    Version 2 accepts progress before ``response.audio.segment_done``. Source
    text is advanced only after an explicitly closed segment is fully consumed;
    sample ranges alone never imply a token-to-audio alignment.
    """
    if not isinstance(branch, dict) or not isinstance(acknowledgement, dict):
        return False
    if acknowledgement.get("type") != "client_consumed_pcm":
        return False
    response_id = str(branch.get("response_id", ""))
    segment_id = acknowledgement.get("speaker_segment_id")
    generation_value = acknowledgement.get("generation_id")
    start_sample = acknowledgement.get("start_sample")
    end_sample = acknowledgement.get("end_sample")
    sample_rate = acknowledgement.get("sample_rate")
    sequence = acknowledgement.get("sequence")
    ack_version = acknowledgement.get("ack_version", 1)
    final = acknowledgement.get("final", False)
    if (
        isinstance(generation_value, bool) or not isinstance(generation_value, int)
        or isinstance(start_sample, bool) or not isinstance(start_sample, int)
        or isinstance(end_sample, bool) or not isinstance(end_sample, int)
        or isinstance(sample_rate, bool) or not isinstance(sample_rate, int)
        or isinstance(sequence, bool) or not isinstance(sequence, int)
        or isinstance(ack_version, bool) or not isinstance(ack_version, int)
    ):
        return False
    if (
        not response_id
        or acknowledgement.get("response_id") != response_id
        or generation_value != branch.get("session_generation_id")
        or not isinstance(segment_id, str)
        or not segment_id
        or ack_version not in {1, 2}
        or acknowledgement.get("consumption_semantics") != "logical_client_device_buffer"
        or start_sample < 0
        or end_sample < start_sample
        or sample_rate != CODEC_SAMPLE_RATE
        or sequence < 0
    ):
        return False
    if ack_version == 2 and not isinstance(final, bool):
        return False
    if ack_version == 1 and "final" in acknowledgement:
        return False

    previous_sequence = int(branch.get("client_consumed_pcm_sequence", -1))
    seen_sequences = branch.get("client_consumed_pcm_ack_by_sequence")
    duplicate_status = _client_consumed_pcm_ack_duplicate_status(
        branch, acknowledgement
    )
    if duplicate_status is not None:
        return duplicate_status
    if sequence <= previous_sequence:
        return False

    if branch.get("playback_cancelled") is True:
        deadline = branch.get("playback_ack_final_deadline_monotonic")
        if (
            ack_version != 2
            or not isinstance(deadline, (int, float))
            or time.monotonic() > float(deadline)
        ):
            return False

    segments = branch.get("playback_segments")
    if not isinstance(segments, dict):
        return False
    segment = segments.get(segment_id)
    if not isinstance(segment, dict):
        return False
    if (
        segment.get("response_id") != response_id
        or segment.get("generation_id") != generation_value
        or segment.get("speaker_segment_id") != segment_id
        or segment.get("sample_rate") != sample_rate
        or segment.get("client_consumed_final") is True
    ):
        return False
    original_start = segment.get("start_sample")
    emitted_end = segment.get("emitted_end_sample")
    consumed_end = segment.get("client_consumed_end_sample", original_start)
    if (
        isinstance(original_start, bool) or not isinstance(original_start, int)
        or isinstance(emitted_end, bool) or not isinstance(emitted_end, int)
        or isinstance(consumed_end, bool) or not isinstance(consumed_end, int)
        or start_sample != original_start
        or end_sample < consumed_end
        or end_sample > emitted_end
    ):
        return False

    if ack_version == 1:
        if (
            segment.get("done_emitted") is not True
            or segment.get("end_sample") != end_sample
            or end_sample <= start_sample
        ):
            return False
    elif final is True:
        cancellation_active = branch.get("playback_cancelled") is True
        if not cancellation_active and (
            end_sample != emitted_end
            or (
                segment.get("done_emitted") is True
                and end_sample != segment.get("end_sample")
            )
        ):
            return False
        if end_sample == start_sample and not cancellation_active:
            return False
    elif end_sample == start_sample:
        return False

    ledger = branch.get("response_event_ledger")
    validate_pcm = getattr(ledger, "validate_pcm_consumed", None)
    record_pcm = getattr(ledger, "record_pcm_consumed", None)
    consumed_pcm = _copy_emitted_pcm_range(segment, consumed_end, end_sample)
    if (
        consumed_pcm is None
        and ack_version == 1
        and not callable(record_pcm)
        and segment.get("done_emitted") is True
        and segment.get("end_sample") == end_sample
        and end_sample > start_sample
    ):
        # Legacy v1 carries only a complete-segment marker. Older producers
        # may not retain PCM bytes; this narrow path can update complete text
        # state, but cannot contribute to the exact PCM ledger.
        consumed_pcm = np.empty(0, dtype=np.float32)
    if consumed_pcm is None:
        logger.warning(
            "[_record_client_consumed_pcm] rejected: emitted PCM is incomplete for segment_id=%s range=%d:%d",
            segment_id,
            consumed_end,
            end_sample,
        )
        return False

    if callable(validate_pcm) and (consumed_pcm.size or (ack_version == 2 and final)):
        try:
            valid_ledger_range = validate_pcm(
                speaker_segment_id=segment_id,
                start_sample=int(consumed_end),
                end_sample=int(end_sample),
                pcm=consumed_pcm,
                sequence=int(sequence),
            )
        except Exception:
            logger.exception("[_record_client_consumed_pcm] ledger rejected ACK during validation")
            return False
        if not valid_ledger_range:
            return False

    segment_snapshot = dict(segment)
    prefix_keys = (
        "spoken_prefix_token_ids", "spoken_prefix_exclusive_cursor",
        "spoken_prefix_text_parts", "spoken_prefix_text",
        "client_consumed_prefix_token_ids", "client_consumed_prefix_exclusive_cursor",
        "client_consumed_prefix_text_parts", "client_consumed_prefix_text",
        "spoken_prefix_kv_position", "spoken_prefix_ack_provenance",
        "client_consumed_segments",
    )
    missing = object()
    prefix_snapshot = {key: branch.get(key, missing) for key in prefix_keys}
    completed = set(branch.get("client_consumed_segments", ()))
    prior_branch_sequence = branch.get("client_consumed_pcm_sequence", missing)
    prior_completed = branch.get("client_consumed_segments", missing)
    prior_events = branch.get("client_consumed_pcm_events", missing)
    prior_events_list = prior_events if isinstance(prior_events, list) else None
    prior_events_length = len(prior_events_list) if prior_events_list is not None else 0
    prior_seen_sequence = (
        seen_sequences.get(sequence, missing)
        if isinstance(seen_sequences, dict) else missing
    )
    prior_seen_sequence_map = branch.get("client_consumed_pcm_ack_by_sequence", missing)
    prior_retained_samples = branch.get("emitted_pcm_retained_samples", missing)
    emitted_packets = segment.get("emitted_pcm_packets")
    original_packet_list = emitted_packets if isinstance(emitted_packets, list) else None
    original_packets = list(original_packet_list) if original_packet_list is not None else None

    def restore_branch_ack() -> None:
        if original_packet_list is not None and original_packets is not None:
            original_packet_list[:] = original_packets
        segment.clear()
        segment.update(segment_snapshot)
        if original_packet_list is not None:
            segment["emitted_pcm_packets"] = original_packet_list
        for key, value in prefix_snapshot.items():
            if value is missing:
                branch.pop(key, None)
            else:
                branch[key] = value
        if prior_branch_sequence is missing:
            branch.pop("client_consumed_pcm_sequence", None)
        else:
            branch["client_consumed_pcm_sequence"] = prior_branch_sequence
        if prior_completed is missing:
            branch.pop("client_consumed_segments", None)
        else:
            branch["client_consumed_segments"] = prior_completed
        if prior_events_list is not None and prior_events is prior_events_list:
            del prior_events_list[prior_events_length:]
        elif prior_events is missing:
            branch.pop("client_consumed_pcm_events", None)
        else:
            branch["client_consumed_pcm_events"] = prior_events
        if isinstance(seen_sequences, dict):
            if prior_seen_sequence is missing:
                seen_sequences.pop(sequence, None)
            else:
                seen_sequences[sequence] = prior_seen_sequence
        if prior_seen_sequence_map is missing:
            branch.pop("client_consumed_pcm_ack_by_sequence", None)
        elif not isinstance(seen_sequences, dict):
            branch["client_consumed_pcm_ack_by_sequence"] = prior_seen_sequence_map
        if prior_retained_samples is missing:
            branch.pop("emitted_pcm_retained_samples", None)
        else:
            branch["emitted_pcm_retained_samples"] = prior_retained_samples

    def commit_branch_ack(_projection: Any = None) -> None:
        try:
            segment["client_consumed_end_sample"] = int(end_sample)
            segment["pcm_samples_consumed"] = int(end_sample - original_start)
            segment["pcm_consumed_start_sample"] = int(original_start)
            segment["pcm_consumed_end_sample"] = int(end_sample)
            segment["consumption_sequence"] = int(sequence)
            segment["client_consumed"] = bool(
                segment.get("done_emitted") is True
                and end_sample == segment.get("end_sample")
            )
            segment["client_consumed_final"] = (
                bool(final) if ack_version == 2 else segment["client_consumed"]
            )

            if segment.get("client_consumed") is True and segment_id not in completed:
                if not _commit_client_consumed_segment(branch, segment_id):
                    raise RuntimeError("complete PCM segment source projection failed")

            branch["client_consumed_pcm_sequence"] = sequence
            if segment.get("client_consumed") is True and segment_id not in completed:
                branch["client_consumed_segments"] = tuple((*completed, segment_id))
            event_row = {
                **acknowledgement,
                "response_id": response_id,
                "generation_id": generation_value,
                "speaker_segment_id": segment_id,
                "consumption_semantics": "logical_client_device_buffer",
            }
            branch.setdefault("client_consumed_pcm_events", []).append(event_row)
            accepted_sequences = branch.setdefault("client_consumed_pcm_ack_by_sequence", {})
            if isinstance(accepted_sequences, dict):
                accepted_sequences[sequence] = _canonical_client_consumed_pcm_ack(
                    acknowledgement
                )
            if ack_version == 2 and final is True:
                segment["client_consumed_final"] = True
            _trim_emitted_pcm_after_ack(
                branch,
                segment,
                consumed_end=int(end_sample),
                final=bool(ack_version == 2 and final is True),
            )
        except Exception:
            restore_branch_ack()
            raise

    try:
        if callable(record_pcm) and (consumed_pcm.size or (ack_version == 2 and final)):
            record_pcm(
                speaker_segment_id=segment_id,
                start_sample=int(consumed_end),
                end_sample=int(end_sample),
                pcm=consumed_pcm,
                sequence=int(sequence),
                final=bool(final) if ack_version == 2 else bool(
                    segment.get("done_emitted") is True and end_sample == segment.get("end_sample")
                ),
                on_commit=commit_branch_ack,
            )
        else:
            commit_branch_ack()
    except Exception:
        restore_branch_ack()
        logger.exception("[_record_client_consumed_pcm] failed to commit validated ACK and ledger projection")
        return False
    return True


def _copy_emitted_pcm_range(
    segment: Mapping[str, Any], start_sample: int, end_sample: int
) -> np.ndarray | None:
    """Copy one exact emitted PCM interval without inferring token alignment."""
    if end_sample == start_sample:
        return np.empty(0, dtype=np.float32)
    packets = segment.get("emitted_pcm_packets")
    if not isinstance(packets, list):
        return None
    cursor = int(start_sample)
    pieces: list[np.ndarray] = []
    for packet in packets:
        if not isinstance(packet, Mapping):
            return None
        packet_start = packet.get("start_sample")
        packet_end = packet.get("end_sample")
        packet_pcm = packet.get("pcm")
        if (
            isinstance(packet_start, bool) or not isinstance(packet_start, int)
            or isinstance(packet_end, bool) or not isinstance(packet_end, int)
        ):
            return None
        if packet_end <= cursor or packet_start >= end_sample:
            continue
        if packet_start > cursor or packet_end <= cursor:
            return None
        array = np.asarray(packet_pcm, dtype=np.float32).reshape(-1)
        if packet_end - packet_start != array.size:
            return None
        take_end = min(int(packet_end), int(end_sample))
        first = cursor - int(packet_start)
        last = take_end - int(packet_start)
        pieces.append(np.array(array[first:last], dtype=np.float32, copy=True))
        cursor = take_end
        if cursor == end_sample:
            break
    if cursor != end_sample:
        return None
    return np.concatenate(pieces).astype(np.float32, copy=False)

def _commit_client_consumed_segment(branch: dict[str, Any], segment_id: str) -> bool:
    segments = branch.get("playback_segments")
    segment = segments.get(segment_id) if isinstance(segments, dict) else None
    if not isinstance(segment, dict):
        return False
    completed = set(branch.get("client_consumed_segments", ()))
    if segment_id in completed:
        return False
    ledger = branch.get("response_event_ledger")
    project = getattr(ledger, "project", None)
    if callable(project):
        projection = project()
        source_rows = tuple(projection.confirmed_source_tokens)
        source_ids = tuple(
            int(row["token_id"])
            for row in source_rows
            if isinstance(row.get("token_id"), int)
            and not isinstance(row.get("token_id"), bool)
        )
        source_parts = tuple(str(row.get("text_piece", "")) for row in source_rows)
        base_cursor = int(branch.get(
            "spoken_prefix_base_cursor", branch.get("turn_start_cache_position", 0)
        ))
        cursor = projection.confirmed_exclusive_kv_cursor
        if not source_ids:
            cursor = base_cursor
        physical_cursor = (
            int(cursor)
            if isinstance(cursor, int) and not isinstance(cursor, bool)
            else None
        )
        branch["spoken_prefix_token_ids"] = source_ids
        branch["spoken_prefix_text_parts"] = source_parts
        branch["spoken_prefix_text"] = str(projection.confirmed_text)
        branch["spoken_prefix_exclusive_cursor"] = physical_cursor
        branch["spoken_prefix_kv_position"] = physical_cursor
        branch["spoken_prefix_ack_provenance"] = "response_event_ledger_pcm_consumption"
        branch["client_consumed_prefix_token_ids"] = source_ids
        branch["client_consumed_prefix_text_parts"] = source_parts
        branch["client_consumed_prefix_text"] = str(projection.confirmed_text)
        branch["client_consumed_prefix_exclusive_cursor"] = physical_cursor
        branch["spoken_prefix_source_token_seq"] = tuple(
            int(row["source_token_seq"])
            for row in source_rows
            if isinstance(row.get("source_token_seq"), int)
            and not isinstance(row.get("source_token_seq"), bool)
        )
        branch["spoken_prefix_physical_cursor_contiguous"] = bool(
            projection.physical_cursor_contiguous
        )
        return True
    packets = tuple(segment.get("token_packets", ()))
    if branch.get("spoken_prefix_exclusive_cursor") is not None:
        cursor = int(branch["spoken_prefix_exclusive_cursor"])
    elif (
        packets
        and isinstance(packets[0], Mapping)
        and isinstance(packets[0].get("exclusive_kv_cursor"), int)
        and not isinstance(packets[0].get("exclusive_kv_cursor"), bool)
    ):
        cursor = int(packets[0]["exclusive_kv_cursor"]) - 1
    elif branch.get("spoken_prefix_base_cursor") is not None:
        cursor = int(branch["spoken_prefix_base_cursor"])
    else:
        cursor = int(branch.get("turn_start_cache_position", 0))
    prefix_ids = list(int(value) for value in branch.get("spoken_prefix_token_ids", ()))
    prefix_parts = list(branch.get("spoken_prefix_text_parts", ()))
    regular_prefix_valid = True

    def log_invalid_packet(packet_index: int, packet: Any, expected_cursor: int, reason: str) -> None:
        packet_fields = packet if isinstance(packet, Mapping) else {}
        logger.warning(
            "[_commit_client_consumed_segment] rejected token packet "
            "segment_id=%s packet_index=%d reason=%s token_cursor=%s "
            "expected_cursor=%d token_id=%s text_piece=%r",
            segment_id,
            packet_index,
            reason,
            packet_fields.get("exclusive_kv_cursor"),
            expected_cursor,
            packet_fields.get("token_id"),
            packet_fields.get("text_piece"),
        )

    for packet_index, packet in enumerate(packets):
        if not isinstance(packet, Mapping):
            regular_prefix_valid = False
            break
        token_id = packet.get("token_id")
        token_cursor = packet.get("exclusive_kv_cursor")
        text_piece = packet.get("text_piece")
        reason = None
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            reason = "invalid_token_id"
        elif isinstance(token_cursor, bool) or not isinstance(token_cursor, int):
            reason = "invalid_token_cursor"
        elif token_cursor != cursor + 1:
            reason = "non_contiguous_cursor"
        elif not isinstance(text_piece, str):
            reason = "invalid_text_piece"
        if reason is not None:
            regular_prefix_valid = False
            break
        prefix_ids.append(token_id)
        prefix_parts.append(text_piece)
        cursor = token_cursor

    # Keep a separate prefix whose only advancement path is a complete client
    # consumption ACK. The live speech path may update spoken_prefix_* when PCM
    # is emitted; that frontier is useful for streaming but is too early for
    # history or a length-limited response's canonical KV rebuild.
    if not tuple(branch.get("client_consumed_segments", ())):
        if (
            packets
            and isinstance(packets[0], Mapping)
            and isinstance(packets[0].get("exclusive_kv_cursor"), int)
            and not isinstance(packets[0].get("exclusive_kv_cursor"), bool)
        ):
            consumed_cursor = int(packets[0]["exclusive_kv_cursor"]) - 1
        elif branch.get("spoken_prefix_base_cursor") is not None:
            consumed_cursor = int(branch["spoken_prefix_base_cursor"])
        elif branch.get("client_consumed_prefix_exclusive_cursor") is not None:
            consumed_cursor = int(branch["client_consumed_prefix_exclusive_cursor"])
        else:
            consumed_cursor = int(branch.get("turn_start_cache_position", 0))
    elif branch.get("client_consumed_prefix_exclusive_cursor") is not None:
        consumed_cursor = int(branch["client_consumed_prefix_exclusive_cursor"])
    elif (
        packets
        and isinstance(packets[0], Mapping)
        and isinstance(packets[0].get("exclusive_kv_cursor"), int)
        and not isinstance(packets[0].get("exclusive_kv_cursor"), bool)
    ):
        consumed_cursor = int(packets[0]["exclusive_kv_cursor"]) - 1
    elif branch.get("spoken_prefix_base_cursor") is not None:
        consumed_cursor = int(branch["spoken_prefix_base_cursor"])
    else:
        consumed_cursor = int(branch.get("turn_start_cache_position", 0))
    consumed_ids = list(int(value) for value in branch.get(
        "client_consumed_prefix_token_ids", ()
    ))
    consumed_parts = list(branch.get("client_consumed_prefix_text_parts", ()))
    for packet_index, packet in enumerate(packets):
        if not isinstance(packet, Mapping):
            log_invalid_packet(packet_index, packet, consumed_cursor + 1, "packet_not_mapping")
            return False
        token_id = packet.get("token_id")
        token_cursor = packet.get("exclusive_kv_cursor")
        text_piece = packet.get("text_piece")
        reason = None
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            reason = "invalid_token_id"
        elif isinstance(token_cursor, bool) or not isinstance(token_cursor, int):
            reason = "invalid_token_cursor"
        elif token_cursor != consumed_cursor + 1:
            reason = "non_contiguous_cursor"
        elif not isinstance(text_piece, str):
            reason = "invalid_text_piece"
        if reason is not None:
            log_invalid_packet(packet_index, packet, consumed_cursor + 1, reason)
            return False
        consumed_ids.append(token_id)
        consumed_parts.append(text_piece)
        consumed_cursor = token_cursor

    if regular_prefix_valid:
        branch["spoken_prefix_token_ids"] = tuple(prefix_ids)
        branch["spoken_prefix_exclusive_cursor"] = cursor
        branch["spoken_prefix_text_parts"] = tuple(prefix_parts)
        branch["spoken_prefix_text"] = "".join(prefix_parts)
    branch["client_consumed_prefix_token_ids"] = tuple(consumed_ids)
    branch["client_consumed_prefix_exclusive_cursor"] = consumed_cursor
    branch["client_consumed_prefix_text_parts"] = tuple(consumed_parts)
    branch["client_consumed_prefix_text"] = "".join(consumed_parts)
    branch["client_consumed_pcm_alignment"] = "complete_segment_only; intra_segment_alignment_missing"
    branch.setdefault("client_consumed_pcm_metrics", {})["intra_segment_alignment_missing"] = True
    return True

def _record_emitted_pcm_packet(branch: dict[str, Any], packet: Any) -> None:
    """Track exact server-emitted sample ranges per speaker subsegment."""
    if not isinstance(branch, dict):
        return
    if branch.get("playback_cancelled") is True or branch.get("speech_ack_retired") is True:
        raise RuntimeError("cancelled response cannot emit additional PCM")
    try:
        response_id = str(branch["response_id"])
        generation_id = int(branch["session_generation_id"])
        segment_id = str(packet.speaker_segment_id)
        packet_generation = int(packet.generation_id)
        start_sample = int(packet.start_sample)
        end_sample = int(packet.end_sample)
        packet_id = int(packet.packet_id)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("emitted PCM packet has no complete segment identity") from exc
    if (
        not response_id or not segment_id or packet_generation != generation_id
        or start_sample < 0 or end_sample <= start_sample or packet_id < 0
    ):
        raise RuntimeError("emitted PCM packet has an invalid sample range or identity")
    segments = branch.setdefault("playback_segments", {})
    if not isinstance(segments, dict):
        raise RuntimeError("playback segment registry is not a mapping")
    segment = segments.setdefault(segment_id, {
        "response_id": response_id,
        "generation_id": generation_id,
        "speaker_segment_id": segment_id,
        "start_sample": start_sample,
        "end_sample": None,
        "emitted_end_sample": start_sample,
        "client_consumed_end_sample": start_sample,
        "sample_rate": CODEC_SAMPLE_RATE,
        "done_emitted": False,
        "token_packets": (),
        "packet_ids": [],
        "emitted_pcm_packets": [],
    })
    if (
        segment.get("response_id") != response_id
        or segment.get("generation_id") != generation_id
        or segment.get("speaker_segment_id") != segment_id
        or segment.get("sample_rate") != CODEC_SAMPLE_RATE
        or segment.get("done_emitted") is True
        or start_sample != int(segment.get("emitted_end_sample", -1))
    ):
        raise RuntimeError("emitted PCM packet is stale, non-contiguous, or follows segment_done")
    pcm = np.asarray(packet.pcm, dtype=np.float32).reshape(-1)
    if pcm.size != end_sample - start_sample or not np.isfinite(pcm).all():
        raise RuntimeError("emitted PCM packet has a sample-range mismatch")
    emitted_pcm_packets = segment.setdefault("emitted_pcm_packets", [])
    emitted_pcm_packets.append({
        "start_sample": start_sample,
        "end_sample": end_sample,
        "pcm": np.array(pcm, dtype=np.float32, copy=True),
        "packet_id": packet_id,
    })
    segment["emitted_end_sample"] = end_sample
    _enforce_emitted_pcm_retention(branch)
    segment["pcm_samples_emitted"] = int(end_sample - int(segment["start_sample"]))
    segment["packet_ids"].append(packet_id)
    ledger = branch.get("response_event_ledger")
    record_sent = getattr(ledger, "record_pcm_sent", None)
    if callable(record_sent):
        record_sent(packet)

def _complete_client_pcm_segment(
    branch: dict[str, Any],
    *,
    speaker_segment_id: str,
    token_packets: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any] | None:
    """Close one emitted unit and return its wire marker with exact sample bounds."""
    if not isinstance(branch, dict):
        return None
    if branch.get("playback_cancelled") is True or branch.get("speech_ack_retired") is True:
        return None
    segment_id = str(speaker_segment_id)
    segments = branch.get("playback_segments")
    segment = segments.get(segment_id) if isinstance(segments, dict) else None
    if not isinstance(segment, dict) or segment.get("done_emitted") is True:
        return None
    start_sample = segment.get("start_sample")
    end_sample = segment.get("emitted_end_sample")
    generation_id = segment.get("generation_id")
    response_id = segment.get("response_id")
    if (
        isinstance(start_sample, bool) or not isinstance(start_sample, int)
        or isinstance(end_sample, bool) or not isinstance(end_sample, int)
        or end_sample < start_sample
        or generation_id != branch.get("session_generation_id")
        or response_id != branch.get("response_id")
    ):
        return None
    owned_packets = tuple(dict(packet) for packet in token_packets)
    token_cursors = [
        int(packet["exclusive_kv_cursor"])
        for packet in owned_packets
        if isinstance(packet.get("exclusive_kv_cursor"), int)
        and not isinstance(packet.get("exclusive_kv_cursor"), bool)
    ]
    token_ids = [
        int(packet["token_id"])
        for packet in owned_packets
        if isinstance(packet.get("token_id"), int)
        and not isinstance(packet.get("token_id"), bool)
    ]
    consumed_end = segment.get("client_consumed_end_sample", start_sample)
    ledger = branch.get("response_event_ledger")
    record_closed = getattr(ledger, "record_closed_tts_unit", None)
    if callable(record_closed):
        record_closed(
            speaker_segment_id=segment_id,
            end_sample=int(end_sample),
            source_tokens=owned_packets,
        )
    segment.update({
        "end_sample": end_sample,
        "done_emitted": True,
        "token_packets": owned_packets,
        "original_token_ids": token_ids,
        "original_token_cursor_start": min(token_cursors) - 1 if token_cursors else None,
        "original_token_cursor_end": max(token_cursors) if token_cursors else None,
        "pcm_samples_emitted": int(end_sample - start_sample),
        "pcm_samples_consumed": int(max(0, int(consumed_end) - start_sample)),
        "client_consumed_end_sample": int(consumed_end),
        "client_consumed": int(consumed_end) == int(end_sample),
        "client_consumed_final": bool(segment.get("client_consumed_final", False)),
        "termination_reason": (
            str(owned_packets[0].get("termination_reason"))
            if owned_packets and isinstance(owned_packets[0].get("termination_reason"), str)
            else None
        ),
    })
    if segment["client_consumed"]:
        completed = set(branch.get("client_consumed_segments", ()))
        if segment_id not in completed:
            _commit_client_consumed_segment(branch, segment_id)
            branch["client_consumed_segments"] = tuple((*completed, segment_id))
    return {
        "type": "response.audio.segment_done",
        "response_id": str(response_id),
        "generation_id": int(generation_id),
        "speaker_segment_id": segment_id,
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
        "sample_rate": CODEC_SAMPLE_RATE,
    }

def _require_closed_tts_codec_eos(metrics: Mapping[str, Any]) -> None:
    """Require sampled codec EOS before a closed TTS unit can advance KV."""
    if not isinstance(metrics, Mapping) or metrics.get("eos_detected") is not True:
        raise RuntimeError("Qwen Talker closed-unit segment did not reach codec EOS")

def _turn_local_vocalized_prefix_snapshot(branch: dict[str, Any]) -> tuple[int | None, str]:
    lock = branch.get("spoken_prefix_lock")
    if lock is None:
        return None, str(branch.get("spoken_prefix_text", ""))
    with lock:
        position = branch.get("spoken_prefix_kv_position")
        return (None if position is None else int(position), str(branch.get("spoken_prefix_text", "")))

def _wait_for_listener_backlog_admission(
    queue_depth: Callable[[], int],
    generation_is_current: Callable[[], bool],
    *,
    callbacks_active: bool = False,
    listener_work_pending: Callable[[], bool] | None = None,
    wait_for_listener_clear: Callable[[Callable[[], bool]], bool] | None = None,
) -> ListenerBacklogAdmission:
    """Wait cooperatively for listener demand without relying on ingress depth.

    A dispatcher-provided condition waiter covers both queued frames and rows
    it already owns. The standalone fallback is intentionally non-blocking; a
    caller with active demand must supply the wait callback rather than infer
    idleness from an empty ingress queue.
    """
    if not callable(queue_depth):
        raise TypeError("queue_depth must be callable")
    if not callable(generation_is_current):
        raise TypeError("generation_is_current must be callable")
    if wait_for_listener_clear is not None and not callable(wait_for_listener_clear):
        raise TypeError("wait_for_listener_clear must be callable or None")
    if listener_work_pending is not None and not callable(listener_work_pending):
        raise TypeError("listener_work_pending must be callable or None")

    if not generation_is_current():
        return ListenerBacklogAdmission.ABORTED
    del callbacks_active  # Diagnostic state only; shared listener work is authoritative.
    if wait_for_listener_clear is not None:
        admitted = bool(wait_for_listener_clear(generation_is_current))
        if not generation_is_current():
            return ListenerBacklogAdmission.ABORTED
        return (
            ListenerBacklogAdmission.ADMITTED
            if admitted
            else ListenerBacklogAdmission.ABORTED
        )

    pending = bool(listener_work_pending()) if listener_work_pending is not None else False
    if queue_depth() == 0 and not pending:
        return ListenerBacklogAdmission.ADMITTED
    return ListenerBacklogAdmission.ABORTED
