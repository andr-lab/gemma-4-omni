"""Response speech ACK stream lifecycle helpers."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from duplex_kernel.audio_stream import AudioPacket
from duplex_kernel.playout import PlayoutFrame

@dataclass(frozen=True)
class _ClosedTTSSegment:
    """Event-loop fence emitted after one text-owned Qwen unit is fully drained."""

    speaker_segment_id: str
    source_tokens: tuple[dict[str, Any], ...]

def _ack_emitted_audio_packet(
    packet: Any,
    *,
    generation_id: int,
    response_segment_id: str,
    live_stream: Any | None,
    closed_unit_streams: Mapping[str, Any],
) -> bool:
    """Advance token speech watermarks only for a released PCM packet's frames."""
    try:
        packet_generation = int(packet.generation_id)
        packet_segment = str(packet.speaker_segment_id)
        first_frame_id = int(packet.first_frame_id)
        last_frame_id = int(packet.last_frame_id)
    except (AttributeError, TypeError, ValueError):
        return False
    if (
        packet_generation != int(generation_id)
        or not packet_segment
        or first_frame_id < 0
        or last_frame_id < first_frame_id
    ):
        return False
    stream = closed_unit_streams.get(packet_segment)
    if stream is None and packet_segment == str(response_segment_id):
        stream = live_stream
    if stream is None:
        # Static or non-speech audio has no token stream to acknowledge.
        return True
    try:
        stream_generation = int(stream.generation_id)
        stream_segment = int(stream.segment_id)
    except (AttributeError, TypeError, ValueError):
        return False
    acknowledge = getattr(stream, "ack_emitted_audio_frame", None)
    if stream_generation != packet_generation or not callable(acknowledge):
        return False
    for frame_id in range(first_frame_id, last_frame_id + 1):
        acknowledge(
            frame_id,
            generation_id=packet_generation,
            segment_id=stream_segment,
        )
    return True

def _response_speech_ack_lock(branch: dict[str, Any]) -> Any:
    lock = branch.get("speech_ack_lock")
    if lock is None:
        lock = threading.RLock()
        branch["speech_ack_lock"] = lock
    return lock

def _register_response_speech_ack_streams(
    branch: dict[str, Any],
    *,
    response_id: str,
    generation_id: int,
    response_segment_id: str,
    live_stream: Any | None,
    closed_unit_streams: dict[str, Any],
    closed_unit_events: dict[str, threading.Event],
    required: bool,
) -> bool:
    """Bind V1/V2 speech ACK resources to their exact response branch."""
    if not isinstance(branch, dict):
        return False
    lock = _response_speech_ack_lock(branch)
    with lock:
        if (
            branch.get("speech_ack_retired") is True
            or str(branch.get("response_id", "")) != str(response_id)
            or int(branch.get("session_generation_id", -1)) != int(generation_id)
            or str(branch.get("speaker_segment_id", "")) != str(response_segment_id)
        ):
            return False
        if live_stream is not None:
            try:
                if int(live_stream.generation_id) != int(generation_id):
                    return False
            except (AttributeError, TypeError, ValueError):
                return False
        branch["speech_ack_resources"] = {
            "response_id": str(response_id),
            "generation_id": int(generation_id),
            "response_segment_id": str(response_segment_id),
            "live_stream": live_stream,
            "closed_unit_streams": closed_unit_streams,
            "closed_unit_events": closed_unit_events,
        }
        branch["speech_ack_required"] = bool(required)
        return True

def _response_speech_ack_resources(
    branch: dict[str, Any],
    *,
    response_id: str,
    generation_id: int,
    response_segment_id: str,
) -> dict[str, Any] | None:
    """Resolve resources only when the response and generation still own them."""
    if not isinstance(branch, dict):
        return None
    lock = _response_speech_ack_lock(branch)
    with lock:
        if (
            branch.get("speech_ack_retired") is True
            or str(branch.get("response_id", "")) != str(response_id)
            or int(branch.get("session_generation_id", -1)) != int(generation_id)
            or str(branch.get("speaker_segment_id", "")) != str(response_segment_id)
        ):
            return None
        resources = branch.get("speech_ack_resources")
        if resources is None:
            if branch.get("speech_ack_required") is True:
                return None
            return {
                "response_id": str(response_id),
                "generation_id": int(generation_id),
                "response_segment_id": str(response_segment_id),
                "live_stream": None,
                "closed_unit_streams": {},
                "closed_unit_events": {},
            }
        if (
            str(resources.get("response_id", "")) != str(response_id)
            or int(resources.get("generation_id", -1)) != int(generation_id)
            or str(resources.get("response_segment_id", "")) != str(response_segment_id)
        ):
            return None
        return resources

def _make_finish_audio_segment_callback(
    *,
    active_output_segment_id: dict[str, str],
    playout: Any,
    packetizer: Any,
    ack_resources_branch: dict[str, Any],
    response_id: str,
    generation_id: int,
    response_segment_id: str,
    packetize_frames_as_packets: Callable[[list[PlayoutFrame]], list[AudioPacket]],
    queue_response_output: Callable[..., bool],
    audio_queue: Any,
) -> Callable[[str, Sequence[Mapping[str, Any]]], None]:
    """Build the response-scoped callback that flushes a closed TTS unit."""

    def finish_audio_segment(
        segment_id: str,
        source_tokens: Sequence[Mapping[str, Any]],
    ) -> None:
        normalized_segment_id = str(segment_id)
        if normalized_segment_id != active_output_segment_id["value"]:
            raise RuntimeError("closed TTS segment does not own the active playout")
        if playout is None or packetizer is None:
            raise RuntimeError("closed TTS segment has no playout or packetizer")
        ack_resources = _response_speech_ack_resources(
            ack_resources_branch,
            response_id=response_id,
            generation_id=generation_id,
            response_segment_id=response_segment_id,
        )
        if ack_resources is None:
            raise RuntimeError("closed TTS segment lost its response speech ACK owner")
        emitted_events = ack_resources["closed_unit_events"]
        if normalized_segment_id in emitted_events:
            raise RuntimeError("closed TTS segment emission identity was reused")
        emitted_events[normalized_segment_id] = threading.Event()
        packets = packetize_frames_as_packets(
            playout.flush_segment(
                generation_id=generation_id,
                speaker_segment_id=normalized_segment_id,
            )
        )
        packets.extend(packetizer.flush_segment(generation_id, normalized_segment_id))
        burst_items = len(packets) + 1
        for packet in packets:
            if not queue_response_output(
                packet,
                queue_kind="audio",
                output_queue=audio_queue,
                burst_items=burst_items,
            ):
                return
        marker = _ClosedTTSSegment(
            speaker_segment_id=normalized_segment_id,
            source_tokens=tuple(dict(row) for row in source_tokens),
        )
        queue_response_output(
            marker,
            queue_kind="audio",
            output_queue=audio_queue,
            burst_items=burst_items,
        )

    return finish_audio_segment

def _retire_response_speech_ack_streams(branch: dict[str, Any]) -> None:
    """Fence late ACKs and release stream references after response cleanup."""
    if not isinstance(branch, dict):
        return
    lock = _response_speech_ack_lock(branch)
    with lock:
        branch["speech_ack_retired"] = True
        resources = branch.get("speech_ack_resources")
        if isinstance(resources, dict):
            resources["live_stream"] = None
            closed_streams = resources.get("closed_unit_streams")
            if isinstance(closed_streams, dict):
                closed_streams.clear()
            closed_events = resources.get("closed_unit_events")
            if isinstance(closed_events, dict):
                closed_events.clear()

def _ack_response_speech_packet(
    branch: dict[str, Any],
    packet: Any,
    *,
    response_id: str,
    generation_id: int,
    response_segment_id: str,
) -> bool:
    """Acknowledge released PCM through resources owned by this response only."""
    if not isinstance(branch, dict):
        return False
    lock = _response_speech_ack_lock(branch)
    with lock:
        resources = _response_speech_ack_resources(
            branch,
            response_id=response_id,
            generation_id=generation_id,
            response_segment_id=response_segment_id,
        )
        if resources is None:
            return False
        return _ack_emitted_audio_packet(
            packet,
            generation_id=int(generation_id),
            response_segment_id=str(response_segment_id),
            live_stream=resources["live_stream"],
            closed_unit_streams=resources["closed_unit_streams"],
        )
