import asyncio
import queue
import threading
from types import SimpleNamespace

import numpy as np

from duplex_kernel.config import AUDIO_FRAME_SAMPLES
from duplex_kernel.playout import SpeakerPlayoutBuffer
from duplex_kernel.server import (
    _ack_emitted_audio_packet,
    _audio_packet_matches_emission_owner,
    _cancel_tts_segment_emission_waiters,
    _legacy_audio_chunk_conditioning_horizon,
    _qwen_text_token_surfaces,
    _release_pcm_packet_then_ack,
    _signal_emitted_tts_segment,
    _wait_for_emitted_tts_segment,
)


class _FakeSpeechStream:
    def __init__(self, *, generation_id=7, segment_id=2):
        self.generation_id = generation_id
        self.segment_id = segment_id
        self.frames = []

    def ack_emitted_audio_frame(self, frame_id, *, generation_id, segment_id):
        self.frames.append((frame_id, generation_id, segment_id))


def _packet(*, generation_id=7, segment_id="response.seg0", first=4, last=6):
    return SimpleNamespace(
        generation_id=generation_id,
        speaker_segment_id=segment_id,
        first_frame_id=first,
        last_frame_id=last,
    )


def test_legacy_chunk_horizon_releases_current_chunk_and_keeps_stateful_delay() -> None:
    legacy = SpeakerPlayoutBuffer(delay_frames=0)
    legacy.begin_segment(generation_id=4, speaker_segment_id="legacy-response")
    next_frame_id = 0
    released_ids = []
    chunk = np.zeros(2 * AUDIO_FRAME_SAMPLES, dtype=np.float32)

    for _ in range(3):
        horizon = _legacy_audio_chunk_conditioning_horizon(next_frame_id, chunk.size)
        pushed = legacy.push_chunk(
            chunk,
            first_frame_id=next_frame_id,
            generation_id=4,
            speaker_segment_id="legacy-response",
        )
        next_frame_id += len(pushed)
        released_chunk_ids = [
            frame.frame_id
            for frame in legacy.release_ready(conditioning_frame_id=horizon)
        ]
        assert released_chunk_ids == list(range(next_frame_id - 2, next_frame_id))
        released_ids.extend(released_chunk_ids)

    assert released_ids == [0, 1, 2, 3, 4, 5]

    stateful = SpeakerPlayoutBuffer(delay_frames=1)
    stateful.begin_segment(generation_id=5, speaker_segment_id="stateful-response")
    pushed = stateful.push_chunk(
        chunk,
        first_frame_id=0,
        generation_id=5,
        speaker_segment_id="stateful-response",
    )
    assert [frame.frame_id for frame in pushed] == [0, 1]
    assert stateful.release_ready(conditioning_frame_id=0) == []
    assert [frame.frame_id for frame in stateful.release_ready(conditioning_frame_id=1)] == [0]
    assert [frame.frame_id for frame in stateful.release_ready(conditioning_frame_id=2)] == [1]
    pushed = stateful.push_chunk(
        chunk,
        first_frame_id=2,
        generation_id=5,
        speaker_segment_id="stateful-response",
    )
    assert [frame.frame_id for frame in pushed] == [2, 3]
    assert [frame.frame_id for frame in stateful.release_ready(conditioning_frame_id=3)] == [2]
    assert [frame.frame_id for frame in stateful.release_ready(conditioning_frame_id=4)] == [3]


def test_emission_wrapper_acknowledges_each_frame_only_after_successful_release():
    events = []
    stream = _FakeSpeechStream()
    packet = _packet()

    async def release(_packet):
        events.append("pcm_released")
        return True

    def acknowledge(released_packet):
        events.append("speech_ack")
        return _ack_emitted_audio_packet(
            released_packet,
            generation_id=7,
            response_segment_id="response.seg0",
            live_stream=None,
            closed_unit_streams={"response.seg0": stream},
        )

    assert asyncio.run(_release_pcm_packet_then_ack(
        packet,
        validate=lambda released_packet: _audio_packet_matches_emission_owner(
            released_packet,
            generation_id=7,
            speaker_segment_id="response.seg0",
        ),
        release=release,
        acknowledge=acknowledge,
    ))
    assert events == ["pcm_released", "speech_ack"]
    assert stream.frames == [(4, 7, 2), (5, 7, 2), (6, 7, 2)]


def test_failed_release_and_stale_packet_identity_do_not_acknowledge():
    stream = _FakeSpeechStream()
    packet = _packet()
    events = []

    async def failed_release(_packet):
        events.append("pcm_release_failed")
        return False

    def acknowledge(_packet):
        events.append("speech_ack")
        return True

    assert not asyncio.run(_release_pcm_packet_then_ack(
        packet, release=failed_release, acknowledge=acknowledge
    ))
    assert events == ["pcm_release_failed"]

    assert not _ack_emitted_audio_packet(
        _packet(generation_id=8),
        generation_id=7,
        response_segment_id="response.seg0",
        live_stream=stream,
        closed_unit_streams={},
    )
    assert _ack_emitted_audio_packet(
        _packet(segment_id="old.segment"),
        generation_id=7,
        response_segment_id="response.seg0",
        live_stream=stream,
        closed_unit_streams={},
    )
    assert stream.frames == []


def test_buffered_old_generation_packet_is_dropped_before_pcm_and_ack():
    current_generation = {"value": 8}
    stream = _FakeSpeechStream(generation_id=7)
    packet = _packet(generation_id=7)
    events = []

    async def release_old_packet(released_packet):
        if released_packet.generation_id != current_generation["value"]:
            events.append("stale_pcm_dropped")
            return False
        events.append("pcm_released")
        return True

    def acknowledge_old_packet(released_packet):
        events.append("speech_ack")
        return _ack_emitted_audio_packet(
            released_packet,
            generation_id=current_generation["value"],
            response_segment_id="response.seg0",
            live_stream=stream,
            closed_unit_streams={},
        )

    # The generation changes while the packet is still buffered for release.
    assert not asyncio.run(_release_pcm_packet_then_ack(
        packet, release=release_old_packet, acknowledge=acknowledge_old_packet
    ))
    assert events == ["stale_pcm_dropped"]
    assert stream.frames == []


def test_buffered_packet_with_wrong_generation_or_segment_is_rejected_before_release():
    events = []

    async def release(_packet):
        events.append("pcm_released")
        return True

    def validate(packet):
        return _audio_packet_matches_emission_owner(
            packet,
            generation_id=8,
            speaker_segment_id="response.seg1",
        )

    for packet in (
        _packet(generation_id=7, segment_id="response.seg1"),
        _packet(generation_id=8, segment_id="response.seg0"),
    ):
        assert not asyncio.run(_release_pcm_packet_then_ack(
            packet,
            validate=validate,
            release=release,
            acknowledge=lambda _packet: events.append("speech_ack") or True,
        ))
    assert events == []


def test_packet_identity_guard_accepts_first_and_next_eos_segment_owners():
    active_segment = {"value": "response.seg0"}
    assert _audio_packet_matches_emission_owner(
        _packet(segment_id="response.seg0"),
        generation_id=7,
        speaker_segment_id=active_segment["value"],
    )
    # The previous unit's close marker has drained before begin_audio_segment
    # switches ownership to the next segment.
    active_segment["value"] = "response.seg1"
    assert _audio_packet_matches_emission_owner(
        _packet(segment_id="response.seg1"),
        generation_id=7,
        speaker_segment_id=active_segment["value"],
    )
    assert not _audio_packet_matches_emission_owner(
        _packet(segment_id="response.seg0"),
        generation_id=7,
        speaker_segment_id=active_segment["value"],
    )


def test_static_qwen_surfaces_keep_whitespace_and_reconstruct_text():
    text = " hello world!"
    surfaces = _qwen_text_token_surfaces(text, ((1, 6), (7, 12), (12, 13)))
    assert surfaces == (" hello", " world", "!")
    assert "".join(surfaces) == text


def test_two_closed_tts_units_wait_for_each_pcm_segment_marker_without_deadlock():
    emitted_events = {name: threading.Event() for name in ("unit0", "unit1")}
    audio_queue = queue.Queue()
    producer_done = threading.Event()
    drained = []

    def producer():
        audio_queue.put(("pcm", "unit0"))
        audio_queue.put(("marker", "unit0"))
        if not _wait_for_emitted_tts_segment(emitted_events, "unit0"):
            return
        audio_queue.put(("pcm", "unit1"))
        audio_queue.put(("marker", "unit1"))
        if _wait_for_emitted_tts_segment(emitted_events, "unit1"):
            drained.append("both")
        producer_done.set()

    thread = threading.Thread(target=producer)
    thread.start()
    assert audio_queue.get(timeout=1) == ("pcm", "unit0")
    assert audio_queue.get(timeout=1) == ("marker", "unit0")
    assert not producer_done.is_set()
    assert _signal_emitted_tts_segment(emitted_events, "unit0")
    assert audio_queue.get(timeout=1) == ("pcm", "unit1")
    assert audio_queue.get(timeout=1) == ("marker", "unit1")
    assert not producer_done.is_set()
    assert _signal_emitted_tts_segment(emitted_events, "unit1")
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert producer_done.is_set()
    assert drained == ["both"]


def test_cancel_wakes_segment_release_waiter_and_clears_stream_registry():
    emitted_events = {"unit0": threading.Event()}
    cancelled = threading.Event()
    started = threading.Event()
    result = []

    def wait_for_segment():
        started.set()
        result.append(_wait_for_emitted_tts_segment(
            emitted_events,
            "unit0",
            stop_check=cancelled.is_set,
        ))

    thread = threading.Thread(target=wait_for_segment)
    thread.start()
    assert started.wait(timeout=1)
    cancelled.set()
    _cancel_tts_segment_emission_waiters(emitted_events)
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert emitted_events == {}
    assert result == [False]
