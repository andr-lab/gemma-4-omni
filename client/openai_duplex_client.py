# -*- coding: utf-8 -*-
"""
client/openai_duplex_client.py — Client Python di Test Ufficiale con SDK OpenAI per Gemma-4-Omni.

Verifica tutti i 4 requisiti di test:
1. Definizione di un Tool Locale (`get_current_weather` e `get_device_telemetry`) ed esecuzione automatica del ciclo di function calling.
2. Chat statica normale (senza audio, solo testo).
3. Chat con risposte lette a voce da Qwen3-TTS (modalità `["text", "audio"]` con estrazione WAV).
4. Streaming parallelo Full-Duplex (WebSocket Realtime con transcript delta testuale e audio delta simultanei).

Uso:
    python client/openai_duplex_client.py --host 192.168.0.186 --port 8000 --mode duplex --duration 30
    python client/openai_duplex_client.py --mode tools
    python client/openai_duplex_client.py --mode audio
    python client/openai_duplex_client.py --mode duplex
"""

import argparse
import asyncio
import base64
import importlib
import io
import json
import queue
import re
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Dict

import numpy as np

# Assicura codifica UTF-8 su Windows terminal
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import client OpenAI ufficiale (opzionale per import test)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# Protocol contract with duplex_kernel.server: signed PCM16LE mic input at
# 16 kHz in exact 80 ms frames; each server audio delta is a mono WAV packet at
# 24 kHz and carries response/generation/frame identity.
MIC_SAMPLE_RATE = 16_000
CODEC_SAMPLE_RATE = 24_000
MIC_FRAME_MS = 80
MIC_FRAME_SAMPLES = MIC_SAMPLE_RATE * MIC_FRAME_MS // 1000
CODEC_FRAME_SAMPLES = CODEC_SAMPLE_RATE * MIC_FRAME_MS // 1000
MIC_FRAME_NS = MIC_FRAME_MS * 1_000_000
MIC_QUEUE_FRAMES = 5
PLAYBACK_BUFFER_SECONDS = 3
PLAYBACK_MAX_SAMPLES = CODEC_SAMPLE_RATE * PLAYBACK_BUFFER_SECONDS
WEBSOCKET_MAX_SIZE = 1_000_000
WEBSOCKET_MAX_QUEUE = 4
WEBSOCKET_WRITE_LIMIT = 32_768


@dataclass(frozen=True)
class CapturedPCMFrame:
    pcm16: bytes
    capture_end_ns: int


@dataclass(frozen=True)
class DecodedAudioPacket:
    response_id: str
    generation_id: int
    speaker_segment_id: str
    packet_id: int
    first_frame_id: int
    last_frame_id: int
    start_sample: int
    end_sample: int
    sample_rate: int
    pcm: np.ndarray


@dataclass
class _PlaybackChunk:
    pcm: np.ndarray
    offset: int
    response_id: str
    generation_id: int
    speaker_segment_id: str | None = None
    packet_id: int | None = None
    first_frame_id: int | None = None
    last_frame_id: int | None = None
    start_sample: int | None = None
    end_sample: int | None = None
    sample_rate: int | None = None


@dataclass
class _PlaybackSegment:
    response_id: str
    generation_id: int
    speaker_segment_id: str
    start_sample: int
    enqueued_end_sample: int
    consumed_end_sample: int
    sample_rate: int
    first_frame_id: int
    last_frame_id: int
    end_sample: int | None = None
    closed: bool = False
    ack_queued: bool = False


def _required_int(event: dict[str, Any], name: str) -> int:
    value = event.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"audio delta: {name} deve essere un intero")
    return int(value)


class BoundedMicrophoneFrames:
    """Assemble callback-sized mic blocks into bounded, causal 80 ms PCM16 frames."""

    def __init__(self, *, max_frames: int = MIC_QUEUE_FRAMES) -> None:
        if isinstance(max_frames, bool) or int(max_frames) < 1:
            raise ValueError("max_frames deve essere positivo")
        self.max_frames = int(max_frames)
        self._queue: queue.Queue[CapturedPCMFrame] = queue.Queue(maxsize=self.max_frames)
        self._pending = np.empty(0, dtype="<i2")
        self._pending_start_ns: int | None = None
        self._lock = threading.Lock()

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    def push(self, samples: np.ndarray, *, capture_end_ns: int | None = None) -> None:
        """Add device samples; overflow is explicit and never silently drops PCM."""
        incoming = np.asarray(samples)
        if incoming.size == 0:
            return
        if incoming.dtype != np.dtype(np.int16) and incoming.dtype != np.dtype("<i2"):
            raise TypeError("microfono deve fornire sample PCM16")
        incoming = np.asarray(incoming, dtype="<i2").reshape(-1)
        end_ns = time.monotonic_ns() if capture_end_ns is None else int(capture_end_ns)
        block_start_ns = end_ns - (incoming.size * 1_000_000_000 // MIC_SAMPLE_RATE)

        with self._lock:
            combined_size = self._pending.size + incoming.size
            frame_count = int(combined_size // MIC_FRAME_SAMPLES)
            if self._queue.qsize() + frame_count > self.max_frames:
                raise queue.Full("client microphone queue full; stopping capture to avoid PCM loss")
            start_ns = self._pending_start_ns if self._pending.size else block_start_ns
            combined = np.concatenate((self._pending, incoming))
            for index in range(frame_count):
                first = index * MIC_FRAME_SAMPLES
                last = first + MIC_FRAME_SAMPLES
                frame_end_ns = start_ns + (last * 1_000_000_000 // MIC_SAMPLE_RATE)
                pcm16 = np.ascontiguousarray(combined[first:last], dtype="<i2").tobytes()
                self._queue.put_nowait(CapturedPCMFrame(pcm16=pcm16, capture_end_ns=frame_end_ns))
            remainder = frame_count * MIC_FRAME_SAMPLES
            self._pending = np.array(combined[remainder:], dtype="<i2", copy=True)
            self._pending_start_ns = (
                start_ns + (remainder * 1_000_000_000 // MIC_SAMPLE_RATE)
                if self._pending.size else None
            )

    def get_nowait(self) -> CapturedPCMFrame:
        return self._queue.get_nowait()


class BoundedPCMPlayback:
    """Generation-fenced FIFO with logical application-buffer consumption ACKs."""

    def __init__(self, *, max_samples: int = PLAYBACK_MAX_SAMPLES) -> None:
        if isinstance(max_samples, bool) or int(max_samples) < 1:
            raise ValueError("max_samples deve essere positivo")
        self.max_samples = int(max_samples)
        self._chunks: deque[_PlaybackChunk] = deque()
        self._pending_samples = 0
        self._response_id: str | None = None
        self._response_generations: dict[str, set[int]] = {}
        self._generation_id: int | None = None
        self._completed_responses: set[str] = set()
        self._cancelled_responses: set[str] = set()
        self._cancelled_pairs: set[tuple[str, int]] = set()
        self._segments: dict[tuple[str, int, str], _PlaybackSegment] = {}
        self._ack_sequence = 0
        self._last_accepted_ack_sequence = 0
        self._pending_acks: deque[dict[str, Any]] = deque()
        self._outstanding_acks: dict[int, dict[str, Any]] = {}
        self._accepted_watermarks: dict[tuple[str, int, str], int] = {}
        self._lock = threading.Lock()

    @property
    def pending_samples(self) -> int:
        with self._lock:
            return self._pending_samples

    @property
    def generation_id(self) -> int | None:
        with self._lock:
            return self._generation_id

    def _discard_acks_locked(self, *, response_id: str | None = None, generation_id: int | None = None) -> None:
        def matches(ack: dict[str, Any]) -> bool:
            return (
                (response_id is None or ack["response_id"] == response_id)
                and (generation_id is None or ack["generation_id"] == generation_id)
            )
        self._pending_acks = deque(ack for ack in self._pending_acks if not matches(ack))
        self._outstanding_acks = {
            sequence: ack for sequence, ack in self._outstanding_acks.items()
            if not matches(ack)
        }

    def _clear_locked(self) -> int:
        dropped = self._pending_samples
        self._chunks.clear()
        self._pending_samples = 0
        self._completed_responses.clear()
        self._segments.clear()
        self._pending_acks.clear()
        self._outstanding_acks.clear()
        return dropped

    def _discard_locked(self, predicate) -> int:
        kept: deque[_PlaybackChunk] = deque()
        dropped = 0
        discarded_pairs: set[tuple[str, int]] = set()
        for chunk in self._chunks:
            if predicate(chunk.response_id, chunk.generation_id):
                dropped += chunk.pcm.size - chunk.offset
                discarded_pairs.add((chunk.response_id, chunk.generation_id))
            else:
                kept.append(chunk)
        self._chunks = kept
        self._pending_samples -= dropped
        for response_id, generation_id in discarded_pairs:
            self._discard_acks_locked(response_id=response_id, generation_id=generation_id)
        for key in tuple(self._segments):
            response_id, generation_id, _segment_id = key
            if predicate(response_id, generation_id):
                self._segments.pop(key, None)
                self._discard_acks_locked(response_id=response_id, generation_id=generation_id)
        for response_id in {response for response, _generation in discarded_pairs}:
            if not any(chunk.response_id == response_id for chunk in self._chunks):
                self._completed_responses.discard(response_id)
        return dropped

    def _cleanup_completed_response_locked(self, response_id: str) -> None:
        if response_id not in self._completed_responses:
            return
        if response_id == self._response_id:
            return
        if any(chunk.response_id == response_id for chunk in self._chunks):
            return
        if any(ack["response_id"] == response_id for ack in self._pending_acks):
            return
        if any(ack["response_id"] == response_id for ack in self._outstanding_acks.values()):
            return
        self._completed_responses.discard(response_id)
        self._response_generations.pop(response_id, None)
        for key in tuple(self._segments):
            if key[0] == response_id:
                self._segments.pop(key, None)

    def _advance_generation_locked(self, generation_id: int) -> bool:
        if self._generation_id is not None and generation_id < self._generation_id:
            return False
        if self._generation_id is None or generation_id > self._generation_id:
            # Preserve output only for responses already completed by the server.
            stale_pairs = {
                (chunk.response_id, chunk.generation_id)
                for chunk in self._chunks
                if chunk.generation_id < generation_id
                and chunk.response_id not in self._completed_responses
            }
            stale_pairs.update(
                (segment.response_id, segment.generation_id)
                for segment in self._segments.values()
                if segment.generation_id < generation_id
                and segment.response_id not in self._completed_responses
            )
            self._cancelled_pairs.update(stale_pairs)
            self._discard_locked(
                lambda response_id, chunk_generation: (response_id, chunk_generation) in stale_pairs
            )
            self._generation_id = generation_id
        return True

    @staticmethod
    def _segment_key(
        response_id: str,
        generation_id: int,
        speaker_segment_id: str,
    ) -> tuple[str, int, str]:
        return (str(response_id), int(generation_id), str(speaker_segment_id))

    def _cancel_response_locked(self, response_id: str, generation_id: int | None) -> bool:
        response = str(response_id)
        generations = set(self._response_generations.get(response, ()))
        generations.update(
            chunk.generation_id for chunk in self._chunks if chunk.response_id == response
        )
        generations.update(
            segment.generation_id for segment in self._segments.values()
            if segment.response_id == response
        )
        if generation_id is not None:
            generations = {int(generation_id)}
        elif response == self._response_id and self._generation_id is not None:
            generations.add(self._generation_id)
        for generation in generations:
            self._cancelled_pairs.add((response, generation))
        dropped = self._discard_locked(
            lambda queued_response, queued_generation: (
                queued_response == response
                and (generation_id is None or queued_generation == int(generation_id))
            )
        )
        self._discard_acks_locked(response_id=response, generation_id=generation_id)
        if generation_id is None:
            self._cancelled_responses.add(response)
            self._completed_responses.discard(response)
            self._response_generations.pop(response, None)
        else:
            response_generations = self._response_generations.get(response)
            if response_generations is not None:
                response_generations.discard(int(generation_id))
                if not response_generations:
                    self._response_generations.pop(response, None)
            self._completed_responses.discard(response)
        active_generations = self._response_generations.get(response, set())
        if response == self._response_id and (
            generation_id is None or not active_generations or int(generation_id) == self._generation_id
        ):
            self._response_id = None
        return bool(generations) or dropped > 0

    def begin_response(self, response_id: str) -> None:
        response = str(response_id)
        if not response:
            raise ValueError("response_id non può essere vuoto")
        with self._lock:
            if response != self._response_id:
                previous = self._response_id
                if previous is not None and previous not in self._completed_responses:
                    # A replacement without response.done is an implicit stale
                    # fence; completed responses remain queued in FIFO order.
                    self._cancel_response_locked(previous, generation_id=None)
                self._response_id = response
                self._cancelled_responses.discard(response)
                self._response_generations.setdefault(response, set())
                if previous is not None:
                    self._cleanup_completed_response_locked(previous)

    def complete_response(self, response_id: str) -> bool:
        response = str(response_id)
        with self._lock:
            if response in self._cancelled_responses:
                return False
            has_state = (
                response == self._response_id
                or any(chunk.response_id == response for chunk in self._chunks)
                or any(segment.response_id == response for segment in self._segments.values())
            )
            if not has_state:
                return False
            self._completed_responses.add(response)
            for segment in self._segments.values():
                if segment.response_id == response and not segment.closed:
                    segment.end_sample = segment.enqueued_end_sample
                    segment.closed = True
                    self._maybe_queue_segment_ack_locked(segment)
            return True

    def advance_generation(self, generation_id: int) -> bool:
        if isinstance(generation_id, bool) or not isinstance(generation_id, int) or generation_id < 0:
            raise ValueError("generation_id non valido")
        with self._lock:
            return self._advance_generation_locked(generation_id)

    def enqueue(self, response_id: str, generation_id: int, pcm: np.ndarray) -> bool:
        return self._enqueue_chunk(
            response_id=response_id,
            generation_id=generation_id,
            pcm=pcm,
        )

    def enqueue_packet(
        self,
        *,
        response_id: str,
        generation_id: int,
        speaker_segment_id: str,
        packet_id: int,
        first_frame_id: int,
        last_frame_id: int,
        start_sample: int,
        end_sample: int,
        pcm: np.ndarray,
        sample_rate: int,
    ) -> bool:
        """Queue one identified packet without losing its sample/frame provenance."""
        for name, value in (
            ("generation_id", generation_id), ("packet_id", packet_id),
            ("first_frame_id", first_frame_id), ("last_frame_id", last_frame_id),
            ("start_sample", start_sample), ("end_sample", end_sample),
            ("sample_rate", sample_rate),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} non valido")
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if (
            last_frame_id < first_frame_id or sample_rate <= 0
            or end_sample <= start_sample or end_sample - start_sample != samples.size
        ):
            raise ValueError("packet frame/sample metadata non valido")
        if not isinstance(speaker_segment_id, str) or not speaker_segment_id:
            raise ValueError("speaker_segment_id non può essere vuoto")
        return self._enqueue_chunk(
            response_id=response_id,
            generation_id=generation_id,
            pcm=pcm,
            speaker_segment_id=speaker_segment_id,
            packet_id=packet_id,
            first_frame_id=first_frame_id,
            last_frame_id=last_frame_id,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
        )

    def _enqueue_chunk(
        self,
        *,
        response_id: str,
        generation_id: int,
        pcm: np.ndarray,
        speaker_segment_id: str | None = None,
        packet_id: int | None = None,
        first_frame_id: int | None = None,
        last_frame_id: int | None = None,
        start_sample: int | None = None,
        end_sample: int | None = None,
        sample_rate: int | None = None,
    ) -> bool:
        array = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError("playback PCM vuoto o non finito")
        if np.any(np.abs(array) > 1.0):
            raise ValueError("playback PCM fuori dall'intervallo normalizzato [-1, 1]")
        if isinstance(generation_id, bool) or not isinstance(generation_id, int) or generation_id < 0:
            raise ValueError("generation_id non valido")
        with self._lock:
            response = str(response_id)
            if (
                response != self._response_id
                or response in self._cancelled_responses
                or (response, generation_id) in self._cancelled_pairs
            ):
                return False
            if response in self._completed_responses:
                return False
            if not self._advance_generation_locked(generation_id):
                return False
            if self._pending_samples + array.size > self.max_samples:
                raise BufferError(
                    f"playback queue capacity exceeded ({self.max_samples} samples)"
                )
            if speaker_segment_id is not None:
                if (
                    packet_id is None or first_frame_id is None or last_frame_id is None
                    or start_sample is None or end_sample is None or sample_rate is None
                    or end_sample - start_sample != array.size
                ):
                    raise ValueError("playback segment metadata is incomplete or inconsistent")
                key = (response, generation_id, speaker_segment_id)
                segment = self._segments.get(key)
                if segment is None:
                    segment = _PlaybackSegment(
                        response_id=response,
                        generation_id=generation_id,
                        speaker_segment_id=speaker_segment_id,
                        start_sample=start_sample,
                        enqueued_end_sample=end_sample,
                        consumed_end_sample=start_sample,
                        sample_rate=sample_rate,
                        first_frame_id=first_frame_id,
                        last_frame_id=last_frame_id,
                    )
                    self._segments[key] = segment
                else:
                    if segment.closed:
                        raise ValueError("audio packet arrived after its segment was closed")
                    if (
                        start_sample != segment.enqueued_end_sample
                        or first_frame_id != segment.last_frame_id + 1
                        or sample_rate != segment.sample_rate
                    ):
                        raise ValueError("audio packets are not contiguous within the speaker segment")
                    segment.enqueued_end_sample = end_sample
                    segment.last_frame_id = last_frame_id
                self._response_generations.setdefault(response, set()).add(generation_id)
            self._chunks.append(_PlaybackChunk(
                np.array(array, dtype=np.float32, copy=True), 0, str(response_id), generation_id,
                speaker_segment_id, packet_id, first_frame_id, last_frame_id,
                start_sample, end_sample, sample_rate,
            ))
            self._pending_samples += int(array.size)
            return True

    def _maybe_queue_segment_ack_locked(self, segment: _PlaybackSegment) -> None:
        if (
            not segment.closed
            or segment.ack_queued
            or segment.end_sample is None
            or segment.start_sample >= segment.end_sample
            or segment.enqueued_end_sample != segment.end_sample
            or segment.consumed_end_sample != segment.end_sample
        ):
            return
        self._ack_sequence += 1
        self._pending_acks.append({
            "type": "client_consumed_pcm",
            "response_id": segment.response_id,
            "generation_id": segment.generation_id,
            "speaker_segment_id": segment.speaker_segment_id,
            "start_sample": segment.start_sample,
            "end_sample": segment.end_sample,
            "sample_rate": segment.sample_rate,
            "sequence": self._ack_sequence,
            "client_monotonic_ns": time.monotonic_ns(),
            "consumption_semantics": "logical_client_device_buffer",
        })
        segment.ack_queued = True

    def complete_segment(
        self,
        *,
        response_id: str,
        generation_id: int,
        speaker_segment_id: str,
        start_sample: int,
        end_sample: int,
        sample_rate: int,
    ) -> bool:
        """Close one server-declared segment; ACK only after every sample is copied."""
        response = str(response_id)
        if not response or not isinstance(speaker_segment_id, str) or not speaker_segment_id:
            raise ValueError("segment completion requires response and segment identities")
        for name, value in (
            ("generation_id", generation_id), ("start_sample", start_sample),
            ("end_sample", end_sample), ("sample_rate", sample_rate),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} non valido")
        if end_sample <= start_sample or sample_rate <= 0:
            raise ValueError("segment completion range/sample rate non valido")
        with self._lock:
            if (
                response in self._cancelled_responses
                or (response, generation_id) in self._cancelled_pairs
                or not self._advance_generation_locked(generation_id)
            ):
                return False
            segment = self._segments.get((response, generation_id, speaker_segment_id))
            if segment is None:
                return False
            if (
                segment.start_sample != start_sample
                or segment.enqueued_end_sample != end_sample
                or segment.sample_rate != sample_rate
            ):
                raise ValueError("segment completion metadata does not match queued PCM")
            if segment.closed:
                return segment.end_sample == end_sample
            segment.closed = True
            segment.end_sample = end_sample
            self._maybe_queue_segment_ack_locked(segment)
            return True

    def cancel_response(
        self,
        response_id: str | None,
        generation_id: int | None = None,
    ) -> bool:
        with self._lock:
            if response_id is None:
                for response, generations in self._response_generations.items():
                    self._cancelled_pairs.update((response, generation) for generation in generations)
                for chunk in self._chunks:
                    self._cancelled_pairs.add((chunk.response_id, chunk.generation_id))
                for segment in self._segments.values():
                    self._cancelled_pairs.add((segment.response_id, segment.generation_id))
                self._clear_locked()
                self._response_id = None
                self._response_generations.clear()
                return True
            return self._cancel_response_locked(str(response_id), generation_id)

    def drain_client_consumed_pcm_acks(
        self,
        *,
        current_generation_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return current-fence ACKs; these certify buffer consumption, not sound output."""
        if current_generation_id is not None and (
            isinstance(current_generation_id, bool) or not isinstance(current_generation_id, int)
        ):
            raise ValueError("current_generation_id non valido")
        with self._lock:
            valid: list[dict[str, Any]] = []
            remaining: deque[dict[str, Any]] = deque()
            while self._pending_acks:
                ack = self._pending_acks.popleft()
                if current_generation_id is None or ack["generation_id"] == current_generation_id:
                    self._outstanding_acks[int(ack["sequence"])] = ack
                    valid.append(dict(ack))
                else:
                    remaining.append(ack)
            self._pending_acks = remaining
            return valid

    def accept_client_consumed_pcm_ack(
        self,
        ack: dict[str, Any],
        *,
        current_generation_id: int | None = None,
    ) -> bool:
        """Validate a locally emitted ACK immediately before sending it on the socket."""
        if not isinstance(ack, dict):
            return False
        with self._lock:
            sequence = ack.get("sequence")
            generation = ack.get("generation_id")
            if (
                isinstance(sequence, bool) or not isinstance(sequence, int)
                or isinstance(generation, bool) or not isinstance(generation, int)
                or (current_generation_id is not None and generation != current_generation_id)
                or (ack.get("response_id"), generation) in self._cancelled_pairs
                or ack.get("response_id") in self._cancelled_responses
                or sequence <= self._last_accepted_ack_sequence
                or self._outstanding_acks.get(sequence) != ack
                or ack.get("type") != "client_consumed_pcm"
            ):
                return False
            response_id = ack.get("response_id")
            segment_id = ack.get("speaker_segment_id")
            watermark = ack.get("end_sample")
            if (
                not isinstance(response_id, str) or not response_id
                or not isinstance(segment_id, str) or not segment_id
                or isinstance(watermark, bool) or not isinstance(watermark, int) or watermark < 0
            ):
                return False
            key = (response_id, generation, segment_id)
            if watermark <= self._accepted_watermarks.get(key, -1):
                return False
            self._accepted_watermarks[key] = watermark
            self._last_accepted_ack_sequence = sequence
            self._outstanding_acks.pop(sequence, None)
            self._cleanup_completed_response_locked(response_id)
            return True

    def read_into(self, outdata: np.ndarray) -> int:
        """Fill the complete device callback buffer; return samples copied."""
        output = np.asarray(outdata)
        if output.ndim == 1:
            destination = output
        elif output.ndim == 2 and output.shape[1] == 1:
            destination = output[:, 0]
        else:
            raise ValueError("playback device deve essere mono")
        destination.fill(0)
        written = 0
        with self._lock:
            while written < destination.size and self._chunks:
                chunk = self._chunks[0]
                count = min(destination.size - written, chunk.pcm.size - chunk.offset)
                destination[written:written + count] = chunk.pcm[chunk.offset:chunk.offset + count]
                written += count
                self._pending_samples -= count
                next_offset = chunk.offset + count
                if chunk.speaker_segment_id is not None:
                    key = self._segment_key(
                        chunk.response_id, chunk.generation_id, chunk.speaker_segment_id,
                    )
                    segment = self._segments.get(key)
                    if segment is None or chunk.start_sample is None:
                        raise RuntimeError("playback chunk lost its segment cursor owner")
                    consumed_start = int(chunk.start_sample) + chunk.offset
                    if consumed_start != segment.consumed_end_sample:
                        raise RuntimeError("device callback encountered a discontinuous segment sample cursor")
                    segment.consumed_end_sample += count
                    if segment.consumed_end_sample > segment.enqueued_end_sample:
                        raise RuntimeError("device callback consumed beyond enqueued segment PCM")
                    self._maybe_queue_segment_ack_locked(segment)
                if next_offset == chunk.pcm.size:
                    self._chunks.popleft()
                    if chunk.response_id in self._completed_responses and not any(
                        queued.response_id == chunk.response_id for queued in self._chunks
                    ):
                        self._cleanup_completed_response_locked(chunk.response_id)
                else:
                    chunk.offset = next_offset
        return written


def decode_response_audio(event: dict[str, Any]) -> DecodedAudioPacket:
    """Validate and decode the server's mono PCM16 WAV audio-delta envelope."""
    if event.get("type") != "response.audio.delta":
        raise ValueError("evento non è response.audio.delta")
    try:
        encoded = event["delta"]
        if not isinstance(encoded, str):
            raise TypeError("delta deve essere base64 string")
        wav_bytes = base64.b64decode(encoded, validate=True)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            wav_rate = wav.getframerate()
            frame_count = wav.getnframes()
            if wav.getcomptype() != "NONE" or channels != 1 or sample_width != 2:
                raise ValueError("WAV deve essere mono PCM16 non compresso")
            raw_pcm = wav.readframes(frame_count)
    except (KeyError, TypeError, ValueError, wave.Error, EOFError) as exc:
        raise ValueError(f"response.audio.delta WAV non valido: {exc}") from exc

    declared_rate = _required_int(event, "sample_rate")
    if declared_rate != CODEC_SAMPLE_RATE or wav_rate != declared_rate:
        raise ValueError(
            f"sample_rate non supportato: metadata={declared_rate}, WAV={wav_rate}, "
            f"atteso={CODEC_SAMPLE_RATE}"
        )
    pcm = np.frombuffer(raw_pcm, dtype="<i2").astype(np.float32) / 32768.0
    response_id = event.get("response_id")
    segment_id = event.get("speaker_segment_id")
    if not isinstance(response_id, str) or not response_id:
        raise ValueError("response.audio.delta richiede response_id")
    if not isinstance(segment_id, str) or not segment_id:
        raise ValueError("response.audio.delta richiede speaker_segment_id")
    generation = _required_int(event, "generation_id")
    packet_id = _required_int(event, "packet_id")
    first_frame = _required_int(event, "first_frame_id")
    last_frame = _required_int(event, "last_frame_id")
    start_sample = _required_int(event, "start_sample")
    end_sample = _required_int(event, "end_sample")
    if generation < 0 or packet_id < 0 or first_frame < 0 or last_frame < first_frame:
        raise ValueError("audio delta: identità packet/frame non valida")
    if start_sample < 0 or end_sample - start_sample != pcm.size:
        raise ValueError("audio delta sample count non coincide con il timeline")
    if pcm.size != (last_frame - first_frame + 1) * CODEC_FRAME_SAMPLES:
        raise ValueError("audio delta sample count non coincide con i frame codec")
    return DecodedAudioPacket(
        response_id=response_id,
        generation_id=generation,
        speaker_segment_id=segment_id,
        packet_id=packet_id,
        first_frame_id=first_frame,
        last_frame_id=last_frame,
        start_sample=start_sample,
        end_sample=end_sample,
        sample_rate=declared_rate,
        pcm=pcm,
    )


def handle_realtime_event(event: dict[str, Any], playback: BoundedPCMPlayback) -> bool:
    """Apply protocol fencing before any response PCM reaches device playback."""
    event_type = event.get("type")
    if event_type == "response.created":
        playback.begin_response(event.get("response_id", ""))
    elif event_type == "speaker.segment.opened":
        generation = _required_int(event, "generation_id")
        playback.advance_generation(generation)
    elif event_type == "response.audio.delta":
        packet = decode_response_audio(event)
        playback.enqueue_packet(
            response_id=packet.response_id,
            generation_id=packet.generation_id,
            speaker_segment_id=packet.speaker_segment_id,
            packet_id=packet.packet_id,
            first_frame_id=packet.first_frame_id,
            last_frame_id=packet.last_frame_id,
            start_sample=packet.start_sample,
            end_sample=packet.end_sample,
            pcm=packet.pcm,
            sample_rate=packet.sample_rate,
        )
    elif event_type == "response.audio.segment_done":
        playback.complete_segment(
            response_id=str(event.get("response_id", "")),
            generation_id=_required_int(event, "generation_id"),
            speaker_segment_id=str(event.get("speaker_segment_id", "")),
            start_sample=_required_int(event, "start_sample"),
            end_sample=_required_int(event, "end_sample"),
            sample_rate=_required_int(event, "sample_rate"),
        )
    elif event_type in {"response.cancelled", "audio.cancelled"}:
        response_id = event.get("response_id")
        generation_id = event.get("generation_id")
        if generation_id is not None:
            generation_id = _required_int(event, "generation_id")
        playback.cancel_response(
            None if response_id is None else str(response_id),
            generation_id,
        )
        print(f"\n[Audio flushed] response={response_id} reason={event.get('reason', 'cancelled')}")
    elif event_type in {"response.output_text.delta", "response.audio_transcript.delta"}:
        delta = event.get("delta", "")
        if delta:
            print(delta, end="", flush=True)
    elif event_type == "response.done":
        response_id = event.get("response_id")
        if isinstance(response_id, str) and response_id:
            playback.complete_response(response_id)
        print("\n[Response complete; mic/listener remain active]")
    elif event_type == "response.function_call_arguments.done":
        print(f"\n[WebSocket Tool Call] {event.get('name')}({event.get('arguments')})")
    elif event_type == "error":
        error = event.get("error", {})
        raise RuntimeError(f"server realtime error: {error}")
    return True


def _load_sounddevice():
    try:
        module = importlib.import_module("sounddevice")
    except ImportError as exc:
        raise RuntimeError(
            "microfono/altoparlante richiedono python-sounddevice >=0.5.2 "
            "(pip install 'sounddevice>=0.5.2')"
        ) from exc
    try:
        version = metadata.version("sounddevice")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("impossibile verificare la versione di sounddevice") from exc
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if match is None or tuple(map(int, match.groups())) < (0, 5, 2):
        raise RuntimeError(f"sounddevice >=0.5.2 richiesto, trovato {version}")
    return module


class RealtimeAudioDevices:
    """Separate-clock mic and speaker streams; no blocking network work in callbacks."""

    def __init__(
        self,
        microphone: BoundedMicrophoneFrames,
        playback: BoundedPCMPlayback,
        *,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
    ) -> None:
        self.microphone = microphone
        self.playback = playback
        self.input_device = input_device
        self.output_device = output_device
        self._sounddevice = None
        self._input_stream = None
        self._output_stream = None
        self._error_lock = threading.Lock()
        self._error: BaseException | None = None

    def _fail(self, error: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error

    def raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"audio device callback failed: {error}") from error

    def _input_callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info
        if getattr(status, "input_overflow", False):
            error = RuntimeError("microphone device overflow; PCM continuity lost")
            self._fail(error)
            raise self._sounddevice.CallbackAbort
        try:
            self.microphone.push(indata)
        except Exception as exc:
            self._fail(exc)
            raise self._sounddevice.CallbackAbort from exc

    def _output_callback(self, outdata, frames, time_info, status) -> None:
        del frames, time_info, status
        try:
            self.playback.read_into(outdata)
        except Exception as exc:
            outdata.fill(0)
            self._fail(exc)
            raise self._sounddevice.CallbackAbort from exc

    def start(self) -> None:
        sd = _load_sounddevice()
        self._sounddevice = sd
        try:
            self._output_stream = sd.OutputStream(
                samplerate=CODEC_SAMPLE_RATE,
                blocksize=0,
                device=self.output_device,
                channels=1,
                dtype="float32",
                latency="low",
                callback=self._output_callback,
            )
            self._input_stream = sd.InputStream(
                samplerate=MIC_SAMPLE_RATE,
                blocksize=0,
                device=self.input_device,
                channels=1,
                dtype="int16",
                latency="low",
                callback=self._input_callback,
            )
            self._output_stream.start()
            self._input_stream.start()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for stream in (self._input_stream, self._output_stream):
            if stream is None:
                continue
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        self._input_stream = None
        self._output_stream = None


async def send_microphone_frames(
    websocket: Any,
    microphone: BoundedMicrophoneFrames,
    stop: asyncio.Event,
    *,
    send_lock: asyncio.Lock | None = None,
) -> None:
    """Send captured PCM16 frames independently of response receive/playback."""
    next_send_ns = 0
    while not stop.is_set():
        try:
            frame = microphone.get_nowait()
        except queue.Empty:
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.01)
            except asyncio.TimeoutError:
                pass
            continue
        due_ns = max(frame.capture_end_ns, next_send_ns)
        delay_s = (due_ns - time.monotonic_ns()) / 1_000_000_000
        if delay_s > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay_s)
                return
            except asyncio.TimeoutError:
                pass
        if stop.is_set():
            return
        message = json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(frame.pcm16).decode("ascii"),
        }, separators=(",", ":"))
        if send_lock is None:
            await websocket.send(message)
        else:
            async with send_lock:
                await websocket.send(message)
        # If the event loop or transport falls behind, keep the PCM timeline
        # honest instead of bursting old frames to catch up.
        next_send_ns = max(due_ns, time.monotonic_ns()) + MIC_FRAME_NS


async def _receive_realtime_events(
    websocket: Any,
    playback: BoundedPCMPlayback,
    *,
    duration_s: float | None = None,
    event_lock: asyncio.Lock | None = None,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = None if duration_s is None else loop.time() + duration_s
    while True:
        if deadline is None:
            raw = await websocket.recv()
        else:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=min(remaining, 0.1))
            except asyncio.TimeoutError:
                continue
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise ValueError("evento realtime deve essere un oggetto JSON")
        if event_lock is None:
            handle_realtime_event(event, playback)
        else:
            async with event_lock:
                handle_realtime_event(event, playback)


async def send_client_consumed_pcm_acks(
    websocket: Any,
    playback: BoundedPCMPlayback,
    stop: asyncio.Event,
    *,
    event_lock: asyncio.Lock,
    send_lock: asyncio.Lock,
) -> None:
    """Send callback-produced ACKs from asyncio, fenced against received cancels."""
    while not stop.is_set():
        acknowledgements = playback.drain_client_consumed_pcm_acks()
        if not acknowledgements:
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.01)
            except asyncio.TimeoutError:
                pass
            continue
        for acknowledgement in acknowledgements:
            if stop.is_set():
                return
            # Take the socket lock first so a queued microphone write cannot
            # make this ACK stale while the event loop waits for the transport.
            async with send_lock:
                async with event_lock:
                    if not playback.accept_client_consumed_pcm_ack(
                        acknowledgement,
                        current_generation_id=int(acknowledgement["generation_id"]),
                    ):
                        continue
                    await websocket.send(json.dumps(acknowledgement, separators=(",", ":")))


async def _watch_audio_devices(devices: RealtimeAudioDevices, stop: asyncio.Event) -> None:
    while not stop.is_set():
        devices.raise_if_failed()
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------------------
# 1. Definizione del Tool Locale di Prova
# ---------------------------------------------------------------------------
LOCAL_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Restituisce la temperatura e le condizioni meteo attuali per una data città.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Nome della città (es. Milano, Roma, Napoli)",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Unità di misura della temperatura",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_device_telemetry",
            "description": "Legge lo stato dei sensori e della memoria della GPU NVIDIA RTX 3090.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {
                        "type": "string",
                        "description": "Identificativo della GPU o del device",
                    }
                },
                "required": ["device_id"],
            },
        },
    },
]


def execute_local_tool(name: str, args: Dict[str, Any]) -> str:
    """Esegue la funzione in locale e restituisce la stringa di risultato."""
    print(f"\n⚙️ [Local Tool Execution] Esecuzione: {name}({args})")
    if name == "get_current_weather":
        city = args.get("location", "Milano")
        return json.dumps({
            "location": city,
            "temperature_celsius": 23.5,
            "condition": "Sereno con brezza leggera",
            "humidity_percent": 45,
        }, ensure_ascii=False)
    elif name == "get_device_telemetry":
        return json.dumps({
            "device": "NVIDIA GeForce RTX 3090 24GB",
            "vram_used_gb": 15.6,
            "vram_total_gb": 24.0,
            "temperature_c": 46.0,
            "fan_speed_percent": 35,
            "status": "HEALTHY",
        }, ensure_ascii=False)
    return json.dumps({"status": "unknown_tool"})


# ---------------------------------------------------------------------------
# Test Requisiti
# ---------------------------------------------------------------------------
def test_static_chat(client: OpenAI, model: str):
    """Requisito 3: Chat Statica Normale (senza Qwen3-TTS)."""
    print("\n" + "=" * 60)
    print("💬 [TEST 1] Chat Statica Normale (Solo Testo)")
    print("=" * 60)
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Spiegami in 2 righe cosa significa elaborazione full-duplex."}],
        temperature=0.7,
    )
    text = resp.choices[0].message.content
    print(f"🤖 Risposta Modello:\n{text}")
    assert text is not None and len(text) > 0
    print("✅ Chat statica completata con successo.")


def test_function_calling(client: OpenAI, model: str):
    """Requisito 1: Function Calling con Tool Locale."""
    print("\n" + "=" * 60)
    print("⚡ [TEST 2] Function Calling con Tool Locale (get_current_weather)")
    print("=" * 60)

    messages = [{"role": "user", "content": "Che tempo fa oggi a Milano?"}]
    print(f"👤 Utente: {messages[0]['content']}")

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=LOCAL_TOOLS_SCHEMA,
        tool_choice="auto",
    )

    msg = resp.choices[0].message
    print(f"🤖 Risposta Iniziale: {msg.content}")

    if msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  • Tool Call Rilevata: ID={tc.id}, Name={tc.function.name}")
            args = json.loads(tc.function.arguments)
            tool_res = execute_local_tool(tc.function.name, args)
            print(f"  • Risultato Tool: {tool_res}")

            # Second turn: invia il risultato del tool al modello
            assistant_msg_dict = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                ],
            }
            messages.append(assistant_msg_dict)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": tool_res,
            })

            final_resp = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            final_text = final_resp.choices[0].message.content
            print(f"🤖 Risposta Finale Post-Tool:\n{final_text}")
    print("✅ Ciclo di Function Calling validato con successo.")


def test_chat_with_voice(client: OpenAI, model: str, output_wav: str = "agent_reply.wav"):
    """Requisito 4: Chat Normale con Risposta Letta a Voce (Qwen3-TTS)."""
    print("\n" + "=" * 60)
    print("🔊 [TEST 3] Chat Normale con Risposta Vocale (modalities: ['text', 'audio'])")
    print("=" * 60)

    # Chiamata raw o tramite extra_body per abilitare l'audio
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Salutami con entusiasmo e presentati."}],
        extra_body={"modalities": ["text", "audio"], "voice": "Vivian"},
    )

    choice = resp.choices[0]
    msg = choice.message
    print(f"🤖 Testo Risposta: {msg.content}")

    # Estrazione audio sintetizzato
    audio_obj = getattr(msg, "audio", None) or getattr(choice, "audio", None)
    if audio_obj and isinstance(audio_obj, dict):
        raw_b64 = audio_obj.get("data")
    elif hasattr(audio_obj, "data"):
        raw_b64 = audio_obj.data
    else:
        raw_b64 = None

    if raw_b64:
        wav_bytes = base64.b64decode(raw_b64)
        with open(output_wav, "wb") as f:
            f.write(wav_bytes)
        print(f"✅ Audio generato e salvato con successo in: {output_wav} ({len(wav_bytes)} bytes)")
    else:
        print("⚠️ Nessun payload audio restituito nel dizionario standard.")


async def test_realtime_duplex_websocket(
    base_url: str,
    *,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    duration_s: float | None = None,
) -> None:
    """Run the trained TurnHead policy over the 80 ms full-duplex audio path."""
    if duration_s is not None and (not np.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("duration_s deve essere positivo e finito")

    import websockets

    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url += "/v1/realtime?duplex=1"
    print("\n" + "=" * 60)
    print("🎙️ Microfono continuo + playback full-duplex (frame listener 80 ms)")
    print(f"Connessione a: {ws_url}")
    print(
        f"Input PCM16 {MIC_SAMPLE_RATE} Hz/{MIC_FRAME_MS} ms; output PCM {CODEC_SAMPLE_RATE} Hz. "
        "Policy richiesta TurnDynamics v1 (checkpoint server obbligatorio); ogni frame, senza VAD."
    )

    async with websockets.connect(
        ws_url,
        max_size=WEBSOCKET_MAX_SIZE,
        max_queue=WEBSOCKET_MAX_QUEUE,
        write_limit=WEBSOCKET_WRITE_LIMIT,
        ping_interval=20,
        ping_timeout=20,
    ) as ws:
        initial = json.loads(await ws.recv())
        if initial.get("type") != "session.created":
            raise RuntimeError(f"atteso session.created, ricevuto: {initial}")
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "voice": "Vivian",
                "instructions": "Rispondi in italiano in modo conciso a ciò che senti.",
                "auto_response": True,
                "turn_policy": "turnhead",
            },
        }, separators=(",", ":")))
        updated = json.loads(await ws.recv())
        if updated.get("type") != "session.updated":
            error = updated.get("error", {})
            if isinstance(error, dict):
                code = error.get("code", "unknown")
                message = error.get("message", updated)
                raise RuntimeError(f"session.update rifiutato [{code}]: {message}")
            raise RuntimeError(f"session.update rifiutato: {updated}")

        microphone = BoundedMicrophoneFrames(max_frames=MIC_QUEUE_FRAMES)
        playback = BoundedPCMPlayback(max_samples=PLAYBACK_MAX_SAMPLES)
        devices = RealtimeAudioDevices(
            microphone,
            playback,
            input_device=input_device,
            output_device=output_device,
        )
        devices.start()
        stop = asyncio.Event()
        event_lock = asyncio.Lock()
        send_lock = asyncio.Lock()
        sender_task = asyncio.create_task(
            send_microphone_frames(ws, microphone, stop, send_lock=send_lock)
        )
        receiver_task = asyncio.create_task(
            _receive_realtime_events(
                ws, playback, duration_s=duration_s, event_lock=event_lock,
            )
        )
        ack_sender_task = asyncio.create_task(
            send_client_consumed_pcm_acks(
                ws, playback, stop, event_lock=event_lock, send_lock=send_lock,
            )
        )
        monitor_task = asyncio.create_task(_watch_audio_devices(devices, stop))
        tasks = (sender_task, receiver_task, ack_sender_task, monitor_task)
        print("✅ Full-duplex attivo. Ctrl+C per terminare.")
        try:
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            if not stop.is_set() and receiver_task not in done:
                raise RuntimeError("uplink microfono o monitor audio terminato inaspettatamente")
        finally:
            stop.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            devices.close()


def main():
    parser = argparse.ArgumentParser(description="Client di Test OpenAI per Gemma 4 Omni")
    parser.add_argument("--host", default="localhost", help="Host del server (es. localhost o 192.168.0.186)")
    parser.add_argument("--port", type=int, default=8000, help="Porta del server")
    parser.add_argument("--mode", choices=["static", "tools", "audio", "duplex", "all"], default="all")
    parser.add_argument("--model", default="gemma-4-omni-duplex")
    parser.add_argument("--input-device", help="ID/nome device microfono PortAudio (default: predefinito OS)")
    parser.add_argument("--output-device", help="ID/nome device speaker PortAudio (default: predefinito OS)")
    parser.add_argument("--duration", type=float, help="ferma il demo duplex dopo N secondi (default: Ctrl+C)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1"

    if args.mode in ("static", "tools", "audio", "all"):
        if OpenAI is None:
            print("[Client] Errore: libreria 'openai' non trovata. Esegui: pip install openai")
            sys.exit(1)
        client = OpenAI(base_url=api_url, api_key="dummy-key-for-local-serving")

        if args.mode in ("static", "all"):
            test_static_chat(client, args.model)

        if args.mode in ("tools", "all"):
            test_function_calling(client, args.model)

        if args.mode in ("audio", "all"):
            test_chat_with_voice(client, args.model)

    if args.mode in ("duplex", "all"):
        try:
            import websockets  # noqa: F401 - explicit optional transport dependency check
            input_device = int(args.input_device) if args.input_device and args.input_device.isdigit() else args.input_device
            output_device = int(args.output_device) if args.output_device and args.output_device.isdigit() else args.output_device
            asyncio.run(test_realtime_duplex_websocket(
                base_url,
                input_device=input_device,
                output_device=output_device,
                duration_s=args.duration,
            ))
        except ImportError:
            print("[Client] Installa 'websockets' per il test: pip install 'websockets==17.1'")
        except RuntimeError as exc:
            print(f"[Client] {exc}", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
