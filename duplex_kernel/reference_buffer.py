from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from duplex_kernel.scheduler import DEFAULT_SAMPLE_RATE

logger = logging.getLogger("duplex_kernel.reference_buffer")

DEFAULT_FRAME_DURATION_MS = 80
DEFAULT_FRAME_SAMPLES = DEFAULT_SAMPLE_RATE * DEFAULT_FRAME_DURATION_MS // 1000
DEFAULT_REFERENCE_DELAY_FRAMES = 1
NANOSECONDS = 1_000_000_000


@dataclass(frozen=True)
class AssistantReferenceFrame:
    """One already-emitted assistant frame normalized to the listener rate."""

    frame_id: int
    pcm: np.ndarray
    emitted_at_ns: int
    sample_rate: int = DEFAULT_SAMPLE_RATE

    def __post_init__(self) -> None:
        array = np.asarray(self.pcm, dtype=np.float32).reshape(-1)
        if not array.size:
            raise ValueError("assistant reference PCM vuoto")
        if not np.isfinite(array).all():
            raise ValueError("assistant reference PCM deve contenere solo sample finite")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "pcm", np.array(array, dtype=np.float32, copy=True))
        object.__setattr__(self, "emitted_at_ns", int(self.emitted_at_ns))
        object.__setattr__(self, "sample_rate", int(self.sample_rate))


class AssistantReferenceBuffer:
    """Thread-safe timestamped buffer with a strict one-frame visibility delay.

    ``append`` is called only after an audio frame was sent to the client. A
    listener at frame ``t`` can retrieve exactly frame ``t-delay_frames``;
    future entries and gaps are never substituted with another frame.
    """

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
        delay_frames: int = DEFAULT_REFERENCE_DELAY_FRAMES,
        max_frames: int = 256,
    ) -> None:
        if sample_rate <= 0 or frame_duration_ms <= 0:
            raise ValueError("sample_rate e frame_duration_ms devono essere positivi")
        if delay_frames < 0:
            raise ValueError("delay_frames non può essere negativo")
        if max_frames <= 0:
            raise ValueError("max_frames deve essere positivo")
        frame_samples = sample_rate * frame_duration_ms
        if frame_samples % 1000:
            raise ValueError("frame_duration_ms deve produrre un numero intero di sample")
        self.sample_rate = int(sample_rate)
        self.frame_duration_ms = int(frame_duration_ms)
        self.frame_samples = frame_samples // 1000
        self.frame_duration_ns = (self.frame_samples * NANOSECONDS) // self.sample_rate
        self.delay_frames = int(delay_frames)
        self.max_frames = int(max_frames)
        self._frames: dict[int, AssistantReferenceFrame] = {}
        self._lock = threading.RLock()
        self._appended = 0
        self._evicted = 0

    @staticmethod
    def _resample(pcm: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return np.array(pcm, dtype=np.float32, copy=True)
        if source_rate <= 0:
            raise ValueError("sample_rate sorgente non valido")
        target_length = int(round(len(pcm) * target_rate / source_rate))
        if target_length <= 0:
            raise ValueError("PCM troppo breve dopo il resampling")
        if len(pcm) == 1:
            return np.full(target_length, float(pcm[0]), dtype=np.float32)
        source_positions = np.linspace(0.0, float(len(pcm) - 1), num=len(pcm), dtype=np.float64)
        target_positions = np.linspace(0.0, float(len(pcm) - 1), num=target_length, dtype=np.float64)
        return np.interp(target_positions, source_positions, pcm).astype(np.float32)

    def _normalize_frame(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        array = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if not array.size:
            raise ValueError("assistant reference PCM vuoto")
        if not np.isfinite(array).all():
            raise ValueError("assistant reference PCM deve contenere solo sample finite")
        expected_source = int(round(self.frame_samples * sample_rate / self.sample_rate))
        if len(array) != expected_source:
            raise ValueError(
                f"assistant reference deve durare esattamente {self.frame_duration_ms} ms "
                f"({expected_source} sample a {sample_rate} Hz)"
            )
        normalized = self._resample(array, sample_rate, self.sample_rate)
        if normalized.size != self.frame_samples:
            raise RuntimeError("resampling assistant reference non allineato al frame")
        return normalized

    def append(
        self,
        pcm: np.ndarray,
        *,
        frame_id: int,
        emitted_at_ns: int | None = None,
        sample_rate: int | None = None,
    ) -> AssistantReferenceFrame:
        """Store one *actually emitted* frame, converting it to listener PCM."""
        frame_id = int(frame_id)
        if frame_id < 0:
            raise ValueError("frame_id non può essere negativo")
        source_rate = self.sample_rate if sample_rate is None else int(sample_rate)
        normalized = self._normalize_frame(pcm, source_rate)
        frame = AssistantReferenceFrame(
            frame_id=frame_id,
            pcm=normalized,
            emitted_at_ns=time.monotonic_ns() if emitted_at_ns is None else int(emitted_at_ns),
            sample_rate=self.sample_rate,
        )
        with self._lock:
            if frame_id in self._frames:
                raise ValueError(f"assistant reference frame duplicato: {frame_id}")
            self._frames[frame_id] = frame
            self._appended += 1
            while len(self._frames) > self.max_frames:
                oldest = min(self._frames)
                del self._frames[oldest]
                self._evicted += 1
        return frame

    def append_chunk(
        self,
        pcm: np.ndarray,
        *,
        first_frame_id: int,
        emitted_at_ns: int | None = None,
        sample_rate: int,
    ) -> list[AssistantReferenceFrame]:
        """Split an emitted 24 kHz (or other-rate) aligned chunk into frames."""
        source_rate = int(sample_rate)
        array = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if not array.size:
            raise ValueError("assistant reference chunk vuoto")
        if not np.isfinite(array).all():
            raise ValueError("assistant reference chunk deve contenere solo sample finite")
        normalized = self._resample(array, source_rate, self.sample_rate)
        if len(normalized) % self.frame_samples:
            raise ValueError("assistant reference chunk non contiene un numero intero di frame")
        start_ns = time.monotonic_ns() if emitted_at_ns is None else int(emitted_at_ns)
        frames = []
        for offset in range(0, len(normalized), self.frame_samples):
            frames.append(self.append(
                normalized[offset:offset + self.frame_samples],
                frame_id=int(first_frame_id) + offset // self.frame_samples,
                emitted_at_ns=start_ns + (offset // self.frame_samples) * self.frame_duration_ns,
                sample_rate=self.sample_rate,
            ))
        return frames

    def visible_for(self, *, listener_frame_id: int) -> AssistantReferenceFrame | None:
        """Return only the exact frame made visible by the configured delay."""
        listener_frame_id = int(listener_frame_id)
        target = listener_frame_id - self.delay_frames
        if target < 0:
            return None
        with self._lock:
            return self._frames.get(target)

    def visible_frames_for(self, *, listener_frame_id: int) -> tuple[AssistantReferenceFrame, ...]:
        """Return all buffered frames whose source id is no later than the target."""
        target = int(listener_frame_id) - self.delay_frames
        if target < 0:
            return ()
        with self._lock:
            return tuple(self._frames[key] for key in sorted(self._frames) if key <= target)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "frames_appended": self._appended,
                "frames_buffered": len(self._frames),
                "frames_evicted": self._evicted,
                "delay_frames": self.delay_frames,
            }

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()


__all__ = [
    "DEFAULT_FRAME_DURATION_MS",
    "DEFAULT_FRAME_SAMPLES",
    "DEFAULT_REFERENCE_DELAY_FRAMES",
    "AssistantReferenceFrame",
    "AssistantReferenceBuffer",
]
