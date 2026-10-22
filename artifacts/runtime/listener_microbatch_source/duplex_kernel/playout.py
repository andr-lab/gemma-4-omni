"""Speaker-side playout buffering for frozen-Qwen future conditioning.

The buffer belongs exclusively to the speaker path.  It never blocks or
re-times the Gemma listener/scheduler clock.  A frame is released only after
its conditioning horizon has advanced by the configured number of 80 ms
frames; missing or duplicate frames are rejected rather than silently filled.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import numpy as np

from duplex_kernel.config import AUDIO_FRAME_MS, AUDIO_FRAME_SAMPLES, CODEC_SAMPLE_RATE


@dataclass(frozen=True)
class PlayoutFrame:
    frame_id: int
    pcm: np.ndarray
    generation_id: int = 0
    speaker_segment_id: str = "legacy"

    def __post_init__(self) -> None:
        if self.generation_id is None or isinstance(self.generation_id, bool):
            raise ValueError("PlayoutFrame generation_id obbligatorio")
        if not str(self.speaker_segment_id):
            raise ValueError("PlayoutFrame speaker_segment_id obbligatorio")
        if int(self.frame_id) < 0:
            raise ValueError("PlayoutFrame frame_id non può essere negativo")
        array = np.asarray(self.pcm, dtype=np.float32).reshape(-1)
        if not array.size or not np.isfinite(array).all():
            raise ValueError("playout PCM vuoto o non finito")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "pcm", np.array(array, dtype=np.float32, copy=True))


class SpeakerPlayoutBuffer:
    """Ordered, lossless speaker buffer with an explicit frame delay.

    ``conditioning_frame_id`` is the newest V9 frame available to Qwen.  With
    delay ``D``, frame ``t`` becomes releasable at conditioning frame ``t+D``.
    The implementation uses frame IDs, not sleeps, so tests and the realtime
    scheduler remain deterministic and independent.
    """

    def __init__(
        self,
        *,
        delay_frames: int,
        sample_rate: int = CODEC_SAMPLE_RATE,
        frame_duration_ms: int = AUDIO_FRAME_MS,
        max_frames: int = 256,
    ) -> None:
        if delay_frames < 0 or sample_rate <= 0 or frame_duration_ms <= 0 or max_frames <= 0:
            raise ValueError("configurazione playout non valida")
        frame_samples_float = sample_rate * frame_duration_ms
        if frame_samples_float % 1000:
            raise ValueError("frame_duration_ms deve produrre sample interi")
        self.delay_frames = int(delay_frames)
        self.sample_rate = int(sample_rate)
        self.frame_duration_ms = int(frame_duration_ms)
        self.frame_samples = int(frame_samples_float // 1000)
        self.max_frames = int(max_frames)
        self._lock = threading.RLock()
        self._pending: dict[int, PlayoutFrame] = {}
        self._next_input_id = 0
        self._next_release_id = 0
        self._active_generation_id: int | None = None
        self._active_speaker_segment_id: str | None = None
        self._pushed = 0
        self._released = 0

    @property
    def buffered_frames(self) -> int:
        with self._lock:
            return len(self._pending)

    def begin_segment(self, *, generation_id: int, speaker_segment_id: str) -> None:
        """Start a fresh segment; frame IDs are relative to this segment."""
        with self._lock:
            if generation_id is None or isinstance(generation_id, bool):
                raise ValueError("generation_id è obbligatorio")
            segment = str(speaker_segment_id)
            if not segment:
                raise ValueError("speaker_segment_id è obbligatorio")
            if self._pending:
                raise RuntimeError("non è possibile cambiare segmento con PCM pending")
            self._next_input_id = 0
            self._next_release_id = 0
            self._active_generation_id = int(generation_id)
            self._active_speaker_segment_id = segment

    def _resolve_identity(
        self,
        generation_id: int | None,
        speaker_segment_id: str | None,
    ) -> tuple[int, str]:
        if self._active_generation_id is None:
            # Compatibility mode for the pre-P7 tests/callers. Production
            # callers always establish identity with begin_segment().
            generation = 0 if generation_id is None else int(generation_id)
            segment = "legacy" if speaker_segment_id is None else str(speaker_segment_id)
        else:
            generation = self._active_generation_id if generation_id is None else int(generation_id)
            segment = self._active_speaker_segment_id if speaker_segment_id is None else str(speaker_segment_id)
            if generation != self._active_generation_id or segment != self._active_speaker_segment_id:
                raise ValueError("frame appartiene a un altro generation/segment")
        if not segment:
            raise ValueError("speaker_segment_id è obbligatorio")
        return generation, segment

    def _push_unlocked(
        self,
        pcm: np.ndarray,
        *,
        frame_id: int,
        generation_id: int | None = None,
        speaker_segment_id: str | None = None,
    ) -> PlayoutFrame:
        frame_id = int(frame_id)
        if frame_id != self._next_input_id:
            raise ValueError(f"frame playout atteso {self._next_input_id}, ricevuto {frame_id}")
        generation, segment = self._resolve_identity(generation_id, speaker_segment_id)
        array = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if array.size != self.frame_samples:
            raise ValueError(f"frame playout deve contenere {self.frame_samples} sample")
        if not np.isfinite(array).all():
            raise ValueError("frame playout non finito")
        if len(self._pending) >= self.max_frames:
            raise BufferError("speaker playout buffer pieno")
        frame = PlayoutFrame(
            frame_id=frame_id,
            pcm=array,
            generation_id=generation,
            speaker_segment_id=segment,
        )
        self._pending[frame_id] = frame
        self._next_input_id += 1
        self._pushed += 1
        return frame

    def push(
        self,
        pcm: np.ndarray,
        *,
        frame_id: int,
        generation_id: int | None = None,
        speaker_segment_id: str | None = None,
    ) -> PlayoutFrame:
        with self._lock:
            return self._push_unlocked(
                pcm,
                frame_id=frame_id,
                generation_id=generation_id,
                speaker_segment_id=speaker_segment_id,
            )

    def push_chunk(
        self,
        pcm: np.ndarray,
        *,
        first_frame_id: int | None = None,
        generation_id: int | None = None,
        speaker_segment_id: str | None = None,
    ) -> list[PlayoutFrame]:
        """Split one generated PCM chunk into exact 80 ms frames."""
        with self._lock:
            array = np.asarray(pcm, dtype=np.float32).reshape(-1)
            if first_frame_id is not None and int(first_frame_id) != self._next_input_id:
                raise ValueError("first_frame_id non contiguo rispetto al buffer")
            if not array.size or array.size % self.frame_samples:
                raise ValueError("chunk playout non allineato a frame")
            return [
                self._push_unlocked(
                    array[offset:offset + self.frame_samples],
                    frame_id=self._next_input_id,
                    generation_id=generation_id,
                    speaker_segment_id=speaker_segment_id,
                )
                for offset in range(0, len(array), self.frame_samples)
            ]

    def release_ready(self, *, conditioning_frame_id: int) -> list[PlayoutFrame]:
        """Release contiguous frames whose future-conditioning horizon is ready."""
        with self._lock:
            limit = int(conditioning_frame_id) - self.delay_frames
            released: list[PlayoutFrame] = []
            while self._next_release_id <= limit:
                frame = self._pending.pop(self._next_release_id, None)
                if frame is None:
                    # A gap is held, never substituted by a later frame.
                    break
                released.append(frame)
                self._next_release_id += 1
                self._released += 1
            return released

    def _flush_pending(self) -> list[PlayoutFrame]:
        released: list[PlayoutFrame] = []
        while self._next_release_id < self._next_input_id:
            frame = self._pending.pop(self._next_release_id, None)
            if frame is None:
                raise RuntimeError(f"gap nel buffer playout al frame {self._next_release_id}")
            released.append(frame)
            self._next_release_id += 1
            self._released += 1
        return released

    def flush(self) -> list[PlayoutFrame]:
        """Legacy clean-end helper; prefer flush_segment in P7 production."""
        with self._lock:
            return self._flush_pending()

    def flush_segment(self, *, generation_id: int, speaker_segment_id: str) -> list[PlayoutFrame]:
        """Release accepted PCM only when the requested segment is active."""
        with self._lock:
            if self._active_generation_id != int(generation_id) or self._active_speaker_segment_id != str(speaker_segment_id):
                return []
            released = self._flush_pending()
            self._active_generation_id = None
            self._active_speaker_segment_id = None
            self._next_input_id = 0
            self._next_release_id = 0
            return released

    def cancel_segment(self, *, generation_id: int, speaker_segment_id: str) -> int:
        """Discard only pending PCM belonging to the named active segment."""
        with self._lock:
            if self._active_generation_id != int(generation_id) or self._active_speaker_segment_id != str(speaker_segment_id):
                return 0
            discarded = len(self._pending)
            self._pending.clear()
            self._active_generation_id = None
            self._active_speaker_segment_id = None
            self._next_input_id = 0
            self._next_release_id = 0
            return discarded

    def cancel(self) -> None:
        """Teardown-only reset of the entire playout instance."""
        with self._lock:
            self._pending.clear()
            self._next_input_id = 0
            self._next_release_id = 0
            self._active_generation_id = None
            self._active_speaker_segment_id = None

    def stats(self) -> dict[str, int | str | None]:
        with self._lock:
            return {
                "delay_frames": self.delay_frames,
                "frames_pushed": self._pushed,
                "frames_released": self._released,
                "frames_buffered": len(self._pending),
                "next_input_frame_id": self._next_input_id,
                "next_release_frame_id": self._next_release_id,
                "generation_id": self._active_generation_id,
                "speaker_segment_id": self._active_speaker_segment_id,
            }


__all__ = ["PlayoutFrame", "SpeakerPlayoutBuffer"]
