"""80 ms duplex clock and lossless PCM ingress transport.

The scheduler owns timing only.  It deliberately knows nothing about TurnHead,
Gemma, V9, Qwen, or the WebSocket protocol; those components consume complete
``AudioFrame`` values through the injected callbacks.
"""
from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_FRAME_DURATION_MS = 80
NANOSECONDS = 1_000_000_000


@dataclass(frozen=True)
class AudioFrame:
    """One causal microphone frame, exactly 80 ms at 16 kHz by default."""

    frame_id: int
    pcm: np.ndarray
    capture_start_ns: int
    capture_end_ns: int
    enqueue_ns: int


@dataclass
class DuplexTick:
    """Timing envelope for one scheduler tick.

    ``policy_ns`` is the timestamp at which policy processing completed.  The
    other phase fields are start/end timestamps and remain ``None`` when the
    corresponding callback was not configured.
    """

    frame_id: int
    capture_start_ns: int
    capture_end_ns: int
    enqueue_ns: int | None
    deadline_ns: int | None = None
    observed_ns: int | None = None
    listener_start_ns: int | None = None
    listener_end_ns: int | None = None
    policy_ns: int | None = None
    speaker_start_ns: int | None = None
    speaker_end_ns: int | None = None


class FrameClock:
    """Clock interface used by ``DuplexScheduler``."""

    period_ns: int
    start_ns: int

    def deadline_ns(self, tick_index: int) -> int:
        raise NotImplementedError

    def now_ns(self) -> int:
        raise NotImplementedError

    def wait_until(self, deadline_ns: int) -> None:
        raise NotImplementedError


class VirtualFrameClock(FrameClock):
    """Deterministic clock for exact tick-count and causality tests."""

    def __init__(self, *, start_ns: int = 0, period_ms: int = DEFAULT_FRAME_DURATION_MS) -> None:
        if period_ms <= 0:
            raise ValueError("period_ms deve essere positivo")
        self.start_ns = int(start_ns)
        self.period_ns = int(period_ms * 1_000_000)
        self._now_ns = self.start_ns

    def deadline_ns(self, tick_index: int) -> int:
        if tick_index < 0:
            raise ValueError("tick_index non può essere negativo")
        return self.start_ns + int(tick_index) * self.period_ns

    def now_ns(self) -> int:
        return self._now_ns

    def wait_until(self, deadline_ns: int) -> None:
        if deadline_ns < self._now_ns:
            raise ValueError("deadline virtuale retrocedente")
        self._now_ns = int(deadline_ns)


class MonotonicFrameClock(FrameClock):
    """Real-time monotonic clock with absolute deadlines (no cumulative drift)."""

    def __init__(self, *, start_ns: int | None = None, period_ms: int = DEFAULT_FRAME_DURATION_MS) -> None:
        if period_ms <= 0:
            raise ValueError("period_ms deve essere positivo")
        self.start_ns = time.monotonic_ns() if start_ns is None else int(start_ns)
        self.period_ns = int(period_ms * 1_000_000)

    def deadline_ns(self, tick_index: int) -> int:
        if tick_index < 0:
            raise ValueError("tick_index non può essere negativo")
        return self.start_ns + int(tick_index) * self.period_ns

    def now_ns(self) -> int:
        return time.monotonic_ns()

    def wait_until(self, deadline_ns: int) -> None:
        while True:
            remaining_ns = int(deadline_ns) - time.monotonic_ns()
            if remaining_ns <= 0:
                return
            # A short sleep avoids a busy spin while retaining sub-ms accuracy
            # close to the deadline on ordinary Linux runners.
            time.sleep(min(remaining_ns / NANOSECONDS, 0.002))


class AudioIngressQueue:
    """Thread-safe, numbered, lossless transport for microphone PCM.

    ``append`` accepts arbitrary PCM chunk sizes and emits complete frames.
    ``put`` is the strict API for callers that already own 1280-sample frames.
    A bounded queue raises ``queue.Full`` rather than silently dropping data.
    """

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
        max_frames: int = 0,
    ) -> None:
        if sample_rate <= 0 or frame_duration_ms <= 0:
            raise ValueError("sample_rate e frame_duration_ms devono essere positivi")
        frame_samples = sample_rate * frame_duration_ms
        if frame_samples % 1000:
            raise ValueError("frame_duration_ms deve produrre un numero intero di sample")
        if max_frames < 0:
            raise ValueError("max_frames non può essere negativo")
        self.sample_rate = int(sample_rate)
        self.frame_duration_ms = int(frame_duration_ms)
        self.frame_samples = frame_samples // 1000
        self.frame_duration_ns = (self.frame_samples * NANOSECONDS) // self.sample_rate
        self._queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=int(max_frames))
        self._lock = threading.RLock()
        self._next_frame_id = 0
        self._last_consumed_frame_id: int | None = None
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start_ns: int | None = None
        self._closed = False
        self._stats = {
            "input_samples_received": 0,
            "input_frames_enqueued": 0,
            "input_frames_consumed": 0,
            "dropped_input_frames": 0,
            "duplicate_input_frames": 0,
            "out_of_order_input_frames": 0,
        }

    @property
    def pending_samples(self) -> int:
        with self._lock:
            return int(self._buffer.size)

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    def _validate_pcm(self, pcm: np.ndarray) -> np.ndarray:
        array = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if not array.size:
            raise ValueError("PCM vuoto")
        if not np.isfinite(array).all():
            raise ValueError("PCM deve contenere solo sample finite")
        return np.array(array, dtype=np.float32, copy=True)

    def _sample_time_ns(self, samples: int) -> int:
        return (int(samples) * NANOSECONDS) // self.sample_rate

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("AudioIngressQueue chiusa")

    def _ensure_capacity(self, frame_count: int) -> None:
        if self._queue.maxsize and self._queue.qsize() + frame_count > self._queue.maxsize:
            self._stats["dropped_input_frames"] += int(frame_count)
            raise queue.Full("AudioIngressQueue piena: nessun frame è stato scartato silenziosamente")

    def _enqueue_locked(
        self,
        pcm: np.ndarray,
        *,
        capture_start_ns: int,
        frame_id: int | None = None,
        enqueue_ns: int | None = None,
    ) -> AudioFrame:
        self._ensure_capacity(1)
        if frame_id is None:
            frame_id = self._next_frame_id
        frame_id = int(frame_id)
        self._next_frame_id = max(self._next_frame_id, frame_id + 1)
        start = int(capture_start_ns)
        frame = AudioFrame(
            frame_id=frame_id,
            pcm=np.array(pcm, dtype=np.float32, copy=True),
            capture_start_ns=start,
            capture_end_ns=start + self.frame_duration_ns,
            enqueue_ns=time.monotonic_ns() if enqueue_ns is None else int(enqueue_ns),
        )
        self._queue.put_nowait(frame)
        self._stats["input_frames_enqueued"] += 1
        return frame

    def put(
        self,
        pcm: np.ndarray,
        *,
        capture_start_ns: int | None = None,
        frame_id: int | None = None,
        enqueue_ns: int | None = None,
    ) -> AudioFrame:
        """Enqueue one complete frame and return its immutable envelope."""
        array = self._validate_pcm(pcm)
        if array.size != self.frame_samples:
            raise ValueError(f"un frame deve contenere esattamente {self.frame_samples} sample")
        with self._lock:
            self._ensure_open()
            if self._buffer.size:
                raise ValueError("non è possibile usare put mentre esiste un PCM parziale")
            start = time.monotonic_ns() if capture_start_ns is None else int(capture_start_ns)
            frame = self._enqueue_locked(array, capture_start_ns=start, frame_id=frame_id, enqueue_ns=enqueue_ns)
            self._stats["input_samples_received"] += self.frame_samples
            return frame

    def append(self, pcm: np.ndarray, *, capture_start_ns: int | None = None) -> list[AudioFrame]:
        """Append arbitrary PCM and enqueue every newly completed 80 ms frame."""
        array = self._validate_pcm(pcm)
        with self._lock:
            self._ensure_open()
            if self._buffer.size:
                expected = self._buffer_start_ns + self._sample_time_ns(self._buffer.size)
                start = expected if capture_start_ns is None else int(capture_start_ns)
                if start != expected:
                    raise ValueError("capture_start_ns non contiguo rispetto al PCM parziale")
            else:
                start = time.monotonic_ns() if capture_start_ns is None else int(capture_start_ns)
                self._buffer_start_ns = start
            combined = np.concatenate((self._buffer, array))
            full_count = combined.size // self.frame_samples
            self._ensure_capacity(int(full_count))
            self._stats["input_samples_received"] += int(array.size)
            frames: list[AudioFrame] = []
            for index in range(int(full_count)):
                offset = index * self.frame_samples
                frame_start = start + self._sample_time_ns(offset)
                frames.append(self._enqueue_locked(
                    combined[offset:offset + self.frame_samples],
                    capture_start_ns=frame_start,
                ))
            remainder_offset = int(full_count) * self.frame_samples
            self._buffer = np.array(combined[remainder_offset:], dtype=np.float32, copy=True)
            self._buffer_start_ns = (
                start + self._sample_time_ns(remainder_offset) if self._buffer.size else None
            )
            return frames

    def get(self, timeout: float | None = None) -> AudioFrame:
        frame = self._queue.get(timeout=timeout)
        with self._lock:
            previous = self._last_consumed_frame_id
            if previous is not None:
                if frame.frame_id == previous:
                    self._stats["duplicate_input_frames"] += 1
                elif frame.frame_id != previous + 1:
                    self._stats["out_of_order_input_frames"] += 1
            self._last_consumed_frame_id = frame.frame_id
            self._stats["input_frames_consumed"] += 1
        return frame

    def get_nowait(self) -> AudioFrame:
        return self.get(timeout=0.0)

    def stats(self) -> dict[str, int]:
        with self._lock:
            result = dict(self._stats)
            result["input_frames_received"] = result["input_frames_enqueued"]
            return result

    def clear(self) -> None:
        """Drop queued data only during explicit session teardown/reset."""
        with self._lock:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._buffer = np.empty(0, dtype=np.float32)
            self._buffer_start_ns = None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.clear()


Callback = Callable[[AudioFrame | None, DuplexTick], None]


class DuplexScheduler:
    """Run one causal tick per 80 ms and dispatch independent phase callbacks."""

    def __init__(
        self,
        ingress: AudioIngressQueue,
        *,
        clock: FrameClock | None = None,
        on_listener: Callback | None = None,
        on_policy: Callback | None = None,
        on_speaker: Callback | None = None,
        deadline_jitter_budget_ms: float = 20.0,
    ) -> None:
        if deadline_jitter_budget_ms < 0:
            raise ValueError("deadline_jitter_budget_ms non può essere negativo")
        self.ingress = ingress
        self.clock = clock or MonotonicFrameClock()
        self.on_listener = on_listener
        self.on_policy = on_policy
        self.on_speaker = on_speaker
        self.deadline_jitter_budget_ns = int(deadline_jitter_budget_ms * 1_000_000)
        self._next_tick_index = 0
        self._ticks = 0
        self._missing_ticks = 0
        self._duplicate_ticks = 0
        self._out_of_order_ticks = 0
        self._deadline_misses = 0
        self._jitters_ns: list[int] = []
        self._last_tick_frame_id: int | None = None
        self._input_frames_during_speaking = 0
        self._history: list[DuplexTick] = []

    def _phase_now(self) -> int:
        return self.clock.now_ns()

    def _dispatch(self, callback: Callback | None, frame: AudioFrame | None, tick: DuplexTick) -> int | None:
        if callback is None:
            return None
        started = self._phase_now()
        callback(frame, tick)
        return started

    def tick(self) -> DuplexTick:
        """Wait for and execute the next tick."""
        index = self._next_tick_index
        deadline = self.clock.deadline_ns(index)
        self.clock.wait_until(deadline)
        actual = self._phase_now()
        lateness = max(0, actual - deadline)
        self._jitters_ns.append(lateness)
        if lateness > self.deadline_jitter_budget_ns:
            self._deadline_misses += 1

        try:
            frame = self.ingress.get_nowait()
        except queue.Empty:
            frame = None
            self._missing_ticks += 1
        frame_id = frame.frame_id if frame is not None else index
        if self._last_tick_frame_id is not None:
            if frame_id == self._last_tick_frame_id:
                self._duplicate_ticks += 1
            elif frame_id != self._last_tick_frame_id + 1:
                self._out_of_order_ticks += 1
        self._last_tick_frame_id = frame_id
        capture_start = frame.capture_start_ns if frame is not None else deadline
        capture_end = frame.capture_end_ns if frame is not None else capture_start + self.clock.period_ns
        tick = DuplexTick(
            frame_id=frame_id,
            capture_start_ns=capture_start,
            capture_end_ns=capture_end,
            enqueue_ns=frame.enqueue_ns if frame is not None else None,
            deadline_ns=deadline,
            observed_ns=actual,
        )

        if self.on_listener is not None:
            tick.listener_start_ns = self._phase_now()
            self.on_listener(frame, tick)
            tick.listener_end_ns = self._phase_now()
        if self.on_policy is not None:
            self.on_policy(frame, tick)
            tick.policy_ns = self._phase_now()
        if self.on_speaker is not None:
            if frame is not None:
                self._input_frames_during_speaking += 1
            tick.speaker_start_ns = self._phase_now()
            self.on_speaker(frame, tick)
            tick.speaker_end_ns = self._phase_now()

        self._next_tick_index += 1
        self._ticks += 1
        self._history.append(tick)
        return tick

    def run_ticks(self, count: int) -> list[DuplexTick]:
        if count < 0:
            raise ValueError("count non può essere negativo")
        return [self.tick() for _ in range(int(count))]

    def run_for(self, duration_s: float) -> list[DuplexTick]:
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("duration_s deve essere finita e non negativa")
        exact_ticks = duration_s * NANOSECONDS / self.clock.period_ns
        count = round(exact_ticks)
        if not math.isclose(exact_ticks, count, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("duration_s deve essere un multiplo esatto del periodo del frame")
        return self.run_ticks(count)

    def stats(self) -> dict[str, int | float | None]:
        ingress = self.ingress.stats()
        if self._jitters_ns:
            jitter_ms = np.asarray(self._jitters_ns, dtype=np.float64) / 1_000_000.0
            jitter_p99 = float(np.percentile(jitter_ms, 99))
        else:
            jitter_p99 = 0.0
        return {
            "ticks": self._ticks,
            "missing_ticks": self._missing_ticks,
            "input_frames_received": int(ingress["input_frames_enqueued"]),
            "samples_received": int(ingress["input_samples_received"]),
            "duplicate_ticks": self._duplicate_ticks,
            "out_of_order_ticks": self._out_of_order_ticks,
            "deadline_misses": self._deadline_misses,
            "deadline_miss_rate": self._deadline_misses / self._ticks if self._ticks else 0.0,
            "tick_jitter_p99_ms": round(jitter_p99, 3),
            "input_frames_during_speaking": self._input_frames_during_speaking,
            **ingress,
        }

    @property
    def history(self) -> tuple[DuplexTick, ...]:
        return tuple(self._history)


__all__ = [
    "AudioFrame",
    "AudioIngressQueue",
    "DuplexScheduler",
    "DuplexTick",
    "FrameClock",
    "MonotonicFrameClock",
    "VirtualFrameClock",
]
