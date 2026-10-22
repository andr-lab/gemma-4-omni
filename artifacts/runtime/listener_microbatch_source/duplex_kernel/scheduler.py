"""80 ms duplex clock and lossless PCM ingress transport.

The scheduler owns timing only.  It deliberately knows nothing about TurnHead,
Gemma, V9, Qwen, or the WebSocket protocol; those components consume complete
``AudioFrame`` values through the injected callbacks.
"""
from __future__ import annotations

import hashlib
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


logger = logging.getLogger("duplex_kernel.scheduler")

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_FRAME_DURATION_MS = 80
REALTIME_INGRESS_QUEUE_FRAMES = 3  # 240 ms queued PCM at the 80 ms server frame rate.
DIRECT_V2_INGRESS_QUEUE_FRAMES = 6  # 480 ms bounded direct-audio v2 burst buffer.
NANOSECONDS = 1_000_000_000


@dataclass(frozen=True)
class AudioFrame:
    """One causal microphone frame, exactly 80 ms at 16 kHz by default."""

    frame_id: int
    pcm: np.ndarray
    capture_start_ns: int
    capture_end_ns: int
    enqueue_ns: int
    transport_sha256: str | None = None
    transport_serialization: str = "normalized_float32_little_endian"
    capture_timestamp_source: str = "server_receive_clock"


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
    dispatch_mode: str | None = None
    listener_batch_id: str | None = None
    listener_batch_size: int | None = None
    listener_batch_started_ns: int | None = None
    listener_batch_finished_ns: int | None = None
    microbatch_queue_wait_ms: float | None = None
    microbatch_compute_ms: float | None = None
    microbatch_model_lock_wait_ms: float | None = None
    microbatch_model_lock_hold_ms: float | None = None
    microbatch_capacity_growth_ms: float | None = None
    microbatch_frontend_encode_ms: float | None = None
    microbatch_mel_prepare_ms: float | None = None
    microbatch_frontend_copy_ms: float | None = None
    microbatch_frontend_copy_cuda_ms: float | None = None
    microbatch_audio_tower_lock_wait_ms: float | None = None
    microbatch_audio_tower_lock_hold_ms: float | None = None
    microbatch_audio_tower_workspace_lock_wait_ms: float | None = None
    microbatch_audio_tower_workspace_lock_hold_ms: float | None = None
    microbatch_audio_tower_row_copy_cuda_ms: float | None = None
    microbatch_audio_tower_graph_submit_ms: float | None = None
    microbatch_audio_tower_cuda_ms: float | None = None
    microbatch_audio_tower_batch_cuda_span_ms: float | None = None
    microbatch_audio_tower_execution_mode: str | None = None
    microbatch_native_submit_ms: float | None = None
    microbatch_native_cuda_ms: float | None = None
    microbatch_native_control6_raw_score_ms: float | None = None
    microbatch_decision_view_fork_count: int | None = None
    microbatch_decision_view_replayed_q2_count: int | None = None
    microbatch_decision_view_cache_hit: bool | None = None
    microbatch_decision_view_ms: float | None = None
    microbatch_decision_view_context_mode: str | None = None
    microbatch_native_output_health_ms: float | None = None
    microbatch_native_output_health_checked: bool | None = None
    microbatch_native_output_health_status: str | None = None
    microbatch_native_output_health_nonfinite: bool | None = None
    microbatch_native_output_health_generation: int | None = None
    microbatch_native_output_health_session_generation: int | None = None
    microbatch_native_output_health_state_epoch_id: str | None = None
    microbatch_native_output_health_row_frame_id: int | None = None
    microbatch_native_output_health_row_state_version: int | None = None
    microbatch_native_output_health_row_sequence_position: int | None = None
    microbatch_native_output_health_row_cache_position: int | None = None
    microbatch_native_output_health_native_context_id: int | None = None
    microbatch_semantic_prefix_request_ms: float | None = None
    microbatch_semantic_prefix_build_ms: float | None = None
    microbatch_semantic_prefix_replay_ms: float | None = None
    microbatch_semantic_prefix_import_ms: float | None = None
    microbatch_semantic_prefix_refresh_event: bool | None = None
    microbatch_semantic_prefix_refresh_event_id: str | None = None
    microbatch_semantic_prefix_refresh_revision: int | None = None
    microbatch_semantic_prefix_len: int | None = None
    microbatch_control6_policy_ms: float | None = None
    microbatch_cuda_completion_wait_ms: float | None = None
    microbatch_error: str | None = None


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

    def anchor_epoch(self, start_ns: int) -> None:
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

    def anchor_epoch(self, start_ns: int) -> None:
        self.start_ns = int(start_ns)
        self._now_ns = max(self._now_ns, self.start_ns)


class MonotonicFrameClock(FrameClock):
    """Real-time monotonic clock with absolute deadlines (no cumulative drift)."""

    def __init__(self, *, start_ns: int | None = None, period_ms: int = DEFAULT_FRAME_DURATION_MS) -> None:
        if period_ms <= 0:
            raise ValueError("period_ms deve essere positivo")
        self.start_ns = time.monotonic_ns() if start_ns is None else int(start_ns)
        self.period_ns = int(period_ms * 1_000_000)

    def reset_epoch(self) -> None:
        """Start a fresh real-time cadence from the current monotonic instant."""
        self.start_ns = time.monotonic_ns()

    def anchor_epoch(self, start_ns: int) -> None:
        """Anchor absolute frame deadlines to a host monotonic enqueue stamp."""
        self.start_ns = int(start_ns)

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
        self._frame_condition = threading.Condition(self._lock)
        self._next_frame_id = 0
        self._last_consumed_frame_id: int | None = None
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start_ns: int | None = None
        self._transport_buffer = bytearray()
        self._transport_serialization: str | None = None
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

    @property
    def max_frames(self) -> int:
        """Configured queue capacity; zero means unbounded."""
        return self._queue.maxsize

    def _validate_pcm(self, pcm: np.ndarray) -> np.ndarray:
        array = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if not array.size:
            raise ValueError("PCM vuoto")
        if not np.isfinite(array).all():
            raise ValueError("PCM deve contenere solo sample finite")
        return np.array(array, dtype=np.float32, copy=True)

    @staticmethod
    def _normalize_transport(
        transport_bytes: bytes | bytearray | memoryview | None,
        array: np.ndarray,
    ) -> tuple[bytes, str, int]:
        if transport_bytes is None:
            raw = np.ascontiguousarray(array, dtype="<f4").tobytes()
            return raw, "normalized_float32_little_endian", 4
        raw = bytes(transport_bytes)
        if len(raw) != array.size * 2:
            raise ValueError("transport PCM deve essere PCM16 little-endian allineato ai sample")
        decoded = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if not np.array_equal(decoded, array):
            raise ValueError("transport PCM non corrisponde al PCM passato al listener")
        return raw, "pcm_s16le", 2

    @staticmethod
    def _transport_sha256(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

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
        transport_bytes: bytes | bytearray | memoryview | None = None,
        transport_serialization: str | None = None,
    ) -> AudioFrame:
        self._ensure_capacity(1)
        if frame_id is None:
            frame_id = self._next_frame_id
        frame_id = int(frame_id)
        self._next_frame_id = max(self._next_frame_id, frame_id + 1)
        start = int(capture_start_ns)
        array = np.array(pcm, dtype=np.float32, copy=True)
        # ``AudioFrame`` is a frozen envelope; make its ndarray immutable too,
        # otherwise a downstream consumer could silently change the bytes that
        # the provenance hash describes after enqueue.
        array.setflags(write=False)
        if transport_bytes is None:
            raw_transport = np.ascontiguousarray(array, dtype="<f4").tobytes()
            serialization = "normalized_float32_little_endian"
        else:
            raw_transport = bytes(transport_bytes)
            serialization = str(transport_serialization or "pcm_s16le")
        expected_bytes = array.size * (2 if serialization == "pcm_s16le" else 4)
        if len(raw_transport) != expected_bytes:
            raise ValueError("transport provenance non allineata al frame")
        frame = AudioFrame(
            frame_id=frame_id,
            pcm=array,
            capture_start_ns=start,
            capture_end_ns=start + self.frame_duration_ns,
            enqueue_ns=time.monotonic_ns() if enqueue_ns is None else int(enqueue_ns),
            transport_sha256=self._transport_sha256(raw_transport),
            transport_serialization=serialization,
            capture_timestamp_source="server_receive_clock",
        )
        self._queue.put_nowait(frame)
        self._stats["input_frames_enqueued"] += 1
        self._frame_condition.notify()
        return frame

    def put(
        self,
        pcm: np.ndarray,
        *,
        capture_start_ns: int | None = None,
        frame_id: int | None = None,
        enqueue_ns: int | None = None,
        transport_bytes: bytes | bytearray | memoryview | None = None,
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
            raw_transport, serialization, _ = self._normalize_transport(transport_bytes, array)
            frame = self._enqueue_locked(
                array,
                capture_start_ns=start,
                frame_id=frame_id,
                enqueue_ns=enqueue_ns,
                transport_bytes=raw_transport,
                transport_serialization=serialization,
            )
            self._stats["input_samples_received"] += self.frame_samples
            return frame

    def append(
        self,
        pcm: np.ndarray,
        *,
        capture_start_ns: int | None = None,
        transport_bytes: bytes | bytearray | memoryview | None = None,
    ) -> list[AudioFrame]:
        """Append arbitrary PCM and enqueue every newly completed 80 ms frame."""
        array = self._validate_pcm(pcm)
        raw_transport, serialization, bytes_per_sample = self._normalize_transport(transport_bytes, array)
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
            if self._buffer.size and self._transport_serialization != serialization:
                raise ValueError("transport serialization non coerente con il PCM parziale")
            combined = np.concatenate((self._buffer, array))
            combined_transport = bytes(self._transport_buffer) + raw_transport
            full_count = combined.size // self.frame_samples
            self._ensure_capacity(int(full_count))
            self._stats["input_samples_received"] += int(array.size)
            frames: list[AudioFrame] = []
            for index in range(int(full_count)):
                offset = index * self.frame_samples
                frame_start = start + self._sample_time_ns(offset)
                frame_transport = combined_transport[offset * bytes_per_sample:(offset + self.frame_samples) * bytes_per_sample]
                frames.append(self._enqueue_locked(
                    combined[offset:offset + self.frame_samples],
                    capture_start_ns=frame_start,
                    transport_bytes=frame_transport,
                    transport_serialization=serialization,
                ))
            remainder_offset = int(full_count) * self.frame_samples
            self._buffer = np.array(combined[remainder_offset:], dtype=np.float32, copy=True)
            remainder_transport_offset = remainder_offset * bytes_per_sample
            self._transport_buffer = bytearray(combined_transport[remainder_transport_offset:])
            self._transport_serialization = serialization if self._buffer.size else None
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

    def wait_for_frame_or_stop(self, stop_event: threading.Event) -> bool:
        """Wait for a complete queued frame or a queue/scheduler shutdown."""
        with self._frame_condition:
            self._frame_condition.wait_for(
                lambda: self._queue.qsize() > 0 or self._closed or stop_event.is_set()
            )
            return (
                self._queue.qsize() > 0
                and not self._closed
                and not stop_event.is_set()
            )

    def wake_waiters(self) -> None:
        """Wake a scheduler so it can observe stop/reset state and recheck input."""
        with self._frame_condition:
            self._frame_condition.notify_all()

    def stats(self) -> dict[str, int]:
        with self._lock:
            result = dict(self._stats)
            result["input_frames_received"] = result["input_frames_enqueued"]
            result["queue_depth_frames"] = self._queue.qsize()
            result["queue_capacity_frames"] = self._queue.maxsize
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
            self._transport_buffer.clear()
            self._transport_serialization = None
            self._frame_condition.notify_all()

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
        on_error: Callable[[BaseException], None] | None = None,
        deadline_jitter_budget_ms: float = 20.0,
        history_limit: int = 4096,
        anchor_first_input: bool = False,
        ready_frame_dispatch: bool = False,
    ) -> None:
        if deadline_jitter_budget_ms < 0:
            raise ValueError("deadline_jitter_budget_ms non può essere negativo")
        if history_limit < 0:
            raise ValueError("history_limit non può essere negativo")
        if anchor_first_input and ready_frame_dispatch:
            raise ValueError("first-input anchor and ready-frame dispatch are mutually exclusive")
        self.ingress = ingress
        self.clock = clock or MonotonicFrameClock()
        self.on_listener = on_listener
        self.on_policy = on_policy
        self.on_speaker = on_speaker
        self.on_error = on_error
        self.anchor_first_input = bool(anchor_first_input)
        self.ready_frame_dispatch = bool(ready_frame_dispatch)
        self.history_limit = int(history_limit)
        self.deadline_jitter_budget_ns = int(deadline_jitter_budget_ms * 1_000_000)
        self._next_tick_index = 0
        self._ticks = 0
        self._missing_ticks = 0
        self._duplicate_ticks = 0
        self._out_of_order_ticks = 0
        self._deadline_misses = 0
        self._skipped_overdue_deadlines = 0
        self._jitters_ns: list[int] = []
        self._last_tick_frame_id: int | None = None
        self._input_frames_during_speaking = 0
        self._history: list[DuplexTick] = []
        self._lifecycle_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._callbacks_active = threading.Event()
        self._first_input_anchor_pending = self.anchor_first_input
        self._first_input_anchor_ns: int | None = None

    def _phase_now(self) -> int:
        return self.clock.now_ns()

    def start(self, *, name: str = "duplex-scheduler") -> None:
        """Start the clock loop with a fresh epoch; the loop is the queue consumer."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            # Sessions allocate their scheduler before expensive per-session
            # model/listener initialization. Anchoring the clock at
            # construction time makes start() catch up elapsed 80 ms deadlines
            # after initialization, producing artificial idle ticks and a
            # false ingress backlog. Restarting begins a fresh cadence too.
            if isinstance(self.clock, MonotonicFrameClock):
                self.clock.reset_epoch()
                self._next_tick_index = 0
            elif self.ready_frame_dispatch:
                # Ready-frame mode has no absolute scheduler grid. The index
                # remains diagnostic/logical state and restarts with the loop.
                self._next_tick_index = 0
            self._first_input_anchor_pending = self.anchor_first_input
            self._first_input_anchor_ns = None
            self._stop_event.clear()
            self._error = None
            self._thread = threading.Thread(target=self._run_forever, name=name, daemon=True)
            self._thread.start()

    def _run_forever(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self.tick() is None:
                    return
        except BaseException as exc:
            self._error = exc
            logger.exception("DuplexScheduler arrestato per errore")
            if self.on_error is not None:
                self.on_error(exc)

    def stop(self, *, timeout: float = 2.0) -> None:
        """Stop the clock loop and wait up to ``timeout`` for callbacks to exit."""
        with self._lifecycle_lock:
            thread = self._thread
            self._stop_event.set()
            self.ingress.wake_waiters()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        with self._lifecycle_lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> BaseException | None:
        return self._error

    @property
    def callbacks_active(self) -> bool:
        """Whether a real input frame is inside listener/policy/speaker callbacks."""
        return self._callbacks_active.is_set()

    @property
    def first_input_anchor_applied(self) -> bool:
        return self._first_input_anchor_ns is not None

    @property
    def first_input_anchor_ns(self) -> int | None:
        return self._first_input_anchor_ns

    def tick(self) -> DuplexTick | None:
        """Wait for and execute the next tick."""
        index = self._next_tick_index
        frame: AudioFrame | None = None
        guarded_dequeue = False
        if self.ready_frame_dispatch:
            # The ingress condition is notified only when a complete 80 ms
            # AudioFrame is enqueued. Condition.wait_for rechecks the queue,
            # close, and stop predicate under its associated lock; the
            # lifecycle lock then makes stop-vs-dequeue admission atomic.
            # Python 3.11 Condition contract:
            # https://docs.python.org/3.11/library/threading.html#threading.Condition.wait_for
            while True:
                if not self.ingress.wait_for_frame_or_stop(self._stop_event):
                    return None
                with self._lifecycle_lock:
                    if self._stop_event.is_set():
                        return None
                    try:
                        frame = self.ingress.get_nowait()
                    except queue.Empty:
                        # Queue reset between condition wake and admission.
                        continue
                    index = self._next_tick_index
                    deadline = int(frame.capture_end_ns)
                    actual = self._phase_now()
                    self._callbacks_active.set()
                break
            guarded_dequeue = True
        elif self._first_input_anchor_pending:
            while True:
                if not self.ingress.wait_for_frame_or_stop(self._stop_event):
                    return None
                # Serialize callback admission with stop(). If stop wins this
                # lock after the condition wake but before dequeue, leave the
                # queued frame untouched and exit without invoking callbacks.
                with self._lifecycle_lock:
                    if self._stop_event.is_set():
                        return None
                    try:
                        frame = self.ingress.get_nowait()
                    except queue.Empty:
                        # An explicit ingress reset can clear the queue between
                        # notification and dequeue. Recheck the condition instead
                        # of anchoring to an incomplete or stale buffer.
                        continue
                    # A restarted virtual/test scheduler may retain a logical
                    # tick index even though first-input anchoring starts a new
                    # phase. Reset that index only for this opt-in first frame;
                    # the default scheduler keeps its captured active behavior.
                    self._next_tick_index = 0
                    index = 0
                    self.clock.anchor_epoch(frame.enqueue_ns)
                    self._first_input_anchor_ns = frame.enqueue_ns
                    self._first_input_anchor_pending = False
                    # Mark this frame as admitted before releasing the lifecycle
                    # lock; a subsequent stop treats it as an in-flight tick.
                    self._callbacks_active.set()
                    deadline = self.clock.deadline_ns(index)
                    actual = self._phase_now()
                break
        else:
            anchored = self.anchor_first_input and self._first_input_anchor_ns is not None
            if anchored:
                # Opt-in anchored mode dispatches only real complete input
                # frames. Keep the next absolute deadline while ingress is
                # empty; after input arrives, wait only if that deadline is
                # still in the future. Late frames catch up one-per-callback
                # without synthetic idle ticks or skipped logical slots.
                while frame is None:
                    if self._stop_event.is_set():
                        return None
                    if not self.ingress.wait_for_frame_or_stop(self._stop_event):
                        return None
                    index = self._next_tick_index
                    deadline = self.clock.deadline_ns(index)
                    actual = self._phase_now()
                    if deadline > actual:
                        self.clock.wait_until(deadline)
                        actual = self._phase_now()
                    # Stop and queue admission share the lifecycle guard. A
                    # reset may empty ingress during a deadline wait; in that
                    # case, keep this same tick index and wait for real input.
                    with self._lifecycle_lock:
                        if self._stop_event.is_set():
                            return None
                        try:
                            frame = self.ingress.get_nowait()
                        except queue.Empty:
                            frame = None
                        if frame is not None:
                            self._callbacks_active.set()
                guarded_dequeue = True
            else:
                index = self._next_tick_index
                deadline = self.clock.deadline_ns(index)
                self.clock.wait_until(deadline)
                actual = self._phase_now()
        lateness = max(0, actual - deadline)
        self._jitters_ns.append(lateness)
        if lateness > self.deadline_jitter_budget_ns:
            self._deadline_misses += 1

        if frame is None and not guarded_dequeue:
            try:
                frame = self.ingress.get_nowait()
            except queue.Empty:
                frame = None
                self._missing_ticks += 1
        frame_id = frame.frame_id if frame is not None else index
        # A missing scheduler tick has no input identity. Do not compare its
        # synthetic frame_id with the next real ingress frame: doing so turns
        # ordinary queue starvation into a false duplicate/out-of-order event.
        if frame is not None:
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
            dispatch_mode=(
                "ready_frame_no_fixed_grid" if self.ready_frame_dispatch else None
            ),
        )

        # A missing input frame is an idle tick, not a callback invocation:
        # listener/policy/speaker are driven only by the scheduler when a
        # complete frame has been consumed from the ingress queue.
        if frame is not None:
            self._callbacks_active.set()
            try:
                if self.on_listener is not None:
                    tick.listener_start_ns = self._phase_now()
                    self.on_listener(frame, tick)
                    tick.listener_end_ns = self._phase_now()
                if self.on_policy is not None:
                    self.on_policy(frame, tick)
                    tick.policy_ns = self._phase_now()
                if self.on_speaker is not None:
                    self._input_frames_during_speaking += 1
                    tick.speaker_start_ns = self._phase_now()
                    self.on_speaker(frame, tick)
                    tick.speaker_end_ns = self._phase_now()
            finally:
                self._callbacks_active.clear()

        self._next_tick_index += 1
        self._ticks += 1
        if self.history_limit:
            self._history.append(tick)
            if len(self._history) > self.history_limit:
                del self._history[:-self.history_limit]
        return tick

    def dispatch_shared_frame(self, frame: AudioFrame, tick: DuplexTick) -> DuplexTick:
        """Run this session's ordinary callbacks for a frame already batched globally.

        The shared listener dispatcher owns ingress dequeue and listener
        inference. This method deliberately starts at callback fanout so the
        session scheduler cannot consume the same ingress frame a second time.
        """
        if not isinstance(frame, AudioFrame) or not isinstance(tick, DuplexTick):
            raise TypeError("shared dispatch requires one AudioFrame and DuplexTick")
        if int(frame.frame_id) != int(tick.frame_id):
            raise ValueError("shared dispatch frame and tick identities differ")
        self._callbacks_active.set()
        try:
            if self.on_listener is not None:
                # The shared batch timestamps cover the listener transaction;
                # do not replace them with only the post-batch callback cost.
                if tick.listener_start_ns is None:
                    tick.listener_start_ns = self._phase_now()
                self.on_listener(frame, tick)
                if tick.listener_end_ns is None:
                    tick.listener_end_ns = self._phase_now()
            if self.on_policy is not None:
                self.on_policy(frame, tick)
                tick.policy_ns = self._phase_now()
            if self.on_speaker is not None:
                self._input_frames_during_speaking += 1
                tick.speaker_start_ns = self._phase_now()
                self.on_speaker(frame, tick)
                tick.speaker_end_ns = self._phase_now()
        except BaseException as exc:
            self._error = exc
            if self.on_error is not None:
                self.on_error(exc)
        finally:
            self._callbacks_active.clear()

        if self._last_tick_frame_id is not None:
            if frame.frame_id == self._last_tick_frame_id:
                self._duplicate_ticks += 1
            elif frame.frame_id != self._last_tick_frame_id + 1:
                self._out_of_order_ticks += 1
        self._last_tick_frame_id = int(frame.frame_id)
        self._next_tick_index += 1
        self._ticks += 1
        if self.history_limit:
            self._history.append(tick)
            if len(self._history) > self.history_limit:
                del self._history[:-self.history_limit]
        return tick

    def run_ticks(self, count: int) -> list[DuplexTick]:
        if count < 0:
            raise ValueError("count non può essere negativo")
        ticks: list[DuplexTick] = []
        for _ in range(int(count)):
            tick = self.tick()
            if tick is None:
                break
            ticks.append(tick)
        return ticks

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
            "skipped_overdue_deadlines": self._skipped_overdue_deadlines,
            "deadline_miss_rate": self._deadline_misses / self._ticks if self._ticks else 0.0,
            "tick_jitter_p99_ms": round(jitter_p99, 3),
            "input_frames_during_speaking": self._input_frames_during_speaking,
            "callbacks_active": self.callbacks_active,
            "first_input_anchor_enabled": self.anchor_first_input,
            "first_input_anchor_applied": self.first_input_anchor_applied,
            "first_input_anchor_ns": self._first_input_anchor_ns,
            "ready_frame_dispatch_enabled": self.ready_frame_dispatch,
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
    "DIRECT_V2_INGRESS_QUEUE_FRAMES",
    "REALTIME_INGRESS_QUEUE_FRAMES",
    "VirtualFrameClock",
]
