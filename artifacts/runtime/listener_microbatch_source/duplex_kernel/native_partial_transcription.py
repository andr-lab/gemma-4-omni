"""Coalesced, observational Gemma transcription of live audio snapshots."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

NATIVE_PARTIAL_MIN_VOICE_FRAMES = 13  # 1.04 s on the 80 ms listener clock.
NATIVE_PARTIAL_CADENCE_FRAMES = 19  # 1.52 s between snapshots.


@dataclass(frozen=True)
class NativeTranscriptSnapshot:
    """One bounded causal audio snapshot owned by an utterance epoch."""

    utterance_id: str
    pcm: np.ndarray
    first_frame_id: int
    last_frame_id: int
    generation_id: int
    final: bool = False
    epoch_id: int = 0


def should_request_partial_snapshot(
    *,
    enabled: bool,
    voice_active: bool,
    response_busy: bool,
    frame_id: int,
    onset_frame_id: int | None,
    last_snapshot_frame_id: int | None,
) -> bool:
    """Apply the opt-in ASR cadence without work on each listener frame."""
    if not enabled or not voice_active or response_busy or onset_frame_id is None:
        return False
    current = int(frame_id)
    onset = int(onset_frame_id)
    if current < onset:
        return False
    observed_frames = current - onset + 1
    if observed_frames < NATIVE_PARTIAL_MIN_VOICE_FRAMES:
        return False
    return (
        last_snapshot_frame_id is None
        or current - int(last_snapshot_frame_id) >= NATIVE_PARTIAL_CADENCE_FRAMES
    )


class NativePartialTranscriber:
    """Run at most one ASR job and retain only its newest pending snapshot.

    A final snapshot cannot be displaced by a newer partial until it completes.
    Cancellation is cooperative at the ASR decoder's next token boundary; a
    running audio-feature prefill remains atomic and cannot be interrupted.
    """

    def __init__(
        self,
        *,
        transcribe: Callable[..., Any],
        publish: Callable[[NativeTranscriptSnapshot, str], None],
        is_current: Callable[[NativeTranscriptSnapshot], bool],
        max_samples: int,
    ) -> None:
        if isinstance(max_samples, bool) or int(max_samples) < 1:
            raise ValueError("max_samples must be positive")
        self._transcribe = transcribe
        self._publish = publish
        self._is_current = is_current
        self._max_samples = int(max_samples)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="native-partial-asr")
        self._lock = threading.RLock()
        self._active_snapshot: NativeTranscriptSnapshot | None = None
        self._active_cancel: threading.Event | None = None
        self._pending: NativeTranscriptSnapshot | None = None
        self._paused = False
        self._closed = False

    @staticmethod
    def _should_replace_pending(
        incoming: NativeTranscriptSnapshot,
        pending: NativeTranscriptSnapshot,
    ) -> bool:
        if incoming.final != pending.final:
            return incoming.final
        if incoming.utterance_id != pending.utterance_id:
            return True
        return incoming.last_frame_id >= pending.last_frame_id

    def submit(self, snapshot: NativeTranscriptSnapshot) -> bool:
        if not isinstance(snapshot, NativeTranscriptSnapshot):
            raise TypeError("snapshot must be NativeTranscriptSnapshot")
        if not str(snapshot.utterance_id).strip():
            raise ValueError("utterance_id must be non-empty")
        if int(snapshot.first_frame_id) < 0 or int(snapshot.last_frame_id) < int(snapshot.first_frame_id):
            raise ValueError("snapshot frame range must be non-negative and ordered")
        pcm = np.asarray(snapshot.pcm, dtype=np.float32).reshape(-1)
        if pcm.size == 0 or not np.isfinite(pcm).all():
            raise ValueError("native transcript PCM must be non-empty and finite")
        if pcm.size > self._max_samples:
            raise ValueError("native transcript PCM exceeds the 30-second contract")
        owned = replace(snapshot, pcm=np.array(pcm, dtype=np.float32, copy=True))

        with self._lock:
            if self._closed:
                return False
            if self._paused:
                if self._pending is None or self._should_replace_pending(owned, self._pending):
                    self._pending = owned
                return True
            if self._active_cancel is None:
                self._start_locked(owned)
                return True
            if self._pending is None or self._should_replace_pending(owned, self._pending):
                self._pending = owned
            return True

    def _start_locked(self, snapshot: NativeTranscriptSnapshot) -> None:
        cancel = threading.Event()
        self._active_snapshot = snapshot
        self._active_cancel = cancel
        self._executor.submit(self._run, snapshot, cancel)

    @staticmethod
    def _transcript(result: Any) -> str:
        if isinstance(result, str):
            return result.strip()
        if not isinstance(result, dict) or result.get("cancelled"):
            return ""
        return str(result.get("text", "")).strip()

    def _run(self, snapshot: NativeTranscriptSnapshot, cancel: threading.Event) -> None:
        current_snapshot = snapshot
        current_cancel = cancel
        while True:
            transcript = ""
            try:
                if not current_cancel.is_set() and self._is_current(current_snapshot):
                    result = self._transcribe(current_snapshot.pcm, stop_check=current_cancel.is_set)
                    transcript = self._transcript(result)
            except Exception:
                # ASR failures are observational and must not terminate the
                # listener or the worker that owns the latest coalesced update.
                transcript = ""

            with self._lock:
                may_publish = (
                    not self._closed
                    and not current_cancel.is_set()
                    and self._active_cancel is current_cancel
                )
            is_current = may_publish and self._is_current(current_snapshot)
            if is_current and (transcript or current_snapshot.final):
                # Keep the cancel fence atomic with publication, but invoke the
                # external session predicate outside the worker lock to avoid
                # inversion with the server's generation lock.
                with self._lock:
                    if (
                        not self._closed
                        and not current_cancel.is_set()
                        and self._active_cancel is current_cancel
                    ):
                        self._publish(current_snapshot, transcript)

            with self._lock:
                if self._closed or self._active_cancel is not current_cancel:
                    return
                if self._paused:
                    # The active decoder has cooperatively stopped. Keep the
                    # latest bounded snapshot parked until response work ends.
                    self._active_cancel = None
                    self._active_snapshot = None
                    return
                pending = self._pending
                self._pending = None
                if pending is None:
                    self._active_cancel = None
                    self._active_snapshot = None
                    return
                current_snapshot = pending
                current_cancel = threading.Event()
                self._active_snapshot = current_snapshot
                self._active_cancel = current_cancel

    def cancel(self) -> None:
        """Cancel the active decode and discard the single pending snapshot."""
        with self._lock:
            if self._active_cancel is not None:
                self._active_cancel.set()
            self._pending = None

    def pause(self) -> None:
        """Defer ASR while response work owns the accelerator.

        The running decode is stopped cooperatively and retained as the
        fallback pending snapshot. Newer snapshots replace it using the same
        one-job/one-latest coalescing rule as normal operation.
        """
        with self._lock:
            if self._closed or self._paused:
                return
            self._paused = True
            if self._active_cancel is not None:
                active = self._active_snapshot
                if active is not None and (
                    self._pending is None or self._should_replace_pending(active, self._pending)
                ):
                    self._pending = active
                self._active_cancel.set()

    def resume(self) -> None:
        """Restart the newest parked snapshot after response work ends."""
        with self._lock:
            if self._closed or not self._paused:
                return
            self._paused = False
            if self._active_cancel is None and self._pending is not None:
                pending = self._pending
                self._pending = None
                self._start_locked(pending)

    def wait_idle(self, *, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            with self._lock:
                if self._active_cancel is None and self._pending is None:
                    return True
            time.sleep(0.005)
        with self._lock:
            return self._active_cancel is None and self._pending is None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._active_cancel is not None:
                self._active_cancel.set()
            self._pending = None
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = [
    "NATIVE_PARTIAL_CADENCE_FRAMES",
    "NATIVE_PARTIAL_MIN_VOICE_FRAMES",
    "NativePartialTranscriber",
    "NativeTranscriptSnapshot",
    "should_request_partial_snapshot",
]
