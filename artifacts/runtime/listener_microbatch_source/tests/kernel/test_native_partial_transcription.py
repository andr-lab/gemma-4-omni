from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from duplex_kernel.native_partial_transcription import (
    NATIVE_PARTIAL_CADENCE_FRAMES,
    NATIVE_PARTIAL_MIN_VOICE_FRAMES,
    NativeTranscriptSnapshot,
    NativePartialTranscriber,
    should_request_partial_snapshot,
)


def _snapshot(value: float, *, last_frame_id: int, final: bool = False) -> NativeTranscriptSnapshot:
    return NativeTranscriptSnapshot(
        utterance_id="utt-1",
        pcm=np.full(1280, value, dtype=np.float32),
        first_frame_id=0,
        last_frame_id=last_frame_id,
        generation_id=4,
        final=final,
    )


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for worker state")


def test_worker_keeps_only_latest_pending_snapshot_without_fifo() -> None:
    started = threading.Event()
    release = threading.Event()
    calls: list[float] = []
    published: list[tuple[str, int, str]] = []

    def transcribe(pcm, *, stop_check):
        value = float(pcm[0])
        calls.append(value)
        if value == 1.0:
            started.set()
            assert release.wait(2.0)
        assert not stop_check()
        return {"text": f"snapshot-{int(value)}"}

    worker = NativePartialTranscriber(
        transcribe=transcribe,
        publish=lambda snapshot, text: published.append((snapshot.utterance_id, snapshot.last_frame_id, text)),
        is_current=lambda _snapshot: True,
        max_samples=16000 * 30,
    )
    try:
        assert worker.submit(_snapshot(1.0, last_frame_id=9))
        assert started.wait(2.0)
        assert worker.submit(_snapshot(2.0, last_frame_id=19))
        assert worker.submit(_snapshot(3.0, last_frame_id=29))
        release.set()
        assert worker.wait_idle(timeout=2.0)
    finally:
        worker.close()

    assert calls == [1.0, 3.0]
    assert published == [("utt-1", 9, "snapshot-1"), ("utt-1", 29, "snapshot-3")]


def test_worker_does_not_let_new_partial_overwrite_pending_final_snapshot() -> None:
    started = threading.Event()
    release = threading.Event()
    calls: list[float] = []

    def transcribe(pcm, *, stop_check):
        value = float(pcm[0])
        calls.append(value)
        if value == 1.0:
            started.set()
            assert release.wait(2.0)
        assert not stop_check()
        return {"text": str(int(value))}

    worker = NativePartialTranscriber(
        transcribe=transcribe,
        publish=lambda *_args: None,
        is_current=lambda _snapshot: True,
        max_samples=16000 * 30,
    )
    try:
        assert worker.submit(_snapshot(1.0, last_frame_id=9))
        assert started.wait(2.0)
        assert worker.submit(_snapshot(2.0, last_frame_id=19, final=True))
        assert worker.submit(_snapshot(3.0, last_frame_id=29))
        release.set()
        assert worker.wait_idle(timeout=2.0)
    finally:
        worker.close()

    assert calls == [1.0, 2.0]


def test_worker_cancellation_discards_running_result_and_pending_snapshot() -> None:
    started = threading.Event()
    cancelled = threading.Event()
    release = threading.Event()
    published: list[str] = []

    def transcribe(_pcm, *, stop_check):
        started.set()
        while not release.is_set():
            if stop_check():
                cancelled.set()
                return {"text": "stale transcript", "cancelled": True}
            time.sleep(0.005)
        return {"text": "stale transcript"}

    worker = NativePartialTranscriber(
        transcribe=transcribe,
        publish=lambda _snapshot, text: published.append(text),
        is_current=lambda _snapshot: True,
        max_samples=16000 * 30,
    )
    try:
        assert worker.submit(_snapshot(1.0, last_frame_id=9))
        assert started.wait(2.0)
        assert worker.submit(_snapshot(2.0, last_frame_id=19))
        worker.cancel()
        assert cancelled.wait(2.0)
        release.set()
        assert worker.wait_idle(timeout=2.0)
    finally:
        worker.close()

    assert cancelled.is_set()
    assert published == []


def test_worker_pause_coalesces_latest_and_resumes_after_response_busy() -> None:
    started = threading.Event()
    calls: list[float] = []
    published: list[tuple[int, str]] = []

    def transcribe(pcm, *, stop_check):
        value = float(pcm[0])
        calls.append(value)
        if value == 1.0:
            started.set()
            while not stop_check():
                time.sleep(0.002)
            return {"text": "cancelled old decode", "cancelled": True}
        return {"text": f"snapshot-{int(value)}"}

    worker = NativePartialTranscriber(
        transcribe=transcribe,
        publish=lambda snapshot, text: published.append((snapshot.last_frame_id, text)),
        is_current=lambda _snapshot: True,
        max_samples=16000 * 30,
    )
    try:
        assert worker.submit(_snapshot(1.0, last_frame_id=9))
        assert started.wait(2.0)
        worker.pause()
        assert worker.submit(_snapshot(2.0, last_frame_id=19))
        assert worker.submit(_snapshot(3.0, last_frame_id=29, final=True))
        time.sleep(0.02)
        assert calls == [1.0], "paused ASR must not start queued work during assistant output"
        worker.resume()
        assert worker.wait_idle(timeout=2.0)
    finally:
        worker.close()

    assert calls == [1.0, 3.0]
    assert published == [(29, "snapshot-3")]


def test_worker_cancel_after_current_check_before_publish_cannot_leak_stale_snapshot() -> None:
    current_checks = 0
    checked_before_publish = threading.Event()
    release_current_check = threading.Event()
    published: list[str] = []

    def is_current(_snapshot):
        nonlocal current_checks
        current_checks += 1
        if current_checks == 2:
            checked_before_publish.set()
            assert release_current_check.wait(2.0)
        return True

    worker = NativePartialTranscriber(
        transcribe=lambda _pcm, *, stop_check: {"text": "stale snapshot"},
        publish=lambda _snapshot, transcript: published.append(transcript),
        is_current=is_current,
        max_samples=16000 * 30,
    )
    try:
        assert worker.submit(_snapshot(1.0, last_frame_id=9))
        assert checked_before_publish.wait(2.0)
        # Mirrors opt-out/commit advancing the server transcript epoch before
        # it waits for the worker lock: the current check already returned,
        # but the publication fence must still discard this snapshot.
        worker.cancel()
        release_current_check.set()
        assert worker.wait_idle(timeout=2.0)
    finally:
        release_current_check.set()
        worker.close()

    assert current_checks == 2
    assert published == []


def test_worker_rejects_audio_outside_finite_30_second_contract() -> None:
    worker = NativePartialTranscriber(
        transcribe=lambda _pcm, *, stop_check: {"text": "unused"},
        publish=lambda *_args: None,
        is_current=lambda _snapshot: True,
        max_samples=16000 * 30,
    )
    try:
        oversized = NativeTranscriptSnapshot(
            utterance_id="utt-1",
            pcm=np.zeros(16000 * 30 + 1, dtype=np.float32),
            first_frame_id=0,
            last_frame_id=1,
            generation_id=4,
        )
        with pytest.raises(ValueError, match="30-second"):
            worker.submit(oversized)

        nonfinite = _snapshot(1.0, last_frame_id=9)
        nonfinite.pcm[0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            worker.submit(nonfinite)
    finally:
        worker.close()


def test_partial_schedule_waits_for_one_second_voice_and_uses_coarse_cadence() -> None:
    def due(frame_id: int, last_snapshot_frame_id: int | None = None, *, busy: bool = False) -> bool:
        return should_request_partial_snapshot(
            enabled=True,
            voice_active=True,
            response_busy=busy,
            frame_id=frame_id,
            onset_frame_id=0,
            last_snapshot_frame_id=last_snapshot_frame_id,
        )

    assert NATIVE_PARTIAL_MIN_VOICE_FRAMES * 80 >= 1000
    assert NATIVE_PARTIAL_CADENCE_FRAMES * 80 >= 1500
    assert not due(NATIVE_PARTIAL_MIN_VOICE_FRAMES - 2)
    assert due(NATIVE_PARTIAL_MIN_VOICE_FRAMES - 1)
    assert not due(NATIVE_PARTIAL_MIN_VOICE_FRAMES - 1 + NATIVE_PARTIAL_CADENCE_FRAMES - 1,
                   NATIVE_PARTIAL_MIN_VOICE_FRAMES - 1)
    assert due(NATIVE_PARTIAL_MIN_VOICE_FRAMES - 1 + NATIVE_PARTIAL_CADENCE_FRAMES,
               NATIVE_PARTIAL_MIN_VOICE_FRAMES - 1)
    assert not due(50, busy=True)
    assert not should_request_partial_snapshot(
        enabled=False,
        voice_active=True,
        response_busy=False,
        frame_id=20,
        onset_frame_id=0,
        last_snapshot_frame_id=None,
    )
