"""Shared response, speech tracking, and execution state for the duplex server."""
from __future__ import annotations

import asyncio
import hashlib
import os
import queue
import threading
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np

from duplex_kernel.audio_stream import AudioPacket, NetworkPacketizer
from duplex_kernel.playout import SpeakerPlayoutBuffer
from duplex_kernel.session import GenerationToken
from duplex_kernel.config import USER_SAMPLE_RATE, V9_TRAINING_REFERENCE_SHA256

POLICY_FRAME_MS = 80
POLICY_FRAME_SAMPLES = USER_SAMPLE_RATE * POLICY_FRAME_MS // 1000
REALTIME_AUDIO_PACKET_CAPACITY = 4
DEFAULT_SPEAK_ENDPOINT_QUIET_MS = 800


class _ResponseAudioAdmissionState:
    """Response watermark and sent-frame count guarded by policy_causality_lock."""

    def __init__(self) -> None:
        self.latest_input_frame_id = -1
        self.input_watermark = -1
        self.response_start_watermark = -1
        self.emitted_audio_frames = 0

    def update_input_frame_locked(self, frame_id: int) -> None:
        self.latest_input_frame_id = max(self.latest_input_frame_id, int(frame_id))

    def mark_audio_sent_locked(self, frames: int) -> None:
        if self.emitted_audio_frames == 0:
            self.input_watermark = self.latest_input_frame_id
        self.emitted_audio_frames += max(0, int(frames))

    def reset_locked(self) -> None:
        self.response_start_watermark = self.latest_input_frame_id
        self.input_watermark = -1
        self.emitted_audio_frames = 0

    def has_post_response_voice_locked(self, speech_tracker: "_CausalUserSpeechTracker") -> bool:
        watermark = self.response_start_watermark
        return bool(
            watermark >= 0
            and (
                speech_tracker.has_onset_after(watermark)
                or (
                    speech_tracker.last_voiced_frame_id is not None
                    and speech_tracker.last_voiced_frame_id > watermark
                )
            )
        )

    def snapshot_locked(self, speech_tracker: "_CausalUserSpeechTracker") -> tuple[int, bool]:
        has_post_watermark_voice = (
            self.input_watermark >= 0
            and (
                speech_tracker.has_onset_after(self.input_watermark)
                or (
                    speech_tracker.last_voiced_frame_id is not None
                    and speech_tracker.last_voiced_frame_id > self.input_watermark
                )
            )
        )
        return self.emitted_audio_frames, has_post_watermark_voice


def _configured_speak_endpoint_frames() -> int:
    raw = os.environ.get("DUPLEX_SPEAK_ENDPOINT_QUIET_MS", str(DEFAULT_SPEAK_ENDPOINT_QUIET_MS))
    try:
        quiet_ms = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("DUPLEX_SPEAK_ENDPOINT_QUIET_MS deve essere un multiplo positivo di 80 ms") from exc
    if quiet_ms <= 0 or quiet_ms % POLICY_FRAME_MS:
        raise ValueError("DUPLEX_SPEAK_ENDPOINT_QUIET_MS deve essere un multiplo positivo di 80 ms")
    return quiet_ms // POLICY_FRAME_MS


class _CausalUserSpeechTracker:
    """Track voiced runs and endpoints from current, already-consumed PCM frames."""

    # The initial estimate is deliberately below the target room-noise range.
    # It is replaced from quiet frames before it can become a hard VAD floor.
    INITIAL_NOISE_FLOOR_RMS = 0.001
    MIN_VOICE_RMS = 0.002
    VOICE_TO_NOISE_RATIO = 1.10
    VOICED_AUTOCORRELATION = 0.32
    VOICED_SPECTRAL_FLATNESS = 0.22
    FRICATIVE_HIGH_TO_MID_RATIO = 2.5
    FRICATIVE_ENVELOPE_RATIO = 1.35
    NOISE_SPECTRAL_FLATNESS_MIN = 0.30
    NOISE_MAX_PITCH_CORRELATION = 0.20
    NOISE_MAX_HIGH_TO_MID_RATIO = 2.2
    NOISE_MAX_ENVELOPE_RATIO = 1.30
    STATIONARY_NOISE_CONFIRM_FRAMES = 5
    STATIONARY_NOISE_RELATIVE_SPREAD = 0.30

    def __init__(
        self,
        *,
        endpoint_quiet_frames: int,
        initial_noise_floor_rms: float = INITIAL_NOISE_FLOOR_RMS,
    ) -> None:
        if isinstance(endpoint_quiet_frames, bool) or int(endpoint_quiet_frames) < 1:
            raise ValueError("endpoint_quiet_frames must be a positive integer")
        if (
            isinstance(initial_noise_floor_rms, bool)
            or not np.isfinite(float(initial_noise_floor_rms))
            or float(initial_noise_floor_rms) <= 0.0
        ):
            raise ValueError("initial_noise_floor_rms must be finite and positive")
        self.endpoint_quiet_frames = int(endpoint_quiet_frames)
        self.noise_floor_rms = float(initial_noise_floor_rms)
        self._quiet_rms_samples: deque[float] = deque(
            (self.noise_floor_rms,) * 7,
            maxlen=15,
        )
        self._stationary_noise_rms: deque[float] = deque(
            maxlen=self.STATIONARY_NOISE_CONFIRM_FRAMES,
        )
        self.last_voiced_frame_id: int | None = None
        self.run_onset_frame_id: int | None = None
        self.voice_run_active = False
        self.quiet_frames = 0
        self.endpoint_frame_id: int | None = None
        self.endpoint_source_cutoff_sample_id: int | None = None
        self.speech_complete = False

    @staticmethod
    def _voice_features(frame: np.ndarray) -> tuple[float, float, float, float, float]:
        """Return RMS, pitch correlation, spectral flatness, high/mid ratio, and envelope ratio.

        One zero-padded FFT supplies both the spectral shape and autocorrelation
        evidence.  The high-band branch keeps unvoiced fricatives eligible; the
        classifier does not require pitch periodicity for every speech frame.
        """
        centered = np.asarray(frame, dtype=np.float64)
        rms = float(np.sqrt(np.mean(centered * centered)))
        centered = centered - float(np.mean(centered))
        if centered.size < 8:
            return rms, 0.0, 1.0, 0.0, 1.0

        subframe_count = min(4, max(1, centered.size // 240))
        chunk_size = centered.size // subframe_count
        usable = centered[:chunk_size * subframe_count].reshape(subframe_count, chunk_size)
        chunk_rms = np.sqrt(np.mean(np.square(usable), axis=1))
        envelope_ratio = float(
            np.quantile(chunk_rms, 0.8) / max(float(np.quantile(chunk_rms, 0.2)), 1e-8)
        )

        nfft = 1 << (2 * centered.size - 1).bit_length()
        windowed = centered * np.hanning(centered.size)
        spectrum = np.fft.rfft(windowed, n=nfft)
        power = np.square(np.abs(spectrum))
        frequencies = np.fft.rfftfreq(nfft, d=1.0 / USER_SAMPLE_RATE)
        valid = (frequencies >= 100.0) & (frequencies <= min(10_000.0, USER_SAMPLE_RATE / 2.0))
        valid_power = power[valid]
        valid_frequencies = frequencies[valid]
        if valid_power.size:
            flatness = float(
                np.exp(np.mean(np.log(valid_power + 1e-18)))
                / max(float(np.mean(valid_power)), 1e-18)
            )
            low_mid = float(np.sum(power[(frequencies >= 250.0) & (frequencies < 4_000.0)]))
            high = float(np.sum(power[(frequencies >= 4_000.0) & (frequencies <= min(10_000.0, USER_SAMPLE_RATE / 2.0))]))
            high_to_mid = high / max(low_mid, 1e-18)
            # An 80 ms aggregate can dilute a short /s/ or /f/ into the room
            # floor. Preserve its local band shape across the 20 ms subframes.
            subframe_frequencies = np.fft.rfftfreq(1024, d=1.0 / USER_SAMPLE_RATE)
            subframe_low = (subframe_frequencies >= 250.0) & (subframe_frequencies < 4_000.0)
            subframe_high = (subframe_frequencies >= 4_000.0) & (
                subframe_frequencies <= min(10_000.0, USER_SAMPLE_RATE / 2.0)
            )
            for chunk in usable:
                chunk = chunk - float(np.mean(chunk))
                chunk_power = np.square(
                    np.abs(np.fft.rfft(chunk * np.hanning(chunk.size), n=1024))
                )
                chunk_low_energy = float(np.sum(chunk_power[subframe_low]))
                chunk_high_energy = float(np.sum(chunk_power[subframe_high]))
                high_to_mid = max(
                    high_to_mid,
                    chunk_high_energy / max(chunk_low_energy, 1e-18),
                )
        else:
            flatness = 1.0
            high_to_mid = 0.0

        # Linear autocorrelation from the same spectrum; the chosen lag band
        # covers typical adult and child speech fundamentals at 24 kHz.
        autocorrelation = np.fft.irfft(power, n=nfft)[: nfft // 2]
        normalizer = max(float(autocorrelation[0]), 1e-18)
        min_lag = max(1, int(USER_SAMPLE_RATE / 400.0))
        max_lag = min(int(USER_SAMPLE_RATE / 70.0), autocorrelation.size - 1)
        pitch_correlation = (
            float(np.max(autocorrelation[min_lag:max_lag + 1])) / normalizer
            if max_lag >= min_lag else 0.0
        )
        return rms, pitch_correlation, flatness, high_to_mid, envelope_ratio

    def _classify_voice(self, frame: np.ndarray) -> tuple[bool, float]:
        features = self._voice_features(frame)
        return self._classify_voice_features(features), float(features[0])

    def _classify_voice_features(
        self,
        features: tuple[float, float, float, float, float],
    ) -> bool:
        rms, pitch_correlation, flatness, high_to_mid, envelope_ratio = features
        floor = self.noise_floor_rms
        above_floor = rms >= max(self.MIN_VOICE_RMS, floor * self.VOICE_TO_NOISE_RATIO)
        voiced_structure = (
            pitch_correlation >= self.VOICED_AUTOCORRELATION
            and flatness <= 0.65
        ) or (
            flatness <= self.VOICED_SPECTRAL_FLATNESS
            and pitch_correlation >= 0.10
        )
        fricative_structure = (
            high_to_mid >= self.FRICATIVE_HIGH_TO_MID_RATIO
            and envelope_ratio >= self.FRICATIVE_ENVELOPE_RATIO
        )
        return bool(above_floor and (voiced_structure or fricative_structure))

    def _observe_quiet_noise_floor(
        self,
        features: tuple[float, float, float, float, float],
    ) -> None:
        rms, pitch_correlation, flatness, high_to_mid, envelope_ratio = features
        if not np.isfinite(rms):
            return

        noise_like = (
            flatness >= self.NOISE_SPECTRAL_FLATNESS_MIN
            and pitch_correlation <= self.NOISE_MAX_PITCH_CORRELATION
            and high_to_mid <= self.NOISE_MAX_HIGH_TO_MID_RATIO
            and envelope_ratio <= self.NOISE_MAX_ENVELOPE_RATIO
        )
        if not noise_like:
            self._stationary_noise_rms.clear()
            return

        if rms > 0.01:
            # A single loud transient is not a room-noise estimate. Require a
            # stable run of noise-shaped frames before admitting high RMS rows.
            self._stationary_noise_rms.append(float(rms))
            if len(self._stationary_noise_rms) < self.STATIONARY_NOISE_CONFIRM_FRAMES:
                return
            samples = np.asarray(self._stationary_noise_rms, dtype=np.float64)
            median = float(np.median(samples))
            relative_spread = (
                float(np.quantile(samples, 0.9) - np.quantile(samples, 0.1))
                / max(median, 1e-8)
            )
            if relative_spread > self.STATIONARY_NOISE_RELATIVE_SPREAD:
                return
        else:
            self._stationary_noise_rms.clear()

        # This rolling median follows quiet room-noise changes. Frames with
        # periodic or fricative speech structure never enter it, even when
        # their absolute RMS is low. High-RMS noise must also pass the
        # multi-frame stationarity check above.
        self._quiet_rms_samples.append(max(0.00025, float(rms)))
        if self._quiet_rms_samples:
            self.noise_floor_rms = float(np.median(self._quiet_rms_samples))

    @staticmethod
    def _robust_noise_rms(frame: np.ndarray) -> float:
        """Estimate stationary Gaussian noise without letting a single click set the floor."""
        values = np.asarray(frame, dtype=np.float64).reshape(-1)
        if not values.size:
            return 0.0
        centered = values - float(np.median(values))
        # Median absolute deviation is unchanged by sparse high-amplitude
        # transients; the Gaussian consistency factor reports an RMS-like
        # scale for the stationary room-noise frames admitted below.
        return float(np.median(np.abs(centered)) / 0.6744897501960817)

    @classmethod
    def _noise_floor_features(
        cls,
        frame: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        """Measure noise shape after clipping sparse transients to a robust bound.

        The voice classifier still sees the untouched PCM. This copy is only
        used to decide whether a quiet frame can update the noise estimate, so
        a single click cannot make an otherwise stationary row look
        non-stationary and stall floor adaptation.
        """
        values = np.asarray(frame, dtype=np.float64).reshape(-1)
        if not values.size:
            return 0.0, 0.0, 1.0, 0.0, 1.0

        center = float(np.median(values))
        noise_rms = cls._robust_noise_rms(values)
        # Keep weak fricative/noise structure visible when MAD is near zero,
        # while bounding isolated clicks well below speech-scale excursions.
        clip_radius = max(8.0 * noise_rms, 4.0 * cls.MIN_VOICE_RMS)
        clipped = np.clip(values, center - clip_radius, center + clip_radius)
        _, pitch_correlation, flatness, high_to_mid, envelope_ratio = cls._voice_features(clipped)
        return noise_rms, pitch_correlation, flatness, high_to_mid, envelope_ratio

    def observe(self, frame_id: int, pcm: np.ndarray) -> tuple[bool, bool]:
        frame = np.asarray(pcm, dtype=np.float32).reshape(-1)
        voiced = False
        rms = 0.0
        was_voice_run_active = self.voice_run_active
        if frame.size and np.isfinite(frame).all():
            features = self._voice_features(frame)
            rms = float(features[0])
            voiced = self._classify_voice_features(features)
        else:
            features = None
        if not voiced and not was_voice_run_active and frame.size and np.isfinite(frame).all():
            self._observe_quiet_noise_floor(self._noise_floor_features(frame))
        new_onset = voiced and not self.voice_run_active
        endpoint = False
        if voiced:
            if new_onset:
                self.run_onset_frame_id = int(frame_id)
                self.speech_complete = False
                self.endpoint_frame_id = None
                self.endpoint_source_cutoff_sample_id = None
            self.voice_run_active = True
            self.quiet_frames = 0
            self.last_voiced_frame_id = int(frame_id)
        elif self.voice_run_active:
            self.quiet_frames += 1
            if self.quiet_frames >= self.endpoint_quiet_frames:
                self.voice_run_active = False
                self.speech_complete = True
                self.endpoint_frame_id = int(frame_id)
                self.endpoint_source_cutoff_sample_id = (
                    int(frame_id) * int(frame.size) + int(frame.size)
                )
                endpoint = True
        return bool(new_onset), endpoint

    def has_onset_after(self, watermark_frame_id: int) -> bool:
        return (
            self.run_onset_frame_id is not None
            and self.run_onset_frame_id > int(watermark_frame_id)
        )

    def consume_endpoint(self) -> None:
        """Prevent one completed utterance from authorizing multiple SPEAKs."""
        self.speech_complete = False
        self.endpoint_frame_id = None
        self.endpoint_source_cutoff_sample_id = None


class _SchedulerEventBuffer:
    """Bounded thread-to-async bridge with one coalesced loop notification."""

    OVERFLOW = object()

    def __init__(self, loop: asyncio.AbstractEventLoop, *, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("scheduler event capacity must be positive")
        self._loop = loop
        self._maxsize = int(maxsize)
        self._events: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()
        self._wake = asyncio.Event()
        self._wake_scheduled = False
        self._closed = False
        self._overflowed = False

    @property
    def qsize(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def overflowed(self) -> bool:
        with self._lock:
            return self._overflowed

    def _schedule_wake(self) -> bool:
        try:
            self._loop.call_soon_threadsafe(self._wake.set)
        except RuntimeError:
            with self._lock:
                self._closed = True
                self._events.clear()
            return False
        return True

    def put_nowait(self, event: dict[str, Any]) -> bool:
        notify = False
        overflowed_now = False
        with self._lock:
            if self._closed or self._overflowed:
                return False
            if len(self._events) >= self._maxsize:
                self._overflowed = True
                overflowed_now = True
            else:
                self._events.append(dict(event))
            if not self._wake_scheduled:
                self._wake_scheduled = True
                notify = True
        notified = not notify or self._schedule_wake()
        return notified and not overflowed_now

    async def get(self) -> dict[str, Any] | object | None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            with self._lock:
                if self._overflowed:
                    self._events.clear()
                    return self.OVERFLOW
                if self._closed:
                    self._events.clear()
                    self._wake_scheduled = False
                    return None
                if self._events:
                    event = self._events.popleft()
                    if self._events:
                        # Keep the wake level-triggered while draining queued
                        # events; producers need not enqueue per-event callbacks.
                        self._wake.set()
                    else:
                        self._wake_scheduled = False
                    return event
                self._wake_scheduled = False

    def close(self) -> None:
        notify = False
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if not self._wake_scheduled:
                self._wake_scheduled = True
                notify = True
        if notify:
            self._schedule_wake()


class _RealtimeResponseControl:
    """Thread-shared handles used by scheduler STOP/FLUSH callbacks."""

    def __init__(
        self,
        token: GenerationToken,
        playout: SpeakerPlayoutBuffer,
        packetizer: NetworkPacketizer,
        audio_queue: queue.Queue[AudioPacket],
        text_queue: queue.Queue[str],
        output_cancel: threading.Event | None = None,
        cancel_reason: dict[str, str | None] | None = None,
    ) -> None:
        self.token = token
        self.playout = playout
        self.packetizer = packetizer
        self.audio_queue = audio_queue
        self.text_queue = text_queue
        self.output_cancel = output_cancel
        self.cancel_reason = cancel_reason
        self.segment_opened = False
        self.segment_close_requested = False
        self.segment_advances = 0
        self.backchannel_requested = False


class _EngineSynthesisOwner:
    """Fence legacy global stop and concurrent response-local cancellation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: tuple[str, str] | None = None
        self._parallel_owners: dict[tuple[str, str], Callable[[], Any]] = {}

    def claim(
        self,
        session_id: str,
        response_id: str,
        *,
        allow_concurrent: bool = False,
        cancel: Callable[[], Any] | None = None,
    ) -> None:
        owner = (str(session_id), str(response_id))
        with self._lock:
            if allow_concurrent:
                if self._owner is not None or len(self._parallel_owners) >= 2:
                    raise RuntimeError("capacità synthesis parallela esaurita")
                if owner in self._parallel_owners:
                    raise RuntimeError(f"synthesis già posseduta da {owner!r}")
                if cancel is None:
                    raise ValueError("parallel synthesis requires a response-local cancel callback")
                self._parallel_owners[owner] = cancel
                return
            if self._owner is not None or self._parallel_owners:
                raise RuntimeError(f"engine synthesis già posseduta da {self._owner!r}")
            self._owner = owner

    def release(self, session_id: str, response_id: str) -> None:
        owner = (str(session_id), str(response_id))
        with self._lock:
            if self._owner == owner:
                self._owner = None
            self._parallel_owners.pop(owner, None)

    def is_owned(self, session_id: str, response_id: str | None) -> bool:
        if response_id is None:
            return False
        with self._lock:
            owner = (str(session_id), str(response_id))
            return self._owner == owner or owner in self._parallel_owners

    def stop_if_owned(self, session_id: str, response_id: str, stop: Callable[[], Any]) -> bool:
        """Make owner validation and the shared engine stop one atomic action."""
        with self._lock:
            owner = (str(session_id), str(response_id))
            if owner in self._parallel_owners:
                self._parallel_owners[owner]()
                return True
            if self._owner != owner:
                return False
            stop()
            return True


class _InferenceExecutionGate:
    """Allow two pooled responses together while excluding legacy callers."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._legacy_active = False
        self._pooled_active = 0

    @contextmanager
    def legacy(self):
        with self._condition:
            while self._legacy_active or self._pooled_active:
                self._condition.wait()
            self._legacy_active = True
        try:
            yield
        finally:
            with self._condition:
                self._legacy_active = False
                self._condition.notify_all()

    @contextmanager
    def pooled(self):
        with self._condition:
            while self._legacy_active:
                self._condition.wait()
            self._pooled_active += 1
        try:
            yield
        finally:
            with self._condition:
                self._pooled_active -= 1
                self._condition.notify_all()


def _verify_v9_training_reference(conditioning: dict[str, Any]) -> Path:
    """Verify both the pinned metadata and the reference WAV bytes on disk."""
    reference_path = Path(str(conditioning.get("reference_wav_path") or "")).expanduser()
    if not reference_path.is_file():
        raise ValueError("il reference voice training V9 non è disponibile")
    declared_sha256 = str(conditioning.get("reference_wav_sha256") or "").lower()
    digest = hashlib.sha256()
    try:
        with reference_path.open("rb") as reference_file:
            for block in iter(lambda: reference_file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError("impossibile leggere il reference voice training V9") from exc
    actual_sha256 = digest.hexdigest()
    if (
        declared_sha256 != V9_TRAINING_REFERENCE_SHA256
        or actual_sha256 != V9_TRAINING_REFERENCE_SHA256
    ):
        raise ValueError("lo SHA del reference voice non corrisponde al training V9")
    return reference_path
