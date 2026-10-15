"""Bounded CPU-only faster-whisper worker for real session audio archives."""
from __future__ import annotations

import importlib.util
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

DEFAULT_ARCHIVE_MODEL_PATH = Path(
    os.environ.get(
        "DUPLEX_ARCHIVE_ASR_MODEL_PATH",
        "/workspace/.hf_cache/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120",
    )
)


class CPUAudioArchiveWorker:
    """Serialize archival ASR on CPU and bound the in-flight PCM/work queue."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_ARCHIVE_MODEL_PATH,
        *,
        cpu_threads: int = 4,
        max_pending: int = 4,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        if cpu_threads <= 0 or max_pending <= 0:
            raise ValueError("cpu_threads and max_pending must be positive")
        self.model_path = Path(model_path)
        self.cpu_threads = int(cpu_threads)
        self.max_pending = int(max_pending)
        self._model_factory = model_factory
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(self.max_pending)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-archive-cpu")
        self._closed = False

    @property
    def available(self) -> bool:
        """Whether this worker can run with its local-only CPU model setup."""
        with self._model_lock:
            if self._closed or not self.model_path.is_dir():
                return False
            if self._model_factory is not None:
                return True
        try:
            return importlib.util.find_spec("faster_whisper") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def submit(self, pcm: np.ndarray, *, language: str | None = None) -> Future[dict[str, Any]] | None:
        """Queue copied mono float32 PCM; return None when bounded capacity is full."""
        audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if not audio.size or not np.isfinite(audio).all():
            raise ValueError("audio archive requires non-empty finite PCM")
        with self._model_lock:
            if self._closed:
                raise RuntimeError("audio archive worker is closed")
        if not self._slots.acquire(blocking=False):
            return None
        try:
            future = self._executor.submit(self._transcribe, audio.copy(), language)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _future: self._slots.release())
        return future

    def _get_model(self) -> Any:
        with self._model_lock:
            if self._model is not None:
                return self._model
            if self._closed:
                raise RuntimeError("audio archive worker is closed")
            if not self.model_path.is_dir():
                raise FileNotFoundError(f"cached faster-whisper snapshot missing: {self.model_path}")
            factory = self._model_factory
            if factory is None:
                from faster_whisper import WhisperModel

                factory = WhisperModel
            self._model = factory(
                str(self.model_path),
                device="cpu",
                compute_type="int8",
                cpu_threads=self.cpu_threads,
                num_workers=1,
                local_files_only=True,
            )
            return self._model

    def _transcribe(self, pcm: np.ndarray, language: str | None) -> dict[str, Any]:
        started = time.perf_counter()
        model = self._get_model()
        # faster-whisper expects a mono 16 kHz waveform for ndarray input.
        segments, info = model.transcribe(pcm, language=language, beam_size=5)
        # The API returns a lazy generator; exhaust it before committing text.
        text = "".join(str(segment.text) for segment in segments).strip()
        return {
            "text": text,
            "language": getattr(info, "language", None),
            "duration_s": getattr(info, "duration", None),
            "asr_ms": (time.perf_counter() - started) * 1000.0,
            "model": "faster-whisper-small",
            "device": "cpu",
            "compute_type": "int8",
            "cpu_threads": self.cpu_threads,
        }

    def shutdown(self, *, wait: bool = False, cancel_futures: bool = True) -> None:
        """Close this worker instance; a running ASR call may finish before exit."""
        with self._model_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)


# The server shares this single model/executor across all WebSocket sessions.
CPU_AUDIO_ARCHIVE_WORKER = CPUAudioArchiveWorker()
