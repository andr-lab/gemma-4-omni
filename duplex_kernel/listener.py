"""Continuous Gemma audio listener and causal assistant self-reference.

P3 deliberately stops at the native Gemma audio tower.  It does not run ASR,
text generation, V9, or the Qwen speaker.  The scheduler supplies one complete
16 kHz/80 ms frame at a time; the listener keeps a bounded causal context and
returns the last valid Gemma audio embedding for that context.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from duplex_kernel.scheduler import DEFAULT_SAMPLE_RATE

logger = logging.getLogger("duplex_kernel.listener")

DEFAULT_FRAME_DURATION_MS = 80
DEFAULT_FRAME_SAMPLES = DEFAULT_SAMPLE_RATE * DEFAULT_FRAME_DURATION_MS // 1000
DEFAULT_REFERENCE_DELAY_FRAMES = 1
DEFAULT_CONTEXT_FRAMES = 16
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

    ``append`` is called only after an audio frame was sent to the client.  A
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
            # Exact lookup is intentional: returning the latest available frame
            # would turn a transport gap into a temporal reference leak.
            frame = self._frames.get(target)
            return frame

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


@dataclass
class GemmaListenerState:
    """Mutable, session-owned state for the continuous listener."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS
    context_frames: int = DEFAULT_CONTEXT_FRAMES
    next_frame_id: int = 0
    total_samples: int = 0
    listener_state_version: int = 0
    listener_input_frames: int = 0
    dropped_listener_frames: int = 0
    nonfinite_hidden_count: int = 0
    empty_hidden_count: int = 0
    future_reference_leaks: int = 0
    wrong_reference_frames: int = 0
    last_audio_timestamp: int = 0
    last_reference_frame_id: int | None = None
    last_hidden: Any | None = field(default=None, repr=False)
    user_pcm_history: deque[np.ndarray] = field(default_factory=deque, repr=False)
    assistant_pcm_history: deque[np.ndarray] = field(default_factory=deque, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    _closed: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.frame_duration_ms <= 0 or self.context_frames <= 0:
            raise ValueError("configurazione GemmaListenerState non valida")
        frame_samples = self.sample_rate * self.frame_duration_ms
        if frame_samples % 1000:
            raise ValueError("frame_duration_ms deve produrre un numero intero di sample")
        self.user_pcm_history = deque(self.user_pcm_history, maxlen=int(self.context_frames))
        self.assistant_pcm_history = deque(self.assistant_pcm_history, maxlen=int(self.context_frames))

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_duration_ms // 1000

    @property
    def processed_frames(self) -> int:
        return self.listener_input_frames

    def stats(self) -> dict[str, int | None]:
        with self._lock:
            return {
                "listener_input_frames": self.listener_input_frames,
                "dropped_listener_frames": self.dropped_listener_frames,
                "samples_received": self.total_samples,
                "samples_accounted_for": self.listener_input_frames * self.frame_samples,
                "listener_state_version": self.listener_state_version,
                "nonfinite_hidden_count": self.nonfinite_hidden_count,
                "empty_hidden_count": self.empty_hidden_count,
                "future_reference_leaks": self.future_reference_leaks,
                "wrong_reference_frames": self.wrong_reference_frames,
                "last_audio_timestamp": self.last_audio_timestamp,
                "last_reference_frame_id": self.last_reference_frame_id,
            }

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.user_pcm_history.clear()
            self.assistant_pcm_history.clear()
            self.last_hidden = None

    def reset(self) -> None:
        with self._lock:
            self.next_frame_id = 0
            self.total_samples = 0
            self.listener_state_version = 0
            self.listener_input_frames = 0
            self.dropped_listener_frames = 0
            self.nonfinite_hidden_count = 0
            self.empty_hidden_count = 0
            self.future_reference_leaks = 0
            self.wrong_reference_frames = 0
            self.last_audio_timestamp = 0
            self.last_reference_frame_id = None
            self.last_hidden = None
            self.user_pcm_history.clear()
            self.assistant_pcm_history.clear()
            self._closed = False


class ListenerBackend(Protocol):
    """Backend boundary; unit tests may inject a deterministic implementation."""

    def encode(
        self,
        user_pcm: np.ndarray,
        assistant_reference: AssistantReferenceFrame | None,
        state: GemmaListenerState,
    ) -> Any:
        ...


class GemmaAudioTowerBackend:
    """Native Transformers 5 Gemma audio-tower backend.

    It extracts a causal rolling dual-channel mel stream ``[user, assistant]``
    and calls ``Gemma4Model.get_audio_features`` directly.  No tokenizer, LLM
    logits, ASR decode, or text generation is involved.
    """

    def __init__(self, thinker: Any, *, processor: Any | None = None, device: str | None = None) -> None:
        wrapper = thinker
        model = getattr(wrapper, "model", wrapper)
        gemma_model = getattr(model, "model", None)
        if gemma_model is None or not callable(getattr(gemma_model, "get_audio_features", None)):
            raise TypeError("thinker non espone Gemma4Model.get_audio_features")
        self.model = model
        self.gemma_model = gemma_model
        self.processor = processor or getattr(wrapper, "processor", None)
        self.feature_extractor = getattr(self.processor, "feature_extractor", self.processor)
        if not callable(self.feature_extractor):
            raise TypeError("processor Gemma audio non disponibile")
        self._lock = threading.Lock()
        config = getattr(gemma_model, "config", None)
        text_config = getattr(config, "text_config", None)
        self.hidden_size = int(getattr(text_config, "hidden_size", 1536))
        try:
            import torch
            parameter = next(self.gemma_model.audio_tower.parameters())
            self.device = torch.device(device or parameter.device)
            self.dtype = parameter.dtype
        except (ImportError, AttributeError, StopIteration) as exc:
            raise TypeError("audio tower Gemma non disponibile") from exc

    def _extract(self, waves: list[np.ndarray]) -> list[np.ndarray]:
        kwargs = {
            "padding": False,
            "truncation": False,
            "max_length": None,
            "pad_to_multiple_of": None,
            "return_tensors": "np",
        }
        try:
            features = self.feature_extractor(waves, **kwargs)
        except TypeError:
            # Keeps compatibility with the same official extractor when an
            # older processor omits one of the optional keyword arguments.
            features = self.feature_extractor(waves, padding=False)
        values = features["input_features"]
        masks = features.get("input_features_mask")
        result: list[np.ndarray] = []
        for index, value in enumerate(values):
            array = np.asarray(value, dtype=np.float32)
            if masks is not None:
                mask = np.asarray(masks[index], dtype=bool)
                array = array[mask]
            if array.ndim != 2 or array.shape[-1] != 128:
                raise RuntimeError(f"feature Gemma inattesa: {array.shape}")
            result.append(array)
        return result

    @staticmethod
    def _pad_rows(array: np.ndarray, rows: int) -> np.ndarray:
        if len(array) == rows:
            return array
        if len(array) > rows:
            return array[:rows]
        return np.pad(array, ((0, rows - len(array)), (0, 0)), mode="constant")

    def encode(
        self,
        user_pcm: np.ndarray,
        assistant_reference: AssistantReferenceFrame | None,
        state: GemmaListenerState,
    ) -> Any:
        del user_pcm, assistant_reference
        with self._lock:
            user_wave = np.concatenate(tuple(state.user_pcm_history), axis=0)
            assistant_wave = np.concatenate(tuple(state.assistant_pcm_history), axis=0)
            user_mel, assistant_mel = self._extract([user_wave, assistant_wave])
            rows = max(len(user_mel), len(assistant_mel))
            if rows <= 0:
                raise RuntimeError("Gemma audio tower non ha prodotto feature")
            user_mel = self._pad_rows(user_mel, rows)
            assistant_mel = self._pad_rows(assistant_mel, rows)
            interleaved = np.stack((user_mel, assistant_mel), axis=1).reshape(rows * 2, 128)
            import torch
            features = torch.from_numpy(interleaved).unsqueeze(0).to(self.device, dtype=self.dtype)
            mask = torch.ones((1, interleaved.shape[0]), dtype=torch.bool, device=self.device)
            with torch.inference_mode():
                output = self.gemma_model.get_audio_features(
                    input_features=features,
                    input_features_mask=mask,
                    return_dict=True,
                )
            pooled = getattr(output, "pooler_output", None)
            valid_mask = getattr(output, "attention_mask", None)
            if pooled is None or valid_mask is None:
                raise RuntimeError("Gemma audio tower non ha restituito pooler_output/mask")
            valid = pooled[0][valid_mask[0].to(pooled.device)]
            if valid.numel() == 0:
                raise RuntimeError("Gemma audio tower ha restituito hidden vuoto")
            return valid[-1:].detach()


@dataclass(frozen=True)
class ListenerOutput:
    """Result of one committed listener tick."""

    hidden_t: Any
    audio_timestamp: int
    listener_state_version: int
    finite: bool
    frame_id: int
    reference_frame_id: int | None = None


class GemmaListener:
    """Frame-locked causal listener with a replaceable model backend."""

    def __init__(
        self,
        backend: ListenerBackend | None = None,
        *,
        thinker: Any | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS,
        reference_delay_frames: int = DEFAULT_REFERENCE_DELAY_FRAMES,
    ) -> None:
        if backend is None:
            if thinker is None:
                raise ValueError("GemmaListener richiede backend o thinker")
            backend = GemmaAudioTowerBackend(thinker)
        if sample_rate <= 0 or frame_duration_ms <= 0:
            raise ValueError("sample_rate e frame_duration_ms devono essere positivi")
        frame_samples = sample_rate * frame_duration_ms
        if frame_samples % 1000:
            raise ValueError("frame_duration_ms deve produrre un numero intero di sample")
        if reference_delay_frames < 0:
            raise ValueError("reference_delay_frames non può essere negativo")
        self.backend = backend
        self.sample_rate = int(sample_rate)
        self.frame_duration_ms = int(frame_duration_ms)
        self.frame_samples = frame_samples // 1000
        self.reference_delay_frames = int(reference_delay_frames)

    @classmethod
    def from_thinker(cls, thinker: Any, **kwargs: Any) -> "GemmaListener":
        return cls(thinker=thinker, **kwargs)

    @staticmethod
    def _validate_user_pcm(pcm: np.ndarray, expected_samples: int) -> np.ndarray:
        array = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if array.size != expected_samples:
            raise ValueError(f"un listener frame deve contenere esattamente {expected_samples} sample")
        if not np.isfinite(array).all():
            raise ValueError("listener PCM deve contenere solo sample finite")
        return np.array(array, dtype=np.float32, copy=True)

    def _coerce_reference(
        self,
        reference: AssistantReferenceFrame | np.ndarray | None,
        *,
        frame_id: int,
    ) -> AssistantReferenceFrame | None:
        allowed = frame_id - self.reference_delay_frames
        if reference is None:
            return None
        if isinstance(reference, AssistantReferenceFrame):
            source_id = reference.frame_id
            if source_id > allowed:
                raise ValueError(
                    f"assistant reference futura: frame {source_id} visibile al listener {frame_id} "
                    f"(massimo causale {allowed})"
                )
            if source_id != allowed:
                raise ValueError(
                    f"assistant reference non allineata: frame {source_id}, atteso {allowed}"
                )
            if reference.sample_rate != self.sample_rate or reference.pcm.size != self.frame_samples:
                raise ValueError("assistant reference non normalizzata al frame listener")
            return reference
        array = np.asarray(reference, dtype=np.float32).reshape(-1)
        if array.size != self.frame_samples:
            raise ValueError(f"assistant reference deve contenere esattamente {self.frame_samples} sample")
        if not np.isfinite(array).all():
            raise ValueError("assistant reference deve contenere solo sample finite")
        if allowed < 0:
            raise ValueError("assistant reference fornita prima del primo frame causale")
        return AssistantReferenceFrame(
            frame_id=allowed,
            pcm=array,
            emitted_at_ns=(allowed + 1) * self.frame_duration_ms * 1_000_000,
            sample_rate=self.sample_rate,
        )

    @staticmethod
    def _hidden_health(hidden: Any) -> tuple[bool, bool]:
        try:
            numel = int(hidden.numel()) if hasattr(hidden, "numel") else int(np.asarray(hidden).size)
        except Exception:
            numel = 0
        if numel <= 0:
            return False, False
        try:
            if hasattr(hidden, "is_floating_point"):
                import torch
                finite = bool(torch.isfinite(hidden).all().item())
            else:
                finite = bool(np.isfinite(np.asarray(hidden)).all())
        except Exception:
            finite = False
        return finite, True

    def step(
        self,
        user_pcm_t: np.ndarray,
        assistant_reference_t: AssistantReferenceFrame | np.ndarray | None,
        state: GemmaListenerState,
    ) -> ListenerOutput:
        """Consume exactly one 80 ms user frame and return its causal hidden."""
        if not isinstance(state, GemmaListenerState):
            raise TypeError("state deve essere GemmaListenerState")
        with state._lock:
            if state._closed:
                raise RuntimeError("GemmaListenerState chiuso")
            if state.sample_rate != self.sample_rate or state.frame_duration_ms != self.frame_duration_ms:
                raise ValueError("state e listener hanno sample-rate/frame-duration diversi")
            frame_id = state.next_frame_id
            user = self._validate_user_pcm(user_pcm_t, self.frame_samples)
            try:
                reference = self._coerce_reference(assistant_reference_t, frame_id=frame_id)
            except ValueError as exc:
                message = str(exc)
                if "futura" in message or "causale" in message and "prima" not in message:
                    state.future_reference_leaks += 1
                else:
                    state.wrong_reference_frames += 1
                raise

            # Commit the causal input context before calling the backend.  The
            # state is locked, so the backend sees a coherent aligned pair.
            state.user_pcm_history.append(user)
            state.assistant_pcm_history.append(
                np.zeros(self.frame_samples, dtype=np.float32)
                if reference is None else np.array(reference.pcm, dtype=np.float32, copy=True)
            )
            try:
                hidden = self.backend.encode(user, reference, state)
            except BaseException:
                state.user_pcm_history.pop()
                state.assistant_pcm_history.pop()
                state.dropped_listener_frames += 1
                raise
            finite, nonempty = self._hidden_health(hidden)
            if not nonempty:
                state.empty_hidden_count += 1
                state.user_pcm_history.pop()
                state.assistant_pcm_history.pop()
                state.dropped_listener_frames += 1
                raise RuntimeError("listener hidden vuoto")
            if not finite:
                state.nonfinite_hidden_count += 1
                state.user_pcm_history.pop()
                state.assistant_pcm_history.pop()
                state.dropped_listener_frames += 1
                raise RuntimeError("listener hidden non finito")

            state.listener_input_frames += 1
            state.total_samples += self.frame_samples
            state.next_frame_id += 1
            state.listener_state_version += 1
            state.last_audio_timestamp = state.next_frame_id * self.frame_duration_ms * 1_000_000
            state.last_reference_frame_id = None if reference is None else reference.frame_id
            if hasattr(hidden, "detach"):
                state.last_hidden = hidden.detach().clone()
            else:
                state.last_hidden = np.array(hidden, copy=True)
            return ListenerOutput(
                hidden_t=hidden,
                audio_timestamp=state.last_audio_timestamp,
                listener_state_version=state.listener_state_version,
                finite=finite,
                frame_id=frame_id,
                reference_frame_id=state.last_reference_frame_id,
            )

    def stats(self, state: GemmaListenerState) -> dict[str, int | None]:
        return state.stats()


__all__ = [
    "AssistantReferenceBuffer",
    "AssistantReferenceFrame",
    "GemmaAudioTowerBackend",
    "GemmaListener",
    "GemmaListenerState",
    "ListenerBackend",
    "ListenerOutput",
]
