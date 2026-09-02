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

from duplex_kernel.gemma_representation import (
    AUDIO_FEATURES_PER_FRAME,
    AUDIO_START_AFTER_TEXT,
    TEXT_BLOCK_LEN,
    build_audio_placeholder_inputs,
    interleave_duplex_mel,
)
from duplex_kernel.scheduler import DEFAULT_SAMPLE_RATE

logger = logging.getLogger("duplex_kernel.listener")

DEFAULT_FRAME_DURATION_MS = 80
DEFAULT_FRAME_SAMPLES = DEFAULT_SAMPLE_RATE * DEFAULT_FRAME_DURATION_MS // 1000
DEFAULT_REFERENCE_DELAY_FRAMES = 1
DEFAULT_CONTEXT_FRAMES = 64
MEL_PER_LISTENER_FRAME = 8
LISTENER_MEL_CHANNELS = 2
LISTENER_FEATURES_PER_FRAME = AUDIO_FEATURES_PER_FRAME
LISTENER_TEXT_BLOCK_LEN = TEXT_BLOCK_LEN
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
    text_token_ids: tuple[int, ...] = field(default_factory=tuple, repr=False)
    # Gemma language-model KV state.  The audio tower remains the native
    # encoder; only its final multimodal LM is advanced four soft tokens per
    # listener frame.  These fields are backend-owned but session-lifetime.
    gemma_lm_cache: Any | None = field(default=None, repr=False, compare=False)
    gemma_lm_shared_kv_states: Any | None = field(default=None, repr=False, compare=False)
    gemma_lm_audio_frames: int = 0
    gemma_lm_context_start_frame: int = 0
    gemma_lm_text_context: tuple[int, ...] = field(default_factory=tuple, repr=False)
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

    @property
    def next_input_frame_id(self) -> int:
        """ID that will be assigned to the next successful listener step."""
        with self._lock:
            return int(self.next_frame_id)

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

    def invalidate_gemma_lm_state(self) -> None:
        """Discard a partial LM transaction; history will be replayed next tick."""
        with self._lock:
            self.gemma_lm_cache = None
            self.gemma_lm_shared_kv_states = None
            self.gemma_lm_audio_frames = 0
            self.gemma_lm_context_start_frame = 0
            self.gemma_lm_text_context = ()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.user_pcm_history.clear()
            self.assistant_pcm_history.clear()
            self.gemma_lm_cache = None
            self.gemma_lm_shared_kv_states = None
            self.gemma_lm_audio_frames = 0
            self.gemma_lm_context_start_frame = 0
            self.gemma_lm_text_context = ()
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
            self.text_token_ids = ()
            self.gemma_lm_cache = None
            self.gemma_lm_shared_kv_states = None
            self.gemma_lm_audio_frames = 0
            self.gemma_lm_context_start_frame = 0
            self.gemma_lm_text_context = ()
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


@dataclass(frozen=True)
class ListenerBackendOutput:
    """Both native audio soft tokens and the canonical multimodal LM hidden."""

    gemma_hidden_t: Any
    audio_tower_hidden: Any | None = None


class GemmaAudioTowerBackend:
    """Gemma multimodal backend used by the continuous listener.

    The V9-facing value is *not* ``pooler_output``.  For a real Thinker it is
    the final hidden state of the multimodal language model at the first audio
    feature of the current 80 ms frame, before V9's causal convolution.  The
    direct audio-tower path remains only as a compatibility fallback for the
    small deterministic test backend.
    """

    def __init__(
        self,
        thinker: Any,
        *,
        processor: Any | None = None,
        device: str | None = None,
        text_token_ids: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        wrapper = thinker
        model = getattr(wrapper, "model", wrapper)
        gemma_model = getattr(model, "model", None)
        if gemma_model is None or not callable(getattr(gemma_model, "get_audio_features", None)):
            raise TypeError("thinker non espone Gemma4Model.get_audio_features")
        self.model = model
        self.multimodal_model = model if callable(model) else None
        self.gemma_model = gemma_model
        self.processor = processor or getattr(wrapper, "processor", None)
        self.feature_extractor = getattr(self.processor, "feature_extractor", self.processor)
        if not callable(self.feature_extractor):
            raise TypeError("processor Gemma audio non disponibile")
        self._lock = threading.Lock()
        self.text_token_ids = tuple(int(value) for value in (text_token_ids or ()))
        config = getattr(gemma_model, "config", None)
        text_config = getattr(config, "text_config", None)
        self.hidden_size = int(getattr(text_config, "hidden_size", 1536))
        self.audio_token_id = int(getattr(config, "audio_token_id", 258881))
        self.boa_token_id = int(getattr(config, "boa_token_id", 256000))
        self.eoa_token_id = int(getattr(config, "eoa_token_id", 258883))
        self.bos_token_id = int(getattr(config, "bos_token_id", 2))
        self.pad_token_id = int(getattr(text_config, "pad_token_id", 0))
        language_model = getattr(self.gemma_model, "language_model", None)
        self._lm_has_shared_kv = any(
            bool(getattr(getattr(layer, "self_attn", None), "is_kv_shared_layer", False))
            for layer in getattr(language_model, "layers", ())
        )
        self._shape_stable_lm = None
        self._frame_pad_ple: Any | None = None
        if language_model is not None:
            from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM

            # This is a zero-weight-copy functional view.  Its cache/workspace
            # is listener-owned; the normal Thinker keeps its native forward.
            self._shape_stable_lm = ShapeStableGemmaListenerLM(language_model, target_length=291)
        try:
            import torch
            parameter = next(self.gemma_model.audio_tower.parameters())
            self.device = torch.device(device or parameter.device)
            self.dtype = parameter.dtype
        except (ImportError, AttributeError, StopIteration) as exc:
            raise TypeError("audio tower Gemma non disponibile") from exc

    def _extract(self, waves: list[np.ndarray], *, rows: int | None = None) -> list[np.ndarray]:
        """Extract the same 100 fps mel convention used by the cache builder."""
        kwargs = {
            "padding": False,
            "truncation": False,
            "max_length": None,
            "pad_to_multiple_of": None,
            "return_tensors": "np",
        }
        prepared = []
        for wave in waves:
            # With the semicausal 160-sample left pad, one sentinel sample
            # makes an N*1280-sample history yield exactly N*8 mel rows.
            prepared.append(np.concatenate((np.asarray(wave, dtype=np.float32), np.zeros(1, dtype=np.float32))))
        try:
            features = self.feature_extractor(prepared, **kwargs)
        except TypeError:
            # Keeps compatibility with the same official extractor when an
            # older processor omits one of the optional keyword arguments.
            features = self.feature_extractor(prepared, padding=False)
        values = features["input_features"]
        result: list[np.ndarray] = []
        for value in values:
            array = np.asarray(value, dtype=np.float32)
            if array.ndim != 2 or array.shape[-1] != 128:
                raise RuntimeError(f"feature Gemma inattesa: {array.shape}")
            if rows is not None:
                array = self._pad_rows(array, rows)
            result.append(array)
        return result

    @staticmethod
    def _pad_rows(array: np.ndarray, rows: int) -> np.ndarray:
        if len(array) == rows:
            return array
        if len(array) > rows:
            return array[:rows]
        return np.pad(array, ((0, rows - len(array)), (0, 0)), mode="constant")

    def _build_multimodal_inputs(self, text_ids: tuple[int, ...], audio_feature_rows: int, *, device: Any):
        return build_audio_placeholder_inputs(
            batch_size=1,
            audio_feature_rows=audio_feature_rows,
            device=device,
            audio_token_id=self.audio_token_id,
            bos_token_id=self.bos_token_id,
            boa_token_id=self.boa_token_id,
            eoa_token_id=self.eoa_token_id,
            pad_token_id=self.pad_token_id,
            text_token_ids=text_ids,
        )

    def _forward_incremental_lm(
        self,
        *,
        embeds: Any,
        per_layer: Any,
        attention: Any,
        positions: Any,
        state: GemmaListenerState,
    ) -> Any:
        """Run one shape-stable listener-only Gemma LM chunk and commit its KV."""
        import torch

        if self._shape_stable_lm is None:
            raise RuntimeError("runtime LM shape-stable non disponibile")
        from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

        position_ids = positions.view(1, -1)
        # The native Gemma mask factory remains the source of truth.  The
        # listener view consumes it without changing the Thinker modules.
        mask_mapping = create_masks_for_generate(
            self.gemma_model.config,
            embeds,
            attention,
            past_key_values=state.gemma_lm_cache,
            position_ids=position_ids,
        )
        with torch.inference_mode():
            output = self._shape_stable_lm.forward(
                inputs_embeds=embeds,
                per_layer_inputs=per_layer,
                attention_mask=mask_mapping,
                position_ids=position_ids,
                past_key_values=state.gemma_lm_cache,
                shared_kv_states=state.gemma_lm_shared_kv_states,
            )
        state.gemma_lm_cache = output.past_key_values
        state.gemma_lm_shared_kv_states = getattr(output.past_key_values, "shared_layers", None)
        if self._lm_has_shared_kv and state.gemma_lm_shared_kv_states is None:
            raise RuntimeError("Gemma listener non ha costruito lo shared-KV session-owned")
        return output.last_hidden_state

    def _frame_pad_per_layer_inputs(self, language_model: Any) -> Any:
        """Return immutable PLE for the four PAD identities used by every tick."""
        if self._frame_pad_ple is None:
            import torch

            pad_ids = torch.full(
                (1, AUDIO_FEATURES_PER_FRAME), self.pad_token_id,
                dtype=torch.long, device=self.device,
            )
            with torch.inference_mode():
                pad_embeds = language_model.embed_tokens(pad_ids)
                self._frame_pad_ple = language_model.get_per_layer_inputs(pad_ids, pad_embeds).detach()
        return self._frame_pad_ple

    def _run_incremental_lm(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        audio_soft: Any,
        state: GemmaListenerState,
        text_ids: tuple[int, ...],
    ) -> Any:
        """Advance Gemma's final multimodal LM by one four-token frame.

        Gemma's native audio tower is causal in time (right context zero), but
        its output is a complete fixed context.  The expensive LM stage can
        therefore consume only the newly available four audio soft tokens;
        its DynamicCache and shared-KV side state live in the listener state.
        A context rollover replays the bounded window frame-by-frame, keeping
        the same numerical state machine as the offline representation path.
        """
        import torch

        lm = self.gemma_model.language_model
        lm_dtype = lm.embed_tokens.weight.dtype
        frame_index = len(state.user_pcm_history) - 1
        context_start = state.next_frame_id - frame_index
        context_changed = (
            state.gemma_lm_cache is None
            or state.gemma_lm_text_context != text_ids
            or state.gemma_lm_audio_frames != state.next_frame_id
        )
        stable_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
        stable_attention = attention_mask[:, :AUDIO_START_AFTER_TEXT].to(
            device=self.device, dtype=torch.long
        )
        stable_embeds = lm.embed_tokens(stable_ids)
        stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)
        frame_rows = AUDIO_FEATURES_PER_FRAME

        if context_changed:
            # Recreate the prefix, then replay every audio frame as a distinct
            # Q=4 call.  Joining prefix+first-frame changes the BF16 GEMM shape
            # and is therefore not representation-equivalent.
            state.gemma_lm_cache = None
            state.gemma_lm_shared_kv_states = None
            state.gemma_lm_audio_frames = 0
            self._forward_incremental_lm(
                embeds=stable_embeds,
                per_layer=stable_ple,
                attention=stable_attention,
                positions=torch.arange(AUDIO_START_AFTER_TEXT, device=self.device, dtype=torch.long),
                state=state,
            )
            frame_ple = self._frame_pad_per_layer_inputs(lm)
            for cached_frame in range(frame_index + 1):
                start = cached_frame * frame_rows
                frame_audio = audio_soft[:, start:start + frame_rows, :].to(dtype=lm_dtype)
                start_position = AUDIO_START_AFTER_TEXT + start
                positions = torch.arange(
                    start_position, start_position + frame_rows,
                    device=self.device, dtype=torch.long,
                )
                frame_attention = torch.ones(
                    (1, start_position + frame_rows), device=self.device, dtype=torch.long
                )
                frame_attention[:, :AUDIO_START_AFTER_TEXT] = stable_attention
                hidden = self._forward_incremental_lm(
                    embeds=frame_audio,
                    per_layer=frame_ple,
                    attention=frame_attention,
                    positions=positions,
                    state=state,
                )
                current_hidden = hidden[:, 0, :]
                state.gemma_lm_audio_frames = cached_frame + 1
            state.gemma_lm_context_start_frame = context_start
            state.gemma_lm_text_context = text_ids
            return current_hidden

        start = frame_index * frame_rows
        frame_audio = audio_soft[:, start:start + frame_rows, :].to(dtype=lm_dtype)
        frame_ple = self._frame_pad_per_layer_inputs(lm)
        start_position = AUDIO_START_AFTER_TEXT + state.gemma_lm_audio_frames * frame_rows
        positions = torch.arange(
            start_position, start_position + frame_rows,
            device=self.device, dtype=torch.long,
        )
        attention = torch.ones(
            (1, start_position + frame_rows), device=self.device, dtype=torch.long
        )
        attention[:, :AUDIO_START_AFTER_TEXT] = stable_attention
        hidden = self._forward_incremental_lm(
            embeds=frame_audio,
            per_layer=frame_ple,
            attention=attention,
            positions=positions,
            state=state,
        )
        state.gemma_lm_audio_frames += 1
        state.gemma_lm_context_start_frame = context_start
        state.gemma_lm_text_context = text_ids
        return hidden[:, 0, :]

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
            canonical = self.multimodal_model is not None
            if canonical and state.context_frames > DEFAULT_CONTEXT_FRAMES:
                raise ValueError(
                    f"contesto listener Gemma oltre il massimo attestato di {DEFAULT_CONTEXT_FRAMES} frame"
                )
            target_rows = state.context_frames * MEL_PER_LISTENER_FRAME if canonical else None
            user_mel, assistant_mel = self._extract([user_wave, assistant_wave], rows=target_rows)
            rows = target_rows or max(len(user_mel), len(assistant_mel))
            if rows <= 0:
                raise RuntimeError("Gemma audio tower non ha prodotto feature")
            user_mel = self._pad_rows(user_mel, rows)
            assistant_mel = self._pad_rows(assistant_mel, rows)
            import torch
            if canonical:
                # Keep the exact frame-block order used by the offline dataset:
                # [u[t, 0:8], a[t, 0:8]] for each 80-ms frame.  Row-wise
                # interleaving would change Gemma's 2-D subsampler receptive field.
                interleaved = interleave_duplex_mel(user_mel, assistant_mel)
            else:
                # The deterministic fixture backend is intentionally allowed
                # to use short arbitrary feature rows; it never enters the
                # production multimodal path.
                interleaved = np.stack((user_mel, assistant_mel), axis=1).reshape(rows * 2, 128)
            features = torch.from_numpy(interleaved).unsqueeze(0).to(self.device, dtype=self.dtype)
            feature_mask = torch.ones((1, interleaved.shape[0]), dtype=torch.bool, device=self.device)

            if not canonical:
                # Compatibility-only path for the deterministic P3 fixture. A
                # real Thinker can never enter this branch.
                with torch.inference_mode():
                    output = self.gemma_model.get_audio_features(
                        input_features=features,
                        input_features_mask=feature_mask,
                        return_dict=True,
                    )
                pooled = getattr(output, "pooler_output", None)
                valid_mask = getattr(output, "attention_mask", None)
                if pooled is None or valid_mask is None:
                    raise RuntimeError("Gemma audio tower non ha restituito pooler_output/mask")
                valid = pooled[0][valid_mask[0].to(pooled.device)]
                if valid.numel() == 0:
                    raise RuntimeError("Gemma audio tower ha restituito hidden vuoto")
                hidden = valid[-1:].detach()
                return ListenerBackendOutput(gemma_hidden_t=hidden, audio_tower_hidden=hidden)

            text_ids = tuple(getattr(state, "text_token_ids", ())) or self.text_token_ids
            ids, attention = self._build_multimodal_inputs(text_ids, interleaved.shape[0], device=self.device)
            # Run the native audio tower once, then advance only the final LM
            # by the newly available four audio soft tokens.  The LM helper
            # owns the DynamicCache/shared-KV state and returns the first token
            # of the current 80-ms frame.
            with torch.inference_mode():
                audio_output = self.gemma_model.get_audio_features(
                    input_features=features,
                    input_features_mask=feature_mask,
                    return_dict=True,
                )
            pooled = getattr(audio_output, "pooler_output", None)
            valid_mask = getattr(audio_output, "attention_mask", None)
            if pooled is None or valid_mask is None:
                raise RuntimeError("Gemma audio tower non ha restituito pooler_output/mask")
            valid = pooled[0][valid_mask[0].to(pooled.device)]
            if valid.numel() == 0 or valid.shape[-1] != self.hidden_size:
                raise RuntimeError("Gemma audio tower ha restituito hidden vuoto/incompatibile")
            audio_soft = valid.unsqueeze(0).to(device=self.device, dtype=self.dtype)
            hidden = self._run_incremental_lm(
                input_ids=ids,
                attention_mask=attention,
                audio_soft=audio_soft,
                state=state,
                text_ids=text_ids,
            ).detach()
            if hidden.shape[-1] != self.hidden_size:
                raise RuntimeError(f"hidden Gemma multimodale inatteso: {tuple(hidden.shape)}")
            if not torch.isfinite(hidden).all():
                raise RuntimeError("hidden Gemma multimodale non finito")
            return ListenerBackendOutput(
                gemma_hidden_t=hidden,
                audio_tower_hidden=audio_soft[0].detach(),
            )


@dataclass(frozen=True)
class ListenerOutput:
    """Result of one committed listener tick.

    ``gemma_hidden_t`` is the only canonical V9 input. ``hidden_t`` remains a
    read-only compatibility alias for older control-plane tests.
    """

    gemma_hidden_t: Any
    audio_timestamp: int
    listener_state_version: int
    finite: bool
    frame_id: int
    reference_frame_id: int | None = None
    audio_tower_hidden: Any | None = None

    @property
    def hidden_t(self) -> Any:
        return self.gemma_hidden_t


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
        text_token_ids: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        if backend is None:
            if thinker is None:
                raise ValueError("GemmaListener richiede backend o thinker")
            backend = GemmaAudioTowerBackend(thinker, text_token_ids=text_token_ids)
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
        self.text_token_ids = tuple(int(value) for value in (text_token_ids or ()))

    @classmethod
    def from_thinker(cls, thinker: Any, **kwargs: Any) -> "GemmaListener":
        return cls(thinker=thinker, **kwargs)

    def set_text_context(self, state: GemmaListenerState, token_ids: list[int] | tuple[int, ...]) -> None:
        """Set optional causal text context for the next listener frames."""
        if not isinstance(state, GemmaListenerState):
            raise TypeError("state deve essere GemmaListenerState")
        with state._lock:
            state.text_token_ids = tuple(int(value) for value in token_ids)

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
                encoded = self.backend.encode(user, reference, state)
                if isinstance(encoded, ListenerBackendOutput):
                    hidden = encoded.gemma_hidden_t
                    audio_tower_hidden = encoded.audio_tower_hidden
                else:
                    hidden = encoded
                    audio_tower_hidden = None
            except BaseException:
                state.user_pcm_history.pop()
                state.assistant_pcm_history.pop()
                # The functional LM may have appended part of the failing
                # chunk before an exception.  Never reuse that partial cache:
                # the next valid tick will deterministically replay history.
                state.invalidate_gemma_lm_state()
                state.dropped_listener_frames += 1
                raise
            finite, nonempty = self._hidden_health(hidden)
            if not nonempty:
                state.empty_hidden_count += 1
                state.user_pcm_history.pop()
                state.assistant_pcm_history.pop()
                state.invalidate_gemma_lm_state()
                state.dropped_listener_frames += 1
                raise RuntimeError("listener hidden vuoto")
            if not finite:
                state.nonfinite_hidden_count += 1
                state.user_pcm_history.pop()
                state.assistant_pcm_history.pop()
                state.invalidate_gemma_lm_state()
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
                gemma_hidden_t=hidden,
                audio_timestamp=state.last_audio_timestamp,
                listener_state_version=state.listener_state_version,
                finite=finite,
                frame_id=frame_id,
                reference_frame_id=state.last_reference_frame_id,
                audio_tower_hidden=audio_tower_hidden,
            )

    def stats(self, state: GemmaListenerState) -> dict[str, int | None]:
        return state.stats()


__all__ = [
    "AssistantReferenceBuffer",
    "AssistantReferenceFrame",
    "GemmaAudioTowerBackend",
    "ListenerBackendOutput",
    "GemmaListener",
    "GemmaListenerState",
    "ListenerBackend",
    "ListenerOutput",
]
