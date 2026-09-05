"""Continuous Gemma audio listener and causal assistant self-reference.

P3 deliberately stops at the native Gemma audio tower.  It does not run ASR,
text generation, V9, or the Qwen speaker.  The scheduler supplies one complete
16 kHz/80 ms frame at a time; the listener keeps a bounded causal context and
returns the last valid Gemma audio embedding for that context.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from duplex_kernel.gemma_representation import (
    AUDIO_FEATURES_PER_FRAME,
    AUDIO_START_AFTER_TEXT,
    CANONICAL_LM_PREFILL_TARGET_LEN,
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
NATIVE_LM_MAX_SEQ_LEN = 2048
NATIVE_LM_PREFIX_LEN = AUDIO_START_AFTER_TEXT
NATIVE_LM_FRAME_CAPACITY = (NATIVE_LM_MAX_SEQ_LEN - NATIVE_LM_PREFIX_LEN) // AUDIO_FEATURES_PER_FRAME
NANOSECONDS = 1_000_000_000


from duplex_kernel.reference_buffer import (
    AssistantReferenceBuffer,
    AssistantReferenceFrame,
)



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
    gemma_lm_native: Any | None = field(default=None, repr=False, compare=False)
    gemma_lm_audio_frames: int = 0
    # Start of the rolling audio-feature window; never used to reset LM KV.
    gemma_lm_context_start_frame: int = 0
    gemma_lm_text_context: tuple[int, ...] = field(default_factory=tuple, repr=False)
    gemma_lm_terminal_error: str | None = field(default=None, repr=False, compare=False)
    native_last_hidden: Any | None = field(default=None, repr=False, compare=False)
    user_pcm_history: deque[np.ndarray] = field(default_factory=deque, repr=False)
    assistant_pcm_history: deque[np.ndarray] = field(default_factory=deque, repr=False)
    user_mel_history: deque[np.ndarray] = field(default_factory=deque, repr=False)
    assistant_mel_history: deque[np.ndarray] = field(default_factory=deque, repr=False)
    user_pcm_tail: np.ndarray | None = field(default=None, repr=False)
    assistant_pcm_tail: np.ndarray | None = field(default=None, repr=False)
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
        self.user_mel_history = deque(self.user_mel_history, maxlen=int(self.context_frames))
        self.assistant_mel_history = deque(self.assistant_mel_history, maxlen=int(self.context_frames))

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
            native = self.gemma_lm_native
            native_reusable = native is not None
            if native is not None:
                try:
                    native.reset()
                except BaseException:
                    # A failed stream drain means the CUDA context is not safe
                    # to reuse. Close it best-effort, then preserve the
                    # original backend exception while leaving state replayable.
                    native_reusable = False
                    logger.exception("native Gemma reset fallito; contesto scartato")
                    try:
                        native.close()
                    except BaseException:
                        logger.exception("chiusura contesto nativo fallita")
            self.gemma_lm_native = native if native_reusable else None
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
            self.user_mel_history.clear()
            self.assistant_mel_history.clear()
            self.user_pcm_tail = None
            self.assistant_pcm_tail = None
            native = self.gemma_lm_native
            try:
                if native is not None:
                    native.close()
            finally:
                self.gemma_lm_cache = None
                self.gemma_lm_shared_kv_states = None
                self.gemma_lm_native = None
                self.gemma_lm_terminal_error = None
                self.native_last_hidden = None
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
            native = self.gemma_lm_native
            try:
                if native is not None:
                    native.close()
            finally:
                self.gemma_lm_cache = None
                self.gemma_lm_shared_kv_states = None
                self.gemma_lm_native = None
                self.gemma_lm_terminal_error = None
                self.native_last_hidden = None
            self.gemma_lm_audio_frames = 0
            self.gemma_lm_context_start_frame = 0
            self.gemma_lm_text_context = ()
            self.last_hidden = None
            self.user_pcm_history.clear()
            self.assistant_pcm_history.clear()
            self.user_mel_history.clear()
            self.assistant_mel_history.clear()
            self.user_pcm_tail = None
            self.assistant_pcm_tail = None
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
        use_native_lm: bool | None = None,
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
        self._lock = threading.RLock()
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
        self._use_native_lm = (
            bool(use_native_lm)
            if use_native_lm is not None
            else os.environ.get("GEMMA4_NATIVE_LISTENER", "0") == "1"
        )
        self._frame_pad_ple: Any | None = None
        if language_model is not None:
            from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM

            # This is a zero-weight-copy functional view.  Its cache/workspace
            # is listener-owned; the normal Thinker keeps its native forward.
            self._shape_stable_lm = ShapeStableGemmaListenerLM(
                language_model, target_length=CANONICAL_LM_PREFILL_TARGET_LEN
            )
        try:
            import torch
            parameter = next(self.gemma_model.audio_tower.parameters())
            self.device = torch.device(device or parameter.device)
            self.dtype = parameter.dtype
            self._cuda_graph: Any | None = None
            self._static_features: Any | None = None
            self._static_shift: Any | None = None
            self._static_new_slice: Any | None = None
            self._static_mask: Any | None = None
            self._graph_output: Any | None = None
            self._graph_stream: Any | None = None
            self._waiting_for_lm = threading.Event()
            self._frame_done_event = threading.Event()
            self._frame_done_event.set()
            self._tick_started_event = threading.Event()
            self._tick_started_event.set()
            self._frame_counter = 0
            self._active_streaming = False
            if self.device.type == "cuda":
                self._warmup_cuda_graph()
        except (ImportError, AttributeError, StopIteration) as exc:
            raise TypeError("audio tower Gemma non disponibile") from exc

    def _warmup_cuda_graph(self) -> None:
        if self.device.type != "cuda":
            return
        import torch
        if self._cuda_graph is None and self.multimodal_model is not None:
            self._static_features = torch.zeros((1, 1024, 128), device=self.device, dtype=self.dtype)
            self._static_shift = torch.empty((1, 1008, 128), device=self.device, dtype=self.dtype)
            self._static_new_slice = torch.empty((1, 16, 128), device=self.device, dtype=self.dtype)
            self._static_mask = torch.ones((1, 1024), device=self.device, dtype=torch.bool)
            self._graph_stream = torch.cuda.Stream()
            self._graph_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._graph_stream):
                for _ in range(3):
                    with torch.inference_mode():
                        self._graph_output = self.gemma_model.get_audio_features(
                            input_features=self._static_features,
                            input_features_mask=self._static_mask,
                            return_dict=True,
                        )
            torch.cuda.current_stream().wait_stream(self._graph_stream)
            self._cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._cuda_graph, stream=self._graph_stream):
                with torch.inference_mode():
                    self._graph_output = self.gemma_model.get_audio_features(
                        input_features=self._static_features,
                        input_features_mask=self._static_mask,
                        return_dict=True,
                    )

    def _extract_single_frame(
        self,
        wave_1280: np.ndarray,
        past_160: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Incremental 80 ms mel extraction (8 mel rows), bit-exact to full rolling history."""
        fe = self.feature_extractor
        if not hasattr(fe, "frame_length") or not hasattr(fe, "mel_filters"):
            res = self._extract([wave_1280], rows=8)[0]
            return res, wave_1280[-160:]

        curr = np.asarray(wave_1280, dtype=np.float32)
        if past_160 is None or len(past_160) == 0:
            past = np.zeros(160, dtype=np.float32)
        else:
            past = np.asarray(past_160, dtype=np.float32)
            if len(past) < 160:
                past = np.pad(past, (160 - len(past), 0), mode="constant")
            elif len(past) > 160:
                past = past[-160:]

        padded_1441 = np.concatenate((past, curr, np.zeros(1, dtype=np.float32)))
        frames_to_process = np.lib.stride_tricks.sliding_window_view(padded_1441, window_shape=321)[::160]

        if fe.preemphasis > 0.0:
            if getattr(fe, "preemphasis_htk_flavor", False):
                first_in_frame = frames_to_process[..., :1] * (1.0 - fe.preemphasis)
                rest_in_frame = frames_to_process[..., 1:-1] - fe.preemphasis * frames_to_process[..., :-2]
                frames = np.concatenate([first_in_frame, rest_in_frame], axis=-1)
            else:
                frames = frames_to_process[..., 1:] - fe.preemphasis * frames_to_process[..., :-1]
        else:
            frames = frames_to_process[..., :-1]

        frames = frames * fe.window
        stft = np.fft.rfft(frames, n=fe.fft_length, axis=-1)
        mag = np.abs(stft)
        mel = np.matmul(mag, fe.mel_filters)
        log_mel = np.log(mel + fe.mel_floor)
        if fe.per_bin_mean is not None:
            log_mel = log_mel - fe.per_bin_mean
        if fe.per_bin_stddev is not None:
            log_mel = log_mel / fe.per_bin_stddev
        return log_mel.astype(np.float32), curr[-160:]

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
        input_ids: Any | None = None,
    ) -> Any:
        """Run one listener-only Gemma LM chunk and commit its KV.

        The canonical ShapeStable implementation is the default. Set
        ``GEMMA4_NATIVE_LISTENER=1`` (or pass ``use_native_lm=True``) only
        after the native BF16 parity gate has been accepted for the deployed
        checkpoint.
        """
        import torch

        lm = self.gemma_model.language_model
        if self._use_native_lm:
            if state.gemma_lm_terminal_error is not None:
                raise RuntimeError(
                    "executor Gemma nativo in stato terminale; chiamare state.reset() "
                    f"({state.gemma_lm_terminal_error})"
                )
            from duplex_kernel.native_listener import NativeGemmaListenerLM

            native = state.gemma_lm_native
            if native is None:
                try:
                    native = NativeGemmaListenerLM(
                        lm,
                        max_seq_len=NATIVE_LM_MAX_SEQ_LEN,
                        stream_q_len=AUDIO_FEATURES_PER_FRAME,
                    )
                except BaseException as exc:
                    state.gemma_lm_terminal_error = str(exc)
                    raise
                state.gemma_lm_native = native
            q_len = int(embeds.shape[1])
            if q_len == AUDIO_START_AFTER_TEXT:
                if input_ids is None or attention is None:
                    raise RuntimeError("prime nativo richiede input_ids e mask canonici")
                if (
                    input_ids.ndim != 2
                    or input_ids.shape[0] != 1
                    or input_ids.shape[1] < AUDIO_START_AFTER_TEXT
                    or input_ids.dtype != torch.long
                    or input_ids.device != embeds.device
                ):
                    raise ValueError("input_ids nativi non hanno il prefisso canonico")
                prefix_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
                if int(prefix_ids[0, 0].item()) != self.bos_token_id or int(prefix_ids[0, -1].item()) != self.boa_token_id:
                    raise ValueError("prefisso nativo privo di BOS/BOA canonici")
                prefix_attention = attention[:, :AUDIO_START_AFTER_TEXT]
                expected_mask = torch.zeros_like(prefix_attention, dtype=torch.bool)
                expected_mask[:, 0] = True
                expected_mask[:, -1] = True
                if not bool(torch.equal(prefix_attention.to(dtype=torch.bool), expected_mask)):
                    raise ValueError("mask nativa non coincide con BOS + 32 slot + BOA")
                try:
                    native.reset()
                    prefix_attention_contiguous = torch.empty(
                        tuple(prefix_attention.shape), device=self.device, dtype=torch.int64
                    )
                    prefix_attention_contiguous.copy_(prefix_attention)
                    native.prime_async(
                        embeds,
                        per_layer,
                        prefix_attention_contiguous,
                    )
                except BaseException as exc:
                    state.gemma_lm_terminal_error = str(exc)
                    raise
                state.gemma_lm_cache = native
                state.gemma_lm_shared_kv_states = None
                return native.output_tensor
            if q_len != AUDIO_FEATURES_PER_FRAME or state.gemma_lm_cache is None:
                raise RuntimeError("stato nativo Gemma non inizializzato o query non canonica")
            try:
                output = native.step_async(embeds)
            except BaseException as exc:
                state.gemma_lm_terminal_error = str(exc)
                raise
            state.gemma_lm_cache = native
            state.gemma_lm_shared_kv_states = None
            return output

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

    def prime_steady_state(self, state: GemmaListenerState, text_token_ids: tuple[int, ...] = ()) -> None:
        """Prime the listener state to steady-state so frame 0 executes in <=80ms without cold replay."""
        import torch
        text_ids = tuple(int(x) for x in (text_token_ids or self.text_token_ids))
        lm = self.gemma_model.language_model
        input_ids, attention_mask = self._build_multimodal_inputs(text_ids, 1024, device=self.device)
        stable_attention = attention_mask[:, :AUDIO_START_AFTER_TEXT].to(device=self.device, dtype=torch.long)
        stable_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
        stable_embeds = lm.embed_tokens(stable_ids)
        stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)
        state.gemma_lm_cache = None
        state.gemma_lm_shared_kv_states = None
        state.gemma_lm_audio_frames = 0
        self._forward_incremental_lm(
            embeds=stable_embeds,
            per_layer=stable_ple,
            attention=stable_attention,
            positions=torch.arange(AUDIO_START_AFTER_TEXT, device=self.device, dtype=torch.long),
            state=state,
            input_ids=input_ids,
        )
        state.gemma_lm_text_context = text_ids
        state.gemma_lm_terminal_error = None
        state.gemma_lm_context_start_frame = 0
        state.next_frame_id = 0

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
        The rolling audio feature window does not reset or replay the LM KV:
        the decoder remains continuous across frame 63/64/65.
        """
        import torch

        lm = self.gemma_model.language_model
        lm_dtype = lm.embed_tokens.weight.dtype
        frame_index = len(state.user_pcm_history) - 1
        context_start = state.next_frame_id - frame_index
        context_changed = (
            state.gemma_lm_cache is None
            or (self._use_native_lm and state.gemma_lm_native is None)
            or state.gemma_lm_text_context != text_ids
            or state.gemma_lm_audio_frames != state.next_frame_id
        )
        stable_attention = (
            attention_mask[:, :AUDIO_START_AFTER_TEXT].to(device=self.device, dtype=torch.long)
            if not (self._use_native_lm and not context_changed)
            else None
        )
        frame_rows = AUDIO_FEATURES_PER_FRAME

        if context_changed:
            stable_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
            stable_embeds = lm.embed_tokens(stable_ids)
            stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)
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
                input_ids=input_ids,
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

        if self._use_native_lm and not context_changed:
            frame_audio = audio_soft
        else:
            start = frame_index * frame_rows
            frame_audio = audio_soft[:, start:start + frame_rows, :]
        frame_audio = frame_audio.to(dtype=lm_dtype)
        if tuple(frame_audio.shape) != (1, frame_rows, self.hidden_size):
            raise RuntimeError(f"frame audio nativo inatteso: {tuple(frame_audio.shape)}")
        frame_ple = self._frame_pad_per_layer_inputs(lm)
        if self._use_native_lm:
            # The C executor owns absolute position and persistent mask state.
            positions = attention = None
        else:
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
        canonical = self.multimodal_model is not None
        if canonical and state.context_frames > DEFAULT_CONTEXT_FRAMES:
            raise ValueError(
                f"contesto listener Gemma oltre il massimo attestato di {DEFAULT_CONTEXT_FRAMES} frame"
            )
        if getattr(self, "_use_native_lm", False) and state.next_frame_id >= NATIVE_LM_FRAME_CAPACITY:
            message = (
                "capacità LM nativa esaurita: "
                f"{NATIVE_LM_FRAME_CAPACITY} frame dopo il prefisso; chiamare state.reset()"
            )
            state.gemma_lm_terminal_error = message
            raise RuntimeError(message)
        target_rows = state.context_frames * MEL_PER_LISTENER_FRAME if canonical else None
        import torch

        # 1. Incremental mel extraction (outside lock)
        is_steady_state = (
            canonical
            and self.device.type == "cuda"
            and hasattr(state, "user_mel_history")
            and len(state.user_mel_history) >= 64
            and self._cuda_graph is not None
        )

        if is_steady_state:
            last_u_pcm = state.user_pcm_history[-1]
            last_a_pcm = state.assistant_pcm_history[-1]
            u_mel_8, state.user_pcm_tail = self._extract_single_frame(last_u_pcm, state.user_pcm_tail)
            a_mel_8, state.assistant_pcm_tail = self._extract_single_frame(last_a_pcm, state.assistant_pcm_tail)
            state.user_mel_history.append(u_mel_8)
            state.assistant_mel_history.append(a_mel_8)
            interleaved_8 = interleave_duplex_mel(u_mel_8, a_mel_8)
            if self._static_shift is None or self._static_new_slice is None or self._graph_stream is None:
                raise RuntimeError("CUDA graph feature workspace non inizializzato")
            current_stream = torch.cuda.current_stream(device=self.device)
            # Reuse the fixed feature workspace only after the previous graph
            # replay has finished, then fence the graph consumer after the
            # current-stream updates. Both edges are device-side waits.
            current_stream.wait_stream(self._graph_stream)
            self._static_shift.copy_(self._static_features[:, 16:, :])
            self._static_features[:, :-16, :].copy_(self._static_shift)
            self._static_new_slice.copy_(torch.from_numpy(interleaved_8).unsqueeze(0), non_blocking=True)
            self._static_features[:, -16:, :].copy_(self._static_new_slice)
            self._graph_stream.wait_stream(current_stream)
            self._cuda_graph.replay()
            current_stream.wait_stream(self._graph_stream)
            audio_output = self._graph_output
            interleaved_rows = 1024
        elif canonical and hasattr(state, "user_mel_history") and len(state.user_pcm_history) > 0:
            last_u_pcm = state.user_pcm_history[-1]
            last_a_pcm = state.assistant_pcm_history[-1]
            u_mel_8, state.user_pcm_tail = self._extract_single_frame(last_u_pcm, state.user_pcm_tail)
            a_mel_8, state.assistant_pcm_tail = self._extract_single_frame(last_a_pcm, state.assistant_pcm_tail)
            state.user_mel_history.append(u_mel_8)
            state.assistant_mel_history.append(a_mel_8)
            u_mel_all = np.concatenate(list(state.user_mel_history), axis=0)
            a_mel_all = np.concatenate(list(state.assistant_mel_history), axis=0)
            rows = target_rows or len(u_mel_all)
            user_mel = self._pad_rows(u_mel_all, rows)
            assistant_mel = self._pad_rows(a_mel_all, rows)
            interleaved = interleave_duplex_mel(user_mel, assistant_mel)
            features = torch.from_numpy(interleaved).unsqueeze(0).to(self.device, dtype=self.dtype)
            feature_mask = torch.ones((1, interleaved.shape[0]), dtype=torch.bool, device=self.device)
            if interleaved.shape[0] == 1024 and self.device.type == "cuda":
                if self._cuda_graph is None:
                    self._warmup_cuda_graph()
                if self._graph_stream is None:
                    raise RuntimeError("CUDA graph stream non inizializzato")
                current_stream = torch.cuda.current_stream(device=self.device)
                current_stream.wait_stream(self._graph_stream)
                self._static_features.copy_(features)
                self._graph_stream.wait_stream(current_stream)
                self._cuda_graph.replay()
                current_stream.wait_stream(self._graph_stream)
                audio_output = self._graph_output
            else:
                with torch.inference_mode():
                    audio_output = self.gemma_model.get_audio_features(
                        input_features=features,
                        input_features_mask=feature_mask,
                        return_dict=True,
                    )
            interleaved_rows = interleaved.shape[0]
        else:
            user_wave = np.concatenate(tuple(state.user_pcm_history), axis=0)
            assistant_wave = np.concatenate(tuple(state.assistant_pcm_history), axis=0)
            user_mel, assistant_mel = self._extract([user_wave, assistant_wave], rows=target_rows)
            rows = target_rows or max(len(user_mel), len(assistant_mel))
            if rows <= 0:
                raise RuntimeError("Gemma audio tower non ha prodotto feature")
            user_mel = self._pad_rows(user_mel, rows)
            assistant_mel = self._pad_rows(assistant_mel, rows)
            interleaved = np.stack((user_mel, assistant_mel), axis=1).reshape(rows * 2, 128)
            features = torch.from_numpy(interleaved).unsqueeze(0).to(self.device, dtype=self.dtype)
            feature_mask = torch.ones((1, interleaved.shape[0]), dtype=torch.bool, device=self.device)
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
        native_session_steady = bool(
            self._use_native_lm
            and state.gemma_lm_native is not None
            and state.gemma_lm_cache is not None
            and state.gemma_lm_text_context == text_ids
            and state.gemma_lm_audio_frames == state.next_frame_id
        )
        if native_session_steady:
            # The native incremental path consumes only the newest Q=4 rows;
            # avoid rebuilding full GPU multimodal IDs/masks and boolean-index
            # temporaries once the persistent LM transaction is established.
            ids = attention = None
        else:
            ids, attention = self._build_multimodal_inputs(
                text_ids, interleaved_rows, device=self.device
            )
        pooled = getattr(audio_output, "pooler_output", None)
        valid_mask = getattr(audio_output, "attention_mask", None)
        if pooled is None or valid_mask is None:
            raise RuntimeError("Gemma audio tower non ha restituito pooler_output/mask")
        if native_session_steady:
            if pooled.ndim != 3 or pooled.shape[1] < AUDIO_FEATURES_PER_FRAME:
                raise RuntimeError("Gemma audio tower native ha restituito shape inattesa")
            audio_soft = pooled[:, -AUDIO_FEATURES_PER_FRAME:, :].to(
                device=self.device, dtype=self.dtype
            )
        else:
            valid = pooled[0][valid_mask[0].to(pooled.device)]
            if valid.numel() == 0 or valid.shape[-1] != self.hidden_size:
                raise RuntimeError("Gemma audio tower ha restituito hidden vuoto/incompatibile")
            audio_soft = valid.unsqueeze(0).to(device=self.device, dtype=self.dtype)

        # 3. Multimodal LM forward (inside lock with dedicated GPU)
        t_before_lock = time.perf_counter()
        while not self._lock.acquire(blocking=False):
            time.sleep(0.0001)
        state.last_lock_wait_ms = (time.perf_counter() - t_before_lock) * 1000.0
        try:
            hidden = self._run_incremental_lm(
                input_ids=ids,
                attention_mask=attention,
                audio_soft=audio_soft,
                state=state,
                text_ids=text_ids,
            ).detach()
        finally:
            self._lock.release()
        if hidden.shape[-1] != self.hidden_size:
            raise RuntimeError(f"hidden Gemma multimodale inatteso: {tuple(hidden.shape)}")
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
                finite = bool(torch.isfinite(hidden.sum()).item())
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
        if hasattr(self.backend, "_waiting_for_lm") and self.backend._waiting_for_lm is not None:
            self.backend._waiting_for_lm.set()
        if hasattr(self.backend, "_tick_started_event") and self.backend._tick_started_event is not None:
            self.backend._tick_started_event.set()
        try:
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

                # Snapshot all rolling histories because encode() advances them
                # before the LM transaction is known to succeed. This also
                # restores a frame evicted by a full deque on failure.
                user_pcm_before = tuple(state.user_pcm_history)
                assistant_pcm_before = tuple(state.assistant_pcm_history)
                user_mel_before = tuple(state.user_mel_history)
                assistant_mel_before = tuple(state.assistant_mel_history)
                user_tail_before = None if state.user_pcm_tail is None else state.user_pcm_tail.copy()
                assistant_tail_before = None if state.assistant_pcm_tail is None else state.assistant_pcm_tail.copy()

                def restore_feature_state() -> None:
                    state.user_pcm_history.clear()
                    state.user_pcm_history.extend(user_pcm_before)
                    state.assistant_pcm_history.clear()
                    state.assistant_pcm_history.extend(assistant_pcm_before)
                    state.user_mel_history.clear()
                    state.user_mel_history.extend(user_mel_before)
                    state.assistant_mel_history.clear()
                    state.assistant_mel_history.extend(assistant_mel_before)
                    state.user_pcm_tail = user_tail_before
                    state.assistant_pcm_tail = assistant_tail_before

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
                    restore_feature_state()
                    # The functional LM may have appended part of the failing
                    # chunk before an exception.  Never reuse that partial cache:
                    # the next valid tick will deterministically replay history.
                    state.invalidate_gemma_lm_state()
                    state.dropped_listener_frames += 1
                    raise
                native_fast_path = bool(
                    getattr(self.backend, "_use_native_lm", False)
                    and state.gemma_lm_native is not None
                )
                if native_fast_path:
                    # Native step_async has a persistent output tensor. Avoid a
                    # scalar GPU reduction/readback in the frame hot path;
                    # native launch/lifecycle errors remain fail-closed.
                    finite = True
                    nonempty = int(hidden.numel()) > 0
                else:
                    finite, nonempty = self._hidden_health(hidden)
                if not nonempty:
                    state.empty_hidden_count += 1
                    restore_feature_state()
                    state.invalidate_gemma_lm_state()
                    state.dropped_listener_frames += 1
                    raise RuntimeError("listener hidden vuoto")
                if not finite:
                    state.nonfinite_hidden_count += 1
                    restore_feature_state()
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
                    detached_hidden = hidden.detach()
                    if native_fast_path:
                        import torch
                        if (
                            state.native_last_hidden is None
                            or state.native_last_hidden.shape != detached_hidden.shape
                            or state.native_last_hidden.device != detached_hidden.device
                            or state.native_last_hidden.dtype != detached_hidden.dtype
                        ):
                            state.native_last_hidden = torch.empty_like(detached_hidden)
                        state.native_last_hidden.copy_(detached_hidden, non_blocking=True)
                        state.last_hidden = state.native_last_hidden
                    else:
                        state.last_hidden = detached_hidden.clone()
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
        finally:
            if hasattr(self.backend, "_waiting_for_lm") and self.backend._waiting_for_lm is not None:
                self.backend._waiting_for_lm.clear()
            if hasattr(self.backend, "_frame_done_event") and self.backend._frame_done_event is not None:
                self.backend._frame_done_event.set()
            if hasattr(self.backend, "_frame_counter") and self.backend._frame_counter is not None:
                self.backend._frame_counter += 1

    def prime_steady_state(self, state: GemmaListenerState) -> None:
        if hasattr(self.backend, "prime_steady_state"):
            self.backend.prime_steady_state(state)

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
