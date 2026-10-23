"""Exact incremental Q2 execution for the Gemma 4 audio Conformer tower.

The implementation follows the installed ``Gemma4AudioModel`` modules and
keeps only the causal state needed for the next eight Mel rows.  Prepared
batches write a separate staging bank; the canonical bank changes only when
the returned transaction is committed.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Sequence

import torch
import torch.nn.functional as F


MEL_TICK_ROWS = 8
MEL_BINS = 128
Q2_TOKENS = 2
MAX_PAST_TOKENS = 12
CACHED_PAST_TOKENS = 13
ATTENTION_CHUNK_SIZE = 12
_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class AudioStreamCursor:
    """Stable session identity and the cursor before this Mel tick."""

    session_id: str
    generation_id: int
    state_epoch_id: str
    frame_id: int
    sequence_position: int

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("audio stream session_id must be non-empty")
        if not isinstance(self.state_epoch_id, str) or not self.state_epoch_id:
            raise ValueError("audio stream state_epoch_id must be non-empty")
        for name in ("generation_id", "frame_id", "sequence_position"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"audio stream {name} must be a non-negative integer")

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.session_id, self.generation_id, self.state_epoch_id)


@dataclass(slots=True)
class _CudaGraphWorkspace:
    graph: Any
    source_features: torch.Tensor
    source_mask: torch.Tensor
    input_features: torch.Tensor
    input_mask: torch.Tensor
    slot_ids: torch.Tensor
    reset_rows: torch.Tensor
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]


class PreparedStreamingAudioBatch:
    """A staged tower result whose state advances only on ``commit``."""

    def __init__(
        self,
        owner: "StreamingConformerAudioTower",
        *,
        cursors: tuple[AudioStreamCursor, ...],
        slot_ids: tuple[int, ...],
        slot_tensor: torch.Tensor,
        replacing_keys: tuple[tuple[str, int, str] | None, ...],
        expected_before: tuple[tuple[int, int] | None, ...],
        raw_tokens: torch.Tensor,
        soft_tokens: torch.Tensor,
        subsampled: torch.Tensor,
        layer_tokens: tuple[torch.Tensor, ...],
        reset_reasons: tuple[str, ...],
        execution_mode: str,
    ) -> None:
        self._owner = owner
        self.cursors = cursors
        self.slot_ids = slot_ids
        self._slot_tensor = slot_tensor
        self.replacing_keys = replacing_keys
        self.expected_before = expected_before
        self.raw_tokens = raw_tokens
        self.soft_tokens = soft_tokens
        self.subsampled = subsampled
        self.layer_tokens = layer_tokens
        self.reset_reasons = reset_reasons
        self.execution_mode = execution_mode
        self._committed = False
        self._aborted = False
        self._rollback_record: dict[str, Any] | None = None

    @property
    def committed(self) -> bool:
        return self._committed

    def commit(self) -> None:
        if self._committed:
            raise RuntimeError("streaming audio batch was already committed")
        if self._aborted:
            raise RuntimeError("streaming audio batch was aborted")
        self._owner._commit(self)
        self._committed = True

    def rollback_committed(self) -> None:
        """Restore the prior state if its enclosing listener transaction fails."""
        if not self._committed or self._rollback_record is None:
            raise RuntimeError("streaming audio batch has no committed state to roll back")
        self._owner._rollback(self)
        self._committed = False
        self._rollback_record = None

    def abort(self) -> None:
        if self._committed:
            raise RuntimeError("committed streaming audio batch cannot be aborted")
        if self._aborted:
            raise RuntimeError("streaming audio batch was already aborted")
        self._owner._abort(self)
        self._aborted = True


class StreamingConformerAudioTower:
    """Exact per-session incremental execution of Gemma4AudioModel.

    ``prepare_batch`` accepts exactly one ``[N,8,128]`` real Mel tick per
    session and returns the final two projected tower rows plus the single
    official ``embed_audio`` transform used by Gemma4Model.  The key/value,
    causal-convolution, and subsampler caches are double buffered by session
    slot, never by microbatch row index.
    """

    def __init__(
        self,
        audio_tower: Any,
        embed_audio: Any,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        max_sessions: int = 64,
        capture_cuda_graphs: bool = False,
        optimize_cuda: bool = False,
    ) -> None:
        if not callable(getattr(audio_tower, "subsample_conv_projection", None)):
            raise TypeError("audio_tower must be an installed Gemma4AudioModel")
        if not callable(embed_audio):
            raise TypeError("embed_audio must be Gemma4MultimodalEmbedder")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions < 2:
            raise ValueError("max_sessions must be an integer >= 2")
        self.audio_tower = audio_tower
        self.embed_audio = embed_audio
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = dtype
        self.max_sessions = max_sessions
        self.optimize_cuda = bool(optimize_cuda)
        self._forward_wrappers: tuple[tuple[Any, Any, bool, Any], ...] = ()
        self._closed = False
        self._compiled_compute_step = None
        self._validate_model()

        config = audio_tower.config
        self.hidden_size = int(config.hidden_size)
        self.num_layers = int(config.num_hidden_layers)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.output_size = int(config.output_proj_dims)
        self.text_hidden_size = int(getattr(embed_audio, "text_hidden_size", self.output_size))
        query_positions = torch.arange(Q2_TOKENS, device=self.device)[:, None]
        key_positions = torch.arange(CACHED_PAST_TOKENS + Q2_TOKENS, device=self.device)[None, :]
        cached_distances = CACHED_PAST_TOKENS - key_positions[:, :CACHED_PAST_TOKENS] + query_positions
        current_distances = query_positions - (
            key_positions[:, CACHED_PAST_TOKENS:] - CACHED_PAST_TOKENS
        )
        distances = torch.cat((cached_distances, current_distances), dim=1)
        self._attention_window_mask = (distances >= 0) & (distances < MAX_PAST_TOKENS)
        self._relative_position_indices = MAX_PAST_TOKENS - distances.clamp(
            min=0, max=MAX_PAST_TOKENS
        )
        channels0, channels1 = tuple(int(x) for x in config.subsampling_conv_channels)
        self.subsample_frequency0 = (MEL_BINS + 1) // 2
        self.subsample_frequency1 = (self.subsample_frequency0 + 1) // 2
        if channels0 % 4:
            raise ValueError("first subsampler channel count must be divisible by four")

        shape = (max_sessions,)
        self._mel_cache = torch.zeros(
            (*shape, 1, 1, MEL_BINS), device=self.device, dtype=self.dtype
        )
        self._mel_mask_cache = torch.zeros((*shape, 1), device=self.device, dtype=torch.bool)
        self._subsample_cache = torch.zeros(
            (*shape, channels0, 1, self.subsample_frequency0),
            device=self.device,
            dtype=self.dtype,
        )
        self._subsample_mask_cache = torch.zeros(
            (*shape, 1), device=self.device, dtype=torch.bool
        )
        kv_shape = (
            max_sessions,
            self.num_layers,
            CACHED_PAST_TOKENS,
            self.num_heads,
            self.head_dim,
        )
        self._keys = torch.zeros(kv_shape, device=self.device, dtype=torch.float32)
        self._values = torch.zeros_like(self._keys)
        self._conv_inputs = torch.zeros(
            (max_sessions, self.num_layers, 4, self.hidden_size),
            device=self.device,
            dtype=self.dtype,
        )
        self._token_valid = torch.zeros(
            (max_sessions, CACHED_PAST_TOKENS), device=self.device, dtype=torch.bool
        )
        self._stage_mel_cache = torch.empty_like(self._mel_cache)
        self._stage_mel_mask_cache = torch.empty_like(self._mel_mask_cache)
        self._stage_subsample_cache = torch.empty_like(self._subsample_cache)
        self._stage_subsample_mask_cache = torch.empty_like(self._subsample_mask_cache)
        self._stage_keys = torch.empty_like(self._keys)
        self._stage_values = torch.empty_like(self._values)
        self._stage_conv_inputs = torch.empty_like(self._conv_inputs)
        self._stage_token_valid = torch.empty_like(self._token_valid)
        self._slots_by_key: dict[tuple[str, int, str], int] = {}
        self._expected_positions: dict[tuple[str, int, str], tuple[int, int]] = {}
        self._key_by_slot: dict[int, tuple[str, int, str]] = {}
        self._active_batch: PreparedStreamingAudioBatch | None = None
        self._lock = threading.RLock()
        self._graphs: dict[int, _CudaGraphWorkspace] = {}
        # Optional diagnostic-only CUDA event pairs. Kept None in serving so
        # neither Python profiling ranges nor event recording affect requests.
        self._profile_layer_events: tuple[tuple[Any, Any], ...] | None = None
        self._capture_cuda_graphs = bool(capture_cuda_graphs)

        if self.optimize_cuda:
            if self.device.type != "cuda" or self.dtype is not torch.bfloat16:
                raise ValueError("optimize_cuda requires CUDA BF16 audio modules")
            from duplex_kernel.streaming_conformer_kernels import (
                initialize_triton_kernels,
                install_gemma4_linear_wrappers,
            )

            # Define the Triton JIT kernel before Dynamo sees _execute. The
            # wrappers are confined to these two audio module instances.
            initialize_triton_kernels()
            self._forward_wrappers = install_gemma4_linear_wrappers(
                (self.audio_tower, self.embed_audio),
                block_n=32,
                block_k=128,
            )

        try:
            # The source positional encoder is fixed (12..0) and independent
            # of input length. Project it once with the startup BF16 policy.
            with torch.no_grad():
                relative_encoding = self.audio_tower.rel_pos_enc(
                    torch.empty(
                        (1, Q2_TOKENS, self.hidden_size),
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                self._relative_keys = tuple(
                    layer.self_attn.relative_k_proj(relative_encoding)
                    .view(-1, self.num_heads, self.head_dim)
                    .to(dtype=torch.float32)
                    for layer in self.audio_tower.layers
                )
        except BaseException:
            self._restore_forward_wrappers()
            raise

        if self._capture_cuda_graphs and self.device.type != "cuda":
            self._restore_forward_wrappers()
            raise ValueError("CUDA graph capture requires a CUDA device")
        if self.optimize_cuda:
            try:
                self._compiled_compute_step = torch.compile(
                    self._compute_step,
                    fullgraph=True,
                    dynamic=False,
                    options={"emulate_precision_casts": True},
                )
            except BaseException:
                self._restore_forward_wrappers()
                raise

    @classmethod
    def from_gemma_model(
        cls,
        gemma_model: Any,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        max_sessions: int = 64,
        capture_cuda_graphs: bool = False,
        optimize_cuda: bool = False,
    ) -> "StreamingConformerAudioTower":
        audio_tower = getattr(gemma_model, "audio_tower", None)
        embed_audio = getattr(gemma_model, "embed_audio", None)
        if audio_tower is None or embed_audio is None:
            raise TypeError("Gemma4Model must expose audio_tower and embed_audio")
        return cls(
            audio_tower,
            embed_audio,
            device=device,
            dtype=dtype,
            max_sessions=max_sessions,
            capture_cuda_graphs=capture_cuda_graphs,
            optimize_cuda=optimize_cuda,
        )

    @classmethod
    def from_gemma_model_optimized_cuda(
        cls,
        gemma_model: Any,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        max_sessions: int = 128,
        max_graph_batch_size: int = 4,
    ) -> "StreamingConformerAudioTower":
        """Construct, compile, and capture CUDA BF16 graphs under one flag scope."""
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions < 2:
            raise ValueError("max_sessions must be an integer >= 2")
        if (
            isinstance(max_graph_batch_size, bool)
            or not isinstance(max_graph_batch_size, int)
            or max_graph_batch_size < 1
            or max_graph_batch_size > max_sessions
        ):
            raise ValueError("max_graph_batch_size must be within persistent session capacity")
        if torch.device(device).type != "cuda" or dtype is not torch.bfloat16:
            raise ValueError("optimized CUDA startup requires CUDA BF16")

        matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
        flag_name = "allow_bf16_reduced_precision_reduction"
        if matmul is None or not hasattr(matmul, flag_name):
            raise RuntimeError(
                "PyTorch does not expose allow_bf16_reduced_precision_reduction; optimized BF16 is unsupported"
            )
        previous_flag = getattr(matmul, flag_name)
        tower = None
        try:
            setattr(matmul, flag_name, False)
            tower = cls.from_gemma_model(
                gemma_model,
                device=device,
                dtype=dtype,
                max_sessions=max_sessions,
                capture_cuda_graphs=True,
                optimize_cuda=True,
            )
            for batch_size in range(1, max_graph_batch_size + 1):
                tower.capture_graph(batch_size)
            return tower
        except BaseException:
            if tower is not None:
                try:
                    tower.close()
                except BaseException:
                    pass
            raise
        finally:
            setattr(matmul, flag_name, previous_flag)

    def _validate_model(self) -> None:
        config = self.audio_tower.config
        exact = {
            "attention_chunk_size": ATTENTION_CHUNK_SIZE,
            "attention_context_left": MAX_PAST_TOKENS + 1,
            "attention_context_right": 0,
            "conv_kernel_size": 5,
        }
        for name, expected in exact.items():
            actual = int(getattr(config, name, -1))
            if actual != expected:
                raise ValueError(f"Gemma audio {name} must be {expected}, found {actual}")
        if int(config.hidden_size) % int(config.num_attention_heads):
            raise ValueError("Gemma audio hidden_size must divide evenly across attention heads")
        if len(self.audio_tower.layers) != int(config.num_hidden_layers):
            raise ValueError("Gemma audio layer count differs from its configuration")
        if self.audio_tower.training or self.embed_audio.training:
            raise ValueError("incremental Gemma audio tower requires eval mode")
        parameter = next(self.audio_tower.parameters())
        embed_parameter = next(self.embed_audio.parameters())
        if parameter.device != self.device or embed_parameter.device != self.device:
            raise ValueError("audio tower and embed_audio must already be on the requested device")
        if parameter.dtype != self.dtype or embed_parameter.dtype != self.dtype:
            raise ValueError("audio tower and embed_audio must already use the requested dtype")
        layers = self.audio_tower.subsample_conv_projection
        for name in ("layer0", "layer1"):
            layer = getattr(layers, name, None)
            if layer is None or tuple(layer.conv.kernel_size) != (3, 3):
                raise ValueError("Gemma audio subsampler must use its 3x3 convolution layers")
            if tuple(layer.conv.stride) != (2, 2) or tuple(layer.conv.padding) != (1, 1):
                raise ValueError("Gemma audio subsampler must use stride 2 and symmetric padding 1")
            if layer.conv.bias is not None:
                raise ValueError("Gemma audio subsampler convolutions must remain biasless")

    @property
    def committed_positions(self) -> tuple[tuple[str, int, str, int, int], ...]:
        return tuple(
            sorted(
                (key[0], key[1], key[2], expected[0], expected[1])
                for key, expected in self._expected_positions.items()
            )
        )

    def reset(self, session_id: str, generation_id: int | None = None, state_epoch_id: str | None = None) -> None:
        """Clear one identity, or every generation/epoch for a session."""
        with self._lock:
            self._ensure_open()
            if self._active_batch is not None:
                raise RuntimeError("cannot reset streaming audio state while a batch is prepared")
            selected = [
                key
                for key in self._slots_by_key
                if key[0] == session_id
                and (generation_id is None or key[1] == generation_id)
                and (state_epoch_id is None or key[2] == state_epoch_id)
            ]
            for key in selected:
                slot = self._slots_by_key.pop(key)
                self._expected_positions.pop(key, None)
                self._key_by_slot.pop(slot, None)
                self._zero_slots((slot,))

    def reset_session(self, slot_idx: int) -> None:
        """Release exactly one persistent slot after its owner session is idle."""
        if isinstance(slot_idx, bool) or not isinstance(slot_idx, int):
            raise TypeError("slot_idx must be an integer")
        if slot_idx < 0 or slot_idx >= self.max_sessions:
            raise IndexError(f"slot_idx {slot_idx} is outside [0, {self.max_sessions})")
        with self._lock:
            self._ensure_open()
            if self._active_batch is not None:
                raise RuntimeError("cannot reset a streaming audio slot while a batch is prepared")
            key = self._key_by_slot.pop(slot_idx, None)
            if key is not None:
                self._slots_by_key.pop(key, None)
                self._expected_positions.pop(key, None)
            self._zero_slots((slot_idx,))

    def capture_graph(self, batch_size: int) -> None:
        try:
            self._capture_graph_impl(batch_size)
        except BaseException:
            if self.optimize_cuda and self._active_batch is None and not self.committed_positions:
                try:
                    self.close()
                except BaseException:
                    pass
            raise

    def _capture_graph_impl(self, batch_size: int) -> None:
        """Warm and capture one static [N,8,128] -> post-embed graph.

        Warmup and capture only write the staging bank; no session metadata or
        canonical cache is advanced.
        """
        if self.device.type != "cuda":
            raise RuntimeError("CUDA graph capture requires a CUDA device")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
            or batch_size > self.max_sessions
        ):
            raise ValueError(
                f"streaming Conformer graph batch size must be within [1, {self.max_sessions}]"
            )
        with self._lock:
            self._ensure_open()
            if self._active_batch is not None:
                raise RuntimeError("cannot capture while a streaming batch is prepared")
            if batch_size in self._graphs:
                return
            static = _CudaGraphWorkspace(
                graph=None,
                source_features=torch.zeros(
                    (batch_size, MEL_TICK_ROWS, MEL_BINS), device=self.device, dtype=self.dtype
                ),
                source_mask=torch.ones(
                    (batch_size, MEL_TICK_ROWS), device=self.device, dtype=torch.bool
                ),
                input_features=torch.zeros(
                    (batch_size, MEL_TICK_ROWS, MEL_BINS), device=self.device, dtype=self.dtype
                ),
                input_mask=torch.ones(
                    (batch_size, MEL_TICK_ROWS), device=self.device, dtype=torch.bool
                ),
                slot_ids=torch.arange(batch_size, device=self.device, dtype=torch.long),
                reset_rows=torch.ones(batch_size, device=self.device, dtype=torch.bool),
                outputs=(None, None, None, ()),
            )
            graph_stream = torch.cuda.Stream(device=self.device)
            current_stream = torch.cuda.current_stream(device=self.device)
            graph_stream.wait_stream(current_stream)
            with torch.cuda.stream(graph_stream), torch.inference_mode():
                for _ in range(2):
                    static.input_features.copy_(static.source_features)
                    static.input_mask.copy_(static.source_mask)
                    self.step_q2_cached(
                        static.input_features,
                        static.input_mask,
                        static.slot_ids,
                        static.reset_rows,
                    )
            graph_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=graph_stream), torch.inference_mode():
                static.input_features.copy_(static.source_features)
                static.input_mask.copy_(static.source_mask)
                outputs = self.step_q2_cached(
                    static.input_features,
                    static.input_mask,
                    static.slot_ids,
                    static.reset_rows,
                )
            graph_stream.synchronize()
            static.graph = graph
            static.outputs = outputs
            self._graphs[batch_size] = static

    def prepare_batch(
        self,
        mel_tick: torch.Tensor,
        cursors: Sequence[AudioStreamCursor],
        *,
        input_mask: torch.Tensor | None = None,
        use_cuda_graph: bool = False,
        graph_event_pair: tuple[Any, Any] | None = None,
    ) -> PreparedStreamingAudioBatch:
        rows = tuple(cursors)
        if not isinstance(mel_tick, torch.Tensor) or mel_tick.ndim != 3:
            raise ValueError("streaming Q2 Mel input must be [N,8,128]")
        n = len(rows)
        if n <= 0 or tuple(mel_tick.shape) != (n, MEL_TICK_ROWS, MEL_BINS):
            raise ValueError("streaming Q2 Mel input must match [N,8,128] cursors")
        if mel_tick.device != self.device or mel_tick.dtype != self.dtype:
            raise ValueError("streaming Q2 Mel input device/dtype differs from the audio tower")
        if len({row.key for row in rows}) != n or len({row.session_id for row in rows}) != n:
            raise ValueError("a streaming audio batch requires distinct session identities")
        if input_mask is None:
            input_mask = torch.ones((n, MEL_TICK_ROWS), device=self.device, dtype=torch.bool)
        elif (
            not isinstance(input_mask, torch.Tensor)
            or tuple(input_mask.shape) != (n, MEL_TICK_ROWS)
            or input_mask.device != self.device
        ):
            raise ValueError("streaming input mask must have shape [N,8] on the audio device")
        input_mask = input_mask.to(dtype=torch.bool)

        with self._lock:
            self._ensure_open()
            if self._active_batch is not None:
                raise RuntimeError("previous streaming audio batch has not committed or aborted")
            slot_ids: list[int] = []
            replacing_keys: list[tuple[str, int, str] | None] = []
            expected_before: list[tuple[int, int] | None] = []
            reset_reasons: list[str] = []
            unavailable = set(self._key_by_slot)
            for row in rows:
                key = row.key
                if key in self._slots_by_key:
                    slot = self._slots_by_key[key]
                    replacing = None
                    expected = self._expected_positions.get(key)
                    if expected is None:
                        reason = "reset"
                    elif expected == (row.frame_id, row.sequence_position):
                        reason = "continue"
                    else:
                        reason = "gap_or_cursor_mismatch"
                else:
                    old_keys = [existing for existing in self._slots_by_key if existing[0] == row.session_id]
                    if old_keys:
                        replacing = old_keys[0]
                        slot = self._slots_by_key[replacing]
                    else:
                        replacing = None
                        free = next(
                            (candidate for candidate in range(self.max_sessions) if candidate not in unavailable),
                            None,
                        )
                        if free is None:
                            raise RuntimeError(
                                f"streaming audio session bank exhausted ({self.max_sessions} sessions)"
                            )
                        slot = free
                        unavailable.add(slot)
                    expected = None
                    reason = "identity_reset"
                slot_ids.append(slot)
                replacing_keys.append(replacing)
                expected_before.append(expected)
                reset_reasons.append(reason)
            if len(set(slot_ids)) != n:
                raise RuntimeError("streaming audio batch maps multiple rows to one persistent state slot")

            slots = torch.tensor(slot_ids, device=self.device, dtype=torch.long)
            reset_rows = torch.tensor(
                [reason != "continue" for reason in reset_reasons],
                device=self.device,
                dtype=torch.bool,
            )
            graph_replay = use_cuda_graph and self.device.type == "cuda"
            if self.device.type == "cuda" and self.dtype is torch.bfloat16 and not graph_replay:
                raise RuntimeError(
                    "CUDA BF16 Q2 supports only startup-captured graph replay; use W240 for this form"
                )
            if graph_replay:
                if n not in self._graphs:
                    raise RuntimeError(
                        f"no startup-captured streaming Q2 graph for N={n}; use W240 or recapture at startup"
                    )
                static = self._graphs[n]
                static.source_features.copy_(mel_tick)
                static.source_mask.copy_(input_mask)
                static.slot_ids.copy_(slots)
                static.reset_rows.copy_(reset_rows)
                if graph_event_pair is not None:
                    graph_event_pair[0].record()
                static.graph.replay()
                if graph_event_pair is not None:
                    graph_event_pair[1].record()
                raw, soft, subsampled, layers = static.outputs
            else:
                with torch.inference_mode():
                    raw, soft, subsampled, layers = self.step_q2_cached(
                        mel_tick, input_mask, slots, reset_rows
                    )

            prepared = PreparedStreamingAudioBatch(
                self,
                cursors=rows,
                slot_ids=tuple(slot_ids),
                slot_tensor=slots,
                replacing_keys=tuple(replacing_keys),
                expected_before=tuple(expected_before),
                raw_tokens=raw.detach().clone(),
                soft_tokens=soft.detach().clone(),
                subsampled=subsampled.detach().clone(),
                layer_tokens=tuple(value.detach().clone() for value in layers),
                reset_reasons=tuple(reset_reasons),
                execution_mode="cuda_graph" if graph_replay else "eager",
            )
            self._active_batch = prepared
            return prepared

    def _execute(
        self,
        mel_tick: torch.Tensor,
        input_mask: torch.Tensor,
        slots: torch.Tensor,
        reset_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        compute_step = self._compiled_compute_step or self._compute_step
        (
            raw,
            soft,
            subsampled_tick,
            layer_tokens,
            next_mel,
            next_mel_mask,
            next_subsample,
            next_subsample_mask,
            next_keys,
            next_values,
            next_conv_inputs,
            next_token_valid,
        ) = compute_step(mel_tick, input_mask, slots, reset_rows)
        with torch.no_grad():
            for bank, next_state in zip(
                self._staging_banks(),
                (
                    next_mel,
                    next_mel_mask,
                    next_subsample,
                    next_subsample_mask,
                    next_keys,
                    next_values,
                    next_conv_inputs,
                    next_token_valid,
                ),
                strict=True,
            ):
                bank.index_copy_(0, slots, next_state)
        return raw, soft, subsampled_tick, layer_tokens

    def _compute_step(
        self,
        mel_tick: torch.Tensor,
        input_mask: torch.Tensor,
        slots: torch.Tensor,
        reset_rows: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        n = mel_tick.shape[0]
        mel_prev = self._mel_cache.index_select(0, slots)
        mel_prev_mask = self._mel_mask_cache.index_select(0, slots)
        sub_prev = self._subsample_cache.index_select(0, slots)
        sub_prev_mask = self._subsample_mask_cache.index_select(0, slots)
        keys = self._keys.index_select(0, slots)
        values = self._values.index_select(0, slots)
        conv_inputs = self._conv_inputs.index_select(0, slots)
        token_valid = self._token_valid.index_select(0, slots)
        mel_prev = torch.where(reset_rows[:, None, None, None], torch.zeros_like(mel_prev), mel_prev)
        mel_prev_mask = torch.where(reset_rows[:, None], torch.zeros_like(mel_prev_mask), mel_prev_mask)
        sub_prev = torch.where(reset_rows[:, None, None, None], torch.zeros_like(sub_prev), sub_prev)
        sub_prev_mask = torch.where(reset_rows[:, None], torch.zeros_like(sub_prev_mask), sub_prev_mask)
        keys = torch.where(reset_rows[:, None, None, None, None], torch.zeros_like(keys), keys)
        values = torch.where(reset_rows[:, None, None, None, None], torch.zeros_like(values), values)
        conv_inputs = torch.where(
            reset_rows[:, None, None, None], torch.zeros_like(conv_inputs), conv_inputs
        )
        token_valid = torch.where(
            reset_rows[:, None], torch.zeros_like(token_valid), token_valid
        )
        current_mel_mask = input_mask[:, None, :, None]
        masked_previous_mel = mel_prev * mel_prev_mask[:, None, :, None].to(self.dtype)
        masked_current_mel = mel_tick.unsqueeze(1) * current_mel_mask.to(self.dtype)
        conv0_input = torch.cat((masked_previous_mel, masked_current_mel), dim=2)
        subsample = self.audio_tower.subsample_conv_projection
        hidden0 = _stream_subsample_layer(subsample.layer0, conv0_input)
        mask0 = input_mask[:, ::2]

        hidden1_input = torch.cat(
            (
                sub_prev * sub_prev_mask[:, None, :, None].to(self.dtype),
                hidden0 * mask0[:, None, :, None].to(self.dtype),
            ),
            dim=2,
        )
        hidden1 = _stream_subsample_layer(subsample.layer1, hidden1_input)
        mask1 = mask0[:, ::2]
        batch, channels, time, freq = hidden1.shape
        subsampled = subsample.input_proj_linear(
            hidden1.permute(0, 2, 3, 1).contiguous().reshape(batch, time, channels * freq)
        )

        hidden_tick = subsampled
        valid_now = mask1
        layer_rows: list[torch.Tensor] = []
        next_keys_by_layer: list[torch.Tensor] = []
        next_values_by_layer: list[torch.Tensor] = []
        next_conv_inputs_by_layer: list[torch.Tensor] = []
        for layer_index, layer in enumerate(self.audio_tower.layers):
            profile_range = None
            if self._profile_layer_events is not None:
                self._profile_layer_events[layer_index][0].record()
                profile_range = torch.profiler.record_function(
                    f"streaming_conformer.layer_{layer_index}"
                )
                profile_range.__enter__()
            hidden_tick = layer.feed_forward1(hidden_tick)
            residual = hidden_tick
            gradient_clip = min(
                layer.gradient_clipping,
                torch.finfo(layer.norm_pre_attn.weight.dtype).max,
            )
            normalized = layer.norm_pre_attn(
                torch.clamp(hidden_tick, -gradient_clip, gradient_clip)
            )
            attention = layer.self_attn
            qkv_shape = (-1, Q2_TOKENS, self.num_heads, self.head_dim)
            query = attention.q_proj(normalized).float().view(qkv_shape)
            new_key = attention.k_proj(normalized).float().view(qkv_shape)
            new_value = attention.v_proj(normalized).float().view(qkv_shape)
            query = query * attention.q_scale * F.softplus(attention.per_dim_scale)
            new_key = new_key * attention.k_scale
            all_keys = torch.cat((keys[:, layer_index], new_key), dim=1)
            all_values = torch.cat((values[:, layer_index], new_value), dim=1)
            q_heads = query.permute(0, 2, 1, 3)
            logits_ac = torch.matmul(q_heads, all_keys.permute(0, 2, 3, 1))
            position_keys = self._relative_keys[layer_index][self._relative_position_indices]
            position_keys = position_keys.permute(2, 0, 3, 1).unsqueeze(0)
            logits_bd = torch.matmul(q_heads.unsqueeze(-2), position_keys)
            logits = logits_ac + logits_bd.squeeze(-2)
            cap = attention.softcap.to(dtype=logits.dtype)
            logits = torch.tanh(logits / cap) * cap
            key_valid = torch.cat((token_valid, valid_now), dim=1)
            attention_valid = self._attention_window_mask[None, :, :] & key_valid[:, None, :]
            logits = logits.masked_fill(
                ~attention_valid[:, None, :, :],
                float(layer.config.attention_invalid_logits_value),
            )
            weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(all_values.dtype)
            attended = torch.matmul(
                weights,
                all_values.permute(0, 2, 1, 3),
            ).permute(0, 2, 1, 3).reshape(n, Q2_TOKENS, self.hidden_size)
            attended = attention.post(attended.to(dtype=attention.post.linear.weight.dtype))
            staged_keys = torch.cat(
                (keys[:, layer_index, Q2_TOKENS:], new_key), dim=1
            )[:, -CACHED_PAST_TOKENS:]
            staged_values = torch.cat(
                (values[:, layer_index, Q2_TOKENS:], new_value), dim=1
            )[:, -CACHED_PAST_TOKENS:]
            next_keys_by_layer.append(staged_keys)
            next_values_by_layer.append(staged_values)
            post_clip = min(
                layer.gradient_clipping,
                torch.finfo(layer.norm_post_attn.weight.dtype).max,
            )
            hidden_tick = layer.norm_post_attn(torch.clamp(attended, -post_clip, post_clip))
            hidden_tick = hidden_tick + residual

            light = layer.lconv1d
            residual = hidden_tick
            normalized = light.pre_layer_norm(hidden_tick)
            gated = F.glu(light.linear_start(normalized), dim=-1)
            conv_window = torch.cat((conv_inputs[:, layer_index], gated), dim=1)
            staged_conv_inputs = conv_window[:, -4:, :]
            next_conv_inputs_by_layer.append(staged_conv_inputs)
            conv = F.conv1d(
                conv_window.transpose(1, 2),
                light.depthwise_conv1d.weight,
                bias=light.depthwise_conv1d.bias,
                stride=1,
                padding=0,
                dilation=light.depthwise_conv1d.dilation,
                groups=light.depthwise_conv1d.groups,
            ).transpose(1, 2)
            conv_clip = min(
                light.gradient_clipping,
                torch.finfo(light.linear_start.linear.weight.dtype).max,
            )
            conv = light.conv_norm(torch.clamp(conv, -conv_clip, conv_clip))
            conv = light.act_fn(conv)
            hidden_tick = light.linear_end(conv) + residual

            hidden_tick = layer.feed_forward2(hidden_tick)
            out_clip = min(
                layer.gradient_clipping,
                torch.finfo(layer.norm_out.weight.dtype).max,
            )
            hidden_tick = layer.norm_out(torch.clamp(hidden_tick, -out_clip, out_clip))
            layer_rows.append(hidden_tick)
            if profile_range is not None:
                profile_range.__exit__(None, None, None)
                self._profile_layer_events[layer_index][1].record()

        keys = torch.stack(next_keys_by_layer, dim=1)
        values = torch.stack(next_values_by_layer, dim=1)
        conv_inputs = torch.stack(next_conv_inputs_by_layer, dim=1)
        token_valid = torch.cat((token_valid[:, Q2_TOKENS:], valid_now), dim=1)[
            :, -CACHED_PAST_TOKENS:
        ]
        raw = self.audio_tower.output_proj(hidden_tick)
        soft = self.embed_audio(raw)
        subsampled_tick = subsampled
        layer_tokens = tuple(layer_rows)
        return (
            raw,
            soft,
            subsampled_tick,
            layer_tokens,
            mel_tick[:, -1:, :].unsqueeze(1),
            input_mask[:, -1:],
            hidden0[:, :, -1:, :],
            mask0[:, -1:],
            keys,
            values,
            conv_inputs,
            token_valid,
        )

    def step_q2_cached(
        self,
        mel_tick: torch.Tensor,
        input_mask: torch.Tensor,
        slots: torch.Tensor,
        reset_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """Run a static Q2 step and write next state to staging banks.

        Session ownership and canonical advancement remain managed by
        ``prepare_batch`` and the prepared batch's commit/abort transaction.
        """
        return self._execute(mel_tick, input_mask, slots, reset_rows)

    def _commit(self, batch: PreparedStreamingAudioBatch) -> None:
        with self._lock:
            if self._active_batch is not batch:
                raise RuntimeError("streaming audio transaction is no longer active")
            try:
                for row, expected in zip(batch.cursors, batch.expected_before, strict=True):
                    if self._expected_positions.get(row.key) != expected:
                        raise RuntimeError("cursor changed before streaming commit")
                slots = batch._slot_tensor
                canonical = self._canonical_banks()
                staged = self._staging_banks()
                backups = tuple(bank.index_select(0, slots).clone() for bank in canonical)
                old_entries = tuple(
                    (
                        self._slots_by_key.get(row.key),
                        self._expected_positions.get(row.key),
                        self._key_by_slot.get(slot),
                        replacing,
                        self._slots_by_key.get(replacing) if replacing is not None else None,
                        self._expected_positions.get(replacing) if replacing is not None else None,
                    )
                    for row, slot, replacing in zip(
                        batch.cursors, batch.slot_ids, batch.replacing_keys, strict=True
                    )
                )
                try:
                    for target, source in zip(canonical, staged, strict=True):
                        target.index_copy_(0, slots, source.index_select(0, slots))
                    for row, slot, replacing in zip(
                        batch.cursors, batch.slot_ids, batch.replacing_keys, strict=True
                    ):
                        if replacing is not None:
                            self._slots_by_key.pop(replacing, None)
                            self._expected_positions.pop(replacing, None)
                        self._slots_by_key[row.key] = slot
                        self._key_by_slot[slot] = row.key
                        self._expected_positions[row.key] = (
                            row.frame_id + 1,
                            row.sequence_position + 1,
                        )
                except BaseException:
                    for target, previous in zip(canonical, backups, strict=True):
                        target.index_copy_(0, slots, previous)
                    self._restore_entries(batch, old_entries)
                    raise
                batch._rollback_record = {
                    "slots": slots,
                    "backups": backups,
                    "old_entries": old_entries,
                }
                self._active_batch = None
            except BaseException:
                self._active_batch = None
                batch._aborted = True
                raise

    def _rollback(self, batch: PreparedStreamingAudioBatch) -> None:
        with self._lock:
            record = batch._rollback_record
            if record is None:
                raise RuntimeError("streaming audio rollback record is missing")
            for target, previous in zip(self._canonical_banks(), record["backups"], strict=True):
                target.index_copy_(0, record["slots"], previous)
            self._restore_entries(batch, record["old_entries"])

    def _restore_entries(self, batch: PreparedStreamingAudioBatch, old_entries: tuple[Any, ...]) -> None:
        for row, slot, old in zip(batch.cursors, batch.slot_ids, old_entries, strict=True):
            old_slot, old_position, old_key_at_slot, replacing, replacing_slot, replacing_position = old
            self._slots_by_key.pop(row.key, None)
            self._expected_positions.pop(row.key, None)
            if self._key_by_slot.get(slot) == row.key:
                self._key_by_slot.pop(slot, None)
            if old_slot is not None:
                self._slots_by_key[row.key] = old_slot
            if old_position is not None:
                self._expected_positions[row.key] = old_position
            if old_key_at_slot is not None:
                self._key_by_slot[slot] = old_key_at_slot
            if replacing is not None:
                if replacing_slot is not None:
                    self._slots_by_key[replacing] = replacing_slot
                if replacing_position is not None:
                    self._expected_positions[replacing] = replacing_position

    def _abort(self, batch: PreparedStreamingAudioBatch) -> None:
        with self._lock:
            if self._active_batch is not batch:
                raise RuntimeError("streaming audio transaction is no longer active")
            self._active_batch = None

    def _canonical_banks(self) -> tuple[torch.Tensor, ...]:
        return (
            self._mel_cache,
            self._mel_mask_cache,
            self._subsample_cache,
            self._subsample_mask_cache,
            self._keys,
            self._values,
            self._conv_inputs,
            self._token_valid,
        )

    def _staging_banks(self) -> tuple[torch.Tensor, ...]:
        return (
            self._stage_mel_cache,
            self._stage_mel_mask_cache,
            self._stage_subsample_cache,
            self._stage_subsample_mask_cache,
            self._stage_keys,
            self._stage_values,
            self._stage_conv_inputs,
            self._stage_token_valid,
        )

    def _zero_slots(self, slots: Sequence[int]) -> None:
        ids = torch.tensor(tuple(slots), device=self.device, dtype=torch.long)
        with torch.no_grad():
            for bank in self._canonical_banks():
                bank.index_fill_(0, ids, 0)
            for bank in self._staging_banks():
                bank.index_fill_(0, ids, 0)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("StreamingConformerAudioTower is closed")

    def close(self) -> None:
        """Release session state and restore wrapped forwards when idle."""
        with self._lock:
            if self._active_batch is not None:
                raise RuntimeError("cannot close StreamingConformerAudioTower with an active batch")
            if self._closed:
                return
            self._zero_slots(tuple(range(self.max_sessions)))
            self._slots_by_key.clear()
            self._expected_positions.clear()
            self._key_by_slot.clear()
            self._graphs.clear()
            self._compiled_compute_step = None
            self._restore_forward_wrappers()
            self._closed = True

    def _restore_forward_wrappers(self) -> None:
        if not self._forward_wrappers:
            return
        from duplex_kernel.streaming_conformer_kernels import restore_gemma4_linear_wrappers

        restore_gemma4_linear_wrappers(self._forward_wrappers)
        self._forward_wrappers = ()


def _stream_subsample_layer(layer: Any, input_states: torch.Tensor) -> torch.Tensor:
    """Run one source-identical conv/LN/ReLU layer without temporal repadding."""
    conv = layer.conv
    hidden = F.conv2d(
        input_states.to(dtype=conv.weight.dtype),
        conv.weight,
        conv.bias,
        stride=conv.stride,
        padding=(0, 1),
        dilation=conv.dilation,
        groups=conv.groups,
    )
    hidden = layer.act(layer.norm(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())
    return hidden
