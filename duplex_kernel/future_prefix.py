"""Future-prefix buffer and frozen-Qwen incremental speaker runtime.

This module implements:
1. FuturePrefixBuffer: ordered, bounded storage for incremental V9 listener projections
   with contiguous frame IDs, lookahead checking, and horizon-limited visibility.
2. build_causal_prefix_mask: 4D causal attention mask enforcing that audio queries
   can only attend to future prefix conditioning up to min(audio_index + lookahead, available_prefix_end).
3. FuturePrefixQwen: orchestration wrapper around frozen Qwen Talker performing
   exact frame-by-frame forward evaluation via full-sequence recomputation.
"""
from __future__ import annotations

import inspect
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from duplex_kernel.config import QWEN_NATIVE_EXECUTOR_MIN_CACHE_LEN
from duplex_kernel.native_trace import (
    NATIVE_TRACE_CHAIN_ALGORITHM,
    build_trace_chain,
)
from duplex_kernel.qwen_conditioning import (
    QWEN_AUDIO_CONDITIONING_RECIPE,
    aggregate_qwen_code_embeddings,
    build_qwen_audio_conditioning,
    build_qwen_audio_conditioning_sequence,
)
from duplex_kernel.talker_state import (
    DynamicConditioningKV,
    SpeakerSegment,
    TalkerState,
)


@dataclass(frozen=True)
class SpeakerStepOutput:
    """Output of one acoustic or terminal Qwen query."""

    logits: torch.Tensor
    last_hidden: torch.Tensor
    frame_id: int
    visible_prefix_end: int
    is_terminal_query: bool = False


@dataclass(frozen=True)
class ConditioningFrame:
    """One V9 frame with generation/segment fencing metadata."""

    frame_id: int
    h_pfx: torch.Tensor
    produced_at_ns: int
    generation_id: int
    segment_id: str

    def __post_init__(self) -> None:
        tensor = torch.as_tensor(self.h_pfx).detach().clone()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape != (1, 2048):
            raise ValueError(f"h_pfx conditioning deve avere shape (1,2048), got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError("h_pfx conditioning non finito")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "h_pfx", tensor)
        object.__setattr__(self, "produced_at_ns", int(self.produced_at_ns))
        object.__setattr__(self, "generation_id", int(self.generation_id))
        object.__setattr__(self, "segment_id", str(self.segment_id))


class ConditioningFrameBuffer:
    """Thread-safe future-conditioning buffer with generation fencing.

    Frame IDs are global listener IDs. ``begin_segment`` receives the last
    listener frame already consumed as its anchor; the first newly produced
    frame (anchor + 1) becomes relative frame zero for Qwen. A matching frame
    at the anchor is also accepted for callers that explicitly stage one.
    ``wait_for_through`` blocks the speaker only, never the listener scheduler.
    """

    def __init__(self, *, max_frames: int = 512) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames deve essere positivo")
        self.max_frames = int(max_frames)
        self._frames: dict[int, ConditioningFrame] = {}
        self._base_frame_id: int | None = None
        self._generation_id: int | None = None
        self._segment_id: str | None = None
        self._condition = threading.Condition(threading.RLock())
        self._closed = False
        self._history_lost_at: int | None = None
        self._stats = {
            "frames_appended": 0,
            "frames_evicted": 0,
            "frames_rejected_stale": 0,
            "frames_rejected_duplicate": 0,
            "waits": 0,
            "wait_time_ns": 0,
            "stale_consumed": 0,
        }

    @property
    def base_frame_id(self) -> int | None:
        with self._condition:
            return self._base_frame_id

    def _available_through_locked(self) -> int:
        if self._base_frame_id is None:
            return -1
        relative = 0
        while self._base_frame_id + relative in self._frames:
            relative += 1
        return relative - 1

    def _raise_if_history_lost_locked(self) -> None:
        if self._history_lost_at is not None:
            raise RuntimeError(
                "conditioning history irrecoverably lost: "
                f"frame_id={self._history_lost_at} was evicted before a consumer watermark"
            )

    @property
    def available_through(self) -> int:
        with self._condition:
            self._raise_if_history_lost_locked()
            return self._available_through_locked()

    def begin_segment(self, *, generation_id: int, segment_id: str, base_frame_id: int) -> None:
        if base_frame_id < 0:
            raise ValueError("base_frame_id non può essere negativo")
        with self._condition:
            self._base_frame_id = int(base_frame_id)
            self._generation_id = int(generation_id)
            self._segment_id = str(segment_id)
            self._history_lost_at = None
            # Never re-fence frames owned by another generation/segment.  A
            # producer may pre-stage frames only when it already authenticated
            # the exact new fence; all other buffered material is stale.
            self._frames = {
                frame_id: frame
                for frame_id, frame in self._frames.items()
                if frame_id >= self._base_frame_id
                and frame.generation_id == self._generation_id
                and frame.segment_id == self._segment_id
            }
            # begin_response anchors at the last listener frame already
            # consumed. If an authenticated producer staged the next frame
            # before this call, make that first live frame relative ID zero.
            if (
                self._base_frame_id not in self._frames
                and self._base_frame_id + 1 in self._frames
            ):
                self._base_frame_id += 1
            elif self._frames and min(self._frames) > self._base_frame_id + 1:
                # A pre-staged future frame proves that the first live frame is
                # already absent; waiting for it would otherwise time out forever.
                self._history_lost_at = self._base_frame_id + 1
            self._condition.notify_all()

    def append(self, frame: ConditioningFrame) -> bool:
        if not isinstance(frame, ConditioningFrame):
            raise TypeError("frame deve essere ConditioningFrame")
        with self._condition:
            if self._closed:
                return False
            if self._base_frame_id is not None:
                if frame.frame_id < self._base_frame_id:
                    self._stats["frames_rejected_stale"] += 1
                    return False
                if self._generation_id is not None and frame.generation_id != self._generation_id:
                    self._stats["frames_rejected_stale"] += 1
                    return False
                if self._segment_id is not None and frame.segment_id != self._segment_id:
                    self._stats["frames_rejected_stale"] += 1
                    return False
            if frame.frame_id in self._frames:
                self._stats["frames_rejected_duplicate"] += 1
                return False
            self._frames[frame.frame_id] = frame
            # Session begin_response anchors to the most recently consumed
            # listener ID, while the next scheduler tick necessarily has the
            # following ID. Treat that first live frame as Qwen-relative 0;
            # otherwise available_through remains -1 forever waiting for the
            # already-consumed anchor frame to be generated a second time.
            if (
                self._base_frame_id is not None
                and len(self._frames) == 1
                and self._base_frame_id not in self._frames
                and frame.frame_id == self._base_frame_id + 1
            ):
                self._base_frame_id = frame.frame_id
            while len(self._frames) > self.max_frames:
                evicted_id = min(self._frames)
                del self._frames[evicted_id]
                self._stats["frames_evicted"] += 1
                if (
                    self._base_frame_id is not None
                    and evicted_id >= self._base_frame_id
                    and self._history_lost_at is None
                ):
                    # There is no consumer acknowledgement API in this buffer's
                    # caller contract. Treat eviction of any frame from the
                    # active causal prefix as unrecoverable instead of silently
                    # skipping IDs or leaving waiters blocked on the old base.
                    self._history_lost_at = evicted_id
            self._stats["frames_appended"] += 1
            self._condition.notify_all()
            return True

    def append_values(
        self,
        *,
        frame_id: int,
        h_pfx: torch.Tensor,
        produced_at_ns: int,
        generation_id: int,
        segment_id: str,
    ) -> bool:
        return self.append(ConditioningFrame(frame_id, h_pfx, produced_at_ns, generation_id, segment_id))

    def _snapshot_locked(self, through: int | None = None) -> tuple[ConditioningFrame, ...]:
        self._raise_if_history_lost_locked()
        if self._base_frame_id is None:
            return ()
        available_through = self._available_through_locked()
        last = available_through if through is None else min(int(through), available_through)
        if last < 0:
            return ()
        return tuple(self._frames[self._base_frame_id + index] for index in range(last + 1))

    def snapshot(self, *, through: int | None = None) -> tuple[ConditioningFrame, ...]:
        with self._condition:
            return self._snapshot_locked(through)

    def wait_for_through(
        self,
        relative_frame_id: int,
        *,
        stop_check: Any | None = None,
        timeout_s: float | None = None,
    ) -> tuple[ConditioningFrame, ...]:
        if relative_frame_id < 0:
            raise ValueError("relative_frame_id non può essere negativo")
        started = time.monotonic_ns()
        with self._condition:
            self._stats["waits"] += 1
        while True:
            # Do not invoke stop_check while holding the buffer lock.  The
            # callback normally acquires DuplexSession._generation_lock,
            # whereas begin_segment acquires those locks in the opposite
            # order; keeping callbacks outside the condition prevents a
            # generation-transition deadlock.
            if stop_check is not None and stop_check():
                with self._condition:
                    self._stats["wait_time_ns"] += time.monotonic_ns() - started
                return ()
            with self._condition:
                if self._closed:
                    self._stats["wait_time_ns"] += time.monotonic_ns() - started
                    return ()
                try:
                    self._raise_if_history_lost_locked()
                except RuntimeError:
                    self._stats["wait_time_ns"] += time.monotonic_ns() - started
                    raise
                if self._available_through_locked() >= relative_frame_id:
                    result = self._snapshot_locked(relative_frame_id)
                    self._stats["wait_time_ns"] += time.monotonic_ns() - started
                    return result
                if timeout_s is not None and (time.monotonic_ns() - started) / 1_000_000_000 >= timeout_s:
                    self._stats["wait_time_ns"] += time.monotonic_ns() - started
                    return ()
                self._condition.wait(timeout=0.02)

    def invalidate(self) -> None:
        with self._condition:
            self._frames.clear()
            self._base_frame_id = None
            self._generation_id = None
            self._segment_id = None
            self._history_lost_at = None
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._frames.clear()
            self._history_lost_at = None
            self._condition.notify_all()

    def stats(self) -> dict[str, int | None]:
        with self._condition:
            result = dict(self._stats)
            result["frames_buffered"] = len(self._frames)
            result["available_through"] = self._available_through_locked()
            result["base_frame_id"] = self._base_frame_id
            result["history_lost_at_frame_id"] = self._history_lost_at
            return result


class FuturePrefixBuffer:
    """Ordered, bounded storage for incoming V9 listener projection frames."""

    def __init__(
        self,
        hidden_dim: int | None = None,
        max_frames: int = 256,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("hidden_dim deve essere positivo")
        if max_frames <= 0:
            raise ValueError("max_frames deve essere positivo")

        self.hidden_dim: int | None = int(hidden_dim) if hidden_dim is not None else None
        self.max_frames: int = int(max_frames)
        self.device: torch.device = torch.device(device)
        self.dtype: torch.dtype = dtype
        self._frames: list[torch.Tensor] = []

    @property
    def buffered_frames(self) -> int:
        """Number of frames currently stored in the buffer."""
        return len(self._frames)

    @property
    def available_through(self) -> int:
        """Latest contiguous frame ID available (-1 if empty)."""
        return len(self._frames) - 1

    def push(self, frame_id: int, hidden: torch.Tensor) -> None:
        """Push a new frame into the buffer with strict contiguous ID check."""
        expected_id = len(self._frames)
        if frame_id != expected_id:
            raise ValueError(f"frame_id non contiguo: atteso {expected_id}, ricevuto {frame_id}")

        if not isinstance(hidden, torch.Tensor):
            raise TypeError("hidden deve essere un torch.Tensor")

        # Accept (hidden_dim,) or (1, hidden_dim)
        if hidden.dim() == 1:
            h = hidden
        elif hidden.dim() == 2 and hidden.shape[0] == 1:
            h = hidden.squeeze(0)
        else:
            raise ValueError(
                f"forma hidden {tuple(hidden.shape)} non valida, atteso (hidden_dim,) o (1, hidden_dim)"
            )

        if self.hidden_dim is None:
            self.hidden_dim = h.numel()
        elif h.numel() != self.hidden_dim:
            raise ValueError(
                f"dimensione hidden {h.numel()} non valida, atteso hidden_dim={self.hidden_dim} (2048 o custom)"
            )

        if not torch.isfinite(h).all():
            raise ValueError("hidden frame contiene valori non finiti (NaN o Inf)")

        if len(self._frames) >= self.max_frames:
            raise BufferError(f"future prefix buffer pieno (max_frames={self.max_frames})")

        self._frames.append(h.detach().clone())

    def can_emit(self, output_frame_id: int, lookahead_frames: int) -> bool:
        """Check whether enough future prefix frames are available for emitting output_frame_id."""
        if output_frame_id < 0 or lookahead_frames < 0:
            raise ValueError("output_frame_id e lookahead_frames devono essere non negativi")
        if not self._frames:
            return False
        target = output_frame_id + int(lookahead_frames)
        # When target reaches or exceeds bounded max_frames, the full prefix horizon is ready
        target_clamped = min(target, self.max_frames - 1)
        return self.available_through >= target_clamped

    def visible_prefix_end(self, output_frame_id: int, lookahead_frames: int) -> int:
        """Calculate the highest visible prefix frame index for output_frame_id."""
        if output_frame_id < 0 or lookahead_frames < 0:
            raise ValueError("output_frame_id e lookahead_frames devono essere non negativi")
        if self.available_through < 0:
            return -1
        target = output_frame_id + int(lookahead_frames)
        return min(target, self.available_through)

    def as_tensor(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return [1, max_frames, hidden_dim] zero-padded tensor of prefix representations."""
        dev = torch.device(device) if device is not None else self.device
        dt = dtype if dtype is not None else self.dtype
        if self.hidden_dim is None:
            raise ValueError("hidden_dim non ancora determinato")

        tensor = torch.zeros((1, self.max_frames, self.hidden_dim), device=dev, dtype=dt)
        if self._frames:
            stacked = torch.stack(self._frames, dim=0).to(device=dev, dtype=dt)
            tensor[0, : stacked.shape[0], :] = stacked
        return tensor

    def cancel(self) -> None:
        """Discard all pending frames without emission."""
        self._frames.clear()


def build_causal_prefix_mask(
    batch_size: int,
    prefix_start: int,
    prefix_frames: int,
    audio_start: int,
    audio_frames: int,
    available_prefix_end: int,
    lookahead_frames: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    total_length: int | None = None,
) -> torch.Tensor:
    """Build a 4D [B, 1, Q, K] causal mask restricting prefix visibility to lookahead horizon.

    Audio queries j=0..audio_frames-1 can only attend to prefix slots up to:
        visible_end = min(audio_index + lookahead, available_prefix_end).
    Prefix tokens beyond this index are masked with torch.finfo(dtype).min.
    """
    if lookahead_frames < 0:
        raise ValueError("lookahead_frames non può essere negativo")
    if prefix_frames < 0 or audio_frames < 0:
        raise ValueError("prefix_frames e audio_frames devono essere non negativi")
    if prefix_start < 0 or audio_start < prefix_start + prefix_frames:
        raise ValueError("audio_start deve seguire tutti gli slot del prefisso")

    if total_length is None:
        total_length = max(prefix_start + prefix_frames, audio_start + audio_frames)

    mask_value = torch.finfo(dtype).min
    mask = torch.zeros((batch_size, 1, total_length, total_length), device=device, dtype=dtype)

    # 1. Standard causal mask (upper triangular positions are masked)
    causal = torch.triu(torch.ones((total_length, total_length), device=device, dtype=torch.bool), diagonal=1)
    mask[:, 0].masked_fill_(causal, mask_value)

    # 2. Restrict prefix visibility for audio/BOS queries
    prefix_end = prefix_start + prefix_frames
    for audio_index in range(audio_frames):
        query = audio_start + audio_index
        if query >= total_length:
            break
        visible_end = min(audio_index + int(lookahead_frames), int(available_prefix_end))
        if visible_end < prefix_frames - 1:
            first_masked = max(0, visible_end + 1)
            mask[:, 0, query, prefix_start + first_masked : prefix_end] = mask_value

    return mask


# Compatibility export for callers that only need the codebook sum.  Audio
# decoder inputs must use build_qwen_audio_conditioning() below so the text
# term cannot be accidentally omitted.
aggregate_code_embeddings = aggregate_qwen_code_embeddings


def _aggregate_committed_codes(
    talker_model: Any,
    committed: list[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Aggregate a list of committed code frames into sequence embeddings."""
    if not committed:
        return torch.zeros((1, 0, 4), device=device, dtype=dtype)

    first = committed[0]
    if first.is_floating_point():
        # Custom or fake floating-point embeddings
        tensors = [c.to(device=device, dtype=dtype).view(1, 1, -1) for c in committed]
        return torch.cat(tensors, dim=1)

    # Discrete RVQ codebook indices
    codes_list = [c.to(device=device, dtype=torch.long).view(1, 1, -1) for c in committed]
    codes = torch.cat(codes_list, dim=1)
    return aggregate_code_embeddings(talker_model, codes)


class _SlotKVCache:
    """Minimal position-addressable KV cache for Qwen's decoder.

    ``DynamicCache`` only appends tokens, while the duplex sequence contains
    conditioning slots before the audio stream whose values become known over
    time.  This cache keeps fixed-size K/V tensors and writes by the explicit
    absolute ``cache_position`` supplied by the official Qwen forward.  The
    attention mask still decides which slots a query may read.
    """

    is_compileable = False
    # Transformers 5.x inspects this as a per-layer iterable when selecting
    # the mask size, even though our explicit 4-D masks return early.
    is_sliding = [False]

    def __init__(self, *, batch_size: int, max_cache_len: int) -> None:
        if batch_size <= 0 or max_cache_len <= 0:
            raise ValueError("batch_size e max_cache_len devono essere positivi")
        self.batch_size = int(batch_size)
        self.max_cache_len = int(max_cache_len)
        self._keys: dict[int, torch.Tensor] = {}
        self._values: dict[int, torch.Tensor] = {}
        self._max_position = -1
        self._layer_max_positions: dict[int, int] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = None if cache_kwargs is None else cache_kwargs.get("cache_position")
        query_length = int(key_states.shape[-2])
        if positions is None:
            start = self.get_seq_length(layer_idx)
            positions = torch.arange(
                start,
                start + query_length,
                device=key_states.device,
                dtype=torch.long,
            )
        positions = torch.as_tensor(positions, device=key_states.device, dtype=torch.long).reshape(-1)
        if positions.numel() != query_length:
            raise ValueError("cache_position non allineata alla query Qwen")
        position_max: int | None = None
        if positions.numel():
            position_min = int(positions.min().item())
            position_max = int(positions.max().item())
            if position_min < 0 or position_max >= self.max_cache_len:
                raise OverflowError("Qwen slot KV oltre la capacità preallocata")

        layer = int(layer_idx)
        if layer not in self._keys:
            self._keys[layer] = torch.zeros(
                (key_states.shape[0], key_states.shape[1], self.max_cache_len, key_states.shape[-1]),
                device=key_states.device,
                dtype=key_states.dtype,
            )
            self._values[layer] = torch.zeros(
                (value_states.shape[0], value_states.shape[1], self.max_cache_len, value_states.shape[-1]),
                device=value_states.device,
                dtype=value_states.dtype,
            )
        keys = self._keys[layer]
        values = self._values[layer]
        if keys.shape[0] != key_states.shape[0] or keys.shape[1] != key_states.shape[1]:
            raise ValueError("dimensione batch/head incompatibile con Qwen slot KV")
        keys.index_copy_(2, positions, key_states)
        values.index_copy_(2, positions, value_states)
        if position_max is not None:
            layer = int(layer_idx)
            self._max_position = max(self._max_position, position_max)
            self._layer_max_positions[layer] = max(
                self._layer_max_positions.get(layer, -1),
                position_max,
            )
        return keys, values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._layer_max_positions.get(int(layer_idx), -1) + 1

    def get_max_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self.max_cache_len

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        del query_length, layer_idx
        return self.max_cache_len, 0

    def __len__(self) -> int:
        return len(self._keys)

    def reset(self) -> None:
        for tensor in (*self._keys.values(), *self._values.values()):
            tensor.zero_()
        self._max_position = -1
        self._layer_max_positions.clear()


class FuturePrefixQwen:
    """Incremental orchestration wrapper around frozen Qwen Talker.

    Real Qwen Talker models use a position-addressable KV cache: stable ICL and
    text are prefetched once, each available V9 slot is cached once, and each
    audio frame appends exactly one decoder token.  A full-sequence path remains
    only for the small deterministic fake backends used by contract tests.
    """

    audio_conditioning_recipe = QWEN_AUDIO_CONDITIONING_RECIPE

    def __init__(
        self,
        talker: Any,
        lookahead_frames: int,
        max_prefix_frames: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        max_audio_frames: int = 64,
        terminal_query_budget: int = 1,
        runtime_backend: str = "pytorch",
        input_observer: Any | None = None,
        trace_enabled: bool = False,
        trace_sequence_id: int | None = None,
        trace_expected_steps: int | None = None,
    ) -> None:
        if lookahead_frames < 0:
            raise ValueError("lookahead_frames deve essere non negativo")
        if max_prefix_frames <= 0:
            raise ValueError("max_prefix_frames deve essere positivo")
        if max_audio_frames <= 0:
            raise ValueError("max_audio_frames deve essere positivo")
        if terminal_query_budget < 0:
            raise ValueError("terminal_query_budget non può essere negativo")
        if runtime_backend not in {"pytorch", "native"}:
            raise ValueError("runtime_backend deve essere 'pytorch' o 'native'")
        if runtime_backend == "native" and lookahead_frames != 0:
            raise ValueError("il backend Qwen native richiede lookahead_frames=0 per JOINT Q=2")
        if runtime_backend == "native" and terminal_query_budget > 1:
            raise ValueError("il backend Qwen native supporta al massimo una terminal query Q=1")
        if runtime_backend == "native" and max_audio_frames > max_prefix_frames:
            raise ValueError(
                "il backend Qwen native richiede max_audio_frames <= max_prefix_frames"
            )
        if runtime_backend == "native" and dtype != torch.bfloat16:
            raise ValueError("il backend Qwen native richiede dtype=torch.bfloat16")
        if runtime_backend == "native" and torch.device(device).type != "cuda":
            raise ValueError("il backend Qwen native richiede device CUDA")
        if trace_enabled and runtime_backend != "native":
            raise ValueError("il native trace richiede runtime_backend='native'")
        if trace_enabled and (trace_expected_steps is None or int(trace_expected_steps) <= 0):
            raise ValueError("il native trace richiede trace_expected_steps positivo")

        self.talker = talker
        self.lookahead_frames = int(lookahead_frames)
        self.max_prefix_frames = int(max_prefix_frames)
        self.max_audio_frames = int(max_audio_frames)
        self.terminal_query_budget = int(terminal_query_budget)
        self.prefix_capacity = self.max_prefix_frames
        self.acoustic_frame_budget = self.max_audio_frames
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.dtype = dtype

        self.buffer = DynamicConditioningKV(
            max_slots=self.max_prefix_frames,
            hidden_size=2048,
            device=self.device,
            dtype=self.dtype,
        )

        self._current_frame_id = 0
        self._terminal_queries_used = 0
        self._text: str | None = None
        self._icl: torch.Tensor | None = None
        self._text_emb: torch.Tensor | None = None
        self._response_text_conditioning: torch.Tensor | None = None
        self._bos: torch.Tensor | None = None
        self._tts_pad_embed: torch.Tensor | None = None
        self._trailing_text_hidden: torch.Tensor | None = None
        self._committed_codes: list[torch.Tensor] = []
        self._voice_conditioning: Any = None
        self._voice_signature: str | None = None
        self._started = False
        self._slot_kv = self._supports_slot_kv()
        self._slot_cache: _SlotKVCache | None = None
        self._runtime_backend = runtime_backend
        self._input_observer = input_observer
        self._trace_enabled = bool(trace_enabled)
        self._trace_sequence_id = trace_sequence_id
        self._trace_expected_steps = None if trace_expected_steps is None else int(trace_expected_steps)
        self._native_trace_steps: list[dict[str, Any]] = []
        self._native_qwen: Any | None = None
        self._legacy_state = TalkerState()
        self._legacy_generation = 0
        self._active_talker_state: TalkerState | None = None
        self._last_talker_state: TalkerState | None = None
        self._prefix_start = 0
        self._audio_start = 0
        self._cached_prefix_frames = 0
        self._lifecycle = {
            "begin_calls": 0,
            "native_context_creates": 0,
            "native_context_reuses": 0,
            "slot_cache_creates": 0,
            "slot_cache_reuses": 0,
            "stable_prefill_runs": 0,
            "stable_prefill_reuses": 0,
            "native_cache_imports": 0,
        }

    def _resolve_voice_signature(
        self,
        voice_conditioning: Any,
    ) -> str | None:
        signature_fn = getattr(
            self.talker,
            "voice_conditioning_signature",
            None,
        )

        if callable(signature_fn):
            return signature_fn(
                voice_conditioning
            )

        if voice_conditioning is None:
            return None

        # A backend test-double without the production voice protocol may only
        # reuse a prefix when the exact same opaque object is reused.
        return (
            "opaque:"
            f"{id(voice_conditioning)}"
        )

    def _build_prefill_parts(
        self,
        text: str,
        voice_conditioning: Any,
    ):
        builder = getattr(
            self.talker,
            "build_prefill_parts",
        )

        try:
            parameters = inspect.signature(
                builder
            ).parameters
        except (
            TypeError,
            ValueError,
        ):
            parameters = {}

        if "voice_conditioning" in parameters:
            return builder(
                text,
                voice_conditioning=voice_conditioning,
            )

        if voice_conditioning is not None:
            raise TypeError(
                "Talker backend non supporta "
                "voice_conditioning"
            )

        return builder(
            text
        )

    def _supports_slot_kv(self) -> bool:
        """Return whether the backend exposes the native Qwen cache contract."""
        model = getattr(self.talker, "talker", self.talker)
        model = getattr(model, "model", model)
        if not all(hasattr(model, name) for name in ("layers", "rotary_emb", "forward")):
            return False
        try:
            parameters = inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            return False
        return "past_key_values" in parameters and "cache_position" in parameters

    def _position_ids(self, positions: torch.Tensor, batch_size: int) -> torch.Tensor:
        values = positions.to(device=self.device, dtype=torch.long).view(1, -1)
        return values.unsqueeze(0).expand(3, batch_size, -1)

    def _observe_qwen_input(
        self,
        *,
        phase: str,
        frame_id: int | None,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> None:
        observer = self._input_observer
        if observer is None:
            return
        observer(
            {
                "phase": str(phase),
                "frame_id": frame_id,
                "inputs_embeds": inputs_embeds.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
                "position_ids": position_ids.detach().clone(),
                "cache_position": cache_position.detach().clone(),
                "tts_pad_embed": (
                    None
                    if self._tts_pad_embed is None
                    else self._tts_pad_embed.detach().clone()
                ),
                "speaker_conditioning_signature": self._voice_signature,
            }
        )

    def _slot_mask(
        self,
        positions: torch.Tensor,
        *,
        visible_prefix_end: int,
        q_pos_int: int | None = None,
    ) -> torch.Tensor:
        """Build [B,1,Q,K] additive mask for position-addressed cache slots."""
        if self._slot_cache is None:
            raise RuntimeError("slot KV non inizializzata")
        total = self._slot_cache.max_cache_len
        mask_val = torch.finfo(self.dtype).min
        keys = getattr(self, "_keys_arange", None)
        if keys is None or keys.shape[0] != total:
            keys = torch.arange(total, device=self.device, dtype=torch.long)
            self._keys_arange = keys

        if positions.numel() == 1 and q_pos_int is not None:
            allowed = keys <= q_pos_int
            if q_pos_int >= self._audio_start:
                prefix_limit = self._prefix_start + int(visible_prefix_end)
                prefix_slots = (keys >= self._prefix_start) & (keys < self._audio_start)
                allowed &= (~prefix_slots) | (keys <= prefix_limit)
            else:
                allowed &= keys <= q_pos_int

            mask = torch.full(
                (self._icl.shape[0], 1, 1, total),
                mask_val,
                device=self.device,
                dtype=self.dtype,
            )
            mask[0, 0, 0, allowed] = 0
            return mask

        q_positions = positions.to(device=self.device, dtype=torch.long).reshape(-1)
        q_len = q_positions.numel()
        mask = torch.full(
            (self._icl.shape[0], 1, q_len, total),
            mask_val,
            device=self.device,
            dtype=self.dtype,
        )
        for row in range(q_len):
            q_pos = int(q_positions[row].item())
            allowed = keys <= q_pos
            if q_pos >= self._audio_start:
                prefix_limit = self._prefix_start + int(visible_prefix_end)
                prefix_slots = (keys >= self._prefix_start) & (keys < self._audio_start)
                allowed &= (~prefix_slots) | (keys <= prefix_limit)
            else:
                allowed &= keys <= q_pos
            mask[0, 0, row, allowed] = 0
        return mask

    def _prefill_stable(self) -> None:
        """Cache ICL+text once at their native absolute positions."""
        if self._slot_cache is None or self._icl is None or self._text_emb is None:
            raise RuntimeError("prefill Qwen non inizializzato")
        stable_parts = [self._icl, self._text_emb]
        if self._response_text_conditioning is not None:
            stable_parts.append(self._response_text_conditioning)
        stable = torch.cat(stable_parts, dim=1)
        positions = torch.arange(stable.shape[1], device=self.device, dtype=torch.long)
        mask = self._slot_mask(positions, visible_prefix_end=-1)
        position_ids = self._position_ids(positions, stable.shape[0])
        self._observe_qwen_input(
            phase="stable",
            frame_id=None,
            inputs_embeds=stable,
            attention_mask=mask,
            position_ids=position_ids,
            cache_position=positions,
        )
        model = getattr(getattr(self.talker, "talker", self.talker), "model", self.talker)
        with torch.inference_mode():
            model(
                inputs_embeds=stable,
                attention_mask=mask,
                position_ids=position_ids,
                cache_position=positions,
                past_key_values=self._slot_cache,
                use_cache=True,
            )

    def _cache_prefix_through(self, frame_id: int) -> None:
        """Cache newly visible conditioning slots in one causal range.

        Batching the first available horizon is important numerically: it
        keeps the K/V for those mutually causal prefix slots on the same
        native attention path as the full-sequence reference.  Later frames
        append one slot at a time as the lookahead window advances.
        """
        if self._slot_cache is None or self._icl is None:
            return
        frame_id = int(frame_id)
        expected = self._cached_prefix_frames
        if frame_id < expected - 1:
            return
        if self.buffer.available_through < frame_id:
            raise RuntimeError("conditioning KV richiesto prima che il buffer sia contiguo")
        if self._icl.shape[0] != 1:
            raise ValueError("FuturePrefixQwen KV supporta solo batch=1")
        values = self.buffer.slice(expected, frame_id + 1)
        if values.shape != (
            1,
            frame_id - expected + 1,
            self.buffer.hidden_size,
        ):
            raise ValueError(f"conditioning KV inatteso: {tuple(values.shape)}")
        positions = torch.arange(
            self._prefix_start + expected,
            self._prefix_start + frame_id + 1,
            device=self.device,
            dtype=torch.long,
        )
        mask = self._slot_mask(positions, visible_prefix_end=frame_id)
        position_ids = self._position_ids(positions, 1)
        self._observe_qwen_input(
            phase="prefix_cache",
            frame_id=frame_id,
            inputs_embeds=values,
            attention_mask=mask,
            position_ids=position_ids,
            cache_position=positions,
        )
        model = getattr(getattr(self.talker, "talker", self.talker), "model", self.talker)
        with torch.inference_mode():
            model(
                inputs_embeds=values,
                attention_mask=mask,
                position_ids=position_ids,
                cache_position=positions,
                past_key_values=self._slot_cache,
                use_cache=True,
            )
        self._cached_prefix_frames = frame_id + 1

    def _cache_prefix_frame(self, frame_id: int, h_pfx: torch.Tensor) -> None:
        """Compatibility wrapper; slots are now cached on first visibility."""
        del h_pfx
        self._cache_prefix_through(frame_id)

    def _step_frame_slot_kv(self, codec_head: Any) -> SpeakerStepOutput:
        if self._slot_cache is None or self._bos is None or self._icl is None:
            raise RuntimeError("slot KV Qwen non inizializzata")
        frame_id = self._current_frame_id
        visible_end = self.buffer.visible_prefix_end(frame_id, self.lookahead_frames)
        if visible_end < 0:
            raise RuntimeError("nessun conditioning frame disponibile")

        model = getattr(getattr(self.talker, "talker", self.talker), "model", self.talker)

        # In L=0 o quando un singolo prefix slot (visible_end) deve essere cachato
        # contestualmente al frame audio corrente, utilizziamo la modalita' JOINT 1-PASS
        # (un singolo forward pass attraverso i 28 layer di Qwen invece di due sequenziali):
        if (
            self._cached_prefix_frames == visible_end
            and visible_end <= self.buffer.available_through
        ):
            pos_pfx = self._prefix_start + visible_end
            pos_aud = self._audio_start + frame_id

            # Prefix embedding direttamente da buffer
            # Singleton slices keep the backing stride of the fixed storage;
            # collapse the size-one dimensions so the native ABI receives its
            # canonical compact [1, 1, 2048] descriptor without allocating.
            h_pfx = self.buffer.frame(visible_end).view(
                1, 1, self.buffer.hidden_size
            )

            # Audio token embedding
            if frame_id == 0:
                aud_emb = self._bos
            else:
                if self._tts_pad_embed is None:
                    raise RuntimeError("tts_pad_embed non inizializzato per il frame audio")
                aud_emb = build_qwen_audio_conditioning(
                    getattr(self.talker, "talker", self.talker),
                    torch.stack(self._committed_codes[-1:]).view(1, 1, -1),
                    tts_pad_embed=self._tts_pad_embed,
                    trailing_text_hidden=self._trailing_text_hidden,
                    generation_step=frame_id - 1,
                ).to(device=self.device, dtype=self.dtype)

            joint_embeds = torch.cat([h_pfx, aud_emb], dim=1)
            positions = torch.tensor([pos_pfx, pos_aud], device=self.device, dtype=torch.long)

            mask = self._slot_mask(positions, visible_prefix_end=visible_end)
            # Query 0 (prefix) non deve vedere Query 1 (audio)
            mask[0, 0, 0, pos_aud] = torch.finfo(self.dtype).min

            pos_ids = self._position_ids(positions, 1)
            self._observe_qwen_input(
                phase="frame",
                frame_id=frame_id,
                inputs_embeds=joint_embeds,
                attention_mask=mask,
                position_ids=pos_ids,
                cache_position=positions,
            )

            if self._native_qwen is not None:
                if self._cached_prefix_frames != visible_end:
                    raise RuntimeError(
                        "Qwen native richiede una sequenza contigua di frame prefix per JOINT Q=2"
                    )
                hidden = self._native_qwen.step_joint_async(
                    h_pfx.contiguous(),
                    aud_emb.contiguous(),
                    prefix_position=pos_pfx,
                    audio_position=pos_aud,
                )
                if self._trace_enabled:
                    trace_step = self._native_qwen.trace_snapshot(output_hidden=hidden)
                    # Supplementary consumer cross-check: native_abi_debug_copy remains
                    # the attestation source; this digest only proves the copied output is
                    # the same tensor subsequently consumed by the codec head.
                    trace_step["output_hidden_consumed"] = {
                        **self._native_qwen._trace_digest_record(hidden),
                        "source": "native_runtime_consumer",
                    }
                    trace_step["sequence_id"] = self._trace_sequence_id
                    trace_step["frame_id"] = int(frame_id)
                    self._native_trace_steps.append(trace_step)
                logits = codec_head(hidden)
                self._cached_prefix_frames = visible_end + 1
                return SpeakerStepOutput(
                    logits=logits,
                    last_hidden=hidden,
                    frame_id=frame_id,
                    visible_prefix_end=visible_end,
                )

            with torch.inference_mode():
                out = model(
                    inputs_embeds=joint_embeds,
                    attention_mask=mask,
                    position_ids=pos_ids,
                    cache_position=positions,
                    past_key_values=self._slot_cache,
                    use_cache=True,
                )
                hidden = out.last_hidden_state[:, 1:, :]  # audio token hidden state
                logits = codec_head(hidden)

            self._cached_prefix_frames = visible_end + 1

        else:
            # Fallback 2-pass per sequenze multi-slot backlog
            if self._cached_prefix_frames <= visible_end:
                self._cache_prefix_through(visible_end)
            if self._cached_prefix_frames <= visible_end:
                raise RuntimeError(
                    f"conditioning KV insufficiente: cached={self._cached_prefix_frames}, required={visible_end + 1}"
                )
            if frame_id == 0:
                embedding = self._bos
            else:
                if self._tts_pad_embed is None:
                    raise RuntimeError("tts_pad_embed non inizializzato per il frame audio")
                embedding = build_qwen_audio_conditioning(
                    getattr(self.talker, "talker", self.talker),
                    torch.stack(self._committed_codes[-1:]).view(1, 1, -1),
                    tts_pad_embed=self._tts_pad_embed,
                    trailing_text_hidden=self._trailing_text_hidden,
                    generation_step=frame_id - 1,
                ).to(device=self.device, dtype=self.dtype)
            pos_int = self._audio_start + frame_id
            position = torch.tensor([pos_int], device=self.device, dtype=torch.long)
            mask = self._slot_mask(position, visible_prefix_end=visible_end, q_pos_int=pos_int)
            position_ids = self._position_ids(position, 1)
            self._observe_qwen_input(
                phase="audio",
                frame_id=frame_id,
                inputs_embeds=embedding,
                attention_mask=mask,
                position_ids=position_ids,
                cache_position=position,
            )
            with torch.inference_mode():
                out = model(
                    inputs_embeds=embedding,
                    attention_mask=mask,
                    position_ids=position_ids,
                    cache_position=position,
                    past_key_values=self._slot_cache,
                    use_cache=True,
                )
                hidden = out.last_hidden_state
                logits = codec_head(hidden)

        if hidden.shape[1] != 1:
            raise RuntimeError("Qwen KV non ha restituito una sola posizione audio")
        return SpeakerStepOutput(
            logits=logits,
            last_hidden=hidden,
            frame_id=frame_id,
            visible_prefix_end=visible_end,
        )

    def _step_terminal_slot_kv(self, codec_head: Any) -> SpeakerStepOutput:
        """Run the extra terminal-only query after the last acoustic frame."""
        if self._slot_cache is None or self._bos is None or self._icl is None:
            raise RuntimeError("slot KV Qwen non inizializzata")
        if self._current_frame_id <= 0 or not self._committed_codes:
            raise RuntimeError("query terminale richiede almeno un frame acustico committato")
        if self._terminal_queries_used >= self.terminal_query_budget:
            raise RuntimeError("terminal query budget esaurito")

        visible_end = self.buffer.visible_prefix_end(
            self._current_frame_id - 1, self.lookahead_frames
        )
        if visible_end < 0:
            raise RuntimeError("nessun conditioning frame disponibile per EOS")
        if self._cached_prefix_frames <= visible_end:
            self._cache_prefix_through(visible_end)
        if self._cached_prefix_frames <= visible_end:
            raise RuntimeError("conditioning insufficiente per query terminale")
        if self._tts_pad_embed is None:
            raise RuntimeError("tts_pad_embed non inizializzato per la query terminale")

        talker_model = getattr(self.talker, "talker", self.talker)
        embedding = build_qwen_audio_conditioning(
            talker_model,
            torch.stack(self._committed_codes[-1:]).view(1, 1, -1),
            tts_pad_embed=self._tts_pad_embed,
            trailing_text_hidden=self._trailing_text_hidden,
            generation_step=self._current_frame_id - 1,
        ).to(device=self.device, dtype=self.dtype)
        pos_int = self._audio_start + self._current_frame_id
        if pos_int >= self._slot_cache.max_cache_len:
            raise RuntimeError("query terminale oltre la cache dichiarata")
        if self._native_qwen is not None:
            hidden = self._native_qwen.step_terminal_async(
                embedding.contiguous(),
                prefix_visible_position=self._prefix_start + visible_end,
                audio_position=pos_int,
            )
            logits = codec_head(hidden)
            self._terminal_queries_used += 1
            return SpeakerStepOutput(
                logits=logits,
                last_hidden=hidden,
                frame_id=self._current_frame_id,
                visible_prefix_end=visible_end,
                is_terminal_query=True,
            )
        position = torch.tensor([pos_int], device=self.device, dtype=torch.long)
        mask = self._slot_mask(position, visible_prefix_end=visible_end, q_pos_int=pos_int)
        model = getattr(getattr(self.talker, "talker", self.talker), "model", self.talker)
        with torch.inference_mode():
            out = model(
                inputs_embeds=embedding,
                attention_mask=mask,
                position_ids=self._position_ids(position, 1),
                cache_position=position,
                past_key_values=self._slot_cache,
                use_cache=True,
            )
            hidden = out.last_hidden_state
            logits = codec_head(hidden)
        self._terminal_queries_used += 1
        return SpeakerStepOutput(
            logits=logits,
            last_hidden=hidden,
            frame_id=self._current_frame_id,
            visible_prefix_end=visible_end,
            is_terminal_query=True,
        )

    def _required_max_cache_len(self, prefix_start: int) -> int:
        """Cache length = prefix capacity + acoustic budget + terminal queries."""
        return (
            int(prefix_start)
            + self.prefix_capacity
            + self.acoustic_frame_budget
            + self.terminal_query_budget
        )

    @staticmethod
    def _native_capacity(required_max_cache_len: int) -> int:
        return max(
            int(required_max_cache_len),
            int(QWEN_NATIVE_EXECUTOR_MIN_CACHE_LEN),
        )

    def _native_layout_compatible(self, required_max_cache_len: int) -> bool:
        """Return whether the persistent native executor can serve the layout."""
        if self._runtime_backend != "native" or self._native_qwen is None:
            return False
        try:
            layer_count = len(self.talker.talker.model.layers)
        except (AttributeError, TypeError):
            return False
        return (
            self._native_qwen.max_cache_len >= int(required_max_cache_len)
            and self._native_qwen.execute_layers == layer_count
        )

    def _reset_turn_state(self, voice_conditioning: Any) -> None:
        self._voice_conditioning = voice_conditioning
        self.buffer.clear_frames()
        self._current_frame_id = 0
        self._terminal_queries_used = 0
        self._committed_codes.clear()
        self._native_trace_steps.clear()
        self._cached_prefix_frames = 0
        self._started = True

    def _discard_native_executor(self) -> None:
        native = self._native_qwen
        if native is None:
            return
        native.close()
        self._native_qwen = None

    def _abort_failed_begin(self) -> None:
        """Release partial turn state without masking the primary exception."""
        try:
            self._discard_native_executor()
        except Exception:
            # The begin-time failure is the actionable exception; native close
            # is best-effort here and public close() remains failure-visible.
            self._native_qwen = None
        for state in (
            self._active_talker_state,
            self._last_talker_state,
            self._legacy_state,
        ):
            if state is None or state.segment is None:
                continue
            state.segment.static_icl_kv = None
            state.segment.acoustic_kv = None
            state.segment.bos_state = None
        self._slot_cache = None
        self._icl = None
        self._text_emb = None
        self._response_text_conditioning = None
        self._bos = None
        self._tts_pad_embed = None
        self._trailing_text_hidden = None
        self._voice_conditioning = None
        self._voice_signature = None
        self._keys_arange = None
        self._current_frame_id = 0
        self._terminal_queries_used = 0
        self._committed_codes.clear()
        self._cached_prefix_frames = 0
        self.buffer.clear_frames()
        self._active_talker_state = None
        self._last_talker_state = None
        self._legacy_state = TalkerState()
        self._started = False

    def _prepare_segment_layout(
        self,
        text: str,
        voice_conditioning: Any = None,
        trailing_text_hidden: torch.Tensor | None = None,
        response_text_conditioning: torch.Tensor | None = None,
    ) -> None:
        """Prepare the stable segment layout and reusable native state."""
        if not text:
            raise ValueError("text non può essere vuoto")
        self._lifecycle["begin_calls"] += 1
        new_voice_signature = (
            self._resolve_voice_signature(
                voice_conditioning
            )
        )

        new_trailing_text_hidden = None
        if trailing_text_hidden is not None:
            new_trailing_text_hidden = trailing_text_hidden
            if new_trailing_text_hidden.dim() == 2:
                # Public shorthand: one batch with T trailing text states.
                new_trailing_text_hidden = new_trailing_text_hidden.unsqueeze(0)
            if new_trailing_text_hidden.dim() != 3:
                raise ValueError("trailing_text_hidden deve avere shape (B,T,D)")
            new_trailing_text_hidden = new_trailing_text_hidden.to(
                device=self.device,
                dtype=self.dtype,
            )

        new_response_text_conditioning = None
        if response_text_conditioning is not None:
            new_response_text_conditioning = response_text_conditioning
            if new_response_text_conditioning.dim() == 2:
                new_response_text_conditioning = new_response_text_conditioning.unsqueeze(0)
            if (
                new_response_text_conditioning.dim() != 3
                or new_response_text_conditioning.shape[0] != 1
                or new_response_text_conditioning.shape[1] <= 0
            ):
                raise ValueError("response_text_conditioning deve avere shape (1,T,D) con T positivo")
            new_response_text_conditioning = new_response_text_conditioning.to(
                device=self.device,
                dtype=self.dtype,
            )
            if not bool(torch.isfinite(new_response_text_conditioning).all()):
                raise ValueError("response_text_conditioning contiene valori non finiti")

        response_conditioning_unchanged = (
            self._response_text_conditioning is None
            and new_response_text_conditioning is None
        ) or (
            self._response_text_conditioning is not None
            and new_response_text_conditioning is not None
            and torch.equal(self._response_text_conditioning, new_response_text_conditioning)
        )

        if self._runtime_backend == "native" and not self._slot_kv:
            self._started = False
            raise RuntimeError("Qwen native richiede il percorso reale slot-KV del Talker")

        # An identical-text turn keeps the prefetched Python tensors and the
        # position-addressable cache.  import_cache_async clears the native
        # arena and copies only the stable prefix, so dynamic native state from
        # the preceding turn cannot leak into the new one.
        required_max_cache_len = self._required_max_cache_len(self._prefix_start)
        if (
            self._runtime_backend == "native"
            and text == self._text
            and new_voice_signature
                == self._voice_signature
            and response_conditioning_unchanged
            and self._slot_cache is not None
            and self._slot_cache.max_cache_len == required_max_cache_len
            and self._native_layout_compatible(required_max_cache_len)
        ):
            self._trailing_text_hidden = new_trailing_text_hidden
            self._reset_turn_state(voice_conditioning)
            try:
                self._native_qwen.import_cache_async(
                    self._slot_cache,
                    stable_length=self._prefix_start,
                    prefix_start=self._prefix_start,
                    audio_start=self._audio_start,
                )
                if self._trace_enabled:
                    self._native_qwen.set_trace_expected_steps(self._trace_expected_steps)
            except Exception:
                self._abort_failed_begin()
                raise
            self._lifecycle["native_context_reuses"] += 1
            self._lifecycle["slot_cache_reuses"] += 1
            self._lifecycle["stable_prefill_reuses"] += 1
            self._lifecycle["native_cache_imports"] += 1
            return

        # Build the new stable tensors before replacing the current turn.  For
        # a compatible changed-text turn, only the stable KV is recomputed; the
        # native context and its Python slot allocation remain persistent.
        try:
            parts = self._build_prefill_parts(
                text,
                voice_conditioning,
            )
            new_icl = parts[0].to(device=self.device, dtype=self.dtype)
            new_text_emb = parts[1].to(device=self.device, dtype=self.dtype)
            new_bos = parts[2].to(device=self.device, dtype=self.dtype)
            new_tts_pad_embed = parts[3].to(device=self.device, dtype=self.dtype)
            if (
                new_response_text_conditioning is not None
                and new_response_text_conditioning.shape[-1] != new_icl.shape[-1]
            ):
                raise ValueError(
                    "response_text_conditioning width deve coincidere con il Talker: "
                    f"{new_response_text_conditioning.shape[-1]} != {new_icl.shape[-1]}"
                )
        except Exception:
            self._abort_failed_begin()
            raise

        new_prefix_start = int(
            new_icl.shape[1]
            + new_text_emb.shape[1]
            + (
                0
                if new_response_text_conditioning is None
                else new_response_text_conditioning.shape[1]
            )
        )
        new_audio_start = new_prefix_start + self.max_prefix_frames
        new_required_max_cache_len = self._required_max_cache_len(new_prefix_start)
        reuse_native_layout = self._native_layout_compatible(new_required_max_cache_len)
        reuse_slot_cache = (
            self._runtime_backend == "native"
            and self._slot_cache is not None
            and self._slot_cache.max_cache_len == new_required_max_cache_len
        )

        if self._runtime_backend == "native" and not reuse_native_layout:
            # Only an insufficient native arena (or an incompatible model
            # topology) destroys the persistent executor. The Python slot cache
            # is managed independently below.
            try:
                self._discard_native_executor()
            except Exception:
                self._abort_failed_begin()
                raise

        self._voice_conditioning = voice_conditioning
        self._voice_signature = (
            new_voice_signature
        )
        self._text = text
        self._icl = new_icl
        self._text_emb = new_text_emb
        self._response_text_conditioning = new_response_text_conditioning
        self._bos = new_bos
        self._tts_pad_embed = new_tts_pad_embed
        self._trailing_text_hidden = new_trailing_text_hidden
        # Real Qwen Talker uses the fixed P5 hidden size.  The fallback
        # backends used by legacy contract tests may expose a smaller custom
        # hidden size; keep that compatibility without allocating per frame.
        hidden_size = int(self._icl.shape[-1])
        if self.buffer.hidden_size != hidden_size:
            self.buffer = DynamicConditioningKV(
                max_slots=self.max_prefix_frames,
                hidden_size=hidden_size,
                device=self.device,
                dtype=self.dtype,
            )
        self._prefix_start = new_prefix_start
        self._audio_start = new_audio_start
        self._reset_turn_state(voice_conditioning)

        if self._slot_kv and self._icl.shape[0] != 1:
            self._started = False
            raise ValueError("FuturePrefixQwen KV supporta solo batch=1")

        if self._slot_kv:
            if reuse_slot_cache:
                try:
                    self._slot_cache.reset()
                except Exception:
                    self._abort_failed_begin()
                    raise
            else:
                self._slot_cache = _SlotKVCache(
                    batch_size=1,
                    max_cache_len=new_required_max_cache_len,
                )
                if self._runtime_backend == "native":
                    self._lifecycle["slot_cache_creates"] += 1

        if self._slot_kv:
            try:
                self._prefill_stable()
            except Exception:
                self._abort_failed_begin()
                raise
            if self._runtime_backend == "native":
                self._lifecycle["stable_prefill_runs"] += 1
                from duplex_kernel.native_qwen_talker import NativeQwenTalker

                try:
                    if self._native_qwen is None:
                        native_kwargs = {
                            "max_cache_len": self._native_capacity(new_required_max_cache_len),
                            "execute_layers": len(self.talker.talker.model.layers),
                        }
                        if self._trace_enabled:
                            native_kwargs["trace_enabled"] = True
                        self._native_qwen = NativeQwenTalker(
                            self.talker.talker,
                            **native_kwargs,
                        )
                        self._lifecycle["native_context_creates"] += 1
                    self._native_qwen.import_cache_async(
                        self._slot_cache,
                        stable_length=self._prefix_start,
                        prefix_start=self._prefix_start,
                        audio_start=self._audio_start,
                    )
                    if self._trace_enabled:
                        self._native_qwen.set_trace_expected_steps(self._trace_expected_steps)
                    if reuse_native_layout:
                        self._lifecycle["native_context_reuses"] += 1
                    if reuse_slot_cache:
                        self._lifecycle["slot_cache_reuses"] += 1
                    self._lifecycle["native_cache_imports"] += 1
                except Exception:
                    self._abort_failed_begin()
                    raise

    def open_segment(
        self,
        text_plan: str,
        state: TalkerState,
        *,
        segment_id: str,
        generation_id: int,
        voice_conditioning: Any = None,
        trailing_text_hidden: torch.Tensor | None = None,
        response_text_conditioning: torch.Tensor | None = None,
    ) -> SpeakerSegment:
        if state is not self._legacy_state:
            if self.max_prefix_frames != 64:
                raise ValueError("P5 richiede esattamente 64 conditioning slots")
            if self.max_audio_frames > 64:
                raise ValueError("P5 SpeakerSegment non può superare 64 frame")

        if state.segment is not None and not state.segment.closed:
            raise RuntimeError(
                "chiudere lo SpeakerSegment corrente prima di aprirne un altro"
            )
        if (
            self._active_talker_state is not None
            and self._active_talker_state is not state
            and self._active_talker_state.segment is not None
            and not self._active_talker_state.segment.closed
        ):
            raise RuntimeError("esiste già uno SpeakerSegment attivo")

        self._prepare_segment_layout(
            text_plan,
            voice_conditioning=voice_conditioning,
            trailing_text_hidden=trailing_text_hidden,
            response_text_conditioning=response_text_conditioning,
        )

        self.buffer.begin_segment(
            generation_id=generation_id,
            segment_id=segment_id,
        )

        segment = SpeakerSegment(
            segment_id=str(segment_id),
            generation_id=int(generation_id),
            text_plan=str(text_plan),
            static_icl_kv=self._slot_cache,
            conditioning_slots=self.buffer,
            bos_state=self._bos,
            acoustic_kv=self._slot_cache,
        )

        state.open_segment(segment)
        self._active_talker_state = state
        self._last_talker_state = state

        return segment

    def insert_conditioning(
        self,
        h_pfx_t: torch.Tensor,
        frame_id: int,
        state: TalkerState,
    ) -> None:
        if state is not self._active_talker_state:
            raise RuntimeError("TalkerState non attivo")
        segment = state.segment
        if segment is None or segment.closed:
            raise RuntimeError("SpeakerSegment non aperto")

        self.buffer.insert(
            frame_id,
            h_pfx_t,
            generation_id=segment.generation_id,
            segment_id=segment.segment_id,
        )

    def begin(
        self,
        text: str,
        voice_conditioning: Any = None,
        trailing_text_hidden: torch.Tensor | None = None,
        response_text_conditioning: torch.Tensor | None = None,
    ) -> None:
        """Initialize a legacy turn through a synthetic TalkerState."""
        if (
            self._legacy_state.segment is not None
            and not self._legacy_state.segment.closed
        ):
            self.close_segment(
                "legacy_begin",
                self._legacy_state,
            )

        self._legacy_generation += 1

        self.open_segment(
            text,
            self._legacy_state,
            segment_id=f"legacy-{self._legacy_generation}",
            generation_id=self._legacy_generation,
            voice_conditioning=voice_conditioning,
            trailing_text_hidden=trailing_text_hidden,
            response_text_conditioning=response_text_conditioning,
        )

    def push_v9_frame(self, frame_id: int, h_pfx: torch.Tensor) -> None:
        """Push a projected V9 listener frame and cache it exactly once."""
        state = self._active_talker_state
        if state is None:
            raise RuntimeError("SpeakerSegment non aperto")
        self.insert_conditioning(h_pfx, frame_id, state)

    def can_step(self) -> bool:
        """Check whether the next frame can be evaluated under the lookahead policy."""
        if not self._started:
            return False
        return self.buffer.can_emit(self._current_frame_id, self.lookahead_frames)

    def step(self, state: TalkerState) -> SpeakerStepOutput:
        if state is not self._active_talker_state:
            raise RuntimeError("TalkerState non attivo")
        if state.segment is None or state.segment.closed:
            raise RuntimeError("SpeakerSegment non aperto")
        return self.step_frame()

    def step_frame(self, *, allow_reduced_horizon: bool = False) -> SpeakerStepOutput:
        """Evaluate one frame with slot KV (or the fake fallback) and a 4D mask.

        ``allow_reduced_horizon`` is reserved for a clean end-of-segment when
        fewer than ``lookahead_frames`` future conditioning frames exist.  It
        never permits a frame beyond ``available_through`` to become visible.
        """
        active_state = self._active_talker_state
        if (
            not self._started
            or self._icl is None
            or self._text_emb is None
            or self._bos is None
            or active_state is None
            or active_state.segment is None
            or active_state.segment.closed
        ):
            raise RuntimeError("begin() deve essere chiamato prima di step_frame()")

        if not self.can_step():
            if not allow_reduced_horizon or self.buffer.available_through < 0:
                raise RuntimeError(
                    f"Lookahead non sufficiente per frame {self._current_frame_id}: "
                    f"available_through={self.buffer.available_through}, "
                    f"lookahead={self.lookahead_frames}"
                )

        talker_model = getattr(self.talker, "talker", self.talker)
        model = getattr(talker_model, "model", talker_model)
        codec_head = getattr(talker_model, "codec_head", lambda h: h)

        batch_size = self._icl.shape[0]
        if self._current_frame_id >= self.max_audio_frames:
            raise RuntimeError(
                f"audio frame {self._current_frame_id} oltre max_audio_frames={self.max_audio_frames}"
            )
        if self._runtime_backend == "native":
            if self._native_qwen is None:
                raise RuntimeError("Qwen native executor non inizializzato; rifiuto fallback PyTorch")
            if self._cached_prefix_frames != self.buffer.visible_prefix_end(
                self._current_frame_id, self.lookahead_frames
            ):
                raise RuntimeError(
                    "Qwen native richiede esattamente un nuovo frame prefix per ogni step"
                )
        if self._slot_kv:
            output = self._step_frame_slot_kv(codec_head)
            active_state.note_acoustic_append(output.frame_id)
            return output

        prefix = self.buffer.as_tensor()
        if prefix.shape[0] != batch_size:
            prefix = prefix.expand(batch_size, -1, -1)

        icl = self._icl
        text = self._text_emb
        bos = self._bos

        prefix_start = icl.shape[1] + text.shape[1]
        prefix_end = prefix_start + self.max_prefix_frames
        audio_start = prefix_end

        # Reconstruct a fixed-size sequence:
        # [ICL | text | prefix slots | BOS | previous audio inputs | dummies].
        # Keeping the sequence length fixed is important: the Qwen reference
        # path evaluates all output positions in one native-length forward.
        parts = [icl, text, prefix, bos]
        committed_audio_frames = len(self._committed_codes)
        if committed_audio_frames:
            if self._tts_pad_embed is None:
                raise RuntimeError("tts_pad_embed non inizializzato per gli audio input")
            committed_codes = torch.stack(self._committed_codes).view(1, committed_audio_frames, -1)
            audio_inputs = build_qwen_audio_conditioning_sequence(
                talker_model,
                committed_codes,
                tts_pad_embed=self._tts_pad_embed,
                trailing_text_hidden=self._trailing_text_hidden,
            ).to(device=self.device, dtype=self.dtype)
            parts.append(audio_inputs)
        dummy_frames = self.max_audio_frames - 1 - committed_audio_frames
        if dummy_frames < 0:
            raise RuntimeError("troppi code audio per il fixed sequence buffer")
        if dummy_frames:
            parts.append(torch.zeros(
                (batch_size, dummy_frames, bos.shape[-1]),
                device=bos.device,
                dtype=bos.dtype,
            ))

        embeddings = torch.cat(parts, dim=1)
        total_length = embeddings.shape[1]
        audio_frames = self.max_audio_frames

        # 4D Attention mask with lookahead horizon
        attention_mask = build_causal_prefix_mask(
            batch_size=batch_size,
            prefix_start=prefix_start,
            prefix_frames=self.max_prefix_frames,
            audio_start=audio_start,
            audio_frames=audio_frames,
            available_prefix_end=self.buffer.available_through,
            lookahead_frames=self.lookahead_frames,
            dtype=embeddings.dtype,
            device=embeddings.device,
            total_length=total_length,
        )

        position_ids = (
            torch.arange(total_length, device=embeddings.device, dtype=torch.long)
            .view(1, 1, total_length)
            .expand(3, batch_size, total_length)
        )

        with torch.inference_mode():
            out = model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            if hasattr(out, "last_hidden_state"):
                hidden = out.last_hidden_state
            elif isinstance(out, torch.Tensor):
                hidden = out
            else:
                hidden = out[0]

            # The current query is BOS (frame 0) or the corresponding
            # committed-code slot.  Future dummy rows are intentionally not
            # selected and cannot affect it through the causal mask.
            query_position = audio_start + self._current_frame_id
            if hidden.shape[1] == 1:
                # Small test/dummy backends may already return only the query.
                last_hidden = hidden
            else:
                last_hidden = hidden[:, query_position:query_position + 1, :]
                if last_hidden.shape[1] != 1:
                    raise RuntimeError("Qwen non ha restituito la posizione audio richiesta")
            logits = codec_head(last_hidden)

        visible_end = self.buffer.visible_prefix_end(
            self._current_frame_id, self.lookahead_frames
        )

        output = SpeakerStepOutput(
            logits=logits,
            last_hidden=last_hidden,
            frame_id=self._current_frame_id,
            visible_prefix_end=visible_end,
        )
        active_state.note_acoustic_append(output.frame_id)
        return output

    def can_terminal_query(self) -> bool:
        """Whether one terminal query can be issued without an acoustic frame."""
        return bool(
            self._started
            and self._current_frame_id > 0
            and self._current_frame_id <= self.acoustic_frame_budget
            and self._terminal_queries_used < self.terminal_query_budget
        )

    def step_terminal(self) -> SpeakerStepOutput:
        """Evaluate the EOS query after the last committed acoustic frame."""
        active_state = self._active_talker_state
        if (
            not self._started
            or self._icl is None
            or self._text_emb is None
            or active_state is None
            or active_state.segment is None
            or active_state.segment.closed
        ):
            raise RuntimeError("begin() deve essere chiamato prima di step_terminal()")
        if not self.can_terminal_query():
            raise RuntimeError("terminal query non disponibile")
        if self._slot_kv:
            output = self._step_terminal_slot_kv(
                getattr(getattr(self.talker, "talker", self.talker), "codec_head", lambda h: h)
            )
        else:
            talker_model = getattr(self.talker, "talker", self.talker)
            model = getattr(talker_model, "model", talker_model)
            codec_head = getattr(talker_model, "codec_head", lambda h: h)
            prefix = self.buffer.as_tensor()
            if prefix.shape[0] != self._icl.shape[0]:
                prefix = prefix.expand(self._icl.shape[0], -1, -1)
            if self._tts_pad_embed is None:
                raise RuntimeError("tts_pad_embed non inizializzato per la query terminale")
            committed = torch.stack(self._committed_codes).view(1, len(self._committed_codes), -1)
            audio_inputs = build_qwen_audio_conditioning_sequence(
                talker_model,
                committed,
                tts_pad_embed=self._tts_pad_embed,
                trailing_text_hidden=self._trailing_text_hidden,
            ).to(device=self.device, dtype=self.dtype)
            audio_capacity = self.acoustic_frame_budget + self.terminal_query_budget
            dummy_frames = audio_capacity - 1 - len(self._committed_codes)
            if dummy_frames < 0:
                raise RuntimeError("query terminale oltre acoustic/cache budget")
            parts = [self._icl, self._text_emb, prefix, self._bos, audio_inputs]
            if dummy_frames:
                parts.append(torch.zeros(
                    (self._icl.shape[0], dummy_frames, self._bos.shape[-1]),
                    device=self.device,
                    dtype=self.dtype,
                ))
            embeddings = torch.cat(parts, dim=1)
            prefix_start = self._icl.shape[1] + self._text_emb.shape[1]
            audio_start = prefix_start + self.prefix_capacity
            total_length = int(embeddings.shape[1])
            attention_mask = build_causal_prefix_mask(
                batch_size=self._icl.shape[0],
                prefix_start=prefix_start,
                prefix_frames=self.prefix_capacity,
                audio_start=audio_start,
                audio_frames=audio_capacity,
                available_prefix_end=min(
                    self.buffer.available_through,
                    self._current_frame_id - 1 + self.lookahead_frames,
                ),
                lookahead_frames=self.lookahead_frames,
                dtype=embeddings.dtype,
                device=embeddings.device,
                total_length=total_length,
            )
            position_ids = torch.arange(total_length, device=self.device, dtype=torch.long).view(1, 1, total_length).expand(3, self._icl.shape[0], total_length)
            with torch.inference_mode():
                out = model(inputs_embeds=embeddings, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                hidden = out.last_hidden_state
                query_position = audio_start + self._current_frame_id
                last_hidden = hidden[:, query_position : query_position + 1, :]
                output = SpeakerStepOutput(
                    logits=codec_head(last_hidden),
                    last_hidden=last_hidden,
                    frame_id=self._current_frame_id,
                    visible_prefix_end=self.buffer.visible_prefix_end(self._current_frame_id - 1, self.lookahead_frames),
                    is_terminal_query=True,
                )
            self._terminal_queries_used += 1
        if hasattr(active_state, "note_terminal_query"):
            active_state.note_terminal_query(output.frame_id)
        return output

    def native_trace_report(self) -> dict[str, Any]:
        """Return native ABI evidence collected for the current segment."""
        if not self._trace_enabled or self._native_qwen is None:
            raise RuntimeError("native trace non disponibile")
        report = self._native_qwen.trace_info()
        steps = [dict(step) for step in self._native_trace_steps]
        report.update({
            "trace_chain_algorithm": NATIVE_TRACE_CHAIN_ALGORITHM,
            "native_completion": self._native_qwen.trace_completion(),
            "steps": steps,
        })
        report["expected_step_count"] = int(report["native_completion"]["expected_step_count"])
        report["trace_complete"] = bool(report["native_completion"]["complete"])
        report["trace_chain_head_sha256"] = build_trace_chain(steps)
        return report

    def runtime_info(self) -> dict[str, Any]:
        """Return native runtime, segment, and persistent lifecycle information."""
        lifecycle = dict(self._lifecycle)
        state = self._active_talker_state or self._last_talker_state
        segment = None if state is None else state.segment
        conditioning = self.buffer
        p5_info = {
            "segment_open": bool(
                self._active_talker_state is not None
                and segment is not None
                and not segment.closed
            ),
            "segment_id": None if segment is None else segment.segment_id,
            "generation_id": None if segment is None else segment.generation_id,
            "conditioning_slots": conditioning.max_slots,
            "prefix_capacity": self.prefix_capacity,
            "acoustic_frame_budget": self.acoustic_frame_budget,
            "terminal_query_budget": self.terminal_query_budget,
            "terminal_queries_used": self._terminal_queries_used,
            "conditioning_available_through": conditioning.available_through,
            "conditioning_writes": conditioning.writes,
            "conditioning_duplicate_writes": conditioning.duplicate_writes,
            "conditioning_out_of_order_writes": conditioning.out_of_order_writes,
            "acoustic_kv_position": -1 if state is None else state.acoustic_kv_position,
            "terminal_queries": 0 if segment is None else segment.terminal_queries,
            "terminal_query_position": -1 if segment is None else segment.terminal_query_position,
            "terminal_query_frame_id": -1 if segment is None else segment.terminal_query_position,
            "unexpected_cache_resets": 0 if state is None else state.unexpected_cache_resets,
            "duplicate_cache_appends": 0 if state is None else state.duplicate_cache_appends,
            "segment_opens": 0 if state is None else state.segment_opens,
            "segment_closes": 0 if state is None else state.segment_closes,
        }
        if self._native_qwen is None:
            return {
                "runtime_backend": self._runtime_backend,
                "native_enabled": False,
                **p5_info,
                **lifecycle,
            }
        native_runtime = self._native_qwen.runtime_info()
        return {
            "runtime_backend": self._runtime_backend,
            "native_enabled": True,
            **native_runtime,
            **p5_info,
            "native_terminal_query_position": native_runtime.get("terminal_query_position", -1),
            "native_terminal_visible_prefix_position": native_runtime.get(
                "terminal_visible_prefix_position", -1
            ),
            **lifecycle,
        }

    def close_segment(
        self,
        reason: str,
        state: TalkerState,
    ) -> None:
        if state is not self._active_talker_state:
            raise RuntimeError("TalkerState non attivo")
        segment = state.segment
        if segment is None:
            raise RuntimeError("SpeakerSegment non aperto")

        state.close_segment(reason)
        segment.static_icl_kv = None
        segment.acoustic_kv = None
        segment.bos_state = None

        self._started = False
        self._current_frame_id = 0
        self._terminal_queries_used = 0
        self._committed_codes.clear()
        self._cached_prefix_frames = 0
        self.buffer.clear_frames()

        self._active_talker_state = None

    def close(self) -> None:
        """Release native and Python turn-local cache resources."""
        states = (
            self._active_talker_state,
            self._last_talker_state,
            self._legacy_state,
        )
        close_error: BaseException | None = None
        try:
            active = self._active_talker_state
            if active is not None and active.segment is not None and not active.segment.closed:
                self.close_segment("close", active)
            self._discard_native_executor()
        except BaseException as error:
            close_error = error
        finally:
            for state in states:
                if state is None or state.segment is None:
                    continue
                state.segment.static_icl_kv = None
                state.segment.acoustic_kv = None
                state.segment.bos_state = None
            self._slot_cache = None
            self._icl = None
            self._text_emb = None
            self._response_text_conditioning = None
            self._bos = None
            self._tts_pad_embed = None
            self._trailing_text_hidden = None
            self._voice_conditioning = None
            self._keys_arange = None
            self._current_frame_id = 0
            self._terminal_queries_used = 0
            self._committed_codes.clear()
            self._cached_prefix_frames = 0
            self.buffer.clear_frames()
            self._active_talker_state = None
            self._last_talker_state = None
            self._legacy_state = TalkerState()
            self._started = False
        if close_error is not None:
            raise close_error

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def commit_codes(self, codes: Any) -> None:
        """Commit discrete codes for the current frame, advancing the speaker sequence."""
        state = self._active_talker_state
        segment = None if state is None else state.segment
        if (
            not self._started
            or state is None
            or segment is None
            or segment.closed
        ):
            raise RuntimeError("begin() deve essere chiamato prima di commit_codes()")
        if state.acoustic_kv_position != self._current_frame_id:
            raise RuntimeError(
                "commit_codes richiede un forward Qwen riuscito per il frame corrente"
            )
        if segment.frames_generated != self._current_frame_id:
            raise RuntimeError("commit_codes duplicato o non contiguo")

        if isinstance(codes, torch.Tensor):
            c = codes.detach()
        else:
            c = torch.as_tensor(codes)

        c = c.squeeze()
        if c.dim() == 0:
            c = c.view(1)
        if c.dim() != 1 or c.numel() != 16:
            raise ValueError("codes deve contenere esattamente i 16 codebook Qwen")
        if not torch.isfinite(c.to(torch.float32)).all():
            raise ValueError("codes contiene valori non finiti")

        self._committed_codes.append(c.to(device=self.device, dtype=torch.long))
        segment.frames_generated += 1
        self._current_frame_id += 1


__all__ = [
    "ConditioningFrame",
    "ConditioningFrameBuffer",
    "DynamicConditioningKV",
    "FuturePrefixBuffer",
    "FuturePrefixQwen",
    "SpeakerSegment",
    "SpeakerStepOutput",
    "TalkerState",
    "aggregate_code_embeddings",
    "build_causal_prefix_mask",
]
