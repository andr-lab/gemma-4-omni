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


@dataclass(frozen=True)
class SpeakerStepOutput:
    """Output of a single incremental speaker forward step."""

    logits: torch.Tensor
    last_hidden: torch.Tensor
    frame_id: int
    visible_prefix_end: int


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

    Frame IDs are global listener IDs. ``begin_segment`` establishes the
    relative frame zero consumed by the Qwen speaker; ``wait_for_through``
    blocks the speaker only, never the 80 ms listener scheduler.
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
        self._stats = {
            "frames_appended": 0,
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

    @property
    def available_through(self) -> int:
        with self._condition:
            if self._base_frame_id is None:
                return -1
            relative = 0
            while self._base_frame_id + relative in self._frames:
                relative += 1
            return relative - 1

    def begin_segment(self, *, generation_id: int, segment_id: str, base_frame_id: int) -> None:
        if base_frame_id < 0:
            raise ValueError("base_frame_id non può essere negativo")
        with self._condition:
            self._base_frame_id = int(base_frame_id)
            self._generation_id = int(generation_id)
            self._segment_id = str(segment_id)
            retained = {
                frame_id: frame for frame_id, frame in self._frames.items() if frame_id >= self._base_frame_id
            }
            # Frames produced before the segment started may seed frame zero,
            # but are re-fenced to the new segment before being consumable.
            self._frames = {
                frame_id: ConditioningFrame(
                    frame_id=frame.frame_id,
                    h_pfx=frame.h_pfx,
                    produced_at_ns=frame.produced_at_ns,
                    generation_id=self._generation_id,
                    segment_id=self._segment_id,
                )
                for frame_id, frame in retained.items()
            }
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
            while len(self._frames) > self.max_frames:
                del self._frames[min(self._frames)]
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
        if self._base_frame_id is None:
            return ()
        last = self.available_through if through is None else min(int(through), self.available_through)
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
                if self.available_through >= relative_frame_id:
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
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._frames.clear()
            self._condition.notify_all()

    def stats(self) -> dict[str, int | None]:
        with self._condition:
            result = dict(self._stats)
            result["frames_buffered"] = len(self._frames)
            result["available_through"] = self.available_through
            result["base_frame_id"] = self._base_frame_id
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


def aggregate_code_embeddings(talker_model: Any, codes: torch.Tensor) -> torch.Tensor:
    """Sum embeddings CB0..CB15 matching official Qwen Talker recipe."""
    model = getattr(talker_model, "talker", talker_model)
    if not hasattr(model, "get_input_embeddings"):
        if hasattr(talker_model, "model") and hasattr(talker_model.model, "get_input_embeddings"):
            model = talker_model.model
        else:
            # Fallback for mock/fake models without embedding layers
            hidden_dim = getattr(getattr(model, "config", None), "hidden_size", 4)
            return torch.zeros((codes.shape[0], codes.shape[1], hidden_dim), device=codes.device, dtype=torch.float32)

    parts = [model.get_input_embeddings()(codes[..., :1])]
    for codebook in range(1, min(16, codes.shape[-1])):
        predictor_emb = model.code_predictor.get_input_embeddings()[codebook - 1]
        parts.append(predictor_emb(codes[..., codebook : codebook + 1]))
    return torch.cat(parts, dim=-2).sum(dim=-2)


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
        if positions.numel() and (int(positions.min()) < 0 or int(positions.max()) >= self.max_cache_len):
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
        if positions.numel():
            self._max_position = max(self._max_position, int(positions.max().item()))
        return keys, values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self._max_position + 1

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


class FuturePrefixQwen:
    """Incremental orchestration wrapper around frozen Qwen Talker.

    Real Qwen Talker models use a position-addressable KV cache: stable ICL and
    text are prefetched once, each available V9 slot is cached once, and each
    audio frame appends exactly one decoder token.  A full-sequence path remains
    only for the small deterministic fake backends used by contract tests.
    """

    def __init__(
        self,
        talker: Any,
        lookahead_frames: int,
        max_prefix_frames: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        max_audio_frames: int = 64,
    ) -> None:
        if lookahead_frames < 0:
            raise ValueError("lookahead_frames deve essere non negativo")
        if max_prefix_frames <= 0:
            raise ValueError("max_prefix_frames deve essere positivo")
        if max_audio_frames <= 0:
            raise ValueError("max_audio_frames deve essere positivo")

        self.talker = talker
        self.lookahead_frames = int(lookahead_frames)
        self.max_prefix_frames = int(max_prefix_frames)
        self.max_audio_frames = int(max_audio_frames)
        self.device = torch.device(device)
        self.dtype = dtype

        self.buffer = FuturePrefixBuffer(
            hidden_dim=None,
            max_frames=self.max_prefix_frames,
            device=self.device,
            dtype=self.dtype,
        )

        self._current_frame_id = 0
        self._text: str | None = None
        self._icl: torch.Tensor | None = None
        self._text_emb: torch.Tensor | None = None
        self._bos: torch.Tensor | None = None
        self._committed_codes: list[torch.Tensor] = []
        self._voice_conditioning: Any = None
        self._started = False
        self._slot_kv = self._supports_slot_kv()
        self._slot_cache: _SlotKVCache | None = None
        self._prefix_start = 0
        self._audio_start = 0
        self._cached_prefix_frames = 0

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

    def _slot_mask(
        self,
        positions: torch.Tensor,
        *,
        visible_prefix_end: int,
    ) -> torch.Tensor:
        """Build [B,1,Q,K] additive mask for position-addressed cache slots."""
        if self._slot_cache is None:
            raise RuntimeError("slot KV non inizializzata")
        q_positions = positions.to(device=self.device, dtype=torch.long).reshape(-1)
        total = self._slot_cache.max_cache_len
        mask = torch.full(
            (self._icl.shape[0], 1, q_positions.numel(), total),
            torch.finfo(self.dtype).min,
            device=self.device,
            dtype=self.dtype,
        )
        keys = torch.arange(total, device=self.device, dtype=torch.long)
        for row, query_position in enumerate(q_positions.tolist()):
            allowed = keys <= int(query_position)
            if int(query_position) >= self._audio_start:
                prefix_limit = self._prefix_start + int(visible_prefix_end)
                prefix_slots = (keys >= self._prefix_start) & (keys < self._audio_start)
                allowed &= (~prefix_slots) | (keys <= prefix_limit)
            else:
                # A prefix query can see only the stable prefix and slots up
                # to itself; future slots remain zero and masked.
                allowed &= keys <= int(query_position)
            mask[0, 0, row, allowed] = 0
        return mask

    def _prefill_stable(self) -> None:
        """Cache ICL+text once at their native absolute positions."""
        if self._slot_cache is None or self._icl is None or self._text_emb is None:
            raise RuntimeError("prefill Qwen non inizializzato")
        stable = torch.cat((self._icl, self._text_emb), dim=1)
        positions = torch.arange(stable.shape[1], device=self.device, dtype=torch.long)
        mask = self._slot_mask(positions, visible_prefix_end=-1)
        model = getattr(getattr(self.talker, "talker", self.talker), "model", self.talker)
        with torch.inference_mode():
            model(
                inputs_embeds=stable,
                attention_mask=mask,
                position_ids=self._position_ids(positions, stable.shape[0]),
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
        values = self.buffer.as_tensor(device=self.device, dtype=self.dtype)[
            :, expected:frame_id + 1, :
        ]
        if values.shape != (1, frame_id - expected + 1, 2048):
            raise ValueError(f"conditioning KV inatteso: {tuple(values.shape)}")
        positions = torch.arange(
            self._prefix_start + expected,
            self._prefix_start + frame_id + 1,
            device=self.device,
            dtype=torch.long,
        )
        mask = self._slot_mask(positions, visible_prefix_end=frame_id)
        model = getattr(getattr(self.talker, "talker", self.talker), "model", self.talker)
        with torch.inference_mode():
            model(
                inputs_embeds=values,
                attention_mask=mask,
                position_ids=self._position_ids(positions, 1),
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
        if self._cached_prefix_frames <= visible_end:
            self._cache_prefix_through(visible_end)
        if self._cached_prefix_frames <= visible_end:
            raise RuntimeError(
                f"conditioning KV insufficiente: cached={self._cached_prefix_frames}, required={visible_end + 1}"
            )
        if frame_id == 0:
            embedding = self._bos
        else:
            embedding = _aggregate_committed_codes(
                getattr(self.talker, "talker", self.talker),
                self._committed_codes[-1:],
                device=self.device,
                dtype=self.dtype,
            )
        position = torch.tensor([self._audio_start + frame_id], device=self.device, dtype=torch.long)
        mask = self._slot_mask(position, visible_prefix_end=visible_end)
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
        if hidden.shape[1] != 1:
            raise RuntimeError("Qwen KV non ha restituito una sola posizione audio")
        return SpeakerStepOutput(
            logits=logits,
            last_hidden=hidden,
            frame_id=frame_id,
            visible_prefix_end=visible_end,
        )

    def begin(self, text: str, voice_conditioning: Any = None) -> None:
        """Initialize generation state and prefill parts for a new utterance."""
        if not text:
            raise ValueError("text non può essere vuoto")

        # Voice payload ownership stays with the caller/engine.  The frozen
        # Qwen wrapper is deliberately not mutated here: a shared TalkerModel
        # must not retain one session's voice configuration.
        self._voice_conditioning = voice_conditioning
        self._text = text
        parts = self.talker.build_prefill_parts(text)
        self._icl = parts[0].to(device=self.device, dtype=self.dtype)
        self._text_emb = parts[1].to(device=self.device, dtype=self.dtype)
        self._bos = parts[2].to(device=self.device, dtype=self.dtype)

        # Infer hidden dimension if not already fixed
        if self.buffer.hidden_dim is None:
            self.buffer.hidden_dim = self._icl.shape[-1]

        self.buffer.cancel()
        self._current_frame_id = 0
        self._committed_codes.clear()
        self._prefix_start = int(self._icl.shape[1] + self._text_emb.shape[1])
        self._audio_start = self._prefix_start + self.max_prefix_frames
        self._cached_prefix_frames = 0
        self._slot_cache = None
        self._started = True
        if self._slot_kv:
            if self._icl.shape[0] != 1:
                raise ValueError("FuturePrefixQwen KV supporta solo batch=1")
            self._slot_cache = _SlotKVCache(
                batch_size=1,
                max_cache_len=self._audio_start + self.max_audio_frames,
            )
            self._prefill_stable()

    def push_v9_frame(self, frame_id: int, h_pfx: torch.Tensor) -> None:
        """Push a projected V9 listener frame and cache it exactly once."""
        self.buffer.push(frame_id, h_pfx)

    def can_step(self) -> bool:
        """Check whether the next frame can be evaluated under the lookahead policy."""
        if not self._started:
            return False
        return self.buffer.can_emit(self._current_frame_id, self.lookahead_frames)

    def step_frame(self, *, allow_reduced_horizon: bool = False) -> SpeakerStepOutput:
        """Evaluate one frame with slot KV (or the fake fallback) and a 4D mask.

        ``allow_reduced_horizon`` is reserved for a clean end-of-segment when
        fewer than ``lookahead_frames`` future conditioning frames exist.  It
        never permits a frame beyond ``available_through`` to become visible.
        """
        if not self._started or self._icl is None or self._text_emb is None or self._bos is None:
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
        if self._slot_kv:
            return self._step_frame_slot_kv(codec_head)

        prefix = self.buffer.as_tensor(device=self.device, dtype=self.dtype)
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
            audio_inputs = _aggregate_committed_codes(
                talker_model, self._committed_codes, device=self.device, dtype=self.dtype
            )
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

        return SpeakerStepOutput(
            logits=logits,
            last_hidden=last_hidden,
            frame_id=self._current_frame_id,
            visible_prefix_end=visible_end,
        )

    def commit_codes(self, codes: Any) -> None:
        """Commit discrete codes for the current frame, advancing the speaker sequence."""
        if not self._started:
            raise RuntimeError("begin() deve essere chiamato prima di commit_codes()")

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
        self._current_frame_id += 1


__all__ = [
    "ConditioningFrame",
    "ConditioningFrameBuffer",
    "FuturePrefixBuffer",
    "FuturePrefixQwen",
    "SpeakerStepOutput",
    "aggregate_code_embeddings",
    "build_causal_prefix_mask",
]
