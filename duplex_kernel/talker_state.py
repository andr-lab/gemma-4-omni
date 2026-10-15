from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


P5_CONDITIONING_SLOTS = 64
P5_TALKER_HIDDEN_SIZE = 2048


def is_same_device(d1: Any, d2: Any) -> bool:
    """True se d1 e d2 fanno riferimento allo stesso device logico o fisico.

    Risolve la discrepanza nativa PyTorch tra torch.device('cuda:0') e torch.device('cuda')
    quando entrambi rappresentano la stessa GPU 0 (o il device CUDA corrente).
    """
    try:
        dev1 = d1 if isinstance(d1, torch.device) else torch.device(d1)
        dev2 = d2 if isinstance(d2, torch.device) else torch.device(d2)
    except Exception:
        return False
    if dev1 == dev2:
        return True
    if dev1.type == dev2.type:
        idx1 = 0 if dev1.index is None else dev1.index
        idx2 = 0 if dev2.index is None else dev2.index
        return idx1 == idx2
    return False


@dataclass
class DynamicConditioningKV:
    max_slots: int = P5_CONDITIONING_SLOTS
    hidden_size: int = P5_TALKER_HIDDEN_SIZE
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32

    _storage: torch.Tensor = field(init=False, repr=False)
    _available_through: int = field(default=-1, init=False)

    generation_id: int | None = field(default=None, init=False)
    segment_id: str | None = field(default=None, init=False)

    writes: int = field(default=0, init=False)
    duplicate_writes: int = field(default=0, init=False)
    out_of_order_writes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if self.device.type == "cuda" and self.device.index is None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        if self.max_slots <= 0:
            raise ValueError("max_slots deve essere positivo")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size deve essere positivo")
        self._storage = torch.zeros(
            (1, self.max_slots, self.hidden_size),
            device=self.device,
            dtype=self.dtype,
        )


    @property
    def available_through(self) -> int:
        return int(self._available_through)

    @property
    def buffered_frames(self) -> int:
        return self._available_through + 1

    def begin_segment(
        self,
        *,
        generation_id: int,
        segment_id: str,
    ) -> None:
        self._storage.zero_()
        self._available_through = -1
        self.generation_id = int(generation_id)
        self.segment_id = str(segment_id)

    def clear_frames(self) -> None:
        self._storage.zero_()
        self._available_through = -1

    def insert(
        self,
        frame_id: int,
        h_pfx: torch.Tensor,
        *,
        generation_id: int,
        segment_id: str,
    ) -> None:
        frame_id = int(frame_id)

        if generation_id != self.generation_id:
            raise RuntimeError("conditioning generation_id stale")
        if str(segment_id) != self.segment_id:
            raise RuntimeError("conditioning segment_id stale")
        if frame_id < 0 or frame_id >= self.max_slots:
            raise IndexError(
                f"conditioning frame {frame_id} fuori [0,{self.max_slots})"
            )

        expected = self._available_through + 1
        if frame_id < expected:
            self.duplicate_writes += 1
            raise RuntimeError(
                f"conditioning duplicato: {frame_id}, expected={expected}"
            )
        if frame_id > expected:
            self.out_of_order_writes += 1
            raise RuntimeError(
                f"conditioning non contiguo: {frame_id}, expected={expected}"
            )

        if not isinstance(h_pfx, torch.Tensor):
            raise TypeError("h_pfx deve essere torch.Tensor")

        value = h_pfx
        if value.ndim == 1:
            value = value.view(1, 1, -1)
        elif value.ndim == 2:
            value = value.unsqueeze(1)

        if value.shape != (1, 1, self.hidden_size):
            raise ValueError(
                f"h_pfx shape inattesa: {tuple(value.shape)}"
            )
        if not is_same_device(value.device, self.device):
            raise ValueError(
                f"h_pfx device {value.device} != {self.device}"
            )
        if value.dtype != self.dtype:
            raise ValueError(
                f"h_pfx dtype {value.dtype} != {self.dtype}"
            )

        self._storage[:, frame_id:frame_id + 1, :].copy_(
            value,
            non_blocking=True,
        )
        self._available_through = frame_id
        self.writes += 1

    def frame(self, frame_id: int) -> torch.Tensor:
        frame_id = int(frame_id)
        if frame_id < 0 or frame_id > self._available_through:
            raise RuntimeError(
                f"conditioning frame {frame_id} non disponibile"
            )
        return self._storage[:, frame_id:frame_id + 1, :]

    def slice(self, start: int, stop: int) -> torch.Tensor:
        if (
            start < 0
            or stop < start
            or stop > self._available_through + 1
        ):
            raise RuntimeError("conditioning slice non disponibile")
        return self._storage[:, start:stop, :]

    def as_tensor(self) -> torch.Tensor:
        return self._storage

    def visible_prefix_end(
        self,
        output_frame_id: int,
        lookahead_frames: int = 0,
    ) -> int:
        if self._available_through < 0:
            return -1
        return min(
            int(output_frame_id) + int(lookahead_frames),
            self._available_through,
        )

    def can_emit(
        self,
        output_frame_id: int,
        lookahead_frames: int = 0,
    ) -> bool:
        target = min(
            int(output_frame_id) + int(lookahead_frames),
            self.max_slots - 1,
        )
        return self._available_through >= target


@dataclass
class SpeakerSegment:
    segment_id: str
    generation_id: int
    text_plan: str

    static_icl_kv: Any | None
    conditioning_slots: DynamicConditioningKV
    bos_state: torch.Tensor | None
    acoustic_kv: Any | None

    frames_generated: int = 0
    emitted_frames: int = 0
    terminal_queries: int = 0
    terminal_query_position: int = -1

    closed: bool = False
    close_reason: str | None = None


@dataclass
class TalkerState:
    segment: SpeakerSegment | None = None
    decoder: Any | None = field(default=None, repr=False)

    acoustic_kv_position: int = -1
    expected_next_audio_position: int = 0

    unexpected_cache_resets: int = 0
    duplicate_cache_appends: int = 0

    segment_opens: int = 0
    segment_closes: int = 0

    def open_segment(self, segment: SpeakerSegment) -> None:
        if self.segment is not None and not self.segment.closed:
            raise RuntimeError("esiste già uno SpeakerSegment aperto")

        self.segment = segment
        self.acoustic_kv_position = -1
        self.expected_next_audio_position = 0
        self.segment_opens += 1

    def note_acoustic_append(self, frame_id: int) -> None:
        frame_id = int(frame_id)

        if frame_id < self.expected_next_audio_position:
            self.duplicate_cache_appends += 1
            raise RuntimeError(
                f"acoustic KV duplicata al frame {frame_id}"
            )

        if frame_id != self.expected_next_audio_position:
            self.unexpected_cache_resets += 1
            raise RuntimeError(
                "acoustic KV position non contigua: "
                f"{frame_id} != {self.expected_next_audio_position}"
            )

        self.acoustic_kv_position = frame_id
        self.expected_next_audio_position = frame_id + 1

    def note_terminal_query(self, frame_id: int) -> None:
        if self.segment is None or self.segment.closed:
            raise RuntimeError("query terminale senza SpeakerSegment aperto")
        if self.segment.terminal_queries >= 1:
            raise RuntimeError("query terminale duplicata")
        self.segment.terminal_queries += 1
        self.segment.terminal_query_position = int(frame_id)

    def close_segment(self, reason: str) -> None:
        if self.segment is None:
            return
        if self.segment.closed:
            return

        self.segment.closed = True
        self.segment.close_reason = str(reason)
        self.segment_closes += 1


__all__ = [
    "DynamicConditioningKV",
    "P5_CONDITIONING_SLOTS",
    "P5_TALKER_HIDDEN_SIZE",
    "SpeakerSegment",
    "TalkerState",
    "is_same_device",
]
