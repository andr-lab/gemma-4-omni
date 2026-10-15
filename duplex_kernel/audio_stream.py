"""P7 audio timeline, decode accumulation, and transport packetization.

The duplex controller still advances at 80 ms.  This module deliberately has
no session or generation ownership: callers provide the canonical
``generation_id``/``speaker_segment_id`` identity at the playout boundary.
Code2Wav aggregation and network packet aggregation are separate state
machines, so neither one can infer the other's chunk size.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Any, Callable, Iterable

import numpy as np
import torch

from duplex_kernel.config import (
    AUDIO_BOUNDARY_CROSSFADE_SAMPLES,
    AUDIO_FRAME_MS,
    AUDIO_FRAME_SAMPLES,
    CODE2WAV_DECODE_FRAMES,
    CODE2WAV_LEFT_CONTEXT_FRAMES,
    CODEC_SAMPLE_RATE,
    NETWORK_PACKET_FRAMES,
)


@dataclass(frozen=True)
class DecodedAudioChunk:
    """PCM for newly decoded codec frames, excluding any left-context PCM."""

    pcm: np.ndarray
    first_frame_id: int
    last_frame_id: int
    decoded_frame_ids: tuple[int, ...]
    context_frame_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        array = np.asarray(self.pcm, dtype=np.float32).reshape(-1)
        frame_ids = tuple(int(value) for value in self.decoded_frame_ids)
        if not frame_ids:
            raise ValueError("DecodedAudioChunk deve contenere almeno un frame")
        if int(self.first_frame_id) != frame_ids[0] or int(self.last_frame_id) != frame_ids[-1]:
            raise ValueError("identity frame chunk incoerente")
        if tuple(frame_ids) != tuple(range(frame_ids[0], frame_ids[-1] + 1)):
            raise ValueError("frame chunk non contigui")
        if array.size != len(frame_ids) * AUDIO_FRAME_SAMPLES:
            raise ValueError("PCM chunk non allineato ai frame codec")
        if not np.isfinite(array).all():
            raise ValueError("PCM chunk non finito")
        object.__setattr__(self, "pcm", np.array(array, dtype=np.float32, copy=True))
        object.__setattr__(self, "first_frame_id", int(self.first_frame_id))
        object.__setattr__(self, "last_frame_id", int(self.last_frame_id))
        object.__setattr__(self, "decoded_frame_ids", frame_ids)
        object.__setattr__(self, "context_frame_ids", tuple(int(value) for value in self.context_frame_ids))

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        return np.asarray(self.pcm, dtype=dtype)

    def __len__(self) -> int:
        return int(self.pcm.size)


class PacketSequenceNumber:
    """Monotonic, thread-safe transport identity for one connection."""

    def __init__(self, start: int = 0) -> None:
        if isinstance(start, bool) or int(start) < 0:
            raise ValueError("start sequence deve essere un intero non negativo")
        self._next = int(start)
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            value = self._next
            self._next += 1
            return value


class AudioFrameTimeline:
    """Sample-exact timeline for PCM frames that were actually emitted.

    A frame is registered only when its PCM crosses the output boundary.  A
    canceled/residual frame is therefore absent rather than represented as a
    synthetic gap.  One timeline instance represents one active segment.
    """

    def __init__(
        self,
        *,
        sample_rate: int = CODEC_SAMPLE_RATE,
        frame_duration_ms: int = AUDIO_FRAME_MS,
    ) -> None:
        if sample_rate <= 0 or frame_duration_ms <= 0:
            raise ValueError("sample_rate e frame_duration_ms devono essere positivi")
        frame_samples = int(sample_rate) * int(frame_duration_ms)
        if frame_samples % 1000:
            raise ValueError("frame_duration_ms deve produrre sample interi")
        self.sample_rate = int(sample_rate)
        self.frame_duration_ms = int(frame_duration_ms)
        self.frame_samples = frame_samples // 1000
        self._generation_id: int | None = None
        self._speaker_segment_id: str | None = None
        self._next_frame_id: int | None = None
        self._next_sample = 0
        self._frame_count = 0

    @property
    def generation_id(self) -> int | None:
        return self._generation_id

    @property
    def speaker_segment_id(self) -> str | None:
        return self._speaker_segment_id

    @property
    def next_frame_id(self) -> int | None:
        return self._next_frame_id

    @property
    def total_samples(self) -> int:
        return self._next_sample

    @property
    def emitted_frames(self) -> int:
        return self._frame_count

    @property
    def pcm_gap_samples(self) -> int:
        return 0

    @property
    def pcm_overlap_samples(self) -> int:
        return 0

    def begin_segment(
        self,
        *,
        generation_id: int,
        speaker_segment_id: str,
        first_frame_id: int = 0,
    ) -> None:
        if generation_id is None or isinstance(generation_id, bool):
            raise ValueError("generation_id è obbligatorio")
        segment = str(speaker_segment_id)
        if not segment:
            raise ValueError("speaker_segment_id è obbligatorio")
        first = int(first_frame_id)
        if first < 0:
            raise ValueError("first_frame_id non può essere negativo")
        if self._frame_count:
            if self._generation_id == int(generation_id) and self._speaker_segment_id == segment:
                return
            raise RuntimeError("timeline contiene già PCM di un altro segment")
        self._generation_id = int(generation_id)
        self._speaker_segment_id = segment
        self._next_frame_id = first
        self._next_sample = 0
        self._frame_count = 0

    def _validate_identity(
        self,
        generation_id: int | None,
        speaker_segment_id: str | None,
        *,
        first_frame_id: int | None = None,
    ) -> None:
        if generation_id is None or isinstance(generation_id, bool):
            raise ValueError("generation_id è obbligatorio")
        segment = str(speaker_segment_id)
        if not segment:
            raise ValueError("speaker_segment_id è obbligatorio")
        if self._generation_id is None:
            self.begin_segment(
                generation_id=int(generation_id),
                speaker_segment_id=segment,
                first_frame_id=0 if first_frame_id is None else int(first_frame_id),
            )
        if self._generation_id != int(generation_id) or self._speaker_segment_id != segment:
            raise ValueError("frame appartiene a un altro generation/segment")

    def append(
        self,
        frame_id: int,
        pcm: np.ndarray,
        *,
        generation_id: int | None,
        speaker_segment_id: str | None,
    ) -> tuple[int, int]:
        frame = int(frame_id)
        self._validate_identity(generation_id, speaker_segment_id, first_frame_id=frame)
        if self._next_frame_id is None:
            self._next_frame_id = frame
        if frame != self._next_frame_id:
            raise ValueError(f"frame timeline atteso {self._next_frame_id}, ricevuto {frame}")
        array = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if array.size != self.frame_samples:
            raise ValueError(f"frame timeline deve contenere {self.frame_samples} sample")
        if not np.isfinite(array).all():
            raise ValueError("frame timeline non finito")
        start = self._next_sample
        end = start + int(array.size)
        self._next_sample = end
        self._next_frame_id += 1
        self._frame_count += 1
        return start, end

    record_emitted_frame = append
    push = append

    def reset(self) -> None:
        self._generation_id = None
        self._speaker_segment_id = None
        self._next_frame_id = None
        self._next_sample = 0
        self._frame_count = 0


@dataclass(frozen=True)
class AudioPacket:
    packet_id: int
    generation_id: int
    speaker_segment_id: str
    first_frame_id: int
    last_frame_id: int
    start_sample: int
    end_sample: int
    pcm: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.packet_id, bool) or int(self.packet_id) < 0:
            raise ValueError("packet_id non valido")
        if self.generation_id is None or isinstance(self.generation_id, bool):
            raise ValueError("generation_id packet obbligatorio")
        segment = str(self.speaker_segment_id)
        if not segment:
            raise ValueError("speaker_segment_id packet obbligatorio")
        first = int(self.first_frame_id)
        last = int(self.last_frame_id)
        start = int(self.start_sample)
        end = int(self.end_sample)
        array = np.asarray(self.pcm, dtype=np.float32).reshape(-1)
        if first < 0 or last < first or start < 0 or end < start:
            raise ValueError("intervallo packet non valido")
        if end - start != int(array.size):
            raise ValueError("end_sample non coincide con la lunghezza PCM")
        if not array.size or not np.isfinite(array).all():
            raise ValueError("packet PCM vuoto o non finito")
        object.__setattr__(self, "packet_id", int(self.packet_id))
        object.__setattr__(self, "generation_id", int(self.generation_id))
        object.__setattr__(self, "speaker_segment_id", segment)
        object.__setattr__(self, "first_frame_id", first)
        object.__setattr__(self, "last_frame_id", last)
        object.__setattr__(self, "start_sample", start)
        object.__setattr__(self, "end_sample", end)
        object.__setattr__(self, "pcm", np.array(array, dtype=np.float32, copy=True))


class DecodeAccumulator:
    """Decode one RVQ frame at a time into bounded Code2Wav chunks."""

    def __init__(
        self,
        decode_fn: Callable[[list[list[int]]], Any],
        *,
        chunk_frames: int = CODE2WAV_DECODE_FRAMES,
        frame_samples: int = AUDIO_FRAME_SAMPLES,
        left_context_frames: int = CODE2WAV_LEFT_CONTEXT_FRAMES,
        crossfade_samples: int = AUDIO_BOUNDARY_CROSSFADE_SAMPLES,
    ) -> None:
        if not callable(decode_fn):
            raise TypeError("decode_fn deve essere callable")
        if not 2 <= int(chunk_frames) <= 4:
            raise ValueError("chunk_frames deve essere tra 2 e 4")
        if int(frame_samples) <= 0:
            raise ValueError("frame_samples deve essere positivo")
        if not 0 <= int(left_context_frames) <= 72:
            raise ValueError("left_context_frames deve essere tra 0 e 72")
        if int(crossfade_samples) < 0 or int(crossfade_samples) > int(frame_samples):
            raise ValueError("crossfade_samples non valido")
        self.decode_fn = decode_fn
        self.chunk_frames = int(chunk_frames)
        self.frame_samples = int(frame_samples)
        self.left_context_frames = int(left_context_frames)
        self.crossfade_samples = int(crossfade_samples)
        self._pending: list[tuple[int, list[int]]] = []
        self._history: deque[tuple[int, list[int]]] = deque(
            maxlen=max(1, self.left_context_frames)
        )
        self._previous_last_sample: float | None = None
        self._next_frame_id = 0
        self._decoded_calls = 0
        self._emitted_samples = 0
        self._cancelled = False

    @property
    def pending_frames(self) -> int:
        return len(self._pending)

    @property
    def decoded_calls(self) -> int:
        return self._decoded_calls

    @property
    def emitted_samples(self) -> int:
        return self._emitted_samples

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def context_frames(self) -> int:
        return len(self._history)

    @staticmethod
    def _normalize_codes(codes: Any) -> list[int]:
        array = np.asarray(codes)
        if array.size != 16:
            raise ValueError("rvq16_codes deve contenere esattamente 16 codebook")
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
                raise ValueError("rvq16_codes deve contenere indici interi")
        try:
            result = [int(value) for value in array.reshape(-1)]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("rvq16_codes non valido") from exc
        if any(value < 0 for value in result):
            raise ValueError("rvq16_codes non può contenere indici negativi")
        return result

    @staticmethod
    def _as_numpy(decoded: Any) -> np.ndarray:
        if hasattr(decoded, "detach"):
            decoded = decoded.detach()
        if hasattr(decoded, "cpu"):
            decoded = decoded.cpu()
        if hasattr(decoded, "numpy"):
            decoded = decoded.numpy()
        return np.asarray(decoded, dtype=np.float32).reshape(-1)

    def _decode_pending(self, *, terminal_fade: bool = False) -> DecodedAudioChunk:
        if not self._pending:
            raise RuntimeError("nessun frame pending")
        new_frames = tuple(frame_id for frame_id, _ in self._pending)
        if self.left_context_frames:
            context = tuple(list(self._history)[-self.left_context_frames:])
        else:
            context = ()
        decode_inputs = [codes for _, codes in context] + [codes for _, codes in self._pending]
        decoded = self._as_numpy(self.decode_fn([list(codes) for codes in decode_inputs]))
        self._decoded_calls += 1
        new_sample_count = len(new_frames) * self.frame_samples
        if decoded.size < new_sample_count:
            raise ValueError(
                "Code2Wav non ha prodotto abbastanza sample PCM per i nuovi frame: "
                f"{decoded.size} < {new_sample_count}"
            )
        if not np.isfinite(decoded).all():
            raise ValueError("Code2Wav ha prodotto PCM non finito")
        pcm = np.array(decoded[-new_sample_count:], dtype=np.float32, copy=True)
        if pcm.size != len(new_frames) * self.frame_samples:
            raise ValueError("PCM emesso non corrisponde ai nuovi frame codec")
        if self._previous_last_sample is not None and self.crossfade_samples:
            count = min(self.crossfade_samples, pcm.size)
            alpha = np.linspace(0.0, 1.0, count, endpoint=True, dtype=np.float32)
            pcm[:count] = self._previous_last_sample * (1.0 - alpha) + pcm[:count] * alpha
        if terminal_fade and self.crossfade_samples:
            count = min(self.crossfade_samples, pcm.size)
            alpha = np.linspace(1.0, 0.0, count, endpoint=True, dtype=np.float32)
            pcm[-count:] *= alpha
        chunk = DecodedAudioChunk(
            pcm=pcm,
            first_frame_id=new_frames[0],
            last_frame_id=new_frames[-1],
            decoded_frame_ids=new_frames,
            context_frame_ids=tuple(frame_id for frame_id, _ in context),
        )
        self._previous_last_sample = float(pcm[-1])
        if self.left_context_frames:
            for item in self._pending:
                self._history.append(item)
        self._pending.clear()
        self._emitted_samples += int(pcm.size)
        return chunk

    def push(self, frame_id: int, rvq16_codes: Any) -> list[DecodedAudioChunk]:
        if self._cancelled:
            raise RuntimeError("DecodeAccumulator è stato cancellato")
        frame = int(frame_id)
        if frame != self._next_frame_id:
            raise ValueError(f"frame accumulator atteso {self._next_frame_id}, ricevuto {frame}")
        codes = self._normalize_codes(rvq16_codes)
        self._pending.append((frame, codes))
        self._next_frame_id += 1
        if len(self._pending) < self.chunk_frames:
            return []
        return [self._decode_pending()]

    def flush(self, *, clean_end: bool = True) -> list[DecodedAudioChunk]:
        if self._cancelled:
            return []
        if not self._pending:
            return []
        if not clean_end:
            self._pending.clear()
            return []
        return [self._decode_pending(terminal_fade=True)]

    def cancel(self) -> None:
        self._pending.clear()
        self._history.clear()
        self._cancelled = True


class DeviceRVQCodeRing:
    """Per-response GPU ring for exact RVQ history and decoder windows."""

    def __init__(
        self,
        *,
        capacity: int = 74,
        history_limit: int = 72,
        device: Any | None = None,
    ) -> None:
        if int(capacity) < 1 or not 0 <= int(history_limit) <= int(capacity):
            raise ValueError("invalid RVQ ring capacity/history limit")
        self.capacity = int(capacity)
        self.history_limit = int(history_limit)
        self._device: torch.device | None = None
        self._storage: torch.Tensor | None = None
        self._frame_ids: list[int | None] = [None] * self.capacity
        self._write = 0
        self._count = 0
        self._last_stream: Any | None = None
        if device is not None:
            self._allocate(torch.device(device))

    def _allocate(self, device: torch.device) -> None:
        requested_device = torch.device(device)
        self._storage = torch.empty(
            (self.capacity, 16), dtype=torch.long, device=requested_device
        )
        # Tensor.device resolves aliases such as "cuda" to the allocated index
        # (for example "cuda:0"), so later comparisons use the concrete device.
        self._device = self._storage.device
        self._last_stream = self._current_stream()

    def _current_stream(self) -> Any | None:
        if self._device is None or self._device.type != "cuda":
            return None
        return torch.cuda.current_stream(self._device)

    def _order_current_stream(self) -> Any | None:
        current = self._current_stream()
        previous = self._last_stream
        if current is not None and previous is not None:
            if int(current.cuda_stream) != int(previous.cuda_stream):
                # Device-side dependency only: do not block the Python caller.
                current.wait_stream(previous)
        return current

    @property
    def count(self) -> int:
        return self._count

    @property
    def device(self) -> torch.device | None:
        return self._device

    def reset(self) -> None:
        self._order_current_stream()
        self._write = 0
        self._count = 0
        self._frame_ids = [None] * self.capacity
        self._last_stream = self._current_stream()

    def append(self, frame_id: int, codes: torch.Tensor) -> None:
        if not isinstance(codes, torch.Tensor):
            raise TypeError("RVQ device ring requires a torch.Tensor")
        row = codes.reshape(-1)
        if row.numel() != 16:
            raise ValueError("RVQ frame must contain exactly 16 codebooks")
        if row.dtype != torch.long:
            row = row.to(dtype=torch.long)
        if self._storage is None:
            self._allocate(row.device)
        if row.device != self._device:
            raise ValueError(f"RVQ frame device {row.device} != ring device {self._device}")
        stream = self._order_current_stream()
        assert self._storage is not None
        self._storage[self._write].copy_(row, non_blocking=True)
        self._frame_ids[self._write] = int(frame_id)
        self._write = (self._write + 1) % self.capacity
        self._count = min(self.capacity, self._count + 1)
        self._last_stream = stream

    def window(self, *, limit: int | None = None) -> tuple[torch.Tensor, tuple[int, ...]]:
        maximum = self.history_limit if limit is None else int(limit)
        if maximum < 0 or maximum > self.capacity:
            raise ValueError("invalid RVQ history window limit")
        take = min(self._count, maximum)
        if take == 0:
            device = self._device or torch.device("cpu")
            return torch.empty((0, 16), dtype=torch.long, device=device), ()
        current = self._order_current_stream()
        assert self._storage is not None
        start = (self._write - take) % self.capacity
        first = min(take, self.capacity - start)
        pieces = [self._storage[start : start + first]]
        indexes = list(range(start, start + first))
        if first < take:
            pieces.append(self._storage[: take - first])
            indexes.extend(range(take - first))
        ordered = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        # Return a private snapshot; later ring writes cannot mutate a decoder input.
        ordered = ordered.clone()
        self._last_stream = current
        ids = tuple(self._frame_ids[index] for index in indexes)
        if any(frame_id is None for frame_id in ids):
            raise RuntimeError("RVQ ring metadata is incomplete")
        return ordered, tuple(int(frame_id) for frame_id in ids)


class TensorDecodeAccumulator:
    """Streaming Code2Wav accumulator that keeps RVQ IDs on their device."""

    def __init__(
        self,
        decode_fn: Callable[[torch.Tensor], Any],
        *,
        history: DeviceRVQCodeRing,
        chunk_frames: int = CODE2WAV_DECODE_FRAMES,
        frame_samples: int = AUDIO_FRAME_SAMPLES,
        left_context_frames: int = CODE2WAV_LEFT_CONTEXT_FRAMES,
        crossfade_samples: int = AUDIO_BOUNDARY_CROSSFADE_SAMPLES,
    ) -> None:
        if not callable(decode_fn):
            raise TypeError("decode_fn deve essere callable")
        if not isinstance(history, DeviceRVQCodeRing):
            raise TypeError("history must be a DeviceRVQCodeRing")
        if not 2 <= int(chunk_frames) <= 4:
            raise ValueError("chunk_frames deve essere tra 2 e 4")
        if int(frame_samples) <= 0:
            raise ValueError("frame_samples deve essere positivo")
        if not 0 <= int(left_context_frames) <= history.capacity:
            raise ValueError("left_context_frames non valido per il ring")
        if int(crossfade_samples) < 0 or int(crossfade_samples) > int(frame_samples):
            raise ValueError("crossfade_samples non valido")
        self.decode_fn = decode_fn
        self.history = history
        self.chunk_frames = int(chunk_frames)
        self.frame_samples = int(frame_samples)
        self.left_context_frames = int(left_context_frames)
        self.crossfade_samples = int(crossfade_samples)
        self._pending: list[tuple[int, torch.Tensor]] = []
        self._previous_last_sample: float | None = None
        self._next_frame_id = 0
        self._decoded_calls = 0
        self._emitted_samples = 0
        self._cancelled = False

    @property
    def pending_frames(self) -> int:
        return len(self._pending)

    @property
    def decoded_calls(self) -> int:
        return self._decoded_calls

    @property
    def emitted_samples(self) -> int:
        return self._emitted_samples

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def context_frames(self) -> int:
        return min(self.history.count, self.left_context_frames)

    @staticmethod
    def _as_numpy(decoded: Any) -> np.ndarray:
        if hasattr(decoded, "detach"):
            decoded = decoded.detach()
        if hasattr(decoded, "cpu"):
            decoded = decoded.cpu()
        if hasattr(decoded, "numpy"):
            decoded = decoded.numpy()
        return np.asarray(decoded, dtype=np.float32).reshape(-1)

    def _decode_pending(self, *, terminal_fade: bool = False) -> DecodedAudioChunk:
        if not self._pending:
            raise RuntimeError("nessun frame pending")
        new_frames = tuple(frame_id for frame_id, _ in self._pending)
        context, context_ids = self.history.window(limit=self.left_context_frames)
        pending_codes = torch.stack([codes for _, codes in self._pending], dim=0)
        decode_codes = torch.cat((context, pending_codes), dim=0).unsqueeze(0)
        decoded = self._as_numpy(self.decode_fn(decode_codes))
        self._decoded_calls += 1
        new_sample_count = len(new_frames) * self.frame_samples
        if decoded.size < new_sample_count:
            raise ValueError(
                "Code2Wav non ha prodotto abbastanza sample PCM per i nuovi frame: "
                f"{decoded.size} < {new_sample_count}"
            )
        if not np.isfinite(decoded).all():
            raise ValueError("Code2Wav ha prodotto PCM non finito")
        pcm = np.array(decoded[-new_sample_count:], dtype=np.float32, copy=True)
        if self._previous_last_sample is not None and self.crossfade_samples:
            count = min(self.crossfade_samples, pcm.size)
            alpha = np.linspace(0.0, 1.0, count, endpoint=True, dtype=np.float32)
            pcm[:count] = self._previous_last_sample * (1.0 - alpha) + pcm[:count] * alpha
        if terminal_fade and self.crossfade_samples:
            count = min(self.crossfade_samples, pcm.size)
            alpha = np.linspace(1.0, 0.0, count, endpoint=True, dtype=np.float32)
            pcm[-count:] *= alpha
        chunk = DecodedAudioChunk(
            pcm=pcm,
            first_frame_id=new_frames[0],
            last_frame_id=new_frames[-1],
            decoded_frame_ids=new_frames,
            context_frame_ids=context_ids,
        )
        self._previous_last_sample = float(pcm[-1])
        for frame_id, codes in self._pending:
            self.history.append(frame_id, codes)
        self._pending.clear()
        self._emitted_samples += int(pcm.size)
        return chunk

    def push(self, frame_id: int, rvq16_codes: torch.Tensor) -> list[DecodedAudioChunk]:
        if self._cancelled:
            raise RuntimeError("TensorDecodeAccumulator è stato cancellato")
        frame = int(frame_id)
        if frame != self._next_frame_id:
            raise ValueError(f"frame accumulator atteso {self._next_frame_id}, ricevuto {frame}")
        if not isinstance(rvq16_codes, torch.Tensor):
            raise TypeError("rvq16_codes deve essere un torch.Tensor")
        row = rvq16_codes.reshape(-1)
        if row.numel() != 16:
            raise ValueError("rvq16_codes deve contenere esattamente 16 codebook")
        if row.dtype != torch.long:
            row = row.to(dtype=torch.long)
        if self.history.device is not None and row.device != self.history.device:
            raise ValueError(f"RVQ frame device {row.device} != ring device {self.history.device}")
        self._pending.append((frame, row))
        self._next_frame_id += 1
        if len(self._pending) < self.chunk_frames:
            return []
        return [self._decode_pending()]

    def flush(self, *, clean_end: bool = True) -> list[DecodedAudioChunk]:
        if self._cancelled:
            return []
        if not self._pending:
            return []
        if not clean_end:
            self._pending.clear()
            return []
        return [self._decode_pending(terminal_fade=True)]

    def cancel(self) -> None:
        self._pending.clear()
        self.history.reset()
        self._cancelled = True


@dataclass(frozen=True)
class _PendingFrame:
    frame_id: int
    pcm: np.ndarray


class NetworkPacketizer:
    """Aggregate emitted 80 ms playout frames into bounded network packets."""

    def __init__(
        self,
        *,
        sequence: PacketSequenceNumber | None = None,
        packet_frames: int = NETWORK_PACKET_FRAMES,
        sample_rate: int = CODEC_SAMPLE_RATE,
        frame_duration_ms: int = AUDIO_FRAME_MS,
    ) -> None:
        if not 2 <= int(packet_frames) <= 4:
            raise ValueError("packet_frames deve essere tra 2 e 4")
        if sample_rate <= 0 or frame_duration_ms <= 0:
            raise ValueError("sample_rate/frame_duration_ms non validi")
        frame_samples = int(sample_rate) * int(frame_duration_ms)
        if frame_samples % 1000:
            raise ValueError("frame_duration_ms deve produrre sample interi")
        self.sequence = sequence or PacketSequenceNumber()
        self.packet_frames = int(packet_frames)
        self.sample_rate = int(sample_rate)
        self.frame_duration_ms = int(frame_duration_ms)
        self.frame_samples = frame_samples // 1000
        self._active_key: tuple[int, str] | None = None
        self._next_frame_id: int | None = None
        self._pending: list[Any] = []
        self._timeline: AudioFrameTimeline | None = None

    @property
    def pending_frames(self) -> int:
        return len(self._pending)

    @property
    def emitted_samples(self) -> int:
        return 0 if self._timeline is None else self._timeline.total_samples

    def _frame_identity(self, frame: Any) -> tuple[int, str, int, np.ndarray]:
        try:
            generation = getattr(frame, "generation_id")
            segment = str(getattr(frame, "speaker_segment_id"))
            frame_id = int(getattr(frame, "frame_id"))
            pcm = np.asarray(getattr(frame, "pcm"), dtype=np.float32).reshape(-1)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError("push richiede un PlayoutFrame con identity completa") from exc
        if generation is None or isinstance(generation, bool) or not segment:
            raise ValueError("PlayoutFrame generation/segment obbligatori")
        if frame_id < 0 or pcm.size != self.frame_samples:
            raise ValueError(f"PlayoutFrame deve contenere {self.frame_samples} sample")
        if not np.isfinite(pcm).all():
            raise ValueError("PlayoutFrame PCM non finito")
        return int(generation), segment, frame_id, np.array(pcm, dtype=np.float32, copy=True)

    def _activate(self, generation: int, segment: str, frame_id: int) -> None:
        self._active_key = (generation, segment)
        self._next_frame_id = frame_id
        self._timeline = AudioFrameTimeline(
            sample_rate=self.sample_rate,
            frame_duration_ms=self.frame_duration_ms,
        )
        self._timeline.begin_segment(
            generation_id=generation,
            speaker_segment_id=segment,
            first_frame_id=frame_id,
        )

    def _emit_pending(self) -> AudioPacket:
        if not self._pending or self._timeline is None or self._active_key is None:
            raise RuntimeError("packetizer senza frame pending")
        generation, segment = self._active_key
        start = self._timeline.total_samples
        for frame in self._pending:
            self._timeline.append(
                frame.frame_id,
                frame.pcm,
                generation_id=generation,
                speaker_segment_id=segment,
            )
        end = self._timeline.total_samples
        first = int(self._pending[0].frame_id)
        last = int(self._pending[-1].frame_id)
        pcm = np.concatenate([frame.pcm for frame in self._pending]).astype(np.float32, copy=False)
        packet = AudioPacket(
            packet_id=self.sequence.next(),
            generation_id=generation,
            speaker_segment_id=segment,
            first_frame_id=first,
            last_frame_id=last,
            start_sample=start,
            end_sample=end,
            pcm=pcm,
        )
        self._pending.clear()
        return packet

    def push(self, frame: Any) -> list[AudioPacket]:
        generation, segment, frame_id, pcm = self._frame_identity(frame)
        key = (generation, segment)
        if self._active_key is None:
            self._activate(generation, segment, frame_id)
        elif self._active_key != key:
            raise ValueError("non è possibile mescolare generation/segment nello stesso packetizer")
        if self._next_frame_id is None or frame_id != self._next_frame_id:
            raise ValueError(f"packetizer frame atteso {self._next_frame_id}, ricevuto {frame_id}")
        # Keep a small normalized frame object so callers cannot mutate queued
        # PCM while the packet is being assembled.
        self._pending.append(_PendingFrame(frame_id=frame_id, pcm=pcm))
        self._next_frame_id += 1
        if len(self._pending) < self.packet_frames:
            return []
        return [self._emit_pending()]

    def flush_segment(self, generation_id: int, speaker_segment_id: str) -> list[AudioPacket]:
        key = (int(generation_id), str(speaker_segment_id))
        if self._active_key != key:
            return []
        packets = [self._emit_pending()] if self._pending else []
        self._active_key = None
        self._next_frame_id = None
        self._pending.clear()
        self._timeline = None
        return packets

    def cancel_segment(self, generation_id: int, speaker_segment_id: str) -> int:
        key = (int(generation_id), str(speaker_segment_id))
        if self._active_key != key:
            return 0
        discarded = len(self._pending)
        self._pending.clear()
        self._active_key = None
        self._next_frame_id = None
        self._timeline = None
        return discarded

    def cancel_all(self) -> int:
        discarded = len(self._pending)
        self._pending.clear()
        self._active_key = None
        self._next_frame_id = None
        self._timeline = None
        return discarded


def iter_audio_packets(
    pcm: np.ndarray,
    *,
    generation_id: int,
    speaker_segment_id: str,
    sequence: PacketSequenceNumber | None = None,
    packet_frames: int = NETWORK_PACKET_FRAMES,
    sample_rate: int = CODEC_SAMPLE_RATE,
) -> Iterable[AudioPacket]:
    """Packetize an already decoded WAV through the canonical P7 path."""
    array = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if array.size == 0 or array.size % AUDIO_FRAME_SAMPLES:
        raise ValueError("PCM fallback non allineato ai frame atomici P7")
    from duplex_kernel.playout import PlayoutFrame

    packetizer = NetworkPacketizer(
        sequence=sequence,
        packet_frames=packet_frames,
        sample_rate=sample_rate,
    )
    for frame_id, start in enumerate(range(0, array.size, AUDIO_FRAME_SAMPLES)):
        yield from packetizer.push(PlayoutFrame(
            frame_id=frame_id,
            generation_id=generation_id,
            speaker_segment_id=speaker_segment_id,
            pcm=array[start:start + AUDIO_FRAME_SAMPLES],
        ))
    yield from packetizer.flush_segment(generation_id, speaker_segment_id)


__all__ = [
    "AUDIO_FRAME_SAMPLES",
    "AudioFrameTimeline",
    "AudioPacket",
    "DecodeAccumulator",
    "DecodedAudioChunk",
    "NetworkPacketizer",
    "PacketSequenceNumber",
    "iter_audio_packets",
]
