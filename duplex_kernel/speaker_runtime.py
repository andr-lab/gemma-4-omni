from __future__ import annotations

import logging
import math
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Generic, TypeVar

import numpy as np
import torch

from duplex_kernel.future_prefix import FuturePrefixQwen, SpeakerStepOutput
from duplex_kernel.models.talker import TalkerModel
from duplex_kernel.subtalker_graph import SubtalkerGraph

logger = logging.getLogger("duplex_kernel.speaker_runtime")


def _rescale_code2wav_preclamp_graph_safe(
    waveform: torch.Tensor,
    *,
    max_peak: float = 0.95,
) -> torch.Tensor:
    """Apply the eager HAR scalar without host reads so CUDA capture can record it.

    The reduction spans the complete raw decoder output, including left context,
    before Qwen applies its hard clamp. Runtime finiteness is checked after the
    graph returns to the existing Code2Wav decode guard.
    """
    if not isinstance(waveform, torch.Tensor) or not waveform.is_floating_point():
        raise TypeError("pre-clamp waveform must be a floating-point Tensor")
    if not math.isfinite(float(max_peak)) or not 0.0 < float(max_peak) < 1.0:
        raise ValueError("max_peak must be finite and between 0 and 1")
    if waveform.numel() == 0:
        return waveform
    peak = waveform.detach().float().abs().amax()
    scale = torch.clamp(peak / float(max_peak), min=1.0)
    finite_peak = torch.isfinite(peak)
    scale = torch.where(finite_peak, scale, torch.full_like(scale, float("nan")))
    return waveform / scale.to(dtype=waveform.dtype)


@contextmanager
def _capture_code2wav_graph_har_preclamp(model: Any):
    """Install the eager-equivalent pre-clamp transform for one graph capture."""
    from duplex_kernel.models.talker import capture_code2wav_preclamp

    with capture_code2wav_preclamp(
        model, _rescale_code2wav_preclamp_graph_safe
    ):
        yield


_T = TypeVar("_T")
_U = TypeVar("_U")


class _TrackingQueue(queue.Queue):
    """Queue that records its exact peak item count inside ``_put``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_depth = 0

    def _put(self, item: object) -> None:
        super()._put(item)
        if item is not None:
            self.max_depth = max(self.max_depth, len(self.queue))


class AsyncFramePlayout(Generic[_T, _U]):
    """Bounded ordered worker for decoded speaker frames.

    Submission is non-dropping and frame-id ordered.  ``wait_for`` provides a
    causal readiness fence for a listener consuming the previous emitted
    frame, while decode failures are retained and raised on both the fence and
    ``close``.  The worker drains accepted jobs before shutdown.
    """

    def __init__(
        self,
        process: Callable[[_T], _U],
        publish: Callable[[int, _U], None],
        *,
        expected_frames: int,
        max_queue: int = 4,
    ) -> None:
        if expected_frames <= 0 or max_queue <= 0:
            raise ValueError("expected_frames e max_queue devono essere positivi")
        self._process = process
        self._publish = publish
        self._queue: _TrackingQueue = _TrackingQueue(maxsize=int(max_queue))
        self._ready = [threading.Event() for _ in range(int(expected_frames))]
        self._next_submit = 0
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="speaker-playout", daemon=True)
        self._thread.start()

    def _set_error(self, error: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error

    def _get_error(self) -> BaseException | None:
        with self._error_lock:
            return self._error

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            frame_id, payload = job
            try:
                error = self._get_error()
                if error is None:
                    result = self._process(payload)
                    self._publish(frame_id, result)
            except BaseException as error:
                self._set_error(error)
            finally:
                self._ready[frame_id].set()
                self._queue.task_done()

    @property
    def queue_max_depth(self) -> int:
        """Return the exact peak number of queued jobs."""
        return int(self._queue.max_depth)

    @property
    def error(self) -> BaseException | None:
        """Return the first worker failure, if any, without clearing it."""
        return self._get_error()

    def submit(self, frame_id: int, payload: _T) -> None:
        frame_id = int(frame_id)
        if frame_id != self._next_submit:
            raise ValueError(f"speaker playout atteso {self._next_submit}, ricevuto {frame_id}")
        if self._closed:
            raise RuntimeError("speaker playout chiuso")
        if frame_id >= len(self._ready):
            raise ValueError("speaker playout oltre expected_frames")
        error = self._get_error()
        if error is not None:
            raise RuntimeError(f"speaker playout fallito: {error}") from error
        self._queue.put((frame_id, payload))
        self._next_submit += 1

    def submit_nowait(self, frame_id: int, payload: _T) -> bool:
        """Try to enqueue one ordered frame without blocking the caller."""
        frame_id = int(frame_id)
        if frame_id != self._next_submit:
            raise ValueError(f"speaker playout atteso {self._next_submit}, ricevuto {frame_id}")
        if self._closed:
            raise RuntimeError("speaker playout chiuso")
        if frame_id >= len(self._ready):
            raise ValueError("speaker playout oltre expected_frames")
        error = self._get_error()
        if error is not None:
            raise RuntimeError(f"speaker playout fallito: {error}") from error
        try:
            self._queue.put_nowait((frame_id, payload))
        except queue.Full:
            return False
        self._next_submit += 1
        return True

    def complete(self, frame_id: int, result: _U) -> None:
        """Publish a synchronously processed frame and release its fence."""
        frame_id = int(frame_id)
        if frame_id != self._next_submit:
            raise ValueError(f"speaker playout atteso {self._next_submit}, ricevuto {frame_id}")
        if self._closed:
            raise RuntimeError("speaker playout chiuso")
        error = self._get_error()
        if error is not None:
            raise RuntimeError(f"speaker playout fallito: {error}") from error
        self._next_submit += 1
        try:
            self._publish(frame_id, result)
        except BaseException as error:
            self._set_error(error)
            raise
        finally:
            self._ready[frame_id].set()

    def wait_for(self, frame_id: int, *, timeout: float | None = None) -> bool:
        frame_id = int(frame_id)
        if frame_id < 0 or frame_id >= len(self._ready):
            raise ValueError("frame_id speaker playout fuori range")
        if not self._ready[frame_id].wait(timeout=timeout):
            return False
        error = self._get_error()
        if error is not None:
            raise RuntimeError(f"speaker playout fallito: {error}") from error
        return True

    def close(self) -> None:
        if not self._closed:
            self._queue.join()
            self._closed = True
            self._queue.put(None)
            self._thread.join()
        error = self._get_error()
        if error is not None:
            raise RuntimeError(f"speaker playout fallito: {error}") from error


@dataclass(frozen=True)
class SpeakerStepResult:
    """Result of a single 80 ms speaker frame execution."""

    frame_id: int
    cb0_tensor: torch.Tensor
    rvq16_tensor: torch.Tensor
    c0_embed: torch.Tensor
    last_hidden: torch.Tensor
    visible_prefix_end: int
    event: torch.cuda.Event | None = None


class Code2WavFrameGraph:
    """Persistent CUDA graph for the fixed one-frame Code2Wav decode."""

    def __init__(
        self,
        model: Any,
        *,
        device: str = "cuda:0",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")

        self.static_codes: torch.Tensor | None = None
        self.static_audio: torch.Tensor | None = None
        self.graph: torch.cuda.CUDAGraph | None = None

        self.captured = False
        self.capture_error: str | None = None
        self.replays = 0

    @torch.inference_mode()
    def _decode_fixed_frame(
        self,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        """Run the fixed decoder body without Qwen's capture-unsafe trim.

        Qwen's public ``decode`` slices each waveform with a CUDA-derived
        length (``audio[:length]``), which CUDA graph capture rejects. For a
        valid one-frame input the decoder body already returns exactly 1920
        samples, so skipping only that no-op trim is numerically identical.
        """
        decoder = getattr(self.model, "decoder", None)
        if decoder is None:
            return self.model.decode(codes).audio_values[0]
        audio_values = decoder.chunked_decode(
            torch.clamp(codes, min=0).transpose(1, 2)
        ).squeeze(1)
        return audio_values[0]

    @torch.inference_mode()
    def capture(
        self,
        stream: torch.cuda.Stream,
    ) -> "Code2WavFrameGraph":
        if self.captured:
            return self

        if not torch.cuda.is_available():
            raise RuntimeError("Code2WavFrameGraph richiede CUDA")

        self.static_codes = torch.zeros(
            (1, 1, 16),
            dtype=torch.long,
            device=self.device,
        )

        try:
            # Allocate every lazy workspace before graph capture.
            with torch.cuda.stream(stream):
                for _ in range(4):
                    warm = self._decode_fixed_frame(self.static_codes)
                    warm = warm.detach().to(dtype=torch.float32)

            stream.synchronize()

            graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(graph, stream=stream):
                captured_audio = self._decode_fixed_frame(self.static_codes)
                captured_audio = captured_audio.detach().to(
                    dtype=torch.float32
                )

            stream.synchronize()

            if captured_audio.ndim != 1:
                raise RuntimeError(
                    "Code2Wav one-frame output non 1-D: "
                    f"{tuple(captured_audio.shape)}"
                )

            if captured_audio.numel() != 1920:
                raise RuntimeError(
                    "Code2Wav one-frame output inatteso: "
                    f"{captured_audio.numel()} != 1920"
                )

            self.graph = graph
            self.static_audio = captured_audio
            self.captured = True
            self.capture_error = None
            return self

        except Exception as exc:
            self.graph = None
            self.static_audio = None
            self.captured = False
            self.capture_error = repr(exc)
            raise

    @torch.inference_mode()
    def decode(
        self,
        rvq16_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.captured
            or self.graph is None
            or self.static_codes is None
            or self.static_audio is None
        ):
            raise RuntimeError(
                "Code2WavFrameGraph non catturato"
            )

        if rvq16_tensor.numel() != 16:
            raise ValueError(
                "Code2Wav codes devono contenere esattamente 16 valori"
            )

        codes = rvq16_tensor.reshape(1, 1, 16)

        from duplex_kernel.talker_state import is_same_device
        if not is_same_device(codes.device, self.device):
            raise ValueError(
                f"Code2Wav codes device {codes.device} != {self.device}"
            )

        if codes.dtype != torch.long:
            raise ValueError(
                f"Code2Wav codes dtype {codes.dtype} != torch.long"
            )

        self.static_codes.copy_(codes, non_blocking=True)
        self.graph.replay()
        self.replays += 1

        return self.static_audio

    def runtime_info(self) -> dict[str, Any]:
        return {
            "captured": self.captured,
            "capture_error": self.capture_error,
            "replays": self.replays,
            "input_shape": (
                list(self.static_codes.shape)
                if self.static_codes is not None
                else None
            ),
            "output_shape": (
                list(self.static_audio.shape)
                if self.static_audio is not None
                else None
            ),
            "output_dtype": (
                str(self.static_audio.dtype)
                if self.static_audio is not None
                else None
            ),
        }



class Code2WavWindowGraph:
    """Per-response CUDA graph for one exact Code2Wav RVQ window shape."""

    CONTEXT_FRAMES = 72
    NEW_FRAMES = 2
    WINDOW_FRAMES = CONTEXT_FRAMES + NEW_FRAMES
    FRAME_SAMPLES = 1920

    def __init__(
        self,
        model: Any,
        *,
        device: str = "cuda:0",
        window_frames: int = WINDOW_FRAMES,
        allow_odd_window: bool = False,
    ) -> None:
        window_frames_valid = (
            isinstance(window_frames, Integral)
            and not isinstance(window_frames, bool)
            and 1 <= int(window_frames) <= self.WINDOW_FRAMES
            and (allow_odd_window or int(window_frames) % self.NEW_FRAMES == 0)
        )
        if (
            not window_frames_valid
        ):
            expected = (
                f"an integer in [1, {self.WINDOW_FRAMES}]"
                if allow_odd_window
                else f"an even integer in [{self.NEW_FRAMES}, {self.WINDOW_FRAMES}]"
            )
            raise ValueError(f"window_frames must be {expected}")
        self.model = model
        self.window_frames = int(window_frames)
        self.output_crop_frames = min(self.NEW_FRAMES, self.window_frames)
        self.context_frames = self.window_frames - self.output_crop_frames
        self.frame_samples = int(getattr(model, "decode_upsample_rate", self.FRAME_SAMPLES))
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.static_codes: torch.Tensor | None = None
        self.static_audio: torch.Tensor | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_stream: torch.cuda.Stream | None = None
        self.captured = False
        self.capture_error: str | None = None
        self.replays = 0

    @torch.inference_mode()
    def _decode_fixed_window(self, codes: torch.Tensor) -> torch.Tensor:
        decoder = getattr(self.model, "decoder", None)
        if decoder is None:
            raise RuntimeError("Code2Wav graph richiede il decoder chunked_decode")
        audio = decoder.chunked_decode(torch.clamp(codes, min=0).transpose(1, 2))
        audio = audio.squeeze(1)
        if audio.ndim != 2 or audio.shape[0] != 1:
            raise RuntimeError(f"Code2Wav window output inatteso: {tuple(audio.shape)}")
        # Qwen's public decode crops to valid_CB0_frames * decode_upsample_rate.
        # DecodeAccumulator supplies only nonnegative RVQ IDs, so this fixed
        # crop is identical while remaining capture-safe.
        if self.frame_samples != self.FRAME_SAMPLES:
            raise RuntimeError(
                f"Code2Wav frame_samples {self.frame_samples} != {self.FRAME_SAMPLES}"
            )
        expected = self.window_frames * self.frame_samples
        if audio.shape[1] < expected:
            raise RuntimeError(f"Code2Wav output {audio.shape[1]} < fixed crop {expected}")
        return audio[0, :expected]

    @torch.inference_mode()
    def capture(
        self,
        stream: torch.cuda.Stream,
        *,
        pool: Any | None = None,
    ) -> "Code2WavWindowGraph":
        if self.captured:
            return self
        if not torch.cuda.is_available() or self.device.type != "cuda":
            raise RuntimeError("Code2WavWindowGraph richiede CUDA")
        self.static_codes = torch.zeros(
            (1, self.window_frames, 16), dtype=torch.long, device=self.device
        )
        try:
            producer_stream = torch.cuda.current_stream(device=self.device)
            if int(producer_stream.cuda_stream) != int(stream.cuda_stream):
                stream.wait_stream(producer_stream)
            with torch.cuda.stream(stream):
                for _ in range(4):
                    warm = self._decode_fixed_window(self.static_codes)
                    if warm.numel() != self.window_frames * self.FRAME_SAMPLES:
                        raise RuntimeError(
                            "Code2Wav window warmup shape inattesa: "
                            f"{warm.numel()} != {self.window_frames * self.FRAME_SAMPLES}"
                        )
            stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            graph_context = (
                torch.cuda.graph(graph, stream=stream)
                if pool is None
                else torch.cuda.graph(graph, stream=stream, pool=pool)
            )
            with _capture_code2wav_graph_har_preclamp(self.model):
                with graph_context:
                    captured_audio = self._decode_fixed_window(self.static_codes).to(torch.float32)
            stream.synchronize()
            expected = self.window_frames * self.FRAME_SAMPLES
            if captured_audio.numel() != expected:
                raise RuntimeError(
                    f"Code2Wav window graph output {captured_audio.numel()} != {expected}"
                )
            self.graph = graph
            self.static_audio = captured_audio.reshape(-1)
            self.capture_stream = stream
            self.captured = True
            self.capture_error = None
            return self
        except Exception as exc:
            self.graph = None
            self.static_audio = None
            self.capture_stream = None
            self.captured = False
            self.capture_error = repr(exc)
            raise

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if not self.captured or self.graph is None or self.static_codes is None or self.static_audio is None:
            raise RuntimeError("Code2WavWindowGraph non catturato")
        if tuple(codes.shape) != (1, self.window_frames, 16):
            raise ValueError(
                f"Code2Wav window codes shape {tuple(codes.shape)} != (1, {self.window_frames}, 16)"
            )
        if codes.dtype != torch.long:
            raise ValueError(f"Code2Wav codes dtype {codes.dtype} != torch.long")
        if codes.device != self.device:
            raise ValueError(f"Code2Wav codes device {codes.device} != {self.device}")
        self.static_codes.copy_(codes, non_blocking=True)
        self.graph.replay()
        self.replays += 1
        # Copy the newly decoded tail before this slot can replay again. EOS
        # flush can contain one pending frame, so a one-frame exact graph owns
        # and returns that full frame without padding or touching RVQ history.
        return self.static_audio[-self.output_crop_frames * self.FRAME_SAMPLES :].clone()

    def supports_shape(self, codes: torch.Tensor) -> bool:
        return bool(
            self.captured
            and tuple(codes.shape) == (1, self.window_frames, 16)
        )

    @property
    def static_buffers_bytes(self) -> int:
        total = 0
        for tensor in (self.static_codes, self.static_audio):
            if tensor is not None:
                total += int(tensor.numel() * tensor.element_size())
        return total

    def runtime_info(self) -> dict[str, Any]:
        return {
            "captured": self.captured,
            "capture_error": self.capture_error,
            "replays": self.replays,
            "window_frames": self.window_frames,
            "context_frames": self.context_frames,
            "new_frames": self.output_crop_frames,
            "input_shape": list(self.static_codes.shape) if self.static_codes is not None else None,
            "output_shape": list(self.static_audio.shape) if self.static_audio is not None else None,
            "output_dtype": str(self.static_audio.dtype) if self.static_audio is not None else None,
            "static_buffers_bytes": self.static_buffers_bytes,
            "capture_stream_device": (
                str(self.capture_stream.device) if self.capture_stream is not None else None
            ),
        }


class Code2WavWindowGraphFamily:
    """Atomic per-response family of independent, exact-shape Code2Wav graphs.

    Every window owns its graph and static I/O, while all shapes within this
    response family share the first graph's private CUDA allocator pool. This
    follows PyTorch's documented independent-output sharing condition: these
    decoder graphs consume independent RVQ inputs, do not consume one another's
    outputs, and must never replay concurrently. Each response slot owns a
    separate family/pool because separate slots can run concurrently. The
    returned audio crop is cloned before a later replay can reuse pool memory.
    """

    WINDOW_FRAMES = tuple(
        range(
            Code2WavWindowGraph.NEW_FRAMES,
            Code2WavWindowGraph.WINDOW_FRAMES + 1,
            Code2WavWindowGraph.NEW_FRAMES,
        )
    )
    ODD_TAIL_WINDOW_FRAMES = tuple(range(1, Code2WavWindowGraph.WINDOW_FRAMES + 1))

    def __init__(
        self,
        model: Any,
        *,
        device: str = "cuda:0",
        include_odd_tail: bool = False,
    ) -> None:
        self.model = model
        self.include_odd_tail = bool(include_odd_tail)
        self.window_frames = (
            self.ODD_TAIL_WINDOW_FRAMES if self.include_odd_tail else self.WINDOW_FRAMES
        )
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._graphs: dict[int, Code2WavWindowGraph] = {}
        self.captured = False
        self.capture_error: str | None = None

    @property
    def graphs(self) -> dict[int, Code2WavWindowGraph]:
        """Return a snapshot so callers cannot mutate the captured family."""
        return dict(self._graphs)

    @property
    def static_buffers_bytes(self) -> int:
        return sum(graph.static_buffers_bytes for graph in self._graphs.values())

    @torch.inference_mode()
    def capture(self) -> "Code2WavWindowGraphFamily":
        if self.captured:
            return self
        if not torch.cuda.is_available() or self.device.type != "cuda":
            raise RuntimeError("Code2WavWindowGraphFamily richiede CUDA")

        pending: dict[int, Code2WavWindowGraph] = {}
        shared_pool: Any | None = None
        try:
            capture_stream = torch.cuda.Stream(device=self.device, priority=0)
            for window_frames in self.window_frames:
                graph = Code2WavWindowGraph(
                    self.model,
                    device=str(self.device),
                    window_frames=window_frames,
                    allow_odd_window=(
                        self.include_odd_tail and window_frames % 2 != 0
                    ),
                )
                # Captures are sequential on this family-local stream. The
                # first graph creates the per-response private pool; later
                # exact shapes share it under the independent-output contract
                # documented by PyTorch's CUDA graph API.
                graph.capture(capture_stream, pool=shared_pool)
                capture_stream.synchronize()
                if not graph.captured:
                    raise RuntimeError(
                        f"Code2Wav graph for window {window_frames} did not capture"
                    )
                pending[window_frames] = graph
                if shared_pool is None:
                    if graph.graph is None:
                        raise RuntimeError(
                            f"Code2Wav graph for window {window_frames} has no CUDA graph handle"
                        )
                    shared_pool = graph.graph.pool()
            if tuple(pending) != self.window_frames:
                raise RuntimeError("Code2Wav graph family is incomplete")
        except Exception as exc:
            self._graphs = {}
            self.captured = False
            self.capture_error = repr(exc)
            raise

        self._graphs = pending
        self.captured = True
        self.capture_error = None
        return self

    def supports_shape(self, codes: torch.Tensor) -> bool:
        shape = tuple(codes.shape)
        if (
            not self.captured
            or len(shape) != 3
            or shape[0] != 1
            or shape[2] != 16
            or shape[1] not in self._graphs
        ):
            return False
        return self._graphs[shape[1]].supports_shape(codes)

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if not self.supports_shape(codes):
            raise ValueError(
                "Code2Wav graph family does not support exact codes shape "
                f"{tuple(codes.shape)}"
            )
        return self._graphs[int(codes.shape[1])].decode(codes)

    def runtime_info(self) -> dict[str, Any]:
        return {
            "captured": self.captured,
            "capture_error": self.capture_error,
            "window_frames": list(self._graphs),
            "graph_count": len(self._graphs),
            "odd_tail_enabled": self.include_odd_tail,
            "static_buffers_bytes": self.static_buffers_bytes,
            "graphs": {
                str(window): graph.runtime_info()
                for window, graph in self._graphs.items()
            },
        }

class SpeakerRuntime:
    """Unified production speaker runtime managing Qwen talker and SubTalker graphs.

    All token selection and tensor operations remain GPU-resident to eliminate
    host PCIe round-trips and CPU blocking.
    """

    def __init__(
        self,
        speaker: FuturePrefixQwen,
        subtalker_graph: SubtalkerGraph,
        talker: TalkerModel,
        *,
        device: str = "cuda:0",
        code2wav_graph: Code2WavFrameGraph | None = None,
    ) -> None:
        self.speaker = speaker
        self.subtalker_graph = subtalker_graph
        self.talker = talker
        self.device = device
        self.code2wav_graph = code2wav_graph
        self._current_frame_id = 0

    def set_code2wav_graph(
        self,
        graph: Code2WavFrameGraph,
    ) -> None:
        if not isinstance(graph, Code2WavFrameGraph):
            raise TypeError("graph deve essere Code2WavFrameGraph")
        if not graph.captured:
            raise RuntimeError("Code2Wav graph non catturato")
        self.code2wav_graph = graph

    @property
    def current_frame_id(self) -> int:
        return self._current_frame_id

    def begin(self, text: str, voice_conditioning: dict[str, Any] | None = None) -> None:
        """Initialize a new utterance turn."""
        self.speaker.begin(text, voice_conditioning=voice_conditioning)
        self._current_frame_id = 0

    def push_v9_frame(self, frame_id: int, conditioning_frame: torch.Tensor) -> None:
        """Push causal V9 conditioning vector (1, 2048) or (2048,)."""
        self.speaker.push_v9_frame(frame_id, conditioning_frame)

    @torch.inference_mode()
    def step(
        self,
        conditioning_frame: torch.Tensor | None = None,
        *,
        frame_id: int | None = None,
        record_event: bool = True,
        do_sample: bool = True,
    ) -> SpeakerStepResult:
        """Execute one 80 ms speaker tick entirely on GPU.

        1. Pushes conditioning vector (if provided).
        2. Evaluates Qwen slot-KV forward step.
        3. Selects CB0 token directly in VRAM.
        4. Replays 15 SubTalker CUDA graphs for CB1..15.
        5. Assembles codes_16 on CUDA and commits to speaker state.
        6. Records optional CUDA event for asynchronous synchronization.
        """
        fid = self._current_frame_id if frame_id is None else int(frame_id)

        if conditioning_frame is not None:
            self.speaker.push_v9_frame(fid, conditioning_frame)

        # 1. Forward Qwen talker backbone
        step_out: SpeakerStepOutput = self.speaker.step_frame()

        # 2. Select CB0 token on GPU
        # In greedy or sampling mode, keep selection entirely on CUDA.
        # Note: Code2Wav RVQ Codebook 0 has strictly 2048 entries.
        cb0_logits = step_out.logits[:, :1, :2048]
        if not do_sample:
            c0_tensor = cb0_logits.argmax(dim=-1).squeeze().view(1)
        else:
            # TopK/TopP GPU sampling on CB0 logits (strictly in [0, 2047])
            topk_vals, topk_idx = torch.topk(cb0_logits, k=20, dim=-1)
            probs = torch.softmax(topk_vals / 0.80, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            mask = (cum - probs) > 0.90
            probs[mask] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            sampled_idx = torch.multinomial(probs.view(1, -1), num_samples=1)
            c0_tensor = topk_idx.view(1, -1).gather(1, sampled_idx).squeeze().view(1)
        c0_tensor = torch.clamp(c0_tensor, 0, 2047)

        # 3. GPU embedding projection
        c0_embed = self.talker.talker.get_input_embeddings()(c0_tensor.view(1, 1))

        # 4. Replay 15 SubTalker CUDA graphs
        sub_codes = self.subtalker_graph.generate(
            step_out.last_hidden,
            c0_token=c0_tensor,
            c0_embed=c0_embed,
            do_sample=do_sample,
            return_tensor=True,
        )

        # 5. Assemble codes_16 on GPU and commit
        if not isinstance(sub_codes, torch.Tensor):
            sub_codes = torch.tensor(sub_codes, device=self.device, dtype=torch.long)
        codes_16 = torch.cat([c0_tensor.view(1), sub_codes.view(-1)], dim=0)
        codes_16 = torch.clamp(codes_16, 0, 2047)
        self.speaker.commit_codes(codes_16)

        # 6. Record CUDA event
        event: torch.cuda.Event | None = None
        if record_event and torch.cuda.is_available():
            event = torch.cuda.Event(enable_timing=True)
            event.record()

        self._current_frame_id = fid + 1

        return SpeakerStepResult(
            frame_id=fid,
            cb0_tensor=c0_tensor,
            rvq16_tensor=codes_16,
            c0_embed=c0_embed,
            last_hidden=step_out.last_hidden,
            visible_prefix_end=step_out.visible_prefix_end,
            event=event,
        )

    def decode_frame_tensor(
        self,
        rvq16_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Decode one frame and keep the waveform on CUDA for stream handoff."""
        if self.talker.code2wav is None:
            raise RuntimeError(
                "Code2Wav non presente nel TalkerModel"
            )

        if self.code2wav_graph is not None:
            return self.code2wav_graph.decode(rvq16_tensor)

        codes = rvq16_tensor.view(1, 1, 16)
        with torch.inference_mode():
            return (
                self.talker.code2wav.decode(codes)
                .audio_values[0]
                .detach()
                .to(dtype=torch.float32)
            )

    def decode_frame(self, rvq16_tensor: torch.Tensor) -> np.ndarray:
        """Decode single 16-code frame to 24 kHz audio (1920 samples)."""
        return self.decode_frame_tensor(rvq16_tensor).cpu().numpy().astype(np.float32)


__all__ = [
    "Code2WavFrameGraph",
    "SpeakerStepResult",
    "SpeakerRuntime",
]
