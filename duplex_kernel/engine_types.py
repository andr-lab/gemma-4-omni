from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable

import torch


class ListenerBacklogAdmission(Enum):
    ADMITTED = "admitted"
    ABORTED = "aborted"


class GenerationAdmissionAborted(RuntimeError):
    """A realtime generation became stale before its first audio prefill."""


class _CompiledTalkerDecodeError(RuntimeError):
    """A compiled Qwen Q1 call violated its startup/runtime contract."""


class _NativeTalkerDecodeError(RuntimeError):
    """The opted-in native audio-Q1 path violated its runtime contract."""


def _cuda_tensors_in(value: Any):
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _cuda_tensors_in(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _cuda_tensors_in(item)


def _wait_for_cuda_producer_event(tensor: Any, producer_event: Any | None) -> None:
    """Order a CUDA tensor handoff on the consumer stream without host sync."""
    if (
        producer_event is None
        or not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cuda"
        or not torch.cuda.is_available()
    ):
        return
    consumer_stream = torch.cuda.current_stream(device=tensor.device)
    consumer_stream.wait_event(producer_event)
    tensor.record_stream(consumer_stream)


def _fence_current_cuda_stage(device: Any) -> None:
    selected = torch.device(device)
    if selected.type != 'cuda' or not torch.cuda.is_available():
        return
    event = torch.cuda.Event(enable_timing=False)
    event.record(torch.cuda.current_stream(device=selected))
    event.synchronize()


def _private_thinker_response_stream(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run one synchronous Thinker response on an isolated CUDA stream."""
    @wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        device = torch.device(getattr(self, "device", "cpu"))
        if device.type != "cuda" or not torch.cuda.is_available():
            return method(self, *args, **kwargs)
        stream = torch.cuda.Stream(device=device, priority=0)
        producer_tensors = list(_cuda_tensors_in(kwargs.get("input_audio")))
        if producer_tensors:
            producer_stream = torch.cuda.current_stream(device=device)
            if int(producer_stream.cuda_stream) != int(stream.cuda_stream):
                stream.wait_stream(producer_stream)
            for tensor in producer_tensors:
                tensor.record_stream(stream)
        try:
            with torch.cuda.stream(stream):
                return method(self, *args, **kwargs)
        finally:
            # The result may contain hidden/KV tensors consumed by the Talker.
            # Fence only this response stream so those outputs survive handoff.
            stream.synchronize()
    return wrapped


def _private_talker_slot_stream(method: Callable[..., Any]) -> Callable[..., Any]:
    """Route every GPU op in a leased Talker response through its slot stream."""
    @wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        talk_context = kwargs.get("talk_context")
        if talk_context is None:
            return method(self, *args, **kwargs)
        stream = getattr(talk_context, "cuda_stream", None)
        if stream is None:
            device = torch.device(getattr(self, "device", "cpu"))
            if device.type == "cuda" and torch.cuda.is_available() and hasattr(talk_context, "cuda_stream"):
                raise RuntimeError("leased Talker slot has no private CUDA response stream")
            return method(self, *args, **kwargs)
        if not torch.cuda.is_available():
            return method(self, *args, **kwargs)
        stream_device = torch.device(stream.device)
        engine_device = torch.device(self.device)
        if stream_device.type != "cuda" or engine_device.type != "cuda":
            raise RuntimeError("leased Talker response stream must match a CUDA engine")
        expected_index = engine_device.index
        actual_index = stream_device.index
        if expected_index is not None and actual_index not in (None, expected_index):
            raise RuntimeError("leased Talker response stream is on the wrong CUDA device")
        input_tensors = list(_cuda_tensors_in((args, kwargs)))
        caller_stream = torch.cuda.current_stream(device=stream_device)
        if int(caller_stream.cuda_stream) != int(stream.cuda_stream):
            # One producer handoff at response entry; all subsequent stages and
            # Q1 graph replay share the leased slot stream.
            stream.wait_stream(caller_stream)
        for tensor in input_tensors:
            if tensor.device.type == "cuda" and (
                actual_index is None or tensor.device.index in (None, actual_index)
            ):
                tensor.record_stream(stream)
        with torch.cuda.stream(stream):
            return method(self, *args, **kwargs)
    return wrapped


class _NativeQ1GraphResponseLease:
    """Response-scoped fence over a persistent slot-owned Q1 CUDA graph."""

    def __init__(self, executor: Any, slot_id: int) -> None:
        self._executor = executor
        self.slot_id = int(slot_id)
        self._closed = False

    def step_audio_graph_async(self, audio_embed: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("native Q1 graph response lease is closed")
        return self._executor.step_audio_graph_async(audio_embed)

    def q1_graph_info(self) -> dict[str, Any]:
        return dict(self._executor.q1_graph_info())

    def close(self) -> None:
        if not self._closed:
            self._executor.fence_q1_graph()
            self._closed = True


class _QwenDynamicCacheQ1(torch.nn.Module):
    """Compile only Qwen's one-audio-query forward while preserving its cache."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds: torch.Tensor, past_key_values: Any):
        output = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return output.last_hidden_state[:, -1:, :], output.past_key_values


@dataclass(frozen=True)
class V9PrefixSequence:
    """Verified runtime sequence produced from Gemma pre-conv hidden states."""

    h_pfx: torch.Tensor
    raw_stage: str
    anchor: torch.Tensor

    def __post_init__(self) -> None:
        if self.raw_stage != "pre_causal_conv":
            raise ValueError("V9 runtime richiede hidden Gemma pre_causal_conv")
        if self.h_pfx.ndim != 3 or self.h_pfx.shape[-1] != 2048:
            raise ValueError("h_pfx runtime deve avere shape (B,T,2048)")
        if self.anchor.ndim != 2 or self.anchor.shape[-1] != 1536:
            raise ValueError("anchor runtime deve avere shape (B,1536)")
