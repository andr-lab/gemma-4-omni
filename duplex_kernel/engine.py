# -*- coding: utf-8 -*-
"""
duplex_kernel/engine.py — PipelineEngine (Thinker → V9 → Talker → Code2Wav).

Kernel narrow autocontenuto (ADR-029, ispirato a DwarfStar): carica i modelli
addestrati direttamente (bf16, niente unsloth/LoRA/bnb per il thinker) e
orchestra la pipeline in un processo singolo con KV persistente per sessione.

Flusso (ADR-022/023/024):
1. Thinker (Gemma 4 E2B bf16) genera testo + hidden states (1536).
2. Hidden → V9: h_tx (media) + h_raw_style (16 frame) → causal_conv → h_pfx
   (1, 16, 2048). h_pfx è anche h_t per la turnhead (media sui frame).
3. Talker Qwen3-TTS: prefill [ICL | h_pfx | BOS] (prefix KV persistente),
   loop autoregressivo CB0 (top_k=20, T=0.6) + sub-talker (top_k=20, T=0.8).
4. Code2Wav: RVQ → PCM 24kHz, trim + peak norm.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import re
import threading
import time
from contextlib import nullcontext
from enum import Enum
from functools import wraps
from copy import deepcopy
from numbers import Integral
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
from duplex_kernel import config
import torch.nn.functional as F

from duplex_kernel.audio_stream import (
    AUDIO_BOUNDARY_CROSSFADE_SAMPLES,
    AUDIO_FRAME_SAMPLES,
    CODE2WAV_DECODE_FRAMES,
    DecodeAccumulator,
    DeviceRVQCodeRing,
    TensorDecodeAccumulator,
)
from duplex_kernel.config import (
    ARIA_FRAMES_PER_WORD,
    ARIA_MAX_FRAMES,
    ARIA_MIN_FRAMES,
    ARIA_OVERHEAD_FRAMES,
    ARIA_BACKCHANNEL_FRAMES_PER_WORD,
    ARIA_BACKCHANNEL_OVERHEAD_FRAMES,
    ARIA_BACKCHANNEL_MAX_FRAMES,
    CODE2WAV_LEFT_CONTEXT_FRAMES,
    FROZEN_QWEN_CONDITIONING_WAIT_TIMEOUT_S,
    USER_SAMPLE_RATE,
    SPEAKER_REFERENCE_SHA256,
    SPEAKER_REFERENCE_WAV,
    FROZEN_QWEN_LOOKAHEAD_FRAMES,
    TALKER_EOS_TOKEN_ID,
    V9_PROFILE_CAUSAL_L0,
    V9_PROFILE_QUALITY_FINAL_ORACLE,
    V9_QUALITY_RECIPE,
    V9_CAUSAL_L0_RECIPE,
)
from duplex_kernel.future_prefix import FuturePrefixQwen
from duplex_kernel.dataset_eos_contract import DEFAULT_WINDOW_FRAMES, TERMINAL_QUERY_BUDGET
from duplex_kernel.models.talker import load_talker, rescale_code2wav_preclamp
from duplex_kernel.speaker_runtime import (
    Code2WavWindowGraph,
    Code2WavWindowGraphFamily,
)
from duplex_kernel.models.thinker import load_thinker, normalize_eos_token_ids
from duplex_kernel.models.v9 import IncrementalV9Projector, load_v9_conditioning
from duplex_kernel.thinker_cache import (
    CachePrefixCopyError,
    copy_static_cache_prefix,
    longest_common_prefix_length,
    static_cache_sequence_length,
    static_cache_storage_bytes,
)
from duplex_kernel.qwen_conditioning import (
    QWEN_AUDIO_CONDITIONING_RECIPE,
    build_qwen_audio_conditioning,
)
from duplex_kernel.frame_residual_adapter import (
    FrameResidualAdapter,
    FrameResidualQueryFiLMAdapter,
    apply_response_state_residual,
    load_frame_residual_adapter,
)
from duplex_kernel.audio_response_protocol import (
    AUDIO_RESPONSE_ACTIONS,
    AUDIO_RESPONSE_PROTOCOL_V2,
    CONTROLLER_GUIDED_SAY_V1,
    AUDIO_RESPONSE_V2_MAX_ACTION_TOKENS,
    AUDIO_RESPONSE_V2_MAX_HEARD_TOKENS,
    AUDIO_RESPONSE_V2_MAX_NEW_TOKENS,
    AUDIO_RESPONSE_V2_MAX_SAY_TOKENS,
    AudioResponseProtocolError,
    AudioResponseV2ActionConstraint,
    AudioResponseV2Collector,
    finish_audio_response_v2_with_say_hidden,
)
from duplex_kernel.audio_context_capture import (
    AudioContextHiddenSnapshot,
    capture_audio_context_hidden as _capture_audio_context_hidden,
)
from duplex_kernel.tts_quality_diagnostics import apply_diagnostic_frame_budget
from duplex_kernel.thinker_text_budget import graceful_budget_action, last_complete_word_prefix


# qwen-tts 0.1.1's official Qwen3TTSTalkerForConditionalGeneration.generate
# sets talker_kwargs["min_new_tokens"] = 2 before honoring
# codec_eos_token_id. Mirror that contract in the custom CB0 loops (installed
# source: qwen3tts_src/qwen_tts/core/models/modeling_qwen3_tts.py, generate()).
QWEN_TALKER_MIN_NEW_TOKENS = 2

logger = logging.getLogger("duplex_kernel.engine")
_NO_FORCED_TOKEN_OVERRIDE = object()


def _code2wav_graph_supports_shape(graph: Any, codes: torch.Tensor) -> bool:
    if graph is None:
        return False
    supports_shape = getattr(graph, "supports_shape", None)
    if callable(supports_shape):
        return bool(supports_shape(codes))
    # Compatibility for graph doubles predating the explicit exact-shape API.
    return tuple(codes.shape) == (1, Code2WavWindowGraph.WINDOW_FRAMES, 16)


def _record_code2wav_graph_use(
    metrics: dict[str, Any], graph: Any, codes: torch.Tensor
) -> None:
    window_frames = int(codes.shape[1])
    metrics["code2wav_backend"] = (
        "cuda_graph_window_family"
        if isinstance(graph, Code2WavWindowGraphFamily)
        else f"cuda_graph_window{window_frames}"
    )
    metrics["code2wav_graph_captured"] = bool(graph.captured)
    metrics["code2wav_graph_window_calls"] = (
        metrics.get("code2wav_graph_window_calls", 0) + 1
    )
    key = f"code2wav_graph_window_{window_frames}_calls"
    metrics[key] = metrics.get(key, 0) + 1


def _decode_generated_response_texts(
    tokenizer: Any,
    generated_token_ids: list[int] | tuple[int, ...],
) -> tuple[str, str]:
    """Return human-facing cleaned text and the exact protocol decode separately."""
    protocol_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    raw_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
    display_text = re.sub(r"\s*<unused\d+>\s*", " ", raw_text)
    display_text = re.sub(r"<bos>|<eos>|<pad>|<end_of_turn>", " ", display_text)
    display_text = re.sub(r"[ \t]+", " ", display_text).strip()
    return display_text, protocol_text


_TALKER_COMPILED_DECODE_ENV = "DUPLEX_TALKER_COMPILED_DECODE"
_TALKER_NATIVE_AUDIO_Q1_ENV = "DUPLEX_TALKER_NATIVE_AUDIO_Q1"
_TALKER_NATIVE_AUDIO_Q1_GRAPH_ENV = "DUPLEX_TALKER_NATIVE_AUDIO_Q1_GRAPH"
_QWEN_NATIVE_LIBRARY_ENV = "QWEN_NATIVE_LIBRARY_PATH"
_QWEN_NATIVE_AUDIO_Q1_LIBRARY = "/workspace/.tmp/native_qwen_audio_q1/libqwen_talker_audio_q1.so"
_V9_DUAL_AUDIO_CONTEXT_ENV = "DUPLEX_V9_DUAL_AUDIO_CONTEXT_EXPERIMENTAL"
_V9_TEXT_REPLAY_PROMPT = "Rispondi in italiano in una o due frasi brevi."
_V9_CONDITIONING_SOURCES = {
    "live_generated",
    "v9_text_replay",
    "audio_prefill_frame_raw",
}
_TALKER_CUDA_TIMING_PROBE_LOCK = threading.Lock()
_TALKER_CUDA_TIMING_PROBE_CLAIMED = False


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



def _warmup_diag_tensor(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach()
    mask = torch.isfinite(value) if value.is_floating_point() else None
    finite = mask is None or bool(mask.all().item())
    values = value[mask] if mask is not None else value
    numeric = values.float() if values.numel() else values
    return {
        "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
        "finite": finite,
        "nonfinite_count": int(value.numel() - mask.sum().item()) if mask is not None else 0,
        "finite_min": float(numeric.min().item()) if numeric.numel() else None,
        "finite_max": float(numeric.max().item()) if numeric.numel() else None,
    }


def _warmup_diag_rotary_state(root_module: Any) -> list[dict[str, Any]]:
    """Snapshot rotary buffers independently; inference tensors have no version counter."""
    if not isinstance(root_module, torch.nn.Module):
        return []
    snapshots: list[dict[str, Any]] = []

    def tensor_state(value: torch.Tensor) -> dict[str, Any]:
        info: dict[str, Any] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "is_meta": bool(value.is_meta),
        }
        if value.is_meta:
            return info
        try:
            finite_mask = torch.isfinite(value)
            nonfinite = int((~finite_mask).sum().item())
            info["finite"] = nonfinite == 0
            info["nonfinite_count"] = nonfinite
            finite_values = value[finite_mask]
            if finite_values.numel():
                finite_values = finite_values.float()
                info["finite_min"] = float(finite_values.min().item())
                info["finite_max"] = float(finite_values.max().item())
            else:
                info["finite_min"] = None
                info["finite_max"] = None
            if nonfinite:
                first = torch.nonzero(~finite_mask, as_tuple=False)[0]
                index = tuple(int(part) for part in first.tolist())
                info["first_nonfinite_index"] = list(index)
                info["first_nonfinite_value"] = repr(float(value[index].item()))
        except Exception as exc:
            info["diagnostic_error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:240],
            }
        return info

    try:
        modules = root_module.named_modules()
        for name, module in modules:
            try:
                if not hasattr(module, "inv_freq") and not hasattr(module, "original_inv_freq"):
                    continue
                row: dict[str, Any] = {
                    "module_name": name or "<root>",
                    "module_type": type(module).__name__,
                    "rope_type": getattr(module, "rope_type", None),
                    "max_seq_len_cached": getattr(module, "max_seq_len_cached", None),
                    "original_max_seq_len": getattr(module, "original_max_seq_len", None),
                    "attributes": {},
                    "registered_buffers": {},
                }
                for attr in ("inv_freq", "original_inv_freq", "attention_scaling"):
                    value = getattr(module, attr, None)
                    if isinstance(value, torch.Tensor):
                        row["attributes"][attr] = tensor_state(value)
                    elif isinstance(value, (int, float)):
                        number = float(value)
                        row["attributes"][attr] = {
                            "value": number if math.isfinite(number) else None,
                            "finite": math.isfinite(number),
                            "type": type(value).__name__,
                        }
                    elif value is not None:
                        row["attributes"][attr] = {"type": type(value).__name__}
                try:
                    buffers = module.named_buffers(recurse=False)
                    for buffer_name, value in buffers:
                        row["registered_buffers"][buffer_name] = tensor_state(value)
                except Exception as exc:
                    row["buffer_iteration_error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc)[:240],
                    }
                snapshots.append(row)
            except Exception as exc:
                snapshots.append({
                    "module_name": name or "<root>",
                    "module_type": type(module).__name__,
                    "module_snapshot_error": {
                        "type": type(exc).__name__,
                        "message": str(exc)[:240],
                    },
                })
    except Exception as exc:
        snapshots.append({
            "module_traversal_error": {
                "type": type(exc).__name__,
                "message": str(exc)[:240],
            }
        })
    return snapshots

def _warmup_diag_tensors(value: Any, path: str = "root") -> list[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        return [(path, value)]
    if isinstance(value, dict):
        return [pair for key, item in value.items() for pair in _warmup_diag_tensors(item, f"{path}.{key}")]
    if isinstance(value, (tuple, list)):
        return [pair for index, item in enumerate(value) for pair in _warmup_diag_tensors(item, f"{path}[{index}]")]
    found: list[tuple[str, torch.Tensor]] = []
    for attr in ("audio_values", "last_hidden_state", "hidden_states", "sample", "waveform"):
        item = getattr(value, attr, None)
        if item is not None:
            found.extend(_warmup_diag_tensors(item, f"{path}.{attr}"))
    return found


def _warmup_diag_backends() -> dict[str, Any]:
    try:
        autocast_enabled = bool(torch.is_autocast_enabled("cuda"))
    except TypeError:
        autocast_enabled = bool(torch.is_autocast_enabled())
    try:
        autocast_dtype = str(torch.get_autocast_dtype("cuda"))
    except (AttributeError, TypeError):
        getter = getattr(torch, "get_autocast_gpu_dtype", None)
        autocast_dtype = str(getter()) if callable(getter) else None
    cuda = getattr(torch.backends, "cuda", None)
    return {
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "sdpa_flash_allowed": bool(cuda.flash_sdp_enabled()) if cuda else None,
        "sdpa_mem_efficient_allowed": bool(cuda.mem_efficient_sdp_enabled()) if cuda else None,
        "sdpa_math_allowed": bool(cuda.math_sdp_enabled()) if cuda else None,
        "sdpa_cudnn_allowed": bool(cuda.cudnn_sdp_enabled()) if cuda and callable(getattr(cuda, "cudnn_sdp_enabled", None)) else None,
        "cuda_autocast_enabled": autocast_enabled, "cuda_autocast_dtype": autocast_dtype,
    }


def _warmup_diag_profile(profiler: Any) -> dict[str, Any]:
    if profiler is None:
        return {"available": False, "operators": []}
    needles = ("cudnn", "flash", "scaled_dot", "attention", "sdp", "autocast", "convolution")
    events = []
    try:
        for event in profiler.key_averages():
            name = str(getattr(event, "key", ""))
            if any(needle in name.lower() for needle in needles):
                events.append({
                    "name": name[:180], "calls": int(getattr(event, "count", 0)),
                    "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0)),
                    "device_time_total_us": float(getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0))),
                })
    except Exception as exc:
        return {"available": False, "error": type(exc).__name__, "operators": []}
    return {"available": True, "operators": events[:40]}

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


def _cache_sequence_length(cache: Any) -> int:
    get_seq_length = getattr(cache, "get_seq_length", None)
    if not callable(get_seq_length):
        raise TypeError("Qwen Q1 requires a cache exposing get_seq_length()")
    length = get_seq_length()
    if isinstance(length, torch.Tensor):
        length = int(length.item())
    return int(length)


def _claim_talker_cuda_timing_probe() -> bool:
    """Enable one opt-in direct-audio Talker response per server process."""
    global _TALKER_CUDA_TIMING_PROBE_CLAIMED
    with _TALKER_CUDA_TIMING_PROBE_LOCK:
        if _TALKER_CUDA_TIMING_PROBE_CLAIMED:
            return False
        _TALKER_CUDA_TIMING_PROBE_CLAIMED = True
        return True


def _summarize_timing_values_ms(values: list[float], *, pending_count: int = 0) -> dict[str, Any]:
    """Summarize completed GPU event timings without synchronizing the device."""
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))

    def quantile(q: float) -> float | None:
        if not ordered:
            return None
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 3)

    return {
        "sample_count": len(ordered),
        "pending_count": int(pending_count),
        "p50_ms": quantile(0.50),
        "p95_ms": quantile(0.95),
        "max_ms": round(max(ordered), 3) if ordered else None,
    }


def _completed_cuda_event_elapsed_ms(start_event: Any, end_event: Any) -> float | None:
    """Return a GPU duration only when both events completed; never synchronize."""
    try:
        if not start_event.query() or not end_event.query():
            return None
        elapsed_ms = float(start_event.elapsed_time(end_event))
        if math.isfinite(elapsed_ms) and elapsed_ms >= 0.0:
            return elapsed_ms
    except Exception:
        # Optional telemetry must never fail an otherwise valid audio response.
        return None
    return None


def _record_completed_cuda_timing(
    metrics: dict[str, Any], key: str, start_event: Any, end_event: Any
) -> None:
    elapsed_ms = _completed_cuda_event_elapsed_ms(start_event, end_event)
    if elapsed_ms is None:
        metrics[f"{key}_pending"] = True
        return
    metrics[key] = elapsed_ms
    metrics[f"{key}_pending"] = False


def _summarize_cuda_event_pairs(event_pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    """Read only completed event pairs; query/elapsed_time never wait for CUDA."""
    values: list[float] = []
    pending = 0
    for start_event, end_event in event_pairs:
        elapsed_ms = _completed_cuda_event_elapsed_ms(start_event, end_event)
        if elapsed_ms is None:
            pending += 1
        else:
            values.append(elapsed_ms)
    return _summarize_timing_values_ms(values, pending_count=pending)
AUDIO_TEXT_PREFIX_CACHE_REUSE_ENV = "DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE"
AUDIO_THINKER_CACHE_BUCKETS = (512, 1024, 2048, 4096)


def select_audio_thinker_cache_bucket(
    prefill_tokens: int,
    output_budget: int,
    *,
    max_cache_len: int = AUDIO_THINKER_CACHE_BUCKETS[-1],
) -> int:
    """Return the smallest static-cache bucket covering one direct-audio turn.

    ``prefill_tokens`` must be the effective processor ``input_ids`` length,
    including audio placeholders. The bucket reserves the full requested
    generation budget; it never truncates either side to fit.
    """
    for name, value in (
        ("prefill_tokens", prefill_tokens),
        ("output_budget", output_budget),
        ("max_cache_len", max_cache_len),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} deve essere un intero")
    prefill_tokens = int(prefill_tokens)
    output_budget = int(output_budget)
    max_cache_len = int(max_cache_len)
    if prefill_tokens < 1:
        raise ValueError("prefill_tokens deve essere almeno 1")
    if output_budget < 1:
        raise ValueError("output_budget deve essere almeno 1")
    if max_cache_len < 1:
        raise ValueError("max_cache_len deve essere almeno 1")

    required_tokens = prefill_tokens + output_budget
    available = tuple(bucket for bucket in AUDIO_THINKER_CACHE_BUCKETS if bucket <= max_cache_len)
    largest_available = available[-1] if available else 0
    for bucket in available:
        if required_tokens <= bucket:
            return bucket
    raise ValueError(
        "prompt audio Thinker + budget supera i bucket StaticCache disponibili "
        f"({prefill_tokens} + {output_budget} = {required_tokens}; "
        f"massimo disponibile {largest_available}, massimo supportato "
        f"{AUDIO_THINKER_CACHE_BUCKETS[-1]})"
    )


def _estimate_talker_frame_budget(
    text: str,
    n_words: int,
    min_frames: int,
    *,
    short_streaming: bool,
) -> tuple[int, int, int]:
    """Return spoken units, digit-expansion units, and bounded CB0 frames.

    The caller supplies the ordinary word count to preserve established
    pacing. Each additional digit in a numeric token gets one spoken unit,
    since numeric IDs and codes are normally read digit by digit. Native EOS
    remains the stopping condition; this only supplies enough bounded budget.
    """
    base_units = max(1, int(n_words))
    numeric_extra_units = 0
    for token in text.split():
        numeric_token = token.strip(".,;:!?()[]{}\"'“”‘’")
        if re.fullmatch(r"\d+(?:[-.,/]\d+)*", numeric_token):
            numeric_extra_units += max(0, sum(char.isdecimal() for char in numeric_token) - 1)
    spoken_units = base_units + numeric_extra_units
    if short_streaming:
        # Backchannels use an EOS-driven safety ceiling. A short text-derived
        # estimate can terminate valid speech before the model emits EOS.
        frame_budget = ARIA_BACKCHANNEL_MAX_FRAMES
    else:
        frame_budget = min(
            ARIA_MAX_FRAMES,
            max(
                int(min_frames),
                int(spoken_units * ARIA_FRAMES_PER_WORD) + ARIA_OVERHEAD_FRAMES,
            ),
        )
    return spoken_units, numeric_extra_units, frame_budget


def _native_q1_import_audio_frame_budget(
    text_frame_budget: int,
    *,
    live_qwen_text_stream: Any | None,
    live_frame_cap: int,
) -> int:
    """Reserve bounded headroom when live SAY text grows after Q1 import."""
    text_frame_budget = int(text_frame_budget)
    live_frame_cap = int(live_frame_cap)
    if text_frame_budget <= 0 or live_frame_cap <= 0:
        raise ValueError("native Q1 frame budgets devono essere positivi")
    if text_frame_budget > live_frame_cap:
        raise ValueError("text frame budget supera il cap live native Q1")
    return live_frame_cap if live_qwen_text_stream is not None else text_frame_budget


def _apply_live_frame_residual(
    adapter: Any,
    selected_response_hidden: torch.Tensor,
    qwen_audio_input: torch.Tensor,
    *,
    generation_step: int,
    tail_hold: bool,
) -> tuple[torch.Tensor, bool]:
    """Apply the already-selected live packet hidden at its global Q1 step."""
    if isinstance(generation_step, bool) or int(generation_step) < 0:
        raise ValueError("generation_step deve essere non negativo")
    if int(generation_step) == 0 and bool(getattr(adapter, "preserve_frame0", False)):
        return qwen_audio_input, bool(tail_hold)
    return adapter(selected_response_hidden, qwen_audio_input), bool(tail_hold)


def _audio_text_prefix_cache_reuse_from_env() -> bool:
    """Read the opt-in cross-response audio-prefix cache switch (default off)."""
    value = os.environ.get(AUDIO_TEXT_PREFIX_CACHE_REUSE_ENV)
    if value is None:
        return False
    value = value.strip()
    if value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{AUDIO_TEXT_PREFIX_CACHE_REUSE_ENV} must be exactly '0' or '1'")


def _build_backchannel_token_trie(
    phrases: list[str],
    tokenizer: Any,
    *,
    eos_token_ids: set[int],
) -> dict[tuple[int, ...], frozenset[int]]:
    """Map each gold-phrase prefix to only its legal next Gemma token IDs.

    Phrase text comes from the train-split BACKCHANNEL targets bundled with
    the active checkpoint. Token IDs are always produced by the loaded
    tokenizer; there are no hand-authored IDs or fallback phrases.
    """
    eos = {int(token_id) for token_id in eos_token_ids}
    if not eos:
        raise ValueError("backchannel trie richiede almeno un token EOS reale")
    if not phrases:
        raise ValueError("backchannel trie vuoto: nessuna frase gold disponibile")

    prefixes: dict[tuple[int, ...], set[int]] = {}
    encoded_sequences: set[tuple[int, ...]] = set()
    for phrase in phrases:
        if not isinstance(phrase, str):
            raise TypeError("backchannel lexicon contiene una frase non testuale")
        phrase = phrase.strip()
        word_count = len(phrase.split())
        if not phrase or not 1 <= word_count <= 3 or "?" in phrase or "？" in phrase:
            raise ValueError(f"frase backchannel fuori contratto (1–3 parole, non domanda): {phrase!r}")
        encoded = tokenizer.encode(phrase, add_special_tokens=False)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            if len(encoded) != 1:
                raise ValueError("tokenizer ha prodotto un batch inatteso per una frase")
            encoded = encoded[0]
        sequence = tuple(int(token_id) for token_id in encoded)
        if not sequence:
            raise ValueError(f"tokenizer non produce token per frase backchannel {phrase!r}")
        if any(token_id in eos or token_id < 0 for token_id in sequence):
            raise ValueError(f"frase backchannel contiene un token EOS/negativo: {phrase!r}")
        if sequence in encoded_sequences:
            continue
        encoded_sequences.add(sequence)
        for index, token_id in enumerate(sequence):
            prefix = sequence[:index]
            prefixes.setdefault(prefix, set()).add(token_id)
        prefixes.setdefault(sequence, set()).update(eos)

    if not encoded_sequences:
        raise ValueError("backchannel trie vuoto dopo la tokenizzazione")
    return {prefix: frozenset(next_ids) for prefix, next_ids in prefixes.items()}


def _resolve_talker_voice_conditioning(
    talker: Any,
    voice_conditioning: dict[str, Any] | None,
) -> dict[str, Any] | None:
    resolver = getattr(
        talker,
        "resolve_voice_conditioning",
        None,
    )
    if callable(resolver):
        return resolver(voice_conditioning)
    return voice_conditioning


def _build_talker_prefill_parts(
    talker: Any,
    text: str,
    voice_conditioning: dict[str, Any] | None,
    *,
    raw_text_token_ids: list[int] | tuple[int, ...] | None = None,
    include_text_eos: bool = True,
):
    builder = getattr(
        talker,
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

    kwargs: dict[str, Any] = {}
    if "voice_conditioning" in parameters:
        kwargs["voice_conditioning"] = voice_conditioning
    elif voice_conditioning is not None:
        raise TypeError("Talker backend non supporta voice_conditioning")
    if "raw_text_token_ids" in parameters:
        kwargs["raw_text_token_ids"] = raw_text_token_ids
    elif raw_text_token_ids is not None:
        raise TypeError("Talker backend non supporta raw_text_token_ids live")
    if "include_text_eos" in parameters:
        kwargs["include_text_eos"] = include_text_eos
    elif not include_text_eos:
        raise TypeError("Talker backend non supporta prefill senza text EOS")
    if kwargs:
        return builder(text, **kwargs)
    if voice_conditioning is not None:
        raise TypeError("Talker backend non supporta voice_conditioning")
    return builder(text)


def _split_text_for_frame_segments(
    text: str,
    *,
    total_frames: int,
    segment_frames: int,
) -> tuple[str, ...]:
    """Partition a completed response across acoustic segment budgets.

    Segment rollovers reset Qwen's local audio history, so replaying the whole
    response text at each rollover would restart the utterance.  The pre-QLoRA
    runtime already has the complete Thinker text; assign contiguous word
    spans in proportion to each segment's acoustic budget instead.
    """
    if total_frames <= 0 or segment_frames <= 0:
        raise ValueError("frame budgets devono essere positivi")
    words = text.split()
    if not words:
        raise ValueError("text non può essere vuoto")
    segment_count = (int(total_frames) + int(segment_frames) - 1) // int(segment_frames)
    if segment_count == 1:
        return (text,)
    if segment_count > len(words):
        raise ValueError(
            "continuazione Qwen richiede almeno una parola per segmento: "
            f"segments={segment_count}, words={len(words)}"
        )

    result: list[str] = []
    word_start = 0
    for segment_index in range(segment_count):
        if segment_index == segment_count - 1:
            word_end = len(words)
        else:
            frame_end = min(
                int(total_frames),
                (segment_index + 1) * int(segment_frames),
            )
            rounded_end = (
                frame_end * len(words) + int(total_frames) // 2
            ) // int(total_frames)
            min_end = word_start + 1
            max_end = len(words) - (segment_count - segment_index - 1)
            word_end = min(max(rounded_end, min_end), max_end)
        result.append(" ".join(words[word_start:word_end]))
        word_start = word_end

    if word_start != len(words) or any(not part for part in result):
        raise RuntimeError("partizione testo/acoustic frame non contigua")
    return tuple(result)


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


def _validate_response_bundle_qlora_compatibility(
    *,
    bundle_enabled: bool,
    gemma_qlora_enabled: bool,
    native_control6_enabled: bool,
    response_lora_ablated: bool,
) -> None:
    """Only native Control6 with the bundle response LoRA ablated may stack adapters."""
    if bundle_enabled and gemma_qlora_enabled and not (
        native_control6_enabled and response_lora_ablated
    ):
        raise ValueError(
            "duplex response bundle + Gemma QLoRA is supported only with native Control6 "
            "and --ablate-response-lora"
        )


class ThinkerSampler:
    """Shared token-sampling pipeline used by HF and native Thinker decoders."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        device: Any,
        eos_token_ids: set[int],
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        min_new_tokens: int = 0,
        backchannel_trie: dict[tuple[int, ...], Any] | None = None,
        runtime_metrics: dict[str, Any] | None = None,
        on_generation_start: Callable[[], Any] | None = None,
        on_generated_token: Callable[[int, int, str, torch.Tensor], Any] | None = None,
        on_candidate_token: Callable[[int, str, bool], bool] | None = None,
        on_candidate_fallback: Callable[[list[tuple[int, str, bool]]], Any] | None = None,
        on_action_logits_probe: Callable[[dict[str, Any]], Any] | None = None,
        generation_id: int | None = None,
        state: Any | None = None,
        best_native_logit_candidate_id: Callable[[list[int], torch.Tensor], int | None] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device
        self.eos_token_ids = set(eos_token_ids)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.min_new_tokens = int(min_new_tokens)
        self.backchannel_trie = backchannel_trie
        self.runtime_metrics = runtime_metrics if runtime_metrics is not None else {}
        self.on_generation_start = on_generation_start
        self.on_generated_token_callback = on_generated_token
        self.on_candidate_token = on_candidate_token
        self.on_candidate_fallback = on_candidate_fallback
        self.on_action_logits_probe = on_action_logits_probe
        self.generation_id = generation_id
        self.state = state
        self.best_native_logit_candidate_id = best_native_logit_candidate_id
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator deve essere torch.Generator o None")
        self.generator = generator
        self.last_decoded_output = ""
        self.forced_action_logprobs: list[torch.Tensor] = []
        self.forced_action_overrides = 0
        self.forced_structure_logprobs: list[torch.Tensor] = []
        self.forced_structure_overrides = 0
        self.action_logits_probe_emitted = False

        self.min_new_tokens_processor = None
        if self.min_new_tokens > 0 and self.eos_token_ids:
            from transformers.generation.logits_process import MinNewTokensLengthLogitsProcessor

            self.min_new_tokens_processor = MinNewTokensLengthLogitsProcessor(
                prompt_length_to_skip=0,
                min_new_tokens=self.min_new_tokens,
                eos_token_id=sorted(self.eos_token_ids),
                device=str(device),
            )

    def sample_token(
        self,
        curr_logits: torch.Tensor,
        generated_ids: Sequence[int],
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Apply the Thinker loop's exact penalty, filters, temperature and nucleus sampling."""
        history = [int(token_id) for token_id in generated_ids]
        if self.repetition_penalty != 1.0 and history:
            curr_logits = curr_logits.clone()
            for prev_id in set(history):
                if curr_logits[0, prev_id] < 0:
                    curr_logits[0, prev_id] = curr_logits[0, prev_id] * self.repetition_penalty
                else:
                    curr_logits[0, prev_id] = curr_logits[0, prev_id] / self.repetition_penalty
        if self.min_new_tokens_processor is not None:
            generated_tensor = torch.tensor([history], device=curr_logits.device, dtype=torch.long)
            curr_logits = self.min_new_tokens_processor(generated_tensor, curr_logits)
        if self.backchannel_trie is not None:
            allowed = self.backchannel_trie.get(tuple(history))
            if not allowed:
                raise RuntimeError(
                    "BACKCHANNEL token prefix non appartiene a nessuna frase train; "
                    f"generated_prefix={history!r}"
                )
            allowed_ids = torch.tensor(sorted(allowed), device=curr_logits.device, dtype=torch.long)
            if int(allowed_ids.min()) < 0 or int(allowed_ids.max()) >= curr_logits.shape[-1]:
                raise RuntimeError("BACKCHANNEL trie contiene ID fuori dal vocabolario Thinker")
            allowed_mask = torch.zeros_like(curr_logits, dtype=torch.bool)
            allowed_mask.index_fill_(-1, allowed_ids, True)
            curr_logits = curr_logits.masked_fill(~allowed_mask, float("-inf"))
            self.runtime_metrics["backchannel_trie_steps"] = (
                self.runtime_metrics.get("backchannel_trie_steps", 0) + 1
            )
        if self.temperature <= 1e-5:
            token_tensor = curr_logits.argmax(dim=-1, keepdim=True)
            candidate_scores = curr_logits
        else:
            probs = torch.softmax(curr_logits / self.temperature, dim=-1)
            if 0.0 < self.top_p < 1.0:
                values, indices = torch.sort(probs, descending=True)
                keep = torch.cumsum(values, dim=-1) <= self.top_p
                keep[..., 0] = True
                probs = torch.zeros_like(probs).scatter(-1, indices, values * keep)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            candidate_scores = probs
            token_tensor = torch.multinomial(probs, 1, generator=self.generator)
        return token_tensor, int(token_tensor.item()), candidate_scores

    def sample_initial(
        self,
        logits: torch.Tensor,
        generated_ids: Sequence[int] = (),
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Sample the first token, then apply the exact initial-force contract."""
        forced_id = self.on_generation_start() if self.on_generation_start is not None else None
        candidate = self.sample_token(logits, generated_ids)
        if forced_id is None:
            return candidate
        if isinstance(forced_id, bool) or not isinstance(forced_id, int):
            raise RuntimeError("audio response initial forced token id must be an integer")
        if forced_id < 0 or forced_id >= int(logits.shape[-1]):
            raise RuntimeError("audio response initial forced token id outside Thinker vocabulary")
        self.forced_structure_logprobs.append(
            torch.log_softmax(logits.float(), dim=-1)[0, forced_id].detach()
        )
        if candidate[1] != forced_id:
            self.forced_structure_overrides += 1
        return torch.tensor([[forced_id]], device=self.device, dtype=torch.long), forced_id, candidate[2]

    def select_valid_candidate(
        self,
        candidate: tuple[torch.Tensor, int, torch.Tensor],
        *,
        source_logits: torch.Tensor,
        generated_ids: Sequence[int],
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        token_tensor, token_id, candidate_scores = candidate
        callback = self.on_candidate_token
        if callback is None:
            return candidate
        if self.generation_id is not None and self.state is not None and not self.state.is_generation_current(
            generation_id=self.generation_id
        ):
            return candidate

        self.runtime_metrics["audio_response_protocol_v2_candidate_checks"] = (
            self.runtime_metrics.get("audio_response_protocol_v2_candidate_checks", 0) + 1
        )
        history = [int(value) for value in generated_ids]

        def candidate_text(candidate_id: int) -> tuple[str, bool]:
            terminal = candidate_id in self.eos_token_ids
            if terminal:
                return "", True
            decoded = self.tokenizer.decode([*history, int(candidate_id)], skip_special_tokens=True)
            if not decoded.startswith(self.last_decoded_output):
                raise RuntimeError("Thinker token decode is not prefix-stable for audio response parsing")
            return decoded[len(self.last_decoded_output):], False

        text, terminal = candidate_text(token_id)
        if callback(token_id, text, terminal):
            return candidate
        rejected: list[tuple[int, str, bool]] = [(token_id, text, terminal)]
        flat_scores = candidate_scores.reshape(-1)
        candidate_count = min(64, int(flat_scores.shape[0]))
        top_scores, top_ids = torch.topk(flat_scores, k=candidate_count)
        for score, candidate_value in zip(top_scores.tolist(), top_ids.tolist()):
            if not math.isfinite(float(score)) or float(score) <= 0.0 and self.temperature > 1e-5:
                continue
            candidate_id = int(candidate_value)
            if candidate_id == token_id:
                continue
            text, terminal = candidate_text(candidate_id)
            self.runtime_metrics["audio_response_protocol_v2_candidate_checks"] += 1
            if not callback(candidate_id, text, terminal):
                rejected.append((candidate_id, text, terminal))
                continue
            self.forced_structure_logprobs.append(
                torch.log_softmax(source_logits.float(), dim=-1)[0, candidate_id].detach()
            )
            self.forced_structure_overrides += 1
            self.runtime_metrics["audio_response_protocol_v2_predecode_replacements"] = (
                self.runtime_metrics.get("audio_response_protocol_v2_predecode_replacements", 0) + 1
            )
            return torch.tensor([[candidate_id]], device=self.device, dtype=torch.long), candidate_id, candidate_scores

        if self.on_candidate_fallback is not None:
            fallback = self.on_candidate_fallback(rejected)
            fallback_id = None
            if isinstance(fallback, (tuple, list, set, frozenset)):
                legal_pool: list[int] = []
                for candidate_value in fallback:
                    if isinstance(candidate_value, bool) or not isinstance(candidate_value, Integral):
                        raise RuntimeError("audio response candidate pool IDs must be integers")
                    candidate_id = int(candidate_value)
                    if candidate_id < 0 or candidate_id >= int(source_logits.shape[-1]):
                        raise RuntimeError("audio response candidate pool ID outside Thinker vocabulary")
                    candidate_text_value, candidate_terminal = candidate_text(candidate_id)
                    self.runtime_metrics["audio_response_protocol_v2_candidate_checks"] += 1
                    if callback(candidate_id, candidate_text_value, candidate_terminal):
                        legal_pool.append(candidate_id)
                best = (
                    None if self.best_native_logit_candidate_id is None
                    else self.best_native_logit_candidate_id(legal_pool, source_logits)
                )
                if best is not None:
                    self.runtime_metrics["audio_response_protocol_v2_lexicon_pool_overrides"] = (
                        self.runtime_metrics.get("audio_response_protocol_v2_lexicon_pool_overrides", 0) + 1
                    )
                    return torch.tensor([[best]], device=self.device, dtype=torch.long), best, candidate_scores
            elif fallback is not None:
                fallback_id = fallback
            if fallback_id is not None:
                if isinstance(fallback_id, bool) or not isinstance(fallback_id, int):
                    raise RuntimeError("audio response candidate fallback id must be an integer")
                if fallback_id < 0 or fallback_id >= int(source_logits.shape[-1]):
                    raise RuntimeError("audio response candidate fallback id outside Thinker vocabulary")
                fallback_text, fallback_terminal = candidate_text(fallback_id)
                if not callback(fallback_id, fallback_text, fallback_terminal):
                    raise AudioResponseProtocolError("bounded_closure_not_valid")
                self.forced_structure_logprobs.append(
                    torch.log_softmax(source_logits.float(), dim=-1)[0, fallback_id].detach()
                )
                self.forced_structure_overrides += 1
                self.runtime_metrics["audio_response_protocol_v2_predecode_replacements"] = (
                    self.runtime_metrics.get("audio_response_protocol_v2_predecode_replacements", 0) + 1
                )
                self.runtime_metrics["audio_response_protocol_v2_bounded_closures"] = (
                    self.runtime_metrics.get("audio_response_protocol_v2_bounded_closures", 0) + 1
                )
                return torch.tensor([[fallback_id]], device=self.device, dtype=torch.long), fallback_id, candidate_scores
        raise AudioResponseProtocolError("no_legal_candidate_within_top64")

    def after_generated_token(
        self,
        *,
        token_index: int,
        token_id: int,
        token_hidden: torch.Tensor,
        generated_ids: Sequence[int],
        source_logits: torch.Tensor,
        sampled_next: tuple[torch.Tensor, int, torch.Tensor],
        forced_override: Any = _NO_FORCED_TOKEN_OVERRIDE,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Run the token callback and apply its forced-next override to a pre-sampled candidate."""
        callback = self.on_generated_token_callback
        if callback is None or (
            self.generation_id is not None
            and self.state is not None
            and not self.state.is_generation_current(generation_id=self.generation_id)
        ):
            return sampled_next
        decoded = self.tokenizer.decode(list(generated_ids), skip_special_tokens=True)
        if not decoded.startswith(self.last_decoded_output):
            raise RuntimeError("Thinker token decode is not prefix-stable for audio response parsing")
        token_text = decoded[len(self.last_decoded_output):]
        self.last_decoded_output = decoded
        forced = (
            callback(token_index, int(token_id), token_text, token_hidden.detach())
            if forced_override is _NO_FORCED_TOKEN_OVERRIDE
            else forced_override
        )
        if forced is None:
            return sampled_next
        forced_kind = "action"
        if isinstance(forced, tuple):
            if len(forced) != 2:
                raise RuntimeError("audio response forced token tuple must contain id and kind")
            forced, forced_kind = forced
        if isinstance(forced, bool) or not isinstance(forced, int):
            raise RuntimeError("audio response forced token id must be an integer")
        if forced_kind not in {"action", "structure"}:
            raise RuntimeError("audio response forced token kind is invalid")
        if forced < 0 or forced >= int(source_logits.shape[-1]):
            raise RuntimeError("audio response forced token id outside Thinker vocabulary")
        forced_logprob = torch.log_softmax(source_logits.float(), dim=-1)[0, forced].detach()
        if forced_kind == "action" and self.on_action_logits_probe is not None and not self.action_logits_probe_emitted:
            label_token_ids: dict[str, list[int]] = {}
            encode = getattr(self.tokenizer, "encode", None)
            if callable(encode):
                for label in AUDIO_RESPONSE_ACTIONS:
                    try:
                        encoded = encode(label, add_special_tokens=False)
                        if callable(getattr(encoded, "detach", None)):
                            encoded = encoded.detach().cpu().reshape(-1).tolist()
                        elif callable(getattr(encoded, "tolist", None)):
                            encoded = encoded.tolist()
                        while isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
                            encoded = encoded[0]
                        label_token_ids[label] = [int(value) for value in encoded]
                    except Exception:
                        label_token_ids[label] = []
            distinct_first_ids = list(dict.fromkeys(
                values[0] for values in label_token_ids.values()
                if values and 0 <= values[0] < int(source_logits.shape[-1])
            ))
            if distinct_first_ids:
                candidate_ids = torch.tensor(distinct_first_ids, device=source_logits.device, dtype=torch.long)
                candidate_logits = source_logits[0].index_select(0, candidate_ids)
                raw_top_id_tensor = source_logits[0].argmax().reshape(1).to(candidate_logits.dtype)
                probe_values = torch.cat((raw_top_id_tensor, candidate_logits), dim=0).detach().float().cpu().tolist()
                raw_argmax_token_id = int(round(probe_values[0]))
                first_token_logits_by_id = {
                    token_id: float(value) for token_id, value in zip(distinct_first_ids, probe_values[1:])
                }
            else:
                raw_argmax_token_id = int(source_logits[0].argmax().item())
                first_token_logits_by_id = {}
            action_first_logits = {
                label: first_token_logits_by_id[values[0]]
                for label, values in label_token_ids.items()
                if values and values[0] in first_token_logits_by_id
            }
            raw_argmax_action_candidates = [
                label for label, values in label_token_ids.items()
                if values and values[0] == raw_argmax_token_id
            ]
            best_first_logit = max(action_first_logits.values(), default=None)
            best_first_action_candidates = (
                [label for label, value in action_first_logits.items() if value == best_first_logit]
                if best_first_logit is not None else []
            )
            self.on_action_logits_probe({
                "raw_argmax_token_id": raw_argmax_token_id,
                "sampled_token_id": int(sampled_next[1]),
                "forced_token_id": int(forced),
                "action_label_token_ids": label_token_ids,
                "action_label_first_token_logits": action_first_logits,
                "raw_argmax_action_candidates": raw_argmax_action_candidates,
                "best_first_logit_action_candidates": best_first_action_candidates,
            })
            self.action_logits_probe_emitted = True
        if forced_kind == "action":
            self.forced_action_logprobs.append(forced_logprob)
            if sampled_next[1] != forced:
                self.forced_action_overrides += 1
        else:
            self.forced_structure_logprobs.append(forced_logprob)
            if sampled_next[1] != forced:
                self.forced_structure_overrides += 1
        return torch.tensor([[forced]], device=self.device, dtype=torch.long), forced, sampled_next[2]

    async def after_generated_token_async(
        self,
        *,
        token_index: int,
        token_id: int,
        token_hidden: torch.Tensor,
        generated_ids: Sequence[int],
        source_logits: torch.Tensor,
        sampled_next: tuple[torch.Tensor, int, torch.Tensor],
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Await async response callbacks without blocking the decoder loop."""
        callback = self.on_generated_token_callback
        if callback is None or (
            self.generation_id is not None
            and self.state is not None
            and not self.state.is_generation_current(generation_id=self.generation_id)
        ):
            return sampled_next
        decoded = self.tokenizer.decode(list(generated_ids), skip_special_tokens=True)
        if not decoded.startswith(self.last_decoded_output):
            raise RuntimeError("Thinker token decode is not prefix-stable for audio response parsing")
        token_text = decoded[len(self.last_decoded_output):]
        self.last_decoded_output = decoded
        forced = callback(token_index, int(token_id), token_text, token_hidden.detach())
        if inspect.isawaitable(forced):
            forced = await forced
        return self.after_generated_token(
            token_index=token_index,
            token_id=token_id,
            token_hidden=token_hidden,
            generated_ids=generated_ids,
            source_logits=source_logits,
            sampled_next=sampled_next,
            forced_override=forced,
        )


class PipelineEngine:
    """Engine a 3 stadi: Thinker + V9 + Talker(Qwen3-TTS) + Code2Wav."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        runtime_backend: str = "pytorch",
        allow_talker_eos: bool = True,
        v9_profile: str = V9_PROFILE_CAUSAL_L0,
        enable_duplex_control6: bool = False,
        duplex_control6_adapter_dir: str | None = None,
        duplex_control6_head_path: str | None = None,
        duplex_control6_manifest_path: str | None = None,
        duplex_response_bundle_dir: str | None = None,
        ablate_response_lora: bool = False,
        v9_conditioning_source: str = "live_generated",
        ablate_session_voice_conditioning: bool = False,
        diagnostic_talker_seed: int | None = None,
        controller_guided_talk_contexts: int = 0,
        gemma_qlora_adapter_dir: str | None = None,
        thinker_prefill_chunk_tokens: int = 16,
        thinker_audio_prefill_chunk_tokens: int = 0,
        thinker_compile_mode: str = "off",
        thinker_static_cache_max_len: int = 4096,
        frame_residual_adapter_checkpoint_path: str | None = None,
        enable_gemma_native_control6_policy: bool = False,
        native_control6_adapter_path: str | None = None,
    ) -> None:
        if runtime_backend not in {"pytorch", "native"}:
            raise ValueError("runtime_backend deve essere 'pytorch' o 'native'")
        if v9_profile not in {V9_PROFILE_QUALITY_FINAL_ORACLE, V9_PROFILE_CAUSAL_L0}:
            raise ValueError(f"v9_profile non valido: {v9_profile!r}")
        self.device = device
        self.dtype = dtype
        self.runtime_backend = runtime_backend
        self.virtual_prefix_mode = "legacy"
        self.allow_talker_eos = bool(allow_talker_eos)
        self.v9_profile = v9_profile
        if v9_conditioning_source not in _V9_CONDITIONING_SOURCES:
            raise ValueError(
                "v9_conditioning_source deve essere 'live_generated', 'v9_text_replay' "
                "o 'audio_prefill_frame_raw'"
            )
        self.v9_conditioning_source = v9_conditioning_source
        if not isinstance(ablate_session_voice_conditioning, bool):
            raise TypeError("ablate_session_voice_conditioning must be a bool")
        self.ablate_session_voice_conditioning = ablate_session_voice_conditioning
        if diagnostic_talker_seed is not None and (
            isinstance(diagnostic_talker_seed, bool)
            or not isinstance(diagnostic_talker_seed, Integral)
            or not 0 <= int(diagnostic_talker_seed) <= 0xFFFFFFFF
        ):
            raise ValueError("diagnostic_talker_seed must be an integer in [0, 2**32-1]")
        self.diagnostic_talker_seed = (
            None if diagnostic_talker_seed is None else int(diagnostic_talker_seed)
        )
        if controller_guided_talk_contexts not in {0, 2}:
            raise ValueError("controller_guided_talk_contexts deve essere 0 o 2")
        self.controller_guided_talk_contexts = int(controller_guided_talk_contexts)
        if config.CODE2WAV_GRAPH_FAMILY_ENABLED and self.controller_guided_talk_contexts != 2:
            raise ValueError(
                "GEMMA4_OMNI_CODE2WAV_GRAPH_FAMILY=1 requires the two-slot response talk-context pool"
            )
        self.response_talk_context_pool = None
        self.response_talk_context_memory: dict[str, Any] | None = None
        self.audio_text_prefix_cache_reuse_enabled = _audio_text_prefix_cache_reuse_from_env()
        self.duplex_response_bundle_dir = duplex_response_bundle_dir
        if not isinstance(ablate_response_lora, bool):
            raise TypeError("ablate_response_lora must be a bool")
        self.ablate_response_lora = ablate_response_lora
        if (
            frame_residual_adapter_checkpoint_path is None
            and runtime_backend == "native"
            and os.path.isfile(config.FRAME_RESIDUAL_ADAPTER_CKPT)
        ):
            frame_residual_adapter_checkpoint_path = config.FRAME_RESIDUAL_ADAPTER_CKPT
        self.frame_residual_adapter_checkpoint_path = frame_residual_adapter_checkpoint_path
        self.frame_residual_adapter: FrameResidualAdapter | FrameResidualQueryFiLMAdapter | None = None
        self.frame_residual_adapter_architecture: str | None = None
        self.frame_residual_adapter_enabled = bool(frame_residual_adapter_checkpoint_path)
        self.frame_residual_adapter_resolved_path: str | None = None
        self.audio_response_protocol: str | None = None
        self.direct_v2_audio_warmup_status = "not_applicable"
        v9_dual_audio_setting = os.environ.get(_V9_DUAL_AUDIO_CONTEXT_ENV, "0").strip()
        if v9_dual_audio_setting not in {"0", "1"}:
            raise ValueError(f"{_V9_DUAL_AUDIO_CONTEXT_ENV} must be exactly '0' or '1'")
        self.v9_dual_audio_context_experimental = v9_dual_audio_setting == "1"
        compiled_decode_setting = os.environ.get(_TALKER_COMPILED_DECODE_ENV, "0").strip()
        if compiled_decode_setting not in {"0", "1"}:
            raise ValueError(f"{_TALKER_COMPILED_DECODE_ENV} must be exactly '0' or '1'")
        self._talker_compiled_decode_requested = compiled_decode_setting == "1"
        native_decode_setting = os.environ.get(_TALKER_NATIVE_AUDIO_Q1_ENV, "0").strip()
        if native_decode_setting not in {"0", "1"}:
            raise ValueError(f"{_TALKER_NATIVE_AUDIO_Q1_ENV} must be exactly '0' or '1'")
        self._talker_native_audio_q1_requested = native_decode_setting == "1"
        native_graph_setting = os.environ.get(_TALKER_NATIVE_AUDIO_Q1_GRAPH_ENV, "0").strip()
        if native_graph_setting not in {"0", "1"}:
            raise ValueError(f"{_TALKER_NATIVE_AUDIO_Q1_GRAPH_ENV} must be exactly '0' or '1'")
        self._talker_native_audio_q1_graph_requested = native_graph_setting == "1"
        if self._talker_native_audio_q1_graph_requested and not self._talker_native_audio_q1_requested:
            raise ValueError(f"{_TALKER_NATIVE_AUDIO_Q1_GRAPH_ENV}=1 requires {_TALKER_NATIVE_AUDIO_Q1_ENV}=1")
        if self._talker_native_audio_q1_graph_requested and self.controller_guided_talk_contexts != 2:
            raise ValueError("native Q1 graph requires the two-slot response talk-context pool")
        self._native_q1_graph_max_audio_frames = max(1024, int(ARIA_MAX_FRAMES))
        self._native_q1_graph_prefill_headroom = 1024
        self._native_q1_slot_executors: dict[int, Any] = {}
        self._native_q1_startup_warmup_active = False
        self.talker_q1_native_graph_slots_captured = 0
        if self._talker_native_audio_q1_requested and not os.environ.get(_QWEN_NATIVE_LIBRARY_ENV):
            os.environ[_QWEN_NATIVE_LIBRARY_ENV] = _QWEN_NATIVE_AUDIO_Q1_LIBRARY
        self.talker_q1_native_status = (
            "requested" if self._talker_native_audio_q1_requested else "disabled"
        )
        self.talker_q1_native_warmup_calls = 0
        self.talker_q1_native_fallback_reason: str | None = None
        self._compiled_talker_q1 = None
        self.talker_q1_backend = "eager"
        self.talker_q1_compile_status = (
            "requested"
            if self._talker_compiled_decode_requested and not self._talker_native_audio_q1_requested
            else "skipped_native_precedence"
            if self._talker_compiled_decode_requested
            else "disabled"
        )
        self.talker_q1_compile_fallback_reason: str | None = None
        self._talker_q1_compile_warmup_calls = 0
        self._talker_q1_dynamo_before: tuple[int, int] | None = None
        self.enable_duplex_control6 = bool(enable_duplex_control6 or duplex_response_bundle_dir)
        self.duplex_control6_adapter_dir = duplex_control6_adapter_dir
        self.duplex_control6_head_path = duplex_control6_head_path
        self.duplex_control6_manifest_path = duplex_control6_manifest_path
        self.duplex_control6_runtime: dict[str, Any] | None = None
        self.enable_gemma_native_control6_policy = bool(enable_gemma_native_control6_policy)
        semantic_prefix_env = os.environ.get("DUPLEX_NATIVE_CONTROL6_SEMANTIC_PREFIX", "0").strip()
        if semantic_prefix_env not in {"0", "1"}:
            raise ValueError("DUPLEX_NATIVE_CONTROL6_SEMANTIC_PREFIX must be 0 or 1")
        self.native_control6_semantic_prefix_enabled = semantic_prefix_env == "1"
        if self.native_control6_semantic_prefix_enabled:
            from duplex_kernel.native_control6_prompt import (
                NATIVE_CONTROL6_PREFIX_CONTRACT,
                NATIVE_CONTROL6_PREFIX_SCHEMA_VERSION,
            )

            env_schema = os.environ.get("DUPLEX_NATIVE_CONTROL6_PREFIX_SCHEMA_VERSION", "2").strip()
            env_contract = os.environ.get(
                "DUPLEX_NATIVE_CONTROL6_PREFIX_CONTRACT", NATIVE_CONTROL6_PREFIX_CONTRACT
            ).strip()
            if env_schema != str(NATIVE_CONTROL6_PREFIX_SCHEMA_VERSION):
                raise ValueError("native Control6 semantic-prefix schema env does not match runtime")
            if env_contract != NATIVE_CONTROL6_PREFIX_CONTRACT:
                raise ValueError("native Control6 semantic-prefix contract env does not match runtime")
        self.native_control6_adapter_path = native_control6_adapter_path
        self.native_control6_runtime: Any | None = None
        self.native_control6_report: dict[str, Any] | None = None
        self.native_control6_semantic_prefix_metadata: dict[str, Any] | None = None
        self._backchannel_token_trie: dict[tuple[int, ...], frozenset[int]] = {}
        self._backchannel_say_phrases: tuple[str, ...] = ()
        self._backchannel_phrase_count = 0
        self.gemma_qlora_adapter_dir = gemma_qlora_adapter_dir
        self.gemma_qlora_adapter_loaded: str | None = None
        if (
            isinstance(thinker_prefill_chunk_tokens, bool)
            or not isinstance(thinker_prefill_chunk_tokens, Integral)
            or int(thinker_prefill_chunk_tokens) < 0
        ):
            raise ValueError("thinker_prefill_chunk_tokens deve essere >= 0; 0 disabilita il chunking")
        self.thinker_prefill_chunk_tokens = int(thinker_prefill_chunk_tokens)
        if (
            isinstance(thinker_audio_prefill_chunk_tokens, bool)
            or not isinstance(thinker_audio_prefill_chunk_tokens, Integral)
            or int(thinker_audio_prefill_chunk_tokens) < 0
        ):
            raise ValueError(
                "thinker_audio_prefill_chunk_tokens deve essere >= 0; "
                "0 disabilita il chunking direct-audio"
            )
        self.thinker_audio_prefill_chunk_tokens = int(thinker_audio_prefill_chunk_tokens)
        if thinker_compile_mode not in {"off", "default", "reduce-overhead"}:
            raise ValueError(
                "thinker_compile_mode deve essere 'off', 'default' o 'reduce-overhead'"
            )
        if (
            isinstance(thinker_static_cache_max_len, bool)
            or not isinstance(thinker_static_cache_max_len, Integral)
            or int(thinker_static_cache_max_len) < 8
        ):
            raise ValueError("thinker_static_cache_max_len deve essere un intero >= 8")
        self.thinker_compile_mode = thinker_compile_mode
        self.thinker_static_cache_max_len = int(thinker_static_cache_max_len)
        self._compiled_thinker = None
        self.v9_recipe: str | None = None
        self.qwen_audio_recipe = QWEN_AUDIO_CONDITIONING_RECIPE
        self.thinker = None
        self._thinker_wrapper = None
        self.tokenizer = None
        self.talker = None
        self.v9 = None
        self.causal_conv = None
        self.v9_stream: IncrementalV9Projector | None = None
        self.subtalker_graph: Any | None = None
        self.residual_sampler: Any | None = None
        self.listener: Any | None = None
        # The real server must not silently downgrade to a non-causal or
        # missing listener; tests/embedded compatibility engines may opt out.
        self.enforce_production_listener = True
        self.enable_listener_silence_skip = os.getenv(
            "GEMMA4_OMNI_LISTENER_SILENCE_SKIP", "0"
        ).strip().lower() in {"1", "true", "yes"}
        self._talker_lock = threading.RLock()
        self._thinker_lock = threading.Lock()
        self._thinker_audio_forward_active = threading.Event()
        self._thinker_audio_prefill_lock_hold_max_ms = 0.0
        self._mtp_ready = False  # MTP parallelo: disattivato (fallback generate HF bit-exact)
        self.loaded = False
        self._stop_event = False
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    def load(
        self,
        *,
        workspace: str = "/workspace",
        speaker_reference_wav: str | None = None,
        speaker_reference_sha256: str | None = None,
    ) -> bool:
        try:
            t0 = time.time()
            self.workspace = Path(workspace)
            configured_control6_paths = (
                self.duplex_control6_adapter_dir,
                self.duplex_control6_head_path,
                self.duplex_control6_manifest_path,
            )

            def workspace_path(value: str | None) -> str | None:
                if value is None:
                    return None
                candidate = Path(value)
                return str(candidate if candidate.is_absolute() else Path(workspace) / candidate)

            if self.enable_gemma_native_control6_policy:
                if not self.native_control6_adapter_path:
                    raise ValueError("gemma_native_control6 requires --native-control6-adapter")
                if not self.ablate_response_lora and not self.gemma_qlora_adapter_dir:
                    raise ValueError("gemma_native_control6 requires response LoRA disabled")
                self.native_control6_adapter_path = workspace_path(self.native_control6_adapter_path)
                if not Path(self.native_control6_adapter_path).is_file():
                    raise FileNotFoundError(f"native Control6 checkpoint missing: {self.native_control6_adapter_path}")
                if self.native_control6_semantic_prefix_enabled:
                    # This small CPU metadata read happens before the large model
                    # is loaded, so a stale listener/checkpoint contract fails early.
                    from duplex_kernel.native_control6_prompt import validate_native_control6_semantic_prefix_metadata

                    checkpoint_metadata = torch.load(
                        self.native_control6_adapter_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    self.native_control6_semantic_prefix_metadata = (
                        validate_native_control6_semantic_prefix_metadata(checkpoint_metadata)
                    )
            elif self.native_control6_adapter_path is not None:
                raise ValueError("--native-control6-adapter requires --enable-gemma-native-control6")
            if (
                self.ablate_response_lora
                and not self.duplex_response_bundle_dir
                and not self.enable_gemma_native_control6_policy
            ):
                raise ValueError("--ablate-response-lora richiede --duplex-response-bundle-dir")
            if self.duplex_response_bundle_dir:
                if any(configured_control6_paths):
                    raise ValueError(
                        "il response bundle include il proprio adapter/head; non combinarlo con asset legacy"
                    )
                _validate_response_bundle_qlora_compatibility(
                    bundle_enabled=True,
                    gemma_qlora_enabled=bool(self.gemma_qlora_adapter_dir),
                    native_control6_enabled=self.enable_gemma_native_control6_policy,
                    response_lora_ablated=self.ablate_response_lora,
                )
                self.duplex_response_bundle_dir = workspace_path(self.duplex_response_bundle_dir)
                bundle_root = Path(self.duplex_response_bundle_dir)
                if not bundle_root.is_dir() or not (bundle_root / "bundle_manifest.json").is_file():
                    raise FileNotFoundError(
                        f"duplex response bundle incompleto: {bundle_root}/bundle_manifest.json"
                    )
                from duplex_kernel.audio_response_protocol import normalize_audio_response_protocol
                from duplex_kernel.duplex_control6 import read_duplex_response_bundle

                bundle_contract = read_duplex_response_bundle(self.duplex_response_bundle_dir)
                self.audio_response_protocol = normalize_audio_response_protocol(
                    bundle_contract["manifest"].get("response_protocol")
                )
            elif self.enable_duplex_control6:
                if not self.duplex_control6_adapter_dir or not self.duplex_control6_manifest_path:
                    raise ValueError(
                        "--enable-duplex-control6 richiede adapter LoRA e training manifest; "
                        "la policy non può fare fallback silenzioso"
                    )
                if self.gemma_qlora_adapter_dir:
                    raise ValueError(
                        "specificare --gemma-qlora-adapter-dir per il solo smoke oppure "
                        "--enable-duplex-control6 per policy+adapter, non entrambi"
                    )
                self.duplex_control6_adapter_dir = workspace_path(self.duplex_control6_adapter_dir)
                self.duplex_control6_head_path = workspace_path(self.duplex_control6_head_path)
                self.duplex_control6_manifest_path = workspace_path(self.duplex_control6_manifest_path)
                if not Path(self.duplex_control6_adapter_dir).is_dir():
                    raise FileNotFoundError(f"duplex_control6 adapter mancante: {self.duplex_control6_adapter_dir}")
                if not Path(self.duplex_control6_manifest_path).is_file():
                    raise FileNotFoundError(f"duplex_control6 manifest mancante: {self.duplex_control6_manifest_path}")
            elif any(configured_control6_paths):
                raise ValueError("asset duplex_control6 richiedono --enable-duplex-control6 esplicito")
            if (
                self._talker_native_audio_q1_requested
                and self.audio_response_protocol != AUDIO_RESPONSE_PROTOCOL_V2
            ):
                raise ValueError(
                    f"{_TALKER_NATIVE_AUDIO_Q1_ENV}=1 requires {AUDIO_RESPONSE_PROTOCOL_V2}"
                )
            if self.gemma_qlora_adapter_dir:
                candidate = Path(self.gemma_qlora_adapter_dir)
                self.gemma_qlora_adapter_dir = str(
                    candidate if candidate.is_absolute() else Path(workspace) / candidate
                )
                if not Path(self.gemma_qlora_adapter_dir).is_dir():
                    raise FileNotFoundError(f"Gemma QLoRA adapter mancante: {self.gemma_qlora_adapter_dir}")
            th = load_thinker(workspace, device=self.device, dtype=self.dtype)
            self._thinker_wrapper = th
            if self.duplex_response_bundle_dir:
                from duplex_kernel.duplex_control6 import load_duplex_response_bundle_runtime

                self.duplex_control6_runtime = load_duplex_response_bundle_runtime(
                    th,
                    bundle_dir=self.duplex_response_bundle_dir,
                    device=self.device,
                    apply_response_lora=not self.ablate_response_lora,
                )
                logger.info(
                    "[engine] response bundle merged before listener binding bundle_id=%s adapter=%s head=%s protocol=%s",
                    self.duplex_control6_runtime["bundle_id"],
                    self.duplex_control6_runtime["adapter_path"],
                    self.duplex_control6_runtime["control_head"],
                    self.audio_response_protocol or "legacy_v1",
                )
            elif self.enable_duplex_control6:
                from duplex_kernel.duplex_control6 import load_control6_runtime

                self.duplex_control6_runtime = load_control6_runtime(
                    th,
                    adapter_dir=self.duplex_control6_adapter_dir,
                    head_path=self.duplex_control6_head_path,
                    manifest_path=self.duplex_control6_manifest_path,
                    device=self.device,
                )
                logger.info(
                    "[engine] duplex_control6 merged before listener pointer binding run_id=%s base=%s adapter=%s",
                    self.duplex_control6_runtime["run_id"],
                    self.duplex_control6_runtime["base_path"],
                    self.duplex_control6_runtime["adapter_path"],
                )
            if self.gemma_qlora_adapter_dir:
                from duplex_kernel.duplex_control6 import merge_gemma_adapter_into_thinker

                base_path = merge_gemma_adapter_into_thinker(th, self.gemma_qlora_adapter_dir)
                self.gemma_qlora_adapter_loaded = str(Path(self.gemma_qlora_adapter_dir).resolve())
                logger.info(
                    "[engine] Gemma QLoRA adapter-only merged before listener pointer binding base=%s adapter=%s",
                    base_path,
                    self.gemma_qlora_adapter_loaded,
                )
            if (
                self._talker_compiled_decode_requested
                and self.audio_response_protocol != AUDIO_RESPONSE_PROTOCOL_V2
            ):
                raise ValueError(
                    f"{_TALKER_COMPILED_DECODE_ENV}=1 requires response protocol "
                    f"{AUDIO_RESPONSE_PROTOCOL_V2}"
                )
            self.thinker = th.model
            self.tokenizer = th.tokenizer
            if self.enable_gemma_native_control6_policy:
                if self.native_control6_semantic_prefix_enabled:
                    live_boa_token_id = getattr(getattr(self.thinker, "config", None), "boa_token_id", None)
                    checkpoint_boa_token_id = self.native_control6_semantic_prefix_metadata[
                        "semantic_prefix_boa_token_id"
                    ]
                    if live_boa_token_id is None or int(live_boa_token_id) != int(checkpoint_boa_token_id):
                        raise RuntimeError(
                            "native Control6 checkpoint BOA token differs from the loaded Gemma config: "
                            f"checkpoint={checkpoint_boa_token_id!r}, live={live_boa_token_id!r}"
                        )
                from training.duplex_qlora.native_control6 import (
                    load_native_control6_adapter,
                    select_tied_output_rows,
                )
                output_head = getattr(self.thinker, "lm_head", None)
                output_weight = getattr(output_head, "weight", None)
                if output_weight is None:
                    raise RuntimeError("loaded Gemma thinker has no native lm_head.weight")
                live_rows = select_tied_output_rows(output_weight)
                self.native_control6_runtime, self.native_control6_report = load_native_control6_adapter(
                    self.native_control6_adapter_path,
                    base_rows=live_rows,
                    device=self.device,
                )
                self.native_control6_runtime.eval()
                for parameter in self.native_control6_runtime.parameters():
                    parameter.requires_grad_(False)
                logger.info(
                    "[engine] native Gemma Control6 loaded checkpoint=%s rows=6 hidden=1536 rank=%d response_lora_applied=false",
                    self.native_control6_adapter_path,
                    self.native_control6_runtime.rank,
                )
            if self.enable_duplex_control6:
                runtime_lexicon_path = (self.duplex_control6_runtime or {}).get("backchannel_lexicon_path")
                lexicon_path = (
                    Path(runtime_lexicon_path)
                    if runtime_lexicon_path
                    else Path(self.duplex_control6_manifest_path).with_name("backchannel_lexicon.json")
                )
                if not lexicon_path.is_file():
                    raise FileNotFoundError(
                        f"backchannel lexicon train mancante nel bundle attivo: {lexicon_path}"
                    )
                with lexicon_path.open("r", encoding="utf-8") as handle:
                    lexicon = json.load(handle)
                if not isinstance(lexicon, dict) or lexicon.get("schema_version") != 1:
                    raise ValueError("schema backchannel_lexicon.json non supportato")
                if lexicon.get("language") != "it":
                    raise ValueError("backchannel_lexicon.json deve dichiarare language='it'")
                if lexicon.get("source_split", "train") != "train":
                    raise ValueError("backchannel_lexicon.json deve provenire dallo split train")
                phrases = lexicon.get("phrases")
                if not isinstance(phrases, list) or any(not isinstance(phrase, str) for phrase in phrases):
                    raise ValueError("backchannel_lexicon.json deve contenere phrases: list[str]")
                eos_value = getattr(getattr(self.thinker, "generation_config", None), "eos_token_id", None)
                if eos_value is None:
                    eos_value = getattr(getattr(self.thinker, "config", None), "eos_token_id", None)
                eos_ids = normalize_eos_token_ids(eos_value)
                if not eos_ids:
                    eos_ids = normalize_eos_token_ids(getattr(self.tokenizer, "eos_token_id", None))
                    eos_ids.update(normalize_eos_token_ids(getattr(self.tokenizer, "eot_id", None)))
                self._backchannel_token_trie = _build_backchannel_token_trie(
                    phrases, self.tokenizer, eos_token_ids=eos_ids
                )
                self._backchannel_say_phrases = tuple(dict.fromkeys(phrase.strip() for phrase in phrases))
                self._backchannel_phrase_count = len(self._backchannel_say_phrases)
                logger.info(
                    "[engine] BACKCHANNEL constrained to %d train phrases / %d token prefixes from %s",
                    self._backchannel_phrase_count,
                    len(self._backchannel_token_trie),
                    lexicon_path,
                )
            from duplex_kernel.listener import GemmaListener
            self.listener = GemmaListener.from_thinker(
                th,
                runtime_profile="production",
                enable_silence_skip=self.enable_listener_silence_skip,
            )
            if hasattr(self.listener, "backend") and hasattr(self.listener.backend, "_lock"):
                self._thinker_lock = self.listener.backend._lock
            logger.info(
                "[engine] thinker + continuous listener in %.1fs | prefill_chunk_tokens=%d "
                "audio_lm_chunk_tokens=%d compile=%s static_cache=%d",
                time.time() - t0,
                self.thinker_prefill_chunk_tokens,
                self.thinker_audio_prefill_chunk_tokens,
                self.thinker_compile_mode,
                self.thinker_static_cache_max_len,
            )

            if self.ablate_session_voice_conditioning:
                resolved_speaker_wav = None
                resolved_speaker_sha = None
            else:
                resolved_speaker_wav = speaker_reference_wav or SPEAKER_REFERENCE_WAV or None
                resolved_speaker_sha = speaker_reference_sha256 or SPEAKER_REFERENCE_SHA256 or None

            if (
                resolved_speaker_wav is None
            ) != (
                resolved_speaker_sha is None
            ):
                raise ValueError(
                    "speaker reference path/SHA "
                    "incompleti"
                )

            self.talker = load_talker(
                workspace,
                device=self.device,
                speaker_reference_wav=(
                    resolved_speaker_wav
                ),
                speaker_reference_sha256=(
                    resolved_speaker_sha
                ),
            )
            logger.info(
                "[engine] talker+code2wav recipe=%s in %.1fs",
                self.qwen_audio_recipe,
                time.time() - t0,
            )

            self.v9, self.causal_conv = load_v9_conditioning(
                profile=self.v9_profile,
                device=self.device,
                dtype=self.dtype,
            )
            self.v9_recipe = getattr(self.v9, "_v9_recipe", None)
            if self.v9_recipe is None:
                raise RuntimeError("V9 loader non ha attestato la recipe")
            self.v9_stream = (
                IncrementalV9Projector(self.v9, self.causal_conv, max_seq_len=16_384)
                if self.v9_profile == V9_PROFILE_CAUSAL_L0
                else None
            )
            logger.info(
                "[engine] V9 profile=%s recipe=%s in %.1fs",
                self.v9_profile,
                self.v9_recipe,
                time.time() - t0,
            )

            # CUDA graph sub-talker (ADR-026): prefill + 14 step autoregressivi
            # catturati in CUDA graph statici, sampling esterno bit-exact con RNG
            # globale (warper T->top_k->top_p come transformers). Fail-soft: eager.
            from duplex_kernel.subtalker_graph import SubtalkerGraph

            cp = self.talker.talker.code_predictor
            self.subtalker_graph = SubtalkerGraph(
                cp,
                talker_embed=self.talker.talker.get_input_embeddings(),
                device=self.device,
            ).capture()
            logger.info("[engine] subtalker graph: %s", "catturato" if self.subtalker_graph.captured else "eager")
            if self.controller_guided_talk_contexts:
                from duplex_kernel.response_talk_context import ResponseTalkContextPool

                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    torch.cuda.synchronize(device=self.device)
                    allocated_before = int(torch.cuda.memory_allocated(device=self.device))
                    reserved_before = int(torch.cuda.memory_reserved(device=self.device))
                else:
                    allocated_before = reserved_before = 0
                talk_graphs = [self.subtalker_graph]
                for slot_index in range(1, self.controller_guided_talk_contexts):
                    slot_graph = SubtalkerGraph(
                        cp,
                        talker_embed=self.talker.talker.get_input_embeddings(),
                        device=self.device,
                    ).capture()
                    talk_graphs.append(slot_graph)
                code2wav_graphs = None
                code2wav_graph_prerequisites = (
                    torch.cuda.is_available()
                    and str(self.device).startswith("cuda")
                    and CODE2WAV_DECODE_FRAMES == Code2WavWindowGraph.NEW_FRAMES
                    and getattr(self.talker, "code2wav", None) is not None
                )
                if config.CODE2WAV_GRAPH_FAMILY_ENABLED and not code2wav_graph_prerequisites:
                    raise RuntimeError(
                        "requested Code2Wav graph family requires CUDA, "
                        "CODE2WAV_DECODE_FRAMES=2, and a loaded Code2Wav decoder"
                    )
                if config.CODE2WAV_GRAPH_FAMILY_ENABLED:
                    code2wav_graphs = []
                    family_capture_started = time.perf_counter()
                    for slot_index in range(self.controller_guided_talk_contexts):
                        family = Code2WavWindowGraphFamily(
                            self.talker.code2wav,
                            device=str(self.device),
                            include_odd_tail=config.CODE2WAV_GRAPH_ODD_TAIL_ENABLED,
                        ).capture()
                        code2wav_graphs.append(family)
                    family_capture_seconds = time.perf_counter() - family_capture_started
                    logger.info(
                        "[engine] Code2Wav exact-window graph family captured slots=%d/%d "
                        "graphs_per_slot=%d odd_tail=%s windows=%s capture_seconds=%.3f "
                        "static_buffers_bytes_per_slot=%d",
                        sum(family.captured for family in code2wav_graphs),
                        len(code2wav_graphs),
                        len(code2wav_graphs[0].window_frames) if code2wav_graphs else 0,
                        config.CODE2WAV_GRAPH_ODD_TAIL_ENABLED,
                        list(code2wav_graphs[0].window_frames) if code2wav_graphs else [],
                        family_capture_seconds,
                        code2wav_graphs[0].static_buffers_bytes if code2wav_graphs else 0,
                    )
                elif code2wav_graph_prerequisites:
                    code2wav_graphs = []
                    for slot_index in range(self.controller_guided_talk_contexts):
                        decode_graph = Code2WavWindowGraph(
                            self.talker.code2wav,
                            device=str(self.device),
                        )
                        decode_stream = torch.cuda.Stream(
                            device=torch.device(self.device),
                            priority=0,
                        )
                        decode_graph.capture(decode_stream)
                        decode_stream.synchronize()
                        code2wav_graphs.append(decode_graph)
                    logger.info(
                        "[engine] Code2Wav window graphs catturati: %d/%d",
                        sum(graph.captured for graph in code2wav_graphs),
                        len(code2wav_graphs),
                    )
                self.response_talk_context_pool = ResponseTalkContextPool(
                    talk_graphs, code2wav_graphs=code2wav_graphs
                )
                element_bytes = int(torch.tensor([], dtype=self.dtype).element_size())
                cp_config = getattr(cp, "config", None)
                cp_layers = int(getattr(cp_config, "num_hidden_layers", 0) or 0)
                cp_heads = int(
                    getattr(cp_config, "num_key_value_heads", 0)
                    or getattr(cp_config, "num_attention_heads", 0)
                    or 0
                )
                cp_hidden = int(getattr(cp_config, "hidden_size", 0) or 0)
                cp_head_dim = int(
                    getattr(cp_config, "head_dim", 0)
                    or (cp_hidden // cp_heads if cp_heads else 0)
                )
                qwen_config = getattr(self.talker.talker, "config", None)
                qwen_layers = int(getattr(qwen_config, "num_hidden_layers", 0) or 0)
                qwen_heads = int(
                    getattr(qwen_config, "num_key_value_heads", 0)
                    or getattr(qwen_config, "num_attention_heads", 0)
                    or 0
                )
                qwen_hidden = int(getattr(qwen_config, "hidden_size", 0) or 0)
                qwen_head_dim = int(
                    getattr(qwen_config, "head_dim", 0)
                    or (qwen_hidden // qwen_heads if qwen_heads else 0)
                )
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    torch.cuda.synchronize(device=self.device)
                    allocated_after = int(torch.cuda.memory_allocated(device=self.device))
                    reserved_after = int(torch.cuda.memory_reserved(device=self.device))
                else:
                    allocated_after = reserved_after = 0
                code2wav_graph_infos = (
                    [graph.runtime_info() for graph in code2wav_graphs]
                    if code2wav_graphs is not None
                    else []
                )
                self.response_talk_context_memory = {
                    "capacity": len(talk_graphs),
                    "allocated_before_bytes": allocated_before,
                    "allocated_after_bytes": allocated_after,
                    "allocated_delta_bytes": allocated_after - allocated_before,
                    "reserved_before_bytes": reserved_before,
                    "reserved_after_bytes": reserved_after,
                    "reserved_delta_bytes": reserved_after - reserved_before,
                    "subtalker_static_kv_estimate_bytes_per_slot": (
                        cp_layers * 2 * cp_heads * cp_head_dim
                        * (self.subtalker_graph.max_code_groups + 2)
                        * element_bytes
                    ),
                    "qwen_kv_estimate_bytes_per_token": (
                        qwen_layers * 2 * qwen_heads * qwen_head_dim * element_bytes
                    ),
                    "code2wav_graph_family_enabled": bool(
                        config.CODE2WAV_GRAPH_FAMILY_ENABLED
                    ),
                    "code2wav_graph_odd_tail_enabled": bool(
                        config.CODE2WAV_GRAPH_ODD_TAIL_ENABLED
                    ),
                    "code2wav_graph_capture_status_by_slot": [
                        {
                            "captured": bool(info.get("captured")),
                            "graph_count": int(
                                info.get("graph_count", 1 if info.get("captured") else 0)
                            ),
                            "window_frames": info.get("window_frames"),
                            "static_buffers_bytes": int(info.get("static_buffers_bytes", 0)),
                            "capture_error": info.get("capture_error"),
                        }
                        for info in code2wav_graph_infos
                    ],
                }
                logger.info(
                    "[engine] response_talk_context_pool %s",
                    json.dumps(self.response_talk_context_memory, sort_keys=True),
                )

            if self.thinker_compile_mode != "off":
                # Load Talker/V9 and capture the sub-talker before compiling
                # Thinker. Transformers registers ModelOutput classes lazily
                # while those backends initialize; warming earlier makes the
                # first request recompile when that process-wide registry grows.
                # Keep the compile warmup before `loaded` and server readiness.
                self._prepare_compiled_thinker()

            if self._talker_compiled_decode_requested and not self._talker_native_audio_q1_requested:
                self._prepare_compiled_talker_q1()

            if self.frame_residual_adapter_checkpoint_path is not None:
                if self.runtime_backend != "native":
                    raise ValueError("frame residual adapter requires runtime_backend=native")
                if self.audio_response_protocol != AUDIO_RESPONSE_PROTOCOL_V2:
                    raise ValueError("frame residual adapter requires heard_action_say_v2 direct audio")
                if not self.duplex_response_bundle_dir or not self.ablate_response_lora:
                    raise ValueError("frame residual adapter requires controller bundle with response LoRA ablated")
                if self.controller_guided_talk_contexts != 2:
                    raise ValueError("frame residual adapter requires the two-slot response pool")
                if not bool(getattr(self, "enforce_v9_training_voice", False)) or self.ablate_session_voice_conditioning:
                    raise ValueError("frame residual adapter requires the exact training-default speaker conditioning")
                if self.v9_conditioning_source != "live_generated" or self.v9_dual_audio_context_experimental:
                    raise ValueError("frame residual adapter requires the plain semantic-text Qwen route")
                if getattr(self, "response_intent_local_projector", None) is not None:
                    raise ValueError("frame residual adapter cannot share a response with a prefix projector")
                if not self._talker_native_audio_q1_requested:
                    raise ValueError("frame residual adapter requires requested native audio-Q1")
                adapter_path = Path(self.frame_residual_adapter_checkpoint_path).expanduser()
                if not adapter_path.is_absolute():
                    adapter_path = self.workspace / adapter_path
                self.frame_residual_adapter, payload = load_frame_residual_adapter(
                    adapter_path, device=self.device, dtype=self.dtype
                )
                if self.frame_residual_adapter.architecture_id != "response_hidden_q1_query_film_v2":
                    raise ValueError(
                        "production serving requires the response_hidden_q1_query_film_v2 Query FiLM v2 checkpoint"
                    )
                self.frame_residual_adapter_resolved_path = str(adapter_path.resolve())
                self.frame_residual_adapter_architecture = str(
                    self.frame_residual_adapter.architecture_id
                )
                logger.info(
                    "[engine] frame_residual_q1 loaded architecture=%s schema=%s checkpoint=%s route=plain_native_q1 frozen=true",
                    self.frame_residual_adapter_architecture,
                    payload.get("schema_version"),
                    self.frame_residual_adapter_resolved_path,
                )

            if self.audio_response_protocol == "heard_action_say_v2":
                self._warmup_direct_v2_audio_path()

            if self.ablate_session_voice_conditioning:
                # Startup warmup exercises the configured audio path, but user
                # sessions in this diagnostic run must match the no-reference
                # fixed probe and must not inherit its warmup x-vector.
                self.talker.default_voice_conditioning = None
                if hasattr(self.talker, "voice_conditioning"):
                    delattr(self.talker, "voice_conditioning")
                logger.info(
                    "[engine] session voice-conditioning ablation enabled; "
                    "no speaker reference or profile will be used for sessions"
                )



            if self.diagnostic_talker_seed is not None:
                # One startup seed for the shared sampling stream. With a
                # concurrent pool this is diagnostic initialization only; it
                # does not make individual responses replayable.
                import random

                random.seed(self.diagnostic_talker_seed)
                np.random.seed(self.diagnostic_talker_seed)
                torch.manual_seed(self.diagnostic_talker_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.diagnostic_talker_seed)
                logger.info(
                    "[engine] diagnostic_talker_seed=%d scope=startup_global_not_per_response",
                    self.diagnostic_talker_seed,
                )
            self.loaded = True
            logger.info("[engine] LOAD COMPLETO in %.1fs", time.time() - t0)
            return True
        except Exception as exc:
            logger.exception("Engine load fallito: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Thinker
    # ------------------------------------------------------------------
    def _prepare_compiled_talker_q1(self) -> None:
        """Set up the optional DynamicCache Q1 wrapper before any request runs."""
        if (
            not self._talker_compiled_decode_requested
            or (
                getattr(self, "_talker_native_audio_q1_requested", False)
                and getattr(self, "talker_q1_native_status", None)
                != "eager_fallback_before_ready"
            )
        ):
            return
        if self.audio_response_protocol != AUDIO_RESPONSE_PROTOCOL_V2:
            raise ValueError(
                f"{_TALKER_COMPILED_DECODE_ENV}=1 requires {AUDIO_RESPONSE_PROTOCOL_V2}"
            )
        try:
            model = self.talker.talker.model
            q1_module = _QwenDynamicCacheQ1(model)
            self._compiled_talker_q1 = torch.compile(
                q1_module,
                dynamic=True,
                fullgraph=False,
                mode="default",
            )
        except Exception as exc:
            self._set_talker_q1_eager_fallback(exc)
            return
        self.talker_q1_compile_status = "warming"
        logger.info(
            "[engine] optional Qwen Q1 torch.compile configured dynamic=True fullgraph=False"
        )

    @staticmethod
    def _talker_q1_dynamo_counts() -> tuple[int, int] | None:
        """Read the same lightweight Torch counters used by the exact-version probe."""
        try:
            counters = torch._dynamo.utils.counters
            graph_break_count = sum(
                int(value) for value in counters.get("graph_break", {}).values()
            )
            unique_graphs = int(counters.get("stats", {}).get("unique_graphs", 0))
            return unique_graphs, graph_break_count
        except Exception:
            return None

    def _verify_talker_q1_compile_capture(self) -> None:
        before = self._talker_q1_dynamo_before
        after = self._talker_q1_dynamo_counts()
        if before is None or after is None:
            raise _CompiledTalkerDecodeError(
                "Torch Dynamo counters unavailable; cannot verify compiled Qwen Q1 warmup"
            )
        graph_delta = after[0] - before[0]
        break_delta = after[1] - before[1]
        if graph_delta != 1 or break_delta != 0:
            raise _CompiledTalkerDecodeError(
                "compiled Qwen Q1 warmup did not settle on one dynamic graph "
                f"(unique_graphs_delta={graph_delta}, graph_breaks_delta={break_delta})"
            )

    def _set_talker_q1_eager_fallback(self, exc: BaseException) -> None:
        self._compiled_talker_q1 = None
        self.talker_q1_backend = "eager"
        self.talker_q1_compile_status = "eager_fallback_before_ready"
        self.talker_q1_compile_fallback_reason = (
            f"{type(exc).__name__}: {str(exc)[:240]}"
        )
        logger.warning(
            "[engine] optional Qwen Q1 compile unavailable before readiness; "
            "using eager decode: %s",
            self.talker_q1_compile_fallback_reason,
        )

    def _set_native_talker_q1_fallback(self, exc: BaseException) -> None:
        self.talker_q1_native_status = "eager_fallback_before_ready"
        self.talker_q1_native_fallback_reason = f"{type(exc).__name__}: {str(exc)[:240]}"
        self.talker_q1_backend = "eager"
        logger.warning(
            "[engine] native Qwen audio-Q1 unavailable before readiness; using configured fallback: %s",
            self.talker_q1_native_fallback_reason,
        )

    def _new_native_talker_q1(
        self,
        talker: Any,
        past_key_values: Any,
        *,
        max_audio_frames: int,
        talk_context: Any | None = None,
    ) -> Any:
        """Import the current response KV into its persistent Q1 slot context."""
        if getattr(self, "audio_response_protocol", None) != AUDIO_RESPONSE_PROTOCOL_V2:
            raise _NativeTalkerDecodeError(
                f"{_TALKER_NATIVE_AUDIO_Q1_ENV}=1 requires {AUDIO_RESPONSE_PROTOCOL_V2}"
            )
        max_audio_frames = int(max_audio_frames)
        if max_audio_frames <= 0:
            raise ValueError("max_audio_frames native Q1 deve essere positivo")

        if self._talker_native_audio_q1_graph_requested:
            if talk_context is None:
                if not self._native_q1_startup_warmup_active:
                    raise _NativeTalkerDecodeError(
                        "native Q1 graph serving requires an exclusive response talk-context slot"
                    )
                slot_id = 0
            else:
                if not talk_context.is_current():
                    raise _NativeTalkerDecodeError("stale response talk-context before Q1 graph import")
                slot_id = int(talk_context.slot_id)
            executor = self._native_q1_slot_executors.get(slot_id)
            if executor is None:
                raise _NativeTalkerDecodeError(
                    f"native Q1 graph slot {slot_id} was not captured before readiness"
                )
            try:
                executor.fence_q1_graph()
                executor.import_dynamic_cache_async(
                    past_key_values,
                    max_audio_frames=max_audio_frames,
                )
            except Exception as exc:
                raise _NativeTalkerDecodeError(
                    f"native Q1 graph slot {slot_id} import failed: {type(exc).__name__}: {exc}"
                ) from exc
            return _NativeQ1GraphResponseLease(executor, slot_id)

        from duplex_kernel.native_qwen_talker import NativeQwenTalker

        sequence_length = _cache_sequence_length(past_key_values)
        executor = NativeQwenTalker(
            talker,
            max_cache_len=sequence_length + max_audio_frames,
        )
        try:
            executor.import_dynamic_cache_async(
                past_key_values,
                max_audio_frames=max_audio_frames,
            )
        except Exception:
            executor.close()
            raise
        return executor

    def _native_talker_q1_step(
        self,
        executor: Any,
        inputs_embeds: torch.Tensor,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Advance one audio-only Q1 step; the returned hidden is borrowed until the next step."""
        warmup_probe = (
            bool(getattr(self, "_warmup_diagnostic_active", False))
            and not bool(getattr(self, "_warmup_debug_first_q1_checked", False))
        )
        decoder = getattr(getattr(self, "talker", None), "code2wav", None)
        rotary_before: list[dict[str, Any]] = []
        if warmup_probe:
            rotary_before = _warmup_diag_rotary_state(decoder)
        graph_step_used = False
        try:
            graph_step = getattr(executor, "step_audio_graph_async", None)
            graph_requested = bool(
                getattr(self, "_talker_native_audio_q1_graph_requested", False)
            )
            if graph_requested:
                if not callable(graph_step):
                    raise RuntimeError("native Q1 graph executor missing graph-step method")
                hidden = graph_step(inputs_embeds)
                graph_step_used = True
            else:
                hidden = executor.step_audio_async(inputs_embeds)
            if (
                not isinstance(hidden, torch.Tensor)
                or hidden.ndim != 3
                or hidden.shape[0] != inputs_embeds.shape[0]
                or hidden.shape[1] != 1
                or hidden.shape[-1] != 2048
            ):
                raise RuntimeError("native Qwen Q1 returned an invalid hidden state")
            if warmup_probe:
                try:
                    graph_info: dict[str, Any] = {}
                    info_fn = getattr(executor, "q1_graph_info", None)
                    if callable(info_fn):
                        info = info_fn()
                        if isinstance(info, dict):
                            graph_info = {
                                str(key): value
                                for key, value in info.items()
                                if value is None or isinstance(value, (str, bool, int, float))
                            }
                    output_snapshot = hidden.detach().clone()
                    self._warmup_debug_first_q1 = {
                        "backend": "native_audio_q1_graph" if graph_step_used else "native_audio_q1_eager",
                        "input": _warmup_diag_tensor(inputs_embeds),
                        "output": _warmup_diag_tensor(output_snapshot),
                        "nonfinite": not bool(torch.isfinite(output_snapshot).all().item()),
                        "rotary_buffers_before_q1": rotary_before,
                        "rotary_buffers_after_q1": _warmup_diag_rotary_state(decoder),
                        "q1_graph_info": graph_info,
                    }
                except Exception as diag_exc:
                    self._warmup_debug_first_q1 = {
                        "backend": "native_audio_q1_graph" if graph_step_used else "native_audio_q1_eager",
                        "diagnostic_error": type(diag_exc).__name__,
                        "rotary_buffers_before_q1": rotary_before,
                    }
                finally:
                    self._warmup_debug_first_q1_checked = True
            if metrics is not None:
                metrics["talker_q1_backend"] = "native_audio_q1"
                metrics["native_talker_q1_calls"] = int(
                    metrics.get("native_talker_q1_calls", 0)
                ) + 1
                if graph_step_used:
                    metrics["native_talker_q1_graph_calls"] = int(
                        metrics.get("native_talker_q1_graph_calls", 0)
                    ) + 1
            return hidden
        except Exception as exc:
            if warmup_probe:
                self._warmup_debug_first_q1 = {
                    "backend": "native_audio_q1_graph" if graph_step_used else "native_audio_q1_eager",
                    "input": _warmup_diag_tensor(inputs_embeds),
                    "error": f"{type(exc).__name__}: {exc}",
                    "rotary_buffers_before_q1": rotary_before,
                }
                self._warmup_debug_first_q1_checked = True
            raise _NativeTalkerDecodeError(
                f"native Qwen audio-Q1 failed: {type(exc).__name__}: {exc}"
            ) from exc

    @torch.inference_mode()
    def _warmup_native_talker_q1(
        self,
        text: str,
        h_pfx: torch.Tensor,
        voice_conditioning: dict[str, Any],
    ) -> None:
        """Warm native Q1 and capture each persistent slot before readiness."""
        t = self.talker.talker
        voice = _resolve_talker_voice_conditioning(self.talker, voice_conditioning)
        static_part, text_part, bos_part, tts_pad_embed = _build_talker_prefill_parts(
            self.talker,
            text,
            voice,
        )
        out_static = t.model(inputs_embeds=static_part, use_cache=True)
        dynamic = torch.cat([text_part, h_pfx.to(self.dtype), bos_part], dim=1)
        out_prefix = t.model(
            inputs_embeds=dynamic,
            past_key_values=out_static.past_key_values,
            use_cache=True,
        )
        cache = out_prefix.past_key_values
        codes = torch.zeros((1, 1, 16), device=self.device, dtype=torch.long)
        warmup_query = build_qwen_audio_conditioning(
            t,
            codes,
            tts_pad_embed=tts_pad_embed,
            generation_step=0,
        )

        if self._talker_native_audio_q1_graph_requested:
            pool = self.response_talk_context_pool
            if pool is None or int(pool.capacity) != 2:
                raise _NativeTalkerDecodeError(
                    "native Q1 graph startup requires exactly two prebuilt response slots"
                )
            from duplex_kernel.native_qwen_talker import NativeQwenTalker

            sequence_length = _cache_sequence_length(cache)
            graph_audio_frames = int(self._native_q1_graph_max_audio_frames)
            cache_capacity = (
                max(sequence_length, int(self._native_q1_graph_prefill_headroom))
                + graph_audio_frames
            )
            prepared: dict[int, Any] = {}
            try:
                for slot_id in range(int(pool.capacity)):
                    executor = NativeQwenTalker(t, max_cache_len=cache_capacity)
                    prepared[slot_id] = executor
                    response_stream = pool.response_stream_for_slot(slot_id)
                    if response_stream is None:
                        raise _NativeTalkerDecodeError(
                            f"native Q1 graph slot {slot_id} has no private response stream"
                        )
                    info = executor.prepare_q1_graph(
                        cache,
                        max_audio_frames=graph_audio_frames,
                        warmup_audio_embed=warmup_query,
                        response_stream=response_stream,
                    )
                    pool.bind_q1_stream_for_slot(slot_id, response_stream)
                    graph_info = dict(info)
                    if (
                        not bool(graph_info.get("captured"))
                        or int(graph_info.get("capture_count", 0)) != 1
                        or int(graph_info.get("abi_version", -1)) != 4
                        or not bool(graph_info.get("device_cursor"))
                    ):
                        raise _NativeTalkerDecodeError(
                            f"native Q1 graph slot {slot_id} failed startup attestation: {graph_info}"
                        )
                self._native_q1_slot_executors = prepared
                self.talker_q1_native_graph_slots_captured = len(prepared)
            except BaseException:
                for executor in prepared.values():
                    try:
                        executor.close()
                    except Exception:
                        pass
                self._native_q1_slot_executors.clear()
                raise
            self.talker_q1_backend = "native_audio_q1"
            self.talker_q1_native_status = "ready"
            logger.info(
                "[engine] native Q1 CUDA graphs ready slots=%d audio_capacity=%d cache_capacity=%d",
                len(prepared),
                graph_audio_frames,
                cache_capacity,
            )
            return

        executor = self._new_native_talker_q1(t, cache, max_audio_frames=3)
        try:
            for generation_step in range(3):
                query = build_qwen_audio_conditioning(
                    t,
                    codes,
                    tts_pad_embed=tts_pad_embed,
                    generation_step=generation_step,
                )
                self._native_talker_q1_step(executor, query)
                self.talker_q1_native_warmup_calls += 1
        finally:
            executor.close()
        if self.talker_q1_native_warmup_calls < 3:
            raise _NativeTalkerDecodeError(
                "native Qwen audio-Q1 warmup did not execute three decode positions"
            )
        self.talker_q1_backend = "native_audio_q1"
        self.talker_q1_native_status = "ready"

    def _warmup_check_code2wav_weights(self, stage: str) -> dict[str, Any] | None:
        if not getattr(self, "_warmup_diagnostic_active", False):
            return None
        decoder = getattr(self.talker, "code2wav", None)
        if decoder is None:
            result = {"stage": stage, "available": False}
            self._warmup_debug_code2wav_weights[stage] = result
            return result
        flags: list[tuple[str, torch.Tensor]] = []
        specs: set[tuple[str, str]] = set()
        trainable = 0
        for name, parameter in decoder.named_parameters():
            if parameter.is_floating_point():
                flags.append((name, torch.isfinite(parameter.detach()).all()))
                specs.add((str(parameter.dtype), str(parameter.device)))
                trainable += int(bool(parameter.requires_grad))
        finite_values = torch.stack([flag for _name, flag in flags]).detach().cpu().tolist() if flags else []
        bad = [name for (name, _flag), finite in zip(flags, finite_values) if not finite]
        result = {"stage": stage, "available": True, "parameter_tensor_count": len(flags),
                  "trainable_parameter_tensor_count": trainable, "parameter_specs": sorted(specs),
                  "finite": not bad, "finite_parameter_tensor_count": len(flags) - len(bad),
                  "first_nonfinite_parameter_names": bad[:8]}
        self._warmup_debug_code2wav_weights[stage] = result
        logger.info("[warmup_code2wav_parameter_finiteness] %s", json.dumps(result, sort_keys=True, allow_nan=False))
        if bad:
            raise RuntimeError(f"Code2Wav parameters non-finite at {stage}: {bad[:4]}")
        return result

    def _qwen_q1_forward(
        self,
        model: Any,
        inputs_embeds: torch.Tensor,
        past_key_values: Any,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Run one decode query; only v2 may select the optional compiled path."""
        compiled = (
            getattr(self, "_compiled_talker_q1", None)
            if getattr(self, "audio_response_protocol", None) == AUDIO_RESPONSE_PROTOCOL_V2
            else None
        )
        if compiled is None:
            if metrics is not None:
                metrics.setdefault("talker_q1_backend", "eager")
            warmup_probe = (getattr(self, "_warmup_diagnostic_active", False)
                            and not getattr(self, "_warmup_debug_first_q1_checked", False))
            if warmup_probe:
                self._warmup_debug_first_q1 = {
                    "backend": "eager_torch_q1",
                    "input": _warmup_diag_tensor(inputs_embeds),
                    "runtime_backends": _warmup_diag_backends(),
                }
                self._warmup_check_code2wav_weights("before_first_q1")
            profiler = None
            if warmup_probe and os.environ.get("DUPLEX_WARMUP_CODE2WAV_TRACE") == "1":
                try:
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if inputs_embeds.device.type == "cuda":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    profiler = torch.profiler.profile(
                        activities=activities, record_shapes=False,
                        profile_memory=False, with_stack=False,
                    )
                except Exception as exc:
                    self._warmup_debug_first_q1["profiler_init_error"] = type(exc).__name__
            try:
                if profiler is None:
                    output = model(
                        inputs_embeds=inputs_embeds,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                else:
                    with profiler:
                        output = model(
                            inputs_embeds=inputs_embeds,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
            finally:
                if warmup_probe:
                    self._warmup_check_code2wav_weights("after_first_q1")
                    self._warmup_debug_first_q1_checked = True
            hidden = output.last_hidden_state[:, -1:, :]
            if warmup_probe:
                self._warmup_debug_first_q1.update({
                    "output": _warmup_diag_tensor(hidden),
                    "profiler_backends": _warmup_diag_profile(profiler),
                    "output_cache_length": _cache_sequence_length(output.past_key_values),
                })
                self._warmup_debug_first_q1["nonfinite"] = not self._warmup_debug_first_q1["output"]["finite"]
                if self._warmup_debug_first_q1["nonfinite"]:
                    logger.error("[warmup_q1_nonfinite] %s", json.dumps(
                        self._warmup_debug_first_q1, sort_keys=True, allow_nan=False,
                    ))
            return hidden, output.past_key_values

        try:
            cache_len_before = _cache_sequence_length(past_key_values)
            hidden, returned_cache = compiled(inputs_embeds, past_key_values)
            cache_len_after = _cache_sequence_length(past_key_values)
            if returned_cache is not past_key_values:
                raise RuntimeError("compiled Qwen Q1 replaced its DynamicCache object")
            if cache_len_after != cache_len_before + int(inputs_embeds.shape[1]):
                raise RuntimeError(
                    "compiled Qwen Q1 did not append exactly its input token count "
                    f"(before={cache_len_before}, after={cache_len_after}, "
                    f"input={int(inputs_embeds.shape[1])})"
                )
            if (
                not isinstance(hidden, torch.Tensor)
                or hidden.ndim != 3
                or hidden.shape[0] != inputs_embeds.shape[0]
                or hidden.shape[1] != 1
            ):
                raise RuntimeError("compiled Qwen Q1 returned an invalid last hidden state")
            if self.talker_q1_compile_status == "warming":
                self._talker_q1_compile_warmup_calls += 1
            if metrics is not None:
                metrics["talker_q1_backend"] = "torch_compile_dynamic"
                metrics["compiled_talker_q1_calls"] = int(
                    metrics.get("compiled_talker_q1_calls", 0)
                ) + 1
            return hidden, returned_cache
        except Exception as exc:
            raise _CompiledTalkerDecodeError(
                f"compiled Qwen Q1 failed its DynamicCache contract: {type(exc).__name__}: {exc}"
            ) from exc

    @torch.inference_mode()
    def _warmup_compiled_talker_q1(
        self,
        text: str,
        h_pfx: torch.Tensor,
        voice_conditioning: dict[str, Any],
    ) -> None:
        """Warm three one-token Q1 calls at changing DynamicCache positions."""
        if self._compiled_talker_q1 is None:
            return
        t = self.talker.talker
        voice = _resolve_talker_voice_conditioning(self.talker, voice_conditioning)
        static_part, text_part, bos_part, tts_pad_embed = _build_talker_prefill_parts(
            self.talker,
            text,
            voice,
        )
        # Keep prefix prefill eager and use a fresh cache isolated to warmup.
        out_static = t.model(inputs_embeds=static_part, use_cache=True)
        dynamic = torch.cat([text_part, h_pfx.to(self.dtype), bos_part], dim=1)
        out_prefix = t.model(
            inputs_embeds=dynamic,
            past_key_values=out_static.past_key_values,
            use_cache=True,
        )
        cache = out_prefix.past_key_values
        if _cache_sequence_length(cache) < 2:
            raise _CompiledTalkerDecodeError("compiled Qwen warmup produced an empty/one-token prefix cache")
        codes = torch.zeros((1, 1, 16), device=self.device, dtype=torch.long)
        for generation_step in range(3):
            query = build_qwen_audio_conditioning(
                t,
                codes,
                tts_pad_embed=tts_pad_embed,
                generation_step=generation_step,
            )
            _hidden, cache = self._qwen_q1_forward(t.model, query, cache)
        if self._talker_q1_compile_warmup_calls < 3:
            raise _CompiledTalkerDecodeError(
                "compiled Qwen Q1 warmup did not execute three decode positions"
            )
        self._verify_talker_q1_compile_capture()
        self.talker_q1_backend = "torch_compile_dynamic"
        self.talker_q1_compile_status = "ready"
        if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def _warmup_direct_v2_audio_path(self) -> None:
        seed_text = os.environ.get("DUPLEX_WARMUP_DIAGNOSTIC_SEED")
        if seed_text is None:
            self._warmup_direct_v2_audio_path_impl()
            return
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise RuntimeError("DUPLEX_WARMUP_DIAGNOSTIC_SEED must be an integer") from exc
        device = torch.device(self.device)
        devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        logger.info("[warmup_diagnostic] temporary seeded RNG seed=%d cuda_devices=%s", seed, devices)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self._warmup_diagnostic_active = True
            self._warmup_diagnostic_seed = seed
            self._warmup_debug_last_cb0_logits = None
            self._warmup_debug_last_cb0_id = None
            self._warmup_debug_last_residual_codes = []
            self._warmup_debug_last_decode = None
            self._warmup_debug_code2wav_weights = {}
            self._warmup_debug_first_q1 = None
            self._warmup_debug_first_q1_checked = False
            self._warmup_debug_first_nonfinite_module = None
            self._warmup_debug_decoder_hooks_done = False
            self._warmup_debug_decoder_profiled = False
            self._warmup_debug_decoder_profiler = None
            self._warmup_debug_rvq_codes = []
            try:
                self._warmup_direct_v2_audio_path_impl()
                trace = dict(self._warmup_debug_last_decode or {})
                trace.update({
                    "schema": "warmup_code2wav_seeded_trace_v1",
                    "status": "passed",
                    "seed": seed,
                    "last_cb0_id": self._warmup_debug_last_cb0_id,
                    "last_residual_codes": list(self._warmup_debug_last_residual_codes),
                    "first_q1": self._warmup_debug_first_q1,
                    "code2wav_weight_finiteness": self._warmup_debug_code2wav_weights,
                    "first_nonfinite_module": self._warmup_debug_first_nonfinite_module,
                })
                trace_path = Path(".tmp/recovery_probe/warmup_code2wav_seeded_trace.json")
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text(
                    json.dumps(trace, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                logger.info("[warmup_diagnostic] seeded warmup passed %s", json.dumps(
                    trace, ensure_ascii=False, sort_keys=True, allow_nan=False,
                ))
            finally:
                self._warmup_diagnostic_active = False

    def _warmup_direct_v2_audio_path_impl(self) -> None:
        """Warm the processor-native audio Q1 and Qwen audio path before readiness.

        Warmup output and private caches are discarded. The processor receives
        an existing short speech fixture using the same Gemma4 input dictionary
        and sample-rate ABI as a direct realtime turn. The Talker consumes only
        the validated SAY-token hiddens corresponding to the parsed SAY text.
        """
        from duplex_kernel.audio_response_conditioning import (
            build_audio_response_protocol_v2_messages,
        )

        processor = getattr(self._thinker_wrapper, "processor", None)
        if processor is None or not callable(processor):
            raise RuntimeError("direct v2 warmup requires the Gemma4 audio processor")
        audio_marker = (
            getattr(processor, "audio_token", None)
            or getattr(self.tokenizer, "audio_token", None)
            or "<|audio|>"
        )
        messages = build_audio_response_protocol_v2_messages(
            history_messages=[],
            audio_token=str(audio_marker),
        )
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        warmup_audio_path = (
            Path(SPEAKER_REFERENCE_WAV)
            if SPEAKER_REFERENCE_WAV
            else Path(__file__).resolve().parents[1]
            / "samples_v9"
            / "sample_1_native_qwen_reference.wav"
        )
        if not warmup_audio_path.is_file():
            raise RuntimeError(
                f"direct v2 warmup speech fixture missing: {warmup_audio_path}"
            )
        import soundfile as sf

        warmup_pcm, warmup_sample_rate = sf.read(
            str(warmup_audio_path),
            dtype="float32",
            always_2d=False,
        )
        warmup_pcm = np.asarray(warmup_pcm, dtype=np.float32)
        if warmup_pcm.ndim == 2:
            warmup_pcm = warmup_pcm.mean(axis=1, dtype=np.float32)
        warmup_pcm = warmup_pcm.reshape(-1)
        if int(warmup_sample_rate) != USER_SAMPLE_RATE:
            from scipy.signal import resample_poly

            divisor = math.gcd(int(warmup_sample_rate), USER_SAMPLE_RATE)
            warmup_pcm = resample_poly(
                warmup_pcm,
                USER_SAMPLE_RATE // divisor,
                int(warmup_sample_rate) // divisor,
            ).astype(np.float32, copy=False)
        if not bool(np.max(np.abs(warmup_pcm)) > 1e-7):
            raise RuntimeError("direct v2 warmup speech fixture has no audible signal")
        if warmup_pcm.size == 0 or not np.isfinite(warmup_pcm).all():
            raise RuntimeError("direct v2 warmup speech fixture is empty or non-finite")
        processor_inputs = processor(
            text=[rendered],
            audio=[warmup_pcm],
            sampling_rate=USER_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        if not isinstance(processor_inputs, dict):
            processor_inputs = dict(processor_inputs)
        input_ids = processor_inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError("direct v2 warmup processor returned invalid input_ids")
        prompt_ids = [int(token) for token in input_ids[0].tolist()]
        warmup_collector = AudioResponseV2Collector(expected_action="SPEAK")
        warmup_constraint = AudioResponseV2ActionConstraint(
            expected_action="SPEAK",
            tokenizer=self.tokenizer,
            enforce_json_structure=True,
        )
        warmup_budget = warmup_constraint.preflight_generation_budget(
            warmup_collector,
            max_new_tokens=AUDIO_RESPONSE_V2_MAX_NEW_TOKENS,
            max_heard_tokens=AUDIO_RESPONSE_V2_MAX_HEARD_TOKENS,
            max_action_tokens=AUDIO_RESPONSE_V2_MAX_ACTION_TOKENS,
            max_say_tokens=AUDIO_RESPONSE_V2_MAX_SAY_TOKENS,
        )
        warmup_collector.set_token_caps(
            heard=warmup_budget["heard_token_cap"],
            action=warmup_budget["action_token_cap"],
            say=warmup_budget["say_token_cap"],
        )

        def on_warmup_generation_start() -> int | None:
            return warmup_constraint.initial_token_id()

        def on_warmup_generated_token(
            token_index: int,
            token_id: int,
            token_text: str,
            token_hidden: torch.Tensor,
        ) -> int | tuple[int, str] | None:
            warmup_collector.feed_token(
                token_index=token_index,
                token_id=token_id,
                text=token_text,
                hidden=token_hidden,
            )
            forced_id = warmup_constraint.next_token_id(warmup_collector)
            if forced_id is None:
                return None
            return forced_id, str(warmup_constraint.last_forced_kind or "structure")

        def on_warmup_candidate(
            _token_id: int,
            token_text: str,
            terminal: bool,
        ) -> bool:
            return warmup_constraint.candidate_within_generation_budget(
                warmup_collector,
                token_text,
                terminal=terminal,
                generated_tokens=warmup_collector.generated_token_count,
                budget=warmup_budget,
            )

        def on_warmup_candidate_fallback(rejected: Any) -> Any:
            action_ids = warmup_constraint.action_value_candidate_token_ids(warmup_collector)
            if action_ids:
                return action_ids
            closure_id = warmup_constraint.bounded_field_closure_token_id(
                warmup_collector,
                rejected,
                budget=warmup_budget,
            )
            if closure_id is not None:
                return closure_id
            return warmup_constraint.bounded_say_closure_token_id(
                warmup_collector,
                rejected,
                max_say_words=4,
            )

        response = self.think(
            prompt_ids,
            max_new_tokens=AUDIO_RESPONSE_V2_MAX_NEW_TOKENS,
            min_new_tokens=1,
            temperature=0.1,
            top_p=1.0,
            input_audio=processor_inputs,
            on_generation_start=on_warmup_generation_start,
            on_generated_token=on_warmup_generated_token,
            on_candidate_token=on_warmup_candidate,
            on_candidate_fallback=on_warmup_candidate_fallback,
            disable_thinker_cadence_pacing=True,
        )
        protocol_text = response.get("protocol_text")
        if not isinstance(protocol_text, str):
            raise RuntimeError("direct v2 warmup did not return exact protocol text")
        try:
            parsed_response, say_hidden = finish_audio_response_v2_with_say_hidden(
                warmup_collector,
                expected_text=protocol_text,
            )
        except AudioResponseProtocolError as exc:
            if exc.reason == "token_decode_mismatch":
                logger.error(
                    "[warmup] v2 exact decode mismatch expected=%s collector=%s gen_ids=%s snapshot=%s",
                    getattr(exc, "expected_text_repr", ""),
                    getattr(exc, "collector_text_repr", ""),
                    getattr(exc, "generated_token_ids", []),
                    warmup_collector.private_failure_snapshot(reason=exc.reason),
                )
            raise
        if parsed_response["action"] != "SPEAK":
            raise RuntimeError("direct v2 warmup did not produce its internal SPEAK smoke action")
        expected_hidden_dim = int(getattr(self.v9_stream, "raw_dim", -1))
        if (
            not isinstance(say_hidden, torch.Tensor)
            or say_hidden.ndim != 3
            or say_hidden.shape[0] != 1
            or say_hidden.shape[-1] != expected_hidden_dim
            or say_hidden.shape[1] <= 0
            or not bool(torch.isfinite(say_hidden).all())
        ):
            raise RuntimeError("direct v2 warmup returned invalid SAY-only hidden states")
        warmup_say = parsed_response["say"]
        from duplex_kernel.voices import resolve_voice_conditioning

        warmup_voice_conditioning = resolve_voice_conditioning("gemma")
        if (
            not isinstance(warmup_voice_conditioning, dict)
            or warmup_voice_conditioning.get("mode") != "x_vector_only"
            or warmup_voice_conditioning.get("ref_code") is not None
        ):
            raise RuntimeError(
                "direct v2 warmup requires the canonical gemma x_vector_only profile"
            )
        speaker_embedding = warmup_voice_conditioning.get("speaker_embedding")
        if not isinstance(speaker_embedding, torch.Tensor):
            raise RuntimeError("direct v2 warmup speaker profile has no tensor embedding")
        expected_speaker_dim = int(
            getattr(
                getattr(getattr(self, "talker", None), "tcfg", None),
                "hidden_size",
                2048,
            )
        )
        if (
            speaker_embedding.numel() != expected_speaker_dim
            or not bool(torch.isfinite(speaker_embedding).all())
        ):
            raise RuntimeError(
                "direct v2 warmup speaker embedding is invalid "
                f"(numel={speaker_embedding.numel()}, expected={expected_speaker_dim})"
            )
        frame_residual_warmup = getattr(self, "frame_residual_adapter", None) is not None
        if frame_residual_warmup:
            warmup_prefix = torch.empty(
                (1, 0, 2048), device=self.device, dtype=self.dtype
            )
        else:
            warmup_prefix = self.v9_prefix(say_hidden, 0)
        native_q1_requested = bool(
            getattr(self, "_talker_native_audio_q1_requested", False)
        )
        if (
            not frame_residual_warmup
            and (getattr(self, "_compiled_talker_q1", None) is not None or native_q1_requested)
            and not bool(torch.any(warmup_prefix != 0))
        ):
            if native_q1_requested:
                self._set_native_talker_q1_fallback(
                    RuntimeError("direct v2 warmup produced an all-zero V9 prefix")
                )
                if self._talker_compiled_decode_requested:
                    self.talker_q1_compile_status = "requested"
                    self._prepare_compiled_talker_q1()
            elif getattr(self, "_compiled_talker_q1", None) is not None:
                self._set_talker_q1_eager_fallback(
                    RuntimeError("direct v2 warmup produced an all-zero V9 prefix")
                )
        if native_q1_requested and self.talker_q1_native_status != "eager_fallback_before_ready":
            self.talker_q1_backend = "native_audio_q1"
            self.talker_q1_native_status = "warming"
            try:
                self._warmup_native_talker_q1(
                    warmup_say,
                    warmup_prefix,
                    warmup_voice_conditioning,
                )
            except Exception as exc:
                if self._talker_native_audio_q1_graph_requested:
                    raise
                self._set_native_talker_q1_fallback(exc)
                if self._talker_compiled_decode_requested:
                    self.talker_q1_compile_status = "requested"
                    self._prepare_compiled_talker_q1()
        if getattr(self, "_compiled_talker_q1", None) is not None:
            # Snapshot after the audio-native Thinker warmup so its compiler
            # graphs cannot be counted as Talker Q1 recompilations.
            self._talker_q1_dynamo_before = self._talker_q1_dynamo_counts()
        self._native_q1_startup_warmup_active = bool(
            self._talker_native_audio_q1_graph_requested
        )
        if getattr(self, "_warmup_diagnostic_active", False):
            # Observe the first Q1 query that actually feeds the generated RVQ stream.
            self._warmup_debug_first_q1 = None
            self._warmup_debug_first_q1_checked = False
        warmup_audio = np.zeros(0, dtype=np.float32)
        try:
            warmup_audio = self.talk(
                warmup_say,
                warmup_prefix,
                n_words=max(1, len(warmup_say.split())),
                voice_conditioning=warmup_voice_conditioning,
                min_frames=4,
                frame_residual_response_hidden=(
                    say_hidden if frame_residual_warmup else None
                ),
            )
            if self.talker_q1_compile_status == "warming":
                if self._talker_q1_compile_warmup_calls < 3:
                    self._warmup_compiled_talker_q1(
                        warmup_say,
                        warmup_prefix,
                        warmup_voice_conditioning,
                    )
                else:
                    self._verify_talker_q1_compile_capture()
                    self.talker_q1_backend = "torch_compile_dynamic"
                    self.talker_q1_compile_status = "ready"
        except _CompiledTalkerDecodeError as exc:
            if self.talker_q1_compile_status != "warming":
                raise
            self._set_talker_q1_eager_fallback(exc)
            warmup_audio = self.talk(
                warmup_say,
                warmup_prefix,
                n_words=max(1, len(warmup_say.split())),
                voice_conditioning=warmup_voice_conditioning,
                min_frames=4,
                frame_residual_response_hidden=(
                    say_hidden if frame_residual_warmup else None
                ),
            )
        self._native_q1_startup_warmup_active = False
        if warmup_audio.size and not np.isfinite(warmup_audio).all():
            raise RuntimeError("direct v2 warmup Talker produced non-finite PCM")
        warmup_waveform_samples = int(warmup_pcm.size)
        del response, say_hidden, warmup_audio, warmup_prefix, processor_inputs, warmup_pcm
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.current_stream(self.device).synchronize()
        self.direct_v2_audio_warmup_status = "complete"
        logger.info(
            "[engine] direct v2 audio warmup complete protocol=heard_action_say_v2 action=SPEAK say_hidden_tokens=%d processor_sample_rate=%d waveform_samples=%d",
            len(warmup_collector.say_tokens),
            USER_SAMPLE_RATE,
            warmup_waveform_samples,
        )

    def _new_thinker_static_cache(self, max_cache_len: int | None = None) -> Any:
        from transformers import StaticCache

        # SlidingWindowCache.update keeps a Python ``cumulative_length_int``
        # which Dynamo guards on for every prefill position. Store every KV
        # layer at the full static capacity instead; the original Thinker
        # config remains untouched and continues to define the official
        # sliding-window attention masks.
        cache_config = deepcopy(self.thinker.config)
        text_config = cache_config.get_text_config(decoder=True)
        layer_types = getattr(text_config, "layer_types", None)
        if not layer_types:
            raise RuntimeError(
                "Thinker StaticCache compilata richiede layer_types espliciti per costruire storage full"
            )
        if any(kind not in {"full_attention", "sliding_attention"} for kind in layer_types):
            raise RuntimeError(
                "Thinker StaticCache compilata supporta solo layer Gemma4 full/sliding attention"
            )
        text_config.layer_types = ["full_attention"] * len(layer_types)
        return StaticCache(
            config=cache_config,
            max_batch_size=1,
            max_cache_len=int(
                self.thinker_static_cache_max_len if max_cache_len is None else max_cache_len
            ),
            device=self.device,
            dtype=self.dtype,
        )

    def _compiled_thinker_cache_fingerprint(self) -> tuple[Any, ...]:
        """Identify the in-process model/cache layout a prompt snapshot belongs to."""
        config = getattr(getattr(self, "thinker", None), "config", None)
        get_text_config = getattr(config, "get_text_config", None)
        text_config = get_text_config(decoder=True) if callable(get_text_config) else config
        layer_types = tuple(getattr(text_config, "layer_types", ()) or ())
        return (
            id(getattr(self, "thinker", None)),
            id(getattr(self, "_compiled_thinker", None)),
            id(getattr(self, "tokenizer", None)),
            str(getattr(self, "device", None)),
            str(getattr(self, "dtype", None)),
            int(getattr(self, "thinker_static_cache_max_len", 0)),
            layer_types,
        )

    def _copy_compiled_thinker_prefix(
        self,
        source: Any,
        target: Any,
        token_count: int,
        runtime_metrics: dict[str, Any],
    ) -> Any:
        """Copy an immutable cache prefix while yielding priority to listener frames."""
        started = time.perf_counter()
        self._listener_priority_lock()
        acquired = time.perf_counter()
        try:
            result = copy_static_cache_prefix(source, target, token_count)
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.current_stream().synchronize()
            runtime_metrics["thinker_cache_copy_bytes"] += result.copied_bytes
            runtime_metrics["thinker_cache_copy_tokens"] += result.copied_tokens
            return result
        finally:
            try:
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    torch.cuda.current_stream().synchronize()
            finally:
                runtime_metrics["thinker_cache_copy_ms"] += (time.perf_counter() - started) * 1000.0
                hold_ms = (time.perf_counter() - acquired) * 1000.0
                runtime_metrics["thinker_cache_copy_lock_hold_ms"] += hold_ms
                self._thinker_lock.release()

    @staticmethod
    def _compiled_generation_is_current(state: Any, generation_id: int | None) -> bool:
        if state is None:
            return True
        is_current = getattr(state, "is_generation_current", None)
        if not callable(is_current):
            return True
        return generation_id is not None and bool(is_current(generation_id=generation_id))

    def _publish_compiled_thinker_prompt_cache(
        self,
        state: Any,
        generation_id: int | None,
        prompt_token_ids: list[int],
        source_cache: Any,
        fingerprint: tuple[Any, ...],
        runtime_metrics: dict[str, Any],
    ) -> None:
        """Publish a separate prompt-only snapshot iff its generation is still current."""
        snapshot = self._new_thinker_static_cache(self.thinker_static_cache_max_len)
        try:
            self._copy_compiled_thinker_prefix(
                source_cache, snapshot, len(prompt_token_ids), runtime_metrics
            )
            runtime_metrics["thinker_cache_snapshot_storage_bytes"] = static_cache_storage_bytes(snapshot)
        except CachePrefixCopyError as exc:
            runtime_metrics["thinker_cache_snapshot_reason"] = f"copy_unsupported:{exc}"
            return

        lock = getattr(state, "_generation_lock", None)
        if lock is None:
            if not self._compiled_generation_is_current(state, generation_id):
                runtime_metrics["thinker_cache_publish"] = "stale_generation"
                return
            state.thinker_static_cache = snapshot
            state.thinker_static_cache_ids = list(prompt_token_ids)
            state.thinker_static_cache_fingerprint = fingerprint
            runtime_metrics["thinker_cache_publish"] = "committed"
            return
        with lock:
            if not self._compiled_generation_is_current(state, generation_id):
                state.stale_kv_writes = int(getattr(state, "stale_kv_writes", 0)) + 1
                runtime_metrics["thinker_cache_publish"] = "stale_generation"
                return
            state.thinker_static_cache = snapshot
            state.thinker_static_cache_ids = list(prompt_token_ids)
            state.thinker_static_cache_fingerprint = fingerprint
            runtime_metrics["thinker_cache_publish"] = "committed"

    def _static_thinker_kwargs(
        self,
        input_ids: torch.Tensor,
        cache: Any,
        *,
        start_position: int,
        visible_tokens: int,
        cache_capacity: int | None = None,
        precompute_attention_masks: bool = False,
        attention_mask_prefix: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        capacity = int(
            self.thinker_static_cache_max_len if cache_capacity is None else cache_capacity
        )
        if capacity < 1:
            raise ValueError("capacità StaticCache Thinker deve essere almeno 1")
        if start_position < 0 or visible_tokens < 0:
            raise ValueError("posizione e token visibili StaticCache non possono essere negativi")
        if visible_tokens > capacity or start_position + input_ids.shape[1] > capacity:
            raise ValueError(
                "prefisso Thinker eccede la capacità StaticCache della richiesta "
                f"(visible={visible_tokens}, end={start_position + input_ids.shape[1]}, "
                f"capacity={capacity})"
            )
        cache_position = torch.arange(
            start_position,
            start_position + input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (1, capacity),
            device=input_ids.device,
            dtype=torch.long,
        )
        if attention_mask_prefix is None:
            attention_mask[:, :visible_tokens] = 1
        else:
            if tuple(attention_mask_prefix.shape) != (1, visible_tokens):
                raise ValueError(
                    "prefisso attention_mask statico deve coprire esattamente i token visibili "
                    f"(shape={tuple(attention_mask_prefix.shape)}, visible={visible_tokens})"
                )
            attention_mask[:, :visible_tokens] = attention_mask_prefix.to(
                device=input_ids.device, dtype=torch.long
            )
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": cache_position.unsqueeze(0),
            "cache_position": cache_position,
            "past_key_values": cache,
            "use_cache": True,
            "logits_to_keep": 1,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if precompute_attention_masks:
            from transformers.masking_utils import create_masks_for_generate

            get_input_embeddings = getattr(self.thinker, "get_input_embeddings", None)
            if not callable(get_input_embeddings):
                raise RuntimeError(
                    "Thinker compilato non espone get_input_embeddings per precomputare le maschere Gemma4"
                )
            inputs_embeds = get_input_embeddings()(input_ids)
            kwargs["attention_mask"] = create_masks_for_generate(
                self.thinker.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=cache,
                position_ids=cache_position.unsqueeze(0),
            )
        return kwargs

    @torch.inference_mode()
    def _prepare_compiled_thinker(self) -> None:
        """Compile fixed Q4/Q1 Thinker forwards before the runtime is ready."""
        try:
            from transformers import StaticCache
            from transformers.generation.configuration_utils import CompileConfig
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "thinker compile mode richiede Transformers con CompileConfig e StaticCache"
            ) from exc
        get_compiled_call = getattr(self.thinker, "get_compiled_call", None)
        if not callable(get_compiled_call):
            raise RuntimeError(
                "thinker compile mode richiede PreTrainedModel.get_compiled_call; avvio annullato"
            )

        # Bound Inductor's parallel compiler workers to fit the 14 GB host.
        compile_cache_root = Path(getattr(self, "workspace", Path.cwd())) / ".tmp"
        torchinductor_cache_dir = Path(os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR", str(compile_cache_root / "torchinductor_cache")
        ))
        triton_cache_dir = Path(os.environ.setdefault(
            "TRITON_CACHE_DIR", str(compile_cache_root / "triton_cache")
        ))
        torchinductor_cache_dir.mkdir(parents=True, exist_ok=True)
        triton_cache_dir.mkdir(parents=True, exist_ok=True)
        # The runner points these to /workspace/.tmp, which is host-mounted so
        # compiler artifacts survive a serving-container restart.
        compile_threads = int(os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1"))
        if compile_threads not in {1, 2}:
            raise RuntimeError(
                "thinker compile mode consente 1 o 2 thread Inductor per limitare la RAM host"
            )
        try:
            import torch._inductor.config as inductor_config

            inductor_config.compile_threads = compile_threads
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("impossibile limitare i thread Inductor a 1") from exc

        compile_config = CompileConfig(fullgraph=False, dynamic=False, mode=self.thinker_compile_mode)
        compiled = get_compiled_call(compile_config)
        if not callable(compiled):
            raise RuntimeError("get_compiled_call non ha restituito un forward compilato")
        self._compiled_thinker = compiled

        def _rss_mb() -> float | None:
            try:
                import resource

                # The serving host is Linux; ru_maxrss is KiB there.
                return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
            except (ImportError, AttributeError, OSError):
                return None

        rss_before = _rss_mb()
        cuda_baseline = 0
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
            cuda_baseline = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        scratch = self._new_thinker_static_cache(self.thinker_static_cache_max_len)
        reset = getattr(scratch, "reset", None)
        if not callable(reset):
            raise RuntimeError("StaticCache del runtime non espone reset(); warmup isolato impossibile")

        # StaticCache layers are lazy. Mirror the runtime sequence: the first
        # real-shaped Q4 forward initializes the layers and consumes positions
        # 0..3; compiled Q4/Q1 then see the same cache lengths as serving.
        warm_q4 = torch.zeros((1, 4), device=self.device, dtype=torch.long)
        self.thinker(**self._static_thinker_kwargs(
            warm_q4, scratch, start_position=0, visible_tokens=4,
        ))
        compiled(**self._static_thinker_kwargs(
            warm_q4, scratch, start_position=4, visible_tokens=8,
            precompute_attention_masks=True,
        ))
        warm_q1 = torch.zeros((1, 1), device=self.device, dtype=torch.long)
        compiled(**self._static_thinker_kwargs(
            warm_q1, scratch, start_position=8, visible_tokens=9,
            precompute_attention_masks=True,
        ))
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        reset()
        del scratch, warm_q4, warm_q1
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

        # Direct-audio v2 turns may use a smaller StaticCache. Each capacity
        # changes the static K/V and attention-mask shapes seen by torch.compile,
        # so warm a Q1 graph per supported bucket before readiness. Keep the
        # configured default's Q4/Q1 warmup above for the legacy text path.
        bucket_warmup: list[tuple[int, float, float | None]] = []
        if self.thinker_static_cache_max_len >= AUDIO_THINKER_CACHE_BUCKETS[0]:
            for capacity in AUDIO_THINKER_CACHE_BUCKETS:
                if capacity > self.thinker_static_cache_max_len or capacity == self.thinker_static_cache_max_len:
                    continue
                bucket_started = time.perf_counter()
                bucket_cache = self._new_thinker_static_cache(capacity)
                bucket_reset = getattr(bucket_cache, "reset", None)
                if not callable(bucket_reset):
                    raise RuntimeError("StaticCache Thinker bucket non espone reset()")
                bucket_warm_q4 = torch.zeros((1, 4), device=self.device, dtype=torch.long)
                self.thinker(**self._static_thinker_kwargs(
                    bucket_warm_q4,
                    bucket_cache,
                    start_position=0,
                    visible_tokens=4,
                    cache_capacity=capacity,
                ))
                bucket_warm_q1 = torch.zeros((1, 1), device=self.device, dtype=torch.long)
                compiled(**self._static_thinker_kwargs(
                    bucket_warm_q1,
                    bucket_cache,
                    start_position=4,
                    visible_tokens=5,
                    cache_capacity=capacity,
                    precompute_attention_masks=True,
                ))
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    torch.cuda.synchronize(self.device)
                bucket_reset()
                del bucket_cache, bucket_warm_q4, bucket_warm_q1
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    torch.cuda.empty_cache()
                    resident_mb = torch.cuda.memory_allocated(self.device) / (1024.0 * 1024.0)
                else:
                    resident_mb = None
                bucket_warmup.append((capacity, time.perf_counter() - bucket_started, resident_mb))
        elapsed = time.perf_counter() - started
        cuda_peak = None
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            cuda_peak = max(0, torch.cuda.max_memory_allocated(self.device) - cuda_baseline)
        rss_after = _rss_mb()
        logger.info(
            "[engine] thinker compiled warmup complete mode=%s q4_q1=true compile_threads=%d cache_tokens=%d bucket_q1_warmup=%s warmup_s=%.3f rss_peak_before_mb=%s rss_peak_after_mb=%s cuda_peak_delta_mb=%s torchinductor_cache=%s triton_cache=%s",
            self.thinker_compile_mode,
            compile_threads,
            self.thinker_static_cache_max_len,
            ",".join(
                f"{capacity}:{seconds:.2f}s:{'n/a' if resident_mb is None else f'{resident_mb:.1f}MB'}"
                for capacity, seconds, resident_mb in bucket_warmup
            ) or "none",
            elapsed,
            "n/a" if rss_before is None else f"{rss_before:.1f}",
            "n/a" if rss_after is None else f"{rss_after:.1f}",
            "n/a" if cuda_peak is None else f"{cuda_peak / (1024.0 * 1024.0):.1f}",
            str(torchinductor_cache_dir),
            str(triton_cache_dir),
        )

    def _listener_priority_lock(
        self,
        *,
        stop_check: Callable[[], bool] | None = None,
    ) -> bool:
        """Acquire one fair shared-model quantum, with cancellation cleanup."""
        acquire_fair = getattr(self._thinker_lock, "acquire_fair", None)
        if callable(acquire_fair):
            return bool(acquire_fair("compute", stop_check=stop_check))
        while True:
            if stop_check is not None and stop_check():
                return False
            try:
                if self._thinker_lock.acquire(timeout=0.005):
                    return True
            except TypeError:
                self._thinker_lock.acquire()
                return True

    @staticmethod
    def _wait_for_listener_admission(
        callback: Callable[[], Any] | None,
        runtime_metrics: dict[str, Any] | None,
        phase: str,
    ) -> ListenerBacklogAdmission | None:
        """Yield model work to queued/current listener callbacks before a forward."""
        if not callable(callback):
            return None
        started = time.perf_counter()
        admission = callback()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if runtime_metrics is None:
            if admission is ListenerBacklogAdmission.ABORTED:
                raise GenerationAdmissionAborted("generation became stale before model forward")
            return admission if isinstance(admission, ListenerBacklogAdmission) else None
        total_key = f"listener_admission_{phase}_wait_ms"
        max_key = f"listener_admission_{phase}_wait_max_ms"
        runtime_metrics[total_key] = runtime_metrics.get(total_key, 0.0) + elapsed_ms
        runtime_metrics[max_key] = max(runtime_metrics.get(max_key, 0.0), elapsed_ms)
        if admission is ListenerBacklogAdmission.ABORTED:
            raise GenerationAdmissionAborted("generation became stale before model forward")
        return admission if isinstance(admission, ListenerBacklogAdmission) else None

    @staticmethod
    def _best_native_logit_candidate_id(
        candidate_ids: list[int] | tuple[int, ...],
        source_logits: torch.Tensor,
    ) -> int | None:
        """Choose the highest native-logit ID from a small prevalidated pool."""
        vocab_size = int(source_logits.shape[-1])
        pool = sorted({
            int(token_id)
            for token_id in candidate_ids
            if not isinstance(token_id, bool) and isinstance(token_id, Integral)
            and 0 <= int(token_id) < vocab_size
        })
        if not pool:
            return None
        pool_tensor = torch.tensor(pool, device=source_logits.device, dtype=torch.long)
        pool_logits = source_logits.reshape(-1).index_select(0, pool_tensor)
        finite = torch.isfinite(pool_logits)
        if not bool(finite.any()):
            return None
        finite_indices = torch.nonzero(finite, as_tuple=False).reshape(-1)
        finite_logits = pool_logits.index_select(0, finite_indices)
        best_offset = int(torch.argmax(finite_logits).item())
        pool_offset = int(finite_indices[best_offset].item())
        return int(pool[pool_offset])

    def build_thinker_sampler(
        self,
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        min_new_tokens: int = 0,
        action: str | None = None,
        backchannel_trie: dict[tuple[int, ...], Any] | None = None,
        runtime_metrics: dict[str, Any] | None = None,
        on_generation_start: Callable[[], Any] | None = None,
        on_generated_token: Callable[[int, int, str, torch.Tensor], Any] | None = None,
        on_candidate_token: Callable[[int, str, bool], bool] | None = None,
        on_candidate_fallback: Callable[[list[tuple[int, str, bool]]], Any] | None = None,
        on_action_logits_probe: Callable[[dict[str, Any]], Any] | None = None,
        state: Any | None = None,
        generation_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ThinkerSampler:
        """Build the exact Thinker sampling pipeline for HF or native decode."""
        thinker_model = getattr(self, "thinker", None)
        generation_eos = getattr(getattr(thinker_model, "generation_config", None), "eos_token_id", None)
        model_eos = getattr(getattr(thinker_model, "config", None), "eos_token_id", None)
        if generation_eos is not None:
            eos_ids = normalize_eos_token_ids(generation_eos)
        elif model_eos is not None:
            eos_ids = normalize_eos_token_ids(model_eos)
        else:
            eos_ids = normalize_eos_token_ids(getattr(self.tokenizer, "eos_token_id", None))
            eos_ids.update(normalize_eos_token_ids(getattr(self.tokenizer, "eot_id", None)))

        normalized_action = str(action or "").strip().upper()
        if backchannel_trie is None and normalized_action == "BACKCHANNEL":
            backchannel_trie = getattr(self, "_backchannel_token_trie", None)
            if not backchannel_trie:
                raise RuntimeError("BACKCHANNEL richiede il trie di frasi gold del bundle attivo")
        return ThinkerSampler(
            tokenizer=self.tokenizer,
            device=self.device,
            eos_token_ids=eos_ids,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            backchannel_trie=backchannel_trie,
            runtime_metrics=runtime_metrics,
            on_generation_start=on_generation_start,
            on_generated_token=on_generated_token,
            on_candidate_token=on_candidate_token,
            on_candidate_fallback=on_candidate_fallback,
            on_action_logits_probe=on_action_logits_probe,
            generation_id=generation_id,
            state=state,
            best_native_logit_candidate_id=self._best_native_logit_candidate_id,
            generator=generator,
        )

    def _run_thinker_prefill(
        self,
        input_ids: list[int],
        *,
        past_key_values: Any | None = None,
        past_length: int = 0,
        is_compiled: bool = False,
        static_cache: Any | None = None,
        cache_init_started: float | None = None,
        runtime_metrics: dict[str, Any] | None = None,
        listener_admission_wait: Callable[[], Any] | None = None,
    ) -> tuple[Any, list[torch.Tensor], Any, torch.Tensor | None]:
        """Prefill bounded causal chunks, releasing the listener-shared lock between calls."""
        if not input_ids:
            raise ValueError("prefill thinker richiede almeno un token")
        chunk_tokens = 4 if is_compiled else self.thinker_prefill_chunk_tokens
        chunk_tokens = len(input_ids) if chunk_tokens == 0 else chunk_tokens
        cache = static_cache if is_compiled else past_key_values
        # A prefix branch is copied into already allocated StaticLayer storage,
        # so a Q4 suffix can use the compiled entrypoint from its first chunk.
        cache_initialized = bool(is_compiled and past_length > 0)
        hidden_parts: list[torch.Tensor] = []
        output = None
        final_mask: torch.Tensor | None = None
        processed = 0
        for start in range(0, len(input_ids), chunk_tokens):
            end = min(len(input_ids), start + chunk_tokens)
            chunk = torch.tensor([input_ids[start:end]], device=self.device, dtype=torch.long)
            kwargs: dict[str, Any] = {
                "input_ids": chunk,
                "use_cache": True,
                "logits_to_keep": 1,
                "output_hidden_states": True,
                "return_dict": True,
            }
            if is_compiled:
                # Transformers 5.5 exposes StaticCache.early_initialization,
                # but Gemma4 lacks _get_static_cache_init_shape(). Consume the
                # actual first chunk eagerly to initialize the fresh cache;
                # later fixed Q4 chunks compile, while a shorter tail stays
                # eager. No synthetic tokens are added to a runtime prompt.
                call_model = (
                    self._compiled_thinker
                    if cache_initialized and chunk.shape[1] == 4
                    else self.thinker
                )
            else:
                call_model = self.thinker
                mask_length = past_length + end
                final_mask = torch.ones((1, mask_length), device=self.device, dtype=torch.long)
                kwargs["attention_mask"] = final_mask
                if cache is not None:
                    kwargs["past_key_values"] = cache

            self._wait_for_listener_admission(
                listener_admission_wait, runtime_metrics, "thinker_prefill"
            )
            lock_wait_started = time.perf_counter()
            self._listener_priority_lock()
            lock_acquired = time.perf_counter()
            if runtime_metrics is not None:
                runtime_metrics["thinker_prefill_lock_wait_ms"] = runtime_metrics.get(
                    "thinker_prefill_lock_wait_ms", 0.0
                ) + (lock_acquired - lock_wait_started) * 1000.0
            try:
                if is_compiled:
                    precompute_masks = (
                        cache_initialized
                        and chunk.shape[1] == 4
                        and callable(getattr(self.thinker, "get_input_embeddings", None))
                        and callable(getattr(cache, "get_mask_sizes", None))
                    )
                    kwargs = self._static_thinker_kwargs(
                        chunk,
                        cache,
                        start_position=past_length + start,
                        visible_tokens=past_length + end,
                        precompute_attention_masks=precompute_masks,
                    )
                output = call_model(**kwargs)
                if is_compiled:
                    # StaticCache is updated in place by Transformers.
                    cache = static_cache
                    cache_initialized = True
                    if (cache_init_started is not None and runtime_metrics is not None
                            and "thinker_cache_init_ms" not in runtime_metrics):
                        runtime_metrics["thinker_cache_init_ms"] = (
                            time.perf_counter() - cache_init_started
                        ) * 1000.0
                else:
                    cache = output.past_key_values
                # Complete this bounded GPU section before giving the shared
                # lock to the listener. This makes the measured hold include
                # actual device work instead of leaving queued work behind.
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    torch.cuda.current_stream().synchronize()
            finally:
                if runtime_metrics is not None:
                    hold_ms = (time.perf_counter() - lock_acquired) * 1000.0
                    runtime_metrics["thinker_prefill_lock_hold_ms"] = runtime_metrics.get(
                        "thinker_prefill_lock_hold_ms", 0.0
                    ) + hold_ms
                    runtime_metrics["thinker_prefill_lock_hold_max_ms"] = max(
                        runtime_metrics.get("thinker_prefill_lock_hold_max_ms", 0.0), hold_ms
                    )
                self._thinker_lock.release()
            hidden_parts.append(output.hidden_states[-1])
            processed = end

        if output is None or processed != len(input_ids):
            raise RuntimeError("prefill thinker incompleto")
        return output, hidden_parts, cache, final_mask

    def _prepare_thinker_audio_inputs(
        self,
        input_audio: dict[str, Any],
        prompt_token_ids: list[int],
    ) -> dict[str, torch.Tensor]:
        """Validate and move processor-native Gemma4 audio tensors to the Thinker device."""
        if not isinstance(input_audio, dict):
            raise TypeError("input_audio deve essere il dizionario processor Gemma4")
        input_ids = input_audio.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_audio richiede input_ids Tensor")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError(f"input_audio input_ids deve avere shape (1,L), got {tuple(input_ids.shape)}")
        expected_ids = [int(value) for value in input_ids[0].tolist()]
        if list(prompt_token_ids) != expected_ids:
            raise ValueError("prompt_token_ids non coincide esattamente con input_audio.input_ids")
        attention_mask = input_audio.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if not isinstance(attention_mask, torch.Tensor) or tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError("input_audio attention_mask deve avere la stessa shape di input_ids")
        features = input_audio.get("input_features")
        feature_mask = input_audio.get("input_features_mask")
        if not isinstance(features, torch.Tensor) or features.ndim != 3 or features.shape[0] != 1:
            raise ValueError("input_audio.input_features deve avere shape processor (1,F,M)")
        if not isinstance(feature_mask, torch.Tensor) or feature_mask.ndim != 2 or feature_mask.shape[0] != 1:
            raise ValueError("input_audio.input_features_mask deve avere shape processor (1,F)")
        if features.shape[1] != feature_mask.shape[1]:
            raise ValueError("input_features e input_features_mask non sono allineati")
        return {
            "input_ids": input_ids.to(device=self.device, dtype=torch.long),
            "attention_mask": attention_mask.to(device=self.device, dtype=torch.long),
            "input_features": features.to(device=self.device),
            "input_features_mask": feature_mask.to(device=self.device),
        }

    def _audio_text_cache_prefix(
        self,
        audio_inputs: dict[str, torch.Tensor],
    ) -> tuple[list[int] | None, str | None]:
        """Return only valid hard-text IDs before Gemma4's first audio placeholder.

        Reusing a prefix that includes an audio placeholder would reuse the
        previous utterance's audio-conditioned K/V. Derive the cut from
        Gemma4's public placeholder mask and fail closed when that contract is
        unavailable or the preceding text positions are padded.
        """
        input_ids = audio_inputs["input_ids"]
        attention_mask = audio_inputs["attention_mask"]
        model = getattr(self.thinker, "model", None)
        get_placeholder_mask = getattr(model, "get_placeholder_mask", None)
        if not callable(get_placeholder_mask):
            return None, "audio_placeholder_mask_unavailable"
        try:
            masks = get_placeholder_mask(input_ids=input_ids)
            if not isinstance(masks, (tuple, list)) or len(masks) != 3:
                return None, "audio_placeholder_mask_shape_unsupported"
            audio_mask = masks[2]
            if not isinstance(audio_mask, torch.Tensor) or tuple(audio_mask.shape) != tuple(input_ids.shape):
                return None, "audio_placeholder_mask_shape_unsupported"
            positions = torch.nonzero(audio_mask[0], as_tuple=False).reshape(-1)
            if positions.numel() == 0:
                return None, "audio_placeholder_missing"
            prefix_length = int(positions[0].item())
            if prefix_length <= 0:
                return None, "audio_text_prefix_empty"
            if not bool(torch.all(attention_mask[:, :prefix_length] == 1).item()):
                return None, "audio_text_prefix_contains_padding"
            prefix = [int(token) for token in input_ids[0, :prefix_length].tolist()]
            return prefix, None
        except Exception as exc:
            return None, f"audio_placeholder_mask_error:{type(exc).__name__}"

    def _run_chunked_thinker_audio_language_model(
        self,
        *,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor | None,
        cache: Any | None,
        is_compiled: bool,
        chunk_tokens: int,
        runtime_metrics: dict[str, Any],
        listener_admission_wait: Callable[[], Any] | None,
    ) -> Any:
        """Cooperatively prefill only the text LM after direct-audio assembly.

        Gemma4's audio features, placeholder scatter, and per-layer input tensor
        are prepared by the caller exactly once. Every LM chunk gets a fresh
        Transformers mask for its absolute query positions and current KV cache.
        """
        if chunk_tokens < 1:
            raise ValueError("audio LM chunk size must be positive")
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
            raise RuntimeError("cooperative direct-audio prefill requires batch-one embeddings")
        sequence_length = int(inputs_embeds.shape[1])
        if sequence_length < 1:
            raise RuntimeError("cooperative direct-audio prefill received an empty sequence")
        if attention_mask.ndim != 2 or attention_mask.shape[0] != 1:
            raise RuntimeError("cooperative direct-audio prefill requires a 2D batch-one attention mask")
        if position_ids.ndim != 2 or tuple(position_ids.shape) != (1, sequence_length):
            raise RuntimeError("cooperative direct-audio prefill requires one absolute position per query")
        if per_layer_inputs is not None and (
            per_layer_inputs.ndim != 4
            or tuple(per_layer_inputs.shape[:2]) != tuple(inputs_embeds.shape[:2])
        ):
            raise RuntimeError("Gemma4 per-layer inputs must cover the full direct-audio query once")
        if is_compiled and (
            cache is None
            or not isinstance(cache_position, torch.Tensor)
            or cache_position.ndim != 1
            or int(cache_position.numel()) != sequence_length
        ):
            raise RuntimeError("compiled direct-audio chunks require matching StaticCache positions")

        outer_config = self.thinker.config
        text_config = outer_config.get_text_config()
        bidirectional = getattr(text_config, "use_bidirectional_attention", None)
        if bidirectional is not None:
            raise RuntimeError(
                "cooperative direct-audio prefill supports causal Gemma4 text attention only; "
                f"use_bidirectional_attention={bidirectional!r}"
            )
        if not bool(getattr(text_config, "is_causal", True)):
            raise RuntimeError("cooperative direct-audio prefill requires a causal text config")
        layer_types = getattr(text_config, "layer_types", None)
        layer_count = getattr(text_config, "num_hidden_layers", None)
        if (
            not isinstance(layer_types, (tuple, list))
            or isinstance(layer_count, bool)
            or not isinstance(layer_count, Integral)
            or len(layer_types) != int(layer_count)
            or not layer_types
            or any(layer_type not in {"full_attention", "sliding_attention"} for layer_type in layer_types)
        ):
            raise RuntimeError("cooperative direct-audio prefill found unsupported Gemma4 layer-mask types")
        mask_builder = getattr(self.thinker, "create_masks_for_generate", None)
        language_model = getattr(getattr(self.thinker, "model", None), "language_model", None)
        if not callable(mask_builder) or not callable(language_model):
            raise RuntimeError("cooperative direct-audio prefill requires Gemma4 mask and text-LM APIs")

        backend = (
            getattr(text_config, "_attn_implementation", None)
            or getattr(outer_config, "_attn_implementation", None)
        )
        # The mask equivalence contract is covered for eager/SDPA. Leave other
        # kernels unavailable to this opt-in until their cached multi-query
        # causal alignment has an equivalent direct test.
        supported_backends = {"eager", "sdpa"}
        if backend not in supported_backends:
            raise RuntimeError(
                "cooperative direct-audio prefill does not recognize the active attention backend "
                f"{backend!r}"
            )
        runtime_metrics["thinker_audio_prefill_attention_backend"] = backend

        # Position IDs are generated as one contiguous absolute range by the
        # direct-audio caller. Keep cache/query indexing on that same range.
        first_position = int(position_ids[0, 0].item())
        expected_positions = torch.arange(
            first_position,
            first_position + sequence_length,
            device=position_ids.device,
            dtype=position_ids.dtype,
        ).unsqueeze(0)
        if not torch.equal(position_ids, expected_positions):
            raise RuntimeError("direct-audio chunking requires contiguous absolute Gemma4 position IDs")
        cached_length = 0
        if cache is not None:
            get_seq_length = getattr(cache, "get_seq_length", None)
            if not callable(get_seq_length):
                raise RuntimeError("direct-audio chunk cache does not expose get_seq_length()")
            cached_length_value = get_seq_length()
            cached_length = (
                int(cached_length_value.item())
                if isinstance(cached_length_value, torch.Tensor)
                else int(cached_length_value)
            )
        if cached_length != first_position:
            raise RuntimeError(
                "direct-audio absolute query positions do not start at the current KV length "
                f"(position={first_position}, cache={cached_length})"
            )
        if is_compiled and (
            not torch.equal(cache_position.to(position_ids.device).unsqueeze(0), position_ids)
        ):
            raise RuntimeError("StaticCache positions do not match direct-audio absolute positions")

        hidden_chunks: list[list[torch.Tensor]] = []
        final_hidden_chunks: list[torch.Tensor] = []
        cuda_event_pairs: list[tuple[Any, Any]] = []
        chunk_records: list[dict[str, Any]] = []
        output = None
        for start in range(0, sequence_length, chunk_tokens):
            end = min(start + chunk_tokens, sequence_length)
            query_embeds = inputs_embeds[:, start:end, :]
            query_positions = position_ids[:, start:end]
            absolute_visible_end = first_position + end
            if is_compiled:
                query_attention_mask = attention_mask
            else:
                if attention_mask.shape[1] < absolute_visible_end:
                    raise RuntimeError("direct-audio attention mask omits visible absolute key positions")
                query_attention_mask = attention_mask[:, :absolute_visible_end]

            # Transformers 5.5.0 create_masks_for_generate accepts the query
            # embeddings, 2D visible-token mask, current Cache, and absolute
            # position_ids. Rebuild on each chunk; slicing a full 4D mask would
            # not preserve cached multi-token causal alignment.
            # Source: https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/masking_utils.py
            chunk_masks = mask_builder(
                outer_config,
                query_embeds,
                query_attention_mask,
                cache,
                query_positions,
            )
            if not isinstance(chunk_masks, dict) or any(layer_type not in chunk_masks for layer_type in set(layer_types)):
                raise RuntimeError("Gemma4 mask builder omitted a required full/sliding attention mask")
            for layer_type in set(layer_types):
                mask_value = chunk_masks[layer_type]
                if mask_value is not None:
                    if not isinstance(mask_value, torch.Tensor) or mask_value.ndim != 4:
                        raise RuntimeError(
                            f"unsupported Gemma4 mask object for {layer_type}: {type(mask_value).__name__}"
                        )
                    if int(mask_value.shape[-2]) != end - start:
                        raise RuntimeError(
                            f"Gemma4 {layer_type} mask query rows do not match the current LM chunk"
                        )

            self._wait_for_listener_admission(
                listener_admission_wait, runtime_metrics, "thinker_prefill"
            )
            wait_started = time.perf_counter()
            if not self._listener_priority_lock():
                raise GenerationAdmissionAborted("listener admission cancelled before audio LM chunk")
            lock_acquired = time.perf_counter()
            wait_ms = (lock_acquired - wait_started) * 1000.0
            lm_stream = None
            lm_start_event = lm_end_event = None
            primary_error: BaseException | None = None
            try:
                # Everything after acquiring the shared model lock belongs inside
                # this guard: argument slicing or CUDA event setup can fail too.
                runtime_metrics["thinker_prefill_lock_wait_ms"] = (
                    runtime_metrics.get("thinker_prefill_lock_wait_ms", 0.0) + wait_ms
                )
                language_kwargs: dict[str, Any] = {
                    "per_layer_inputs": (
                        None if per_layer_inputs is None else per_layer_inputs[:, start:end, :, :]
                    ),
                    "attention_mask": chunk_masks,
                    "position_ids": query_positions,
                    "past_key_values": cache,
                    "inputs_embeds": query_embeds,
                    "use_cache": True,
                    "return_dict": True,
                    "output_hidden_states": True,
                    "logits_to_keep": 1,
                }
                if cache_position is not None:
                    language_kwargs["cache_position"] = cache_position[start:end]
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    lm_stream = torch.cuda.current_stream(self.device)
                    lm_start_event = torch.cuda.Event(enable_timing=True)
                    lm_end_event = torch.cuda.Event(enable_timing=True)
                    lm_start_event.record(lm_stream)
                output = language_model(**language_kwargs)
                hidden_states = getattr(output, "hidden_states", None)
                final_hidden = getattr(output, "last_hidden_state", None)
                if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
                    raise RuntimeError("Gemma4 audio LM chunk omitted requested hidden states")
                if not isinstance(final_hidden, torch.Tensor) or tuple(final_hidden.shape[:2]) != (1, end - start):
                    raise RuntimeError("Gemma4 audio LM chunk final hidden rows do not match query positions")
                if any(
                    not isinstance(hidden, torch.Tensor)
                    or tuple(hidden.shape[:2]) != (1, end - start)
                    for hidden in hidden_states
                ):
                    raise RuntimeError("Gemma4 hidden-state rows do not match the current LM chunk")
                if hidden_chunks and len(hidden_states) != len(hidden_chunks[0]):
                    raise RuntimeError("Gemma4 hidden-state tuple changed across audio LM chunks")
                hidden_chunks.append(list(hidden_states))
                final_hidden_chunks.append(final_hidden)
                if is_compiled:
                    cache = cache
                else:
                    cache = getattr(output, "past_key_values", None)
                    if cache is None:
                        raise RuntimeError("Gemma4 audio LM chunk did not return its updated KV cache")
                if lm_end_event is not None:
                    lm_end_event.record(lm_stream)
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                try:
                    # Drain any work queued on the configured stream before the
                    # lock becomes available to a listener. If forward/setup
                    # already failed, keep that original exception authoritative.
                    if lm_stream is not None:
                        try:
                            lm_stream.synchronize()
                        except BaseException:
                            if primary_error is None:
                                raise
                            logger.exception(
                                "secondary CUDA stream drain failure after audio LM prefill error"
                            )
                finally:
                    try:
                        hold_ms = (time.perf_counter() - lock_acquired) * 1000.0
                        self._thinker_audio_prefill_lock_hold_max_ms = max(
                            self._thinker_audio_prefill_lock_hold_max_ms, hold_ms
                        )
                        runtime_metrics["thinker_prefill_lock_hold_ms"] = (
                            runtime_metrics.get("thinker_prefill_lock_hold_ms", 0.0) + hold_ms
                        )
                        runtime_metrics["thinker_prefill_lock_hold_max_ms"] = max(
                            runtime_metrics.get("thinker_prefill_lock_hold_max_ms", 0.0), hold_ms
                        )
                        runtime_metrics["thinker_prefill_lock_80ms_frames"] = (
                            runtime_metrics.get("thinker_prefill_lock_80ms_frames", 0.0) + hold_ms / 80.0
                        )
                    finally:
                        self._thinker_lock.release()
            if lm_start_event is not None and lm_end_event is not None:
                cuda_event_pairs.append((lm_start_event, lm_end_event))
            chunk_records.append({
                "query_start": start,
                "query_end": end,
                "absolute_position_start": first_position + start,
                "lock_wait_ms": wait_ms,
                "lock_hold_ms": hold_ms,
            })

        if output is None or len(final_hidden_chunks) != math.ceil(sequence_length / chunk_tokens):
            raise RuntimeError("cooperative direct-audio LM prefill did not complete every query row")
        merged_hidden_states = tuple(
            torch.cat([part[index] for part in hidden_chunks], dim=1)
            for index in range(len(hidden_chunks[0]))
        )
        merged_final_hidden = torch.cat(final_hidden_chunks, dim=1)
        runtime_metrics["thinker_audio_lm_prefill_chunks"] = len(chunk_records)
        runtime_metrics["thinker_audio_lm_prefill_chunk_tokens"] = int(chunk_tokens)
        runtime_metrics["thinker_audio_lm_prefill_chunk_timings"] = chunk_records
        if cuda_event_pairs:
            event_values = [
                _completed_cuda_event_elapsed_ms(start_event, end_event)
                for start_event, end_event in cuda_event_pairs
            ]
            completed = [value for value in event_values if value is not None]
            runtime_metrics["thinker_lm_prefill_gpu_ms"] = float(sum(completed))
            runtime_metrics["thinker_lm_prefill_gpu_ms_pending"] = len(completed) != len(event_values)
        return SimpleNamespace(
            last_hidden_state=merged_final_hidden,
            hidden_states=merged_hidden_states,
            past_key_values=cache,
            attentions=getattr(output, "attentions", None),
        )

    def _run_thinker_audio_prefill(
        self,
        audio_inputs: dict[str, torch.Tensor],
        *,
        is_compiled: bool,
        static_cache: Any | None,
        past_length: int,
        cache_init_started: float | None,
        runtime_metrics: dict[str, Any],
        listener_admission_wait: Callable[[], Any] | None = None,
        cache_capacity: int | None = None,
        capture_audio_context: bool = False,
        capture_audio_prefill_frame_raw: bool = False,
        audio_context_source_pcm_samples: int | None = None,
    ) -> tuple[Any, list[torch.Tensor], Any, torch.Tensor, AudioContextHiddenSnapshot | None]:
        """Run one direct-audio prefill after any exactly reused text prefix.

        Gemma4 still receives the complete current audio feature/mask pair and
        every placeholder token in the suffix together. The audio tower,
        feature filtering, and placeholder scatter run once; the opt-in may
        split only the subsequent causal text-LM prefill into lock quanta.
        Only hard text token positions strictly before the first audio
        placeholder can be reused from the committed prefix cache.
        """
        # Transformers 5.5.0 Gemma4Model.forward runs get_audio_features(), strips
        # encoder padding, scatters the valid tokens at audio placeholders, builds
        # causal masks, then calls the language model. Keep those exact operations
        # and tensors, but do the stateless audio tower before acquiring the shared
        # listener/Thinker LM lock.
        # Source: https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/gemma4/modeling_gemma4.py
        full_input_ids = audio_inputs["input_ids"]
        input_ids = full_input_ids[:, past_length:]
        if input_ids.shape[1] == 0:
            raise RuntimeError("prefill multimodale non contiene il suffisso audio corrente")
        audio_context_snapshot: AudioContextHiddenSnapshot | None = None
        attention_mask = audio_inputs["attention_mask"]
        model = getattr(self.thinker, "model", None)
        if model is None or not callable(getattr(model, "get_audio_features", None)):
            raise RuntimeError("Gemma4 pinned API get_audio_features non disponibile")
        if not callable(getattr(model, "get_placeholder_mask", None)):
            raise RuntimeError("Gemma4 pinned API get_placeholder_mask non disponibile")
        if not callable(getattr(self.thinker, "create_masks_for_generate", None)):
            raise RuntimeError("Gemma4 pinned API create_masks_for_generate non disponibile")
        if is_compiled and static_cache is None:
            raise ValueError("prefill audio compilato richiede una StaticCache fresca")
        cache_position = None
        lm_attention_mask = attention_mask
        if is_compiled:
            static_kwargs = self._static_thinker_kwargs(
                input_ids,
                static_cache,
                start_position=past_length,
                visible_tokens=int(full_input_ids.shape[1]),
                cache_capacity=cache_capacity,
                attention_mask_prefix=attention_mask,
            )
            lm_attention_mask = static_kwargs["attention_mask"]
            position_ids = static_kwargs["position_ids"]
            cache_position = static_kwargs["cache_position"]

        self._wait_for_listener_admission(
            listener_admission_wait, runtime_metrics, "thinker_prefill"
        )
        self._thinker_audio_forward_active.set()
        audio_start_event = audio_end_event = None
        try:
            audio_stream = (
                torch.cuda.current_stream(self.device)
                if torch.cuda.is_available() and str(self.device).startswith("cuda")
                else None
            )
            audio_started = time.perf_counter()
            if audio_stream is not None:
                audio_start_event = torch.cuda.Event(enable_timing=True)
                audio_end_event = torch.cuda.Event(enable_timing=True)
                audio_start_event.record(audio_stream)
            audio_output = model.get_audio_features(
                audio_inputs["input_features"],
                audio_inputs["input_features_mask"],
                return_dict=True,
            )
            if audio_end_event is not None:
                audio_end_event.record(audio_stream)
            runtime_metrics["thinker_audio_feature_enqueue_ms"] = (
                time.perf_counter() - audio_started
            ) * 1000.0
            audio_features = getattr(audio_output, "pooler_output", None)
            audio_feature_mask = getattr(audio_output, "attention_mask", None)
            if not isinstance(audio_features, torch.Tensor) or audio_features.ndim != 3:
                raise RuntimeError("Gemma4 get_audio_features non ha restituito soft token validi")
            if not isinstance(audio_feature_mask, torch.Tensor) or audio_feature_mask.ndim != 2:
                raise RuntimeError("Gemma4 get_audio_features non ha restituito la mask encoder")
            if tuple(audio_features.shape[:2]) != tuple(audio_feature_mask.shape):
                raise RuntimeError("Gemma4 soft token e mask encoder non sono allineati")
            audio_features = audio_features[
                audio_feature_mask.to(device=audio_features.device, dtype=torch.bool)
            ]

            image_mask, video_mask, audio_mask = model.get_placeholder_mask(input_ids=input_ids)
            if bool(image_mask.any().item()) or bool(video_mask.any().item()):
                raise RuntimeError("prefill direct audio non accetta placeholder image/video")
            multimodal_mask = image_mask | video_mask | audio_mask
            llm_input_ids = input_ids.clone()
            llm_input_ids[multimodal_mask] = int(model.config.text_config.pad_token_id)
            inputs_embeds = model.get_input_embeddings()(llm_input_ids)
            text_config = model.config.get_text_config()
            if bool(getattr(text_config, "hidden_size_per_layer_input", 0)):
                pad_embedding = model.language_model.embed_tokens.weight[
                    model.config.text_config.pad_token_id, :
                ]
                llm_inputs_embeds = torch.where(
                    multimodal_mask[..., None], pad_embedding.view(1, 1, -1), inputs_embeds
                )
                per_layer_inputs = model.language_model.get_per_layer_inputs(
                    llm_input_ids, llm_inputs_embeds
                )
            else:
                per_layer_inputs = None

            expected_audio_values = int(audio_mask.sum().item()) * int(inputs_embeds.shape[-1])
            if int(audio_features.numel()) != expected_audio_values:
                raise RuntimeError(
                    "Gemma4 encoder soft token count non coincide con i placeholder "
                    f"(features={int(audio_features.numel())}, placeholders={expected_audio_values})"
                )
            audio_mask_expanded = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_mask_expanded,
                audio_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
            )
            if not is_compiled:
                position_ids = torch.arange(
                    past_length,
                    past_length + int(inputs_embeds.shape[1]),
                    device=inputs_embeds.device,
                    dtype=torch.long,
                ).unsqueeze(0)
            audio_lm_chunk_tokens = int(
                getattr(self, "thinker_audio_prefill_chunk_tokens", 0)
            )
            if audio_lm_chunk_tokens > 0:
                language_output = self._run_chunked_thinker_audio_language_model(
                    inputs_embeds=inputs_embeds,
                    per_layer_inputs=per_layer_inputs,
                    attention_mask=lm_attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    cache=static_cache if is_compiled else None,
                    is_compiled=is_compiled,
                    chunk_tokens=audio_lm_chunk_tokens,
                    runtime_metrics=runtime_metrics,
                    listener_admission_wait=listener_admission_wait,
                )
            else:
                causal_mask_mapping = self.thinker.create_masks_for_generate(
                    self.thinker.config,
                    inputs_embeds,
                    lm_attention_mask,
                    static_cache if is_compiled else None,
                    position_ids,
                )

                self._wait_for_listener_admission(
                    listener_admission_wait, runtime_metrics, "thinker_prefill"
                )
                lock_wait_started = time.perf_counter()
                self._listener_priority_lock()
                lock_acquired = time.perf_counter()
                runtime_metrics["thinker_prefill_lock_wait_ms"] += (
                    lock_acquired - lock_wait_started
                ) * 1000.0
                lm_prefill_start_event = lm_prefill_end_event = None
                if (
                    capture_audio_prefill_frame_raw
                    or os.environ.get("DUPLEX_THINKER_CUDA_TIMING", "0") == "1"
                ) and audio_stream is not None:
                    lm_prefill_start_event = torch.cuda.Event(enable_timing=True)
                    lm_prefill_end_event = torch.cuda.Event(enable_timing=True)
                    lm_prefill_start_event.record(audio_stream)
                try:
                    language_kwargs: dict[str, Any] = {
                        "per_layer_inputs": per_layer_inputs,
                        "attention_mask": causal_mask_mapping,
                        "position_ids": position_ids,
                        "past_key_values": static_cache if is_compiled else None,
                        "inputs_embeds": inputs_embeds,
                        "use_cache": True,
                        "return_dict": True,
                        "output_hidden_states": True,
                        "logits_to_keep": 1,
                    }
                    if cache_position is not None:
                        language_kwargs["cache_position"] = cache_position
                    language_output = model.language_model(**language_kwargs)
                    if lm_prefill_end_event is not None:
                        lm_prefill_end_event.record(audio_stream)
                    if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                        torch.cuda.current_stream(self.device).synchronize()
                finally:
                    hold_ms = (time.perf_counter() - lock_acquired) * 1000.0
                    self._thinker_audio_prefill_lock_hold_max_ms = max(
                        self._thinker_audio_prefill_lock_hold_max_ms, hold_ms
                    )
                    runtime_metrics["thinker_prefill_lock_hold_ms"] += hold_ms
                    runtime_metrics["thinker_prefill_lock_hold_max_ms"] = max(
                        runtime_metrics["thinker_prefill_lock_hold_max_ms"], hold_ms
                    )
                    runtime_metrics["thinker_prefill_lock_hold_80ms_frames"] = hold_ms / 80.0
                    self._thinker_lock.release()
                if lm_prefill_start_event is not None and lm_prefill_end_event is not None:
                    _record_completed_cuda_timing(
                        runtime_metrics,
                        "thinker_lm_prefill_gpu_ms",
                        lm_prefill_start_event,
                        lm_prefill_end_event,
                    )

            cache = static_cache if is_compiled else language_output.past_key_values
            if cache is None:
                raise RuntimeError("prefill audio Gemma non ha restituito la cache KV")
            if not isinstance(getattr(language_output, "hidden_states", None), (tuple, list)):
                raise RuntimeError("prefill audio Gemma non ha restituito gli hidden states")
            last_hidden = language_output.last_hidden_state
            logits = self.thinker.lm_head(last_hidden[:, -1:, :])
            if (softcap := getattr(text_config, "final_logit_softcapping", None)) is not None:
                logits = logits / softcap
                logits = torch.tanh(logits) * softcap
            output = SimpleNamespace(
                logits=logits,
                past_key_values=cache,
                hidden_states=language_output.hidden_states,
                final_hidden_state=language_output.last_hidden_state,
                attentions=getattr(language_output, "attentions", None),
                audio_hidden_states=audio_features,
            )
            if is_compiled:
                cache_length = static_cache_sequence_length(cache)
                if cache_length != int(full_input_ids.shape[1]):
                    raise RuntimeError(
                        "StaticCache audio e input_ids non sono allineati dopo il prefill "
                        f"(cache={cache_length}, ids={int(full_input_ids.shape[1])})"
                    )
                if cache_init_started is not None:
                    runtime_metrics["thinker_cache_init_ms"] = (
                        time.perf_counter() - cache_init_started
                    ) * 1000.0
            if audio_end_event is not None and audio_start_event is not None:
                _record_completed_cuda_timing(
                    runtime_metrics,
                    "thinker_audio_encoder_gpu_ms",
                    audio_start_event,
                    audio_end_event,
                )
            if capture_audio_context or capture_audio_prefill_frame_raw:
                processor = getattr(self._thinker_wrapper, "processor", None)
                audio_ms_per_token = getattr(processor, "audio_ms_per_token", None)
                if (
                    isinstance(audio_ms_per_token, bool)
                    or not isinstance(audio_ms_per_token, Integral)
                    or int(audio_ms_per_token) <= 0
                ):
                    source_name = (
                        "audio_prefill_frame_raw"
                        if capture_audio_prefill_frame_raw else "V9 dual audio-context"
                    )
                    raise RuntimeError(f"{source_name} requires processor.audio_ms_per_token")
                audio_context_snapshot = _capture_audio_context_hidden(
                    self.thinker,
                    input_ids=full_input_ids,
                    prefill_hidden=(
                        output.final_hidden_state
                        if capture_audio_prefill_frame_raw
                        else output.hidden_states[-1]
                    ),
                    past_length=past_length,
                    audio_ms_per_token=int(audio_ms_per_token),
                    source_pcm_samples=audio_context_source_pcm_samples,
                    require_single_audio_run=capture_audio_prefill_frame_raw,
                )
            return output, [output.hidden_states[-1]], cache, attention_mask, audio_context_snapshot
        finally:
            self._thinker_audio_forward_active.clear()

    @_private_thinker_response_stream
    @torch.inference_mode()
    def think(self, prompt_token_ids: list[int], *, max_new_tokens: int = 64,
              temperature: float = 0.7, top_p: float = 0.9, repetition_penalty: float = 1.15,
              stop_check: Any | None = None,
              state: Any | None = None, generation_id: int | None = None,
              on_text_chunk: Any | None = None, min_new_tokens: int = 0,
              on_generation_start: Any | None = None,
              on_token_id: Callable[[int], Any] | None = None,
              on_generated_token: Callable[[int, int, str, torch.Tensor], int | tuple[int, str] | None] | None = None,
              on_candidate_token: Callable[[int, str, bool], bool] | None = None,
              on_candidate_fallback: Callable[
                  [list[tuple[int, str, bool]]],
                  int | tuple[int, ...] | list[int] | None,
              ] | None = None,
              on_action_logits_probe: Callable[[dict[str, Any]], Any] | None = None,
              action: str | None = None,
              response_protocol: str | None = None,
              input_audio: dict[str, Any] | None = None,
              capture_audio_context: bool = False,
              capture_audio_prefill_frame_raw: bool = False,
              audio_context_source_pcm_samples: int | None = None,
              listener_admission_wait: Callable[[], Any] | None = None,
              disable_thinker_cadence_pacing: bool = False,
              graceful_sentence_completion: bool = False) -> dict[str, Any]:
        """Decode Thinker controllato dal kernel con riuso opportunistico della KV."""
        audio_input_prepare_started = time.perf_counter()
        audio_inputs = (
            None if input_audio is None
            else self._prepare_thinker_audio_inputs(input_audio, prompt_token_ids)
        )
        audio_input_prepare_ms = (time.perf_counter() - audio_input_prepare_started) * 1000.0
        has_multimodal_audio = audio_inputs is not None
        compile_enabled = getattr(self, "_compiled_thinker", None) is not None
        active_audio_response_protocol = (
            getattr(self, "audio_response_protocol", None)
            if response_protocol is None else response_protocol
        )
        audio_response_v2 = (
            has_multimodal_audio
            and active_audio_response_protocol == AUDIO_RESPONSE_PROTOCOL_V2
        )
        if capture_audio_context and capture_audio_prefill_frame_raw:
            raise ValueError("V9 dual context e audio_prefill_frame_raw sono fonti mutuamente esclusive")
        if capture_audio_context and not audio_response_v2:
            raise ValueError("V9 dual audio-context capture is restricted to direct protocol-v2 audio")
        if capture_audio_prefill_frame_raw and (
            not has_multimodal_audio
            or active_audio_response_protocol != CONTROLLER_GUIDED_SAY_V1
        ):
            raise ValueError(
                "audio_prefill_frame_raw richiede input multimodale controller_guided_say_v1"
            )
        cache_capacity = int(
            getattr(self, "thinker_static_cache_max_len", AUDIO_THINKER_CACHE_BUCKETS[-1])
        )
        requested_max_new_tokens = max(0, int(max_new_tokens))
        graceful_extension_tokens = 0
        if graceful_sentence_completion:
            if not has_multimodal_audio or active_audio_response_protocol != CONTROLLER_GUIDED_SAY_V1:
                raise ValueError("graceful_sentence_completion requires controller-guided audio")
            available_cache_tokens = max(
                0, cache_capacity - len(prompt_token_ids) - requested_max_new_tokens
            )
            graceful_extension_tokens = min(32, available_cache_tokens)
        decode_token_limit = requested_max_new_tokens + graceful_extension_tokens
        if compile_enabled and audio_response_v2:
            cache_capacity = select_audio_thinker_cache_bucket(
                int(audio_inputs["input_ids"].shape[1]),
                max_new_tokens,
                max_cache_len=min(
                    int(self.thinker_static_cache_max_len),
                    AUDIO_THINKER_CACHE_BUCKETS[-1],
                ),
            )
        audio_cache_prefix_ids: list[int] | None = None
        audio_cache_prefix_reason: str | None = None
        if has_multimodal_audio:
            if audio_response_v2:
                audio_cache_prefix_reason = "audio_cross_response_prefix_reuse_disabled_for_v2"
            elif getattr(self, "audio_text_prefix_cache_reuse_enabled", False):
                audio_cache_prefix_ids, audio_cache_prefix_reason = self._audio_text_cache_prefix(audio_inputs)
            else:
                audio_cache_prefix_reason = "audio_cross_response_prefix_reuse_disabled"
        normalized_action = str(action or "").strip().upper()
        if has_multimodal_audio:
            # Candidate audio responses are trained to choose their own wording
            # and timing from input features, without the legacy action trie.
            normalized_action = ""
        backchannel_trie = None
        if normalized_action == "BACKCHANNEL":
            backchannel_trie = getattr(self, "_backchannel_token_trie", None)
            if not backchannel_trie:
                raise RuntimeError("BACKCHANNEL richiede il trie di frasi gold del bundle attivo")
        if (
            compile_enabled
            and not audio_response_v2
            and len(prompt_token_ids) + decode_token_limit > self.thinker_static_cache_max_len
        ):
            raise ValueError(
                "prompt Thinker + max_new_tokens supera thinker_static_cache_max_len "
                f"({len(prompt_token_ids)} + {decode_token_limit} > {self.thinker_static_cache_max_len}); "
                "ridurre esplicitamente il budget o aumentare la capacità"
            )
        cached_ids = list(getattr(state, "thinker_cache_ids", [])) if state is not None and not compile_enabled else []
        use_cached = bool(not has_multimodal_audio and cached_ids and len(prompt_token_ids) > len(cached_ids)
                          and prompt_token_ids[:len(cached_ids)] == cached_ids
                          and getattr(state, "thinker_kv", None) is not None
                          and getattr(state, "thinker_hidden", None) is not None)
        compiled_thinker = getattr(self, "_compiled_thinker", None)
        is_compiled = compiled_thinker is not None
        static_cache = None
        past_key_values = None
        past_length = 0
        prefill_ids = prompt_token_ids
        hidden_prefix: list[torch.Tensor] = []
        runtime_metrics: dict[str, Any] = {
            "thinker_prefill_lock_wait_ms": 0.0,
            "thinker_prefill_lock_hold_ms": 0.0,
            "thinker_prefill_lock_hold_max_ms": 0.0,
            "thinker_decode_lock_wait_ms": 0.0,
            "thinker_decode_lock_hold_ms": 0.0,
            "thinker_decode_lock_hold_max_ms": 0.0,
            "thinker_prefill_chunks": 0,
            "thinker_decode_steps": 0,
            "listener_admission_thinker_prefill_wait_ms": 0.0,
            "listener_admission_thinker_prefill_wait_max_ms": 0.0,
            "listener_admission_thinker_decode_wait_ms": 0.0,
            "listener_admission_thinker_decode_wait_max_ms": 0.0,
        }
        if has_multimodal_audio:
            runtime_metrics.update({
                "thinker_audio_prefill": True,
                "thinker_audio_input_prepare_ms": audio_input_prepare_ms,
                "thinker_cache_bucket_capacity": cache_capacity if compile_enabled and audio_response_v2 else None,
                "thinker_cache_reused_tokens": 0,
                "thinker_cache_hit": False,
                "thinker_cache_reason": (
                    f"fresh:{audio_cache_prefix_reason}"
                    if audio_cache_prefix_reason is not None
                    else "audio_prefix_reuse_requires_static_cache"
                    if not compile_enabled
                    else "audio_prefix_no_committed_cache"
                    if state is not None
                    else "audio_prefix_no_session"
                ),
            })
        audio_context_snapshot: AudioContextHiddenSnapshot | None = None
        if capture_audio_context:
            runtime_metrics["v9_dual_audio_context_capture_requested"] = True
        if capture_audio_prefill_frame_raw:
            runtime_metrics["v9_audio_prefill_frame_raw_capture_requested"] = True
        if backchannel_trie is not None:
            runtime_metrics["backchannel_constraint"] = "train_phrase_token_trie"
            runtime_metrics["backchannel_phrase_count"] = int(
                getattr(self, "_backchannel_phrase_count", 0)
            )
            runtime_metrics["backchannel_trie_steps"] = 0
        cache_init_started = None
        compiled_cache_fingerprint = None
        if is_compiled:
            cache_init_started = time.perf_counter()
            compiled_cache_fingerprint = self._compiled_thinker_cache_fingerprint()
            runtime_metrics.update({
                "thinker_cache_reused_tokens": 0,
                "thinker_cache_copy_tokens": 0,
                "thinker_cache_copy_bytes": 0,
                "thinker_cache_copy_ms": 0.0,
                "thinker_cache_copy_lock_hold_ms": 0.0,
                "thinker_cache_hit": False,
                "thinker_cache_reason": (
                    (
                        f"fresh:{audio_cache_prefix_reason}"
                        if audio_cache_prefix_reason is not None
                        else "audio_prefix_no_committed_cache" if state is not None
                        else "audio_prefix_no_session"
                    )
                    if has_multimodal_audio
                    else "no_committed_cache" if state is not None else "no_session"
                ),
            })
            cached_prompt_ids = (
                list(getattr(state, "thinker_static_cache_ids", []))
                if state is not None and (not has_multimodal_audio or audio_cache_prefix_ids is not None)
                else []
            )
            cached_static_cache = (
                getattr(state, "thinker_static_cache", None)
                if state is not None else None
            )
            cached_fingerprint = (
                getattr(state, "thinker_static_cache_fingerprint", None)
                if state is not None else None
            )
            cache_context_ids = (
                audio_cache_prefix_ids if has_multimodal_audio else prompt_token_ids
            )
            if cached_static_cache is not None and cached_prompt_ids:
                if cached_fingerprint != compiled_cache_fingerprint:
                    runtime_metrics["thinker_cache_reason"] = "fingerprint_mismatch"
                else:
                    try:
                        committed_length = static_cache_sequence_length(cached_static_cache)
                    except CachePrefixCopyError as exc:
                        committed_length = -1
                        runtime_metrics["thinker_cache_reason"] = f"length_unreadable:{exc}"
                    if committed_length >= 0 and committed_length != len(cached_prompt_ids):
                        runtime_metrics["thinker_cache_reason"] = "cached_ids_length_mismatch"
                    elif committed_length >= 0:
                        common_prefix = longest_common_prefix_length(cached_prompt_ids, cache_context_ids or [])
                        common_prefix = min(common_prefix, committed_length)
                        if common_prefix == 0:
                            runtime_metrics["thinker_cache_reason"] = "no_common_prefix"
                        elif not has_multimodal_audio and common_prefix >= len(prompt_token_ids):
                            # Cached prompt KV has no final-token logits. Re-prefill
                            # equal/shorter prompts to preserve the ordinary logits boundary.
                            runtime_metrics["thinker_cache_reason"] = "empty_suffix_requires_logits"
                        else:
                            branch = self._new_thinker_static_cache(cache_capacity)
                            try:
                                self._copy_compiled_thinker_prefix(
                                    cached_static_cache, branch, common_prefix, runtime_metrics
                                )
                            except CachePrefixCopyError as exc:
                                runtime_metrics["thinker_cache_reason"] = f"branch_copy_unsupported:{exc}"
                            else:
                                static_cache = branch
                                past_length = common_prefix
                                prefill_ids = prompt_token_ids[common_prefix:]
                                runtime_metrics["thinker_cache_hit"] = True
                                runtime_metrics["thinker_cache_reused_tokens"] = common_prefix
                                runtime_metrics["thinker_cache_reason"] = (
                                    "exact_text_prefix_before_audio" if has_multimodal_audio
                                    else "exact_token_prefix"
                                )
            if static_cache is None:
                static_cache = self._new_thinker_static_cache(cache_capacity)
            runtime_metrics["thinker_cache_init_method"] = "eager_first_real_chunk"
        elif use_cached:
            prefill_ids = prompt_token_ids[len(cached_ids):]
            past_key_values = state.thinker_kv
            past_length = len(cached_ids)
            hidden_prefix = [state.thinker_hidden]
        prefill_started = time.perf_counter()
        try:
            if has_multimodal_audio:
                out, prefill_hidden, past, mask, audio_context_snapshot = self._run_thinker_audio_prefill(
                    audio_inputs,
                    is_compiled=is_compiled,
                    static_cache=static_cache,
                    cache_capacity=cache_capacity,
                    past_length=past_length,
                    cache_init_started=cache_init_started,
                    runtime_metrics=runtime_metrics,
                    listener_admission_wait=listener_admission_wait,
                    capture_audio_context=capture_audio_context,
                    capture_audio_prefill_frame_raw=capture_audio_prefill_frame_raw,
                    audio_context_source_pcm_samples=audio_context_source_pcm_samples,
                )
                if audio_context_snapshot is not None:
                    capture_metrics = {
                        "status": "captured",
                        "audio_placeholder_count": audio_context_snapshot.audio_placeholder_count,
                        "audio_placeholder_run_count": audio_context_snapshot.audio_placeholder_run_count,
                        "audio_placeholder_run_start": audio_context_snapshot.audio_placeholder_run_start,
                        "audio_placeholder_run_end": audio_context_snapshot.audio_placeholder_run_end,
                        "audio_frame_count": audio_context_snapshot.selected_frame_count,
                        "past_length": audio_context_snapshot.past_length,
                        "audio_ms_per_token": audio_context_snapshot.audio_ms_per_token,
                        "raw_frame_period_ms": audio_context_snapshot.raw_frame_period_ms,
                        "source_pcm_samples": audio_context_snapshot.source_pcm_samples,
                    }
                    metric_name = (
                        "v9_audio_prefill_frame_raw_capture"
                        if capture_audio_prefill_frame_raw
                        else "v9_dual_audio_context_capture"
                    )
                    runtime_metrics[metric_name] = capture_metrics
                if is_compiled and audio_cache_prefix_ids is not None and state is not None:
                    self._publish_compiled_thinker_prompt_cache(
                        state,
                        generation_id,
                        audio_cache_prefix_ids,
                        past,
                        compiled_cache_fingerprint,
                        runtime_metrics,
                    )
            else:
                out, prefill_hidden, past, mask = self._run_thinker_prefill(
                    prefill_ids,
                    past_key_values=past_key_values,
                    past_length=past_length,
                    is_compiled=is_compiled,
                    static_cache=static_cache,
                    cache_init_started=cache_init_started,
                    runtime_metrics=runtime_metrics,
                    listener_admission_wait=listener_admission_wait,
                )
        except GenerationAdmissionAborted:
            runtime_metrics["thinker_prefill_aborted_stale_generation"] = True
            return {
                "text": "",
                "tool_calls": [],
                "hidden": None,
                "prompt_len": len(prompt_token_ids),
                "hidden_prompt_len": len(prompt_token_ids),
                "cancelled": True,
                "runtime_metrics": runtime_metrics,
            }
        runtime_metrics["thinker_prefill_ms"] = (time.perf_counter() - prefill_started) * 1000.0
        runtime_metrics["thinker_prefill_chunks"] = math.ceil(
            len(prefill_ids) / max(1, (4 if is_compiled else self.thinker_prefill_chunk_tokens) or len(prefill_ids))
        )
        if has_multimodal_audio:
            runtime_metrics["thinker_prefill_chunks"] = int(
                runtime_metrics.get("thinker_audio_lm_prefill_chunks", 1)
            )
        hidden_parts = hidden_prefix + prefill_hidden
        hidden_prompt_len = len(prefill_ids) if is_compiled else len(prompt_token_ids)
        runtime_metrics["thinker_prefill_tokens"] = len(prefill_ids)
        if is_compiled and not has_multimodal_audio:
            try:
                runtime_metrics["thinker_cache_branch_storage_bytes"] = static_cache_storage_bytes(past)
            except CachePrefixCopyError as exc:
                runtime_metrics["thinker_cache_storage_reason"] = str(exc)
            if state is not None and compiled_cache_fingerprint is not None:
                self._publish_compiled_thinker_prompt_cache(
                    state,
                    generation_id,
                    prompt_token_ids,
                    past,
                    compiled_cache_fingerprint,
                    runtime_metrics,
                )
        logits = out.logits[:, -1, :]
        generated: list[int] = []
        pending_chunk: list[int] = []
        sampler = self.build_thinker_sampler(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            action=normalized_action,
            backchannel_trie=backchannel_trie,
            runtime_metrics=runtime_metrics,
            on_generation_start=on_generation_start,
            on_generated_token=on_generated_token,
            on_candidate_token=on_candidate_token,
            on_candidate_fallback=on_candidate_fallback,
            on_action_logits_probe=on_action_logits_probe,
            state=state,
            generation_id=generation_id,
        )
        eos = sampler.eos_token_ids

        def _sample_token(curr_logits):
            return sampler.sample_token(curr_logits, generated)

        def _select_valid_candidate(
            token_tensor: torch.Tensor,
            token_id: int,
            candidate_scores: torch.Tensor,
            source_logits: torch.Tensor,
        ) -> tuple[torch.Tensor, int]:
            selected = sampler.select_valid_candidate(
                (token_tensor, token_id, candidate_scores),
                source_logits=source_logits,
                generated_ids=generated,
            )
            return selected[0], selected[1]

        decode_started = time.perf_counter()
        thinker_decode_event_start = thinker_decode_event_end = None
        thinker_decode_event_stream = None
        if (
            os.environ.get("DUPLEX_THINKER_CUDA_TIMING", "0") == "1"
            and torch.cuda.is_available()
            and str(self.device).startswith("cuda")
        ):
            thinker_decode_event_stream = torch.cuda.current_stream(device=self.device)
            thinker_decode_event_start = torch.cuda.Event(enable_timing=True)
            thinker_decode_event_end = torch.cuda.Event(enable_timing=True)
            thinker_decode_event_start.record(thinker_decode_event_stream)
        if graceful_sentence_completion:
            runtime_metrics["thinker_budget_requested_tokens"] = requested_max_new_tokens
            runtime_metrics["thinker_budget_hard_limit_tokens"] = decode_token_limit
            runtime_metrics["thinker_budget_extension_capacity_tokens"] = graceful_extension_tokens
        cancelled_before_first_token = False
        terminal_reason = "budget_exhausted"
        next_tok, next_val, next_candidate_scores = sampler.sample_initial(logits, generated)

        for step in range(decode_token_limit):
            if stop_check is not None and stop_check():
                cancelled_before_first_token = not generated
                terminal_reason = "cancelled_before_first_token" if not generated else "cancelled_after_token"
                break
            next_tok, next_val = _select_valid_candidate(
                next_tok,
                next_val,
                next_candidate_scores,
                logits,
            )
            if next_val in eos:
                if len(generated) < min_new_tokens:
                    raise RuntimeError(
                        "MinNewTokensLengthLogitsProcessor ha restituito EOS prima della soglia "
                        f"({len(generated)} < {min_new_tokens})"
                    )
                # Keep the actual model-emitted terminator in the raw token
                # trace, but never feed it through decode, text/TTS, cache, or
                # another Thinker forward.
                if on_token_id is not None:
                    on_token_id(int(next_val))
                terminal_reason = "eos"
                break
            token, value = next_tok, next_val
            generated.append(value)
            if on_token_id is not None:
                on_token_id(int(value))
            pending_chunk.append(value)
            if not is_compiled or has_multimodal_audio:
                mask = torch.cat([mask, torch.ones((1, 1), device=self.device, dtype=mask.dtype)], dim=1)

            # Priority yield: if listener is waiting for LM, yield immediately
            if (hasattr(self, "listener") and hasattr(self.listener, "backend")
                    and hasattr(self.listener.backend, "_waiting_for_lm")):
                while self.listener.backend._waiting_for_lm.is_set():
                    time.sleep(0.0005)

            if not disable_thinker_cadence_pacing:
                # Legacy conversational text path paces one token every three
                # listener frames. V2 synthesizes only after a complete,
                # validated SAY, so per-token pacing here adds latency without
                # pacing emitted audio. V2 still yields to listener admission
                # before each GPU forward below.
                if (hasattr(self, "listener") and hasattr(self.listener, "backend")
                        and hasattr(self.listener.backend, "_frame_counter")
                        and getattr(self.listener.backend, "_active_streaming", False)):
                    target_frame = self.listener.backend._frame_counter + 3
                    while self.listener.backend._frame_counter < target_frame:
                        time.sleep(0.001)
                elif hasattr(self, "listener") and hasattr(self.listener, "backend"):
                    time.sleep(0.001)

            thinker_call = self._compiled_thinker if is_compiled else self.thinker
            # A listener can register while the thinker is blocked acquiring
            # the shared Gemma lock. Recheck after acquire so a decode token
            # cannot jump ahead of the audio frame that just became ready.
            if not audio_response_v2:
                self._wait_for_listener_admission(
                    listener_admission_wait, runtime_metrics, "thinker_decode"
                )
            if stop_check is not None and stop_check():
                terminal_reason = "cancelled_before_token_forward"
                break
            lock_wait_started = time.perf_counter()
            self._listener_priority_lock()
            lock_acquired = time.perf_counter()
            runtime_metrics["thinker_decode_lock_wait_ms"] += (lock_acquired - lock_wait_started) * 1000.0
            try:
                if is_compiled:
                    tok_tensor = torch.tensor([[value]], device=self.device, dtype=torch.long)
                    current_position = len(prompt_token_ids) + len(generated) - 1
                    out = thinker_call(**self._static_thinker_kwargs(
                        tok_tensor,
                        past,
                        start_position=current_position,
                        visible_tokens=current_position + 1,
                        cache_capacity=cache_capacity,
                        precompute_attention_masks=(
                            callable(getattr(self.thinker, 'get_input_embeddings', None))
                            and callable(getattr(past, 'get_mask_sizes', None))
                        ),
                        attention_mask_prefix=(mask if has_multimodal_audio else None),
                    ))
                else:
                    out = thinker_call(input_ids=token, attention_mask=mask, past_key_values=past,
                                       use_cache=True, logits_to_keep=1,
                                       output_hidden_states=True, return_dict=True)
                if is_compiled:
                    # Keep only per-step outputs consumed after the shared lock.
                    # The response-owned cache remains untouched.
                    out = SimpleNamespace(
                        past_key_values=out.past_key_values,
                        hidden_states=(out.hidden_states[-1].clone(),),
                        logits=out.logits.clone(),
                    )
            finally:
                try:
                    _fence_current_cuda_stage(self.device)
                finally:
                    hold_ms = (time.perf_counter() - lock_acquired) * 1000.0
                    runtime_metrics['thinker_decode_lock_hold_ms'] += hold_ms
                    runtime_metrics['thinker_decode_lock_hold_max_ms'] = max(
                        runtime_metrics.get('thinker_decode_lock_hold_max_ms', 0.0), hold_ms
                    )
                    self._thinker_lock.release()
            runtime_metrics["thinker_decode_steps"] += 1
            past = out.past_key_values
            hidden_parts.append(out.hidden_states[-1])
            logits = out.logits[:, -1, :]

            # Pre-sample next token immediately while listener is executing on GPU
            next_candidate = _sample_token(logits)
            next_tok, next_val, next_candidate_scores = sampler.after_generated_token(
                token_index=len(generated) - 1,
                token_id=int(generated[-1]),
                token_hidden=out.hidden_states[-1][:, -1:, :],
                generated_ids=generated,
                source_logits=logits,
                sampled_next=next_candidate,
            )

            if graceful_sentence_completion and len(generated) >= requested_max_new_tokens:
                budget_text = self.tokenizer.decode(generated, skip_special_tokens=True)
                budget_action = graceful_budget_action(
                    budget_text, len(generated), requested_max_new_tokens, decode_token_limit
                )
                if budget_action == "sentence_complete":
                    runtime_metrics["thinker_budget_extension_used_tokens"] = (
                        len(generated) - requested_max_new_tokens
                    )
                    terminal_reason = "sentence_boundary_after_budget"
                    break

            if on_text_chunk is not None and (generation_id is None or state is None or state.is_generation_current(generation_id=generation_id)):
                piece = self.tokenizer.decode(pending_chunk, skip_special_tokens=True).strip()
                if piece and (len(pending_chunk) >= 8 or piece.endswith((".", "!", "?", ":", ";"))):
                    on_text_chunk(piece, torch.cat(hidden_parts, dim=1), hidden_prompt_len)
                    pending_chunk.clear()
        budget_generated_before_trim = None
        budget_kept_tokens = None
        if graceful_sentence_completion and terminal_reason == "budget_exhausted":
            budget_generated_before_trim = len(generated)
            budget_kept_tokens, budget_text = last_complete_word_prefix(
                generated,
                lambda ids: self.tokenizer.decode(list(ids), skip_special_tokens=True),
            )
            runtime_metrics["thinker_budget_partial"] = True
            runtime_metrics["thinker_budget_extension_used_tokens"] = max(
                0, budget_generated_before_trim - requested_max_new_tokens
            )
            runtime_metrics["thinker_budget_kept_tokens"] = int(budget_kept_tokens)
            runtime_metrics["thinker_budget_dropped_tokens"] = int(
                budget_generated_before_trim - budget_kept_tokens
            )
            generated = generated[:budget_kept_tokens]
            terminal_reason = "budget_partial"
        elif graceful_sentence_completion:
            runtime_metrics["thinker_budget_partial"] = False
            runtime_metrics["thinker_budget_kept_tokens"] = len(generated)

        if on_text_chunk is not None and pending_chunk and (generation_id is None or state is None or state.is_generation_current(generation_id=generation_id)):
            piece = self.tokenizer.decode(pending_chunk, skip_special_tokens=True).strip()
            if piece:
                on_text_chunk(piece, torch.cat(hidden_parts, dim=1), hidden_prompt_len)
        if not generated:
            if cancelled_before_first_token:
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    torch.cuda.current_stream().synchronize()
                if thinker_decode_event_start is not None and thinker_decode_event_end is not None:
                    thinker_decode_event_end.record(thinker_decode_event_stream)
                    _record_completed_cuda_timing(
                        runtime_metrics,
                        "thinker_decode_gpu_ms",
                        thinker_decode_event_start,
                        thinker_decode_event_end,
                    )
                runtime_metrics["thinker_decode_ms"] = (time.perf_counter() - decode_started) * 1000.0
                for key, value in tuple(runtime_metrics.items()):
                    if key.endswith("_ms"):
                        runtime_metrics[key] = round(float(value), 3)
                return {
                    "text": "",
                    "gen_ids": [],
                    "hidden": torch.cat(hidden_parts, dim=1),
                    "prompt_len": len(prompt_token_ids),
                    "hidden_prompt_len": hidden_prompt_len,
                    "past_key_values": None if is_compiled else past,
                    "runtime_metrics": runtime_metrics,
                    "cancelled": True,
                    "terminal_reason": terminal_reason,
                }
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.current_stream().synchronize()
            if thinker_decode_event_start is not None and thinker_decode_event_end is not None:
                thinker_decode_event_end.record(thinker_decode_event_stream)
                _record_completed_cuda_timing(
                    runtime_metrics,
                    "thinker_decode_gpu_ms",
                    thinker_decode_event_start,
                    thinker_decode_event_end,
                )
            runtime_metrics["thinker_decode_ms"] = (time.perf_counter() - decode_started) * 1000.0
            if sampler.forced_action_logprobs:
                runtime_metrics["native_action_logprob"] = float(
                    torch.stack(sampler.forced_action_logprobs).sum().item()
                )
            runtime_metrics["forced_action_token_count"] = len(sampler.forced_action_logprobs)
            runtime_metrics["forced_action_override_count"] = sampler.forced_action_overrides
            runtime_metrics["forced_structure_token_count"] = len(sampler.forced_structure_logprobs)
            runtime_metrics["forced_structure_override_count"] = sampler.forced_structure_overrides
            for key, value in tuple(runtime_metrics.items()):
                if key.endswith("_ms"):
                    runtime_metrics[key] = round(float(value), 3)
            error = RuntimeError("Thinker ha prodotto una risposta vuota")
            error.terminal_reason = terminal_reason
            error.terminal_token_id = int(next_val) if terminal_reason == "eos" else None
            error.empty_eos = terminal_reason == "eos"
            error.runtime_metrics = {
                **runtime_metrics,
                "terminal_reason": terminal_reason,
                "terminal_token_id": error.terminal_token_id,
            }
            raise error
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.current_stream().synchronize()
        if thinker_decode_event_start is not None and thinker_decode_event_end is not None:
            thinker_decode_event_end.record(thinker_decode_event_stream)
            _record_completed_cuda_timing(
                runtime_metrics,
                "thinker_decode_gpu_ms",
                thinker_decode_event_start,
                thinker_decode_event_end,
            )
        runtime_metrics["thinker_decode_ms"] = (time.perf_counter() - decode_started) * 1000.0
        if sampler.forced_action_logprobs:
            runtime_metrics["native_action_logprob"] = float(
                torch.stack(sampler.forced_action_logprobs).sum().item()
            )
        runtime_metrics["forced_action_token_count"] = len(sampler.forced_action_logprobs)
        runtime_metrics["forced_action_override_count"] = sampler.forced_action_overrides
        runtime_metrics["forced_structure_token_count"] = len(sampler.forced_structure_logprobs)
        runtime_metrics["forced_structure_override_count"] = sampler.forced_structure_overrides
        for key, value in tuple(runtime_metrics.items()):
            if key.endswith("_ms"):
                runtime_metrics[key] = round(float(value), 3)
        hidden = torch.cat(hidden_parts, dim=1)
        if budget_generated_before_trim is not None:
            expected_hidden_tokens = hidden_prompt_len + budget_generated_before_trim
            if int(hidden.shape[1]) != expected_hidden_tokens:
                raise RuntimeError(
                    "budget_partial_hidden_alignment_mismatch:"
                    f" hidden={int(hidden.shape[1])} expected={expected_hidden_tokens}"
                )
            hidden = hidden[:, :hidden_prompt_len + budget_kept_tokens, :]
        if state is not None and not is_compiled and not has_multimodal_audio:
            with state._generation_lock:
                if generation_id is None or state.is_generation_current(generation_id=generation_id):
                    state.thinker_kv = past
                    state.thinker_cache_ids = list(prompt_token_ids) + generated
                    state.thinker_hidden = hidden.detach()
                else:
                    state.stale_kv_writes += 1
        cleaned_text, protocol_text = _decode_generated_response_texts(
            self.tokenizer, generated,
        )
        result = {
            "text": cleaned_text,
            "protocol_text": protocol_text,
            "gen_ids": generated,
            "hidden": hidden,
            "prompt_len": len(prompt_token_ids),
            "hidden_prompt_len": hidden_prompt_len,
            # No serving caller consumes this cache. Do not keep the
            # 4096-token mutable StaticCache alive through V9/Talker.
            "past_key_values": None if is_compiled else past,
            "runtime_metrics": runtime_metrics,
            "terminal_reason": terminal_reason,
        }
        if audio_context_snapshot is not None:
            # This private tensor is consumed only by the in-process V9 bridge;
            # it is never included in events, history, or telemetry.
            result["audio_context_snapshot"] = audio_context_snapshot
        return result

    def transcribe_audio(
        self,
        pcm: np.ndarray,
        *,
        max_new_tokens: int = 64,
        stop_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        if self._thinker_wrapper is None:
            raise RuntimeError("engine non caricato")
        lock_wait_ms = 0.0
        lock_hold_ms = 0.0
        prefill_lock_wait_ms = 0.0
        prefill_lock_hold_ms = 0.0
        started = time.perf_counter()

        def prefill_step(forward):
            nonlocal prefill_lock_wait_ms, prefill_lock_hold_ms
            if stop_check is not None and stop_check():
                return None
            wait_started = time.perf_counter()
            self._listener_priority_lock()
            lock_acquired = time.perf_counter()
            prefill_lock_wait_ms += (lock_acquired - wait_started) * 1000.0
            try:
                if stop_check is not None and stop_check():
                    return None
                result = forward()
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    torch.cuda.current_stream().synchronize()
                return result
            finally:
                prefill_lock_hold_ms += (time.perf_counter() - lock_acquired) * 1000.0
                self._thinker_lock.release()

        def decode_step(forward):
            nonlocal lock_wait_ms, lock_hold_ms
            wait_started = time.perf_counter()
            acquired = self._listener_priority_lock(stop_check=stop_check)
            lock_acquired = time.perf_counter()
            lock_wait_ms += (lock_acquired - wait_started) * 1000.0
            if not acquired:
                return None
            try:
                result = forward()
            finally:
                lock_hold_ms += (time.perf_counter() - lock_acquired) * 1000.0
                self._thinker_lock.release()
            # Give the listener's scheduler thread a chance to run after each
            # single Q1 forward. Token bookkeeping and stop checks stay outside
            # the GPU lock in ThinkerModel.transcribe_audio.
            time.sleep(0)
            return result

        compile_enabled = getattr(self, "_compiled_thinker", None) is not None
        cache_alloc_ms = 0.0
        transcribe_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "decode_step": decode_step,
            "prefill_step": prefill_step,
        }
        if stop_check is not None:
            transcribe_kwargs["stop_check"] = stop_check
        if compile_enabled:
            cache_started = time.perf_counter()
            static_cache = self._new_thinker_static_cache(self.thinker_static_cache_max_len)
            cache_alloc_ms = (time.perf_counter() - cache_started) * 1000.0

            def compiled_decode_forward(token: torch.Tensor, cache: Any, attention_mask: torch.Tensor) -> Any:
                position = int(cache.get_seq_length())
                if position >= self.thinker_static_cache_max_len:
                    raise ValueError(
                        "decode ASR supererebbe la capacità della cache statica "
                        f"({position} >= {self.thinker_static_cache_max_len})"
                    )
                visible_tokens = position + 1
                if int(attention_mask.shape[-1]) != visible_tokens:
                    raise RuntimeError(
                        "cache e maschera ASR non sono allineate prima del decode "
                        f"(cache={position}, mask={int(attention_mask.shape[-1])})"
                    )
                out = self._compiled_thinker(**self._static_thinker_kwargs(
                    token,
                    cache,
                    start_position=position,
                    visible_tokens=visible_tokens,
                    precompute_attention_masks=True,
                    attention_mask_prefix=attention_mask,
                ))
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    torch.cuda.current_stream().synchronize()
                return out

            transcribe_kwargs.update({
                "past_key_values": static_cache,
                "decode_forward": compiled_decode_forward,
                "static_cache_max_len": self.thinker_static_cache_max_len,
            })
        result = self._thinker_wrapper.transcribe_audio(pcm, **transcribe_kwargs)
        total_ms = (time.perf_counter() - started) * 1000.0
        timings = result.setdefault("phase_timings_ms", {})
        cache_info = result.get("cache_info", {})
        if compile_enabled:
            timings["asr_static_cache_alloc_ms"] = round(cache_alloc_ms, 3)
        if cache_info.get("prefill_sequence_length") is not None:
            timings["asr_cache_prefill_tokens"] = int(cache_info["prefill_sequence_length"])
        if cache_info.get("capacity_tokens") is not None:
            timings["asr_cache_capacity_tokens"] = int(cache_info["capacity_tokens"])
        timings["asr_compiled_q1_decode"] = bool(cache_info.get("compiled_q1_decode", False))
        timings["shared_lm_lock_wait_ms"] = round(lock_wait_ms, 3)
        timings["shared_lm_lock_hold_ms"] = round(lock_hold_ms, 3)
        timings["shared_lm_prefill_lock_wait_ms"] = round(prefill_lock_wait_ms, 3)
        timings["shared_lm_prefill_lock_hold_ms"] = round(prefill_lock_hold_ms, 3)
        timings["total_asr_ms"] = round(total_ms, 3)
        logger.info(
            "[asr_timing] feature_prepare_ms=%.3f audio_prefill_ms=%.3f decode_ms=%.3f "
            "decode_steps=%d lock_wait_ms=%.3f lock_hold_ms=%.3f total_ms=%.3f compiled_q1=%s cache_tokens=%s",
            float(timings.get("feature_prepare_ms", 0.0)),
            float(timings.get("audio_prefill_ms", 0.0)),
            float(timings.get("decode_ms", 0.0)),
            int(timings.get("decode_steps", 0)),
            lock_wait_ms,
            lock_hold_ms,
            total_ms,
            bool(cache_info.get("compiled_q1_decode", False)),
            cache_info.get("prefill_sequence_length", "n/a"),
        )
        return result

    # ------------------------------------------------------------------
    # V9 → h_pfx (16×2048) — h_t per la turnhead = media di h_pfx
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def controller_say_v9_input(
        self,
        live_hidden: torch.Tensor,
        live_prompt_len: int,
        say_text: str,
        *,
        stop_check: Callable[[], bool] | None = None,
    ) -> tuple[torch.Tensor, int, dict[str, Any]] | None:
        """Select live SAY hiddens or an isolated text-only replay for V9.

        ``v9_text_replay`` is an opt-in diagnostic conditioning path. It keeps
        the controller's generated text unchanged and teacher-forces only that
        text after the short text prompt used by the fixed-text ablation. Its
        KV is never reused: ``use_cache=False`` keeps this forward private.
        """
        source = getattr(self, "v9_conditioning_source", "live_generated")
        if source == "live_generated":
            return live_hidden, int(live_prompt_len), {"source": source}
        if source != "v9_text_replay":
            raise RuntimeError(f"v9_conditioning_source non supportata: {source!r}")
        if not isinstance(say_text, str) or not say_text.strip():
            raise ValueError("v9_text_replay richiede un SAY non vuoto")

        tokenize_started = time.perf_counter()
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": _V9_TEXT_REPLAY_PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(chat, return_tensors="pt")["input_ids"].to(self.device)
        target_ids = self.tokenizer(
            say_text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.device)
        if (
            prompt_ids.ndim != 2
            or target_ids.ndim != 2
            or prompt_ids.shape[0] != 1
            or target_ids.shape[0] != 1
            or prompt_ids.shape[1] == 0
            or target_ids.shape[1] == 0
        ):
            raise RuntimeError("v9_text_replay_ids_invalid")
        input_ids = torch.cat((prompt_ids, target_ids), dim=1)
        attention_mask = torch.ones_like(input_ids)
        prompt_len = int(prompt_ids.shape[1])
        target_len = int(target_ids.shape[1])
        tokenize_ms = (time.perf_counter() - tokenize_started) * 1000.0

        lock_wait_started = time.perf_counter()
        acquired = self._listener_priority_lock(stop_check=stop_check)
        lock_acquired = time.perf_counter()
        if not acquired:
            return None
        try:
            # Official Gemma 4 docs define per-layer hidden-state sequence
            # outputs and use_cache as the reusable KV switch:
            # https://huggingface.co/docs/transformers/model_doc/gemma4
            result = self.thinker(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
                logits_to_keep=1,
            )
        finally:
            lock_released = time.perf_counter()
            self._thinker_lock.release()

        if stop_check is not None and stop_check():
            return None
        hidden_states = getattr(result, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("v9_text_replay_hidden_states_missing")
        hidden = hidden_states[-1]
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 3
            or hidden.shape[0] != 1
            or int(hidden.shape[1]) != int(input_ids.shape[1])
        ):
            raise RuntimeError("v9_text_replay_hidden_shape_mismatch")
        hidden_target_len = int(hidden.shape[1]) - prompt_len
        if hidden_target_len != target_len:
            raise RuntimeError(
                "v9_text_replay_target_alignment_mismatch: "
                f"hidden={hidden_target_len} target={target_len}"
            )
        metrics = {
            "source": source,
            "prompt_tokens": prompt_len,
            "target_tokens": target_len,
            "hidden_target_tokens": hidden_target_len,
            "tokenize_ms": tokenize_ms,
            "lock_wait_ms": (lock_acquired - lock_wait_started) * 1000.0,
            "lock_hold_ms": (lock_released - lock_acquired) * 1000.0,
            "prefill_ms": (lock_released - lock_acquired) * 1000.0,
            "kv_reused": False,
        }
        return hidden.detach(), prompt_len, metrics

    @torch.inference_mode()
    def v9_prefix(self, hidden: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Return the profile-selected prefix for legacy callers.

        Canonical/runtime engines use the causal-L0 stateful projector.  The
        batch quality profile remains available only when explicitly selected
        as the diagnostic oracle.
        """
        if self.v9_profile == V9_PROFILE_CAUSAL_L0:
            return self.v9_prefix_stateful(hidden, prompt_len).h_pfx
        if self.v9_profile != V9_PROFILE_QUALITY_FINAL_ORACLE:
            raise RuntimeError(f"V9_PROFILE_RECIPE_MISMATCH: {self.v9_profile!r}")
        if self.v9_recipe != V9_QUALITY_RECIPE:
            raise RuntimeError(f"V9_PROFILE_RECIPE_MISMATCH: {self.v9_recipe!r}")
        lh = hidden  # (1, L, 1536)
        gen = lh[:, prompt_len:, :]
        h_tx = gen.mean(dim=1, keepdim=True).to(self.dtype)
        T_pfx = 16
        seq_trans = gen.transpose(1, 2)
        h_raw_style = F.interpolate(seq_trans, size=T_pfx, mode="linear", align_corners=False).transpose(1, 2)
        cc = self.causal_conv(h_raw_style.to(self.dtype))
        out = self.v9(cc, h_tx)  # (1, 16, 2048)
        if not torch.isfinite(out).all():
            raise RuntimeError("V9 ha prodotto h_pfx non finito")
        return out

    @torch.inference_mode()
    def v9_prefix_stateful(
        self, hidden: torch.Tensor, prompt_len: int, *,
        listener_admission_wait: Callable[[], Any] | None = None,
        listener_priority: bool = False,
        stop_check: Callable[[], bool] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> V9PrefixSequence:
        """Project Gemma final-layer (pre-conv) states one frame at a time.

        ``hidden`` is the final hidden sequence returned by Gemma; no causal
        convolution has been applied yet.  The first generated hidden is the
        fixed causal anchor and every subsequent frame goes through exactly
        one ``IncrementalV9Projector.step`` call.
        """
        if self.v9_profile != V9_PROFILE_CAUSAL_L0:
            raise RuntimeError(
                "V9_PROFILE_RECIPE_MISMATCH: v9_prefix_stateful() è riservato a causal_l0"
            )
        if self.v9_recipe != V9_CAUSAL_L0_RECIPE:
            raise RuntimeError(f"V9_PROFILE_RECIPE_MISMATCH: {self.v9_recipe!r}")
        if self.v9_stream is None:
            raise RuntimeError("V9 incrementale non caricato")
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != self.v9_stream.raw_dim:
            raise ValueError(
                f"hidden Gemma deve avere shape (1,T,{self.v9_stream.raw_dim}), got {tuple(hidden.shape)}"
            )
        start = int(prompt_len)
        if start < 0 or start >= hidden.shape[1]:
            raise ValueError("prompt_len non lascia hidden Gemma generati")
        raw = hidden[:, start:, :]
        anchor = raw[:, :1, :].squeeze(1).to(device=self.v9_stream.projector.audio_in.weight.device,
                                             dtype=self.v9_stream.projector.audio_in.weight.dtype)
        state = self.v9_stream.new_state(
            batch_size=1,
            device=self.v9_stream.projector.audio_in.weight.device,
            dtype=self.v9_stream.projector.audio_in.weight.dtype,
            max_seq_len=raw.shape[1],
        )
        projected = []
        for index in range(raw.shape[1]):
            if listener_priority:
                self._wait_for_listener_admission(
                    listener_admission_wait, metrics, "v9_prefix"
                )
                if not self._listener_priority_lock(stop_check=stop_check):
                    raise GenerationAdmissionAborted(
                        "response cancelled before V9 projection stage"
                    )
                try:
                    if stop_check is not None and stop_check():
                        raise GenerationAdmissionAborted(
                            "response cancelled before V9 projector step"
                        )
                    frame = self.v9_stream.step(raw[:, index, :], anchor, state)
                finally:
                    self._thinker_lock.release()
            else:
                frame = self.v9_stream.step(raw[:, index, :], anchor, state)
            projected.append(frame.unsqueeze(1))
        h_pfx = torch.cat(projected, dim=1)
        if not torch.isfinite(h_pfx).all():
            raise RuntimeError("V9 runtime ha prodotto h_pfx non finito")
        return V9PrefixSequence(h_pfx=h_pfx.detach(), raw_stage="pre_causal_conv", anchor=anchor.detach())

    @torch.inference_mode()
    def v9_prefix_audio_context(
        self,
        audio_context_hidden: torch.Tensor,
        semantic_anchor_hidden: torch.Tensor,
    ) -> V9PrefixSequence:
        """Project real input-audio states with a separate generated-SAY anchor.

        This opt-in experiment keeps the V9 causal raw stream tied to actual
        Gemma audio-placeholder positions.  The text anchor is computed from
        the already-validated post-token hidden states of the response's SAY
        field.  No second Thinker pass or prompt/history tokens are introduced.
        """
        if self.v9_profile != V9_PROFILE_CAUSAL_L0:
            raise RuntimeError("V9_PROFILE_RECIPE_MISMATCH: dual audio-context requires causal_l0")
        if self.v9_recipe != V9_CAUSAL_L0_RECIPE:
            raise RuntimeError(f"V9_PROFILE_RECIPE_MISMATCH: {self.v9_recipe!r}")
        if self.v9_stream is None:
            raise RuntimeError("V9 incrementale non caricato")
        if (
            not isinstance(audio_context_hidden, torch.Tensor)
            or audio_context_hidden.ndim != 3
            or audio_context_hidden.shape[0] != 1
            or audio_context_hidden.shape[-1] != self.v9_stream.raw_dim
            or audio_context_hidden.shape[1] <= 0
        ):
            raise ValueError(
                "audio_context_hidden deve avere shape (1,T,1536) con T>0"
            )
        if not isinstance(semantic_anchor_hidden, torch.Tensor):
            raise TypeError("semantic_anchor_hidden deve essere un Tensor")
        if semantic_anchor_hidden.ndim == 3 and semantic_anchor_hidden.shape[1] == 1:
            semantic_anchor_hidden = semantic_anchor_hidden[:, 0, :]
        if (
            semantic_anchor_hidden.ndim != 2
            or tuple(semantic_anchor_hidden.shape) != (1, self.v9_stream.raw_dim)
        ):
            raise ValueError("semantic_anchor_hidden deve avere shape (1,1536) o (1,1,1536)")
        if not bool(torch.isfinite(audio_context_hidden).all()) or not bool(
            torch.isfinite(semantic_anchor_hidden).all()
        ):
            raise ValueError("dual V9 input contiene valori non finiti")

        projector_weight = self.v9_stream.projector.audio_in.weight
        raw = audio_context_hidden.to(device=projector_weight.device, dtype=projector_weight.dtype)
        anchor = semantic_anchor_hidden.to(device=projector_weight.device, dtype=projector_weight.dtype)
        state = self.v9_stream.new_state(
            batch_size=1,
            device=projector_weight.device,
            dtype=projector_weight.dtype,
            max_seq_len=int(raw.shape[1]),
        )
        projected = [
            self.v9_stream.step(raw[:, index, :], anchor, state).unsqueeze(1)
            for index in range(int(raw.shape[1]))
        ]
        h_pfx = torch.cat(projected, dim=1)
        if not bool(torch.isfinite(h_pfx).all()):
            raise RuntimeError("V9 dual audio-context ha prodotto h_pfx non finito")
        return V9PrefixSequence(
            h_pfx=h_pfx.detach(),
            raw_stage="pre_causal_conv",
            anchor=anchor.detach(),
        )

    # ------------------------------------------------------------------
    # Talker
    # ------------------------------------------------------------------
    def _codec_eos_token_id(self) -> int:
        """Return the Talker's authoritative codec EOS token."""
        if self.talker is None:
            raise RuntimeError("talker non caricato")

        value = getattr(self.talker.tcfg, "codec_eos_token_id", None)
        if value is None:
            value = getattr(self.talker.tcfg, "codec_eos_id", None)
        if value is None:
            value = TALKER_EOS_TOKEN_ID

        if isinstance(value, bool) or not isinstance(value, Integral):
            raise RuntimeError(f"codec EOS non intero: {value!r}")
        eos = int(value)
        vocab = int(self.talker.talker.codec_head.out_features)
        if eos < 2048 or eos >= vocab:
            raise RuntimeError(
                f"codec EOS fuori vocab o nel range audio: eos={eos}, vocab={vocab}"
            )
        return eos

    def _codec_eos_allowed(self, generated_cb0_frames: int) -> bool:
        """Match Qwen's min_new_tokens=2 for the acoustic CB0 sequence."""
        return bool(self.allow_talker_eos) and int(generated_cb0_frames) >= QWEN_TALKER_MIN_NEW_TOKENS

    def _mask_cb0_logits(
        self,
        logits: torch.Tensor,
        *,
        allow_eos: bool = True,
    ) -> torch.Tensor:
        """Apply the CB0 vocabulary mask without changing sampling parameters."""
        if logits.ndim == 1:
            lg = logits.clone()
        elif logits.ndim == 2:
            if logits.shape[0] != 1:
                raise ValueError(f"CB0 batch deve essere 1, got {tuple(logits.shape)}")
            lg = logits[0].clone()
        elif logits.ndim == 3:
            if logits.shape[0] != 1 or logits.shape[1] < 1:
                raise ValueError(f"CB0 shape inattesa: {tuple(logits.shape)}")
            lg = logits[0, -1].clone()
        else:
            raise ValueError(f"CB0 logits devono essere 1-D, 2-D o 3-D, got {logits.ndim}-D")

        if not torch.is_floating_point(lg):
            raise ValueError("CB0 logits devono essere floating point")
        if lg.numel() <= 2048:
            raise ValueError(f"CB0 vocab inatteso: {lg.numel()} <= 2048")
        if not bool(torch.isfinite(lg).all()):
            raise RuntimeError("CB0 logits non finiti")

        eos = self._codec_eos_token_id() if allow_eos else None
        eos_logit = None if eos is None else lg[eos].clone()
        lg[2048:] = -float("inf")
        if eos is not None:
            assert eos_logit is not None
            lg[eos] = eos_logit
        return lg

    def _cb0_sample(
        self,
        logits: torch.Tensor,
        temperature: float = 0.6,
        *,
        allow_eos: bool = True,
    ) -> int:
        lg = self._mask_cb0_logits(logits, allow_eos=allow_eos)
        topk_vals, topk_idx = torch.topk(lg, k=min(20, lg.shape[-1]), dim=-1)
        filtered = torch.full_like(lg, -float("inf"))
        filtered.scatter_(-1, topk_idx, topk_vals)
        probs = F.softmax(filtered / temperature, dim=-1)
        if not bool(torch.isfinite(probs).all()):
            raise RuntimeError("probabilità CB0 non finite")
        if float(probs.sum().item()) <= 0.0:
            raise RuntimeError("distribuzione CB0 vuota")
        return int(torch.multinomial(probs, 1).item())

    @torch.no_grad()
    def stop_generation(self) -> None:
        """Richiede l'arresto al prossimo frame; sicuro da chiamare dal WS."""
        self._stop_event = True

    @_private_talker_slot_stream
    @torch.inference_mode()
    def talk_stateful(
        self,
        text: str,
        h_pfx: torch.Tensor | None = None,
        *,
        n_words: int,
        voice_conditioning: dict[str, Any] | None = None,
        stop_check: Any | None = None,
        state: Any | None = None,
        on_audio_chunk: Any | None = None,
        on_audio_chunk_with_horizon: Any | None = None,
        on_cb0_frame: Any | None = None,
        min_frames: int = ARIA_MIN_FRAMES,
        metrics: dict[str, Any] | None = None,
        generation_id: int | None = None,
        lookahead_frames: int = FROZEN_QWEN_LOOKAHEAD_FRAMES,
        conditioning_buffer: Any | None = None,
        trailing_text_hidden: torch.Tensor | None = None,
        response_text_conditioning: torch.Tensor | None = None,
        talk_context: Any | None = None,
        max_frames: int | None = None,
        listener_admission_wait: Callable[[], Any] | None = None,
        quality_diagnostic: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Generate speech from a frame-stateful V9 prefix.

        Real Qwen uses ``FuturePrefixQwen``'s position-addressable slot KV:
        stable ICL/text, each conditioning slot, and each audio query are
        committed once.  The 4-D mask still enforces the bounded lookahead;
        deterministic fake backends retain the full-sequence fallback.
        """
        talker_started = time.perf_counter()
        talker_lock_acquired = talker_started
        if metrics is not None:
            metrics.update({
                "eos_detected": False,
                "eos_frame_id": None,
                "generated_cb0_frames": 0,
                "generated_pcm_samples": 0,
                "terminated_by_budget": False,
                "terminated_by": None,
                "cancelled": False,
                "effective_min_frames": int(min_frames),
                "effective_max_frames": None,
                "terminated_by_segment_capacity": False,
                "intermediate_segment_eos": [],
                "effective_min_frames": int(min_frames),
            })
        if self.talker is None or self.subtalker_graph is None:
            raise RuntimeError("talker e SubTalker non caricati")
        if talk_context is not None:
            if self.runtime_backend != "native":
                raise RuntimeError(
                    "controller-guided response pool requires FuturePrefixQwen native"
                )
            if conditioning_buffer is not None:
                raise ValueError(
                    "controller-guided response pool requires the generated SAY V9 prefix, "
                    "not the listener conditioning buffer"
                )
            if not talk_context.is_current():
                raise GenerationAdmissionAborted("response talk lease is stale before Talker start")
        active_subtalker_graph = (
            talk_context.subtalker_graph
            if talk_context is not None
            else self.subtalker_graph
        )
        if metrics is not None:
            metrics["qwen_audio_conditioning_recipe"] = getattr(
                self,
                "qwen_audio_recipe",
                QWEN_AUDIO_CONDITIONING_RECIPE,
            )
        if lookahead_frames < 0 or lookahead_frames >= 64:
            raise ValueError("lookahead_frames deve essere compreso tra 0 e 63")
        if conditioning_buffer is None:
            if h_pfx is None:
                raise ValueError("h_pfx è obbligatorio senza conditioning_buffer")
            if h_pfx.ndim == 2:
                h_pfx = h_pfx.unsqueeze(0)
            if h_pfx.ndim != 3 or h_pfx.shape[0] != 1 or h_pfx.shape[-1] != 2048:
                raise ValueError(f"h_pfx deve avere shape (1,T,2048), got {tuple(h_pfx.shape)}")
            if h_pfx.shape[1] == 0:
                raise ValueError("h_pfx non può essere vuoto")
        elif not callable(getattr(conditioning_buffer, "wait_for_through", None)):
            raise TypeError("conditioning_buffer non espone wait_for_through")
        if response_text_conditioning is not None:
            if conditioning_buffer is None:
                raise ValueError("response_text_conditioning richiede il conditioning_buffer realtime")
            if h_pfx is not None:
                raise ValueError("response_text_conditioning e h_pfx non possono essere usati insieme")
            if response_text_conditioning.ndim == 2:
                response_text_conditioning = response_text_conditioning.unsqueeze(0)
            if (
                response_text_conditioning.ndim != 3
                or response_text_conditioning.shape[0] != 1
                or response_text_conditioning.shape[1] <= 0
                or response_text_conditioning.shape[-1] != 2048
            ):
                raise ValueError(
                    "response_text_conditioning deve avere shape (1,T,2048), "
                    f"got {tuple(response_text_conditioning.shape)}"
                )
            if not bool(torch.isfinite(response_text_conditioning).all()):
                raise ValueError("response_text_conditioning contiene valori non finiti")

        short_streaming = (
            (on_audio_chunk is not None or on_audio_chunk_with_horizon is not None)
            and min_frames < ARIA_MIN_FRAMES
        )
        spoken_units, numeric_extra_units, T_gen = _estimate_talker_frame_budget(
            text,
            n_words,
            min_frames,
            short_streaming=short_streaming,
        )
        requested_T_gen = T_gen
        if max_frames is not None:
            if int(max_frames) <= 0:
                raise ValueError("max_frames deve essere positivo")
            T_gen = min(T_gen, int(max_frames))
        if conditioning_buffer is None:
            if h_pfx is None or h_pfx.shape[1] < 1:
                raise ValueError("Qwen richiede almeno un frame h_pfx")
            # In a pooled stateful response, each real SAY prefix slot is
            # inserted once, then the native Q1 audio suffix continues with
            # the same KV. Legacy stateless generation remains prefix-bounded.
            if talk_context is None:
                T_gen = min(T_gen, int(h_pfx.shape[1]))
        segment_capacity = (
            max(1, int(T_gen))
            if talk_context is not None
            else DEFAULT_WINDOW_FRAMES
        )
        max_prefix_capacity = (
            int(h_pfx.shape[1])
            if talk_context is not None and h_pfx is not None
            else segment_capacity
        )
        if metrics is not None:
            metrics.update({
                "spoken_text_units": spoken_units,
                "numeric_extra_spoken_units": numeric_extra_units,
                "requested_acoustic_frame_budget": requested_T_gen,
                "acoustic_frame_budget": T_gen,
                "response_text_conditioning_source": (
                    "audio_response_protocol_v2.say_hidden_v9"
                    if response_text_conditioning is not None else None
                ),
                "effective_max_frames": int(T_gen),
                "future_prefix_max_prefix_frames": int(max_prefix_capacity),
                "future_prefix_max_audio_frames": int(segment_capacity),
                "future_prefix_prefix_frames_provided": (
                    int(h_pfx.shape[1]) if h_pfx is not None else 0
                ),
                "talk_context_slot_id": (
                    int(talk_context.slot_id) if talk_context is not None else None
                ),
                "talk_context_response_id": (
                    talk_context.response_id if talk_context is not None else None
                ),
            })
        if T_gen < QWEN_TALKER_MIN_NEW_TOKENS:
            if metrics is not None:
                metrics["no_speech_reason"] = "acoustic_budget_below_qwen_min_new_tokens"
            return np.zeros(0, dtype=np.float32)
        if T_gen <= 0:
            raise ValueError("acoustic frame budget vuoto")
        segment_texts = (text,)
        if conditioning_buffer is not None:
            segment_texts = _split_text_for_frame_segments(
                text,
                total_frames=T_gen,
                segment_frames=segment_capacity,
            )
            if len(segment_texts) > 1 and trailing_text_hidden is not None:
                raise ValueError(
                    "continuazione Qwen con trailing_text_hidden richiede "
                    "una suddivisione hidden/testo esplicita"
                )
        if self.runtime_backend == "native" and conditioning_buffer is None and h_pfx is None:
            raise ValueError("Qwen native richiede un h_pfx per ogni frame generato")

        talker_lock_wait_started = time.perf_counter()
        talker_lock_scope = self._talker_lock if talk_context is None else nullcontext()
        with talker_lock_scope:
            talker_lock_acquired = time.perf_counter()
            if metrics is not None:
                metrics["talker_lock_wait_ms"] = (
                    (talker_lock_acquired - talker_lock_wait_started) * 1000.0
                    if talk_context is None else 0.0
                )
            # Response cancellation is lease-local in pooled mode. Reset only
            # the process-wide shutdown flag when a new response actually begins.
            self._stop_event = False
            if state is not None and generation_id is not None and not state.is_generation_current(generation_id=generation_id):
                state.stale_kv_writes += 1
                if metrics is not None:
                    metrics["talker_lock_hold_ms"] = (time.perf_counter() - talker_lock_acquired) * 1000.0
                    metrics["talker_total_ms"] = (time.perf_counter() - talker_started) * 1000.0
                return np.zeros(0, dtype=np.float32)

            effective_voice = (
                _resolve_talker_voice_conditioning(
                    self.talker,
                    voice_conditioning,
                )
            )

            if metrics is not None:
                metrics.update({
                    "voice_conditioning_mode": (
                        None
                        if effective_voice is None
                        else effective_voice.get(
                            "mode"
                        )
                    ),
                    "speaker_reference_sha256": (
                        None
                        if effective_voice is None
                        else effective_voice.get(
                            "reference_wav_sha256"
                        )
                    ),
                    "speaker_embedding_present": (
                        effective_voice is not None
                        and effective_voice.get(
                            "speaker_embedding"
                        ) is not None
                    ),
                })

            had_voice = hasattr(self.talker, "voice_conditioning")
            previous_voice = getattr(self.talker, "voice_conditioning", None)
            if effective_voice is not None and talk_context is None:
                self.talker.voice_conditioning = effective_voice

            def cancelled() -> bool:
                if talk_context is None and self._stop_event:
                    return True
                if talk_context is not None and (
                    not talk_context.is_current() or talk_context.is_cancelled()
                ):
                    return True
                if stop_check is not None and stop_check():
                    return True
                return state is not None and generation_id is not None and not state.is_generation_current(
                    generation_id=generation_id
                )

            def run_gpu_stage(site: str, forward: Callable[[], Any]) -> Any:
                if talk_context is None:
                    return forward()
                self._wait_for_listener_admission(
                    listener_admission_wait, metrics, "talker"
                )
                wait_started = time.perf_counter()
                if not self._listener_priority_lock(stop_check=cancelled):
                    raise GenerationAdmissionAborted(
                        f"response cancelled before FuturePrefix GPU stage {site}"
                    )
                acquired = time.perf_counter()
                if metrics is not None:
                    metrics[f"listener_priority_{site}_lock_wait_ms"] = (
                        metrics.get(f"listener_priority_{site}_lock_wait_ms", 0.0)
                        + (acquired - wait_started) * 1000.0
                    )
                try:
                    if cancelled():
                        raise GenerationAdmissionAborted(
                            f"response cancelled before FuturePrefix GPU stage {site}"
                        )
                    return forward()
                finally:
                    held_ms = (time.perf_counter() - acquired) * 1000.0
                    if metrics is not None:
                        metrics[f"listener_priority_{site}_lock_hold_ms"] = (
                            metrics.get(f"listener_priority_{site}_lock_hold_ms", 0.0)
                            + held_ms
                        )
                    self._thinker_lock.release()
            speaker: FuturePrefixQwen | None = None
            eos_detected = False
            eos_frame_id: int | None = None
            eos_query_logits: list[tuple[str, int, torch.Tensor]] = []
            try:
                if self.runtime_backend == "native" and lookahead_frames != 0:
                    raise ValueError("Qwen native talk_stateful richiede lookahead=0")
                speaker = FuturePrefixQwen(
                    self.talker,
                    lookahead_frames=lookahead_frames,
                    max_prefix_frames=max_prefix_capacity,
                    device=self.device,
                    dtype=next(self.talker.talker.model.parameters()).dtype,
                    max_audio_frames=segment_capacity,
                    terminal_query_budget=TERMINAL_QUERY_BUDGET,
                    runtime_backend=self.runtime_backend,
                    virtual_prefix_mode=getattr(self, "virtual_prefix_mode", "legacy"),
                    prefix_key_log_gate=float(
                        getattr(self, "response_intent_prefix_key_log_gate", -8.0)
                    ),
                )
                begin_kwargs: dict[str, Any] = {
                    "voice_conditioning": effective_voice,
                }
                if response_text_conditioning is not None:
                    begin_kwargs["response_text_conditioning"] = response_text_conditioning
                if trailing_text_hidden is not None:
                    begin_kwargs["trailing_text_hidden"] = trailing_text_hidden
                if talk_context is not None:
                    talk_context.future_prefix_qwen = speaker
                prefix_started = time.perf_counter()
                run_gpu_stage(
                    "qwen_prefill",
                    lambda: speaker.begin(segment_texts[0], **begin_kwargs),
                )
                prefix_runtime = speaker.runtime_info()
                if talk_context is not None and (
                    prefix_runtime.get("runtime_backend") != "native"
                    or prefix_runtime.get("native_enabled") is not True
                ):
                    raise RuntimeError(
                        "controller-guided response pool did not activate native FuturePrefixQwen"
                    )
                if metrics is not None:
                    metrics.update({
                        "future_prefix_runtime_backend": prefix_runtime.get("runtime_backend"),
                        "virtual_prefix_mode": prefix_runtime.get("virtual_prefix_mode", "legacy"),
                        "prefix_key_log_gate": prefix_runtime.get("prefix_key_log_gate", -8.0),
                        "future_prefix_native_enabled": bool(prefix_runtime.get("native_enabled")),
                        "future_prefix_q1_backend": getattr(self, "talker_q1_backend", "eager"),
                        "future_prefix_prefix_frames_provided": (
                            int(h_pfx.shape[1]) if h_pfx is not None else 0
                        ),
                        "future_prefix_max_prefix_frames": int(speaker.max_prefix_frames),
                        "future_prefix_max_audio_frames": int(speaker.max_audio_frames),
                    })
                if metrics is not None:
                    metrics["talker_prefix_ms"] = metrics.get("talker_prefix_ms", 0.0) + (
                        time.perf_counter() - prefix_started
                    ) * 1000.0
                segment_offset = 0
                segment_rollovers = 0
                terminal_queries = 0
                compat_static_pushed = False
                eos_token_id = self._codec_eos_token_id() if self.allow_talker_eos else None

                def record_eos_query(kind: str, frame_id: int, logits: torch.Tensor) -> None:
                    if metrics is None or eos_token_id is None:
                        return
                    if logits.ndim == 3:
                        row = logits[0, -1]
                    elif logits.ndim == 2:
                        row = logits[0]
                    else:
                        row = logits
                    # Keep a snapshot because compiled backends may reuse their
                    # output storage on the next frame. No host sync or forward
                    # is introduced in the generation loop.
                    eos_query_logits.append((kind, int(frame_id), row.detach().clone()))

                def publish_eos_diagnostics() -> None:
                    if not eos_query_logits or eos_token_id is None or metrics is None:
                        return
                    raw = torch.stack([row for _, _, row in eos_query_logits]).float()
                    masked = torch.full_like(raw, -float("inf"))
                    masked[:, :2048] = raw[:, :2048]
                    masked[:, eos_token_id] = raw[:, eos_token_id]
                    eos_logits = masked[:, eos_token_id]
                    log_probability = eos_logits - torch.logsumexp(masked, dim=-1)
                    # Transfer one compact record table after all inference,
                    # preserving device execution and sampling behavior.
                    table = torch.stack((
                        eos_logits,
                        (masked > eos_logits.unsqueeze(-1)).sum(dim=-1).float() + 1.0,
                        log_probability,
                        log_probability.exp(),
                        masked.argmax(dim=-1).float(),
                    ), dim=-1).cpu().tolist()
                    metrics["cb0_eos_diagnostics"] = {
                        "codec_eos_token_id": int(eos_token_id),
                        "support_size": 2049,
                        "probability_definition": "raw_untempered_after_cb0_mask_before_topk",
                        "queries": [
                            {
                                "kind": kind,
                                "frame_id": frame_id,
                                "eos_logit": float(values[0]),
                                "eos_rank": int(values[1]),
                                "eos_log_probability": float(values[2]),
                                "eos_probability": float(values[3]),
                                "argmax_id": int(values[4]),
                            }
                            for (kind, frame_id, _), values in zip(eos_query_logits, table)
                        ],
                    }

                if metrics is not None:
                    metrics.update({
                        "requested_acoustic_frame_budget": requested_T_gen,
                        "prefix_capacity": segment_capacity,
                        "acoustic_frame_budget": T_gen,
                        "terminal_query_budget": TERMINAL_QUERY_BUDGET,
                        "segment_rollovers": 0,
                        "text_segments": len(segment_texts),
                        "terminal_queries": 0,
                    })

                def sync_conditioning(required_frame_id: int) -> bool:
                    nonlocal compat_static_pushed
                    required_local = int(required_frame_id)
                    if required_local < 0:
                        return False
                    required_local = min(speaker.max_prefix_frames - 1, required_local)
                    speaker_buffer = getattr(speaker, "buffer", None)
                    if speaker_buffer is None:
                        if conditioning_buffer is not None:
                            return False
                        if not compat_static_pushed and h_pfx is not None:
                            stop = min(segment_offset + segment_capacity, int(h_pfx.shape[1]))
                            for local_id in range(stop - segment_offset):
                                speaker.push_v9_frame(local_id, h_pfx[:, segment_offset + local_id, :])
                            compat_static_pushed = True
                        return True
                    already_pushed = speaker_buffer.buffered_frames
                    if conditioning_buffer is None:
                        assert h_pfx is not None
                        start = segment_offset + already_pushed
                        stop = min(segment_offset + speaker.max_prefix_frames, int(h_pfx.shape[1]))
                        for local_id in range(already_pushed, stop - segment_offset):
                            speaker.push_v9_frame(local_id, h_pfx[:, segment_offset + local_id, :])
                        return speaker.buffer.available_through >= required_local

                    required_global = segment_offset + required_local
                    frames = conditioning_buffer.wait_for_through(
                        required_global,
                        stop_check=cancelled,
                        timeout_s=FROZEN_QWEN_CONDITIONING_WAIT_TIMEOUT_S,
                    )
                    if not frames:
                        return False
                    for relative_id, frame in enumerate(frames):
                        if relative_id < segment_offset + already_pushed:
                            continue
                        local_id = relative_id - segment_offset
                        if local_id >= speaker.max_prefix_frames:
                            break
                        speaker.push_v9_frame(local_id, frame.h_pfx)
                    return speaker.buffer.available_through >= required_local

                codes: list[list[int]] = []
                streamed_audio: list[np.ndarray] = []
                preclamp_audio: list[np.ndarray] = []
                capture_preclamp = bool(
                    isinstance(quality_diagnostic, dict)
                    and os.environ.get("DUPLEX_CAPTURE_CODE2WAV_PRECLAMP", "0") == "1"
                )

                def record_preclamp(value: torch.Tensor) -> None:
                    array = value.detach().to(dtype=torch.float32).cpu().numpy()
                    preclamp_audio.append(np.ascontiguousarray(array.reshape(-1)))

                def preclamp_summary() -> dict[str, Any]:
                    values = np.concatenate(preclamp_audio) if preclamp_audio else np.empty(0, dtype=np.float32)
                    finite = values[np.isfinite(values)]
                    return {
                        "samples": int(values.size),
                        "nonfinite_count": int(values.size - finite.size),
                        "finite_peak_abs": float(np.max(np.abs(finite))) if finite.size else None,
                        "finite_min": float(np.min(finite)) if finite.size else None,
                        "finite_max": float(np.max(finite)) if finite.size else None,
                        "finite_outside_unit_count": int(np.count_nonzero(np.abs(finite) > 1.0)),
                    }

                def capture_preclamp_failure(chunk_codes: list[list[int]], exc: BaseException) -> None:
                    if not capture_preclamp or not preclamp_audio:
                        return
                    capture = quality_diagnostic.get("capture")
                    if not callable(capture):
                        return
                    metadata = dict(quality_diagnostic.get("metadata") or {})
                    metadata["code2wav_preclamp"] = preclamp_summary()
                    metadata["decode_failure"] = type(exc).__name__
                    try:
                        capture({
                            "metadata": metadata,
                            "say_hidden": quality_diagnostic.get("say_hidden"),
                            "v9_prefix": h_pfx,
                            "rvq_codes": np.asarray(chunk_codes, dtype=np.int32),
                            "preclamp_pcm": np.concatenate(preclamp_audio),
                        })
                    except Exception as capture_error:
                        logger.warning(
                            "[code2wav pre-clamp diagnostic] failure capture failed: %s",
                            str(capture_error)[:180],
                        )

                streaming_requested = on_audio_chunk is not None or on_audio_chunk_with_horizon is not None

                def decode_chunk(chunk_codes: list[list[int]]) -> np.ndarray:
                    tensor = torch.tensor(
                        chunk_codes,
                        device=self.device,
                        dtype=torch.long,
                    ).unsqueeze(0)
                    decode_started = time.perf_counter()
                    try:
                        code2wav_graph = (
                            talk_context.code2wav_graph if talk_context is not None else None
                        )
                        if (
                            code2wav_graph is not None
                            and not capture_preclamp
                            and not getattr(self, "_warmup_diagnostic_active", False)
                            and _code2wav_graph_supports_shape(code2wav_graph, tensor)
                        ):
                            decoded = run_gpu_stage(
                                "code2wav_decode", lambda: code2wav_graph.decode(tensor)
                            )
                            if metrics is not None:
                                _record_code2wav_graph_use(
                                    metrics, code2wav_graph, tensor
                                )
                        else:
                            def _har_preclamp_callback(val: torch.Tensor) -> torch.Tensor:
                                if capture_preclamp:
                                    record_preclamp(val)
                                return rescale_code2wav_preclamp(val, max_peak=0.95)

                            decoded = run_gpu_stage(
                                "code2wav_decode",
                                lambda: self.talker.decode(
                                    tensor,
                                    preclamp_callback=_har_preclamp_callback,
                                ),
                            )
                            if metrics is not None:
                                metrics["code2wav_backend"] = "qwen_chunked_decode"
                                metrics["code2wav_eager_calls"] = (
                                    metrics.get("code2wav_eager_calls", 0) + 1
                                )
                    except BaseException as exc:
                        capture_preclamp_failure(chunk_codes, exc)
                        raise
                    if metrics is not None:
                        metrics["code2wav_decode_ms"] = metrics.get("code2wav_decode_ms", 0.0) + (
                            time.perf_counter() - decode_started
                        ) * 1000.0
                        metrics["code2wav_decode_calls"] = metrics.get("code2wav_decode_calls", 0) + 1
                    if hasattr(decoded, "detach"):
                        decoded = decoded.detach().cpu().numpy()
                    chunk = np.asarray(decoded, dtype=np.float32).reshape(-1)
                    if chunk.size == 0:
                        error = RuntimeError("Code2Wav ha prodotto PCM vuoto")
                        capture_preclamp_failure(chunk_codes, error)
                        raise error
                    if not np.isfinite(chunk).all():
                        error = RuntimeError("Code2Wav ha prodotto audio non finito")
                        capture_preclamp_failure(chunk_codes, error)
                        raise error
                    return chunk

                accumulator = DecodeAccumulator(
                    decode_fn=decode_chunk,
                    chunk_frames=CODE2WAV_DECODE_FRAMES,
                    frame_samples=AUDIO_FRAME_SAMPLES,
                    left_context_frames=CODE2WAV_LEFT_CONTEXT_FRAMES,
                    crossfade_samples=AUDIO_BOUNDARY_CROSSFADE_SAMPLES,
                )

                def emit_decoded_chunk(decoded_chunk: Any) -> None:
                    if talk_context is not None:
                        if cancelled():
                            return
                        talk_context.callback_started()
                    try:
                        if talk_context is not None and cancelled():
                            return
                        streamed_audio.append(decoded_chunk.pcm)
                        if on_audio_chunk_with_horizon is not None:
                            # The horizon belongs to the last NEW codec frame;
                            # context frames are never exposed as output horizon.
                            on_audio_chunk_with_horizon(decoded_chunk.pcm, decoded_chunk.last_frame_id)
                        elif on_audio_chunk is not None:
                            on_audio_chunk(decoded_chunk.pcm)
                    finally:
                        if talk_context is not None and talk_context.is_current():
                            talk_context.callback_finished()

                intermediate_segment_eos: list[int] = []
                segment_capacity_exhausted = False

                def advance_segment() -> bool:
                    nonlocal segment_offset, segment_rollovers, compat_static_pushed
                    next_segment = segment_rollovers + 1
                    if conditioning_buffer is not None and next_segment >= len(segment_texts):
                        return False
                    segment_offset = len(codes)
                    segment_rollovers = next_segment
                    compat_static_pushed = False
                    segment_text = (
                        segment_texts[next_segment]
                        if conditioning_buffer is not None
                        else text
                    )
                    prefix_started = time.perf_counter()
                    speaker.begin(segment_text, **begin_kwargs)
                    if metrics is not None:
                        metrics["talker_prefix_ms"] = metrics.get("talker_prefix_ms", 0.0) + (
                            time.perf_counter() - prefix_started
                        ) * 1000.0
                    if metrics is not None:
                        metrics["segment_rollovers"] = segment_rollovers
                    return True

                while len(codes) < T_gen:
                    if cancelled():
                        break
                    local_frame_id = len(codes) - segment_offset
                    if local_frame_id >= segment_capacity:
                        # Explicit continuation: reset local acoustic/prefix ids,
                        # keep the declared 64-slot layout, and resume with the
                        # next text span rather than replaying the utterance.
                        if not advance_segment():
                            segment_capacity_exhausted = True
                            break
                        local_frame_id = 0
                    required_conditioning = local_frame_id + lookahead_frames
                    if not sync_conditioning(required_conditioning):
                        if cancelled():
                            break
                        required_global = segment_offset + min(
                            speaker.max_prefix_frames - 1,
                            required_conditioning,
                        )
                        raise TimeoutError(
                            "Qwen conditioning frame non disponibile: "
                            f"relative={required_global} entro "
                            f"{FROZEN_QWEN_CONDITIONING_WAIT_TIMEOUT_S:.3f}s"
                        )
                    def run_step_and_sample() -> tuple[Any, int]:
                        step_output = speaker.step_frame(
                            allow_reduced_horizon=conditioning_buffer is None
                        )
                        if not torch.isfinite(step_output.logits).all():
                            raise RuntimeError(
                                f"Talker codec_head non finito al frame {step_output.frame_id}"
                            )
                        record_eos_query("acoustic", len(codes), step_output.logits)
                        sampled_c0 = self._cb0_sample(
                            step_output.logits,
                            allow_eos=self._codec_eos_allowed(len(codes)),
                        )
                        return step_output, sampled_c0

                    step, c0 = run_gpu_stage(
                        "future_prefix_step",
                        run_step_and_sample,
                    )
                    if eos_token_id is not None and c0 == eos_token_id:
                        if conditioning_buffer is not None and advance_segment():
                            intermediate_segment_eos.append(int(len(codes)))
                            if metrics is not None:
                                metrics["intermediate_segment_eos"] = list(intermediate_segment_eos)
                            continue
                        eos_detected = True
                        eos_frame_id = int(len(codes))
                        if metrics is not None:
                            metrics.update({"eos_detected": True, "eos_frame_id": eos_frame_id})
                        break
                    last_h = step.last_hidden
                    def run_subtalker() -> Any:
                        c0_embed = self.talker.talker.get_input_embeddings()(
                            torch.tensor([[c0]], device=self.device)
                        )
                        if metrics is not None:
                            metrics["subtalker_graph_calls"] = metrics.get("subtalker_graph_calls", 0) + 1
                            metrics["subtalker_graph_captured"] = bool(
                                getattr(active_subtalker_graph, "captured", False)
                            )
                            metrics["subtalker_graph_backend"] = (
                                "cuda_graph" if metrics["subtalker_graph_captured"] else "eager"
                            )
                        return active_subtalker_graph.generate(
                            last_h, c0, c0_embed=c0_embed
                        )

                    residual = run_gpu_stage("subtalker", run_subtalker)
                    if isinstance(residual, torch.Tensor):
                        residual = residual.reshape(-1).tolist()
                    residual = [int(value) for value in residual]
                    if len(residual) != 15:
                        raise RuntimeError(f"SubTalker ha prodotto {len(residual)} residui, attesi 15")
                    frame_codes = [c0] + residual
                    global_frame_id = len(codes)
                    speaker.commit_codes(frame_codes)
                    codes.append(frame_codes)
                    if on_cb0_frame is not None and not cancelled():
                        on_cb0_frame(global_frame_id, time.monotonic_ns())
                    if cancelled():
                        accumulator.cancel()
                        break
                    if streaming_requested:
                        for decoded_chunk in accumulator.push(global_frame_id, frame_codes):
                            if not cancelled():
                                emit_decoded_chunk(decoded_chunk)
                            else:
                                accumulator.cancel()
                                break

                # At an acoustic budget boundary, issue one extra terminal query.
                # It may emit EOS but never becomes a 65th acoustic frame.
                budget_reached = len(codes) >= T_gen
                can_terminal_query = callable(getattr(speaker, "can_terminal_query", None)) and speaker.can_terminal_query()
                if budget_reached and not eos_detected and can_terminal_query:
                    local_count = len(codes) - segment_offset
                    if sync_conditioning(max(0, local_count - 1 + lookahead_frames)):
                        def run_terminal_and_sample() -> tuple[Any, int]:
                            terminal_output = speaker.step_terminal()
                            if not torch.isfinite(terminal_output.logits).all():
                                raise RuntimeError("Talker terminal codec_head non finito")
                            record_eos_query("terminal", len(codes), terminal_output.logits)
                            sampled_c0 = self._cb0_sample(
                                terminal_output.logits,
                                allow_eos=self.allow_talker_eos,
                            )
                            return terminal_output, sampled_c0

                        terminal_step, terminal_c0 = run_gpu_stage(
                            "future_prefix_terminal",
                            run_terminal_and_sample,
                        )
                        terminal_queries += 1
                        if metrics is not None:
                            metrics["terminal_queries"] = terminal_queries
                        if eos_token_id is not None and terminal_c0 == eos_token_id:
                            eos_detected = True
                            eos_frame_id = int(len(codes))
                            if metrics is not None:
                                metrics.update({"eos_detected": True, "eos_frame_id": eos_frame_id})

                if cancelled():
                    accumulator.cancel()
                    if metrics is not None:
                        metrics.update({
                            "generated_cb0_frames": len(codes),
                            "generated_pcm_samples": int(sum(audio.size for audio in streamed_audio)),
                            "eos_detected": eos_detected,
                            "eos_frame_id": eos_frame_id,
                            "terminated_by_budget": False,
                            "terminated_by": "cancel",
                            "cancelled": True,
                            "terminated_by_segment_capacity": False,
                        })
                    return np.zeros(0, dtype=np.float32)

                if streaming_requested:
                    for decoded_chunk in accumulator.flush(clean_end=True):
                        if not cancelled():
                            emit_decoded_chunk(decoded_chunk)
                        else:
                            accumulator.cancel()
                            break

                if streaming_requested:
                    wav = np.concatenate(streamed_audio) if streamed_audio else np.zeros(0, dtype=np.float32)
                else:
                    wav = decode_chunk(codes) if codes else np.zeros(0, dtype=np.float32)
                if metrics is not None:
                    metrics.update({
                        "generated_cb0_frames": len(codes),
                        "generated_pcm_samples": int(len(wav)),
                        "eos_detected": eos_detected,
                        "eos_frame_id": eos_frame_id,
                        "terminated_by_budget": (
                            not eos_detected and len(codes) >= T_gen
                        ),
                        "terminated_by": (
                            "eos" if eos_detected
                            else "budget" if len(codes) >= T_gen
                            else "segment_capacity" if segment_capacity_exhausted
                            else "cancel"
                        ),
                        "cancelled": False,
                        "effective_min_frames": int(min_frames),
                        "effective_max_frames": int(T_gen),
                        "terminated_by_segment_capacity": segment_capacity_exhausted,
                        "intermediate_segment_eos": list(intermediate_segment_eos),
                        "terminal_queries": terminal_queries,
                        "segment_rollovers": segment_rollovers,
                    })
                return self._postprocess(wav) if wav.size else wav
            except GenerationAdmissionAborted:
                if metrics is not None:
                    metrics.update({
                        "cancelled": True,
                        "terminated_by": "cancel",
                        "terminated_by_budget": False,
                        "generated_cb0_frames": len(codes) if "codes" in locals() else 0,
                        "generated_pcm_samples": int(sum(audio.size for audio in streamed_audio))
                        if "streamed_audio" in locals() else 0,
                    })
                return np.zeros(0, dtype=np.float32)
            finally:
                publish_diagnostics = locals().get("publish_eos_diagnostics")
                if callable(publish_diagnostics):
                    publish_diagnostics()
                if speaker is not None:
                    # Observability only: capture the actual backend state before
                    # close() tears down the per-turn native executor.  This is
                    # consumed by the P7 verifier and does not alter sampling
                    # or the generated PCM.
                    if metrics is not None:
                        runtime_info = getattr(speaker, "runtime_info", None)
                        metrics["native_runtime_info"] = (
                            runtime_info() if callable(runtime_info) else {
                                "runtime_backend": self.runtime_backend,
                                "native_enabled": False,
                            }
                        )
                        metrics["future_prefix_runtime_backend"] = metrics[
                            "native_runtime_info"
                        ].get("runtime_backend")
                        metrics["future_prefix_native_enabled"] = bool(
                            metrics["native_runtime_info"].get("native_enabled")
                        )
                        metrics["future_prefix_q1_backend"] = getattr(
                            self, "talker_q1_backend", "eager"
                        )
                        metrics["future_prefix_prefix_frames_provided"] = (
                            int(h_pfx.shape[1]) if h_pfx is not None else 0
                        )
                        metrics["subtalker_graph_captured"] = bool(
                            getattr(active_subtalker_graph, "captured", False)
                        )
                        metrics["future_prefix_max_prefix_frames"] = int(
                            getattr(speaker, "max_prefix_frames", 0)
                        )
                        metrics["future_prefix_max_audio_frames"] = int(
                            getattr(speaker, "max_audio_frames", 0)
                        )
                        metrics["future_prefix_frames_buffered"] = int(
                            getattr(getattr(speaker, "buffer", None), "buffered_frames", 0)
                        )
                        metrics["future_prefix_audio_frame_index"] = int(
                            getattr(
                                speaker,
                                "_current_frame_id",
                                len(codes) if "codes" in locals() else 0,
                            )
                        )
                    speaker.close()
                    if talk_context is not None and talk_context.is_current():
                        talk_context.future_prefix_qwen = None
                if talk_context is None:
                    if had_voice:
                        self.talker.voice_conditioning = previous_voice
                    elif hasattr(self.talker, "voice_conditioning"):
                        delattr(self.talker, "voice_conditioning")
                if metrics is not None:
                    metrics["talker_lock_hold_ms"] = (
                        (time.perf_counter() - talker_lock_acquired) * 1000.0
                        if talk_context is None else 0.0
                    )
                    metrics["talker_total_ms"] = (time.perf_counter() - talker_started) * 1000.0
                    for key, value in tuple(metrics.items()):
                        if key.endswith("_ms"):
                            metrics[key] = round(float(value), 3)

    @_private_talker_slot_stream
    @torch.inference_mode()
    def talk(self, text: str, h_pfx: torch.Tensor, *, n_words: int,
             voice_conditioning: dict[str, Any] | None = None,
             stop_check: Any | None = None, state: Any | None = None,
             on_audio_chunk: Any | None = None, on_cb0_frame: Any | None = None,
             min_frames: int = ARIA_MIN_FRAMES, max_frames: int | None = None,
             metrics: dict[str, Any] | None = None,
             generation_id: int | None = None,
             trailing_text_hidden: torch.Tensor | None = None,
             raw_text_token_ids: list[int] | tuple[int, ...] | None = None,
             include_text_eos: bool = True,
             live_qwen_text_stream: Any | None = None,
             live_frame_residual_hidden_stream: Any | None = None,
             frame_residual_response_hidden: torch.Tensor | None = None,
             listener_admission_wait: Callable[[], Any] | None = None,
             quality_diagnostic: dict[str, Any] | None = None,
             talk_context: Any | None = None) -> np.ndarray:
        if metrics is not None:
            metrics.update({
                "eos_detected": False,
                "eos_frame_id": None,
                "generated_cb0_frames": 0,
                "generated_pcm_samples": 0,
                "terminated_by_budget": False,
                "terminated_by": None,
                "rvq_frames": 0,
                "active_pcm_samples": 0,
                "active_pcm_duration_ms": 0.0,
                "effective_min_frames": int(min_frames),
                "effective_max_frames": None,
                "qwen_audio_conditioning_recipe": getattr(
                    self,
                    "qwen_audio_recipe",
                    QWEN_AUDIO_CONDITIONING_RECIPE,
                ),
                "listener_admission_talker_wait_ms": 0.0,
                "listener_admission_talker_wait_max_ms": 0.0,
                "talker_q1_backend": (
                    self.talker_q1_backend
                    if getattr(self, "audio_response_protocol", None) == AUDIO_RESPONSE_PROTOCOL_V2
                    else "eager"
                ),
                "compiled_talker_q1_calls": 0,
                "native_talker_q1_calls": 0,
                "native_talker_q1_graph_calls": 0,
                "native_talker_q1_graph_requested": bool(
                    getattr(self, "_talker_native_audio_q1_graph_requested", False)
                ),
            })
        if talk_context is None:
            self._stop_event = False
        elif not talk_context.is_current():
            if metrics is not None:
                metrics["cancelled"] = True
                metrics["stale_talk_context"] = True
            return np.zeros(0, dtype=np.float32)

        frame_residual_adapter = getattr(self, "frame_residual_adapter", None)
        if frame_residual_response_hidden is not None:
            if frame_residual_adapter is None:
                raise RuntimeError("response hidden passed without a loaded frame residual adapter")
            if (
                frame_residual_response_hidden.ndim != 3
                or tuple(frame_residual_response_hidden.shape[:1]) != (1,)
                or int(frame_residual_response_hidden.shape[1]) < 1
                or int(frame_residual_response_hidden.shape[-1]) != 1536
            ):
                raise ValueError("frame residual response hidden must have shape (1,T,1536)")
        if live_qwen_text_stream is not None:
            if raw_text_token_ids is None or len(raw_text_token_ids) != 1:
                raise ValueError("live Qwen response requires only token 0 in prefill")
            if include_text_eos:
                raise ValueError("live Qwen prefill must omit text EOS")
            if trailing_text_hidden is not None:
                raise ValueError("live Qwen text tail cannot be combined with static trailing hiddens")
        if live_frame_residual_hidden_stream is not None and frame_residual_adapter is None:
            raise RuntimeError("live response hidden requires a loaded frame residual adapter")

        def talk_stopped() -> bool:
            context_cancelled = (
                bool(talk_context.is_cancelled()) if talk_context is not None
                else bool(self._stop_event)
            )
            return context_cancelled or bool(stop_check is not None and stop_check())

        effective_voice = (
            _resolve_talker_voice_conditioning(
                self.talker,
                voice_conditioning,
            )
        )
        if effective_voice is not None and talk_context is None:
            # Il builder nativo può consumare questo payload; lo manteniamo sul
            # wrapper per non duplicare il protocollo Qwen nel loop caldo.
            self.talker.voice_conditioning = effective_voice
        if state is not None and generation_id is not None and not state.is_generation_current(generation_id=generation_id):
            state.stale_kv_writes += 1
            return np.zeros(0, dtype=np.float32)
        t = self.talker.talker
        tcfg = self.talker.tcfg
        short_streaming = on_audio_chunk is not None and min_frames < ARIA_MIN_FRAMES
        spoken_units, numeric_extra_units, T_gen = _estimate_talker_frame_budget(
            text,
            n_words,
            min_frames,
            short_streaming=short_streaming,
        )
        text_derived_T_gen = T_gen
        diagnostic_max_frames = (
            quality_diagnostic.get("max_acoustic_frame_budget")
            if isinstance(quality_diagnostic, dict)
            else None
        )
        T_gen = apply_diagnostic_frame_budget(
            T_gen,
            diagnostic_max_frames=diagnostic_max_frames,
            absolute_max_frames=ARIA_MAX_FRAMES,
        )
        if max_frames is not None:
            configured_max_frames = int(max_frames)
            if configured_max_frames <= 0:
                raise ValueError("max_frames must be positive when provided")
            T_gen = min(T_gen, configured_max_frames)
            if metrics is not None:
                metrics["configured_max_frames"] = configured_max_frames
        else:
            configured_max_frames = ARIA_MAX_FRAMES
        live_frame_cap = min(ARIA_MAX_FRAMES, configured_max_frames)
        live_eos_frame_cap = live_frame_cap
        if diagnostic_max_frames is not None:
            live_eos_frame_cap = min(live_eos_frame_cap, int(diagnostic_max_frames))
        if metrics is not None:
            metrics.update({
                "spoken_text_units": spoken_units,
                "numeric_extra_spoken_units": numeric_extra_units,
                "requested_acoustic_frame_budget": text_derived_T_gen,
                "diagnostic_max_acoustic_frame_budget": diagnostic_max_frames,
                "acoustic_frame_budget": T_gen,
                "effective_min_frames": int(min_frames),
                "effective_max_frames": int(T_gen),
            })
        if T_gen < QWEN_TALKER_MIN_NEW_TOKENS:
            if metrics is not None:
                metrics["no_speech_reason"] = "acoustic_budget_below_qwen_min_new_tokens"
            return np.zeros(0, dtype=np.float32)
        cuda_timing_requested = bool(
            metrics is not None
            and os.environ.get("DUPLEX_TALKER_CUDA_TIMING", "0") == "1"
            and callable(listener_admission_wait)
            and torch.cuda.is_available()
            and torch.device(self.device).type == "cuda"
        )
        cuda_timing_enabled = (
            _claim_talker_cuda_timing_probe() if cuda_timing_requested else False
        )
        native_v2_talker = bool(
            getattr(self, "audio_response_protocol", None) == AUDIO_RESPONSE_PROTOCOL_V2
            and getattr(self, "talker_q1_backend", None) == "native_audio_q1"
            and getattr(self, "talker_q1_native_status", None) == "ready"
            and getattr(self, "_talker_native_audio_q1_requested", False)
        )
        if frame_residual_response_hidden is not None or live_frame_residual_hidden_stream is not None:
            if not native_v2_talker and not getattr(self, "allow_eager_frame_residual", True):
                raise RuntimeError("frame residual candidate requires the actual native Qwen audio-Q1 route")
            if h_pfx.ndim != 3 or tuple(h_pfx.shape) != (1, 0, 2048):
                raise RuntimeError("frame residual candidate requires plain Qwen semantic prefill with zero prefix slots")
        compiled_v2_talker = bool(
            getattr(self, "audio_response_protocol", None) == AUDIO_RESPONSE_PROTOCOL_V2
            and getattr(self, "talker_q1_backend", None) == "torch_compile_dynamic"
            and getattr(self, "talker_q1_compile_status", None) == "ready"
            and getattr(self, "_compiled_talker_q1", None) is not None
        )
        prioritized_v2_talker = native_v2_talker or compiled_v2_talker
        response_context_gpu_stages = talk_context is not None
        priority_lock_talker = prioritized_v2_talker or response_context_gpu_stages
        timing_stream = (
            torch.cuda.current_stream(device=self.device)
            if cuda_timing_enabled else None
        )
        cuda_timing_pairs: dict[str, list[tuple[Any, Any]]] = {
            stage: [] for stage in (
                "static_prefill",
                "dynamic_prefill",
                "codec_head",
                "subtalker",
                "next_qwen_model",
                "code2wav_decode",
            )
        }
        cuda_timing_record_failures = {stage: 0 for stage in cuda_timing_pairs}
        admission_samples_by_site: dict[str, list[float]] = {}
        admission_samples_by_phase: dict[str, list[float]] = {}

        def begin_cuda_timing(stage: str) -> tuple[Any, Any] | None:
            if not cuda_timing_enabled:
                return None
            try:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(timing_stream)
                return start_event, end_event
            except Exception:
                cuda_timing_record_failures[stage] += 1
                return None

        def finish_cuda_timing(stage: str, pair: tuple[Any, Any] | None) -> None:
            if pair is None:
                return
            try:
                pair[1].record(timing_stream)
                cuda_timing_pairs[stage].append(pair)
            except Exception:
                cuda_timing_record_failures[stage] += 1

        def admit_listener(phase: str, site: str) -> None:
            if (
                prioritized_v2_talker
                and phase == "talker"
                and site in {"codec_head", "subtalker", "next_qwen_model"}
            ):
                return
            started = time.perf_counter() if cuda_timing_enabled else 0.0
            self._wait_for_listener_admission(
                listener_admission_wait, metrics, phase
            )
            if cuda_timing_enabled and callable(listener_admission_wait):
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                admission_samples_by_site.setdefault(site, []).append(elapsed_ms)
                admission_samples_by_phase.setdefault(phase, []).append(elapsed_ms)

        def run_talker_gpu_step(site: str, forward: Callable[[], Any]) -> Any:
            if not priority_lock_talker:
                return forward()
            wait_started = time.perf_counter()
            self._listener_priority_lock()
            acquired = time.perf_counter()
            wait_ms = (acquired - wait_started) * 1000.0
            if metrics is not None:
                wait_key = f"listener_priority_{site}_lock_wait_ms"
                wait_max_key = f"listener_priority_{site}_lock_wait_max_ms"
                metrics[wait_key] = metrics.get(wait_key, 0.0) + wait_ms
                metrics[wait_max_key] = max(
                    metrics.get(wait_max_key, 0.0), wait_ms
                )
            try:
                return forward()
            finally:
                held_ms = (time.perf_counter() - acquired) * 1000.0
                if metrics is not None:
                    hold_key = f"listener_priority_{site}_lock_hold_ms"
                    hold_max_key = f"listener_priority_{site}_lock_hold_max_ms"
                    metrics[hold_key] = metrics.get(hold_key, 0.0) + held_ms
                    metrics[hold_max_key] = max(
                        metrics.get(hold_max_key, 0.0), held_ms
                    )
                self._thinker_lock.release()

        def log_private_cuda_timing() -> None:
            if not cuda_timing_enabled:
                return
            event_summary = {
                stage: _summarize_cuda_event_pairs(pairs)
                for stage, pairs in cuda_timing_pairs.items()
            }
            phase_totals = {
                phase: {
                    "total_ms": round(sum(samples), 3),
                    **_summarize_timing_values_ms(samples),
                }
                for phase, samples in admission_samples_by_phase.items()
            }
            report = {
                "schema": "talker_cuda_stage_timing_v1",
                "method": "current_stream_cuda_events_query_only_no_per_frame_sync",
                "frame_count": len(codes),
                "stages": event_summary,
                "event_record_failures": dict(cuda_timing_record_failures),
                "admission_wait_ms_by_site": {
                    site: _summarize_timing_values_ms(samples)
                    for site, samples in admission_samples_by_site.items()
                },
                "admission_wait_ms_by_phase": phase_totals,
            }
            if metrics is not None:
                metrics["_talker_cuda_stage_timing_private"] = report
            logger.info(
                "[talker_cuda_stage_timing_private] %s",
                json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True),
            )

        (
            static_part,
            text_part,
            bos_part,
            tts_pad_embed,
        ) = _build_talker_prefill_parts(
            self.talker,
            text,
            effective_voice,
            raw_text_token_ids=raw_text_token_ids,
            include_text_eos=include_text_eos,
        )
        # Il solo blocco statico [assistant role + codec ICL] è riutilizzabile
        # tra turni; testo e h_pfx restano nel prefill dinamico del turno.
        if (
            state is not None
            and state.talker_prefix.is_ready()
            and effective_voice is None
        ):
            # DynamicCache viene aggiornata in-place da alcuni modelli Qwen:
            # clona una volta per turno, mai una volta per frame.
            import copy
            prefix_kv = copy.deepcopy(state.talker_prefix.kv)
        else:
            admit_listener("talker_prefill", "static_prefill")
            timing_pair = begin_cuda_timing("static_prefill")
            out_static = run_talker_gpu_step(
                "static_prefill",
                lambda: t.model(inputs_embeds=static_part, use_cache=True),
            )
            finish_cuda_timing("static_prefill", timing_pair)
            prefix_kv = out_static.past_key_values
            if state is not None:
                with state._generation_lock:
                    if generation_id is None or state.is_generation_current(generation_id=generation_id):
                        state.talker_prefix.store(prefix_kv)
                        state.talker_prefix.last_hidden = out_static.last_hidden_state[:, -1:, :].detach()
                    else:
                        state.stale_kv_writes += 1
        dynamic = torch.cat([text_part, h_pfx.to(self.dtype), bos_part], dim=1)
        admit_listener("talker_prefill", "dynamic_prefill")
        timing_pair = begin_cuda_timing("dynamic_prefill")
        out_pfx = run_talker_gpu_step(
            "dynamic_prefill",
            lambda: t.model(inputs_embeds=dynamic, past_key_values=prefix_kv, use_cache=True),
        )
        finish_cuda_timing("dynamic_prefill", timing_pair)
        past_kv = out_pfx.past_key_values
        if talk_context is not None:
            talk_context.qwen_kv = past_kv
        ack_initial_text = getattr(live_qwen_text_stream, "ack_initial_token", None)
        if callable(ack_initial_text):
            # The initial ID is now in the successful dynamic Qwen prefill;
            # only this point commits its source Gemma packet to the cursor.
            ack_initial_text()
        native_executor = None
        if native_v2_talker:
            # The native context is private to this response.  It imports the
            # exact eager HF prefill cache and never publishes cross-turn KV.
            # Live SAY extends its text-derived acoustic budget as Gemma emits
            # safe prefixes, so reserve the existing bounded live cap at import
            # time; the loop still stops at its dynamically updated T_gen.
            native_q1_import_frames = _native_q1_import_audio_frame_budget(
                T_gen,
                live_qwen_text_stream=live_qwen_text_stream,
                live_frame_cap=live_frame_cap,
            )
            native_executor = self._new_native_talker_q1(
                t,
                past_kv,
                max_audio_frames=native_q1_import_frames,
                talk_context=talk_context,
            )
            if metrics is not None:
                metrics["native_q1_import_audio_frame_capacity"] = int(native_q1_import_frames)
                metrics["native_q1_live_frame_cap"] = int(live_frame_cap)
            if talk_context is not None:
                talk_context.native_q1_executor = native_executor
        last_h = out_pfx.last_hidden_state[:, -1:, :]
        if state is not None:
            with state._generation_lock:
                if generation_id is None or state.is_generation_current(generation_id=generation_id):
                    state.turn_kv.start_from_prefix(state.talker_prefix)
                    state.turn_kv.append(past_kv)
                else:
                    state.stale_kv_writes += 1
        if not torch.isfinite(last_h).all():
            raise RuntimeError("Talker prefill ha prodotto hidden non finito")
        eos_detected = False
        eos_frame_id: int | None = None
        terminal_queries = 0
        eos_token_id = self._codec_eos_token_id() if self.allow_talker_eos else None
        codes: list[Any] = []
        streamed_audio: list[np.ndarray] = (
            talk_context.pcm_chunks if talk_context is not None else []
        )
        frame_residual_calls = 0
        frame_residual_q1_steps = 0
        frame_residual_tail_hold_calls = 0
        frame_residual_delta_sq_sum: torch.Tensor | None = None
        frame_residual_base_sq_sum: torch.Tensor | None = None
        frame_residual_element_count = 0
        live_first_packet_hidden: torch.Tensor | None = None
        live_first_packet_hidden_object: torch.Tensor | None = None
        live_previous_packet_hidden: torch.Tensor | None = None
        live_previous_packet_hidden_object: torch.Tensor | None = None
        live_hidden_trace_tensors: list[tuple[dict[str, Any], torch.Tensor, torch.Tensor | None]] = []

        def apply_frame_residual(query: torch.Tensor, generation_step: int) -> torch.Tensor:
            nonlocal frame_residual_calls, frame_residual_q1_steps, frame_residual_tail_hold_calls
            nonlocal frame_residual_delta_sq_sum, frame_residual_base_sq_sum
            nonlocal frame_residual_element_count
            nonlocal live_first_packet_hidden
            nonlocal live_first_packet_hidden_object
            nonlocal live_previous_packet_hidden
            nonlocal live_previous_packet_hidden_object
            adapter = frame_residual_adapter
            response_states = frame_residual_response_hidden
            residual_step = generation_step
            live_tail_hold = False
            live_state = None
            if live_frame_residual_hidden_stream is not None:
                live_state = live_frame_residual_hidden_stream.hidden_for_frame(
                    generation_step,
                    stop_check=talk_stopped,
                )
                response_states = live_state.hidden
                live_tail_hold = bool(live_state.tail_hold)
                _wait_for_cuda_producer_event(
                    response_states,
                    getattr(live_state, "producer_event", None),
                )
            if adapter is None or response_states is None:
                return query
            if live_state is not None:
                packet_position = (
                    int(live_state.packet_exclusive_kv_cursor)
                    if live_state.packet_exclusive_kv_cursor is not None else None
                )
                source_hidden = response_states.detach().float().reshape(-1)
                hidden_norm = torch.linalg.vector_norm(source_hidden)
                hidden_cosine: torch.Tensor | None = None
                if (
                    live_previous_packet_hidden is not None
                    and response_states is not live_previous_packet_hidden_object
                ):
                    hidden_cosine = F.cosine_similarity(
                        source_hidden.unsqueeze(0),
                        live_previous_packet_hidden.unsqueeze(0),
                        dim=-1,
                    ).reshape(())
                if response_states is not live_previous_packet_hidden_object:
                    if live_first_packet_hidden is None:
                        live_first_packet_hidden = source_hidden.detach().clone()
                        live_first_packet_hidden_object = response_states
                    live_previous_packet_hidden = source_hidden.detach().clone()
                    live_previous_packet_hidden_object = response_states
                trace_entry: dict[str, Any] = {
                    "qwen_frame_step": int(generation_step),
                    "gemma_token_id": live_state.gemma_token_id,
                    "gemma_kv_position": packet_position,
                    "qwen_token_id": live_state.qwen_token_id,
                    "qwen_packet_token_index": live_state.packet_token_index,
                    "qwen_packet_token_count": live_state.packet_token_count,
                    "tail_hold": bool(live_state.tail_hold),
                }
                if metrics is not None:
                    if len(live_hidden_trace_tensors) < 256:
                        live_hidden_trace_tensors.append((trace_entry, hidden_norm, hidden_cosine))
                logger.debug(
                    "[turn-local frame residual] qwen_frame=%d gemma_token=%s gemma_kv=%s "
                    "qwen_token=%s qwen_span=%s/%s tail_hold=%s",
                    int(generation_step), live_state.gemma_token_id, packet_position,
                    live_state.qwen_token_id, live_state.packet_token_index,
                    live_state.packet_token_count, bool(live_state.tail_hold),
                )
            if live_state is not None:
                base_query = query.detach().clone()
                conditioned, tail_hold = _apply_live_frame_residual(
                    adapter,
                    response_states,
                    query,
                    generation_step=generation_step,
                    tail_hold=live_tail_hold,
                )
            else:
                base_query = query.detach().clone()
                conditioned, tail_hold = apply_response_state_residual(
                    adapter, response_states, query, residual_step
                )
            delta = conditioned.float() - base_query.float()
            base = base_query.float()
            delta_sq, base_sq = delta.square().sum(), base.square().sum()
            frame_residual_delta_sq_sum = (
                delta_sq if frame_residual_delta_sq_sum is None else frame_residual_delta_sq_sum + delta_sq
            )
            frame_residual_base_sq_sum = (
                base_sq if frame_residual_base_sq_sum is None else frame_residual_base_sq_sum + base_sq
            )
            frame_residual_element_count += int(delta.numel())
            frame_residual_q1_steps += 1
            if not (
                generation_step == 0
                and bool(getattr(adapter, "preserve_frame0", False))
            ):
                frame_residual_calls += 1
            frame_residual_tail_hold_calls += int(tail_hold or live_tail_hold)
            return conditioned

        def finalize_frame_residual_metrics() -> None:
            if metrics is None or frame_residual_adapter is None or (
                frame_residual_response_hidden is None
                and live_frame_residual_hidden_stream is None
            ):
                return
            denominator = max(1, frame_residual_element_count)
            delta_sq_sum = (
                frame_residual_delta_sq_sum
                if frame_residual_delta_sq_sum is not None
                else torch.zeros((), device=self.device, dtype=torch.float32)
            )
            base_sq_sum = (
                frame_residual_base_sq_sum
                if frame_residual_base_sq_sum is not None
                else torch.zeros((), device=self.device, dtype=torch.float32)
            )
            summary_values: list[torch.Tensor] = [
                delta_sq_sum.detach().float(),
                base_sq_sum.detach().float(),
                frame_residual_adapter.scale.detach().float().reshape(()),
            ]
            first_last_index: int | None = None
            if (
                live_first_packet_hidden is not None
                and live_previous_packet_hidden is not None
                and live_first_packet_hidden_object is not live_previous_packet_hidden_object
            ):
                first_last_index = len(summary_values)
                summary_values.append(F.cosine_similarity(
                    live_first_packet_hidden.unsqueeze(0),
                    live_previous_packet_hidden.unsqueeze(0),
                    dim=-1,
                ).reshape(()))
            trace_scalar_indexes: list[tuple[int, int | None]] = []
            for _entry, norm, cosine in live_hidden_trace_tensors:
                norm_index = len(summary_values)
                summary_values.append(norm.detach().float().reshape(()))
                cosine_index = None
                if cosine is not None:
                    cosine_index = len(summary_values)
                    summary_values.append(cosine.detach().float().reshape(()))
                trace_scalar_indexes.append((norm_index, cosine_index))
            scalar_values = torch.stack(summary_values).detach().to(device="cpu").tolist()
            delta_sq_value = float(scalar_values[0])
            base_sq_value = float(scalar_values[1])
            delta_rms = float((delta_sq_value / denominator) ** 0.5)
            base_rms = float((base_sq_value / denominator) ** 0.5)
            first_last_hidden_cosine = (
                float(scalar_values[first_last_index])
                if first_last_index is not None else None
            )
            trace = []
            for (entry, _norm, _cosine), (norm_index, cosine_index) in zip(
                live_hidden_trace_tensors, trace_scalar_indexes
            ):
                trace.append({
                    **entry,
                    "gemma_hidden_norm": float(scalar_values[norm_index]),
                    "gemma_hidden_cosine_previous_packet": (
                        float(scalar_values[cosine_index])
                        if cosine_index is not None else None
                    ),
                })
            metrics.update({
                "frame_residual_adapter": str(
                    getattr(frame_residual_adapter, "architecture_id", "unknown")
                ),
                "frame_residual_checkpoint": self.frame_residual_adapter_resolved_path,
                "frame_residual_calls": frame_residual_calls,
                "frame_residual_q1_steps": frame_residual_q1_steps,
                "frame_residual_tail_hold_calls": frame_residual_tail_hold_calls,
                "frame_residual_qwen_q1_only": True,
                "frame_residual_prefix_tokens_added": 0,
                "frame_residual_learned_scale": float(scalar_values[2]),
                "frame_residual_delta_sq_sum": delta_sq_value,
                "frame_residual_base_sq_sum": base_sq_value,
                "frame_residual_rms": delta_rms,
                "frame_residual_base_rms": base_rms,
                "frame_residual_rms_to_base_ratio": delta_rms / base_rms if base_rms > 0.0 else 0.0,
                "frame_residual_live_first_last_hidden_cosine": first_last_hidden_cosine,
                "frame_residual_live_hidden_trace": trace,
            })
        if talk_context is not None:
            talk_context.hook_state = {
                "on_audio_chunk": on_audio_chunk,
                "generation_id": generation_id,
            }
        preclamp_audio: list[np.ndarray] = []
        streaming_requested = on_audio_chunk is not None

        capture_preclamp = bool(
            isinstance(quality_diagnostic, dict)
            and os.environ.get("DUPLEX_CAPTURE_CODE2WAV_PRECLAMP", "0") == "1"
        )

        def record_preclamp(value: torch.Tensor) -> None:
            array = value.detach().to(dtype=torch.float32).cpu().numpy()
            preclamp_audio.append(np.ascontiguousarray(array.reshape(-1)))

        def preclamp_summary() -> dict[str, Any]:
            values = np.concatenate(preclamp_audio) if preclamp_audio else np.empty(0, dtype=np.float32)
            finite = values[np.isfinite(values)]
            return {
                "samples": int(values.size),
                "nonfinite_count": int(values.size - finite.size),
                "finite_peak_abs": float(np.max(np.abs(finite))) if finite.size else None,
                "finite_min": float(np.min(finite)) if finite.size else None,
                "finite_max": float(np.max(finite)) if finite.size else None,
                "finite_outside_unit_count": int(np.count_nonzero(np.abs(finite) > 1.0)),
            }

        def rvq_codes_to_numpy(chunk_codes: Any, *, dtype: Any = np.int64) -> np.ndarray:
            if isinstance(chunk_codes, torch.Tensor):
                values = chunk_codes.detach().to(device="cpu", dtype=torch.int64).numpy()
            elif (
                isinstance(chunk_codes, (list, tuple))
                and chunk_codes
                and isinstance(chunk_codes[0], torch.Tensor)
            ):
                values = torch.stack(list(chunk_codes), dim=0).detach().to(
                    device="cpu", dtype=torch.int64
                ).numpy()
            else:
                values = np.asarray(chunk_codes, dtype=np.int64)
            return np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(-1, 16))

        def capture_preclamp_failure(chunk_codes: Any, exc: BaseException) -> None:
            if not capture_preclamp or not preclamp_audio:
                return
            capture = quality_diagnostic.get("capture")
            if not callable(capture):
                return
            metadata = dict(quality_diagnostic.get("metadata") or {})
            metadata["code2wav_preclamp"] = preclamp_summary()
            metadata["decode_failure"] = type(exc).__name__
            try:
                capture({
                    "metadata": metadata,
                    "say_hidden": quality_diagnostic.get("say_hidden"),
                    "v9_prefix": h_pfx,
                    "rvq_codes": rvq_codes_to_numpy(chunk_codes, dtype=np.int32),
                    "preclamp_pcm": np.concatenate(preclamp_audio),
                })
            except Exception as capture_error:
                logger.warning(
                    "[code2wav pre-clamp diagnostic] failure capture failed: %s",
                    str(capture_error)[:180],
                )

        def decode_chunk(chunk_codes: Any) -> np.ndarray:
            if isinstance(chunk_codes, torch.Tensor):
                tensor = chunk_codes
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0)
                elif tensor.ndim != 3:
                    raise ValueError(f"RVQ decode tensor must have rank 2 or 3, got {tensor.ndim}")
                target_device = torch.device(self.device)
                if (
                    target_device.type == tensor.device.type
                    and target_device.index is None
                ):
                    # An unindexed CUDA device names the current device; keep the
                    # already allocated tensor on its concrete device (cuda:N).
                    target_device = tensor.device
                if tensor.device != target_device:
                    tensor = tensor.to(device=target_device)
                if tensor.dtype != torch.long:
                    tensor = tensor.to(dtype=torch.long)
            elif (
                isinstance(chunk_codes, (list, tuple))
                and chunk_codes
                and isinstance(chunk_codes[0], torch.Tensor)
            ):
                tensor = torch.stack(list(chunk_codes), dim=0).unsqueeze(0)
            else:
                tensor = torch.tensor(
                    chunk_codes,
                    device=self.device,
                    dtype=torch.long,
                ).unsqueeze(0)
            if tensor.shape[0] != 1 or tensor.shape[2] != 16:
                raise ValueError(f"RVQ decode tensor must have shape (1,T,16), got {tuple(tensor.shape)}")
            warmup_trace_base = None
            warmup_codebook_size_config = None
            warmup_codebook_bounds = None
            warmup_invalid_counts = None
            if getattr(self, "_warmup_diagnostic_active", False):
                code_array = rvq_codes_to_numpy(chunk_codes, dtype=np.int64)
                self._warmup_debug_rvq_codes.extend(code_array.tolist())
                rvq_path = Path(".tmp/recovery_probe/warmup_rvq_codes_seed_20261006_q1eager.npy")
                rvq_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(rvq_path, np.asarray(self._warmup_debug_rvq_codes, dtype=np.int32))
                rvq_path.with_suffix(".json").write_text(json.dumps({
                    "schema": "warmup_rvq_replay_v1", "seed": self._warmup_diagnostic_seed,
                    "q1_backend": getattr(self, "talker_q1_backend", None), "dtype": "int32",
                    "shape": [len(self._warmup_debug_rvq_codes), 16],
                }, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                decoder = getattr(self.talker, "code2wav", None)
                decoder_config = getattr(decoder, "config", None)
                raw_size = getattr(decoder_config, "codebook_size", None)
                if raw_size is None:
                    raw_size = getattr(getattr(decoder_config, "decoder_config", None), "codebook_size", None)
                if isinstance(raw_size, int) and not isinstance(raw_size, bool):
                    warmup_codebook_size_config = int(raw_size)
                    warmup_codebook_bounds = [int(raw_size)] * 16
                elif isinstance(raw_size, (list, tuple)) and len(raw_size) == 16:
                    warmup_codebook_size_config = [int(value) for value in raw_size]
                    warmup_codebook_bounds = list(warmup_codebook_size_config)
                else:
                    warmup_codebook_size_config = None
                    warmup_codebook_bounds = None
                warmup_invalid_counts = [
                    int(np.count_nonzero((code_array[:, index] < 0) | (code_array[:, index] >= warmup_codebook_bounds[index])))
                    if warmup_codebook_bounds is not None and warmup_codebook_bounds[index] > 0
                    else None
                    for index in range(16)
                ]
                warmup_trace_base = {
                    "seed": getattr(self, "_warmup_diagnostic_seed", None),
                    "decode_input_shape": list(tensor.shape),
                    "decode_input_dtype": str(tensor.dtype),
                    "decode_input_device": str(tensor.device),
                    "rvq_frames": int(code_array.shape[0]),
                    "rvq_codebook_ranges": [
                        {"min": int(code_array[:, index].min()), "max": int(code_array[:, index].max())}
                        for index in range(16)
                    ],
                    "configured_codebook_size": warmup_codebook_size_config,
                    "cb0_configured_codebook_size": warmup_codebook_bounds[0] if warmup_codebook_bounds else None,
                    "residual_configured_codebook_sizes": warmup_codebook_bounds[1:] if warmup_codebook_bounds else None,
                    "cb0_out_of_range_count": warmup_invalid_counts[0],
                    "residual_out_of_range_counts": warmup_invalid_counts[1:],
                    "last_rvq_frames": code_array[-4:].tolist(),
                    "last_cb0_id": getattr(self, "_warmup_debug_last_cb0_id", None),
                    "last_residual_codes": list(getattr(self, "_warmup_debug_last_residual_codes", [])),
                }
                if any(count for count in warmup_invalid_counts if count is not None):
                    warmup_trace_base.update({"schema": "warmup_rvq_code_range_failure_v1", "status": "failed"})
                    logger.error("[warmup_rvq_code_range] %s", json.dumps(
                        warmup_trace_base, ensure_ascii=False, sort_keys=True, allow_nan=False,
                    ))
                    trace_path = Path(".tmp/recovery_probe/warmup_code2wav_failure.json")
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    trace_path.write_text(
                        json.dumps(warmup_trace_base, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                    raise RuntimeError("RVQ codes exceed configured Code2Wav codebook bounds")
            def decode_with_warmup_diagnostics() -> Any:
                decoder = getattr(self.talker, "code2wav", None)
                handles = []
                diagnostic_hooks = bool(getattr(self, "_warmup_diagnostic_active", False)
                    and not self._warmup_debug_decoder_hooks_done and isinstance(decoder, torch.nn.Module))
                if diagnostic_hooks:
                    self._warmup_debug_decoder_hooks_done = True
                    def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any, *, module_name: str):
                        if self._warmup_debug_first_nonfinite_module is not None:
                            return
                        try:
                            for output_path, value in _warmup_diag_tensors(output):
                                if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
                                    self._warmup_debug_first_nonfinite_module = {
                                        "module_name": module_name or "<root-child>",
                                        "module_type": type(module).__name__,
                                        "output_path": output_path,
                                        "inputs": [{"path": p, **_warmup_diag_tensor(t)}
                                                   for p, t in _warmup_diag_tensors(inputs)[:12]],
                                        "output": _warmup_diag_tensor(value),
                                        "rotary_buffers": (
                                            _warmup_diag_rotary_state(module)
                                            if "rotary" in module_name.lower()
                                            else []
                                        ),
                                        "runtime_backends": _warmup_diag_backends(),
                                    }
                                    logger.error("[warmup_code2wav_first_nonfinite_module] %s", json.dumps(
                                        self._warmup_debug_first_nonfinite_module, sort_keys=True, allow_nan=False,
                                    ))
                                    return
                        except Exception as exc:
                            logger.warning("[warmup_code2wav_hook] module=%s error=%s",
                                           module_name, type(exc).__name__)
                    for name, module in decoder.named_modules():
                        handles.append(module.register_forward_hook(
                            lambda mod, args, out, module_name=name:
                                hook(mod, args, out, module_name=module_name)))
                profiler = None
                if (getattr(self, "_warmup_diagnostic_active", False)
                    and os.environ.get("DUPLEX_WARMUP_CODE2WAV_TRACE") == "1"
                    and not self._warmup_debug_decoder_profiled):
                    self._warmup_debug_decoder_profiled = True
                    try:
                        activities = [torch.profiler.ProfilerActivity.CPU]
                        if tensor.device.type == "cuda":
                            activities.append(torch.profiler.ProfilerActivity.CUDA)
                        profiler = torch.profiler.profile(
                            activities=activities, record_shapes=False,
                            profile_memory=False, with_stack=False,
                        )
                    except Exception as exc:
                        self._warmup_debug_decoder_profiler = {
                            "available": False, "error": type(exc).__name__, "operators": [],
                        }
                try:
                    def invoke() -> Any:
                        code2wav_graph = (
                            talk_context.code2wav_graph if talk_context is not None else None
                        )
                        if (
                            code2wav_graph is not None
                            and not capture_preclamp
                            and not getattr(self, "_warmup_diagnostic_active", False)
                            and _code2wav_graph_supports_shape(code2wav_graph, tensor)
                        ):
                            if metrics is not None:
                                _record_code2wav_graph_use(
                                    metrics, code2wav_graph, tensor
                                )
                            return code2wav_graph.decode(tensor)
                        if metrics is not None:
                            metrics["code2wav_backend"] = "qwen_chunked_decode"
                            metrics["code2wav_eager_calls"] = (
                                metrics.get("code2wav_eager_calls", 0) + 1
                            )
                        def _har_preclamp_callback(val: torch.Tensor) -> torch.Tensor:
                            if capture_preclamp:
                                record_preclamp(val)
                            return rescale_code2wav_preclamp(val, max_peak=0.95)

                        return self.talker.decode(
                            tensor,
                            preclamp_callback=_har_preclamp_callback,
                        )
                    if profiler is None:
                        return run_talker_gpu_step("code2wav_decode", invoke)
                    with profiler:
                        return run_talker_gpu_step("code2wav_decode", invoke)
                finally:
                    if profiler is not None:
                        self._warmup_debug_decoder_profiler = _warmup_diag_profile(profiler)
                    for handle in handles:
                        handle.remove()
            code2wav_timing_pair = begin_cuda_timing("code2wav_decode")
            try:
                decoded = decode_with_warmup_diagnostics()
            except BaseException as exc:
                capture_preclamp_failure(chunk_codes, exc)
                raise
            finally:
                finish_cuda_timing("code2wav_decode", code2wav_timing_pair)
            decoded_dtype = str(decoded.dtype) if isinstance(decoded, torch.Tensor) else str(type(decoded).__name__)
            decoded_device = str(decoded.device) if isinstance(decoded, torch.Tensor) else None
            if hasattr(decoded, "detach"):
                decoded = decoded.detach().cpu().numpy()
            chunk = np.asarray(decoded, dtype=np.float32).reshape(-1)
            if chunk.size == 0:
                error = RuntimeError("Code2Wav ha prodotto PCM vuoto")
                capture_preclamp_failure(chunk_codes, error)
                raise error
            if getattr(self, "_warmup_diagnostic_active", False):
                code_array = rvq_codes_to_numpy(chunk_codes, dtype=np.int64)
                finite_pcm = chunk[np.isfinite(chunk)]
                decoder = getattr(self.talker, "code2wav", None)
                decoder_parameters = list(decoder.parameters()) if decoder is not None else []
                decoder_specs = sorted({(str(param.dtype), str(param.device)) for param in decoder_parameters})
                last_logits = getattr(self, "_warmup_debug_last_cb0_logits", None)
                cb0_stats = None
                if isinstance(last_logits, torch.Tensor):
                    logits = last_logits.detach().float()
                    finite_logits = torch.isfinite(logits)
                    finite_values = logits[finite_logits]
                    cb0_stats = {
                        "finite": bool(finite_logits.all().item()),
                        "finite_count": int(finite_logits.sum().item()),
                        "total_count": int(logits.numel()),
                        "min": float(finite_values.min().item()) if finite_values.numel() else None,
                        "max": float(finite_values.max().item()) if finite_values.numel() else None,
                    }
                self._warmup_debug_last_decode = {
                    "seed": getattr(self, "_warmup_diagnostic_seed", None),
                    "decode_input_shape": list(tensor.shape),
                    "decode_input_dtype": str(tensor.dtype),
                    "decode_input_device": str(tensor.device),
                    "rvq_frames": int(code_array.shape[0]),
                    "configured_codebook_size": warmup_codebook_size_config,
                    "cb0_configured_codebook_size": warmup_codebook_bounds[0] if warmup_codebook_bounds else None,
                    "residual_configured_codebook_sizes": warmup_codebook_bounds[1:] if warmup_codebook_bounds else None,
                    "cb0_out_of_range_count": warmup_invalid_counts[0] if warmup_invalid_counts else None,
                    "residual_out_of_range_counts": warmup_invalid_counts[1:] if warmup_invalid_counts else None,
                    "decoder_trainable_parameter_tensor_count": sum(
                        bool(param.requires_grad) for param in decoder_parameters
                    ),
                    "decoder_training_mode": bool(getattr(decoder, "training", False)),
                    "rvq_codebook_ranges": [
                        {"min": int(code_array[:, index].min()), "max": int(code_array[:, index].max())}
                        for index in range(code_array.shape[1])
                    ],
                    "last_rvq_frames": code_array[-4:].tolist(),
                    "last_cb0_id": getattr(self, "_warmup_debug_last_cb0_id", None),
                    "last_cb0_logits": cb0_stats,
                    "last_residual_codes": list(getattr(self, "_warmup_debug_last_residual_codes", [])),
                    "decoder_parameter_specs": decoder_specs,
                    "decoder_parameter_tensor_count": len(decoder_parameters),
                    "decoded_dtype": decoded_dtype,
                    "decoded_device": decoded_device,
                    "pcm_samples": int(chunk.size),
                    "pcm_nonfinite_count": int(chunk.size - finite_pcm.size),
                    "pcm_nan_count": int(np.isnan(chunk).sum()),
                    "pcm_positive_inf_count": int(np.isposinf(chunk).sum()),
                    "pcm_negative_inf_count": int(np.isneginf(chunk).sum()),
                    "pcm_finite_min": float(finite_pcm.min()) if finite_pcm.size else None,
                    "pcm_finite_max": float(finite_pcm.max()) if finite_pcm.size else None,
                    "pcm_finite_abs_peak": float(np.max(np.abs(finite_pcm))) if finite_pcm.size else None,
                    "subtalker_graph_type": type(getattr(self, "subtalker_graph", None)).__name__,
                    "talker_q1_backend": getattr(self, "talker_q1_backend", None),
                    "talker_q1_native_status": getattr(self, "talker_q1_native_status", None),
                    "talker_q1_compile_status": getattr(self, "talker_q1_compile_status", None),
                    "first_q1": self._warmup_debug_first_q1,
                    "first_nonfinite_module": self._warmup_debug_first_nonfinite_module,
                    "code2wav_weight_finiteness": self._warmup_debug_code2wav_weights,
                    "runtime_backends": _warmup_diag_backends(),
                    "profiler_backends": self._warmup_debug_decoder_profiler,
                    "rvq_replay_path": ".tmp/recovery_probe/warmup_rvq_codes_seed_20261006_q1eager.npy",
                }
            if not np.isfinite(chunk).all():
                if getattr(self, "_warmup_diagnostic_active", False):
                    trace = dict(self._warmup_debug_last_decode or {})
                    trace.update({"schema": "warmup_code2wav_failure_v2", "status": "failed"})
                    logger.error("[warmup_code2wav_nonfinite] %s", json.dumps(
                        trace, ensure_ascii=False, sort_keys=True, allow_nan=False,
                    ))
                    try:
                        trace_path = Path(".tmp/recovery_probe/warmup_code2wav_failure.json")
                        trace_path.parent.mkdir(parents=True, exist_ok=True)
                        trace_path.write_text(
                            json.dumps(trace, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                            encoding="utf-8",
                        )
                    except Exception as trace_exc:
                        logger.error("[warmup_code2wav_nonfinite] trace write failed: %s", str(trace_exc)[:200])
                raise RuntimeError("Code2Wav ha prodotto audio non finito")
            return chunk

        if (
            frame_residual_response_hidden is not None
            or live_frame_residual_hidden_stream is not None
        ) and streaming_requested:
            rvq_history = getattr(talk_context, "rvq_history", None)
            if rvq_history is None:
                rvq_history = DeviceRVQCodeRing(device=self.device)
            accumulator = TensorDecodeAccumulator(
                decode_fn=decode_chunk,
                history=rvq_history,
                chunk_frames=CODE2WAV_DECODE_FRAMES,
                frame_samples=AUDIO_FRAME_SAMPLES,
                left_context_frames=CODE2WAV_LEFT_CONTEXT_FRAMES,
                crossfade_samples=AUDIO_BOUNDARY_CROSSFADE_SAMPLES,
            )
        else:
            accumulator = DecodeAccumulator(
                decode_fn=decode_chunk,
                chunk_frames=CODE2WAV_DECODE_FRAMES,
                frame_samples=AUDIO_FRAME_SAMPLES,
                left_context_frames=CODE2WAV_LEFT_CONTEXT_FRAMES,
                crossfade_samples=AUDIO_BOUNDARY_CROSSFADE_SAMPLES,
            )

        def emit_decoded_chunk(decoded_chunk: Any) -> None:
            streamed_audio.append(decoded_chunk.pcm)
            if talk_context is not None:
                talk_context.callback_started()
            try:
                on_audio_chunk(decoded_chunk.pcm)
            finally:
                if talk_context is not None:
                    talk_context.callback_finished()

        frame_id = 0
        while True:
            if live_qwen_text_stream is not None:
                live_text = live_qwen_text_stream.visible_text
                if live_text:
                    live_words = max(1, len(live_text.split()))
                    live_spoken_units, live_numeric_extra, live_text_budget = (
                        _estimate_talker_frame_budget(
                            live_text,
                            live_words,
                            min_frames,
                            short_streaming=short_streaming,
                        )
                    )
                    # Each native text token needs one Q1 query after a CB0
                    # frame; reserve two additional frames to feed native text
                    # EOS and sample codec EOS after the tail drains.
                    live_text_budget = max(
                        live_text_budget,
                        live_qwen_text_stream.observed_qwen_token_count + 2,
                    )
                    live_text_budget = apply_diagnostic_frame_budget(
                        live_text_budget,
                        diagnostic_max_frames=diagnostic_max_frames,
                        absolute_max_frames=ARIA_MAX_FRAMES,
                    )
                    live_text_budget = min(live_frame_cap, live_text_budget)
                    if live_text_budget > T_gen:
                        T_gen = live_text_budget
                    if metrics is not None:
                        metrics.update({
                            "spoken_text_units": live_spoken_units,
                            "numeric_extra_spoken_units": live_numeric_extra,
                            "requested_acoustic_frame_budget": live_text_budget,
                            "acoustic_frame_budget": T_gen,
                            "effective_max_frames": int(T_gen),
                            "live_qwen_text_token_count": int(
                                live_qwen_text_stream.observed_qwen_token_count
                            ),
                        })
                # The text-derived budget is only an initial estimate. Once
                # the live text EOS has crossed Q1 and the source FIFO is fully
                # drained, keep sampling the real codec stream until its EOS
                # token or the configured hard cap. Otherwise the one final
                # Q1 could predict an ordinary audio code which this old
                # budget-exhaustion path would silently discard.
                if (
                    self.allow_talker_eos
                    and live_qwen_text_stream.codec_eos_allowed
                ):
                    if T_gen < live_eos_frame_cap:
                        T_gen = live_eos_frame_cap
                    if metrics is not None:
                        metrics.update({
                            "acoustic_frame_budget": int(T_gen),
                            "effective_max_frames": int(T_gen),
                            "live_text_eos_drained_frame_cap": int(live_eos_frame_cap),
                        })
            if frame_id >= T_gen:
                break
            if talk_stopped():
                break
            admit_listener("talker", "codec_head")
            if talk_stopped():
                break
            timing_pair = begin_cuda_timing("codec_head")
            lg = run_talker_gpu_step(
                "codec_head",
                lambda: t.codec_head(last_h)[:, -1, :],
            )
            finish_cuda_timing("codec_head", timing_pair)
            if not torch.isfinite(lg).all():
                raise RuntimeError(f"Talker codec_head non finito al frame {len(codes)}")
            allow_codec_eos = self._codec_eos_allowed(len(codes))
            if live_qwen_text_stream is not None:
                allow_codec_eos = allow_codec_eos and live_qwen_text_stream.codec_eos_allowed
            c0 = self._cb0_sample(lg, allow_eos=allow_codec_eos)
            if eos_token_id is not None and c0 == eos_token_id:
                eos_detected = True
                eos_frame_id = int(frame_id)
                if metrics is not None:
                    metrics.update({
                        "eos_detected": True,
                        "eos_frame_id": eos_frame_id,
                    })
                break
            # Sub-talker: CUDA graph (ADR-026, 15 residui in ~12ms) oppure
            # fallback eager bit-exact (stesso warper order di transformers).
            timing_pair = begin_cuda_timing("subtalker")
            def generate_residuals() -> Any:
                c0_embed = t.get_input_embeddings()(torch.tensor([[c0]], device=self.device))
                graph = (
                    talk_context.subtalker_graph
                    if talk_context is not None else self.subtalker_graph
                )
                if metrics is not None:
                    metrics["subtalker_graph_calls"] = metrics.get("subtalker_graph_calls", 0) + 1
                    metrics["subtalker_graph_captured"] = bool(getattr(graph, "captured", False))
                    metrics["subtalker_graph_backend"] = (
                        "cuda_graph" if metrics["subtalker_graph_captured"] else "eager"
                    )
                return graph.generate(
                    last_h, c0, c0_embed=c0_embed, return_tensor=True,
                )

            resid_output = run_talker_gpu_step("subtalker", generate_residuals)
            finish_cuda_timing("subtalker", timing_pair)
            if (
                frame_residual_response_hidden is not None
                or live_frame_residual_hidden_stream is not None
            ) and isinstance(resid_output, torch.Tensor):
                residual_codes = resid_output.reshape(-1)
                if residual_codes.numel() != 15:
                    raise RuntimeError(
                        f"SubTalker ha prodotto {residual_codes.numel()} residui, attesi 15"
                    )
                frame_codes = torch.cat((
                    torch.tensor([int(c0)], device=self.device, dtype=torch.long),
                    residual_codes.to(device=self.device, dtype=torch.long),
                ), dim=0)
                resid = None
            else:
                if isinstance(resid_output, torch.Tensor):
                    resid = [int(value) for value in resid_output.detach().cpu().tolist()]
                else:
                    resid = [int(value) for value in resid_output]
                frame_codes = [int(c0)] + [int(value) for value in resid]
            if getattr(self, "_warmup_diagnostic_active", False):
                self._warmup_debug_last_cb0_logits = lg.detach()
                self._warmup_debug_last_cb0_id = int(c0)
                last_residual_codes = (
                    frame_codes[1:].detach().cpu().tolist()
                    if isinstance(frame_codes, torch.Tensor) else list(resid)
                )
                self._warmup_debug_last_residual_codes = list(last_residual_codes)
            frame_code_count = frame_codes.numel() if isinstance(frame_codes, torch.Tensor) else len(frame_codes)
            if frame_code_count != 16:
                raise RuntimeError(f"SubTalker ha prodotto {frame_code_count - 1} residui, attesi 15")
            codes.append(frame_codes)
            current = state is None or generation_id is None or state.is_generation_current(generation_id=generation_id)
            if not current:
                state.stale_kv_writes += 1
                accumulator.cancel()
                break
            if on_cb0_frame is not None:
                on_cb0_frame(frame_id, time.monotonic_ns())
            if streaming_requested:
                for decoded_chunk in accumulator.push(frame_id, frame_codes):
                    if talk_stopped():
                        accumulator.cancel()
                        break
                    emit_decoded_chunk(decoded_chunk)
            if len(codes) >= T_gen:
                break
            last_codes = codes[-1]
            cur = (
                last_codes.view(1, 1, -1)
                if isinstance(last_codes, torch.Tensor)
                else torch.tensor(last_codes, device=self.device, dtype=torch.long).view(1, 1, -1)
            )
            qwen_trailing_text_hidden = trailing_text_hidden
            qwen_generation_step = len(codes) - 1
            live_text_token_pending = False
            if live_qwen_text_stream is not None:
                next_text_id = live_qwen_text_stream.take_trailing_token(
                    stop_check=talk_stopped,
                )
                if next_text_id is None:
                    qwen_trailing_text_hidden = None
                    qwen_generation_step = None
                else:
                    live_text_token_pending = True
                    text_id_tensor = torch.tensor(
                        [[int(next_text_id)]],
                        device=self.device,
                        dtype=torch.long,
                    )
                    qwen_trailing_text_hidden = t.text_projection(
                        t.get_text_embeddings()(text_id_tensor)
                    )
                    # Live input advances its own cursor once per Q1 query.
                    qwen_generation_step = 0
            nxt = build_qwen_audio_conditioning(
                t,
                cur,
                tts_pad_embed=tts_pad_embed,
                trailing_text_hidden=qwen_trailing_text_hidden,
                generation_step=qwen_generation_step,
            )
            nxt = apply_frame_residual(nxt, generation_step=len(codes) - 1)
            admit_listener("talker", "next_qwen_model")
            if talk_stopped():
                break
            timing_pair = begin_cuda_timing("next_qwen_model")
            if native_executor is not None:
                q1_hidden = run_talker_gpu_step(
                    "next_qwen_model",
                    lambda: self._native_talker_q1_step(
                        native_executor,
                        nxt,
                        metrics=metrics,
                    ),
                )
                q1_cache = past_kv
            else:
                q1_hidden, q1_cache = run_talker_gpu_step(
                    "next_qwen_model",
                    lambda: self._qwen_q1_forward(
                        t.model,
                        nxt,
                        past_kv,
                        metrics=metrics,
                    ),
                )
            finish_cuda_timing("next_qwen_model", timing_pair)
            past_kv = q1_cache
            if talk_context is not None:
                talk_context.qwen_kv = past_kv
            if live_text_token_pending and live_qwen_text_stream is not None:
                ack_trailing_text = getattr(live_qwen_text_stream, "ack_trailing_token", None)
                if callable(ack_trailing_text):
                    # Keep the packet head and its cursor in flight until Qwen
                    # has actually committed this text ID through Q1.
                    ack_trailing_text()
            last_h = q1_hidden
            frame_id += 1

        if (
            live_qwen_text_stream is not None
            and not talk_stopped()
            and len(codes) >= T_gen
            and not live_qwen_text_stream.codec_eos_allowed
        ):
            capacity_error = RuntimeError(
                "live SAY exhausted ARIA_MAX_FRAMES before Qwen text EOS and tail drain"
            )
            live_qwen_text_stream.fail(capacity_error)
            raise capacity_error

        # The legacy path also reserves one non-acoustic EOS query.  It uses
        # the final committed code as input and never appends to `codes`.
        if (
            len(codes) >= T_gen
            and not eos_detected
            and self.allow_talker_eos
            and codes
            and (
                live_qwen_text_stream is None
                or live_qwen_text_stream.codec_eos_allowed
            )
        ):
            last_codes = codes[-1]
            cur = (
                last_codes.view(1, 1, -1)
                if isinstance(last_codes, torch.Tensor)
                else torch.tensor(last_codes, device=self.device, dtype=torch.long).view(1, 1, -1)
            )
            terminal_input = build_qwen_audio_conditioning(
                t,
                cur,
                tts_pad_embed=tts_pad_embed,
                trailing_text_hidden=trailing_text_hidden,
                generation_step=len(codes) - 1,
            )
            terminal_input = apply_frame_residual(
                terminal_input, generation_step=len(codes) - 1
            )
            if native_executor is not None:
                terminal_hidden = run_talker_gpu_step(
                    "terminal_qwen_model",
                    lambda: self._native_talker_q1_step(
                        native_executor,
                        terminal_input,
                        metrics=metrics,
                    ),
                )
            else:
                terminal_hidden, past_kv = run_talker_gpu_step(
                    "terminal_qwen_model",
                    lambda: self._qwen_q1_forward(
                        t.model,
                        terminal_input,
                        past_kv,
                        metrics=metrics,
                    ),
                )
            terminal_queries = 1
            terminal_logits = run_talker_gpu_step(
                "terminal_codec_head",
                lambda: t.codec_head(terminal_hidden)[:, -1, :],
            )
            terminal_c0 = self._cb0_sample(terminal_logits, allow_eos=True)
            if eos_token_id is not None and terminal_c0 == eos_token_id:
                eos_detected = True
                eos_frame_id = int(len(codes))
                if metrics is not None:
                    metrics.update({"eos_detected": True, "eos_frame_id": eos_frame_id})

        if native_executor is not None:
            # Persist the slot graph across responses, but fence it before
            # Code2Wav flush and before the pool may lease this slot again.
            if metrics is not None and callable(getattr(native_executor, "q1_graph_info", None)):
                metrics["native_talker_q1_graph_runtime"] = native_executor.q1_graph_info()
            native_executor.close()
            native_executor = None
            if talk_context is not None:
                talk_context.native_q1_executor = None

        cancelled = talk_stopped()
        if cancelled:
            accumulator.cancel()
            active_pcm_samples = int(sum(audio.size for audio in streamed_audio))
            if metrics is not None:
                metrics.update({
                    "generated_cb0_frames": len(codes),
                    "generated_pcm_samples": active_pcm_samples,
                    "active_pcm_samples": active_pcm_samples,
                    "active_pcm_duration_ms": active_pcm_samples * 1000.0 / 24000.0,
                    "rvq_frames": len(codes),
                    "eos_detected": eos_detected,
                    "eos_frame_id": eos_frame_id,
                    "terminated_by_budget": False,
                    "terminated_by": "cancel",
                    "terminal_queries": terminal_queries,
                    "cancelled": True,
                })
            log_private_cuda_timing()
            return np.zeros(0, dtype=np.float32)
        if streaming_requested:
            for decoded_chunk in accumulator.flush(clean_end=True):
                if talk_stopped():
                    accumulator.cancel()
                    break
                emit_decoded_chunk(decoded_chunk)
            if accumulator.cancelled:
                active_pcm_samples = int(sum(audio.size for audio in streamed_audio))
                if metrics is not None:
                    metrics.update({
                        "generated_cb0_frames": len(codes),
                        "generated_pcm_samples": active_pcm_samples,
                        "active_pcm_samples": active_pcm_samples,
                        "active_pcm_duration_ms": active_pcm_samples * 1000.0 / 24000.0,
                        "rvq_frames": len(codes),
                        "terminated_by": "cancel",
                        "terminated_by_budget": False,
                        "cancelled": True,
                    })
                log_private_cuda_timing()
                return np.zeros(0, dtype=np.float32)
            wav = np.concatenate(streamed_audio) if streamed_audio else np.zeros(0, dtype=np.float32)
        else:
            wav = decode_chunk(codes) if codes else np.zeros(0, dtype=np.float32)
        if wav.size and not np.isfinite(wav).all():
            raise RuntimeError("Code2Wav ha prodotto audio non finito")
        output = self._postprocess(wav) if wav.size else wav
        if isinstance(quality_diagnostic, dict) and codes and output.size:
            try:
                capture = quality_diagnostic.get("capture")
                if not callable(capture):
                    raise TypeError("diagnostic capture callback missing")

                def diagnostic_array(value: Any, *, dtype: Any = np.float32) -> np.ndarray:
                    if isinstance(value, torch.Tensor):
                        value = value.detach().to(device="cpu", dtype=torch.float32).numpy()
                    elif (
                        isinstance(value, (list, tuple))
                        and value
                        and isinstance(value[0], torch.Tensor)
                    ):
                        value = torch.stack(list(value), dim=0).detach().to(
                            device="cpu", dtype=torch.float32
                        ).numpy()
                    return np.ascontiguousarray(np.asarray(value, dtype=dtype))

                metadata = dict(quality_diagnostic.get("metadata") or {})
                if capture_preclamp:
                    metadata["code2wav_preclamp"] = preclamp_summary()
                metadata.update({
                        "generated_cb0_frames": len(codes),
                        "eos_detected": bool(eos_detected),
                        "eos_frame_id": eos_frame_id,
                        "terminated_by_budget": bool(not eos_detected and len(codes) >= T_gen),
                        "talker_q1_backend": (
                            "native_audio_q1" if native_v2_talker else
                            "torch_compile_dynamic" if compiled_v2_talker else "eager"
                        ),
                        "acoustic_frame_budget": int(T_gen),
                        "text_derived_acoustic_frame_budget": (
                            metrics.get("requested_acoustic_frame_budget")
                            if metrics is not None else None
                        ),
                        "diagnostic_max_acoustic_frame_budget": (
                            metrics.get("diagnostic_max_acoustic_frame_budget")
                            if metrics is not None else None
                        ),
                    })
                if metrics is not None:
                    metadata["talker_metrics"] = {
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (str, bool, int))
                        or (isinstance(value, float) and np.isfinite(value))
                    }
                    metadata["talker_runtime_summary"] = {
                        key: metrics[key]
                        for key in (
                            "native_talker_q1_calls",
                            "compiled_talker_q1_calls",
                            "eos_detected",
                            "eos_frame_id",
                            "generated_cb0_frames",
                            "generated_pcm_samples",
                            "terminated_by_budget",
                            "terminated_by",
                            "rvq_frames",
                            "active_pcm_samples",
                            "active_pcm_duration_ms",
                            "effective_min_frames",
                            "effective_max_frames",
                        )
                        if key in metrics
                    }
                capture({
                    "metadata": metadata,
                    "audio_context_hidden": diagnostic_array(
                        quality_diagnostic.get("audio_context_hidden")
                    ) if quality_diagnostic.get("audio_context_hidden") is not None else None,
                    "semantic_anchor_hidden": diagnostic_array(
                        quality_diagnostic.get("semantic_anchor_hidden")
                    ) if quality_diagnostic.get("semantic_anchor_hidden") is not None else None,
                    "say_hidden": diagnostic_array(quality_diagnostic.get("say_hidden")),
                    "v9_prefix": diagnostic_array(h_pfx),
                    "rvq_codes": diagnostic_array(codes, dtype=np.int32),
                    "preclamp_pcm": (
                        diagnostic_array(np.concatenate(preclamp_audio))
                        if preclamp_audio else None
                    ),
                    "streamed_pcm": diagnostic_array(
                        np.concatenate(streamed_audio) if streamed_audio else np.empty(0, dtype=np.float32)
                    ),
                    "final_pcm": diagnostic_array(output),
                })
            except Exception as exc:
                logger.warning("[tts quality diagnostic] capture failed: %s", str(exc)[:240])
        if metrics is not None:
            active_pcm_samples = (
                int(sum(audio.size for audio in streamed_audio))
                if streaming_requested else int(wav.size)
            )
            terminated_by = (
                "eos" if eos_detected
                else "budget" if len(codes) >= T_gen
                else "cancel"
            )
            metrics.update({
                "generated_cb0_frames": len(codes),
                "generated_pcm_samples": int(len(output)),
                "active_pcm_samples": active_pcm_samples,
                "active_pcm_duration_ms": active_pcm_samples * 1000.0 / 24000.0,
                "rvq_frames": len(codes),
                "eos_detected": eos_detected,
                "eos_frame_id": eos_frame_id,
                "terminated_by_budget": (
                    not eos_detected and len(codes) >= T_gen
                ),
                "terminated_by": terminated_by,
                "effective_min_frames": int(min_frames),
                "effective_max_frames": int(T_gen),
                "terminal_queries": terminal_queries,
            })
        finalize_frame_residual_metrics()
        log_private_cuda_timing()
        return output

    @staticmethod
    def _postprocess(wav: np.ndarray) -> np.ndarray:
        """Apply the transparent linear output cleanup used by production serving.

        Level control uses one scalar for the whole clip;
        there is no sample-wise waveshaping, artificial high-shelf filtering,
        or nonlinear distortion.
        """
        sample_rate = 24000
        signal = np.asarray(wav, dtype=np.float32).reshape(-1).copy()
        if signal.size == 0:
            return signal
        if not np.isfinite(signal).all():
            signal = np.nan_to_num(signal, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Detect the last active 20 ms frame against an adaptive noise estimate,
        # then retain a 250 ms reverb tail.
        frame_samples = max(1, int(round(0.020 * sample_rate)))
        frame_rms = np.asarray(
            [
                np.sqrt(np.mean(np.square(signal[offset : offset + frame_samples], dtype=np.float64)))
                for offset in range(0, signal.size, frame_samples)
            ],
            dtype=np.float64,
        )
        if frame_rms.size:
            noise_floor = float(np.percentile(frame_rms, 20.0))
            activity_threshold = max(0.0025, noise_floor * 2.5)
            active_frames = np.flatnonzero(frame_rms > activity_threshold)
            if active_frames.size:
                tail_samples = int(round(0.250 * sample_rate))
                active_end = min(signal.size, (int(active_frames[-1]) + 1) * frame_samples)
                signal = signal[: min(signal.size, active_end + tail_samples)].copy()

        # Cosine fades soften plosive/click edges and leave a 250 ms tail.
        fade_in_samples = min(signal.size, int(round(0.030 * sample_rate)))
        if fade_in_samples:
            phase = np.linspace(0.0, np.pi, fade_in_samples, endpoint=True, dtype=np.float64)
            signal[:fade_in_samples] *= (0.5 - 0.5 * np.cos(phase)).astype(np.float32)
        fade_out_samples = min(signal.size, int(round(0.250 * sample_rate)))
        if fade_out_samples:
            phase = np.linspace(0.0, np.pi, fade_out_samples, endpoint=True, dtype=np.float64)
            signal[-fade_out_samples:] *= (0.5 + 0.5 * np.cos(phase)).astype(np.float32)

        peak = float(np.max(np.abs(signal))) if signal.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)))) if signal.size else 0.0
        if peak <= 1e-8 or rms <= 1e-8:
            return np.zeros_like(signal)

        target_rms = 10.0 ** (-14.0 / 20.0)
        ceiling = 10.0 ** (-1.5 / 20.0)
        desired_gain = target_rms / rms
        requested_peak = peak * desired_gain
        knee_start = ceiling * 10.0 ** (-3.0 / 20.0)  # -4.5 dBFS to -1.5 dBFS knee
        if requested_peak <= knee_start:
            gain = desired_gain
        else:
            capped_gain = ceiling / peak
            blend = min(1.0, max(0.0, (requested_peak - knee_start) / (ceiling - knee_start)))
            blend = blend * blend * (3.0 - 2.0 * blend)
            gain = desired_gain * (1.0 - blend) + capped_gain * blend
        signal *= np.float32(gain)
        return signal.astype(np.float32, copy=False)

    def unload(self) -> None:
        self.loaded = False
        for executor in self._native_q1_slot_executors.values():
            try:
                executor.close()
            except Exception:
                logger.exception("[engine] native Q1 graph slot close failed")
        self._native_q1_slot_executors.clear()
        self.subtalker_graph = None
        self.response_talk_context_pool = None
        self.response_talk_context_memory = None
        self.listener = None
        self.thinker = self._thinker_wrapper = self.talker = self.v9 = self.causal_conv = None
        self.v9_stream = None
        torch.cuda.empty_cache()


__all__ = ["PipelineEngine", "V9PrefixSequence"]
