#!/usr/bin/env python3
"""Compare offline and causal 80 ms Gemma audio-tower features on a real WAV.

Run inside the existing Mondaini listener container with its server venv.  The
script loads only ``Gemma4AudioModel`` from the local Gemma checkpoint and
never starts, stops, or modifies the serving process or any checkpoint.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np


DEFAULT_INPUT = Path("/workspace/artifacts/conversation_quality_20261014/inputs/clean_initial.wav")
DEFAULT_OUTPUT = Path("/workspace/artifacts/causal_audio_tower_benchmark/benchmark.json")
DEFAULT_WORKSPACE = Path("/workspace")
DEFAULT_RUNTIME_SOURCE = Path("/workspace/artifacts/runtime/listener_microbatch_source")
SAMPLE_RATE = 16_000
TICK_MS = 80
SAMPLES_PER_TICK = SAMPLE_RATE * TICK_MS // 1000
MEL_BINS = 128
MEL_ROWS_PER_TICK = 8
WINDOW_ROWS = 60
HISTORY_ROWS = WINDOW_ROWS - MEL_ROWS_PER_TICK
TOKENS_PER_TICK = 2
Q_START = WINDOW_ROWS // 4 - TOKENS_PER_TICK
COSINE_TARGET = 0.995


def _stats(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _feature_stats(offline: np.ndarray, streaming: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "offline_shape": list(offline.shape),
        "streaming_shape": list(streaming.shape),
        "same_shape": bool(offline.shape == streaming.shape),
    }
    if offline.shape != streaming.shape:
        return result
    delta = np.asarray(streaming, dtype=np.float64) - np.asarray(offline, dtype=np.float64)
    offline64 = np.asarray(offline, dtype=np.float64)
    streaming64 = np.asarray(streaming, dtype=np.float64)
    flat_left = offline64.reshape(-1)
    flat_right = streaming64.reshape(-1)
    denominator = float(np.linalg.norm(flat_left) * np.linalg.norm(flat_right))
    row_denominator = np.linalg.norm(offline64, axis=1) * np.linalg.norm(streaming64, axis=1)
    row_cosine = np.divide(
        np.sum(offline64 * streaming64, axis=1),
        row_denominator,
        out=np.ones_like(row_denominator),
        where=row_denominator > 0,
    )
    result.update({
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "exact_equal_elements": int(np.count_nonzero(delta == 0.0)),
        "element_count": int(delta.size),
        "whole_matrix_cosine": (
            float(np.dot(flat_left, flat_right) / denominator) if denominator > 0 else None
        ),
        "per_row_cosine": {
            "mean": float(np.mean(row_cosine)),
            "p05": float(np.percentile(row_cosine, 5)),
            "p50": float(np.percentile(row_cosine, 50)),
            "min": float(np.min(row_cosine)),
        },
        "rows_with_abs_delta_over_1e_5": int(np.count_nonzero(np.max(np.abs(delta), axis=1) > 1e-5)),
        "first_differing_rows_over_1e_5": np.flatnonzero(
            np.max(np.abs(delta), axis=1) > 1e-5
        )[:20].astype(int).tolist(),
    })
    return result


def _output_hidden(output: Any):
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, (tuple, list)) and output:
        hidden = output[0]
    if hidden is None or getattr(hidden, "ndim", None) != 3:
        raise RuntimeError("Gemma4AudioModel did not return rank-3 last_hidden_state")
    return hidden


def _one_forward(model: Any, features: Any, mask: Any, torch: Any):
    with torch.inference_mode():
        return model(input_features=features, attention_mask=mask, return_dict=True)


def _timed_forward(model: Any, features: Any, mask: Any, torch: Any) -> tuple[Any, float, float]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    started_ns = time.perf_counter_ns()
    start_event.record()
    output = _one_forward(model, features, mask, torch)
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    return output, float(start_event.elapsed_time(end_event)), wall_ms


def _capture_graph(model: Any, static_features: Any, static_mask: Any, torch: Any, warmup: int):
    """Warm up and capture the exact eager forward with fixed input buffers."""
    device = static_features.device
    capture_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device=device)
    capture_stream.wait_stream(current_stream)
    with torch.cuda.stream(capture_stream):
        for _ in range(max(1, warmup)):
            _one_forward(model, static_features, static_mask, torch)
    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    capture_started_ns = time.perf_counter_ns()
    with torch.cuda.graph(graph, stream=capture_stream):
        static_output = _one_forward(model, static_features, static_mask, torch)
        static_hidden = _output_hidden(static_output)
    torch.cuda.synchronize(device)
    capture_wall_ms = (time.perf_counter_ns() - capture_started_ns) / 1_000_000.0
    return graph, static_hidden, capture_wall_ms


def _graph_replay_timing(graph: Any, torch: Any, device: Any) -> tuple[float, float]:
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start_ns = time.perf_counter_ns()
    start_event.record()
    graph.replay()
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_start_ns) / 1_000_000.0
    return float(start_event.elapsed_time(end_event)), wall_ms


def _graph_copy_replay_timing(graph: Any, static_features: Any, source_features: Any,
                              torch: Any, device: Any) -> tuple[float, float]:
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start_ns = time.perf_counter_ns()
    start_event.record()
    static_features.copy_(source_features)
    graph.replay()
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_start_ns) / 1_000_000.0
    return float(start_event.elapsed_time(end_event)), wall_ms


def _graph_copy_mask_replay_timing(graph: Any, static_features: Any, static_mask: Any,
                                   source_features: Any, source_mask: Any,
                                   torch: Any, device: Any) -> tuple[float, float]:
    """Measure GPU staging-buffer copies plus replay, including mask updates."""
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start_ns = time.perf_counter_ns()
    start_event.record()
    static_features.copy_(source_features)
    static_mask.copy_(source_mask)
    graph.replay()
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_start_ns) / 1_000_000.0
    return float(start_event.elapsed_time(end_event)), wall_ms


def _measure_graph_shape(model: Any, input_array: np.ndarray, eager_hidden: np.ndarray,
                         iterations: int, warmup: int, torch: Any, device: Any,
                         input_series: np.ndarray | None = None) -> dict[str, Any]:
    """Measure graph replay; optionally replay a sequence through one static buffer."""
    static_features, static_mask = _tensor_for_features(input_array, model, device, torch)
    graph, graph_hidden, capture_wall_ms = _capture_graph(
        model, static_features, static_mask, torch, warmup
    )
    # Synchronize and compare on the very same input used by eager.
    graph_cuda_ms, graph_wall_ms = _graph_replay_timing(graph, torch, device)
    graph_hidden_np = graph_hidden[0].float().cpu().numpy()
    eager_vs_graph = _feature_stats(eager_hidden, graph_hidden_np)

    replay_cuda: list[float] = []
    replay_wall: list[float] = []
    for _ in range(iterations):
        cuda_ms, wall_ms = _graph_replay_timing(graph, torch, device)
        replay_cuda.append(cuda_ms)
        replay_wall.append(wall_ms)

    result: dict[str, Any] = {
        "status": "measured",
        "backend": "torch.cuda.CUDAGraph replay of the same Gemma4AudioModel forward",
        "input_shape": list(static_features.shape),
        "output_shape": list(graph_hidden.shape),
        "static_input_buffers": True,
        "capture_warmup_forwards": max(1, warmup),
        "capture_wall_ms_excluded_from_latency": capture_wall_ms,
        "first_replay_cuda_ms": graph_cuda_ms,
        "first_replay_wall_ms": graph_wall_ms,
        "replay_only_iterations": len(replay_cuda),
        "replay_only_cuda_ms": _stats(replay_cuda),
        "replay_only_wall_ms": _stats(replay_wall),
        "same_input_eager_vs_graph": eager_vs_graph,
    }
    if input_series is not None:
        source_series = torch.as_tensor(input_series, device=device, dtype=static_features.dtype)
        if source_series.ndim != 3 or tuple(source_series.shape[1:]) != tuple(static_features.shape[1:]):
            raise ValueError(
                f"graph input series has shape {tuple(source_series.shape)}, expected [N,{static_features.shape[1]},{static_features.shape[2]}]"
            )
        series_cuda: list[float] = []
        series_wall: list[float] = []
        first_series_hidden: np.ndarray | None = None
        for index in range(int(source_series.shape[0])):
            cuda_ms, wall_ms = _graph_copy_replay_timing(
                graph, static_features, source_series[index : index + 1], torch, device
            )
            series_cuda.append(cuda_ms)
            series_wall.append(wall_ms)
            if index == 0:
                first_series_hidden = graph_hidden[0].float().cpu().numpy()
        result["input_series_count"] = int(source_series.shape[0])
        result["copy_plus_replay_cuda_ms"] = _stats(series_cuda)
        result["copy_plus_replay_wall_ms"] = _stats(series_wall)
        if first_series_hidden is None:
            raise RuntimeError("graph input series had no replayed rows")
        result["input_series_first_output"] = _feature_stats(eager_hidden, first_series_hidden)
        del source_series
    del graph, graph_hidden, static_features, static_mask
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return result


def _read_and_resample(path: Path) -> tuple[np.ndarray, int, int, int]:
    import soundfile as sf
    from scipy.signal import resample_poly

    decoded, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if decoded.size == 0:
        raise ValueError(f"empty WAV: {path}")
    mono = decoded.mean(axis=1, dtype=np.float32)
    source_samples = int(mono.size)
    if int(source_rate) != SAMPLE_RATE:
        ratio = Fraction(SAMPLE_RATE, int(source_rate))
        mono = resample_poly(mono, ratio.numerator, ratio.denominator).astype(np.float32, copy=False)
    target_samples = 125 * SAMPLES_PER_TICK
    if mono.size < target_samples:
        raise ValueError(
            f"input has {mono.size} resampled samples; need {target_samples} for 125 complete ticks"
        )
    return np.asarray(mono[:target_samples], dtype=np.float32), int(source_rate), source_samples, int(mono.size)


def _load_runtime_preprocessor(workspace: Path, source_root: Path):
    # The active service imports this runtime overlay. Put it first so the
    # incremental extractor below comes from the same serving source tree.
    sys.path.insert(0, str(source_root))
    sys.path.insert(1, str(workspace))
    from duplex_kernel.listener import GemmaAudioTowerBackend
    from duplex_kernel.models.thinker import resolve_thinker_path
    import transformers

    model_path = Path(resolve_thinker_path(str(workspace)))
    processor = transformers.AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    extractor = getattr(processor, "feature_extractor", processor)
    adapter = object.__new__(GemmaAudioTowerBackend)
    adapter.feature_extractor = extractor
    return model_path, processor, extractor, adapter


def _extract_mels(adapter: Any, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    offline = adapter._extract([waveform])[0]
    if offline.ndim != 2 or offline.shape[1] != MEL_BINS:
        raise RuntimeError(f"offline extractor returned unexpected shape {offline.shape}")
    tail: np.ndarray | None = None
    chunks: list[np.ndarray] = []
    for tick in range(125):
        start = tick * SAMPLES_PER_TICK
        pcm = waveform[start : start + SAMPLES_PER_TICK]
        rows, tail = adapter._extract_single_frame(pcm, tail)
        if rows.shape != (MEL_ROWS_PER_TICK, MEL_BINS):
            raise RuntimeError(f"streaming extractor tick {tick} returned {rows.shape}")
        chunks.append(np.asarray(rows, dtype=np.float32))
    streaming = np.concatenate(chunks, axis=0)
    return np.asarray(offline, dtype=np.float32), streaming, chunks


def _stream_windows(chunks: list[np.ndarray]):
    history: deque[np.ndarray] = deque(maxlen=HISTORY_ROWS)
    windows: list[np.ndarray] = []
    for chunk in chunks:
        prior = np.stack(tuple(history), axis=0) if history else np.empty((0, MEL_BINS), dtype=np.float32)
        prior = prior[-HISTORY_ROWS:]
        if prior.shape[0] < HISTORY_ROWS:
            prior = np.pad(
                prior,
                ((HISTORY_ROWS - prior.shape[0], 0), (0, 0)),
                mode="constant",
            )
        window = np.concatenate((prior, chunk), axis=0)
        if window.shape != (WINDOW_ROWS, MEL_BINS):
            raise RuntimeError(f"causal window has unexpected shape {window.shape}")
        windows.append(np.asarray(window, dtype=np.float32))
        history.extend(chunk)
    return windows


def _window_series(chunks: list[np.ndarray], window_rows: int) -> tuple[list[np.ndarray], list[int]]:
    """Build rolling user-Mel windows, recording how many leading rows are padding."""
    history_rows = window_rows - MEL_ROWS_PER_TICK
    if history_rows < 0:
        raise ValueError(f"window_rows={window_rows} is shorter than one {MEL_ROWS_PER_TICK}-row tick")
    history: deque[np.ndarray] = deque(maxlen=history_rows or None)
    windows: list[np.ndarray] = []
    leading_padding: list[int] = []
    for chunk in chunks:
        prior = np.stack(tuple(history), axis=0) if history else np.empty((0, MEL_BINS), dtype=np.float32)
        if history_rows:
            prior = prior[-history_rows:]
        pad_rows = history_rows - prior.shape[0]
        if pad_rows:
            prior = np.pad(prior, ((pad_rows, 0), (0, 0)), mode="constant")
        window = np.concatenate((prior, chunk), axis=0)
        if window.shape != (window_rows, MEL_BINS):
            raise RuntimeError(f"{window_rows}-row window has unexpected shape {window.shape}")
        windows.append(np.asarray(window, dtype=np.float32))
        leading_padding.append(int(pad_rows))
        history.extend(chunk)
    return windows, leading_padding


def _load_audio_tower(model_path: Path, torch: Any, transformers: Any):
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel

    config = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    audio_config = getattr(config, "audio_config", None)
    if audio_config is None:
        raise RuntimeError("local Gemma checkpoint has no audio_config")
    model, loading_info = Gemma4AudioModel.from_pretrained(
        str(model_path),
        config=audio_config,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    model.eval()
    return model, config, audio_config, loading_info


def _tower_key(name: str) -> bool:
    lowered = str(name).lower()
    return "audio_tower" in lowered or lowered.startswith("audio_model.") or lowered.startswith("layers.")


def _tensor_for_features(array: np.ndarray, model: Any, device: Any, torch: Any):
    parameter = next(model.parameters())
    features = torch.as_tensor(array, device=device, dtype=parameter.dtype).unsqueeze(0).contiguous()
    mask = torch.ones((1, array.shape[0]), device=device, dtype=torch.bool)
    return features, mask


def _cosine_summary(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    left = torch.as_tensor(reference, dtype=torch.float32)
    right = torch.as_tensor(candidate, dtype=torch.float32)
    if left.shape != right.shape:
        return {"same_shape": False, "reference_shape": list(left.shape), "candidate_shape": list(right.shape)}
    per_token = F.cosine_similarity(left, right, dim=-1).numpy().reshape(-1)
    return {
        "same_shape": True,
        "shape": list(left.shape),
        "token_count": int(per_token.size),
        "mean": float(np.mean(per_token)),
        "p05": float(np.percentile(per_token, 5)),
        "p50": float(np.percentile(per_token, 50)),
        "p95": float(np.percentile(per_token, 95)),
        "min": float(np.min(per_token)),
        "max": float(np.max(per_token)),
        "count_below_0_995": int(np.count_nonzero(per_token < COSINE_TARGET)),
        "fraction_at_or_above_0_995": float(np.mean(per_token >= COSINE_TARGET)),
        "target_mean_at_or_above_0_995": bool(float(np.mean(per_token)) >= COSINE_TARGET),
    }


def _tick_phase_indices(window_rows: int, tick_count: int) -> tuple[int, list[int], list[int]]:
    history_rows = window_rows - MEL_ROWS_PER_TICK
    steady_start = (history_rows + MEL_ROWS_PER_TICK - 1) // MEL_ROWS_PER_TICK
    startup = list(range(min(steady_start, tick_count)))
    steady = list(range(min(steady_start, tick_count), tick_count))
    return steady_start, startup, steady


def _phase_metric(reference: np.ndarray, outputs: list[np.ndarray], indices: list[int],
                  reference_tick_indices: list[int] | None = None) -> dict[str, Any]:
    if not indices:
        return {"tick_count": 0, "cosine": None}
    if reference_tick_indices is None:
        reference_tick_indices = list(range(len(outputs)))
    reference_tokens = np.concatenate([
        reference[2 * reference_tick_indices[index] : 2 * reference_tick_indices[index] + 2]
        for index in indices
    ], axis=0)
    candidate_tokens = np.concatenate([outputs[index] for index in indices], axis=0)
    return {"tick_count": len(indices), "cosine": _cosine_summary(reference_tokens, candidate_tokens)}


def _measure_sweep_graph(model: Any, window_rows: int, windows: list[np.ndarray],
                         leading_padding: list[int], offline_tokens: np.ndarray,
                         mask_mode: str, iterations: int, warmup: int, smoke_ticks: int,
                         torch: Any, device: Any,
                         reference_tick_indices: list[int] | None = None) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Replay one static-shape graph over real rolling windows and return 250 Q2 tokens."""
    parameter = next(model.parameters())
    source_numpy = np.stack(windows, axis=0)
    source_features = torch.as_tensor(source_numpy, device=device, dtype=parameter.dtype).contiguous()
    source_masks_numpy = np.ones((len(windows), window_rows), dtype=np.bool_)
    if mask_mode == "false_invalid_leading":
        for index, pad_rows in enumerate(leading_padding):
            if pad_rows:
                source_masks_numpy[index, :pad_rows] = False
    elif mask_mode != "all_true":
        raise ValueError(f"unknown mask_mode={mask_mode}")
    source_masks = torch.as_tensor(source_masks_numpy, device=device, dtype=torch.bool).contiguous()
    static_features = source_features[0:1].clone().contiguous()
    static_mask = source_masks[0:1].clone().contiguous()
    graph, graph_hidden, capture_wall_ms = _capture_graph(model, static_features, static_mask, torch, warmup)
    expected_tokens = window_rows // 4
    q_start = expected_tokens - TOKENS_PER_TICK
    if tuple(graph_hidden.shape) != (1, expected_tokens, int(graph_hidden.shape[-1])):
        raise RuntimeError(f"graph W={window_rows} returned unexpected output shape {tuple(graph_hidden.shape)}")

    # Check graph/eager parity on the exact first window before measuring a series.
    eager_hidden = _output_hidden(_one_forward(model, source_features[0:1], source_masks[0:1], torch))
    _graph_copy_mask_replay_timing(
        graph, static_features, static_mask, source_features[0:1], source_masks[0:1], torch, device
    )
    eager_vs_graph = _feature_stats(eager_hidden[0].float().cpu().numpy(), graph_hidden[0].float().cpu().numpy())

    # Small graph smoke precedes the full 125-tick pass.
    smoke_shapes: list[list[int]] = []
    for index in range(min(smoke_ticks, len(windows))):
        _graph_copy_mask_replay_timing(
            graph, static_features, static_mask,
            source_features[index : index + 1], source_masks[index : index + 1], torch, device,
        )
        if not torch.isfinite(graph_hidden).all().item():
            raise RuntimeError(f"non-finite graph output in smoke tick {index} for W={window_rows}")
        smoke_shapes.append(list(graph_hidden.shape))

    replay_cuda: list[float] = []
    replay_wall: list[float] = []
    for _ in range(iterations):
        cuda_ms, wall_ms = _graph_replay_timing(graph, torch, device)
        replay_cuda.append(cuda_ms)
        replay_wall.append(wall_ms)

    output_pairs: list[np.ndarray] = []
    copy_cuda: list[float] = []
    copy_wall: list[float] = []
    for index in range(len(windows)):
        cuda_ms, wall_ms = _graph_copy_mask_replay_timing(
            graph, static_features, static_mask,
            source_features[index : index + 1], source_masks[index : index + 1], torch, device,
        )
        copy_cuda.append(cuda_ms)
        copy_wall.append(wall_ms)
        output_pairs.append(graph_hidden[0, q_start : q_start + TOKENS_PER_TICK].float().cpu().numpy())
    concat = np.concatenate(output_pairs, axis=0)
    terminal_hidden = graph_hidden[0].float().cpu().numpy()
    if reference_tick_indices is None:
        reference_tick_indices = list(range(len(windows)))
    steady_start, _, _ = _tick_phase_indices(window_rows, max(reference_tick_indices, default=-1) + 1)
    startup_indices = [index for index, tick in enumerate(reference_tick_indices) if tick < steady_start]
    steady_indices = [index for index, tick in enumerate(reference_tick_indices) if tick >= steady_start]
    startup_metric = _phase_metric(offline_tokens, output_pairs, startup_indices, reference_tick_indices)
    steady_metric = _phase_metric(offline_tokens, output_pairs, steady_indices, reference_tick_indices)
    result = {
        "status": "measured",
        "window_rows": window_rows,
        "input_shape": [1, window_rows, MEL_BINS],
        "output_shape": list(graph_hidden.shape),
        "mask_convention": mask_mode,
        "mask_true_rows_per_tick_min": int(source_masks_numpy.sum(axis=1).min()),
        "mask_true_rows_per_tick_max": int(source_masks_numpy.sum(axis=1).max()),
        "leading_padding_rows_first_tick": int(leading_padding[0]),
        "leading_padding_rows_last_tick": int(leading_padding[-1]),
        "q_start": q_start,
        "selected_output_rows": [q_start, q_start + TOKENS_PER_TICK],
        "tick_count": len(windows),
        "reference_tick_indices": [reference_tick_indices[0], reference_tick_indices[-1]] if reference_tick_indices else [],
        "concatenated_output_shape": list(concat.shape),
        "cosine_vs_same_offline_250_tokens": _cosine_summary(offline_tokens, concat),
        "phase_split": {
            "steady_context_first_tick": steady_start,
            "startup": startup_metric,
            "steady_full_history": steady_metric,
            "steady_tick_definition": f"at least {window_rows - MEL_ROWS_PER_TICK} prior real Mel rows; leading padding is zero",
        },
        "latency": {
            "backend": "CUDA Graph replay of Gemma4AudioModel",
            "copy_plus_replay_copy_scope": "GPU staging tensor to static graph feature and mask buffers; excludes CPU-to-GPU staging",
            "replay_only_cuda_ms": _stats(replay_cuda),
            "replay_only_wall_ms": _stats(replay_wall),
            "copy_plus_replay_cuda_ms": _stats(copy_cuda),
            "copy_plus_replay_wall_ms": _stats(copy_wall),
            "p95_replay_cuda_below_0_6ms": bool(np.percentile(replay_cuda, 95) < 0.6),
            "p95_copy_plus_replay_cuda_below_0_6ms": bool(np.percentile(copy_cuda, 95) < 0.6),
        },
        "graph_validation": {
            "capture_warmup_forwards": max(1, warmup),
            "capture_wall_ms_excluded_from_latency": capture_wall_ms,
            "smoke_ticks": len(smoke_shapes),
            "smoke_output_shapes": smoke_shapes,
            "same_input_eager_vs_graph": eager_vs_graph,
        },
    }
    del graph, graph_hidden, eager_hidden, static_features, static_mask, source_features, source_masks
    torch.cuda.synchronize(device)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return result, concat, terminal_hidden


def _measure_prefix_series(model: Any, mel: np.ndarray, offline_tokens: np.ndarray,
                           torch: Any, device: Any) -> tuple[dict[str, Any], np.ndarray]:
    """Measure exact no-padding prefixes, selecting the last two tower rows each tick."""
    parameter = next(model.parameters())
    pairs: list[np.ndarray] = []
    cuda_ms: list[float] = []
    wall_ms: list[float] = []
    lengths: list[int] = []
    for tick in range(125):
        prefix_rows = (tick + 1) * MEL_ROWS_PER_TICK
        prefix = mel[:prefix_rows]
        features = torch.as_tensor(prefix, device=device, dtype=parameter.dtype).unsqueeze(0).contiguous()
        mask = torch.ones((1, prefix_rows), device=device, dtype=torch.bool)
        output, cuda_time, wall_time = _timed_forward(model, features, mask, torch)
        hidden = _output_hidden(output)
        if hidden.shape[1] < TOKENS_PER_TICK:
            raise RuntimeError(f"prefix tick {tick} returned too few tower rows: {tuple(hidden.shape)}")
        pairs.append(hidden[0, -TOKENS_PER_TICK:].float().cpu().numpy())
        cuda_ms.append(cuda_time)
        wall_ms.append(wall_time)
        lengths.append(prefix_rows)
    concat = np.concatenate(pairs, axis=0)
    return ({
        "status": "measured",
        "mask_convention": "all_true; no padding; exact user-Mel prefix [0:(tick+1)*8]",
        "prefix_lengths_mel_rows": {"first": lengths[0], "last": lengths[-1], "step": MEL_ROWS_PER_TICK},
        "selection": "last 2 output rows of each variable-length prefix",
        "tick_count": len(pairs),
        "concatenated_output_shape": list(concat.shape),
        "cosine_vs_same_offline_250_tokens": _cosine_summary(offline_tokens, concat),
        "eager_prefix_forward_cuda_ms": _stats(cuda_ms),
        "eager_prefix_forward_wall_ms": _stats(wall_ms),
        "interpretation_limit": "Each prefix is independently position-encoded from index 0; this isolates left padding/truncation but is not the fixed-window serving contract.",
    }, concat)


def _profile_audio_stages(model: Any, input_array: np.ndarray, mask_array: np.ndarray,
                          torch: Any, device: Any) -> dict[str, Any]:
    """Collect lightweight CUDA-event and host-hook timings for major tower stages."""
    features, mask = _tensor_for_features(input_array, model, device, torch)
    mask.copy_(torch.as_tensor(mask_array[None, :], device=device, dtype=torch.bool))
    stages: list[tuple[str, Any]] = []
    subsample = getattr(model, "subsample_conv_projection", None)
    if subsample is not None:
        stages.append(("subsample_conv_projection", subsample))
    layers = getattr(model, "layers", None)
    if layers is not None:
        stages.extend((f"conformer_layer_{index}", layer) for index, layer in enumerate(layers))
    projection = getattr(model, "output_proj", None)
    if projection is not None:
        stages.append(("output_proj", projection))
    records: dict[str, dict[str, Any]] = {}
    handles = []
    for name, module in stages:
        state: dict[str, Any] = {}
        def pre_hook(_module: Any, _inputs: Any, *, key: str = name, slot: dict[str, Any] = state):
            slot["start"] = torch.cuda.Event(enable_timing=True)
            slot["start"].record()
            slot["host_start_ns"] = time.perf_counter_ns()
        def post_hook(_module: Any, _inputs: Any, _output: Any, *, key: str = name, slot: dict[str, Any] = state):
            slot["end"] = torch.cuda.Event(enable_timing=True)
            slot["end"].record()
            slot["host_ms"] = (time.perf_counter_ns() - slot["host_start_ns"]) / 1_000_000.0
        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
        records[name] = state
    try:
        _timed_forward(model, features, mask, torch)
        torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()
    for state in records.values():
        if "start" in state and "end" in state:
            state["cuda_ms"] = float(state["start"].elapsed_time(state["end"]))
    layer_values = [records[f"conformer_layer_{index}"] for index in range(len(layers or [])) if f"conformer_layer_{index}" in records]
    return {
        "status": "measured",
        "input_shape": list(features.shape),
        "mask_true_rows": int(mask_array.sum()),
        "method": "one eager forward with per-module CUDA events and host hook durations; hooks add diagnostic overhead",
        "event_interpretation": "Eager module event intervals include GPU queue idle/launch-dispatch gaps while Python issues operators; summing them is not pure kernel time and can exceed CUDA Graph replay. Use per-op profiler values for op attribution and graph replay for the serving-path latency comparison.",
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "module_cuda_ms": {
            name: state.get("cuda_ms") for name, state in records.items() if "cuda_ms" in state
        },
        "module_host_hook_ms": {
            name: state.get("host_ms") for name, state in records.items() if "host_ms" in state
        },
        "conformer_layer_cuda_ms_summary": _stats([state["cuda_ms"] for state in layer_values if "cuda_ms" in state]),
        "conformer_layer_host_hook_ms_summary": _stats([state["host_ms"] for state in layer_values if "host_ms" in state]),
        "modules_found": [name for name, _ in stages],
    }


def _write_report(output_path: Path, report: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import soundfile  # noqa: F401
    import torch
    import transformers

    report: dict[str, Any] = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "container": args.container,
            "active_runtime_source": str(args.runtime_source),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": None,
            "backend": "Gemma4AudioModel eager CUDA forward",
            "cuda_graph": "measured below with static feature and mask buffers",
        },
        "input": {"path": str(args.input), "target_sample_rate_hz": SAMPLE_RATE},
        "contract": {
            "ticks": 125,
            "tick_ms": TICK_MS,
            "mel_rows_per_tick": MEL_ROWS_PER_TICK,
            "streaming_window_rows": WINDOW_ROWS,
            "streaming_history_rows": HISTORY_ROWS,
            "startup_padding": "left zeros to 52 historical Mel rows",
            "all_input_mask_rows_valid": True,
            "tower_tokens_per_tick": TOKENS_PER_TICK,
            "selected_q_start": Q_START,
            "selected_rows_per_tick": "[13:15]",
            "concatenated_stream_tokens_expected": 250,
            "cosine_reference_target": COSINE_TARGET,
            "target_streaming_tower_cuda_ms": 0.6,
            "target_interpretation": "reported without changing windows, inputs, or references",
        },
        "notes": [
            "The tower-only process runs inside the existing container; the serving PID and checkpoints are not modified.",
            "The offline feature extractor and incremental 80 ms extractor are both sourced from the active runtime overlay.",
            "A 60-row window is 15 output tokens by stride geometry; conformer context across layers can exceed this geometric overlap.",
        ],
    }

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected runtime")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    report["runtime"]["gpu"] = {
        "name": props.name,
        "total_bytes": int(total_bytes),
        "free_bytes_at_start": int(free_before),
        "device_index": 0,
    }
    report["runtime"]["gpu_memory_guard_min_free_bytes_before_load"] = int(args.min_free_gib * 1024**3)
    if free_before < args.min_free_gib * 1024**3:
        report["status"] = "skipped_insufficient_gpu_headroom"
        report["runtime"]["memory_guard_reason"] = "tower was not loaded to preserve serving headroom"
        return report

    waveform, source_rate, source_samples, resampled_samples = _read_and_resample(args.input)
    report["input"].update({
        "source_rate_hz": source_rate,
        "source_samples": source_samples,
        "source_duration_seconds": source_samples / source_rate,
        "resampled_samples_before_trim": resampled_samples,
        "measured_samples": int(waveform.size),
        "measured_duration_seconds": waveform.size / SAMPLE_RATE,
        "resampler": "scipy.signal.resample_poly, same gcd-ratio path as server WAV upload",
        "peak_abs": float(np.max(np.abs(waveform))),
        "rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
    })

    model_path, processor, extractor, adapter = _load_runtime_preprocessor(args.workspace, args.runtime_source)
    report["model"] = {"path": str(model_path), "processor": type(processor).__name__, "feature_extractor": type(extractor).__name__}
    report["preprocessing"] = {
        "extractor_source_file": str(Path(sys.modules["duplex_kernel.listener"].__file__).resolve()),
        "extractor_source_method": "GemmaAudioTowerBackend._extract and _extract_single_frame",
    }
    offline_mel, streaming_mel, mel_chunks = _extract_mels(adapter, waveform)
    windows = _stream_windows(mel_chunks)
    report["preprocessing"]["offline_vs_streaming_mel"] = _feature_stats(offline_mel, streaming_mel)
    report["preprocessing"]["offline_extractor_config"] = {
        name: getattr(extractor, name, None)
        for name in ("sampling_rate", "feature_size", "frame_length", "hop_length", "fft_length", "preemphasis", "mel_floor")
        if getattr(extractor, name, None) is not None
    }

    audio_model, config, audio_config, loading_info = _load_audio_tower(model_path, torch, transformers)
    parameter_count = sum(int(parameter.numel()) for parameter in audio_model.parameters())
    parameter_bytes = sum(int(parameter.numel() * parameter.element_size()) for parameter in audio_model.parameters())
    missing_keys = [str(key) for key in loading_info.get("missing_keys", ())]
    mismatched_keys = loading_info.get("mismatched_keys", ())
    mismatch_names = [str(item[0] if isinstance(item, (tuple, list)) else item) for item in mismatched_keys]
    unexpected_keys = [str(key) for key in loading_info.get("unexpected_keys", ())]
    report["model"].update({
        "audio_tower_class": type(audio_model).__name__,
        "audio_tower_parameters": parameter_count,
        "audio_tower_parameter_bytes": parameter_bytes,
        "audio_tower_dtype": str(next(audio_model.parameters()).dtype),
        "audio_config": audio_config.to_dict() if hasattr(audio_config, "to_dict") else str(audio_config),
        "weight_load": {
            "expected_state_dict_entries": len(audio_model.state_dict()),
            "missing_key_count": len(missing_keys),
            "missing_key_sample": missing_keys[:20],
            "mismatched_key_count": len(mismatch_names),
            "mismatched_key_sample": mismatch_names[:20],
            "unexpected_key_count": len(unexpected_keys),
            "unexpected_tower_key_count": sum(_tower_key(key) for key in unexpected_keys),
            "unexpected_non_tower_key_count": sum(not _tower_key(key) for key in unexpected_keys),
            "unexpected_key_sample": unexpected_keys[:20],
            "tower_weights_complete": not missing_keys and not mismatch_names,
        },
    })
    modeling_source = Path(
        sys.modules["transformers.models.gemma4.modeling_gemma4"].__file__
    ).resolve()
    report["model"]["audio_tower_temporal_geometry"] = {
        "transformers_modeling_source": str(modeling_source),
        "subsampling_conv_layers": 2,
        "subsampling_kernel_time": 3,
        "subsampling_stride_time": 2,
        "subsampling_padding_time": 1,
        "subsampling_receptive_field_mel_rows": 7,
        "subsampling_output_stride_mel_rows": 4,
        "conformer_layers": int(getattr(audio_config, "num_hidden_layers", 12)),
        "attention_context_left_tokens": int(getattr(audio_config, "attention_context_left", 13)),
        "attention_context_right_tokens": int(getattr(audio_config, "attention_context_right", 0)),
        "attention_chunk_tokens": int(getattr(audio_config, "attention_chunk_size", 12)),
        "lightconv_depthwise_groups": int(getattr(audio_config, "hidden_size", 1024)),
        "lightconv_kernel_tokens": int(getattr(audio_config, "conv_kernel_size", 5)),
        "lightconv_causal_left_context_tokens": int(getattr(audio_config, "conv_kernel_size", 5)) - 1,
        "interpretation": (
            "A 60-Mel-row input geometrically yields 15 tokens, but each conformer layer "
            "also applies left-context attention and a causal depthwise convolution; "
            "stacked layer context means this slice does not guarantee equivalence to "
            "the same positions from the full-clip offline pass."
        ),
    }
    if missing_keys or mismatch_names:
        report["status"] = "tower_weight_load_incomplete"
        return report
    free_after_load, _ = torch.cuda.mem_get_info(device)
    estimated_activation_reserve = int(args.min_forward_reserve_gib * 1024**3)
    if free_after_load < parameter_bytes + estimated_activation_reserve:
        report["status"] = "skipped_insufficient_gpu_headroom_after_load"
        report["runtime"]["gpu"]["free_bytes_after_load"] = int(free_after_load)
        report["runtime"]["gpu"]["required_free_bytes_before_tower_move"] = int(
            parameter_bytes + estimated_activation_reserve
        )
        report["runtime"]["gpu"]["process_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        return report
    audio_model.to(device)
    audio_model.eval()
    torch.cuda.reset_peak_memory_stats(device)

    # Run a few real streaming windows before the full 125-tick measurement.
    smoke_features, smoke_mask = _tensor_for_features(windows[0], audio_model, device, torch)
    smoke: dict[str, Any] = {"requested_ticks": int(args.smoke_ticks), "passed": False}
    try:
        for index in range(args.warmup):
            _one_forward(audio_model, smoke_features, smoke_mask, torch)
        torch.cuda.synchronize(device)
        smoke_cuda_ms: list[float] = []
        smoke_wall_ms: list[float] = []
        for index in range(args.smoke_ticks):
            tick_features, tick_mask = _tensor_for_features(windows[index], audio_model, device, torch)
            output, cuda_ms, wall_ms = _timed_forward(audio_model, tick_features, tick_mask, torch)
            hidden = _output_hidden(output)
            if tuple(hidden.shape[1:]) != (15, hidden.shape[-1]) or hidden.shape[-1] <= 0:
                raise RuntimeError(f"60-row smoke returned unexpected tower shape {tuple(hidden.shape)}")
            smoke_cuda_ms.append(cuda_ms)
            smoke_wall_ms.append(wall_ms)
        smoke = {
            "requested_ticks": int(args.smoke_ticks),
            "passed": True,
            "output_shape_per_tick": list(hidden.shape),
            "cuda_ms": _stats(smoke_cuda_ms),
            "wall_ms": _stats(smoke_wall_ms),
        }
    except Exception as exc:
        smoke.update({"error": f"{type(exc).__name__}: {exc}"})
        report["smoke"] = smoke
        report["status"] = "smoke_failed"
        return report
    report["smoke"] = smoke

    stream_tokens: list[np.ndarray] = []
    stream_cuda_ms: list[float] = []
    stream_wall_ms: list[float] = []
    stream_output_shape: list[int] | None = None
    first_eager_stream_hidden: np.ndarray | None = None
    try:
        for tick, window in enumerate(windows):
            features, mask = _tensor_for_features(window, audio_model, device, torch)
            output, cuda_ms, wall_ms = _timed_forward(audio_model, features, mask, torch)
            hidden = _output_hidden(output)
            if tick == 0:
                first_eager_stream_hidden = hidden[0].float().cpu().numpy()
            if hidden.shape[1] != 15:
                raise RuntimeError(f"tick {tick} returned {hidden.shape[1]} tokens for the 60-row input")
            selected = hidden[:, Q_START : Q_START + TOKENS_PER_TICK, :]
            if tuple(selected.shape[1:]) != (TOKENS_PER_TICK, hidden.shape[-1]):
                raise RuntimeError(f"tick {tick} did not produce the requested q_start={Q_START} pair")
            stream_tokens.append(selected[0].float().cpu().numpy())
            stream_cuda_ms.append(cuda_ms)
            stream_wall_ms.append(wall_ms)
            stream_output_shape = list(hidden.shape)
    except Exception as exc:
        report["streaming"] = {
            "completed_ticks": len(stream_tokens),
            "requested_ticks": 125,
            "error": f"{type(exc).__name__}: {exc}",
        }
        report["status"] = "streaming_failed"
        return report

    stream_concat = np.concatenate(stream_tokens, axis=0)
    report["streaming"] = {
        "completed_ticks": len(stream_tokens),
        "output_shape_per_tick": stream_output_shape,
        "selected_rows": [Q_START, Q_START + TOKENS_PER_TICK],
        "concatenated_output_shape": list(stream_concat.shape),
        "cuda_ms": _stats(stream_cuda_ms),
        "wall_ms": _stats(stream_wall_ms),
        "cuda_target_ms_below_0_6": bool(float(np.percentile(stream_cuda_ms, 95)) < 0.6),
        "latency_target_uses_p95": True,
    }

    # The full-clip reference uses the complete first 10 s, as one tower call.
    offline_features, offline_mask = _tensor_for_features(offline_mel, audio_model, device, torch)
    offline_output, offline_cuda_ms, offline_wall_ms = _timed_forward(
        audio_model, offline_features, offline_mask, torch
    )
    offline_hidden = _output_hidden(offline_output)
    offline_tokens = offline_hidden[0].float().cpu().numpy()
    report["offline"] = {
        "input_shape": list(offline_features.shape),
        "output_shape": list(offline_hidden.shape),
        "cuda_ms_single_call": offline_cuda_ms,
        "wall_ms_single_call": offline_wall_ms,
    }
    report["offline_vs_streaming"] = _cosine_summary(offline_tokens, stream_concat)
    report["offline_vs_streaming"]["same_token_count_expected_250"] = bool(
        offline_tokens.shape[0] == 250 and stream_concat.shape[0] == 250
    )

    # Separate production-width baseline: first 10 s plus 24 zero Mel rows,
    # with every row valid as in the current Gemma tower call contract.
    baseline_array = np.pad(offline_mel, ((0, max(0, 1024 - offline_mel.shape[0])), (0, 0)))
    baseline_array = baseline_array[:1024]
    if free_after_load > 0:
        free_before_baseline, _ = torch.cuda.mem_get_info(device)
    else:
        free_before_baseline = 0
    baseline: dict[str, Any] = {
        "input_shape": [1, 1024, MEL_BINS],
        "valid_mask_rows": 1024,
        "separate_from_causal_60_row_measurement": True,
        "backend": "eager",
        "free_bytes_before": int(free_before_baseline),
    }
    baseline_eager_hidden: np.ndarray | None = None
    if free_before_baseline < estimated_activation_reserve:
        baseline["status"] = "skipped_insufficient_gpu_headroom"
    else:
        try:
            baseline_features, baseline_mask = _tensor_for_features(baseline_array, audio_model, device, torch)
            for _ in range(max(1, args.baseline_warmup)):
                _one_forward(audio_model, baseline_features, baseline_mask, torch)
            baseline_cuda_ms: list[float] = []
            baseline_wall_ms: list[float] = []
            baseline_output_shape: list[int] | None = None
            for _ in range(args.baseline_iterations):
                output, cuda_ms, wall_ms = _timed_forward(audio_model, baseline_features, baseline_mask, torch)
                hidden = _output_hidden(output)
                baseline_eager_hidden = hidden[0].float().cpu().numpy()
                baseline_output_shape = list(hidden.shape)
                baseline_cuda_ms.append(cuda_ms)
                baseline_wall_ms.append(wall_ms)
            baseline.update({
                "status": "measured",
                "output_shape": baseline_output_shape,
                "iterations": len(baseline_cuda_ms),
                "cuda_ms": _stats(baseline_cuda_ms),
                "wall_ms": _stats(baseline_wall_ms),
            })
        except Exception as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            baseline.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    report["baseline_1024"] = baseline

    # Capture the same full forward in CUDA Graphs. The 60-row graph is replayed
    # for each real causal window through one static input buffer; the 1024-row
    # graph is a separate production-width baseline. Graph and eager references
    # use the exact same inputs, with no window or reference adjustments.
    graph_results: dict[str, Any] = {
        "semantics": "same model, weights, masks, input arrays, and output selection as eager",
        "timing": "CUDA event device time plus host wall time; capture excluded",
        "input_update": "GPU-to-GPU copy into static input buffer, included in copy_plus_replay metrics",
    }
    graph_free, _ = torch.cuda.mem_get_info(device)
    graph_results["free_bytes_before_capture"] = int(graph_free)
    if graph_free < estimated_activation_reserve:
        graph_results["status"] = "skipped_insufficient_gpu_headroom"
        graph_results["reason"] = "preserve active serving headroom before CUDA Graph capture"
    elif first_eager_stream_hidden is None:
        graph_results["status"] = "skipped_missing_eager_reference"
    else:
        try:
            all_windows = np.stack(windows, axis=0)
            graph_results["input_60"] = _measure_graph_shape(
                audio_model,
                windows[0],
                first_eager_stream_hidden,
                args.graph_iterations,
                args.graph_warmup,
                torch,
                device,
                input_series=all_windows,
            )
        except Exception as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            graph_results["input_60"] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        graph_free_after_60, _ = torch.cuda.mem_get_info(device)
        graph_results["free_bytes_after_input_60"] = int(graph_free_after_60)
        if baseline_eager_hidden is None or baseline.get("status") != "measured":
            graph_results["input_1024"] = {
                "status": "skipped_missing_eager_1024_reference",
            }
        elif graph_free_after_60 < estimated_activation_reserve:
            graph_results["input_1024"] = {
                "status": "skipped_insufficient_gpu_headroom",
                "free_bytes_before_capture": int(graph_free_after_60),
            }
        else:
            try:
                graph_results["input_1024"] = _measure_graph_shape(
                    audio_model,
                    baseline_array,
                    baseline_eager_hidden,
                    args.graph_iterations,
                    args.graph_warmup,
                    torch,
                    device,
                )
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                graph_results["input_1024"] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
        graph_results["free_bytes_after_input_1024"] = int(torch.cuda.mem_get_info(device)[0])
        graph_results["status"] = "completed"
    report["cuda_graph"] = graph_results
    for shape_result in (graph_results.get("input_60"), graph_results.get("input_1024")):
        if isinstance(shape_result, dict) and shape_result.get("status") == "measured":
            replay_p95 = shape_result["replay_only_cuda_ms"]["p95"]
            shape_result["replay_only_cuda_target_ms_below_0_6"] = bool(replay_p95 < 0.6)
            if "copy_plus_replay_cuda_ms" in shape_result:
                shape_result["copy_plus_replay_cuda_target_ms_below_0_6"] = bool(
                    shape_result["copy_plus_replay_cuda_ms"]["p95"] < 0.6
                )

    report["runtime"]["gpu"].update({
        "free_bytes_after_benchmark": int(torch.cuda.mem_get_info(device)[0]),
        "process_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "process_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    })
    report["status"] = "completed"
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Measure the requested user-only rolling-window sweep without touching serving."""
    import torch
    import transformers

    report: dict[str, Any] = {
        "status": "running",
        "benchmark": "user-only Mel rolling-window sweep against same offline 10-second tower output",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "container": args.container,
            "active_runtime_source": str(args.runtime_source),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "backend": "Gemma4AudioModel CUDA Graph; one graph per fixed window/mask shape",
            "precision_baseline": {
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "tower_dtype": "checkpoint-loaded BF16 (FP32 attention internals verified from installed source)",
            },
            "tf32_ablation": "temporary process-local torch.set_float32_matmul_precision('high'); restored after the isolated graph experiment",
        },
        "input": {"path": str(args.input), "target_sample_rate_hz": SAMPLE_RATE},
        "contract": {
            "ticks": 125,
            "tick_ms": TICK_MS,
            "mel_rows_per_tick": MEL_ROWS_PER_TICK,
            "window_rows": [60, 120, 160, 192, 224, 240, 480, 640, 800, 1024],
            "window_update": "window_rows-8 prior rows (capped by real history) plus 8 new user Mel rows",
            "selected_output": "last 2 tower rows; q_start=window_rows//4-2",
            "offline_reference": "same real WAV, resampled and trimmed to 10.0s, full 1000-row offline Mel, same loaded tower/checkpoint",
            "mask_ablation": "W1024 all-true (historical behavior) versus false for invalid leading zero-pad rows",
            "prefix_ablation": "exact user-Mel prefixes of 8,16,...,1000 rows, no padding, select last 2 rows",
            "latency_target_ms": 0.6,
            "latency_target_is_reported_not_enforced": True,
        },
        "notes": [
            "All graph windows use the same real user-only Mel stream; no assistant-history features enter these measurements.",
            "Graph copy+replay timing copies from a pre-staged GPU tensor into static graph buffers; CPU-to-GPU transfer and Mel extraction are excluded.",
            "Startup and steady metrics use the same 250-pair offline reference; steady begins only when the complete requested prior Mel context exists.",
            "A fixed Mel window resets the tower positional coordinate system at the window start; a larger window is measured, not assumed equivalent.",
        ],
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected runtime")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    report["runtime"]["gpu"] = {
        "name": props.name,
        "total_bytes": int(total_bytes),
        "free_bytes_at_start": int(free_before),
        "device_index": 0,
        "active_serving_is_not_managed_by_this_script": True,
    }
    if free_before < args.min_free_gib * 1024**3:
        report["status"] = "skipped_insufficient_gpu_headroom"
        report["runtime"]["memory_guard_reason"] = "preserve active serving headroom; tower was not loaded"
        return report

    waveform, source_rate, source_samples, resampled_samples = _read_and_resample(args.input)
    report["input"].update({
        "source_rate_hz": source_rate,
        "source_samples": source_samples,
        "source_duration_seconds": source_samples / source_rate,
        "resampled_samples_before_trim": resampled_samples,
        "measured_samples": int(waveform.size),
        "measured_duration_seconds": waveform.size / SAMPLE_RATE,
        "peak_abs": float(np.max(np.abs(waveform))),
        "rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
    })
    model_path, processor, extractor, adapter = _load_runtime_preprocessor(args.workspace, args.runtime_source)
    offline_mel, streaming_mel, chunks = _extract_mels(adapter, waveform)
    report["model"] = {
        "path": str(model_path),
        "processor": type(processor).__name__,
        "feature_extractor": type(extractor).__name__,
    }
    report["preprocessing"] = {
        "extractor_source_file": str(Path(sys.modules["duplex_kernel.listener"].__file__).resolve()),
        "extractor_source_method": "GemmaAudioTowerBackend._extract and _extract_single_frame",
        "offline_shape": list(offline_mel.shape),
        "streaming_shape": list(streaming_mel.shape),
        "offline_vs_streaming_mel": _feature_stats(offline_mel, streaming_mel),
    }
    if offline_mel.shape != (1000, MEL_BINS) or streaming_mel.shape != (1000, MEL_BINS):
        raise RuntimeError(f"expected 1000x128 offline and streaming mels, got {offline_mel.shape}/{streaming_mel.shape}")

    audio_model, config, audio_config, loading_info = _load_audio_tower(model_path, torch, transformers)
    parameter_count = sum(int(parameter.numel()) for parameter in audio_model.parameters())
    parameter_bytes = sum(int(parameter.numel() * parameter.element_size()) for parameter in audio_model.parameters())
    missing = [str(key) for key in loading_info.get("missing_keys", ())]
    mismatches = [str(item[0] if isinstance(item, (tuple, list)) else item) for item in loading_info.get("mismatched_keys", ())]
    unexpected = [str(key) for key in loading_info.get("unexpected_keys", ())]
    report["model"].update({
        "audio_tower_class": type(audio_model).__name__,
        "audio_tower_parameters": parameter_count,
        "audio_tower_parameter_bytes": parameter_bytes,
        "audio_tower_dtype": str(next(audio_model.parameters()).dtype),
        "audio_config": audio_config.to_dict() if hasattr(audio_config, "to_dict") else str(audio_config),
        "attention_implementation": getattr(audio_config, "_attn_implementation", getattr(audio_model.config, "_attn_implementation", None)),
        "weight_load": {
            "expected_state_dict_entries": len(audio_model.state_dict()),
            "missing_key_count": len(missing),
            "missing_key_sample": missing[:20],
            "mismatched_key_count": len(mismatches),
            "mismatched_key_sample": mismatches[:20],
            "unexpected_key_count": len(unexpected),
            "unexpected_tower_key_count": sum(_tower_key(key) for key in unexpected),
            "unexpected_non_tower_key_count": sum(not _tower_key(key) for key in unexpected),
            "tower_weights_complete": not missing and not mismatches,
        },
    })
    if missing or mismatches:
        report["status"] = "tower_weight_load_incomplete"
        return report
    modeling_source = Path(sys.modules["transformers.models.gemma4.modeling_gemma4"].__file__).resolve()
    report["model"]["audio_tower_source_geometry"] = {
        "source_file": str(modeling_source),
        "subsample_module": "Gemma4AudioSubSampleConvProjection; two Conv2d temporal kernel=3 stride=2 padding=1 layers; receptive field=7 Mel rows, stride=4",
        "conformer_layers": len(getattr(audio_model, "layers", [])),
        "attention_context_left_tokens": int(getattr(audio_config, "attention_context_left", 13)),
        "attention_context_right_tokens": int(getattr(audio_config, "attention_context_right", 0)),
        "attention_chunk_tokens": int(getattr(audio_config, "attention_chunk_size", 12)),
        "lightconv_kernel_tokens": int(getattr(audio_config, "conv_kernel_size", 5)),
        "attention_source_behavior": "Gemma4AudioAttention manually casts q/k/v to FP32, uses matmul for q-k and relative positions, then FP32 softmax; no SDPA path in this model forward.",
        "attention_dtype_implication": "model/input/linear/conv weights are BF16; internal attention matmuls/softmax include FP32 ops, the only target of the temporary TF32 precision experiment.",
    }
    free_after_load, _ = torch.cuda.mem_get_info(device)
    reserve = int(args.min_forward_reserve_gib * 1024**3)
    report["runtime"]["gpu"]["free_bytes_after_cpu_load"] = int(free_after_load)
    report["runtime"]["gpu"]["tower_parameter_bytes"] = int(parameter_bytes)
    if free_after_load < parameter_bytes + reserve:
        report["status"] = "skipped_insufficient_gpu_headroom_after_load"
        report["runtime"]["gpu"]["required_free_bytes_before_tower_move"] = int(parameter_bytes + reserve)
        return report
    audio_model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)

    offline_features, offline_mask = _tensor_for_features(offline_mel, audio_model, device, torch)
    offline_output, offline_cuda_ms, offline_wall_ms = _timed_forward(audio_model, offline_features, offline_mask, torch)
    offline_hidden = _output_hidden(offline_output)
    offline_tokens = offline_hidden[0].float().cpu().numpy()
    report["offline_reference"] = {
        "input_shape": list(offline_features.shape),
        "output_shape": list(offline_hidden.shape),
        "cuda_ms_single_call": offline_cuda_ms,
        "wall_ms_single_call": offline_wall_ms,
        "token_count": int(offline_tokens.shape[0]),
    }
    if offline_tokens.shape != (250, int(offline_hidden.shape[-1])):
        raise RuntimeError(f"expected 250 offline tower tokens, got {offline_tokens.shape}")

    windows_by_size: dict[int, list[np.ndarray]] = {}
    padding_by_size: dict[int, list[int]] = {}
    for width in (60, 120, 160, 192, 224, 240, 480, 640, 800, 1024):
        windows_by_size[width], padding_by_size[width] = _window_series(chunks, width)
    report["window_sweep"] = {
        "mask_convention": "all_true, including leading zero pad rows; W1024 false-leading mask is a separate ablation",
        "windows": {},
    }
    baseline_stream_by_size: dict[int, np.ndarray] = {}
    terminal_hidden_by_size: dict[int, np.ndarray] = {}
    latency_by_size: dict[int, dict[str, Any]] = {}
    for width, windows in windows_by_size.items():
        free_now, _ = torch.cuda.mem_get_info(device)
        row_result: dict[str, Any] = {
            "status": "pending",
            "window_rows": width,
            "history_rows": width - MEL_ROWS_PER_TICK,
            "q_start": width // 4 - TOKENS_PER_TICK,
            "padding_rows_by_tick": {
                "first": padding_by_size[width][0],
                "last": padding_by_size[width][-1],
                "min": min(padding_by_size[width]),
                "max": max(padding_by_size[width]),
            },
            "free_bytes_before_capture": int(free_now),
        }
        if free_now < reserve:
            row_result.update({"status": "skipped_insufficient_gpu_headroom", "reason": "preserve active serving headroom"})
            report["window_sweep"]["windows"][str(width)] = row_result
            continue
        try:
            measured, outputs, terminal_hidden = _measure_sweep_graph(
                audio_model, width, windows, padding_by_size[width], offline_tokens,
                "all_true", args.graph_iterations, args.graph_warmup, args.smoke_ticks, torch, device,
            )
            row_result.update(measured)
            baseline_stream_by_size[width] = outputs
            terminal_hidden_by_size[width] = terminal_hidden
            latency_by_size[width] = measured["latency"]
        except Exception as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            row_result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        row_result["free_bytes_after_capture"] = int(torch.cuda.mem_get_info(device)[0])
        report["window_sweep"]["windows"][str(width)] = row_result
        _write_report(args.output, report)

    # The parent explicitly requested this single-mask ablation at the largest width.
    if "1024" in report["window_sweep"]["windows"] and 1024 in baseline_stream_by_size:
        free_now, _ = torch.cuda.mem_get_info(device)
        if free_now >= reserve:
            try:
                false_result, false_outputs, _false_terminal = _measure_sweep_graph(
                    audio_model, 1024, windows_by_size[1024], padding_by_size[1024], offline_tokens,
                    "false_invalid_leading", args.graph_iterations, args.graph_warmup, args.smoke_ticks, torch, device,
                )
                report["window_sweep"]["windows"]["1024"]["false_invalid_leading_mask_ablation"] = false_result
                report["window_sweep"]["windows"]["1024"]["mask_ablation_cosine_delta_mean_vs_all_true"] = (
                    false_result["cosine_vs_same_offline_250_tokens"]["mean"]
                    - report["window_sweep"]["windows"]["1024"]["cosine_vs_same_offline_250_tokens"]["mean"]
                )
                _write_report(args.output, report)
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                report["window_sweep"]["windows"]["1024"]["false_invalid_leading_mask_ablation"] = {
                    "status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()
                }
        else:
            report["window_sweep"]["windows"]["1024"]["false_invalid_leading_mask_ablation"] = {
                "status": "skipped_insufficient_gpu_headroom", "free_bytes_before_capture": int(free_now)
            }

    # Small, auxiliary CPU-only cache for the sibling's projection-boundary test.
    # It stores the already-measured terminal last_hidden_state, not a checkpoint.
    cached_widths = [width for width in (60, 640, 1024) if width in terminal_hidden_by_size]
    if len(cached_widths) == 3:
        cache_path = args.output.with_name(args.output.stem + "_terminal_hidden.npz")
        np.savez_compressed(cache_path, **{
            f"window_{width}_tick124_last_hidden_state": terminal_hidden_by_size[width].astype(np.float32, copy=False)
            for width in cached_widths
        })
        report["terminal_output_cache"] = {
            "path": str(cache_path),
            "format": "NumPy compressed archive; arrays are CPU float32",
            "windows": cached_widths,
            "tick": 124,
            "mask_convention": "all_true",
            "tensor": "Gemma4AudioModel.last_hidden_state after its output_proj; exact final rolling window from the graph sweep",
            "purpose": "reusable output for isolated downstream projection-boundary timing without another tower load/forward",
        }
        _write_report(args.output, report)

    prefix_result, prefix_outputs = _measure_prefix_series(audio_model, streaming_mel, offline_tokens, torch, device)
    report["prefix_no_padding_ablation"] = prefix_result
    _write_report(args.output, report)

    # Lightweight stage and ATen-op profiling for representative inputs. The source profile
    # labels ops, and per-module CUDA events provide block-level device time without claiming
    # that ATen operator labels are literal CUDA kernel names.
    report["stage_profile"] = {}
    for width in (60, 640, 1024):
        if str(width) not in report["window_sweep"]["windows"] or report["window_sweep"]["windows"][str(width)].get("status") != "measured":
            report["stage_profile"][str(width)] = {"status": "skipped_no_window_measurement"}
            continue
        tick = min(max(0, (width - MEL_ROWS_PER_TICK + 7) // MEL_ROWS_PER_TICK), 124)
        profile_input = windows_by_size[width][tick]
        profile_mask = np.ones(width, dtype=np.bool_)
        if width == 1024:
            profile_mask[: padding_by_size[width][tick]] = False
        try:
            stage_report = _profile_audio_stages(audio_model, profile_input, profile_mask, torch, device)
            parameter_dtype = next(audio_model.parameters()).dtype
            stage_report["input_dtype"] = str(parameter_dtype)
            stage_report["attention_implementation"] = report["model"]["attention_implementation"]
            try:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=False,
                    with_stack=False,
                ) as profiler:
                    features, mask = _tensor_for_features(profile_input, audio_model, device, torch)
                    mask.copy_(torch.as_tensor(profile_mask[None, :], device=device, dtype=torch.bool))
                    _one_forward(audio_model, features, mask, torch)
                    torch.cuda.synchronize(device)
                averages = profiler.key_averages(group_by_input_shape=True)
                top_ops = []
                for event in sorted(averages, key=lambda item: float(getattr(item, "self_device_time_total", getattr(item, "self_cuda_time_total", 0.0))), reverse=True):
                    name = str(getattr(event, "key", getattr(event, "name", "unknown")))
                    if not name.startswith("aten::"):
                        continue
                    cuda_time = float(getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0)))
                    cpu_time = float(getattr(event, "self_cpu_time_total", 0.0))
                    top_ops.append({
                        "operator": name,
                        "input_shapes": getattr(event, "input_shapes", None),
                        "self_device_ms": cuda_time / 1000.0,
                        "self_cpu_ms": cpu_time / 1000.0,
                        "calls": int(getattr(event, "count", 0)),
                    })
                    if len(top_ops) >= 20:
                        break
                stage_report["torch_profiler"] = {
                    "status": "measured",
                    "scope": "ATen operator aggregates grouped by input shape; operator names are not asserted to equal individual CUDA kernels",
                    "top_20_self_device_time_ops": top_ops,
                }
            except Exception as profile_exc:
                stage_report["torch_profiler"] = {"status": "failed", "error": f"{type(profile_exc).__name__}: {profile_exc}"}
                if isinstance(profile_exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            report["stage_profile"][str(width)] = stage_report
        except Exception as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            report["stage_profile"][str(width)] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        _write_report(args.output, report)

    # A reversible process-local FP32 matmul precision experiment. First compare one
    # representative last-tick input; only run full 250-token validation if speedup >=5%.
    measured_windows = [
        (width, item) for width, item in report["window_sweep"]["windows"].items()
        if item.get("status") == "measured" and int(width) in baseline_stream_by_size
    ]
    best_width = max(
        (int(width) for width, _ in measured_windows),
        key=lambda width: (
            report["window_sweep"]["windows"][str(width)]["cosine_vs_same_offline_250_tokens"]["mean"],
            -report["window_sweep"]["windows"][str(width)]["latency"]["copy_plus_replay_cuda_ms"]["p95"],
        ),
    ) if measured_windows else 60
    tf32_result: dict[str, Any] = {
        "status": "pending",
        "precision_before": torch.get_float32_matmul_precision(),
        "allow_tf32_before": bool(torch.backends.cuda.matmul.allow_tf32),
        "selection": {"fixed_shape": 60, "best_mean_cosine_shape": best_width},
        "minimum_speedup_for_full250_quality_validation": 0.05,
        "scope": "temporary process-local precision setting on the isolated tower benchmark only",
    }
    original_precision = torch.get_float32_matmul_precision()
    original_allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    try:
        torch.set_float32_matmul_precision("high")
        tf32_result["precision_during_test"] = torch.get_float32_matmul_precision()
        tf32_result["allow_tf32_during_test"] = bool(torch.backends.cuda.matmul.allow_tf32)
        point_results = {}
        for width in sorted({60, best_width}):
            baseline_row = report["window_sweep"]["windows"][str(width)]
            if baseline_row.get("status") != "measured":
                point_results[str(width)] = {"status": "skipped_missing_baseline"}
                continue
            parameter = next(audio_model.parameters())
            source_features = torch.as_tensor(windows_by_size[width][-1][None, :, :], device=device, dtype=parameter.dtype).contiguous()
            source_mask = torch.ones((1, width), device=device, dtype=torch.bool)
            static_features = source_features.clone()
            static_mask = source_mask.clone()
            graph, graph_hidden, capture_wall = _capture_graph(audio_model, static_features, static_mask, torch, args.graph_warmup)
            replay_cuda, replay_wall = [], []
            copy_cuda, copy_wall = [], []
            for _ in range(args.tf32_iterations):
                c, w = _graph_replay_timing(graph, torch, device)
                replay_cuda.append(c); replay_wall.append(w)
            for _ in range(args.tf32_iterations):
                c, w = _graph_copy_mask_replay_timing(graph, static_features, static_mask, source_features, source_mask, torch, device)
                copy_cuda.append(c); copy_wall.append(w)
            high_hidden = graph_hidden[0].float().cpu().numpy()
            baseline_precision_lat = baseline_row["latency"]["replay_only_cuda_ms"]["p50"]
            baseline_copy_lat = baseline_row["latency"]["copy_plus_replay_cuda_ms"]["p50"]
            speedup = max(
                (baseline_precision_lat - float(np.percentile(replay_cuda, 50))) / baseline_precision_lat if baseline_precision_lat else 0.0,
                (baseline_copy_lat - float(np.percentile(copy_cuda, 50))) / baseline_copy_lat if baseline_copy_lat else 0.0,
            )
            point = {
                "status": "measured",
                "window_rows": width,
                "input_tick": 124,
                "same_input_baseline_vs_tf32_full_hidden": _feature_stats(
                    high_hidden if terminal_hidden_by_size.get(width) is None else terminal_hidden_by_size[width], high_hidden
                ),
                "tf32_replay_only_cuda_ms": _stats(replay_cuda),
                "tf32_replay_only_wall_ms": _stats(replay_wall),
                "tf32_copy_plus_replay_cuda_ms": _stats(copy_cuda),
                "tf32_copy_plus_replay_wall_ms": _stats(copy_wall),
                "baseline_replay_p50_cuda_ms": baseline_precision_lat,
                "baseline_copy_plus_replay_p50_cuda_ms": baseline_copy_lat,
                "best_observed_speedup_fraction": float(speedup),
                "capture_wall_ms_excluded": capture_wall,
            }
            if speedup >= 0.05:
                full, full_tokens, _tf32_terminal = _measure_sweep_graph(
                    audio_model, width, windows_by_size[width], padding_by_size[width], offline_tokens,
                    "all_true", args.graph_iterations, args.graph_warmup, args.smoke_ticks, torch, device,
                )
                point["full250_quality_validation"] = {
                    "status": "measured_due_to_speedup",
                    "cosine_vs_offline": full["cosine_vs_same_offline_250_tokens"],
                    "same_input_default_vs_tf32_q2": _feature_stats(baseline_stream_by_size[width], full_tokens),
                    "latency": full["latency"],
                }
                del full_tokens
            else:
                point["full250_quality_validation"] = {
                    "status": "not_run_no_material_speedup",
                    "reason": "representative CUDA Graph p50 speedup was below 5%",
                }
            point_results[str(width)] = point
            tf32_result["shape_results"] = point_results
            report["tf32_fp32_matmul_ablation"] = tf32_result
            _write_report(args.output, report)
            del graph, graph_hidden, static_features, static_mask, source_features, source_mask
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        tf32_result.update({"status": "completed", "shape_results": point_results})
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        tf32_result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cuda.matmul.allow_tf32 = original_allow_tf32
        tf32_result["precision_after_restore"] = torch.get_float32_matmul_precision()
        tf32_result["allow_tf32_after_restore"] = bool(torch.backends.cuda.matmul.allow_tf32)
        tf32_result["restored_original_precision_settings"] = (
            torch.get_float32_matmul_precision() == original_precision
            and bool(torch.backends.cuda.matmul.allow_tf32) == original_allow_tf32
        )
    report["tf32_fp32_matmul_ablation"] = tf32_result
    report["runtime"]["gpu"].update({
        "free_bytes_after_sweep": int(torch.cuda.mem_get_info(device)[0]),
        "process_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "process_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    })
    report["status"] = "completed"
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def run_w240_startup_prefix_ablation(args: argparse.Namespace) -> dict[str, Any]:
    """Use exact growing prefixes only until W240 is full, then the fixed window graph."""
    import torch
    import transformers

    if not args.output.is_file():
        raise FileNotFoundError(f"existing completed sweep JSON required: {args.output}")
    report = json.loads(args.output.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise RuntimeError("the completed primary window sweep is required before the W240 startup ablation")
    report["startup_prefix_transition_w240"] = {"status": "running"}
    _write_report(args.output, report)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected runtime")
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    report["startup_prefix_transition_w240"]["gpu_free_bytes_before"] = int(free_before)
    if free_before < args.min_free_gib * 1024**3:
        report["startup_prefix_transition_w240"].update({
            "status": "skipped_insufficient_gpu_headroom",
            "gpu_total_bytes": int(total_bytes),
        })
        _write_report(args.output, report)
        return report

    waveform, source_rate, source_samples, _ = _read_and_resample(args.input)
    model_path, processor, extractor, adapter = _load_runtime_preprocessor(args.workspace, args.runtime_source)
    offline_mel, streaming_mel, chunks = _extract_mels(adapter, waveform)
    mel_identity = _feature_stats(offline_mel, streaming_mel)
    if offline_mel.shape != (1000, MEL_BINS) or streaming_mel.shape != (1000, MEL_BINS):
        raise RuntimeError(f"expected 1000x128 Mel input, got {offline_mel.shape}/{streaming_mel.shape}")
    audio_model, _config, _audio_config, loading_info = _load_audio_tower(model_path, torch, transformers)
    missing = [str(key) for key in loading_info.get("missing_keys", ())]
    mismatches = [str(item[0] if isinstance(item, (tuple, list)) else item) for item in loading_info.get("mismatched_keys", ())]
    free_after_load, _ = torch.cuda.mem_get_info(device)
    parameter_bytes = sum(int(parameter.numel() * parameter.element_size()) for parameter in audio_model.parameters())
    reserve = int(args.min_forward_reserve_gib * 1024**3)
    if missing or mismatches:
        raise RuntimeError(f"tower weight load incomplete: missing={len(missing)} mismatch={len(mismatches)}")
    if free_after_load < parameter_bytes + reserve:
        report["startup_prefix_transition_w240"].update({
            "status": "skipped_insufficient_gpu_headroom_after_load",
            "gpu_free_bytes_after_load": int(free_after_load),
            "tower_parameter_bytes": int(parameter_bytes),
        })
        _write_report(args.output, report)
        return report
    audio_model.to(device).eval()
    offline_features, offline_mask = _tensor_for_features(offline_mel, audio_model, device, torch)
    offline_output = _one_forward(audio_model, offline_features, offline_mask, torch)
    offline_tokens = _output_hidden(offline_output)[0].float().cpu().numpy()

    prefix_ticks = 29  # 232 rows precede tick 29; its current 8 rows complete W=240.
    startup_pairs: list[np.ndarray] = []
    startup_cuda: list[float] = []
    startup_wall: list[float] = []
    parameter = next(audio_model.parameters())
    for tick in range(prefix_ticks):
        rows = (tick + 1) * MEL_ROWS_PER_TICK
        prefix_features = torch.as_tensor(streaming_mel[:rows], device=device, dtype=parameter.dtype).unsqueeze(0).contiguous()
        prefix_mask = torch.ones((1, rows), device=device, dtype=torch.bool)
        output, cuda_ms, wall_ms = _timed_forward(audio_model, prefix_features, prefix_mask, torch)
        hidden = _output_hidden(output)
        startup_pairs.append(hidden[0, -TOKENS_PER_TICK:].float().cpu().numpy())
        startup_cuda.append(cuda_ms)
        startup_wall.append(wall_ms)

    all_steady_windows, all_padding = _window_series(chunks, 240)
    steady_start_tick = 29
    steady_windows = all_steady_windows[steady_start_tick:]
    steady_padding = all_padding[steady_start_tick:]
    if len(steady_windows) != 96 or any(steady_padding):
        raise RuntimeError(f"expected 96 full-history W240 windows, got {len(steady_windows)} with padding={steady_padding[:1]}")
    steady_graph, steady_pairs, _steady_terminal = _measure_sweep_graph(
        audio_model, 240, steady_windows, steady_padding, offline_tokens,
        "all_true", args.graph_iterations, args.graph_warmup, args.smoke_ticks,
        torch, device, reference_tick_indices=list(range(steady_start_tick, 125)),
    )
    all_pairs = startup_pairs + [steady_pairs[index * 2 : index * 2 + 2] for index in range(len(steady_pairs) // 2)]
    if len(all_pairs) != 125:
        raise RuntimeError(f"combined startup/steady stream produced {len(all_pairs)} ticks, expected 125")
    all_tokens = np.concatenate(all_pairs, axis=0)
    actual_tick_indices = list(range(125))
    startup_metric = _phase_metric(offline_tokens, all_pairs, list(range(prefix_ticks)), actual_tick_indices)
    steady_metric = _phase_metric(offline_tokens, all_pairs, list(range(prefix_ticks, 125)), actual_tick_indices)
    last40_metric = _phase_metric(offline_tokens, all_pairs, list(range(85, 125)), actual_tick_indices)
    report["startup_prefix_transition_w240"] = {
        "status": "measured",
        "semantics": "ticks 0..28 use no-left-padding exact prefixes of 8..232 user Mel rows; tick 29 onward uses a fixed W240 rolling window of 232 prior rows plus 8 new rows",
        "mask_convention": "all true for real Mel rows; startup has no padding rows",
        "input_source_rate_hz": int(source_rate),
        "input_source_samples": int(source_samples),
        "mel_offline_vs_streaming": mel_identity,
        "tower_weight_load": {
            "missing_key_count": len(missing),
            "mismatched_key_count": len(mismatches),
            "tower_weights_complete": not missing and not mismatches,
        },
        "offline_reference_shape": [int(offline_tokens.shape[0]), int(offline_tokens.shape[1])],
        "tick_count": len(all_pairs),
        "startup": {
            "ticks": [0, prefix_ticks - 1],
            "prefix_mel_rows": [MEL_ROWS_PER_TICK, prefix_ticks * MEL_ROWS_PER_TICK],
            "mask_convention": "all true; no padding",
            "cosine_vs_offline": startup_metric["cosine"],
            "eager_cuda_ms": _stats(startup_cuda),
            "eager_wall_ms": _stats(startup_wall),
        },
        "steady": {
            "ticks": [steady_start_tick, 124],
            "window_rows": 240,
            "history_rows": 232,
            "q_start": 58,
            "cosine_vs_offline": steady_metric["cosine"],
            "graph": steady_graph["latency"],
        },
        "last_40_ticks": {
            "ticks": [85, 124],
            "phase": "steady W240 window",
            "cosine_vs_offline": last40_metric["cosine"],
        },
        "combined_250_tokens_vs_offline": _cosine_summary(offline_tokens, all_tokens),
        "full_window_graph_validation": steady_graph["graph_validation"],
        "gpu_free_bytes_after": int(torch.cuda.mem_get_info(device)[0]),
        "decision_note": "Uses an unpadded real-prefix startup to avoid injecting synthetic leading Mel frames; the W240 steady graph result is measured separately above.",
    }
    _write_report(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--runtime-source", type=Path, default=DEFAULT_RUNTIME_SOURCE)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--container", default="gemma4-listener-control6-microbatch-20261014")
    parser.add_argument("--smoke-ticks", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--baseline-warmup", type=int, default=2)
    parser.add_argument("--baseline-iterations", type=int, default=20)
    parser.add_argument("--graph-warmup", type=int, default=3)
    parser.add_argument("--graph-iterations", type=int, default=20)
    parser.add_argument("--tf32-iterations", type=int, default=20)
    parser.add_argument("--sweep-only", action="store_true", help="run the multi-window graph/fidelity sweep")
    parser.add_argument(
        "--w240-startup-prefix-ablation",
        action="store_true",
        help="append the no-left-padding prefix startup followed by fixed W240 graph ticks to an existing sweep JSON",
    )
    parser.add_argument("--min-free-gib", type=float, default=2.5)
    parser.add_argument("--min-forward-reserve-gib", type=float, default=1.5)
    args = parser.parse_args()
    if args.smoke_ticks < 1 or args.smoke_ticks > 8:
        parser.error("--smoke-ticks must be from 1 to 8")
    if args.baseline_iterations < 1:
        parser.error("--baseline-iterations must be positive")
    if args.graph_warmup < 1 or args.graph_iterations < 1 or args.tf32_iterations < 1:
        parser.error("graph warmup, graph iterations, and TF32 iterations must be positive")
    if args.sweep_only and args.w240_startup_prefix_ablation:
        parser.error("choose either --sweep-only or --w240-startup-prefix-ablation")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.w240_startup_prefix_ablation:
            report = run_w240_startup_prefix_ablation(args)
        else:
            report = run_sweep(args) if args.sweep_only else run(args)
    except Exception as exc:
        report = {
            "status": "failed",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "input": {"path": str(args.input)},
            "runtime": {"container": args.container, "active_runtime_source": str(args.runtime_source)},
        }
    _write_report(args.output, report)
    print(json.dumps({
        "status": report.get("status"),
        "output": str(args.output),
        "streaming": report.get("streaming"),
        "baseline_1024": report.get("baseline_1024"),
        "cuda_graph": report.get("cuda_graph"),
        "window_sweep": report.get("window_sweep"),
        "prefix_no_padding_ablation": report.get("prefix_no_padding_ablation"),
        "tf32_fp32_matmul_ablation": report.get("tf32_fp32_matmul_ablation"),
        "startup_prefix_transition_w240": report.get("startup_prefix_transition_w240"),
    }, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
