"""Parity and CUDA Graph benchmark for the incremental Gemma 4 Q2 audio tower."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import platform
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any

import numpy as np


WAV_RELATIVE = Path("artifacts/conversation_quality_20261014/inputs/clean_initial.wav")
REPORT_RELATIVE = Path(
    "artifacts/causal_audio_tower_benchmark/streaming_conformer_kv_cache_20261020.json"
)
SAMPLE_RATE = 16_000
TICK_SAMPLES = 1_280
MEL_TICK_ROWS = 8
MEL_BINS = 128
TICK_COUNT = 125
BENCHMARK_ITERATIONS = 500
LATENCY_TARGET_MS = 0.80


def _load_streaming_module(workspace: Path):
    name = "duplex_kernel.streaming_conformer"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = workspace / "duplex_kernel" / "streaming_conformer.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load streaming implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_frontend_helpers(workspace: Path, runtime_source: Path):
    sys.path.insert(0, str(runtime_source))
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
    resampled_samples = int(mono.size)
    target_samples = TICK_COUNT * TICK_SAMPLES
    if resampled_samples < target_samples:
        raise ValueError(f"need {target_samples} resampled samples; got {resampled_samples}")
    return np.asarray(mono[:target_samples], dtype=np.float32), int(source_rate), source_samples, resampled_samples


def _extract_mels(adapter: Any, waveform: np.ndarray):
    offline = np.asarray(adapter._extract([waveform])[0], dtype=np.float32)
    if offline.shape != (TICK_COUNT * MEL_TICK_ROWS, MEL_BINS):
        raise RuntimeError(f"offline Mel shape is {offline.shape}, expected [1000,128]")
    chunks: list[np.ndarray] = []
    tail: np.ndarray | None = None
    for tick in range(TICK_COUNT):
        start = tick * TICK_SAMPLES
        current, tail = adapter._extract_single_frame(
            waveform[start:start + TICK_SAMPLES], tail
        )
        current = np.asarray(current, dtype=np.float32)
        if current.shape != (MEL_TICK_ROWS, MEL_BINS):
            raise RuntimeError(f"streaming Mel tick {tick} has shape {current.shape}")
        chunks.append(current)
    return offline, np.concatenate(chunks, axis=0), chunks


def _checkpoint_inventory(model_path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        selected_keys = [
            key for key in weight_map
            if key.startswith(("model.audio_tower.", "model.embed_audio."))
        ]
        shards = sorted({weight_map[key] for key in selected_keys})
        shard_paths = [model_path / name for name in shards]
        if not all(path.is_file() for path in shard_paths):
            raise FileNotFoundError("one or more audio checkpoint shard paths are missing")
    else:
        shard_paths = [model_path / "model.safetensors"]
        if not shard_paths[0].is_file():
            raise FileNotFoundError(f"Gemma safetensors not found: {shard_paths[0]}")
        shards = [shard_paths[0].name]
        selected_keys = []
    prefixes: Counter[str] = Counter()
    selected_tensor_bytes = 0
    selected_tensor_count = 0
    selected_key_names: list[str] = []
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            if index_path.is_file():
                keys = [key for key in selected_keys if key in available]
            else:
                keys = [
                    key for key in available
                    if key.startswith(("model.audio_tower.", "model.embed_audio."))
                ]
                selected_keys.extend(keys)
            for key in keys:
                prefix = "model.audio_tower." if key.startswith("model.audio_tower.") else "model.embed_audio."
                prefixes[prefix] += 1
                tensor = handle.get_tensor(key)
                selected_tensor_bytes += tensor.numel() * tensor.element_size()
                selected_tensor_count += 1
                selected_key_names.append(key)
    if prefixes["model.audio_tower."] == 0:
        raise RuntimeError("safetensors contains no model.audio_tower.* tensors")
    if "model.embed_audio.embedding_projection.weight" not in selected_key_names:
        raise RuntimeError("safetensors is missing model.embed_audio.embedding_projection.weight")
    return {
        "index_path": str(index_path) if index_path.is_file() else None,
        "index_present": index_path.is_file(),
        "shard_paths": [str(path) for path in shard_paths],
        "shard_file_sizes_bytes": {str(path): path.stat().st_size for path in shard_paths},
        "selected_prefix_counts": dict(prefixes),
        "selected_tensor_count": selected_tensor_count,
        "selected_tensor_bytes": selected_tensor_bytes,
        "all_safetensors_tensor_keys": len(selected_key_names),
        "full_language_model_loaded": False,
    }


def _load_tower_and_embed(model_path: Path, device: Any, dtype: Any):
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4AudioModel,
        Gemma4MultimodalEmbedder,
    )

    config = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    audio_config = config.audio_config
    text_config = config.text_config
    audio_model = Gemma4AudioModel(audio_config).eval().to(device=device, dtype=dtype)
    embed_audio = Gemma4MultimodalEmbedder(audio_config, text_config).eval().to(
        device=device, dtype=dtype
    )
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        prefixes = ("model.audio_tower.", "model.embed_audio.")
        shard_names = sorted({
            weight_map[key] for key in weight_map if key.startswith(prefixes)
        })
        shard_paths = [model_path / name for name in shard_names]
    else:
        shard_paths = [model_path / "model.safetensors"]
    audio_state: dict[str, Any] = {}
    embed_state: dict[str, Any] = {}
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("model.audio_tower."):
                    audio_state[key[len("model.audio_tower."):]] = handle.get_tensor(key)
                elif key.startswith("model.embed_audio."):
                    embed_state[key[len("model.embed_audio."):]] = handle.get_tensor(key)
    audio_result = audio_model.load_state_dict(audio_state, strict=True)
    embed_result = embed_audio.load_state_dict(embed_state, strict=True)
    if audio_result.missing_keys or audio_result.unexpected_keys:
        raise RuntimeError(f"audio tower state mismatch: {audio_result}")
    if embed_result.missing_keys or embed_result.unexpected_keys:
        raise RuntimeError(f"embed_audio state mismatch: {embed_result}")
    return audio_model, embed_audio, config


def _run_raw(audio_model: Any, features: Any, mask: Any, torch: Any):
    with torch.inference_mode():
        output = audio_model(
            input_features=features,
            attention_mask=mask,
            return_dict=True,
        )
    raw = getattr(output, "last_hidden_state", None)
    if raw is None or raw.ndim != 3:
        raise RuntimeError("Gemma4AudioModel did not return a rank-three last_hidden_state")
    return raw


def _feature_tensor(array: np.ndarray, device: Any, dtype: Any, torch: Any):
    features = torch.as_tensor(array, device=device, dtype=dtype).unsqueeze(0).contiguous()
    mask = torch.ones((1, int(array.shape[0])), device=device, dtype=torch.bool)
    return features, mask


def _cosine_statistics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    left = torch.as_tensor(reference, dtype=torch.float32)
    right = torch.as_tensor(candidate, dtype=torch.float32)
    if left.shape != right.shape:
        return {"same_shape": False, "reference_shape": list(left.shape), "candidate_shape": list(right.shape)}
    scores = F.cosine_similarity(left, right, dim=-1).numpy().reshape(-1)
    return {
        "same_shape": True,
        "shape": list(left.shape),
        "token_count": int(scores.size),
        "mean": float(np.mean(scores)),
        "p05": float(np.percentile(scores, 5)),
        "p50": float(np.percentile(scores, 50)),
        "p95": float(np.percentile(scores, 95)),
        "min": float(np.min(scores)),
    }


def _similarity_summary(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"cosine shapes differ: {reference.shape} vs {candidate.shape}")
    return {
        "whole_sequence": _cosine_statistics(reference, candidate),
        "after_first_13_tokens": _cosine_statistics(reference[13:], candidate[13:]),
    }


def _cached_quality_thresholds(summary: dict[str, Any]) -> dict[str, Any]:
    after = summary["after_first_13_tokens"]
    return {
        **summary,
        "required_after_first_13_tokens": {"mean_minimum": 0.9995, "minimum_minimum": 0.998},
        "passed_after_first_13_tokens": bool(
            after["mean"] >= 0.9995 and after["min"] >= 0.998
        ),
    }


def _quick_gpu_smoke(workspace: Path, torch: Any):
    if not torch.cuda.is_available():
        raise RuntimeError("the required RTX 3090 CUDA device is unavailable")
    module = _load_streaming_module(workspace)
    from transformers.models.gemma4 import modeling_gemma4
    from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4TextConfig

    config = Gemma4AudioConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        subsampling_conv_channels=(128, 4),
        conv_kernel_size=5,
        attention_chunk_size=12,
        attention_context_left=13,
        attention_context_right=0,
        output_proj_dims=32,
        rms_norm_eps=1e-6,
    )
    text_config = Gemma4TextConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        layer_types=["full_attention"],
        rope_parameters={
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1_000_000.0,
        },
    )
    device = torch.device("cuda:0")
    audio_model = modeling_gemma4.Gemma4AudioModel(config).eval().to(
        device=device, dtype=torch.bfloat16
    )
    embed_audio = modeling_gemma4.Gemma4MultimodalEmbedder(config, text_config).eval().to(
        device=device, dtype=torch.bfloat16
    )
    matmul = torch.backends.cuda.matmul
    previous_reduction = matmul.allow_bf16_reduced_precision_reduction
    try:
        matmul.allow_bf16_reduced_precision_reduction = False
        streaming = module.StreamingConformerAudioTower(
            audio_model,
            embed_audio,
            device=device,
            dtype=torch.bfloat16,
            max_sessions=4,
            capture_cuda_graphs=True,
            optimize_cuda=True,
        )
        streaming.capture_graph(1)
    finally:
        matmul.allow_bf16_reduced_precision_reduction = previous_reduction
    generator = torch.Generator(device=device).manual_seed(509)
    with torch.inference_mode():
        for frame in range(2):
            tick = torch.randn((1, 8, 128), generator=generator, device=device, dtype=torch.bfloat16)
            row = module.AudioStreamCursor("smoke", 0, "smoke-epoch", frame, frame)
            prepared = streaming.prepare_batch(tick, (row,), use_cuda_graph=True)
            if tuple(prepared.raw_tokens.shape) != (1, 2, 32):
                raise RuntimeError(f"GPU smoke raw output has shape {tuple(prepared.raw_tokens.shape)}")
            if tuple(prepared.soft_tokens.shape) != (1, 2, 32):
                raise RuntimeError(f"GPU smoke post-embed output has shape {tuple(prepared.soft_tokens.shape)}")
            if not bool(torch.isfinite(prepared.soft_tokens).all().item()):
                raise RuntimeError("GPU smoke produced non-finite post-embed output")
            prepared.commit()
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "status": "passed",
        "model": "tiny installed Gemma4AudioModel + Gemma4MultimodalEmbedder",
        "ticks": 2,
        "device": torch.cuda.get_device_name(device),
        "free_bytes_after_smoke": int(free_bytes),
        "total_bytes": int(total_bytes),
        "committed_positions": streaming.committed_positions,
    }


class _ClockSampler:
    def __init__(self, device_index: int = 0, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml: Any | None = None
        try:
            import pynvml

            self._pynvml = pynvml
            self._pynvml.nvmlInit()
            self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except ImportError:
            import ctypes
            import ctypes.util

            library = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
            self._nvml_lib = ctypes.CDLL(library)
            self._nvml_lib.nvmlInit_v2.restype = ctypes.c_int
            self._nvml_lib.nvmlShutdown.restype = ctypes.c_int
            self._nvml_lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self._nvml_lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
            self._nvml_lib.nvmlDeviceGetClockInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint),
            ]
            self._nvml_lib.nvmlDeviceGetClockInfo.restype = ctypes.c_int
            self._nvml_lib.nvmlDeviceGetPowerUsage.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
            ]
            self._nvml_lib.nvmlDeviceGetPowerUsage.restype = ctypes.c_int

            class _Utilization(ctypes.Structure):
                _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

            self._utilization_type = _Utilization
            self._nvml_lib.nvmlDeviceGetUtilizationRates.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_Utilization),
            ]
            self._nvml_lib.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
            self._check_nvml(self._nvml_lib.nvmlInit_v2(), "nvmlInit_v2")
            handle = ctypes.c_void_p()
            self._check_nvml(
                self._nvml_lib.nvmlDeviceGetHandleByIndex_v2(
                    device_index, ctypes.byref(handle)
                ),
                "nvmlDeviceGetHandleByIndex_v2",
            )
            self._handle = handle

    @staticmethod
    def _check_nvml(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"NVML {operation} failed with code {status}")

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, name="q2-nvml-telemetry", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pynvml is not None:
            self._pynvml.nvmlShutdown()
        else:
            self._check_nvml(self._nvml_lib.nvmlShutdown(), "nvmlShutdown")

    def _read_nvml_sample(self) -> dict[str, float]:
        if self._pynvml is not None:
            nvml = self._pynvml
            util = nvml.nvmlDeviceGetUtilizationRates(self._handle)
            return {
                "sm_clock_mhz": float(nvml.nvmlDeviceGetClockInfo(self._handle, nvml.NVML_CLOCK_SM)),
                "memory_clock_mhz": float(nvml.nvmlDeviceGetClockInfo(self._handle, nvml.NVML_CLOCK_MEM)),
                "gpu_utilization_percent": float(util.gpu),
                "memory_utilization_percent": float(util.memory),
                "power_watts": float(nvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0,
            }
        import ctypes

        util = self._utilization_type()
        self._check_nvml(
            self._nvml_lib.nvmlDeviceGetUtilizationRates(self._handle, ctypes.byref(util)),
            "nvmlDeviceGetUtilizationRates",
        )
        values: dict[str, int] = {}
        for name, clock_type in (("sm_clock_mhz", 1), ("memory_clock_mhz", 2)):
            value = ctypes.c_uint()
            self._check_nvml(
                self._nvml_lib.nvmlDeviceGetClockInfo(
                    self._handle, clock_type, ctypes.byref(value)
                ),
                "nvmlDeviceGetClockInfo",
            )
            values[name] = int(value.value)
        power = ctypes.c_uint()
        self._check_nvml(
            self._nvml_lib.nvmlDeviceGetPowerUsage(self._handle, ctypes.byref(power)),
            "nvmlDeviceGetPowerUsage",
        )
        return {
            "sm_clock_mhz": float(values["sm_clock_mhz"]),
            "memory_clock_mhz": float(values["memory_clock_mhz"]),
            "gpu_utilization_percent": float(util.gpu),
            "memory_utilization_percent": float(util.memory),
            "power_watts": float(power.value) / 1000.0,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self._read_nvml_sample())
            except Exception:
                pass
            self._stop.wait(self.interval_s)


def _telemetry_summary(samples: list[dict[str, float]]) -> dict[str, Any]:
    if not samples:
        return {"status": "no_samples"}
    keys = tuple(samples[0])
    return {
        "status": "measured",
        "sample_count": len(samples),
        "sampling_interval_ms": 50,
        "samples": samples,
        "mean": {key: float(np.mean([row[key] for row in samples])) for key in keys},
        "min": {key: float(np.min([row[key] for row in samples])) for key in keys},
        "max": {key: float(np.max([row[key] for row in samples])) for key in keys},
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(np.mean(array)),
        "p05_ms": float(np.percentile(array, 5)),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
    }


def _bench_graph_shape(
    streaming: Any,
    audio_model: Any,
    embed_audio: Any,
    device: Any,
    dtype: Any,
    torch: Any,
    real_tick: np.ndarray,
    batch_size: int,
    iteration_count: int = BENCHMARK_ITERATIONS,
) -> dict[str, Any]:
    module = sys.modules["duplex_kernel.streaming_conformer"]
    if batch_size not in streaming._graphs:
        raise RuntimeError(f"production optimized graph N={batch_size} was not captured at startup")
    source = torch.as_tensor(real_tick, device=device, dtype=dtype).view(
        1, MEL_TICK_ROWS, MEL_BINS
    ).expand(batch_size, -1, -1).contiguous()
    ids = tuple(f"bench-{row}" for row in range(batch_size))
    warm_rows = tuple(module.AudioStreamCursor(sid, 0, "bench-epoch", 0, 0) for sid in ids)
    positions_before_warmup = streaming.committed_positions
    with torch.inference_mode():
        warm = streaming.prepare_batch(source, warm_rows, use_cuda_graph=True)
        warm.abort()
        for _ in range(8):
            warm = streaming.prepare_batch(source, warm_rows, use_cuda_graph=True)
            warm.abort()
    if streaming.committed_positions != positions_before_warmup:
        raise RuntimeError("CUDA graph replay warmup changed committed session cursors")

    cuda_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iteration_count)
    ]
    transaction_times_ms: list[float] = []
    torch.cuda.synchronize(device)
    with _ClockSampler() as telemetry, torch.inference_mode():
        for iteration in range(iteration_count):
            order = tuple(range(batch_size))
            if batch_size == 2 and iteration % 2:
                order = (1, 0)
            rows = tuple(
                module.AudioStreamCursor(ids[row], 0, "bench-epoch", iteration, iteration)
                for row in order
            )
            wall_start_ns = time.perf_counter_ns()
            input_rows = source[list(order)]
            prepared = streaming.prepare_batch(
                input_rows,
                rows,
                use_cuda_graph=True,
                graph_event_pair=cuda_events[iteration],
            )
            prepared.commit()
            transaction_times_ms.append(
                (time.perf_counter_ns() - wall_start_ns) / 1_000_000.0
            )
        torch.cuda.synchronize(device)
    cuda_times = [float(start.elapsed_time(end)) for start, end in cuda_events]
    expected = tuple((sid, 0, "bench-epoch", iteration_count, iteration_count) for sid in ids)
    measured_positions = tuple(
        position for position in streaming.committed_positions if position[0] in ids
    )
    if measured_positions != expected:
        raise RuntimeError(
            f"500 measured graph replays did not advance each session exactly: {measured_positions}"
        )
    return {
        "status": "measured",
        "batch_size": batch_size,
        "input_shape": [batch_size, MEL_TICK_ROWS, MEL_BINS],
        "input_source": "final real WAV Mel tick repeated for N rows",
        "raw_output_shape": [batch_size, 2, int(audio_model.config.output_proj_dims)],
        "post_embed_output_shape": [batch_size, 2, int(embed_audio.text_hidden_size)],
        "measured_iterations": len(cuda_times),
        "warmup_replays_excluded": 8,
        "graph_capture_and_warmup_advanced_state": False,
        "every_measured_replay_committed_state": True,
        "n2_session_order_reversed_every_other_replay": batch_size == 2,
        "state_after_measured_replays": measured_positions,
        "cuda_graph_replay_only_latency": _latency_summary(cuda_times),
        "python_prepare_commit_clone_transaction_latency": _latency_summary(transaction_times_ms),
        "latency_target_mean_and_p50_below_0_80_ms": bool(
            float(np.mean(cuda_times)) < LATENCY_TARGET_MS
            and float(np.percentile(cuda_times, 50)) < LATENCY_TARGET_MS
        ),
        "nvml_telemetry": _telemetry_summary(telemetry.samples),
        "cuda_events_preallocated_before_measured_loop": True,
        "graph_event_bracket": "immediately before and after ws.graph.replay() only",
        "no_host_copy_or_synchronization_inside_measured_replay_loop": True,
    }


def _run_real_parity(
    audio_model: Any,
    embed_audio: Any,
    waveform: np.ndarray,
    adapter: Any,
    torch: Any,
    streaming: Any,
    extracted_mels: tuple[np.ndarray, np.ndarray, list[np.ndarray]] | None = None,
):
    device = next(audio_model.parameters()).device
    dtype = next(audio_model.parameters()).dtype
    offline_mel, streaming_mel, chunks = extracted_mels or _extract_mels(adapter, waveform)
    mel_delta = np.asarray(streaming_mel, dtype=np.float64) - np.asarray(offline_mel, dtype=np.float64)
    offline_features, offline_mask = _feature_tensor(offline_mel, device, dtype, torch)
    matmul = torch.backends.cuda.matmul
    previous_reduction = matmul.allow_bf16_reduced_precision_reduction
    try:
        matmul.allow_bf16_reduced_precision_reduction = True
        with torch.inference_mode():
            offline_raw_tensor = _run_raw(audio_model, offline_features, offline_mask, torch)
            offline_soft_tensor = embed_audio(offline_raw_tensor)
            offline_raw = offline_raw_tensor[0].float().cpu().numpy()
            offline_soft = offline_soft_tensor[0].float().cpu().numpy()
    finally:
        matmul.allow_bf16_reduced_precision_reduction = previous_reduction

    module = sys.modules["duplex_kernel.streaming_conformer"]
    cached_raw: list[np.ndarray] = []
    cached_soft: list[np.ndarray] = []
    with torch.inference_mode():
        for tick, chunk in enumerate(chunks):
            features = torch.as_tensor(chunk, device=device, dtype=dtype).unsqueeze(0)
            row = module.AudioStreamCursor("real-parity", 0, "real-epoch", tick, tick)
            prepared = streaming.prepare_batch(
                features,
                (row,),
                use_cuda_graph=bool(streaming.optimize_cuda),
            )
            cached_raw.append(prepared.raw_tokens[0].float().cpu().numpy())
            cached_soft.append(prepared.soft_tokens[0].float().cpu().numpy())
            prepared.commit()
    cached_raw_np = np.concatenate(cached_raw, axis=0)
    cached_soft_np = np.concatenate(cached_soft, axis=0)

    rolling_raw: list[np.ndarray] = []
    rolling_soft: list[np.ndarray] = []
    try:
        matmul.allow_bf16_reduced_precision_reduction = True
        with torch.inference_mode():
            for tick in range(TICK_COUNT):
                end = (tick + 1) * MEL_TICK_ROWS
                start = max(0, end - 240)
                window = streaming_mel[start:end]
                features, mask = _feature_tensor(window, device, dtype, torch)
                raw = _run_raw(audio_model, features, mask, torch)
                soft = embed_audio(raw)
                rolling_raw.append(raw[0, -2:].float().cpu().numpy())
                rolling_soft.append(soft[0, -2:].float().cpu().numpy())
    finally:
        matmul.allow_bf16_reduced_precision_reduction = previous_reduction
    rolling_raw_np = np.concatenate(rolling_raw, axis=0)
    rolling_soft_np = np.concatenate(rolling_soft, axis=0)
    if offline_raw.shape[0] != 250 or cached_raw_np.shape != offline_raw.shape:
        raise RuntimeError("offline and cached Q2 output must both contain exactly 250 tokens")
    return {
        "mel_features": {
            "shape": list(offline_mel.shape),
            "offline_equals_incremental_bitwise": bool(np.array_equal(offline_mel, streaming_mel)),
            "max_absolute_delta": float(np.max(np.abs(mel_delta))),
        },
        "tower_1536": {
            "offline_shape": list(offline_raw.shape),
            "cached_streaming_vs_offline": _cached_quality_thresholds(
                _similarity_summary(offline_raw, cached_raw_np)
            ),
            "rolling_w240_vs_offline": _similarity_summary(offline_raw, rolling_raw_np),
            "cached_streaming_vs_w240": _similarity_summary(rolling_raw_np, cached_raw_np),
        },
        "post_embed_production": {
            "offline_shape": list(offline_soft.shape),
            "cached_streaming_vs_offline": _cached_quality_thresholds(
                _similarity_summary(offline_soft, cached_soft_np)
            ),
            "rolling_w240_vs_offline": _similarity_summary(offline_soft, rolling_soft_np),
            "cached_streaming_vs_w240": _similarity_summary(rolling_soft_np, cached_soft_np),
        },
        "cached_streaming_output_shape": list(cached_raw_np.shape),
        "rolling_w240_output_shape": list(rolling_raw_np.shape),
        "cache_state_after_125_ticks": streaming.committed_positions,
    }


def _capture_production_streaming_tower(
    audio_model: Any, embed_audio: Any, *, device: Any, dtype: Any
):
    """Use the exact optimized class factory called by the serving builder."""
    from types import SimpleNamespace

    module = sys.modules["duplex_kernel.streaming_conformer"]
    gemma_model = SimpleNamespace(audio_tower=audio_model, embed_audio=embed_audio)
    return module.StreamingConformerAudioTower.from_gemma_model_optimized_cuda(
        gemma_model,
        device=device,
        dtype=dtype,
        max_sessions=128,
        max_graph_batch_size=4,
    )


def _compile_audit(args: argparse.Namespace, *, profile_only: bool = False) -> dict[str, Any]:
    """Full WAV compile, graph profiler, and replay audit on selected audio weights only."""
    import shutil
    import torch

    workspace = Path(args.workspace).resolve()
    runtime_source = Path(args.runtime_source).resolve()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    audit: dict[str, Any] = {
        "status": "started",
        "compile_mode": "torch.compile(fullgraph=True, dynamic=False, backend=inductor)",
        "checkpoint_dtype": "bfloat16",
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "reduced_precision_policy": {
            "constructor_relative_r_projection_warmup_compile_and_capture": False,
            "reference_offline_and_graph_replay": True,
            "global_flag_restored_in_try_finally": True,
        },
        "free_bytes_before_selected_weights": int(free_before),
        "total_gpu_bytes": int(total_bytes),
        "selected_checkpoint_only": True,
        "language_model_loaded": False,
        "quality_criteria": {
            "after_first_13_tokens": {"mean_minimum": 0.9995, "minimum_minimum": 0.998},
            "reference": "official full 125-tick offline Gemma4AudioModel with BF16 reduction=True",
        },
        "latency_target_ms": LATENCY_TARGET_MS,
        "latency_target_applies_to": "compiled CUDA Graph replay only",
        "profile_iteration_count": 10,
        "latency_iteration_count": BENCHMARK_ITERATIONS,
    }
    if free_before < 4 * 1024**3:
        audit.update({
            "status": "not_run_insufficient_gpu_headroom",
            "reason": "less than 4 GiB free before loading selected audio weights",
        })
        return audit

    try:
        model_path, _processor, _extractor, adapter = _load_frontend_helpers(workspace, runtime_source)
        audit["resolved_checkpoint_path"] = str(model_path)
        audit["checkpoint_tensors"] = _checkpoint_inventory(model_path)
        audio_model, embed_audio, config = _load_tower_and_embed(
            model_path, device, torch.bfloat16
        )
        free_after_weights, _ = torch.cuda.mem_get_info(device)
        audit["free_bytes_after_selected_weights"] = int(free_after_weights)
        if free_after_weights < 3 * 1024**3:
            audit.update({
                "status": "not_run_insufficient_gpu_headroom_after_weights",
                "reason": "less than 3 GiB remained after selected audio weights",
            })
            return audit

        waveform, source_rate, source_samples, resampled_samples = _read_and_resample(
            workspace / WAV_RELATIVE
        )
        offline_mel, streaming_mel, chunks = _extract_mels(adapter, waveform)
        if not np.array_equal(offline_mel, streaming_mel):
            raise RuntimeError("offline and incremental Mel features are not bitwise equal")
        audit["wav"] = {
            "path": str(workspace / WAV_RELATIVE),
            "source_rate_hz": source_rate,
            "source_samples": source_samples,
            "resampled_samples_before_trim": resampled_samples,
            "compared_duration_seconds": 10.0,
            "ticks": len(chunks),
            "mel_rows": int(offline_mel.shape[0]),
            "mel_offline_equals_incremental_bitwise": True,
            "mel_max_abs_delta": 0.0,
        }
        audit["offline_original_bf16_reduced_precision_reduction"] = True
        audit["offline_source_sample_rate"] = source_rate
        audit["offline_source_samples"] = source_samples
        audit["offline_resampled_samples"] = resampled_samples
        audit["comparison_ticks"] = TICK_COUNT
        matmul = torch.backends.cuda.matmul
        previous_reduction = matmul.allow_bf16_reduced_precision_reduction

        # Construct the reference with the original runtime reduction policy.
        try:
            matmul.allow_bf16_reduced_precision_reduction = True
            offline_features, offline_mask = _feature_tensor(
                offline_mel, device, torch.bfloat16, torch
            )
            with torch.inference_mode():
                offline_raw = _run_raw(audio_model, offline_features, offline_mask, torch)
                offline_soft = embed_audio(offline_raw)
        finally:
            matmul.allow_bf16_reduced_precision_reduction = previous_reduction
        if offline_raw.shape[1] != 250:
            raise RuntimeError(f"full offline tower returned {offline_raw.shape[1]} tokens, expected 250")
        audit["official_offline_output_shapes"] = {
            "tower_1536": list(offline_raw.shape),
            "post_embed": list(offline_soft.shape),
        }

        module = _load_streaming_module(workspace)
        slots = torch.zeros((1,), device=device, dtype=torch.long)
        mask = torch.ones((1, MEL_TICK_ROWS), device=device, dtype=torch.bool)
        reset_first = torch.ones((1,), device=device, dtype=torch.bool)
        reset_next = torch.zeros((1,), device=device, dtype=torch.bool)

        def make_step(tower: Any):
            def step(tick: Any, reset: Any):
                raw, soft, _subsampled, _layers = tower._execute(tick, mask, slots, reset)
                for canonical, staged in zip(
                    tower._canonical_banks(), tower._staging_banks(), strict=True
                ):
                    canonical.index_copy_(0, slots, staged.index_select(0, slots))
                return raw, soft
            return step

        def capture_graph(tower: Any, step_fn: Any, tick_np: np.ndarray):
            tower._zero_slots((0,))
            source_tick = torch.as_tensor(
                tick_np, device=device, dtype=torch.bfloat16
            ).unsqueeze(0).contiguous()
            static_tick = torch.empty_like(source_tick)
            graph_reset = torch.zeros((1,), device=device, dtype=torch.bool)

            def graph_step():
                static_tick.copy_(source_tick)
                return step_fn(static_tick, graph_reset)

            with torch.inference_mode():
                for _ in range(3):
                    graph_step()
            tower._zero_slots((0,))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.inference_mode():
                outputs = graph_step()
            torch.cuda.synchronize(device)
            tower._zero_slots((0,))
            return graph, source_tick, static_tick, graph_reset, outputs

        def classify_cuda_profile_events(profile: Any) -> dict[str, Any]:
            cuda_events = []
            for event in profile.events():
                if "CUDA" not in str(getattr(event, "device_type", "")).upper():
                    continue
                event_name = str(getattr(event, "name", ""))
                if not event_name:
                    continue
                time_range = getattr(event, "time_range", None)
                duration_us = float(getattr(event, "device_time_total", 0.0) or 0.0)
                if duration_us <= 0.0 and time_range is not None:
                    duration_us = max(0.0, float(time_range.end) - float(time_range.start))
                cuda_events.append((event_name, duration_us))

            totals = {key: {"count": 0, "device_time_us": 0.0} for key in (
                "conv", "gemm", "linear", "norm", "pointwise", "other"
            )}
            name_counts: Counter[str] = Counter()
            for name, duration in cuda_events:
                low = name.lower()
                name_counts[name] += 1
                if any(word in low for word in ("cudnn", "conv", "convolution")):
                    category = "conv"
                elif any(word in low for word in ("gemm", "matmul", "cublas", "cutlass", "mm_")):
                    category = "gemm"
                elif "linear" in low:
                    category = "linear"
                elif any(word in low for word in ("norm", "rms")):
                    category = "norm"
                elif any(word in low for word in ("pointwise", "fused", "triton_poi", "copy", "index", "cat")):
                    category = "pointwise"
                else:
                    category = "other"
                totals[category]["count"] += 1
                totals[category]["device_time_us"] += duration
            device_total = sum(value["device_time_us"] for value in totals.values())
            for value in totals.values():
                value["share_of_profiled_device_time"] = (
                    value["device_time_us"] / device_total if device_total else 0.0
                )
            gemm_share = (
                totals["gemm"]["share_of_profiled_device_time"]
                + totals["linear"]["share_of_profiled_device_time"]
            )
            tools = {name: shutil.which(name) for name in ("ncu", "nsys")}
            return {
                "cuda_kernel_event_count": len(cuda_events),
                "gemm_kernel_event_count": totals["gemm"]["count"],
                "linear_kernel_event_count": totals["linear"]["count"],
                "aggregate_device_time_by_category": totals,
                "total_summed_kernel_device_time_ms": device_total / 1000.0,
                "top_kernel_names": [
                    {"name": name, "count": count}
                    for name, count in name_counts.most_common(40)
                ],
                "gemm_and_linear_device_time_share": gemm_share,
                "nsight_tools": tools,
                "effective_bandwidth": {
                    "status": "not_measured_by_torch_profiler",
                    "limitation": "PyTorch CUDA profiler kernel events do not report reliable transferred-byte counts; no Nsight Compute bandwidth section was collected",
                },
            }

        compile_tower = None
        compiled = None
        compiled_tick_outputs_raw: list[Any] = []
        compiled_tick_outputs_soft: list[Any] = []
        compile_started = time.perf_counter()
        # The one-time relative-R projection, Dynamo warmup/compile, all 125
        # quality ticks, and graph capture share the same temporary policy.
        try:
            matmul.allow_bf16_reduced_precision_reduction = False
            compile_tower = module.StreamingConformerAudioTower(
                audio_model, embed_audio, device=device, dtype=torch.bfloat16, max_sessions=4
            )
            compiled = torch.compile(
                make_step(compile_tower),
                fullgraph=True,
                dynamic=False,
                backend="inductor",
            )
            with torch.inference_mode():
                for tick_index, tick_np in enumerate(chunks):
                    tick = torch.as_tensor(
                        tick_np, device=device, dtype=torch.bfloat16
                    ).unsqueeze(0).contiguous()
                    raw, soft = compiled(tick, reset_first if tick_index == 0 else reset_next)
                    compiled_tick_outputs_raw.append(raw.detach().clone())
                    compiled_tick_outputs_soft.append(soft.detach().clone())
            torch.cuda.synchronize(device)
            audit["compile_and_full_125_tick_execution_seconds"] = float(
                time.perf_counter() - compile_started
            )

            compiled_graph, source_tick, static_tick, graph_reset, compiled_static_outputs = capture_graph(
                compile_tower, compiled, chunks[-1]
            )

            eager_tower = module.StreamingConformerAudioTower(
                audio_model, embed_audio, device=device, dtype=torch.bfloat16, max_sessions=4
            )
            eager_step = make_step(eager_tower)
            eager_graph, eager_source, eager_static, eager_reset, eager_outputs = capture_graph(
                eager_tower, eager_step, chunks[-1]
            )
        finally:
            matmul.allow_bf16_reduced_precision_reduction = previous_reduction

        compiled_raw = torch.cat(compiled_tick_outputs_raw, dim=1)
        compiled_soft = torch.cat(compiled_tick_outputs_soft, dim=1)
        raw_quality = _cached_quality_thresholds(_similarity_summary(
            offline_raw[0].float().cpu().numpy(), compiled_raw[0].float().cpu().numpy()
        ))
        soft_quality = _cached_quality_thresholds(_similarity_summary(
            offline_soft[0].float().cpu().numpy(), compiled_soft[0].float().cpu().numpy()
        ))
        quality_pass = bool(
            raw_quality["passed_after_first_13_tokens"]
            and soft_quality["passed_after_first_13_tokens"]
        )
        audit["compiled_fullgraph_vs_official_offline"] = {
            "comparison_ticks": TICK_COUNT,
            "comparison_tokens": int(compiled_raw.shape[1]),
            "reference_reduction_flag": True,
            "tower_1536": raw_quality,
            "post_embed": soft_quality,
            "passed_quality": quality_pass,
        }
        audit["compile_status"] = "compiled_fullgraph_static"

        graph_quality_raw: list[Any] = []
        graph_quality_soft: list[Any] = []
        graph_reference_previous = matmul.allow_bf16_reduced_precision_reduction
        try:
            matmul.allow_bf16_reduced_precision_reduction = True
            compile_tower._zero_slots((0,))
            with torch.inference_mode():
                for tick_index, tick_np in enumerate(chunks):
                    source_tick.copy_(torch.as_tensor(
                        tick_np, device=device, dtype=torch.bfloat16
                    ).unsqueeze(0))
                    if tick_index == 0:
                        graph_reset.fill_(True)
                    else:
                        graph_reset.fill_(False)
                    compiled_graph.replay()
                    graph_quality_raw.append(compiled_static_outputs[0].detach().clone())
                    graph_quality_soft.append(compiled_static_outputs[1].detach().clone())
            torch.cuda.synchronize(device)
        finally:
            matmul.allow_bf16_reduced_precision_reduction = graph_reference_previous
        graph_raw = torch.cat(graph_quality_raw, dim=1)
        graph_soft = torch.cat(graph_quality_soft, dim=1)
        graph_raw_quality = _cached_quality_thresholds(_similarity_summary(
            offline_raw[0].float().cpu().numpy(), graph_raw[0].float().cpu().numpy()
        ))
        graph_soft_quality = _cached_quality_thresholds(_similarity_summary(
            offline_soft[0].float().cpu().numpy(), graph_soft[0].float().cpu().numpy()
        ))
        graph_quality_pass = bool(
            graph_raw_quality["passed_after_first_13_tokens"]
            and graph_soft_quality["passed_after_first_13_tokens"]
        )
        audit["compiled_graph_replay_vs_official_offline"] = {
            "comparison_ticks": TICK_COUNT,
            "comparison_tokens": int(graph_raw.shape[1]),
            "graph_replayed_with_reduction_flag_true": True,
            "tower_1536": graph_raw_quality,
            "post_embed": graph_soft_quality,
            "passed_quality": graph_quality_pass,
        }
        quality_pass = bool(quality_pass and graph_quality_pass)

        # Warmup is outside the profile window. Each captured graph writes the
        # staged tensor banks back to the canonical persistent banks.
        torch.cuda.synchronize(device)
        with torch.inference_mode():
            for _ in range(3):
                compiled_graph.replay()
        torch.cuda.synchronize(device)
        compile_tower._zero_slots((0,))
        profile = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
        )
        profiler_replay_count = 0
        with profile, torch.inference_mode():
            for _ in range(10):
                compiled_graph.replay()
                profiler_replay_count += 1
        torch.cuda.synchronize(device)
        compiled_profile = classify_cuda_profile_events(profile)
        compiled_profile.update({
            "status": "completed",
            "graph_variant": "compiled_fullgraph_static",
            "warmup_replays_excluded": 3,
            "profile_replays": profiler_replay_count,
            "tensor_cache_commit_inside_graph": True,
            "repeated_real_tick": True,
            "logical_session_cursor_metadata": "not advanced by this isolated kernel harness; persistent canonical tower cache banks are advanced by captured commit copies",
        })
        audit["compiled_graph_profiler_10_stateful_replays"] = compiled_profile

        def latency_run(graph: Any, tower: Any, label: str) -> dict[str, Any]:
            tower._zero_slots((0,))
            with torch.inference_mode():
                for _ in range(3):
                    graph.replay()
            torch.cuda.synchronize(device)
            tower._zero_slots((0,))
            events = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(BENCHMARK_ITERATIONS)
            ]
            torch.cuda.synchronize(device)
            with _ClockSampler() as telemetry:
                for start_event, end_event in events:
                    start_event.record()
                    graph.replay()
                    end_event.record()
                torch.cuda.synchronize(device)
            values = [float(start.elapsed_time(end)) for start, end in events]
            return {
                "variant": label,
                "iterations": len(values),
                "cuda_graph_replay_only_latency": _latency_summary(values),
                "acceptance_mean_and_p50_below_0_80_ms": bool(
                    float(np.mean(values)) < LATENCY_TARGET_MS
                    and float(np.percentile(values, 50)) < LATENCY_TARGET_MS
                ),
                "complete_captured_boundary": "[1,8,128] input staging -> exact incremental Q2 tower -> output_proj -> embed_audio -> staged-to-canonical tensor cache writes",
                "measured_replays_advance_tensor_cache": True,
                "replayed_tick": "same final real WAV tick on each replay; sequential persistent history advances through the captured Q2 operator",
                "event_bracket": "preallocated CUDA Events bracket only CUDAGraph.replay; no CPU copies or synchronization inside the 500 replay loop",
                "nvml_telemetry": _telemetry_summary(telemetry.samples),
            }

        compiled_latency = latency_run(compiled_graph, compile_tower, "compiled_fullgraph")
        eager_latency = latency_run(eager_graph, eager_tower, "eager")
        audit["five_hundred_replay_comparison"] = {
            "compiled": compiled_latency,
            "eager": eager_latency,
            "compiled_mean_over_eager_mean": (
                compiled_latency["cuda_graph_replay_only_latency"]["mean_ms"]
                / eager_latency["cuda_graph_replay_only_latency"]["mean_ms"]
            ),
        }
        gemm_share = float(compiled_profile["gemm_and_linear_device_time_share"])
        compiled_latency_pass = bool(compiled_latency["acceptance_mean_and_p50_below_0_80_ms"])
        audit["conditional_triton_branch"] = {
            "activation_condition": "compiled GEMM/linear device-time share >50% AND compiled graph mean and p50 >=0.80 ms",
            "gemm_linear_share_exceeded_50_percent": gemm_share > 0.50,
            "compiled_latency_target_passed": compiled_latency_pass,
            "prototype_required": bool(gemm_share > 0.50 and not compiled_latency_pass),
            "prototype_status": "not_started_until_profiler_and_latency_conditions_are_evaluated",
        }
        audit["acceptance"] = {
            "compiled_cached_vs_full_offline_original_true": {
                "passed": quality_pass,
                "raw_after13_mean": raw_quality["after_first_13_tokens"]["mean"],
                "raw_after13_min": raw_quality["after_first_13_tokens"]["min"],
                "post_embed_after13_mean": soft_quality["after_first_13_tokens"]["mean"],
                "post_embed_after13_min": soft_quality["after_first_13_tokens"]["min"],
            },
            "compiled_graph_latency_mean_and_p50_below_0_80_ms": {
                "passed": compiled_latency_pass,
                "mean_ms": compiled_latency["cuda_graph_replay_only_latency"]["mean_ms"],
                "p50_ms": compiled_latency["cuda_graph_replay_only_latency"]["p50_ms"],
            },
            "run_status_completed": True,
            "status_does_not_imply_acceptance_pass": True,
        }
        audit["status"] = "completed"
        return audit
    except Exception as exc:
        audit.update({
            "status": "compile_or_profile_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        return audit
    finally:
        if "previous_reduction" in locals():
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous_reduction


def run_compile_audit(args: argparse.Namespace) -> dict[str, Any]:
    workspace = Path(args.workspace).resolve()
    report_path = workspace / REPORT_RELATIVE
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = _compile_audit(args, profile_only=bool(args.layer_profile_only))
    if args.layer_profile_only:
        previous = report.get("torch_compile_real_checkpoint_bf16", {})
        if "eager_layer_profile_one_q2_tick" in result:
            previous["eager_layer_profile_one_q2_tick"] = result[
                "eager_layer_profile_one_q2_tick"
            ]
            previous["layer_profile_refresh_status"] = result["status"]
        else:
            previous["layer_profile_refresh_error"] = result
        report["torch_compile_real_checkpoint_bf16"] = previous
    else:
        report["torch_compile_real_checkpoint_bf16"] = result
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print(f"benchmark_json={report_path}")
    return report


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import soundfile as sf
    import torch
    import transformers

    workspace = Path(args.workspace).resolve()
    runtime_source = Path(args.runtime_source).resolve()
    previous_artifact_path = workspace / REPORT_RELATIVE
    previous_artifact = (
        json.loads(previous_artifact_path.read_text(encoding="utf-8"))
        if previous_artifact_path.is_file()
        else {}
    )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    free_before_smoke, total_bytes = torch.cuda.mem_get_info(device)
    smoke = _quick_gpu_smoke(workspace, torch)
    free_after_smoke, _ = torch.cuda.mem_get_info(device)
    if args.smoke_only:
        return {
            "status": "smoke_only_passed",
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(device),
            },
            "free_bytes_before_smoke": int(free_before_smoke),
            "free_bytes_after_smoke": int(free_after_smoke),
            "total_bytes": int(total_bytes),
            "smoke": smoke,
        }
    if free_after_smoke < 3 * 1024**3:
        raise RuntimeError(
            f"only {free_after_smoke} GPU bytes free after the smoke; refusing checkpoint load"
        )

    model_path, processor, extractor, adapter = _load_frontend_helpers(workspace, runtime_source)
    inventory = _checkpoint_inventory(model_path)
    audio_model, embed_audio, full_config = _load_tower_and_embed(
        model_path, device, torch.bfloat16
    )
    free_after_weights, _ = torch.cuda.mem_get_info(device)
    waveform, source_rate, source_samples, resampled_samples = _read_and_resample(
        workspace / WAV_RELATIVE
    )
    module = _load_streaming_module(workspace)
    offline_mel, streaming_mel, chunks = _extract_mels(adapter, waveform)
    if not np.array_equal(offline_mel, streaming_mel):
        raise RuntimeError("offline and incremental Mel features are not bitwise equal")

    matmul = torch.backends.cuda.matmul
    reduction_before_startup = bool(matmul.allow_bf16_reduced_precision_reduction)
    reduction_after_startup = reduction_before_startup
    streaming = None
    wrapped_module_count = 0
    quality_ticks_completed = 0
    parity: dict[str, Any]
    graph_results: dict[str, Any] = {}
    try:
        try:
            streaming = _capture_production_streaming_tower(
                audio_model, embed_audio, device=device, dtype=torch.bfloat16
            )
            wrapped_module_count = len(streaming._forward_wrappers)
        except Exception as exc:
            parity = {
                "status": "optimized_runtime_unavailable",
                "fallback": "W240",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        finally:
            reduction_after_startup = bool(matmul.allow_bf16_reduced_precision_reduction)

        if streaming is not None:
            try:
                parity = _run_real_parity(
                    audio_model,
                    embed_audio,
                    waveform,
                    adapter,
                    torch,
                    streaming,
                    extracted_mels=(offline_mel, streaming_mel, chunks),
                )
                quality_ticks_completed = TICK_COUNT
            except Exception as exc:
                parity = {
                    "status": "optimized_runtime_quality_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

            raw_passed = bool(
                parity.get("tower_1536", {})
                .get("cached_streaming_vs_offline", {})
                .get("passed_after_first_13_tokens", False)
            )
            post_embed_passed = bool(
                parity.get("post_embed_production", {})
                .get("cached_streaming_vs_offline", {})
                .get("passed_after_first_13_tokens", False)
            )
            quality_passed = raw_passed and post_embed_passed
            if quality_passed:
                for batch_size in (1, 2):
                    free_bytes, _ = torch.cuda.mem_get_info(device)
                    if free_bytes < 2 * 1024**3:
                        graph_results[str(batch_size)] = {
                            "status": "skipped_insufficient_gpu_headroom",
                            "free_bytes_before_500_replays": int(free_bytes),
                        }
                        continue
                    try:
                        graph_results[str(batch_size)] = _bench_graph_shape(
                            streaming,
                            audio_model,
                            embed_audio,
                            device,
                            torch.bfloat16,
                            torch,
                            chunks[-1],
                            batch_size,
                        )
                    except Exception as exc:
                        graph_results[str(batch_size)] = {
                            "status": "replay_failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
            else:
                for batch_size in (1, 2):
                    graph_results[str(batch_size)] = {
                        "status": "skipped_quality_failure",
                        "reason": "cached-vs-offline 125-tick quality acceptance did not pass",
                    }
        else:
            for batch_size in (1, 2):
                graph_results[str(batch_size)] = {
                    "status": "skipped_optimized_runtime_unavailable",
                    "reason": "startup compile/capture failed; W240 remains the explicit fallback",
                }
    finally:
        if streaming is not None:
            streaming.close()

    raw_passed = bool(
        parity.get("tower_1536", {})
        .get("cached_streaming_vs_offline", {})
        .get("passed_after_first_13_tokens", False)
    )
    post_embed_passed = bool(
        parity.get("post_embed_production", {})
        .get("cached_streaming_vs_offline", {})
        .get("passed_after_first_13_tokens", False)
    )
    quality_passed = raw_passed and post_embed_passed
    cached_vs_w240_raw = parity.get("tower_1536", {}).get("cached_streaming_vs_w240")
    cached_vs_w240_post = parity.get("post_embed_production", {}).get("cached_streaming_vs_w240")
    cached_vs_w240_raw = (
        _cached_quality_thresholds(cached_vs_w240_raw)
        if cached_vs_w240_raw is not None
        else {"passed_after_first_13_tokens": False}
    )
    cached_vs_w240_post = (
        _cached_quality_thresholds(cached_vs_w240_post)
        if cached_vs_w240_post is not None
        else {"passed_after_first_13_tokens": False}
    )
    cached_vs_w240_passed = bool(
        cached_vs_w240_raw["passed_after_first_13_tokens"]
        and cached_vs_w240_post["passed_after_first_13_tokens"]
    )
    replay_acceptance = {
        shape: (
            {
                "passed": bool(result.get("latency_target_mean_and_p50_below_0_80_ms")),
                "mean_ms": result.get("cuda_graph_replay_only_latency", {}).get("mean_ms"),
                "p50_ms": result.get("cuda_graph_replay_only_latency", {}).get("p50_ms"),
                "target_ms": LATENCY_TARGET_MS,
            }
            if result.get("status") == "measured"
            else {"status": result.get("status"), "passed": False}
        )
        for shape, result in graph_results.items()
    }
    reduction_at_end = bool(matmul.allow_bf16_reduced_precision_reduction)

    config_values = {
        "hidden_size": int(full_config.audio_config.hidden_size),
        "num_hidden_layers": int(full_config.audio_config.num_hidden_layers),
        "num_attention_heads": int(full_config.audio_config.num_attention_heads),
        "subsampling_conv_channels": list(full_config.audio_config.subsampling_conv_channels),
        "conv_kernel_size": int(full_config.audio_config.conv_kernel_size),
        "attention_chunk_size": int(full_config.audio_config.attention_chunk_size),
        "attention_context_left": int(full_config.audio_config.attention_context_left),
        "attention_context_right": int(full_config.audio_config.attention_context_right),
        "output_proj_dims": int(full_config.audio_config.output_proj_dims),
        "rms_norm_eps": float(full_config.audio_config.rms_norm_eps),
        "text_hidden_size": int(full_config.text_config.hidden_size),
    }
    return {
        "status": "completed",
        "started_at_utc": args.started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "exact incremental Gemma4AudioModel Q2 Conformer cache parity and complete frontend CUDA Graph latency",
        "runtime": {
            "container": args.container,
            "workspace": str(workspace),
            "active_service_overlay": str(runtime_source),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "transformers_modeling_source": str(
                Path(transformers.__file__).parent
                / "models"
                / "gemma4"
                / "modeling_gemma4.py"
            ),
        },
        "implementation": {
            "path": str(workspace / "duplex_kernel" / "streaming_conformer.py"),
            "sha256": hashlib.sha256(
                (workspace / "duplex_kernel" / "streaming_conformer.py").read_bytes()
            ).hexdigest(),
            "triton_kernel_path": str(
                workspace / "duplex_kernel" / "streaming_conformer_kernels.py"
            ),
            "triton_kernel_sha256": hashlib.sha256(
                (workspace / "duplex_kernel" / "streaming_conformer_kernels.py").read_bytes()
            ).hexdigest(),
        },
        "model": {
            "resolved_path": str(model_path),
            "loading_boundary": "Gemma4AudioModel and Gemma4MultimodalEmbedder only; no LM instantiated or loaded",
            "audio_config": config_values,
            "checkpoint_tensors": inventory,
        },
        "input": {
            "path": str(workspace / WAV_RELATIVE),
            "sha256": hashlib.sha256((workspace / WAV_RELATIVE).read_bytes()).hexdigest(),
            "source_sample_rate": source_rate,
            "source_samples": source_samples,
            "resampled_sample_rate": SAMPLE_RATE,
            "resampled_samples_before_trim": resampled_samples,
            "samples_used": int(waveform.size),
            "seconds_used": float(waveform.size / SAMPLE_RATE),
            "channels": 1,
            "ticks": TICK_COUNT,
            "mel_rows": TICK_COUNT * MEL_TICK_ROWS,
            "feature_extractor": type(extractor).__name__,
            "processor": type(processor).__name__,
        },
        "gpu_memory": {
            "free_bytes_before_smoke": int(free_before_smoke),
            "free_bytes_after_smoke": int(free_after_smoke),
            "free_bytes_after_selected_weights": int(free_after_weights),
            "total_bytes": int(total_bytes),
        },
        "smoke": smoke,
        "parity": parity,
        "production_streaming_api": {
            "status": (
                "optimized_runtime_ready"
                if streaming is not None
                else "startup_compile_or_capture_failed_fallback_w240"
            ),
            "api_path": str(workspace / "duplex_kernel" / "streaming_conformer.py"),
            "optimize_cuda": True,
            "default_opt_in": False,
            "max_sessions": 128,
            "compiled_function": "_compute_step",
            "compiled_call": "torch.compile(self._compute_step, fullgraph=True, dynamic=False, options={'emulate_precision_casts': True})",
            "selected_row_staging_writes": {
                "operation": "torch.Tensor.index_copy_(0, slots, next_state)",
                "banks": [
                    "_stage_mel_cache",
                    "_stage_mel_mask_cache",
                    "_stage_subsample_cache",
                    "_stage_subsample_mask_cache",
                    "_stage_keys",
                    "_stage_values",
                    "_stage_conv_inputs",
                    "_stage_token_valid",
                ],
                "scope": "N selected rows only; outside Inductor and inside CUDA graph capture through _execute",
            },
            "wrapped_gemma4_linear_instance_count": wrapped_module_count,
            "graph_shapes_captured_before_quality": [1, 2, 3, 4] if streaming is not None else [],
            "quality_ticks": quality_ticks_completed,
            "quality_completed_before_500_replays": quality_ticks_completed == TICK_COUNT,
            "cached_vs_offline_acceptance": {
                "raw_passed": raw_passed,
                "post_embed_passed": post_embed_passed,
                "passed": quality_passed,
                "required_after_first_13_tokens": {
                    "mean_minimum": 0.9995,
                    "minimum_minimum": 0.998,
                },
            },
            "cached_vs_w240": {
                "status": (
                    "measured_threshold_passed" if cached_vs_w240_passed else "measured_threshold_failed"
                    if "tower_1536" in parity
                    else "not_measured_optimized_runtime_unavailable"
                ),
                "tower_1536": cached_vs_w240_raw,
                "post_embed": cached_vs_w240_post,
                "required_after_first_13_tokens": {
                    "mean_minimum": 0.9995,
                    "minimum_minimum": 0.998,
                },
                "passed": cached_vs_w240_passed,
            },
            "replay_target_acceptance_n1_n2": replay_acceptance,
            "overall_acceptance": {
                "cached_vs_offline_after_first_13_tokens": quality_passed,
                "cached_vs_w240_after_first_13_tokens": cached_vs_w240_passed,
                "cuda_graph_latency_mean_and_p50_below_target_n1_n2": bool(
                    replay_acceptance.get("1", {}).get("passed", False)
                    and replay_acceptance.get("2", {}).get("passed", False)
                ),
                "passed": bool(
                    quality_passed
                    and cached_vs_w240_passed
                    and replay_acceptance.get("1", {}).get("passed", False)
                    and replay_acceptance.get("2", {}).get("passed", False)
                ),
            },
            "global_bf16_reduced_precision_reduction": {
                "original_value": reduction_before_startup,
                "value_after_ctor_compile_warmup_and_capture": reduction_after_startup,
                "value_at_end": reduction_at_end,
                "restored_after_startup": reduction_after_startup is reduction_before_startup,
                "restored_at_end": reduction_at_end is reduction_before_startup,
                "scope_value_during_startup": False,
            },
            "fallback": "W240 rolling frontend, with compile/capture diagnostic retained above",
            "status_completed_does_not_imply_acceptance_pass": True,
        },
        "cuda_graph_complete_frontend": {
            "boundary": "[N,8,128] device tensor input through input staging, exact cached tower, one embed_audio, output clone, and transactional persistent-state commit",
            "iterations_per_shape": BENCHMARK_ITERATIONS,
            "iterations_run_only_after_125_tick_offline_quality_pass": True,
            "warmup_compile_and_capture_excluded": True,
            "concurrent_nvml_telemetry": True,
            "target_ms": LATENCY_TARGET_MS,
            "shapes": graph_results,
        },
        "candidate_snapshot_evidence_not_runtime_api": {
            "provenance": "pre-integration candidate harness measurements supplied for this task",
            "status": "snapshot_candidate_not_production_runtime",
            "cached_vs_offline_after_first_13_tokens": {
                "tower_1536_mean": 0.9999464154,
                "tower_1536_min": 0.9993313551,
                "post_embed_mean": 0.9999531507,
                "post_embed_min": 0.9992180467,
                "passed_stated_offline_threshold": True,
            },
            "cached_vs_w240_after_first_13_tokens": {
                "tower_1536_mean": 0.999706924,
                "tower_1536_min": 0.996979177,
                "post_embed_mean": 0.999754608,
                "post_embed_min": 0.997372150,
                "acceptance_status": "measured_not_passed_or_certified",
            },
            "compiled_candidate_replay_cuda_only_ms_n1": {
                "mean": 1.681170,
                "p50": 1.681408,
                "target_0_80_ms_passed": False,
            },
            "compiled_candidate_replay_cuda_only_ms_n2": {
                "mean": 1.848268,
                "p50": 1.848320,
                "target_0_80_ms_passed": False,
            },
            "triton_without_compile_cuda_only_ms_n1_n2": [3.669, 3.911],
            "linear_graph_reference_exact_kernel_p50_ms": {
                "cuda_graph_reference": 0.023552,
                "triton_tile32x128": 0.016384,
                "bitwise_equal": True,
            },
            "prior_compile_report_section": previous_artifact.get(
                "torch_compile_real_checkpoint_bf16"
            ),
        },
        "measurement_limits": {
            "service_state": "active build14 service was left running and unchanged",
            "latency_scope": "audio frontend only; excludes PCM/Mel CPU extractor, native LM/policy, scheduler, TTS, and playback",
            "post_embed_included": True,
            "persistent_state_update_included": True,
            "clock_policy": "read only; no GPU clock changes",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument(
        "--runtime-source",
        default=(
            "/workspace/artifacts/runtime/"
            "control6_runtime_pendingpcm_phaseonly_q2cachemeta_build14_stableturn_q2_context_h3000_eos_20261021"
        ),
    )
    parser.add_argument("--container", default="gemma4-listener-control6-h3000-eos-build14-20261021")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--compile-audit-only", action="store_true")
    parser.add_argument("--layer-profile-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=BENCHMARK_ITERATIONS)
    args = parser.parse_args()
    if args.iterations != BENCHMARK_ITERATIONS:
        raise ValueError("the requested evidence requires exactly 500 iterations")
    args.started_at_utc = datetime.now(timezone.utc).isoformat()
    if args.compile_audit_only or args.layer_profile_only:
        run_compile_audit(args)
        return
    report = run_benchmark(args)
    print(json.dumps(report, indent=2, default=str))
    if not args.smoke_only:
        output = Path(args.workspace).resolve() / REPORT_RELATIVE
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"benchmark_json={output}")


if __name__ == "__main__":
    main()
