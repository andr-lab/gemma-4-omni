from __future__ import annotations

import math
from typing import Any

import torch
from duplex_kernel.speaker_runtime import Code2WavWindowGraph, Code2WavWindowGraphFamily


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
