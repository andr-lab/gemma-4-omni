"""Fixed-tensor QueryFiLM-v2 to native Qwen Q1 parity probe.

Default mode is a CPU-only cache/adapter preflight.  Model loading and the
native forward require the explicit ``--run-gpu`` switch; no download path is
used and Hugging Face offline mode is enabled before the cached model loads.
The response hidden used by the GPU probe is deterministic synthetic h_txt,
because the prior dialogue capture did not preserve that tensor.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER_CHECKPOINT = ROOT / "checkpoints" / "frame_residual_adapter_emovo_query_film_v2_candidate.pt"
DEFAULT_NATIVE_LIBRARY = Path("/workspace/native/qwen_talker/libqwen_talker.so")
EXPECTED_NATIVE_LIBRARY_SHA256 = "0b2303a0902ca61c375b5babe159f6ce6afd93d77291d935bf16e2469cbeff44"
EXPECTED_NATIVE_ABI = 4
EXPECTED_NATIVE_TRACE_ABI = 3
PARITY_LIMITS = {"mae": 0.01, "max_abs": 0.05, "cosine_min": 0.999}
FIXED_SEED = 7319
SYNTHETIC_RESPONSE_TEXT = "Fixed-tensor QueryFiLM/native parity probe."
CACHE_PRODUCER_MODULES = (
    "qwen_tts.core.models.modeling_qwen3_tts",
)
OPTIONAL_CACHE_PRODUCER_PREFIXES = ("transformers.models.qwen3",)
SOURCE_RELATIVE_PATHS = (
    "tools/probe_qwen_native_film_parity.py",
    "tests/kernel/test_qwen_native_film_parity_probe.py",
    "duplex_kernel/engine.py",
    "duplex_kernel/frame_residual_adapter.py",
    "duplex_kernel/future_prefix.py",
    "duplex_kernel/models/talker.py",
    "duplex_kernel/native_qwen_talker.py",
    "duplex_kernel/native_trace.py",
    "duplex_kernel/qwen_conditioning.py",
    "native/qwen_talker/Makefile",
    "native/qwen_talker/qwen_talker.cu",
    "native/qwen_talker/qwen_talker.h",
    "native/qwen_talker/qwen_talker_aten_attention.cpp",
    "native/qwen_talker/qwen_talker_aten_attention.h",
    "native/qwen_talker/qwen_talker_aten_rmsnorm.cpp",
    "native/qwen_talker/qwen_talker_aten_rmsnorm.h",
    "native/qwen_talker/qwen_talker_kernels.cu",
    "native/qwen_talker/qwen_talker_kernels.cuh",
    "native/qwen_talker/generated/qwen_native_topology.h",
    "validation/kernel_gates/QWEN_NATIVE/QWEN_NATIVE_TOPOLOGY.json",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_workspace_import_path(workspace: str | Path, search_path: list[str] | None = None) -> str:
    """Make the workspace packages importable when this script is run by path."""
    target = str(Path(workspace).resolve())
    paths = sys.path if search_path is None else search_path
    qwen_source = str(Path(target) / "qwen3tts_src")
    for value in (target, qwen_source):
        if Path(value).is_dir() and value not in paths:
            # Match TalkerModel._install_shims order: qwen3tts_src precedes root.
            paths.insert(0, value)
    return target


def _expected_qwen_revision(root: Path) -> str:
    source = (root / "duplex_kernel" / "models" / "talker.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PINNED_QWEN_TALKER_REVISION"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and value:
                return value
    raise RuntimeError("PINNED_QWEN_TALKER_REVISION is not a literal string")


def _snapshot_files(root: Path) -> tuple[Path, dict[str, Any]]:
    revision = _expected_qwen_revision(root)
    snapshot = (
        root
        / ".hf_cache"
        / "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
        / "snapshots"
        / revision
    )
    required = ("config.json", "model.safetensors")
    tokenizer_names = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    files: dict[str, Any] = {
        "revision": revision,
        "path": str(snapshot),
        "required": {},
        "tokenizer_assets": {},
    }
    for name in required:
        path = snapshot / name
        files["required"][name] = {
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    for name in tokenizer_names:
        path = snapshot / name
        files["tokenizer_assets"][name] = {
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    tokenizer_json_ready = files["tokenizer_assets"]["tokenizer.json"]["exists"]
    tokenizer_legacy_ready = all(
        files["tokenizer_assets"][name]["exists"]
        for name in ("tokenizer_config.json", "vocab.json", "merges.txt")
    )
    files["tokenizer_loadable_assets"] = bool(tokenizer_json_ready or tokenizer_legacy_ready)
    files["ready"] = bool(
        all(item["exists"] for item in files["required"].values())
        and files["tokenizer_loadable_assets"]
    )
    if files["ready"]:
        files["sha256"] = {
            name: sha256_file(snapshot / name)
            for name in (*required, *(name for name in tokenizer_names if files["tokenizer_assets"][name]["exists"]))
        }
    return snapshot, files


def _module_available(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": spec is not None, "origin": None if spec is None else spec.origin}


def _native_binary_file_identity(
    library_path: str | Path,
    *,
    expected_sha256: str | None = EXPECTED_NATIVE_LIBRARY_SHA256,
) -> dict[str, Any]:
    path = Path(library_path).expanduser().resolve()
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "expected_sha256": expected_sha256,
    }
    if path.is_file():
        result["sha256"] = sha256_file(path)
        result["sha256_matches_expected"] = (
            expected_sha256 is None or result["sha256"] == expected_sha256
        )
    return result


def _native_binary_attestation(
    library_path: str | Path,
    *,
    expected_sha256: str | None = EXPECTED_NATIVE_LIBRARY_SHA256,
    wrapper_path: Path | None = None,
) -> dict[str, Any]:
    path = Path(library_path).expanduser().resolve()
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "expected_sha256": expected_sha256,
        "abi_version": None,
        "trace_abi_version": None,
        "missing_wrapper_symbols": [],
        "load_error": None,
    }
    if not path.is_file():
        return result
    result["sha256"] = sha256_file(path)
    result["sha256_matches_expected"] = (
        expected_sha256 is None or result["sha256"] == expected_sha256
    )
    try:
        library = ctypes.CDLL(str(path))
        abi_fn = library.qwen_talker_abi_version
        trace_abi_fn = library.qwen_talker_trace_abi_version
        abi_fn.restype = ctypes.c_uint32
        trace_abi_fn.restype = ctypes.c_uint32
        result["abi_version"] = int(abi_fn())
        result["trace_abi_version"] = int(trace_abi_fn())
        if wrapper_path is not None:
            wrapper_source = wrapper_path.read_text(encoding="utf-8")
            names = sorted(set(re.findall(r"(?:self\.)?_lib\.(qwen_talker_[A-Za-z0-9_]+)", wrapper_source)))
            missing: list[str] = []
            for name in names:
                try:
                    getattr(library, name)
                except AttributeError:
                    missing.append(name)
            result["wrapper_symbols_checked"] = names
            result["missing_wrapper_symbols"] = missing
    except (AttributeError, OSError) as exc:
        result["load_error"] = f"{type(exc).__name__}: {exc}"
    result["ready"] = bool(
        result["abi_version"] == EXPECTED_NATIVE_ABI
        and result["trace_abi_version"] == EXPECTED_NATIVE_TRACE_ABI
        and result.get("sha256_matches_expected", False)
        and not result["missing_wrapper_symbols"]
        and result["load_error"] is None
    )
    return result


def _cache_producer_source_attestation(workspace: Path) -> dict[str, Any]:
    qwen_source_root = (workspace / "qwen3tts_src").resolve()
    modules: dict[str, Any] = {}
    loaded_names = {
        name
        for name in sys.modules
        if name in CACHE_PRODUCER_MODULES
        or any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in OPTIONAL_CACHE_PRODUCER_PREFIXES
        )
    }
    for name in sorted(loaded_names | set(CACHE_PRODUCER_MODULES)):
        module = sys.modules.get(name)
        filename = None if module is None else getattr(module, "__file__", None)
        source_path = Path(filename).resolve() if isinstance(filename, str) else None
        source_exists = bool(source_path is not None and source_path.is_file())
        record = {
            "available": module is not None,
            "loaded": module is not None,
            "path": None if source_path is None else str(source_path),
            "exists": source_exists,
            "sha256": sha256_file(source_path) if source_exists else None,
        }
        if name.startswith("qwen_tts."):
            record["inside_qwen3tts_src"] = bool(
                source_exists and source_path.is_relative_to(qwen_source_root)
            )
        modules[name] = record
    missing_required = [
        name
        for name in CACHE_PRODUCER_MODULES
        if not modules[name]["loaded"] or not modules[name]["exists"]
    ]
    qwen_record = modules[CACHE_PRODUCER_MODULES[0]]
    ready = (
        not missing_required
        and all(record["exists"] and record.get("sha256") for record in modules.values() if record["loaded"])
        and qwen_record.get("inside_qwen3tts_src") is True
    )
    return {
        "qwen3tts_source_root": str(qwen_source_root),
        "modules": modules,
        "missing_required_modules": missing_required,
        "ready": bool(ready),
    }


def _activate_talker_imports_without_weights(workspace: Path) -> None:
    """Mirror load_talker import activation without constructing/loading a model."""
    from duplex_kernel.models.talker import _install_shims

    _install_shims(str(workspace))
    import qwen_tts.core.models  # noqa: F401
    import safetensors.torch  # noqa: F401
    from transformers import AutoTokenizer  # noqa: F401
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (  # noqa: F401
        Qwen3TTSTokenizerV2Model,
    )


def source_attestation(
    root: str | Path,
    native_library: str | Path,
    *,
    expected_library_sha256: str | None = EXPECTED_NATIVE_LIBRARY_SHA256,
    include_native_abi: bool = True,
) -> dict[str, Any]:
    workspace = Path(root).resolve()
    sources: dict[str, Any] = {}
    for relative in SOURCE_RELATIVE_PATHS:
        path = workspace / relative
        sources[relative] = {
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    checkpoint = workspace / DEFAULT_ADAPTER_CHECKPOINT.relative_to(ROOT)
    sources[str(checkpoint.relative_to(workspace))] = {
        "exists": checkpoint.is_file(),
        "sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
        "size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
    }
    missing_sources = [name for name, record in sources.items() if not record["exists"]]
    cache_producer_sources = _cache_producer_source_attestation(workspace)
    return {
        "workspace": str(workspace),
        "source_hashes": sources,
        "missing_sources": missing_sources,
        "cache_producer_sources": cache_producer_sources,
        "native_library": (
            _native_binary_attestation(
                native_library,
                expected_sha256=expected_library_sha256,
                wrapper_path=workspace / "duplex_kernel" / "native_qwen_talker.py",
            )
            if include_native_abi
            else _native_binary_file_identity(
                native_library,
                expected_sha256=expected_library_sha256,
            )
        ),
        "binary_build_mapping": (
            "Source files and DSO are independently hashed; this artifact does not prove which exact source files were used to build the DSO."
        ),
    }


def verify_cpu_preflight_handoff(
    preflight: dict[str, Any],
    *,
    workspace: str | Path,
    native_library: str | Path,
    current_attestation: dict[str, Any],
    current_snapshot: dict[str, Any],
    verify_native_abi: bool = True,
) -> None:
    """Fail closed unless the current probe inputs match one completed CPU preflight."""
    root = Path(workspace).resolve()
    library = Path(native_library).expanduser().resolve()
    if (
        preflight.get("mode") != "cpu_preflight"
        or preflight.get("ready_for_explicit_gpu_probe") is not True
        or preflight.get("gpu_used") is not False
        or preflight.get("model_loaded") is not False
    ):
        raise RuntimeError("GPU probe requires a ready CPU preflight manifest with no GPU/model load")

    expected = preflight.get("attestation")
    if not isinstance(expected, dict) or expected.get("workspace") != str(root):
        raise RuntimeError("CPU preflight workspace does not match the GPU probe workspace")
    if current_attestation.get("workspace") != str(root):
        raise RuntimeError("current GPU probe attestation resolved a different workspace")
    expected_sources = expected.get("source_hashes")
    current_sources = current_attestation.get("source_hashes")
    checkpoint_relative = DEFAULT_ADAPTER_CHECKPOINT.relative_to(ROOT).as_posix()
    required_sources = {*SOURCE_RELATIVE_PATHS, checkpoint_relative}
    if (
        not isinstance(expected_sources, dict)
        or not required_sources.issubset(expected_sources)
        or not isinstance(current_sources, dict)
        or set(current_sources) != set(expected_sources)
        or expected.get("missing_sources") != []
        or current_attestation.get("missing_sources") != []
    ):
        raise RuntimeError("CPU preflight source inventory is incomplete or changed")
    for relative, expected_record in expected_sources.items():
        current_record = current_sources.get(relative)
        if (
            not isinstance(expected_record, dict)
            or expected_record.get("exists") is not True
            or not isinstance(expected_record.get("sha256"), str)
            or not isinstance(current_record, dict)
            or current_record != expected_record
        ):
            raise RuntimeError(f"GPU probe source differs from CPU preflight: {relative}")

    expected_producers = expected.get("cache_producer_sources")
    if not isinstance(expected_producers, dict) or expected_producers.get("ready") is not True:
        raise RuntimeError("CPU preflight cache-producer import activation is incomplete")

    expected_native = expected.get("native_library")
    current_native = current_attestation.get("native_library")
    if not isinstance(expected_native, dict) or expected_native.get("ready") is not True:
        raise RuntimeError("CPU preflight native DSO attestation is not ready")
    if expected_native.get("path") != str(library) or not isinstance(current_native, dict):
        raise RuntimeError("native DSO path differs from CPU preflight")
    if verify_native_abi:
        if current_native != expected_native:
            raise RuntimeError("native DSO path, hash, ABI, or exports differ from CPU preflight")
    else:
        identity_fields = (
            "path",
            "exists",
            "expected_sha256",
            "sha256",
            "sha256_matches_expected",
        )
        if any(current_native.get(key) != expected_native.get(key) for key in identity_fields):
            raise RuntimeError("native DSO file identity differs from CPU preflight")
        if current_native.get("exists") is not True or current_native.get("sha256_matches_expected") is not True:
            raise RuntimeError("native DSO file identity is not valid")

    expected_snapshot = preflight.get("qwen_talker_snapshot")
    expected_snapshot_hashes = (
        expected_snapshot.get("sha256") if isinstance(expected_snapshot, dict) else None
    )
    if (
        not isinstance(expected_snapshot, dict)
        or expected_snapshot.get("ready") is not True
        or not isinstance(expected_snapshot_hashes, dict)
        or not expected_snapshot_hashes
        or current_snapshot.get("ready") is not True
        or current_snapshot.get("revision") != expected_snapshot.get("revision")
        or current_snapshot.get("sha256") != expected_snapshot_hashes
    ):
        raise RuntimeError("cached Qwen model/tokenizer assets differ from CPU preflight")


def verify_native_binary_handoff(
    preflight: dict[str, Any],
    *,
    native_library: str | Path,
    current_native: dict[str, Any],
) -> None:
    expected = preflight.get("attestation", {}).get("native_library")
    path = Path(native_library).expanduser().resolve()
    if (
        not isinstance(expected, dict)
        or expected.get("ready") is not True
        or expected.get("path") != str(path)
        or current_native != expected
        or current_native.get("ready") is not True
    ):
        raise RuntimeError("native ABI/DSO differs from CPU preflight")


def summarize_tensor(tensor: Any) -> dict[str, Any]:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor summary requires a torch.Tensor")
    values = tensor.detach().float()
    finite = bool(torch.isfinite(values).all().item())
    record: dict[str, Any] = {
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
        "finite": finite,
    }
    if values.numel() and finite:
        record.update(
            minimum=float(values.min().item()),
            maximum=float(values.max().item()),
            mean=float(values.mean().item()),
            rms=float(torch.sqrt(torch.mean(values.square())).item()),
        )
    return record


def _load_adapter(checkpoint: str | Path, *, device: str, dtype: Any):
    from duplex_kernel.frame_residual_adapter import load_frame_residual_adapter

    adapter, metadata = load_frame_residual_adapter(checkpoint, device=device, dtype=dtype)
    adapter.eval().requires_grad_(False)
    return adapter, metadata


def make_cpu_fixture(
    checkpoint: str | Path = DEFAULT_ADAPTER_CHECKPOINT,
    *,
    seed: int = FIXED_SEED,
) -> dict[str, Any]:
    import torch

    adapter, metadata = _load_adapter(checkpoint, device="cpu", dtype=torch.bfloat16)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    response_hidden = torch.randn((1, 1, 1536), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    base_query = torch.randn((1, 1, 2048), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    with torch.inference_mode():
        inference_mode = torch.is_inference_mode_enabled()
        adapted_query = adapter(response_hidden, base_query)
        delta = adapted_query - base_query
        learned_scale = float(adapter.scale.detach().float().cpu().item())
    return {
        "device": "cpu",
        "seed": int(seed),
        "synthetic_response_hidden": True,
        "inference_mode": bool(inference_mode),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_metadata": metadata,
        "learned_scale": learned_scale,
        "response_hidden": response_hidden,
        "base_query": base_query,
        "delta": delta,
        "adapted_query": adapted_query,
        "summaries": {
            "response_hidden": summarize_tensor(response_hidden),
            "base_query": summarize_tensor(base_query),
            "delta": summarize_tensor(delta),
            "adapted_query": summarize_tensor(adapted_query),
            "mean_shift": float(
                adapted_query.float().mean().sub(base_query.float().mean()).item()
            ),
        },
    }


def compare_tensor_pair(name: str, expected: Any, actual: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    record: dict[str, Any] = {"name": str(name), "tolerances": dict(PARITY_LIMITS)}
    if tuple(expected.shape) != tuple(actual.shape):
        record.update(
            passed=False,
            error=f"shape mismatch: {tuple(expected.shape)} != {tuple(actual.shape)}",
        )
        return record
    left = expected.detach().float()
    right = actual.detach().float()
    if not bool(torch.isfinite(left).all().item()) or not bool(torch.isfinite(right).all().item()):
        record.update(passed=False, error="non-finite tensor values")
        return record
    difference = (left - right).abs()
    cosine = float(
        functional.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item()
    )
    metrics = {
        "mae": float(difference.mean().item()),
        "max_abs": float(difference.max().item()),
        "cosine": cosine,
    }
    record.update(
        metrics=metrics,
        passed=bool(
            metrics["mae"] <= PARITY_LIMITS["mae"]
            and metrics["max_abs"] <= PARITY_LIMITS["max_abs"]
            and metrics["cosine"] >= PARITY_LIMITS["cosine_min"]
        ),
    )
    return record


def validate_audio_q1_contract(
    inputs_embeds: Any,
    position_ids: Any,
    cache_position: Any,
    *,
    expected_position: int,
) -> dict[str, Any]:
    import torch

    if not isinstance(inputs_embeds, torch.Tensor) or tuple(inputs_embeds.shape) != (1, 1, 2048):
        raise ValueError("audio Q1 input must have shape [1,1,2048]")
    if inputs_embeds.dtype != torch.bfloat16 or not inputs_embeds.is_contiguous():
        raise ValueError("audio Q1 input must be contiguous BF16")
    if not bool(torch.isfinite(inputs_embeds.float()).all().item()):
        raise ValueError("audio Q1 input contains non-finite values")
    if not isinstance(position_ids, torch.Tensor) or tuple(position_ids.shape) != (3, 1, 1):
        raise ValueError("audio Q1 position_ids must have shape [3,1,1]")
    if not isinstance(cache_position, torch.Tensor) or tuple(cache_position.shape) != (1,):
        raise ValueError("audio Q1 cache_position must have shape [1]")
    positions = [int(value) for value in position_ids.reshape(-1).tolist()]
    cache_value = int(cache_position.item())
    expected = int(expected_position)
    if positions != [expected, expected, expected] or cache_value != expected:
        raise ValueError(
            f"audio Q1 rotary/cache cursor mismatch: positions={positions}, cache={cache_value}, expected={expected}"
        )
    return {"audio_position": expected, "frame_zero_preserved": expected > 0}


def q1_cache_cursor_diagnostic(reference: Any, observer_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Capture absolute cache slots and the next audio-frame cursor before gating."""
    import torch

    frame_ordinal = int(reference._current_frame_id)
    audio_start = int(reference._audio_start)
    expected_position = audio_start + frame_ordinal
    cache = reference._slot_cache
    layer_lengths: dict[str, int] = {}
    if cache is not None:
        for layer in sorted(cache._keys):
            layer_lengths[str(int(layer))] = int(cache.get_seq_length(int(layer)))
    default_length = None if cache is None else int(cache.get_seq_length())

    writes: list[dict[str, Any]] = []
    for event in observer_events:
        positions = event.get("cache_position")
        if not isinstance(positions, torch.Tensor):
            continue
        values = [int(value) for value in positions.detach().cpu().reshape(-1).tolist()]
        writes.append(
            {
                "phase": event.get("phase"),
                "frame_id": event.get("frame_id"),
                "positions": values,
            }
        )
    observed_max = max(
        (position for event in writes for position in event["positions"]),
        default=None,
    )
    all_layers_match = bool(layer_lengths) and all(
        length == expected_position for length in layer_lengths.values()
    )
    return {
        "prefix_start": int(reference._prefix_start),
        "audio_start": audio_start,
        "audio_frame_ordinal": frame_ordinal,
        "next_audio_absolute_position": expected_position,
        "cache_sequence_length_before_q1": default_length,
        "cache_max_absolute_position": (
            None if cache is None else int(getattr(cache, "_max_position", -1))
        ),
        "cache_layer_sequence_lengths": layer_lengths,
        "last_observed_write_max_position": observed_max,
        "last_observed_write_max_plus_one": None if observed_max is None else observed_max + 1,
        "observed_cache_writes": writes,
        "all_cache_layers_match_next_audio_position": all_layers_match,
        "cache_cursor_matches_next_audio_position": (
            default_length == expected_position and all_layers_match
        ),
    }


class _Q1CacheCursorError(RuntimeError):
    def __init__(self, message: str, diagnostic: dict[str, Any]) -> None:
        super().__init__(message)
        self.q1_cache_diagnostic = diagnostic


def finalize_gpu_parity_report(report: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless every output/cache and last-layer tap gate passed."""
    finalized = dict(report)
    all_gates_pass = (
        finalized.get("all_final_parity_gates_pass") is True
        and finalized.get("all_last_layer_qk_tap_gates_pass") is True
    )
    finalized["all_parity_gates_pass"] = all_gates_pass
    finalized["status"] = "complete" if all_gates_pass else "parity_gate_failed"
    return finalized


def validate_gpu_worker_report(
    worker_exit_code: int,
    worker_report: dict[str, Any],
    *,
    expected_cpu_preflight_sha256: str,
    expected_probe_sha256: str,
) -> dict[str, Any]:
    """Preserve worker failures; require full provenance only for success reports."""
    handoff = worker_report.get("cpu_preflight_manifest")
    handoff = handoff if isinstance(handoff, dict) else {}
    handoff_matches = handoff.get("sha256") == expected_cpu_preflight_sha256
    worker_completed = worker_exit_code == 0 and worker_report.get("status") == "complete"
    if not worker_completed:
        primary_error = worker_report.get("error")
        if not primary_error:
            primary_error = (
                f"GPU worker exited with code {worker_exit_code} "
                f"and status {worker_report.get('status')!r}"
            )
        return {
            "worker_completed": False,
            "primary_error": str(primary_error),
            "worker_exit_code": int(worker_exit_code),
            "worker_status": worker_report.get("status"),
            "cpu_preflight_manifest": handoff or None,
            "cpu_preflight_manifest_matches": handoff_matches,
            "probe_source_validation": "not_applicable_worker_failed",
        }

    if not handoff_matches:
        raise RuntimeError("GPU worker did not consume the exact CPU preflight manifest")
    probe_hash = (
        worker_report.get("source_attestation", {}).get("source_hashes", {})
        .get("tools/probe_qwen_native_film_parity.py", {}).get("sha256")
    )
    if probe_hash != expected_probe_sha256:
        raise RuntimeError("GPU worker source hash differs from the passing CPU preflight")
    return {
        "worker_completed": True,
        "worker_exit_code": int(worker_exit_code),
        "worker_status": worker_report.get("status"),
        "cpu_preflight_manifest": handoff,
        "cpu_preflight_manifest_matches": True,
        "probe_source_sha256": probe_hash,
        "probe_source_validation": "matched",
    }


def _assemble_gpu_parity_report(
    *,
    created_at_unix_ns: int,
    completed_at_unix_ns: int,
    qwen_talker_revision: str,
    qwen_snapshot_path: str,
    qwen_snapshot_files: dict[str, Any],
    source_attestation: dict[str, Any],
    cpu_preflight_manifest: dict[str, Any],
    adapter: dict[str, Any],
    fixture: dict[str, Any],
    q1_cursor: dict[str, Any],
    eager_attention_taps_final_layer: dict[str, Any],
    native_vs_eager_last_layer_qk_taps: dict[str, Any],
    native_vs_eager_output_and_cache: dict[str, Any],
    native_runtime: dict[str, Any],
) -> dict[str, Any]:
    """Build the exact GPU artifact from already captured evidence sections."""
    cache_sources = source_attestation.get("cache_producer_sources", {})
    modules = cache_sources.get("modules", {}) if isinstance(cache_sources, dict) else {}
    external_python_source_hashes = {
        name: record["sha256"]
        for name, record in modules.items()
        if isinstance(record, dict)
        and record.get("loaded") is True
        and isinstance(record.get("sha256"), str)
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "mode": "fixed_tensor_q1_film_v2_native_vs_eager",
        "created_at_unix_ns": created_at_unix_ns,
        "completed_at_unix_ns": completed_at_unix_ns,
        "elapsed_ms": (completed_at_unix_ns - created_at_unix_ns) / 1_000_000.0,
        "gpu_used": True,
        "model_loaded": "cached Qwen3-TTS Talker only; Gemma not loaded; Code2Wav excluded",
        "qwen_talker_revision": qwen_talker_revision,
        "qwen_snapshot_path": qwen_snapshot_path,
        "qwen_snapshot_files": qwen_snapshot_files,
        "source_attestation": source_attestation,
        "cpu_preflight_manifest": cpu_preflight_manifest,
        "external_python_source_hashes": external_python_source_hashes,
        "adapter": adapter,
        "fixture": fixture,
        "q1_cursor": q1_cursor,
        "eager_attention_taps_final_layer": eager_attention_taps_final_layer,
        "native_vs_eager_last_layer_qk_taps": native_vs_eager_last_layer_qk_taps,
        "all_last_layer_qk_tap_gates_pass": all(
            item["passed"] for item in native_vs_eager_last_layer_qk_taps.values()
        ),
        "native_vs_eager_output_and_cache": native_vs_eager_output_and_cache,
        "all_final_parity_gates_pass": all(
            item["passed"] for item in native_vs_eager_output_and_cache.values()
        ),
        "native_runtime": native_runtime,
        "limitations": [
            "This is fixed-tensor numerical parity, not speech-quality or conversation evidence.",
            "Synthetic h_txt is not a captured Gemma response hidden state.",
            "Native-vs-eager CUDA execution is intentionally deferred until explicit parent authorization.",
            "Source hashes and the DSO hash do not alone prove which exact source files were used to build the DSO.",
        ],
    }
    return finalize_gpu_parity_report(report)


def gpu_parity_exit_code(report: dict[str, Any]) -> int:
    return 0 if report.get("status") == "complete" and report.get("all_parity_gates_pass") is True else 3


def _preflight_report(
    root: Path,
    native_library: Path,
    *,
    expected_library_sha256: str | None = EXPECTED_NATIVE_LIBRARY_SHA256,
) -> dict[str, Any]:
    import torch

    workspace = root.resolve()
    snapshot, snapshot_record = _snapshot_files(workspace)
    cpu_fixture = make_cpu_fixture(workspace / "checkpoints" / DEFAULT_ADAPTER_CHECKPOINT.name)
    imports = {
        name: _module_available(name)
        for name in ("qwen_tts", "transformers", "safetensors")
    }
    importable = all(record["available"] for record in imports.values())
    if importable:
        _activate_talker_imports_without_weights(workspace)
    assets = source_attestation(
        workspace,
        native_library,
        expected_library_sha256=expected_library_sha256,
    )
    ready = bool(
        snapshot_record["ready"]
        and not assets["missing_sources"]
        and assets["cache_producer_sources"]["ready"]
        and assets["native_library"]["ready"]
        and importable
    )
    return {
        "schema_version": 1,
        "mode": "cpu_preflight",
        "created_at_unix_ns": time.time_ns(),
        "ready_for_explicit_gpu_probe": ready,
        "gpu_used": False,
        "model_loaded": False,
        "torch": {
            "version": str(torch.__version__),
            "cuda_initialized": bool(torch.cuda.is_initialized()),
            "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "qwen_talker_snapshot": snapshot_record,
        "qwen_talker_snapshot_sha256": {
            key: value for key, value in snapshot_record.get("sha256", {}).items()
        },
        "python_imports": imports,
        "adapter_fixture": {
            "synthetic_response_hidden": cpu_fixture["synthetic_response_hidden"],
            "inference_mode": cpu_fixture["inference_mode"],
            "checkpoint_sha256": cpu_fixture["checkpoint_sha256"],
            "learned_scale": cpu_fixture["learned_scale"],
            "summaries": cpu_fixture["summaries"],
        },
        "attestation": assets,
        "limitations": [
            "CPU preflight activates the production Talker import shims and Qwen modules with CUDA hidden, but does not construct/load model weights or perform a Qwen forward.",
            "The fixed response hidden is synthetic because the prior capture did not preserve h_txt tensors.",
            "Source hashes and the DSO hash are recorded independently; build provenance is not inferred from matching hashes.",
        ],
    }


def cpu_preflight(
    root: str | Path = ROOT,
    native_library: str | Path = DEFAULT_NATIVE_LIBRARY,
    *,
    expected_library_sha256: str | None = EXPECTED_NATIVE_LIBRARY_SHA256,
) -> dict[str, Any]:
    """Read cached assets and run the adapter fixture on CPU only."""
    return _preflight_report(
        Path(root), Path(native_library), expected_library_sha256=expected_library_sha256
    )


def _tensor_digest(tensor: Any) -> str:
    raw = tensor.detach().contiguous().cpu().view(__import__("torch").uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _capture_eager_last_layer(model: Any):
    """Install temporary hooks for the final layer's Q/K projection, norm and RoPE."""
    layers = getattr(model, "layers", None)
    try:
        layer_count = len(layers)
        target_layer = layers[-1] if layer_count else None
    except (TypeError, IndexError, KeyError):
        layer_count = 0
        target_layer = None
    if not layer_count or target_layer is None:
        raise RuntimeError("Qwen model does not expose decoder layers")
    attention = getattr(target_layer, "self_attn", None)
    if attention is None:
        raise RuntimeError("last Qwen layer has no self_attn module")
    capture: dict[str, Any] = {
        "rotary_call_count": 0,
        "target_layer_index": layer_count - 1,
        "rotary_function_name": "apply_multimodal_rotary_pos_emb",
    }
    handles = []

    def record(name: str):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            capture[name] = value.detach().clone()
        return hook

    for name in ("q_proj", "k_proj", "q_norm", "k_norm"):
        module = getattr(attention, name, None)
        if module is None or not hasattr(module, "register_forward_hook"):
            raise RuntimeError(f"Qwen final attention missing {name}")
        handles.append(module.register_forward_hook(record(name)))

    forward = attention.forward
    globals_map = getattr(forward, "__globals__", None)
    # Qwen3-TTS TalkerAttention.forward calls the multimodal variant directly.
    # The same module also defines apply_rotary_pos_emb for other model paths;
    # patching that unused helper silently leaves the Q/K/RoPE taps incomplete.
    rotary_name = "apply_multimodal_rotary_pos_emb"
    original_rotary = None if globals_map is None else globals_map.get(rotary_name)
    if not callable(original_rotary):
        for handle in handles:
            handle.remove()
        raise RuntimeError(f"Qwen attention has no discoverable {rotary_name}")

    def rotary_hook(*args, **kwargs):
        result = original_rotary(*args, **kwargs)
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise RuntimeError("unexpected apply_rotary_pos_emb result")
        capture["rotary_call_count"] += 1
        capture["q_rope"] = result[0].detach().clone()
        capture["k_rope"] = result[1].detach().clone()
        return result

    globals_map[rotary_name] = rotary_hook

    def restore() -> None:
        globals_map[rotary_name] = original_rotary
        for handle in handles:
            handle.remove()

    return capture, restore


def _cache_view_for_native(slot_cache: Any, sequence_length: int):
    from types import SimpleNamespace

    layers = []
    for index in range(28):
        if index not in slot_cache._keys or index not in slot_cache._values:
            raise RuntimeError(f"eager cache missing Qwen layer {index}")
        key = slot_cache._keys[index][:, :, :sequence_length, :].contiguous()
        value = slot_cache._values[index][:, :, :sequence_length, :].contiguous()
        layers.append(SimpleNamespace(keys=key, values=value))
    return SimpleNamespace(layers=layers, get_seq_length=lambda: int(sequence_length))


def _native_tap_comparisons(native: Any, eager: dict[str, Any]) -> dict[str, Any]:
    # Tap IDs come from native/qwen_talker/qwen_talker.h.
    specs = (
        ("q_proj", 1, (1, 2048)),
        ("k_proj", 2, (1, 1024)),
        ("q_norm", 4, (16, 128)),
        ("k_norm", 5, (8, 128)),
        ("q_rope", 6, (16, 1, 128)),
        ("k_rope", 7, (8, 1, 128)),
    )
    required = [name for name, _tap_id, _shape in specs]
    missing = [name for name in required if name not in eager]
    if missing:
        raise RuntimeError(
            "eager Q/K/RoPE tap capture is incomplete: "
            f"missing={missing}, rotary_function={eager.get('rotary_function_name')!r}, "
            f"rotary_call_count={eager.get('rotary_call_count')!r}"
        )
    output: dict[str, Any] = {}
    for name, tap_id, native_shape in specs:
        native_value = native.debug_copy_tap_async(tap_id, native_shape)
        eager_value = eager[name]
        if name in {"q_proj", "k_proj"}:
            eager_value = eager_value.reshape(native_shape)
        elif name in {"q_norm", "k_norm", "q_rope", "k_rope"}:
            if eager_value.ndim == 4 and eager_value.shape[0] == 1:
                eager_value = eager_value.squeeze(0)
            eager_value = eager_value.reshape(native_shape)
        output[name] = compare_tensor_pair(name, eager_value, native_value)
    return output


def verify_cache_producer_runtime_sources(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> None:
    if expected.get("ready") is not True:
        raise RuntimeError(
            "CPU preflight is missing required cache-producer module(s): "
            + ", ".join(expected.get("missing_required_modules", []))
        )
    actual_modules = actual.get("modules")
    expected_modules = expected.get("modules")
    if actual.get("ready") is not True or not isinstance(actual_modules, dict):
        raise RuntimeError(
            "required cache-producer module missing at runtime: "
            + ", ".join(actual.get("missing_required_modules", []))
        )
    if not isinstance(expected_modules, dict):
        raise RuntimeError("CPU preflight cache-producer module inventory is invalid")
    if set(actual_modules) != set(expected_modules):
        missing = sorted(set(expected_modules) - set(actual_modules))
        unexpected = sorted(set(actual_modules) - set(expected_modules))
        raise RuntimeError(
            f"runtime cache-producer module set differs from CPU preflight: missing={missing}, unexpected={unexpected}"
        )
    for module, record in expected_modules.items():
        runtime_record = actual_modules[module]
        if not record.get("loaded") or not record.get("exists") or not isinstance(record.get("sha256"), str):
            raise RuntimeError(f"CPU preflight cache-producer module is invalid: {module}")
        if runtime_record.get("path") != record.get("path"):
            raise RuntimeError(
                f"cache-producer module path differs for {module}: "
                f"expected={record.get('path')}, actual={runtime_record.get('path')}"
            )
        if runtime_record.get("sha256") != record.get("sha256"):
            raise RuntimeError(f"cache-producer module source hash differs for {module}: {record.get('path')}")


def run_gpu_probe(
    *,
    workspace: str | Path = Path("/workspace"),
    native_library: str | Path = DEFAULT_NATIVE_LIBRARY,
    output_path: str | Path,
    preflight_manifest_path: str | Path | None = None,
    cpu_preflight_handoff_record: dict[str, Any] | None = None,
    expected_library_sha256: str | None = EXPECTED_NATIVE_LIBRARY_SHA256,
    seed: int = FIXED_SEED,
) -> dict[str, Any]:
    """Run one fixed adapted Q1 input through eager and native Qwen backends."""
    root = Path(workspace).resolve()
    dso = Path(native_library).resolve()
    if preflight_manifest_path is None:
        raise RuntimeError("GPU probe requires a CPU preflight manifest")
    manifest_path = Path(preflight_manifest_path).resolve()
    try:
        preflight = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read CPU preflight manifest: {manifest_path}: {exc}") from exc
    if not isinstance(preflight, dict):
        raise RuntimeError("CPU preflight manifest must contain a JSON object")
    if not dso.is_file():
        raise FileNotFoundError(f"native library missing: {dso}")
    # This is scoped to the child probe process and makes both loader and wrapper
    # resolve the exact binary attested by the preflight.
    os.environ["QWEN_NATIVE_LIBRARY_PATH"] = str(dso)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    attestation = source_attestation(
        root,
        dso,
        expected_library_sha256=expected_library_sha256,
        include_native_abi=False,
    )
    snapshot, snapshot_record = _snapshot_files(root)
    verify_cpu_preflight_handoff(
        preflight,
        workspace=root,
        native_library=dso,
        current_attestation=attestation,
        current_snapshot=snapshot_record,
        verify_native_abi=False,
    )
    consumed_manifest = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "created_at_unix_ns": preflight.get("created_at_unix_ns"),
    }
    if cpu_preflight_handoff_record is not None:
        cpu_preflight_handoff_record.update(consumed_manifest)
    if not snapshot_record["ready"]:
        raise FileNotFoundError(f"pinned cached Qwen snapshot incomplete: {snapshot_record}")

    # Source, weights, and DSO file identity are bound before importing torch.
    # The CUDA runtime dependency is not necessarily loadable until Torch has
    # initialized its shared-library search paths, so validate the DSO ABI next.
    import torch

    native_binary = _native_binary_attestation(
        dso,
        expected_sha256=expected_library_sha256,
        wrapper_path=root / "duplex_kernel" / "native_qwen_talker.py",
    )
    verify_native_binary_handoff(
        preflight,
        native_library=dso,
        current_native=native_binary,
    )
    attestation["native_library"] = native_binary

    if not torch.cuda.is_available():
        raise RuntimeError("explicit GPU probe requires CUDA")
    for package in ("qwen_tts", "transformers", "safetensors"):
        if not _module_available(package)["available"]:
            raise ModuleNotFoundError(f"required cached import unavailable: {package}")

    from duplex_kernel.frame_residual_adapter import load_frame_residual_adapter
    from duplex_kernel.models.talker import load_talker
    from duplex_kernel.native_qwen_talker import (
        NativeQwenTalker,
        QWEN_NATIVE_HEAD_DIM,
        QWEN_NATIVE_NUM_ATTN_HEADS,
        QWEN_NATIVE_NUM_KV_HEADS,
        QWEN_NATIVE_NUM_LAYERS,
    )
    import duplex_kernel.future_prefix as future_prefix_module
    from duplex_kernel.future_prefix import FuturePrefixQwen

    # Verify the actual production import activation before constructing/loading
    # the Talker weights. CPU preflight used the same loader shims and imports.
    _activate_talker_imports_without_weights(root)
    runtime_cache_sources = _cache_producer_source_attestation(root)
    verify_cache_producer_runtime_sources(
        preflight["attestation"]["cache_producer_sources"],
        runtime_cache_sources,
    )
    attestation["cache_producer_sources"] = runtime_cache_sources

    if QWEN_NATIVE_NUM_LAYERS != 28 or QWEN_NATIVE_NUM_ATTN_HEADS != 16 or QWEN_NATIVE_NUM_KV_HEADS != 8 or QWEN_NATIVE_HEAD_DIM != 128:
        raise RuntimeError("native ABI topology differs from the attested Qwen Q1 probe contract")

    device = torch.device("cuda:0")
    qwen_revision = _expected_qwen_revision(root)
    talker = load_talker(
        str(root),
        device=str(device),
        load_code2wav=False,
        qwen_talker_revision=qwen_revision,
    )
    talker.talker.eval().requires_grad_(False)
    adapter, checkpoint_metadata = load_frame_residual_adapter(
        root / "checkpoints" / DEFAULT_ADAPTER_CHECKPOINT.name,
        device=str(device),
        dtype=torch.bfloat16,
    )
    adapter.eval().requires_grad_(False)

    generator = torch.Generator(device=device).manual_seed(int(seed))
    prefix_frames = torch.randn(
        (2, 1, 2048), generator=generator, device=device, dtype=torch.float32
    ).to(torch.bfloat16)
    response_hidden = torch.randn(
        (1, 1, 1536), generator=generator, device=device, dtype=torch.float32
    ).to(torch.bfloat16)
    reference = FuturePrefixQwen(
        talker,
        lookahead_frames=0,
        max_prefix_frames=2,
        max_audio_frames=4,
        device=device,
        dtype=torch.bfloat16,
        runtime_backend="pytorch",
    )
    native = None
    builder_original = future_prefix_module.build_qwen_audio_conditioning
    adapter_application: dict[str, Any] = {"applied_generation_steps": [], "other_generation_steps": []}
    observed_target: dict[str, Any] = {}
    observer_events: list[dict[str, Any]] = []
    reference._input_observer = observer_events.append
    eager_taps: dict[str, Any] = {}
    restore_taps = None
    start_ns = time.time_ns()

    def conditioning_with_film(*args, **kwargs):
        base = builder_original(*args, **kwargs)
        step = kwargs.get("generation_step")
        if step == 1:
            if adapter_application["applied_generation_steps"]:
                raise RuntimeError("fixed probe unexpectedly applied FiLM more than once")
            with torch.inference_mode():
                adapted = adapter(response_hidden, base)
                delta = adapted - base
            adapter_application["applied_generation_steps"].append(int(step))
            observed_target.update(
                base=base.detach().clone(),
                delta=delta.detach().clone(),
                adapted=adapted.detach().clone(),
                h_txt=response_hidden.detach().clone(),
                inference_mode=torch.is_inference_mode_enabled(),
            )
            return adapted
        if step is not None:
            adapter_application["other_generation_steps"].append(int(step))
        return base

    try:
        with torch.inference_mode():
            reference.begin(SYNTHETIC_RESPONSE_TEXT)
            reference.push_v9_frame(0, prefix_frames[0])
            frame0 = reference.step_frame()
            reference.commit_codes(torch.zeros((16,), device=device, dtype=torch.long))
            reference.push_v9_frame(1, prefix_frames[1])
            frame1 = reference.step_frame()
            reference.commit_codes(torch.ones((16,), device=device, dtype=torch.long))
        if reference._current_frame_id != 2 or reference._cached_prefix_frames != 2:
            raise RuntimeError("fixed probe did not preserve initial frames/prefix cache")
        audio_position = int(reference._audio_start + reference._current_frame_id)
        cursor_diagnostic = q1_cache_cursor_diagnostic(reference, observer_events)
        if not cursor_diagnostic["cache_cursor_matches_next_audio_position"]:
            raise _Q1CacheCursorError(
                "eager cache metadata does not reach the next absolute audio Q1 position",
                cursor_diagnostic,
            )

        native = NativeQwenTalker(talker.talker, max_cache_len=reference._slot_cache.max_cache_len)
        if native._native_abi_version != EXPECTED_NATIVE_ABI:
            raise RuntimeError(f"native ABI mismatch after load: {native._native_abi_version}")
        if native._library_sha256 != native_binary["sha256"]:
            raise RuntimeError("NativeQwenTalker resolved a DSO different from CPU preflight")
        native_cache = _cache_view_for_native(reference._slot_cache, audio_position)
        native.import_dynamic_cache_async(native_cache, max_audio_frames=1)

        # Install the builder adapter only around the target generation_step=1
        # eager call. Frame zero and the generation_step=0 query remain unchanged.
        future_prefix_module.build_qwen_audio_conditioning = conditioning_with_film
        model = getattr(getattr(talker, "talker", talker), "model", None)
        if model is None:
            raise RuntimeError("loaded Qwen Talker has no decoder model")
        eager_taps, restore_taps = _capture_eager_last_layer(model)
        with torch.inference_mode():
            eager_result = reference.step_frame()
        if restore_taps is not None:
            restore_taps()
            restore_taps = None
        future_prefix_module.build_qwen_audio_conditioning = builder_original
        if adapter_application["applied_generation_steps"] != [1]:
            raise RuntimeError(f"target FiLM step not observed exactly once: {adapter_application}")
        if eager_result.frame_id != 2 or not observed_target:
            raise RuntimeError("adapted Q1 target input was not observed")

        target_observer = next(
            (
                item
                for item in reversed(observer_events)
                if item.get("phase") == "audio" and item.get("frame_id") == 2
            ),
            None,
        )
        if target_observer is None:
            raise RuntimeError("input observer did not capture target Q1 input")
        q1_contract = validate_audio_q1_contract(
            target_observer["inputs_embeds"],
            target_observer["position_ids"],
            target_observer["cache_position"],
            expected_position=audio_position,
        )
        if not torch.equal(target_observer["inputs_embeds"], observed_target["adapted"]):
            raise RuntimeError("observer input differs from the single adapted query")
        with torch.inference_mode():
            native_hidden = native.step_audio_async(observed_target["adapted"].contiguous())
        torch.cuda.synchronize(device)

        native_last_layer = QWEN_NATIVE_NUM_LAYERS - 1
        eager_cache = reference._slot_cache
        eager_key = eager_cache._keys[native_last_layer][:, :, audio_position : audio_position + 1, :].detach().clone()
        eager_value = eager_cache._values[native_last_layer][:, :, audio_position : audio_position + 1, :].detach().clone()
        native_key_cache, native_value_cache = native.debug_copy_layer_cache_async(native_last_layer)
        native_key = native_key_cache[:, :, audio_position : audio_position + 1, :]
        native_value = native_value_cache[:, :, audio_position : audio_position + 1, :]
        native_taps = _native_tap_comparisons(native, eager_taps)
        if eager_taps.get("rotary_call_count") != QWEN_NATIVE_NUM_LAYERS:
            raise RuntimeError(
                "eager RoPE capture did not cover exactly one attention call per layer: "
                f"{eager_taps.get('rotary_call_count')} != {QWEN_NATIVE_NUM_LAYERS}"
            )
        final_comparisons = {
            "last_hidden": compare_tensor_pair("last_hidden", eager_result.last_hidden, native_hidden),
            "last_layer_audio_key_row": compare_tensor_pair("last_layer_audio_key_row", eager_key, native_key),
            "last_layer_audio_value_row": compare_tensor_pair("last_layer_audio_value_row", eager_value, native_value),
        }
        completed_at_ns = time.time_ns()
        report = _assemble_gpu_parity_report(
            created_at_unix_ns=start_ns,
            completed_at_unix_ns=completed_at_ns,
            qwen_talker_revision=str(talker.qwen_talker_revision),
            qwen_snapshot_path=str(talker.qwen_snapshot_path),
            qwen_snapshot_files=snapshot_record,
            source_attestation=attestation,
            cpu_preflight_manifest=consumed_manifest,
            adapter={
                "checkpoint": str((root / "checkpoints" / DEFAULT_ADAPTER_CHECKPOINT.name).resolve()),
                "checkpoint_sha256": sha256_file(root / "checkpoints" / DEFAULT_ADAPTER_CHECKPOINT.name),
                "metadata": checkpoint_metadata,
                "learned_scale": float(adapter.scale.detach().float().cpu().item()),
                "requires_grad": bool(any(parameter.requires_grad for parameter in adapter.parameters())),
                "inference_mode": bool(observed_target["inference_mode"]),
            },
            fixture={
                "synthetic_h_txt": True,
                "h_txt_reason": "the earlier dialogue capture did not retain the actual response hidden tensor",
                "seed": int(seed),
                "prompt": SYNTHETIC_RESPONSE_TEXT,
                "response_hidden": summarize_tensor(observed_target["h_txt"]),
                "base_query": summarize_tensor(observed_target["base"]),
                "delta": summarize_tensor(observed_target["delta"]),
                "adapted_query": summarize_tensor(observed_target["adapted"]),
                "mean_shift": float(
                    observed_target["adapted"].float().mean().sub(observed_target["base"].float().mean()).item()
                ),
                "frame0_bos_hidden_summary": summarize_tensor(frame0.last_hidden),
                "generation_step0_hidden_summary": summarize_tensor(frame1.last_hidden),
                "adapter_application": adapter_application,
                "query_sha256": _tensor_digest(observed_target["adapted"]),
                "same_adapted_tensor_to_native_and_eager": True,
            },
            q1_cursor={
                **q1_contract,
                **cursor_diagnostic,
                "frame_id": int(eager_result.frame_id),
                "visible_prefix_end": int(eager_result.visible_prefix_end),
                "position_ids": [int(value) for value in target_observer["position_ids"].reshape(-1).tolist()],
                "cache_position": int(target_observer["cache_position"].item()),
                "frame_zero_preserved": int(eager_result.frame_id) > 0,
            },
            eager_attention_taps_final_layer={
                key: summarize_tensor(value)
                for key, value in eager_taps.items()
                if isinstance(value, torch.Tensor)
            },
            native_vs_eager_last_layer_qk_taps=native_taps,
            native_vs_eager_output_and_cache=final_comparisons,
            native_runtime={
                "library_path": native._library_path,
                "library_sha256": native._library_sha256,
                "abi_version": int(native._native_abi_version),
                "trace_abi_version": int(native._lib.qwen_talker_trace_abi_version()),
                "graph_enabled": native._q1_cuda_graph is not None,
                "qwen_topology_sha256": native._topology.get("topology_sha256"),
                "debug_taps_are_final_layer": True,
            },
        )
        return report
    finally:
        future_prefix_module.build_qwen_audio_conditioning = builder_original
        if restore_taps is not None:
            restore_taps()
        if native is not None:
            native.close()
        reference.close()


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu-preflight", action="store_true", help="default: only validate cached assets on CPU")
    mode.add_argument("--run-gpu", action="store_true", help="explicitly run one fixed-tensor Q1 parity probe")
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--native-library", type=Path, default=DEFAULT_NATIVE_LIBRARY)
    parser.add_argument("--expected-library-sha256", default=EXPECTED_NATIVE_LIBRARY_SHA256)
    parser.add_argument(
        "--preflight-manifest",
        type=Path,
        help="CPU preflight JSON whose exact sources, cached weights, and DSO must match this GPU probe",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen_native_film_parity_probe.json"))
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    args = parser.parse_args(argv)
    ensure_workspace_import_path(args.workspace)
    is_gpu = bool(args.run_gpu)
    if not is_gpu:
        # Must precede torch import in the CPU-preflight path.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cpu_preflight_handoff_record: dict[str, Any] = {}
    try:
        if is_gpu:
            if args.preflight_manifest is None:
                raise RuntimeError("GPU mode requires a ready CPU preflight manifest (--preflight-manifest)")
            report = run_gpu_probe(
                workspace=args.workspace,
                native_library=args.native_library,
                output_path=args.output,
                preflight_manifest_path=args.preflight_manifest,
                cpu_preflight_handoff_record=cpu_preflight_handoff_record,
                expected_library_sha256=args.expected_library_sha256,
                seed=args.seed,
            )
        else:
            report = cpu_preflight(
                args.workspace,
                args.native_library,
                expected_library_sha256=args.expected_library_sha256,
            )
        _write_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        if is_gpu:
            return gpu_parity_exit_code(report)
        return 0 if report.get("ready_for_explicit_gpu_probe") else 2
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed_closed",
            "mode": "gpu_probe" if is_gpu else "cpu_preflight",
            "gpu_used": False if not is_gpu else None,
            "error": f"{type(exc).__name__}: {exc}",
        }
        diagnostic = getattr(exc, "q1_cache_diagnostic", None)
        if isinstance(diagnostic, dict):
            failure["q1_cache_diagnostic"] = diagnostic
        if cpu_preflight_handoff_record:
            failure["cpu_preflight_manifest"] = dict(cpu_preflight_handoff_record)
        _write_json(args.output, failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
