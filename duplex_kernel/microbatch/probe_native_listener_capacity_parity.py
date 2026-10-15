#!/usr/bin/env python3
"""Sequential baseline/candidate parity probe for Gemma listener staging capacity.

The GPU worker reuses the existing ``NativeGemmaListenerLM`` ABI and native
diagnostics.  It loads the same attested BF16 checkpoint for each DSO in a
separate process, because loading two DSOs with identical exported C names in
one process risks symbol interposition.  No serving process, native DSO, model,
attention policy, or sequence-capacity setting is modified by this tool.

Run ``--help`` for the two-stage worker/assembler protocol.  A GPU run is only
valid after the managed-container maintenance preflight has pinned both DSOs,
the source manifests, checkpoint, and empty GPU ownership barrier.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Keep both ``python -m ...`` and direct script invocation usable.  The
# sequential executor launches this file as a child process, where Python
# otherwise puts only tools/kernel_gates on sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from duplex_kernel.microbatch.native_listener_capacity_probe_contract import (
    BASELINE_RESET_DIAGNOSTIC_CASES,
    BASELINE_RESET_DIAGNOSTIC_SCHEMA,
    ProbeContractError,
    EXPECTED_DEBUG_LAYER_ROLES,
    EXPECTED_DEBUG_LAYERS,
    REPORT_SCHEMA,
    TOLERANCES,
    WORKER_SCHEMA,
    atomic_write_json,
    compare_tensor_arrays,
    compare_reset_reprime_output,
    assemble_baseline_reset_diagnostic,
    sha256_file,
    validate_worker_report,
)


PROBE_SEED = 20261011
PREFIX_LEN = 34
PREFILL_COMPUTE_LEN = 291
Q_LEN = 4
SLIDING_WINDOW = 512
CROSSING_START = 510
CROSSING_END = 514
LONG_ORDERED_POSITION = 2049
LONG_ORDERED_END = LONG_ORDERED_POSITION + Q_LEN
MAX_SEQ_LEN = 16384
NATIVE_LIBRARY_ENV = "GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"
_VERSIONED_CUDART_NAME = re.compile(r"libcudart\.so\.\d+(?:\.\d+)*\Z")


def load_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema") != "gemma4-native-listener-capacity-manifest/v1":
        raise ProbeContractError("probe manifest schema mismatch")
    for role in ("baseline", "candidate"):
        record = value.get("libraries", {}).get(role)
        if not isinstance(record, dict):
            raise ProbeContractError(f"manifest has no {role} library record")
        if not isinstance(record.get("path"), str) or not record["path"]:
            raise ProbeContractError(f"manifest has no {role} library path")
        if not isinstance(record.get("source_manifest_path"), str):
            raise ProbeContractError(f"manifest has no {role} source manifest path")
        for name in ("sha256", "source_manifest_sha256"):
            digest = record.get(name)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ProbeContractError(f"manifest {role}.{name} is not pinned")
    settings = value.get("runtime")
    if settings != {
        "abi_version": 3,
        "attention_backend_id": 3,
        "rmsnorm_backend_id": 2,
        "max_seq_len": MAX_SEQ_LEN,
        "stream_q_len": Q_LEN,
        "enable_cuda_graph": False,
    }:
        raise ProbeContractError("runtime settings must match the reviewed native policy")
    tolerances = value.get("tolerances")
    if tolerances != TOLERANCES:
        raise ProbeContractError("manifest tolerances differ from the pinned native parity gate")
    return value


def _verify_source_manifest(record: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(record["source_manifest_path"]))
    if not manifest_path.is_file():
        raise ProbeContractError(f"source manifest missing: {manifest_path}")
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash.lower() != str(record["source_manifest_sha256"]).lower():
        raise ProbeContractError("source manifest file hash differs from pinned manifest")
    root_value = record.get("source_root")
    if not isinstance(root_value, str) or not root_value:
        raise ProbeContractError("source_root must point to the immutable build source snapshot")
    root = Path(root_value)
    override_root_value = record.get("source_override_root")
    override_root = Path(override_root_value).resolve() if override_root_value else None
    overrides = record.get("source_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ProbeContractError("source_overrides must be a path map")
    verified_files = 0
    seen_paths: set[str] = set()
    used_overrides: set[str] = set()
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ProbeContractError(f"malformed source manifest row {line_number}")
        relative = parts[1].lstrip("* ")
        if relative in seen_paths:
            raise ProbeContractError(f"duplicate source manifest path at row {line_number}: {relative}")
        seen_paths.add(relative)
        if relative in overrides:
            if override_root is None:
                raise ProbeContractError("source_override_root required when overrides exist")
            source = Path(str(overrides[relative])).resolve()
            allowed_root = override_root
            used_overrides.add(relative)
        else:
            source = (root / relative).resolve()
            allowed_root = root.resolve()
        if allowed_root not in source.parents:
            raise ProbeContractError(f"source manifest path escapes its pinned root: {relative}")
        if not source.is_file() or sha256_file(source).lower() != parts[0].lower():
            raise ProbeContractError(f"source snapshot mismatch at row {line_number}: {relative}")
        verified_files += 1
    if verified_files == 0:
        raise ProbeContractError("source manifest has no source files")
    unused_overrides = sorted(set(overrides) - used_overrides)
    if unused_overrides:
        raise ProbeContractError(f"unused source override entries: {unused_overrides}")
    return {
        "path": str(manifest_path),
        "sha256": actual_manifest_hash,
        "source_root": str(root),
        "verified_file_count": verified_files,
        "source_override_root": str(override_root) if override_root else None,
        "source_override_count": len(overrides),
    }


def _worker_environment(
    base_environment: Mapping[str, str], library_record: Mapping[str, Any]
) -> dict[str, str]:
    """Create an isolated offline child environment pinned to one DSO."""
    library_path = library_record.get("path")
    if not isinstance(library_path, str) or not library_path.strip():
        raise ProbeContractError("worker library path must be non-empty")
    result = dict(base_environment)
    result["HF_HUB_OFFLINE"] = "1"
    result["TRANSFORMERS_OFFLINE"] = "1"
    result[NATIVE_LIBRARY_ENV] = library_path
    return result


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _torch_tensor_sha256(tensor: Any, torch: Any) -> str:
    cpu = tensor.detach().to(device="cpu").contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return _sha_bytes(raw)


def _tensor_array(tensor: Any) -> np.ndarray:
    return tensor.detach().float().to(device="cpu").contiguous().numpy().copy()


def _array_sha256(array: np.ndarray) -> str:
    return _sha_bytes(np.ascontiguousarray(array).view(np.uint8).tobytes())


def _load_bound_worker_arrays(
    report: Mapping[str, Any], supplied_path: str | Path
) -> dict[str, np.ndarray]:
    """Load only the exact, hashed NPZ and tensors attested by a worker report."""
    reported_path = report.get("arrays_path")
    if not isinstance(reported_path, str) or not reported_path.strip():
        raise ProbeContractError("worker arrays_path is missing")
    expected_path = Path(reported_path).resolve()
    actual_path = Path(supplied_path).resolve()
    if actual_path != expected_path:
        raise ProbeContractError(
            f"arrays_path does not match worker report: {actual_path} != {expected_path}"
        )
    if not actual_path.is_file():
        raise ProbeContractError(f"worker tensor archive missing: {actual_path}")
    expected_file_hash = report.get("arrays_file_sha256")
    actual_file_hash = sha256_file(actual_path)
    if not isinstance(expected_file_hash, str) or actual_file_hash.lower() != expected_file_hash.lower():
        raise ProbeContractError("arrays_file_sha256 mismatch")
    metadata = report.get("arrays")
    if not isinstance(metadata, Mapping) or not metadata:
        raise ProbeContractError("worker arrays metadata is missing or empty")
    try:
        archive = np.load(actual_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProbeContractError(f"worker tensor archive is unreadable: {exc}") from exc
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ProbeContractError("worker tensor archive is not an NPZ")
    with archive:
        archive_names = set(archive.files)
        metadata_names = set(metadata)
        if archive_names != metadata_names:
            missing = sorted(metadata_names - archive_names)
            unexpected = sorted(archive_names - metadata_names)
            raise ProbeContractError(
                f"worker array names mismatch; missing={missing}; unexpected={unexpected}"
            )
        loaded: dict[str, np.ndarray] = {}
        for name in sorted(metadata_names):
            expected = metadata[name]
            if not isinstance(expected, Mapping):
                raise ProbeContractError(f"array metadata for {name} is not an object")
            value = np.asarray(archive[name])
            if value.dtype != np.dtype("float32"):
                raise ProbeContractError(f"worker array dtype mismatch for {name}: {value.dtype}")
            if list(value.shape) != expected.get("shape"):
                raise ProbeContractError(f"worker array shape mismatch for {name}")
            if _array_sha256(value).lower() != str(expected.get("sha256", "")).lower():
                raise ProbeContractError(f"worker array sha256 mismatch for {name}")
            if not np.isfinite(value).all():
                raise ProbeContractError(f"worker array contains non-finite values: {name}")
            loaded[name] = value.copy()
    return loaded


def _memory_snapshot(torch: Any, device: Any) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "cuda_free_bytes": int(free_bytes),
        "cuda_total_bytes": int(total_bytes),
        "torch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _sync(torch: Any, device: Any) -> None:
    torch.cuda.synchronize(device)


def _record_tensor(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
    name: str,
    tensor: Any,
) -> None:
    value = _tensor_array(tensor)
    arrays[name] = value
    metadata[name] = {
        "shape": list(value.shape),
        "dtype": "float32_from_bf16_or_native_tensor",
        "sha256": _array_sha256(value),
    }


def _preserve_repeat_evidence(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
    repeat_arrays: Mapping[str, np.ndarray],
    repeat_metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    """Copy baseline self-repeat tensors into the outer failure-safe NPZ set."""
    if set(repeat_arrays) != set(repeat_metadata):
        raise ProbeContractError("repeat array values and metadata names differ")
    for name, value in repeat_arrays.items():
        if not name.startswith("repeat."):
            raise ProbeContractError(f"repeat evidence array lacks repeat namespace: {name}")
        if name in arrays:
            raise ProbeContractError(f"repeat evidence would overwrite an existing array: {name}")
        copied = np.asarray(value).copy()
        item = dict(repeat_metadata[name])
        if item.get("shape") != list(copied.shape) or item.get("sha256") != _array_sha256(copied):
            raise ProbeContractError(f"repeat evidence metadata mismatch for {name}")
        arrays[name] = copied
        metadata[name] = item


def _fixed_inputs(torch: Any, lm: Any, device: Any) -> tuple[Any, Any, Any, Any, str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PROBE_SEED)
    vocab_size = int(getattr(lm.config, "vocab_size", 0))
    if vocab_size < PREFIX_LEN + 2:
        raise ProbeContractError(f"unexpected vocabulary size: {vocab_size}")
    token_ids_cpu = (torch.arange(1, PREFIX_LEN + 1, dtype=torch.long) % (vocab_size - 1)) + 1
    token_ids = token_ids_cpu.reshape(1, PREFIX_LEN).to(device=device)
    prefix_embeds = lm.embed_tokens(token_ids).to(dtype=torch.bfloat16).contiguous()
    prefix_ple = lm.get_per_layer_inputs(token_ids, prefix_embeds).to(dtype=torch.bfloat16).contiguous()
    prefix_attention = torch.ones((1, PREFIX_LEN), device=device, dtype=torch.int64).contiguous()
    frames_cpu = (
        torch.randn((121, 1, Q_LEN, 1536), generator=generator, dtype=torch.float32) * 0.05
    ).to(dtype=torch.bfloat16).contiguous()
    frames = frames_cpu.to(device=device).contiguous()
    input_parts = {
        "token_ids": _torch_tensor_sha256(token_ids, torch),
        "prefix_embeds": _torch_tensor_sha256(prefix_embeds, torch),
        "prefix_ple": _torch_tensor_sha256(prefix_ple, torch),
        "prefix_attention": _torch_tensor_sha256(prefix_attention, torch),
        "frames_q4": _torch_tensor_sha256(frames_cpu, torch),
    }
    input_sha256 = _sha_bytes(json.dumps(input_parts, sort_keys=True).encode("utf-8"))
    return prefix_embeds, prefix_ple, prefix_attention, frames, input_sha256


def _make_dynamic_cache(torch: Any, DynamicCache: Any, lm: Any, layer_values: Mapping[int, tuple[Any, Any]]) -> Any:
    cache = DynamicCache(config=lm.config)
    for layer_idx, (key, value) in sorted(layer_values.items()):
        cache.update(key.contiguous(), value.contiguous(), int(layer_idx))
    return cache


def _versioned_cudart_candidate(value: str | Path) -> str | None:
    candidate = str(value).strip()
    if not candidate:
        return None
    path = Path(candidate)
    if not path.is_absolute():
        # A bare SONAME is only a lookup hint. Never pass it to CDLL: the
        # ambient loader search path could bind a different CUDA runtime.
        return None
    resolved = path.resolve(strict=False)
    if not resolved.is_file() or not _VERSIONED_CUDART_NAME.fullmatch(resolved.name):
        return None
    return str(resolved)


def _load_cuda_runtime_for_cache_copy(
    torch_module: Any,
    *,
    find_library: Any = ctypes.util.find_library,
    cdll_factory: Any = ctypes.CDLL,
    maps_path: str | Path = "/proc/self/maps",
) -> tuple[Any, str, str]:
    """Bind cudaMemcpy from the installed, versioned runtime without aliases."""
    mapped_candidates: list[str] = []
    packaged_candidates: list[str] = []

    def add(target: list[str], value: str | Path | None) -> None:
        if value is None:
            return
        try:
            normalized = _versioned_cudart_candidate(value)
        except OSError:
            return
        if normalized and normalized not in target:
            target.append(normalized)

    # A CUDA runtime already loaded by PyTorch is an authoritative installed
    # path. Resolve it directly instead of depending on an unversioned alias.
    try:
        for line in Path(maps_path).read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or "libcudart.so" not in fields[5]:
                continue
            mapped_path = fields[5].removesuffix(" (deleted)")
            add(mapped_candidates, mapped_path)
    except OSError:
        pass

    # CUDA wheels may bundle the runtime under either torch/lib or the NVIDIA
    # runtime wheel. Search only these installed package locations.
    torch_file = getattr(torch_module, "__file__", None)
    if isinstance(torch_file, str):
        torch_root = Path(torch_file).resolve(strict=False).parent
        for library_dir in (
            torch_root / "lib",
            torch_root.parent / "nvidia" / "cuda_runtime" / "lib",
        ):
            try:
                for path in sorted(library_dir.glob("libcudart.so*")):
                    add(packaged_candidates, path)
            except OSError:
                continue

    # find_library returns a SONAME on some Linux images. Treat that only as a
    # version hint to select an already-established mapped/package path; never
    # let it nominate an ambient loader search result.
    find_library_name: str | None = None
    find_library_path: str | None = None
    try:
        found = find_library("cudart")
    except (OSError, RuntimeError):
        found = None
    if found:
        find_path = Path(found)
        if find_path.is_absolute():
            find_library_path = _versioned_cudart_candidate(find_path)
            if find_library_path is None:
                raise OSError("absolute find_library result is not an existing versioned libcudart file")
        elif _VERSIONED_CUDART_NAME.fullmatch(found.strip()):
            find_library_name = found.strip()

    installed_candidates = list(dict.fromkeys(mapped_candidates + packaged_candidates))
    if find_library_name:
        matching = [
            path for path in installed_candidates
            if Path(path).name == find_library_name or Path(path).name.startswith(find_library_name + ".")
        ]
        if installed_candidates and not matching:
            raise OSError(
                f"find_library runtime {find_library_name} does not match the mapped/packaged CUDA runtime"
            )
        if not mapped_candidates and len(matching) > 1:
            raise OSError(
                f"find_library runtime {find_library_name} matches multiple installed CUDA runtimes"
            )
        if matching:
            installed_candidates = matching + [path for path in installed_candidates if path not in matching]
    if find_library_path:
        if not installed_candidates:
            raise OSError("absolute find_library runtime is outside the mapped/packaged CUDA runtime paths")
        if find_library_path not in installed_candidates:
            raise OSError("absolute find_library runtime does not match the mapped/packaged CUDA runtime")
        if not mapped_candidates:
            # An explicit absolute result disambiguates package copies only
            # after it is verified to be one of those exact installed files.
            installed_candidates = [find_library_path]
    candidates = installed_candidates
    if not mapped_candidates and not find_library_name and not find_library_path and len(packaged_candidates) > 1:
        raise OSError(
            "multiple installed versioned libcudart files found without a mapped runtime or SONAME hint"
        )

    load_errors: list[str] = []
    for candidate in candidates:
        try:
            # Record the exact regular-file identity before binding symbols.
            candidate_sha256 = sha256_file(candidate)
            library = cdll_factory(candidate)
            copy_function = library.cudaMemcpy
            copy_function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            copy_function.restype = ctypes.c_int
            return library, candidate, candidate_sha256
        except (AttributeError, OSError) as exc:
            load_errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    detail = "; ".join(load_errors) if load_errors else "no versioned candidates were found"
    raise OSError(f"unable to load an installed versioned libcudart runtime for cache copy: {detail}")


def _copy_cached_kv_from_native(torch: Any, native: Any, layer_idx: int, cuda_memcpy: Any) -> tuple[Any, Any]:
    layer_types = native.language_model.config.layer_types
    if not isinstance(layer_idx, int) or not 0 <= layer_idx < len(layer_types):
        raise ValueError(f"native cache layer index out of range: {layer_idx!r}")
    seq_len = int(native.current_position)
    if seq_len < 0:
        raise RuntimeError(f"native cache position is negative: {seq_len}")
    head_dim = 512 if layer_types[layer_idx] == "full_attention" else 256
    layer = ctypes.c_int(layer_idx)
    key_ptr = native._lib.gemma4_listener_get_cached_k(native._ctx, layer)
    value_ptr = native._lib.gemma4_listener_get_cached_v(native._ctx, layer)
    if seq_len > 0 and (not key_ptr or not value_ptr):
        raise RuntimeError(f"native cache K/V unavailable for layer {layer_idx}")
    device = native.output_tensor.device
    key = torch.empty((1, 1, seq_len, head_dim), dtype=torch.bfloat16, device=device)
    value = torch.empty((1, 1, seq_len, head_dim), dtype=torch.bfloat16, device=device)
    byte_count = seq_len * head_dim * 2
    key_status = int(cuda_memcpy(
        ctypes.c_void_p(key.data_ptr()), ctypes.c_void_p(key_ptr), ctypes.c_size_t(byte_count), ctypes.c_int(3),
    ))
    if key_status != 0:
        raise RuntimeError(f"cudaMemcpy K failed with code {key_status}")
    value_status = int(cuda_memcpy(
        ctypes.c_void_p(value.data_ptr()), ctypes.c_void_p(value_ptr), ctypes.c_size_t(byte_count), ctypes.c_int(3),
    ))
    if value_status != 0:
        raise RuntimeError(f"cudaMemcpy V failed with code {value_status}")
    return key, value


def _actual_cache_from_native(
    torch: Any,
    DynamicCache: Any,
    lm: Any,
    native: Any,
    *,
    cuda_memcpy: Any,
) -> tuple[Any, dict[str, np.ndarray], str]:
    values: dict[int, tuple[Any, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    hash_parts: dict[str, str] = {}
    for layer_idx, layer in enumerate(lm.layers[:35]):
        attention = layer.self_attn
        if bool(getattr(attention, "is_kv_shared_layer", False)):
            continue
        key, value = _copy_cached_kv_from_native(torch, native, layer_idx, cuda_memcpy)
        values[layer_idx] = (key, value)
        arrays[f"actual_cache.layer{layer_idx:02d}.k"] = _tensor_array(key)
        arrays[f"actual_cache.layer{layer_idx:02d}.v"] = _tensor_array(value)
        hash_parts[f"{layer_idx}.k"] = _torch_tensor_sha256(key, torch)
        hash_parts[f"{layer_idx}.v"] = _torch_tensor_sha256(value, torch)
    digest = _sha_bytes(json.dumps(hash_parts, sort_keys=True).encode("utf-8"))
    return _make_dynamic_cache(torch, DynamicCache, lm, values), arrays, digest


def _synthetic_cache(
    torch: Any,
    DynamicCache: Any,
    lm: Any,
    device: Any,
    seq_len: int,
    seed: int,
) -> tuple[Any, str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values: dict[int, tuple[Any, Any]] = {}
    hash_parts: dict[str, str] = {}
    for layer_idx, layer in enumerate(lm.layers[:35]):
        attention = layer.self_attn
        if bool(getattr(attention, "is_kv_shared_layer", False)):
            continue
        head_dim = int(attention.head_dim)
        key_cpu = (torch.randn((1, 1, seq_len, head_dim), generator=generator) * 0.02).to(torch.bfloat16).contiguous()
        value_cpu = (torch.randn((1, 1, seq_len, head_dim), generator=generator) * 0.02).to(torch.bfloat16).contiguous()
        hash_parts[f"{layer_idx}.k"] = _torch_tensor_sha256(key_cpu, torch)
        hash_parts[f"{layer_idx}.v"] = _torch_tensor_sha256(value_cpu, torch)
        values[layer_idx] = (
            key_cpu.to(device=device).contiguous(),
            value_cpu.to(device=device).contiguous(),
        )
    digest = _sha_bytes(json.dumps(hash_parts, sort_keys=True).encode("utf-8"))
    return _make_dynamic_cache(torch, DynamicCache, lm, values), digest


def _sum_ordered_calls(runtime_info: Mapping[str, Any]) -> int:
    values = runtime_info.get("ordered_calls_by_layer")
    if not isinstance(values, Mapping):
        raise ProbeContractError("native runtime omitted ordered_calls_by_layer")
    return sum(int(value) for value in values.values())


def _run_full_q4_route(
    torch: Any,
    device: Any,
    native: Any,
    inputs: tuple[Any, Any, Any, Any, str],
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    prefix_embeds, prefix_ple, prefix_attention, frames, _ = inputs
    if native.current_position != 0:
        raise ProbeContractError("route must start from a reset native context")
    native.prime_async(prefix_embeds, prefix_ple, prefix_attention)
    _sync(torch, device)
    if native.current_position != PREFIX_LEN:
        raise ProbeContractError(f"prime position mismatch: {native.current_position}")
    prime_info = native.native_runtime_info()
    short_calls_after_prime = _sum_ordered_calls(prime_info)
    if short_calls_after_prime != 0:
        raise ProbeContractError("short canonical prefill unexpectedly entered ordered-long attention")
    outputs: list[np.ndarray] = []
    positions: list[int] = []
    for index in range(120):
        output = native.step_async(frames[index])
        _sync(torch, device)
        expected_position = PREFIX_LEN + (index + 1) * Q_LEN
        if native.current_position != expected_position:
            raise ProbeContractError(
                f"Q4 cursor mismatch at frame {index}: {native.current_position} != {expected_position}"
            )
        native.verify_output_health()
        outputs.append(_tensor_array(output))
        positions.append(expected_position)
    if positions[118] != CROSSING_START or positions[119] != CROSSING_END:
        raise ProbeContractError("Q4 sequence failed to cross the 512-token sliding boundary at 510->514")
    array_key = f"{prefix}.q4_hidden_outputs"
    value = np.stack(outputs, axis=0)
    arrays[array_key] = value
    metadata[array_key] = {"shape": list(value.shape), "dtype": "float32_from_bf16", "sha256": _array_sha256(value)}
    runtime_after = native.native_runtime_info()
    ordered_after_short = _sum_ordered_calls(runtime_after)
    if ordered_after_short != 0:
        raise ProbeContractError("512-crossing route unexpectedly entered >2048-key ordered path")
    return {
        "status": "PASS",
        "position": int(native.current_position),
        "positions": {"after_prime": PREFIX_LEN, "before_crossing": CROSSING_START, "after_crossing": CROSSING_END},
        "compute_q_len": PREFILL_COMPUTE_LEN,
        "valid_q_len": PREFIX_LEN,
        "ordered_calls_after_prime": short_calls_after_prime,
        "ordered_calls_after_crossing512": ordered_after_short,
        "runtime_after_prime": prime_info,
        "runtime_after": runtime_after,
    }


def _candidate_two_context_memory_probe(
    torch: Any,
    device: Any,
    lm: Any,
    NativeGemmaListenerLM: Any,
    runtime: Mapping[str, Any],
    inputs: tuple[Any, Any, Any, Any, str],
) -> tuple[Any, Any, dict[str, Any]]:
    prefix_embeds, prefix_ple, prefix_attention, _frames, _digest = inputs
    before = _memory_snapshot(torch, device)
    first = NativeGemmaListenerLM(
        lm,
        max_seq_len=int(runtime["max_seq_len"]),
        stream_q_len=int(runtime["stream_q_len"]),
        attention_backend=int(runtime["attention_backend_id"]),
        rmsnorm_backend=int(runtime["rmsnorm_backend_id"]),
    )
    torch.cuda.synchronize(device)
    after_first_create = _memory_snapshot(torch, device)
    first.prime_async(prefix_embeds, prefix_ple, prefix_attention)
    torch.cuda.synchronize(device)
    if first.current_position != PREFIX_LEN:
        raise ProbeContractError("first candidate context failed canonical prefill")
    after_first_prime = _memory_snapshot(torch, device)
    second = NativeGemmaListenerLM(
        lm,
        max_seq_len=int(runtime["max_seq_len"]),
        stream_q_len=int(runtime["stream_q_len"]),
        attention_backend=int(runtime["attention_backend_id"]),
        rmsnorm_backend=int(runtime["rmsnorm_backend_id"]),
    )
    torch.cuda.synchronize(device)
    after_second_create = _memory_snapshot(torch, device)
    second.prime_async(prefix_embeds, prefix_ple, prefix_attention)
    torch.cuda.synchronize(device)
    if second.current_position != PREFIX_LEN:
        raise ProbeContractError("second candidate context failed canonical prefill")
    second.verify_output_health()
    after_second_prime = _memory_snapshot(torch, device)
    result = {
        "status": "PASS",
        "context_count": 2,
        "primed_positions": [first.current_position, second.current_position],
        "memory": {
            "before_contexts": before,
            "after_context_1_create": after_first_create,
            "after_context_1_prime": after_first_prime,
            "after_context_2_create": after_second_create,
            "after_context_2_prime": after_second_prime,
        },
        "free_bytes_before": before["cuda_free_bytes"],
        "free_bytes_after_context_1": after_first_prime["cuda_free_bytes"],
        "free_bytes_after_context_2": after_second_prime["cuda_free_bytes"],
        "context_1_runtime": first.native_runtime_info(),
        "context_2_runtime": second.native_runtime_info(),
    }
    return first, second, result


def _run_debug_layers(
    torch: Any,
    device: Any,
    lm: Any,
    native: Any,
    DynamicCache: Any,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cache, cache_sha256 = _synthetic_cache(torch, DynamicCache, lm, device, PREFIX_LEN, PROBE_SEED + 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PROBE_SEED + 2)
    hidden = (torch.randn((1, Q_LEN, 1536), generator=generator) * 0.05).to(torch.bfloat16).to(device).contiguous()
    projected_ple = (torch.randn((1, Q_LEN, 35, 256), generator=generator) * 0.05).to(torch.bfloat16).to(device).contiguous()
    hidden_sha256 = _torch_tensor_sha256(hidden, torch)
    ple_sha256 = _torch_tensor_sha256(projected_ple, torch)
    manifest_path = Path(native.native_runtime_info()["manifest_path"])
    topology = json.loads(manifest_path.read_text(encoding="utf-8"))
    layers_by_index = {int(layer["index"]): layer for layer in topology["layers"]}
    source13 = layers_by_index.get(EXPECTED_DEBUG_LAYER_ROLES["source13_sliding"])
    source14 = layers_by_index.get(EXPECTED_DEBUG_LAYER_ROLES["source14_full"])
    if (
        not isinstance(source13, Mapping)
        or source13.get("layer_type") != "sliding_attention"
        or source13.get("is_kv_shared_layer")
        or not isinstance(source14, Mapping)
        or source14.get("layer_type") != "full_attention"
        or source14.get("is_kv_shared_layer")
    ):
        raise ProbeContractError("loaded topology source13/source14 direct layer roles changed")
    source13_consumers = [
        int(layer["index"])
        for layer in topology["layers"]
        if layer.get("is_kv_shared_layer")
        and layer.get("kv_shared_layer_index") == 13
        and layer.get("layer_type") == "sliding_attention"
    ]
    source14_consumers = [
        int(layer["index"])
        for layer in topology["layers"]
        if layer.get("is_kv_shared_layer")
        and layer.get("kv_shared_layer_index") == 14
        and layer.get("layer_type") == "full_attention"
    ]
    full_layers = [
        int(layer["index"])
        for layer in topology["layers"]
        if layer.get("layer_type") == "full_attention" and not layer.get("is_kv_shared_layer")
    ]
    if not source13_consumers or not source14_consumers or not full_layers:
        raise ProbeContractError("topology lacks required source13/source14/full layer representatives")
    cases = [13, source13_consumers[0], 14, source14_consumers[0]]
    if cases != list(EXPECTED_DEBUG_LAYERS):
        raise ProbeContractError(
            f"loaded topology debug representatives differ from pinned contract: {cases}"
        )
    layer_rows: list[dict[str, Any]] = []
    for layer_idx in cases:
        output = native.debug_execute_layer(
            layer_idx,
            hidden,
            projected_ple,
            cache,
            current_position=PREFIX_LEN,
            query_position=PREFIX_LEN,
        )
        torch.cuda.synchronize(device)
        native.verify_output_health()
        key = f"debug.layer{layer_idx:02d}.output"
        _record_tensor(arrays, metadata, key, output)
        layer_rows.append({"layer_idx": layer_idx, "output_sha256": metadata[key]["sha256"], "status": "PASS"})
    return {
        "status": "PASS",
        "layers": cases,
        "layer_roles": dict(EXPECTED_DEBUG_LAYER_ROLES),
        "source13_consumers": source13_consumers,
        "source14_consumers": source14_consumers,
        "full_attention_layers": full_layers,
        "synthetic_cache_position": PREFIX_LEN,
        "synthetic_cache_sha256": cache_sha256,
        "hidden_input_sha256": hidden_sha256,
        "projected_ple_sha256": ple_sha256,
        "outputs": layer_rows,
    }


def _run_actual_import(
    torch: Any,
    device: Any,
    native: Any,
    cache: Any,
    cache_sha256: str,
    frames: Any,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    native.reset()
    if native.current_position != 0:
        raise ProbeContractError("reset before actual cache import did not return position zero")
    native.import_cache_async(cache, current_position=CROSSING_END)
    torch.cuda.synchronize(device)
    if native.current_position != CROSSING_END:
        raise ProbeContractError("actual cache import did not set position 514")
    output = native.step_async(frames[120])
    torch.cuda.synchronize(device)
    native.verify_output_health()
    _record_tensor(arrays, metadata, "actual_import.q4_output", output)
    if native.current_position != 518:
        raise ProbeContractError(f"actual cache import Q4 ended at {native.current_position}, expected 518")
    return {
        "status": "PASS",
        "position_before": CROSSING_END,
        "position_after": native.current_position,
        "imported_cache_sha256": cache_sha256,
        "output_sha256": metadata["actual_import.q4_output"]["sha256"],
    }


def _run_rebase(
    torch: Any,
    device: Any,
    native: Any,
    frames: Any,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    native.compact_audio_history(expected_cached_frames=120, retain_frames=32)
    torch.cuda.synchronize(device)
    if native.current_position != 162:
        raise ProbeContractError(f"rebase position {native.current_position}, expected 162")
    output = native.step_async(frames[120])
    torch.cuda.synchronize(device)
    native.verify_output_health()
    _record_tensor(arrays, metadata, "rebase.q4_output", output)
    return {
        "status": "PASS",
        "position_before": CROSSING_END,
        "position_after": native.current_position,
        "retained_audio_frames": 32,
        "output_sha256": metadata["rebase.q4_output"]["sha256"],
    }


def _rng_snapshot(torch: Any) -> dict[str, Any]:
    python_sha256 = _sha_bytes(repr(random.getstate()).encode("utf-8"))
    numpy_state = np.random.get_state()
    numpy_digest = hashlib.sha256()
    numpy_digest.update(str(numpy_state[0]).encode("utf-8"))
    numpy_digest.update(np.ascontiguousarray(numpy_state[1]).view(np.uint8).tobytes())
    numpy_digest.update(repr(numpy_state[2:]).encode("utf-8"))
    torch_cpu_sha256 = _torch_tensor_sha256(torch.get_rng_state(), torch)
    cuda_states = torch.cuda.get_rng_state_all()
    cuda_sha256 = [_torch_tensor_sha256(state, torch) for state in cuda_states]
    payload = {
        "python_sha256": python_sha256,
        "numpy_sha256": numpy_digest.hexdigest(),
        "torch_cpu_sha256": torch_cpu_sha256,
        "torch_cuda_sha256": cuda_sha256,
    }
    payload["sha256"] = _sha_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return payload


def _fixed_input_tensor_hashes(torch: Any, inputs: tuple[Any, Any, Any, Any, str]) -> dict[str, str]:
    prefix_embeds, prefix_ple, prefix_attention, frames, _ = inputs
    return {
        "prefix_embeds": _torch_tensor_sha256(prefix_embeds, torch),
        "prefix_ple": _torch_tensor_sha256(prefix_ple, torch),
        "prefix_attention": _torch_tensor_sha256(prefix_attention, torch),
        "frames_q4": _torch_tensor_sha256(frames, torch),
    }


def _record_reset_diagnostic_cache(
    torch: Any,
    native: Any,
    *,
    stage: str,
    case_name: str,
    cuda_memcpy: Any,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    position = int(native.current_position)
    captured: dict[str, str] = {}
    for layer_idx in (13, 14):
        key, value = _copy_cached_kv_from_native(torch, native, layer_idx, cuda_memcpy)
        for kind, tensor in (("k", key), ("v", value)):
            name = f"baseline_reset.{case_name}.{stage}.layer{layer_idx:02d}.{kind}"
            _record_tensor(arrays, metadata, name, tensor)
            captured[f"layer{layer_idx:02d}.{kind}"] = metadata[name]["sha256"]
    return {"position": position, "sha256": captured}


def _record_reset_diagnostic_output(
    outputs: list[np.ndarray],
    *,
    case_name: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
) -> None:
    if not outputs:
        return
    value = np.stack(outputs, axis=0)
    name = f"baseline_reset.{case_name}.q4_hidden_outputs"
    arrays[name] = value
    metadata[name] = {
        "shape": list(value.shape),
        "dtype": "float32_from_bf16",
        "sha256": _array_sha256(value),
    }


def _run_reset_diagnostic_q4_route(
    torch: Any,
    device: Any,
    native: Any,
    inputs: tuple[Any, Any, Any, Any, str],
    *,
    case_name: str,
    cuda_memcpy: Any,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
    record: dict[str, Any],
) -> dict[str, Any]:
    prefix_embeds, prefix_ple, prefix_attention, frames, _ = inputs
    if int(native.current_position) != 0:
        raise ProbeContractError(
            f"{case_name} must start from reset position zero, got {native.current_position}"
        )
    native.prime_async(prefix_embeds, prefix_ple, prefix_attention)
    _sync(torch, device)
    if int(native.current_position) != PREFIX_LEN:
        raise ProbeContractError(
            f"{case_name} prime ended at {native.current_position}, expected {PREFIX_LEN}"
        )
    prime_snapshot = _record_reset_diagnostic_cache(
        torch,
        native,
        stage="after_prime",
        case_name=case_name,
        cuda_memcpy=cuda_memcpy,
        arrays=arrays,
        metadata=metadata,
    )
    outputs: list[np.ndarray] = []
    positions: list[int] = []
    step_health: list[dict[str, Any]] = []
    try:
        for index in range(120):
            output = native.step_async(frames[index])
            _sync(torch, device)
            expected_position = PREFIX_LEN + (index + 1) * Q_LEN
            actual_position = int(native.current_position)
            if actual_position != expected_position:
                raise ProbeContractError(
                    f"{case_name} Q4 position mismatch at frame {index}: "
                    f"{actual_position} != {expected_position}"
                )
            native.verify_output_health()
            outputs.append(_tensor_array(output))
            positions.append(actual_position)
            step_health.append({"frame_index": index, "position": actual_position, "health": "PASS"})
    except Exception:
        _record_reset_diagnostic_output(
            outputs, case_name=case_name, arrays=arrays, metadata=metadata
        )
        record["q4_positions"] = positions
        record["q4_step_health"] = step_health
        raise
    _record_reset_diagnostic_output(
        outputs, case_name=case_name, arrays=arrays, metadata=metadata
    )
    post_snapshot = _record_reset_diagnostic_cache(
        torch,
        native,
        stage="after_q4",
        case_name=case_name,
        cuda_memcpy=cuda_memcpy,
        arrays=arrays,
        metadata=metadata,
    )
    if positions[118] != CROSSING_START or positions[119] != CROSSING_END:
        raise ProbeContractError(f"{case_name} did not cross the 512-token boundary at 510->514")
    ordered_calls = _sum_ordered_calls(native.native_runtime_info())
    if ordered_calls != 0:
        raise ProbeContractError(f"{case_name} unexpectedly entered the ordered-long path")
    record.update({
        "status": "PASS",
        "position_after_prime": prime_snapshot["position"],
        "position_after_q4": post_snapshot["position"],
        "frames": len(outputs),
        "crossing_positions": positions[118:120],
        "ordered_calls": ordered_calls,
        "after_prime_cache": prime_snapshot["sha256"],
        "after_q4_cache": post_snapshot["sha256"],
        "q4_output_sha256": metadata[f"baseline_reset.{case_name}.q4_hidden_outputs"]["sha256"],
        "q4_positions": positions,
        "q4_step_health": step_health,
    })
    return {
        "position": post_snapshot["position"],
        "output_count": len(outputs),
    }


def _save_reset_diagnostic_artifacts(
    report: dict[str, Any],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    report_path: str | Path,
    arrays_path: str | Path,
) -> None:
    report["arrays"] = {name: dict(value) for name, value in metadata.items()}
    if arrays:
        target = Path(arrays_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        report["arrays_path"] = str(target)
        report["arrays_file_sha256"] = sha256_file(target)
    atomic_write_json(report_path, report)


def _diagnostic_case_inputs(
    torch: Any,
    inputs: tuple[Any, Any, Any, Any, str],
    *,
    expected_tensor_hashes: Mapping[str, str],
) -> dict[str, Any]:
    before = _fixed_input_tensor_hashes(torch, inputs)
    rng_before = _rng_snapshot(torch)
    return {
        "input_sha256": inputs[-1],
        "input_integrity": {
            "before_tensor_sha256": before,
            "expected_tensor_sha256": dict(expected_tensor_hashes),
            "before_matches_expected": before == dict(expected_tensor_hashes),
        },
        "rng": {"before": rng_before, "before_sha256": rng_before["sha256"]},
    }


def _finish_diagnostic_case_inputs(
    torch: Any,
    inputs: tuple[Any, Any, Any, Any, str],
    record: dict[str, Any],
    *,
    expected_tensor_hashes: Mapping[str, str],
) -> None:
    after = _fixed_input_tensor_hashes(torch, inputs)
    rng_after = _rng_snapshot(torch)
    integrity = record["input_integrity"]
    integrity["after_tensor_sha256"] = after
    integrity["after_matches_expected"] = after == dict(expected_tensor_hashes)
    integrity["unchanged"] = (
        integrity["before_tensor_sha256"] == after
        and integrity["before_matches_expected"]
        and integrity["after_matches_expected"]
    )
    rng = record["rng"]
    rng["after"] = rng_after
    rng["after_sha256"] = rng_after["sha256"]
    rng["unchanged"] = rng["before_sha256"] == rng["after_sha256"]


def _run_ordered_long(
    torch: Any,
    device: Any,
    lm: Any,
    native: Any,
    DynamicCache: Any,
    frames: Any,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    native.reset()
    cache, cache_sha256 = _synthetic_cache(
        torch, DynamicCache, lm, device, LONG_ORDERED_POSITION, PROBE_SEED + 3
    )
    native.import_cache_async(cache, current_position=LONG_ORDERED_POSITION)
    torch.cuda.synchronize(device)
    before_info = native.native_runtime_info()
    ordered_before = _sum_ordered_calls(before_info)
    output = native.step_async(frames[120])
    torch.cuda.synchronize(device)
    native.verify_output_health()
    after_info = native.native_runtime_info()
    ordered_after = _sum_ordered_calls(after_info)
    _record_tensor(arrays, metadata, "ordered_long.q4_output", output)
    if native.current_position != LONG_ORDERED_END:
        raise ProbeContractError(f"ordered-long Q4 ended at {native.current_position}, expected {LONG_ORDERED_END}")
    if ordered_before != 0 or ordered_after <= ordered_before:
        raise ProbeContractError(
            f"ordered-long dispatch counter did not increase: {ordered_before} -> {ordered_after}"
        )
    return {
        "status": "PASS",
        "position_before": LONG_ORDERED_POSITION,
        "position_after": native.current_position,
        "synthetic_import_cache_sha256": cache_sha256,
        "ordered_calls_before": ordered_before,
        "ordered_calls_after": ordered_after,
        "ordered_call_delta_by_layer": {
            layer: int(after_info["ordered_calls_by_layer"][layer])
            - int(before_info["ordered_calls_by_layer"][layer])
            for layer in before_info["ordered_calls_by_layer"]
        },
        "runtime_after": after_info,
        "output_sha256": metadata["ordered_long.q4_output"]["sha256"],
    }


def run_worker(
    *,
    manifest_path: str | Path,
    role: str,
    report_path: str | Path,
    arrays_path: str | Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": WORKER_SCHEMA,
        "role": role,
        "status": "RUNNING",
        "cases": {},
        "arrays": {},
    }
    atomic_write_json(report_path, report)
    try:
        manifest = load_manifest(manifest_path)
        if role not in ("baseline", "candidate"):
            raise ProbeContractError(f"unknown worker role {role!r}")
        library_record = manifest["libraries"][role]
        source_attestation = _verify_source_manifest(library_record)
        library_path = Path(library_record["path"])
        if not library_path.is_file():
            raise ProbeContractError(f"native library missing: {library_path}")
        library_sha256 = sha256_file(library_path)
        if library_sha256.lower() != str(library_record["sha256"]).lower():
            raise ProbeContractError(f"{role} library hash mismatch")
    except Exception as exc:
        report.update({
            "status": "FAIL",
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        })
        atomic_write_json(report_path, report)
        raise

    report.update({
        "schema": WORKER_SCHEMA,
        "role": role,
        "status": "RUNNING",
        "library_path": str(library_path),
        "library_sha256": library_sha256,
        "source_manifest": source_attestation,
        "source_manifest_sha256": source_attestation["sha256"],
        "source_snapshot_tree": library_record.get("source_root"),
        "manifest_sha256": sha256_file(manifest_path),
        "runtime_contract": manifest["runtime"],
        "cases": {},
        "arrays": {},
    })
    arrays: dict[str, np.ndarray] = {}
    array_metadata: dict[str, dict[str, Any]] = {}
    atomic_write_json(report_path, report)
    torch: Any | None = None
    cuda_runtime: Any | None = None
    context_one: Any | None = None
    context_two: Any | None = None
    previous_library_env = os.environ.get(NATIVE_LIBRARY_ENV)
    try:
        # Pin only this worker process to the DSO already hash-verified above.
        # run_all also passes this explicitly to each child; restoring it in
        # finally keeps direct in-process callers from changing ambient state.
        os.environ[NATIVE_LIBRARY_ENV] = str(library_path)
        import torch
        from transformers.cache_utils import DynamicCache

        from duplex_kernel.models.thinker import load_thinker
        from duplex_kernel.native_listener import NativeGemmaListenerLM

        cuda_runtime, cuda_runtime_path, cuda_runtime_sha256 = _load_cuda_runtime_for_cache_copy(torch)
        report["cuda_runtime_for_cache_copy"] = {
            "path": cuda_runtime_path,
            "sha256": cuda_runtime_sha256,
            "torch_cuda_initialized_at_load": bool(torch.cuda.is_initialized()),
        }
        atomic_write_json(report_path, report)
        if not torch.cuda.is_available():
            raise ProbeContractError("GPU worker requires CUDA; refusing CPU fallback")
        device = torch.device("cuda:0")
        torch.cuda.init()
        pre_model_memory = _memory_snapshot(torch, device)
        thinker = load_thinker("/workspace", device=device, dtype=torch.bfloat16)
        lm = thinker.model.model.language_model
        torch.cuda.synchronize(device)
        model_memory = _memory_snapshot(torch, device)
        inputs = _fixed_inputs(torch, lm, device)
        prefix_embeds, prefix_ple, prefix_attention, frames, input_sha256 = inputs
        report.update({
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_device_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "memory_before_model": pre_model_memory,
            "memory_after_model_and_fixed_inputs": model_memory,
            "input_sha256": input_sha256,
            "fixed_input_sha256": {
                "prefix_embeds": _torch_tensor_sha256(prefix_embeds, torch),
                "prefix_ple": _torch_tensor_sha256(prefix_ple, torch),
                "prefix_attention": _torch_tensor_sha256(prefix_attention, torch),
                "frames_q4": _torch_tensor_sha256(frames, torch),
            },
            "model_checkpoint_files_sha256": None,
            "native_topology_manifest_sha256": None,
        })
        runtime = manifest["runtime"]
        # Explicit policy arguments avoid ambient environment overrides while
        # preserving production defaults and serving state.
        if role == "candidate":
            context_one, context_two, pair_case = _candidate_two_context_memory_probe(
                torch, device, lm, NativeGemmaListenerLM, runtime, inputs
            )
            report["cases"]["two_candidate_contexts"] = pair_case
            report["memory_after_two_candidate_contexts"] = pair_case["memory"]["after_context_2_prime"]
            # Reset the second successfully primed context for the independent
            # debug/import/ordered subcases; it remains the exact same DSO.
            context_two.reset()
        else:
            memory_before_context = _memory_snapshot(torch, device)
            context_one = NativeGemmaListenerLM(
                lm,
                max_seq_len=int(runtime["max_seq_len"]),
                stream_q_len=int(runtime["stream_q_len"]),
                attention_backend=int(runtime["attention_backend_id"]),
                rmsnorm_backend=int(runtime["rmsnorm_backend_id"]),
            )
            torch.cuda.synchronize(device)
            report["memory_single_context"] = {
                "before_create": memory_before_context,
                "after_create": _memory_snapshot(torch, device),
            }
        runtime_info = context_one.native_runtime_info()
        if (
            int(runtime_info["attention_backend_id"]) != 3
            or int(runtime_info["rmsnorm_backend_id"]) != 2
            or int(runtime_info["max_seq_len"]) != MAX_SEQ_LEN
            or runtime_info.get("library_sha256") != library_sha256
        ):
            raise ProbeContractError(f"native runtime policy mismatch: {runtime_info}")
        abi_version = int(context_one._lib.gemma4_listener_abi_version())
        if abi_version != 3:
            raise ProbeContractError(f"native ABI mismatch: {abi_version}")
        report.update({
            "abi_version": abi_version,
            "attention_backend_id": int(runtime_info["attention_backend_id"]),
            "rmsnorm_backend_id": int(runtime_info["rmsnorm_backend_id"]),
            "max_seq_len": int(runtime_info["max_seq_len"]),
            "stream_q_len": int(runtime["stream_q_len"]),
            "native_runtime_initial": runtime_info,
            "model_checkpoint_files_sha256": runtime_info["checkpoint_files_sha256"],
            "native_topology_manifest_sha256": runtime_info["manifest_sha256"],
        })
        report["fixed_input_sha256"]["synthetic_debug_cache34"] = None
        report["fixed_input_sha256"]["synthetic_ordered_cache2049"] = None
        atomic_write_json(report_path, report)

        # Run/debug context two first.  It was primed with context one above in
        # the candidate case; baseline has only the single sequential context.
        if context_two is not None:
            debug_context = context_two
        else:
            debug_context = context_one

        debug_case = _run_debug_layers(
            torch, device, lm, debug_context, DynamicCache, arrays, array_metadata
        )
        report["cases"]["debug_source13_and_others"] = debug_case
        report["fixed_input_sha256"]["synthetic_debug_cache34"] = debug_case["synthetic_cache_sha256"]
        _sync(torch, device)
        atomic_write_json(report_path, report)

        # Main fixed Q4 route: true production prime path (compute 291, valid
        # 34), then 120 native Q4 steps crossing logical position 512.
        context_one.reset()
        main_case = _run_full_q4_route(
            torch, device, context_one, inputs,
            arrays=arrays, metadata=array_metadata, prefix="main",
        )
        main_case["input_sha256"] = inputs[-1]
        report["cases"]["canonical_prefill291_valid34"] = {
            "status": "PASS",
            "position": PREFIX_LEN,
            "compute_q_len": PREFILL_COMPUTE_LEN,
            "valid_q_len": PREFIX_LEN,
            "ordered_calls_after_prime": main_case["ordered_calls_after_prime"],
            "runtime_after_short_prime": main_case["runtime_after_prime"],
        }
        report["cases"]["q4_cross512"] = {
            "status": "PASS",
            "position_before": CROSSING_START,
            "position_after": CROSSING_END,
            "frames": 120,
            "ordered_calls_after_crossing": main_case["ordered_calls_after_crossing512"],
        }
        report["cases"]["short_full_fallback"] = {
            "status": "PASS",
            "reference_case": "prime34 and Q4 through 514 stay below ordered threshold",
            "ordered_calls": main_case["ordered_calls_after_crossing512"],
        }
        actual_cache, actual_cache_arrays, actual_cache_sha256 = _actual_cache_from_native(
            torch, DynamicCache, lm, context_one, cuda_memcpy=cuda_runtime.cudaMemcpy
        )
        arrays.update(actual_cache_arrays)
        for name, value in actual_cache_arrays.items():
            array_metadata[name] = {"shape": list(value.shape), "dtype": "float32_from_bf16", "sha256": _array_sha256(value)}
        report["cases"]["main_q4_route"] = main_case
        report["cases"]["main_cache_snapshot"] = {
            "status": "PASS",
            "position": context_one.current_position,
            "actual_import_cache_sha256": actual_cache_sha256,
            "nonshared_source_layers": len(actual_cache_arrays) // 2,
            "source13_key_sha256": array_metadata["actual_cache.layer13.k"]["sha256"],
            "source13_value_sha256": array_metadata["actual_cache.layer13.v"]["sha256"],
            "source13_key_shape": array_metadata["actual_cache.layer13.k"]["shape"],
            "source13_value_shape": array_metadata["actual_cache.layer13.v"]["shape"],
        }

        rebase_case = _run_rebase(
            torch, device, context_one, frames, arrays, array_metadata
        )
        report["cases"]["cache_rebase"] = rebase_case
        _sync(torch, device)
        atomic_write_json(report_path, report)

        # Exact reset/re-prime control. Baseline repeats the complete route and
        # compares every captured Q4 output and source KV; candidate records the
        # same reset boundary and first Q4 result without doubling long work.
        context_one.reset()
        position_after_reset = context_one.current_position
        context_one.prime_async(prefix_embeds, prefix_ple, prefix_attention)
        _sync(torch, device)
        if context_one.current_position != PREFIX_LEN:
            raise ProbeContractError("re-prime after reset did not return to position 34")
        repeat_output = context_one.step_async(frames[0])
        _sync(torch, device)
        context_one.verify_output_health()
        _record_tensor(arrays, array_metadata, "reset_reprime.first_q4_output", repeat_output)
        reset_comparison = compare_reset_reprime_output(
            arrays["main.q4_hidden_outputs"], arrays["reset_reprime.first_q4_output"]
        )
        report["cases"]["reset_reprime"] = {
            "status": "PASS" if reset_comparison["passed"] else "FAIL",
            "position_after_reset": position_after_reset,
            "position_after_prime": PREFIX_LEN,
            "position_after_q4": context_one.current_position,
            "input_sha256": inputs[-1],
            "output_sha256": array_metadata["reset_reprime.first_q4_output"]["sha256"],
            "first_q4_output_comparison": reset_comparison,
        }
        repeat_comparison: dict[str, Any] | None = None
        if role == "baseline":
            context_one.reset()
            repeat_arrays: dict[str, np.ndarray] = {}
            repeat_meta: dict[str, dict[str, Any]] = {}
            repeat_case = _run_full_q4_route(
                torch, device, context_one, inputs,
                arrays=repeat_arrays, metadata=repeat_meta, prefix="repeat",
            )
            _repeat_cache, repeat_cache_arrays, repeat_cache_sha256 = _actual_cache_from_native(
                torch, DynamicCache, lm, context_one, cuda_memcpy=cuda_runtime.cudaMemcpy
            )
            for name, value in repeat_cache_arrays.items():
                repeat_name = f"repeat.{name}"
                repeat_arrays[repeat_name] = value
                repeat_meta[repeat_name] = {
                    "shape": list(value.shape),
                    "dtype": "float32_from_bf16",
                    "sha256": _array_sha256(value),
                }
            # Preserve every repeat tensor before parity comparison can fail;
            # the outer exception handler writes this complete set to NPZ.
            _preserve_repeat_evidence(arrays, array_metadata, repeat_arrays, repeat_meta)
            original = {
                **{
                    key.removeprefix("main."): value
                    for key, value in arrays.items()
                    if key.startswith("main.")
                },
                **{
                    key: value
                    for key, value in arrays.items()
                    if key.startswith("actual_cache.")
                },
            }
            repeat_values = {
                key.removeprefix("repeat."): value
                for key, value in repeat_arrays.items()
                if key.startswith("repeat.")
            }
            repeat_comparison = compare_tensor_arrays(original, repeat_values)
            report["cases"]["baseline_repeat_control"] = {
                "status": "PASS" if repeat_comparison["passed"] else "FAIL",
                "scope": "full_q4_route_and_cache reset-repeat control",
                "position": repeat_case["position"],
                "input_sha256": inputs[-1],
                "actual_cache_sha256": repeat_cache_sha256,
                "comparison": repeat_comparison,
            }
        else:
            report["cases"]["baseline_repeat_control"] = {
                "status": "NOT_APPLICABLE",
                "reason": "baseline-only repeat control",
            }
        if not reset_comparison["passed"]:
            raise ProbeContractError("reset/re-prime first Q4 output exceeded pinned parity tolerance")
        if repeat_comparison is not None and not repeat_comparison["passed"]:
            raise ProbeContractError("baseline repeat control exceeded pinned parity tolerance")

        fixed_inputs = dict(report["fixed_input_sha256"])
        input_sha256 = _sha_bytes(json.dumps(fixed_inputs, sort_keys=True).encode("utf-8"))
        report["input_sha256"] = input_sha256
        atomic_write_json(report_path, report)

        # A second native context imports the exact source KV copied from the
        # main route and continues one Q4. For baseline it is allocated only
        # after the main context is released; candidate already has two live
        # prime-attested contexts and reuses the second one here.
        if role == "baseline":
            torch.cuda.synchronize(device)
            context_one.close()
            context_one = None
            context_two = NativeGemmaListenerLM(
                lm,
                max_seq_len=int(runtime["max_seq_len"]),
                stream_q_len=int(runtime["stream_q_len"]),
                attention_backend=int(runtime["attention_backend_id"]),
                rmsnorm_backend=int(runtime["rmsnorm_backend_id"]),
            )
            debug_context = context_two
        actual_import_case = _run_actual_import(
            torch,
            device,
            debug_context,
            actual_cache,
            actual_cache_sha256,
            frames,
            arrays,
            array_metadata,
        )
        report["cases"]["cache_import_actual514"] = actual_import_case

        ordered_case = _run_ordered_long(
            torch,
            device,
            lm,
            debug_context,
            DynamicCache,
            frames,
            arrays,
            array_metadata,
        )
        report["cases"]["ordered_long_full"] = ordered_case
        report["fixed_input_sha256"]["synthetic_ordered_cache2049"] = ordered_case["synthetic_import_cache_sha256"]

        report["memory_after_scenarios"] = _memory_snapshot(torch, device)
        report["status"] = "PASS"
        report["arrays"] = array_metadata
        report["arrays_path"] = str(arrays_path)
        np.savez_compressed(arrays_path, **arrays)
        report["arrays_file_sha256"] = sha256_file(arrays_path)
        atomic_write_json(report_path, report)
        validate_worker_report(
            report,
            expected_role=role,
            expected_library_sha256=library_record["sha256"],
            expected_source_manifest_sha256=library_record["source_manifest_sha256"],
        )
        return report
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        report["arrays"] = array_metadata
        if arrays:
            np.savez_compressed(arrays_path, **arrays)
            report["arrays_path"] = str(arrays_path)
            report["arrays_file_sha256"] = sha256_file(arrays_path)
        atomic_write_json(report_path, report)
        raise
    finally:
        for native in (context_two, context_one):
            if native is not None and torch is not None:
                try:
                    torch.cuda.synchronize()
                    native.close()
                except Exception:
                    pass
        if previous_library_env is None:
            os.environ.pop(NATIVE_LIBRARY_ENV, None)
        else:
            os.environ[NATIVE_LIBRARY_ENV] = previous_library_env


def run_baseline_reset_diagnostic(
    *,
    manifest_path: str | Path,
    report_path: str | Path,
    arrays_path: str | Path,
) -> dict[str, Any]:
    """Run only the baseline four-route history diagnostic; never load candidate DSO."""
    report: dict[str, Any] = {
        "schema": BASELINE_RESET_DIAGNOSTIC_SCHEMA,
        "status": "RUNNING",
        "purpose": "baseline-only route-history RCA; candidate DSO remains unmeasured",
        "case_order": list(BASELINE_RESET_DIAGNOSTIC_CASES),
        "runtime_contract": None,
        "route_cases": {},
        "tolerances": dict(TOLERANCES),
    }
    atomic_write_json(report_path, report)
    arrays: dict[str, np.ndarray] = {}
    array_metadata: dict[str, dict[str, Any]] = {}
    torch: Any | None = None
    context_a: Any | None = None
    context_b: Any | None = None
    actual_cache: Any | None = None
    inputs: tuple[Any, Any, Any, Any, str] | None = None
    expected_tensor_hashes: dict[str, str] = {}
    previous_library_env = os.environ.get(NATIVE_LIBRARY_ENV)
    manifest: dict[str, Any] | None = None
    record: Mapping[str, Any] | None = None
    library_path: Path | None = None
    try:
        manifest = load_manifest(manifest_path)
        record = manifest["libraries"]["baseline"]
        source_attestation = _verify_source_manifest(record)
        library_path = Path(str(record["path"]))
        if not library_path.is_file():
            raise ProbeContractError(f"baseline native library missing: {library_path}")
        library_sha256 = sha256_file(library_path)
        if library_sha256.lower() != str(record["sha256"]).lower():
            raise ProbeContractError("baseline native library hash mismatch")
        os.environ[NATIVE_LIBRARY_ENV] = str(library_path)
        import torch
        from transformers.cache_utils import DynamicCache

        from duplex_kernel.models.thinker import load_thinker
        from duplex_kernel.native_listener import NativeGemmaListenerLM

        cuda_runtime, cuda_runtime_path, cuda_runtime_sha256 = _load_cuda_runtime_for_cache_copy(torch)
        report.update({
            "manifest_path": str(Path(manifest_path)),
            "manifest_sha256": sha256_file(manifest_path),
            "baseline_library_path": str(library_path),
            "baseline_library_sha256": library_sha256,
            "baseline_source_manifest": source_attestation,
            "baseline_source_manifest_sha256": source_attestation["sha256"],
            "source_reconstruction_note": manifest.get("source_reconstruction"),
            "candidate_dso_loaded": False,
            "runtime_contract": manifest["runtime"],
            "cuda_runtime_for_cache_copy": {
                "path": cuda_runtime_path,
                "sha256": cuda_runtime_sha256,
                "torch_cuda_initialized_at_load": bool(torch.cuda.is_initialized()),
            },
        })
        atomic_write_json(report_path, report)
        if not torch.cuda.is_available():
            raise ProbeContractError("baseline diagnostic requires CUDA; refusing CPU fallback")
        device = torch.device("cuda:0")
        torch.cuda.init()
        pre_model_memory = _memory_snapshot(torch, device)
        thinker = load_thinker("/workspace", device=device, dtype=torch.bfloat16)
        lm = thinker.model.model.language_model
        torch.cuda.synchronize(device)
        inputs = _fixed_inputs(torch, lm, device)
        expected_tensor_hashes = _fixed_input_tensor_hashes(torch, inputs)
        report.update({
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_device_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "memory_before_model": pre_model_memory,
            "memory_after_model_and_fixed_inputs": _memory_snapshot(torch, device),
            "input_sha256": inputs[-1],
            "fixed_input_tensor_sha256": expected_tensor_hashes,
            "model_checkpoint_files_sha256": None,
            "native_topology_manifest_sha256": None,
            "cases": report["route_cases"],
        })
        atomic_write_json(report_path, report)
        runtime = manifest["runtime"]

        def create_context() -> Any:
            native = NativeGemmaListenerLM(
                lm,
                max_seq_len=int(runtime["max_seq_len"]),
                stream_q_len=int(runtime["stream_q_len"]),
                enable_cuda_graph=False,
                attention_backend=int(runtime["attention_backend_id"]),
                rmsnorm_backend=int(runtime["rmsnorm_backend_id"]),
            )
            torch.cuda.synchronize(device)
            info = native.native_runtime_info()
            if (
                native.enable_cuda_graph
                or int(info["attention_backend_id"]) != 3
                or int(info["rmsnorm_backend_id"]) != 2
                or int(info["max_seq_len"]) != MAX_SEQ_LEN
                or info.get("library_sha256") != library_sha256
            ):
                native.close()
                raise ProbeContractError(f"baseline diagnostic runtime policy mismatch: {info}")
            report.setdefault("context_runtime_attestations", []).append(info)
            report["abi_version"] = int(native._lib.gemma4_listener_abi_version())
            if report["abi_version"] != 3:
                native.close()
                raise ProbeContractError(f"baseline native ABI mismatch: {report['abi_version']}")
            report["attention_backend_id"] = int(info["attention_backend_id"])
            report["rmsnorm_backend_id"] = int(info["rmsnorm_backend_id"])
            report["max_seq_len"] = int(info["max_seq_len"])
            report["stream_q_len"] = int(runtime["stream_q_len"])
            report["model_checkpoint_files_sha256"] = info["checkpoint_files_sha256"]
            report["native_topology_manifest_sha256"] = info["manifest_sha256"]
            atomic_write_json(report_path, report)
            return native

        def initialize_case(case_name: str) -> dict[str, Any]:
            assert inputs is not None
            value = {
                "status": "RUNNING",
                "history": [],
                **_diagnostic_case_inputs(
                    torch, inputs, expected_tensor_hashes=expected_tensor_hashes
                ),
            }
            report["route_cases"][case_name] = value
            atomic_write_json(report_path, report)
            return value

        def finish_case_inputs(case_record: dict[str, Any]) -> None:
            assert inputs is not None
            try:
                _finish_diagnostic_case_inputs(
                    torch,
                    inputs,
                    case_record,
                    expected_tensor_hashes=expected_tensor_hashes,
                )
            except Exception as exc:
                case_record.setdefault("errors", []).append(
                    f"post-route input/RNG attestation failed: {type(exc).__name__}: {exc}"
                )
                case_record["status"] = "FAIL"

        def run_full_case(case_name: str, native: Any, *, reset_first: bool = False) -> dict[str, Any]:
            assert inputs is not None
            case_record = initialize_case(case_name)
            try:
                if reset_first:
                    native.reset()
                    torch.cuda.synchronize(device)
                    case_record["history"].append({"event": "reset", "position": int(native.current_position)})
                _run_reset_diagnostic_q4_route(
                    torch,
                    device,
                    native,
                    inputs,
                    case_name=case_name,
                    cuda_memcpy=cuda_runtime.cudaMemcpy,
                    arrays=arrays,
                    metadata=array_metadata,
                    record=case_record,
                )
            except Exception as exc:
                case_record["status"] = "FAIL"
                case_record["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            finally:
                finish_case_inputs(case_record)
                _save_reset_diagnostic_artifacts(
                    report, arrays, array_metadata,
                    report_path=report_path, arrays_path=arrays_path,
                )
            return case_record

        # A: prime and execute the full fixed Q4 sequence on a fresh context.
        context_a = create_context()
        case_a = run_full_case("fresh_a", context_a)
        if case_a.get("status") == "PASS" and inputs is not None:
            try:
                actual_cache, _discarded_cache_arrays, cache_sha256 = _actual_cache_from_native(
                    torch,
                    DynamicCache,
                    lm,
                    context_a,
                    cuda_memcpy=cuda_runtime.cudaMemcpy,
                )
                case_a["import_source_cache_sha256"] = cache_sha256
                case_a["import_source_position"] = int(context_a.current_position)
            except Exception as exc:
                case_a["import_source_cache_error"] = f"{type(exc).__name__}: {exc}"
        torch.cuda.synchronize(device)
        context_a.close()
        context_a = None
        report["context_a_destroyed_before_fresh_b"] = True
        report["memory_after_context_a_destroy"] = _memory_snapshot(torch, device)
        _save_reset_diagnostic_artifacts(
            report, arrays, array_metadata, report_path=report_path, arrays_path=arrays_path
        )

        # B: allocate a second, fresh context only after A has been destroyed.
        context_b = create_context()
        case_b = run_full_case("fresh_b", context_b)
        if case_b.get("status") == "PASS":
            # C: reset/re-prime the same B context before any import or rebase.
            case_c = run_full_case("same_context_reset", context_b, reset_first=True)
        else:
            case_c = {
                "status": "SKIPPED",
                "reason": "fresh_b failed; same-context reuse was not treated as safe",
                "input_sha256": inputs[-1] if inputs is not None else None,
            }
            report["route_cases"]["same_context_reset"] = case_c
            _save_reset_diagnostic_artifacts(
                report, arrays, array_metadata, report_path=report_path, arrays_path=arrays_path
            )

        # D: import A's actual position-514 cache into B, rebase, then consume
        # the exact same first fixed Q4 frame. Never infer a cache if A failed.
        can_run_import_rebase = (
            actual_cache is not None
            and case_a.get("status") == "PASS"
            and case_b.get("status") == "PASS"
            and case_c.get("status") == "PASS"
        )
        if can_run_import_rebase:
            assert inputs is not None
            case_d = initialize_case("import_rebase")
            try:
                context_b.reset()
                torch.cuda.synchronize(device)
                case_d["history"].append({"event": "reset", "position": int(context_b.current_position)})
                context_b.import_cache_async(actual_cache, current_position=CROSSING_END)
                torch.cuda.synchronize(device)
                if int(context_b.current_position) != CROSSING_END:
                    raise ProbeContractError(
                        f"imported source cache ended at {context_b.current_position}, expected {CROSSING_END}"
                    )
                imported_snapshot = _record_reset_diagnostic_cache(
                    torch, context_b, stage="after_import", case_name="import_rebase",
                    cuda_memcpy=cuda_runtime.cudaMemcpy, arrays=arrays, metadata=array_metadata,
                )
                case_d["history"].append({"event": "import", **imported_snapshot})
                context_b.compact_audio_history(expected_cached_frames=120, retain_frames=32)
                torch.cuda.synchronize(device)
                if int(context_b.current_position) != 162:
                    raise ProbeContractError(
                        f"rebase ended at {context_b.current_position}, expected 162"
                    )
                rebased_snapshot = _record_reset_diagnostic_cache(
                    torch, context_b, stage="after_rebase", case_name="import_rebase",
                    cuda_memcpy=cuda_runtime.cudaMemcpy, arrays=arrays, metadata=array_metadata,
                )
                case_d["history"].append({"event": "rebase", **rebased_snapshot})
                output = context_b.step_async(inputs[3][0])
                torch.cuda.synchronize(device)
                context_b.verify_output_health()
                if int(context_b.current_position) != 162 + Q_LEN:
                    raise ProbeContractError(
                        f"post-rebase Q4 ended at {context_b.current_position}, expected {162 + Q_LEN}"
                    )
                _record_tensor(
                    arrays, array_metadata, "baseline_reset.import_rebase.q4_output", output
                )
                post_snapshot = _record_reset_diagnostic_cache(
                    torch, context_b, stage="after_q4", case_name="import_rebase",
                    cuda_memcpy=cuda_runtime.cudaMemcpy, arrays=arrays, metadata=array_metadata,
                )
                case_d.update({
                    "status": "PASS",
                    "position_after_import": imported_snapshot["position"],
                    "position_after_rebase": rebased_snapshot["position"],
                    "position_after_q4": post_snapshot["position"],
                    "source_cache_sha256": case_a.get("import_source_cache_sha256"),
                    "q4_input_frame_index": 0,
                    "q4_output_sha256": array_metadata[
                        "baseline_reset.import_rebase.q4_output"
                    ]["sha256"],
                    "after_import_cache": imported_snapshot["sha256"],
                    "after_rebase_cache": rebased_snapshot["sha256"],
                    "after_q4_cache": post_snapshot["sha256"],
                })
            except Exception as exc:
                case_d["status"] = "FAIL"
                case_d["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            finally:
                finish_case_inputs(case_d)
                _save_reset_diagnostic_artifacts(
                    report, arrays, array_metadata,
                    report_path=report_path, arrays_path=arrays_path,
                )
        else:
            reasons = []
            if actual_cache is None:
                reasons.append("fresh_a actual cache at position 514 unavailable")
            if case_b.get("status") != "PASS":
                reasons.append("fresh_b did not complete successfully")
            if case_c.get("status") != "PASS":
                reasons.append("same_context_reset did not complete successfully")
            report["route_cases"]["import_rebase"] = {
                "status": "SKIPPED",
                "reason": "; ".join(reasons),
                "input_sha256": inputs[-1] if inputs is not None else None,
            }

        # All route attempts finish before any numerical comparison is allowed
        # to influence the result, so a mismatch cannot truncate the matrix.
        matrix = assemble_baseline_reset_diagnostic(
            case_records=report["route_cases"],
            arrays=arrays,
            expected_input_sha256=inputs[-1],
            tolerances=TOLERANCES,
        )
        report["route_matrix"] = matrix
        report["status"] = matrix["status"]
        report["memory_after_scenarios"] = _memory_snapshot(torch, device)
        report["arrays"] = array_metadata
        report["candidate_status"] = "UNMEASURED"
        _save_reset_diagnostic_artifacts(
            report, arrays, array_metadata, report_path=report_path, arrays_path=arrays_path
        )
        return report
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        for case_name in BASELINE_RESET_DIAGNOSTIC_CASES:
            report["route_cases"].setdefault(
                case_name,
                {"status": "SKIPPED", "reason": "worker setup or earlier unsafe context failure"},
            )
        if inputs is not None:
            report["route_matrix"] = assemble_baseline_reset_diagnostic(
                case_records=report["route_cases"],
                arrays=arrays,
                expected_input_sha256=inputs[-1],
                tolerances=TOLERANCES,
            )
        else:
            report["route_matrix"] = {
                "schema": BASELINE_RESET_DIAGNOSTIC_SCHEMA,
                "status": "FAIL",
                "case_order": list(BASELINE_RESET_DIAGNOSTIC_CASES),
                "cases": dict(report["route_cases"]),
                "comparisons": {},
                "tolerances": dict(TOLERANCES),
                "errors": ["fixed model inputs were not created; no route comparison is valid"],
            }
        _save_reset_diagnostic_artifacts(
            report, arrays, array_metadata, report_path=report_path, arrays_path=arrays_path
        )
        return report
    finally:
        for native in (context_b, context_a):
            if native is not None and torch is not None:
                try:
                    torch.cuda.synchronize()
                    native.close()
                except Exception:
                    pass
        if previous_library_env is None:
            os.environ.pop(NATIVE_LIBRARY_ENV, None)
        else:
            os.environ[NATIVE_LIBRARY_ENV] = previous_library_env


def assemble_report(
    *,
    manifest_path: str | Path,
    baseline_report_path: str | Path,
    baseline_arrays_path: str | Path,
    candidate_report_path: str | Path,
    candidate_arrays_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    baseline = json.loads(Path(baseline_report_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_report_path).read_text(encoding="utf-8"))
    errors: list[str] = []
    loaded_arrays: dict[str, dict[str, np.ndarray]] = {}
    for report, role, arrays_path in (
        (baseline, "baseline", baseline_arrays_path),
        (candidate, "candidate", candidate_arrays_path),
    ):
        try:
            record = manifest["libraries"][role]
            validate_worker_report(
                report,
                expected_role=role,
                expected_library_sha256=record["sha256"],
                expected_source_manifest_sha256=record["source_manifest_sha256"],
            )
            loaded_arrays[role] = _load_bound_worker_arrays(report, arrays_path)
        except ProbeContractError as exc:
            errors.append(f"{role}: {exc}")
    identity_fields = (
        "model_checkpoint_files_sha256",
        "native_topology_manifest_sha256",
        "input_sha256",
    )
    for field in identity_fields:
        if baseline.get(field) != candidate.get(field):
            errors.append(f"baseline/candidate {field} mismatch")
    if baseline.get("fixed_input_sha256") != candidate.get("fixed_input_sha256"):
        errors.append("baseline/candidate fixed_input_sha256 mismatch")
    comparison: dict[str, Any] | None = None
    if not errors:
        try:
            comparison = compare_tensor_arrays(
                {
                    name: value
                    for name, value in loaded_arrays["baseline"].items()
                    if not name.startswith("repeat.")
                },
                loaded_arrays["candidate"],
                tolerances=manifest["tolerances"],
            )
        except ProbeContractError as exc:
            errors.append(f"tensor comparison could not be assembled: {exc}")
        if comparison is not None and not comparison["passed"]:
            errors.append("baseline/candidate tensor metrics exceed pinned tolerances")
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "runtime_contract": manifest["runtime"],
        "source_contract": {
            role: {
                "library_path": manifest["libraries"][role]["path"],
                "library_sha256": manifest["libraries"][role]["sha256"],
                "source_manifest_path": manifest["libraries"][role]["source_manifest_path"],
                "source_manifest_sha256": manifest["libraries"][role]["source_manifest_sha256"],
                "provenance_note": manifest["libraries"][role].get("provenance_note"),
            }
            for role in ("baseline", "candidate")
        },
        "source_reconstruction_note": manifest.get("source_reconstruction"),
        "workers": {"baseline": baseline, "candidate": candidate},
        "comparison": comparison,
        "errors": errors,
        "coverage_boundary": {
            "demonstrated": [
                "canonical prime padded compute length 291 with valid prefix length 34",
                "native Q4-only step across positions 510 to 514",
                "actual-cache import at 514, rebase/reset, source13/source14/full-layer diagnostics",
                "ordered-long full attention after importing a pinned synthetic cache at position 2049",
                "two simultaneously allocated and primed candidate contexts with measured CUDA free bytes",
            ],
            "unsupported_or_unverified": [
                "Q1 listener step API is not exposed; the probe does not invent one",
                "no public history-batch step API exists; compute291/valid34 is the production prime path, not an exhaustive arbitrary batch API sweep",
                "synthetic imported KV at position 2049 validates dispatch/parity only, not a 2049-token model-produced dialogue history",
                "memory measurements cover listener contexts plus loaded model and do not prove peak memory with a loaded Talker or serving pair",
            ],
        },
    }
    atomic_write_json(output_path, report)
    return report


def run_all(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve()
    python = sys.executable
    outputs: dict[str, dict[str, str]] = {}
    base_environment = os.environ.copy()
    manifest = load_manifest(manifest_path)
    for role in ("baseline", "candidate"):
        report_path = directory / f"{role}_worker.json"
        arrays_path = directory / f"{role}_tensors.npz"
        log_path = directory / f"{role}_worker.log"
        command = [
            python,
            str(runner),
            "worker",
            "--manifest", str(manifest_path),
            "--role", role,
            "--report", str(report_path),
            "--arrays", str(arrays_path),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env=_worker_environment(base_environment, manifest["libraries"][role]),
            )
        outputs[role] = {
            "report": str(report_path),
            "arrays": str(arrays_path),
            "log": str(log_path),
            "exit_code": str(completed.returncode),
        }
        atomic_write_json(directory / "sequential_worker_status.json", {
            "schema": "gemma4-native-listener-capacity-sequential-status/v1",
            "workers": outputs,
            "order": ["baseline", "candidate"],
            "concurrency": 1,
        })
        if completed.returncode != 0:
            failure = {
                "schema": REPORT_SCHEMA,
                "status": "FAIL",
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "sequential_workers": outputs,
                "errors": [f"{role} worker exited {completed.returncode}; see its report and log"],
            }
            atomic_write_json(directory / "full_report.json", failure)
            return failure
    return assemble_report(
        manifest_path=manifest_path,
        baseline_report_path=outputs["baseline"]["report"],
        baseline_arrays_path=outputs["baseline"]["arrays"],
        candidate_report_path=outputs["candidate"]["report"],
        candidate_arrays_path=outputs["candidate"]["arrays"],
        output_path=directory / "full_report.json",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run sequential baseline/candidate workers and assemble the full report")
    run_parser.add_argument("--manifest", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    worker_parser = subparsers.add_parser("worker", help="execute one pinned library worker")
    worker_parser.add_argument("--manifest", required=True, type=Path)
    worker_parser.add_argument("--role", required=True, choices=("baseline", "candidate"))
    worker_parser.add_argument("--report", required=True, type=Path)
    worker_parser.add_argument("--arrays", required=True, type=Path)
    diagnostic_parser = subparsers.add_parser(
        "diagnose-baseline",
        help="run baseline-only fresh/reset/import route history diagnostics",
    )
    diagnostic_parser.add_argument("--manifest", required=True, type=Path)
    diagnostic_parser.add_argument("--report", required=True, type=Path)
    diagnostic_parser.add_argument("--arrays", required=True, type=Path)
    assemble_parser = subparsers.add_parser("assemble", help="assemble reports already produced by sequential workers")
    assemble_parser.add_argument("--manifest", required=True, type=Path)
    assemble_parser.add_argument("--baseline-report", required=True, type=Path)
    assemble_parser.add_argument("--baseline-arrays", required=True, type=Path)
    assemble_parser.add_argument("--candidate-report", required=True, type=Path)
    assemble_parser.add_argument("--candidate-arrays", required=True, type=Path)
    assemble_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_all(args.manifest, args.output_dir)
        elif args.command == "worker":
            result = run_worker(
                manifest_path=args.manifest,
                role=args.role,
                report_path=args.report,
                arrays_path=args.arrays,
            )
        elif args.command == "diagnose-baseline":
            result = run_baseline_reset_diagnostic(
                manifest_path=args.manifest,
                report_path=args.report,
                arrays_path=args.arrays,
            )
        else:
            result = assemble_report(
                manifest_path=args.manifest,
                baseline_report_path=args.baseline_report,
                baseline_arrays_path=args.baseline_arrays,
                candidate_report_path=args.candidate_report,
                candidate_arrays_path=args.candidate_arrays,
                output_path=args.output,
            )
        print(json.dumps({"status": result.get("status"), "report": result}, indent=2))
        return 0 if result.get("status") == "PASS" else 1
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc), "type": type(exc).__name__}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
