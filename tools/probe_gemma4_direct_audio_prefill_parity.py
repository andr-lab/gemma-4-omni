"""Fixed-input parity probe for Gemma4 direct-audio LM prefill chunking.

The GPU mode runs the production Gemma4 audio encoder and masked scatter once,
freezes the resulting LM inputs, then compares whole-prefill, whole-repeat, and
the opt-in chunked LM path from the same empty initial KV cache. This is a
numerical diagnostic, not a speech-quality or duplex certification.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any


SCHEMA = "gemma4_direct_audio_prefill_parity_v1"
EXPECTED_AUDIO_SHA256 = "f8778953767e7a856be4f57fb8f0f4bf35917643e70b296c6ffac32314a5677e"
EXPECTED_CONFIG_SHA256 = "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330"
RTOL = 1e-4
ATOL = 1e-5
CHUNK_TOKENS = 64
MODE_B_INSTRUCTIONS = (
    "Sei un assistente vocale italiano. Ascolta con attenzione l'altro interlocutore e "
    "ripeti in una frase breve il nome e ogni cifra del codice che senti, nello stesso ordine "
    "(ad esempio: 'Ho capito Marco, confermo codice 4 8 2 9'). "
    "Non convertire una sequenza di cifre in un numero composto e non inventare."
)
SOURCE_FILES = (
    "tools/probe_gemma4_direct_audio_prefill_parity.py",
    "duplex_kernel/engine.py",
    "duplex_kernel/server.py",
    "duplex_kernel/audio_response_conditioning.py",
    "duplex_kernel/models/thinker.py",
    "client/duplex_stream_e2e.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_bool(value: Any) -> bool:
    return value is True


def _tracked_model_load(load_model: Any, runtime_state: dict[str, Any]) -> Any:
    """Track the boundary around the production weight-loading call."""
    runtime_state["stage"] = "model_loading"
    runtime_state["model_load_started"] = True
    try:
        model = load_model()
    except BaseException:
        _refresh_cuda_runtime_state(runtime_state)
        raise
    runtime_state["model_loaded"] = True
    runtime_state["stage"] = "model_loaded"
    _refresh_cuda_runtime_state(runtime_state)
    return model


def _refresh_cuda_runtime_state(runtime_state: dict[str, Any]) -> None:
    """Read CUDA initialization only when Torch is already imported."""
    torch = sys.modules.get("torch")
    if torch is None:
        runtime_state.setdefault("cuda_initialized", False)
        runtime_state.setdefault("gpu_used", False)
        return
    try:
        initialized = bool(torch.cuda.is_initialized())
    except Exception as exc:  # diagnostic must not replace the primary error
        runtime_state["cuda_initialized"] = None
        runtime_state["gpu_used"] = None
        runtime_state["cuda_state_read_error"] = f"{type(exc).__name__}: {exc}"
        return
    runtime_state["cuda_initialized"] = initialized
    runtime_state["gpu_used"] = initialized
    runtime_state.pop("cuda_state_read_error", None)


def _failure_runtime(runtime_state: dict[str, Any]) -> dict[str, Any]:
    _refresh_cuda_runtime_state(runtime_state)
    return {
        "gpu_used": runtime_state.get("gpu_used", False),
        "cuda_initialized": runtime_state.get("cuda_initialized", False),
        "model_load_started": bool(runtime_state.get("model_load_started", False)),
        "model_loaded": bool(runtime_state.get("model_loaded", False)),
        "stage": runtime_state.get("stage", "preflight"),
        "preflight_manifest": runtime_state.get("preflight_manifest"),
        "cuda_state_read_error": runtime_state.get("cuda_state_read_error"),
    }


def _all_nested_gates_pass(value: Any) -> bool:
    if isinstance(value, dict):
        for key in ("passed", "same_final_next_token_argmax"):
            if key in value and value[key] is not True:
                return False
        return all(_all_nested_gates_pass(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_nested_gates_pass(item) for item in value)
    return True


def assemble_parity_report(
    *,
    worker_exit_code: int,
    comparisons: dict[str, Any],
    source_attestation: dict[str, Any],
    fixture: dict[str, Any],
    worker_error: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    """Fail-closed report builder shared by the worker and CPU tests."""
    whole_repeat = comparisons.get("whole_repeat", {})
    whole_vs_chunk = comparisons.get("whole_vs_chunk64", {})
    masks = comparisons.get("mask_query_rows_exact", {})
    layer_counts = comparisons.get("snapshot_layer_counts", {})
    gates = {
        "worker_exit_zero": worker_exit_code == 0,
        "source_attestation_ready": _strict_bool(source_attestation.get("ready")),
        "audio_fixture_hash_matches": fixture.get("audio_sha256") == EXPECTED_AUDIO_SHA256,
        "encoder_ran_once": fixture.get("encoder_calls") == 1,
        "whole_repeat_passed": _strict_bool(whole_repeat.get("passed"))
        and _all_nested_gates_pass(whole_repeat),
        "whole_vs_chunk64_passed": _strict_bool(whole_vs_chunk.get("passed"))
        and _all_nested_gates_pass(whole_vs_chunk),
        "mask_query_rows_exact": _strict_bool(masks.get("passed"))
        and _all_nested_gates_pass(masks),
        "snapshot_layer_counts_match_config": _strict_bool(layer_counts.get("passed"))
        and _all_nested_gates_pass(layer_counts),
    }
    all_passed = all(gates.values())
    result = {
        "schema": SCHEMA,
        "status": "passed" if all_passed else "failed",
        "worker_exit_code": int(worker_exit_code),
        "worker_error": worker_error,
        "gpu_used": (runtime or {}).get("gpu_used", False),
        "model_loaded": bool((runtime or {}).get("model_loaded", False)),
        "source_attestation": source_attestation,
        "fixture": fixture,
        "runtime": runtime or {},
        "tolerances": {"rtol": RTOL, "atol": ATOL, "dtype": "bfloat16"},
        "comparisons": comparisons,
        "gates": gates,
        "all_parity_gates_pass": all_passed,
        "limitations": list(limitations or []),
    }
    return result


def _safe_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_hashes(workspace: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = (workspace / relative).resolve()
        if not path.is_file() or not path.is_relative_to(workspace.resolve()):
            raise RuntimeError(f"source attestation file missing or outside workspace: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def _manifest_asset_path(workspace: Path, record: Any, field: str) -> Path:
    if not isinstance(record, dict):
        raise RuntimeError(f"CPU preflight {field} record is missing")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"CPU preflight {field} path is missing")
    relative_path = Path(raw_path)
    if relative_path.is_absolute():
        raise RuntimeError(f"CPU preflight {field} path must be workspace-relative")
    resolved = (workspace / relative_path).resolve()
    if not resolved.is_relative_to(workspace):
        raise RuntimeError(f"CPU preflight {field} path escapes the workspace")
    return resolved


def _checkpoint_inventory(model_path: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(model_path.glob("*.safetensors")):
        record: dict[str, Any] = {"name": path.name, "size_bytes": path.stat().st_size}
        # HF LFS cache blobs are named by their content SHA-256. Record that
        # attested object id without rereading multi-GB weights during preflight.
        if path.is_symlink():
            target_name = path.resolve().name
            if len(target_name) == 64 and all(c in "0123456789abcdef" for c in target_name.lower()):
                record["content_sha256_from_hf_blob_name"] = target_name.lower()
        files.append(record)
    return files


def build_cpu_preflight(workspace: Path, audio_path: Path, model_path: Path) -> dict[str, Any]:
    """Attest local inputs/config without loading model weights or initializing CUDA."""
    workspace = workspace.resolve()
    audio_path = audio_path.resolve()
    model_path = model_path.resolve()
    config_path = model_path / "config.json"
    if not audio_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Mode-B audio fixture or cached Gemma config is unavailable")
    if not audio_path.is_relative_to(workspace) or not model_path.is_relative_to(workspace):
        raise RuntimeError("preflight fixture and model cache must reside under the mounted workspace")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    text_config = config.get("text_config", {})
    audio_sha = sha256_file(audio_path)
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    source_hashes = _source_hashes(workspace)
    checkpoint_files = _checkpoint_inventory(model_path)
    checkpoint_inventory_ready = bool(checkpoint_files) and all(
        isinstance(item.get("size_bytes"), int) and item["size_bytes"] > 0
        for item in checkpoint_files
    )
    return {
        "schema": SCHEMA,
        "mode": "cpu_preflight",
        "ready_for_gpu_probe": bool(
            audio_sha == EXPECTED_AUDIO_SHA256
            and config_sha == EXPECTED_CONFIG_SHA256
            and checkpoint_inventory_ready
        ),
        "gpu_used": False,
        "model_weights_loaded": False,
        "audio_fixture": {
            "path": str(audio_path.relative_to(workspace)),
            "sha256": audio_sha,
            "size_bytes": audio_path.stat().st_size,
            "expected_sha256": EXPECTED_AUDIO_SHA256,
        },
        "model": {
            "path": str(model_path.relative_to(workspace)),
            "model_type": config.get("model_type"),
            "config_sha256": config_sha,
            "expected_config_sha256": EXPECTED_CONFIG_SHA256,
            "text": {
                "num_hidden_layers": text_config.get("num_hidden_layers"),
                "layer_types": text_config.get("layer_types"),
                "sliding_window": text_config.get("sliding_window"),
                "use_bidirectional_attention": text_config.get("use_bidirectional_attention"),
                "attention_backend_configured": text_config.get("_attn_implementation"),
            },
            "checkpoint_files": checkpoint_files,
            "checkpoint_inventory_ready": checkpoint_inventory_ready,
        },
        "source_sha256": source_hashes,
        "limitations": [
            "Preflight reads config and file metadata only; it does not open model weight tensors or initialize CUDA.",
            "Weight content hashes are reported only when supplied by a recognized Hugging Face LFS blob name.",
        ],
    }


def validate_preflight_manifest(
    workspace: Path,
    manifest_path: Path,
    *,
    audio_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    """Bind GPU execution to the exact CPU-preflight source and asset hashes."""
    workspace = workspace.resolve()
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file() or not manifest_path.is_relative_to(workspace):
        raise RuntimeError("CPU preflight manifest is missing or outside the mounted workspace")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("ready_for_gpu_probe") is not True:
        raise RuntimeError("CPU preflight manifest is not ready for this parity probe")
    audio_record = manifest.get("audio_fixture", {})
    model_record = manifest.get("model", {})
    expected_audio_path = _manifest_asset_path(workspace, audio_record, "audio fixture")
    expected_model_path = _manifest_asset_path(workspace, model_record, "model")
    audio_path = audio_path.resolve()
    model_path = model_path.resolve()
    if audio_path != expected_audio_path:
        raise RuntimeError("worker audio path differs from the CPU-preflight fixture path")
    if model_path != expected_model_path:
        raise RuntimeError("worker model path differs from the CPU-preflight checkpoint path")
    expected = manifest.get("source_sha256")
    if not isinstance(expected, dict) or set(expected) != set(SOURCE_FILES):
        raise RuntimeError("CPU preflight source hash set is incomplete")
    actual = _source_hashes(workspace)
    mismatches = {
        name: {"expected": expected.get(name), "actual": actual[name]}
        for name in SOURCE_FILES
        if expected.get(name) != actual[name]
    }
    if mismatches:
        raise RuntimeError(f"CPU preflight source mismatch: {mismatches}")
    if (
        not audio_path.is_file()
        or sha256_file(audio_path) != EXPECTED_AUDIO_SHA256
    ):
        raise RuntimeError("CPU preflight audio fixture changed after attestation")
    config_path = model_path / "config.json"
    if not config_path.is_file() or sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("CPU preflight model config changed after attestation")
    expected_inventory = model_record.get("checkpoint_files")
    actual_inventory = _checkpoint_inventory(model_path)
    if (
        model_record.get("checkpoint_inventory_ready") is not True
        or not isinstance(expected_inventory, list)
        or not expected_inventory
        or expected_inventory != actual_inventory
    ):
        raise RuntimeError("CPU preflight checkpoint inventory changed or is incomplete")
    return {
        "path": str(manifest_path.relative_to(workspace)),
        "sha256": sha256_file(manifest_path),
        "source_sha256": actual,
        "audio_sha256": EXPECTED_AUDIO_SHA256,
        "audio_path": str(audio_path.relative_to(workspace)),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "model_path": str(model_path.relative_to(workspace)),
        "checkpoint_files": actual_inventory,
    }


def _copy_cpu_tree(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous().clone()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {key: _copy_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_copy_cpu_tree(item) for item in value]
    raise TypeError(f"unsupported frozen direct-audio input type: {type(value).__name__}")


def _copy_to_device_tree(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _copy_to_device_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_to_device_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_copy_to_device_tree(item, device) for item in value]
    return value


def _cache_tensors(cache: Any) -> dict[str, Any]:
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, (list, tuple)) or not layers:
        raise RuntimeError("Gemma4 parity output did not expose Cache.layers")
    captured: dict[str, Any] = {}
    for index, layer in enumerate(layers):
        for label in ("keys", "values"):
            tensor = getattr(layer, label, None)
            if tensor is None:
                raise RuntimeError(f"Gemma4 cache layer {index} is missing {label}")
            captured[f"cache.layer_{index}.{label}"] = _copy_cpu_tree(tensor)
    return captured


def _snapshot_lm_output(output: Any, model: Any) -> dict[str, Any]:
    import torch

    hidden_states = getattr(output, "hidden_states", None)
    if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
        raise RuntimeError("Gemma4 direct-audio LM output omitted hidden_states")
    final_hidden = getattr(output, "final_hidden_state", None)
    if final_hidden is None:
        final_hidden = getattr(output, "last_hidden_state", None)
    if not isinstance(final_hidden, torch.Tensor):
        raise RuntimeError("Gemma4 direct-audio LM output omitted final hidden state")
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        logits = model.lm_head(final_hidden[:, -1:, :])
        softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
    tensors = {
        "final_hidden": _copy_cpu_tree(final_hidden),
        "final_logits": _copy_cpu_tree(logits),
    }
    for index, hidden in enumerate(hidden_states):
        tensors[f"hidden_state_{index}"] = _copy_cpu_tree(hidden)
    tensors.update(_cache_tensors(getattr(output, "past_key_values", None)))
    return {
        "tensors": tensors,
        "final_next_token_argmax": [int(value) for value in logits.argmax(dim=-1).detach().cpu().reshape(-1).tolist()],
        "layer_count": len(hidden_states),
        "cache_layer_count": len(getattr(output.past_key_values, "layers", ())),
    }


def _tensor_comparison(reference: Any, candidate: Any) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "shape_reference": list(reference.shape),
        "shape_candidate": list(candidate.shape),
        "rtol": RTOL,
        "atol": ATOL,
    }
    if tuple(reference.shape) != tuple(candidate.shape):
        return {**result, "passed": False, "reason": "shape_mismatch"}
    ref = reference.detach().to(device="cpu", dtype=torch.float64)
    got = candidate.detach().to(device="cpu", dtype=torch.float64)
    finite = bool(torch.isfinite(ref).all().item() and torch.isfinite(got).all().item())
    if not finite:
        return {**result, "passed": False, "finite": False, "reason": "non_finite_tensor"}
    delta = (got - ref).abs()
    tolerance = ATOL + RTOL * ref.abs()
    normalized = torch.where(tolerance > 0, delta / tolerance, torch.where(delta == 0, 0.0, math.inf))
    ref_flat = ref.reshape(-1)
    got_flat = got.reshape(-1)
    cosine = float(torch.nn.functional.cosine_similarity(ref_flat, got_flat, dim=0).item())
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    result.update({
        "passed": bool(torch.all(delta <= tolerance).item()),
        "finite": finite,
        "max_abs_error": max_abs,
        "mean_abs_error": float(delta.mean().item()) if delta.numel() else 0.0,
        "relative_l2_error": float(torch.linalg.vector_norm(delta) / torch.clamp(torch.linalg.vector_norm(ref), min=1e-30)),
        "max_normalized_error": float(normalized.max().item()) if normalized.numel() else 0.0,
        "cosine_similarity": cosine,
    })
    return result


def _compare_snapshots(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref_tensors = reference.get("tensors", {})
    candidate_tensors = candidate.get("tensors", {})
    names = sorted(set(ref_tensors) | set(candidate_tensors))
    tensor_results: dict[str, Any] = {}
    for name in names:
        if name not in ref_tensors or name not in candidate_tensors:
            tensor_results[name] = {"passed": False, "reason": "missing_tensor"}
        else:
            tensor_results[name] = _tensor_comparison(ref_tensors[name], candidate_tensors[name])
    argmax_equal = reference.get("final_next_token_argmax") == candidate.get("final_next_token_argmax")
    return {
        "passed": bool(
            all(item.get("passed") is True for item in tensor_results.values()) and argmax_equal
        ),
        "tensors": tensor_results,
        "same_final_next_token_argmax": argmax_equal,
        "final_next_token_argmax_reference": reference.get("final_next_token_argmax"),
        "final_next_token_argmax_candidate": candidate.get("final_next_token_argmax"),
    }


def _expected_gemma4_cache_layout(language_model: Any, text_config: Any) -> dict[str, Any]:
    """Validate Gemma4's configured KV-sharing suffix and expected cache count."""
    transformer_layers = getattr(text_config, "num_hidden_layers", None)
    shared_layers = getattr(text_config, "num_kv_shared_layers", 0)
    if (
        isinstance(transformer_layers, bool)
        or not isinstance(transformer_layers, int)
        or transformer_layers < 1
    ):
        raise RuntimeError("Gemma4 text config has no valid num_hidden_layers")
    if isinstance(shared_layers, bool) or not isinstance(shared_layers, int):
        raise RuntimeError("Gemma4 text config has no valid num_kv_shared_layers")
    if shared_layers < 0 or shared_layers >= transformer_layers:
        raise RuntimeError("Gemma4 KV-sharing count is outside the supported layer range")

    configured_layer_types = getattr(text_config, "layer_types", None)
    if not isinstance(configured_layer_types, (tuple, list)) or len(configured_layer_types) != transformer_layers:
        raise RuntimeError("Gemma4 text config layer_types do not match num_hidden_layers")
    layers = getattr(language_model, "layers", None)
    try:
        actual_layer_count = len(layers)
    except TypeError as exc:
        raise RuntimeError("Gemma4 language model does not expose an indexable layer sequence") from exc
    if actual_layer_count != transformer_layers:
        raise RuntimeError(
            f"Gemma4 language model exposes {actual_layer_count} layers; config requires {transformer_layers}"
        )

    actual_shared_indices: list[int] = []
    for index in range(actual_layer_count):
        try:
            attention = layers[index].self_attn
            is_shared = attention.is_kv_shared_layer
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Gemma4 layer {index} has no readable KV-sharing marker") from exc
        if type(is_shared) is not bool:
            raise RuntimeError(f"Gemma4 layer {index} KV-sharing marker is not boolean")
        if is_shared:
            actual_shared_indices.append(index)

    shared_start = transformer_layers - shared_layers
    expected_shared_indices = list(range(shared_start, transformer_layers))
    if actual_shared_indices != expected_shared_indices:
        raise RuntimeError(
            "Gemma4 KV-sharing layer markers differ from the config suffix: "
            f"actual={actual_shared_indices}, expected={expected_shared_indices}"
        )
    return {
        "transformer_layer_count": transformer_layers,
        "configured_kv_shared_layer_count": shared_layers,
        "shared_layer_indices": actual_shared_indices,
        "expected_unique_cache_layer_count": transformer_layers - shared_layers,
    }


def _snapshot_layer_counts_comparison(
    snapshots: dict[str, dict[str, Any]],
    expected_transformer_layers: int,
    expected_cache_layers: int,
    expected_kv_shared_layers: int,
) -> dict[str, Any]:
    """Require every snapshot to contain all HF hidden states and unique KV slots."""
    if isinstance(expected_transformer_layers, bool) or expected_transformer_layers < 1:
        raise ValueError("expected transformer layer count must be a positive integer")
    if (
        isinstance(expected_cache_layers, bool)
        or expected_cache_layers < 1
        or expected_cache_layers > expected_transformer_layers
    ):
        raise ValueError("expected unique cache layer count must be within the transformer layer range")
    if (
        isinstance(expected_kv_shared_layers, bool)
        or not isinstance(expected_kv_shared_layers, int)
        or expected_kv_shared_layers < 0
        or expected_kv_shared_layers != expected_transformer_layers - expected_cache_layers
    ):
        raise ValueError("unique cache layer count does not match configured KV sharing")
    expected = {
        "hidden_state_count": expected_transformer_layers + 1,
        "cache_layer_count": expected_cache_layers,
    }
    records: dict[str, Any] = {}
    for label in ("whole", "whole_repeat", "chunk64"):
        snapshot = snapshots.get(label, {})
        actual = {
            "hidden_state_count": snapshot.get("layer_count"),
            "cache_layer_count": snapshot.get("cache_layer_count"),
        }
        records[label] = {
            "expected": expected,
            "actual": actual,
            "passed": actual == expected,
        }
    return {
        "passed": all(item["passed"] for item in records.values()),
        "expected_transformer_layer_count": expected_transformer_layers,
        "configured_kv_shared_layer_count": expected_kv_shared_layers,
        "expected_unique_cache_layer_count": expected_cache_layers,
        "validated_shared_layer_indices": list(
            range(expected_transformer_layers - expected_kv_shared_layers, expected_transformer_layers)
        ),
        "snapshots": records,
    }


def _mask_digest(mask: Any) -> dict[str, Any]:
    import torch

    if mask is None:
        return {"kind": "none"}
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"unsupported attention mask type: {type(mask).__name__}")
    raw = mask.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return {
        "kind": "tensor",
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _mask_rows_comparison(full_masks: dict[str, Any], chunk_records: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    records: list[dict[str, Any]] = []
    all_equal = True
    for chunk in chunk_records:
        start = int(chunk["query_start"])
        end = int(chunk["query_end"])
        per_layer: dict[str, Any] = {}
        for layer_type, actual in chunk["masks"].items():
            expected_full = full_masks.get(layer_type)
            if expected_full is None and actual is None:
                passed = True
                reason = "both_none"
            elif not isinstance(expected_full, torch.Tensor) or not isinstance(actual, torch.Tensor):
                passed = False
                reason = "mask_materialization_mismatch"
            else:
                if expected_full.ndim != 4 or actual.ndim != 4:
                    passed = False
                    reason = "unexpected_mask_rank"
                else:
                    expected = expected_full[..., start:end, :end]
                    passed = bool(
                        tuple(expected.shape) == tuple(actual.shape)
                        and torch.equal(expected, actual.detach())
                    )
                    reason = None if passed else "query_rows_differ"
            per_layer[layer_type] = {
                "passed": passed,
                "reason": reason,
                "expected": _mask_digest(expected_full),
                "actual": _mask_digest(actual),
                "query_start": start,
                "query_end": end,
            }
            all_equal = all_equal and passed
        records.append({"query_start": start, "query_end": end, "layer_masks": per_layer})
    return {"passed": all_equal and bool(records), "chunk_count": len(records), "chunks": records}


def _audio_feature_digest(features: Any) -> dict[str, Any]:
    import torch

    raw = features.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "finite": bool(torch.isfinite(features).all().item()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _direct_audio_model_parts(outer_model: Any) -> tuple[Any, Any, Any]:
    """Resolve the production outer/nested Gemma4 APIs used by the engine."""
    gemma4_model = getattr(outer_model, "model", None)
    if gemma4_model is None or not callable(getattr(gemma4_model, "get_audio_features", None)):
        raise RuntimeError("loaded Gemma4 outer model has no nested get_audio_features API")
    language_model = getattr(gemma4_model, "language_model", None)
    if not callable(language_model):
        raise RuntimeError("loaded nested Gemma4 model has no callable language_model")
    mask_builder = getattr(outer_model, "create_masks_for_generate", None)
    if not callable(mask_builder):
        raise RuntimeError("loaded Gemma4 outer model has no create_masks_for_generate API")
    return gemma4_model, language_model, mask_builder


def _run_whole_repeat_snapshot(language_model: Any, make_inputs: Any, outer_model: Any) -> dict[str, Any]:
    """Run the repeat control and snapshot it against the loaded outer model."""
    import torch

    with torch.inference_mode():
        output = language_model(**make_inputs())
        return _snapshot_lm_output(output, outer_model)


def _run_gpu_probe(
    *, workspace: Path, audio_path: Path, model_path: Path, preflight_manifest_path: Path,
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    workspace = workspace.resolve()
    runtime_state["stage"] = "preflight_validation"
    preflight_handoff = validate_preflight_manifest(
        workspace,
        preflight_manifest_path,
        audio_path=audio_path,
        model_path=model_path,
    )
    runtime_state["preflight_manifest"] = preflight_handoff
    # File/source/path identity is checked before importing Torch or resolving
    # the model loader, so a stale handoff cannot reach framework initialization.
    runtime_state["stage"] = "torch_import"
    import torch

    runtime_state["stage"] = "cuda_availability_check"
    _refresh_cuda_runtime_state(runtime_state)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU parity mode requires CUDA; use --cpu-preflight for no-GPU preparation")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before preflight identity was validated")
    device = "cuda:0"

    from client.duplex_stream_e2e import read_audio_16k
    from duplex_kernel.engine import PipelineEngine
    from duplex_kernel.models.thinker import load_thinker, resolve_thinker_path
    from duplex_kernel.server import (
        CONTROLLER_GUIDED_SAY_V1,
        _prepare_direct_audio_processor_inputs,
    )

    model_path = model_path.resolve()
    config_path = model_path / "config.json"
    config_sha = sha256_file(config_path)
    if config_sha != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Gemma config hash changed after CPU preflight")
    loader_model_path = Path(resolve_thinker_path(str(workspace))).resolve()
    if loader_model_path != model_path.resolve():
        raise RuntimeError(
            "production Thinker loader would select a different checkpoint than CPU preflight: "
            f"expected={model_path}, loader={loader_model_path}"
        )
    wrapper = _tracked_model_load(
        lambda: load_thinker(str(workspace), device=device, dtype=torch.bfloat16),
        runtime_state,
    )
    thinker = wrapper.model
    if str(getattr(thinker, "_gemma4_checkpoint_path", "")) != str(model_path):
        raise RuntimeError("loaded Thinker checkpoint path differs from CPU preflight")
    if next(thinker.parameters()).dtype != torch.bfloat16:
        raise RuntimeError("loaded Gemma Thinker is not BF16")
    actual_config_sha = sha256_file(Path(thinker._gemma4_checkpoint_path) / "config.json")
    if actual_config_sha != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("loaded Gemma config differs from attested CPU preflight")

    audio_path = audio_path.resolve()
    if sha256_file(audio_path) != EXPECTED_AUDIO_SHA256:
        raise RuntimeError("Mode-B kickoff WAV hash differs from the pinned input")
    pcm = read_audio_16k(audio_path)
    if pcm.size == 0 or not bool(torch.isfinite(torch.from_numpy(pcm)).all().item()):
        raise RuntimeError("Mode-B kickoff audio is empty or non-finite after 16 kHz ingress conversion")
    probe_host = type("ProbeHost", (), {})()
    probe_host.tokenizer = wrapper.tokenizer
    probe_host._thinker_wrapper = wrapper
    prompt_ids, processor_inputs, processor_ms = _prepare_direct_audio_processor_inputs(
        probe_host,
        language="it",
        history_messages=[],
        pcm=pcm,
        response_protocol=CONTROLLER_GUIDED_SAY_V1,
        session_instructions=MODE_B_INSTRUCTIONS,
        controller_action="SPEAK",
    )

    engine = PipelineEngine.__new__(PipelineEngine)
    engine.device = device
    engine.thinker = thinker
    engine._thinker_wrapper = wrapper
    engine.tokenizer = wrapper.tokenizer
    engine._thinker_lock = threading.Lock()
    engine._thinker_audio_forward_active = threading.Event()
    engine._thinker_audio_prefill_lock_hold_max_ms = 0.0
    engine.thinker_audio_prefill_chunk_tokens = 0
    audio_inputs = engine._prepare_thinker_audio_inputs(processor_inputs, prompt_ids)

    frozen: dict[str, Any] = {}
    full_masks: dict[str, Any] = {}
    encoder_calls = 0
    outer_model = thinker
    gemma4_model, language_model, original_mask_builder = _direct_audio_model_parts(outer_model)
    original_get_audio_features = gemma4_model.get_audio_features
    had_instance_mask_builder = "create_masks_for_generate" in getattr(outer_model, "__dict__", {})

    def counted_get_audio_features(*args: Any, **kwargs: Any) -> Any:
        nonlocal encoder_calls
        encoder_calls += 1
        return original_get_audio_features(*args, **kwargs)

    def capture_mask_builder(*args: Any, **kwargs: Any) -> Any:
        result = original_mask_builder(*args, **kwargs)
        if not full_masks:
            full_masks.update(_copy_cpu_tree(result))
        return result

    def capture_first_lm_inputs(_module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if frozen:
            return
        for key in ("inputs_embeds", "per_layer_inputs", "position_ids", "attention_mask", "past_key_values"):
            frozen[key] = _copy_cpu_tree(kwargs.get(key))

    gemma4_model.get_audio_features = counted_get_audio_features
    outer_model.create_masks_for_generate = capture_mask_builder
    hook = language_model.register_forward_pre_hook(capture_first_lm_inputs, with_kwargs=True)
    try:
        metrics: dict[str, Any] = defaultdict(float)
        with torch.inference_mode():
            whole_output, _, _, _, _ = engine._run_thinker_audio_prefill(
                audio_inputs,
                is_compiled=False,
                static_cache=None,
                past_length=0,
                cache_init_started=None,
                runtime_metrics=metrics,
            )
        whole_snapshot = _snapshot_lm_output(whole_output, outer_model)
        postencoder_features = _audio_feature_digest(whole_output.audio_hidden_states)
    finally:
        hook.remove()
        gemma4_model.get_audio_features = original_get_audio_features
        if had_instance_mask_builder:
            outer_model.create_masks_for_generate = original_mask_builder
        else:
            delattr(outer_model, "create_masks_for_generate")

    if encoder_calls != 1:
        raise RuntimeError(f"direct-audio encoder must run exactly once, got {encoder_calls}")
    if frozen.get("past_key_values") is not None:
        raise RuntimeError("fixed Mode-B fixture unexpectedly started with a nonempty cache")
    if not isinstance(frozen.get("inputs_embeds"), torch.Tensor):
        raise RuntimeError("production direct-audio LM input embeddings were not captured")
    if not isinstance(frozen.get("position_ids"), torch.Tensor):
        raise RuntimeError("production direct-audio absolute positions were not captured")
    if not isinstance(frozen.get("attention_mask"), dict) or not full_masks:
        raise RuntimeError("production full attention masks were not captured")
    if not isinstance(audio_inputs.get("attention_mask"), torch.Tensor):
        raise RuntimeError("production 2D processor attention mask is unavailable")
    frozen["input_attention_mask"] = _copy_cpu_tree(audio_inputs["attention_mask"])
    frozen_input_digests = {
        key: _mask_digest(value) if key == "attention_mask" else _tensor_digest_record(value)
        for key, value in frozen.items()
        if key not in {"past_key_values"} and key != "attention_mask"
    }
    frozen_input_digests["full_attention_masks"] = {
        key: _mask_digest(value) for key, value in full_masks.items()
    }

    def make_whole_inputs() -> dict[str, Any]:
        return {
            "inputs_embeds": _copy_to_device_tree(frozen["inputs_embeds"], device),
            "per_layer_inputs": _copy_to_device_tree(frozen["per_layer_inputs"], device),
            "attention_mask": _copy_to_device_tree(frozen["attention_mask"], device),
            "position_ids": _copy_to_device_tree(frozen["position_ids"], device),
            "past_key_values": None,
            "use_cache": True,
            "return_dict": True,
            "output_hidden_states": True,
            "logits_to_keep": 1,
        }

    def run_whole_repeat() -> dict[str, Any]:
        return _run_whole_repeat_snapshot(language_model, make_whole_inputs, outer_model)

    whole_repeat_snapshot = run_whole_repeat()

    observed_chunks: list[dict[str, Any]] = []
    original_mask_builder = outer_model.create_masks_for_generate
    had_instance_mask_builder = "create_masks_for_generate" in getattr(outer_model, "__dict__", {})

    def observe_chunk_masks(*args: Any, **kwargs: Any) -> Any:
        result = original_mask_builder(*args, **kwargs)
        positions = args[4] if len(args) > 4 else kwargs.get("position_ids")
        embeds = args[1] if len(args) > 1 else kwargs.get("inputs_embeds")
        start = int(positions[0, 0].item())
        end = start + int(embeds.shape[1])
        observed_chunks.append({
            "query_start": start,
            "query_end": end,
            "masks": _copy_cpu_tree(result),
        })
        return result

    outer_model.create_masks_for_generate = observe_chunk_masks
    try:
        chunk_metrics: dict[str, Any] = defaultdict(float)
        with torch.inference_mode():
            chunk_output = engine._run_chunked_thinker_audio_language_model(
                inputs_embeds=_copy_to_device_tree(frozen["inputs_embeds"], device),
                per_layer_inputs=_copy_to_device_tree(frozen["per_layer_inputs"], device),
                attention_mask=_copy_to_device_tree(frozen["input_attention_mask"], device),
                position_ids=_copy_to_device_tree(frozen["position_ids"], device),
                cache_position=None,
                cache=None,
                is_compiled=False,
                chunk_tokens=CHUNK_TOKENS,
                runtime_metrics=chunk_metrics,
                listener_admission_wait=None,
            )
            chunk_snapshot = _snapshot_lm_output(chunk_output, outer_model)
    finally:
        if had_instance_mask_builder:
            outer_model.create_masks_for_generate = original_mask_builder
        else:
            delattr(outer_model, "create_masks_for_generate")

    chunk_mask_comparison = _mask_rows_comparison(full_masks, observed_chunks)
    text_config = outer_model.config.get_text_config()
    cache_layout = _expected_gemma4_cache_layout(language_model, text_config)
    expected_transformer_layers = cache_layout["transformer_layer_count"]
    comparisons = {
        "whole_repeat": _compare_snapshots(whole_snapshot, whole_repeat_snapshot),
        "whole_vs_chunk64": _compare_snapshots(whole_snapshot, chunk_snapshot),
        "mask_query_rows_exact": chunk_mask_comparison,
        "snapshot_layer_counts": _snapshot_layer_counts_comparison(
            {
                "whole": whole_snapshot,
                "whole_repeat": whole_repeat_snapshot,
                "chunk64": chunk_snapshot,
            },
            expected_transformer_layers,
            cache_layout["expected_unique_cache_layer_count"],
            cache_layout["configured_kv_shared_layer_count"],
        ),
    }
    comparisons["snapshot_layer_counts"]["validated_cache_layout"] = cache_layout
    model_root = Path(outer_model._gemma4_checkpoint_path)
    config_sha = sha256_file(model_root / "config.json")
    source_sha = _source_hashes(workspace)
    attestation = {
        "ready": all(source_sha.values()) and config_sha == EXPECTED_CONFIG_SHA256,
        "source_sha256": source_sha,
        "model_config_sha256": config_sha,
        "preflight_manifest": preflight_handoff,
    }
    audio_rel = str(audio_path.relative_to(workspace))
    fixture = {
        "audio_path": audio_rel,
        "audio_sha256": sha256_file(audio_path),
        "audio_sample_rate_hz": 16_000,
        "audio_sample_count": int(pcm.size),
        "audio_encoder_calls": encoder_calls,
        "encoder_calls": encoder_calls,
        "postencoder_valid_audio_features": postencoder_features,
        "frozen_lm_inputs": frozen_input_digests,
        "initial_kv": "empty; past_length=0 for the current direct-audio kickoff",
        "query_tokens": int(frozen["inputs_embeds"].shape[1]),
        "hard_text_prefix_tokens": 0,
        "processor_prepare_ms": processor_ms,
        "scenario_instruction_sha256": hashlib.sha256(MODE_B_INSTRUCTIONS.encode("utf-8")).hexdigest(),
    }
    runtime_state["stage"] = "parity_complete"
    _refresh_cuda_runtime_state(runtime_state)
    runtime = {
        "gpu_used": runtime_state.get("gpu_used", False),
        "cuda_initialized": runtime_state.get("cuda_initialized", False),
        "cuda_state_read_error": runtime_state.get("cuda_state_read_error"),
        "model_load_started": bool(runtime_state.get("model_load_started", False)),
        "model_loaded": bool(runtime_state.get("model_loaded", False)),
        "stage": runtime_state["stage"],
        "device": str(next(outer_model.parameters()).device),
        "dtype": str(next(outer_model.parameters()).dtype),
        "attention_backend": (
            getattr(text_config, "_attn_implementation", None)
            or getattr(outer_model.config, "_attn_implementation", None)
        ),
        "layer_types": list(getattr(text_config, "layer_types", [])),
        "num_hidden_layers": int(getattr(text_config, "num_hidden_layers", 0)),
        "sliding_window": getattr(text_config, "sliding_window", None),
        "compile_mode": "off",
        "compile_limit": "The standalone parity worker loads the production BF16 Thinker without the serving engine's optional compile warmup.",
        "whole_runtime_metrics": dict(metrics),
        "chunk_runtime_metrics": dict(chunk_metrics),
        "chunk_tokens": CHUNK_TOKENS,
        "model_path": str(model_root.relative_to(workspace)),
        "model_config_sha256": config_sha,
    }
    return assemble_parity_report(
        worker_exit_code=0,
        comparisons=comparisons,
        source_attestation=attestation,
        fixture=fixture,
        runtime=runtime,
        limitations=[
            "This probe is fixed-input numerical parity only; it does not certify generated speech or duplex latency.",
            "The production encoder/scatter runs once. Whole-repeat and chunk64 reuse the frozen post-scatter LM inputs.",
            "The current Mode-B kickoff has no cached hard-text prefix; cached-prefix full/sliding mask rows are covered by the focused tiny Gemma CPU test.",
            "The probe uses the uncompiled BF16 text LM path; it does not prove parity under the serving engine's optional compile mode.",
        ],
    )


def _tensor_digest_record(value: Any) -> dict[str, Any]:
    import torch

    if not isinstance(value, torch.Tensor):
        return {"kind": type(value).__name__}
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return {
        "kind": "tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _cpu_preflight_cli(args: argparse.Namespace) -> int:
    from duplex_kernel.models.thinker import resolve_thinker_path

    workspace = args.workspace.resolve()
    model_path = args.model_path.resolve() if args.model_path else Path(
        resolve_thinker_path(str(workspace))
    ).resolve()
    report = build_cpu_preflight(workspace, args.audio.resolve(), model_path)
    report["created_at_unix_ns"] = time.time_ns()
    _safe_json_write(args.output, report)
    print(json.dumps({"status": "ready" if report["ready_for_gpu_probe"] else "failed", "output": str(args.output), "model_weights_loaded": False}))
    return 0 if report["ready_for_gpu_probe"] else 3


def _gpu_cli(args: argparse.Namespace) -> int:
    report: dict[str, Any]
    runtime_state: dict[str, Any] = {
        "stage": "preflight",
        "model_load_started": False,
        "model_loaded": False,
        "cuda_initialized": False,
        "gpu_used": False,
        "preflight_manifest": None,
    }
    try:
        if args.preflight_manifest is None:
            raise RuntimeError("GPU mode requires --preflight-manifest from the same source snapshot")
        workspace = args.workspace.resolve()
        manifest = json.loads(args.preflight_manifest.resolve().read_text(encoding="utf-8"))
        model_record = manifest.get("model", {})
        model_path = (
            args.model_path.resolve()
            if args.model_path
            else (workspace / str(model_record.get("path", ""))).resolve()
        )
        report = _run_gpu_probe(
            workspace=workspace,
            audio_path=args.audio.resolve(),
            model_path=model_path,
            preflight_manifest_path=args.preflight_manifest.resolve(),
            runtime_state=runtime_state,
        )
    except BaseException as exc:
        try:
            handoff = None
            if args.preflight_manifest is not None and args.preflight_manifest.is_file():
                handoff = {
                    "path": str(args.preflight_manifest.resolve()),
                    "sha256": sha256_file(args.preflight_manifest.resolve()),
                }
        except Exception:
            handoff = None
        report = assemble_parity_report(
            worker_exit_code=1,
            worker_error={"type": type(exc).__name__, "message": str(exc)},
            comparisons={},
            source_attestation={"ready": False, "preflight_manifest": handoff},
            fixture={"audio_sha256": None, "encoder_calls": 0},
            runtime=_failure_runtime(runtime_state),
        )
    _safe_json_write(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output), "all_parity_gates_pass": report["all_parity_gates_pass"]}))
    return 0 if report["status"] == "passed" else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("artifacts/scenario_audio_qwen3tts/mode_b_kickoff.wav"),
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--preflight-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-preflight", action="store_true")
    args = parser.parse_args()
    return _cpu_preflight_cli(args) if args.cpu_preflight else _gpu_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
