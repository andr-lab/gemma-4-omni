#!/usr/bin/env python3
"""Production-prefix baseline/candidate capacity probe for the native listener.

The probe uses the serving listener's ShapeStable 34-token prefix path, imports
that frozen DynamicCache into native Q4 contexts, and compares a 505-frame
route through positions 34, 514, and 2054. It is isolated from the older
native-prime diagnostic and runs each DSO in a fresh subprocess.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from duplex_kernel.microbatch.native_listener_capacity_probe_contract import ProbeContractError, atomic_write_json, sha256_file
from duplex_kernel.microbatch.native_listener_capacity_production_contract import (
    CASES,
    DEFAULT_SPEC,
    REPORT_SCHEMA,
    SOURCE_LAYER_HEAD_DIMS,
    WORKER_SCHEMA,
    assemble_production_report,
    validate_production_worker_report,
)
from duplex_kernel.microbatch.probe_native_listener_capacity_parity import (
    MAX_SEQ_LEN,
    NATIVE_LIBRARY_ENV,
    PREFIX_LEN,
    Q_LEN,
    _array_sha256,
    _copy_cached_kv_from_native,
    _load_cuda_runtime_for_cache_copy,
    _memory_snapshot,
    _rng_snapshot,
    _sha_bytes,
    _sum_ordered_calls,
    _sync,
    _tensor_array,
    _torch_tensor_sha256,
    _verify_source_manifest,
    load_manifest,
)


SEED = 20261012
PREFIX_TOKENS = 34
CROSSING_FRAME_INDEX = 120
FRAME_COUNT = 505
CHECKPOINTS = (34, 514, 2054)


def _write_array(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
    name: str,
    tensor: Any,
) -> None:
    value = _tensor_array(tensor)
    arrays[name] = value
    metadata[name] = {
        "shape": list(value.shape),
        "dtype": "float32_from_bf16",
        "sha256": _array_sha256(value),
    }


def _hash_prefix_inputs(torch: Any, inputs: Mapping[str, Any]) -> str:
    parts = {
        name: _torch_tensor_sha256(value, torch)
        for name, value in sorted(inputs.items())
    }
    return _sha_bytes(json.dumps(parts, sort_keys=True).encode("utf-8"))


def _hash_prefix_source(torch: Any, prefix_cache: Any) -> str:
    parts: dict[str, str] = {}
    for layer in range(15):
        entry = prefix_cache.layers[layer]
        keys = getattr(entry, "keys", None)
        values = getattr(entry, "values", None)
        if keys is None or values is None:
            raise ProbeContractError(f"canonical ShapeStable cache has no source layer {layer}")
        if int(keys.shape[2]) != PREFIX_TOKENS or int(values.shape[2]) != PREFIX_TOKENS:
            raise ProbeContractError(f"canonical source layer {layer} is not exactly 34 tokens")
        parts[f"{layer}.k"] = _torch_tensor_sha256(keys, torch)
        parts[f"{layer}.v"] = _torch_tensor_sha256(values, torch)
    return _sha_bytes(json.dumps(parts, sort_keys=True).encode("utf-8"))


def _create_production_prefix(torch: Any, lm: Any, backend: Any, device: Any):
    """Mirror listener._forward_incremental_lm's ShapeStable prefix prefill."""
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

    from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT

    if AUDIO_START_AFTER_TEXT != PREFIX_TOKENS:
        raise ProbeContractError("canonical audio prefix length differs from the pinned 34-token route")
    input_ids, attention = backend._build_multimodal_inputs((), 1024, device=device)
    prefix_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
    prefix_attention = attention[:, :AUDIO_START_AFTER_TEXT].to(
        device=device, dtype=torch.long
    ).contiguous()
    if tuple(prefix_ids.shape) != (1, PREFIX_TOKENS) or tuple(prefix_attention.shape) != (1, PREFIX_TOKENS):
        raise ProbeContractError("production multimodal builder returned a noncanonical prefix")
    with torch.inference_mode():
        prefix_embeds = lm.embed_tokens(prefix_ids).contiguous()
        prefix_ple = lm.get_per_layer_inputs(prefix_ids, prefix_embeds).contiguous()
    prefix_positions = torch.arange(
        PREFIX_TOKENS, device=device, dtype=torch.long
    ).view(1, -1)
    prefix_masks = create_masks_for_generate(
        lm.config,
        prefix_embeds,
        prefix_attention,
        past_key_values=None,
        position_ids=prefix_positions,
    )
    with torch.inference_mode():
        prefix_output = backend._shape_stable_lm.forward(
            inputs_embeds=prefix_embeds,
            per_layer_inputs=prefix_ple,
            attention_mask=prefix_masks,
            position_ids=prefix_positions,
            past_key_values=None,
            shared_kv_states=None,
        )
    source_cache = backend._clone_canonical_prefix_snapshot(prefix_output.past_key_values)
    inputs = {
        "prefix.token_ids": prefix_ids.detach().clone(),
        "prefix.attention": prefix_attention.detach().clone(),
        "prefix.embeddings": prefix_embeds.detach().clone(),
        "prefix.per_layer_inputs": prefix_ple.detach().clone(),
    }
    return inputs, source_cache


def _prepare_frames(torch: Any, device: Any) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    cpu = (torch.randn(
        (FRAME_COUNT, 1, Q_LEN, 1536), generator=generator, dtype=torch.float32
    ) * 0.05).to(dtype=torch.bfloat16).contiguous()
    return cpu.to(device=device).contiguous()


def _native_context(NativeGemmaListenerLM: Any, lm: Any):
    return NativeGemmaListenerLM(
        lm,
        max_seq_len=MAX_SEQ_LEN,
        stream_q_len=Q_LEN,
        enable_cuda_graph=False,
        attention_backend=3,
        rmsnorm_backend=2,
    )


def _runtime_gate(native: Any, library_sha256: str) -> dict[str, Any]:
    info = native.native_runtime_info()
    expected = {
        "attention_backend_id": 3,
        "rmsnorm_backend_id": 2,
        "max_seq_len": MAX_SEQ_LEN,
        "library_sha256": library_sha256,
        "cuda_graph_enabled": False,
        "attention_implementation": "CUTLASS_ORDERED_TILES_V1",
    }
    mismatches = {key: (info.get(key), value) for key, value in expected.items() if info.get(key) != value}
    if mismatches:
        raise ProbeContractError(f"native runtime policy mismatch: {mismatches}")
    abi = int(native._lib.gemma4_listener_abi_version())
    if abi != 3:
        raise ProbeContractError(f"native ABI mismatch: {abi}")
    return info


def _record_native_cache(
    *,
    torch: Any,
    native: Any,
    cuda_memcpy: Any,
    case: str,
    position: int,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
) -> None:
    if int(native.current_position) != int(position):
        raise ProbeContractError(
            f"{case} cache checkpoint position {native.current_position} != {position}"
        )
    for layer in range(15):
        key, value = _copy_cached_kv_from_native(torch, native, layer, cuda_memcpy)
        if int(key.shape[2]) != position or int(value.shape[2]) != position:
            raise ProbeContractError(f"{case} layer {layer} cache shape disagrees with position {position}")
        _write_array(arrays, metadata, f"{case}.cache.position{position}.layer{layer:02d}.k", key)
        _write_array(arrays, metadata, f"{case}.cache.position{position}.layer{layer:02d}.v", value)


def _route(
    *,
    torch: Any,
    device: Any,
    native: Any,
    prefix_cache: Any,
    prefix_inputs: Mapping[str, Any],
    frames: Any,
    cuda_memcpy: Any,
    case_name: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    input_before = _hash_prefix_inputs(torch, {**prefix_inputs, "frames.q4": frames})
    source_before = _hash_prefix_source(torch, prefix_cache)
    rng_before = _rng_snapshot(torch)
    native.reset()
    if int(native.current_position) != 0:
        raise ProbeContractError(f"{case_name} reset did not clear the native cursor")
    native.import_cache_async(prefix_cache, current_position=PREFIX_TOKENS)
    _sync(torch, device)
    if int(native.current_position) != PREFIX_TOKENS:
        raise ProbeContractError(f"{case_name} import did not end at position 34")
    native.verify_output_health()
    outputs: list[np.ndarray] = []
    _record_native_cache(
        torch=torch, native=native, cuda_memcpy=cuda_memcpy,
        case=case_name, position=34, arrays=arrays, metadata=metadata,
    )
    for layer in range(15):
        for kind in ("k", "v"):
            source_name = f"prefix.source_cache.layer{layer:02d}.{kind}"
            imported_name = f"{case_name}.cache.position34.layer{layer:02d}.{kind}"
            if not np.array_equal(arrays[source_name], arrays[imported_name]):
                raise ProbeContractError(
                    f"{case_name} native import differs from exact ShapeStable source at layer {layer}.{kind}"
                )

    for frame_index in range(FRAME_COUNT):
        output = native.step_async(frames[frame_index])
        _sync(torch, device)
        native.verify_output_health()
        expected_position = PREFIX_TOKENS + (frame_index + 1) * Q_LEN
        if int(native.current_position) != expected_position:
            raise ProbeContractError(
                f"{case_name} frame {frame_index} cursor {native.current_position} != {expected_position}"
            )
        outputs.append(_tensor_array(output))
        if frame_index + 1 == CROSSING_FRAME_INDEX:
            if expected_position != CHECKPOINTS[1]:
                raise ProbeContractError("configured Q4 crossing frame does not land at position 514")
            _record_native_cache(
                torch=torch, native=native, cuda_memcpy=cuda_memcpy,
                case=case_name, position=CHECKPOINTS[1], arrays=arrays, metadata=metadata,
            )
    _sync(torch, device)
    if int(native.current_position) != CHECKPOINTS[2]:
        raise ProbeContractError("505-frame Q4 route did not end at position 2054")
    _record_native_cache(
        torch=torch, native=native, cuda_memcpy=cuda_memcpy,
        case=case_name, position=CHECKPOINTS[2], arrays=arrays, metadata=metadata,
    )
    output_array = np.stack(outputs, axis=0)
    output_name = f"{case_name}.output"
    arrays[output_name] = output_array
    metadata[output_name] = {
        "shape": list(output_array.shape),
        "dtype": "float32_from_bf16",
        "sha256": _array_sha256(output_array),
    }
    runtime_after = native.native_runtime_info()
    ordered = _sum_ordered_calls(runtime_after)
    if ordered <= 0:
        raise ProbeContractError(f"{case_name} failed to enter ordered attention beyond 2048")
    input_after = _hash_prefix_inputs(torch, {**prefix_inputs, "frames.q4": frames})
    source_after = _hash_prefix_source(torch, prefix_cache)
    rng_after = _rng_snapshot(torch)
    if input_before != input_after or source_before != source_after:
        raise ProbeContractError(f"{case_name} mutated the frozen prefix, K/V, or Q4 input tensors")
    if rng_before["sha256"] != rng_after["sha256"]:
        raise ProbeContractError(f"{case_name} advanced a random generator")
    return {
        "status": "PASS",
        "prefix_import_position": PREFIX_TOKENS,
        "prefix_import_cache_exact": True,
        "positions_observed": list(CHECKPOINTS),
        "position_after_decode": int(native.current_position),
        "ordered_calls_at_final_position": ordered,
        "input_sha256_before": input_before,
        "input_sha256_after": input_after,
        "rng_sha256_before": rng_before["sha256"],
        "rng_sha256_after": rng_after["sha256"],
        "native_runtime_after": runtime_after,
    }


def _run_candidate_routes(
    *,
    native_one: Any,
    native_two: Any,
    route_case: Callable[[str, Any], dict[str, Any]],
    cleanup_errors: list[str],
) -> dict[str, dict[str, Any]]:
    """Run A on context 1, then B and its reset/import repeat on context 2.

    Both candidate contexts remain resident together for the capacity portion
    of the probe.  This helper owns their cleanup so exceptional route exits
    cannot leak either context.
    """
    records: dict[str, dict[str, Any]] = {}
    try:
        if native_one is native_two:
            raise ProbeContractError("candidate route contexts must be distinct instances")
        context_by_case = {
            "fresh_a": native_one,
            "fresh_b": native_two,
            "same_context_reset_import": native_two,
        }
        for case_name in CASES:
            records[case_name] = route_case(case_name, context_by_case[case_name])
    finally:
        closed: set[int] = set()
        for context in (native_two, native_one):
            if id(context) in closed:
                continue
            closed.add(id(context))
            try:
                context.close()
            except Exception as close_error:
                cleanup_errors.append(f"{type(close_error).__name__}: {close_error}")
    return records


def run_worker(
    *,
    manifest_path: str | Path,
    role: str,
    report_path: str | Path,
    arrays_path: str | Path,
) -> dict[str, Any]:
    """Execute one role in one fresh process and preserve a report on failure."""
    manifest_file = Path(manifest_path).resolve()
    report_file = Path(report_path).resolve()
    arrays_file = Path(arrays_path).resolve()
    manifest = load_manifest(manifest_file)
    if role not in ("baseline", "candidate"):
        raise ProbeContractError(f"unknown worker role: {role!r}")
    record = manifest["libraries"][role]
    report: dict[str, Any] = {
        "schema": WORKER_SCHEMA,
        "role": role,
        "status": "RUNNING",
        "library_sha256": str(record["sha256"]).lower(),
        "source_manifest_sha256": str(record["source_manifest_sha256"]).lower(),
        "arrays_path": str(arrays_file),
        "arrays": {},
        "cases": {},
    }
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    torch: Any | None = None
    native_one: Any | None = None
    native_two: Any | None = None
    previous_dso = os.environ.get(NATIVE_LIBRARY_ENV)
    try:
        source_attestation = _verify_source_manifest(record)
        library_path = Path(str(record["path"]))
        if not library_path.is_file() or sha256_file(library_path).lower() != str(record["sha256"]).lower():
            raise ProbeContractError(f"{role} library path/hash does not match the pinned manifest")
        os.environ[NATIVE_LIBRARY_ENV] = str(library_path)
        import torch as torch_module
        torch = torch_module
        from duplex_kernel.listener import GemmaAudioTowerBackend
        from duplex_kernel.models.thinker import load_thinker
        from duplex_kernel.native_listener import NativeGemmaListenerLM

        cuda_runtime, cuda_runtime_path, cuda_runtime_sha256 = _load_cuda_runtime_for_cache_copy(torch)
        if not torch.cuda.is_available():
            raise ProbeContractError("production capacity worker requires CUDA; no CPU fallback")
        device = torch.device("cuda:0")
        torch.cuda.init()
        memory_before_model = _memory_snapshot(torch, device)
        thinker = load_thinker("/workspace", device=device, dtype=torch.bfloat16)
        lm = thinker.model.model.language_model
        backend = GemmaAudioTowerBackend(thinker, device=str(device), use_native_lm=False)
        torch.cuda.synchronize(device)
        memory_after_model = _memory_snapshot(torch, device)
        prefix_inputs, prefix_cache = _create_production_prefix(torch, lm, backend, device)
        frames = _prepare_frames(torch, device)
        torch.cuda.synchronize(device)
        prefix_source_sha256 = _hash_prefix_source(torch, prefix_cache)
        input_sha256 = _hash_prefix_inputs(torch, {**prefix_inputs, "frames.q4": frames})
        for name, tensor in prefix_inputs.items():
            _write_array(arrays, metadata, name, tensor)
        _write_array(arrays, metadata, "frames.q4", frames)
        for layer in range(15):
            entry = prefix_cache.layers[layer]
            _write_array(arrays, metadata, f"prefix.source_cache.layer{layer:02d}.k", entry.keys)
            _write_array(arrays, metadata, f"prefix.source_cache.layer{layer:02d}.v", entry.values)
        dims = {
            str(layer): int(512 if lm.config.layer_types[layer] == "full_attention" else 256)
            for layer in range(15)
        }
        if {int(layer): value for layer, value in dims.items()} != SOURCE_LAYER_HEAD_DIMS:
            raise ProbeContractError(f"source cache head dimensions differ from pinned Gemma topology: {dims}")

        # The candidate deliberately keeps two imported contexts alive at once.
        # This is a listener-context capacity measurement; no talker is loaded.
        memory_before_contexts = _memory_snapshot(torch, device)
        native_one = _native_context(NativeGemmaListenerLM, lm)
        runtime_initial = _runtime_gate(native_one, str(record["sha256"]).lower())
        memory_after_context1_create = _memory_snapshot(torch, device)
        native_one.import_cache_async(prefix_cache, current_position=PREFIX_TOKENS)
        _sync(torch, device)
        memory_after_context1_import = _memory_snapshot(torch, device)
        if native_one.current_position != PREFIX_TOKENS:
            raise ProbeContractError("first native context did not import canonical prefix")
        if role == "candidate":
            native_two = _native_context(NativeGemmaListenerLM, lm)
            _runtime_gate(native_two, str(record["sha256"]).lower())
            memory_after_context2_create = _memory_snapshot(torch, device)
            native_two.import_cache_async(prefix_cache, current_position=PREFIX_TOKENS)
            _sync(torch, device)
            memory_after_context2_import = _memory_snapshot(torch, device)
            if native_two.current_position != PREFIX_TOKENS:
                raise ProbeContractError("second candidate context did not import canonical prefix")
            context_report = {
                "context_count": 2,
                "both_contexts_imported_at": [int(native_one.current_position), int(native_two.current_position)],
                "memory_before_contexts": memory_before_contexts,
                "memory_after_context_1_create": memory_after_context1_create,
                "memory_after_context_1_import": memory_after_context1_import,
                "memory_after_context_2_create": memory_after_context2_create,
                "memory_after_context_2_import": memory_after_context2_import,
                "context_1_runtime": runtime_initial,
                "context_2_runtime": native_two.native_runtime_info(),
                "scope": "loaded Thinker plus audio tower and two native listener contexts; Talker not loaded",
            }
        else:
            context_report = {
                "context_count": 1,
                "both_contexts_imported_at": [int(native_one.current_position)],
                "memory_before_contexts": memory_before_contexts,
                "memory_after_context_1_create": memory_after_context1_create,
                "memory_after_context_1_import": memory_after_context1_import,
                "max_simultaneous_contexts": 1,
                "scope": "loaded Thinker plus audio tower and one native listener context at a time; Talker not loaded",
            }
        case_records: dict[str, Any] = {}
        if role == "candidate":
            first_context, second_context = native_one, native_two
            native_one = native_two = None

            def route_candidate_case(case_name: str, native: Any) -> dict[str, Any]:
                return _route(
                    torch=torch,
                    device=device,
                    native=native,
                    prefix_cache=prefix_cache,
                    prefix_inputs=prefix_inputs,
                    frames=frames,
                    cuda_memcpy=cuda_runtime.cudaMemcpy,
                    case_name=case_name,
                    arrays=arrays,
                    metadata=metadata,
                )

            case_records = _run_candidate_routes(
                native_one=first_context,
                native_two=second_context,
                route_case=route_candidate_case,
                cleanup_errors=report.setdefault("cleanup_errors", []),
            )
        else:
            case_records["fresh_a"] = _route(
                torch=torch,
                device=device,
                native=native_one,
                prefix_cache=prefix_cache,
                prefix_inputs=prefix_inputs,
                frames=frames,
                cuda_memcpy=cuda_runtime.cudaMemcpy,
                case_name="fresh_a",
                arrays=arrays,
                metadata=metadata,
            )
            native_one.close()
            native_one = None
            memory_after_context1_close = _memory_snapshot(torch, device)
            native_two = _native_context(NativeGemmaListenerLM, lm)
            runtime_second = _runtime_gate(native_two, str(record["sha256"]).lower())
            context_report.update({
                "memory_after_context_1_close": memory_after_context1_close,
                "memory_after_sequential_context_2_create": _memory_snapshot(torch, device),
                "context_2_runtime": runtime_second,
            })
            for case_name in ("fresh_b", "same_context_reset_import"):
                case_records[case_name] = _route(
                    torch=torch,
                    device=device,
                    native=native_two,
                    prefix_cache=prefix_cache,
                    prefix_inputs=prefix_inputs,
                    frames=frames,
                    cuda_memcpy=cuda_runtime.cudaMemcpy,
                    case_name=case_name,
                    arrays=arrays,
                    metadata=metadata,
                )
        report.update({
            "status": "PASS",
            "source_manifest": source_attestation,
            "source_manifest_sha256": source_attestation["sha256"],
            "source_snapshot_tree": source_attestation["source_root"],
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_device_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "memory_before_model": memory_before_model,
            "memory_after_model_and_production_prefix": memory_after_model,
            "cuda_runtime_for_cache_copy": {"path": cuda_runtime_path, "sha256": cuda_runtime_sha256},
            "model_checkpoint_files_sha256": runtime_initial["checkpoint_files_sha256"],
            "native_topology_manifest_sha256": runtime_initial["manifest_sha256"],
            "input_sha256": input_sha256,
            "prefix_source_sha256": prefix_source_sha256,
            "prefix_mode": "shape_stable_dynamic_cache_import",
            "native_prime_async_used": False,
            "q4_frames_source": "deterministic fixed synthetic BF16 Q4 tensors; not audio/dialogue evidence",
            "prefix_source": "canonical empty-text production multimodal placeholders and ShapeStable prefix path",
            "prefix_len": PREFIX_LEN,
            "frame_count": FRAME_COUNT,
            "features_per_frame": Q_LEN,
            "crossing_frame_index": CROSSING_FRAME_INDEX,
            "position_checkpoints": list(CHECKPOINTS),
            "layer_head_dims": dims,
            "tolerances": dict(DEFAULT_SPEC.tolerances),
            "abi_version": 3,
            "attention_backend_id": 3,
            "rmsnorm_backend_id": 2,
            "max_seq_len": MAX_SEQ_LEN,
            "stream_q_len": Q_LEN,
            "initial_runtime": runtime_initial,
            "context_capacity_probe": context_report,
            "cases": case_records,
            "arrays": metadata,
        })
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        for context in (native_two, native_one):
            if context is not None:
                try:
                    context.close()
                except Exception as close_error:
                    report.setdefault("cleanup_errors", []).append(f"{type(close_error).__name__}: {close_error}")
        if previous_dso is None:
            os.environ.pop(NATIVE_LIBRARY_ENV, None)
        else:
            os.environ[NATIVE_LIBRARY_ENV] = previous_dso
        arrays_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arrays_file, **arrays)
        report["arrays_path"] = str(arrays_file)
        report["arrays_file_sha256"] = sha256_file(arrays_file)
        report["arrays"] = metadata
        if report.get("status") == "PASS":
            try:
                validate_production_worker_report(report, expected_role=role)
            except Exception as validation_error:
                report["status"] = "FAIL"
                report["error"] = f"{type(validation_error).__name__}: {validation_error}"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(report_file, report)
    return report


def _child_environment(base: Mapping[str, str], record: Mapping[str, Any]) -> dict[str, str]:
    result = dict(base)
    result["HF_HUB_OFFLINE"] = "1"
    result["TRANSFORMERS_OFFLINE"] = "1"
    result["HF_HUB_DISABLE_TELEMETRY"] = "1"
    result[NATIVE_LIBRARY_ENV] = str(record["path"])
    return result


def run_all(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run baseline then candidate once; a baseline failure leaves candidate UNMEASURED."""
    manifest_file = Path(manifest_path).resolve()
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_file)
    workers: dict[str, Any] = {}
    python = sys.executable
    script = Path(__file__).resolve()
    for role in ("baseline", "candidate"):
        report_path = directory / f"{role}_production_prefix_report.json"
        arrays_path = directory / f"{role}_production_prefix_tensors.npz"
        log_path = directory / f"{role}_worker.log"
        command = [
            python, str(script), "worker", "--manifest", str(manifest_file),
            "--role", role, "--report", str(report_path), "--arrays", str(arrays_path),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env=_child_environment(os.environ, manifest["libraries"][role]),
            )
        workers[role] = {
            "report_path": str(report_path),
            "arrays_path": str(arrays_path),
            "log_path": str(log_path),
            "exit_code": completed.returncode,
        }
        if completed.returncode != 0:
            failure = {
                "schema": REPORT_SCHEMA,
                "status": "FAIL",
                "manifest_path": str(manifest_file),
                "manifest_sha256": sha256_file(manifest_file),
                "sequential_workers": workers,
                "candidate_status": "UNMEASURED" if role == "baseline" else "FAILED",
                "errors": [f"{role} worker exited {completed.returncode}; no automatic retry was attempted"],
            }
            atomic_write_json(directory / "production_prefix_full_report.json", failure)
            return failure
    baseline_report = json.loads(Path(workers["baseline"]["report_path"]).read_text(encoding="utf-8"))
    candidate_report = json.loads(Path(workers["candidate"]["report_path"]).read_text(encoding="utf-8"))
    result = assemble_production_report(
        manifest=manifest,
        baseline_report=baseline_report,
        baseline_arrays_path=workers["baseline"]["arrays_path"],
        candidate_report=candidate_report,
        candidate_arrays_path=workers["candidate"]["arrays_path"],
    )
    result.update({
        "manifest_path": str(manifest_file),
        "manifest_sha256": sha256_file(manifest_file),
        "sequential_workers": workers,
        "execution_order": ["baseline", "candidate"],
        "worker_concurrency": 1,
    })
    atomic_write_json(directory / "production_prefix_full_report.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("diagnose-production-prefix")
    run_parser.add_argument("--manifest", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--manifest", required=True, type=Path)
    worker_parser.add_argument("--role", required=True, choices=("baseline", "candidate"))
    worker_parser.add_argument("--report", required=True, type=Path)
    worker_parser.add_argument("--arrays", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "diagnose-production-prefix":
            result = run_all(args.manifest, args.output_dir)
        else:
            result = run_worker(
                manifest_path=args.manifest,
                role=args.role,
                report_path=args.report,
                arrays_path=args.arrays,
            )
        print(json.dumps({"status": result.get("status"), "report": result}, indent=2))
        return 0 if result.get("status") == "PASS" else 1
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc), "type": type(exc).__name__}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
