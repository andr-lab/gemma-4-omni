#!/usr/bin/env python3
"""Strict, fail-closed N=1/2/4 native listener parity probe.

Default invocation is plan-only and never imports PyTorch or probes CUDA. The
GPU path is intentionally opt-in and expects an already prepared, offline,
isolated RTX 3090 worker plus a safety-evidence sidecar. It pins the real
ShapeStable 34-token prefix -> native cache import route; it never calls
``prime_async``. This probe is numerical/runtime evidence, not dialogue or
end-to-end latency certification.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
import traceback
import uuid
from typing import Any, Mapping, Sequence

import numpy as np


SUPPORTED_BATCHES = (1, 2, 4)
LAYER_COUNT = 35
HIDDEN_SIZE = 1536
Q_PER_FRAME = 4
PREFIX_LEN = 34
MAX_SEQ_LEN = 16_384
MAX_SLIDING_WINDOW = 512
SLIDING_STAGE_CAPACITY = 802  # 512 + 291 - 1, from the native capacity contract.
TOPOLOGY_SHA256 = "53787ab382f05cc6b6c3ca7aa391cc863eb95403c08a827abb009215b2c96b60"
EXPECTED_TORCH_VERSION = "2.12.1+cu130"
CERTIFIED_REFERENCE_RELATIVE_PATH = Path(
    ".tmp/listener_sliding_capacity_candidate_20261011/native/"
    "gemma4_listener/libgemma4_listener.so"
)
CERTIFIED_REFERENCE_SHA256 = "0bf885240d63bf6581142613a6b1ded45c2290efe0e809041e4a6e22a6ed94eb"


def _failure_diagnostics(exc: BaseException, *, stage: str, torch_module: Any) -> dict[str, Any]:
    """Capture failure location and tensor layout metadata without copying data."""
    tensor_frames: list[dict[str, Any]] = []
    tb = exc.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        frame_tensors: list[dict[str, Any]] = []
        for name, value in frame.f_locals.items():
            if not isinstance(value, torch_module.Tensor):
                continue
            try:
                shape = tuple(int(size) for size in value.shape)
                strides = tuple(int(stride) for stride in value.stride())
                expected: list[int] = []
                running = 1
                for size in reversed(shape):
                    expected.append(running)
                    running *= size
                expected_strides = tuple(reversed(expected))
                frame_tensors.append({
                    "local_name": str(name),
                    "shape": list(shape),
                    "strides": list(strides),
                    "expected_row_major_strides": list(expected_strides),
                    "exact_row_major": strides == expected_strides,
                    "torch_is_contiguous": bool(value.is_contiguous()),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "storage_offset": int(value.storage_offset()),
                })
            except BaseException as metadata_exc:
                frame_tensors.append({
                    "local_name": str(name),
                    "metadata_error": f"{type(metadata_exc).__name__}: {metadata_exc}",
                })
        if frame_tensors:
            tensor_frames.append({
                "function": frame.f_code.co_name,
                "file": frame.f_code.co_filename,
                "line": int(tb.tb_lineno),
                "tensors": frame_tensors,
            })
        tb = tb.tb_next
    return {
        "stage": stage,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "tensor_locals": tensor_frames,
    }


class ExactParityError(AssertionError):
    """A shape, dtype, or raw element stream differs from the reference."""


@dataclass(frozen=True, slots=True)
class PhaseResult:
    batch_size: int
    passed: bool
    evidence_id: str


class ParityPhaseGate:
    """Allow N=1, then N=2, then N=4; any failure permanently closes the gate."""

    def __init__(self) -> None:
        self._results: list[PhaseResult] = []
        self._failed = False

    @property
    def results(self) -> tuple[PhaseResult, ...]:
        return tuple(self._results)

    @property
    def next_batch_size(self) -> int | None:
        if self._failed or len(self._results) == len(SUPPORTED_BATCHES):
            return None
        return SUPPORTED_BATCHES[len(self._results)]

    def record(self, batch_size: int, *, passed: bool, evidence_id: str) -> None:
        expected = self.next_batch_size
        if expected is None:
            raise RuntimeError("parity gate is closed; no additional batch size may run")
        if batch_size != expected:
            raise RuntimeError(f"parity phases must run in order; expected N={expected}")
        if not evidence_id.strip():
            raise ValueError("phase evidence_id must be non-empty")
        self._results.append(PhaseResult(batch_size, bool(passed), evidence_id))
        if not passed:
            self._failed = True


def assert_exact_array(name: str, expected: Any, actual: Any) -> None:
    """Require identical shape, dtype and every raw element (no tolerance)."""
    left = np.asarray(expected)
    right = np.asarray(actual)
    if left.shape != right.shape:
        raise ExactParityError(f"{name}: shape {left.shape} != {right.shape}")
    if left.dtype != right.dtype:
        raise ExactParityError(f"{name}: dtype {left.dtype} != {right.dtype}")
    if not np.array_equal(left, right):
        different = np.argwhere(left != right)
        first = tuple(int(value) for value in different[0]) if different.size else None
        raise ExactParityError(
            f"{name}: {len(different)} raw elements differ; first index={first}"
        )


def expected_projection_gemm_count(shared_kv_layers: Sequence[bool]) -> int:
    """Count the one pre-layer PLE GEMM plus each layer's dense projections.

    The native source issues 9 dense projections in a private-KV layer and 7
    when K/V are shared. The row count on each such GEMM must be N*Q.
    """
    if len(shared_kv_layers) != LAYER_COUNT:
        raise ValueError(f"expected exactly {LAYER_COUNT} layer sharing flags")
    return 1 + sum(7 if shared else 9 for shared in shared_kv_layers)


def assert_projection_batch_counters(
    before: Mapping[str, int],
    after: Mapping[str, int],
    *,
    batch_size: int,
    layer_count: int,
    shared_kv_layers: Sequence[bool],
) -> dict[str, int]:
    if batch_size not in SUPPORTED_BATCHES or layer_count != LAYER_COUNT:
        raise AssertionError("counter proof requires N in {1,2,4} and all 35 layers")
    names = (
        "forward_calls", "layers_executed", "projection_gemm_calls",
        "projection_rows", "session_rows", "output_generation",
    )
    delta = {name: int(after[name]) - int(before[name]) for name in names}
    projections = expected_projection_gemm_count(shared_kv_layers)
    expected = {
        "forward_calls": 1,
        "layers_executed": layer_count,
        "projection_gemm_calls": projections,
        "projection_rows": projections * batch_size * Q_PER_FRAME,
        "session_rows": batch_size,
        "output_generation": 1,
    }
    if delta != expected:
        raise AssertionError(f"native forward counters do not prove one batched pass: {delta} != {expected}")
    return delta


def assert_snapshot_exact(
    expected_hidden_bits: np.ndarray,
    actual_hidden_bits: np.ndarray,
    expected_kv: Mapping[int, tuple[np.ndarray, np.ndarray]],
    actual_kv: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    expected_state: Mapping[str, Any],
    actual_state: Mapping[str, Any],
) -> None:
    assert_exact_array("hidden", expected_hidden_bits, actual_hidden_bits)
    if dict(expected_state) != dict(actual_state):
        raise ExactParityError(f"native cursor/state differs: {actual_state} != {expected_state}")
    expected_layers = set(expected_kv)
    actual_layers = set(actual_kv)
    if actual_layers != expected_layers:
        raise ExactParityError(
            f"unique KV source-layer set differs: {sorted(actual_layers)} != {sorted(expected_layers)}"
        )
    for layer_idx in sorted(expected_layers):
        for label, expected, actual in zip(
            ("key", "value"), expected_kv[layer_idx], actual_kv[layer_idx], strict=True
        ):
            assert_exact_array(f"layer {layer_idx} {label}", expected, actual)


def _bf16_bits_as_float32(value: np.ndarray) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.uint16:
        raise TypeError(f"BF16 evidence must be uint16 raw bits, got {raw.dtype}")
    return (raw.astype(np.uint32) << 16).view(np.float32)


def _hash_native_weight_pointer(model: Any, pointer: int, torch_module: Any) -> dict[str, Any]:
    """Resolve a native weight pointer to its exact loaded parameter and hash bytes."""
    pointer = int(pointer)
    if pointer <= 0:
        raise ValueError("native weight pointer must be positive")
    matches = [
        (str(name), parameter)
        for name, parameter in model.named_parameters()
        if int(parameter.data_ptr()) == pointer
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"native weight pointer {pointer:#x} matched {len(matches)} loaded parameters"
        )
    name, parameter = matches[0]
    tensor = parameter.detach().contiguous()
    # PyTorch 2.12 documents Tensor.view(dtype) as a reinterpretation sharing
    # the same bytes; the following CPU copy is only for the one-shot digest.
    raw = tensor.view(torch_module.uint8).cpu().numpy().tobytes()
    return {
        "name": name,
        "data_ptr": pointer,
        "shape": [int(size) for size in parameter.shape],
        "dtype": str(parameter.dtype),
        "device": str(parameter.device),
        "nbytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _array_diff_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    left = np.asarray(expected)
    right = np.asarray(actual)
    metrics: dict[str, Any] = {
        "expected_shape": list(left.shape),
        "actual_shape": list(right.shape),
        "expected_dtype": str(left.dtype),
        "actual_dtype": str(right.dtype),
        "same_shape": left.shape == right.shape,
        "same_dtype": left.dtype == right.dtype,
        "raw_mismatch_count": None,
        "first_mismatch_indices": [],
        "max_abs_diff": None,
        "nonfinite_pair_count": None,
    }
    if left.shape != right.shape or left.dtype != right.dtype:
        return metrics
    different = np.argwhere(left != right)
    metrics["raw_mismatch_count"] = int(len(different))
    metrics["first_mismatch_indices"] = [
        [int(index) for index in row] for row in different[:10]
    ]
    if left.dtype == np.uint16:
        left_float = _bf16_bits_as_float32(left).astype(np.float64)
        right_float = _bf16_bits_as_float32(right).astype(np.float64)
        finite_pairs = np.isfinite(left_float) & np.isfinite(right_float)
        metrics["nonfinite_pair_count"] = int(np.count_nonzero(~finite_pairs))
        if bool(np.any(finite_pairs)):
            metrics["max_abs_diff"] = float(
                np.max(np.abs(left_float[finite_pairs] - right_float[finite_pairs]))
            )
        elif left.size == 0:
            metrics["max_abs_diff"] = 0.0
    return metrics


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot encode {type(value).__name__} in parity evidence JSON")


def _persist_evidence_checkpoint(
    *, output_path: Path, artifact_path: Path,
    tensor_artifacts: Mapping[str, np.ndarray], report: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temp_artifact = artifact_path.with_name(artifact_path.name + ".pending")
    with temp_artifact.open("wb") as handle:
        np.savez_compressed(handle, **tensor_artifacts)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_artifact, artifact_path)
    report["tensor_artifact"] = {
        "path": str(artifact_path),
        "sha256": _sha256_file(artifact_path),
        "arrays": len(tensor_artifacts),
    }
    temp_json = output_path.with_name(output_path.name + ".pending")
    encoded = json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n"
    with temp_json.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_json, output_path)


def _record_native_numerical_probe_stage(
    *,
    output_path: Path,
    report: dict[str, Any],
    tensor_artifacts: dict[str, np.ndarray],
    stage: str,
    metadata: Mapping[str, Any],
    raw_arrays: Mapping[str, bytes],
    comparisons: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    """Persist every raw diagnostic row before enforcing any strict comparison."""
    stage = str(stage).lower()
    if stage not in {"rms", "qproj"}:
        raise ValueError("native numerical stage must be 'rms' or 'qproj'")
    prefix = f"N2/numerical/{stage}"
    expected_shapes: dict[str, tuple[int, int]]
    if stage == "rms":
        expected_shapes = {name: (8, 1536) for name in ("input_bits", "batched_bits", "split_bits")}
        required_names = set(expected_shapes)
    else:
        q_dim = int(metadata.get("q_dim", 0))
        expected_shapes = {
            "input_bits": (8, 1536),
            "actual_bits": (8, q_dim),
            "batched_bits": (8, q_dim),
            "split_bits": (8, q_dim),
        }
        required_names = set(expected_shapes)

    decoded: dict[str, np.ndarray] = {}
    raw_sha256: dict[str, str] = {}
    for name, value in raw_arrays.items():
        raw = bytes(value)
        raw_sha256[str(name)] = hashlib.sha256(raw).hexdigest()
        if len(raw) % 2:
            decoded[str(name)] = np.frombuffer(raw, dtype=np.uint8).copy()
            continue
        flat = np.frombuffer(raw, dtype="<u2").copy()
        shape = expected_shapes.get(str(name))
        if shape and flat.size == shape[0] * shape[1]:
            decoded[str(name)] = flat.reshape(shape)
        else:
            decoded[str(name)] = flat
    for name, array in decoded.items():
        tensor_artifacts[f"{prefix}/{name}"] = array.copy()

    record: dict[str, Any] = {
        "stage": stage,
        "status": "EVIDENCE_SAVED",
        "metadata": dict(metadata),
        "raw_sha256": raw_sha256,
        "arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype), "elements": int(array.size)}
            for name, array in decoded.items()
        },
        "comparisons": {},
    }
    for label, expected_name, actual_name in comparisons:
        expected = decoded.get(expected_name)
        actual = decoded.get(actual_name)
        if expected is None or actual is None:
            record["comparisons"][label] = {"error": "comparison input is absent"}
        else:
            record["comparisons"][label] = _array_diff_metrics(expected, actual)
    report.setdefault("native_numerical_diagnostics", []).append(record)
    _persist_evidence_checkpoint(
        output_path=output_path,
        artifact_path=output_path.with_suffix(".tensors.npz"),
        tensor_artifacts=tensor_artifacts,
        report=report,
    )

    failure: str | None = None
    missing_names = required_names - set(decoded)
    if missing_names:
        failure = f"missing raw diagnostic arrays: {sorted(missing_names)}"
    else:
        for name, shape in expected_shapes.items():
            if decoded[name].shape != shape or decoded[name].dtype != np.dtype("<u2"):
                failure = (
                    f"{name} has shape/dtype {decoded[name].shape}/{decoded[name].dtype}; "
                    f"expected {shape}/uint16"
                )
                break
    if failure is None:
        if stage == "rms":
            expected_metadata = {
                "row_count": 8,
                "rms_dim": 1536,
                "rms_backend_id": 2,
                "rms_batched_block_x": 64,
                "rms_batched_block_y": 8,
                "rms_reference_block_x": 128,
                "rms_reference_block_y": 4,
            }
        else:
            q_dim = int(metadata.get("q_dim", 0))
            expected_metadata = {
                "row_count": 8,
                "rms_dim": 1536,
                "rms_backend_id": 2,
                "q_batched_m": 8,
                "q_split_m": 4,
                "q_k": 1536,
                "q_lda": 1536,
                "q_ldb": 1536,
                "q_ldc": q_dim,
                "cublas_stream_matches_listener": 1,
            }
        metadata_mismatch = {
            name: (metadata.get(name), expected)
            for name, expected in expected_metadata.items()
            if metadata.get(name) != expected
        }
        if not int(metadata.get("forward_generation", 0)):
            metadata_mismatch["forward_generation"] = (metadata.get("forward_generation"), "positive")
        if not int(metadata.get("rms_weight_device_ptr", 0)):
            metadata_mismatch["rms_weight_device_ptr"] = (metadata.get("rms_weight_device_ptr"), "positive")
        if stage == "qproj" and not int(metadata.get("q_weight_device_ptr", 0)):
            metadata_mismatch["q_weight_device_ptr"] = (metadata.get("q_weight_device_ptr"), "positive")
        pin_key = "rms_weight_device_ptr" if stage == "rms" else "q_weight_device_ptr"
        weight_pin = metadata.get("native_weight_pin")
        if not isinstance(weight_pin, Mapping):
            metadata_mismatch["native_weight_pin"] = (weight_pin, "loaded parameter pin")
        else:
            if int(weight_pin.get("data_ptr", 0)) != int(metadata.get(pin_key, 0)):
                metadata_mismatch["native_weight_pin.data_ptr"] = (
                    weight_pin.get("data_ptr"), metadata.get(pin_key),
                )
            if not re.fullmatch(r"[0-9a-f]{64}", str(weight_pin.get("sha256", ""))):
                metadata_mismatch["native_weight_pin.sha256"] = (weight_pin.get("sha256"), "64 lowercase hex")
            if not str(weight_pin.get("name", "")) or not int(weight_pin.get("nbytes", 0)):
                metadata_mismatch["native_weight_pin.parameter"] = weight_pin
        if stage == "qproj" and int(metadata.get("listener_stream_handle", 0)) == 0:
            metadata_mismatch["listener_stream_handle"] = (metadata.get("listener_stream_handle"), "positive")
        if stage == "qproj" and any(
            int(metadata.get(name, 0)) not in (0, 1)
            for name in ("q4_gemv4_opt_in", "q4_gemv4_split0_eligible", "q4_gemv4_split1_eligible")
        ):
            metadata_mismatch["q4_dispatch"] = "flags must be 0/1"
        if metadata_mismatch:
            failure = f"native diagnostic metadata mismatch: {metadata_mismatch}"

    if failure is None:
        try:
            for label, expected_name, actual_name in comparisons:
                if expected_name not in decoded or actual_name not in decoded:
                    raise ExactParityError(f"{label}: diagnostic comparison array is missing")
                assert_exact_array(label, decoded[expected_name], decoded[actual_name])
        except ExactParityError as exc:
            failure = f"{type(exc).__name__}: {exc}"

    if failure is not None:
        record["status"] = "NO-GO"
        record["error"] = failure
        report["status"] = "NO-GO"
        _persist_evidence_checkpoint(
            output_path=output_path,
            artifact_path=output_path.with_suffix(".tensors.npz"),
            tensor_artifacts=tensor_artifacts,
            report=report,
        )
        raise ExactParityError(f"N2 {stage} numerical diagnostic: {failure}")

    record["status"] = "PASS"
    _persist_evidence_checkpoint(
        output_path=output_path,
        artifact_path=output_path.with_suffix(".tensors.npz"),
        tensor_artifacts=tensor_artifacts,
        report=report,
    )
    return record


def _record_native_operator_trace(
    *,
    output_path: Path,
    report: dict[str, Any],
    tensor_artifacts: dict[str, np.ndarray],
    records: Sequence[Mapping[str, Any]],
    stage_names: Sequence[str],
    attention_kind: int,
    row_descriptors: Sequence[Mapping[str, Any]],
    model: Any,
    torch_module: Any,
    runtime_signature: Any,
) -> dict[str, Any]:
    """Archive every N=2 operator boundary before enforcing exact parity."""
    if len(records) != len(stage_names):
        raise ExactParityError("native operator trace returned an incomplete stage table")
    stage_reports: list[dict[str, Any]] = []
    first_mismatch: dict[str, Any] | None = None
    prefix = "N2/operator_trace"
    for stage_id, record in enumerate(records):
        metadata = dict(record["metadata"])
        raw_arrays = dict(record["raw"])
        stage_name = str(stage_names[stage_id])
        if metadata.get("stage_name") != stage_name or metadata.get("stage_id") != stage_id:
            raise ExactParityError(f"native operator trace stage ordering changed at {stage_id}")
        metadata["row_descriptors"] = [dict(row) for row in row_descriptors]
        metadata["runtime_signature"] = {
            "model_config_sha256": runtime_signature.model_config_sha256,
            "checkpoint_sha256": runtime_signature.checkpoint_sha256,
            "topology_sha256": runtime_signature.topology_sha256,
            "native_library_sha256": runtime_signature.native_library_sha256,
            "native_abi_version": runtime_signature.native_abi_version,
            "batch_api_version": runtime_signature.batch_api_version,
            "attention_backend_id": runtime_signature.attention_backend_id,
            "rmsnorm_backend_id": runtime_signature.rmsnorm_backend_id,
            "max_seq_len": runtime_signature.max_seq_len,
            "stream_q_len": runtime_signature.stream_q_len,
        }
        weight_pointer = int(metadata.get("weight_device_ptr", 0))
        weight_pin_error: str | None = None
        if weight_pointer:
            try:
                metadata["native_weight_pin"] = _hash_native_weight_pointer(
                    model, weight_pointer, torch_module,
                )
            except Exception as exc:
                weight_pin_error = f"{type(exc).__name__}: {exc}"
                metadata["native_weight_pin_error"] = weight_pin_error
        array_specs = (
            ("input0_bits", "input0_rows", "input0_cols", "input0_elements"),
            ("input1_bits", "input1_rows", "input1_cols", "input1_elements"),
            ("input2_bits", "input2_rows", "input2_cols", "input2_elements"),
            ("actual_bits", "actual_rows", "actual_cols", "actual_elements"),
            ("reference_bits", "reference_rows", "reference_cols", "reference_elements"),
        )
        decoded: dict[str, np.ndarray] = {}
        array_report: dict[str, Any] = {}
        raw_hashes: dict[str, str] = {}
        raw_bytes: dict[str, int] = {}
        for raw_name, rows_key, cols_key, elements_key in array_specs:
            raw = bytes(raw_arrays.get(raw_name, b""))
            rows = int(metadata[rows_key])
            cols = int(metadata[cols_key])
            elements = int(metadata[elements_key])
            expected_bytes = elements * 2
            raw_hashes[raw_name] = hashlib.sha256(raw).hexdigest()
            raw_bytes[raw_name] = len(raw)
            if len(raw) != expected_bytes:
                raise ExactParityError(
                    f"{stage_name}:{raw_name} raw byte count {len(raw)} != {expected_bytes}"
                )
            if elements:
                if rows <= 0 or cols <= 0 or rows * cols != elements:
                    raise ExactParityError(f"{stage_name}:{raw_name} metadata shape is inconsistent")
                array = np.frombuffer(raw, dtype="<u2").copy().reshape(rows, cols)
            else:
                if rows or cols:
                    raise ExactParityError(f"{stage_name}:{raw_name} empty shape has nonzero dimensions")
                array = np.empty((0,), dtype=np.uint16)
            decoded[raw_name] = array
            array_report[raw_name] = {
                "shape": list(array.shape), "dtype": str(array.dtype),
                "elements": int(array.size), "raw_bytes": len(raw),
            }
            tensor_artifacts[f"{prefix}/{stage_name}/{raw_name}"] = array.copy()

        attention = int(metadata["kind"]) == int(attention_kind)
        comparison: dict[str, Any]
        if attention:
            comparison = {
                "status": "DEFERRED",
                "reason": "reference attention recomputation is omitted until scratch isolation is proven",
            }
        else:
            comparison = _array_diff_metrics(decoded["reference_bits"], decoded["actual_bits"])
            comparison["status"] = (
                "BITWISE_EXACT" if comparison.get("raw_mismatch_count") == 0 else "MISMATCH"
            )
            if first_mismatch is None and comparison["status"] != "BITWISE_EXACT":
                first_mismatch = {"stage_name": stage_name, **comparison}
        stage_reports.append({
            "stage_id": stage_id,
            "stage_name": stage_name,
            "metadata": metadata,
            "array_sha256": raw_hashes,
            "array_bytes": raw_bytes,
            "arrays": array_report,
            "comparison": comparison,
            "weight_pin_error": weight_pin_error,
        })

    trace_report = {
        "status": "MISMATCH" if first_mismatch else "BITWISE_EXACT_WITH_ATTENTION_DEFERRED",
        "trace_api_version": 1,
        "stage_count": len(stage_reports),
        "diagnostic_row_count": 8,
        "row_mapping": "manifest order; N=2 sessions x Q=4 tokens, no padding rows",
        "stage_comparisons": stage_reports,
        "first_mismatch": first_mismatch,
        "attention_reference": "DEFERRED",
    }
    report["native_operator_trace"] = trace_report
    _persist_evidence_checkpoint(
        output_path=output_path,
        artifact_path=output_path.with_suffix(".tensors.npz"),
        tensor_artifacts=tensor_artifacts,
        report=report,
    )
    if first_mismatch is not None:
        raise ExactParityError(
            "native operator trace first bitwise mismatch at "
            f"{first_mismatch['stage_name']}: "
            f"{first_mismatch.get('raw_mismatch_count')} of "
            f"{first_mismatch.get('actual_shape')} output values differ"
        )
    pin_failures = [item for item in stage_reports if item["weight_pin_error"]]
    if pin_failures:
        raise ExactParityError(
            "native operator trace raw arrays were archived, but a native weight pointer "
            f"could not be pinned: {pin_failures[0]['stage_name']} "
            f"{pin_failures[0]['weight_pin_error']}"
        )
    return trace_report


def _assert_snapshot_with_evidence(
    *, output_path: Path, tensor_artifacts: dict[str, np.ndarray],
    report: dict[str, Any], comparison_id: str, row_metadata: Mapping[str, Any],
    expected_hidden_bits: np.ndarray, actual_hidden_bits: np.ndarray,
    expected_kv: Mapping[int, tuple[np.ndarray, np.ndarray]],
    actual_kv: Mapping[int, tuple[np.ndarray, np.ndarray]],
    expected_state: Mapping[str, Any], actual_state: Mapping[str, Any],
    counters: Mapping[str, Any],
) -> None:
    """Persist both raw snapshots and diagnostics before enforcing exact parity."""
    artifact_path = output_path.with_suffix(".tensors.npz")
    copied_expected_hidden = np.asarray(expected_hidden_bits).copy()
    copied_actual_hidden = np.asarray(actual_hidden_bits).copy()
    copied_expected_kv = {
        int(layer): (np.asarray(pair[0]).copy(), np.asarray(pair[1]).copy())
        for layer, pair in expected_kv.items()
    }
    copied_actual_kv = {
        int(layer): (np.asarray(pair[0]).copy(), np.asarray(pair[1]).copy())
        for layer, pair in actual_kv.items()
    }
    prefix = comparison_id.strip("/")
    names: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "hidden": (copied_expected_hidden, copied_actual_hidden),
    }
    for layer in sorted(set(copied_expected_kv) | set(copied_actual_kv)):
        expected_pair = copied_expected_kv.get(layer)
        actual_pair = copied_actual_kv.get(layer)
        for label, offset in (("key", 0), ("value", 1)):
            if expected_pair is not None:
                tensor_artifacts[f"{prefix}/expected/layer{layer}/{label}_bits"] = expected_pair[offset]
            if actual_pair is not None:
                tensor_artifacts[f"{prefix}/actual/layer{layer}/{label}_bits"] = actual_pair[offset]
            if expected_pair is not None and actual_pair is not None:
                names[f"layer{layer}_{label}"] = (expected_pair[offset], actual_pair[offset])
    tensor_artifacts[f"{prefix}/expected_hidden_bits"] = copied_expected_hidden
    tensor_artifacts[f"{prefix}/actual_hidden_bits"] = copied_actual_hidden
    metrics = {
        name: _array_diff_metrics(expected, actual)
        for name, (expected, actual) in names.items()
    }
    record: dict[str, Any] = {
        "comparison_id": comparison_id,
        "status": "EVIDENCE_SAVED",
        "row": dict(row_metadata),
        "counters": dict(counters),
        "expected_state": dict(expected_state),
        "actual_state": dict(actual_state),
        "state_equal": dict(expected_state) == dict(actual_state),
        "expected_kv_layers": sorted(copied_expected_kv),
        "actual_kv_layers": sorted(copied_actual_kv),
        "kv_layer_set_equal": set(copied_expected_kv) == set(copied_actual_kv),
        "arrays": metrics,
    }
    report.setdefault("snapshot_comparisons", []).append(record)
    _persist_evidence_checkpoint(
        output_path=output_path, artifact_path=artifact_path,
        tensor_artifacts=tensor_artifacts, report=report,
    )
    try:
        assert_snapshot_exact(
            copied_expected_hidden, copied_actual_hidden,
            copied_expected_kv, copied_actual_kv,
            expected_state=expected_state, actual_state=actual_state,
        )
    except ExactParityError as exc:
        record["status"] = "NO-GO"
        record["error"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "NO-GO"
        _persist_evidence_checkpoint(
            output_path=output_path, artifact_path=artifact_path,
            tensor_artifacts=tensor_artifacts, report=report,
        )
        detail = next((
            f"{name} metrics={value}"
            for name, value in metrics.items()
            if value.get("raw_mismatch_count") not in (None, 0)
            or not value.get("same_shape") or not value.get("same_dtype")
        ), "no tensor mismatch metrics; state or source-layer metadata differs")
        raise ExactParityError(f"{comparison_id}: {exc}; {detail}") from exc
    record["status"] = "PASS"
    _persist_evidence_checkpoint(
        output_path=output_path, artifact_path=artifact_path,
        tensor_artifacts=tensor_artifacts, report=report,
    )


def _baseline_reference_path(workspace: Path) -> Path:
    return workspace / CERTIFIED_REFERENCE_RELATIVE_PATH


def assert_distinct_input_rows(pcm_rows: Sequence[np.ndarray], soft_rows_bits: Sequence[np.ndarray]) -> None:
    if len(pcm_rows) != len(soft_rows_bits) or len(pcm_rows) not in SUPPORTED_BATCHES:
        raise AssertionError("PCM and audio-soft rows must match an exact supported N")
    pcm_hashes = [hashlib.sha256(np.ascontiguousarray(row, dtype="<f4").tobytes()).hexdigest()
                  for row in pcm_rows]
    if len(set(pcm_hashes)) != len(pcm_hashes):
        raise AssertionError("batch rows contain duplicated PCM histories")
    hashes = [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
              for row in soft_rows_bits]
    if len(set(hashes)) != len(hashes):
        raise AssertionError("batch rows contain duplicated audio-soft-token histories")


def validate_gpu_safety_evidence(document: Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Validate the external GPU-owner/serving snapshot handoff before runtime init."""
    if document.get("schema_version") != 1:
        raise ValueError("GPU safety evidence schema_version must be 1")
    serving = document.get("serving_container")
    if not isinstance(serving, Mapping):
        raise ValueError("GPU safety evidence lacks serving_container identity")
    container_id = str(serving.get("container_id", ""))
    snapshot_id = str(serving.get("snapshot_id", ""))
    if not re.fullmatch(r"[0-9a-fA-F]{12,64}", container_id) or not snapshot_id.strip():
        raise ValueError("serving container id and recoverable snapshot id are required")
    if serving.get("status") != "running":
        raise ValueError("serving container must be recorded running before the probe handoff")
    owners = document.get("gpu_owners")
    if not isinstance(owners, list) or not owners:
        raise ValueError("GPU owner inventory is missing")
    if any(not isinstance(owner, Mapping) or owner.get("known") is not True
           or not str(owner.get("container_id", "")).strip() for owner in owners):
        raise ValueError("unknown GPU owners block the probe")
    if not any(str(owner.get("container_id", "")) == container_id for owner in owners):
        raise ValueError("serving container is absent from the device owner inventory")
    if str(document.get("planned_pause_container_id", "")) != container_id:
        raise ValueError("only the exact snapshotted serving container may be paused")
    if str(document.get("planned_restart_container_id", "")) != container_id:
        raise ValueError("recovery must restart the same serving container id")

    now_utc = now or datetime.now(timezone.utc)
    snapshot_at = _parse_utc(serving.get("snapshot_at_utc"), "snapshot_at_utc")
    checks = document.get("idle_health_checks")
    if not isinstance(checks, list) or len(checks) != 2:
        raise ValueError("exactly two fresh idle health checks are required")
    check_times: list[datetime] = []
    for item in checks:
        if not isinstance(item, Mapping) or item.get("status") != "healthy" \
                or item.get("active_requests") != 0:
            raise ValueError("each health check must be healthy and idle")
        checked_at = _parse_utc(item.get("checked_at_utc"), "checked_at_utc")
        age = (now_utc - checked_at).total_seconds()
        if age < 0 or age > 60 or checked_at < snapshot_at:
            raise ValueError("health checks must be after the snapshot and no older than 60 seconds")
        check_times.append(checked_at)
    if check_times[0] == check_times[1]:
        raise ValueError("the two idle health checks must be distinct observations")
    return {
        "serving_container_id": container_id,
        "snapshot_id": snapshot_id,
        "snapshot_at_utc": snapshot_at.isoformat(),
        "idle_health_checks": [value.isoformat() for value in check_times],
        "gpu_owners_known": len(owners),
    }


def _parse_utc(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    workspace = PurePosixPath(args.workspace)
    baseline_path = PurePosixPath(args.baseline_dso) if args.baseline_dso else (
        workspace / CERTIFIED_REFERENCE_RELATIVE_PATH.as_posix()
    )
    candidate_path = PurePosixPath(args.candidate_dso) if args.candidate_dso else (
        workspace / ".tmp/native_listener_microbatch_candidate_20261012/"
        "native/gemma4_listener_microbatch/libgemma4_listener_microbatch.so"
    )
    results_path = PurePosixPath(args.output_json) if args.output_json else (
        workspace / "validation/kernel_gates/native_listener_microbatch_parity.json"
    )
    return {
        "mode": "plan_only",
        "gpu_initialized": False,
        "batch_order": list(SUPPORTED_BATCHES),
        "stop_on_first_failure": True,
        "prefix": {"source": "ShapeStableGemmaListenerLM DynamicCache import",
                   "valid_tokens": PREFIX_LEN, "native_prime_async": False},
        "native_runtime": {
            "max_seq_len": MAX_SEQ_LEN,
            "stream_q_len": Q_PER_FRAME,
            "layer_count": LAYER_COUNT,
            "topology_sha256": TOPOLOGY_SHA256,
            "sliding_window": MAX_SLIDING_WINDOW,
            "sliding_staging_capacity": SLIDING_STAGE_CAPACITY,
            "backend_ids": {"attention": 3, "rmsnorm": 2},
        },
        "expected_artifacts": {
            "baseline_dso": str(baseline_path),
            "baseline_dso_sha256": CERTIFIED_REFERENCE_SHA256,
            "baseline_role": "independently certified capacity reference; may differ from managed serving runtime",
            "candidate_dso": str(candidate_path),
            "results": str(results_path),
        },
        "numerical_policy": "bitwise BF16 equality for every output element and all unique source-layer K/V rows",
        "latency_claim": "none; policy/frontend end-to-end p95 remains a separate gate",
        "lifecycle_checks": ["stale/cancelled manifest rejected before FFI and counters unchanged",
                             "retire then fresh-generation prefix re-import",
                             "rollover cursor/KV snapshot at an isolated test boundary"],
        "limitations": ["GPU execution requires separate root GO",
                        "runner does not stop, restart, or mutate serving containers"],
    }


def _install_overlay_import_path(
    workspace: str | Path,
    *,
    overlay_root: str | Path | None = None,
):
    """Load the isolated probe package first and its canonical sibling package second.

    The overlay's ``duplex_kernel/__init__.py`` explicitly appends the canonical
    package path. This keeps the pinned microbatch modules in the overlay while
    resolving ordinary Gemma/listener/model modules from the repository.
    """
    import importlib

    workspace_path = Path(workspace).resolve()
    overlay_path = Path(overlay_root).resolve() if overlay_root is not None else (
        Path(__file__).resolve().parents[2]
    )
    overlay_package = overlay_path / "duplex_kernel"
    canonical_package = workspace_path / "duplex_kernel"
    if not (overlay_package / "__init__.py").is_file():
        raise RuntimeError(f"isolated probe package is missing: {overlay_package / '__init__.py'}")
    if not canonical_package.is_dir():
        raise RuntimeError(f"canonical duplex package is missing: {canonical_package}")

    def normalized(value: str) -> str:
        try:
            return str(Path(value or ".").resolve())
        except OSError:
            return str(Path(value or "."))

    remove = {normalized(str(overlay_path)), normalized(str(workspace_path))}
    sys.path[:] = [item for item in sys.path if normalized(item) not in remove]
    sys.path.insert(0, str(workspace_path))
    if overlay_path != workspace_path:
        sys.path.insert(0, str(overlay_path))

    package = importlib.import_module("duplex_kernel")
    package_file = Path(str(getattr(package, "__file__", ""))).resolve()
    expected_package_file = (overlay_package / "__init__.py").resolve()
    if package_file != expected_package_file:
        raise RuntimeError(
            "duplex_kernel resolved from an unexpected package root: "
            f"{package_file} != {expected_package_file}"
        )
    search_roots = {normalized(str(Path(item).resolve())) for item in package.__path__}
    if overlay_path != workspace_path and normalized(str(canonical_package)) not in search_roots:
        raise RuntimeError(
            f"overlay package did not expose the canonical module directory: {sorted(search_roots)}"
        )
    return package


def _import_gpu_path_runtime(workspace: str | Path) -> dict[str, Any]:
    """Import the same modules and symbols used by the GPU path after path pinning."""
    _install_overlay_import_path(workspace)
    from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT
    from duplex_kernel.listener import GemmaAudioTowerBackend
    from duplex_kernel.listener_microbatch import ListenerBatch as FrontendListenerBatch
    from duplex_kernel.listener_microbatch_frontend import ListenerMicrobatchAudioFrontend
    from duplex_kernel.microbatch_contract import (
        BatchRowDescriptor, ListenerBatch, ListenerBatchManifest, ListenerFrame,
        ListenerSessionState, RuntimeSignature,
    )
    from duplex_kernel.models.thinker import load_thinker
    from duplex_kernel.native_listener import NativeGemmaListenerLM
    from duplex_kernel.native_listener_microbatch import (
        NativeGemmaListenerMicrobatch, TRACE_KIND_ATTENTION, TRACE_STAGE_NAMES,
    )
    from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

    if FrontendListenerBatch is not ListenerBatch:
        raise RuntimeError("frontend and probe resolved duplicate ListenerBatch classes")
    return {
        "AUDIO_START_AFTER_TEXT": AUDIO_START_AFTER_TEXT,
        "GemmaAudioTowerBackend": GemmaAudioTowerBackend,
        "ListenerMicrobatchAudioFrontend": ListenerMicrobatchAudioFrontend,
        "BatchRowDescriptor": BatchRowDescriptor,
        "ListenerBatch": ListenerBatch,
        "ListenerBatchManifest": ListenerBatchManifest,
        "ListenerFrame": ListenerFrame,
        "ListenerSessionState": ListenerSessionState,
        "RuntimeSignature": RuntimeSignature,
        "load_thinker": load_thinker,
        "NativeGemmaListenerLM": NativeGemmaListenerLM,
        "NativeGemmaListenerMicrobatch": NativeGemmaListenerMicrobatch,
        "TRACE_KIND_ATTENTION": TRACE_KIND_ATTENTION,
        "TRACE_STAGE_NAMES": TRACE_STAGE_NAMES,
        "ShapeStableGemmaListenerLM": ShapeStableGemmaListenerLM,
        "create_masks_for_generate": create_masks_for_generate,
        "_modules": {
            name: sys.modules[name] for name in (
                "duplex_kernel", "duplex_kernel.gemma_representation", "duplex_kernel.listener",
                "duplex_kernel.listener_microbatch", "duplex_kernel.listener_microbatch_frontend",
                "duplex_kernel.microbatch_contract", "duplex_kernel.models.thinker",
                "duplex_kernel.native_listener", "duplex_kernel.native_listener_microbatch",
                "duplex_kernel.shape_stable_listener", "transformers.models.gemma4.modeling_gemma4",
            )
        },
    }


def _preflight_runtime(args: argparse.Namespace) -> int:
    """CPU-only same-image import/source preflight; never probes CUDA availability."""
    workspace = Path(args.workspace).resolve(strict=True)
    if not args.output_json:
        raise SystemExit("--preflight-runtime requires --output-json")
    if not args.baseline_sha256 or not args.candidate_sha256 or not args.model_config_sha256:
        raise SystemExit("--preflight-runtime requires baseline, candidate, and config SHA pins")
    baseline = Path(args.baseline_dso or _baseline_reference_path(workspace)).resolve(strict=True)
    candidate = Path(args.candidate_dso or workspace / ".tmp/native_listener_microbatch_candidate_20261012/"
                     "native/gemma4_listener_microbatch/libgemma4_listener_microbatch.so").resolve(strict=True)
    for path, expected in ((baseline, args.baseline_sha256), (candidate, args.candidate_sha256)):
        actual = _sha256_file(path)
        if actual != str(expected).lower():
            raise RuntimeError(f"preflight DSO SHA-256 mismatch for {path}: {actual}")
    config_path = workspace / ".hf_cache/models--google--gemma-4-E2B-it/full/config.json"
    if not config_path.is_file() or _sha256_file(config_path) != str(args.model_config_sha256).lower():
        raise RuntimeError("preflight local model config is absent or differs from the requested SHA pin")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeError(f"expected PyTorch {EXPECTED_TORCH_VERSION}, got {torch.__version__}")
    initialized_before = bool(torch.cuda.is_initialized())
    if initialized_before:
        raise RuntimeError("CPU runtime preflight began with CUDA already initialized")
    runtime = _import_gpu_path_runtime(workspace)
    initialized_after = bool(torch.cuda.is_initialized())
    if initialized_after:
        raise RuntimeError("runtime imports initialized CUDA during CPU preflight")
    if runtime["AUDIO_START_AFTER_TEXT"] != PREFIX_LEN:
        raise RuntimeError(f"canonical audio prefix changed: {runtime['AUDIO_START_AFTER_TEXT']}")

    modules = runtime["_modules"]
    overlay_root = Path(__file__).resolve().parents[2]
    overlay_module_names = {
        "duplex_kernel", "duplex_kernel.listener_microbatch",
        "duplex_kernel.listener_microbatch_frontend", "duplex_kernel.microbatch_contract",
        "duplex_kernel.native_listener_microbatch",
    }
    sources: dict[str, dict[str, str]] = {}
    for name, module in modules.items():
        path = Path(str(module.__file__)).resolve(strict=True)
        if name in overlay_module_names:
            expected_root = overlay_root
        elif name.startswith("duplex_kernel."):
            expected_root = workspace
        else:
            expected_root = None
        if expected_root is not None and expected_root.resolve() not in path.parents \
                and path != expected_root.resolve():
            raise RuntimeError(f"runtime module {name} resolved outside its pinned root: {path}")
        sources[name] = {"path": str(path), "sha256": _sha256_file(path)}
    runner_path = Path(__file__).resolve(strict=True)
    if overlay_root != workspace and overlay_root not in runner_path.parents:
        raise RuntimeError("runtime preflight runner is not executing from the isolated overlay")
    sources["probe_runner"] = {"path": str(runner_path), "sha256": _sha256_file(runner_path)}

    output_path = Path(args.output_json)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "mode": "cpu_runtime_import_preflight",
        "torch_version": torch.__version__,
        "cuda_initialized_before_imports": initialized_before,
        "cuda_initialized_after_imports": initialized_after,
        "cuda_availability_probed": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "workspace": str(workspace),
        "overlay_root": str(overlay_root),
        "package_search_path": [str(Path(item).resolve()) for item in sys.modules["duplex_kernel"].__path__],
        "imported_symbols": sorted(key for key in runtime if not key.startswith("_")),
        "module_sources": sources,
        "baseline_dso": {"path": str(baseline), "sha256": _sha256_file(baseline)},
        "candidate_dso": {"path": str(candidate), "sha256": _sha256_file(candidate)},
        "model_config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".pending")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, output_path)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--baseline-dso", default=None)
    parser.add_argument("--candidate-dso", default=None)
    parser.add_argument("--baseline-sha256")
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--model-config-sha256")
    parser.add_argument("--checkpoint-files-sha256")
    parser.add_argument("--gpu-safety-evidence")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-batch-size", type=int, choices=SUPPORTED_BATCHES, default=4,
                        help="stop at N=1, N=2, or N=4; the N=2 trace gate never implies N=4")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute-gpu", action="store_true",
                      help="initialize CUDA and run; only after explicit root GO")
    mode.add_argument("--preflight-runtime", action="store_true",
                      help="import the exact GPU-path Python runtime without initializing CUDA")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.preflight_runtime:
        return _preflight_runtime(args)
    if not args.execute_gpu:
        print(json.dumps(_plan(args), indent=2, sort_keys=True))
        return 0
    required = (args.baseline_sha256, args.candidate_sha256, args.model_config_sha256,
                args.checkpoint_files_sha256, args.gpu_safety_evidence)
    if any(not value for value in required):
        raise SystemExit("--execute-gpu requires all SHA pins and --gpu-safety-evidence")
    return _run_gpu(args)


def _run_gpu(args: argparse.Namespace) -> int:
    """GPU path. Imported lazily so plan and CPU-contract checks stay CPU-only."""
    workspace = Path(args.workspace).resolve(strict=True)
    if str(args.baseline_sha256).lower() != CERTIFIED_REFERENCE_SHA256:
        raise RuntimeError(
            "baseline SHA pin must match the certified sliding-capacity reference "
            f"{CERTIFIED_REFERENCE_SHA256}"
        )
    baseline = Path(args.baseline_dso or _baseline_reference_path(workspace)).resolve(strict=True)
    pinned_baseline = _baseline_reference_path(workspace).resolve(strict=True)
    if baseline != pinned_baseline:
        raise RuntimeError(f"baseline path must be the pinned certified reference: {pinned_baseline}")
    candidate = Path(args.candidate_dso or workspace / ".tmp/native_listener_microbatch_candidate_20261012/"
                     "native/gemma4_listener_microbatch/libgemma4_listener_microbatch.so").resolve(strict=True)
    for path, expected in ((baseline, args.baseline_sha256), (candidate, args.candidate_sha256)):
        if _sha256_file(path) != expected.lower():
            raise RuntimeError(f"DSO SHA-256 mismatch: {path}")
    safety = json.loads(Path(args.gpu_safety_evidence).read_text(encoding="utf-8"))
    safety_record = validate_gpu_safety_evidence(safety)

    model_dir = workspace / ".hf_cache/models--google--gemma-4-E2B-it/full"
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("the pinned local Google Gemma 4 E2B checkpoint is unavailable; no download is allowed")
    if _sha256_file(config_path) != args.model_config_sha256.lower():
        raise RuntimeError("local Gemma config SHA-256 does not match the requested pin")

    # Ensure every model/tokenizer loader remains offline before importing the
    # project model code or touching CUDA.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"] = str(baseline)
    os.environ["GEMMA4_ATEN_SDPA_BRIDGE"] = "1"
    os.environ["GEMMA4_RMSNORM_BACKEND"] = "NATIVE_EXACT"
    os.environ["GEMMA4_ATEN_RMSNORM_BRIDGE"] = "0"

    import torch
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeError(f"expected PyTorch {EXPECTED_TORCH_VERSION}, got {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("the explicitly approved GPU worker has no CUDA device")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if "3090" not in properties.name or tuple(properties.major_minor if hasattr(properties, "major_minor") else
                                               (properties.major, properties.minor)) != (8, 6):
        if "3090" not in properties.name or (properties.major, properties.minor) != (8, 6):
            raise RuntimeError(f"probe is pinned to RTX 3090 sm_86, got {properties.name}")

    runtime = _import_gpu_path_runtime(workspace)
    AUDIO_START_AFTER_TEXT = runtime["AUDIO_START_AFTER_TEXT"]
    GemmaAudioTowerBackend = runtime["GemmaAudioTowerBackend"]
    ListenerMicrobatchAudioFrontend = runtime["ListenerMicrobatchAudioFrontend"]
    BatchRowDescriptor = runtime["BatchRowDescriptor"]
    ListenerBatch = runtime["ListenerBatch"]
    ListenerBatchManifest = runtime["ListenerBatchManifest"]
    ListenerFrame = runtime["ListenerFrame"]
    ListenerSessionState = runtime["ListenerSessionState"]
    RuntimeSignature = runtime["RuntimeSignature"]
    load_thinker = runtime["load_thinker"]
    NativeGemmaListenerLM = runtime["NativeGemmaListenerLM"]
    NativeGemmaListenerMicrobatch = runtime["NativeGemmaListenerMicrobatch"]
    TRACE_KIND_ATTENTION = runtime["TRACE_KIND_ATTENTION"]
    TRACE_STAGE_NAMES = runtime["TRACE_STAGE_NAMES"]
    ShapeStableGemmaListenerLM = runtime["ShapeStableGemmaListenerLM"]
    create_masks_for_generate = runtime["create_masks_for_generate"]

    if AUDIO_START_AFTER_TEXT != PREFIX_LEN:
        raise RuntimeError(f"canonical listener prefix changed from {PREFIX_LEN}")
    thinker = load_thinker(str(workspace), device=device, dtype=torch.bfloat16)
    lm = thinker.model.model.language_model
    if str(getattr(lm, "_gemma4_checkpoint_path", "")) != str(model_dir):
        raise RuntimeError("loaded model did not come from the required local Google E2B checkpoint")
    backend = GemmaAudioTowerBackend(thinker, device=str(device), use_native_lm=False)
    listener_stream = torch.cuda.Stream(device=device, priority=-1)
    if not int(listener_stream.cuda_stream):
        raise RuntimeError("Listener probe requires a non-default CUDA stream")

    with torch.cuda.stream(listener_stream), torch.inference_mode():
        input_ids, attention = backend._build_multimodal_inputs((), 1024, device=device)
        prefix_ids = input_ids[:, :PREFIX_LEN]
        prefix_attention = attention[:, :PREFIX_LEN].to(device=device, dtype=torch.long).contiguous()
        prefix_embeds = lm.embed_tokens(prefix_ids).contiguous()
        prefix_ple = lm.get_per_layer_inputs(prefix_ids, prefix_embeds).contiguous()
        positions = torch.arange(PREFIX_LEN, device=device, dtype=torch.long).view(1, -1)
        prefix_masks = create_masks_for_generate(
            lm.config, prefix_embeds, prefix_attention, past_key_values=None,
            position_ids=positions,
        )
        oracle = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
        prefix_cache = oracle.forward(
            inputs_embeds=prefix_embeds,
            per_layer_inputs=prefix_ple,
            attention_mask=prefix_masks,
            position_ids=positions,
            past_key_values=None,
            shared_kv_states=None,
        ).past_key_values
    prefix_ready = torch.cuda.Event(enable_timing=False)
    prefix_ready.record(listener_stream)

    gate = ParityPhaseGate()
    max_batch_size = int(getattr(args, "max_batch_size", 4))
    if max_batch_size not in SUPPORTED_BATCHES:
        raise ValueError("max_batch_size must be one of 1, 2, or 4")
    run_batch_sizes = tuple(n for n in SUPPORTED_BATCHES if n <= max_batch_size)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING",
        "torch_version": torch.__version__,
        "device": properties.name,
        "baseline_dso": str(baseline),
        "baseline_dso_sha256": args.baseline_sha256.lower(),
        "baseline_role": "independently certified capacity reference; managed serving identity is not inferred from health checks",
        "candidate_dso": str(candidate),
        "candidate_dso_sha256": args.candidate_sha256.lower(),
        "model_config_sha256": args.model_config_sha256.lower(),
        "checkpoint_files_sha256": args.checkpoint_files_sha256.lower(),
        "topology_sha256": TOPOLOGY_SHA256,
        "native_dispatch_environment": {
            "GEMMA4_NATIVE_Q4_GEMV4": os.environ.get("GEMMA4_NATIVE_Q4_GEMV4"),
            "gemma4_native_q4_gemv4_opt_in": os.environ.get("GEMMA4_NATIVE_Q4_GEMV4") == "1",
        },
        "prefix": {"tokens": PREFIX_LEN, "source": "ShapeStable DynamicCache import",
                   "native_prime_async_used": False},
        "capacity": {"max_seq_len": MAX_SEQ_LEN, "q_len": Q_PER_FRAME,
                     "sliding_window": MAX_SLIDING_WINDOW,
                     "sliding_staging_capacity": SLIDING_STAGE_CAPACITY},
        "requested_batch_sizes": list(run_batch_sizes),
        "safety_evidence": safety_record,
        "phases": [],
        "snapshot_comparisons": [],
        "reference_runtime_attestations": [],
        "managed_serving_runtime_identity": "not asserted; serving health checks are maintenance-safety evidence only",
        "latency_certification": "not evaluated",
    }
    run_id = uuid.uuid4().hex
    output_path = Path(args.output_json) if args.output_json else (
        workspace / "validation/kernel_gates/native_listener_microbatch_parity.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path.with_suffix(".tensors.npz")
    tensor_artifacts: dict[str, Any] = {}

    for n in run_batch_sizes:
        if gate.next_batch_size != n:
            break
        baseline_provider = None
        batch_provider = None
        batch_owner = None
        prepared_audio_first = None
        prepared_audio_second = None
        second_manifest = None
        second_audio_soft = None
        second_soft_hashes: list[str] = []
        stage = f"N{n}:setup_pcm_and_reference_context"
        try:
            pcm_rows = _make_pcm_rows(n, phase_seed=n)
            # The existing ABI3 library attests the already loaded model. This
            # context is closed before the candidate batch arena is allocated.
            stage = f"N{n}:construct_abi3_reference_context"
            baseline_provider = NativeGemmaListenerLM(
                lm, max_seq_len=MAX_SEQ_LEN, stream_q_len=Q_PER_FRAME,
                enable_cuda_graph=False, attention_backend=3, rmsnorm_backend=2,
            )
            stage = f"N{n}:attest_abi3_reference_context"
            baseline_info = baseline_provider.native_runtime_info()
            _assert_runtime_pins(baseline_info, args)
            report["reference_runtime_attestations"].append({
                key: baseline_info.get(key) for key in (
                    "library_path", "library_sha256", "manifest_sha256",
                    "checkpoint_path", "checkpoint_files_sha256", "max_seq_len",
                    "attention_backend_id", "rmsnorm_backend_id", "cuda_graph_enabled",
                )
            })
            source_layers = _source_layer_head_dims(lm)
            if len(source_layers) != 15:
                raise RuntimeError(f"expected 15 unique KV sources from the topology, got {len(source_layers)}")
            runtime_signature = _runtime_signature_from_args(
                args, config_sha=args.model_config_sha256.lower(),
                checkpoint_sha=str(baseline_info["checkpoint_files_sha256"]),
            )
            states, frames, rows = _make_probe_rows(
                n, run_id=run_id, pcm_rows=pcm_rows, runtime_signature=runtime_signature,
            )
            manifest = _make_manifest(
                rows, runtime_signature, listener_stream, NativeGemmaListenerMicrobatch,
            )
            probe_batch = ListenerBatch(manifest=manifest, frames=tuple(frames))
            audio_frontend = ListenerMicrobatchAudioFrontend(
                backend.gemma_model, backend._extract_single_frame,
                device=device, dtype=torch.bfloat16,
                audio_tower_row_runner=backend.run_audio_tower_q4,
            )
            stage = f"N{n}:encode_distinct_frontend_rows"
            with torch.cuda.stream(listener_stream), torch.inference_mode():
                prepared_audio_first = audio_frontend.encode_batch(probe_batch, states)
                soft_rows = prepared_audio_first.audio_soft_tokens
                soft_bits = [_bf16_bits(item) for item in soft_rows]
                assert_distinct_input_rows(pcm_rows, soft_bits)
                audio_soft = torch.stack(soft_rows, dim=0).contiguous()
            prefix_ready.synchronize()

            # Reference N=1 calls run serially in one ABI3 context while the
            # ThinkerModel keeps one shared strong reference to frozen weights.
            reference_outputs: list[np.ndarray] = []
            reference_kv: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
            reference_positions: list[int] = []
            for index in range(n):
                stage = f"N{n}:reference_import_prefix_row{index}"
                baseline_provider.reset()
                baseline_provider.import_cache_async(
                    prefix_cache, current_position=PREFIX_LEN,
                    stream=listener_stream, producer_stream=listener_stream,
                )
                stage = f"N{n}:reference_q4_forward_row{index}"
                with torch.cuda.stream(listener_stream):
                    view = baseline_provider.step_async(
                        audio_soft[index:index + 1],
                        stream=listener_stream, producer_stream=listener_stream,
                    )
                    hidden = view.detach().clone()
                    fence = torch.cuda.Event(enable_timing=False)
                    fence.record(listener_stream)
                fence.synchronize()
                expected_position = PREFIX_LEN + Q_PER_FRAME
                if baseline_provider.current_position != expected_position:
                    raise RuntimeError("ABI3 baseline cursor did not advance exactly Q=4")
                reference_positions.append(baseline_provider.current_position)
                reference_outputs.append(_bf16_bits(hidden)[0])
                stage = f"N{n}:reference_kv_export_row{index}"
                kv: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                for layer_idx in source_layers:
                    key, value = baseline_provider.get_cached_kv(layer_idx)
                    kv[layer_idx] = (_bf16_bits(key), _bf16_bits(value))
                    del key, value
                reference_kv.append(kv)

                second_reference_hidden = None
                second_reference_kv = None
                reference_rollover_kv = None
                if n == 1:
                    # N=1 also exercises a two-frame cursor transition and
                    # cache rollover before it is allowed to unlock N=2.
                    prepared_audio_first.commit()
                    prepared_audio_first = None
                    for state in states:
                        state.next_frame_id = 1
                        state.state_version = 1
                        state.sequence_position = 1
                        state.cache_position = PREFIX_LEN + Q_PER_FRAME
                        state.valid_history_tokens = state.cache_position
                    second_pcm_rows = _make_pcm_rows(1, phase_seed=n + 100)
                    second_states, second_frames, second_rows = _make_followup_rows(
                        rows, states, second_pcm_rows,
                    )
                    second_manifest = _make_manifest(
                        second_rows, runtime_signature, listener_stream,
                        NativeGemmaListenerMicrobatch,
                    )
                    second_batch = ListenerBatch(
                        manifest=second_manifest, frames=tuple(second_frames),
                    )
                    stage = "N1:encode_distinct_followup_frontend_row"
                    with torch.cuda.stream(listener_stream), torch.inference_mode():
                        prepared_audio_second = audio_frontend.encode_batch(
                            second_batch, second_states,
                        )
                        second_audio_soft = torch.stack(
                            prepared_audio_second.audio_soft_tokens, dim=0,
                        ).contiguous()
                    assert_distinct_input_rows(
                        second_pcm_rows,
                        [_bf16_bits(item) for item in prepared_audio_second.audio_soft_tokens],
                    )
                    second_soft_hashes = [
                        hashlib.sha256(_bf16_bits(item).tobytes()).hexdigest()
                        for item in prepared_audio_second.audio_soft_tokens
                    ]
                    if second_soft_hashes[0] == hashlib.sha256(soft_bits[0].tobytes()).hexdigest():
                        raise AssertionError("N=1 frame histories produced identical audio soft tokens")
                    if np.array_equal(second_pcm_rows[0], pcm_rows[0]):
                        raise AssertionError("N=1 second PCM frame duplicated the first frame")

                    with torch.cuda.stream(listener_stream), torch.inference_mode():
                        stage = "N1:reference_followup_q4_forward"
                        second_view = baseline_provider.step_async(
                            second_audio_soft,
                            stream=listener_stream, producer_stream=listener_stream,
                        )
                        second_hidden = second_view.detach().clone()
                        second_fence = torch.cuda.Event(enable_timing=False)
                        second_fence.record(listener_stream)
                    second_fence.synchronize()
                    if baseline_provider.current_position != PREFIX_LEN + 2 * Q_PER_FRAME:
                        raise RuntimeError("ABI3 baseline cursor did not advance two exact Q=4 frames")
                    second_reference_hidden = _bf16_bits(second_hidden)[0]
                    second_reference_kv = {}
                    stage = "N1:reference_followup_kv_export"
                    for layer_idx in source_layers:
                        key, value = baseline_provider.get_cached_kv(layer_idx)
                        second_reference_kv[layer_idx] = (_bf16_bits(key), _bf16_bits(value))
                        del key, value
                    baseline_provider.compact_audio_history(
                        expected_cached_frames=2, retain_frames=1,
                        stream=listener_stream,
                    )
                    rollover_fence = torch.cuda.Event(enable_timing=False)
                    rollover_fence.record(listener_stream)
                    rollover_fence.synchronize()
                    if baseline_provider.current_position != PREFIX_LEN + Q_PER_FRAME:
                        raise RuntimeError("ABI3 reference rollover cursor did not retain exactly one Q=4 frame")
                    reference_rollover_kv = {}
                    stage = "N1:reference_rollover_kv_export"
                    for layer_idx in source_layers:
                        key, value = baseline_provider.get_cached_kv(layer_idx)
                        reference_rollover_kv[layer_idx] = (_bf16_bits(key), _bf16_bits(value))
                        del key, value
            baseline_provider.close()
            baseline_provider = None

            # Batch binding owns one true N-row context and closes its temporary
            # ABI3 provider before native batch allocation.
            stage = f"N{n}:construct_candidate_provider"
            batch_provider = NativeGemmaListenerLM(
                lm, max_seq_len=MAX_SEQ_LEN, stream_q_len=Q_PER_FRAME,
                enable_cuda_graph=False, attention_backend=3, rmsnorm_backend=2,
            )
            stage = f"N{n}:construct_candidate_batch_owner"
            batch_owner = NativeGemmaListenerMicrobatch(
                batch_provider, library_path=candidate,
                runtime_signature=runtime_signature, listener_stream=listener_stream,
                max_batch_size=n,
            )
            for row_index, row in enumerate(rows):
                stage = f"N{n}:candidate_import_prefix_row{row_index}"
                batch_owner.import_session(
                    row, prefix_cache, prefix_attention, prefix_len=PREFIX_LEN,
                    producer_stream=listener_stream,
                )
            stage = f"N{n}:candidate_prefix_import_completion"
            prefix_ready.synchronize()
            stage = f"N{n}:read_projection_counters_before_forward"
            counters_before = batch_owner.counters()
            if n == 2:
                stage = "N2:arm_all_stage_operator_trace"
                batch_owner.probe_trace_arm()
            stage = f"N{n}:candidate_batched_q4_forward"
            with torch.cuda.stream(listener_stream), torch.inference_mode():
                result = batch_owner.step_async(
                    manifest, audio_soft, producer_stream=listener_stream,
                )
            stage = f"N{n}:candidate_q4_completion_event"
            result.completion_event.synchronize()
            if n == 2:
                stage = "N2:read_all_stage_operator_trace"
                operator_trace_records = batch_owner.probe_trace_read_all()
                stage = "N2:archive_all_stage_operator_trace_arrays"
                _record_native_operator_trace(
                    output_path=output_path,
                    report=report,
                    tensor_artifacts=tensor_artifacts,
                    records=operator_trace_records,
                    stage_names=TRACE_STAGE_NAMES,
                    attention_kind=TRACE_KIND_ATTENTION,
                    row_descriptors=[asdict(row) for row in rows],
                    model=lm,
                    torch_module=torch,
                    runtime_signature=runtime_signature,
                )
            stage = f"N{n}:read_projection_counters_after_forward"
            counters_after = batch_owner.counters()
            shared_flags = [bool(layer.self_attn.is_kv_shared_layer) for layer in lm.layers]
            if tuple((row.session_id, row.frame_id, row.native_context_id) for row in result.rows) != tuple(
                (row.session_id, row.frame_id, row.native_context_id) for row in rows
            ):
                raise ExactParityError("native output row provenance/order differs from manifest")
            actual_hidden = _bf16_bits(result.tensor)
            for index, row in enumerate(rows):
                stage = f"N{n}:candidate_snapshot_row{index}"
                next_row = _advanced_snapshot_row(row)
                snapshot = batch_owner.snapshot_session(next_row)
                expected_state = _expected_session_state(
                    slot=index, row=next_row, prefix_len=PREFIX_LEN,
                    current_position=PREFIX_LEN + Q_PER_FRAME,
                )
                actual_state = asdict(snapshot.state)
                # Compare each batch row against the individually run N=1 baseline.
                actual_kv = {
                    layer: (_bf16_bits(k), _bf16_bits(v))
                    for layer, (k, v) in snapshot.kv_by_layer.items()
                }
                _assert_snapshot_with_evidence(
                    output_path=output_path, tensor_artifacts=tensor_artifacts,
                    report=report, comparison_id=f"N{n}/{row.session_id}/frame{row.frame_id + 1}",
                    row_metadata=asdict(next_row),
                    expected_hidden_bits=reference_outputs[index],
                    actual_hidden_bits=actual_hidden[index],
                    expected_kv=reference_kv[index], actual_kv=actual_kv,
                    expected_state=expected_state, actual_state=actual_state,
                    counters={"before": counters_before, "after": counters_after},
                )
            counter_delta = assert_projection_batch_counters(
                counters_before, counters_after, batch_size=n,
                layer_count=len(lm.layers), shared_kv_layers=shared_flags,
            )

            # A stale/cancelled manifest must reject before entering native C;
            # its counters remain unchanged and the valid context stays usable.
            stage = f"N{n}:reject_stale_post_step_manifest"
            try:
                batch_owner.step_async(manifest, audio_soft, producer_stream=listener_stream)
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError("stale post-step manifest was accepted")
            if batch_owner.counters() != counters_after:
                raise AssertionError("rejected stale manifest changed native counters")
            stale_rejection_kinds = ["duplicate_frame"]
            current_row0 = _advanced_snapshot_row(rows[0])
            for label, invalid_row in (
                ("wrong_generation", _replace_row(
                    current_row0, session_generation=current_row0.session_generation + 1,
                )),
                ("wrong_epoch", _replace_row(
                    current_row0, state_epoch_id=f"cancelled-{current_row0.state_epoch_id}",
                )),
                ("wrong_context", _replace_row(
                    current_row0, native_context_id=current_row0.native_context_id + 900_000,
                )),
            ):
                stage = f"N{n}:reject_invalid_manifest_{label}"
                invalid_manifest = _make_manifest(
                    (invalid_row,), runtime_signature, listener_stream,
                    NativeGemmaListenerMicrobatch,
                )
                _assert_rejected_manifest_without_native_mutation(
                    batch_owner, invalid_manifest, audio_soft[:1],
                    producer_stream=listener_stream, label=label,
                )
                stale_rejection_kinds.append(label)

            rollover_snapshot = None
            second_counter_delta = None
            if n == 1:
                if second_manifest is None or second_audio_soft is None \
                        or second_reference_hidden is None or second_reference_kv is None \
                        or reference_rollover_kv is None:
                    raise AssertionError("N=1 rollover reference artifacts were not produced")
                second_before = batch_owner.counters()
                stage = "N1:candidate_followup_q4_forward"
                with torch.cuda.stream(listener_stream), torch.inference_mode():
                    second_result = batch_owner.step_async(
                        second_manifest, second_audio_soft,
                        producer_stream=listener_stream,
                    )
                second_result.completion_event.synchronize()
                if tuple(second_result.rows) != tuple(second_manifest.rows):
                    raise ExactParityError("second-frame native output row provenance/order differs")
                second_after = batch_owner.counters()
                second_actual_hidden = _bf16_bits(second_result.tensor)[0]
                second_row = second_manifest.rows[0]
                second_current_row = _advanced_snapshot_row(second_row)
                stage = "N1:candidate_followup_snapshot"
                second_snapshot = batch_owner.snapshot_session(second_current_row)
                second_actual_state = asdict(second_snapshot.state)
                second_expected_state = _expected_session_state(
                    slot=0, row=second_current_row,
                    prefix_len=PREFIX_LEN,
                    current_position=PREFIX_LEN + 2 * Q_PER_FRAME,
                )
                second_actual_kv = {
                    layer: (_bf16_bits(k), _bf16_bits(v))
                    for layer, (k, v) in second_snapshot.kv_by_layer.items()
                }
                _assert_snapshot_with_evidence(
                    output_path=output_path, tensor_artifacts=tensor_artifacts,
                    report=report,
                    comparison_id=f"N1/{second_row.session_id}/frame{second_row.frame_id + 1}",
                    row_metadata=asdict(second_current_row),
                    expected_hidden_bits=second_reference_hidden,
                    actual_hidden_bits=second_actual_hidden,
                    expected_kv=second_reference_kv, actual_kv=second_actual_kv,
                    expected_state=second_expected_state,
                    actual_state=second_actual_state,
                    counters={"before": second_before, "after": second_after},
                )
                second_counter_delta = assert_projection_batch_counters(
                    second_before, second_after, batch_size=1,
                    layer_count=len(lm.layers), shared_kv_layers=shared_flags,
                )

                rollover_before = batch_owner.counters()
                stage = "N1:candidate_rollover"
                batch_owner.rollover_session_async(
                    second_current_row, retained_audio_tokens=Q_PER_FRAME,
                )
                rolled_row = _row_at_position(second_current_row, PREFIX_LEN + Q_PER_FRAME)
                stage = "N1:candidate_rollover_snapshot"
                rollover_snapshot = batch_owner.snapshot_session(rolled_row)
                rollover_actual_state = asdict(rollover_snapshot.state)
                rollover_expected_state = _expected_session_state(
                    slot=0, row=rolled_row,
                    prefix_len=PREFIX_LEN,
                    current_position=PREFIX_LEN + Q_PER_FRAME,
                )
                rollover_actual_kv = {
                    layer: (_bf16_bits(k), _bf16_bits(v))
                    for layer, (k, v) in rollover_snapshot.kv_by_layer.items()
                }
                _assert_snapshot_with_evidence(
                    output_path=output_path, tensor_artifacts=tensor_artifacts,
                    report=report,
                    comparison_id=f"N1/{second_row.session_id}/rollover",
                    row_metadata=asdict(rolled_row),
                    expected_hidden_bits=np.empty((0,), dtype=np.uint16),
                    actual_hidden_bits=np.empty((0,), dtype=np.uint16),
                    expected_kv=reference_rollover_kv, actual_kv=rollover_actual_kv,
                    expected_state=rollover_expected_state,
                    actual_state=rollover_actual_state,
                    counters={"before": rollover_before, "after": batch_owner.counters()},
                )
                prepared_audio_second.commit()
                prepared_audio_second = None

            for index, row in enumerate(rows):
                stage = f"N{n}:retire_session_row{index}"
                current_row = (
                    _row_at_position(_advanced_snapshot_row(second_manifest.rows[0]),
                                     PREFIX_LEN + Q_PER_FRAME)
                    if n == 1 and index == 0 else _advanced_snapshot_row(row)
                )
                batch_owner.release_session_async(current_row)
            if batch_owner._session_by_id:
                raise AssertionError("session retirement did not clear all native bindings")

            stale_reimport_row = rows[0]
            stage = f"N{n}:reject_retired_generation"
            refreshed_row = _fresh_generation_row(
                stale_reimport_row, run_id=run_id,
            )
            try:
                batch_owner.session_state(stale_reimport_row)
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError("retired generation remained addressable after release")
            reimport_before = batch_owner.counters()
            stage = f"N{n}:fresh_generation_prefix_reimport"
            batch_owner.import_session(
                refreshed_row, prefix_cache, prefix_attention,
                prefix_len=PREFIX_LEN, producer_stream=listener_stream,
            )
            stage = f"N{n}:fresh_generation_prefix_snapshot"
            reimport_snapshot = batch_owner.snapshot_session(refreshed_row)
            prefix_reference_kv = _prefix_source_kv_bits(
                lm, prefix_cache, source_layers,
            )
            reimport_expected_state = _expected_session_state(
                slot=0, row=refreshed_row, prefix_len=PREFIX_LEN,
                current_position=PREFIX_LEN,
            )
            reimport_actual_kv = {
                layer: (_bf16_bits(k), _bf16_bits(v))
                for layer, (k, v) in reimport_snapshot.kv_by_layer.items()
            }
            _assert_snapshot_with_evidence(
                output_path=output_path, tensor_artifacts=tensor_artifacts,
                report=report,
                comparison_id=f"N{n}/{refreshed_row.session_id}/fresh-prefix-reimport",
                row_metadata=asdict(refreshed_row),
                expected_hidden_bits=np.empty((0,), dtype=np.uint16),
                actual_hidden_bits=np.empty((0,), dtype=np.uint16),
                expected_kv=prefix_reference_kv, actual_kv=reimport_actual_kv,
                expected_state=reimport_expected_state,
                actual_state=asdict(reimport_snapshot.state),
                counters={"before": reimport_before, "after": batch_owner.counters()},
            )
            stage = f"N{n}:retire_fresh_generation"
            batch_owner.release_session_async(refreshed_row)

            if prepared_audio_first is not None:
                prepared_audio_first.commit()
                prepared_audio_first = None

            phase = {
                "batch_size": n,
                "status": "PASS",
                "comparison": "bitwise exact hidden + every unique-source K/V + slot cursor",
                "prefix_len": PREFIX_LEN,
                "positions": reference_positions,
                "source_kv_layers": source_layers,
                "counter_delta": counter_delta,
                "second_frame_counter_delta": second_counter_delta,
                "distinct_pcm_sha256": [frame.pcm_sha256 for frame in frames],
                "distinct_audio_soft_sha256": [hashlib.sha256(x.tobytes()).hexdigest() for x in soft_bits],
                "second_frame_audio_soft_sha256": (
                    second_soft_hashes
                ),
                "stale_manifest_rejected_before_native_mutation": True,
                "stale_rejection_kinds": stale_rejection_kinds,
                "slots_retired": True,
                "retired_generation_rejected": True,
                "fresh_generation_prefix_reimport_bitexact": True,
                "q4_rollover_bitexact": rollover_snapshot is not None,
                "input_origin": "seeded synthetic PCM transformed by the existing Gemma frontend; not dialogue evidence",
            }
            report["phases"].append(phase)
            gate.record(n, passed=True, evidence_id=f"N{n}-bitexact")
        except BaseException as exc:
            try:
                for transaction in (prepared_audio_first, prepared_audio_second):
                    if transaction is not None and not transaction.committed:
                        transaction.abort()
            except BaseException:
                pass
            diagnostics = _failure_diagnostics(exc, stage=stage, torch_module=torch)
            phase = {
                "batch_size": n,
                "status": "NO-GO",
                "error": f"{type(exc).__name__}: {exc}",
                "failure_diagnostics": diagnostics,
            }
            report["phases"].append(phase)
            gate.record(n, passed=False, evidence_id=f"N{n}-failed")
            report["status"] = "NO-GO"
            report["phase_gate"] = [asdict(item) for item in gate.results]
            _persist_evidence_checkpoint(
                output_path=output_path, artifact_path=artifact_path,
                tensor_artifacts=tensor_artifacts, report=report,
            )
            break
        finally:
            if batch_owner is not None:
                batch_owner.close()
            elif batch_provider is not None and getattr(batch_provider, "_ctx", None):
                batch_provider.close()
            if baseline_provider is not None and getattr(baseline_provider, "_ctx", None):
                baseline_provider.close()

    if report["status"] != "NO-GO":
        if len(gate.results) == len(SUPPORTED_BATCHES):
            report["status"] = "PASS"
        elif max_batch_size == 2 and len(gate.results) == 2:
            report["status"] = "N2_DIAGNOSTIC_COMPLETE"
        else:
            report["status"] = "INCOMPLETE"
    report["phase_gate"] = [asdict(item) for item in gate.results]
    _persist_evidence_checkpoint(
        output_path=output_path, artifact_path=artifact_path,
        tensor_artifacts=tensor_artifacts, report=report,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] in {"PASS", "N2_DIAGNOSTIC_COMPLETE"} else 2


def _runtime_signature_from_args(
    args: argparse.Namespace,
    *,
    config_sha: str,
    checkpoint_sha: str | None = None,
):
    from duplex_kernel.microbatch_contract import RuntimeSignature
    checkpoint = checkpoint_sha or str(args.checkpoint_files_sha256).lower()
    if checkpoint != str(args.checkpoint_files_sha256).lower():
        raise RuntimeError("loaded checkpoint file manifest SHA does not match the requested pin")
    return RuntimeSignature(
        model_id="google/gemma-4-E2B-it",
        model_config_sha256=config_sha,
        checkpoint_sha256=checkpoint,
        topology_sha256=TOPOLOGY_SHA256,
        native_library_sha256=str(args.candidate_sha256).lower(),
        native_abi_version=3,
        batch_api_version=2,
        attention_backend_id=3,
        rmsnorm_backend_id=2,
        max_seq_len=MAX_SEQ_LEN,
        stream_q_len=Q_PER_FRAME,
    )


def _assert_runtime_pins(info: Mapping[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "library_sha256": str(args.baseline_sha256).lower(),
        "manifest_sha256": TOPOLOGY_SHA256,
        "checkpoint_files_sha256": str(args.checkpoint_files_sha256).lower(),
        "max_seq_len": MAX_SEQ_LEN,
        "attention_backend_id": 3,
        "rmsnorm_backend_id": 2,
        "cuda_graph_enabled": False,
    }
    mismatches = {key: (info.get(key), value) for key, value in expected.items()
                  if info.get(key) != value}
    if mismatches:
        raise RuntimeError(f"ABI3 reference runtime is not pinned as requested: {mismatches}")


def _make_pcm_rows(batch_size: int, *, phase_seed: int) -> list[np.ndarray]:
    rows = []
    for index in range(batch_size):
        rng = np.random.default_rng(20261012 + phase_seed * 1009 + index * 7919)
        pcm = (0.05 * rng.standard_normal(1280)).astype(np.float32)
        rows.append(pcm)
    return rows


def _make_probe_rows(batch_size: int, *, run_id: str, pcm_rows: Sequence[np.ndarray], runtime_signature: Any):
    from duplex_kernel.microbatch_contract import BatchRowDescriptor, ListenerFrame, ListenerSessionState
    now = time.monotonic_ns()
    states = []
    frames = []
    rows = []
    for index, pcm in enumerate(pcm_rows):
        sid = f"native-parity-{run_id[:8]}-{index}"
        epoch = f"native-parity-{run_id}-{index}"
        frame = ListenerFrame(
            session_id=sid, generation_id=1, state_epoch_id=epoch,
            frame_id=0, ready_at_ns=now, pcm=pcm,
        )
        native_context_id = 70_000 + index
        state = ListenerSessionState(
            session_id=sid, generation_id=1, state_epoch_id=epoch,
            runtime_signature=runtime_signature,
            next_frame_id=0, state_version=0, sequence_position=0,
            cache_position=PREFIX_LEN, valid_history_tokens=PREFIX_LEN,
            native_context_id=native_context_id,
        )
        row = BatchRowDescriptor(
            session_id=sid, session_generation=1, state_epoch_id=epoch,
            frame_id=0, listener_state_version=0, sequence_position=0,
            cache_position=PREFIX_LEN, valid_history_tokens=PREFIX_LEN,
            native_context_id=native_context_id,
            input_frame_sha256=frame.pcm_sha256,
            reference_pcm_sha256=None, frame_ready_at_ns=now,
            reference_frame_id=None,
        )
        states.append(state)
        frames.append(frame)
        rows.append(row)
    return tuple(states), tuple(frames), tuple(rows)


def _make_manifest(rows: Sequence[Any], runtime_signature: Any, stream: Any, batch_type: Any):
    from duplex_kernel.microbatch_contract import ListenerBatchManifest
    now = time.monotonic_ns()
    return ListenerBatchManifest(
        manifest_id=uuid.uuid4().hex,
        runtime_signature=runtime_signature,
        stream_identity=batch_type._stream_identity(stream),
        stream_priority=-1,
        dispatch_at_ns=now,
        max_coalesce_wait_ns=0,
        dispatch_reason="coalesce_deadline" if len(rows) == 1 else "multiple_ready",
        rows=tuple(rows),
    )


def _advanced_snapshot_row(row: Any):
    from dataclasses import replace
    return replace(
        row,
        frame_id=row.frame_id + 1,
        listener_state_version=row.listener_state_version + 1,
        sequence_position=row.sequence_position + 1,
        cache_position=row.cache_position + Q_PER_FRAME,
        valid_history_tokens=row.valid_history_tokens + Q_PER_FRAME,
    )


def _replace_row(row: Any, **changes: Any):
    from dataclasses import replace
    return replace(row, **changes)


def _assert_rejected_manifest_without_native_mutation(
    batch_owner: Any,
    manifest: Any,
    audio_soft: Any,
    *,
    producer_stream: Any,
    label: str,
) -> None:
    before = batch_owner.counters()
    try:
        batch_owner.step_async(
            manifest, audio_soft, producer_stream=producer_stream,
        )
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError(f"invalid {label} manifest was accepted")
    if batch_owner.counters() != before:
        raise AssertionError(f"rejected {label} manifest changed native counters")


def _make_followup_rows(rows: Sequence[Any], states: Sequence[Any], pcm_rows: Sequence[np.ndarray]):
    from dataclasses import replace
    from duplex_kernel.microbatch_contract import ListenerFrame
    if not (len(rows) == len(states) == len(pcm_rows)):
        raise ValueError("follow-up rows, session states, and PCM must have equal counts")
    now = time.monotonic_ns()
    frames = []
    next_rows = []
    for row, state, pcm in zip(rows, states, pcm_rows, strict=True):
        if state.next_frame_id != row.frame_id + 1:
            raise ValueError("follow-up session state must advance exactly one frame")
        frame = ListenerFrame(
            session_id=state.session_id,
            generation_id=state.generation_id,
            state_epoch_id=state.state_epoch_id,
            frame_id=state.next_frame_id,
            ready_at_ns=now,
            pcm=pcm,
        )
        frames.append(frame)
        next_rows.append(replace(
            row,
            frame_id=state.next_frame_id,
            listener_state_version=state.state_version,
            sequence_position=state.sequence_position,
            cache_position=state.cache_position,
            valid_history_tokens=state.valid_history_tokens,
            input_frame_sha256=frame.pcm_sha256,
            reference_pcm_sha256=None,
            frame_ready_at_ns=now,
            reference_frame_id=None,
        ))
    return tuple(states), tuple(frames), tuple(next_rows)


def _row_at_position(row: Any, position: int):
    from dataclasses import replace
    if position < PREFIX_LEN or position > MAX_SEQ_LEN:
        raise ValueError("snapshot row position is outside the listener cache bounds")
    return replace(row, cache_position=position, valid_history_tokens=position)


def _fresh_generation_row(row: Any, *, run_id: str):
    from dataclasses import replace
    return replace(
        row,
        session_generation=row.session_generation + 1,
        state_epoch_id=f"native-parity-reimport-{run_id}-{row.session_id}",
        frame_id=0,
        listener_state_version=0,
        sequence_position=0,
        cache_position=PREFIX_LEN,
        valid_history_tokens=PREFIX_LEN,
        native_context_id=row.native_context_id + 1_000_000,
    )


def _expected_session_state(
    *, slot: int, row: Any, prefix_len: int, current_position: int,
) -> dict[str, Any]:
    return {
        "slot": slot,
        "native_context_id": row.native_context_id,
        "session_generation": row.session_generation,
        "state_epoch_id": _epoch_hash(row.state_epoch_id),
        "expected_frame_id": row.frame_id,
        "expected_state_version": row.listener_state_version,
        "expected_sequence_position": row.sequence_position,
        "prefix_len": prefix_len,
        "current_position": current_position,
        "valid_kv_len": current_position,
        "max_seq_len": MAX_SEQ_LEN,
        "primed": True,
    }


def _prefix_source_kv_bits(
    language_model: Any, prefix_cache: Any, source_layers: Sequence[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    import torch
    from duplex_kernel.native_listener import NativeGemmaListenerLM
    layers = getattr(prefix_cache, "layers", None)
    shared_layers = getattr(prefix_cache, "shared_layers", None)
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for layer_idx in source_layers:
        pair = shared_layers.get(layer_idx) if hasattr(shared_layers, "get") else None
        if pair is None:
            if layers is None or layer_idx >= len(layers):
                raise RuntimeError(f"prefix DynamicCache lacks source layer {layer_idx}")
            source = layers[layer_idx]
            pair = (getattr(source, "keys", None), getattr(source, "values", None))
        key, value = pair
        key = NativeGemmaListenerLM._canonicalize_cuda_tensor(
            key, name=f"prefix K layer {layer_idx}", dtype=torch.bfloat16,
        )
        value = NativeGemmaListenerLM._canonicalize_cuda_tensor(
            value, name=f"prefix V layer {layer_idx}", dtype=torch.bfloat16,
        )
        expected_head_dim = int(language_model.layers[layer_idx].self_attn.k_proj.weight.shape[0])
        expected_shape = (1, 1, PREFIX_LEN, expected_head_dim)
        if tuple(key.shape) != expected_shape or tuple(value.shape) != expected_shape:
            raise RuntimeError(
                f"prefix source KV layer {layer_idx} has unexpected shapes "
                f"{tuple(key.shape)}/{tuple(value.shape)} != {expected_shape}"
            )
        result[layer_idx] = (_bf16_bits(key), _bf16_bits(value))
    return result


def _source_layer_head_dims(language_model: Any) -> dict[int, int]:
    result = {}
    for index, layer in enumerate(language_model.layers):
        attention = layer.self_attn
        if not bool(attention.is_kv_shared_layer):
            result[index] = int(attention.k_proj.weight.shape[0])
    return result


def _epoch_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little") or 1


def _bf16_bits(tensor: Any) -> np.ndarray:
    import torch
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"strict native parity requires BF16 tensors, got {tensor.dtype}")
    return tensor.detach().contiguous().view(torch.uint16).cpu().numpy().copy()


if __name__ == "__main__":
    raise SystemExit(main())
