"""CPU-only contracts for the production-prefix native capacity comparison.

This version deliberately covers the serving path: a frozen 34-token
ShapeStable DynamicCache is imported into a native listener context.  The old
native-prime diagnostic schema remains separate and unchanged.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from duplex_kernel.microbatch.native_listener_capacity_probe_contract import (
    ProbeContractError,
    TOLERANCES,
    sha256_file,
)


WORKER_SCHEMA = "gemma4-native-listener-capacity-production-prefix-worker/v1"
REPORT_SCHEMA = "gemma4-native-listener-capacity-production-prefix-parity/v1"
CASES = ("fresh_a", "fresh_b", "same_context_reset_import")
POSITIONS = (34, 514, 2054)
SOURCE_LAYER_HEAD_DIMS = {
    **{layer: 256 for layer in range(15)},
    **{layer: 512 for layer in (4, 9, 14)},
}


@dataclass(frozen=True)
class ProductionProbeSpec:
    """Shape contract; small overrides are used only by CPU unit fixtures."""

    prefix_len: int = 34
    frame_count: int = 505
    features_per_frame: int = 4
    hidden_size: int = 1536
    crossing_frame_index: int = 120
    position_checkpoints: tuple[int, ...] = POSITIONS
    source_layers: tuple[int, ...] = tuple(range(15))
    layer_head_dims: Mapping[int, int] = field(default_factory=lambda: dict(SOURCE_LAYER_HEAD_DIMS))
    per_layer_input_shape: tuple[int, ...] = (1, 34, 35, 256)
    tolerances: Mapping[str, float] = field(default_factory=lambda: dict(TOLERANCES))
    worker_schema: str = WORKER_SCHEMA

    @property
    def expected_position(self) -> int:
        return self.prefix_len + self.frame_count * self.features_per_frame

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.frame_count, 1, self.features_per_frame, self.hidden_size)

    @property
    def cases(self) -> tuple[str, ...]:
        return CASES


DEFAULT_SPEC = ProductionProbeSpec()


def expected_array_shapes(spec: ProductionProbeSpec = DEFAULT_SPEC) -> dict[str, tuple[int, ...]]:
    """Return the complete, non-vacuous array inventory required per worker."""
    expected_checkpoints = (
        spec.prefix_len,
        spec.prefix_len + spec.crossing_frame_index * spec.features_per_frame,
        spec.expected_position,
    )
    if not spec.layer_head_dims or tuple(spec.position_checkpoints) != expected_checkpoints:
        raise ProbeContractError("production prefix spec has incomplete layers/checkpoint positions")
    if tuple(sorted(spec.layer_head_dims)) != tuple(spec.source_layers):
        raise ProbeContractError("production prefix spec must name all 15 unique KV source layers")
    shapes: dict[str, tuple[int, ...]] = {
        "prefix.token_ids": (1, spec.prefix_len),
        "prefix.attention": (1, spec.prefix_len),
        "prefix.embeddings": (1, spec.prefix_len, spec.hidden_size),
        "prefix.per_layer_inputs": tuple(spec.per_layer_input_shape),
        "frames.q4": spec.output_shape,
    }
    for layer, head_dim in sorted(spec.layer_head_dims.items()):
        for kind in ("k", "v"):
            shapes[f"prefix.source_cache.layer{layer:02d}.{kind}"] = (
                1, 1, spec.prefix_len, int(head_dim)
            )
    for case in CASES:
        shapes[f"{case}.output"] = spec.output_shape
        for position in spec.position_checkpoints:
            for layer, head_dim in sorted(spec.layer_head_dims.items()):
                for kind in ("k", "v"):
                    shapes[
                        f"{case}.cache.position{position}.layer{layer:02d}.{kind}"
                    ] = (1, 1, position, int(head_dim))
    return shapes


def _require_sha(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ProbeContractError(f"{field_name} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ProbeContractError(f"{field_name} must be hexadecimal") from exc


def _validate_memory_record(value: Any, field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise ProbeContractError(f"{field_name} memory record is missing")
    for key in ("cuda_free_bytes", "cuda_total_bytes", "torch_allocated_bytes", "torch_reserved_bytes"):
        amount = value.get(key)
        if not isinstance(amount, int) or amount < 0:
            raise ProbeContractError(f"{field_name}.{key} must be a nonnegative integer")


def validate_production_worker_report(
    report: Mapping[str, Any],
    *,
    expected_role: str,
    expected_library_sha256: str | None = None,
    expected_source_manifest_sha256: str | None = None,
    spec: ProductionProbeSpec = DEFAULT_SPEC,
) -> None:
    """Reject incomplete or mismatched production-prefix worker evidence."""
    if report.get("schema") != spec.worker_schema:
        raise ProbeContractError("production-prefix worker schema mismatch")
    if report.get("role") != expected_role or expected_role not in {"baseline", "candidate"}:
        raise ProbeContractError("production-prefix worker role mismatch")
    if report.get("status") != "PASS":
        raise ProbeContractError("production-prefix worker status is not PASS")
    for field_name in (
        "library_sha256",
        "source_manifest_sha256",
        "model_checkpoint_files_sha256",
        "native_topology_manifest_sha256",
        "input_sha256",
        "prefix_source_sha256",
        "arrays_file_sha256",
    ):
        _require_sha(report.get(field_name), field_name)
    if expected_library_sha256 and str(report["library_sha256"]).lower() != expected_library_sha256.lower():
        raise ProbeContractError("worker library hash differs from the pinned manifest")
    if expected_source_manifest_sha256 and str(report["source_manifest_sha256"]).lower() != expected_source_manifest_sha256.lower():
        raise ProbeContractError("worker source manifest hash differs from the pinned manifest")

    required_scalars = {
        "abi_version": 3,
        "attention_backend_id": 3,
        "rmsnorm_backend_id": 2,
        "max_seq_len": 16384,
        "stream_q_len": 4,
        "prefix_len": spec.prefix_len,
        "frame_count": spec.frame_count,
        "features_per_frame": spec.features_per_frame,
        "crossing_frame_index": spec.crossing_frame_index,
        "position_checkpoints": list(spec.position_checkpoints),
        "tolerances": dict(spec.tolerances),
        "prefix_mode": "shape_stable_dynamic_cache_import",
        "native_prime_async_used": False,
        "q4_frames_source": "deterministic fixed synthetic BF16 Q4 tensors; not audio/dialogue evidence",
        "prefix_source": "canonical empty-text production multimodal placeholders and ShapeStable prefix path",
    }
    for name, expected in required_scalars.items():
        if report.get(name) != expected:
            raise ProbeContractError(f"{name} differs from the production-prefix contract")
    expected_dims = {str(k): int(v) for k, v in sorted(spec.layer_head_dims.items())}
    if report.get("layer_head_dims") != expected_dims:
        raise ProbeContractError("layer_head_dims differs from the pinned topology contract")

    arrays_path = report.get("arrays_path")
    if not isinstance(arrays_path, str) or not arrays_path.strip():
        raise ProbeContractError("worker arrays_path is missing")
    metadata = report.get("arrays")
    shapes = expected_array_shapes(spec)
    if not isinstance(metadata, Mapping) or set(metadata) != set(shapes):
        raise ProbeContractError("worker array coverage differs from the complete production-prefix inventory")
    for name, shape in shapes.items():
        item = metadata.get(name)
        if not isinstance(item, Mapping) or item.get("shape") != list(shape):
            raise ProbeContractError(f"array metadata shape mismatch for {name}")
        if item.get("dtype") != "float32_from_bf16":
            raise ProbeContractError(f"array dtype contract mismatch for {name}")
        _require_sha(item.get("sha256"), f"arrays.{name}.sha256")

    cases = report.get("cases")
    if not isinstance(cases, Mapping) or set(cases) != set(CASES):
        raise ProbeContractError("case coverage must contain exactly fresh_a, fresh_b, and same_context_reset_import")
    for case_name in CASES:
        case = cases[case_name]
        if not isinstance(case, Mapping) or case.get("status") != "PASS":
            raise ProbeContractError(f"case {case_name} is missing or not PASS")
        if case.get("prefix_import_position") != spec.prefix_len:
            raise ProbeContractError(f"case {case_name} did not import the exact production prefix length")
        if case.get("positions_observed") != list(spec.position_checkpoints):
            raise ProbeContractError(f"case {case_name} did not observe all required cache positions")
        for field_name in ("input_sha256_before", "input_sha256_after", "rng_sha256_before", "rng_sha256_after"):
            _require_sha(case.get(field_name), f"cases.{case_name}.{field_name}")
        if case["input_sha256_before"] != case["input_sha256_after"]:
            raise ProbeContractError(f"case {case_name} mutated frozen production inputs")
        if case["input_sha256_before"] != report["input_sha256"]:
            raise ProbeContractError(f"case {case_name} did not use the worker's pinned frozen inputs")
        if case["rng_sha256_before"] != case["rng_sha256_after"]:
            raise ProbeContractError(f"case {case_name} changed a random generator state")
        if case.get("position_after_decode") != spec.expected_position:
            raise ProbeContractError(f"case {case_name} ended at the wrong sequence position")
        if case.get("prefix_import_cache_exact") is not True:
            raise ProbeContractError(f"case {case_name} did not confirm exact prefix K/V import")
        if not isinstance(case.get("ordered_calls_at_final_position"), int) or case["ordered_calls_at_final_position"] <= 0:
            raise ProbeContractError(f"case {case_name} did not attest ordered attention beyond 2048")

    context = report.get("context_capacity_probe")
    if not isinstance(context, Mapping):
        raise ProbeContractError("context_capacity_probe is missing")
    expected_context_count = 2 if expected_role == "candidate" else 1
    if context.get("context_count") != expected_context_count:
        raise ProbeContractError("context capacity count differs from the worker role")
    if context.get("both_contexts_imported_at") != [spec.prefix_len] * expected_context_count:
        raise ProbeContractError("listener context(s) were not successfully imported at the frozen prefix")
    if expected_role == "candidate":
        _validate_memory_record(context.get("memory_before_contexts"), "memory_before_contexts")
        _validate_memory_record(context.get("memory_after_context_1_import"), "memory_after_context_1_import")
        _validate_memory_record(context.get("memory_after_context_2_import"), "memory_after_context_2_import")


def load_bound_worker_arrays(
    report: Mapping[str, Any],
    supplied_path: str | Path,
    *,
    spec: ProductionProbeSpec = DEFAULT_SPEC,
) -> dict[str, np.ndarray]:
    """Load the exact nonempty NPZ named and hashed by a worker report."""
    reported_path = report.get("arrays_path")
    if not isinstance(reported_path, str) or not reported_path:
        raise ProbeContractError("worker arrays_path is missing")
    expected_path = Path(reported_path).resolve()
    actual_path = Path(supplied_path).resolve()
    if actual_path != expected_path:
        raise ProbeContractError("supplied archive path differs from worker arrays_path")
    if not actual_path.is_file():
        raise ProbeContractError("worker tensor archive is missing")
    if sha256_file(actual_path).lower() != str(report.get("arrays_file_sha256", "")).lower():
        raise ProbeContractError("arrays_file_sha256 mismatch")
    metadata = report.get("arrays")
    shapes = expected_array_shapes(spec)
    if not isinstance(metadata, Mapping) or set(metadata) != set(shapes):
        raise ProbeContractError("array names/coverage mismatch in worker metadata")
    try:
        archive = np.load(actual_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProbeContractError(f"worker archive cannot be loaded: {exc}") from exc
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ProbeContractError("worker tensor archive is not an NPZ")
    with archive:
        if set(archive.files) != set(shapes):
            raise ProbeContractError("NPZ array names differ from required array coverage")
        loaded: dict[str, np.ndarray] = {}
        for name, shape in shapes.items():
            value = np.asarray(archive[name])
            item = metadata[name]
            if value.shape != shape or item.get("shape") != list(shape):
                raise ProbeContractError(f"array shape mismatch for {name}")
            if value.dtype != np.dtype("float32") or item.get("dtype") != "float32_from_bf16":
                raise ProbeContractError(f"array dtype mismatch for {name}")
            digest = hashlib.sha256(np.ascontiguousarray(value).view(np.uint8).tobytes()).hexdigest()
            if digest.lower() != str(item.get("sha256", "")).lower():
                raise ProbeContractError(f"array sha256 mismatch for {name}")
            if not np.isfinite(value).all():
                raise ProbeContractError(f"array contains non-finite values: {name}")
            loaded[name] = value.copy()
    return loaded


def _compare_arrays(
    reference: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    *,
    tolerances: Mapping[str, float],
    exact_prefix: bool = False,
) -> dict[str, Any]:
    if set(reference) != set(actual):
        raise ProbeContractError("comparison arrays have different names")
    worst_max_abs = 0.0
    worst_mae = 0.0
    worst_cosine = 1.0
    failure_names: list[str] = []
    for name in sorted(reference):
        left = np.asarray(reference[name])
        right = np.asarray(actual[name])
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ProbeContractError(f"comparison array shape/dtype mismatch: {name}")
        if exact_prefix and (name.startswith("prefix.") or ".cache.position34." in name):
            passed = np.array_equal(left, right)
            max_abs = float(np.max(np.abs(left - right), initial=0.0))
            mae = float(np.mean(np.abs(left - right))) if left.size else 0.0
            cosine = 1.0 if passed else 0.0
        else:
            delta = np.abs(left.astype(np.float64, copy=False) - right.astype(np.float64, copy=False))
            max_abs = float(np.max(delta, initial=0.0))
            mae = float(np.mean(delta)) if delta.size else 0.0
            left_norm = float(np.linalg.norm(left.astype(np.float64, copy=False).ravel()))
            right_norm = float(np.linalg.norm(right.astype(np.float64, copy=False).ravel()))
            cosine = (
                float(np.dot(left.astype(np.float64, copy=False).ravel(), right.astype(np.float64, copy=False).ravel()) / (left_norm * right_norm))
                if left_norm and right_norm
                else (1.0 if not left_norm and not right_norm else 0.0)
            )
            passed = (
                max_abs <= float(tolerances["max_abs"])
                and mae <= float(tolerances["mae"])
                and cosine >= float(tolerances["cosine_min"])
            )
        worst_max_abs = max(worst_max_abs, max_abs)
        worst_mae = max(worst_mae, mae)
        worst_cosine = min(worst_cosine, cosine)
        if not passed and len(failure_names) < 20:
            failure_names.append(name)
    return {
        "passed": not failure_names,
        "tensor_count": len(reference),
        "worst_max_abs": worst_max_abs,
        "worst_mae": worst_mae,
        "worst_cosine": worst_cosine,
        "failure_examples": failure_names,
        "tolerances": dict(tolerances),
    }


def assemble_production_report(
    *,
    manifest: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    baseline_arrays_path: str | Path,
    candidate_report: Mapping[str, Any],
    candidate_arrays_path: str | Path,
    spec: ProductionProbeSpec = DEFAULT_SPEC,
) -> dict[str, Any]:
    """Assemble the two-worker report without accepting absent or stale evidence."""
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "FAIL",
        "worker_roles": ["baseline", "candidate"],
        "comparisons": {},
        "errors": [],
        "position_contract": list(spec.position_checkpoints),
        "prefix_contract": "production ShapeStable DynamicCache import at exact length 34",
        "native_prime_async_used": False,
        "rollover_compaction": "UNVERIFIED unless separately reached by the recorded route",
        "source_provenance": {},
    }
    try:
        records = manifest.get("libraries")
        if not isinstance(records, Mapping) or set(records) != {"baseline", "candidate"}:
            raise ProbeContractError("manifest library role coverage mismatch")
        runtime = manifest.get("runtime")
        expected_runtime = {
            "abi_version": 3,
            "attention_backend_id": 3,
            "rmsnorm_backend_id": 2,
            "max_seq_len": 16384,
            "stream_q_len": 4,
        }
        if not isinstance(runtime, Mapping) or any(runtime.get(k) != v for k, v in expected_runtime.items()):
            raise ProbeContractError("manifest runtime policy differs from backend-3 production probe")
        if manifest.get("tolerances") != dict(spec.tolerances):
            raise ProbeContractError("manifest tolerances differ from the strict pinned preset")
        result["source_provenance"] = {
            "reconstruction_note": manifest.get("source_reconstruction"),
            "libraries": {
                role: {
                    key: records[role].get(key)
                    for key in (
                        "path", "sha256", "source_manifest_path", "source_manifest_sha256",
                        "source_root", "provenance_note",
                    )
                }
                for role in ("baseline", "candidate")
            },
        }
        for role, report in (("baseline", baseline_report), ("candidate", candidate_report)):
            record = records[role]
            validate_production_worker_report(
                report,
                expected_role=role,
                expected_library_sha256=record.get("sha256"),
                expected_source_manifest_sha256=record.get("source_manifest_sha256"),
                spec=spec,
            )
        baseline_arrays = load_bound_worker_arrays(baseline_report, baseline_arrays_path, spec=spec)
        candidate_arrays = load_bound_worker_arrays(candidate_report, candidate_arrays_path, spec=spec)
        if baseline_report["input_sha256"] != candidate_report["input_sha256"]:
            raise ProbeContractError("baseline and candidate fixed production inputs differ")
        if baseline_report["prefix_source_sha256"] != candidate_report["prefix_source_sha256"]:
            raise ProbeContractError("baseline and candidate ShapeStable prefix K/V snapshots differ")

        def case_arrays(arrays: Mapping[str, np.ndarray], case: str) -> dict[str, np.ndarray]:
            selected = {name: value for name, value in arrays.items() if name.startswith("prefix.")}
            selected.update(
                {f"route.{name[len(case) + 1:]}": value for name, value in arrays.items() if name.startswith(f"{case}.")}
            )
            return selected

        for role, arrays in (("baseline", baseline_arrays), ("candidate", candidate_arrays)):
            source_cache = {
                f"prefix.layer{layer:02d}.{kind}": arrays[f"prefix.source_cache.layer{layer:02d}.{kind}"]
                for layer in spec.source_layers
                for kind in ("k", "v")
            }
            for case in CASES:
                imported_cache = {
                    f"prefix.layer{layer:02d}.{kind}": arrays[
                        f"{case}.cache.position{spec.prefix_len}.layer{layer:02d}.{kind}"
                    ]
                    for layer in spec.source_layers
                    for kind in ("k", "v")
                }
                name = f"{role}_{case}_prefix_import_exact"
                result["comparisons"][name] = _compare_arrays(
                    source_cache,
                    imported_cache,
                    tolerances=spec.tolerances,
                    exact_prefix=True,
                )
            fresh_a = case_arrays(arrays, "fresh_a")
            fresh_b = case_arrays(arrays, "fresh_b")
            reset_import = case_arrays(arrays, "same_context_reset_import")
            result["comparisons"][f"{role}_fresh_repeat"] = _compare_arrays(
                fresh_a,
                fresh_b,
                tolerances=spec.tolerances,
                exact_prefix=True,
            )
            result["comparisons"][f"{role}_reset_import"] = _compare_arrays(
                fresh_b,
                reset_import,
                tolerances=spec.tolerances,
                exact_prefix=True,
            )
        result["comparisons"]["baseline_vs_candidate"] = _compare_arrays(
            baseline_arrays,
            candidate_arrays,
            tolerances=spec.tolerances,
            exact_prefix=True,
        )
        result["worker_evidence"] = {
            role: {
                "library_sha256": worker["library_sha256"],
                "source_manifest_sha256": worker["source_manifest_sha256"],
                "checkpoint_sha256": worker["model_checkpoint_files_sha256"],
                "topology_sha256": worker["native_topology_manifest_sha256"],
                "input_sha256": worker["input_sha256"],
                "prefix_source_sha256": worker["prefix_source_sha256"],
                "arrays_path": worker["arrays_path"],
                "arrays_file_sha256": worker["arrays_file_sha256"],
                "context_capacity_probe": worker["context_capacity_probe"],
            }
            for role, worker in (("baseline", baseline_report), ("candidate", candidate_report))
        }
        if baseline_report["model_checkpoint_files_sha256"] != candidate_report["model_checkpoint_files_sha256"]:
            raise ProbeContractError("worker checkpoint hashes differ")
        if baseline_report["native_topology_manifest_sha256"] != candidate_report["native_topology_manifest_sha256"]:
            raise ProbeContractError("worker topology hashes differ")
        result["status"] = (
            "PASS"
            if all(item["passed"] for item in result["comparisons"].values())
            else "FAIL"
        )
        if result["status"] == "FAIL":
            result["errors"].append("one or more strict production-prefix parity comparisons failed")
    except (ProbeContractError, KeyError, TypeError) as exc:
        result["errors"].append(f"worker validation/assembly failed: {type(exc).__name__}: {exc}")
    return result

