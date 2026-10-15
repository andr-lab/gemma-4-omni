"""Fail-closed contracts and report assembly for the native listener probe.

This module intentionally has no Torch/CUDA imports so the worker's evidence
schema and comparison gates can be exercised on CPU-only hosts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


WORKER_SCHEMA = "gemma4-native-listener-capacity-worker/v1"
REPORT_SCHEMA = "gemma4-native-listener-capacity-parity/v1"
REQUIRED_CASES = (
    "canonical_prefill291_valid34",
    "q4_cross512",
    "short_full_fallback",
    "cache_import_actual514",
    "cache_rebase",
    "reset_reprime",
    "debug_source13_and_others",
    "ordered_long_full",
)
REQUIRED_OUTPUT_ARRAYS = (
    "main.q4_hidden_outputs",
    "rebase.q4_output",
    "reset_reprime.first_q4_output",
    "actual_import.q4_output",
    "ordered_long.q4_output",
)
BASELINE_REPEAT_ARRAYS = (
    "repeat.q4_hidden_outputs",
    *tuple(
        f"repeat.actual_cache.layer{layer:02d}.{kind}"
        for layer in range(15)
        for kind in ("k", "v")
    ),
)
# The pinned Gemma-4-E2B topology has 15 KV-source layers (0..14); the
# remaining layers reuse those cache sources.
NATIVE_CACHE_SOURCE_LAYERS = tuple(range(15))
EXPECTED_DEBUG_LAYER_ROLES = {
    "source13_sliding": 13,
    "source13_sliding_consumer": 15,
    "source14_full": 14,
    "source14_full_consumer": 19,
}
EXPECTED_DEBUG_LAYERS = tuple(EXPECTED_DEBUG_LAYER_ROLES.values())
TOLERANCES = {"cosine_min": 0.999, "mae": 0.01, "max_abs": 0.05}
BASELINE_RESET_DIAGNOSTIC_SCHEMA = "gemma4-native-listener-baseline-reset-diagnostic/v1"
BASELINE_RESET_DIAGNOSTIC_CASES = (
    "fresh_a",
    "fresh_b",
    "same_context_reset",
    "import_rebase",
)
BASELINE_RESET_SOURCE_LAYERS = (13, 14)


class ProbeContractError(ValueError):
    """A missing, inconsistent, or out-of-contract probe artifact."""


def _require_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ProbeContractError(f"{field} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ProbeContractError(f"{field} must be hexadecimal SHA-256") from exc


def validate_worker_report(
    report: Mapping[str, Any],
    *,
    expected_role: str,
    expected_library_sha256: str | None = None,
    expected_source_manifest_sha256: str | None = None,
) -> None:
    """Reject incomplete/incorrect worker evidence before comparisons run."""
    if report.get("schema") != WORKER_SCHEMA:
        raise ProbeContractError("worker schema mismatch")
    if report.get("role") != expected_role:
        raise ProbeContractError(f"role mismatch: {report.get('role')!r}")
    if report.get("status") != "PASS":
        raise ProbeContractError(f"worker status is not PASS: {report.get('status')!r}")
    for name in (
        "library_sha256",
        "source_manifest_sha256",
        "model_checkpoint_files_sha256",
        "native_topology_manifest_sha256",
        "input_sha256",
    ):
        _require_sha256(report.get(name), name)
    if expected_library_sha256 and report["library_sha256"].lower() != expected_library_sha256.lower():
        raise ProbeContractError("library_sha256 differs from pinned manifest")
    if (
        expected_source_manifest_sha256
        and report["source_manifest_sha256"].lower()
        != expected_source_manifest_sha256.lower()
    ):
        raise ProbeContractError("source_manifest_sha256 differs from pinned manifest")
    arrays_path = report.get("arrays_path")
    if not isinstance(arrays_path, str) or not arrays_path.strip():
        raise ProbeContractError("arrays_path is missing")
    _require_sha256(report.get("arrays_file_sha256"), "arrays_file_sha256")
    arrays = report.get("arrays")
    if not isinstance(arrays, Mapping) or not arrays:
        raise ProbeContractError("arrays metadata is missing or empty")
    expected_scalars = {
        "abi_version": 3,
        "attention_backend_id": 3,
        "rmsnorm_backend_id": 2,
        "max_seq_len": 16384,
        "stream_q_len": 4,
    }
    for field, expected in expected_scalars.items():
        if report.get(field) != expected:
            raise ProbeContractError(f"{field} must equal {expected}")

    cases = report.get("cases")
    if not isinstance(cases, Mapping):
        raise ProbeContractError("cases must be an object")
    required = set(REQUIRED_CASES)
    if expected_role == "candidate":
        required.add("two_candidate_contexts")
    else:
        required.add("baseline_repeat_control")
    missing = sorted(required - set(cases))
    if missing:
        raise ProbeContractError(f"missing required cases: {', '.join(missing)}")
    for name in sorted(required):
        case = cases[name]
        if not isinstance(case, Mapping) or case.get("status") != "PASS":
            raise ProbeContractError(f"case {name} is not PASS")

    prime = cases["canonical_prefill291_valid34"]
    if prime.get("position") != 34 or prime.get("compute_q_len") != 291 or prime.get("valid_q_len") != 34:
        raise ProbeContractError("canonical_prefill291_valid34 position/query contract mismatch")
    cross = cases["q4_cross512"]
    if cross.get("position_before") != 510 or cross.get("position_after") != 514:
        raise ProbeContractError("q4_cross512 position contract mismatch")
    short_fallback = cases["short_full_fallback"]
    if short_fallback.get("ordered_calls") != 0:
        raise ProbeContractError("short_full_fallback unexpectedly reported ordered-long calls")
    imported = cases["cache_import_actual514"]
    if imported.get("position_before") != 514 or imported.get("position_after") != 518:
        raise ProbeContractError("cache_import_actual514 position contract mismatch")
    rebase = cases["cache_rebase"]
    if rebase.get("position_before") != 514 or rebase.get("position_after") != 162:
        raise ProbeContractError("cache_rebase position contract mismatch")
    reset = cases["reset_reprime"]
    if reset.get("position_after_reset") != 0 or reset.get("position_after_prime") != 34:
        raise ProbeContractError("reset_reprime position contract mismatch")
    if reset.get("position_after_q4") != 38:
        raise ProbeContractError("reset_reprime Q4 position contract mismatch")
    reset_comparison = reset.get("first_q4_output_comparison")
    if not isinstance(reset_comparison, Mapping) or reset_comparison.get("passed") is not True:
        raise ProbeContractError("reset_reprime first-Q4 parity comparison is missing or failed")
    ordered = cases["ordered_long_full"]
    if ordered.get("position_before") != 2049 or ordered.get("position_after") != 2053:
        raise ProbeContractError("ordered_long_full position contract mismatch")
    if ordered.get("ordered_calls_before") != 0 or ordered.get("ordered_calls_after", 0) <= 0:
        raise ProbeContractError("ordered_long_full did not attest a counter increase")
    debug = cases["debug_source13_and_others"]
    layers = debug.get("layers")
    if layers != list(EXPECTED_DEBUG_LAYERS):
        raise ProbeContractError(
            "debug layer coverage must exactly match distinct source13/source14 topology roles"
        )
    if debug.get("layer_roles") != EXPECTED_DEBUG_LAYER_ROLES:
        raise ProbeContractError("debug layer role mapping differs from pinned topology contract")
    expected_arrays = set(REQUIRED_OUTPUT_ARRAYS)
    if expected_role == "baseline":
        expected_arrays.update(BASELINE_REPEAT_ARRAYS)
    expected_arrays.update(f"debug.layer{int(layer):02d}.output" for layer in layers)
    for layer in NATIVE_CACHE_SOURCE_LAYERS:
        expected_arrays.add(f"actual_cache.layer{layer:02d}.k")
        expected_arrays.add(f"actual_cache.layer{layer:02d}.v")
    if set(arrays) != expected_arrays:
        missing_arrays = sorted(expected_arrays - set(arrays))
        unexpected_arrays = sorted(set(arrays) - expected_arrays)
        raise ProbeContractError(
            f"worker array coverage mismatch; missing={missing_arrays}; unexpected={unexpected_arrays}"
        )
    for name, metadata in arrays.items():
        if not isinstance(metadata, Mapping):
            raise ProbeContractError(f"array metadata for {name} is not an object")
        shape = metadata.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
        ):
            raise ProbeContractError(f"array metadata shape for {name} is missing or empty")
        if metadata.get("dtype") not in {
            "float32_from_bf16_or_native_tensor",
            "float32_from_bf16",
        }:
            raise ProbeContractError(f"array metadata dtype for {name} is unsupported")
        _require_sha256(metadata.get("sha256"), f"arrays.{name}.sha256")
    cache_snapshot = cases.get("main_cache_snapshot")
    if not isinstance(cache_snapshot, Mapping) or cache_snapshot.get("status") != "PASS":
        raise ProbeContractError("main_cache_snapshot is missing or not PASS")
    if cache_snapshot.get("position") != 514:
        raise ProbeContractError("main_cache_snapshot position contract mismatch")
    if cache_snapshot.get("nonshared_source_layers") != len(NATIVE_CACHE_SOURCE_LAYERS):
        raise ProbeContractError("main_cache_snapshot source-layer count mismatch")
    source13_key = arrays["actual_cache.layer13.k"]
    source13_value = arrays["actual_cache.layer13.v"]
    if (
        cache_snapshot.get("source13_key_sha256") != source13_key.get("sha256")
        or cache_snapshot.get("source13_value_sha256") != source13_value.get("sha256")
        or cache_snapshot.get("source13_key_shape") != source13_key.get("shape")
        or cache_snapshot.get("source13_value_shape") != source13_value.get("shape")
    ):
        raise ProbeContractError("main_cache_snapshot source13 metadata mismatch")
    if expected_role == "baseline" and cases["baseline_repeat_control"].get("scope") != "full_q4_route_and_cache reset-repeat control":
        raise ProbeContractError("baseline_repeat_control did not cover the full route and cache")
    if expected_role == "baseline":
        repeat_case = cases["baseline_repeat_control"]
        repeat_comparison = repeat_case.get("comparison")
        if not isinstance(repeat_comparison, Mapping) or repeat_comparison.get("passed") is not True:
            raise ProbeContractError("baseline repeat parity comparison is missing or failed")
        if repeat_case.get("position") != 514:
            raise ProbeContractError("baseline repeat control position contract mismatch")
        _require_sha256(repeat_case.get("input_sha256"), "baseline_repeat_control.input_sha256")

    if expected_role == "candidate":
        pair = cases["two_candidate_contexts"]
        if pair.get("context_count") != 2 or pair.get("primed_positions") != [34, 34]:
            raise ProbeContractError("two_candidate_contexts must prime exactly two contexts")
        memory = (
            pair.get("free_bytes_before"),
            pair.get("free_bytes_after_context_1"),
            pair.get("free_bytes_after_context_2"),
        )
        if any(not isinstance(item, int) or item < 0 for item in memory):
            raise ProbeContractError("two_candidate_contexts memory readings are missing")
        if not memory[0] > memory[1] > memory[2]:
            raise ProbeContractError("candidate context allocations did not reduce free VRAM monotonically")


def _metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    lhs = np.asarray(reference, dtype=np.float64).reshape(-1)
    rhs = np.asarray(actual, dtype=np.float64).reshape(-1)
    delta = np.abs(lhs - rhs)
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm == 0.0 and rhs_norm == 0.0:
        cosine = 1.0
    elif lhs_norm == 0.0 or rhs_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm))
    # Dot rounding can place the computed cosine a few ulps above one.
    cosine = min(1.0, max(-1.0, cosine))
    return {
        "max_abs": float(delta.max(initial=0.0)),
        "mae": float(delta.mean()) if delta.size else 0.0,
        "cosine": cosine,
    }


def compare_tensor_arrays(
    reference: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    *,
    tolerances: Mapping[str, float] = TOLERANCES,
) -> dict[str, Any]:
    """Compare matching named tensors without accepting missing or NaN data."""
    reference_keys = set(reference)
    actual_keys = set(actual)
    if reference_keys != actual_keys:
        missing = sorted(reference_keys - actual_keys)
        extra = sorted(actual_keys - reference_keys)
        raise ProbeContractError(f"missing tensor keys={missing}; unexpected tensor keys={extra}")
    tensors: dict[str, Any] = {}
    passed = True
    for name in sorted(reference_keys):
        left = np.asarray(reference[name])
        right = np.asarray(actual[name])
        if left.shape != right.shape:
            raise ProbeContractError(f"shape mismatch for {name}: {left.shape} != {right.shape}")
        if not np.issubdtype(left.dtype, np.number) or not np.issubdtype(right.dtype, np.number):
            raise ProbeContractError(f"non-numeric tensor data for {name}")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ProbeContractError(f"non-finite tensor data for {name}")
        values = _metrics(left, right)
        item_passed = (
            values["max_abs"] <= float(tolerances["max_abs"])
            and values["mae"] <= float(tolerances["mae"])
            and values["cosine"] >= float(tolerances["cosine_min"])
        )
        tensors[name] = {"shape": list(left.shape), "metrics": values, "passed": item_passed}
        passed = passed and item_passed
    return {"passed": passed, "tolerances": dict(tolerances), "tensors": tensors}


def compare_reset_reprime_output(
    main_outputs: np.ndarray,
    reset_output: np.ndarray,
    *,
    tolerances: Mapping[str, float] = TOLERANCES,
) -> dict[str, Any]:
    """Compare the first reset/re-prime Q4 output with the main route's first Q4."""
    main_values = np.asarray(main_outputs)
    if main_values.ndim < 1 or main_values.shape[0] < 1:
        raise ProbeContractError("main Q4 output route has no first output")
    return compare_tensor_arrays(
        {"first_q4_output": main_values[0]},
        {"first_q4_output": np.asarray(reset_output)},
        tolerances=tolerances,
    )


def _diagnostic_array_key(case: str, stage: str, layer: int, kind: str) -> str:
    return f"baseline_reset.{case}.{stage}.layer{layer:02d}.{kind}"


def _compare_named_arrays(
    reference: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    reference_name: str,
    actual_name: str,
    *,
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    try:
        return compare_tensor_arrays(
            {"tensor": reference[reference_name]},
            {"tensor": actual[actual_name]},
            tolerances=tolerances,
        )
    except (KeyError, ProbeContractError) as exc:
        return {"passed": False, "error": str(exc), "tolerances": dict(tolerances), "tensors": {}}


def _compare_diagnostic_cache(
    arrays: Mapping[str, np.ndarray],
    reference_case: str,
    actual_case: str,
    *,
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {"passed": True, "source_layers": {}}
    for layer in BASELINE_RESET_SOURCE_LAYERS:
        layer_report: dict[str, Any] = {}
        for kind in ("k", "v"):
            prime_left = _diagnostic_array_key(reference_case, "after_prime", layer, kind)
            prime_right = _diagnostic_array_key(actual_case, "after_prime", layer, kind)
            prime = _compare_named_arrays(
                arrays, arrays, prime_left, prime_right, tolerances=tolerances
            )
            post_left = _diagnostic_array_key(reference_case, "after_q4", layer, kind)
            post_right = _diagnostic_array_key(actual_case, "after_q4", layer, kind)
            post: dict[str, Any] = {"passed": True, "rows": {}}
            try:
                reference = np.asarray(arrays[post_left])
                actual = np.asarray(arrays[post_right])
                if reference.shape != actual.shape or reference.ndim != 4 or reference.shape[2] < 34:
                    raise ProbeContractError(
                        f"incompatible post-Q4 cache shapes for layer {layer}.{kind}: "
                        f"{reference.shape} != {actual.shape}"
                    )
                for row_group, start, stop in (
                    ("prefix_rows", 0, 34),
                    ("appended_rows", 34, reference.shape[2]),
                ):
                    row_comparison = compare_tensor_arrays(
                        {row_group: reference[:, :, start:stop, :]},
                        {row_group: actual[:, :, start:stop, :]},
                        tolerances=tolerances,
                    )
                    post["rows"][row_group] = row_comparison
                    post["passed"] = post["passed"] and row_comparison["passed"]
            except (KeyError, ProbeContractError) as exc:
                post = {"passed": False, "error": str(exc), "rows": {}}
            layer_report[kind] = {"after_prime": prime, "after_q4": post}
            comparisons["passed"] = (
                comparisons["passed"] and prime["passed"] and post["passed"]
            )
        comparisons["source_layers"][str(layer)] = layer_report
    return comparisons


def _compare_imported_cache(
    arrays: Mapping[str, np.ndarray],
    *,
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {"passed": True, "source_layers": {}}
    for layer in BASELINE_RESET_SOURCE_LAYERS:
        layer_report: dict[str, Any] = {}
        for kind in ("k", "v"):
            source_name = _diagnostic_array_key("fresh_a", "after_q4", layer, kind)
            imported_name = _diagnostic_array_key("import_rebase", "after_import", layer, kind)
            try:
                source = np.asarray(arrays[source_name])
                imported = np.asarray(arrays[imported_name])
                if source.shape != imported.shape or source.shape != (1, 1, 514, 256 if layer == 13 else 512):
                    raise ProbeContractError(
                        f"same-position imported cache shape mismatch for layer {layer}.{kind}: "
                        f"{source.shape} != {imported.shape}"
                    )
                row_comparisons = {}
                for row_group, start, stop in (("prefix_rows", 0, 34), ("appended_rows", 34, 514)):
                    row_comparisons[row_group] = compare_tensor_arrays(
                        {row_group: source[:, :, start:stop, :]},
                        {row_group: imported[:, :, start:stop, :]},
                        tolerances=tolerances,
                    )
                tensor_report = {
                    "passed": all(item["passed"] for item in row_comparisons.values()),
                    "source_array": source_name,
                    "imported_array": imported_name,
                    "position": 514,
                    "row_count": 514,
                    "rows": row_comparisons,
                }
            except (KeyError, ProbeContractError) as exc:
                tensor_report = {"passed": False, "error": str(exc), "rows": {}}
            layer_report[kind] = tensor_report
            comparisons["passed"] = comparisons["passed"] and tensor_report["passed"]
        comparisons["source_layers"][str(layer)] = layer_report
    return comparisons


def _compare_diagnostic_outputs(
    arrays: Mapping[str, np.ndarray],
    reference_case: str,
    actual_case: str,
    *,
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    left_name = f"baseline_reset.{reference_case}.q4_hidden_outputs"
    right_name = f"baseline_reset.{actual_case}.q4_hidden_outputs"
    try:
        left = np.asarray(arrays[left_name])
        right = np.asarray(arrays[right_name])
        if left.shape != right.shape or left.ndim < 1 or left.shape[0] < 120:
            raise ProbeContractError(
                f"incompatible Q4 output routes: {left.shape} != {right.shape}"
            )
        comparisons = {
            "full_route": compare_tensor_arrays(
                {"q4_outputs": left}, {"q4_outputs": right}, tolerances=tolerances
            ),
            "first_q4": compare_tensor_arrays(
                {"first_q4": left[0]}, {"first_q4": right[0]}, tolerances=tolerances
            ),
            "boundary_q4": compare_tensor_arrays(
                {"boundary_q4": left[118:120]},
                {"boundary_q4": right[118:120]},
                tolerances=tolerances,
            ),
        }
    except (KeyError, ProbeContractError) as exc:
        return {"passed": False, "error": str(exc), "comparisons": {}}
    return {
        "passed": all(value["passed"] for value in comparisons.values()),
        "comparisons": comparisons,
    }


def assemble_baseline_reset_diagnostic(
    *,
    case_records: Mapping[str, Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    expected_input_sha256: str,
    tolerances: Mapping[str, float] = TOLERANCES,
) -> dict[str, Any]:
    """Compare all four baseline route histories without short-circuiting."""
    errors: list[str] = []
    cases = {name: dict(case_records.get(name, {})) for name in BASELINE_RESET_DIAGNOSTIC_CASES}
    for name, case in cases.items():
        if case.get("status") != "PASS":
            errors.append(f"case {name} status is {case.get('status')!r}")
        if case.get("input_sha256") != expected_input_sha256:
            errors.append(f"case {name} input hash differs from the fixed fixture")
        integrity = case.get("input_integrity")
        if not isinstance(integrity, Mapping) or integrity.get("unchanged") is not True:
            errors.append(f"case {name} did not attest unchanged frozen inputs")
        elif (
            integrity.get("before_tensor_sha256")
            != integrity.get("expected_tensor_sha256")
            or integrity.get("after_tensor_sha256")
            != integrity.get("expected_tensor_sha256")
        ):
            errors.append(f"case {name} input tensor hash snapshots differ")
        rng = case.get("rng")
        if (
            not isinstance(rng, Mapping)
            or not isinstance(rng.get("before"), Mapping)
            or not isinstance(rng.get("after"), Mapping)
            or not rng.get("before_sha256")
            or not rng.get("after_sha256")
            or rng.get("before", {}).get("sha256") != rng.get("before_sha256")
            or rng.get("after", {}).get("sha256") != rng.get("after_sha256")
        ):
            errors.append(f"case {name} is missing RNG state snapshots")

    comparisons: dict[str, Any] = {}
    for reference, actual in (("fresh_a", "fresh_b"), ("fresh_b", "same_context_reset"), ("fresh_a", "same_context_reset")):
        key = f"{reference}_vs_{actual}"
        output = _compare_diagnostic_outputs(arrays, reference, actual, tolerances=tolerances)
        cache = _compare_diagnostic_cache(arrays, reference, actual, tolerances=tolerances)
        reference_rng = cases[reference].get("rng", {})
        actual_rng = cases[actual].get("rng", {})
        comparisons[key] = {
            "passed": output["passed"] and cache["passed"],
            "outputs": output,
            "cache": cache,
            "rng": {
                "before_state_equal": reference_rng.get("before_sha256") == actual_rng.get("before_sha256"),
                "after_state_equal": reference_rng.get("after_sha256") == actual_rng.get("after_sha256"),
                "reference_unchanged_during_route": reference_rng.get("unchanged"),
                "actual_unchanged_during_route": actual_rng.get("unchanged"),
            },
        }
        if not comparisons[key]["passed"]:
            errors.append(f"route comparison {key} failed")
    imported_cache = _compare_imported_cache(arrays, tolerances=tolerances)
    comparisons["fresh_a_vs_imported_cache"] = {
        "passed": imported_cache["passed"],
        "cache": imported_cache,
        "scope": "source cache at 514 compared with the same cache after import at 514; no post-rebase parity claim",
    }
    if not imported_cache["passed"]:
        errors.append("same-position imported cache comparison failed")

    fresh_a_position = cases["fresh_a"].get("position_after_q4")
    imported_position = cases["import_rebase"].get("position_after_import")
    if fresh_a_position != 514 or imported_position != 514:
        errors.append(
            f"import cache comparison positions must both be 514: {fresh_a_position!r}, {imported_position!r}"
        )

    required_arrays: dict[str, tuple[str, ...]] = {}
    for case in BASELINE_RESET_DIAGNOSTIC_CASES[:3]:
        required_arrays[case] = (
            f"baseline_reset.{case}.q4_hidden_outputs",
            *(
                _diagnostic_array_key(case, stage, layer, kind)
                for stage in ("after_prime", "after_q4")
                for layer in BASELINE_RESET_SOURCE_LAYERS
                for kind in ("k", "v")
            ),
        )
    required_arrays["import_rebase"] = (
        "baseline_reset.import_rebase.q4_output",
        *(
            _diagnostic_array_key("import_rebase", stage, layer, kind)
            for stage in ("after_import", "after_rebase", "after_q4")
            for layer in BASELINE_RESET_SOURCE_LAYERS
            for kind in ("k", "v")
        ),
    )
    for case, names in required_arrays.items():
        missing = sorted(set(names) - set(arrays))
        if missing:
            errors.append(f"case {case} missing diagnostic arrays: {missing}")
    expected_shapes: dict[str, tuple[int, ...]] = {}
    for case in BASELINE_RESET_DIAGNOSTIC_CASES[:3]:
        expected_shapes[f"baseline_reset.{case}.q4_hidden_outputs"] = (120, 1, 4, 1536)
        for stage, seq_len in (("after_prime", 34), ("after_q4", 514)):
            for layer, head_dim in ((13, 256), (14, 512)):
                for kind in ("k", "v"):
                    expected_shapes[
                        _diagnostic_array_key(case, stage, layer, kind)
                    ] = (1, 1, seq_len, head_dim)
    expected_shapes["baseline_reset.import_rebase.q4_output"] = (1, 4, 1536)
    for stage, seq_len in (("after_import", 514), ("after_rebase", 162), ("after_q4", 166)):
        for layer, head_dim in ((13, 256), (14, 512)):
            for kind in ("k", "v"):
                expected_shapes[
                    _diagnostic_array_key("import_rebase", stage, layer, kind)
                ] = (1, 1, seq_len, head_dim)
    for name, expected_shape in expected_shapes.items():
        if name not in arrays:
            continue
        value = np.asarray(arrays[name])
        if value.shape != expected_shape:
            errors.append(f"diagnostic array {name} shape {value.shape} != {expected_shape}")
        if value.dtype != np.dtype("float32"):
            errors.append(f"diagnostic array {name} dtype {value.dtype} is not float32")
        if not np.isfinite(value).all():
            errors.append(f"diagnostic array {name} contains non-finite values")

    return {
        "schema": BASELINE_RESET_DIAGNOSTIC_SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "case_order": list(BASELINE_RESET_DIAGNOSTIC_CASES),
        "cases": cases,
        "comparisons": comparisons,
        "tolerances": dict(tolerances),
        "errors": errors,
        "interpretation_boundary": [
            "fresh_a vs fresh_b tests sequential fresh-context repeatability",
            "fresh_b vs same_context_reset tests reset/re-prime on a context with no prior import or rebase",
            "import_rebase records the imported-cache plus rebase history and is not treated as an equivalent-state parity comparison",
        ],
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)

