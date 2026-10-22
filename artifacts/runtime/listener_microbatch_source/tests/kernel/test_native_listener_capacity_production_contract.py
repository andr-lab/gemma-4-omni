from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from duplex_kernel.microbatch import probe_native_listener_capacity_production as production_probe
from duplex_kernel.microbatch.native_listener_capacity_production_contract import (
    ProductionProbeSpec,
    ProbeContractError,
    assemble_production_report,
    expected_array_shapes,
    load_bound_worker_arrays,
    validate_production_worker_report,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _small_spec() -> ProductionProbeSpec:
    return ProductionProbeSpec(
        prefix_len=2,
        frame_count=2,
        features_per_frame=1,
        hidden_size=2,
        crossing_frame_index=1,
        position_checkpoints=(2, 3, 4),
        source_layers=(0, 1),
        layer_head_dims={0: 2, 1: 3},
        per_layer_input_shape=(1, 2, 2, 1),
        tolerances={"cosine_min": 0.999, "mae": 0.01, "max_abs": 0.05},
    )


def _worker_fixture(tmp_path: Path, role: str, spec: ProductionProbeSpec, delta: float = 0.0):
    shapes = expected_array_shapes(spec)
    arrays = {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}
    for name in arrays:
        if name.endswith(".output"):
            arrays[name].fill(delta)
    arrays_path = tmp_path / f"{role}.npz"
    np.savez_compressed(arrays_path, **arrays)
    metadata = {
        name: {
            "shape": list(value.shape),
            "dtype": "float32_from_bf16",
            "sha256": _sha(np.ascontiguousarray(value).view(np.uint8).tobytes()),
        }
        for name, value in arrays.items()
    }
    digest = "a" * 64
    cases = {
        case: {
            "status": "PASS",
            "prefix_import_position": spec.prefix_len,
            "prefix_import_cache_exact": True,
            "positions_observed": list(spec.position_checkpoints),
            "input_sha256_before": digest,
            "input_sha256_after": digest,
            "rng_sha256_before": digest,
            "rng_sha256_after": digest,
            "position_after_decode": spec.expected_position,
            "ordered_calls_at_final_position": 1,
        }
        for case in spec.cases
    }
    report = {
        "schema": spec.worker_schema,
        "role": role,
        "status": "PASS",
        "library_sha256": ("1" if role == "baseline" else "2") * 64,
        "source_manifest_sha256": "3" * 64,
        "model_checkpoint_files_sha256": "4" * 64,
        "native_topology_manifest_sha256": "5" * 64,
        "input_sha256": digest,
        "prefix_source_sha256": "b" * 64,
        "prefix_mode": "shape_stable_dynamic_cache_import",
        "native_prime_async_used": False,
        "q4_frames_source": "deterministic fixed synthetic BF16 Q4 tensors; not audio/dialogue evidence",
        "prefix_source": "canonical empty-text production multimodal placeholders and ShapeStable prefix path",
        "prefix_len": spec.prefix_len,
        "frame_count": spec.frame_count,
        "features_per_frame": spec.features_per_frame,
        "crossing_frame_index": spec.crossing_frame_index,
        "position_checkpoints": list(spec.position_checkpoints),
        "layer_head_dims": {str(k): v for k, v in spec.layer_head_dims.items()},
        "tolerances": dict(spec.tolerances),
        "abi_version": 3,
        "attention_backend_id": 3,
        "rmsnorm_backend_id": 2,
        "max_seq_len": 16384,
        "stream_q_len": 4,
        "context_capacity_probe": {
            "context_count": 2 if role == "candidate" else 1,
            "both_contexts_imported_at": [spec.prefix_len, spec.prefix_len]
            if role == "candidate"
            else [spec.prefix_len],
            "memory_before_contexts": {"cuda_free_bytes": 10, "cuda_total_bytes": 100, "torch_allocated_bytes": 1, "torch_reserved_bytes": 2},
            "memory_after_context_1_import": {"cuda_free_bytes": 9, "cuda_total_bytes": 100, "torch_allocated_bytes": 2, "torch_reserved_bytes": 3},
            "memory_after_context_2_import": {"cuda_free_bytes": 8, "cuda_total_bytes": 100, "torch_allocated_bytes": 3, "torch_reserved_bytes": 4},
        },
        "cases": cases,
        "arrays_path": str(arrays_path),
        "arrays_file_sha256": _sha(arrays_path.read_bytes()),
        "arrays": metadata,
    }
    return report, arrays_path, arrays


def test_expected_shapes_require_all_three_routes_all_layers_and_positions():
    shapes = expected_array_shapes(_small_spec())
    for case in ("fresh_a", "fresh_b", "same_context_reset_import"):
        assert shapes[f"{case}.output"] == (2, 1, 1, 2)
        for position in (2, 3, 4):
            for layer, head_dim in ((0, 2), (1, 3)):
                assert shapes[f"{case}.cache.position{position}.layer{layer:02d}.k"] == (
                    1, 1, position, head_dim
                )
                assert shapes[f"{case}.cache.position{position}.layer{layer:02d}.v"] == (
                    1, 1, position, head_dim
                )


def test_worker_report_rejects_missing_route_or_prime_fallback(tmp_path: Path):
    spec = _small_spec()
    report, arrays_path, _ = _worker_fixture(tmp_path, "baseline", spec)
    validate_production_worker_report(report, expected_role="baseline", spec=spec)

    missing_case = copy.deepcopy(report)
    del missing_case["cases"]["same_context_reset_import"]
    with pytest.raises(ProbeContractError, match="case coverage"):
        validate_production_worker_report(missing_case, expected_role="baseline", spec=spec)

    fallback = copy.deepcopy(report)
    fallback["native_prime_async_used"] = True
    with pytest.raises(ProbeContractError, match="native_prime_async"):
        validate_production_worker_report(fallback, expected_role="baseline", spec=spec)

    arrays_path.unlink()


def test_bound_npz_rejects_empty_or_substituted_archives(tmp_path: Path):
    spec = _small_spec()
    report, arrays_path, arrays = _worker_fixture(tmp_path, "baseline", spec)
    loaded = load_bound_worker_arrays(report, arrays_path, spec=spec)
    assert set(loaded) == set(arrays)

    empty_path = tmp_path / "empty.npz"
    np.savez_compressed(empty_path)
    empty_report = copy.deepcopy(report)
    empty_report["arrays_path"] = str(empty_path)
    empty_report["arrays_file_sha256"] = _sha(empty_path.read_bytes())
    with pytest.raises(ProbeContractError, match="array names"):
        load_bound_worker_arrays(empty_report, empty_path, spec=spec)

    substituted_path = tmp_path / "substituted.npz"
    modified = dict(arrays)
    modified[next(iter(modified))] = np.ones_like(modified[next(iter(modified))])
    np.savez_compressed(substituted_path, **modified)
    substituted_report = copy.deepcopy(report)
    substituted_report["arrays_path"] = str(substituted_path)
    substituted_report["arrays_file_sha256"] = _sha(substituted_path.read_bytes())
    with pytest.raises(ProbeContractError, match="sha256 mismatch"):
        load_bound_worker_arrays(substituted_report, substituted_path, spec=spec)

    missing_metadata = copy.deepcopy(report)
    missing_metadata["arrays"].pop(next(iter(missing_metadata["arrays"])))
    with pytest.raises(ProbeContractError, match="array names"):
        load_bound_worker_arrays(missing_metadata, arrays_path, spec=spec)


def test_full_report_compares_fresh_reset_import_and_candidate(tmp_path: Path):
    spec = _small_spec()
    baseline, baseline_path, _ = _worker_fixture(tmp_path, "baseline", spec)
    candidate, candidate_path, _ = _worker_fixture(tmp_path, "candidate", spec)
    manifest = {
        "libraries": {
            "baseline": {"sha256": baseline["library_sha256"], "source_manifest_sha256": "3" * 64},
            "candidate": {"sha256": candidate["library_sha256"], "source_manifest_sha256": "3" * 64},
        },
        "runtime": {"abi_version": 3, "attention_backend_id": 3, "rmsnorm_backend_id": 2, "max_seq_len": 16384, "stream_q_len": 4},
        "tolerances": dict(spec.tolerances),
    }
    result = assemble_production_report(
        manifest=manifest,
        baseline_report=baseline,
        baseline_arrays_path=baseline_path,
        candidate_report=candidate,
        candidate_arrays_path=candidate_path,
        spec=spec,
    )
    assert result["status"] == "PASS", result
    assert result["comparisons"]["baseline_fresh_repeat"]["tensor_count"] > 0
    assert result["comparisons"]["candidate_reset_import"]["passed"] is True
    assert result["comparisons"]["baseline_vs_candidate"]["passed"] is True

    # The reset/import route is anchored to the context that ran fresh_b (B),
    # while the fresh-context repeat remains A -> B.
    reset_npz = np.load(candidate_path)
    reset_arrays = {name: reset_npz[name].copy() for name in reset_npz.files}
    reset_npz.close()
    reset_arrays["fresh_a.output"].fill(0.25)
    np.savez_compressed(candidate_path, **reset_arrays)
    candidate["arrays_file_sha256"] = _sha(candidate_path.read_bytes())
    candidate["arrays"] = {
        name: {
            "shape": list(value.shape),
            "dtype": "float32_from_bf16",
            "sha256": _sha(np.ascontiguousarray(value).view(np.uint8).tobytes()),
        }
        for name, value in reset_arrays.items()
    }
    result = assemble_production_report(
        manifest=manifest,
        baseline_report=baseline,
        baseline_arrays_path=baseline_path,
        candidate_report=candidate,
        candidate_arrays_path=candidate_path,
        spec=spec,
    )
    assert result["comparisons"]["candidate_fresh_repeat"]["passed"] is False
    assert result["comparisons"]["candidate_reset_import"]["passed"] is True

    mutated = copy.deepcopy(candidate)
    mutated["cases"]["same_context_reset_import"]["rng_sha256_after"] = "f" * 64
    result = assemble_production_report(
        manifest=manifest,
        baseline_report=baseline,
        baseline_arrays_path=baseline_path,
        candidate_report=mutated,
        candidate_arrays_path=candidate_path,
        spec=spec,
    )
    assert result["status"] == "FAIL"
    assert "worker validation" in result["errors"][0]


def test_candidate_routes_fresh_b_then_reset_import_on_same_distinct_context_and_closes_both():
    class Context:
        def __init__(self, label: str):
            self.label = label
            self.close_count = 0

        def close(self):
            self.close_count += 1

    native_one = Context("A")
    native_two = Context("B")
    seen = []

    def route_case(case_name, native):
        seen.append((case_name, native))
        if case_name == "same_context_reset_import":
            raise RuntimeError("injected route failure")
        return {"case": case_name}

    cleanup_errors = []
    with pytest.raises(RuntimeError, match="injected route failure"):
        production_probe._run_candidate_routes(
            native_one=native_one,
            native_two=native_two,
            route_case=route_case,
            cleanup_errors=cleanup_errors,
        )

    assert [(name, context.label) for name, context in seen] == [
        ("fresh_a", "A"),
        ("fresh_b", "B"),
        ("same_context_reset_import", "B"),
    ]
    assert native_one is not native_two
    assert native_one.close_count == native_two.close_count == 1
    assert cleanup_errors == []


def test_candidate_route_rejects_aliased_context_and_closes_owned_instance_once():
    class Context:
        close_count = 0

        def close(self):
            self.close_count += 1

    context = Context()
    cleanup_errors = []
    with pytest.raises(ProbeContractError, match="distinct instances"):
        production_probe._run_candidate_routes(
            native_one=context,
            native_two=context,
            route_case=lambda case_name, native: {},
            cleanup_errors=cleanup_errors,
        )
    assert context.close_count == 1
    assert cleanup_errors == []


def test_assembler_fails_closed_on_missing_kv_layer_or_threshold_failure(tmp_path: Path):
    spec = _small_spec()
    baseline, baseline_path, _ = _worker_fixture(tmp_path, "baseline", spec)
    candidate, candidate_path, _ = _worker_fixture(tmp_path, "candidate", spec, delta=1.0)
    manifest = {
        "libraries": {
            "baseline": {"sha256": baseline["library_sha256"], "source_manifest_sha256": "3" * 64},
            "candidate": {"sha256": candidate["library_sha256"], "source_manifest_sha256": "3" * 64},
        },
        "runtime": {"abi_version": 3, "attention_backend_id": 3, "rmsnorm_backend_id": 2, "max_seq_len": 16384, "stream_q_len": 4},
        "tolerances": dict(spec.tolerances),
    }
    result = assemble_production_report(
        manifest=manifest,
        baseline_report=baseline,
        baseline_arrays_path=baseline_path,
        candidate_report=candidate,
        candidate_arrays_path=candidate_path,
        spec=spec,
    )
    assert result["status"] == "FAIL"
    assert result.get("comparisons", {}).get("baseline_vs_candidate", {}).get("passed") is False, result

    bad_candidate = copy.deepcopy(candidate)
    del bad_candidate["arrays"]["fresh_a.cache.position2.layer01.v"]
    with pytest.raises(ProbeContractError, match="array coverage"):
        validate_production_worker_report(bad_candidate, expected_role="candidate", spec=spec)


def test_sequential_runner_stops_after_baseline_failure_without_candidate_retry(tmp_path: Path, monkeypatch):
    manifest = {
        "schema": "gemma4-native-listener-capacity-manifest/v1",
        "runtime": {
            "abi_version": 3, "attention_backend_id": 3, "rmsnorm_backend_id": 2,
            "max_seq_len": 16384, "stream_q_len": 4, "enable_cuda_graph": False,
        },
        "tolerances": {"cosine_min": 0.999, "mae": 0.01, "max_abs": 0.05},
        "libraries": {
            role: {
                "path": f"/workspace/native/{role}.so",
                "sha256": ("a" if role == "baseline" else "b") * 64,
                "source_manifest_path": f"/workspace/artifacts/{role}.sha256",
                "source_manifest_sha256": "c" * 64,
            }
            for role in ("baseline", "candidate")
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls = []

    def failed_baseline(command, **kwargs):
        calls.append(command)
        return production_probe.subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    monkeypatch.setattr(production_probe.subprocess, "run", failed_baseline)
    result = production_probe.run_all(manifest_path, tmp_path / "output")

    assert result["status"] == "FAIL"
    assert result["candidate_status"] == "UNMEASURED"
    assert len(calls) == 1 and "baseline" in calls[0]
    assert "candidate" not in result["sequential_workers"]
