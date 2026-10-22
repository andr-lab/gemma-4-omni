from __future__ import annotations

import numpy as np
import pytest
import json
import hashlib

from duplex_kernel.microbatch.probe_native_listener_capacity_parity import (
    ProbeContractError,
    _actual_cache_from_native,
    _array_sha256,
    _preserve_repeat_evidence,
    compare_reset_reprime_output,
    compare_tensor_arrays,
    assemble_report,
    _verify_source_manifest,
    _worker_environment,
    sha256_file,
    validate_worker_report,
)
from duplex_kernel.microbatch.native_listener_capacity_probe_contract import (
    BASELINE_RESET_DIAGNOSTIC_CASES,
    EXPECTED_DEBUG_LAYERS,
    EXPECTED_DEBUG_LAYER_ROLES,
    NATIVE_CACHE_SOURCE_LAYERS,
    REQUIRED_OUTPUT_ARRAYS,
    assemble_baseline_reset_diagnostic,
)


def fixture_arrays(role: str = "baseline") -> dict[str, np.ndarray]:
    names = set(REQUIRED_OUTPUT_ARRAYS)
    repeat_names = {"repeat.q4_hidden_outputs"}
    for layer in NATIVE_CACHE_SOURCE_LAYERS:
        repeat_names.add(f"repeat.actual_cache.layer{layer:02d}.k")
        repeat_names.add(f"repeat.actual_cache.layer{layer:02d}.v")
    names.update(repeat_names)
    names.update(f"debug.layer{layer:02d}.output" for layer in EXPECTED_DEBUG_LAYERS)
    for layer in NATIVE_CACHE_SOURCE_LAYERS:
        names.add(f"actual_cache.layer{layer:02d}.k")
        names.add(f"actual_cache.layer{layer:02d}.v")
    return {
        name: np.asarray([float(index + 1)], dtype=np.float32)
        for index, name in enumerate(sorted(names))
        if role == "baseline" or name not in repeat_names
    }


def bind_archive(report: dict, path, arrays: dict[str, np.ndarray] | None = None) -> None:
    values = fixture_arrays(report.get("role", "baseline")) if arrays is None else arrays
    np.savez_compressed(path, **values)
    report["arrays_path"] = str(path.resolve())
    report["arrays_file_sha256"] = sha256_file(path)


def worker_report(role: str = "baseline") -> dict:
    arrays = fixture_arrays(role)
    array_metadata = {}
    for name, value in arrays.items():
        array_metadata[name] = {
            "shape": list(value.shape),
            "dtype": "float32_from_bf16" if name.startswith(("actual_cache.", "repeat.actual_cache.")) else "float32_from_bf16_or_native_tensor",
            "sha256": hashlib.sha256(np.ascontiguousarray(value).view(np.uint8).tobytes()).hexdigest(),
        }
    cases = {
        "canonical_prefill291_valid34": {
            "status": "PASS", "position": 34, "compute_q_len": 291, "valid_q_len": 34,
        },
        "q4_cross512": {"status": "PASS", "position_before": 510, "position_after": 514},
        "short_full_fallback": {"status": "PASS", "ordered_calls": 0},
        "cache_import_actual514": {"status": "PASS", "position_before": 514, "position_after": 518},
        "cache_rebase": {"status": "PASS", "position_before": 514, "position_after": 162},
        "reset_reprime": {
            "status": "PASS", "position_after_reset": 0, "position_after_prime": 34,
            "position_after_q4": 38, "first_q4_output_comparison": {"passed": True},
        },
        "debug_source13_and_others": {
            "status": "PASS",
            "layers": list(EXPECTED_DEBUG_LAYERS),
            "layer_roles": dict(EXPECTED_DEBUG_LAYER_ROLES),
        },
        "main_cache_snapshot": {
            "status": "PASS",
            "position": 514,
            "nonshared_source_layers": len(NATIVE_CACHE_SOURCE_LAYERS),
            "source13_key_sha256": array_metadata["actual_cache.layer13.k"]["sha256"],
            "source13_value_sha256": array_metadata["actual_cache.layer13.v"]["sha256"],
            "source13_key_shape": array_metadata["actual_cache.layer13.k"]["shape"],
            "source13_value_shape": array_metadata["actual_cache.layer13.v"]["shape"],
        },
        "ordered_long_full": {
            "status": "PASS", "position_before": 2049, "position_after": 2053,
            "ordered_calls_before": 0, "ordered_calls_after": 1,
        },
        "baseline_repeat_control": {
            "status": "PASS",
            "scope": "full_q4_route_and_cache reset-repeat control",
            "position": 514,
            "input_sha256": "f" * 64,
            "comparison": {"passed": True},
        },
    }
    if role == "candidate":
        cases.pop("baseline_repeat_control")
        cases["two_candidate_contexts"] = {
            "status": "PASS",
            "context_count": 2,
            "primed_positions": [34, 34],
            "free_bytes_before": 8_000,
            "free_bytes_after_context_1": 7_000,
            "free_bytes_after_context_2": 6_000,
        }
    return {
        "schema": "gemma4-native-listener-capacity-worker/v1",
        "role": role,
        "status": "PASS",
        "library_sha256": "a" * 64 if role == "baseline" else "b" * 64,
        "source_manifest_sha256": "c" * 64,
        "abi_version": 3,
        "attention_backend_id": 3,
        "rmsnorm_backend_id": 2,
        "max_seq_len": 16384,
        "stream_q_len": 4,
        "model_checkpoint_files_sha256": "d" * 64,
        "native_topology_manifest_sha256": "e" * 64,
        "input_sha256": "f" * 64,
        "arrays": array_metadata,
        "arrays_path": f"/fixture/{role}.npz",
        "arrays_file_sha256": "e" * 64,
        "cases": cases,
    }


def test_worker_report_requires_each_supported_gpu_path_and_pins_backend() -> None:
    report = worker_report()
    validate_worker_report(report, expected_role="baseline")

    report["cases"].pop("cache_rebase")
    with pytest.raises(ProbeContractError, match="cache_rebase"):
        validate_worker_report(report, expected_role="baseline")


def test_worker_report_requires_reset_and_full_repeat_parity_evidence() -> None:
    report = worker_report()
    report["cases"]["reset_reprime"].pop("first_q4_output_comparison")
    with pytest.raises(ProbeContractError, match="reset_reprime first-Q4 parity"):
        validate_worker_report(report, expected_role="baseline")

    report = worker_report()
    report["cases"]["reset_reprime"]["first_q4_output_comparison"] = {"passed": False}
    with pytest.raises(ProbeContractError, match="reset_reprime first-Q4 parity"):
        validate_worker_report(report, expected_role="baseline")

    report = worker_report()
    report["cases"]["baseline_repeat_control"].pop("comparison")
    with pytest.raises(ProbeContractError, match="baseline repeat parity"):
        validate_worker_report(report, expected_role="baseline")

    report = worker_report()
    report["arrays"].pop("repeat.q4_hidden_outputs")
    with pytest.raises(ProbeContractError, match="array coverage mismatch"):
        validate_worker_report(report, expected_role="baseline")


def test_reset_reprime_compares_actual_first_q4_and_repeat_evidence_survives_failure() -> None:
    main_outputs = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    same_first = np.asarray([1.0, 2.0], dtype=np.float32)
    assert compare_reset_reprime_output(main_outputs, same_first)["passed"] is True

    changed_first = np.asarray([1.0, 2.5], dtype=np.float32)
    comparison = compare_reset_reprime_output(main_outputs, changed_first)
    assert comparison["passed"] is False
    assert comparison["tensors"]["first_q4_output"]["metrics"]["max_abs"] == 0.5

    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, dict] = {}
    repeat_arrays = {
        "repeat.q4_hidden_outputs": np.ones((2, 1), dtype=np.float32),
        "repeat.actual_cache.layer14.k": np.full((1, 1), 14, dtype=np.float32),
    }
    repeat_metadata = {
        name: {"shape": list(value.shape), "dtype": "float32", "sha256": _array_sha256(value)}
        for name, value in repeat_arrays.items()
    }
    _preserve_repeat_evidence(arrays, metadata, repeat_arrays, repeat_metadata)
    failed_repeat = compare_tensor_arrays(
        {"q4": np.asarray([1.0], dtype=np.float32)},
        {"q4": np.asarray([2.0], dtype=np.float32)},
    )
    assert failed_repeat["passed"] is False
    assert set(arrays) == {"repeat.q4_hidden_outputs", "repeat.actual_cache.layer14.k"}
    assert set(metadata) == set(arrays)


def test_worker_report_rejects_missing_archive_or_required_kv_metadata() -> None:
    report = worker_report()
    report.pop("arrays")
    with pytest.raises(ProbeContractError, match="arrays metadata is missing"):
        validate_worker_report(report, expected_role="baseline")

    report = worker_report()
    report["arrays"].pop("actual_cache.layer13.k")
    with pytest.raises(ProbeContractError, match="array coverage mismatch"):
        validate_worker_report(report, expected_role="baseline")


def test_worker_report_requires_distinct_debug_layer_coverage() -> None:
    report = worker_report()
    report["cases"]["debug_source13_and_others"]["layers"] = [13, 13, 13]
    with pytest.raises(ProbeContractError, match="distinct source13/source14 topology roles"):
        validate_worker_report(report, expected_role="baseline")

    report = worker_report()
    report["cases"]["debug_source13_and_others"]["layers"] = [13, 15, 4, 19]
    with pytest.raises(ProbeContractError, match="distinct source13/source14 topology roles"):
        validate_worker_report(report, expected_role="baseline")


def test_worker_report_fails_closed_on_runtime_or_source_mismatch() -> None:
    report = worker_report("candidate")
    report["attention_backend_id"] = 2
    with pytest.raises(ProbeContractError, match="attention_backend_id"):
        validate_worker_report(report, expected_role="candidate")


def _baseline_reset_fixture():
    expected_hash = "f" * 64
    expected_tensors = {"prefix_embeds": "1" * 64, "frames_q4": "2" * 64}
    records = {}
    arrays = {}
    for case in BASELINE_RESET_DIAGNOSTIC_CASES:
        records[case] = {
            "status": "PASS",
            "position_after_q4": 514 if case != "import_rebase" else 166,
            "position_after_import": 514 if case == "import_rebase" else None,
            "input_sha256": expected_hash,
            "input_integrity": {
                "unchanged": True,
                "before_tensor_sha256": expected_tensors,
                "expected_tensor_sha256": expected_tensors,
                "after_tensor_sha256": expected_tensors,
            },
            "rng": {
                "before": {"sha256": "a" * 64},
                "after": {"sha256": "a" * 64},
                "before_sha256": "a" * 64,
                "after_sha256": "a" * 64,
                "unchanged": True,
            },
        }
        if case == "import_rebase":
            arrays[f"baseline_reset.{case}.q4_output"] = np.ones((1, 4, 1536), dtype=np.float32)
            stages = (("after_import", 514), ("after_rebase", 162), ("after_q4", 166))
        else:
            arrays[f"baseline_reset.{case}.q4_hidden_outputs"] = np.ones((120, 1, 4, 1536), dtype=np.float32)
            stages = (("after_prime", 34), ("after_q4", 514))
        for stage, rows in stages:
            for layer, head_dim in ((13, 256), (14, 512)):
                for kind in ("k", "v"):
                    arrays[
                        f"baseline_reset.{case}.{stage}.layer{layer:02d}.{kind}"
                    ] = np.ones((1, 1, rows, head_dim), dtype=np.float32)
    return expected_hash, records, arrays


def test_baseline_reset_matrix_compares_fresh_reset_and_import_rebase_routes() -> None:
    expected_hash, records, arrays = _baseline_reset_fixture()
    report = assemble_baseline_reset_diagnostic(
        case_records=records,
        arrays=arrays,
        expected_input_sha256=expected_hash,
    )
    assert report["status"] == "PASS"
    assert report["case_order"] == list(BASELINE_RESET_DIAGNOSTIC_CASES)
    assert set(report["comparisons"]) == {
        "fresh_a_vs_fresh_b",
        "fresh_b_vs_same_context_reset",
        "fresh_a_vs_same_context_reset",
        "fresh_a_vs_imported_cache",
    }
    assert report["comparisons"]["fresh_b_vs_same_context_reset"]["cache"]["source_layers"]["14"]["k"]["after_q4"]["rows"]["appended_rows"]["passed"]
    assert report["cases"]["import_rebase"]["status"] == "PASS"


def test_baseline_reset_matrix_reports_all_cases_after_first_mismatch() -> None:
    expected_hash, records, arrays = _baseline_reset_fixture()
    arrays["baseline_reset.fresh_b.after_q4.layer14.k"][:, :, 34:, :] *= 2.0
    report = assemble_baseline_reset_diagnostic(
        case_records=records,
        arrays=arrays,
        expected_input_sha256=expected_hash,
    )
    assert report["status"] == "FAIL"
    assert report["comparisons"]["fresh_a_vs_fresh_b"]["cache"]["source_layers"]["14"]["k"]["after_q4"]["rows"]["prefix_rows"]["passed"]
    assert not report["comparisons"]["fresh_a_vs_fresh_b"]["cache"]["source_layers"]["14"]["k"]["after_q4"]["rows"]["appended_rows"]["passed"]
    assert set(report["cases"]) == set(BASELINE_RESET_DIAGNOSTIC_CASES)
    assert "fresh_b_vs_same_context_reset" in report["comparisons"]
    assert "fresh_a_vs_same_context_reset" in report["comparisons"]
    assert "baseline_reset.import_rebase.q4_output" in arrays


def test_baseline_reset_matrix_checks_imported_kv_before_rebase_only() -> None:
    expected_hash, records, arrays = _baseline_reset_fixture()
    arrays["baseline_reset.import_rebase.after_import.layer14.k"][:, :, 34:, :] *= 2.0
    report = assemble_baseline_reset_diagnostic(
        case_records=records,
        arrays=arrays,
        expected_input_sha256=expected_hash,
    )
    imported = report["comparisons"]["fresh_a_vs_imported_cache"]
    assert report["status"] == "FAIL"
    assert not imported["cache"]["source_layers"]["14"]["k"]["passed"]
    assert imported["cache"]["source_layers"]["14"]["k"]["position"] == 514
    assert imported["scope"].endswith("no post-rebase parity claim")


def test_baseline_reset_matrix_fails_closed_on_skips_input_mutation_and_missing_arrays() -> None:
    expected_hash, records, arrays = _baseline_reset_fixture()
    records["fresh_a"]["input_integrity"]["unchanged"] = False
    records["fresh_b"]["status"] = "SKIPPED"
    records["same_context_reset"]["input_sha256"] = "0" * 64
    arrays.pop("baseline_reset.import_rebase.q4_output")
    report = assemble_baseline_reset_diagnostic(
        case_records=records,
        arrays=arrays,
        expected_input_sha256=expected_hash,
    )
    assert report["status"] == "FAIL"
    assert any("fresh_a did not attest unchanged" in item for item in report["errors"])
    assert any("fresh_b status is 'SKIPPED'" in item for item in report["errors"])
    assert any("same_context_reset input hash differs" in item for item in report["errors"])
    assert any("import_rebase missing diagnostic arrays" in item for item in report["errors"])

    expected_hash, records, arrays = _baseline_reset_fixture()
    records["fresh_a"]["input_integrity"]["before_tensor_sha256"] = {"prefix_embeds": "0" * 64}
    records["fresh_a"]["input_integrity"]["after_tensor_sha256"] = {"prefix_embeds": "0" * 64}
    report = assemble_baseline_reset_diagnostic(
        case_records=records,
        arrays=arrays,
        expected_input_sha256=expected_hash,
    )
    assert report["status"] == "FAIL"
    assert any("fresh_a input tensor hash snapshots differ" in item for item in report["errors"])

    report = worker_report("candidate")
    report["source_manifest_sha256"] = None
    with pytest.raises(ProbeContractError, match="source_manifest_sha256"):
        validate_worker_report(report, expected_role="candidate")


def test_candidate_report_requires_both_contexts_primed_and_memory_observed() -> None:
    report = worker_report("candidate")
    validate_worker_report(report, expected_role="candidate")

    report["cases"]["two_candidate_contexts"]["primed_positions"] = [34]
    with pytest.raises(ProbeContractError, match="prime exactly two contexts"):
        validate_worker_report(report, expected_role="candidate")


def test_tensor_comparison_reports_exact_metrics_and_rejects_missing_or_divergent_data() -> None:
    reference = {"hidden": np.asarray([1.0, -2.0, 0.5], dtype=np.float32)}
    same = {"hidden": reference["hidden"].copy()}
    result = compare_tensor_arrays(reference, same)
    assert result["passed"] is True
    assert result["tensors"]["hidden"]["metrics"]["max_abs"] == 0.0

    with pytest.raises(ProbeContractError, match="missing tensor keys"):
        compare_tensor_arrays(reference, {})

    divergent = compare_tensor_arrays(
        reference,
        {"hidden": np.asarray([1.0, -2.0, 0.8], dtype=np.float32)},
    )
    assert divergent["passed"] is False
    assert divergent["tensors"]["hidden"]["metrics"]["max_abs"] > 0.05


def test_comparison_rejects_nonfinite_values_and_shape_drift() -> None:
    with pytest.raises(ProbeContractError, match="non-finite"):
        compare_tensor_arrays(
            {"x": np.asarray([1.0], dtype=np.float32)},
            {"x": np.asarray([np.nan], dtype=np.float32)},
        )
    with pytest.raises(ProbeContractError, match="shape mismatch"):
        compare_tensor_arrays(
            {"x": np.asarray([1.0], dtype=np.float32)},
            {"x": np.asarray([[1.0]], dtype=np.float32)},
        )


def test_baseline_source_overlay_is_hash_pinned_and_rejects_drift(tmp_path) -> None:
    candidate_root = tmp_path / "candidate"
    baseline_root = tmp_path / "baseline"
    candidate_root.mkdir()
    baseline_root.mkdir()
    unchanged = candidate_root / "unchanged.cu"
    overridden = baseline_root / "listener.cu"
    unchanged.write_bytes(b"candidate unchanged source\n")
    overridden.write_bytes(b"preserved pre-change source\n")

    import hashlib

    unchanged_hash = hashlib.sha256(unchanged.read_bytes()).hexdigest()
    override_hash = hashlib.sha256(overridden.read_bytes()).hexdigest()
    source_manifest = tmp_path / "effective.sha256"
    source_manifest.write_text(
        f"{override_hash}  ./listener.cu\n{unchanged_hash}  ./unchanged.cu\n",
        encoding="utf-8",
    )
    manifest_bytes = source_manifest.read_bytes()
    record = {
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_root": str(candidate_root),
        "source_override_root": str(baseline_root),
        "source_overrides": {"./listener.cu": str(overridden)},
    }
    attestation = _verify_source_manifest(record)
    assert attestation["verified_file_count"] == 2
    assert attestation["source_override_count"] == 1

    record["source_overrides"]["./unused.cu"] = str(overridden)
    with pytest.raises(ProbeContractError, match="unused source override"):
        _verify_source_manifest(record)
    del record["source_overrides"]["./unused.cu"]

    escape = tmp_path / "escape.cu"
    escape.write_bytes(overridden.read_bytes())
    record["source_overrides"]["./listener.cu"] = str(escape)
    with pytest.raises(ProbeContractError, match="escapes its pinned root"):
        _verify_source_manifest(record)
    record["source_overrides"]["./listener.cu"] = str(overridden)

    overridden.write_bytes(b"changed after manifest pin\n")
    with pytest.raises(ProbeContractError, match="source snapshot mismatch"):
        _verify_source_manifest(record)


def test_worker_environment_pins_role_library_and_offline_loading() -> None:
    base = {"GEMMA4_NATIVE_LISTENER_LIBRARY_PATH": "/ambient/default.so", "OTHER": "kept"}
    record = {"path": "/workspace/candidate/libgemma4_listener.so"}
    env = _worker_environment(base, record)
    assert env["GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"] == record["path"]
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["OTHER"] == "kept"
    assert base["GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"] == "/ambient/default.so"


def test_report_assembler_preserves_pass_and_turns_mock_worker_failures_into_fail(tmp_path) -> None:
    manifest = {
        "schema": "gemma4-native-listener-capacity-manifest/v1",
        "runtime": {
            "abi_version": 3,
            "attention_backend_id": 3,
            "rmsnorm_backend_id": 2,
            "max_seq_len": 16384,
            "stream_q_len": 4,
            "enable_cuda_graph": False,
        },
        "tolerances": {"cosine_min": 0.999, "mae": 0.01, "max_abs": 0.05},
        "source_reconstruction": "baseline effective snapshot is reconstructed, not build attestation",
        "libraries": {
            "baseline": {
                "path": "/pinned/baseline.so",
                "sha256": "a" * 64,
                "source_manifest_path": "/pinned/baseline.sha256",
                "source_manifest_sha256": "c" * 64,
                "provenance_note": "reconstructed source snapshot; not canonical DSO build attestation",
            },
            "candidate": {
                "path": "/pinned/candidate.so",
                "sha256": "b" * 64,
                "source_manifest_path": "/pinned/candidate.sha256",
                "source_manifest_sha256": "c" * 64,
                "provenance_note": "isolated candidate source snapshot",
            },
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    baseline_report_path = tmp_path / "baseline.json"
    candidate_report_path = tmp_path / "candidate.json"
    baseline_report = worker_report()
    candidate_report = worker_report("candidate")
    baseline_arrays = tmp_path / "baseline.npz"
    candidate_arrays = tmp_path / "candidate.npz"
    bind_archive(baseline_report, baseline_arrays)
    bind_archive(candidate_report, candidate_arrays)
    baseline_report_path.write_text(json.dumps(baseline_report), encoding="utf-8")
    candidate_report_path.write_text(json.dumps(candidate_report), encoding="utf-8")
    output = tmp_path / "assembled.json"

    result = assemble_report(
        manifest_path=manifest_path,
        baseline_report_path=baseline_report_path,
        baseline_arrays_path=baseline_arrays,
        candidate_report_path=candidate_report_path,
        candidate_arrays_path=candidate_arrays,
        output_path=output,
    )
    assert result["status"] == "PASS"
    assert "not build attestation" in result["source_reconstruction_note"]
    assert "not canonical DSO build attestation" in result["source_contract"]["baseline"]["provenance_note"]
    assert output.is_file()

    broken_candidate = worker_report("candidate")
    broken_candidate["cases"].pop("ordered_long_full")
    candidate_report_path.write_text(json.dumps(broken_candidate), encoding="utf-8")
    result = assemble_report(
        manifest_path=manifest_path,
        baseline_report_path=baseline_report_path,
        baseline_arrays_path=baseline_arrays,
        candidate_report_path=candidate_report_path,
        candidate_arrays_path=candidate_arrays,
        output_path=output,
    )
    assert result["status"] == "FAIL"
    assert any("missing required cases" in error for error in result["errors"])


def test_assembler_rejects_unbound_empty_stale_or_hash_mismatched_archives(tmp_path) -> None:
    manifest = {
        "schema": "gemma4-native-listener-capacity-manifest/v1",
        "runtime": {
            "abi_version": 3,
            "attention_backend_id": 3,
            "rmsnorm_backend_id": 2,
            "max_seq_len": 16384,
            "stream_q_len": 4,
            "enable_cuda_graph": False,
        },
        "tolerances": {"cosine_min": 0.999, "mae": 0.01, "max_abs": 0.05},
        "libraries": {
            "baseline": {"path": "/pinned/baseline.so", "sha256": "a" * 64,
                          "source_manifest_path": "/pinned/baseline.sha256", "source_manifest_sha256": "c" * 64},
            "candidate": {"path": "/pinned/candidate.so", "sha256": "b" * 64,
                           "source_manifest_path": "/pinned/candidate.sha256", "source_manifest_sha256": "c" * 64},
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    baseline_report = worker_report()
    candidate_report = worker_report("candidate")
    baseline_report_path = tmp_path / "baseline.json"
    candidate_report_path = tmp_path / "candidate.json"
    baseline_arrays = tmp_path / "baseline.npz"
    candidate_arrays = tmp_path / "candidate.npz"
    bind_archive(baseline_report, baseline_arrays)
    bind_archive(candidate_report, candidate_arrays)
    baseline_report_path.write_text(json.dumps(baseline_report), encoding="utf-8")
    candidate_report_path.write_text(json.dumps(candidate_report), encoding="utf-8")
    output = tmp_path / "full.json"

    substituted = tmp_path / "substituted.npz"
    bind_archive(worker_report(), substituted, {"stale": np.asarray([1.0], dtype=np.float32)})
    result = assemble_report(
        manifest_path=manifest_path,
        baseline_report_path=baseline_report_path,
        baseline_arrays_path=substituted,
        candidate_report_path=candidate_report_path,
        candidate_arrays_path=candidate_arrays,
        output_path=output,
    )
    assert result["status"] == "FAIL"
    assert any("arrays_path does not match" in error for error in result["errors"])

    empty = tmp_path / "empty.npz"
    np.savez_compressed(empty)
    empty_report = worker_report()
    empty_report["arrays_path"] = str(empty.resolve())
    empty_report["arrays_file_sha256"] = sha256_file(empty)
    baseline_report_path.write_text(json.dumps(empty_report), encoding="utf-8")
    result = assemble_report(
        manifest_path=manifest_path,
        baseline_report_path=baseline_report_path,
        baseline_arrays_path=empty,
        candidate_report_path=candidate_report_path,
        candidate_arrays_path=candidate_arrays,
        output_path=output,
    )
    assert result["status"] == "FAIL"
    assert any("array names" in error or "required array" in error for error in result["errors"])

    bind_archive(empty_report, baseline_arrays)
    baseline_report_path.write_text(json.dumps(empty_report), encoding="utf-8")
    np.savez_compressed(candidate_arrays, **{"stale": np.asarray([2.0], dtype=np.float32)})
    result = assemble_report(
        manifest_path=manifest_path,
        baseline_report_path=baseline_report_path,
        baseline_arrays_path=baseline_arrays,
        candidate_report_path=candidate_report_path,
        candidate_arrays_path=candidate_arrays,
        output_path=output,
    )
    assert result["status"] == "FAIL"
    assert any("arrays_file_sha256 mismatch" in error for error in result["errors"])


def test_native_cache_copy_npz_is_consumed_by_complete_assembler(tmp_path) -> None:
    class FakeTensor:
        def __init__(self, shape, pointer):
            self.shape = tuple(shape)
            self.pointer = pointer
            self.device = "cuda:0"

        def data_ptr(self):
            return self.pointer

        def detach(self):
            return self

        def float(self):
            return self

        def to(self, *, device=None):
            assert device == "cpu"
            return self

        def contiguous(self):
            return self

        def view(self, _dtype):
            return self

        def numpy(self):
            return np.full(self.shape, self.pointer, dtype=np.float32)

    class FakeTorch:
        bfloat16 = "bf16"
        uint8 = "uint8"

        def __init__(self):
            self.next_pointer = 1000

        def empty(self, shape, *, dtype, device):
            assert dtype == self.bfloat16 and device == "cuda:0"
            result = FakeTensor(shape, self.next_pointer)
            self.next_pointer += 1
            return result

    class FakeCache:
        def __init__(self, *, config):
            self.config = config
            self.layers = {}

        def update(self, key, value, layer_idx):
            self.layers[layer_idx] = (key, value)

    class FakeCudaMemcpy:
        argtypes = None
        restype = None

        def __init__(self):
            self.calls = []

        def __call__(self, *args):
            self.calls.append(args)
            return 0

    layer_types = ["sliding"] * 14 + ["full_attention"] + ["sliding"] * 20
    lm = type("FakeLM", (), {})()
    lm.config = type("FakeConfig", (), {})()
    lm.layers = [
        type("FakeLayer", (), {"self_attn": type("FakeAttention", (), {"is_kv_shared_layer": index >= 15})()})()
        for index in range(35)
    ]
    native = type("FakeNative", (), {})()
    native._ctx = 77
    native.current_position = 514
    native.output_tensor = type("FakeOutput", (), {"device": "cuda:0"})()
    native.language_model = type("FakeLanguageModel", (), {
        "config": type("FakeNativeConfig", (), {"layer_types": layer_types})(),
    })()
    native._lib = type("FakeNativeLibrary", (), {
        "gemma4_listener_get_cached_k": staticmethod(lambda _ctx, layer: 0x100000 + layer.value),
        "gemma4_listener_get_cached_v": staticmethod(lambda _ctx, layer: 0x200000 + layer.value),
    })()
    memcpy = FakeCudaMemcpy()

    _cache, native_cache_arrays, _cache_sha = _actual_cache_from_native(
        FakeTorch(), FakeCache, lm, native, cuda_memcpy=memcpy,
    )
    assert len(native_cache_arrays) == 2 * len(NATIVE_CACHE_SOURCE_LAYERS)
    assert native_cache_arrays["actual_cache.layer13.k"].shape == (1, 1, 514, 256)
    assert native_cache_arrays["actual_cache.layer14.k"].shape == (1, 1, 514, 512)
    assert len(memcpy.calls) == 2 * len(NATIVE_CACHE_SOURCE_LAYERS)

    manifest = {
        "schema": "gemma4-native-listener-capacity-manifest/v1",
        "runtime": {
            "abi_version": 3, "attention_backend_id": 3, "rmsnorm_backend_id": 2,
            "max_seq_len": 16384, "stream_q_len": 4, "enable_cuda_graph": False,
        },
        "tolerances": {"cosine_min": 0.999, "mae": 0.01, "max_abs": 0.05},
        "libraries": {
            "baseline": {"path": "/pinned/baseline.so", "sha256": "a" * 64,
                          "source_manifest_path": "/pinned/baseline.sha256", "source_manifest_sha256": "c" * 64},
            "candidate": {"path": "/pinned/candidate.so", "sha256": "b" * 64,
                           "source_manifest_path": "/pinned/candidate.sha256", "source_manifest_sha256": "c" * 64},
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    reports = {role: worker_report(role) for role in ("baseline", "candidate")}
    archive_paths = {role: tmp_path / f"{role}.npz" for role in reports}
    report_paths = {role: tmp_path / f"{role}.json" for role in reports}
    for role, report in reports.items():
        role_arrays = fixture_arrays(role)
        role_arrays.update(native_cache_arrays)
        for name, value in native_cache_arrays.items():
            report["arrays"][name] = {
                "shape": list(value.shape), "dtype": "float32_from_bf16", "sha256": _array_sha256(value),
            }
        report["cases"]["main_cache_snapshot"].update({
            "source13_key_sha256": report["arrays"]["actual_cache.layer13.k"]["sha256"],
            "source13_value_sha256": report["arrays"]["actual_cache.layer13.v"]["sha256"],
            "source13_key_shape": report["arrays"]["actual_cache.layer13.k"]["shape"],
            "source13_value_shape": report["arrays"]["actual_cache.layer13.v"]["shape"],
        })
        bind_archive(report, archive_paths[role], role_arrays)
        report_paths[role].write_text(json.dumps(report), encoding="utf-8")

    assembled_path = tmp_path / "assembled.json"
    result = assemble_report(
        manifest_path=manifest_path,
        baseline_report_path=report_paths["baseline"],
        baseline_arrays_path=archive_paths["baseline"],
        candidate_report_path=report_paths["candidate"],
        candidate_arrays_path=archive_paths["candidate"],
        output_path=assembled_path,
    )
    assert result["status"] == "PASS"

    altered = dict(fixture_arrays("candidate"))
    altered.update(native_cache_arrays)
    altered["actual_cache.layer13.k"] = altered["actual_cache.layer13.k"] + 1.0
    candidate = reports["candidate"]
    candidate["arrays"]["actual_cache.layer13.k"]["sha256"] = _array_sha256(altered["actual_cache.layer13.k"])
    candidate["cases"]["main_cache_snapshot"]["source13_key_sha256"] = candidate["arrays"]["actual_cache.layer13.k"]["sha256"]
    bind_archive(candidate, archive_paths["candidate"], altered)
    report_paths["candidate"].write_text(json.dumps(candidate), encoding="utf-8")
    result = assemble_report(
        manifest_path=manifest_path,
        baseline_report_path=report_paths["baseline"],
        baseline_arrays_path=archive_paths["baseline"],
        candidate_report_path=report_paths["candidate"],
        candidate_arrays_path=archive_paths["candidate"],
        output_path=assembled_path,
    )
    assert result["status"] == "FAIL"
    assert any("metrics exceed" in error for error in result["errors"])
