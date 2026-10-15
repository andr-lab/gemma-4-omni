"""CPU-only fail-closed contract tests for the deferred GPU parity probe."""
from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import duplex_kernel.microbatch.probe_native_listener_microbatch_parity as parity_probe

from duplex_kernel.microbatch.probe_native_listener_microbatch_parity import (
    ExactParityError,
    ParityPhaseGate,
    _failure_diagnostics,
    _assert_snapshot_with_evidence,
    _baseline_reference_path,
    assert_exact_array,
    assert_projection_batch_counters,
    assert_snapshot_exact,
    expected_projection_gemm_count,
    validate_gpu_safety_evidence,
)


class NativeListenerMicrobatchParityProbeTests(unittest.TestCase):
    def test_n2_operator_trace_is_opt_in_full_stage_table_and_stops_before_n4(self):
        source = Path(parity_probe.__file__).read_text(encoding="utf-8")
        self.assertIn('"diagnostic_row_count": 8', source)
        self.assertIn("probe_trace_arm()", source)
        self.assertIn("probe_trace_read_all()", source)
        self.assertIn("_record_native_operator_trace(", source)
        self.assertIn('"attention_reference": "DEFERRED"', source)
        self.assertIn('"N2_DIAGNOSTIC_COMPLETE"', source)
        self.assertEqual(parity_probe._parser().parse_args(["--max-batch-size", "2"]).max_batch_size, 2)

    def test_operator_trace_archives_all_stages_before_first_exactness_failure(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            output = Path(directory) / "operator-trace.json"
            stage_names = tuple(f"STAGE_{index}" for index in range(32))
            records = []
            for stage_id, stage_name in enumerate(stage_names):
                kind = 4 if stage_id == 13 else 1
                actual = np.array([stage_id + 10, stage_id + 11], dtype=np.uint16)
                reference = actual.copy()
                if stage_id == 18:
                    actual[1] ^= np.uint16(1)
                metadata = {
                    "stage_id": stage_id,
                    "stage_name": stage_name,
                    "kind": kind,
                    "input0_rows": 2,
                    "input0_cols": 1,
                    "input1_rows": 0,
                    "input1_cols": 0,
                    "input2_rows": 0,
                    "input2_cols": 0,
                    "actual_rows": 2,
                    "actual_cols": 1,
                    "reference_rows": 0 if kind == 4 else 2,
                    "reference_cols": 0 if kind == 4 else 1,
                    "input0_elements": 2,
                    "input1_elements": 0,
                    "input2_elements": 0,
                    "actual_elements": 2,
                    "reference_elements": 0 if kind == 4 else 2,
                    "weight_device_ptr": 0,
                }
                records.append({
                    "metadata": metadata,
                    "raw": {
                        "input0_bits": np.array([stage_id, stage_id + 1], dtype=np.uint16).tobytes(),
                        "input1_bits": b"",
                        "input2_bits": b"",
                        "actual_bits": actual.tobytes(),
                        "reference_bits": b"" if kind == 4 else reference.tobytes(),
                    },
                })

            runtime_signature = SimpleNamespace(
                model_config_sha256="a" * 64,
                checkpoint_sha256="b" * 64,
                topology_sha256="c" * 64,
                native_library_sha256="d" * 64,
                native_abi_version=3,
                batch_api_version=2,
                attention_backend_id=3,
                rmsnorm_backend_id=2,
                max_seq_len=16384,
                stream_q_len=4,
            )
            report: dict[str, object] = {"status": "RUNNING"}
            artifacts: dict[str, np.ndarray] = {}
            with self.assertRaisesRegex(ExactParityError, "STAGE_18"):
                parity_probe._record_native_operator_trace(
                    output_path=output,
                    report=report,
                    tensor_artifacts=artifacts,
                    records=records,
                    stage_names=stage_names,
                    attention_kind=4,
                    row_descriptors=({"session_id": "sid-a"}, {"session_id": "sid-b"}),
                    model=object(),
                    torch_module=object(),
                    runtime_signature=runtime_signature,
                )

            payload = json.loads(output.read_text(encoding="utf-8"))
            trace = payload["native_operator_trace"]
            self.assertEqual(trace["stage_count"], 32)
            self.assertEqual(trace["first_mismatch"]["stage_name"], "STAGE_18")
            npz_path = Path(payload["tensor_artifact"]["path"])
            with np.load(npz_path, allow_pickle=False) as saved:
                self.assertIn("N2/operator_trace/STAGE_31/actual_bits", saved.files)
                self.assertTrue(np.array_equal(
                    saved["N2/operator_trace/STAGE_18/actual_bits"],
                    np.array([[28], [28]], dtype=np.uint16),
                ))
                self.assertEqual(saved["N2/operator_trace/STAGE_13/reference_bits"].size, 0)

    def test_overlay_path_bridge_resolves_probe_and_canonical_modules_once(self):
        workspace = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp_dir:
            overlay = Path(temp_dir) / "payload"
            overlay_package = overlay / "duplex_kernel"
            overlay_package.mkdir(parents=True, exist_ok=True)
            (overlay_package / "__init__.py").touch()
            for mod in ("listener_microbatch_frontend.py", "native_listener_microbatch.py", "microbatch_contract.py"):
                (overlay_package / mod).touch()
            canonical_package = workspace / "duplex_kernel"
            package = SimpleNamespace(
                __file__=str(overlay_package / "__init__.py"),
                __path__=[str(overlay_package), str(canonical_package)],
            )
            prior_path = list(sys.path)
            try:
                with mock.patch.object(importlib, "import_module", return_value=package):
                    resolved = parity_probe._install_overlay_import_path(workspace, overlay_root=overlay)
                self.assertIs(resolved, package)
                self.assertEqual(Path(sys.path[0]).resolve(), overlay.resolve())
                self.assertEqual(Path(sys.path[1]).resolve(), workspace.resolve())
                package_roots = [str(Path(item).resolve()) for item in package.__path__]
                self.assertEqual(package_roots, [str(overlay_package.resolve()), str(canonical_package.resolve())])

                for module_name, expected in (
                    ("duplex_kernel.listener_microbatch_frontend", overlay_package / "listener_microbatch_frontend.py"),
                    ("duplex_kernel.native_listener_microbatch", overlay_package / "native_listener_microbatch.py"),
                    ("duplex_kernel.microbatch_contract", overlay_package / "microbatch_contract.py"),
                    ("duplex_kernel.listener", canonical_package / "listener.py"),
                ):
                    spec = importlib.machinery.PathFinder.find_spec(module_name, package.__path__)
                    self.assertIsNotNone(spec, module_name)
                    self.assertEqual(Path(spec.origin).resolve(), expected.resolve())
            finally:
                sys.path[:] = prior_path

    def test_failure_diagnostics_preserve_strict_stride_evidence(self):
        from types import SimpleNamespace

        class MockTensor:
            shape = (1, 34)
            dtype = "torch.int64"
            device = "cuda:0"

            def stride(self):
                # PyTorch may report a singleton-axis tensor as contiguous even
                # when its exact native row-major stride contract differs.
                return (1, 1)

            def is_contiguous(self):
                return True

            def storage_offset(self):
                return 0

        def raise_layout_error():
            tensor = MockTensor()
            raise ValueError("metadata nativo richiede layout row-major contiguous")

        try:
            raise_layout_error()
        except ValueError as exc:
            diagnostics = _failure_diagnostics(
                exc,
                stage="N1:candidate_import_prefix_row0",
                torch_module=SimpleNamespace(Tensor=MockTensor),
            )
        else:
            self.fail("expected mock row-major layout error")
        self.assertEqual(diagnostics["stage"], "N1:candidate_import_prefix_row0")
        tensor_local = next(
            item
            for frame in diagnostics["tensor_locals"]
            for item in frame["tensors"]
            if item["local_name"] == "tensor"
        )
        self.assertEqual(tensor_local["shape"], [1, 34])
        self.assertEqual(tensor_local["strides"], [1, 1])
        self.assertEqual(tensor_local["expected_row_major_strides"], [34, 1])
        self.assertFalse(tensor_local["exact_row_major"])
        self.assertTrue(tensor_local["torch_is_contiguous"])
        self.assertIn("raise_layout_error", diagnostics["traceback"])

    def test_failed_snapshot_flushes_expected_and_actual_hidden_and_all_unique_kv(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            output = Path(directory) / "probe.json"
            artifacts: dict[str, np.ndarray] = {}
            report: dict[str, object] = {"status": "RUNNING", "snapshot_comparisons": []}
            expected_hidden = np.zeros((4, 1536), dtype=np.uint16)
            actual_hidden = expected_hidden.copy()
            expected_hidden[1, 2] = 0x3F80
            actual_hidden[1, 2] = 0x4000
            expected_kv = {
                layer: (
                    np.full((1, 1, 2, 2), layer, dtype=np.uint16),
                    np.full((1, 1, 2, 2), layer + 1, dtype=np.uint16),
                )
                for layer in range(15)
            }
            actual_kv = {layer: (key.copy(), value.copy()) for layer, (key, value) in expected_kv.items()}
            expected_state = {"generation": 3, "position": 38}
            actual_state = {"generation": 3, "position": 42}
            row = {"session_id": "sid-a", "session_generation": 3, "frame_id": 7}
            counters = {"before": {"forward_calls": 1}, "after": {"forward_calls": 2}}

            with self.assertRaises(ExactParityError):
                _assert_snapshot_with_evidence(
                    output_path=output, tensor_artifacts=artifacts, report=report,
                    comparison_id="N1/sid-a/frame7", row_metadata=row,
                    expected_hidden_bits=expected_hidden, actual_hidden_bits=actual_hidden,
                    expected_kv=expected_kv, actual_kv=actual_kv,
                    expected_state=expected_state, actual_state=actual_state,
                    counters=counters,
                )

            payload = json.loads(output.read_text(encoding="utf-8"))
            record = payload["snapshot_comparisons"][0]
            self.assertEqual(record["status"], "NO-GO")
            self.assertEqual(record["row"], row)
            self.assertEqual(record["counters"], counters)
            self.assertEqual(record["arrays"]["hidden"]["raw_mismatch_count"], 1)
            self.assertEqual(record["arrays"]["hidden"]["first_mismatch_indices"], [[1, 2]])
            self.assertEqual(record["arrays"]["hidden"]["max_abs_diff"], 1.0)
            npz_path = Path(payload["tensor_artifact"]["path"])
            with np.load(npz_path, allow_pickle=False) as saved:
                self.assertTrue(np.array_equal(saved["N1/sid-a/frame7/expected_hidden_bits"], expected_hidden))
                self.assertTrue(np.array_equal(saved["N1/sid-a/frame7/actual_hidden_bits"], actual_hidden))
                for layer in range(15):
                    for side in ("expected", "actual"):
                        for kind in ("key", "value"):
                            self.assertIn(f"N1/sid-a/frame7/{side}/layer{layer}/{kind}_bits", saved.files)
            self.assertEqual(len(record["arrays"]), 31)

    def test_rms_probe_archives_all_eight_rows_before_strict_failure(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            output = Path(directory) / "probe.json"
            expected_input = np.arange(8 * 1536, dtype=np.uint16).reshape(8, 1536)
            expected_output = np.arange(8 * 1536, dtype=np.uint16).reshape(8, 1536)
            actual_output = expected_output.copy()
            actual_output[7, 1535] ^= np.uint16(1)
            report: dict[str, object] = {"status": "RUNNING"}
            artifacts: dict[str, np.ndarray] = {}
            with self.assertRaises(ExactParityError):
                parity_probe._record_native_numerical_probe_stage(
                    output_path=output,
                    report=report,
                    tensor_artifacts=artifacts,
                    stage="rms",
                    metadata={
                        "row_count": 8,
                        "rms_dim": 1536,
                        "rms_backend_id": 2,
                        "rms_batched_block_x": 64,
                        "rms_batched_block_y": 8,
                        "rms_reference_block_x": 128,
                        "rms_reference_block_y": 4,
                        "rms_weight_device_ptr": 111,
                        "forward_generation": 1,
                        "native_weight_pin": {
                            "name": "layers.0.input_layernorm.weight",
                            "data_ptr": 111,
                            "nbytes": 3072,
                            "sha256": "a" * 64,
                        },
                    },
                    raw_arrays={
                        "input_bits": expected_input.tobytes(),
                        "batched_bits": actual_output.tobytes(),
                        "split_bits": expected_output.tobytes(),
                    },
                    comparisons=(("batched_vs_split", "split_bits", "batched_bits"),),
                )
            payload = json.loads(output.read_text(encoding="utf-8"))
            record = payload["native_numerical_diagnostics"][0]
            self.assertEqual(record["status"], "NO-GO")
            self.assertEqual(record["comparisons"]["batched_vs_split"]["raw_mismatch_count"], 1)
            self.assertEqual(record["comparisons"]["batched_vs_split"]["first_mismatch_indices"], [[7, 1535]])
            npz_path = Path(payload["tensor_artifact"]["path"])
            with np.load(npz_path, allow_pickle=False) as saved:
                self.assertTrue(np.array_equal(saved["N2/numerical/rms/input_bits"], expected_input))
                self.assertTrue(np.array_equal(saved["N2/numerical/rms/batched_bits"], actual_output))
                self.assertTrue(np.array_equal(saved["N2/numerical/rms/split_bits"], expected_output))

    def test_qproj_probe_archives_full_rows_and_cublas_dispatch_metadata(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            output = Path(directory) / "probe.json"
            input_bits = np.arange(8 * 1536, dtype=np.uint16).reshape(8, 1536)
            actual = np.arange(8 * 1024, dtype=np.uint16).reshape(8, 1024)
            whole = actual.copy()
            split = actual.copy()
            split[7, 1023] ^= np.uint16(1)
            report: dict[str, object] = {"status": "RUNNING"}
            artifacts: dict[str, np.ndarray] = {}
            with self.assertRaises(ExactParityError):
                parity_probe._record_native_numerical_probe_stage(
                    output_path=output,
                    report=report,
                    tensor_artifacts=artifacts,
                    stage="qproj",
                    metadata={
                        "row_count": 8,
                        "rms_dim": 1536,
                        "rms_backend_id": 2,
                        "rms_weight_device_ptr": 100,
                        "q_weight_device_ptr": 200,
                        "native_weight_pin": {
                            "name": "layers.0.self_attn.q_proj.weight",
                            "data_ptr": 200,
                            "nbytes": 6_291_456,
                            "sha256": "b" * 64,
                        },
                        "forward_generation": 1,
                        "q_dim": 1024,
                        "q_batched_m": 8,
                        "q_split_m": 4,
                        "q_k": 1536,
                        "q_lda": 1536,
                        "q_ldb": 1536,
                        "q_ldc": 1024,
                        "cublas_stream_matches_listener": 1,
                        "listener_stream_handle": 300,
                        "q4_gemv4_opt_in": 0,
                        "q4_gemv4_split0_eligible": 0,
                        "q4_gemv4_split1_eligible": 0,
                    },
                    raw_arrays={
                        "input_bits": input_bits.tobytes(),
                        "actual_bits": actual.tobytes(),
                        "batched_bits": whole.tobytes(),
                        "split_bits": split.tobytes(),
                    },
                    comparisons=(("m8_vs_two_m4", "batched_bits", "split_bits"),),
                )
            payload = json.loads(output.read_text(encoding="utf-8"))
            record = payload["native_numerical_diagnostics"][0]
            self.assertEqual(record["status"], "NO-GO")
            self.assertEqual(record["metadata"]["q_ldc"], 1024)
            self.assertEqual(record["metadata"]["q4_gemv4_opt_in"], 0)
            npz_path = Path(payload["tensor_artifact"]["path"])
            with np.load(npz_path, allow_pickle=False) as saved:
                self.assertEqual(saved["N2/numerical/qproj/input_bits"].shape, (8, 1536))
                self.assertEqual(saved["N2/numerical/qproj/split_bits"].shape, (8, 1024))
                self.assertEqual(saved["N2/numerical/qproj/actual_bits"].shape, (8, 1024))

    def test_native_weight_pin_hashes_the_loaded_parameter_bytes(self):
        import hashlib

        payload = b"exact-loaded-weight-bytes"

        class FakeTensor:
            shape = (3, 4)
            dtype = "torch.bfloat16"
            device = "cuda:0"

            def data_ptr(self):
                return 0xC0FFEE

            def detach(self):
                return self

            def contiguous(self):
                return self

            def view(self, dtype):
                if dtype != "uint8":
                    raise AssertionError(dtype)
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.frombuffer(payload, dtype=np.uint8)

        model = type("Model", (), {"named_parameters": lambda _self: [("layers.0.weight", FakeTensor())]})()
        pin = parity_probe._hash_native_weight_pointer(
            model, 0xC0FFEE, type("Torch", (), {"uint8": "uint8"})(),
        )
        self.assertEqual(pin["name"], "layers.0.weight")
        self.assertEqual(pin["nbytes"], len(payload))
        self.assertEqual(pin["sha256"], hashlib.sha256(payload).hexdigest())
        with self.assertRaisesRegex(RuntimeError, "matched 0"):
            parity_probe._hash_native_weight_pointer(
                model, 0xDEAD, type("Torch", (), {"uint8": "uint8"})(),
            )

    def test_plan_pins_certified_capacity_reference_not_canonical_default(self):
        from pathlib import Path

        workspace = Path("/workspace")
        baseline = _baseline_reference_path(workspace)
        self.assertEqual(
            baseline,
            workspace / ".tmp/listener_sliding_capacity_candidate_20261011/native/"
            "gemma4_listener/libgemma4_listener.so",
        )

    def test_n1_failure_closes_gate_before_n2_or_n4(self):
        gate = ParityPhaseGate()
        self.assertEqual(gate.next_batch_size, 1)
        gate.record(1, passed=False, evidence_id="n1-mismatch")
        self.assertIsNone(gate.next_batch_size)
        with self.assertRaises(RuntimeError):
            gate.record(2, passed=True, evidence_id="must-not-run")

    def test_only_ordered_exact_passes_advance_n1_n2_n4(self):
        gate = ParityPhaseGate()
        for n in (1, 2, 4):
            self.assertEqual(gate.next_batch_size, n)
            gate.record(n, passed=True, evidence_id=f"n{n}-exact")
        self.assertIsNone(gate.next_batch_size)
        self.assertEqual([entry.batch_size for entry in gate.results], [1, 2, 4])

    def test_bitwise_comparison_rejects_same_shape_one_bit_difference(self):
        expected = np.array([0x3F80, 0x0001], dtype=np.uint16)
        candidate = np.array([0x3F80, 0x0000], dtype=np.uint16)
        with self.assertRaises(ExactParityError):
            assert_exact_array("hidden", expected, candidate)

    def test_bitwise_comparison_rejects_shape_or_dtype_change(self):
        expected = np.zeros((4, 1536), dtype=np.uint16)
        with self.assertRaises(ExactParityError):
            assert_exact_array("hidden", expected, expected.reshape(1, 4, 1536))
        with self.assertRaises(ExactParityError):
            assert_exact_array("hidden", expected, expected.astype(np.float32))

    def test_projection_counter_contract_proves_one_gemm_per_projection(self):
        shared = [False] * 15 + [True] * 20
        projection_count = expected_projection_gemm_count(shared)
        before = {
            "forward_calls": 3,
            "layers_executed": 105,
            "projection_gemm_calls": projection_count * 3,
            "projection_rows": projection_count * 40,
            "session_rows": 5,
        }
        after = before | {
            "forward_calls": 4,
            "layers_executed": 140,
            "projection_gemm_calls": projection_count * 4,
            "projection_rows": projection_count * 40 + projection_count * 8,
            "session_rows": 7,
            "output_generation": 9,
        }
        before["output_generation"] = 8
        delta = assert_projection_batch_counters(
            before, after, batch_size=2, layer_count=35,
            shared_kv_layers=shared,
        )
        self.assertEqual(delta["projection_gemm_calls"], projection_count)
        self.assertEqual(delta["projection_rows"], projection_count * 8)
        self.assertEqual(delta["session_rows"], 2)

    def test_counter_check_rejects_hidden_per_session_gemm_loop(self):
        shared = [False] * 15 + [True] * 20
        projection_count = expected_projection_gemm_count(shared)
        before = {key: 0 for key in (
            "forward_calls", "layers_executed", "projection_gemm_calls",
            "projection_rows", "session_rows", "output_generation",
        )}
        after = before | {
            "forward_calls": 1,
            "layers_executed": 35,
            "projection_gemm_calls": projection_count * 2,
            "projection_rows": projection_count * 8,
            "session_rows": 2,
            "output_generation": 1,
        }
        with self.assertRaises(AssertionError):
            assert_projection_batch_counters(
                before, after, batch_size=2, layer_count=35,
                shared_kv_layers=shared,
            )

    def test_snapshot_comparison_covers_cursor_layer_set_and_every_kv_value(self):
        hidden = np.zeros((4, 1536), dtype=np.uint16)
        key0 = np.zeros((1, 1, 34, 512), dtype=np.uint16)
        value0 = np.full_like(key0, 7)
        state = {"slot": 0, "current_position": 34}
        exact = {0: (key0, value0)}
        assert_snapshot_exact(
            hidden, hidden.copy(), exact, {0: (key0.copy(), value0.copy())},
            expected_state=state, actual_state=dict(state),
        )
        with self.assertRaises(ExactParityError):
            assert_snapshot_exact(
                hidden, hidden.copy(), exact, {},
                expected_state=state, actual_state=dict(state),
            )
        changed = value0.copy()
        changed[0, 0, 33, 511] = 6
        with self.assertRaises(ExactParityError):
            assert_snapshot_exact(
                hidden, hidden.copy(), exact, {0: (key0.copy(), changed)},
                expected_state=state, actual_state=dict(state),
            )
        with self.assertRaises(ExactParityError):
            assert_snapshot_exact(
                hidden, hidden.copy(), exact, {0: (key0.copy(), value0.copy())},
                expected_state=state, actual_state={"slot": 0, "current_position": 38},
            )

    def test_gpu_safety_requires_known_owners_and_two_fresh_idle_observations(self):
        now = datetime(2026, 10, 12, 12, 0, 0, tzinfo=timezone.utc)
        doc = {
            "schema_version": 1,
            "serving_container": {
                "container_id": "a" * 64,
                "snapshot_id": "snapshot-42",
                "snapshot_at_utc": "2026-10-12T11:59:10Z",
                "status": "running",
            },
            "gpu_owners": [
                {"known": True, "container_id": "a" * 64},
                {"known": True, "container_id": "b" * 64},
            ],
            "planned_pause_container_id": "a" * 64,
            "planned_restart_container_id": "a" * 64,
            "idle_health_checks": [
                {"status": "healthy", "active_requests": 0,
                 "checked_at_utc": "2026-10-12T11:59:40Z"},
                {"status": "healthy", "active_requests": 0,
                 "checked_at_utc": "2026-10-12T11:59:50Z"},
            ],
        }
        record = validate_gpu_safety_evidence(doc, now=now)
        self.assertEqual(record["serving_container_id"], "a" * 64)
        unknown_owner = {**doc, "gpu_owners": [
            {"known": False, "container_id": "b" * 64},
        ]}
        with self.assertRaises(ValueError):
            validate_gpu_safety_evidence(unknown_owner, now=now)
        one_check = {**doc, "idle_health_checks": doc["idle_health_checks"][:1]}
        with self.assertRaises(ValueError):
            validate_gpu_safety_evidence(one_check, now=now)
        repeated_check = {**doc, "idle_health_checks": [
            doc["idle_health_checks"][0], doc["idle_health_checks"][0],
        ]}
        with self.assertRaises(ValueError):
            validate_gpu_safety_evidence(repeated_check, now=now)


if __name__ == "__main__":
    unittest.main()
