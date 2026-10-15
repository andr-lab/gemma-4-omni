from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools.probe_qwen_native_film_parity import (
    PARITY_LIMITS,
    CACHE_PRODUCER_MODULES,
    DEFAULT_ADAPTER_CHECKPOINT,
    EXPECTED_NATIVE_ABI,
    EXPECTED_NATIVE_TRACE_ABI,
    EXPECTED_NATIVE_LIBRARY_SHA256,
    SOURCE_RELATIVE_PATHS,
    _Q1CacheCursorError,
    _capture_eager_last_layer,
    _native_tap_comparisons,
    _cache_producer_source_attestation,
    _assemble_gpu_parity_report,
    compare_tensor_pair,
    ensure_workspace_import_path,
    make_cpu_fixture,
    finalize_gpu_parity_report,
    gpu_parity_exit_code,
    main,
    q1_cache_cursor_diagnostic,
    run_gpu_probe,
    sha256_file,
    validate_audio_q1_contract,
    validate_gpu_worker_report,
    verify_cache_producer_runtime_sources,
    verify_cpu_preflight_handoff,
)


class QueryFilmNativeParityProbeTests(unittest.TestCase):
    def _isolated_fixture_workspace(self, workspace: Path) -> Path:
        workspace = workspace.resolve()
        repository = Path(__file__).resolve().parents[2]
        temp_root = Path(tempfile.gettempdir()).resolve()
        repository_temp_root = (repository / ".tmp").resolve()
        self.assertTrue(
            workspace.is_relative_to(temp_root) or workspace.is_relative_to(repository_temp_root),
            f"fixture workspace must stay under an explicit temp root: {workspace}",
        )
        self.assertFalse(
            workspace == repository
            or repository.is_relative_to(workspace)
            or workspace.is_relative_to(repository / "native"),
            f"fixture workspace must be isolated from the repository: {workspace}",
        )
        return workspace

    def _cpu_preflight_handoff_fixture(self, workspace: Path):
        workspace = self._isolated_fixture_workspace(workspace)
        self.assertTrue(workspace.is_dir(), f"fixture workspace must already exist: {workspace}")
        source_hashes = {
            path: {"exists": True, "sha256": f"source-{index}"}
            for index, path in enumerate(SOURCE_RELATIVE_PATHS)
        }
        checkpoint = DEFAULT_ADAPTER_CHECKPOINT.relative_to(Path(__file__).resolve().parents[2]).as_posix()
        source_hashes[checkpoint] = {"exists": True, "sha256": "adapter-checkpoint"}
        modules = {
            name: {
                "available": True,
                "path": str(workspace / "qwen3tts_src" / (name.replace(".", "/") + ".py"))
                if name.startswith("qwen_tts.")
                else str(workspace / "site-packages" / (name.replace(".", "/") + ".py")),
                "exists": True,
                "sha256": f"external-{index}",
            }
            for index, name in enumerate(CACHE_PRODUCER_MODULES)
        }
        modules[CACHE_PRODUCER_MODULES[0]]["inside_qwen3tts_src"] = True
        producer_sources = {
            "qwen3tts_source_root": str(workspace / "qwen3tts_src"),
            "modules": modules,
            "ready": True,
        }
        native = {
            "path": str(workspace / "native" / "qwen_talker" / "libqwen_talker.so"),
            "exists": True,
            "expected_sha256": EXPECTED_NATIVE_LIBRARY_SHA256,
            "sha256": EXPECTED_NATIVE_LIBRARY_SHA256,
            "sha256_matches_expected": True,
            "abi_version": EXPECTED_NATIVE_ABI,
            "trace_abi_version": EXPECTED_NATIVE_TRACE_ABI,
            "wrapper_symbols_checked": [],
            "missing_wrapper_symbols": [],
            "load_error": None,
            "ready": True,
        }
        attestation = {
            "workspace": str(workspace),
            "source_hashes": source_hashes,
            "missing_sources": [],
            "cache_producer_sources": producer_sources,
            "native_library": native,
        }
        snapshot = {
            "revision": "fixed-revision",
            "path": str(workspace / ".hf_cache" / "fixed-revision"),
            "ready": True,
            "sha256": {"config.json": "config-hash", "model.safetensors": "weights-hash"},
        }
        preflight = {
            "mode": "cpu_preflight",
            "ready_for_explicit_gpu_probe": True,
            "gpu_used": False,
            "model_loaded": False,
            "attestation": attestation,
            "qwen_talker_snapshot": snapshot,
        }
        return workspace, preflight, copy.deepcopy(attestation), copy.deepcopy(snapshot)

    def test_eager_capture_accepts_serving_talker_modulelist_and_hooks_final_layer(self) -> None:
        class FakeAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Identity()
                self.k_proj = torch.nn.Identity()
                self.q_norm = torch.nn.Identity()
                self.k_norm = torch.nn.Identity()

            def forward(self, hidden):
                query = self.q_norm(self.q_proj(hidden))
                key = self.k_norm(self.k_proj(hidden))
                return apply_multimodal_rotary_pos_emb(query, key, None, None, None, False)

        class FakeLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttention()

        decoder = SimpleNamespace(layers=torch.nn.ModuleList([FakeLayer(), FakeLayer()]))
        serving_talker = SimpleNamespace(talker=SimpleNamespace(model=decoder))
        # Match run_gpu_probe's production extraction: TalkerModel.talker.model.
        model = getattr(getattr(serving_talker, "talker", serving_talker), "model", None)
        rotary_impl = lambda query, key, *_args: (query + 1, key + 1)
        with patch.dict(globals(), {"apply_multimodal_rotary_pos_emb": rotary_impl}):
            capture, restore = _capture_eager_last_layer(model)
            try:
                query, key = decoder.layers[-1].self_attn(torch.tensor([[2.0]]))
            finally:
                restore()
            self.assertIs(FakeAttention.forward.__globals__["apply_multimodal_rotary_pos_emb"], rotary_impl)
            self.assertFalse(decoder.layers[-1].self_attn.q_proj._forward_hooks)

        self.assertEqual(capture["target_layer_index"], 1)
        self.assertEqual(capture["rotary_call_count"], 1)
        for name in ("q_proj", "k_proj", "q_norm", "k_norm", "q_rope", "k_rope"):
            self.assertIn(name, capture)
        self.assertTrue(torch.equal(capture["q_rope"], query))
        self.assertTrue(torch.equal(capture["k_rope"], key))

    def test_native_tap_comparison_reports_missing_rope_keys_without_keyerror(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "eager Q/K/RoPE tap capture is incomplete: missing=.*q_rope.*k_rope",
        ):
            _native_tap_comparisons(SimpleNamespace(), {"q_proj": torch.ones((1, 1))})

    def test_eager_capture_fails_closed_for_missing_or_empty_layers(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not expose decoder layers"):
            _capture_eager_last_layer(SimpleNamespace())
        with self.assertRaisesRegex(RuntimeError, "does not expose decoder layers"):
            _capture_eager_last_layer(SimpleNamespace(layers=torch.nn.ModuleList()))

    def test_cpu_preflight_handoff_accepts_identical_source_assets_and_dso(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace, preflight, current, snapshot = self._cpu_preflight_handoff_fixture(Path(temporary))
            verify_cpu_preflight_handoff(
                preflight,
                workspace=workspace,
                native_library=preflight["attestation"]["native_library"]["path"],
                current_attestation=current,
                current_snapshot=snapshot,
            )

    def test_cpu_preflight_handoff_rejects_source_mutated_after_preflight(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace, preflight, current, snapshot = self._cpu_preflight_handoff_fixture(Path(temporary))
            current["source_hashes"]["duplex_kernel/future_prefix.py"]["sha256"] = "mutated"
            with self.assertRaisesRegex(RuntimeError, "differs from CPU preflight"):
                verify_cpu_preflight_handoff(
                    preflight,
                    workspace=workspace,
                    native_library=preflight["attestation"]["native_library"]["path"],
                    current_attestation=current,
                    current_snapshot=snapshot,
                )

    def test_external_source_attestation_uses_talker_imported_module_without_dotted_find_spec(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = self._isolated_fixture_workspace(Path(temporary))
            for module_name in CACHE_PRODUCER_MODULES:
                package_root = workspace / "qwen3tts_src"
                module_path = package_root.joinpath(*module_name.split(".")).with_suffix(".py")
                module_path.parent.mkdir(parents=True, exist_ok=True)
                module_path.write_text("source only\n", encoding="utf-8")
            sibling_modules = {
                "transformers.models.qwen3_vl.modeling_qwen3_vl": (
                    workspace / "site-packages" / "transformers" / "models" / "qwen3_vl" / "modeling_qwen3_vl.py"
                ),
                "transformers.models.qwen3_moe.modeling_qwen3_moe": (
                    workspace / "site-packages" / "transformers" / "models" / "qwen3_moe" / "modeling_qwen3_moe.py"
                ),
            }
            for sibling_path in sibling_modules.values():
                sibling_path.parent.mkdir(parents=True, exist_ok=True)
                sibling_path.write_text("sibling package source\n", encoding="utf-8")

            with patch.dict(
                sys.modules,
                {
                    CACHE_PRODUCER_MODULES[0]: SimpleNamespace(__file__=str(module_path)),
                    **{
                        name: SimpleNamespace(__file__=str(path))
                        for name, path in sibling_modules.items()
                    },
                },
            ), patch.object(sys, "path", [str(workspace / "site-packages"), str(workspace / "qwen3tts_src")]), patch(
                "tools.probe_qwen_native_film_parity._module_available",
                side_effect=AssertionError("dotted find_spec may import package parents"),
            ):
                attestation = _cache_producer_source_attestation(workspace)

            self.assertTrue(attestation["ready"])
            self.assertTrue(attestation["modules"][CACHE_PRODUCER_MODULES[0]]["inside_qwen3tts_src"])
            self.assertEqual(set(attestation["modules"]), {CACHE_PRODUCER_MODULES[0]})

    def test_gpu_source_mismatch_is_rejected_before_torch_import(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = self._isolated_fixture_workspace(Path(temporary))
            workspace, preflight, current, snapshot = self._cpu_preflight_handoff_fixture(workspace)
            library = Path(preflight["attestation"]["native_library"]["path"])
            library.parent.mkdir(parents=True)
            library.write_bytes(b"fixture dso")
            current["source_hashes"]["duplex_kernel/future_prefix.py"]["sha256"] = "mutated"
            manifest_path = workspace / "preflight.json"
            manifest_path.write_text(json.dumps(preflight), encoding="utf-8")
            handoff_record: dict[str, object] = {}
            torch_imports: list[str] = []
            original_import = __import__

            def import_spy(name, *args, **kwargs):
                if name == "torch":
                    torch_imports.append(name)
                    raise AssertionError("torch imported before source handoff was checked")
                return original_import(name, *args, **kwargs)

            with patch(
                "tools.probe_qwen_native_film_parity.source_attestation",
                return_value=current,
            ), patch(
                "tools.probe_qwen_native_film_parity._snapshot_files",
                return_value=(workspace, snapshot),
            ), patch("builtins.__import__", side_effect=import_spy):
                with self.assertRaisesRegex(RuntimeError, "differs from CPU preflight"):
                    run_gpu_probe(
                        workspace=workspace,
                        native_library=library,
                        output_path=workspace / "gpu.json",
                        preflight_manifest_path=manifest_path,
                        cpu_preflight_handoff_record=handoff_record,
                    )

            self.assertEqual(torch_imports, [])
            self.assertEqual(handoff_record, {})

    def test_gpu_dso_file_identity_mismatch_is_rejected_before_torch_import(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace, preflight, current, snapshot = self._cpu_preflight_handoff_fixture(Path(temporary))
            library = Path(preflight["attestation"]["native_library"]["path"])
            library.parent.mkdir(parents=True, exist_ok=True)
            library.write_bytes(b"fixture dso")
            current["native_library"]["sha256"] = "mutated-dso-hash"
            current["native_library"]["sha256_matches_expected"] = False
            manifest_path = workspace / "preflight.json"
            manifest_path.write_text(json.dumps(preflight), encoding="utf-8")
            torch_imports: list[str] = []
            original_import = __import__

            def import_spy(name, *args, **kwargs):
                if name == "torch":
                    torch_imports.append(name)
                    raise AssertionError("torch imported before DSO file identity was checked")
                return original_import(name, *args, **kwargs)

            with patch(
                "tools.probe_qwen_native_film_parity.source_attestation",
                return_value=current,
            ), patch(
                "tools.probe_qwen_native_film_parity._snapshot_files",
                return_value=(workspace, snapshot),
            ), patch("builtins.__import__", side_effect=import_spy):
                with self.assertRaisesRegex(RuntimeError, "native DSO file identity differs"):
                    run_gpu_probe(
                        workspace=workspace,
                        native_library=library,
                        output_path=workspace / "gpu.json",
                        preflight_manifest_path=manifest_path,
                    )

            self.assertEqual(torch_imports, [])

    def test_gpu_loads_torch_before_native_abi_attestation_but_before_cuda_or_model(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace, preflight, current, snapshot = self._cpu_preflight_handoff_fixture(Path(temporary))
            library = Path(preflight["attestation"]["native_library"]["path"])
            library.parent.mkdir(parents=True, exist_ok=True)
            library.write_bytes(b"fixture dso")
            current_native = current["native_library"]
            file_identity = {
                key: current_native[key]
                for key in ("path", "exists", "expected_sha256", "sha256", "sha256_matches_expected")
            }
            current["native_library"] = file_identity
            manifest_path = workspace / "preflight.json"
            manifest_path.write_text(json.dumps(preflight), encoding="utf-8")
            handoff_record: dict[str, object] = {}
            events: list[str] = []
            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: (_ for _ in ()).throw(AssertionError("CUDA checked before ABI gate"))
                )
            )
            invalid_runtime_native = copy.deepcopy(current_native)
            invalid_runtime_native["abi_version"] = EXPECTED_NATIVE_ABI - 1

            def import_spy(name, *args, **kwargs):
                if name == "torch":
                    events.append("torch")
                    return fake_torch
                return __import__(name, *args, **kwargs)

            def native_attestation_spy(*args, **kwargs):
                events.append("native_abi")
                return invalid_runtime_native

            with patch(
                "tools.probe_qwen_native_film_parity.source_attestation",
                return_value=current,
            ) as source_attestation, patch(
                "tools.probe_qwen_native_film_parity._snapshot_files",
                return_value=(workspace, snapshot),
            ), patch(
                "tools.probe_qwen_native_film_parity._native_binary_attestation",
                side_effect=native_attestation_spy,
            ), patch("builtins.__import__", side_effect=import_spy):
                with self.assertRaisesRegex(RuntimeError, "native ABI/DSO differs from CPU preflight"):
                    run_gpu_probe(
                        workspace=workspace,
                        native_library=library,
                        output_path=workspace / "gpu.json",
                        preflight_manifest_path=manifest_path,
                        cpu_preflight_handoff_record=handoff_record,
                    )

            self.assertEqual(events, ["torch", "native_abi"])
            self.assertEqual(
                handoff_record,
                {
                    "path": str(manifest_path.resolve()),
                    "sha256": sha256_file(manifest_path),
                    "created_at_unix_ns": None,
                },
            )
            source_attestation.assert_called_once_with(
                workspace,
                library,
                expected_library_sha256=EXPECTED_NATIVE_LIBRARY_SHA256,
                include_native_abi=False,
            )

    def test_gpu_cli_requires_cpu_preflight_manifest_before_running_probe(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            with patch("tools.probe_qwen_native_film_parity.run_gpu_probe") as run_probe, patch(
                "tools.probe_qwen_native_film_parity._write_json"
            ) as write_json, patch("builtins.print"):
                exit_code = main(
                    [
                        "--run-gpu",
                        "--workspace",
                        str(workspace),
                        "--output",
                        str(workspace / "missing-preflight.json"),
                    ]
                )

            self.assertEqual(exit_code, 2)
            run_probe.assert_not_called()
            failure_payload = write_json.call_args.args[1]
            self.assertIn("cpu preflight manifest", failure_payload["error"].lower())

    def test_cpu_preflight_attests_only_cache_producer_modules_loaded_by_talker_import(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = self._isolated_fixture_workspace(Path(temporary))
            qwen_path = (
                workspace
                / "qwen3tts_src"
                / "qwen_tts"
                / "core"
                / "models"
                / "modeling_qwen3_tts.py"
            )
            transformers_path = (
                workspace
                / "site-packages"
                / "transformers"
                / "models"
                / "qwen3"
                / "modeling_qwen3.py"
            )
            qwen_path.parent.mkdir(parents=True)
            transformers_path.parent.mkdir(parents=True)
            qwen_path.write_text("qwen cache producer source\n", encoding="utf-8")
            transformers_path.write_text("transformers cache consumer source\n", encoding="utf-8")
            qwen_module = SimpleNamespace(__file__=str(qwen_path))
            with patch.dict(sys.modules, {CACHE_PRODUCER_MODULES[0]: qwen_module}), patch.object(
                sys, "path", [str(workspace / "qwen3tts_src"), str(workspace / "site-packages")]
            ):
                attestation = _cache_producer_source_attestation(workspace)

            self.assertTrue(attestation["ready"])
            self.assertEqual(set(attestation["modules"]), {CACHE_PRODUCER_MODULES[0]})
            self.assertEqual(attestation["modules"][CACHE_PRODUCER_MODULES[0]]["path"], str(qwen_path))
            self.assertEqual(attestation["modules"][CACHE_PRODUCER_MODULES[0]]["sha256"], sha256_file(qwen_path))
            self.assertTrue(transformers_path.is_file())
            verify_cache_producer_runtime_sources(attestation, attestation)

    def test_cache_producer_attestation_fails_closed_when_required_talker_module_is_missing(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = self._isolated_fixture_workspace(Path(temporary))
            missing = _cache_producer_source_attestation(workspace)
            self.assertFalse(missing["ready"])
            with self.assertRaisesRegex(RuntimeError, "missing required cache-producer module"):
                verify_cache_producer_runtime_sources(missing, missing)

    def test_cache_producer_runtime_source_hash_change_is_reported(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = self._isolated_fixture_workspace(Path(temporary))
            qwen_path = workspace / "qwen3tts_src/qwen_tts/core/models/modeling_qwen3_tts.py"
            qwen_path.parent.mkdir(parents=True)
            qwen_path.write_text("qwen cache producer source\n", encoding="utf-8")
            qwen_module = SimpleNamespace(__file__=str(qwen_path))
            with patch.dict(sys.modules, {CACHE_PRODUCER_MODULES[0]: qwen_module}):
                expected = _cache_producer_source_attestation(workspace)
                qwen_path.write_text("changed source\n", encoding="utf-8")
                current = _cache_producer_source_attestation(workspace)
            with self.assertRaisesRegex(RuntimeError, "source hash differs"):
                verify_cache_producer_runtime_sources(expected, current)

    def test_cache_producer_runtime_source_path_change_is_reported(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = self._isolated_fixture_workspace(Path(temporary))
            first_path = workspace / "qwen3tts_src/qwen_tts/core/models/modeling_qwen3_tts.py"
            second_path = workspace / "qwen3tts_src/qwen_tts/core/models/alternate_model.py"
            first_path.parent.mkdir(parents=True)
            second_path.parent.mkdir(parents=True, exist_ok=True)
            first_path.write_text("qwen source\n", encoding="utf-8")
            second_path.write_text("qwen source\n", encoding="utf-8")
            with patch.dict(sys.modules, {CACHE_PRODUCER_MODULES[0]: SimpleNamespace(__file__=str(first_path))}):
                expected = _cache_producer_source_attestation(workspace)
            with patch.dict(sys.modules, {CACHE_PRODUCER_MODULES[0]: SimpleNamespace(__file__=str(second_path))}):
                current = _cache_producer_source_attestation(workspace)
            with self.assertRaisesRegex(RuntimeError, "module path differs"):
                verify_cache_producer_runtime_sources(expected, current)

    def test_q1_diagnostic_distinguishes_audio_ordinal_from_absolute_cache_cursor(self) -> None:
        from duplex_kernel.future_prefix import _SlotKVCache

        cache = _SlotKVCache(batch_size=1, max_cache_len=16)
        events = []
        for phase, frame_id, positions in (
            ("stable", None, [0, 1]),
            ("frame", 0, [2, 4]),
            ("frame", 1, [3, 5]),
        ):
            position_tensor = torch.tensor(positions, dtype=torch.long)
            values = torch.ones((1, 1, len(positions), 2), dtype=torch.float32)
            cache.update(values, values, 0, {"cache_position": position_tensor})
            events.append({"phase": phase, "frame_id": frame_id, "cache_position": position_tensor})

        reference = SimpleNamespace(
            _current_frame_id=2,
            _audio_start=4,
            _prefix_start=2,
            _slot_cache=cache,
        )
        diagnostic = q1_cache_cursor_diagnostic(reference, events)

        self.assertEqual(diagnostic["audio_frame_ordinal"], 2)
        self.assertEqual(diagnostic["next_audio_absolute_position"], 6)
        self.assertEqual(diagnostic["last_observed_write_max_plus_one"], 6)
        self.assertEqual(diagnostic["cache_sequence_length_before_q1"], 6)
        self.assertTrue(diagnostic["cache_cursor_matches_next_audio_position"])

    def test_gpu_failure_json_keeps_q1_cache_cursor_diagnostic(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            manifest_path = workspace / "unused-preflight.json"
            handoff = {
                "path": str(manifest_path),
                "sha256": "consumed-manifest-sha",
                "created_at_unix_ns": 123,
            }
            diagnostic = {
                "audio_frame_ordinal": 2,
                "next_audio_absolute_position": 6,
                "cache_sequence_length_before_q1": 0,
                "last_observed_write_max_plus_one": 6,
            }
            failure = _Q1CacheCursorError("cursor mismatch", diagnostic)

            def fail_after_preflight_handoff(**kwargs):
                kwargs["cpu_preflight_handoff_record"].update(handoff)
                raise failure

            with patch(
                "tools.probe_qwen_native_film_parity.run_gpu_probe",
                side_effect=fail_after_preflight_handoff,
            ), patch("tools.probe_qwen_native_film_parity._write_json") as write_json, patch(
                "builtins.print"
            ):
                exit_code = main(
                    [
                        "--run-gpu",
                        "--workspace",
                        str(workspace),
                        "--preflight-manifest",
                        str(manifest_path),
                        "--output",
                        str(workspace / "cursor-failure.json"),
                    ]
                )

            self.assertEqual(exit_code, 2)
            failure_payload = write_json.call_args.args[1]
            self.assertEqual(failure_payload["q1_cache_diagnostic"], diagnostic)
            self.assertEqual(failure_payload["cpu_preflight_manifest"], handoff)

    def test_tap_mismatch_cannot_be_reported_as_gpu_parity_success(self) -> None:
        pre_fix_report = {
            "status": "complete",
            "all_last_layer_qk_tap_gates_pass": False,
            "all_final_parity_gates_pass": True,
        }
        self.assertTrue(
            pre_fix_report.get("status") == "complete"
            and pre_fix_report.get("all_final_parity_gates_pass")
        )
        report = finalize_gpu_parity_report(pre_fix_report)

        self.assertFalse(report["all_parity_gates_pass"])
        self.assertEqual(report["status"], "parity_gate_failed")
        self.assertEqual(gpu_parity_exit_code(report), 3)

    def test_all_gpu_parity_gates_must_pass_for_success(self) -> None:
        report = finalize_gpu_parity_report(
            {
                "status": "complete",
                "all_last_layer_qk_tap_gates_pass": True,
                "all_final_parity_gates_pass": True,
            }
        )

        self.assertTrue(report["all_parity_gates_pass"])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(gpu_parity_exit_code(report), 0)

    def test_gpu_report_assembly_preserves_verified_sources_and_comparison_gates(self) -> None:
        source_attestation = {
            "cache_producer_sources": {
                "ready": True,
                "modules": {
                    "qwen_tts.core.models.modeling_qwen3_tts": {
                        "loaded": True,
                        "path": "/workspace/qwen3tts_src/qwen_tts/core/models/modeling_qwen3_tts.py",
                        "sha256": "qwen-source-sha",
                    }
                },
            },
            "native_library": {
                "path": "/workspace/native/qwen_talker/libqwen_talker.so",
                "sha256": EXPECTED_NATIVE_LIBRARY_SHA256,
                "abi_version": EXPECTED_NATIVE_ABI,
                "trace_abi_version": EXPECTED_NATIVE_TRACE_ABI,
            },
        }
        manifest = {
            "path": "/workspace/artifacts/cpu_preflight.json",
            "sha256": "preflight-sha",
            "created_at_unix_ns": 123,
        }
        taps = {
            name: {"passed": True, "mae": 0.0, "max_abs": 0.0, "cosine": 1.0}
            for name in ("q_proj", "k_proj", "q_norm", "k_norm", "q_rope", "k_rope")
        }
        final_comparisons = {
            name: {"passed": True, "mae": 0.0, "max_abs": 0.0, "cosine": 1.0}
            for name in ("last_hidden", "last_layer_audio_key_row", "last_layer_audio_value_row")
        }
        report = _assemble_gpu_parity_report(
            created_at_unix_ns=100,
            completed_at_unix_ns=200,
            qwen_talker_revision="fixed-revision",
            qwen_snapshot_path="/workspace/.hf_cache/fixed-revision",
            qwen_snapshot_files={"ready": True, "sha256": {"config.json": "config-sha"}},
            source_attestation=source_attestation,
            cpu_preflight_manifest=manifest,
            adapter={"learned_scale": 0.05, "requires_grad": False, "inference_mode": True},
            fixture={"synthetic_h_txt": True, "delta": {"finite": True, "rms": 0.01}},
            q1_cursor={"frame_id": 2, "position_ids": [2, 2, 2]},
            eager_attention_taps_final_layer={"q_rope": {"finite": True}},
            native_vs_eager_last_layer_qk_taps=taps,
            native_vs_eager_output_and_cache=final_comparisons,
            native_runtime={"library_sha256": EXPECTED_NATIVE_LIBRARY_SHA256, "native_calls": 1},
        )

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["cpu_preflight_manifest"], manifest)
        self.assertEqual(
            report["external_python_source_hashes"],
            {"qwen_tts.core.models.modeling_qwen3_tts": "qwen-source-sha"},
        )
        self.assertTrue(report["all_last_layer_qk_tap_gates_pass"])
        self.assertTrue(report["all_final_parity_gates_pass"])
        self.assertEqual(gpu_parity_exit_code(report), 0)

        taps["q_rope"]["passed"] = False
        failed = _assemble_gpu_parity_report(
            created_at_unix_ns=100,
            completed_at_unix_ns=200,
            qwen_talker_revision="fixed-revision",
            qwen_snapshot_path="/workspace/.hf_cache/fixed-revision",
            qwen_snapshot_files={"ready": True},
            source_attestation=source_attestation,
            cpu_preflight_manifest=manifest,
            adapter={"learned_scale": 0.05},
            fixture={"synthetic_h_txt": True},
            q1_cursor={"frame_id": 2},
            eager_attention_taps_final_layer={},
            native_vs_eager_last_layer_qk_taps=taps,
            native_vs_eager_output_and_cache=final_comparisons,
            native_runtime={"native_calls": 1},
        )
        self.assertEqual(failed["status"], "parity_gate_failed")
        self.assertEqual(gpu_parity_exit_code(failed), 3)

    def test_failed_worker_error_remains_primary_without_success_only_source_hash(self) -> None:
        report = {
            "status": "failed_closed",
            "error": "NameError: name 'external_sources' is not defined",
            "cpu_preflight_manifest": {
                "path": "/workspace/artifacts/cpu_preflight.json",
                "sha256": "preflight-sha",
            },
        }
        summary = validate_gpu_worker_report(
            2,
            report,
            expected_cpu_preflight_sha256="preflight-sha",
            expected_probe_sha256="probe-sha",
        )

        self.assertFalse(summary["worker_completed"])
        self.assertEqual(summary["primary_error"], report["error"])
        self.assertTrue(summary["cpu_preflight_manifest_matches"])
        self.assertEqual(summary["probe_source_validation"], "not_applicable_worker_failed")

    def test_successful_worker_without_probe_source_hash_fails_closed(self) -> None:
        report = {
            "status": "complete",
            "all_parity_gates_pass": True,
            "cpu_preflight_manifest": {"sha256": "preflight-sha"},
            "source_attestation": {"source_hashes": {}},
        }
        with self.assertRaisesRegex(RuntimeError, "GPU worker source hash differs"):
            validate_gpu_worker_report(
                0,
                report,
                expected_cpu_preflight_sha256="preflight-sha",
                expected_probe_sha256="probe-sha",
            )

    def test_missing_tap_gate_fails_closed(self) -> None:
        report = finalize_gpu_parity_report(
            {"status": "complete", "all_final_parity_gates_pass": True}
        )

        self.assertFalse(report["all_parity_gates_pass"])
        self.assertEqual(report["status"], "parity_gate_failed")
        self.assertEqual(gpu_parity_exit_code(report), 3)

    def test_fixed_adapter_fixture_is_cpu_bf16_inference_only_and_repeatable(self) -> None:
        checkpoint = Path(DEFAULT_ADAPTER_CHECKPOINT)
        self.assertTrue(checkpoint.is_file(), checkpoint)

        first = make_cpu_fixture(checkpoint, seed=7319)
        second = make_cpu_fixture(checkpoint, seed=7319)

        self.assertTrue(first["inference_mode"])
        self.assertEqual(first["device"], "cpu")
        self.assertEqual(tuple(first["response_hidden"].shape), (1, 1, 1536))
        self.assertEqual(tuple(first["base_query"].shape), (1, 1, 2048))
        self.assertEqual(first["response_hidden"].dtype, torch.bfloat16)
        self.assertEqual(first["base_query"].dtype, torch.bfloat16)
        self.assertEqual(first["adapted_query"].dtype, torch.bfloat16)
        self.assertFalse(first["response_hidden"].requires_grad)
        self.assertFalse(first["base_query"].requires_grad)
        self.assertFalse(first["adapted_query"].requires_grad)
        self.assertTrue(torch.equal(first["response_hidden"], second["response_hidden"]))
        self.assertTrue(torch.equal(first["base_query"], second["base_query"]))
        self.assertTrue(torch.equal(first["adapted_query"], second["adapted_query"]))
        self.assertIn("mean", first["summaries"]["adapted_query"])
        self.assertIn("rms", first["summaries"]["adapted_query"])
        self.assertIn("learned_scale", first)

    def test_parity_metrics_use_existing_qwen_thresholds_and_reject_divergence(self) -> None:
        expected = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.bfloat16)
        same = compare_tensor_pair("same", expected, expected.clone())
        self.assertTrue(same["passed"])
        self.assertEqual(same["tolerances"], PARITY_LIMITS)

        different = compare_tensor_pair(
            "different",
            expected,
            torch.tensor([[[-1.0, 0.0, 1.0]]], dtype=torch.bfloat16),
        )
        self.assertFalse(different["passed"])

    def test_audio_q1_contract_requires_one_bf16_query_at_matching_position(self) -> None:
        query = torch.zeros((1, 1, 2048), dtype=torch.bfloat16)
        position_ids = torch.full((3, 1, 1), 19, dtype=torch.long)
        cache_position = torch.tensor([19], dtype=torch.long)

        self.assertEqual(
            validate_audio_q1_contract(query, position_ids, cache_position, expected_position=19),
            {"audio_position": 19, "frame_zero_preserved": True},
        )
        with self.assertRaises(ValueError):
            validate_audio_q1_contract(query, position_ids, cache_position, expected_position=18)
        with self.assertRaises(ValueError):
            validate_audio_q1_contract(query.float(), position_ids, cache_position, expected_position=19)

    def test_sha256_helper_reads_exact_file_bytes(self) -> None:
        checkpoint = Path(DEFAULT_ADAPTER_CHECKPOINT)
        self.assertEqual(len(sha256_file(checkpoint)), 64)

    def test_workspace_path_bootstrap_enables_script_by_path_imports(self) -> None:
        search_path: list[str] = []
        expected = ensure_workspace_import_path(Path.cwd(), search_path)
        expected_qwen = str(Path(expected) / "qwen3tts_src")
        self.assertIn(expected, search_path)
        if Path(expected_qwen).is_dir():
            self.assertEqual(search_path[0], expected_qwen)
        ensure_workspace_import_path(Path.cwd(), search_path)
        self.assertEqual(len(search_path), 2 if Path(expected_qwen).is_dir() else 1)

    def test_qwen_shim_source_path_is_added_without_installing_a_package(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = self._isolated_fixture_workspace(Path(temporary))
            source = workspace / "qwen3tts_src"
            source.mkdir()
            search_path: list[str] = []
            ensure_workspace_import_path(workspace, search_path)
            self.assertEqual(search_path, [str(source.resolve()), str(workspace.resolve())])


if __name__ == "__main__":
    unittest.main()
