from __future__ import annotations

import unittest
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext

from tools.probe_gemma4_direct_audio_prefill_parity import (
    EXPECTED_AUDIO_SHA256,
    _direct_audio_model_parts,
    _expected_gemma4_cache_layout,
    _failure_runtime,
    _manifest_asset_path,
    _run_whole_repeat_snapshot,
    _safe_json_write,
    _snapshot_layer_counts_comparison,
    _tracked_model_load,
    _refresh_cuda_runtime_state,
    assemble_parity_report,
)


def _passing_comparisons() -> dict[str, object]:
    return {
        "whole_repeat": {"passed": True, "tensors": {"hidden": {"passed": True}}},
        "whole_vs_chunk64": {
            "passed": True,
            "tensors": {"hidden": {"passed": True}, "logits": {"passed": True}},
            "same_final_next_token_argmax": True,
        },
        "mask_query_rows_exact": {"passed": True, "chunk_count": 3},
        "snapshot_layer_counts": _snapshot_layer_counts_comparison(
            {
                "whole": {"layer_count": 36, "cache_layer_count": 15},
                "whole_repeat": {"layer_count": 36, "cache_layer_count": 15},
                "chunk64": {"layer_count": 36, "cache_layer_count": 15},
            },
            expected_transformer_layers=35,
            expected_cache_layers=15,
            expected_kv_shared_layers=20,
        ),
    }


def _workspace_tempdir() -> tempfile.TemporaryDirectory[str]:
    repo = Path(__file__).resolve().parents[2]
    temp_root = (repo / ".tmp" / "direct_audio_parity_tests").resolve()
    if not temp_root.is_relative_to((repo / ".tmp").resolve()):
        raise AssertionError("test fixture directory escaped the workspace temporary root")
    temp_root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=temp_root)


class ParityReportAssemblyTests(unittest.TestCase):
    def test_identically_truncated_layer_snapshots_fail_config_count_gate(self) -> None:
        snapshots = {
            label: {"layer_count": 36, "cache_layer_count": 14}
            for label in ("whole", "whole_repeat", "chunk64")
        }
        count_comparison = _snapshot_layer_counts_comparison(
            snapshots,
            expected_transformer_layers=35,
            expected_cache_layers=15,
            expected_kv_shared_layers=20,
        )
        self.assertFalse(count_comparison["passed"])
        comparisons = _passing_comparisons()
        comparisons["snapshot_layer_counts"] = count_comparison
        report = assemble_parity_report(
            worker_exit_code=0,
            comparisons=comparisons,
            source_attestation={"ready": True},
            fixture={"audio_sha256": EXPECTED_AUDIO_SHA256, "encoder_calls": 1},
        )
        self.assertFalse(report["gates"]["snapshot_layer_counts_match_config"])
        self.assertEqual(report["status"], "failed")

    def test_missing_snapshot_count_fails_closed(self) -> None:
        comparison = _snapshot_layer_counts_comparison(
            {
                "whole": {"layer_count": 36, "cache_layer_count": 15},
                "whole_repeat": {"layer_count": 36, "cache_layer_count": 15},
                "chunk64": {"layer_count": 36},
            },
            expected_transformer_layers=35,
            expected_cache_layers=15,
            expected_kv_shared_layers=20,
        )
        self.assertFalse(comparison["passed"])

    def test_gemma4_kv_cache_count_uses_verified_shared_layer_suffix(self) -> None:
        transformer_layers = 35
        language_model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(is_kv_shared_layer=index >= 15)
                )
                for index in range(transformer_layers)
            ]
        )
        text_config = SimpleNamespace(
            num_hidden_layers=transformer_layers,
            num_kv_shared_layers=20,
            layer_types=["sliding_attention"] * transformer_layers,
        )
        layout = _expected_gemma4_cache_layout(language_model, text_config)
        self.assertEqual(layout["expected_unique_cache_layer_count"], 15)
        self.assertEqual(layout["shared_layer_indices"], list(range(15, 35)))

        snapshots = {
            label: {"layer_count": 36, "cache_layer_count": 15}
            for label in ("whole", "whole_repeat", "chunk64")
        }
        comparison = _snapshot_layer_counts_comparison(
            snapshots,
            expected_transformer_layers=35,
            expected_cache_layers=layout["expected_unique_cache_layer_count"],
            expected_kv_shared_layers=layout["configured_kv_shared_layer_count"],
        )
        self.assertTrue(comparison["passed"])
        self.assertEqual(comparison["validated_shared_layer_indices"], list(range(15, 35)))

    def test_gemma4_kv_cache_layout_rejects_sharing_markers_outside_config_suffix(self) -> None:
        language_model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(is_kv_shared_layer=(index >= 15 and index != 20))
                )
                for index in range(35)
            ]
        )
        text_config = SimpleNamespace(
            num_hidden_layers=35,
            num_kv_shared_layers=20,
            layer_types=["sliding_attention"] * 35,
        )
        with self.assertRaisesRegex(RuntimeError, "markers differ from the config suffix"):
            _expected_gemma4_cache_layout(language_model, text_config)

    def test_unreadable_cuda_state_is_unknown_not_false(self) -> None:
        state = {"cuda_initialized": False, "gpu_used": False}
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_initialized=lambda: (_ for _ in ()).throw(RuntimeError("cuda query failed"))
            )
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            _refresh_cuda_runtime_state(state)
            runtime = _failure_runtime(state)
        self.assertIsNone(runtime["cuda_initialized"])
        self.assertIsNone(runtime["gpu_used"])
        self.assertEqual(runtime["cuda_state_read_error"], "RuntimeError: cuda query failed")
        report = assemble_parity_report(
            worker_exit_code=1,
            worker_error={"type": "RuntimeError", "message": "probe failed"},
            comparisons={},
            source_attestation={"ready": False},
            fixture={},
            runtime=runtime,
        )
        self.assertIsNone(report["gpu_used"])

    def test_whole_repeat_snapshot_uses_loaded_outer_model(self) -> None:
        outer_model = object()
        output = object()
        lm_kwargs = {"inputs_embeds": object()}
        observed: dict[str, object] = {}

        def language_model(**kwargs: object) -> object:
            observed["kwargs"] = kwargs
            return output

        def snapshot(actual_output: object, actual_model: object) -> dict[str, object]:
            observed["output"] = actual_output
            observed["model"] = actual_model
            return {"snapshot": True}

        fake_torch = SimpleNamespace(inference_mode=nullcontext)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch(
                "tools.probe_gemma4_direct_audio_prefill_parity._snapshot_lm_output",
                side_effect=snapshot,
            ):
                result = _run_whole_repeat_snapshot(
                    language_model,
                    lambda: lm_kwargs,
                    outer_model,
                )

        self.assertEqual(result, {"snapshot": True})
        self.assertIs(observed["kwargs"]["inputs_embeds"], lm_kwargs["inputs_embeds"])
        self.assertIs(observed["output"], output)
        self.assertIs(observed["model"], outer_model)

    def test_preflight_failure_reports_no_model_or_cuda_use(self) -> None:
        state = {
            "stage": "preflight_validation",
            "model_load_started": False,
            "model_loaded": False,
            "gpu_used": False,
            "cuda_initialized": False,
            "preflight_manifest": {"path": "/workspace/preflight.json", "sha256": "abc"},
        }
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))}):
            runtime = _failure_runtime(state)
        self.assertEqual(runtime["stage"], "preflight_validation")
        self.assertFalse(runtime["model_load_started"])
        self.assertFalse(runtime["model_loaded"])
        self.assertFalse(runtime["gpu_used"])
        self.assertEqual(runtime["preflight_manifest"]["sha256"], "abc")

    def test_mid_load_failure_retains_started_but_not_loaded_state(self) -> None:
        state = {
            "stage": "torch_import",
            "model_load_started": False,
            "model_loaded": False,
            "gpu_used": False,
            "cuda_initialized": False,
            "preflight_manifest": {"path": "/workspace/preflight.json", "sha256": "abc"},
        }

        def fail_during_load() -> None:
            raise RuntimeError("checkpoint shard read failed")

        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))}):
            with self.assertRaisesRegex(RuntimeError, "checkpoint shard"):
                _tracked_model_load(fail_during_load, state)
            runtime = _failure_runtime(state)
        self.assertEqual(runtime["stage"], "model_loading")
        self.assertTrue(runtime["model_load_started"])
        self.assertFalse(runtime["model_loaded"])
        self.assertFalse(runtime["gpu_used"])
        self.assertEqual(runtime["preflight_manifest"]["sha256"], "abc")

    def test_late_failure_preserves_loaded_model_and_initialized_cuda(self) -> None:
        state = {
            "stage": "torch_import",
            "model_load_started": False,
            "model_loaded": False,
            "gpu_used": False,
            "cuda_initialized": False,
            "preflight_manifest": {"path": "/workspace/preflight.json", "sha256": "abc"},
        }
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(_tracked_model_load(lambda: object(), state).__class__, object)
            state["stage"] = "audio_prefill"
            runtime = _failure_runtime(state)
        report = assemble_parity_report(
            worker_exit_code=1,
            worker_error={"type": "RuntimeError", "message": "audio prefill failed"},
            comparisons={},
            source_attestation={"ready": False},
            fixture={},
            runtime=runtime,
        )
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["model_loaded"])
        self.assertTrue(report["gpu_used"])
        self.assertEqual(report["runtime"]["stage"], "audio_prefill")
        self.assertEqual(report["runtime"]["preflight_manifest"]["sha256"], "abc")

    def test_worker_resolves_audio_encoder_nested_and_mask_builder_outer(self) -> None:
        encoder = lambda *_args, **_kwargs: None
        language_model = lambda **_kwargs: None
        mask_builder = lambda *_args, **_kwargs: None
        nested = SimpleNamespace(
            get_audio_features=encoder,
            language_model=language_model,
        )
        outer = SimpleNamespace(model=nested, create_masks_for_generate=mask_builder)
        resolved_nested, resolved_language_model, resolved_mask_builder = _direct_audio_model_parts(outer)
        self.assertIs(resolved_nested, nested)
        self.assertIs(resolved_language_model, language_model)
        self.assertIs(resolved_mask_builder, mask_builder)

    def test_worker_rejects_audio_api_missing_from_nested_model(self) -> None:
        outer = SimpleNamespace(
            get_audio_features=lambda *_args: None,
            model=SimpleNamespace(language_model=lambda **_kwargs: None),
            create_masks_for_generate=lambda *_args: None,
        )
        with self.assertRaisesRegex(RuntimeError, "nested get_audio_features"):
            _direct_audio_model_parts(outer)

    def test_success_report_requires_all_parity_gates_and_preflight_identity(self) -> None:
        report = assemble_parity_report(
            worker_exit_code=0,
            comparisons=_passing_comparisons(),
            source_attestation={"ready": True, "sha256": {"engine.py": "engine-sha"}},
            fixture={"audio_sha256": EXPECTED_AUDIO_SHA256, "encoder_calls": 1},
        )
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["all_parity_gates_pass"])
        self.assertEqual(report["fixture"]["encoder_calls"], 1)

    def test_tensor_or_mask_mismatch_fails_without_relaxing_gates(self) -> None:
        comparisons = _passing_comparisons()
        comparisons["whole_vs_chunk64"]["tensors"]["hidden"]["passed"] = False
        report = assemble_parity_report(
            worker_exit_code=0,
            comparisons=comparisons,
            source_attestation={"ready": True, "sha256": {"engine.py": "engine-sha"}},
            fixture={"audio_sha256": EXPECTED_AUDIO_SHA256, "encoder_calls": 1},
        )
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["all_parity_gates_pass"])

    def test_nonzero_worker_exit_remains_primary_failure(self) -> None:
        report = assemble_parity_report(
            worker_exit_code=7,
            worker_error={"type": "RuntimeError", "message": "forward failed"},
            comparisons={},
            source_attestation={"ready": True, "sha256": {}},
            fixture={},
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["worker_error"]["message"], "forward failed")
        self.assertEqual(report["worker_exit_code"], 7)

    def test_missing_preflight_provenance_fails_closed(self) -> None:
        report = assemble_parity_report(
            worker_exit_code=0,
            comparisons=_passing_comparisons(),
            source_attestation={"ready": False, "sha256": {}},
            fixture={"audio_sha256": EXPECTED_AUDIO_SHA256, "encoder_calls": 1},
        )
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["all_parity_gates_pass"])

    def test_report_builder_output_is_json_serializable_for_pass_and_failure(self) -> None:
        with _workspace_tempdir() as temporary:
            root = Path(temporary)
            success = assemble_parity_report(
                worker_exit_code=0,
                comparisons=_passing_comparisons(),
                source_attestation={"ready": True},
                fixture={"audio_sha256": EXPECTED_AUDIO_SHA256, "encoder_calls": 1},
                runtime={"gpu_used": True, "model_loaded": True},
            )
            failure = assemble_parity_report(
                worker_exit_code=7,
                worker_error={"type": "RuntimeError", "message": "forward failed"},
                comparisons={},
                source_attestation={"ready": False},
                fixture={"audio_sha256": None, "encoder_calls": 0},
            )
            _safe_json_write(root / "success.json", success)
            _safe_json_write(root / "failure.json", failure)
            self.assertEqual(json.loads((root / "success.json").read_text())["status"], "passed")
            decoded_failure = json.loads((root / "failure.json").read_text())
            self.assertEqual(decoded_failure["status"], "failed")
            self.assertEqual(decoded_failure["worker_error"]["message"], "forward failed")

    def test_manifest_asset_binding_requires_exact_workspace_relative_paths(self) -> None:
        with _workspace_tempdir() as temporary:
            root = Path(temporary).resolve()
            audio = root / "fixture.wav"
            model = root / "model"
            model.mkdir()
            audio.write_bytes(b"fixture")
            record_audio = {"path": "fixture.wav"}
            record_model = {"path": "model"}
            self.assertEqual(_manifest_asset_path(root, record_audio, "audio"), audio)
            self.assertEqual(_manifest_asset_path(root, record_model, "model"), model)
            with self.assertRaisesRegex(RuntimeError, "escapes the workspace"):
                _manifest_asset_path(root, {"path": "../outside.wav"}, "audio")
            with self.assertRaisesRegex(RuntimeError, "workspace-relative"):
                _manifest_asset_path(root, {"path": str(audio)}, "audio")


if __name__ == "__main__":
    unittest.main()
