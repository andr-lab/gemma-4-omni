"""CPU-only contract tests for the isolated FFN quantization probe."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.benchmarks.benchmark_duplex_ffn_quantization import (
    EXPECTED_TORCHAO,
    _variant_builders,
    benchmark_cases,
    find_ffn_tensor_names,
    is_exact_stable_version,
    native_listener_contract,
    parse_layer_indices,
)


class LayerSelectionTests(unittest.TestCase):
    def test_accepts_one_or_two_distinct_layers(self) -> None:
        self.assertEqual(parse_layer_indices("15", num_layers=35), (15,))
        self.assertEqual(parse_layer_indices("14,34", num_layers=35), (14, 34))

    def test_rejects_empty_duplicate_or_out_of_range_layers(self) -> None:
        for value in ("", "15,15", "35", "-1", "1,2,3", "1,,2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_layer_indices(value, num_layers=35)


class CheckpointTensorSelectionTests(unittest.TestCase):
    def test_selects_the_exact_ffn_tensors_for_one_layer(self) -> None:
        keys = {
            f"model.language_model.layers.{layer}.mlp.{projection}.weight"
            for layer in (1, 10, 15)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        found = find_ffn_tensor_names(keys, layer_index=1)
        self.assertEqual(
            found,
            {
                projection: f"model.language_model.layers.1.mlp.{projection}.weight"
                for projection in ("gate_proj", "up_proj", "down_proj")
            },
        )

    def test_fails_closed_on_missing_or_ambiguous_ffn_weights(self) -> None:
        complete = {
            f"model.language_model.layers.15.mlp.{projection}.weight"
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        with self.assertRaises(ValueError):
            find_ffn_tensor_names(complete - {next(iter(complete))}, layer_index=15)
        ambiguous = complete | {
            f"model.decoder.layers.15.mlp.{projection}.weight"
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        with self.assertRaises(ValueError):
            find_ffn_tensor_names(ambiguous, layer_index=15)


class BenchmarkContractTests(unittest.TestCase):
    def test_torchao_probe_pin_is_exact_and_matches_torch_212_support_window(self) -> None:
        self.assertEqual(EXPECTED_TORCHAO, "0.18.0")
        self.assertTrue(is_exact_stable_version("0.18.0", EXPECTED_TORCHAO))
        for version in ("0.18.0rc1", "0.18.0.dev0", "0.18.0.post1", "0.18.0+cu130", "0.18.1"):
            with self.subTest(version=version):
                self.assertFalse(is_exact_stable_version(version, EXPECTED_TORCHAO))

    def test_covers_listener_q4_decode_q1_and_prefill_q4_shapes(self) -> None:
        self.assertEqual(
            [(case.name, case.token_count) for case in benchmark_cases()],
            [("listener_q4", 4), ("thinker_decode_q1", 1), ("thinker_prefill_q4", 4)],
        )

    def test_report_states_native_bf16_abi_boundary(self) -> None:
        contract = native_listener_contract()
        self.assertEqual(contract["query_shape"], [1, 4, 1536])
        self.assertEqual(contract["query_dtype"], "bfloat16")
        self.assertEqual(contract["weight_dtype_required"], "bfloat16")
        self.assertFalse(contract["quantized_shared_ffn_supported"])
        self.assertFalse(contract["native_listener_integration_tested"])

    def test_absent_quantizers_leave_a_valid_bf16_only_probe(self) -> None:
        with (
            patch("tools.benchmarks.benchmark_duplex_ffn_quantization._torchao_configs", return_value=None),
            patch("tools.benchmarks.benchmark_duplex_ffn_quantization._bnb_version", return_value=None),
        ):
            builders, notes = _variant_builders(None, device=None, torch=None)
        self.assertEqual(builders, [])
        self.assertEqual(len(notes), 2)


if __name__ == "__main__":
    unittest.main()
