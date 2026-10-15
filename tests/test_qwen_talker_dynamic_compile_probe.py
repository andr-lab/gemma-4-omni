import numpy as np

from tools.benchmarks.benchmark_qwen_talker_dynamic_compile import (
    compiler_capture_summary,
    exact_array_equal,
    parity_decision,
    step_parity_pass,
    tensor_metrics,
)


def test_exact_array_equality_handles_numpy_inputs_and_shape_mismatch():
    reference = np.asarray([[1.0, 2.0]], dtype=np.float32)

    assert exact_array_equal(reference, reference.copy()) is True
    assert exact_array_equal(reference, np.asarray([[1.0, 2.01]], dtype=np.float32)) is False
    assert exact_array_equal(reference, np.asarray([1.0, 2.0], dtype=np.float32)) is False


def test_tensor_metrics_report_scale_aware_errors_and_pointwise_failures():
    reference = np.asarray([10.0, 0.5, -4.0], dtype=np.float32)
    candidate = np.asarray([10.05, 0.55, -3.98], dtype=np.float32)

    result = tensor_metrics(reference, candidate, pointwise_atol=0.02, pointwise_rtol=0.02)

    assert result["shape_equal"] is True
    assert result["all_finite"] is True
    assert result["relative_l2"] < 0.02
    assert result["cosine_similarity"] > 0.999
    assert result["pointwise_failed_count"] == 1


def test_parity_gate_requires_numerical_bounds_and_exact_greedy_ids():
    metrics = tensor_metrics(
        np.asarray([1.0, 3.0, -2.0], dtype=np.float32),
        np.asarray([1.001, 3.002, -2.001], dtype=np.float32),
    )

    accepted = parity_decision(metrics=metrics, reference_ids=[44], candidate_ids=[44])
    changed_token = parity_decision(metrics=metrics, reference_ids=[44], candidate_ids=[45])

    assert accepted["pass"] is True
    assert changed_token["numerical_ok"] is True
    assert changed_token["pass"] is False


def test_parity_gate_rejects_numerical_divergence_even_when_token_ids_match():
    metrics = tensor_metrics(
        np.asarray([10.0, 0.01], dtype=np.float32),
        np.asarray([9.0, 1.0], dtype=np.float32),
    )

    decision = parity_decision(metrics=metrics, reference_ids=[7], candidate_ids=[7])

    assert decision["greedy_ids_equal"] is True
    assert decision["numerical_ok"] is False
    assert decision["pass"] is False


def test_step_gate_keeps_lower_rank_and_residual_differences_diagnostic_only():
    good = {
        "hidden_decision": {"pass": True},
        "cb0_decision": {"pass": True},
        "cb0_top5": {"ids_equal": True},
        "subtalker_residual_ids": {"all_15_equal": True},
        "all_16_codebook_ids_equal": True,
        "cache_mutation_one_token_each": True,
        "eager_returned_same_cache_object": True,
        "compiled_returned_same_cache_object": True,
    }
    numerical_differences = {
        **good,
        "cb0_top5": {"ids_equal": False},
        "subtalker_residual_ids": {"all_15_equal": False},
        "all_16_codebook_ids_equal": False,
    }

    assert step_parity_pass(good) is True
    assert step_parity_pass(numerical_differences) is True


def test_step_gate_still_rejects_greedy_cb0_change():
    good = {
        "hidden_decision": {"pass": True},
        "cb0_decision": {"pass": True},
        "cache_mutation_one_token_each": True,
        "eager_returned_same_cache_object": True,
        "compiled_returned_same_cache_object": True,
    }

    assert step_parity_pass({**good, "cb0_decision": {"pass": False}}) is False


def test_capture_summary_labels_partial_capture_and_leaves_recompile_count_unknown():
    before = {"unique_graphs": 0, "calls_captured": 0, "graph_break_count": 0}
    after = {"unique_graphs": 2, "calls_captured": 7, "graph_break_count": 3}

    result = compiler_capture_summary(before, after)

    assert result["q1_graph_captured"] is True
    assert result["partial_capture_with_fallback"] is True
    assert result["unique_graphs_delta"] == 2
    assert result["calls_captured_delta"] == 7
    assert result["recompile_count"] is None


def test_capture_summary_makes_zero_graph_compilation_inconclusive():
    stats = {"unique_graphs": 0, "calls_captured": 0, "graph_break_count": 0}

    result = compiler_capture_summary(stats, stats)

    assert result["q1_graph_captured"] is False
    assert result["capture_status"] == "no_graph_captured"
