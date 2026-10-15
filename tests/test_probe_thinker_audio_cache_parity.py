from __future__ import annotations

import torch
import pytest

from tools.probe_thinker_audio_cache_parity import _compare_passes, _compare_tensors


def _pass(*, say_span=(0, 4), offset=0.0):
    return {
        "label": "fixture",
        "cache_metrics": {"thinker_cache_reused_tokens": 1},
        "logits": torch.tensor([[0.1 + offset, 0.9 - offset, -0.5 + offset]]),
        "prefill_hidden": torch.tensor([[[0.0, 0.0], [1.0 + offset, 2.0 + offset]]]),
        "generated_hidden": torch.tensor([[[3.0 + offset, 4.0 + offset]]]),
        "generated_ids": [7],
        "say_layout": [{"token_index": 0, "token_id": 7, "char_span": list(say_span)}],
        "say_tokens": [{"hidden": torch.tensor([[[5.0 + offset, 6.0 + offset]]])}],
        "protocol": {"status": "valid"},
    }


def _cached(*, say_span=(0, 4), offset=0.0):
    result = _pass(say_span=say_span, offset=offset)
    result["prefill_hidden"] = result["prefill_hidden"][:, 1:, :]
    return result


def test_cache_parity_requires_logits_hidden_token_and_say_span_alignment() -> None:
    fresh = _pass()
    cached = _cached(offset=0.001)

    matched = _compare_passes(fresh, cached, prefix_len=1, atol=0.01, rtol=0.01)

    assert matched["status"] == "pass"
    assert matched["top5_token_ids_equal"] is True
    assert matched["say_layout_equal"] is True
    assert matched["prefix_tokens_reused"] == matched["expected_text_prefix_tokens"] == 1

    shifted_say = _compare_passes(
        fresh, _cached(say_span=(1, 5)), prefix_len=1, atol=0.01, rtol=0.01,
    )
    assert shifted_say["status"] == "fail"
    assert shifted_say["say_layout_equal"] is False

    changed_logits = _compare_passes(
        fresh, _cached(offset=0.5), prefix_len=1, atol=0.01, rtol=0.01,
    )
    assert changed_logits["status"] == "fail"
    assert changed_logits["top5_token_ids_equal"] is False


def test_tensor_comparison_persists_relative_l2_cosine_and_max_vector_norms() -> None:
    compared = _compare_tensors(
        torch.tensor([[3.0, 4.0]]), torch.tensor([[3.0, 5.0]]), atol=0.01, rtol=0.01,
    )

    assert compared["within_tolerance"] is False
    assert compared["l2_error"] == 1.0
    assert compared["relative_l2_error"] == 0.2
    assert compared["cosine_similarity"] == torch.nn.functional.cosine_similarity(
        torch.tensor([[3.0, 4.0]]), torch.tensor([[3.0, 5.0]]), dim=1,
    ).item()
    assert compared["max_vector_l2_error"] == 1.0
    assert compared["reference_max_vector_l2_norm"] == 5.0
    assert compared["candidate_max_vector_l2_norm"] == pytest.approx(torch.sqrt(torch.tensor(34.0)).item())

    mismatch = _compare_tensors(torch.zeros((1, 2)), torch.zeros((1, 3)), atol=0.01, rtol=0.01)
    assert mismatch["within_tolerance"] is False
    assert mismatch["l2_error"] is None
    assert mismatch["cosine_similarity"] is None
