import numpy as np
import pytest

from tools.benchmarks.benchmark_qwen_talker_static_decode import (
    compare_tensors,
    select_cache_capacity,
)


def test_static_cache_bucket_covers_prefill_and_decode_without_truncation():
    assert select_cache_capacity(prefill_tokens=470, decode_steps=16) == 512
    assert select_cache_capacity(prefill_tokens=497, decode_steps=16) == 1024


def test_static_cache_bucket_fails_closed_when_no_bucket_fits():
    with pytest.raises(ValueError, match="exceeds largest static cache bucket"):
        select_cache_capacity(prefill_tokens=1000, decode_steps=25)


def test_tensor_parity_reports_bf16_sized_differences_and_top_id():
    reference = np.asarray([[0.0, 1.0, -2.0]], dtype=np.float32)
    candidate = np.asarray([[0.001, 1.002, -1.999]], dtype=np.float32)

    result = compare_tensors(reference, candidate, atol=0.01, rtol=0.01)

    assert result["shape_equal"] is True
    assert result["allclose"] is True
    assert result["max_abs_diff"] == pytest.approx(0.002, abs=1e-7)
    assert result["reference_argmax"] == result["candidate_argmax"] == 1


def test_tensor_parity_rejects_top_id_change_even_with_close_values():
    reference = np.asarray([[1.0, 0.999]], dtype=np.float32)
    candidate = np.asarray([[0.999, 1.0]], dtype=np.float32)

    result = compare_tensors(reference, candidate, atol=0.01, rtol=0.01)

    assert result["allclose"] is True
    assert result["reference_argmax"] == 0
    assert result["candidate_argmax"] == 1
    assert result["top_id_equal"] is False
