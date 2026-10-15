from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from duplex_kernel.engine import (
    PipelineEngine,
    select_audio_thinker_cache_bucket,
)


@pytest.mark.parametrize(
    ("prefill_tokens", "output_budget", "expected"),
    [
        (384, 128, 512),
        (385, 128, 1024),
        (896, 128, 1024),
        (897, 128, 2048),
        (1920, 128, 2048),
        (1921, 128, 4096),
        (3968, 128, 4096),
    ],
)
def test_audio_thinker_cache_bucket_uses_effective_prefill_and_budget(
    prefill_tokens: int,
    output_budget: int,
    expected: int,
) -> None:
    assert select_audio_thinker_cache_bucket(prefill_tokens, output_budget) == expected


def test_audio_thinker_cache_bucket_fails_instead_of_truncating_over_capacity() -> None:
    with pytest.raises(ValueError, match=r"4129.*4096"):
        select_audio_thinker_cache_bucket(4001, 128)


@pytest.mark.parametrize(
    ("prefill_tokens", "output_budget"),
    [(0, 1), (-1, 1), (1, 0), (True, 1), (1, 1.5)],
)
def test_audio_thinker_cache_bucket_rejects_invalid_lengths(
    prefill_tokens: int,
    output_budget: int,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        select_audio_thinker_cache_bucket(prefill_tokens, output_budget)


def test_static_thinker_kwargs_uses_request_capacity_without_global_mutation() -> None:
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.thinker_static_cache_max_len = 4096
    engine.thinker = SimpleNamespace(config=SimpleNamespace())
    tokens = torch.tensor([[5]], dtype=torch.long)

    kwargs = engine._static_thinker_kwargs(
        tokens,
        object(),
        start_position=384,
        visible_tokens=385,
        cache_capacity=512,
    )

    assert tuple(kwargs["attention_mask"].shape) == (1, 512)
    assert kwargs["attention_mask"][0, :385].sum().item() == 385
    assert kwargs["attention_mask"][0, 385:].sum().item() == 0
    assert kwargs["position_ids"].tolist() == [[384]]
    assert engine.thinker_static_cache_max_len == 4096
