from __future__ import annotations

import pytest
import torch
from transformers import PretrainedConfig, StaticCache

from duplex_kernel.thinker_cache import (
    CachePrefixCopyError,
    copy_static_cache_prefix,
    longest_common_prefix_length,
)


def _tiny_static_cache(*, max_cache_len: int = 8) -> StaticCache:
    config = PretrainedConfig()
    config.num_hidden_layers = 2
    config.layer_types = ["full_attention", "full_attention"]
    config.sliding_window = None
    config.attention_chunk_size = None
    return StaticCache(
        config=config,
        max_cache_len=max_cache_len,
        max_batch_size=1,
        device="cpu",
        dtype=torch.float32,
    )


def _fill_static_cache(cache: StaticCache, length: int) -> None:
    for layer_index, layer in enumerate(cache.layers):
        keys = torch.arange(length * 8, dtype=torch.float32).reshape(1, 2, length, 4)
        values = keys + 1000.0 + layer_index
        cache.update(keys, values, layer_index)


def test_longest_common_prefix_is_exact_and_stops_at_first_difference() -> None:
    assert longest_common_prefix_length([11, 12, 13], [11, 12, 99, 13]) == 2
    assert longest_common_prefix_length([11, 12], [11, 12, 13]) == 2
    assert longest_common_prefix_length([], [11]) == 0


def test_static_cache_prefix_copy_keeps_storage_layout_and_isolates_decode_branch() -> None:
    source = _tiny_static_cache()
    _fill_static_cache(source, 5)
    target = _tiny_static_cache()

    copied = copy_static_cache_prefix(source, target, 3)

    assert copied.copied_tokens == 3
    assert copied.layer_count == 2
    assert copied.copied_bytes == 2 * 2 * 1 * 2 * 3 * 4 * 4
    assert int(target.get_seq_length()) == 3
    for source_layer, target_layer in zip(source.layers, target.layers, strict=True):
        assert source_layer.keys.shape == target_layer.keys.shape
        assert torch.equal(target_layer.keys[:, :, :3], source_layer.keys[:, :, :3])
        assert torch.equal(target_layer.values[:, :, :3], source_layer.values[:, :, :3])
        assert torch.count_nonzero(target_layer.keys[:, :, 3:]) == 0

    source_prefix = source.layers[0].keys[:, :, :3].clone()
    next_key = torch.full((1, 2, 1, 4), -7.0)
    target.update(next_key, next_key, 0)

    assert int(source.get_seq_length()) == 5
    assert int(target.get_seq_length()) == 4
    assert torch.equal(source.layers[0].keys[:, :, :3], source_prefix)
    assert not torch.equal(source.layers[0].keys[:, :, 3:4], next_key)


def test_static_cache_prefix_copy_fails_closed_when_source_is_too_short() -> None:
    source = _tiny_static_cache()
    _fill_static_cache(source, 2)
    target = _tiny_static_cache()

    with pytest.raises(CachePrefixCopyError, match="source cache length"):
        copy_static_cache_prefix(source, target, 3)

