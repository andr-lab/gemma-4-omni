from __future__ import annotations

import pytest
import torch

from duplex_kernel.future_prefix import _SlotKVCache


def _cache_write(
    cache: _SlotKVCache,
    layer: int,
    positions: list[int],
) -> None:
    position_tensor = torch.tensor(positions, dtype=torch.long)
    values = torch.ones((1, 1, len(positions), 2), dtype=torch.float32)
    cache.update(
        values,
        values,
        layer,
        {"cache_position": position_tensor},
    )


def test_absolute_cache_length_tracks_joint_prefix_and_audio_positions() -> None:
    cache = _SlotKVCache(batch_size=1, max_cache_len=16)

    # Fixed layout: stable prefix_start=2, two prefix slots, audio_start=4.
    _cache_write(cache, 0, [0, 1])
    assert cache.get_seq_length() == 2
    _cache_write(cache, 0, [2, 4])  # joint frame 0: prefix row 2 + audio row 4
    assert cache.get_seq_length() == 5
    _cache_write(cache, 0, [3, 5])  # joint frame 1: prefix row 3 + audio row 5

    # The next absolute Q1 position is audio_start + frame_id = 4 + 2 = 6.
    assert cache.get_seq_length() == 6


def test_absolute_cache_length_is_independent_per_layer_for_append_fallback() -> None:
    cache = _SlotKVCache(batch_size=1, max_cache_len=16)
    key = torch.ones((1, 1, 2, 2), dtype=torch.float32)

    cache.update(key, key, 0)
    assert cache.get_seq_length(0) == 2
    assert cache.get_seq_length(1) == 0

    cache.update(key, key, 1)
    assert cache.get_seq_length(0) == 2
    assert cache.get_seq_length(1) == 2

    one = torch.ones((1, 1, 1, 2), dtype=torch.float32)
    cache.update(one, one, 0)
    cache.update(one, one, 1)
    assert cache.get_seq_length(0) == 3
    assert cache.get_seq_length(1) == 3


def test_explicit_positions_update_each_layer_and_reset_clears_lengths() -> None:
    cache = _SlotKVCache(batch_size=1, max_cache_len=8)
    for layer in (0, 1):
        _cache_write(cache, layer, [0, 1])
        _cache_write(cache, layer, [2, 4])
    assert cache.get_seq_length(0) == 5
    assert cache.get_seq_length(1) == 5

    cache.reset()
    assert cache.get_seq_length(0) == 0
    assert cache.get_seq_length(1) == 0
    assert cache._max_position == -1


def test_empty_position_update_does_not_change_cache_length() -> None:
    cache = _SlotKVCache(batch_size=1, max_cache_len=8)
    _cache_write(cache, 0, [3])
    empty = torch.empty((1, 1, 0, 2), dtype=torch.float32)
    cache.update(empty, empty, 0, {"cache_position": torch.empty((0,), dtype=torch.long)})
    assert cache.get_seq_length(0) == 4


@pytest.mark.parametrize("position", [-1, 8])
def test_explicit_position_bounds_remain_fail_closed(position: int) -> None:
    cache = _SlotKVCache(batch_size=1, max_cache_len=8)
    with pytest.raises(OverflowError):
        _cache_write(cache, 0, [position])
    assert cache.get_seq_length(0) == 0
    assert cache._max_position == -1

