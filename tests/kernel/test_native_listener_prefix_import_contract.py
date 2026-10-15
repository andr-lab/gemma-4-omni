"""CPU contract checks for native cache imports with actual semantic prefixes."""
from __future__ import annotations

import pytest
import torch

from duplex_kernel.native_listener import NativeGemmaListenerLM


def _uninitialized_provider() -> NativeGemmaListenerLM:
    provider = object.__new__(NativeGemmaListenerLM)
    provider.stream_q_len = 4
    provider._import_cache_with_prefix = object()
    return provider


def test_prefix_import_requires_length_and_mask_as_a_pair() -> None:
    provider = _uninitialized_provider()
    with pytest.raises(ValueError, match="devono essere forniti insieme"):
        provider.import_cache_async(object(), current_position=141, prefix_len=141)


def test_prefix_import_rejects_non_q4_aligned_cursor_before_cuda_access() -> None:
    provider = _uninitialized_provider()
    with pytest.raises(ValueError, match="multiplo della lunghezza Q4"):
        provider.import_cache_async(
            object(),
            current_position=142,
            prefix_len=141,
            prefix_mask=torch.ones((1, 141), dtype=torch.int64),
        )


def test_prefix_import_bounds_actual_length_to_native_workspace() -> None:
    provider = _uninitialized_provider()
    with pytest.raises(ValueError, match="compreso tra 1 e 291"):
        provider.import_cache_async(
            object(),
            current_position=292,
            prefix_len=292,
            prefix_mask=torch.ones((1, 292), dtype=torch.int64),
        )


def test_prefix_import_reaches_cuda_tensor_validation(monkeypatch) -> None:
    provider = _uninitialized_provider()
    monkeypatch.setattr(
        NativeGemmaListenerLM,
        "_canonicalize_cuda_tensor",
        classmethod(lambda _cls, tensor, **_kwargs: tensor),
    )
    with pytest.raises(ValueError, match="deve essere CUDA"):
        provider.import_cache_async(
            object(),
            current_position=3,
            prefix_len=3,
            prefix_mask=torch.ones((1, 3), dtype=torch.int64),
        )
