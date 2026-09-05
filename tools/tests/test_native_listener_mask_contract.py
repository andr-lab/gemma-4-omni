"""Regression tests for the native listener's fail-closed prime contract."""
from __future__ import annotations

import pytest
import torch

from duplex_kernel.native_listener import NativeGemmaListenerLM


def test_prime_requires_canonical_ple_and_attention_mask() -> None:
    listener = object.__new__(NativeGemmaListenerLM)
    prefix = torch.empty((1, 34, 1536), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="PLE e prefix_attention espliciti"):
        listener._validate_prime_inputs(prefix, None, None)


def test_prime_requires_mask_even_when_ple_is_present() -> None:
    listener = object.__new__(NativeGemmaListenerLM)
    prefix = torch.empty((1, 34, 1536), dtype=torch.bfloat16)
    ple = torch.empty((1, 34, 35, 256), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="PLE e prefix_attention espliciti"):
        listener._validate_prime_inputs(prefix, ple, None)
