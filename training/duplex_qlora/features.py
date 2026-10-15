"""Native Gemma listener feature contract shared by stage 2 and runtime."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from duplex_kernel.acoustic_phase_features import ACOUSTIC_PHASE_FEATURE_CONTRACT

FEATURE_CONTRACT = {
    "queries_per_tick": 4,
    "input_dim": 1536,
    "pooling": "last_audio_query",
    "tick_ms": 80,
    "source": "native_gemma_listener",
}


def listener_feature_contract(input_dim: int = 1536) -> dict[str, Any]:
    """Return the strict native-only ABI contract for supported head widths."""
    if isinstance(input_dim, bool) or input_dim not in {1536, 1540}:
        raise ValueError(f"listener head input_dim must be 1536 or 1540, got {input_dim!r}")
    contract = dict(FEATURE_CONTRACT)
    if input_dim == 1536:
        return contract
    contract["input_dim"] = 1540
    contract["acoustic_phase"] = deepcopy(ACOUSTIC_PHASE_FEATURE_CONTRACT)
    return contract


def last_audio_query(hidden: Any):
    """Return query index 3 from a four-query Gemma tick; reject proxy features."""
    shape = tuple(getattr(hidden, "shape", ()))
    if len(shape) == 3 and shape[0] == 1:
        hidden = hidden[0]
        shape = tuple(hidden.shape)
    if len(shape) != 2 or shape != (4, 1536):
        raise ValueError(
            "duplex_control6 richiede quattro query Gemma ordinate con shape "
            f"(4, 1536); ricevuto {shape} (la feature V9 2048 non è compatibile)"
        )
    try:
        import torch

        if not bool(torch.isfinite(hidden).all()):
            raise ValueError("hidden Gemma contiene valori non finiti")
    except TypeError:
        # Permit array-like CPU fixtures without loosening shape validation.
        import numpy as np

        if not bool(np.isfinite(hidden).all()):
            raise ValueError("hidden Gemma contiene valori non finiti")
    return hidden[3]


__all__ = ["FEATURE_CONTRACT", "last_audio_query", "listener_feature_contract"]
