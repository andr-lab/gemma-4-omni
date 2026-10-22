"""Native Gemma listener feature contract shared by stage 2 and runtime."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from duplex_kernel.acoustic_phase_features import ACOUSTIC_PHASE_FEATURE_CONTRACT
from duplex_kernel.gemma_representation import AUDIO_FEATURES_PER_FRAME
from duplex_kernel.listener_microbatch_frontend import MEL_WINDOW_ROWS
from duplex_kernel.native_control6_phase import NATIVE_CONTROL6_NATIVE1540_PHASE_T_NAMES

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


def native_control6_feature_contract(input_dim: int = 1536) -> dict[str, Any]:
    """Return the Control6 ABI consumed by the current production listener.

    This is deliberately distinct from the historical Q4 pooled-query feature
    contract above. Runtime Control6 consumes ``ListenerOutput.gemma_hidden_t``;
    the production listener generates two Gemma audio rows per 80 ms tick from
    a W240 Mel window and returns the first current-frame LM hidden as that
    canonical tensor.
    """
    if isinstance(input_dim, bool) or input_dim not in {1536, 1540}:
        raise ValueError(f"native Control6 feature input_dim must be 1536 or 1540, got {input_dim!r}")
    contract: dict[str, Any] = {
        "queries_per_tick": AUDIO_FEATURES_PER_FRAME,
        "input_dim": int(input_dim),
        "pooling": "ListenerOutput.gemma_hidden_t[0]",
        "tick_ms": 80,
        "source": "native_gemma_listener",
        "hidden_contract": "gemma4_listener_final_norm_frame_v1",
        "hidden_source": "ListenerOutput.gemma_hidden_t",
        "hidden_norm": "gemma_model.language_model.norm",
        "audio_tower_queries_per_tick": AUDIO_FEATURES_PER_FRAME,
        "mel_window_rows": MEL_WINDOW_ROWS,
    }
    if input_dim == 1540:
        contract["phase_profile"] = "native1540"
        contract["phase_schema"] = {
            "profile": "native1540",
            "width": 4,
            "names": list(NATIVE_CONTROL6_NATIVE1540_PHASE_T_NAMES),
        }
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


__all__ = [
    "FEATURE_CONTRACT",
    "last_audio_query",
    "listener_feature_contract",
    "native_control6_feature_contract",
]
