"""Frozen-runtime bridge from Gemma response states to a Qwen prefix."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


RESPONSE_INTENT_INPUT_DIM = 1536
RESPONSE_INTENT_OUTPUT_DIM = 2048
RESPONSE_INTENT_PREFIX_CAPACITY = 256


class ResponseIntentSafetyCapError(RuntimeError):
    """A non-EOS 256-frame cutoff that must remain a failed response."""

    def __init__(self, generated_frames: int, runtime_metrics: dict[str, Any] | None) -> None:
        super().__init__(
            "response_intent_repeat_v1_safety_cap_without_natural_eos: "
            f"reached {int(generated_frames)} RVQ frames"
        )
        self.runtime_metrics = dict(runtime_metrics or {})
        self.runtime_metrics_phase = "talker"


class ResponseIntentProjector(nn.Module):
    """The trained linear map; keep parameter names compatible with its checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(RESPONSE_INTENT_INPUT_DIM, RESPONSE_INTENT_OUTPUT_DIM)

    def forward(self, h_intent: torch.Tensor) -> torch.Tensor:
        if h_intent.ndim not in (2, 3) or int(h_intent.shape[-1]) != RESPONSE_INTENT_INPUT_DIM:
            raise ValueError(
                f"h_intent must be (B,{RESPONSE_INTENT_INPUT_DIM}) or "
                f"(B,T,{RESPONSE_INTENT_INPUT_DIM}), got {tuple(h_intent.shape)}"
            )
        return self.linear(h_intent)


def pool_response_intent(response_hidden: torch.Tensor) -> torch.Tensor:
    """Pool final-normalized hiddens from the generated SAY span only."""
    if (
        not isinstance(response_hidden, torch.Tensor)
        or response_hidden.ndim != 3
        or int(response_hidden.shape[0]) != 1
        or int(response_hidden.shape[1]) <= 0
        or int(response_hidden.shape[2]) != RESPONSE_INTENT_INPUT_DIM
    ):
        shape = tuple(response_hidden.shape) if isinstance(response_hidden, torch.Tensor) else None
        raise ValueError(f"response_hidden must be (1,T,{RESPONSE_INTENT_INPUT_DIM}), got {shape}")
    if not bool(torch.isfinite(response_hidden).all()):
        raise ValueError("response_hidden contains non-finite values")
    return (response_hidden.mean(dim=1) + response_hidden[:, -1, :]) * 0.5


def build_response_intent_prefix(
    response_hidden: torch.Tensor,
    projector: ResponseIntentProjector,
    *,
    capacity: int = RESPONSE_INTENT_PREFIX_CAPACITY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return h_intent and its repeated fixed-capacity conditioning prefix."""
    if isinstance(capacity, bool) or int(capacity) != RESPONSE_INTENT_PREFIX_CAPACITY:
        raise ValueError(
            f"response_intent_repeat_v1 requires prefix capacity {RESPONSE_INTENT_PREFIX_CAPACITY}"
        )
    h_intent = pool_response_intent(response_hidden)
    projected = projector(h_intent)
    if projected.shape != (1, RESPONSE_INTENT_OUTPUT_DIM):
        raise ValueError(f"projector output must be (1,{RESPONSE_INTENT_OUTPUT_DIM})")
    if not bool(torch.isfinite(projected).all()):
        raise ValueError("response-intent projector returned non-finite values")
    repeated = projected.unsqueeze(1).expand(1, RESPONSE_INTENT_PREFIX_CAPACITY, -1)
    return h_intent, repeated


def load_response_intent_projector(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> ResponseIntentProjector:
    """Load the exact trained linear checkpoint and freeze it for serving."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"response-intent checkpoint not found: {path}")
    payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("response-intent checkpoint must be a mapping")
    contract = payload.get("training_contract")
    if not isinstance(contract, dict) or int(contract.get("prefix_capacity", -1)) != RESPONSE_INTENT_PREFIX_CAPACITY:
        raise ValueError("response-intent checkpoint was not trained for fixed prefix capacity 256")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("response-intent checkpoint has no state_dict")
    model = ResponseIntentProjector()
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)
    return model


def response_intent_termination(
    *,
    eos_detected: bool,
    cancelled: bool,
    generated_frames: int,
    frame_cap: int = RESPONSE_INTENT_PREFIX_CAPACITY,
) -> str:
    """Label natural EOS, cancellation, or the candidate's hard safety cutoff."""
    if eos_detected:
        return "eos"
    if cancelled:
        return "cancel"
    if int(generated_frames) >= int(frame_cap):
        return "safety_cap"
    return "incomplete"


def response_intent_frame_budget(*, ordinary_budget: int, is_backchannel: bool) -> int:
    """Use a 256-frame SPEAK safety cap and retain the established short BC budget."""
    if is_backchannel:
        return max(1, min(32, int(ordinary_budget)))
    return RESPONSE_INTENT_PREFIX_CAPACITY
