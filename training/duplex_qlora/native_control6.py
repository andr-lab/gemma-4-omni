"""Native Gemma output-row LoRA for six-way conversational control.

This module adapts only the six reserved output rows. The caller supplies the
actual tied Gemma output-embedding rows and final-normalized listener hidden
states; no separate classifier or response-generation adapter is involved.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .schema import ACTIONS

SCHEMA_VERSION = 1
MODEL_ID = "google/gemma-4-E2B-it"
HIDDEN_CONTRACT = "gemma4_listener_final_norm_frame_v1"
HIDDEN_SOURCE = "ListenerOutput.gemma_hidden_t"
HIDDEN_NORM = "gemma_model.language_model.norm"
HIDDEN_WIDTH = 1536
PHASE_T_WIDTH = 4
DELTA_INPUT_WIDTH = HIDDEN_WIDTH + PHASE_T_WIDTH
TICK_MS = 80
VOCAB_SIZE = 262144
BASE_WEIGHT_SOURCE = "lm_head.weight (tied to model.language_model.embed_tokens.weight)"

# The six rows are intentionally mapped by action name. Token order does not
# imply action order; token 256000 is Gemma's <|audio> token and is excluded.
ACTION_TO_TOKEN_ID: dict[str, int] = {
    "LISTEN": 256001,       # <unused89>
    "SPEAK": 256002,        # <unused90>
    "CONTINUE": 256003,     # <unused91>
    "YIELD": 256004,        # <unused92>
    "BACKCHANNEL": 256005,  # <unused93>
    "REPLAN": 256006,       # <unused94>
}
TOKEN_ID_TO_STRING: dict[int, str] = {
    256001: "<unused89>",
    256002: "<unused90>",
    256003: "<unused91>",
    256004: "<unused92>",
    256005: "<unused93>",
    256006: "<unused94>",
}

if set(ACTION_TO_TOKEN_ID) != set(ACTIONS):  # fail at import if enum drifted
    raise RuntimeError("native Control6 token map must explicitly cover schema.ACTIONS")
if len(set(ACTION_TO_TOKEN_ID.values())) != len(ACTIONS):
    raise RuntimeError("native Control6 action token ids must be unique")
if 256000 in ACTION_TO_TOKEN_ID.values():
    raise RuntimeError("<|audio> must never be used as a Control6 action token")


def action_token_ids(actions: tuple[str, ...] | list[str] = ACTIONS) -> tuple[int, ...]:
    """Return token ids in the requested action-name order, never enum indices."""
    names = tuple(actions)
    if len(names) != len(ACTIONS) or set(names) != set(ACTIONS):
        raise ValueError(f"actions must be a permutation of {ACTIONS}, got {names}")
    return tuple(ACTION_TO_TOKEN_ID[name] for name in names)


def select_tied_output_rows(weight_or_module: Any) -> torch.Tensor:
    """Select approved rows from Gemma's actual output head.

    Gemma 4 E2B config has `tie_word_embeddings=true`; pass `lm_head.weight`
    (or its tied `model.language_model.embed_tokens.weight`).
    """
    weight = getattr(weight_or_module, "weight", weight_or_module)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError("expected the actual two-dimensional Gemma output embedding weight")
    if weight.shape[1] != HIDDEN_WIDTH or weight.shape[0] < VOCAB_SIZE:
        raise ValueError(
            f"expected tied Gemma embedding with shape [>= {VOCAB_SIZE}, {HIDDEN_WIDTH}], "
            f"got {tuple(weight.shape)}"
        )
    ids = torch.tensor(action_token_ids(), dtype=torch.long, device=weight.device)
    return weight.index_select(0, ids).detach().clone()


def base_rows_bundle(weight_or_module: Any, *, model_id: str = MODEL_ID) -> dict[str, Any]:
    """Build a small, auditable bundle from an already-loaded Gemma model."""
    if model_id != MODEL_ID:
        raise ValueError(f"unexpected model id {model_id!r}; expected {MODEL_ID!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture": "native_gemma_tied_output_rows",
        "model_id": model_id,
        "base_weight_source": BASE_WEIGHT_SOURCE,
        "hidden_size": HIDDEN_WIDTH,
        "vocab_size": VOCAB_SIZE,
        "action_to_token_id": dict(ACTION_TO_TOKEN_ID),
        "token_id_to_string": {str(k): v for k, v in TOKEN_ID_TO_STRING.items()},
        "actions": list(ACTIONS),
        "base_rows": select_tied_output_rows(weight_or_module).cpu(),
    }


def validate_base_rows_bundle(bundle: Mapping[str, Any]) -> torch.Tensor:
    expected_map = dict(ACTION_TO_TOKEN_ID)
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported native Control6 base-row schema")
    if bundle.get("architecture") != "native_gemma_tied_output_rows":
        raise ValueError("base rows are not marked as native tied Gemma output rows")
    if bundle.get("model_id") != MODEL_ID:
        raise ValueError(f"base-row model mismatch: {bundle.get('model_id')!r}")
    if bundle.get("base_weight_source") != BASE_WEIGHT_SOURCE:
        raise ValueError("base rows must come from the actual tied Gemma lm_head.weight")
    if bundle.get("action_to_token_id") != expected_map:
        raise ValueError("base-row action/token map mismatch")
    if bundle.get("actions") != list(ACTIONS):
        raise ValueError("base-row action ordering mismatch")
    rows = bundle.get("base_rows")
    if not isinstance(rows, torch.Tensor) or tuple(rows.shape) != (len(ACTIONS), HIDDEN_WIDTH):
        raise ValueError(f"base_rows must have shape [{len(ACTIONS)}, {HIDDEN_WIDTH}]")
    if not rows.is_floating_point() or not torch.isfinite(rows).all():
        raise ValueError("base rows must be finite floating-point weights")
    return rows


class NativeControl6LoRA(nn.Module):
    """Rank-r delta on six native LM-head rows; all base weights stay frozen."""

    def __init__(self, base_rows: torch.Tensor, *, rank: int = 16, alpha: float = 16.0) -> None:
        super().__init__()
        if tuple(base_rows.shape) != (len(ACTIONS), HIDDEN_WIDTH):
            raise ValueError(
                f"base_rows must have shape [{len(ACTIONS)}, {HIDDEN_WIDTH}], got {tuple(base_rows.shape)}"
            )
        if not base_rows.is_floating_point() or not torch.isfinite(base_rows).all():
            raise ValueError("base_rows must be finite floating-point weights")
        if rank <= 0 or rank > HIDDEN_WIDTH:
            raise ValueError(f"rank must be in [1, {HIDDEN_WIDTH}]")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        # No gradient and no serialization of the large base model's selected rows.
        self.register_buffer("base_rows", base_rows.detach().clone(), persistent=False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        # Gemma's frozen tied-output projection remains 1536-wide. Only the
        # trainable delta receives the four causal phase values.
        self.lora_A = nn.Parameter(torch.empty(self.rank, DELTA_INPUT_WIDTH, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(len(ACTIONS), self.rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(
        self,
        normalized_hidden: torch.Tensor,
        *,
        phase_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(normalized_hidden, torch.Tensor) or normalized_hidden.ndim < 1:
            raise TypeError("normalized_hidden must be a tensor with a feature dimension")
        if normalized_hidden.shape[-1] != HIDDEN_WIDTH:
            raise ValueError(
                f"Gemma final-normalized hidden width must be {HIDDEN_WIDTH}, "
                f"got {tuple(normalized_hidden.shape)}"
            )
        if not torch.isfinite(normalized_hidden).all():
            raise ValueError("normalized_hidden contains non-finite values")
        hidden = normalized_hidden.to(dtype=torch.float32)
        if phase_t is None:
            phase = torch.zeros(
                (*hidden.shape[:-1], PHASE_T_WIDTH),
                dtype=torch.float32,
                device=hidden.device,
            )
        else:
            if not isinstance(phase_t, torch.Tensor) or tuple(phase_t.shape) != (
                *hidden.shape[:-1], PHASE_T_WIDTH
            ):
                raise ValueError(
                    f"phase_t shape must be {(*hidden.shape[:-1], PHASE_T_WIDTH)}"
                )
            if not torch.isfinite(phase_t).all():
                raise ValueError("phase_t contains non-finite values")
            phase = phase_t.to(device=hidden.device, dtype=torch.float32)
        # PyTorch documents F.linear as y = x A^T + b:
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
        # Reproduce the native Gemma output-projection input/weight dtype for
        # frozen logits, then accumulate the learned delta in FP32.
        native_hidden = normalized_hidden.to(dtype=self.base_rows.dtype)
        base = F.linear(native_hidden, self.base_rows).to(dtype=torch.float32)
        delta_input = torch.cat((hidden, phase), dim=-1)
        delta = F.linear(F.linear(delta_input, self.lora_A), self.lora_B) * self.scaling
        return base + delta

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {"lora_A": self.lora_A.detach().cpu(), "lora_B": self.lora_B.detach().cpu()}

    def load_adapter_state_dict(self, state: Mapping[str, torch.Tensor]) -> bool:
        if set(state) != {"lora_A", "lora_B"}:
            raise ValueError("native Control6 adapter must contain only lora_A and lora_B")
        lora_a = state["lora_A"]
        migrated_legacy_phase = tuple(lora_a.shape) == (self.rank, HIDDEN_WIDTH)
        if migrated_legacy_phase:
            # Preserve the effect of pre-phase checkpoints exactly. The new
            # phase columns start at zero and remain untrained until fine-tuned.
            lora_a = torch.nn.functional.pad(lora_a, (0, PHASE_T_WIDTH))
        if tuple(lora_a.shape) != tuple(self.lora_A.shape):
            raise ValueError("native Control6 lora_A shape mismatch")
        if tuple(state["lora_B"].shape) != tuple(self.lora_B.shape):
            raise ValueError("native Control6 lora_B shape mismatch")
        with torch.no_grad():
            self.lora_A.copy_(lora_a)
            self.lora_B.copy_(state["lora_B"])
        return migrated_legacy_phase


def save_base_rows(path: str | Path, weight_or_module: Any, *, model_id: str = MODEL_ID) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(base_rows_bundle(weight_or_module, model_id=model_id), temporary)
    temporary.replace(destination)
    return destination


def load_base_rows(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    # `weights_only=True` keeps this tensor/primitive bundle within PyTorch's
    # restricted loader contract: https://docs.pytorch.org/docs/stable/generated/torch.load.html
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict):
        raise ValueError("base-row bundle must contain a dictionary")
    rows = validate_base_rows_bundle(bundle)
    return rows, bundle


def load_native_control6_adapter(
    path: str | Path,
    base_rows: torch.Tensor | None = None,
    *,
    device: str | torch.device = "cpu",
) -> tuple[NativeControl6LoRA, dict[str, Any]]:
    """Load only this six-row adapter against caller-supplied native Gemma rows."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("native Control6 adapter must contain a dictionary")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "architecture": "native_gemma_selected_output_rows_lora",
        "model_id": MODEL_ID,
        "action_to_token_id": dict(ACTION_TO_TOKEN_ID),
        "token_id_to_string": {str(k): v for k, v in TOKEN_ID_TO_STRING.items()},
        "actions": list(ACTIONS),
        "hidden_contract": HIDDEN_CONTRACT,
        "base_weight_source": BASE_WEIGHT_SOURCE,
    }
    mismatches = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatches:
        raise ValueError(f"native Control6 adapter contract mismatch: {mismatches}")
    rank = payload.get("rank")
    alpha = payload.get("alpha")
    if not isinstance(rank, int) or not isinstance(alpha, (int, float)):
        raise ValueError("native Control6 adapter is missing rank/alpha")
    saved_base_rows = payload.get("base_rows")
    if not isinstance(saved_base_rows, torch.Tensor) or tuple(saved_base_rows.shape) != (len(ACTIONS), HIDDEN_WIDTH):
        raise ValueError("native Control6 adapter must embed the exact frozen six LM-head rows")
    if not saved_base_rows.is_floating_point() or not torch.isfinite(saved_base_rows).all():
        raise ValueError("native Control6 embedded base rows must be finite floating-point weights")
    if base_rows is None:
        base_rows = saved_base_rows
    elif (
        tuple(base_rows.shape) != tuple(saved_base_rows.shape)
        or not torch.equal(base_rows.detach().cpu().to(saved_base_rows.dtype), saved_base_rows.cpu())
    ):
        raise ValueError("supplied live Gemma lm_head rows differ from the six rows saved in the adapter")
    model = NativeControl6LoRA(base_rows, rank=rank, alpha=float(alpha))
    adapter_state = payload.get("adapter_state_dict")
    if not isinstance(adapter_state, dict):
        raise ValueError("native Control6 adapter has no adapter_state_dict")
    migrated_legacy_phase = model.load_adapter_state_dict(adapter_state)
    model.to(device=device)
    model.eval()
    report = payload.get("report")
    load_report = dict(report) if isinstance(report, dict) else {}
    if migrated_legacy_phase:
        load_report["phase_t_checkpoint_migration"] = (
            "legacy 1536-column LoRA A zero-extended in memory; phase columns are untrained"
        )
    return model, load_report


__all__ = [
    "ACTION_TO_TOKEN_ID",
    "ACTIONS",
    "BASE_WEIGHT_SOURCE",
    "HIDDEN_CONTRACT",
    "HIDDEN_NORM",
    "HIDDEN_SOURCE",
    "HIDDEN_WIDTH",
    "MODEL_ID",
    "NativeControl6LoRA",
    "TICK_MS",
    "TOKEN_ID_TO_STRING",
    "VOCAB_SIZE",
    "action_token_ids",
    "load_base_rows",
    "load_native_control6_adapter",
    "save_base_rows",
    "select_tied_output_rows",
    "validate_base_rows_bundle",
]
