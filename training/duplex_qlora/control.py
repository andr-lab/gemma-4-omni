"""Six-action head trained only on merged-base causal listener features."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .features import FEATURE_CONTRACT, listener_feature_contract
from .schema import ACTIONS

HEAD_ARCHITECTURES = (
    "linear",
    "mlp_256_gelu_dropout_0.1",
    "native_phase_fusion_mlp",
)


class ControlHead(nn.Module):
    input_dim = 1536
    actions = ACTIONS
    pooling = "last_audio_query"
    tick_ms = 80

    def __init__(self, *, architecture: str = "linear", input_dim: int = 1536) -> None:
        super().__init__()
        if architecture not in HEAD_ARCHITECTURES:
            raise ValueError(
                f"control head architecture deve essere una tra {HEAD_ARCHITECTURES}, ricevuto {architecture!r}"
            )
        self.architecture = architecture
        self.input_dim = int(input_dim)
        self.feature_contract = listener_feature_contract(self.input_dim)
        if architecture == "native_phase_fusion_mlp":
            if self.input_dim != 1540:
                raise ValueError(
                    "native_phase_fusion_mlp richiede 1536 feature native + 4 feature phase"
                )
            self.native_norm = nn.LayerNorm(1536)
            self.native_projection = nn.Linear(1536, 256)
            self.native_activation = nn.GELU()
            self.phase_projection = nn.Linear(4, 32)
            self.phase_activation = nn.GELU()
            self.classifier = nn.Linear(288, len(self.actions))
        else:
            self.norm = nn.LayerNorm(self.input_dim)
        if architecture == "linear":
            self.classifier = nn.Linear(self.input_dim, len(self.actions))
        elif architecture == "mlp_256_gelu_dropout_0.1":
            self.classifier = nn.Sequential(
                nn.Linear(self.input_dim, 256),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(256, len(self.actions)),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"control head richiede width {self.input_dim}, ricevuto {tuple(features.shape)}"
            )
        if self.architecture == "native_phase_fusion_mlp":
            native = self.native_activation(
                self.native_projection(self.native_norm(features[..., :1536]))
            )
            phase = features[..., 1536:]
            # RMS fields are clipped to [-6, 0]; scaling by six maps them to
            # [-1, 0]. The fourth field is already normalized to [-1, 1].
            phase = torch.cat((phase[..., :3] / 6.0, phase[..., 3:4]), dim=-1)
            phase = self.phase_activation(self.phase_projection(phase))
            return self.classifier(torch.cat((native, phase), dim=-1))
        return self.classifier(self.norm(features))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "input_dim": self.input_dim,
            "pooling": self.pooling,
            "tick_ms": self.tick_ms,
            "architecture": self.architecture,
            "actions": list(self.actions),
            "feature_contract": self.feature_contract,
            "state_dict": {name: value.detach().cpu() for name, value in self.state_dict().items()},
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)


def load_control_head(path: str | Path, *, device: str | torch.device = "cpu") -> ControlHead:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"duplex_control6 control head mancante: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"duplex_control6 control head non leggibile: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("duplex_control6 checkpoint deve contenere un object")
    input_dim = payload.get("input_dim")
    try:
        feature_contract = listener_feature_contract(input_dim)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"duplex_control6 input_dim non supportato: {input_dim!r}") from exc
    expected = {
        "schema_version": 1,
        "input_dim": input_dim,
        "pooling": "last_audio_query",
        "tick_ms": 80,
        "actions": list(ACTIONS),
        "feature_contract": feature_contract,
    }
    mismatches = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatches:
        raise ValueError(f"duplex_control6 checkpoint contract mismatch: {mismatches}")
    architecture = payload.get("architecture", "linear")
    if not isinstance(architecture, str) or architecture not in HEAD_ARCHITECTURES:
        raise ValueError(f"duplex_control6 architettura non supportata: {architecture!r}")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("duplex_control6 checkpoint privo di state_dict")
    model = ControlHead(architecture=architecture, input_dim=input_dim)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(f"duplex_control6 control head state non compatibile: {exc}") from exc
    model.to(device=device)
    model.eval()
    return model


__all__ = ["ACTIONS", "HEAD_ARCHITECTURES", "ControlHead", "load_control_head"]
