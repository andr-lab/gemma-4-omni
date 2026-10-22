"""Response-token residuals for Qwen's existing audio-query embeddings."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn


def response_state_at_cursor(
    response_hidden_by_qwen_token: torch.Tensor,
    generation_step: int,
) -> tuple[torch.Tensor, bool]:
    """Select the response state for Qwen's existing audio-text cursor.

    Qwen consumes text token 0 at the first audio step and then advances one
    raw-text token per generated RVQ frame. Once the response text is
    exhausted, the final state is held; no hidden state is synthesized.
    """
    if (
        not isinstance(response_hidden_by_qwen_token, torch.Tensor)
        or response_hidden_by_qwen_token.ndim != 3
        or int(response_hidden_by_qwen_token.shape[0]) != 1
        or int(response_hidden_by_qwen_token.shape[1]) <= 0
    ):
        raise ValueError("aligned response hidden must have shape (1,T,D), T > 0")
    if isinstance(generation_step, bool) or int(generation_step) < 0:
        raise ValueError("generation_step must be a nonnegative integer")
    index = min(int(generation_step), int(response_hidden_by_qwen_token.shape[1]) - 1)
    return response_hidden_by_qwen_token[:, index : index + 1, :], index < int(generation_step)


def apply_response_state_residual(
    adapter: "FrameResidualAdapter | FrameResidualQueryFiLMAdapter",
    response_hidden_by_qwen_token: torch.Tensor,
    qwen_audio_input: torch.Tensor,
    generation_step: int,
) -> tuple[torch.Tensor, bool]:
    """Add one cursor-matched response residual to an existing Qwen Q1 input."""
    if not isinstance(qwen_audio_input, torch.Tensor) or qwen_audio_input.ndim != 3:
        raise ValueError("Qwen audio input must have shape (1,1,D)")
    if int(qwen_audio_input.shape[0]) != 1 or int(qwen_audio_input.shape[1]) != 1:
        raise ValueError("one Qwen generation step requires a (1,1,D) audio input")
    local_hidden, tail_hold = response_state_at_cursor(
        response_hidden_by_qwen_token, generation_step
    )
    if int(generation_step) == 0 and bool(getattr(adapter, "preserve_frame0", False)):
        return qwen_audio_input, tail_hold
    return adapter(local_hidden, qwen_audio_input), tail_hold


class FrameResidualAdapter(nn.Module):
    """Map aligned Gemma response states into bounded Qwen Q1 input residuals.

    The adapter changes only the existing audio-query input.  It adds no Qwen
    token, KV slot, prefix position, or audio target; a zero-initialized output
    projection is therefore an exact frozen-Qwen baseline at initialization.
    """

    architecture = "FrameResidualAdapter LayerNorm(1536)-Linear(256)-SiLU-Linear(2048), Qwen-Q1 query residual"
    architecture_id = "response_hidden_q1_input_residual_v1"
    schema_version = 1
    preserve_frame0 = False

    def __init__(
        self,
        input_dim: int = 1536,
        output_dim: int = 2048,
        bottleneck_dim: int = 256,
        *,
        initial_scale: float = 0.05,
        max_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        if not (math.isfinite(initial_scale) and math.isfinite(max_scale)):
            raise ValueError("adapter scales must be finite")
        if not (0.0 < initial_scale < max_scale):
            raise ValueError("scales must satisfy 0 < initial_scale < max_scale")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.max_scale = float(max_scale)
        initial_fraction = float(initial_scale) / float(max_scale)
        self.scale_logit = nn.Parameter(
            torch.tensor(math.log(initial_fraction / (1.0 - initial_fraction)))
        )
        self.norm = nn.LayerNorm(self.input_dim)
        self.down = nn.Linear(self.input_dim, self.bottleneck_dim)
        self.activation = nn.SiLU()
        self.output = nn.Linear(self.bottleneck_dim, self.output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @property
    def scale(self) -> torch.Tensor:
        return self.max_scale * torch.sigmoid(self.scale_logit)

    def residual(
        self,
        response_hidden: torch.Tensor,
        qwen_audio_input: torch.Tensor,
        *,
        detach_scale: bool = False,
    ) -> torch.Tensor:
        if response_hidden.ndim != 3 or qwen_audio_input.ndim != 3:
            raise ValueError("response hidden and Qwen audio input must be (B,T,D)")
        if tuple(response_hidden.shape[:2]) != tuple(qwen_audio_input.shape[:2]):
            raise ValueError("batch/frame dimensions must match")
        if int(response_hidden.shape[-1]) != self.input_dim:
            raise ValueError(f"response hidden width must be {self.input_dim}")
        if int(qwen_audio_input.shape[-1]) != self.output_dim:
            raise ValueError(f"Qwen audio input width must be {self.output_dim}")
        if response_hidden.device != qwen_audio_input.device:
            raise ValueError("response hidden and Qwen audio input must share a device")
        hidden = response_hidden.to(dtype=self.norm.weight.dtype)
        base = qwen_audio_input
        mapped = self.output(self.activation(self.down(self.norm(hidden))))
        amplitude = torch.sqrt(torch.mean(base.float().square(), dim=-1, keepdim=True))
        # Keep the trainable mapping and bounded scale in FP32, then cast once
        # at the Qwen query boundary. Tensor.to(dtype=...) returns the requested
        # dtype: https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.to.html
        scale = self.scale.float()
        if detach_scale:
            scale = scale.detach()
        residual_fp32 = torch.tanh(mapped.float()) * scale * amplitude
        return residual_fp32.to(dtype=qwen_audio_input.dtype)

    def forward(self, response_hidden: torch.Tensor, qwen_audio_input: torch.Tensor) -> torch.Tensor:
        return qwen_audio_input + self.residual(response_hidden, qwen_audio_input)


class FrameResidualQueryFiLMAdapter(nn.Module):
    """Query-conditioned Gemma response residual for the existing Qwen Q1 input."""

    architecture = "FrameResidualQueryFiLM LayerNorm(1536+2048)-Linear(256)-FiLM-SiLU-Linear(2048), Qwen-Q1 query residual"
    architecture_id = "response_hidden_q1_query_film_v2"
    schema_version = 2
    preserve_frame0 = True

    def __init__(
        self,
        input_dim: int = 1536,
        output_dim: int = 2048,
        bottleneck_dim: int = 256,
        *,
        initial_scale: float = 0.05,
        max_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        if not (math.isfinite(initial_scale) and math.isfinite(max_scale)):
            raise ValueError("adapter scales must be finite")
        if not (0.0 < initial_scale < max_scale):
            raise ValueError("scales must satisfy 0 < initial_scale < max_scale")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.max_scale = float(max_scale)
        initial_fraction = float(initial_scale) / float(max_scale)
        self.scale_logit = nn.Parameter(
            torch.tensor(math.log(initial_fraction / (1.0 - initial_fraction)))
        )
        self.query_norm = nn.LayerNorm(self.output_dim)
        self.query_down = nn.Linear(self.output_dim, self.bottleneck_dim)
        self.response_norm = nn.LayerNorm(self.input_dim)
        self.response_down = nn.Linear(self.input_dim, self.bottleneck_dim)
        self.response_activation = nn.SiLU()
        self.film = nn.Linear(self.bottleneck_dim, 2 * self.bottleneck_dim)
        self.activation = nn.SiLU()
        self.output = nn.Linear(self.bottleneck_dim, self.output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @property
    def scale(self) -> torch.Tensor:
        return self.max_scale * torch.sigmoid(self.scale_logit)

    def residual(
        self,
        response_hidden: torch.Tensor,
        qwen_audio_input: torch.Tensor,
        *,
        detach_scale: bool = False,
    ) -> torch.Tensor:
        if response_hidden.ndim != 3 or qwen_audio_input.ndim != 3:
            raise ValueError("response hidden and Qwen audio input must be (B,T,D)")
        if tuple(response_hidden.shape[:2]) != tuple(qwen_audio_input.shape[:2]):
            raise ValueError("batch/frame dimensions must match")
        if int(response_hidden.shape[-1]) != self.input_dim:
            raise ValueError(f"response hidden width must be {self.input_dim}")
        if int(qwen_audio_input.shape[-1]) != self.output_dim:
            raise ValueError(f"Qwen audio input width must be {self.output_dim}")
        if response_hidden.device != qwen_audio_input.device:
            raise ValueError("response hidden and Qwen audio input must share a device")

        dtype = self.query_norm.weight.dtype
        query = self.query_down(self.query_norm(qwen_audio_input.to(dtype=dtype)))
        response = self.response_activation(
            self.response_down(self.response_norm(response_hidden.to(dtype=dtype)))
        )
        gamma, beta = self.film(response).chunk(2, dim=-1)
        fused = self.activation(query * (1.0 + torch.tanh(gamma)) + beta)
        mapped = self.output(fused)
        amplitude = torch.sqrt(torch.mean(qwen_audio_input.float().square(), dim=-1, keepdim=True))
        scale = self.scale.float()
        if detach_scale:
            scale = scale.detach()
        residual_fp32 = torch.tanh(mapped.float()) * scale * amplitude
        return residual_fp32.to(dtype=qwen_audio_input.dtype)

    def forward(self, response_hidden: torch.Tensor, qwen_audio_input: torch.Tensor) -> torch.Tensor:
        return qwen_audio_input + self.residual(response_hidden, qwen_audio_input)


def project_norm_preserving_query(
    query: torch.Tensor,
    delta: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Project ``query + delta`` onto the sphere with the original query norm."""
    if not isinstance(query, torch.Tensor) or not isinstance(delta, torch.Tensor):
        raise TypeError("query and delta must be tensors")
    if query.shape != delta.shape or query.ndim < 1:
        raise ValueError("query and delta must have the same non-scalar shape")
    if query.device != delta.device or not query.is_floating_point() or not delta.is_floating_point():
        raise ValueError("query and delta must be floating tensors on the same device")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")

    query_fp32 = query.float()
    modulated = query_fp32 + delta.float()
    query_norm = torch.linalg.vector_norm(query_fp32, ord=2, dim=-1, keepdim=True)
    modulated_norm = torch.linalg.vector_norm(modulated, ord=2, dim=-1, keepdim=True)
    projected = modulated * (query_norm / modulated_norm.clamp_min(float(eps)))
    # The stated projection is undefined when q + delta is exactly zero;
    # returning q for that degenerate case preserves the norm and direction.
    projected = torch.where(modulated_norm > float(eps), projected, query_fp32)
    return projected.to(dtype=query.dtype)


class NormPreservingQueryFiLMAdapter(FrameResidualQueryFiLMAdapter):
    """Opt-in NP-FiLM candidate; it shares v2 weights but projects output norm."""

    architecture = FrameResidualQueryFiLMAdapter.architecture + ", spherical query-norm projection"
    architecture_id = "response_hidden_q1_query_film_v2_np_candidate"

    def forward(self, response_hidden: torch.Tensor, qwen_audio_input: torch.Tensor) -> torch.Tensor:
        delta = self.residual(response_hidden, qwen_audio_input)
        return project_norm_preserving_query(qwen_audio_input, delta)


def load_frame_residual_adapter(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[FrameResidualAdapter, dict[str, Any]]:
    """Load only the opt-in residual adapter; reject incompatible checkpoints."""
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"unsupported frame residual checkpoint schema: {path}")
    schema_version = int(payload.get("schema_version", -1))
    architecture = payload.get("architecture")
    architecture_id = payload.get("architecture_id")
    if schema_version == FrameResidualAdapter.schema_version and architecture == FrameResidualAdapter.architecture:
        adapter_type = FrameResidualAdapter
    elif schema_version == FrameResidualQueryFiLMAdapter.schema_version and architecture == FrameResidualQueryFiLMAdapter.architecture:
        adapter_type = FrameResidualQueryFiLMAdapter
    else:
        raise RuntimeError(f"frame residual checkpoint architecture mismatch: {path}")
    if architecture_id is not None and architecture_id != adapter_type.architecture_id:
        raise RuntimeError(f"frame residual checkpoint architecture ID mismatch: {path}")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"frame residual checkpoint has no state_dict: {path}")
    adapter = adapter_type().to(device=device, dtype=dtype)
    adapter.load_state_dict(state, strict=True)
    adapter.eval().requires_grad_(False)
    contract = payload.get("training_contract")
    if not isinstance(contract, dict) or not str(contract.get("runtime_route", "")).startswith("plain native Qwen Q1"):
        raise RuntimeError(f"frame residual checkpoint does not declare the plain Q1 route: {path}")
    return adapter, payload


__all__ = [
    "FrameResidualAdapter",
    "FrameResidualQueryFiLMAdapter",
    "NormPreservingQueryFiLMAdapter",
    "apply_response_state_residual",
    "load_frame_residual_adapter",
    "project_norm_preserving_query",
    "response_state_at_cursor",
]
