"""Functional, shape-stable Gemma text-LM view for the realtime listener.

The class intentionally owns no ``nn.Parameter`` and never mutates a Gemma
module or a Transformers attention registry.  It reads the already-resident
Gemma weights through the original module objects, while each listener session
continues to own its own KV and shared-KV state.
"""
from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ShapeStableListenerOutput:
    """Listener-local result with no model-global execution state."""

    last_hidden_state: torch.Tensor
    past_key_values: Any
    shared_kv_states: UserDict


class _StaticPaddingWorkspace:
    """Reusable listener-only shape buffers, serialized by the listener backend."""

    def __init__(self, target_length: int) -> None:
        self.target_length = int(target_length)
        self._buffers: dict[tuple[str, tuple[int, ...], torch.dtype, torch.device], torch.Tensor] = {}
        self._mask_buffers: dict[tuple[tuple[int, ...], torch.dtype, torch.device], torch.Tensor] = {}

    @staticmethod
    def _axis(axis: int, ndim: int) -> int:
        return axis if axis >= 0 else ndim + axis

    def padded(self, value: torch.Tensor, *, axis: int, slot: str = "default") -> torch.Tensor:
        axis = self._axis(axis, value.ndim)
        length = int(value.shape[axis])
        if length >= self.target_length:
            return value
        shape = list(value.shape)
        shape[axis] = self.target_length
        key = (slot, tuple(shape), value.dtype, value.device)
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = torch.zeros(shape, dtype=value.dtype, device=value.device)
            self._buffers[key] = buffer
        target = [slice(None)] * value.ndim
        target[axis] = slice(0, length)
        buffer[tuple(target)].copy_(value)
        return buffer

    def padded_mask(self, mask: torch.Tensor, *, key_length: int) -> torch.Tensor:
        if key_length >= self.target_length:
            return mask
        shape = list(mask.shape)
        shape[-1] = self.target_length
        key = (tuple(shape), mask.dtype, mask.device)
        buffer = self._mask_buffers.get(key)
        fill = torch.finfo(mask.dtype).min if mask.is_floating_point() else False
        if buffer is None:
            buffer = torch.full(shape, fill, dtype=mask.dtype, device=mask.device)
            self._mask_buffers[key] = buffer
        buffer[..., :key_length].copy_(mask)
        return buffer

    @property
    def allocated_bytes(self) -> int:
        tensors = (*self._buffers.values(), *self._mask_buffers.values())
        return sum(int(value.numel() * value.element_size()) for value in tensors)


class ShapeStableGemmaListenerLM:
    """Listener-specific functional execution over shared frozen Gemma weights.

    The implementation mirrors the current Transformers 5 Gemma text decoder,
    but substitutes only the operations proven necessary by the BF16 candidate:
    all linear projections, every RMSNorm, dense MLPs, and the K/V dimension of
    SDPA are shape-padded to ``target_length``.  The native 4-D masks, RoPE,
    DynamicCache, and shared-KV topology are retained unchanged.
    """

    def __init__(self, language_model: Any, *, target_length: int, stream_length: int = 4) -> None:
        if int(target_length) <= 0:
            raise ValueError("target_length deve essere positivo")
        if not hasattr(language_model, "layers") or not hasattr(language_model, "config"):
            raise TypeError("language_model Gemma non valido")
        self.language_model = language_model
        self.target_length = int(target_length)
        self.stream_length = int(stream_length)
        self._workspace_prefill = _StaticPaddingWorkspace(self.target_length)
        self._workspace_stream = _StaticPaddingWorkspace(self.stream_length)
        self._workspace = self._workspace_stream
        self._source_parameter_data_ptrs = {
            name: int(parameter.data_ptr())
            for name, parameter in language_model.named_parameters()
        }
        self._created_allocated_bytes = self._cuda_memory("memory_allocated")
        self._created_reserved_bytes = self._cuda_memory("memory_reserved")

    @staticmethod
    def _cuda_memory(method: str) -> int | None:
        if not torch.cuda.is_available():
            return None
        return int(getattr(torch.cuda, method)())

    def weight_sharing_report(self) -> dict[str, Any]:
        current = {
            name: int(parameter.data_ptr())
            for name, parameter in self.language_model.named_parameters()
        }
        return {
            "listener_parameter_count": 0,
            "source_parameter_count": len(self._source_parameter_data_ptrs),
            "source_parameter_data_ptrs": dict(self._source_parameter_data_ptrs),
            "shared_weight_storage": current == self._source_parameter_data_ptrs,
            "created_allocated_bytes": self._created_allocated_bytes,
            "created_reserved_bytes": self._created_reserved_bytes,
            "current_allocated_bytes": self._cuda_memory("memory_allocated"),
            "current_reserved_bytes": self._cuda_memory("memory_reserved"),
            "workspace_static_bytes": self._workspace_prefill.allocated_bytes + self._workspace_stream.allocated_bytes,
        }

    @staticmethod
    def _token_axis(value: torch.Tensor) -> int:
        # This deliberately mirrors the candidate harness.  Q/K/V RMSNorms are
        # rank-4 and therefore receive the same -2 dimension treatment as the
        # validated diagnostic path.
        return -2 if value.ndim >= 3 else 0

    def _padded_linear(self, linear: Any, value: torch.Tensor) -> torch.Tensor:
        axis = self._token_axis(value)
        original_length = int(value.shape[axis])
        padded = self._workspace.padded(value, axis=axis)
        output = F.linear(padded, linear.weight, linear.bias)
        if padded is value:
            return output
        slices = [slice(None)] * output.ndim
        slices[axis] = slice(0, original_length)
        return output[tuple(slices)]

    def _padded_rmsnorm(self, norm: Any, value: torch.Tensor) -> torch.Tensor:
        axis = self._token_axis(value)
        original_length = int(value.shape[axis])
        padded = self._workspace.padded(value, axis=axis)
        mean_squared = padded.float().pow(2).mean(-1, keepdim=True) + float(norm.eps)
        output = padded.float() * torch.pow(mean_squared, -0.5)
        weight = getattr(norm, "weight", None)
        if weight is not None:
            output = output * weight.float()
        output = output.type_as(padded)
        if padded is value:
            return output
        slices = [slice(None)] * output.ndim
        slices[axis] = slice(0, original_length)
        return output[tuple(slices)]

    def _project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        token_identity: torch.Tensor | None,
    ) -> torch.Tensor | None:
        lm = self.language_model
        if not getattr(lm, "hidden_size_per_layer_input", 0):
            return None
        if token_identity is None:
            raise ValueError("il listener shape-stable richiede PLE precomputata")
        projection = self._padded_linear(lm.per_layer_model_projection, inputs_embeds)
        projection = projection * lm.per_layer_model_projection_scale
        projection = projection.reshape(
            *inputs_embeds.shape[:-1],
            lm.config.num_hidden_layers,
            lm.hidden_size_per_layer_input,
        )
        projection = self._padded_rmsnorm(lm.per_layer_projection_norm, projection)
        return (projection + token_identity) * lm.per_layer_input_scale

    def _attention(
        self,
        attention: Any,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        shared_kv_states: UserDict,
        past_key_values: Any | None,
    ) -> torch.Tensor:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

        if getattr(attention.config, "_attn_implementation", None) != "sdpa":
            raise RuntimeError("ShapeStableGemmaListenerLM richiede il backend Gemma SDPA attestato")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        cos, sin = position_embeddings

        query_states = self._padded_linear(attention.q_proj, hidden_states).view(hidden_shape)
        query_states = self._padded_rmsnorm(attention.q_norm, query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)

        if attention.is_kv_shared_layer and past_key_values is not None:
            # Gemma's exact shared-KV contract is indexed by the source layer,
            # not by attention type.  ``shared_layers`` lives on this listener's
            # DynamicCache, never on the resident model.
            key_states, value_states = past_key_values.shared_layers[attention.kv_shared_layer_index]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self._padded_linear(attention.k_proj, hidden_states).view(hidden_shape)
            value_states = (
                self._padded_linear(attention.v_proj, hidden_states).view(hidden_shape)
                if attention.v_proj is not None
                else key_states
            )
            key_states = self._padded_rmsnorm(attention.k_norm, key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            value_states = self._padded_rmsnorm(attention.v_norm, value_states).transpose(1, 2)

        if past_key_values is not None:
            if not attention.is_kv_shared_layer:
                key_states, value_states = past_key_values.update(key_states, value_states, attention.layer_idx)
            if attention.store_full_length_kv:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                past_key_values.shared_layers[attention.layer_idx] = (key_states, value_states)

        key_length = int(key_states.shape[-2])
        # K and V have the same shape, but both must remain live until SDPA
        # consumes them; distinct slots prevent an in-place workspace reuse.
        key_padded = self._workspace.padded(key_states, axis=-2, slot="attention_key")
        value_padded = self._workspace.padded(value_states, axis=-2, slot="attention_value")
        mask_padded = (
            self._workspace.padded_mask(attention_mask, key_length=key_length)
            if isinstance(attention_mask, torch.Tensor)
            else attention_mask
        )
        attn_output, _ = sdpa_attention_forward(
            attention,
            query_states,
            key_padded,
            value_padded,
            mask_padded,
            dropout=attention.attention_dropout if attention.training else 0.0,
            scaling=attention.scaling,
            sliding_window=attention.sliding_window,
            position_ids=position_ids,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self._padded_linear(attention.o_proj, attn_output)

    def _mlp(self, mlp: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        original_length = int(hidden_states.shape[-2])
        padded = self._workspace.padded(hidden_states, axis=-2)
        gate = self._padded_linear(mlp.gate_proj, padded)
        up = self._padded_linear(mlp.up_proj, padded)
        output = self._padded_linear(mlp.down_proj, mlp.act_fn(gate) * up)
        return output[:, :original_length, :]

    def _decoder_layer(
        self,
        layer: Any,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        shared_kv_states: UserDict,
        past_key_values: Any | None,
    ) -> torch.Tensor:
        if bool(getattr(layer, "enable_moe_block", False)):
            raise RuntimeError("ShapeStableGemmaListenerLM non supporta checkpoint Gemma MoE")
        residual = hidden_states
        hidden_states = self._padded_rmsnorm(layer.input_layernorm, hidden_states)
        hidden_states = self._attention(
            layer.self_attn,
            hidden_states,
            position_embeddings,
            attention_mask,
            position_ids,
            shared_kv_states,
            past_key_values,
        )
        hidden_states = self._padded_rmsnorm(layer.post_attention_layernorm, hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self._padded_rmsnorm(layer.pre_feedforward_layernorm, hidden_states)
        hidden_states = self._mlp(layer.mlp, hidden_states)
        hidden_states = self._padded_rmsnorm(layer.post_feedforward_layernorm, hidden_states)
        hidden_states = residual + hidden_states

        if getattr(layer, "hidden_size_per_layer_input", 0):
            if per_layer_input is None:
                raise ValueError("PLE mancante per decoder layer Gemma")
            residual = hidden_states
            hidden_states = self._padded_linear(layer.per_layer_input_gate, hidden_states)
            hidden_states = layer.act_fn(hidden_states) * per_layer_input
            hidden_states = self._padded_linear(layer.per_layer_projection, hidden_states)
            hidden_states = self._padded_rmsnorm(layer.post_per_layer_input_norm, hidden_states)
            hidden_states = residual + hidden_states
        return hidden_states * layer.layer_scalar

    @torch.inference_mode()
    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
        attention_mask: dict[str, torch.Tensor | None],
        position_ids: torch.Tensor,
        past_key_values: Any | None,
        shared_kv_states: UserDict | None,
    ) -> ShapeStableListenerOutput:
        lm = self.language_model
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != lm.config.hidden_size:
            raise ValueError(f"inputs_embeds Gemma inattesi: {tuple(inputs_embeds.shape)}")
        q_len = int(inputs_embeds.shape[1])
        if q_len <= self.stream_length:
            self._workspace = self._workspace_stream
        elif q_len <= self.target_length:
            self._workspace = self._workspace_prefill
        else:
            raise ValueError("query listener oltre il target shape-stable")
        # Match Gemma4TextModel(use_cache=True): the cache is listener-session
        # state, created only when this invocation begins a new listener stream.
        if past_key_values is None:
            from transformers.cache_utils import DynamicCache

            past_key_values = DynamicCache(config=lm.config)
        projected_ple = self._project_per_layer_inputs(inputs_embeds, per_layer_inputs)
        position_embeddings = {
            layer_type: lm.rotary_emb(inputs_embeds, position_ids, layer_type)
            for layer_type in lm.unique_layer_types
        }
        shared = shared_kv_states if shared_kv_states is not None else UserDict()
        hidden_states = inputs_embeds
        for index, layer in enumerate(lm.layers[: lm.config.num_hidden_layers]):
            per_layer_input = projected_ple[:, :, index, :] if projected_ple is not None else None
            layer_type = lm.config.layer_types[index]
            hidden_states = self._decoder_layer(
                layer,
                hidden_states,
                per_layer_input,
                position_embeddings[layer_type],
                attention_mask[layer_type],
                position_ids,
                shared,
                past_key_values,
            )
        hidden_states = self._padded_rmsnorm(lm.norm, hidden_states)
        return ShapeStableListenerOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            shared_kv_states=shared,
        )


__all__ = ["ShapeStableGemmaListenerLM", "ShapeStableListenerOutput"]
