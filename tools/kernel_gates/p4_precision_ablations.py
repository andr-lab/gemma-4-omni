#!/usr/bin/env python3
"""Ordered precision/backend ablations for the Gemma P4 numeric blocker.

The canonical full BF16 result is collected once with no patches.  Each
candidate patches only the incremental execution, starts from a fresh cache,
and is compared against that untouched native full result.  This keeps the
oracle independent and makes a candidate's effect attributable.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.thinker import load_thinker  # noqa: E402
from tools.kernel_gates.p4_numeric_bisect import (  # noqa: E402
    Capture,
    _compare,
    _fixture,
    _incremental,
    _native_full,
)


ORDERED_CANDIDATES = (
    "baseline",
    "bf16_no_reduced_reduction",
    "eager_attention",
    "kv_projection_fp32",
    "qkv_projection_fp32",
    "attention_linear_fp32",
    "all_linear_fp32",
    "all_linear_padded_full_shape",
    "shape_padded_fp32_activation",
    "shape_padded_fp32_mlp_elementwise",
    "kv_projection_padded",
    "qkv_projection_padded",
    "attention_linear_padded",
    "mlp_linear_padded",
    "per_layer_linear_padded",
    "shape_padded_full_attention",
    "shape_padded_rmsnorm_input_only",
    "shape_padded_rmsnorm_input_preff",
    "shape_padded_rmsnorm_input_preff_postper",
    "shape_padded_math_rmsnorm",
    "transformer_linear_padded_rmsnorm",
    "no_o_projection_padded_rmsnorm",
    "no_mlp_padded_rmsnorm",
    "shape_padded_rmsnorm_input_postattn_preff",
    "shape_padded_rmsnorm_input_postattn_preff_postff",
    "shape_padded_rmsnorm_main",
    "shape_padded_rmsnorm",
    "shape_padded_rmsnorm_kv",
    "shape_padded_all_ops",
    "shape_padded_all_ops_qk",
    "shape_padded_all_ops_kv",
    "shape_padded_all_ops_qk_l0",
    "shape_padded_all_ops_qk_l1",
    "shape_padded_all_ops_qk_l2",
    "shape_padded_all_ops_qk_l3",
    "shape_padded_all_ops_qk_l4",
    "shape_padded_all_ops_qk_l8",
    "shape_padded_cache_rmsnorm",
    "shape_padded_qk_rmsnorm",
    "shape_padded_qk_rmsnorm_postper",
    "shape_padded_static_cache_rmsnorm",
    "router_projection_fp32",
    "moe_experts_fp32",
)


def _selected(hidden: torch.Tensor, positions: list[int]) -> torch.Tensor:
    return hidden[:, positions, :].detach().cpu().contiguous()


class NullCapture:
    def __init__(self, positions: set[int]):
        self.positions = positions
        self.final: torch.Tensor | None = None

    def install(self, language_model: Any) -> None:
        del language_model

    def store_final(self, hidden: torch.Tensor) -> None:
        self.final = hidden[:, sorted(self.positions), :].detach().cpu().contiguous()

    def close(self) -> None:
        return None


def _linear_fp32_forward(module: nn.Module, inputs: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    if args or kwargs:
        raise TypeError(f"linear ablation inattesa: args={len(args)} kwargs={list(kwargs)}")
    weight = getattr(module, "weight").float()
    bias = getattr(module, "bias", None)
    result = F.linear(inputs.float(), weight, bias.float() if bias is not None else None)
    return result.to(dtype=inputs.dtype)


def _linear_padded_forward(module: nn.Module, inputs: torch.Tensor, target_length: int, *args: Any, **kwargs: Any) -> torch.Tensor:
    if args or kwargs:
        raise TypeError(f"linear padding ablation inattesa: args={len(args)} kwargs={list(kwargs)}")
    if inputs.ndim < 2 or target_length <= 0:
        return module._p4_original_forward(inputs)
    token_axis = -2 if inputs.ndim >= 3 else 0
    query_length = inputs.shape[token_axis]
    if query_length >= target_length:
        return module._p4_original_forward(inputs)
    padding_shape = list(inputs.shape)
    padding_shape[token_axis] = target_length - query_length
    padded = torch.cat((inputs, torch.zeros(padding_shape, dtype=inputs.dtype, device=inputs.device)), dim=token_axis)
    result = F.linear(padded, getattr(module, "weight"), getattr(module, "bias", None))
    slices = [slice(None)] * result.ndim
    slices[token_axis] = slice(0, query_length)
    return result[tuple(slices)]


@contextmanager
def _patch_linear_modules(modules: list[nn.Module]) -> Iterator[bool]:
    originals: list[tuple[nn.Module, Any]] = []
    for module in modules:
        originals.append((module, module.forward))
        module.forward = lambda inputs, *args, _module=module, **kwargs: _linear_fp32_forward(
            _module, inputs, *args, **kwargs
        )
    try:
        yield bool(modules)
    finally:
        for module, original in originals:
            module.forward = original


@contextmanager
def _patch_padded_linear_modules(modules: list[nn.Module], target_length: int) -> Iterator[bool]:
    originals: list[tuple[nn.Module, Any]] = []
    for module in modules:
        original = module.forward
        originals.append((module, original))
        module._p4_original_forward = original
        module.forward = lambda inputs, *args, _module=module, **kwargs: _linear_padded_forward(
            _module, inputs, target_length, *args, **kwargs
        )
    try:
        yield bool(modules)
    finally:
        for module, original in originals:
            module.forward = original
            if hasattr(module, "_p4_original_forward"):
                delattr(module, "_p4_original_forward")


@contextmanager
def _patch_activation_modules(modules: list[nn.Module]) -> Iterator[bool]:
    originals: list[tuple[nn.Module, Any]] = []
    for module in modules:
        original = module.forward
        originals.append((module, original))
        module._p4_original_forward = original
        module.forward = lambda inputs, *args, _module=module, **kwargs: _module._p4_original_forward(
            inputs.float(), *args, **kwargs
        ).to(dtype=inputs.dtype)
    try:
        yield bool(modules)
    finally:
        for module, original in originals:
            module.forward = original
            if hasattr(module, "_p4_original_forward"):
                delattr(module, "_p4_original_forward")


@contextmanager
def _patch_rmsnorm_modules(modules: list[nn.Module], target_length: int) -> Iterator[bool]:
    originals: list[tuple[nn.Module, Any]] = []
    for module in modules:
        original = module.forward
        originals.append((module, original))

        def forward(inputs: torch.Tensor, _module=module) -> torch.Tensor:
            if inputs.ndim < 2 or inputs.shape[-2 if inputs.ndim >= 3 else 0] >= target_length:
                return _module._p4_original_forward(inputs)
            token_axis = -2 if inputs.ndim >= 3 else 0
            query_length = inputs.shape[token_axis]
            padding_shape = list(inputs.shape)
            padding_shape[token_axis] = target_length - query_length
            padded = torch.cat((inputs, torch.zeros(padding_shape, dtype=inputs.dtype, device=inputs.device)), dim=token_axis)
            result = _module._p4_original_forward(padded)
            slices = [slice(None)] * result.ndim
            slices[token_axis] = slice(0, query_length)
            return result[tuple(slices)]

        module._p4_original_forward = original
        module.forward = forward
    try:
        yield bool(modules)
    finally:
        for module, original in originals:
            module.forward = original
            if hasattr(module, "_p4_original_forward"):
                delattr(module, "_p4_original_forward")


@contextmanager
def _patch_mlp_shape_modules(modules: list[nn.Module], target_length: int) -> Iterator[bool]:
    originals: list[tuple[nn.Module, Any]] = []
    for module in modules:
        original = module.forward
        originals.append((module, original))

        def forward(inputs: torch.Tensor, _module=module) -> torch.Tensor:
            if inputs.ndim < 3 or inputs.shape[-2] >= target_length:
                return _module._p4_original_forward(inputs)
            query_length = inputs.shape[-2]
            padding_shape = list(inputs.shape)
            padding_shape[-2] = target_length - query_length
            padded = torch.cat((inputs, torch.zeros(padding_shape, dtype=inputs.dtype, device=inputs.device)), dim=-2)
            gate = _module.gate_proj(padded)
            up = _module.up_proj(padded)
            activated = _module.act_fn(gate)
            product = activated * up
            result = _module.down_proj(product)
            return result[..., :query_length, :]

        module._p4_original_forward = original
        module.forward = forward
    try:
        yield bool(modules)
    finally:
        for module, original in originals:
            module.forward = original
            if hasattr(module, "_p4_original_forward"):
                delattr(module, "_p4_original_forward")


@contextmanager
def _patch_mlp_elementwise(modules: list[nn.Module]) -> Iterator[bool]:
    originals: list[tuple[nn.Module, Any]] = []
    for module in modules:
        original = module.forward
        originals.append((module, original))

        def forward(inputs: torch.Tensor, _module=module) -> torch.Tensor:
            gate = _module.gate_proj(inputs)
            up = _module.up_proj(inputs)
            activated = _module.act_fn(gate)
            product = (activated.float() * up.float()).to(dtype=inputs.dtype)
            return _module.down_proj(product)

        module.forward = forward
    try:
        yield bool(modules)
    finally:
        for module, original in originals:
            module.forward = original


@contextmanager
def _patch_padded_kv_attention(lm: Any, target_length: int) -> Iterator[bool]:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gemma4 import modeling_gemma4

    key = "p4_padded_kv"
    sentinel = object()
    old_backend_name = lm.config._attn_implementation
    old_backend = ALL_ATTENTION_FUNCTIONS.get_interface(
        old_backend_name,
        modeling_gemma4.eager_attention_forward,
    )
    old_registered = ALL_ATTENTION_FUNCTIONS._global_mapping.get(old_backend_name, sentinel)

    def padded_kv(module: Any, query: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor,
                  attention_mask: torch.Tensor | None, dropout: float = 0.0,
                  scaling: float | None = None, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        key_length = key_states.shape[-2]
        if key_length >= target_length or not isinstance(attention_mask, torch.Tensor):
            return old_backend(module, query, key_states, value_states, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        pad = target_length - key_length
        key_pad = torch.cat((key_states, torch.zeros((*key_states.shape[:-2], pad, key_states.shape[-1]), dtype=key_states.dtype, device=key_states.device)), dim=-2)
        value_pad = torch.cat((value_states, torch.zeros((*value_states.shape[:-2], pad, value_states.shape[-1]), dtype=value_states.dtype, device=value_states.device)), dim=-2)
        if attention_mask.is_floating_point():
            fill_value = torch.finfo(attention_mask.dtype).min
        else:
            fill_value = False
        mask_pad = torch.full((*attention_mask.shape[:-1], target_length), fill_value, dtype=attention_mask.dtype, device=attention_mask.device)
        mask_pad[..., :key_length] = attention_mask
        padded_kwargs = dict(kwargs)
        # SDPA's implicit causal diagonal is local to the q chunk; once K is
        # shape-padded the native 4-D mask must be the sole causal source.
        padded_kwargs["is_causal"] = False
        return old_backend(module, query, key_pad, value_pad, mask_pad, dropout=dropout, scaling=scaling, **padded_kwargs)

    # Keep config._attn_implementation unchanged so the native mask factory
    # still emits Gemma's sliding/full 4-D mask.
    ALL_ATTENTION_FUNCTIONS._global_mapping[old_backend_name] = padded_kv
    try:
        yield True
    finally:
        if old_registered is sentinel:
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(old_backend_name, None)
        else:
            ALL_ATTENTION_FUNCTIONS._global_mapping[old_backend_name] = old_registered


@contextmanager
def _patch_padded_qk_attention(lm: Any, target_length: int, layer_indices: set[int] | None = None) -> Iterator[bool]:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gemma4 import modeling_gemma4

    sentinel = object()
    old_backend_name = lm.config._attn_implementation
    old_backend = ALL_ATTENTION_FUNCTIONS.get_interface(
        old_backend_name,
        modeling_gemma4.eager_attention_forward,
    )
    old_registered = ALL_ATTENTION_FUNCTIONS._global_mapping.get(old_backend_name, sentinel)

    def padded_qk(module: Any, query: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor,
                  attention_mask: torch.Tensor | None, dropout: float = 0.0,
                  scaling: float | None = None, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        position_ids = kwargs.get("position_ids")
        if not (
            (layer_indices is None or int(getattr(module, "layer_idx", -1)) in layer_indices)
            and query.ndim == 4
            and key_states.ndim == 4
            and isinstance(position_ids, torch.Tensor)
            and query.shape[-2] < target_length
            and key_states.shape[-2] < target_length
            and int(position_ids.max().item()) < target_length
        ):
            return old_backend(module, query, key_states, value_states, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        query_length = query.shape[-2]
        key_length = key_states.shape[-2]
        batch_size, num_heads, _, head_dim = query.shape
        query_indices = position_ids.to(device=query.device, dtype=torch.long)
        # Place Q and the native 4-D mask at absolute positions.  This keeps
        # the cache-aware causal/sliding semantics while matching the full
        # sequence shape seen by the backend.
        query_pad = torch.zeros((batch_size, num_heads, target_length, head_dim), dtype=query.dtype, device=query.device)
        query_pad.scatter_(2, query_indices[:, None, :, None].expand(batch_size, num_heads, query_length, head_dim), query)
        key_pad = torch.cat((key_states, torch.zeros((*key_states.shape[:-2], target_length - key_length, key_states.shape[-1]), dtype=key_states.dtype, device=key_states.device)), dim=-2)
        value_pad = torch.cat((value_states, torch.zeros((*value_states.shape[:-2], target_length - key_length, value_states.shape[-1]), dtype=value_states.dtype, device=value_states.device)), dim=-2)
        if not isinstance(attention_mask, torch.Tensor):
            return old_backend(module, query, key_states, value_states, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        if attention_mask.is_floating_point():
            fill_value = torch.finfo(attention_mask.dtype).min
        else:
            fill_value = False
        mask_pad = torch.full((*attention_mask.shape[:-2], target_length, target_length), fill_value, dtype=attention_mask.dtype, device=attention_mask.device)
        for batch_index in range(batch_size):
            mask_pad[batch_index, ..., query_indices[batch_index], :key_length] = attention_mask[batch_index]
        padded_kwargs = dict(kwargs)
        padded_kwargs["is_causal"] = False
        output, weights = old_backend(module, query_pad, key_pad, value_pad, mask_pad, dropout=dropout, scaling=scaling, **padded_kwargs)
        selected = output.gather(1, query_indices[:, :, None, None].expand(batch_size, query_length, output.shape[-2], output.shape[-1]))
        if weights is None:
            selected_weights = None
        else:
            selected_weights = weights.gather(2, query_indices[:, None, :, None].expand(batch_size, weights.shape[1], query_length, weights.shape[-1]))
        return selected, selected_weights

    ALL_ATTENTION_FUNCTIONS._global_mapping[old_backend_name] = padded_qk
    try:
        yield True
    finally:
        if old_registered is sentinel:
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(old_backend_name, None)
        else:
            ALL_ATTENTION_FUNCTIONS._global_mapping[old_backend_name] = old_registered


@contextmanager
def _patch_padded_attention(target_length: int) -> Iterator[bool]:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gemma4 import modeling_gemma4

    key = "p4_padded_eager"
    sentinel = object()
    old_backend = ALL_ATTENTION_FUNCTIONS._global_mapping.get(key, sentinel)
    old_rope = modeling_gemma4.apply_rotary_pos_emb

    def padded_eager(module: Any, query: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor,
                     attention_mask: torch.Tensor | None, dropout: float = 0.0,
                     scaling: float | None = None, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward

        if query.ndim != 4 or key_states.ndim != 4 or not isinstance(attention_mask, torch.Tensor):
            return eager_attention_forward(module, query, key_states, value_states, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        query_length = query.shape[-2]
        key_length = key_states.shape[-2]
        if query_length >= target_length or key_length >= target_length:
            return eager_attention_forward(module, query, key_states, value_states, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        query_pad = torch.cat((query, torch.zeros((*query.shape[:-2], target_length - query_length, query.shape[-1]), dtype=query.dtype, device=query.device)), dim=-2)
        key_pad = torch.cat((key_states, torch.zeros((*key_states.shape[:-2], target_length - key_length, key_states.shape[-1]), dtype=key_states.dtype, device=key_states.device)), dim=-2)
        value_pad = torch.cat((value_states, torch.zeros((*value_states.shape[:-2], target_length - key_length, value_states.shape[-1]), dtype=value_states.dtype, device=value_states.device)), dim=-2)
        mask_pad = torch.full((*attention_mask.shape[:-2], target_length, target_length), torch.finfo(attention_mask.dtype).min, dtype=attention_mask.dtype, device=attention_mask.device)
        mask_pad[..., :query_length, :key_length] = attention_mask
        output, weights = eager_attention_forward(module, query_pad, key_pad, value_pad, mask_pad, dropout=dropout, scaling=scaling, **kwargs)
        return output[..., :query_length, :], weights[..., :query_length, :]

    def padded_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
        if x.ndim >= 3 and x.shape[1] < target_length:
            query_length = x.shape[1]
            x = torch.cat((x, torch.zeros((*x.shape[:1], target_length - query_length, *x.shape[2:]), dtype=x.dtype, device=x.device)), dim=1)
            cos = torch.cat((cos, torch.ones((*cos.shape[:1], target_length - query_length, *cos.shape[2:]), dtype=cos.dtype, device=cos.device)), dim=1)
            sin = torch.cat((sin, torch.zeros((*sin.shape[:1], target_length - query_length, *sin.shape[2:]), dtype=sin.dtype, device=sin.device)), dim=1)
            return old_rope(x, cos, sin, unsqueeze_dim=unsqueeze_dim)[:, :query_length]
        return old_rope(x, cos, sin, unsqueeze_dim=unsqueeze_dim)

    ALL_ATTENTION_FUNCTIONS._global_mapping[key] = padded_eager
    modeling_gemma4.apply_rotary_pos_emb = padded_rope
    try:
        yield True
    finally:
        if old_backend is sentinel:
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(key, None)
        else:
            ALL_ATTENTION_FUNCTIONS._global_mapping[key] = old_backend
        modeling_gemma4.apply_rotary_pos_emb = old_rope


@contextmanager
def _candidate(lm: Any, name: str) -> Iterator[bool]:
    if name == "baseline":
        yield True
        return
    if name in {"bf16_no_reduced_reduction", "shape_padded_math_rmsnorm"}:
        if name == "shape_padded_math_rmsnorm":
            linear_modules = [module for module in lm.modules() if isinstance(module, nn.Linear)]
            rmsnorm_modules = [
                module
                for layer in lm.layers
                for module in (layer.input_layernorm, layer.pre_feedforward_layernorm)
            ]
            target_length = int(getattr(lm, "_p4_padding_length", 0))
            with _patch_padded_linear_modules(linear_modules, target_length) as linear_applicable:
                with _patch_rmsnorm_modules(rmsnorm_modules, target_length) as norm_applicable:
                    yield linear_applicable and norm_applicable
        else:
            yield True
        return
    if name == "eager_attention":
        old = lm.config._attn_implementation
        lm.config._attn_implementation = "eager"
        try:
            yield True
        finally:
            lm.config._attn_implementation = old
        return
    if name in {"kv_projection_fp32", "qkv_projection_fp32", "attention_linear_fp32", "all_linear_fp32"}:
        modules: list[nn.Module] = []
        if name == "all_linear_fp32":
            modules = [module for module in lm.modules() if isinstance(module, nn.Linear)]
        else:
            for layer in lm.layers:
                attention = layer.self_attn
                paths = {
                    "kv_projection_fp32": ("k_proj", "v_proj"),
                    "qkv_projection_fp32": ("q_proj", "k_proj", "v_proj"),
                    "attention_linear_fp32": ("q_proj", "k_proj", "v_proj", "o_proj"),
                }[name]
                for path in paths:
                    module = getattr(attention, path, None)
                    if module is not None:
                        modules.append(module)
        with _patch_linear_modules(modules) as applicable:
            yield applicable
        return
    if name == "all_linear_padded_full_shape":
        modules = [module for module in lm.modules() if isinstance(module, nn.Linear)]
        target_length = int(getattr(lm, "_p4_padding_length", 0))
        with _patch_padded_linear_modules(modules, target_length) as applicable:
            yield applicable
        return
    if name == "shape_padded_fp32_activation":
        linear_modules = [module for module in lm.modules() if isinstance(module, nn.Linear)]
        activation_modules = [
            module
            for layer in lm.layers
            for module in (getattr(getattr(layer, "mlp", None), "act_fn", None), getattr(layer, "act_fn", None))
            if isinstance(module, nn.Module)
        ]
        target_length = int(getattr(lm, "_p4_padding_length", 0))
        with _patch_padded_linear_modules(linear_modules, target_length) as linear_applicable:
            with _patch_activation_modules(activation_modules) as activation_applicable:
                yield linear_applicable and activation_applicable
        return
    if name == "shape_padded_fp32_mlp_elementwise":
        linear_modules = [module for module in lm.modules() if isinstance(module, nn.Linear)]
        mlp_modules = [layer.mlp for layer in lm.layers if getattr(layer, "mlp", None) is not None]
        target_length = int(getattr(lm, "_p4_padding_length", 0))
        with _patch_padded_linear_modules(linear_modules, target_length) as linear_applicable:
            with _patch_mlp_elementwise(mlp_modules) as mlp_applicable:
                yield linear_applicable and mlp_applicable
        return
    if name in {"kv_projection_padded", "qkv_projection_padded", "attention_linear_padded", "mlp_linear_padded", "per_layer_linear_padded"}:
        modules: list[nn.Module] = []
        for layer in lm.layers:
            if name == "mlp_linear_padded":
                paths = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
            elif name == "per_layer_linear_padded":
                paths = ("per_layer_input_gate", "per_layer_projection")
            else:
                paths = {
                    "kv_projection_padded": ("self_attn.k_proj", "self_attn.v_proj"),
                    "qkv_projection_padded": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                    "attention_linear_padded": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"),
                }[name]
            for path in paths:
                module: Any = layer
                try:
                    for part in path.split("."):
                        module = getattr(module, part)
                except AttributeError:
                    continue
                if isinstance(module, nn.Linear):
                    modules.append(module)
        target_length = int(getattr(lm, "_p4_padding_length", 0))
        with _patch_padded_linear_modules(modules, target_length) as applicable:
            yield applicable
        return
    if name in {
        "shape_padded_rmsnorm_input_only",
        "shape_padded_rmsnorm_input_preff",
        "shape_padded_rmsnorm_input_preff_postper",
        "shape_padded_rmsnorm_input_postattn_preff",
        "shape_padded_rmsnorm_input_postattn_preff_postff",
        "shape_padded_rmsnorm_main",
        "shape_padded_rmsnorm",
    }:
        linear_modules = [module for module in lm.modules() if isinstance(module, nn.Linear)]
        all_rmsnorm_modules = [module for module in lm.modules() if type(module).__name__ == "Gemma4RMSNorm"]
        if name == "shape_padded_rmsnorm_input_only":
            rmsnorm_modules = [layer.input_layernorm for layer in lm.layers]
        elif name == "shape_padded_rmsnorm_input_preff":
            rmsnorm_modules = [
                module
                for layer in lm.layers
                for module in (layer.input_layernorm, layer.pre_feedforward_layernorm)
            ]
        elif name == "shape_padded_rmsnorm_input_preff_postper":
            rmsnorm_modules = [
                module
                for layer in lm.layers
                for module in (layer.input_layernorm, layer.pre_feedforward_layernorm, layer.post_per_layer_input_norm)
            ]
        elif name == "shape_padded_rmsnorm_input_postattn_preff":
            rmsnorm_modules = [
                module
                for layer in lm.layers
                for module in (layer.input_layernorm, layer.post_attention_layernorm, layer.pre_feedforward_layernorm)
            ]
        elif name == "shape_padded_rmsnorm_input_postattn_preff_postff":
            rmsnorm_modules = [
                module
                for layer in lm.layers
                for module in (
                    layer.input_layernorm,
                    layer.post_attention_layernorm,
                    layer.pre_feedforward_layernorm,
                    layer.post_feedforward_layernorm,
                )
            ]
        elif name == "shape_padded_rmsnorm_main":
            rmsnorm_modules = [
                module
                for layer in lm.layers
                for module in (
                    getattr(layer, "input_layernorm", None),
                    getattr(layer, "post_attention_layernorm", None),
                    getattr(layer, "pre_feedforward_layernorm", None),
                    getattr(layer, "post_feedforward_layernorm", None),
                    getattr(layer, "post_per_layer_input_norm", None),
                )
                if module is not None
            ]
        else:
            rmsnorm_modules = all_rmsnorm_modules
        target_length = int(getattr(lm, "_p4_padding_length", 0))
        with _patch_padded_linear_modules(linear_modules, target_length) as linear_applicable:
            with _patch_rmsnorm_modules(rmsnorm_modules, target_length) as norm_applicable:
                yield linear_applicable and norm_applicable
        return
    if name == "shape_padded_rmsnorm_kv":
        linear_modules = [module for module in lm.modules() if isinstance(module, nn.Linear)]
        rmsnorm_modules = [module for module in lm.modules() if type(module).__name__ == "Gemma4RMSNorm"]
        target_length = int(getattr(lm, "_p4_padding_length", 0))
        with _patch_padded_linear_modules(linear_modules, target_length) as linear_applicable:
            with _patch_rmsnorm_modules(rmsnorm_modules, target_length) as norm_applicable:
                with _patch_padded_kv_attention(lm, target_length) as attention_applicable:
                    yield linear_applicable and norm_applicable and attention_applicable
        return
    if name in {"transformer_linear_padded_rmsnorm", "no_o_projection_padded_rmsnorm", "no_mlp_padded_rmsnorm"}:
        all_linear = [(module_name, module) for module_name, module in lm.named_modules() if isinstance(module, nn.Linear)]
        if name == "transformer_linear_padded_rmsnorm":
            modules = [module for module_name, module in all_linear if "per_layer" not in module_name]
        elif name == "no_o_projection_padded_rmsnorm":
            modules = [module for module_name, module in all_linear if "o_proj" not in module_name]
        else:
            modules = [module for module_name, module in all_linear if ".mlp." not in module_name]
        rmsnorm_modules = [
            module
            for layer in lm.layers
            for module in (layer.input_layernorm, layer.pre_feedforward_layernorm)
        ]
        target_length = int(getattr(lm, "_p4_padding_length", 0))
        with _patch_padded_linear_modules(modules, target_length) as linear_applicable:
            with _patch_rmsnorm_modules(rmsnorm_modules, target_length) as norm_applicable:
                yield linear_applicable and norm_applicable
        return
    if name in {
        "shape_padded_all_ops",
        "shape_padded_all_ops_qk",
        "shape_padded_all_ops_kv",
        "shape_padded_all_ops_qk_l0",
        "shape_padded_all_ops_qk_l1",
        "shape_padded_all_ops_qk_l2",
        "shape_padded_all_ops_qk_l3",
        "shape_padded_all_ops_qk_l4",
        "shape_padded_all_ops_qk_l8",
    }:
        linear_modules = [module for module in lm.modules() if isinstance(module, nn.Linear)]
        rmsnorm_modules = [module for module in lm.modules() if type(module).__name__ == "Gemma4RMSNorm"]
        mlp_modules = [layer.mlp for layer in lm.layers if getattr(layer, "mlp", None) is not None]
        target_length = int(getattr(lm, "_p4_padding_length", 0))
        with _patch_padded_linear_modules(linear_modules, target_length) as linear_applicable:
            with _patch_rmsnorm_modules(rmsnorm_modules, target_length) as norm_applicable:
                with _patch_mlp_shape_modules(mlp_modules, target_length) as mlp_applicable:
                    if name.startswith("shape_padded_all_ops_qk"):
                        if name == "shape_padded_all_ops_qk":
                            qk_layers = None
                        else:
                            last_layer = int(name.rsplit("_l", 1)[1])
                            qk_layers = set(range(last_layer + 1))
                        with _patch_padded_qk_attention(lm, target_length, qk_layers) as attention_applicable:
                            yield linear_applicable and norm_applicable and mlp_applicable and attention_applicable
                    elif name == "shape_padded_all_ops_kv":
                        with _patch_padded_kv_attention(lm, target_length) as attention_applicable:
                            yield linear_applicable and norm_applicable and mlp_applicable and attention_applicable
                    else:
                        yield linear_applicable and norm_applicable and mlp_applicable
        return
    if name in {"shape_padded_qk_rmsnorm", "shape_padded_qk_rmsnorm_postper"}:
        linear_modules = [module for module in lm.modules() if isinstance(module, nn.Linear)]
        rmsnorm_modules = [
            module
            for layer in lm.layers
            for module in (
                layer.input_layernorm,
                layer.pre_feedforward_layernorm,
                *("shape_padded_qk_rmsnorm_postper" == name and (layer.post_per_layer_input_norm,) or ()),
            )
        ]
        target_length = int(getattr(lm, "_p4_padding_length", 0))
        with _patch_padded_linear_modules(linear_modules, target_length) as linear_applicable:
            with _patch_rmsnorm_modules(rmsnorm_modules, target_length) as norm_applicable:
                with _patch_padded_qk_attention(lm, target_length) as attention_applicable:
                    yield linear_applicable and norm_applicable and attention_applicable
        return
    if name == "shape_padded_cache_rmsnorm":
        linear_modules = [module for module in lm.modules() if isinstance(module, nn.Linear)]
        rmsnorm_modules = [
            module
            for layer in lm.layers
            for module in (layer.input_layernorm, layer.pre_feedforward_layernorm)
        ]
        target_length = int(getattr(lm, "_p4_padding_length", 0))
        with _patch_padded_linear_modules(linear_modules, target_length) as linear_applicable:
            with _patch_rmsnorm_modules(rmsnorm_modules, target_length) as norm_applicable:
                with _patch_padded_kv_attention(lm, target_length) as attention_applicable:
                    yield linear_applicable and norm_applicable and attention_applicable
        return
    if name == "shape_padded_static_cache_rmsnorm":
        linear_modules = [module for module in lm.modules() if isinstance(module, nn.Linear)]
        rmsnorm_modules = [
            module
            for layer in lm.layers
            for module in (layer.input_layernorm, layer.pre_feedforward_layernorm)
        ]
        target_length = int(getattr(lm, "_p4_padding_length", 0))
        old_static_length = getattr(lm, "_p4_static_cache_length", None)
        lm._p4_static_cache_length = int(getattr(lm, "_p4_sequence_length", target_length))
        try:
            with _patch_padded_linear_modules(linear_modules, target_length) as linear_applicable:
                with _patch_rmsnorm_modules(rmsnorm_modules, target_length) as norm_applicable:
                    yield linear_applicable and norm_applicable
        finally:
            if old_static_length is None:
                delattr(lm, "_p4_static_cache_length")
            else:
                lm._p4_static_cache_length = old_static_length
        return
    if name == "shape_padded_full_attention":
        linear_modules = [module for module in lm.modules() if isinstance(module, nn.Linear)]
        target_length = int(getattr(lm, "_p4_padding_length", 0))
        old_backend = lm.config._attn_implementation
        lm.config._attn_implementation = "p4_padded_eager"
        try:
            with _patch_padded_linear_modules(linear_modules, target_length) as linear_applicable:
                with _patch_padded_attention(target_length) as attention_applicable:
                    yield linear_applicable and attention_applicable
        finally:
            lm.config._attn_implementation = old_backend
        return
    if name == "router_projection_fp32":
        modules = [
            layer.router.proj
            for layer in lm.layers
            if getattr(layer, "enable_moe_block", False) and getattr(layer, "router", None) is not None
        ]
        with _patch_linear_modules(modules) as applicable:
            yield applicable
        return
    if name == "moe_experts_fp32":
        # Expert kernels are model/version-specific.  Do not silently patch
        # only the router and call that an expert ablation.
        applicable = any(bool(getattr(layer, "enable_moe_block", False)) for layer in lm.layers)
        if applicable:
            raise NotImplementedError("expert FP32 shadow kernel non implementato per questa versione Gemma")
        yield False
        return
    raise ValueError(f"candidate sconosciuto: {name}")


def _base_full(model: Any, ids: torch.Tensor, attention: torch.Tensor, features: torch.Tensor, feature_mask: torch.Tensor, positions: list[int]) -> torch.Tensor:
    with torch.inference_mode():
        output = model(
            input_ids=ids,
            attention_mask=attention,
            input_features=features,
            input_features_mask=feature_mask,
            use_cache=True,
            return_dict=True,
        )
    return _selected(output.last_hidden_state, positions)


def _markdown_section(report: dict[str, Any]) -> str:
    lines = [
        "## Ordered mixed-precision/backend ablations",
        "",
        "Ogni riga parte da una cache nuova e confronta l'incrementale candidato con lo stesso `Gemma4Model.forward` full BF16 non patchato. `applicable=false` significa che il checkpoint non contiene quel blocco (questo E2B non abilita MoE).",
        "",
        "| ordine | candidato | applicable | MAE | max abs | cosine min | P4 parity | audio p50/p95 ms | note |",
        "|---:|---|:---:|---:|---:|---:|:---:|---:|---|",
    ]
    for index, item in enumerate(report["candidates"], 1):
        cmp = item.get("comparison", {})
        audio_timing = item.get("timing", {}).get("incremental_audio_frames", {})
        lines.append(
            f"| {index} | `{item['name']}` | {item['applicable']} | {cmp.get('mae')} | "
            f"{cmp.get('max_abs')} | {cmp.get('cosine')} | {item.get('p4_parity')} | "
            f"{audio_timing.get('p50_ms')}/{audio_timing.get('p95_ms')} | {item.get('note', '')} |"
        )
    lines.extend([
        "",
        "La scelta di produzione resta sospesa finché un candidato non supera anche il gate completo (100 sequenze/5000 frame), la latenza p95 < 80 ms e la memoria residente Gemma+Qwen. Il percorso FP32 globale resta diagnostico.",
        "",
    ])
    return "\n".join(lines)


def run(
    frames: int,
    seed: int,
    output_dir: Path,
    markdown_path: Path,
    padding_length: int | None = None,
    candidate_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    thinker = load_thinker("/workspace", device="cuda", dtype=torch.bfloat16)
    model = thinker.model.model
    try:
        ids, attention, features, feature_mask, audio_soft, audio_start = _fixture(model, frames, seed)
        positions = (ids[0] == int(model.config.audio_token_id)).nonzero(as_tuple=False).flatten().tolist()
        model.language_model._p4_padding_length = int(padding_length or ids.shape[1])
        model.language_model._p4_sequence_length = int(ids.shape[1])
        candidates: list[dict[str, Any]] = []
        selected_candidates = candidate_names or ORDERED_CANDIDATES
        old_reduced_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        if "bf16_no_reduced_reduction" in selected_candidates:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        old_sdp_flags = (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )
        if "shape_padded_math_rmsnorm" in selected_candidates:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        base = _base_full(model, ids, attention, features, feature_mask, positions)
        unknown = set(selected_candidates) - set(ORDERED_CANDIDATES)
        if unknown:
            raise ValueError(f"candidate sconosciuti: {sorted(unknown)}")
        for name in selected_candidates:
            capture = NullCapture(set(range(len(positions))))
            # _incremental stores final rows indexed by absolute positions.  A
            # compact local comparison is enough here; the bisect tool carries
            # the detailed per-layer trace.
            capture.positions = set(positions)
            with _candidate(model.language_model, name) as applicable:
                # An unavailable block is measured as an explicit no-op so
                # the report remains total without claiming it was tested.
                result = _incremental(
                    model,
                    mode_name=f"ablation_{name}",
                    input_ids=ids,
                    attention_mask=attention,
                    audio_soft=audio_soft,
                    audio_start=audio_start,
                    capture=capture,
                    cache_length=getattr(model.language_model, "_p4_static_cache_length", None),
                )
            if capture.final is None:
                raise RuntimeError(f"candidate {name} non ha prodotto hidden finale")
            comparison = _compare(base, capture.final)
            p4_parity = bool(
                comparison.get("available")
                and not comparison.get("shape_mismatch")
                and float(comparison.get("mae", float("inf"))) <= 1e-3
                and float(comparison.get("cosine", -1.0)) >= 0.9999
            )
            note = ""
            if not applicable:
                note = "blocco assente nel checkpoint"
            candidates.append({
                "name": name,
                "applicable": applicable,
                "comparison": comparison,
                "p4_parity": p4_parity,
                "timing": result.timing(),
                "peak_allocated_bytes": result.peak_allocated,
                "peak_reserved_bytes": result.peak_reserved,
                "note": note,
            })
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduced_reduction
        torch.backends.cuda.enable_flash_sdp(old_sdp_flags[0])
        torch.backends.cuda.enable_mem_efficient_sdp(old_sdp_flags[1])
        torch.backends.cuda.enable_math_sdp(old_sdp_flags[2])
        report = {
            "schema_version": 1,
            "phase": "P4",
            "tool": "p4_precision_ablations",
            "status": "diagnostic",
            "fixture": {
                "frames": frames,
                "seed": seed,
                "audio_positions": positions,
                "audio_start": audio_start,
                "padding_length": int(model.language_model._p4_padding_length),
            },
            "oracle": "untouched native Gemma4Model.forward BF16",
            "candidate_contract": "stable prefill + 4 audio soft tokens/frame + native masks/cache; no Qwen/V9 changes",
            "candidates": candidates,
        }
        json_path = output_dir / "P4_PRECISION_ABLATIONS.json"
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        marker = "## Ordered mixed-precision/backend ablations"
        existing = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else "# P4 Numeric Bisect\n"
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n" + _markdown_section(report)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(existing, encoding="utf-8")
        report["json_path"] = str(json_path)
        report["markdown_path"] = str(markdown_path)
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        del thinker, model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "validation" / "kernel_gates" / "numeric_bisect")
    parser.add_argument("--padding-length", type=int, default=None)
    parser.add_argument("--candidates", default=None, help="lista separata da virgole; default: tutte le ablation")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs" / "P4_NUMERIC_BISECT.md")
    args = parser.parse_args()
    candidate_names = tuple(value for value in args.candidates.split(",") if value) if args.candidates else None
    report = run(args.frames, args.seed, args.output_dir, args.markdown, args.padding_length, candidate_names)
    print(json.dumps({"candidates": report["candidates"], "json_path": report["json_path"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
