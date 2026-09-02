#!/usr/bin/env python3
"""Layer-by-layer numerical bisect for the Gemma P4 representation blocker.

The full path is the native multimodal ``Gemma4Model.forward``.  The two
incremental paths use the same precomputed audio soft tokens, text prefix,
PLE, positions and native hybrid masks; only the transition shape/cache differs
(and, for the diagnostic path, the LM is executed in FP32).

This is a diagnostic tool, not a gate and not a replacement oracle.  Reports
contain hashes and scalar comparisons, never full intermediate tensors.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.gemma_representation import build_audio_placeholder_inputs  # noqa: E402
from duplex_kernel.models.thinker import load_thinker  # noqa: E402


AUDIO_FEATURES_PER_FRAME = 4
SIGNIFICANT_MAE = 1e-3
SIGNIFICANT_COSINE = 0.9999
COMPONENT_ORDER = (
    "input_hidden",
    "pre_attention_norm",
    "q_projection",
    "q_norm",
    "k_projection",
    "k_norm",
    "v_projection",
    "v_norm",
    "attention_context",
    "attention_output",
    "post_attention_norm",
    "post_attention_residual",
    "dense_mlp_output",
    "router_logits",
    "router_probabilities",
    "router_weights",
    "router_topk",
    "moe_output",
    "post_feedforward_norm",
    "post_mlp_hidden",
    "shared_kv",
    "normal_kv",
    "final_hidden",
)
COMPONENT_ORDER_INDEX = {name: index for index, name in enumerate(COMPONENT_ORDER)}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _finite(value: Any) -> bool:
    return bool(torch.isfinite(value).all().item()) if isinstance(value, torch.Tensor) else math.isfinite(float(value))


def _tensor_hash(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    # NumPy has no bfloat16 dtype on the runtime used by the kernel.  Hash the
    # raw bytes through a uint8 view so BF16/FP32/int tensors are all covered.
    payload = cpu.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values]
    return {
        "samples": len(numbers),
        "p50_ms": _percentile(numbers, 50),
        "p95_ms": _percentile(numbers, 95),
        "p99_ms": _percentile(numbers, 99),
        "max_ms": max(numbers) if numbers else None,
    }


def _position_list(position_ids: torch.Tensor | None, query_length: int) -> list[int]:
    if position_ids is None:
        return list(range(query_length))
    value = position_ids.detach()
    while value.ndim > 2:
        value = value[0]
    if value.ndim == 2:
        value = value[0]
    return [int(item) for item in value.reshape(-1).tolist()]


def _output_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _output_tensor(output[0])
    return None


def _to_query_rows(value: torch.Tensor, query_length: int) -> torch.Tensor | None:
    """Normalize module outputs to ``(query, features...)`` for B=1."""
    if value.ndim == 0:
        return None
    if value.ndim == 4 and value.shape[0] == 1 and value.shape[2] == query_length:
        # Attention projections before/after head transposes: (B,H,Q,D).
        return value[0].transpose(0, 1)
    if value.ndim >= 3 and value.shape[0] == 1 and value.shape[1] == query_length:
        return value[0]
    if value.ndim == 2 and value.shape[0] == query_length:
        return value
    if value.ndim == 2 and value.shape[0] % max(query_length, 1) == 0:
        # Router and expert modules flatten B*Q.
        return value.reshape(-1, query_length, value.shape[-1])[0]
    return None


def _cache_pair(cache: Any, layer_index: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    if cache is None:
        return None
    layers = getattr(cache, "layers", None)
    if layers is not None and layer_index < len(layers):
        layer = layers[layer_index]
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor):
            return keys, values
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if isinstance(key_cache, (list, tuple)) and isinstance(value_cache, (list, tuple)):
        if layer_index < len(key_cache) and layer_index < len(value_cache):
            keys, values = key_cache[layer_index], value_cache[layer_index]
            if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor):
                return keys, values
    return None


def _prefix_state(value: torch.Tensor, position: int) -> torch.Tensor:
    if value.ndim >= 3:
        # Gemma K/V states are (B, H, sequence, D).  For unusual layouts this
        # still selects the penultimate dimension, which is the sequence axis.
        end = min(position + 1, value.shape[-2])
        return value[..., :end, :]
    return value


def _router_margin(logits: torch.Tensor) -> float | None:
    if logits.ndim != 1 or logits.numel() < 2:
        return None
    ordered = torch.sort(logits.float(), descending=True).values
    return float((ordered[0] - ordered[1]).item())


@dataclass
class Capture:
    audio_positions: set[int]
    records: dict[tuple[int, str], dict[int, torch.Tensor]] = field(default_factory=lambda: defaultdict(dict))
    active_positions: dict[int, list[int]] = field(default_factory=dict)
    pending: dict[int, dict[str, Any]] = field(default_factory=dict)
    handles: list[Any] = field(default_factory=list)

    def _store(self, layer: int, component: str, positions: list[int], value: Any) -> None:
        tensor = _output_tensor(value)
        if tensor is None:
            return
        rows = _to_query_rows(tensor.detach(), len(positions))
        if rows is None or rows.shape[0] != len(positions):
            return
        target = self.records[(layer, component)]
        for index, position in enumerate(positions):
            if position not in self.audio_positions:
                continue
            row = rows[index].detach().cpu().contiguous()
            target[position] = row

    def _store_state(self, layer: int, component: str, positions: list[int], value: Any) -> None:
        if not isinstance(value, torch.Tensor):
            return
        target = self.records[(layer, component)]
        for position in positions:
            if position not in self.audio_positions:
                continue
            target[position] = _prefix_state(value.detach(), position).cpu().contiguous()

    def _layer_pre(self, layer: int, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        del module
        hidden = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden, torch.Tensor):
            return
        positions = _position_list(kwargs.get("position_ids"), hidden.shape[1])
        self.active_positions[layer] = positions
        self.pending[layer] = {"positions": positions, "input": hidden.detach().cpu().contiguous()}
        self._store(layer, "input_hidden", positions, hidden)

    def _module_hook(self, layer: int, component: str, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        del module, kwargs
        positions = self.active_positions.get(layer, [])
        if not positions:
            return
        if component in {"post_attention_norm", "post_feedforward_norm"}:
            tensor = _output_tensor(output)
            if tensor is not None:
                self.pending.setdefault(layer, {})[component] = tensor.detach().cpu().contiguous()
        elif component == "router":
            if isinstance(output, (tuple, list)) and len(output) >= 3:
                self._store(layer, "router_probabilities", positions, output[0])
                self._store(layer, "router_weights", positions, output[1])
                self._store(layer, "router_topk", positions, output[2])
        else:
            self._store(layer, component, positions, output)

    def _layer_post(self, layer: int, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        del module, args
        pending = self.pending.pop(layer, None)
        positions = self.active_positions.get(layer, [])
        if pending is not None and positions:
            input_hidden = pending.get("input")
            post_attention = pending.get("post_attention_norm")
            if isinstance(input_hidden, torch.Tensor) and isinstance(post_attention, torch.Tensor):
                residual = input_hidden + post_attention
                self._store(layer, "post_attention_residual", positions, residual)
                pending["post_attention_residual"] = residual
            post_feedforward = pending.get("post_feedforward_norm")
            residual = pending.get("post_attention_residual")
            if isinstance(residual, torch.Tensor) and isinstance(post_feedforward, torch.Tensor):
                self._store(layer, "post_mlp_hidden", positions, residual + post_feedforward)

        shared = kwargs.get("shared_kv_states")
        if isinstance(shared, Mapping):
            for key, pair in shared.items():
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    continue
                for suffix, value in zip(("key", "value"), pair):
                    self._store_state(layer, f"shared_kv.{key}.{suffix}", positions, value)

        cache = kwargs.get("past_key_values")
        pair = _cache_pair(cache, layer)
        if pair is not None:
            self._store_state(layer, "normal_kv.key", positions, pair[0])
            self._store_state(layer, "normal_kv.value", positions, pair[1])
        del output

    def install(self, language_model: Any) -> None:
        for layer_index, layer in enumerate(language_model.layers):
            self.handles.append(
                layer.register_forward_pre_hook(
                    lambda module, args, kwargs, index=layer_index: self._layer_pre(index, module, args, kwargs),
                    with_kwargs=True,
                )
            )
            self.handles.append(
                layer.register_forward_hook(
                    lambda module, args, kwargs, output, index=layer_index: self._layer_post(index, module, args, kwargs, output),
                    with_kwargs=True,
                )
            )
            module_specs = (
                ("input_layernorm", "pre_attention_norm"),
                ("self_attn.q_proj", "q_projection"),
                ("self_attn.q_norm", "q_norm"),
                ("self_attn.k_proj", "k_projection"),
                ("self_attn.k_norm", "k_norm"),
                ("self_attn.v_proj", "v_projection"),
                ("self_attn.v_norm", "v_norm"),
                ("self_attn.o_proj", "attention_output"),
                ("post_attention_layernorm", "post_attention_norm"),
                ("mlp", "dense_mlp_output"),
                ("router.proj", "router_logits"),
                ("router", "router"),
                ("experts", "moe_output"),
                ("post_feedforward_layernorm", "post_feedforward_norm"),
            )
            for path, component in module_specs:
                module: Any = layer
                try:
                    for name in path.split("."):
                        module = getattr(module, name)
                except AttributeError:
                    continue
                if component == "attention_output":
                    self.handles.append(
                        module.register_forward_pre_hook(
                            lambda mod, args, kwargs, index=layer_index: self._module_hook(
                                index, "attention_context", mod, args, kwargs, args[0] if args else None
                            ),
                            with_kwargs=True,
                        )
                    )
                self.handles.append(
                    module.register_forward_hook(
                        lambda mod, args, kwargs, output, index=layer_index, name=component: self._module_hook(
                            index, name, mod, args, kwargs, output
                        ),
                        with_kwargs=True,
                    )
                )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def store_final(self, hidden: torch.Tensor) -> None:
        positions = sorted(self.audio_positions)
        for position in positions:
            if position < hidden.shape[1]:
                self.records[(-1, "final_hidden")][position] = hidden[0, position].detach().cpu().contiguous()


@dataclass
class ModeResult:
    name: str
    capture: Capture
    output_shape: list[int]
    total_ms: float
    stable_ms: float | None
    audio_ms: list[float]
    peak_allocated: int
    peak_reserved: int

    def timing(self) -> dict[str, Any]:
        return {
            "total": _timing_summary([self.total_ms]),
            "stable_prefill": _timing_summary([self.stable_ms]) if self.stable_ms is not None else None,
            "incremental_audio_frames": _timing_summary(self.audio_ms),
            "instrumented": True,
        }


def _native_full(
    model: Any,
    *,
    mode_name: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_features: torch.Tensor,
    input_features_mask: torch.Tensor,
    capture: Capture,
    audio_soft: torch.Tensor | None = None,
) -> ModeResult:
    device = input_ids.device
    capture.install(model.language_model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        if mode_name == "full_fp32":
            if audio_soft is None:
                raise ValueError("full_fp32 richiede audio_soft precomputato")
            text_config = model.config.get_text_config()
            audio_mask = input_ids == int(model.config.audio_token_id)
            llm_ids = torch.where(audio_mask, torch.as_tensor(text_config.pad_token_id, device=device), input_ids)
            inputs_embeds = model.language_model.embed_tokens(llm_ids)
            pad_embedding = model.language_model.embed_tokens.weight[int(text_config.pad_token_id)]
            llm_inputs_embeds = torch.where(audio_mask.unsqueeze(-1), pad_embedding.view(1, 1, -1), inputs_embeds)
            # Keep the raw identity PLE here.  Gemma4TextModel performs the
            # single context projection using the merged (audio-soft) main
            # embeddings, exactly as the native multimodal forward does.
            per_layer_inputs = model.language_model.get_per_layer_inputs(llm_ids, llm_inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_mask.unsqueeze(-1), audio_soft.to(device=device, dtype=inputs_embeds.dtype)
            )
            output = model.language_model(
                inputs_embeds=inputs_embeds,
                per_layer_inputs=per_layer_inputs,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        else:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                input_features_mask=input_features_mask,
                use_cache=True,
                return_dict=True,
            )
    _sync(device)
    elapsed = (time.perf_counter() - started) * 1000.0
    capture.store_final(output.last_hidden_state)
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    capture.close()
    return ModeResult(
        name=mode_name,
        capture=capture,
        output_shape=list(output.last_hidden_state.shape),
        total_ms=elapsed,
        stable_ms=None,
        audio_ms=[],
        peak_allocated=peak_allocated,
        peak_reserved=peak_reserved,
    )


def _incremental(
    model: Any,
    *,
    mode_name: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    audio_soft: torch.Tensor,
    audio_start: int,
    capture: Capture,
    cache_length: int | None = None,
) -> ModeResult:
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

    device = input_ids.device
    lm = model.language_model
    capture.install(lm)
    lm_dtype = lm.embed_tokens.weight.dtype
    batch_size = input_ids.shape[0]
    audio_tokens = audio_soft.shape[1]
    stable_ids = input_ids[:, :audio_start]
    stable_attention = attention_mask[:, :audio_start].to(device=device, dtype=torch.long)
    stable_embeds = lm.embed_tokens(stable_ids)
    stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)
    pad_ids = torch.full((batch_size, AUDIO_FEATURES_PER_FRAME), int(getattr(model.config.text_config, "pad_token_id", 0)), dtype=torch.long, device=device)
    pad_embeds = lm.embed_tokens(pad_ids)
    pad_ple = lm.get_per_layer_inputs(pad_ids, pad_embeds)

    past: Any | None = None
    if cache_length is not None:
        from transformers.cache_utils import StaticCache
        past = StaticCache(config=model.config, max_cache_len=int(cache_length))
    shared: Any | None = None
    shared_enabled = any(bool(getattr(getattr(layer, "self_attn", None), "is_kv_shared_layer", False)) for layer in lm.layers)
    chunks: list[torch.Tensor] = []

    def call(embeds: torch.Tensor, per_layer: torch.Tensor, positions: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        nonlocal past, shared
        position_ids = positions.view(1, -1).expand(batch_size, -1)
        mask_mapping = create_masks_for_generate(
            model.config,
            embeds,
            attn,
            past_key_values=past,
            position_ids=position_ids,
        )
        kwargs: dict[str, Any] = {
            "inputs_embeds": embeds,
            "per_layer_inputs": per_layer,
            "attention_mask": mask_mapping,
            "position_ids": position_ids,
            "past_key_values": past,
            "use_cache": True,
            "return_dict": True,
        }
        if shared_enabled:
            kwargs["return_shared_kv_states"] = True
            if shared is not None:
                kwargs["shared_kv_states"] = shared
        output = lm(**kwargs)
        past = output.past_key_values
        next_shared = getattr(output, "shared_kv_states", None)
        if next_shared is None:
            next_shared = getattr(past, "shared_layers", None)
        if shared_enabled and next_shared is None:
            raise RuntimeError("Gemma incremental non ha restituito shared_kv_states")
        shared = next_shared
        return output.last_hidden_state

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    total_started = time.perf_counter()
    stable_started = time.perf_counter()
    with torch.inference_mode():
        stable_hidden = call(
            stable_embeds,
            stable_ple,
            torch.arange(audio_start, device=device, dtype=torch.long),
            stable_attention,
        )
    _sync(device)
    stable_ms = (time.perf_counter() - stable_started) * 1000.0
    chunks.append(stable_hidden)

    audio_times: list[float] = []
    audio_soft = audio_soft.to(device=device, dtype=lm_dtype)
    for start in range(0, audio_tokens, AUDIO_FEATURES_PER_FRAME):
        frame_audio = audio_soft[:, start:start + AUDIO_FEATURES_PER_FRAME, :]
        positions = torch.arange(audio_start + start, audio_start + start + AUDIO_FEATURES_PER_FRAME, device=device, dtype=torch.long)
        attn = torch.ones((batch_size, audio_start + start + AUDIO_FEATURES_PER_FRAME), device=device, dtype=torch.long)
        attn[:, :audio_start] = stable_attention
        frame_started = time.perf_counter()
        with torch.inference_mode():
            frame_hidden = call(frame_audio, pad_ple, positions, attn)
        _sync(device)
        audio_times.append((time.perf_counter() - frame_started) * 1000.0)
        chunks.append(frame_hidden)

    # EOA is not needed to influence any audio position, but running it keeps
    # the state transition shape identical to the canonical helper.
    eoa_ids = input_ids[:, -1:]
    eoa_embeds = lm.embed_tokens(eoa_ids)
    eoa_ple = lm.get_per_layer_inputs(eoa_ids, eoa_embeds)
    eoa_position = audio_start + audio_tokens
    eoa_attn = torch.ones((batch_size, eoa_position + 1), device=device, dtype=torch.long)
    eoa_attn[:, :audio_start] = stable_attention
    with torch.inference_mode():
        eoa_hidden = call(
            eoa_embeds,
            eoa_ple,
            torch.full((1,), eoa_position, device=device, dtype=torch.long),
            eoa_attn,
        )
    _sync(device)
    chunks.append(eoa_hidden[:, :1, :])
    result = torch.cat(chunks, dim=1)
    capture.store_final(result)
    total_ms = (time.perf_counter() - total_started) * 1000.0
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    capture.close()
    return ModeResult(
        name=mode_name,
        capture=capture,
        output_shape=list(result.shape),
        total_ms=total_ms,
        stable_ms=stable_ms,
        audio_ms=audio_times,
        peak_allocated=peak_allocated,
        peak_reserved=peak_reserved,
    )


def _load_gemma_model() -> tuple[Any, Any]:
    thinker = load_thinker("/workspace", device="cuda", dtype=torch.bfloat16)
    model = thinker.model.model
    return thinker, model


def _fixture(model: Any, frames: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    device = torch.device("cuda")
    torch.manual_seed(seed)
    mel_rows = frames * 8
    input_features = torch.randn((1, mel_rows * 2, 128), device=device, dtype=torch.bfloat16)
    input_features_mask = torch.ones((1, input_features.shape[1]), device=device, dtype=torch.bool)
    with torch.inference_mode():
        audio_output = model.get_audio_features(
            input_features=input_features,
            input_features_mask=input_features_mask,
            return_dict=True,
        )
    pooled = audio_output.pooler_output
    valid_mask = audio_output.attention_mask.to(pooled.device)
    rows = [pooled[index][valid_mask[index]] for index in range(pooled.shape[0])]
    if len({int(row.shape[0]) for row in rows}) != 1:
        raise RuntimeError("fixture Gemma con numero di audio token non uniforme")
    audio_soft = torch.stack(rows, dim=0)
    if audio_soft.shape[1] % AUDIO_FEATURES_PER_FRAME:
        raise RuntimeError(f"audio soft token non divisibili per 4: {tuple(audio_soft.shape)}")
    config = model.config
    ids, attention = build_audio_placeholder_inputs(
        batch_size=1,
        audio_feature_rows=int(audio_soft.shape[1] * 4),
        device=device,
        audio_token_id=int(getattr(config, "audio_token_id", 258881)),
        bos_token_id=int(getattr(config, "bos_token_id", 2)),
        boa_token_id=int(getattr(config, "boa_token_id", 256000)),
        eoa_token_id=int(getattr(config, "eoa_token_id", 258883)),
        pad_token_id=int(getattr(config.text_config, "pad_token_id", 0)),
    )
    audio_positions = (ids[0] == int(getattr(config, "audio_token_id", 258881))).nonzero(as_tuple=False).flatten().tolist()
    if len(audio_positions) != audio_soft.shape[1]:
        raise RuntimeError(f"placeholder/audio soft mismatch: {len(audio_positions)} != {audio_soft.shape[1]}")
    audio_start = int(audio_positions[0])
    return ids, attention, input_features, input_features_mask, audio_soft, audio_start


def _compare(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"available": False, "shape_mismatch": False}
    if tuple(a.shape) != tuple(b.shape):
        return {
            "available": True,
            "shape_mismatch": True,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }
    a_finite = bool(torch.isfinite(a).all().item()) if a.is_floating_point() else True
    b_finite = bool(torch.isfinite(b).all().item()) if b.is_floating_point() else True
    result: dict[str, Any] = {
        "available": True,
        "shape_mismatch": False,
        "shape": list(a.shape),
        "hash_a": _tensor_hash(a),
        "hash_b": _tensor_hash(b),
        "finite_a": a_finite,
        "finite_b": b_finite,
    }
    if a.dtype == torch.bool or not a.is_floating_point():
        mismatch = (a != b).float()
        result.update({
            "mae": float(mismatch.mean().item()),
            "max_abs": float(mismatch.max().item()) if mismatch.numel() else 0.0,
            "cosine": None,
            "mismatch_rate": float(mismatch.mean().item()),
        })
        return result
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    delta = (af - bf).abs()
    result.update({
        "mae": float(delta.mean().item()) if delta.numel() else 0.0,
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "cosine": float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0), dim=-1, eps=1e-12).item()) if af.numel() else 1.0,
        "mismatch_rate": None,
    })
    return result


def _record_sort_key(key: tuple[int, str]) -> tuple[int, int, str]:
    layer, component = key
    base = component.split(".", 1)[0]
    return (layer, COMPONENT_ORDER_INDEX.get(base, len(COMPONENT_ORDER)), component)


def _aggregate(events: list[dict[str, Any]], pair: str) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        comparison = event.get(pair)
        if comparison and comparison.get("available") and not comparison.get("shape_mismatch"):
            groups[(int(event["layer"]), str(event["component"]))].append(comparison)
    summary: list[dict[str, Any]] = []
    for (layer, component), values in sorted(groups.items(), key=lambda item: _record_sort_key(item[0])):
        maes = [float(value["mae"]) for value in values if value.get("mae") is not None]
        maxima = [float(value["max_abs"]) for value in values if value.get("max_abs") is not None]
        cosines = [float(value["cosine"]) for value in values if value.get("cosine") is not None]
        mismatches = [float(value["mismatch_rate"]) for value in values if value.get("mismatch_rate") is not None]
        summary.append({
            "layer": layer,
            "component": component,
            "positions": len(values),
            "mae_mean": sum(maes) / len(maes) if maes else None,
            "mae_max": max(maes) if maes else None,
            "max_abs": max(maxima) if maxima else None,
            "cosine_min": min(cosines) if cosines else None,
            "mismatch_rate": sum(mismatches) / len(mismatches) if mismatches else None,
            "finite": all(bool(value.get("finite_a", False) and value.get("finite_b", False)) for value in values),
        })
    return summary


def _is_significant(summary: dict[str, Any]) -> bool:
    if summary.get("mismatch_rate") is not None:
        return float(summary["mismatch_rate"]) > 0.0
    mae = summary.get("mae_max")
    cosine = summary.get("cosine_min")
    return bool((mae is not None and float(mae) > SIGNIFICANT_MAE) or (cosine is not None and float(cosine) < SIGNIFICANT_COSINE))


def _first_divergence(summary: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in summary:
        if item["layer"] < 0 or item["component"] == "final_hidden":
            continue
        if _is_significant(item):
            return item
    return None


def _router_report(events: list[dict[str, Any]], pair: str) -> dict[str, Any]:
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("component") != "router_topk":
            continue
        comparison = event.get(pair)
        if comparison and comparison.get("available") and not comparison.get("shape_mismatch"):
            by_layer[int(event["layer"])].append(event)
    layers: list[dict[str, Any]] = []
    first_layer = None
    margins: list[float] = []
    for layer in sorted(by_layer):
        values = by_layer[layer]
        mismatch = [float(item[pair].get("mismatch_rate", 0.0)) for item in values]
        mismatch_rate = sum(mismatch) / len(mismatch) if mismatch else 0.0
        for item in values:
            if float(item[pair].get("mismatch_rate", 0.0)) > 0.0:
                margin_key = "full_fp32_router_margin" if pair.startswith("full_fp32") else "full_router_margin"
                margin = item.get(margin_key)
                if margin is not None:
                    margins.append(float(margin))
        if mismatch_rate > 0.0 and first_layer is None:
            first_layer = layer
        layers.append({
            "layer": layer,
            "positions": len(values),
            "expert_mismatch_rate": mismatch_rate,
            "mismatch_positions": sum(value > 0.0 for value in mismatch),
        })
    return {
        "layers": layers,
        "first_router_mismatch_layer": first_layer,
        "router_logit_margin_at_mismatch_mean": sum(margins) / len(margins) if margins else None,
        "router_logit_margin_at_mismatch_min": min(margins) if margins else None,
    }


def _build_events(results: dict[str, ModeResult], audio_start: int, audio_positions: list[int]) -> list[dict[str, Any]]:
    del audio_positions
    full = results["full_bf16"].capture.records
    full_fp32 = results["full_fp32"].capture.records
    all_keys = set(full) | set(full_fp32)
    for result in results.values():
        all_keys.update(result.capture.records)
    events: list[dict[str, Any]] = []
    for layer, component in sorted(all_keys, key=_record_sort_key):
        positions = set()
        for result in results.values():
            positions.update(result.capture.records.get((layer, component), {}))
        for position in sorted(positions):
            a = full.get((layer, component), {}).get(position)
            d = full_fp32.get((layer, component), {}).get(position)
            b = results["incremental_bf16"].capture.records.get((layer, component), {}).get(position)
            c = results["incremental_fp32"].capture.records.get((layer, component), {}).get(position)
            event: dict[str, Any] = {
                "event": "p4_numeric_bisect",
                "layer": layer,
                "component": component,
                "position": position,
                "frame_id": (position - audio_start) // AUDIO_FEATURES_PER_FRAME if position >= audio_start else None,
                "audio_token_in_frame": (position - audio_start) % AUDIO_FEATURES_PER_FRAME if position >= audio_start else None,
                "full_bf16_vs_incremental_bf16": _compare(a, b),
                # This comparison includes the BF16->FP32 precision change and
                # is retained for attribution, not as the FP32 parity oracle.
                "full_bf16_vs_incremental_fp32": _compare(a, c),
                "full_fp32_vs_incremental_fp32": _compare(d, c),
                "full_bf16_vs_full_fp32": _compare(a, d),
            }
            full_logits = full.get((layer, "router_logits"), {}).get(position)
            full_fp32_logits = full_fp32.get((layer, "router_logits"), {}).get(position)
            event["full_router_margin"] = _router_margin(full_logits) if full_logits is not None else None
            event["full_fp32_router_margin"] = _router_margin(full_fp32_logits) if full_fp32_logits is not None else None
            if component == "router_topk":
                event["expert_ids"] = {
                    "full_bf16": a.tolist() if a is not None else None,
                    "incremental_bf16": b.tolist() if b is not None else None,
                    "full_fp32": d.tolist() if d is not None else None,
                    "incremental_fp32": c.tolist() if c is not None else None,
                }
            events.append(event)
    return events


def _representation_summary(events: list[dict[str, Any]], pair: str) -> dict[str, Any]:
    selected = [event for event in events if event["component"] == "final_hidden" and event.get(pair, {}).get("available")]
    comparisons = [event[pair] for event in selected if not event[pair].get("shape_mismatch")]
    maes = [float(item["mae"]) for item in comparisons if item.get("mae") is not None]
    maxima = [float(item["max_abs"]) for item in comparisons if item.get("max_abs") is not None]
    cosines = [float(item["cosine"]) for item in comparisons if item.get("cosine") is not None]
    return {
        "audio_tokens": len(selected),
        "mae": sum(maes) / len(maes) if maes else None,
        "mae_max_position": max(maes) if maes else None,
        "max_abs": max(maxima) if maxima else None,
        "cosine_min": min(cosines) if cosines else None,
        "shape_mismatches": sum(bool(event[pair].get("shape_mismatch")) for event in selected),
        "nonfinite": sum(not bool(item.get("finite_a", False) and item.get("finite_b", False)) for item in comparisons),
    }


def _markdown(report: dict[str, Any]) -> str:
    first_b = report["first_divergence"]["full_bf16_vs_incremental_bf16"]
    first_c = report["first_divergence"]["full_fp32_vs_incremental_fp32"]
    def first_line(item: dict[str, Any] | None) -> str:
        if item is None:
            return "nessuna divergenza significativa rilevata"
        return (
            f"layer `{item['layer']}`, componente `{item['component']}`, "
            f"MAE max `{item.get('mae_max')}`, cosine min `{item.get('cosine_min')}`"
        )
    lines = [
        "# P4 Numeric Bisect — Gemma incremental BF16",
        "",
        f"**Status:** diagnostico; non è un gate e non modifica l'oracle canonico.  ",
        f"**Fixture:** {report['fixture']['frames']} frame, seed `{report['fixture']['seed']}`, "
        f"audio soft hash `{report['fixture']['audio_soft_sha256']}`.",
        "",
        "## Primo punto divergente",
        "",
        f"- Full BF16 vs incremental BF16: {first_line(first_b)}.",
        f"- Full FP32 diagnostico vs incremental FP32: {first_line(first_c)}.",
        f"- Significatività: MAE > `{SIGNIFICANT_MAE}` oppure cosine < `{SIGNIFICANT_COSINE}`; per top-k basta un expert mismatch.",
        "",
        "## Representation finale",
        "",
        "| confronto | MAE | max abs | cosine min | shape mismatch | non-finite |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, key in (
        ("full BF16 → incremental BF16", "full_bf16_vs_incremental_bf16"),
        ("full FP32 → incremental FP32 (diagnostico)", "full_fp32_vs_incremental_fp32"),
        ("full BF16 → full FP32 (precision drift)", "full_bf16_vs_full_fp32"),
    ):
        item = report["representation"][key]
        lines.append(
            f"| {name} | {item['mae']} | {item['max_abs']} | {item['cosine_min']} | "
            f"{item['shape_mismatches']} | {item['nonfinite']} |"
        )
    lines.extend([
        "",
        "## MoE routing",
        "",
        "| confronto | primo layer mismatch | mismatch rate per layer | margin al mismatch |",
        "|---|---:|---|---:|",
    ])
    for name, key in (("BF16 incremental", "full_bf16_vs_incremental_bf16"), ("FP32 incremental", "full_fp32_vs_incremental_fp32")):
        item = report["router"][key]
        rates = ", ".join(f"L{x['layer']}={x['expert_mismatch_rate']:.6g}" for x in item["layers"])
        lines.append(f"| {name} | {item['first_router_mismatch_layer']} | {rates or 'n/a'} | {item['router_logit_margin_at_mismatch_min']} |")
    lines.extend([
        "",
        "## Precision ablation presente",
        "",
        "- `full_bf16`: forward nativo Gemma con pesi/attivazioni BF16.",
        "- `incremental_bf16`: prefill stabile `[BOS|text|BOA]` separato, poi 4 soft token/frame con cache native.",
        "- `full_fp32` + `incremental_fp32`: percorso nativo e transizione incrementale con il language model interamente FP32 **solo come riferimento diagnostico**.",
        "- Non sono ancora state applicate shadow projection, attention FP32 selettiva o router FP32: saranno valutate dopo il bisect.",
        "",
        "## Timing/memoria del probe strumentato",
        "",
        "| variante | total ms | stable p50 ms | audio p50/p95/p99 ms | peak allocated | peak reserved |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, item in report["modes"].items():
        timing = item["timing"]
        audio = timing.get("incremental_audio_frames") or {}
        lines.append(
            f"| {name} | {timing['total']['p50_ms']} | "
            f"{(timing['stable_prefill'] or {}).get('p50_ms')} | "
            f"{audio.get('p50_ms')}/{audio.get('p95_ms')}/{audio.get('p99_ms')} | "
            f"{item['peak_allocated_bytes']} | {item['peak_reserved_bytes']} |"
        )
    lines.extend([
        "",
        "## Interpretazione",
        "",
        "Il percorso full BF16 sopra è l'oracle nativo e non viene ridefinito in base al risultato incremental; il full FP32 è solo il controllo diagnostico della precisione C. "
        "Questo report localizza il primo scarto numerico e separa la possibile amplificazione MoE dalla causa iniziale; "
        "non autorizza ancora la scelta di una variante di produzione.",
        "",
    ])
    return "\n".join(lines)


def run(*, frames: int, seed: int, output_dir: Path, markdown_path: Path) -> dict[str, Any]:
    if frames <= 0:
        raise ValueError("frames deve essere positivo")
    output_dir.mkdir(parents=True, exist_ok=True)
    thinker, model = _load_gemma_model()
    device = torch.device("cuda")
    try:
        ids, attention, features, feature_mask, audio_soft, audio_start = _fixture(model, frames, seed)
        audio_positions = (ids[0] == int(getattr(model.config, "audio_token_id", 258881))).nonzero(as_tuple=False).flatten().tolist()
        captures = {
            "full_bf16": Capture(set(audio_positions)),
            "incremental_bf16": Capture(set(audio_positions)),
            "full_fp32": Capture(set(audio_positions)),
            "incremental_fp32": Capture(set(audio_positions)),
        }
        results: dict[str, ModeResult] = {}
        results["full_bf16"] = _native_full(
            model,
            mode_name="full_bf16",
            input_ids=ids,
            attention_mask=attention,
            input_features=features,
            input_features_mask=feature_mask,
            capture=captures["full_bf16"],
        )
        results["incremental_bf16"] = _incremental(
            model,
            mode_name="incremental_bf16",
            input_ids=ids,
            attention_mask=attention,
            audio_soft=audio_soft,
            audio_start=audio_start,
            capture=captures["incremental_bf16"],
        )
        model.language_model.float()
        results["full_fp32"] = _native_full(
            model,
            mode_name="full_fp32",
            input_ids=ids,
            attention_mask=attention,
            input_features=features,
            input_features_mask=feature_mask,
            capture=captures["full_fp32"],
            audio_soft=audio_soft,
        )
        results["incremental_fp32"] = _incremental(
            model,
            mode_name="incremental_fp32",
            input_ids=ids,
            attention_mask=attention,
            audio_soft=audio_soft,
            audio_start=audio_start,
            capture=captures["incremental_fp32"],
        )
        events = _build_events(results, audio_start, audio_positions)
        comparison_pairs = (
            "full_bf16_vs_incremental_bf16",
            "full_bf16_vs_incremental_fp32",
            "full_fp32_vs_incremental_fp32",
            "full_bf16_vs_full_fp32",
        )
        summaries = {key: _aggregate(events, key) for key in comparison_pairs}
        first = {key: _first_divergence(summaries[key]) for key in comparison_pairs}
        router = {
            "full_bf16_vs_incremental_bf16": _router_report(events, "full_bf16_vs_incremental_bf16"),
            "full_fp32_vs_incremental_fp32": _router_report(events, "full_fp32_vs_incremental_fp32"),
        }
        report: dict[str, Any] = {
            "schema_version": 1,
            "phase": "P4",
            "tool": "p4_numeric_bisect",
            "status": "diagnostic",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "fixture": {
                "frames": len(audio_positions) // AUDIO_FEATURES_PER_FRAME,
                "audio_tokens": len(audio_positions),
                "seed": seed,
                "audio_start": audio_start,
                "audio_soft_sha256": _tensor_hash(audio_soft),
                "audio_soft_shape": list(audio_soft.shape),
                "input_features_shape": list(features.shape),
                "attention_shape": list(attention.shape),
            },
            "oracle": {
                "implementation": "native Gemma4Model.forward",
                "dtype": "BF16",
                "selected_hidden": "final multimodal LM hidden at every audio token, pre-V9 causal_conv",
                "mask": "native create_masks_for_generate sliding_attention/full_attention",
            },
            "incremental": {
                "prefill": "stable [BOS|text|BOA] once per context",
                "audio_chunk": AUDIO_FEATURES_PER_FRAME,
                "cache": "DynamicCache + native shared_kv_states",
                "fp32_mode": "diagnostic full language_model.float(), not production candidate",
            },
            "thresholds_for_significant_divergence": {
                "mae_gt": SIGNIFICANT_MAE,
                "cosine_lt": SIGNIFICANT_COSINE,
                "expert_mismatch_gt": 0.0,
            },
            "modes": {
                name: {
                    "output_shape": result.output_shape,
                    "timing": result.timing(),
                    "peak_allocated_bytes": result.peak_allocated,
                    "peak_reserved_bytes": result.peak_reserved,
                }
                for name, result in results.items()
            },
            "representation": {
                key: _representation_summary(events, key)
                for key in comparison_pairs
            },
            "first_divergence": first,
            "router": router,
            "component_summary": summaries,
            "events": len(events),
        }
        raw_path = output_dir / "raw_trace.jsonl"
        raw_path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8")
        report["raw_trace"] = str(raw_path)
        report["raw_trace_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        report_path = output_dir / "P4_NUMERIC_BISECT.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        report["report_path"] = str(report_path)
        report["markdown_path"] = str(markdown_path)
        # Rewrite once so the output paths are themselves part of the report.
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        del thinker, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "validation" / "kernel_gates" / "numeric_bisect")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs" / "P4_NUMERIC_BISECT.md")
    args = parser.parse_args()
    report = run(frames=args.frames, seed=args.seed, output_dir=args.output_dir, markdown_path=args.markdown)
    print(json.dumps({
        "first_bf16": report["first_divergence"]["full_bf16_vs_incremental_bf16"],
        "first_fp32": report["first_divergence"]["full_fp32_vs_incremental_fp32"],
        "representation": report["representation"],
        "report_path": report["report_path"],
        "markdown_path": report["markdown_path"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
