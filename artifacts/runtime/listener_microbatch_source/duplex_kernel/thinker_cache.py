"""Exact token-prefix reuse helpers for the compiled Thinker cache."""
# Transformers 5.5.0 `StaticLayer` uses fixed backing storage and advances its
# logical length in-place. Its inherited generic `crop()` slices K/V tensors
# without repairing `cumulative_length`; branch reuse therefore copies the
# committed prefix into fresh layer storage rather than cropping or decoding
# against shared storage.
# Source: https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/cache_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers.cache_utils import StaticLayer


class CachePrefixCopyError(RuntimeError):
    """The supplied cache pair does not satisfy the physical StaticLayer contract."""


@dataclass(frozen=True)
class CachePrefixCopyResult:
    copied_tokens: int
    copied_bytes: int
    layer_count: int


def longest_common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the number of leading token IDs that match exactly."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and int(left[index]) == int(right[index]):
        index += 1
    return index


def _cache_length(cache: object) -> int:
    get_seq_length = getattr(cache, "get_seq_length", None)
    if not callable(get_seq_length):
        raise CachePrefixCopyError("cache non espone get_seq_length()")
    try:
        value = get_seq_length()
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise CachePrefixCopyError("get_seq_length() deve restituire uno scalare")
            return int(value.item())
        return int(value)
    except CachePrefixCopyError:
        raise
    except Exception as exc:
        raise CachePrefixCopyError(f"get_seq_length() non leggibile: {exc}") from exc


def static_cache_sequence_length(cache: object) -> int:
    """Read a cache's committed position count or fail closed."""
    return _cache_length(cache)


def static_cache_storage_bytes(cache: object) -> int:
    """Return allocated K/V backing storage, including unused static capacity."""
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, (list, tuple)) or not layers:
        raise CachePrefixCopyError("cache non espone layer misurabili")
    total = 0
    for index, layer in enumerate(layers):
        if not _is_full_static_layer(layer):
            raise CachePrefixCopyError(f"layer {index} non è una StaticLayer full-attention")
        if not bool(getattr(layer, "is_initialized", False)):
            continue
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise CachePrefixCopyError(f"storage K/V non leggibile a layer {index}")
        total += keys.numel() * keys.element_size()
        total += values.numel() * values.element_size()
    return total


def _is_full_static_layer(layer: object) -> bool:
    return isinstance(layer, StaticLayer) and not bool(getattr(layer, "is_sliding", False))


def copy_static_cache_prefix(
    source_cache: object,
    target_cache: object,
    token_count: int,
) -> CachePrefixCopyResult:
    """Copy only a valid prefix into a fresh, matching physical full-attention cache.

    Transformers 5.5.0's StaticLayer keeps fixed-size backing tensors and updates
    them in place. The inherited generic ``crop()`` only slices the K/V tensors;
    it leaves StaticLayer's cumulative length unchanged. A branch is therefore
    initialized and receives only the prefix slices plus matching per-layer
    cumulative lengths.
    """
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise ValueError("token_count deve essere un intero non negativo")
    if source_cache is target_cache:
        raise CachePrefixCopyError("source e target devono essere cache isolate")
    if token_count == 0:
        return CachePrefixCopyResult(copied_tokens=0, copied_bytes=0, layer_count=0)

    source_layers = getattr(source_cache, "layers", None)
    target_layers = getattr(target_cache, "layers", None)
    if not isinstance(source_layers, (list, tuple)) or not isinstance(target_layers, (list, tuple)):
        raise CachePrefixCopyError("source/target devono esporre le layer della cache")
    if not source_layers or len(source_layers) != len(target_layers):
        raise CachePrefixCopyError("numero di layer source/target non compatibile")

    source_length = _cache_length(source_cache)
    if token_count > source_length:
        raise CachePrefixCopyError(
            f"token_count={token_count} supera source cache length={source_length}"
        )

    pairs: list[tuple[StaticLayer, StaticLayer]] = []
    for index, (source_layer, target_layer) in enumerate(zip(source_layers, target_layers, strict=True)):
        if not _is_full_static_layer(source_layer) or not _is_full_static_layer(target_layer):
            raise CachePrefixCopyError(f"layer {index} non è una StaticLayer full-attention")
        if type(source_layer) is not type(target_layer):
            raise CachePrefixCopyError(f"tipo layer source/target diverso a indice {index}")
        if source_layer.get_seq_length() != source_length:
            raise CachePrefixCopyError(f"layer {index} ha una lunghezza diversa dalla cache source")
        if target_layer.get_seq_length() != 0:
            raise CachePrefixCopyError(f"layer target {index} non è vuoto")
        if not source_layer.is_initialized or source_layer.keys is None or source_layer.values is None:
            raise CachePrefixCopyError(f"layer source {index} non è inizializzato")
        if token_count > int(source_layer.keys.shape[-2]):
            raise CachePrefixCopyError(f"layer source {index} non contiene l'intero prefisso richiesto")
        if int(source_layer.max_cache_len) != int(target_layer.max_cache_len):
            raise CachePrefixCopyError(f"capacità StaticLayer diversa a indice {index}")
        if tuple(source_layer.keys.shape) != tuple(source_layer.values.shape):
            raise CachePrefixCopyError(f"forme K/V source non compatibili a indice {index}")
        if source_layer.keys.dim() != 4:
            raise CachePrefixCopyError(f"forma K/V source non rank-4 a indice {index}")
        pairs.append((source_layer, target_layer))

    copied_bytes = 0
    for index, (source_layer, target_layer) in enumerate(pairs):
        try:
            key_shape = source_layer.keys.shape
            value_shape = source_layer.values.shape
            if not target_layer.is_initialized:
                empty_keys = torch.empty(
                    (key_shape[0], key_shape[1], 0, key_shape[3]),
                    device=source_layer.keys.device,
                    dtype=source_layer.keys.dtype,
                )
                empty_values = torch.empty(
                    (value_shape[0], value_shape[1], 0, value_shape[3]),
                    device=source_layer.values.device,
                    dtype=source_layer.values.dtype,
                )
                target_layer.lazy_initialization(empty_keys, empty_values)
            if (
                tuple(target_layer.keys.shape) != tuple(source_layer.keys.shape)
                or tuple(target_layer.values.shape) != tuple(source_layer.values.shape)
                or target_layer.keys.device != source_layer.keys.device
                or target_layer.values.device != source_layer.values.device
                or target_layer.keys.dtype != source_layer.keys.dtype
                or target_layer.values.dtype != source_layer.values.dtype
            ):
                raise CachePrefixCopyError(f"storage StaticLayer incompatibile a indice {index}")
            if token_count:
                target_layer.keys[:, :, :token_count, :].copy_(
                    source_layer.keys[:, :, :token_count, :]
                )
                target_layer.values[:, :, :token_count, :].copy_(
                    source_layer.values[:, :, :token_count, :]
                )
            target_layer.cumulative_length.copy_(
                torch.full_like(target_layer.cumulative_length, token_count)
            )
            copied_bytes += (
                source_layer.keys[:, :, :token_count, :].numel() * source_layer.keys.element_size()
                + source_layer.values[:, :, :token_count, :].numel()
                * source_layer.values.element_size()
            )
        except CachePrefixCopyError:
            raise
        except Exception as exc:
            raise CachePrefixCopyError(f"copia StaticLayer fallita a indice {index}: {exc}") from exc

    if _cache_length(target_cache) != token_count:
        raise CachePrefixCopyError("lunghezza target non coincide col prefisso copiato")
    return CachePrefixCopyResult(
        copied_tokens=token_count,
        copied_bytes=copied_bytes,
        layer_count=len(pairs),
    )


__all__ = [
    "CachePrefixCopyError",
    "CachePrefixCopyResult",
    "copy_static_cache_prefix",
    "longest_common_prefix_length",
    "static_cache_sequence_length",
    "static_cache_storage_bytes",
]
