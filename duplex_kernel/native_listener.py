# -*- coding: utf-8 -*-
"""duplex_kernel/native_listener.py — Python adapter for native Gemma 4 Listener LM.

Loads libgemma4_listener.so via ctypes, binds PyTorch parameter data_ptr()
with zero weight duplication, validates cryptographic topology hashes,
and exposes the single-call gemma4_listener_step_async interface.

Conforms strictly to P4-NATIVE Implementation Plan §6, §7, §22.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import math
import os
import re
import threading
from pathlib import Path
from typing import Any

import torch

from duplex_kernel.config import NATIVE_LISTENER_CUDA_PROFILE_ENABLED
from duplex_kernel.gemma_representation import (
    AUDIO_FEATURES_PER_FRAME,
    AUDIO_START_AFTER_TEXT,
    CANONICAL_LM_PREFILL_TARGET_LEN,
)

logger = logging.getLogger("duplex_kernel.native_listener")

_ATTESTED_CHECKPOINTS_LOCK = threading.Lock()
_VERIFIED_CHECKPOINT_SIGNATURES: set[tuple[Any, ...]] = set()

GEMMA4_NUM_LAYERS = 35
GEMMA4_HIDDEN_SIZE = 1536
GEMMA4_PLE_DIM = 256
NATIVE_REBASE_MAX_AUDIO_FRAMES = 64
NATIVE_LM_CUSTOM_MAX_SEQ_LEN = 4096
NATIVE_LM_EXACT_MAX_SEQ_LEN = 16384

# Attention backend IDs are part of the P4 runtime evidence contract.  ID 1
# remains the literal repeat_kv oracle; ID 2 is the opt-in direct-GQA candidate.
GEMMA4_ATTN_CUSTOM = 0
GEMMA4_ATTN_ATEN_REPEAT_EXACT = 1
GEMMA4_ATTN_ATEN_GQA_EXACT = 2
GEMMA4_ATTN_NATIVE_SPLITK_GQA = 3
GEMMA4_ATTN_ATEN_EXACT = GEMMA4_ATTN_ATEN_REPEAT_EXACT  # legacy source name
GEMMA4_RMS_CUSTOM_FAST = 0
GEMMA4_RMS_ATEN_EXACT = 1
GEMMA4_RMS_NATIVE_EXACT = 2

GEMMA4_FULL_ATTENTION_LAYERS = (4, 9, 14, 19, 24, 29, 34)
GEMMA4_SPLITK_STATE_SAFE_HYBRID_LAYERS = (19, 24, 29, 34)
GEMMA4_ORDERED_LAYER_MASK = GEMMA4_SPLITK_STATE_SAFE_HYBRID_LAYERS

# Public backend-3 implementation metadata. Backend IDs stay stable; the
# implementation string distinguishes the new four-stage ordered core from
# historical split-K reports.
GEMMA4_ORDERED_ATTENTION_IMPLEMENTATION = "CUTLASS_ORDERED_TILES_V1"
GEMMA4_ORDERED_KEY_TILE_SIZE = 128
GEMMA4_ORDERED_ACTIVATION_MIN_KEYS_EXCLUSIVE = 2048
GEMMA4_ORDERED_REDUCTION_ORDER = "ASCENDING_KEY_TILE"
GEMMA4_ORDERED_STAGE_SEQUENCE = ("A_QK", "B_SOFTMAX", "C_PV", "D_OUTPUT_FOLD")
GEMMA4_ORDERED_Q_PACKING = "BITWISE_Q_MAJOR_TO_HEAD_MAJOR"


def parse_splitk_layer_mask(value: str | None) -> tuple[int, ...]:
    """Parse the explicit full-attention Split-K layer policy.

    The default for backend 3 is STATE_SAFE_HYBRID: only shared-KV consumer
    layers may use approximate attention, so no persistent source KV is
    written from an approximate result. Sliding layers are never eligible.
    """
    normalized = "" if value is None else str(value).strip().upper()
    if not normalized or normalized == "STATE_SAFE_HYBRID":
        return GEMMA4_SPLITK_STATE_SAFE_HYBRID_LAYERS
    if normalized == "ALL_FULL":
        return GEMMA4_FULL_ATTENTION_LAYERS
    if normalized == "NONE":
        return ()
    try:
        layers = tuple(sorted(int(token.strip()) for token in normalized.split(",") if token.strip()))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mask Split-K non valida: {value!r}") from exc
    if not layers or len(layers) != len(set(layers)):
        raise ValueError(f"mask Split-K non valida: {value!r}")
    if any(layer not in GEMMA4_FULL_ATTENTION_LAYERS for layer in layers):
        raise ValueError(
            f"mask Split-K contiene layer non full-attention: {value!r}; "
            f"ammessi={GEMMA4_FULL_ATTENTION_LAYERS}"
        )
    return layers


def splitk_layer_mask_from_bits(mask: int) -> tuple[int, ...]:
    """Decode the native uint64 layer mask for runtime evidence."""
    mask = int(mask)
    if mask < 0:
        raise ValueError(f"mask Split-K negativa: {mask}")
    return tuple(layer for layer in GEMMA4_FULL_ATTENTION_LAYERS if mask & (1 << layer))


def parse_attention_backend(value: str) -> int:
    """Parse only explicitly supported attention backend names/IDs."""
    normalized = str(value).strip().upper()
    if normalized in {"0", "CUSTOM"}:
        return GEMMA4_ATTN_CUSTOM
    if normalized in {"1", "ATEN_EXACT", "ATEN_REPEAT_EXACT"}:
        return GEMMA4_ATTN_ATEN_REPEAT_EXACT
    if normalized in {"2", "ATEN_GQA_EXACT"}:
        return GEMMA4_ATTN_ATEN_GQA_EXACT
    if normalized in {"3", "NATIVE_SPLITK_GQA", "ATEN_SPLITK_GQA"}:
        return GEMMA4_ATTN_NATIVE_SPLITK_GQA
    raise ValueError(f"backend attenzione non supportato: {value!r}")


def validate_cuda_graph_attention_backend(enable_cuda_graph: bool, attention_backend: int) -> None:
    """Graph replay currently preserves only the native CUSTOM Q4 route."""
    if enable_cuda_graph and int(attention_backend) != GEMMA4_ATTN_CUSTOM:
        raise RuntimeError(
            "CUDA Graph Q4 richiede GEMMA4_ATTENTION_BACKEND=CUSTOM; "
            "backend ATEN/NATIVE_SPLITK resta sul percorso legacy"
        )


def attention_backend_from_env() -> int:
    """Resolve the legacy low-level environment contract.

    This deliberately retains CUSTOM when no selector is present. Production
    startup uses :func:`resolve_native_attention_backend` and passes the
    resolved policy through the native creation API instead of mutating env.
    """
    configured = os.environ.get("GEMMA4_ATTENTION_BACKEND")
    if configured is not None and configured.strip():
        return parse_attention_backend(configured)
    bridge = os.environ.get("GEMMA4_ATEN_SDPA_BRIDGE")
    return (
        GEMMA4_ATTN_ATEN_REPEAT_EXACT
        if bridge is not None and bridge != "0"
        else GEMMA4_ATTN_CUSTOM
    )


def resolve_native_attention_backend(explicit: str | int | None = None) -> int:
    """Resolve the canonical production attention policy, fail-closed.

    An explicit selector always wins. With no selector, production defaults to
    backend 3 (CUTLASS_ORDERED_TILES_V1). Invalid environment values are not
    silently converted to the compatibility backend.
    """
    if explicit is not None:
        try:
            return parse_attention_backend(explicit)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"backend attenzione esplicito non valido: {explicit!r}") from exc
    configured = os.environ.get("GEMMA4_ATTENTION_BACKEND")
    if configured is None or not configured.strip():
        return GEMMA4_ATTN_NATIVE_SPLITK_GQA
    try:
        return parse_attention_backend(configured)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "GEMMA4_ATTENTION_BACKEND non valido; startup production rifiutato: "
            f"{configured!r}"
        ) from exc


def assert_production_native_runtime(info: dict[str, Any]) -> None:
    """Assert the live production native policy, including the opt-in graph route."""
    graph_enabled = info.get("cuda_graph_enabled") is True
    if graph_enabled:
        required = {
            "enabled": True,
            "cuda_graph_enabled": True,
            "cuda_graph_stream_non_default": True,
            "checkpoint_path": "nonempty",
            "attention_backend_id": GEMMA4_ATTN_CUSTOM,
            "attention_backend": "CUSTOM",
            "rmsnorm_backend": "NATIVE_EXACT",
        }
    else:
        required = {
            "enabled": True,
            "cuda_graph_enabled": False,
            "checkpoint_path": "nonempty",
            "attention_backend_id": GEMMA4_ATTN_NATIVE_SPLITK_GQA,
            "attention_backend": "NATIVE_SPLITK_GQA",
            "attention_implementation": GEMMA4_ORDERED_ATTENTION_IMPLEMENTATION,
            "splitk_layer_mask_layers": list(GEMMA4_ORDERED_LAYER_MASK),
            "rmsnorm_backend": "NATIVE_EXACT",
        }
    mismatches = {}
    for key, expected in required.items():
        actual = info.get(key)
        if expected == "nonempty":
            matches = isinstance(actual, str) and bool(actual)
        elif key == "splitk_layer_mask_layers":
            matches = actual is not None and list(actual) == expected
        else:
            matches = actual == expected
        if not matches:
            mismatches[key] = (actual, expected)
    if mismatches:
        raise RuntimeError(f"runtime native production non conforme: {mismatches}")


def parse_rmsnorm_backend(value: str | int) -> int:
    """Parse the explicit RMSNorm policy used by native creation."""
    normalized = str(value).strip().upper()
    if normalized in {"0", "CUSTOM_FAST"}:
        return GEMMA4_RMS_CUSTOM_FAST
    if normalized in {"1", "ATEN_EXACT"}:
        return GEMMA4_RMS_ATEN_EXACT
    if normalized in {"2", "NATIVE_EXACT"}:
        return GEMMA4_RMS_NATIVE_EXACT
    raise ValueError(f"backend RMSNorm non supportato: {value!r}")


def attention_backend_name(backend: int) -> str:
    names = {
        GEMMA4_ATTN_CUSTOM: "CUSTOM",
        GEMMA4_ATTN_ATEN_REPEAT_EXACT: "ATEN_REPEAT_EXACT",
        GEMMA4_ATTN_ATEN_GQA_EXACT: "ATEN_GQA_EXACT",
        GEMMA4_ATTN_NATIVE_SPLITK_GQA: "NATIVE_SPLITK_GQA",
    }
    try:
        return names[int(backend)]
    except KeyError as exc:
        raise ValueError(f"backend attenzione sconosciuto: {backend}") from exc


def attention_implementation_metadata(backend: int) -> dict[str, int | str | list[str]] | None:
    """Return verifiable implementation metadata for the selected backend."""
    if int(backend) != GEMMA4_ATTN_NATIVE_SPLITK_GQA:
        return None
    return {
        "attention_backend": "NATIVE_SPLITK_GQA",
        "attention_implementation": GEMMA4_ORDERED_ATTENTION_IMPLEMENTATION,
        "key_tile_size": GEMMA4_ORDERED_KEY_TILE_SIZE,
        "activation_min_keys_exclusive": GEMMA4_ORDERED_ACTIVATION_MIN_KEYS_EXCLUSIVE,
        "reduction_order": GEMMA4_ORDERED_REDUCTION_ORDER,
        "stage_sequence": list(GEMMA4_ORDERED_STAGE_SEQUENCE),
        "q_packing": GEMMA4_ORDERED_Q_PACKING,
    }


_MANIFEST_RELATIVE_PATH = Path("validation/kernel_gates/P4_NATIVE/GEMMA4_NATIVE_TOPOLOGY.json")


def _manifest_path() -> Path:
    candidates = (
        Path("/workspace") / _MANIFEST_RELATIVE_PATH,
        Path(__file__).resolve().parents[1] / _MANIFEST_RELATIVE_PATH,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"manifest topology non trovato: {_MANIFEST_RELATIVE_PATH}")


def _load_attested_manifest() -> dict[str, Any]:
    path = _manifest_path()
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    claimed_sha = manifest.get("manifest_sha256")
    if not isinstance(claimed_sha, str):
        raise RuntimeError("manifest topology privo di manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    computed_sha = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if computed_sha != claimed_sha:
        raise RuntimeError("manifest topology con hash interno non valido")
    return manifest


def _verify_attested_checkpoint(manifest: dict[str, Any]) -> Path:
    checkpoint_path = manifest.get("checkpoint_path")
    checkpoint_files = manifest.get("checkpoint_files")
    if not isinstance(checkpoint_path, str) or not isinstance(checkpoint_files, dict):
        raise RuntimeError("manifest privo di attestazione checkpoint completa")
    root = Path(checkpoint_path)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint attestato non trovato: {root}")
    resolved_root = root.resolve()
    files: list[tuple[str, str, Path, int, int]] = []
    for relative_name, expected_sha in sorted(checkpoint_files.items()):
        if not isinstance(relative_name, str) or not isinstance(expected_sha, str):
            raise RuntimeError("entry checkpoint_files non valida")
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError("percorso checkpoint non relativo al root attestato")
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"file checkpoint attestato non trovato: {path}")
        resolved_path = path.resolve()
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise RuntimeError("percorso checkpoint risolto fuori dal root attestato") from exc
        stat = resolved_path.stat()
        files.append((relative_name, expected_sha, resolved_path, int(stat.st_size), int(stat.st_mtime_ns)))
    signature: tuple[Any, ...] = (
        str(resolved_root),
        tuple((name, expected_sha, size, mtime_ns) for name, expected_sha, _, size, mtime_ns in files),
    )
    with _ATTESTED_CHECKPOINTS_LOCK:
        if signature in _VERIFIED_CHECKPOINT_SIGNATURES:
            return root
        for relative_name, expected_sha, path, _, _ in files:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha = digest.hexdigest()
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"hash checkpoint non valido per {relative_name}: {actual_sha} != {expected_sha}"
                )
        for _, _, path, size, mtime_ns in files:
            stat = path.stat()
            if int(stat.st_size) != size or int(stat.st_mtime_ns) != mtime_ns:
                raise RuntimeError(f"file checkpoint modificato durante attestazione: {path}")
        _VERIFIED_CHECKPOINT_SIGNATURES.add(signature)
    return root


def _load_compiled_listener_topology() -> dict[str, Any]:
    """Read the topology table compiled into this repository's native source.

    Runtime loading no longer depends on a generated validation manifest. The
    checked-in generated header is the source used by the C++ build, and the
    DSO's topology id is compared with its value before any session is created.
    """
    repo_root = Path(__file__).resolve().parents[1]
    candidates = (
        Path("/workspace/native/gemma4_listener/generated_topology.h"),
        repo_root / "native" / "gemma4_listener" / "generated_topology.h",
    )
    header_path = next((path for path in candidates if path.is_file()), None)
    if header_path is None:
        raise FileNotFoundError("native listener generated_topology.h is required at runtime")
    source = header_path.read_text(encoding="utf-8")

    def macro(name: str) -> str:
        match = re.search(rf"^#define\s+{re.escape(name)}\s+([^\s]+)", source, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"native topology source lacks {name}")
        return match.group(1)

    def number(name: str) -> int:
        return int(macro(name), 0)

    def scalar(value: str) -> float:
        return float(value.strip().rstrip("fF"))

    array = re.search(
        r"static\s+const\s+GemmaLayerSpec\s+gemma4_layers\s*\[.*?\]\s*=\s*\{(.*?)\};",
        source,
        re.DOTALL,
    )
    if array is None:
        raise RuntimeError("native topology source lacks the gemma4_layers table")
    raw_rows = re.findall(r"\{\s*([^{}]+?)\s*\}", array.group(1))
    layer_count = number("GEMMA4_NUM_LAYERS")
    if len(raw_rows) != layer_count:
        raise RuntimeError(f"native topology has {len(raw_rows)} rows, expected {layer_count}")

    hidden = number("GEMMA4_HIDDEN_SIZE")
    ple_dim = number("GEMMA4_PLE_DIM")
    num_heads = number("GEMMA4_NUM_ATTN_HEADS")
    num_kv_heads = number("GEMMA4_NUM_KV_HEADS")
    layers: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(raw_rows):
        fields = [field.strip() for field in raw_row.split(",")]
        if len(fields) != 19:
            raise RuntimeError(f"native topology layer {row_index} has {len(fields)} fields")
        index = int(fields[0])
        if index != row_index:
            raise RuntimeError(f"native topology row order mismatch at layer {row_index}")
        if fields[1] not in {"GEMMA_ATTN_SLIDING", "GEMMA_ATTN_FULL"}:
            raise RuntimeError(f"native topology has unknown attention type at layer {index}")
        layer_type = "sliding_attention" if fields[1] == "GEMMA_ATTN_SLIDING" else "full_attention"
        head_dim, intermediate, window = map(int, fields[2:5])
        rope_base = scalar(fields[6])
        shared = fields[7] == "true"
        shared_source = int(fields[8])
        store_full = fields[9] == "true"
        rope_type = "proportional" if layer_type == "full_attention" else "default"
        rope_spec: dict[str, Any] = {"rope_type": rope_type, "rope_theta": rope_base}
        if layer_type == "full_attention":
            # The native full-attention RoPE path rotates one quarter of the
            # 512-wide head; this is explicit in gemma4_listener.cu.
            rope_spec["partial_rotary_factor"] = 0.25
        layer = {
            "index": index,
            "layer_type": layer_type,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "rope_scaling": {layer_type: rope_spec},
            "intermediate_size": intermediate,
            "sliding_window": window,
            "attention_scaling": scalar(fields[5]),
            "layer_scalar": scalar(fields[10]),
            "is_kv_shared_layer": shared,
            "kv_shared_layer_index": shared_source,
            "store_full_length_kv": store_full,
            "input_layernorm_eps": scalar(fields[11]),
            "post_attention_layernorm_eps": scalar(fields[12]),
            "pre_feedforward_layernorm_eps": scalar(fields[13]),
            "post_feedforward_layernorm_eps": scalar(fields[14]),
            "q_norm_eps": scalar(fields[15]),
            "k_norm_eps": scalar(fields[16]),
            "v_norm_eps": scalar(fields[17]),
            "post_per_layer_input_norm_eps": scalar(fields[18]),
            "q_proj_shape": (num_heads * head_dim, hidden),
            "k_proj_shape": (num_kv_heads * head_dim, hidden),
            "v_proj_shape": (num_kv_heads * head_dim, hidden),
            "o_proj_shape": (hidden, num_heads * head_dim),
            "gate_proj_shape": (intermediate, hidden),
            "up_proj_shape": (intermediate, hidden),
            "down_proj_shape": (hidden, intermediate),
            "q_norm_shape": (head_dim,),
            "k_norm_shape": (head_dim,),
            "per_layer_input_gate_shape": (ple_dim, hidden),
            "per_layer_projection_shape": (hidden, ple_dim),
        }
        layers.append(layer)

    return {
        "manifest_sha256": macro("GEMMA4_TOPOLOGY_MANIFEST_SHA256").strip('"'),
        "canonical_prefix_len": number("GEMMA4_CANONICAL_PREFIX_LEN"),
        "canonical_lm_prefill_target_len": number("GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN"),
        "final_norm_eps": scalar(macro("GEMMA4_FINAL_NORM_EPS")),
        "hidden_size": hidden,
        "num_layers": layer_count,
        "ple_dim": ple_dim,
        "per_layer_model_projection_shape": (layer_count * ple_dim, hidden),
        "layers": tuple(layers),
    }


class Gemma4TensorMeta(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("dtype", ctypes.c_uint32),
        ("ndim", ctypes.c_uint32),
        ("contiguous", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("dims", ctypes.c_uint64 * 4),
        ("strides", ctypes.c_uint64 * 4),
    ]


class Gemma4ListenerConfig(ctypes.Structure):
    _fields_ = [
        ("max_seq_len", ctypes.c_uint32),
        ("stream_q_len", ctypes.c_uint32),
        ("enable_cuda_graph", ctypes.c_bool),
        ("output_tensor_ptr", ctypes.c_void_p),
        ("frame_pad_ple_ptr", ctypes.c_void_p),
        ("weight_meta", ctypes.POINTER(Gemma4TensorMeta)),
        ("weight_meta_count", ctypes.c_uint32),
        ("output_meta", ctypes.POINTER(Gemma4TensorMeta)),
        ("frame_pad_ple_meta", ctypes.POINTER(Gemma4TensorMeta)),
    ]


class Gemma4LayerWeights(ctypes.Structure):
    _fields_ = [
        ("input_norm_w", ctypes.c_void_p),
        ("post_attn_norm_w", ctypes.c_void_p),
        ("pre_ff_norm_w", ctypes.c_void_p),
        ("post_ff_norm_w", ctypes.c_void_p),
        ("q_proj_w", ctypes.c_void_p),
        ("k_proj_w", ctypes.c_void_p),
        ("v_proj_w", ctypes.c_void_p),
        ("o_proj_w", ctypes.c_void_p),
        ("q_norm_w", ctypes.c_void_p),
        ("k_norm_w", ctypes.c_void_p),
        ("v_norm_w", ctypes.c_void_p),
        ("gate_proj_w", ctypes.c_void_p),
        ("up_proj_w", ctypes.c_void_p),
        ("down_proj_w", ctypes.c_void_p),
        ("ple_gate_w", ctypes.c_void_p),
        ("ple_proj_w", ctypes.c_void_p),
        ("ple_norm_w", ctypes.c_void_p),
    ]


class Gemma4ListenerWeights(ctypes.Structure):
    _fields_ = [
        ("final_norm_w", ctypes.c_void_p),
        ("per_layer_model_proj_w", ctypes.c_void_p),
        ("per_layer_proj_norm_w", ctypes.c_void_p),
        ("per_layer_model_proj_scale", ctypes.c_float),
        ("per_layer_input_scale", ctypes.c_float),
        ("layers", Gemma4LayerWeights * GEMMA4_NUM_LAYERS),
    ]


class Gemma4ImportedKV(ctypes.Structure):
    _fields_ = [
        ("k_ptr", ctypes.c_void_p),
        ("k_meta", ctypes.POINTER(Gemma4TensorMeta)),
        ("v_ptr", ctypes.c_void_p),
        ("v_meta", ctypes.POINTER(Gemma4TensorMeta)),
    ]


class Gemma4ImportedCache(ctypes.Structure):
    _fields_ = [("layers", Gemma4ImportedKV * GEMMA4_NUM_LAYERS)]


_NATIVE_LISTENER_LIBRARY_PATH_ENV = "GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"


def _find_library() -> str:
    configured = os.environ.get(_NATIVE_LISTENER_LIBRARY_PATH_ENV)
    if configured is not None:
        if not configured.strip():
            raise ValueError(f"{_NATIVE_LISTENER_LIBRARY_PATH_ENV} must be a non-empty path")
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"{_NATIVE_LISTENER_LIBRARY_PATH_ENV} does not exist: {path}"
            )
        return str(path.resolve())
    candidates = [
        Path("/workspace/native/gemma4_listener/libgemma4_listener.so"),
        Path(__file__).resolve().parents[1] / "native" / "gemma4_listener" / "libgemma4_listener.so",
        Path("native/gemma4_listener/libgemma4_listener.so").resolve(),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    raise FileNotFoundError("libgemma4_listener.so non trovata nei percorsi candidati")


class _NativeQ4CudaProfiler:
    """Bounded, opt-in CUDA-event samples around the native Q4 enqueue call."""

    sampling_frequency_frames = 8
    sample_cap = 64

    def __init__(self, enabled: bool, *, event_factory: Any | None = None) -> None:
        self.enabled = bool(enabled)
        self._event_factory = event_factory or (
            lambda **kwargs: torch.cuda.Event(**kwargs)
        )
        self._native_step_calls = 0
        self._sample_attempts = 0
        self._pending: list[dict[str, Any]] = []
        self._completed: list[dict[str, Any]] = []
        self._failure: dict[str, Any] | None = None

    def _fail(self, stage: str, frame_id: int | None, exc: BaseException) -> RuntimeError:
        self._failure = {
            "stage": stage,
            "frame_id": frame_id,
            "error": f"{type(exc).__name__}: {exc}",
        }
        return RuntimeError(f"native listener CUDA profile failed during {stage}")

    def _poll_completed(self, *, raise_on_error: bool) -> None:
        if not self.enabled or not self._pending:
            return
        remaining: list[dict[str, Any]] = []
        pending = list(self._pending)
        for index, sample in enumerate(pending):
            try:
                if sample["end_event"].query():
                    # CUDA Event.elapsed_time is read only after a nonblocking
                    # query reports completion; this path never synchronizes.
                    elapsed_ms = float(sample["start_event"].elapsed_time(sample["end_event"]))
                    self._completed.append({
                        "sample_index": sample["sample_index"],
                        "native_call_index": sample["native_call_index"],
                        "frame_id": sample["frame_id"],
                        "elapsed_ms": elapsed_ms,
                        # The listener callback fills its end stamps in a
                        # finally block after this nonblocking event harvest.
                        # Keep the owned context live until summary/sidecar
                        # serialization so an early poll cannot freeze nulls.
                        "host_context": sample["host_context"],
                    })
                else:
                    remaining.append(sample)
            except BaseException as exc:
                self._failure = {
                    "stage": "event_query_or_elapsed_time",
                    "frame_id": sample.get("frame_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                remaining.append(sample)
                if raise_on_error:
                    self._pending = remaining + pending[index + 1:]
                    raise RuntimeError("native listener CUDA profile event query failed") from exc
        self._pending = remaining

    def begin_step(
        self,
        stream: Any,
        *,
        frame_id: int | None,
        host_context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        native_call_index = self._native_step_calls
        self._native_step_calls += 1
        if native_call_index % self.sampling_frequency_frames != 0:
            return None
        self._poll_completed(raise_on_error=True)
        if self._sample_attempts >= self.sample_cap:
            return None
        if frame_id is None or isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            error = ValueError("sampled native Q4 call requires its host frame_id")
            raise self._fail("frame_id_validation", frame_id, error)
        if (
            not isinstance(host_context, dict)
            or not isinstance(host_context.get("session_id"), str)
            or not host_context.get("session_id")
            or host_context.get("frame_id") != frame_id
        ):
            error = ValueError("sampled native Q4 call requires matching host session/frame timestamps")
            raise self._fail("host_context_validation", frame_id, error)
        self._sample_attempts += 1
        sample = {
            "sample_index": self._sample_attempts - 1,
            "native_call_index": native_call_index,
            "frame_id": frame_id,
            "host_context": host_context,
        }
        try:
            start_event = self._event_factory(enable_timing=True)
            end_event = self._event_factory(enable_timing=True)
            start_event.record(stream)
        except BaseException as exc:
            raise self._fail("event_create_or_start_record", frame_id, exc) from exc
        sample["start_event"] = start_event
        sample["end_event"] = end_event
        return sample

    def finish_step(self, sample: dict[str, Any] | None, stream: Any) -> None:
        if sample is None:
            return
        try:
            sample["end_event"].record(stream)
        except BaseException as exc:
            raise self._fail("end_record", sample.get("frame_id"), exc) from exc
        if len(self._pending) >= self.sample_cap:
            error = RuntimeError("native listener CUDA profile pending ring exceeded its bound")
            raise self._fail("pending_ring_bound", sample.get("frame_id"), error)
        self._pending.append(sample)

    def abort_step(self, sample: dict[str, Any] | None, exc: BaseException) -> None:
        if sample is None:
            return
        self._failure = {
            "stage": "native_q4_enqueue",
            "frame_id": sample.get("frame_id"),
            "error": f"{type(exc).__name__}: {exc}",
        }

    def summary(
        self,
        *,
        session_id: str,
        runtime_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.enabled:
            self._poll_completed(raise_on_error=False)
        info = runtime_info if isinstance(runtime_info, dict) else {}
        if not self.enabled:
            status = "not_requested"
        elif self._failure is not None:
            status = "failed"
        elif self._sample_attempts == 0 or self._pending:
            status = "incomplete"
        else:
            status = "complete"
        return {
            "schema_version": 1,
            "enabled": self.enabled,
            "status": status,
            "session_id": str(session_id),
            "sampling_frequency_frames": self.sampling_frequency_frames,
            "sample_cap": self.sample_cap,
            "native_step_call_count": self._native_step_calls,
            "sampled_event_pair_count": len(self._completed) + len(self._pending),
            "completed_sample_count": len(self._completed),
            "pending_sample_count": len(self._pending),
            "pending_frame_ids": [sample.get("frame_id") for sample in self._pending],
            "pending_samples": [
                {
                    "sample_index": sample.get("sample_index"),
                    "native_call_index": sample.get("native_call_index"),
                    "frame_id": sample.get("frame_id"),
                    "host_context": dict(sample.get("host_context", {})),
                }
                for sample in self._pending
            ],
            "failure": self._failure,
            "runtime": {
                key: info.get(key)
                for key in (
                    "cuda_graph_enabled",
                    "cuda_graph_capture_launches",
                    "cuda_graph_replay_launches",
                    "cuda_graph_step_launches",
                    "attention_backend_id",
                    "attention_backend",
                    "library_sha256",
                )
            },
            "samples": [dict(sample) for sample in self._completed],
            "host_join_contract": (
                "sample.host_context contains the captured host stamps for its session_id/frame_id; "
                "the sidecar sampled_host_tick_timings table preserves those rows even when the "
                "bounded listener.tick_timings history has rolled over"
            ),
        }


class NativeGemmaListenerLM:
    """DS4-style native model-specific executor for Gemma 4 Listener LM.

    Single C ABI boundary per frame, device-resident KV and activation arena,
    zero weight duplication (borrowed data_ptr()); numerical parity is guarded
    by external gates before production use.
    """

    def __init__(
        self,
        language_model: Any,
        *,
        max_seq_len: int = 4096,
        stream_q_len: int = 4,
        enable_cuda_graph: bool = False,
        attention_backend: str | int | None = None,
        rmsnorm_backend: str | int | None = None,
        attention_backend_source: str | None = None,
        frame_pad_ple_tensor: torch.Tensor | None = None,
    ) -> None:
        self.language_model = language_model
        self._q4_cuda_profiler = (
            _NativeQ4CudaProfiler(NATIVE_LISTENER_CUDA_PROFILE_ENABLED)
            if NATIVE_LISTENER_CUDA_PROFILE_ENABLED else None
        )
        self.enable_cuda_graph = bool(enable_cuda_graph)
        self.max_seq_len = int(max_seq_len)
        self.stream_q_len = int(stream_q_len)
        if self.stream_q_len != 4:
            raise ValueError("il backend Gemma nativo richiede stream_q_len=4")
        if self.max_seq_len < AUDIO_START_AFTER_TEXT + self.stream_q_len:
            raise ValueError("max_seq_len troppo piccolo per prefix canonico + un frame")
        self._attention_backend_env_present = bool(
            os.environ.get("GEMMA4_ATTENTION_BACKEND", "").strip()
        )
        self._attention_backend_source = attention_backend_source or (
            "environment" if self._attention_backend_env_present else "legacy_default"
        )
        self._explicit_attention_backend = (
            None if attention_backend is None else resolve_native_attention_backend(attention_backend)
        )
        self.attention_backend = (
            self._explicit_attention_backend
            if self._explicit_attention_backend is not None
            else attention_backend_from_env()
        )
        validate_cuda_graph_attention_backend(self.enable_cuda_graph, self.attention_backend)
        self._explicit_rmsnorm_backend = (
            None if rmsnorm_backend is None else parse_rmsnorm_backend(rmsnorm_backend)
        )
        self.splitk_layer_mask_layers = (
            parse_splitk_layer_mask(os.environ.get("GEMMA4_SPLITK_LAYER_MASK"))
            if self.attention_backend == GEMMA4_ATTN_NATIVE_SPLITK_GQA
            else ()
        )
        if (
            self.attention_backend == GEMMA4_ATTN_NATIVE_SPLITK_GQA
            and self.splitk_layer_mask_layers != GEMMA4_ORDERED_LAYER_MASK
        ):
            raise ValueError(
                "CUTLASS_ORDERED_TILES_V1 richiede esattamente layer mask "
                f"{GEMMA4_ORDERED_LAYER_MASK}"
            )
        exact_sdpa = self.attention_backend in {
            GEMMA4_ATTN_ATEN_REPEAT_EXACT,
            GEMMA4_ATTN_ATEN_GQA_EXACT,
            GEMMA4_ATTN_NATIVE_SPLITK_GQA,
        }
        max_allowed_seq_len = (
            NATIVE_LM_EXACT_MAX_SEQ_LEN if exact_sdpa else NATIVE_LM_CUSTOM_MAX_SEQ_LEN
        )
        if self.max_seq_len > max_allowed_seq_len:
            raise ValueError(
                f"max_seq_len={self.max_seq_len} oltre la capacità del backend "
                f"({max_allowed_seq_len})"
            )
        self._pinned_tensors: list[torch.Tensor] = []
        self._cache_import_tensors: list[torch.Tensor] = []
        self._last_execution_stream: torch.cuda.Stream | None = None

        lib_path = _find_library()
        self._lib_path = lib_path
        logger.info("[native_listener] Caricamento %s...", lib_path)
        self._lib = ctypes.CDLL(lib_path)
        self._setup_c_signatures()

        # 1. Attest ABI and Topology Hash
        abi_ver = self._lib.gemma4_listener_abi_version()
        if abi_ver != 3:
            raise RuntimeError(f"Mismatch ABI version nativa: atteso 3, ottenuto {abi_ver}")
        abi_sizes = {
            "config": int(self._lib.gemma4_listener_abi_config_size()),
            "tensor_meta": int(self._lib.gemma4_listener_abi_tensor_meta_size()),
            "layer_weights": int(self._lib.gemma4_listener_abi_layer_weights_size()),
            "weights": int(self._lib.gemma4_listener_abi_weights_size()),
            "imported_kv": int(self._lib.gemma4_listener_abi_imported_kv_size()),
            "imported_cache": int(self._lib.gemma4_listener_abi_imported_cache_size()),
        }
        expected_abi_sizes = {
            "config": ctypes.sizeof(Gemma4ListenerConfig),
            "tensor_meta": ctypes.sizeof(Gemma4TensorMeta),
            "layer_weights": ctypes.sizeof(Gemma4LayerWeights),
            "weights": ctypes.sizeof(Gemma4ListenerWeights),
            "imported_kv": ctypes.sizeof(Gemma4ImportedKV),
            "imported_cache": ctypes.sizeof(Gemma4ImportedCache),
        }
        if abi_sizes != expected_abi_sizes:
            raise RuntimeError(f"layout ABI nativa incompatibile: {abi_sizes} != {expected_abi_sizes}")
        abi_fields = {
            "tensor_meta": (Gemma4TensorMeta, (
                "ptr", "dtype", "ndim", "contiguous", "reserved", "dims", "strides",
            )),
            "config": (Gemma4ListenerConfig, (
                "max_seq_len", "stream_q_len", "enable_cuda_graph", "output_tensor_ptr",
                "frame_pad_ple_ptr", "weight_meta", "weight_meta_count", "output_meta",
                "frame_pad_ple_meta",
            )),
            "layer_weights": (Gemma4LayerWeights, (
                "input_norm_w", "post_attn_norm_w", "pre_ff_norm_w", "post_ff_norm_w",
                "q_proj_w", "k_proj_w", "v_proj_w", "o_proj_w", "q_norm_w", "k_norm_w",
                "v_norm_w", "gate_proj_w", "up_proj_w", "down_proj_w", "ple_gate_w",
                "ple_proj_w", "ple_norm_w",
            )),
            "weights": (Gemma4ListenerWeights, (
                "final_norm_w", "per_layer_model_proj_w", "per_layer_proj_norm_w",
                "per_layer_model_proj_scale", "per_layer_input_scale", "layers",
            )),
            "imported_kv": (Gemma4ImportedKV, ("k_ptr", "k_meta", "v_ptr", "v_meta")),
            "imported_cache": (Gemma4ImportedCache, ("layers",)),
        }
        for kind, (struct_type, fields) in abi_fields.items():
            offsets = (ctypes.c_size_t * len(fields))()
            count = int(self._lib[f"gemma4_listener_abi_{kind}_offsets"](offsets, len(fields)))
            expected_offsets = [int(getattr(struct_type, field).offset) for field in fields]
            if count != len(fields) or list(offsets) != expected_offsets:
                raise RuntimeError(
                    f"offset ABI nativa incompatibile per {kind}: "
                    f"{list(offsets)} != {expected_offsets} (count={count})"
                )

        topology = _load_compiled_listener_topology()
        self._manifest_path = ""
        self._checkpoint_files_sha256 = ""
        self._checkpoint_root = getattr(language_model, "_gemma4_checkpoint_path", None)
        if self._checkpoint_root is not None and not Path(self._checkpoint_root).is_dir():
            raise FileNotFoundError(f"loaded Gemma checkpoint directory does not exist: {self._checkpoint_root}")
        manifest_sha = self._lib.gemma4_listener_manifest_sha256().decode("utf-8")
        self._manifest_sha256 = topology["manifest_sha256"]
        if manifest_sha != topology["manifest_sha256"]:
            raise RuntimeError(
                "native DSO topology id differs from the compiled header "
                f"({manifest_sha} != {topology['manifest_sha256']})"
            )
        self._validate_model_topology(language_model, topology)
        logger.info("[native_listener] Native ABI v%d and compiled model topology verified", abi_ver)

        parameters = iter(language_model.parameters())
        first_parameter = next(parameters, None)
        if first_parameter is None or first_parameter.device.type != "cuda":
            raise ValueError("il checkpoint nativo deve risiedere interamente su CUDA")
        device = first_parameter.device
        if device.index != torch.cuda.current_device():
            raise ValueError("il checkpoint nativo deve usare il device CUDA corrente")
        for parameter in parameters:
            if parameter.device != device:
                raise ValueError("i pesi nativi devono risiedere su un solo device CUDA")
        self._model_device = device
        self._graph_stream: torch.cuda.Stream | None = None
        self._graph_capture_pending = self.enable_cuda_graph
        self._graph_capture_launches = 0
        self._graph_replay_launches = 0
        self._graph_step_launches = 0
        if self.enable_cuda_graph:
            self._graph_stream = torch.cuda.Stream(device=device, priority=0)
            if int(self._graph_stream.cuda_stream) == 0:
                raise RuntimeError("native listener CUDA Graph richiede uno stream CUDA non-default")
            # PyTorch pooled streams are non-blocking. This device-side wait
            # orders prior model work before any graph-owned C ABI operation.
            self._graph_stream.wait_stream(torch.cuda.current_stream(device=device))

        # 2. Register borrowed parameter pointers
        self._weights_struct = self._build_weights_struct(language_model)

        # 3. Preallocate persistent device output tensor and PAD PLE identities
        self.output_tensor = torch.empty(
            (1, self.stream_q_len, GEMMA4_HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=self._model_device,
        )
        # Accept the non-contiguous Q=4 view produced by the rolling audio
        # tensor without allocating a temporary on every frame.
        self._frame_input = torch.empty_like(self.output_tensor)

        if frame_pad_ple_tensor is None:
            pad_token_id = int(getattr(getattr(language_model, "config", None), "pad_token_id", 0))
            pad_ids = torch.full((1, self.stream_q_len), pad_token_id, device=device, dtype=torch.long)
            with torch.inference_mode():
                pad_embeds = language_model.embed_tokens(pad_ids)
                self.frame_pad_ple = language_model.get_per_layer_inputs(pad_ids, pad_embeds)
        else:
            if not isinstance(frame_pad_ple_tensor, torch.Tensor):
                raise TypeError("frame_pad_ple_tensor deve essere un torch.Tensor")
            self.frame_pad_ple = frame_pad_ple_tensor
        if (
            self.frame_pad_ple.device != device
            or self.frame_pad_ple.dtype != torch.bfloat16
            or not self.frame_pad_ple.is_contiguous()
            or tuple(self.frame_pad_ple.shape) != (1, self.stream_q_len, GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM)
        ):
            raise ValueError("frame_pad_ple deve essere CUDA BF16 contiguo con shape [1,Q,35,256]")
        self._weight_meta = self._build_weight_meta(language_model)
        self._output_meta = self._tensor_meta(self.output_tensor, dtype_code=1)
        self._frame_meta = self._tensor_meta(self._frame_input, dtype_code=1)
        self._frame_pad_ple_meta = self._tensor_meta(self.frame_pad_ple, dtype_code=1)
        self._frame_meta_ptr = ctypes.pointer(self._frame_meta)
        self._prefix_embeds_meta: Gemma4TensorMeta | None = None
        self._prefix_ple_meta: Gemma4TensorMeta | None = None
        self._prefix_attention_meta: Gemma4TensorMeta | None = None

        # 4. Create native context
        self._config_struct = Gemma4ListenerConfig(
            max_seq_len=self.max_seq_len,
            stream_q_len=self.stream_q_len,
            enable_cuda_graph=enable_cuda_graph,
            output_tensor_ptr=int(self.output_tensor.data_ptr()),
            frame_pad_ple_ptr=int(self.frame_pad_ple.data_ptr()),
            weight_meta=ctypes.cast(self._weight_meta, ctypes.POINTER(Gemma4TensorMeta)),
            weight_meta_count=len(self._weight_meta),
            output_meta=ctypes.pointer(self._output_meta),
            frame_pad_ple_meta=ctypes.pointer(self._frame_pad_ple_meta),
        )

        self._ctx = ctypes.c_void_p()
        self._active_prefix_len = 0
        if self._explicit_attention_backend is not None or self._explicit_rmsnorm_backend is not None:
            ret = self._lib.gemma4_listener_create_with_policy(
                ctypes.byref(self._config_struct),
                ctypes.byref(self._weights_struct),
                ctypes.byref(self._ctx),
                ctypes.c_int32(
                    -1 if self._explicit_attention_backend is None else self._explicit_attention_backend
                ),
                ctypes.c_int32(
                    -1 if self._explicit_rmsnorm_backend is None else self._explicit_rmsnorm_backend
                ),
            )
            create_name = "gemma4_listener_create_with_policy"
        else:
            ret = self._lib.gemma4_listener_create(
                ctypes.byref(self._config_struct),
                ctypes.byref(self._weights_struct),
                ctypes.byref(self._ctx),
            )
            create_name = "gemma4_listener_create"
        if ret != 0 or not self._ctx:
            raise RuntimeError(f"{create_name} failed with code {ret}")

        logger.info("[native_listener] Context creato con successo (Arena e Output residenti su GPU).")

    @staticmethod
    def _validate_model_topology(lm: Any, manifest: dict[str, Any]) -> None:
        def check_shape(tensor: Any, expected_shape: Any, label: str) -> None:
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(expected_shape):
                actual = None if tensor is None else tuple(tensor.shape)
                raise ValueError(f"{label}: shape inattesa {actual} != {tuple(expected_shape)}")

        def check_float(actual: Any, expected: Any, label: str) -> None:
            actual_float = float(actual)
            expected_float = float(expected)
            if not math.isfinite(actual_float) or not math.isfinite(expected_float):
                raise ValueError(f"{label}: valore non finito")
            if abs(actual_float - expected_float) > 1e-7:
                raise ValueError(f"{label}: valore inatteso {actual} != {expected}")

        config = getattr(lm, "config", None)
        layers = list(getattr(lm, "layers", ()))
        if config is None or len(layers) != GEMMA4_NUM_LAYERS:
            raise ValueError("checkpoint Gemma incompatibile: numero layer inatteso")
        if int(getattr(config, "hidden_size", -1)) != GEMMA4_HIDDEN_SIZE:
            raise ValueError("checkpoint Gemma incompatibile: hidden_size inatteso")
        if int(getattr(config, "num_hidden_layers", -1)) != GEMMA4_NUM_LAYERS:
            raise ValueError("checkpoint Gemma incompatibile: num_hidden_layers inatteso")
        if int(getattr(lm, "hidden_size_per_layer_input", -1)) != GEMMA4_PLE_DIM:
            raise ValueError("checkpoint Gemma incompatibile: PLE dim inattesa")
        if int(manifest.get("canonical_prefix_len", -1)) != AUDIO_START_AFTER_TEXT:
            raise ValueError("manifest topology incompatibile: prefix canonico inatteso")
        if int(manifest.get("canonical_lm_prefill_target_len", -1)) != 291:
            raise ValueError("manifest topology incompatibile: target prefill inatteso")
        ple_projection = getattr(lm, "per_layer_model_projection", None)
        check_shape(
            getattr(ple_projection, "weight", None),
            manifest.get("per_layer_model_projection_shape", ()),
            "per_layer_model_projection",
        )
        check_shape(
            getattr(getattr(lm, "per_layer_projection_norm", None), "weight", None),
            (GEMMA4_PLE_DIM,),
            "per_layer_projection_norm",
        )
        for scale_name in ("per_layer_model_projection_scale", "per_layer_input_scale"):
            scale = getattr(lm, scale_name, None)
            if scale is None or not math.isfinite(float(scale)):
                raise ValueError(f"checkpoint Gemma has invalid {scale_name}")
        check_float(getattr(getattr(lm, "norm", None), "eps", None), manifest["final_norm_eps"], "final_norm_eps")
        rope_parameters = getattr(config, "rope_parameters", None)
        if not isinstance(rope_parameters, dict):
            raise ValueError("checkpoint Gemma privo di rope_parameters attestabili")

        expected_layers = manifest.get("layers", ())
        if len(expected_layers) != GEMMA4_NUM_LAYERS:
            raise ValueError("manifest topology incompatibile: numero layer inatteso")
        for index, (layer, expected) in enumerate(zip(layers, expected_layers)):
            attention = layer.self_attn
            if int(expected.get("index", -1)) != index:
                raise ValueError(f"manifest layer index inatteso: {index}")
            actual_type = str(config.layer_types[index])
            if actual_type != expected["layer_type"]:
                raise ValueError(f"layer {index}: attention type inatteso")
            actual_num_heads = getattr(
                attention,
                "num_attention_heads",
                getattr(attention, "num_heads", getattr(config, "num_attention_heads", -1)),
            )
            if int(actual_num_heads) != int(expected["num_attention_heads"]):
                raise ValueError(f"layer {index}: num_heads inatteso")
            actual_num_kv_heads = getattr(
                attention,
                "num_key_value_heads",
                getattr(config, "num_key_value_heads", -1),
            )
            if int(actual_num_kv_heads) != int(expected["num_key_value_heads"]):
                raise ValueError(f"layer {index}: num_key_value_heads inatteso")
            if int(attention.head_dim) != int(expected["head_dim"]):
                raise ValueError(f"layer {index}: head_dim inatteso")
            rope_type = actual_type
            expected_rope = expected["rope_scaling"][rope_type]
            actual_rope = rope_parameters.get(rope_type)
            if not isinstance(actual_rope, dict):
                raise ValueError(f"layer {index}: rope_parameters inattesi")
            if str(actual_rope.get("rope_type")) != str(expected_rope["rope_type"]):
                raise ValueError(f"layer {index}: rope_type inatteso")
            check_float(actual_rope.get("rope_theta"), expected_rope["rope_theta"], f"layer {index} rope_theta")
            if "partial_rotary_factor" in expected_rope:
                check_float(
                    actual_rope.get("partial_rotary_factor"),
                    expected_rope["partial_rotary_factor"],
                    f"layer {index} partial_rotary_factor",
                )
            actual_window = int(getattr(attention, "sliding_window", 0) or 0)
            expected_window = int(expected.get("sliding_window", 0) or 0)
            if actual_window != expected_window:
                raise ValueError(f"layer {index}: sliding_window inattesa")
            check_float(getattr(attention, "scaling", None), expected["attention_scaling"], f"layer {index} scaling")
            check_float(getattr(layer, "layer_scalar", None), expected["layer_scalar"], f"layer {index} layer_scalar")
            if int(getattr(layer, "hidden_size_per_layer_input", -1)) != GEMMA4_PLE_DIM:
                raise ValueError(f"layer {index}: PLE dim inattesa")
            actual_shared = bool(attention.is_kv_shared_layer)
            expected_shared = bool(expected["is_kv_shared_layer"])
            if actual_shared != expected_shared:
                raise ValueError(f"layer {index}: shared-KV topology inattesa")
            if actual_shared:
                # transformers 5.14 removed the per-module source-index
                # attribute; the attested Gemma topology still identifies the
                # source by the full-attention store layer (14) versus the
                # sliding store layer (13).
                actual_shared_index = getattr(attention, "kv_shared_layer_index", None)
                if actual_shared_index is None:
                    actual_shared_index = 14 if actual_type == "full_attention" else 13
                if int(actual_shared_index) != int(expected["kv_shared_layer_index"]):
                    raise ValueError(f"layer {index}: shared-KV source inattesa")
            if bool(getattr(attention, "store_full_length_kv", False)) != bool(expected["store_full_length_kv"]):
                raise ValueError(f"layer {index}: store_full_length_kv inatteso")

            for name, module in (
                ("input_layernorm", layer.input_layernorm),
                ("post_attention_layernorm", layer.post_attention_layernorm),
                ("pre_feedforward_layernorm", layer.pre_feedforward_layernorm),
                ("post_feedforward_layernorm", layer.post_feedforward_layernorm),
                ("post_per_layer_input_norm", layer.post_per_layer_input_norm),
            ):
                check_shape(module.weight, (GEMMA4_HIDDEN_SIZE,), f"layer {index} {name}")
            source_attention = attention
            if actual_shared:
                source_index = int(expected["kv_shared_layer_index"])
                source_attention = layers[source_index].self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                module = getattr(attention, name, None)
                if module is None and name in {"k_proj", "v_proj"}:
                    module = getattr(source_attention, name, None)
                check_shape(module.weight if module is not None else None,
                            expected[f"{name}_shape"], f"layer {index} {name}")
            q_norm = getattr(attention, "q_norm", None) or getattr(source_attention, "q_norm", None)
            k_norm = getattr(attention, "k_norm", None) or getattr(source_attention, "k_norm", None)
            v_norm = getattr(attention, "v_norm", None) or getattr(source_attention, "v_norm", None)
            for name, module in (("q_norm", q_norm), ("k_norm", k_norm)):
                check_shape(module.weight if module is not None else None,
                            (int(expected["head_dim"]),), f"layer {index} {name}")
            if getattr(v_norm, "weight", None) is not None:
                raise ValueError(f"layer {index} v_norm: il checkpoint nativo atteso non ha peso")
            for name, module in (
                ("input_layernorm_eps", layer.input_layernorm),
                ("post_attention_layernorm_eps", layer.post_attention_layernorm),
                ("pre_feedforward_layernorm_eps", layer.pre_feedforward_layernorm),
                ("post_feedforward_layernorm_eps", layer.post_feedforward_layernorm),
                ("q_norm_eps", q_norm),
                ("k_norm_eps", k_norm),
                ("v_norm_eps", v_norm),
                ("post_per_layer_input_norm_eps", layer.post_per_layer_input_norm),
            ):
                check_float(getattr(module, "eps", None), expected[name], f"layer {index} {name}")
            for name in ("gate_proj", "up_proj", "down_proj"):
                check_shape(getattr(layer.mlp, name).weight, expected[f"{name}_shape"], f"layer {index} {name}")
            check_shape(layer.per_layer_input_gate.weight, expected["per_layer_input_gate_shape"], f"layer {index} per_layer_input_gate")
            check_shape(layer.per_layer_projection.weight, expected["per_layer_projection_shape"], f"layer {index} per_layer_projection")

    def _setup_c_signatures(self) -> None:
        self._lib.gemma4_listener_manifest_sha256.restype = ctypes.c_char_p
        self._lib.gemma4_listener_manifest_sha256.argtypes = []

        self._lib.gemma4_listener_abi_version.restype = ctypes.c_int
        self._lib.gemma4_listener_abi_version.argtypes = []
        for name in (
            "gemma4_listener_abi_config_size",
            "gemma4_listener_abi_tensor_meta_size",
            "gemma4_listener_abi_layer_weights_size",
            "gemma4_listener_abi_weights_size",
            "gemma4_listener_abi_imported_kv_size",
            "gemma4_listener_abi_imported_cache_size",
        ):
            function = getattr(self._lib, name)
            function.restype = ctypes.c_size_t
            function.argtypes = []
        for name in (
            "gemma4_listener_abi_config_offsets",
            "gemma4_listener_abi_tensor_meta_offsets",
            "gemma4_listener_abi_layer_weights_offsets",
            "gemma4_listener_abi_weights_offsets",
            "gemma4_listener_abi_imported_kv_offsets",
            "gemma4_listener_abi_imported_cache_offsets",
        ):
            function = getattr(self._lib, name)
            function.restype = ctypes.c_size_t
            function.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]

        self._lib.gemma4_listener_create.restype = ctypes.c_int
        self._lib.gemma4_listener_create.argtypes = [
            ctypes.POINTER(Gemma4ListenerConfig),
            ctypes.POINTER(Gemma4ListenerWeights),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.gemma4_listener_create_with_policy.restype = ctypes.c_int
        self._lib.gemma4_listener_create_with_policy.argtypes = [
            ctypes.POINTER(Gemma4ListenerConfig),
            ctypes.POINTER(Gemma4ListenerWeights),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int32,
            ctypes.c_int32,
        ]

        self._lib.gemma4_listener_import_cache_async.restype = ctypes.c_int
        self._lib.gemma4_listener_import_cache_async.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4ImportedCache),
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self._import_cache_with_prefix = getattr(
            self._lib, "gemma4_listener_import_cache_with_prefix_async", None
        )
        if self._import_cache_with_prefix is not None:
            self._import_cache_with_prefix.restype = ctypes.c_int
            self._import_cache_with_prefix.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(Gemma4ImportedCache),
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.POINTER(Gemma4TensorMeta),
                ctypes.c_uint32,
                ctypes.c_void_p,
            ]

        self._lib.gemma4_listener_debug_execute_layer.restype = ctypes.c_int
        self._lib.gemma4_listener_debug_execute_layer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.POINTER(Gemma4ImportedCache),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]

        self._lib.gemma4_listener_debug_attention_splitk.restype = ctypes.c_int
        self._lib.gemma4_listener_debug_attention_splitk.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_uint32,
            ctypes.c_double,
            ctypes.c_void_p,
        ]

        self._lib.gemma4_listener_debug_rmsnorm_trace.restype = ctypes.c_int
        self._lib.gemma4_listener_debug_rmsnorm_trace.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.gemma4_listener_debug_rmsnorm_native_exact.restype = ctypes.c_int
        self._lib.gemma4_listener_debug_rmsnorm_native_exact.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]

        self._lib.gemma4_listener_debug_linear.restype = ctypes.c_int
        self._lib.gemma4_listener_debug_linear.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self._lib.gemma4_listener_debug_gelu_tanh.restype = ctypes.c_int
        self._lib.gemma4_listener_debug_gelu_tanh.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self._lib.gemma4_listener_debug_mul.restype = ctypes.c_int
        self._lib.gemma4_listener_debug_mul.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]

        self._lib.gemma4_aten_rmsnorm_parity_async.restype = ctypes.c_int
        self._lib.gemma4_aten_rmsnorm_parity_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_double,
            ctypes.c_void_p,
        ]

        self._lib.gemma4_listener_debug_attention.restype = ctypes.c_int
        self._lib.gemma4_listener_debug_attention.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_double,
            ctypes.c_void_p,
        ]
        self._lib.gemma4_listener_debug_attention_gqa.restype = ctypes.c_int
        self._lib.gemma4_listener_debug_attention_gqa.argtypes = list(
            self._lib.gemma4_listener_debug_attention.argtypes
        )

        self._lib.gemma4_listener_prime_async.restype = ctypes.c_int
        self._lib.gemma4_listener_prime_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]

        self._lib.gemma4_listener_prime_debug.restype = ctypes.c_int
        self._lib.gemma4_listener_prime_debug.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_bool,
            ctypes.c_void_p,
        ]

        self._lib.gemma4_listener_step_async.restype = ctypes.c_int
        self._lib.gemma4_listener_step_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
        ]
        self._lib.gemma4_listener_rebase_async.restype = ctypes.c_int
        self._lib.gemma4_listener_rebase_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]

        self._lib.gemma4_listener_step_debug.restype = ctypes.c_int
        self._lib.gemma4_listener_step_debug.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_uint32,
            ctypes.c_bool,
            ctypes.c_void_p,
        ]


        self._lib.gemma4_listener_output_ptr.restype = ctypes.c_void_p
        self._lib.gemma4_listener_output_ptr.argtypes = [ctypes.c_void_p]
        self._lib.gemma4_listener_output_health.restype = ctypes.c_int
        self._lib.gemma4_listener_output_health.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)
        ]

        self._lib.gemma4_listener_debug_ptr.restype = ctypes.c_void_p
        self._lib.gemma4_listener_debug_ptr.argtypes = [ctypes.c_void_p, ctypes.c_char_p]


        self._lib.gemma4_listener_reset.restype = ctypes.c_int
        self._lib.gemma4_listener_reset.argtypes = [ctypes.c_void_p]

        self._lib.gemma4_listener_destroy.restype = None
        self._lib.gemma4_listener_destroy.argtypes = [ctypes.c_void_p]

        for name in (
            "gemma4_listener_runtime_attention_backend",
            "gemma4_listener_runtime_rmsnorm_backend",
            "gemma4_listener_runtime_max_seq_len",
            "gemma4_listener_runtime_current_position",
            "gemma4_listener_current_position",
        ):
            function = getattr(self._lib, name)
            function.restype = ctypes.c_uint32
            function.argtypes = [ctypes.c_void_p]
        self._lib.gemma4_listener_runtime_splitk_layer_mask.restype = ctypes.c_uint64
        self._lib.gemma4_listener_runtime_splitk_layer_mask.argtypes = [ctypes.c_void_p]
        self._lib.gemma4_listener_runtime_derived_weight_bytes.restype = ctypes.c_size_t
        self._lib.gemma4_listener_runtime_derived_weight_bytes.argtypes = [ctypes.c_void_p]
        self._lib.gemma4_listener_runtime_attention_implementation.restype = ctypes.c_char_p
        self._lib.gemma4_listener_runtime_attention_implementation.argtypes = [ctypes.c_void_p]
        self._lib.gemma4_listener_runtime_ordered_calls_by_layer.restype = ctypes.c_uint64
        self._lib.gemma4_listener_runtime_ordered_calls_by_layer.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32
        ]
        self._lib.gemma4_listener_runtime_ordered_key_tile_size.restype = ctypes.c_uint32
        self._lib.gemma4_listener_runtime_ordered_key_tile_size.argtypes = [ctypes.c_void_p]
        self._lib.gemma4_listener_runtime_ordered_min_keys_exclusive.restype = ctypes.c_uint32
        self._lib.gemma4_listener_runtime_ordered_min_keys_exclusive.argtypes = [ctypes.c_void_p]
        self._lib.gemma4_listener_runtime_ordered_workspace_bytes.restype = ctypes.c_size_t
        self._lib.gemma4_listener_runtime_ordered_workspace_bytes.argtypes = [ctypes.c_void_p]
        for name in (
            "gemma4_listener_runtime_ordered_pack_launches",
            "gemma4_listener_runtime_ordered_stage_a_launches",
            "gemma4_listener_runtime_ordered_stage_b_launches",
            "gemma4_listener_runtime_ordered_stage_c_launches",
            "gemma4_listener_runtime_ordered_stage_d_launches",
        ):
            function = getattr(self._lib, name)
            function.restype = ctypes.c_uint64
            function.argtypes = [ctypes.c_void_p]

        self._lib.gemma4_listener_get_cached_k.restype = ctypes.c_void_p
        self._lib.gemma4_listener_get_cached_k.argtypes = [ctypes.c_void_p, ctypes.c_int]

        self._lib.gemma4_listener_get_cached_v.restype = ctypes.c_void_p
        self._lib.gemma4_listener_get_cached_v.argtypes = [ctypes.c_void_p, ctypes.c_int]

    @staticmethod
    def _tensor_meta(tensor: torch.Tensor | None, *, dtype_code: int) -> Gemma4TensorMeta:
        meta = Gemma4TensorMeta()
        if tensor is None:
            return meta
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            raise ValueError("metadata nativo richiede tensor CUDA")
        expected_dtype = {
            1: torch.bfloat16,
            2: torch.int64,
            3: torch.float32,
            4: torch.bool,
        }.get(int(dtype_code))
        if expected_dtype is None or tensor.dtype != expected_dtype:
            raise ValueError(f"dtype metadata nativo inatteso: {tensor.dtype} != {expected_dtype}")
        shape = tuple(int(value) for value in tensor.shape)
        if len(shape) > 4 or any(value <= 0 for value in shape):
            raise ValueError(f"rank/shape tensor nativo inattesi: {shape}")
        expected_strides = []
        running = 1
        for size in reversed(shape):
            expected_strides.append(running)
            running *= size
        expected_strides = tuple(reversed(expected_strides))
        if tuple(int(value) for value in tensor.stride()) != expected_strides:
            raise ValueError("metadata nativo richiede layout row-major contiguous")
        strides = tuple(int(value) for value in tensor.stride())
        meta.ptr = int(tensor.data_ptr())
        meta.dtype = int(dtype_code)
        meta.ndim = len(shape)
        meta.contiguous = 1
        meta.reserved = 0
        meta.dims = (ctypes.c_uint64 * 4)(*(shape + (0,) * (4 - len(shape))))
        meta.strides = (ctypes.c_uint64 * 4)(*(strides + (0,) * (4 - len(strides))))
        return meta

    def _ptr(self, tensor: torch.Tensor | None) -> int:
        if tensor is None:
            return 0
        if tensor.device.type != "cuda" or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError("tutti i pesi nativi devono essere tensor CUDA BF16 contigui")
        self._pinned_tensors.append(tensor)
        return int(tensor.data_ptr())

    def _build_weight_meta(self, lm: Any) -> Any:
        tensors: list[torch.Tensor | None] = [
            getattr(getattr(lm, "norm", None), "weight", None),
            getattr(getattr(lm, "per_layer_model_projection", None), "weight", None),
            getattr(getattr(lm, "per_layer_projection_norm", None), "weight", None),
        ]
        for layer in lm.layers[:GEMMA4_NUM_LAYERS]:
            attn = layer.self_attn
            tensors.extend([
                getattr(layer.input_layernorm, "weight", None),
                getattr(layer.post_attention_layernorm, "weight", None),
                getattr(layer.pre_feedforward_layernorm, "weight", None),
                getattr(layer.post_feedforward_layernorm, "weight", None),
                getattr(attn.q_proj, "weight", None),
                getattr(attn.k_proj, "weight", None) if not attn.is_kv_shared_layer else None,
                getattr(attn.v_proj, "weight", None) if not attn.is_kv_shared_layer else None,
                getattr(attn.o_proj, "weight", None),
                getattr(attn.q_norm, "weight", None),
                getattr(attn.k_norm, "weight", None) if not attn.is_kv_shared_layer else None,
                getattr(attn.v_norm, "weight", None) if not attn.is_kv_shared_layer else None,
                getattr(layer.mlp.gate_proj, "weight", None),
                getattr(layer.mlp.up_proj, "weight", None),
                getattr(layer.mlp.down_proj, "weight", None),
                getattr(layer.per_layer_input_gate, "weight", None),
                getattr(layer.per_layer_projection, "weight", None),
                getattr(layer.post_per_layer_input_norm, "weight", None),
            ])
        expected_count = 3 + GEMMA4_NUM_LAYERS * 17
        if len(tensors) != expected_count:
            raise RuntimeError(f"metadata pesi nativi incompleti: {len(tensors)} != {expected_count}")
        metadata = (Gemma4TensorMeta * expected_count)()
        for index, tensor in enumerate(tensors):
            metadata[index] = self._tensor_meta(tensor, dtype_code=1)
        return metadata

    def _build_weights_struct(self, lm: Any) -> Gemma4ListenerWeights:
        w = Gemma4ListenerWeights()
        w.final_norm_w = self._ptr(getattr(getattr(lm, "norm", None), "weight", None))

        if hasattr(lm, "per_layer_model_projection") and hasattr(lm.per_layer_model_projection, "weight"):
            w.per_layer_model_proj_w = self._ptr(lm.per_layer_model_projection.weight)
        if hasattr(lm, "per_layer_projection_norm") and hasattr(lm.per_layer_projection_norm, "weight"):
            w.per_layer_proj_norm_w = self._ptr(lm.per_layer_projection_norm.weight)

        w.per_layer_model_proj_scale = float(getattr(lm, "per_layer_model_projection_scale", 1.0))
        w.per_layer_input_scale = float(getattr(lm, "per_layer_input_scale", 1.0))

        for i, layer in enumerate(lm.layers[:GEMMA4_NUM_LAYERS]):
            lw = w.layers[i]
            lw.input_norm_w = self._ptr(layer.input_layernorm.weight)
            lw.post_attn_norm_w = self._ptr(layer.post_attention_layernorm.weight)
            lw.pre_ff_norm_w = self._ptr(layer.pre_feedforward_layernorm.weight)
            lw.post_ff_norm_w = self._ptr(layer.post_feedforward_layernorm.weight)

            attn = layer.self_attn
            lw.q_proj_w = self._ptr(attn.q_proj.weight)
            lw.q_norm_w = self._ptr(getattr(attn.q_norm, "weight", None))

            if not attn.is_kv_shared_layer:
                lw.k_proj_w = self._ptr(attn.k_proj.weight if attn.k_proj is not None else None)
                lw.v_proj_w = self._ptr(attn.v_proj.weight if attn.v_proj is not None else None)
                lw.k_norm_w = self._ptr(getattr(attn.k_norm, "weight", None))
                lw.v_norm_w = self._ptr(getattr(attn.v_norm, "weight", None))
            else:
                lw.k_proj_w = 0
                lw.v_proj_w = 0
                lw.k_norm_w = 0
                lw.v_norm_w = 0

            lw.o_proj_w = self._ptr(attn.o_proj.weight)

            mlp = layer.mlp
            lw.gate_proj_w = self._ptr(mlp.gate_proj.weight)
            lw.up_proj_w = self._ptr(mlp.up_proj.weight)
            lw.down_proj_w = self._ptr(mlp.down_proj.weight)

            if getattr(layer, "hidden_size_per_layer_input", 0):
                lw.ple_gate_w = self._ptr(layer.per_layer_input_gate.weight)
                lw.ple_proj_w = self._ptr(layer.per_layer_projection.weight)
                lw.ple_norm_w = self._ptr(getattr(layer.post_per_layer_input_norm, "weight", None))
            else:
                lw.ple_gate_w = 0
                lw.ple_proj_w = 0
                lw.ple_norm_w = 0

        return w

    def _cuda_stream(self, stream: torch.cuda.Stream | None) -> torch.cuda.Stream:
        if stream is not None:
            if torch.device(stream.device) != self.output_tensor.device:
                raise ValueError("CUDA stream e output nativo devono usare lo stesso device")
            return stream
        return torch.cuda.current_stream(device=self.output_tensor.device)

    def _resolve_stream(
        self,
        stream: torch.cuda.Stream | None,
        producer_stream: torch.cuda.Stream | None,
    ) -> torch.cuda.Stream:
        requested = self._cuda_stream(stream)
        caller_current = self._cuda_stream(None)
        if self.enable_cuda_graph:
            active = self._graph_stream if stream is None else requested
            if active is None or int(active.cuda_stream) == 0:
                raise RuntimeError("native listener CUDA Graph non può usare il legacy default stream")
            producer = (
                self._cuda_stream(producer_stream)
                if producer_stream is not None
                else requested if stream is not None else caller_current
            )
        else:
            active = requested
            producer = self._cuda_stream(producer_stream) if producer_stream is not None else active
        if active.cuda_stream != producer.cuda_stream:
            # Device-side ordering only: inputs are produced on producer and
            # native work is enqueued on the session graph stream.
            active.wait_stream(producer)
        return active

    def _record_graph_step_success(self) -> None:
        if not self.enable_cuda_graph:
            return
        self._graph_step_launches += 1
        if self._graph_capture_pending:
            # The native C ABI captures and launches the first graph in one
            # successful step_async call for the current cache generation.
            self._graph_capture_launches += 1
            self._graph_capture_pending = False
        else:
            self._graph_replay_launches += 1

    def _mark_graph_capture_pending(self) -> None:
        if self.enable_cuda_graph:
            self._graph_capture_pending = True

    @classmethod
    def _canonicalize_cuda_tensor(cls, tensor: torch.Tensor, *, name: str, dtype: torch.dtype) -> torch.Tensor:
        cls._require_cuda_tensor(tensor, name=name, dtype=dtype)
        shape = tuple(int(value) for value in tensor.shape)
        if len(shape) > 4:
            raise ValueError(f"{name} rank inatteso: {len(shape)}")
        running = 1
        expected = []
        for size in reversed(shape):
            expected.append(running)
            running *= size
        expected_strides = tuple(reversed(expected))
        if tuple(int(value) for value in tensor.stride()) == expected_strides:
            return tensor
        canonical = torch.empty(shape, dtype=dtype, device=tensor.device)
        canonical.copy_(tensor)
        return canonical

    @staticmethod
    def _require_cuda_contiguous(tensor: torch.Tensor, *, name: str, dtype: torch.dtype) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} deve essere un torch.Tensor")
        if tensor.device.type != "cuda" or tensor.dtype != dtype:
            raise ValueError(f"{name} deve essere CUDA {dtype} contiguo")
        shape = tuple(int(value) for value in tensor.shape)
        running = 1
        expected = []
        for size in reversed(shape):
            expected.append(running)
            running *= size
        if tuple(int(value) for value in tensor.stride()) != tuple(reversed(expected)):
            raise ValueError(f"{name} deve avere stride row-major contiguo")

    @staticmethod
    def _require_cuda_tensor(tensor: torch.Tensor, *, name: str, dtype: torch.dtype) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} deve essere un torch.Tensor")
        if tensor.device.type != "cuda" or tensor.dtype != dtype:
            raise ValueError(f"{name} deve essere CUDA {dtype}")

    def _validate_prime_inputs(
        self,
        prefix_embeds: torch.Tensor,
        prefix_ple: torch.Tensor | None,
        prefix_attention: torch.Tensor | None,
    ) -> tuple[int, torch.cuda.Stream]:
        if prefix_ple is None or prefix_attention is None:
            raise ValueError("prime nativo richiede PLE e prefix_attention espliciti")
        self._require_cuda_contiguous(prefix_embeds, name="prefix_embeds", dtype=torch.bfloat16)
        self._require_cuda_contiguous(prefix_ple, name="prefix_ple", dtype=torch.bfloat16)
        self._require_cuda_contiguous(prefix_attention, name="prefix_attention", dtype=torch.int64)
        if prefix_embeds.ndim != 3 or tuple(prefix_embeds.shape[:1]) != (1,) or prefix_embeds.shape[2] != GEMMA4_HIDDEN_SIZE:
            raise ValueError(f"prefix_embeds inatteso: {tuple(prefix_embeds.shape)}")
        prefix_len = int(prefix_embeds.shape[1])
        expected_ple = (1, prefix_len, GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM)
        if tuple(prefix_ple.shape) != expected_ple:
            raise ValueError(f"prefix_ple inatteso: atteso {expected_ple}, ottenuto {tuple(prefix_ple.shape)}")
        if tuple(prefix_attention.shape) != (1, prefix_len):
            raise ValueError(f"prefix_attention inatteso: atteso {(1, prefix_len)}, ottenuto {tuple(prefix_attention.shape)}")
        if prefix_embeds.device != self.output_tensor.device or prefix_ple.device != prefix_embeds.device or prefix_attention.device != prefix_embeds.device:
            raise ValueError("tutti gli input prime devono risiedere sul device CUDA del listener")
        if prefix_len != AUDIO_START_AFTER_TEXT:
            raise ValueError(
                f"prefix_len canonico inatteso: {prefix_len}, atteso {AUDIO_START_AFTER_TEXT}"
            )
        if not bool(torch.all((prefix_attention == 0) | (prefix_attention == 1)).item()):
            raise ValueError("prefix_attention deve contenere esclusivamente 0/1")
        self._prefix_embeds_meta = self._tensor_meta(prefix_embeds, dtype_code=1)
        self._prefix_ple_meta = self._tensor_meta(prefix_ple, dtype_code=1)
        self._prefix_attention_meta = self._tensor_meta(prefix_attention, dtype_code=2)
        return prefix_len, self._cuda_stream(None)

    def _prepare_imported_cache(
        self,
        past_key_values: Any,
        current_position: int,
    ) -> tuple[Gemma4ImportedCache, list[Gemma4TensorMeta], list[torch.Tensor]]:
        layers = getattr(past_key_values, "layers", None)
        source_layer_count = sum(
            not bool(getattr(layer.self_attn, "is_kv_shared_layer", False))
            for layer in self.language_model.layers[:GEMMA4_NUM_LAYERS]
        )
        if layers is None or len(layers) < source_layer_count:
            raise ValueError(
                "DynamicCache canonica priva dei layer KV sorgente richiesti: "
                f"{len(layers) if layers is not None else 0} < {source_layer_count}"
            )
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length) and int(get_seq_length(0)) != current_position:
            raise ValueError("DynamicCache canonica con lunghezza inattesa")

        imported = Gemma4ImportedCache()
        descriptors: list[Gemma4TensorMeta] = []
        imported_tensors: list[torch.Tensor] = []
        shared_layers = getattr(past_key_values, "shared_layers", None)
        for layer_idx in range(GEMMA4_NUM_LAYERS):
            attention = self.language_model.layers[layer_idx].self_attn
            if bool(getattr(attention, "is_kv_shared_layer", False)):
                continue
            source = None
            if shared_layers is not None and hasattr(shared_layers, "get"):
                source = shared_layers.get(layer_idx)
            if source is None:
                source_layer = layers[layer_idx]
                keys = getattr(source_layer, "keys", None)
                values = getattr(source_layer, "values", None)
            else:
                keys, values = source
            head_dim = int(attention.head_dim)
            expected_shape = (1, 1, current_position, head_dim)
            keys = self._canonicalize_cuda_tensor(
                keys, name=f"cache K layer {layer_idx}", dtype=torch.bfloat16
            )
            values = self._canonicalize_cuda_tensor(
                values, name=f"cache V layer {layer_idx}", dtype=torch.bfloat16
            )
            if tuple(keys.shape) != expected_shape or tuple(values.shape) != expected_shape:
                raise ValueError(
                    f"cache KV layer {layer_idx} inattesa: "
                    f"{tuple(keys.shape)}/{tuple(values.shape)} != {expected_shape}"
                )
            k_meta = self._tensor_meta(keys, dtype_code=1)
            v_meta = self._tensor_meta(values, dtype_code=1)
            descriptors.extend((k_meta, v_meta))
            entry = imported.layers[layer_idx]
            entry.k_ptr = int(keys.data_ptr())
            entry.k_meta = ctypes.pointer(descriptors[-2])
            entry.v_ptr = int(values.data_ptr())
            entry.v_meta = ctypes.pointer(descriptors[-1])
            imported_tensors.extend((keys, values))
        return imported, descriptors, imported_tensors

    def import_cache_async(
        self,
        past_key_values: Any,
        current_position: int = AUDIO_START_AFTER_TEXT,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
        *,
        prefix_len: int | None = None,
        prefix_mask: torch.Tensor | None = None,
    ) -> None:
        """Import an exact ShapeStable DynamicCache snapshot with D2D K/V copies.

        The snapshot position may be the canonical prefix or any later logical
        position. Only the non-shared source layers are described. Native
        shared layers already alias sources 13 and 14, matching Gemma's
        shared-KV topology.
        Descriptor validation is synchronous, while every K/V payload transfer
        is device-to-device and ordered on ``stream``.
        """
        current_position = int(current_position)
        prefix_aware = prefix_len is not None or prefix_mask is not None
        if prefix_aware:
            if prefix_len is None or prefix_mask is None:
                raise ValueError("prefix_len e prefix_mask devono essere forniti insieme")
            prefix_len = int(prefix_len)
            if not 1 <= prefix_len <= CANONICAL_LM_PREFILL_TARGET_LEN:
                raise ValueError(
                    "prefix_len deve essere compreso tra 1 e "
                    f"{CANONICAL_LM_PREFILL_TARGET_LEN}"
                )
            if current_position < prefix_len or (current_position - prefix_len) % self.stream_q_len:
                raise ValueError(
                    "current_position deve essere prefix_len più un multiplo della lunghezza Q4"
                )
            if self._import_cache_with_prefix is None:
                raise RuntimeError(
                    "la libreria nativa non espone l'import con prefisso semantico"
                )
            prefix_mask = self._canonicalize_cuda_tensor(
                prefix_mask, name="prefix_mask", dtype=torch.int64
            )
            self._require_cuda_contiguous(
                prefix_mask, name="prefix_mask", dtype=torch.int64
            )
            if prefix_mask.device != self.output_tensor.device:
                raise ValueError("prefix_mask deve risiedere sul device del listener")
            if tuple(prefix_mask.shape) != (1, prefix_len):
                raise ValueError(
                    f"prefix_mask deve avere forma {(1, prefix_len)}, "
                    f"ottenuta {tuple(prefix_mask.shape)}"
                )
        elif current_position < AUDIO_START_AFTER_TEXT:
            raise ValueError(
                f"cache import richiede posizione >= {AUDIO_START_AFTER_TEXT}, "
                f"ottenuta {current_position}"
            )
        imported, descriptors, imported_tensors = self._prepare_imported_cache(
            past_key_values, current_position
        )
        active_stream = self._resolve_stream(stream, producer_stream)
        for tensor in imported_tensors:
            tensor.record_stream(active_stream)
        c_stream = ctypes.c_void_p(active_stream.cuda_stream)
        if prefix_aware:
            assert prefix_mask is not None and prefix_len is not None
            prefix_mask.record_stream(active_stream)
            prefix_meta = self._tensor_meta(prefix_mask, dtype_code=2)
            ret = self._import_cache_with_prefix(
                self._ctx,
                ctypes.byref(imported),
                ctypes.c_uint32(current_position),
                ctypes.c_void_p(prefix_mask.data_ptr()),
                ctypes.byref(prefix_meta),
                ctypes.c_uint32(prefix_len),
                c_stream,
            )
        else:
            ret = self._lib.gemma4_listener_import_cache_async(
                self._ctx,
                ctypes.byref(imported),
                ctypes.c_uint32(current_position),
                c_stream,
            )
        if ret != 0:
            api_name = (
                "gemma4_listener_import_cache_with_prefix_async"
                if prefix_aware else "gemma4_listener_import_cache_async"
            )
            raise RuntimeError(f"{api_name} failed with code {ret}")
        self._cache_import_tensors = imported_tensors
        self._active_prefix_len = (
            int(prefix_len) if prefix_aware and prefix_len is not None
            else AUDIO_START_AFTER_TEXT
        )
        if prefix_aware and prefix_mask is not None:
            self._cache_import_tensors.append(prefix_mask)
        self._last_execution_stream = active_stream
        self._mark_graph_capture_pending()

    def debug_execute_layer(
        self,
        layer_idx: int,
        hidden_input: torch.Tensor,
        projected_ple: torch.Tensor,
        past_key_values: Any,
        current_position: int = AUDIO_START_AFTER_TEXT,
        query_position: int = AUDIO_START_AFTER_TEXT,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Execute one layer from exact oracle hidden/PLE/KV tensors.

        This is diagnostic only: it does not advance ``current_position`` or
        mark the context primed. Reset the context between independent probes.
        """
        layer_idx = int(layer_idx)
        if not 0 <= layer_idx < GEMMA4_NUM_LAYERS:
            raise ValueError(f"layer_idx inatteso: {layer_idx}")
        hidden_input = self._canonicalize_cuda_tensor(
            hidden_input, name="debug hidden_input", dtype=torch.bfloat16
        )
        projected_ple = self._canonicalize_cuda_tensor(
            projected_ple, name="debug projected_ple", dtype=torch.bfloat16
        )
        if tuple(hidden_input.shape) != (1, self.stream_q_len, GEMMA4_HIDDEN_SIZE):
            raise ValueError(f"debug hidden_input inatteso: {tuple(hidden_input.shape)}")
        expected_ple = (
            1, self.stream_q_len, GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM
        )
        if tuple(projected_ple.shape) != expected_ple:
            raise ValueError(f"debug projected_ple inatteso: {tuple(projected_ple.shape)}")
        imported, descriptors, imported_tensors = self._prepare_imported_cache(
            past_key_values, int(current_position)
        )
        # Keep descriptor storage alive until the C ABI call returns.
        active_stream = self._resolve_stream(stream, producer_stream)
        for tensor in (*imported_tensors, hidden_input, projected_ple):
            tensor.record_stream(active_stream)
        hidden_meta = self._tensor_meta(hidden_input, dtype_code=1)
        ple_meta = self._tensor_meta(projected_ple, dtype_code=1)
        c_stream = ctypes.c_void_p(active_stream.cuda_stream)
        ret = self._lib.gemma4_listener_debug_execute_layer(
            self._ctx,
            ctypes.c_uint32(layer_idx),
            ctypes.c_void_p(hidden_input.data_ptr()),
            ctypes.byref(hidden_meta),
            ctypes.c_void_p(projected_ple.data_ptr()),
            ctypes.byref(ple_meta),
            ctypes.byref(imported),
            ctypes.c_uint32(int(current_position)),
            ctypes.c_uint32(int(query_position)),
            c_stream,
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_debug_execute_layer failed with code {ret}")
        self._cache_import_tensors = imported_tensors
        self._last_execution_stream = active_stream
        self.output_tensor.record_stream(active_stream)
        return self.output_tensor

    def debug_rmsnorm_trace(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> dict[str, torch.Tensor]:
        """Trace the current custom RMSNorm on a borrowed exact input.

        Diagnostic-only: the custom production kernel is launched unchanged
        and its FP32 intermediates are copied into caller-owned CUDA buffers.
        """
        self._require_cuda_contiguous(input_tensor, name="rmsnorm input", dtype=torch.bfloat16)
        self._require_cuda_contiguous(weight, name="rmsnorm weight", dtype=torch.bfloat16)
        if input_tensor.ndim != 3 or int(input_tensor.shape[0]) != 1:
            raise ValueError("rmsnorm input deve avere shape [1, rows, dim]")
        rows = int(input_tensor.shape[1])
        dim = int(input_tensor.shape[2])
        if rows <= 0 or rows > self.stream_q_len or tuple(weight.shape) != (dim,):
            raise ValueError("shape RMSNorm inattesa")
        if not math.isfinite(float(eps)) or float(eps) < 0:
            raise ValueError("eps RMSNorm deve essere finito e non negativo")

        active_stream = self._resolve_stream(stream, producer_stream)
        output = torch.empty_like(input_tensor)
        input_fp32 = torch.empty((rows, dim), dtype=torch.float32, device=input_tensor.device)
        x_squared_fp32 = torch.empty_like(input_fp32)
        normalized_fp32 = torch.empty_like(input_fp32)
        weighted_fp32 = torch.empty_like(input_fp32)
        row_sum_fp32 = torch.empty((rows,), dtype=torch.float32, device=input_tensor.device)
        row_mean_fp32 = torch.empty_like(row_sum_fp32)
        mean_plus_eps_fp32 = torch.empty_like(row_sum_fp32)
        pow_neg_half_fp32 = torch.empty_like(row_sum_fp32)
        for tensor in (
            input_tensor, weight, output, input_fp32, x_squared_fp32,
            normalized_fp32, weighted_fp32, row_sum_fp32, row_mean_fp32,
            mean_plus_eps_fp32, pow_neg_half_fp32,
        ):
            tensor.record_stream(active_stream)

        input_meta = self._tensor_meta(input_tensor, dtype_code=1)
        weight_meta = self._tensor_meta(weight, dtype_code=1)
        output_meta = self._tensor_meta(output, dtype_code=1)
        ret = self._lib.gemma4_listener_debug_rmsnorm_trace(
            self._ctx,
            ctypes.c_void_p(input_tensor.data_ptr()),
            ctypes.byref(input_meta),
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.byref(weight_meta),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.byref(output_meta),
            ctypes.c_uint32(rows),
            ctypes.c_uint32(dim),
            ctypes.c_double(float(eps)),
            ctypes.c_void_p(input_fp32.data_ptr()),
            ctypes.c_void_p(x_squared_fp32.data_ptr()),
            ctypes.c_void_p(row_sum_fp32.data_ptr()),
            ctypes.c_void_p(row_mean_fp32.data_ptr()),
            ctypes.c_void_p(mean_plus_eps_fp32.data_ptr()),
            ctypes.c_void_p(pow_neg_half_fp32.data_ptr()),
            ctypes.c_void_p(normalized_fp32.data_ptr()),
            ctypes.c_void_p(weighted_fp32.data_ptr()),
            ctypes.c_void_p(active_stream.cuda_stream),
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_debug_rmsnorm_trace failed with code {ret}")
        return {
            "input_bf16": input_tensor,
            "input_fp32": input_fp32,
            "x_squared_fp32": x_squared_fp32,
            "row_sum_fp32": row_sum_fp32,
            "row_mean_fp32": row_mean_fp32,
            "mean_plus_eps_fp32": mean_plus_eps_fp32,
            "pow_neg_half_fp32": pow_neg_half_fp32,
            "normalized_fp32": normalized_fp32,
            "weighted_fp32": weighted_fp32,
            "final_bf16": output,
        }

    def debug_rmsnorm_native_exact_trace(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor | None,
        *,
        eps: float,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> dict[str, torch.Tensor]:
        """Trace the staged NATIVE_EXACT RMSNorm implementation.

        This diagnostic entrypoint exposes the native square, ATen-compatible
        reduction, rsqrt, FP32 pointwise values, and final BF16 output. It is
        intentionally independent of the selected runtime backend.
        """
        self._require_cuda_contiguous(
            input_tensor, name="native exact RMSNorm input", dtype=torch.bfloat16
        )
        if input_tensor.ndim != 3 or int(input_tensor.shape[0]) != 1:
            raise ValueError("native exact RMSNorm input deve avere shape [1, rows, dim]")
        rows = int(input_tensor.shape[1])
        dim = int(input_tensor.shape[2])
        if rows <= 0 or rows > 4 * GEMMA4_NUM_LAYERS or dim not in (256, 512, 1536):
            raise ValueError("shape native exact RMSNorm inattesa")
        if weight is not None:
            if weight.device != input_tensor.device or weight.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError("native exact RMSNorm weight deve essere CUDA BF16 o FP32")
            if not weight.is_contiguous() or tuple(weight.shape) != (dim,):
                raise ValueError("shape/layout native exact RMSNorm weight inattesi")
            weight_fp32 = weight.float() if weight.dtype == torch.bfloat16 else weight
        else:
            weight_fp32 = None
        if not math.isfinite(float(eps)) or float(eps) < 0:
            raise ValueError("eps native exact RMSNorm deve essere finito e non negativo")

        active_stream = self._resolve_stream(stream, producer_stream)
        output = torch.empty_like(input_tensor)
        squared = torch.empty((rows, dim), dtype=torch.float32, device=input_tensor.device)
        row_sum = torch.empty((rows,), dtype=torch.float32, device=input_tensor.device)
        mean = torch.empty((rows,), dtype=torch.float32, device=input_tensor.device)
        mean_plus_eps = torch.empty_like(mean)
        scale = torch.empty_like(mean)
        normalized = torch.empty_like(squared)
        weighted = torch.empty_like(squared)
        tensors = (
            input_tensor, output, squared, row_sum, mean, mean_plus_eps, scale,
            normalized, weighted,
        )
        for tensor in tensors:
            tensor.record_stream(active_stream)
        if weight_fp32 is not None:
            weight_fp32.record_stream(active_stream)
        input_meta = self._tensor_meta(input_tensor, dtype_code=1)
        output_meta = self._tensor_meta(output, dtype_code=1)
        ret = self._lib.gemma4_listener_debug_rmsnorm_native_exact(
            self._ctx,
            ctypes.c_void_p(input_tensor.data_ptr()),
            ctypes.byref(input_meta),
            ctypes.c_void_p(weight_fp32.data_ptr() if weight_fp32 is not None else 0),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.byref(output_meta),
            ctypes.c_uint32(rows),
            ctypes.c_uint32(dim),
            ctypes.c_double(float(eps)),
            ctypes.c_void_p(squared.data_ptr()),
            ctypes.c_void_p(row_sum.data_ptr()),
            ctypes.c_void_p(mean.data_ptr()),
            ctypes.c_void_p(mean_plus_eps.data_ptr()),
            ctypes.c_void_p(scale.data_ptr()),
            ctypes.c_void_p(normalized.data_ptr()),
            ctypes.c_void_p(weighted.data_ptr()),
            ctypes.c_void_p(active_stream.cuda_stream),
        )
        if ret != 0:
            raise RuntimeError(
                f"gemma4_listener_debug_rmsnorm_native_exact failed with code {ret}"
            )
        return {
            "input_bf16": input_tensor,
            "x_squared_fp32": squared,
            "row_sum_fp32": row_sum,
            "row_mean_fp32": mean,
            "mean_plus_eps_fp32": mean_plus_eps,
            "pow_neg_half_fp32": scale,
            "normalized_fp32": normalized,
            "weighted_fp32": weighted,
            "final_bf16": output,
        }

    def debug_linear(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Run the native cuBLAS linear primitive on exact diagnostic inputs."""
        self._require_cuda_contiguous(input_tensor, name="linear input", dtype=torch.bfloat16)
        self._require_cuda_contiguous(weight, name="linear weight", dtype=torch.bfloat16)
        if input_tensor.ndim != 3 or int(input_tensor.shape[0]) != 1 or weight.ndim != 2:
            raise ValueError("linear tensors devono avere shape [1, rows, in] e [out, in]")
        rows, in_dim = (int(input_tensor.shape[1]), int(input_tensor.shape[2]))
        out_dim = int(weight.shape[0])
        if rows <= 0 or rows > self.stream_q_len or tuple(weight.shape) != (out_dim, in_dim):
            raise ValueError("shape linear inattesa")
        active_stream = self._resolve_stream(stream, producer_stream)
        output = torch.empty((1, rows, out_dim), dtype=torch.bfloat16, device=input_tensor.device)
        for tensor in (input_tensor, weight, output):
            tensor.record_stream(active_stream)
        input_meta = self._tensor_meta(input_tensor, dtype_code=1)
        weight_meta = self._tensor_meta(weight, dtype_code=1)
        output_meta = self._tensor_meta(output, dtype_code=1)
        ret = self._lib.gemma4_listener_debug_linear(
            self._ctx,
            ctypes.c_void_p(input_tensor.data_ptr()), ctypes.byref(input_meta),
            ctypes.c_void_p(weight.data_ptr()), ctypes.byref(weight_meta),
            ctypes.c_void_p(output.data_ptr()), ctypes.byref(output_meta),
            ctypes.c_uint32(rows), ctypes.c_uint32(out_dim), ctypes.c_uint32(in_dim),
            ctypes.c_void_p(active_stream.cuda_stream),
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_debug_linear failed with code {ret}")
        return output

    def debug_gelu_tanh(
        self,
        input_tensor: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Run the native GELU-tanh primitive on an exact diagnostic input."""
        self._require_cuda_contiguous(input_tensor, name="GELU input", dtype=torch.bfloat16)
        if input_tensor.ndim != 3 or int(input_tensor.shape[0]) != 1:
            raise ValueError("GELU input deve avere shape [1, rows, dim]")
        rows, dim = (int(input_tensor.shape[1]), int(input_tensor.shape[2]))
        if rows <= 0 or rows > self.stream_q_len or dim <= 0:
            raise ValueError("shape GELU inattesa")
        active_stream = self._resolve_stream(stream, producer_stream)
        output = torch.empty_like(input_tensor)
        input_meta = self._tensor_meta(input_tensor, dtype_code=1)
        output_meta = self._tensor_meta(output, dtype_code=1)
        for tensor in (input_tensor, output):
            tensor.record_stream(active_stream)
        ret = self._lib.gemma4_listener_debug_gelu_tanh(
            self._ctx,
            ctypes.c_void_p(input_tensor.data_ptr()), ctypes.byref(input_meta),
            ctypes.c_void_p(output.data_ptr()), ctypes.byref(output_meta),
            ctypes.c_uint32(rows), ctypes.c_uint32(dim),
            ctypes.c_void_p(active_stream.cuda_stream),
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_debug_gelu_tanh failed with code {ret}")
        return output

    def debug_mul(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Run the native BF16 elementwise multiply on exact inputs."""
        for name, tensor in (("mul lhs", lhs), ("mul rhs", rhs)):
            self._require_cuda_contiguous(tensor, name=name, dtype=torch.bfloat16)
        if lhs.ndim != 3 or tuple(lhs.shape) != tuple(rhs.shape) or int(lhs.shape[0]) != 1:
            raise ValueError("mul tensors devono avere stessa shape [1, rows, dim]")
        rows, dim = (int(lhs.shape[1]), int(lhs.shape[2]))
        if rows <= 0 or rows > self.stream_q_len or dim <= 0:
            raise ValueError("shape mul inattesa")
        active_stream = self._resolve_stream(stream, producer_stream)
        output = torch.empty_like(lhs)
        lhs_meta = self._tensor_meta(lhs, dtype_code=1)
        rhs_meta = self._tensor_meta(rhs, dtype_code=1)
        output_meta = self._tensor_meta(output, dtype_code=1)
        for tensor in (lhs, rhs, output):
            tensor.record_stream(active_stream)
        ret = self._lib.gemma4_listener_debug_mul(
            self._ctx,
            ctypes.c_void_p(lhs.data_ptr()), ctypes.byref(lhs_meta),
            ctypes.c_void_p(rhs.data_ptr()), ctypes.byref(rhs_meta),
            ctypes.c_void_p(output.data_ptr()), ctypes.byref(output_meta),
            ctypes.c_uint32(rows), ctypes.c_uint32(dim),
            ctypes.c_void_p(active_stream.cuda_stream),
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_debug_mul failed with code {ret}")
        return output

    def aten_rmsnorm_parity(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor | None = None,
        *,
        eps: float,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Run the literal installed-PyTorch ATen RMSNorm sequence.

        Diagnostic-only and intentionally allocation-per-call. The C bridge
        borrows the BF16 input/optional-weight/output storage and orders ATen
        work on the caller's CUDA stream. A null weight computes the
        unweighted RMSNorm used by Gemma V norms.
        """
        self._require_cuda_contiguous(input_tensor, name="ATen RMSNorm input", dtype=torch.bfloat16)
        if weight is not None:
            self._require_cuda_contiguous(weight, name="ATen RMSNorm weight", dtype=torch.bfloat16)
            if weight.device != input_tensor.device:
                raise ValueError("ATen RMSNorm input e weight devono usare lo stesso device")
        if input_tensor.ndim != 3 or int(input_tensor.shape[0]) != 1:
            raise ValueError("ATen RMSNorm input deve avere shape [1, rows, dim]")
        rows = int(input_tensor.shape[1])
        dim = int(input_tensor.shape[2])
        if rows <= 0 or rows > self.max_seq_len or (
            weight is not None and tuple(weight.shape) != (dim,)
        ):
            raise ValueError("shape ATen RMSNorm inattesa")
        if not math.isfinite(float(eps)) or float(eps) < 0:
            raise ValueError("eps ATen RMSNorm deve essere finito e non negativo")
        active_stream = self._resolve_stream(stream, producer_stream)
        output = torch.empty_like(input_tensor)
        tensors = (input_tensor, output, weight) if weight is not None else (input_tensor, output)
        for tensor in tensors:
            tensor.record_stream(active_stream)
        ret = self._lib.gemma4_aten_rmsnorm_parity_async(
            ctypes.c_void_p(input_tensor.data_ptr()),
            ctypes.c_void_p(weight.data_ptr() if weight is not None else 0),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_uint32(rows),
            ctypes.c_uint32(dim),
            ctypes.c_double(float(eps)),
            ctypes.c_void_p(active_stream.cuda_stream),
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_aten_rmsnorm_parity_async failed with code {ret}")
        return output

    def debug_attention_native_splitk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        base_position: int,
        prefix_attention: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        scale: float = 1.0,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Run the isolated native split-K GQA primitive (backend ID 3).

        The diagnostic contract uses production q-major Q=[1,4,8,512],
        K/V=[1,1,K,512], causal masking from ``base_position``, and an
        optional canonical int64 prefix mask.  A caller-provided ``output``
        keeps allocation outside timed calls; production execution uses the
        persistent native output arena instead.
        """
        tensors = (("query", query), ("key", key), ("value", value))
        for name, tensor in tensors:
            self._require_cuda_tensor(tensor, name=name, dtype=torch.bfloat16)
            if not tensor.is_contiguous() or tensor.ndim != 4:
                raise ValueError(f"{name} deve essere CUDA BF16 contiguo rank-4")
        if tuple(query.shape) != (1, 4, 8, 512):
            raise ValueError(f"query split-K q-major inatteso: {tuple(query.shape)}")
        if tuple(key.shape) != tuple(value.shape) or int(key.shape[0]) != 1 or int(key.shape[1]) != 1 or int(key.shape[3]) != 512:
            raise ValueError("K/V split-K incompatibili")
        if int(key.shape[2]) <= 2048:
            raise ValueError("split-K richiede K > 2048")
        if not math.isfinite(float(scale)):
            raise ValueError("scale split-K deve essere finito")
        base_position = int(base_position)
        if base_position < 0 or base_position + 4 > self.max_seq_len:
            raise ValueError("base_position split-K fuori capacita")
        prefix_len = 0
        prefix_meta_ptr = None
        prefix_ptr = ctypes.c_void_p(0)
        if prefix_attention is not None:
            self._require_cuda_tensor(prefix_attention, name="prefix_attention", dtype=torch.int64)
            if not prefix_attention.is_contiguous() or prefix_attention.ndim != 2 or int(prefix_attention.shape[0]) != 1:
                raise ValueError("prefix_attention split-K deve essere CUDA int64 [1,prefix_len]")
            prefix_len = int(prefix_attention.shape[1])
            if prefix_len <= 0 or prefix_len > int(key.shape[2]):
                raise ValueError("prefix_attention split-K fuori capacita")
            prefix_meta = self._tensor_meta(prefix_attention, dtype_code=2)
            prefix_meta_ptr = ctypes.byref(prefix_meta)
            prefix_ptr = ctypes.c_void_p(prefix_attention.data_ptr())
        if output is None:
            output = torch.empty_like(query)
        else:
            self._require_cuda_tensor(output, name="output", dtype=torch.bfloat16)
            if not output.is_contiguous() or tuple(output.shape) != tuple(query.shape):
                raise ValueError("output split-K deve essere CUDA BF16 contiguo come query")
        active_stream = self._resolve_stream(stream, producer_stream)
        for tensor in (query, key, value, output):
            tensor.record_stream(active_stream)
        if prefix_attention is not None:
            prefix_attention.record_stream(active_stream)
        query_meta = self._tensor_meta(query, dtype_code=1)
        key_meta = self._tensor_meta(key, dtype_code=1)
        value_meta = self._tensor_meta(value, dtype_code=1)
        output_meta = self._tensor_meta(output, dtype_code=1)
        ret = self._lib.gemma4_listener_debug_attention_splitk(
            self._ctx,
            ctypes.c_void_p(query.data_ptr()), ctypes.byref(query_meta),
            ctypes.c_void_p(key.data_ptr()), ctypes.byref(key_meta),
            ctypes.c_void_p(value.data_ptr()), ctypes.byref(value_meta),
            prefix_ptr, prefix_meta_ptr, ctypes.c_uint32(prefix_len),
            ctypes.c_void_p(output.data_ptr()), ctypes.byref(output_meta),
            ctypes.c_uint32(base_position), ctypes.c_double(float(scale)),
            ctypes.c_void_p(active_stream.cuda_stream),
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_debug_attention_splitk failed with code {ret}")
        self._last_execution_stream = active_stream
        output.record_stream(active_stream)
        return output

    def debug_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor,
        scale: float = 1.0,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
        *,
        backend: int = GEMMA4_ATTN_ATEN_REPEAT_EXACT,
    ) -> torch.Tensor:
        """Run a diagnostic public-ATen SDPA backend on borrowed CUDA tensors.

        The default is the permanent repeat_kv oracle.  ``backend=2`` selects
        the direct-GQA candidate and is intentionally explicit for benchmarks.
        Both paths preserve the caller-provided explicit mask and do not alter
        the native listener position or cache.
        """
        tensors = (("query", query), ("key", key), ("value", value))
        for name, tensor in tensors:
            self._require_cuda_tensor(tensor, name=name, dtype=torch.bfloat16)
            if not tensor.is_contiguous() or tensor.ndim != 4:
                raise ValueError(f"{name} deve essere CUDA BF16 contiguo rank-4")
        if not isinstance(attn_mask, torch.Tensor):
            raise TypeError("attn_mask deve essere un torch.Tensor")
        self._require_cuda_tensor(attn_mask, name="attn_mask", dtype=attn_mask.dtype)
        if attn_mask.dtype not in (torch.bfloat16, torch.float32, torch.bool):
            raise ValueError("attn_mask deve essere BF16, FP32 o BOOL")
        if not attn_mask.is_contiguous() or attn_mask.ndim != 4:
            raise ValueError("attn_mask deve essere contiguo rank-4")
        batch, query_heads, query_len, head_dim = (int(v) for v in query.shape)
        if tuple(int(v) for v in key.shape[:1]) != (batch,) or tuple(int(v) for v in value.shape[:1]) != (batch,):
            raise ValueError("batch Q/K/V incompatibile")
        if tuple(key.shape) != tuple(value.shape) or int(key.shape[3]) != head_dim:
            raise ValueError("shape K/V incompatibile con Q")
        if query_heads % int(key.shape[1]) != 0:
            raise ValueError("Q heads devono essere multipli dei KV heads")
        if tuple(int(v) for v in attn_mask.shape[2:]) != (query_len, int(key.shape[2])):
            raise ValueError("shape attn_mask incompatibile con Q/K")
        if int(attn_mask.shape[0]) not in (1, batch) or int(attn_mask.shape[1]) not in (1, query_heads):
            raise ValueError("attn_mask non broadcastable sulle teste")

        active_stream = self._resolve_stream(stream, producer_stream)
        output = torch.empty_like(query)
        for tensor in (query, key, value, attn_mask, output):
            tensor.record_stream(active_stream)
        query_meta = self._tensor_meta(query, dtype_code=1)
        key_meta = self._tensor_meta(key, dtype_code=1)
        value_meta = self._tensor_meta(value, dtype_code=1)
        mask_dtype_code = {
            torch.bfloat16: 1,
            torch.float32: 3,
            torch.bool: 4,
        }[attn_mask.dtype]
        mask_meta = self._tensor_meta(attn_mask, dtype_code=mask_dtype_code)
        output_meta = self._tensor_meta(output, dtype_code=1)
        if backend == GEMMA4_ATTN_ATEN_REPEAT_EXACT:
            bridge = self._lib.gemma4_listener_debug_attention
        elif backend == GEMMA4_ATTN_ATEN_GQA_EXACT:
            bridge = self._lib.gemma4_listener_debug_attention_gqa
        else:
            raise ValueError(f"backend attenzione diagnostico non supportato: {backend}")
        ret = bridge(
            ctypes.c_void_p(query.data_ptr()),
            ctypes.byref(query_meta),
            ctypes.c_void_p(key.data_ptr()),
            ctypes.byref(key_meta),
            ctypes.c_void_p(value.data_ptr()),
            ctypes.byref(value_meta),
            ctypes.c_void_p(attn_mask.data_ptr()),
            ctypes.byref(mask_meta),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.byref(output_meta),
            ctypes.c_double(float(scale)),
            ctypes.c_void_p(active_stream.cuda_stream),
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_debug_attention failed with code {ret}")
        output.record_stream(active_stream)
        return output

    def debug_attention_gqa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor,
        scale: float = 1.0,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Run the direct-GQA diagnostic backend (ID 2)."""
        return self.debug_attention(
            query, key, value, attn_mask, scale, stream, producer_stream,
            backend=GEMMA4_ATTN_ATEN_GQA_EXACT,
        )

    def prime_async(
        self,
        prefix_embeds: torch.Tensor,
        prefix_ple: torch.Tensor | None = None,
        prefix_attention: torch.Tensor | None = None,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Prime the canonical [BOS|text|BOA] prefix and its exact mask."""
        if prefix_ple is None or prefix_attention is None:
            raise ValueError("prime nativo richiede PLE e prefix_attention espliciti")
        prefix_embeds = self._canonicalize_cuda_tensor(prefix_embeds, name="prefix_embeds", dtype=torch.bfloat16)
        prefix_ple = self._canonicalize_cuda_tensor(prefix_ple, name="prefix_ple", dtype=torch.bfloat16)
        prefix_attention = self._canonicalize_cuda_tensor(prefix_attention, name="prefix_attention", dtype=torch.int64)
        prefix_len, _ = self._validate_prime_inputs(prefix_embeds, prefix_ple, prefix_attention)
        active_stream = self._resolve_stream(stream, producer_stream)
        for tensor in (prefix_embeds, prefix_ple, prefix_attention):
            tensor.record_stream(active_stream)
        c_stream = ctypes.c_void_p(active_stream.cuda_stream)
        p_embeds = ctypes.c_void_p(prefix_embeds.data_ptr())
        p_ple = ctypes.c_void_p(prefix_ple.data_ptr())
        p_attention = ctypes.c_void_p(prefix_attention.data_ptr())

        ret = self._lib.gemma4_listener_prime_async(
            self._ctx,
            p_embeds,
            ctypes.byref(self._prefix_embeds_meta),
            p_ple,
            ctypes.byref(self._prefix_ple_meta),
            p_attention,
            ctypes.byref(self._prefix_attention_meta),
            ctypes.c_uint32(prefix_len),
            c_stream,
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_prime_async failed with code {ret}")
        self._active_prefix_len = prefix_len
        self._mark_graph_capture_pending()

    def prime_debug(
        self,
        prefix_embeds: torch.Tensor,
        prefix_ple: torch.Tensor | None = None,
        prefix_attention: torch.Tensor | None = None,
        num_layers: int = 1,
        apply_final_norm: bool = False,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Prime the prefix for diagnostics using the exact canonical mask."""
        if prefix_ple is None or prefix_attention is None:
            raise ValueError("prime nativo richiede PLE e prefix_attention espliciti")
        prefix_embeds = self._canonicalize_cuda_tensor(prefix_embeds, name="prefix_embeds", dtype=torch.bfloat16)
        prefix_ple = self._canonicalize_cuda_tensor(prefix_ple, name="prefix_ple", dtype=torch.bfloat16)
        prefix_attention = self._canonicalize_cuda_tensor(prefix_attention, name="prefix_attention", dtype=torch.int64)
        prefix_len, _ = self._validate_prime_inputs(prefix_embeds, prefix_ple, prefix_attention)
        active_stream = self._resolve_stream(stream, producer_stream)
        for tensor in (prefix_embeds, prefix_ple, prefix_attention):
            tensor.record_stream(active_stream)
        c_stream = ctypes.c_void_p(active_stream.cuda_stream)
        p_embeds = ctypes.c_void_p(prefix_embeds.data_ptr())
        p_ple = ctypes.c_void_p(prefix_ple.data_ptr())
        p_attention = ctypes.c_void_p(prefix_attention.data_ptr())

        ret = self._lib.gemma4_listener_prime_debug(
            self._ctx,
            p_embeds,
            ctypes.byref(self._prefix_embeds_meta),
            p_ple,
            ctypes.byref(self._prefix_ple_meta),
            p_attention,
            ctypes.byref(self._prefix_attention_meta),
            ctypes.c_uint32(prefix_len),
            ctypes.c_uint32(num_layers),
            ctypes.c_bool(apply_final_norm),
            c_stream,
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_prime_debug failed with code {ret}")
        self._active_prefix_len = prefix_len

    def step_async(
        self,
        frame_audio_soft: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
        *,
        profile_frame_id: int | None = None,
        profile_host_context: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Executes one continuous 80 ms audio frame (Q=4 soft tokens).

        Returns a borrowed persistent output view [1, 4, 1536]. It remains
        valid until the next native invocation or ``close()``; consumers must
        finish using it before advancing the executor again.
        """
        consumer_stream = self._cuda_stream(stream)
        active_stream = self._resolve_stream(stream, producer_stream)
        self._require_cuda_tensor(frame_audio_soft, name="frame_audio_soft", dtype=torch.bfloat16)
        if frame_audio_soft.device != self.output_tensor.device:
            raise ValueError("frame_audio_soft deve risiedere sul device del listener nativo")
        if tuple(frame_audio_soft.shape) != (1, self.stream_q_len, GEMMA4_HIDDEN_SIZE):
            raise ValueError(f"frame_audio_soft inatteso: {tuple(frame_audio_soft.shape)}")
        frame_audio_soft.record_stream(active_stream)
        with torch.cuda.stream(active_stream):
            self._frame_input.copy_(frame_audio_soft, non_blocking=True)
        self._frame_input.record_stream(active_stream)
        c_stream = ctypes.c_void_p(active_stream.cuda_stream)
        p_audio = ctypes.c_void_p(self._frame_input.data_ptr())
        profiler = self._q4_cuda_profiler
        profile_sample = (
            profiler.begin_step(
                active_stream,
                frame_id=profile_frame_id,
                host_context=profile_host_context,
            )
            if profiler is not None else None
        )
        try:
            ret = self._lib.gemma4_listener_step_async(
                self._ctx,
                p_audio,
                self._frame_meta_ptr,
                c_stream,
            )
        except BaseException as exc:
            if profiler is not None:
                profiler.abort_step(profile_sample, exc)
            raise
        if ret != 0:
            exc = RuntimeError(f"gemma4_listener_step_async failed with code {ret}")
            if profiler is not None:
                profiler.abort_step(profile_sample, exc)
            raise exc
        if profiler is not None:
            profiler.finish_step(profile_sample, active_stream)
        self._last_execution_stream = active_stream
        self._record_graph_step_success()
        # Order consumers on their current stream after graph replay. This is
        # a device-side wait; no host synchronization is introduced.
        self.output_tensor.record_stream(active_stream)
        if consumer_stream.cuda_stream != active_stream.cuda_stream:
            consumer_stream.wait_stream(active_stream)
            self.output_tensor.record_stream(consumer_stream)

        return self.output_tensor

    @property
    def q4_cuda_profile_enabled(self) -> bool:
        return self._q4_cuda_profiler is not None

    def native_q4_cuda_profile_summary(
        self,
        *,
        session_id: str,
        runtime_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._q4_cuda_profiler is None:
            return {
                "schema_version": 1,
                "enabled": False,
                "status": "not_requested",
                "session_id": str(session_id),
                "sampling_frequency_frames": _NativeQ4CudaProfiler.sampling_frequency_frames,
                "sample_cap": _NativeQ4CudaProfiler.sample_cap,
                "native_step_call_count": 0,
                "sampled_event_pair_count": 0,
                "completed_sample_count": 0,
                "pending_sample_count": 0,
                "pending_frame_ids": [],
                "pending_samples": [],
                "failure": None,
                "runtime": {
                    key: runtime_info.get(key) if isinstance(runtime_info, dict) else None
                    for key in (
                        "cuda_graph_enabled",
                        "cuda_graph_capture_launches",
                        "cuda_graph_replay_launches",
                        "cuda_graph_step_launches",
                        "attention_backend_id",
                        "attention_backend",
                        "library_sha256",
                    )
                },
                "samples": [],
                "host_join_contract": "no sampled rows when native Q4 profiling is disabled",
            }
        return self._q4_cuda_profiler.summary(session_id=session_id, runtime_info=runtime_info)

    def compact_audio_history(
        self,
        *,
        expected_cached_frames: int,
        retain_frames: int,
        stream: torch.cuda.Stream | None = None,
    ) -> int:
        """Rebase positions while retaining the canonical prompt and recent audio KV.

        Older audio KV entries are removed; retained keys are RoPE-shifted to
        the new local positions. Their hidden-state conditioning is preserved,
        but this is bounded-history approximation, not full-history replay.
        """
        expected_cached_frames = int(expected_cached_frames)
        retain_frames = int(retain_frames)
        current_position = self.current_position
        prefix_len = int(self._active_prefix_len)
        if prefix_len <= 0:
            raise RuntimeError("contesto nativo non ha un prefisso attivo")
        expected_position = prefix_len + expected_cached_frames * AUDIO_FEATURES_PER_FRAME
        if current_position != expected_position:
            raise RuntimeError(
                f"posizione KV nativa inattesa: {current_position} != {expected_position}"
            )
        if not 0 < retain_frames < expected_cached_frames:
            raise ValueError("retain_frames deve essere positivo e minore della cache nativa")
        if retain_frames > NATIVE_REBASE_MAX_AUDIO_FRAMES:
            raise ValueError(
                f"retain_frames oltre il limite nativo ({NATIVE_REBASE_MAX_AUDIO_FRAMES})"
            )
        retained_tokens = retain_frames * AUDIO_FEATURES_PER_FRAME
        active_stream = self._resolve_stream(stream, stream)
        ret = self._lib.gemma4_listener_rebase_async(
            self._ctx,
            ctypes.c_uint32(retained_tokens),
            ctypes.c_void_p(active_stream.cuda_stream),
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_rebase_async failed with code {ret}")
        self._last_execution_stream = active_stream
        expected_after = prefix_len + retained_tokens
        if self.current_position != expected_after:
            raise RuntimeError(
                f"posizione KV nativa dopo rebase inattesa: {self.current_position} != {expected_after}"
            )
        self._mark_graph_capture_pending()
        return expected_cached_frames - retain_frames

    def wait_for_output(self, stream: torch.cuda.Stream | None = None) -> None:
        """Fence the last borrowed output before consuming it on ``stream``.

        Execution is stream-affine and same-stream consumers are already
        ordered. A different CUDA stream gets a device-side wait; no host
        synchronization is performed.
        """
        producer = self._last_execution_stream
        if producer is None:
            return
        consumer = self._cuda_stream(stream)
        if consumer.cuda_stream != producer.cuda_stream:
            consumer.wait_stream(producer)
        self.output_tensor.record_stream(consumer)

    def step_debug(
        self,
        frame_audio_soft: torch.Tensor,
        num_layers: int = 1,
        apply_final_norm: bool = False,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Executes the first num_layers decoder layers for fine-grained debugging/parity checks.

        Returns output_tensor containing the intermediate hidden states after layer num_layers-1.
        """
        active_stream = self._resolve_stream(stream, producer_stream)
        self._require_cuda_tensor(frame_audio_soft, name="frame_audio_soft", dtype=torch.bfloat16)
        if frame_audio_soft.device != self.output_tensor.device:
            raise ValueError("frame_audio_soft deve risiedere sul device del listener nativo")
        if tuple(frame_audio_soft.shape) != (1, self.stream_q_len, GEMMA4_HIDDEN_SIZE):
            raise ValueError(f"frame_audio_soft inatteso: {tuple(frame_audio_soft.shape)}")
        frame_audio_soft.record_stream(active_stream)
        with torch.cuda.stream(active_stream):
            self._frame_input.copy_(frame_audio_soft, non_blocking=True)
        self._frame_input.record_stream(active_stream)
        c_stream = ctypes.c_void_p(active_stream.cuda_stream)
        p_audio = ctypes.c_void_p(self._frame_input.data_ptr())

        ret = self._lib.gemma4_listener_step_debug(
            self._ctx,
            p_audio,
            self._frame_meta_ptr,
            ctypes.c_uint32(num_layers),
            ctypes.c_bool(apply_final_norm),
            c_stream,
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_step_debug failed with code {ret}")
        self._last_execution_stream = active_stream
        self.output_tensor.record_stream(active_stream)

        return self.output_tensor

    def copy_debug_tensor(self, name: str, shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Copies an internal device scratch buffer into a new PyTorch tensor for inspection."""
        ptr = self._lib.gemma4_listener_debug_ptr(self._ctx, name.encode("utf-8"))
        if not ptr:
            raise ValueError(f"Unknown or null debug buffer: {name}")
        out = torch.empty(shape, dtype=dtype, device=self.output_tensor.device)
        libcudart = ctypes.CDLL("libcudart.so")
        ret = libcudart.cudaMemcpy(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(out.numel() * out.element_size()),
            ctypes.c_int(3),  # cudaMemcpyDeviceToDevice
        )
        if ret != 0:
            raise RuntimeError(f"cudaMemcpy failed with code {ret}")
        return out

    def get_cached_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Copies the cached K and V tensors for layer_idx up to current_position."""
        if not 0 <= int(layer_idx) < GEMMA4_NUM_LAYERS:
            raise ValueError(f"layer_idx inatteso: {layer_idx}")
        k_ptr = self._lib.gemma4_listener_get_cached_k(self._ctx, ctypes.c_int(layer_idx))
        v_ptr = self._lib.gemma4_listener_get_cached_v(self._ctx, ctypes.c_int(layer_idx))
        head_dim = 512 if self.language_model.config.layer_types[layer_idx] == "full_attention" else 256
        seq_len = self.current_position
        if seq_len > 0 and (not k_ptr or not v_ptr):
            raise RuntimeError("cache K/V nativa non disponibile")
        k_out = torch.empty((1, 1, seq_len, head_dim), dtype=torch.bfloat16, device=self.output_tensor.device)
        v_out = torch.empty((1, 1, seq_len, head_dim), dtype=torch.bfloat16, device=self.output_tensor.device)
        libcudart = ctypes.CDLL("libcudart.so")
        k_ret = libcudart.cudaMemcpy(
            ctypes.c_void_p(k_out.data_ptr()),
            ctypes.c_void_p(k_ptr),
            ctypes.c_size_t(seq_len * head_dim * 2),
            ctypes.c_int(3),
        )
        if k_ret != 0:
            raise RuntimeError(f"cudaMemcpy K failed with code {k_ret}")
        v_ret = libcudart.cudaMemcpy(
            ctypes.c_void_p(v_out.data_ptr()),
            ctypes.c_void_p(v_ptr),
            ctypes.c_size_t(seq_len * head_dim * 2),
            ctypes.c_int(3),
        )
        if v_ret != 0:
            raise RuntimeError(f"cudaMemcpy V failed with code {v_ret}")
        return k_out, v_out

    def reset(self) -> None:
        """Reset the KV position and invalidate the primed prefix."""
        ret = self._lib.gemma4_listener_reset(self._ctx)
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_reset failed with code {ret}")
        self._cache_import_tensors = []
        self._active_prefix_len = 0
        self._last_execution_stream = None
        self._mark_graph_capture_pending()

    def verify_output_health(self) -> bool:
        """Synchronously validate the last output; never call in steady-state."""
        status = ctypes.c_int(0)
        rc = int(self._lib.gemma4_listener_output_health(self._ctx, ctypes.byref(status)))
        if rc < 0:
            raise RuntimeError(f"health check output nativo fallito: rc={rc}")
        if rc != 0 or int(status.value) != 0:
            raise RuntimeError("output nativo non finito; contesto marcato poisoned")
        return True

    @property
    def current_position(self) -> int:
        return int(self._lib.gemma4_listener_runtime_current_position(self._ctx))

    def native_runtime_info(self) -> dict[str, int | str | bool | None | list[str]]:
        """Return backend/capacity facts queried from the live native context."""
        attention = int(self._lib.gemma4_listener_runtime_attention_backend(self._ctx))
        rmsnorm = int(self._lib.gemma4_listener_runtime_rmsnorm_backend(self._ctx))
        attention_names = {
            GEMMA4_ATTN_CUSTOM: "CUSTOM",
            GEMMA4_ATTN_ATEN_REPEAT_EXACT: "ATEN_REPEAT_EXACT",
            GEMMA4_ATTN_ATEN_GQA_EXACT: "ATEN_GQA_EXACT",
            GEMMA4_ATTN_NATIVE_SPLITK_GQA: "NATIVE_SPLITK_GQA",
        }
        rmsnorm_names = {0: "CUSTOM_FAST", 1: "ATEN_EXACT", 2: "NATIVE_EXACT"}
        if attention not in attention_names or rmsnorm not in rmsnorm_names:
            raise RuntimeError(
                f"backend nativo sconosciuto: attention={attention}, rmsnorm={rmsnorm}"
            )
        splitk_layer_mask = int(self._lib.gemma4_listener_runtime_splitk_layer_mask(self._ctx))
        splitk_layer_mask_layers = splitk_layer_mask_from_bits(splitk_layer_mask)
        implementation_raw = self._lib.gemma4_listener_runtime_attention_implementation(self._ctx)
        implementation = implementation_raw.decode("ascii") if implementation_raw else None
        if splitk_layer_mask_layers == GEMMA4_SPLITK_STATE_SAFE_HYBRID_LAYERS:
            splitk_layer_policy = "STATE_SAFE_HYBRID"
        elif splitk_layer_mask_layers == GEMMA4_FULL_ATTENTION_LAYERS:
            splitk_layer_policy = "ALL_FULL"
        else:
            splitk_layer_policy = ",".join(str(layer) for layer in splitk_layer_mask_layers) or "NONE"
        ordered_calls_by_layer = {
            str(layer): int(self._lib.gemma4_listener_runtime_ordered_calls_by_layer(
                self._ctx, ctypes.c_uint32(layer)
            ))
            for layer in range(GEMMA4_NUM_LAYERS)
        }
        library_path = Path(self._lib_path)
        try:
            library_sha256 = hashlib.sha256(library_path.read_bytes()).hexdigest()
        except OSError:
            library_sha256 = None
        return {
            "enabled": True,
            "native_q4_cuda_profile_enabled": self._q4_cuda_profiler is not None,
            "cuda_graph_enabled": self.enable_cuda_graph,
            "cuda_graph_stream_non_default": bool(
                self._graph_stream is not None and int(self._graph_stream.cuda_stream) != 0
            ),
            "cuda_graph_capture_pending": bool(self._graph_capture_pending),
            "cuda_graph_capture_launches": int(self._graph_capture_launches),
            "cuda_graph_replay_launches": int(self._graph_replay_launches),
            "cuda_graph_step_launches": int(self._graph_step_launches),
            "library_path": str(library_path),
            "library_sha256": library_sha256,
            "manifest_sha256": str(getattr(self, "_manifest_sha256", "")),
            "manifest_path": str(getattr(self, "_manifest_path", "")),
            "checkpoint_path": str(getattr(self.language_model, "_gemma4_checkpoint_path", "")),
            "checkpoint_files_sha256": str(getattr(self, "_checkpoint_files_sha256", "")),
            "attention_backend_id": attention,
            "attention_backend": attention_names[attention],
            "attention_backend_source": self._attention_backend_source,
            "attention_backend_env_present": self._attention_backend_env_present,
            "attention_implementation": implementation,
            "ordered_key_tile_size": int(self._lib.gemma4_listener_runtime_ordered_key_tile_size(self._ctx)),
            "ordered_stage_sequence": list(GEMMA4_ORDERED_STAGE_SEQUENCE) if implementation else [],
            "ordered_q_packing": GEMMA4_ORDERED_Q_PACKING if implementation else None,
            "ordered_min_keys_exclusive": int(self._lib.gemma4_listener_runtime_ordered_min_keys_exclusive(self._ctx)),
            "ordered_workspace_bytes": int(self._lib.gemma4_listener_runtime_ordered_workspace_bytes(self._ctx)),
            "ordered_pack_launches": int(self._lib.gemma4_listener_runtime_ordered_pack_launches(self._ctx)),
            "ordered_stage_a_launches": int(self._lib.gemma4_listener_runtime_ordered_stage_a_launches(self._ctx)),
            "ordered_stage_b_launches": int(self._lib.gemma4_listener_runtime_ordered_stage_b_launches(self._ctx)),
            "ordered_stage_c_launches": int(self._lib.gemma4_listener_runtime_ordered_stage_c_launches(self._ctx)),
            "ordered_stage_d_launches": int(self._lib.gemma4_listener_runtime_ordered_stage_d_launches(self._ctx)),
            "ordered_calls_by_layer": ordered_calls_by_layer,
            "splitk_layer_mask": splitk_layer_mask,
            "splitk_layer_mask_layers": splitk_layer_mask_layers,
            "splitk_layer_policy": splitk_layer_policy,
            "rmsnorm_backend_id": rmsnorm,
            "rmsnorm_backend": rmsnorm_names[rmsnorm],
            "max_seq_len": int(self._lib.gemma4_listener_runtime_max_seq_len(self._ctx)),
            "current_position": int(self._lib.gemma4_listener_runtime_current_position(self._ctx)),
            "derived_weight_bytes": int(self._lib.gemma4_listener_runtime_derived_weight_bytes(self._ctx)),
        }

    @property
    def runtime_info(self) -> dict[str, int | str | bool | None | list[str]]:
        return self.native_runtime_info()

    def close(self) -> None:
        """Release native storage after draining the executor stream."""
        ctx = getattr(self, "_ctx", None)
        if ctx:
            self._lib.gemma4_listener_destroy(ctx)
            self._ctx = None
        self._pinned_tensors.clear()
        self._cache_import_tensors.clear()
        self._last_execution_stream = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors must not mask interpreter shutdown or a prior CUDA
            # failure. The C destructor still performs best-effort draining.
            self._ctx = None


__all__ = [
    "NativeGemmaListenerLM",
    "Gemma4TensorMeta",
    "Gemma4ListenerConfig",
    "Gemma4ListenerWeights",
    "Gemma4ImportedKV",
    "Gemma4ImportedCache",
    "GEMMA4_ATTN_CUSTOM",
    "GEMMA4_ATTN_ATEN_REPEAT_EXACT",
    "GEMMA4_ATTN_ATEN_GQA_EXACT",
    "GEMMA4_ATTN_NATIVE_SPLITK_GQA",
    "GEMMA4_ATTN_ATEN_EXACT",
    "GEMMA4_RMS_CUSTOM_FAST",
    "GEMMA4_RMS_ATEN_EXACT",
    "GEMMA4_RMS_NATIVE_EXACT",
    "GEMMA4_ORDERED_ATTENTION_IMPLEMENTATION",
    "GEMMA4_ORDERED_KEY_TILE_SIZE",
    "GEMMA4_ORDERED_ACTIVATION_MIN_KEYS_EXCLUSIVE",
    "GEMMA4_ORDERED_REDUCTION_ORDER",
    "GEMMA4_ORDERED_STAGE_SEQUENCE",
    "GEMMA4_ORDERED_Q_PACKING",
    "GEMMA4_ORDERED_LAYER_MASK",
    "GEMMA4_FULL_ATTENTION_LAYERS",
    "attention_implementation_metadata",
    "GEMMA4_SPLITK_STATE_SAFE_HYBRID_LAYERS",
    "parse_attention_backend",
    "parse_splitk_layer_mask",
    "splitk_layer_mask_from_bits",
    "attention_backend_from_env",
    "resolve_native_attention_backend",
    "parse_rmsnorm_backend",
    "assert_production_native_runtime",
    "attention_backend_name",
]
