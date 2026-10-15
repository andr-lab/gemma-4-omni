"""Python adapter for the opt-in native Qwen3-TTS Talker backbone.

The wrapper owns orchestration and borrowed-tensor lifetimes. The native
executor supports both the established joint Q=2 slot-KV path and the
recurrent audio-only Q1 path from an eager DynamicCache prefill. Codec-head
sampling and SubTalker remain external to both paths.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import torch

from duplex_kernel.future_prefix import _SlotKVCache
from duplex_kernel.native_trace import (
    NATIVE_TRACE_ABI_VERSION,
    NATIVE_TRACE_ATTESTATION_SOURCE,
    NATIVE_TRACE_HASH_ALGORITHM,
    NATIVE_TRACE_NATIVE_CHAIN_ALGORITHM,
    NATIVE_TRACE_SCHEMA_VERSION,
    tensor_digest,
)

logger = logging.getLogger("duplex_kernel.native_qwen_talker")

QWEN_NATIVE_NUM_LAYERS = 28
QWEN_NATIVE_HIDDEN_SIZE = 2048
QWEN_NATIVE_NUM_ATTN_HEADS = 16
QWEN_NATIVE_NUM_KV_HEADS = 8
QWEN_NATIVE_HEAD_DIM = 128
QWEN_NATIVE_META_BF16 = 1
QWEN_NATIVE_META_FP32 = 3
QWEN_NATIVE_META_INT64 = 4
QWEN_NATIVE_GLOBAL_WEIGHT_META_COUNT = 2
QWEN_NATIVE_LAYER_WEIGHT_META_COUNT = 11
QWEN_NATIVE_WEIGHT_META_COUNT = QWEN_NATIVE_GLOBAL_WEIGHT_META_COUNT + QWEN_NATIVE_NUM_LAYERS * QWEN_NATIVE_LAYER_WEIGHT_META_COUNT


class QwenTensorMeta(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("dtype", ctypes.c_uint32),
        ("ndim", ctypes.c_uint32),
        ("contiguous", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("dims", ctypes.c_uint64 * 4),
        ("strides", ctypes.c_uint64 * 4),
        ("capacity_bytes", ctypes.c_uint64),
    ]


class QwenLayerWeights(ctypes.Structure):
    _fields_ = [
        ("input_norm_w", ctypes.c_void_p),
        ("post_attention_norm_w", ctypes.c_void_p),
        ("q_proj_w", ctypes.c_void_p),
        ("k_proj_w", ctypes.c_void_p),
        ("v_proj_w", ctypes.c_void_p),
        ("o_proj_w", ctypes.c_void_p),
        ("q_norm_w", ctypes.c_void_p),
        ("k_norm_w", ctypes.c_void_p),
        ("gate_proj_w", ctypes.c_void_p),
        ("up_proj_w", ctypes.c_void_p),
        ("down_proj_w", ctypes.c_void_p),
    ]


class QwenWeights(ctypes.Structure):
    _fields_ = [
        ("final_norm_w", ctypes.c_void_p),
        ("rotary_inv_freq", ctypes.c_void_p),
        ("layers", QwenLayerWeights * QWEN_NATIVE_NUM_LAYERS),
    ]


class QwenConfig(ctypes.Structure):
    _fields_ = [
        ("max_cache_len", ctypes.c_uint32),
        ("execute_layers", ctypes.c_uint32),
        ("weight_meta", ctypes.POINTER(QwenTensorMeta)),
        ("weight_meta_count", ctypes.c_uint32),
    ]


class QwenImportedKV(ctypes.Structure):
    _fields_ = [
        ("k_ptr", ctypes.c_void_p),
        ("k_meta", ctypes.POINTER(QwenTensorMeta)),
        ("v_ptr", ctypes.c_void_p),
        ("v_meta", ctypes.POINTER(QwenTensorMeta)),
    ]


class QwenImportedCache(ctypes.Structure):
    _fields_ = [("layers", QwenImportedKV * QWEN_NATIVE_NUM_LAYERS)]


class QwenTraceStep(ctypes.Structure):
    _fields_ = [
        ("step_index", ctypes.c_uint64),
        ("trace_generation", ctypes.c_uint64),
        ("expected_step_count", ctypes.c_uint64),
        ("completed_step_count", ctypes.c_uint64),
        ("stable_length", ctypes.c_uint32),
        ("prefix_start", ctypes.c_uint32),
        ("audio_start", ctypes.c_uint32),
        ("prefix_position", ctypes.c_uint32),
        ("audio_position", ctypes.c_uint32),
        ("cache_layers_written", ctypes.c_uint32),
        ("trace_enabled", ctypes.c_uint32),
        ("native_complete", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("joint_steps", ctypes.c_uint64),
        ("cache_imports", ctypes.c_uint64),
        ("cuda_error_state", ctypes.c_int32),
        ("native_previous_chain", ctypes.c_uint8 * 32),
        ("native_chain", ctypes.c_uint8 * 32),
    ]


def _find_library() -> str:
    configured = os.environ.get("QWEN_NATIVE_LIBRARY_PATH")
    candidates = tuple(
        Path(value)
        for value in (configured,) if value
    ) + (
        Path("/workspace/native/qwen_talker/libqwen_talker.so"),
        Path(__file__).resolve().parents[1] / "native" / "qwen_talker" / "libqwen_talker.so",
        Path("native/qwen_talker/libqwen_talker.so").resolve(),
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("libqwen_talker.so non trovata; eseguire make in native/qwen_talker")


def _compiled_qwen_contract(header_path: Path | None = None) -> dict[str, Any]:
    """Read the native ABI dimensions from its generated compile-time header.

    Runtime binding uses this small source contract plus the loaded model's
    actual config and tensors. It deliberately does not require a separate
    topology manifest or a precomputed checkpoint attestation.
    """
    candidates = (header_path,) if header_path is not None else (
        Path("/workspace/native/qwen_talker/generated/qwen_native_topology.h"),
        Path(__file__).resolve().parents[1] / "native/qwen_talker/generated/qwen_native_topology.h",
    )
    path = next((item for item in candidates if item is not None and item.is_file()), None)
    if path is None:
        raise FileNotFoundError("header ABI Qwen nativa generated/qwen_native_topology.h non trovato")
    source = path.read_text(encoding="utf-8")

    def integer(name: str) -> int:
        match = re.search(rf"^#define\s+{re.escape(name)}\s+(\d+)u?\s*$", source, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"header ABI Qwen privo di {name}")
        return int(match.group(1))

    def scalar(name: str) -> float:
        match = re.search(
            rf"^static constexpr float {re.escape(name)}\s*=\s*([-+0-9.eE]+)f;\s*$",
            source,
            re.MULTILINE,
        )
        if match is None:
            raise RuntimeError(f"header ABI Qwen privo di {name}")
        return float(match.group(1))

    contract = {
        "num_hidden_layers": integer("QWEN_NATIVE_NUM_LAYERS"),
        "hidden_size": integer("QWEN_NATIVE_HIDDEN_SIZE"),
        "intermediate_size": integer("QWEN_NATIVE_INTERMEDIATE_SIZE"),
        "num_attention_heads": integer("QWEN_NATIVE_NUM_ATTN_HEADS"),
        "num_key_value_heads": integer("QWEN_NATIVE_NUM_KV_HEADS"),
        "num_key_value_groups": integer("QWEN_NATIVE_NUM_KV_GROUPS"),
        "head_dim": integer("QWEN_NATIVE_HEAD_DIM"),
        "max_position_embeddings": integer("QWEN_NATIVE_MAX_POSITION_EMBEDDINGS"),
        "codec_vocab_size": integer("QWEN_NATIVE_CODEC_VOCAB_SIZE"),
        "rope_interleaved": bool(integer("QWEN_NATIVE_ROPE_INTERLEAVED")),
        "mrope_section": [
            integer("QWEN_NATIVE_MROPE_SECTION_0"),
            integer("QWEN_NATIVE_MROPE_SECTION_1"),
            integer("QWEN_NATIVE_MROPE_SECTION_2"),
        ],
        "hidden_act_silu": bool(integer("QWEN_NATIVE_HIDDEN_ACT_SILU")),
        "rms_norm_eps": scalar("QWEN_NATIVE_RMS_NORM_EPS"),
        "rope_theta": scalar("QWEN_NATIVE_ROPE_THETA"),
        "attention_scale": scalar("QWEN_NATIVE_ATTENTION_SCALE"),
    }
    python_contract = {
        "num_hidden_layers": QWEN_NATIVE_NUM_LAYERS,
        "hidden_size": QWEN_NATIVE_HIDDEN_SIZE,
        "num_attention_heads": QWEN_NATIVE_NUM_ATTN_HEADS,
        "num_key_value_heads": QWEN_NATIVE_NUM_KV_HEADS,
        "head_dim": QWEN_NATIVE_HEAD_DIM,
    }
    for name, expected in python_contract.items():
        if contract[name] != expected:
            raise RuntimeError(
                f"contratto Python/ABI Qwen incoerente per {name}: {expected} != {contract[name]}"
            )
    if contract["num_attention_heads"] // contract["num_key_value_heads"] != contract["num_key_value_groups"]:
        raise RuntimeError("gruppi KV nel header ABI Qwen non coerenti con le teste")
    return contract


def _validate_qwen_runtime_config(config: Any, contract: dict[str, Any]) -> dict[str, Any]:
    """Check loaded config against compile-time ABI dimensions and constants."""
    config_values = {
        "num_hidden_layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "num_key_value_groups": int(config.num_attention_heads) // int(config.num_key_value_heads),
        "head_dim": int(getattr(config, "head_dim", int(config.hidden_size) // int(config.num_attention_heads))),
        "max_position_embeddings": int(config.max_position_embeddings),
        "codec_vocab_size": int(config.vocab_size),
        "hidden_act_silu": str(config.hidden_act).lower() == "silu",
        "rms_norm_eps": float(config.rms_norm_eps),
        "rope_theta": float(config.rope_theta),
    }
    rope_config = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    config_values["rope_interleaved"] = bool(rope_config.get("interleaved", False))
    config_values["mrope_section"] = [int(value) for value in rope_config.get("mrope_section", [])]
    config_values["attention_scale"] = config_values["head_dim"] ** -0.5

    for name in (
        "num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
        "num_key_value_heads", "head_dim", "max_position_embeddings", "codec_vocab_size",
        "num_key_value_groups",
        "rope_interleaved", "mrope_section", "hidden_act_silu",
    ):
        if config_values[name] != contract[name]:
            raise ValueError(
                f"config Qwen incompatibile con ABI compilata per {name}: "
                f"{config_values[name]} != {contract[name]}"
            )
    for name in ("rms_norm_eps", "rope_theta", "attention_scale"):
        if not math.isclose(config_values[name], contract[name], rel_tol=1e-7, abs_tol=1e-12):
            raise ValueError(
                f"config Qwen incompatibile con ABI compilata per {name}: "
                f"{config_values[name]} != {contract[name]}"
            )
    return config_values


def _runtime_topology(model: Any) -> dict[str, Any]:
    """Build validated weight descriptors directly from the loaded Qwen model."""
    contract = _compiled_qwen_contract()
    config_values = _validate_qwen_runtime_config(model.config, contract)
    hidden = contract["hidden_size"]
    intermediate = contract["intermediate_size"]
    kv_width = contract["num_key_value_heads"] * contract["head_dim"]
    expected_dtype = torch.bfloat16

    def descriptor(name: str, tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> dict[str, Any]:
        actual_shape = tuple(int(value) for value in tensor.shape)
        if actual_shape != shape:
            raise ValueError(f"shape Qwen inattesa per {name}: {actual_shape} != {shape}")
        if tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError(f"dtype/contig Qwen inattesi per {name}: {tensor.dtype}")
        return {
            "name": name,
            "shape": list(actual_shape),
            "stride": [int(value) for value in tensor.stride()],
            "dtype": str(tensor.dtype),
        }

    global_descriptors = [
        descriptor("model.norm.weight", model.norm.weight, (hidden,), expected_dtype),
        descriptor(
            "model.rotary_emb.inv_freq", model.rotary_emb.inv_freq,
            (contract["head_dim"] // 2,), expected_dtype,
        ),
    ]
    layer_descriptors = []
    if len(model.layers) != contract["num_hidden_layers"]:
        raise ValueError(
            f"numero layer Qwen inatteso: {len(model.layers)} != {contract['num_hidden_layers']}"
        )
    for layer_index, layer in enumerate(model.layers):
        prefix = f"model.layers.{layer_index}."
        weights = [
            descriptor(prefix + "input_layernorm.weight", layer.input_layernorm.weight, (hidden,), expected_dtype),
            descriptor(prefix + "post_attention_layernorm.weight", layer.post_attention_layernorm.weight, (hidden,), expected_dtype),
            descriptor(prefix + "self_attn.q_proj.weight", layer.self_attn.q_proj.weight, (hidden, hidden), expected_dtype),
            descriptor(prefix + "self_attn.k_proj.weight", layer.self_attn.k_proj.weight, (kv_width, hidden), expected_dtype),
            descriptor(prefix + "self_attn.v_proj.weight", layer.self_attn.v_proj.weight, (kv_width, hidden), expected_dtype),
            descriptor(prefix + "self_attn.o_proj.weight", layer.self_attn.o_proj.weight, (hidden, hidden), expected_dtype),
            descriptor(prefix + "self_attn.q_norm.weight", layer.self_attn.q_norm.weight, (contract["head_dim"],), expected_dtype),
            descriptor(prefix + "self_attn.k_norm.weight", layer.self_attn.k_norm.weight, (contract["head_dim"],), expected_dtype),
            descriptor(prefix + "mlp.gate_proj.weight", layer.mlp.gate_proj.weight, (intermediate, hidden), expected_dtype),
            descriptor(prefix + "mlp.up_proj.weight", layer.mlp.up_proj.weight, (intermediate, hidden), expected_dtype),
            descriptor(prefix + "mlp.down_proj.weight", layer.mlp.down_proj.weight, (hidden, intermediate), expected_dtype),
        ]
        layer_descriptors.append({"layer_index": layer_index, "weights": weights})

    runtime_contract = {
        "schema": "QWEN_NATIVE_RUNTIME_CONTRACT_V1",
        "config": config_values,
        "layer_descriptors": layer_descriptors,
        "global_descriptors": global_descriptors,
    }
    runtime_contract["topology_sha256"] = hashlib.sha256(
        json.dumps(runtime_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return runtime_contract


def _meta(tensor: torch.Tensor) -> QwenTensorMeta:
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
        raise ValueError("il binding Qwen richiede tensor CUDA")
    dtype_code = {
        torch.bfloat16: QWEN_NATIVE_META_BF16,
        torch.float32: QWEN_NATIVE_META_FP32,
        torch.int64: QWEN_NATIVE_META_INT64,
    }.get(tensor.dtype)
    if dtype_code is None:
        raise ValueError(f"dtype Qwen inatteso: {tensor.dtype}")
    if not tensor.is_contiguous() or tensor.ndim > 4 or tensor.ndim == 0:
        raise ValueError(f"tensor Qwen deve essere contiguo rank 1..4: {tuple(tensor.shape)}")
    expected_strides: list[int] = []
    running = 1
    for size in reversed(tensor.shape):
        expected_strides.append(running)
        running *= int(size)
    if tuple(reversed(expected_strides)) != tuple(int(value) for value in tensor.stride()):
        raise ValueError(f"tensor Qwen deve avere stride compatto: {tuple(tensor.stride())}")
    result = QwenTensorMeta()
    result.ptr = int(tensor.data_ptr())
    result.dtype = dtype_code
    result.ndim = tensor.ndim
    result.contiguous = 1
    for index, value in enumerate(tensor.shape):
        result.dims[index] = int(value)
        result.strides[index] = int(tensor.stride()[index])
    result.capacity_bytes = int(tensor.numel() * tensor.element_size())
    return result


def _dynamic_cache_kv_layers(cache: Any) -> tuple[int, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Return the exact HF DynamicCache KV tensors without casting or cloning.

    This adapter intentionally accepts the public ``layers``/``keys``/``values``
    shape exposed by Transformers DynamicCache while keeping validation usable
    with CPU fixtures. The executor applies CUDA/device checks before borrowing
    any pointer.
    """
    layers = getattr(cache, "layers", None)
    get_seq_length = getattr(cache, "get_seq_length", None)
    if not isinstance(layers, (list, tuple)) or not callable(get_seq_length):
        raise ValueError("cache Qwen deve esporre DynamicCache.layers e get_seq_length()")
    if len(layers) != QWEN_NATIVE_NUM_LAYERS:
        raise ValueError(
            f"DynamicCache Qwen deve contenere {QWEN_NATIVE_NUM_LAYERS} layer, ricevuti {len(layers)}"
        )
    try:
        sequence_length = int(get_seq_length())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("DynamicCache Qwen ha una lunghezza sequenza non valida") from exc
    if sequence_length <= 0:
        raise ValueError("DynamicCache Qwen deve contenere almeno un token prefissato")

    result: list[tuple[torch.Tensor, torch.Tensor]] = []
    expected_shape = (1, QWEN_NATIVE_NUM_KV_HEADS, sequence_length, QWEN_NATIVE_HEAD_DIM)
    for index, layer in enumerate(layers):
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError(f"DynamicCache layer {index} non contiene tensori K/V")
        if tuple(key.shape) != expected_shape or tuple(value.shape) != expected_shape:
            raise ValueError(
                f"shape DynamicCache inattesa al layer {index}: "
                f"K={tuple(key.shape)} V={tuple(value.shape)} attesa={expected_shape}"
            )
        if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise ValueError(f"DynamicCache layer {index} deve essere BF16")
        if key.device != value.device:
            raise ValueError(f"DynamicCache K/V su device diversi al layer {index}")
        if not key.is_contiguous() or not value.is_contiguous():
            raise ValueError(f"DynamicCache layer {index} deve essere contiguo")
        result.append((key, value))
    return sequence_length, result


def _advance_audio_q1_position(current_position: int, limit_position: int) -> int:
    """Return the next causal Q1 position, rejecting use past the response budget."""
    current_position = int(current_position)
    limit_position = int(limit_position)
    if current_position < 0 or limit_position <= current_position:
        raise RuntimeError("limite frame audio della risposta raggiunto")
    return current_position + 1


def _stream_handle(device: torch.device) -> int:
    return int(torch.cuda.current_stream(device=device).cuda_stream)


class NativeQwenTalker:
    """Opt-in native Qwen decoder with borrowed PyTorch weight storage.

    ``last_hidden`` and ``debug_hidden`` are persistent borrowed outputs:
    callers must consume them before the next executor step/debug call.
    """

    def __init__(
        self,
        talker: Any,
        *,
        max_cache_len: int,
        execute_layers: int = QWEN_NATIVE_NUM_LAYERS,
        trace_enabled: bool = False,
    ) -> None:
        self.talker = talker
        self.model = getattr(talker, "model", talker)
        self.max_cache_len = int(max_cache_len)
        self.execute_layers = int(execute_layers)
        if self.execute_layers <= 0 or self.execute_layers > QWEN_NATIVE_NUM_LAYERS:
            raise ValueError("execute_layers Qwen fuori intervallo")
        first = next(self.model.parameters(), None)
        if first is None or first.device.type != "cuda":
            raise ValueError("Qwen native richiede pesi CUDA")
        self.device = first.device
        self._topology = _runtime_topology(self.model)
        self._trace_enabled = bool(trace_enabled)
        self._library_path = _find_library()
        self._library_sha256 = hashlib.sha256(Path(self._library_path).read_bytes()).hexdigest()
        self._weight_refs: list[torch.Tensor] = []
        self._weight_meta_refs: list[QwenTensorMeta] = []
        self._weights = self._bind_weights()
        self._weight_meta_array = (QwenTensorMeta * len(self._weight_meta_refs))(*self._weight_meta_refs)
        self._config = QwenConfig(
            max_cache_len=self.max_cache_len,
            execute_layers=self.execute_layers,
            weight_meta=self._weight_meta_array,
            weight_meta_count=len(self._weight_meta_refs),
        )
        self._lib = ctypes.CDLL(self._library_path)
        self._setup_signatures()
        self._ctx = ctypes.c_void_p()
        status = self._lib.qwen_talker_create(
            ctypes.byref(self._config), ctypes.byref(self._weights), ctypes.byref(self._ctx)
        )
        if status != 0 or not self._ctx:
            raise RuntimeError(f"qwen_talker_create failed: {status}")
        self.last_hidden = torch.empty((1, 1, QWEN_NATIVE_HIDDEN_SIZE), device=self.device, dtype=torch.bfloat16)
        self._q1_graph_device_state = torch.empty((2,), device=self.device, dtype=torch.int32)
        self._q1_graph_last_device_cursor: int | None = None
        self._q1_graph_last_device_error = 0
        self._q1_graph_state_dirty = False
        self.debug_hidden = torch.empty((1, 2, QWEN_NATIVE_HIDDEN_SIZE), device=self.device, dtype=torch.bfloat16)
        self._cache_refs: list[torch.Tensor] = []
        self._audio_q1_imported = False
        self._audio_q1_next_position: int | None = None
        self._audio_q1_limit_position: int | None = None
        self._joint_prefix_start: int | None = None
        self._joint_audio_start: int | None = None
        self._joint_next_prefix: int | None = None
        self._joint_next_audio: int | None = None
        self._closed = False
        # Whole-step plain-Q1 CUDA graph state. One NativeQwenTalker owns one
        # pool slot and one persistent graph; the native cache and static
        # buffers remain at stable addresses across response imports.
        self._q1_graph_stream: torch.cuda.Stream | None = None
        self._q1_cuda_graph: torch.cuda.CUDAGraph | None = None
        self._q1_graph_input: torch.Tensor | None = None
        self._q1_graph_event: torch.cuda.Event | None = None
        self._q1_graph_capacity: int | None = None
        self._q1_graph_max_audio_frames: int | None = None
        self._q1_graph_weight_signature: tuple[tuple[Any, ...], ...] | None = None
        self._q1_graph_capture_count = 0
        self._q1_graph_failure: str | None = None
        if self._trace_enabled:
            self.set_trace_enabled(True)

    def _setup_signatures(self) -> None:
        self._lib.qwen_talker_abi_version.restype = ctypes.c_uint32
        self._native_abi_version = int(self._lib.qwen_talker_abi_version())
        if self._native_abi_version not in (3, 4):
            raise RuntimeError("ABI Qwen nativa incompatibile")
        self._lib.qwen_talker_create.restype = ctypes.c_int
        self._lib.qwen_talker_create.argtypes = [
            ctypes.POINTER(QwenConfig), ctypes.POINTER(QwenWeights), ctypes.POINTER(ctypes.c_void_p)
        ]
        self._lib.qwen_talker_import_cache_async.restype = ctypes.c_int
        self._lib.qwen_talker_import_cache_async.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(QwenImportedCache), ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
        ]
        self._lib.qwen_talker_import_audio_cache_async.restype = ctypes.c_int
        self._lib.qwen_talker_import_audio_cache_async.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(QwenImportedCache), ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_void_p,
        ]
        self._lib.qwen_talker_step_joint_async.restype = ctypes.c_int
        self._lib.qwen_talker_step_joint_async.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta), ctypes.c_void_p,
            ctypes.POINTER(QwenTensorMeta), ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_float, ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta), ctypes.c_void_p,
        ]
        self._lib.qwen_talker_step_terminal_async.restype = ctypes.c_int
        self._lib.qwen_talker_step_terminal_async.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta),
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_float, ctypes.c_void_p,
            ctypes.POINTER(QwenTensorMeta), ctypes.c_void_p,
        ]
        self._lib.qwen_talker_step_audio_async.restype = ctypes.c_int
        self._lib.qwen_talker_step_audio_async.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta),
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float,
            ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta), ctypes.c_void_p,
        ]
        if self._native_abi_version >= 4:
            self._lib.qwen_talker_q1_graph_bind_stream.restype = ctypes.c_int
            self._lib.qwen_talker_q1_graph_bind_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self._lib.qwen_talker_q1_graph_capture_begin.restype = ctypes.c_int
            self._lib.qwen_talker_q1_graph_capture_begin.argtypes = [ctypes.c_void_p]
            self._lib.qwen_talker_q1_graph_capture_end.restype = ctypes.c_int
            self._lib.qwen_talker_q1_graph_capture_end.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            self._lib.qwen_talker_q1_graph_can_replay.restype = ctypes.c_int
            self._lib.qwen_talker_q1_graph_can_replay.argtypes = [ctypes.c_void_p]
            self._lib.qwen_talker_q1_graph_replay_commit.restype = ctypes.c_int
            self._lib.qwen_talker_q1_graph_replay_commit.argtypes = [ctypes.c_void_p]
            self._lib.qwen_talker_q1_graph_invalidate.restype = ctypes.c_int
            self._lib.qwen_talker_q1_graph_invalidate.argtypes = [ctypes.c_void_p]
        self._lib.qwen_talker_debug_execute_layer_async.restype = ctypes.c_int
        self._lib.qwen_talker_debug_execute_layer_async.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta),
            ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta), ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta), ctypes.c_void_p,
        ]
        self._lib.qwen_talker_debug_copy_tap_async.restype = ctypes.c_int
        self._lib.qwen_talker_debug_copy_tap_async.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta),
            ctypes.c_uint32, ctypes.c_void_p,
        ]
        self._lib.qwen_talker_debug_copy_input_rms_mean_async.restype = ctypes.c_int
        self._lib.qwen_talker_debug_copy_input_rms_mean_async.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta), ctypes.c_void_p,
        ]
        self._lib.qwen_talker_debug_copy_layer_cache_async.restype = ctypes.c_int
        self._lib.qwen_talker_debug_copy_layer_cache_async.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta),
            ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta), ctypes.c_void_p,
        ]
        self._lib.qwen_talker_debug_copy_layer_cache_positions_async.restype = ctypes.c_int
        self._lib.qwen_talker_debug_copy_layer_cache_positions_async.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta), ctypes.c_void_p,
            ctypes.POINTER(QwenTensorMeta), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
        ]
        self._lib.qwen_talker_trace_abi_version.restype = ctypes.c_uint32
        self._lib.qwen_talker_trace_set_enabled.restype = ctypes.c_int
        self._lib.qwen_talker_trace_set_enabled.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self._lib.qwen_talker_trace_set_expected_steps.restype = ctypes.c_int
        self._lib.qwen_talker_trace_set_expected_steps.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self._lib.qwen_talker_trace_copy_effective_mask_async.restype = ctypes.c_int
        self._lib.qwen_talker_trace_copy_effective_mask_async.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta), ctypes.c_void_p,
        ]
        self._lib.qwen_talker_trace_copy_inputs_async.restype = ctypes.c_int
        self._lib.qwen_talker_trace_copy_inputs_async.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta),
            ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta), ctypes.c_void_p,
        ]
        self._lib.qwen_talker_trace_copy_position_ids_async.restype = ctypes.c_int
        self._lib.qwen_talker_trace_copy_position_ids_async.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta), ctypes.c_void_p,
        ]
        self._lib.qwen_talker_trace_copy_output_async.restype = ctypes.c_int
        self._lib.qwen_talker_trace_copy_output_async.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta), ctypes.c_void_p,
        ]
        self._lib.qwen_talker_trace_copy_cache_rows_async.restype = ctypes.c_int
        self._lib.qwen_talker_trace_copy_cache_rows_async.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta),
            ctypes.c_void_p, ctypes.POINTER(QwenTensorMeta), ctypes.c_void_p,
        ]
        self._lib.qwen_talker_trace_last_step.restype = ctypes.c_int
        self._lib.qwen_talker_trace_last_step.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(QwenTraceStep),
        ]
        for name, restype in {
            "qwen_talker_runtime_workspace_bytes": ctypes.c_uint64,
            "qwen_talker_runtime_joint_steps": ctypes.c_uint64,
            "qwen_talker_runtime_terminal_query_count": ctypes.c_uint64,
            "qwen_talker_runtime_terminal_prefix_cache_appends": ctypes.c_uint64,
            "qwen_talker_runtime_terminal_audio_cache_appends": ctypes.c_uint64,
            "qwen_talker_runtime_terminal_query_position": ctypes.c_uint32,
            "qwen_talker_runtime_terminal_visible_prefix_position": ctypes.c_uint32,
            "qwen_talker_runtime_cache_imports": ctypes.c_uint64,
            "qwen_talker_runtime_execute_layers": ctypes.c_uint32,
            "qwen_talker_runtime_current_prefix_position": ctypes.c_uint32,
            "qwen_talker_runtime_current_audio_position": ctypes.c_uint32,
            "qwen_talker_runtime_max_cache_len": ctypes.c_uint32,
            "qwen_talker_runtime_last_error": ctypes.c_int32,
            "qwen_talker_runtime_attention_graph_captured": ctypes.c_int32,
            "qwen_talker_runtime_rmsnorm_graph_captured": ctypes.c_int32,
        }.items():
            self._lib[name].restype = restype
            self._lib[name].argtypes = [ctypes.c_void_p]
        if self._native_abi_version >= 4:
            self._lib.qwen_talker_q1_graph_snapshot_device_state.restype = ctypes.c_int
            self._lib.qwen_talker_q1_graph_snapshot_device_state.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ]
            for name, restype in {
                "qwen_talker_runtime_q1_device_cursor_supported": ctypes.c_int32,
                "qwen_talker_runtime_q1_graph_captured": ctypes.c_int32,
                "qwen_talker_runtime_q1_graph_replays": ctypes.c_uint64,
            }.items():
                self._lib[name].restype = restype
                self._lib[name].argtypes = [ctypes.c_void_p]
        self._lib.qwen_talker_destroy.argtypes = [ctypes.c_void_p]
        self._lib.qwen_talker_destroy.restype = None
        if self._lib.qwen_talker_trace_abi_version() != NATIVE_TRACE_ABI_VERSION:
            raise RuntimeError("ABI trace Qwen nativa incompatibile")

    def _ensure_open(self) -> None:
        if self._closed or not self._ctx:
            raise RuntimeError("Qwen native executor già chiuso")

    def _expected_descriptor(self, name: str) -> dict[str, Any]:
        for layer in self._topology["layer_descriptors"]:
            for descriptor in layer["weights"]:
                if descriptor["name"] == name:
                    return descriptor
        for descriptor in self._topology["global_descriptors"]:
            if descriptor["name"] == name:
                return descriptor
        raise RuntimeError(f"peso Qwen non presente nel contratto runtime: {name}")

    def _bind_tensor(self, name: str, tensor: torch.Tensor) -> int:
        expected = self._expected_descriptor(name)
        actual_shape = [int(value) for value in tensor.shape]
        if actual_shape != expected["shape"] or [int(value) for value in tensor.stride()] != expected["stride"]:
            raise ValueError(f"shape/stride inattesi per {name}: {actual_shape}/{list(tensor.stride())}")
        if str(tensor.dtype) != expected["dtype"] or not tensor.is_contiguous():
            raise ValueError(f"dtype/contig inattesi per {name}")
        self._weight_refs.append(tensor)
        self._weight_meta_refs.append(_meta(tensor))
        return int(tensor.data_ptr())

    def _bind_weights(self) -> QwenWeights:
        model = self.model
        result = QwenWeights()
        result.final_norm_w = self._bind_tensor("model.norm.weight", model.norm.weight)
        result.rotary_inv_freq = self._bind_tensor("model.rotary_emb.inv_freq", model.rotary_emb.inv_freq)
        for layer_index, layer in enumerate(model.layers):
            prefix = f"model.layers.{layer_index}."
            result.layers[layer_index] = QwenLayerWeights(
                input_norm_w=self._bind_tensor(prefix + "input_layernorm.weight", layer.input_layernorm.weight),
                post_attention_norm_w=self._bind_tensor(prefix + "post_attention_layernorm.weight", layer.post_attention_layernorm.weight),
                q_proj_w=self._bind_tensor(prefix + "self_attn.q_proj.weight", layer.self_attn.q_proj.weight),
                k_proj_w=self._bind_tensor(prefix + "self_attn.k_proj.weight", layer.self_attn.k_proj.weight),
                v_proj_w=self._bind_tensor(prefix + "self_attn.v_proj.weight", layer.self_attn.v_proj.weight),
                o_proj_w=self._bind_tensor(prefix + "self_attn.o_proj.weight", layer.self_attn.o_proj.weight),
                q_norm_w=self._bind_tensor(prefix + "self_attn.q_norm.weight", layer.self_attn.q_norm.weight),
                k_norm_w=self._bind_tensor(prefix + "self_attn.k_norm.weight", layer.self_attn.k_norm.weight),
                gate_proj_w=self._bind_tensor(prefix + "mlp.gate_proj.weight", layer.mlp.gate_proj.weight),
                up_proj_w=self._bind_tensor(prefix + "mlp.up_proj.weight", layer.mlp.up_proj.weight),
                down_proj_w=self._bind_tensor(prefix + "mlp.down_proj.weight", layer.mlp.down_proj.weight),
            )
        return result

    def _current_model_weight_tensors(self) -> list[torch.Tensor]:
        model = self.model
        result = [model.norm.weight, model.rotary_emb.inv_freq]
        for layer in model.layers:
            result.extend(
                (
                    layer.input_layernorm.weight,
                    layer.post_attention_layernorm.weight,
                    layer.self_attn.q_proj.weight,
                    layer.self_attn.k_proj.weight,
                    layer.self_attn.v_proj.weight,
                    layer.self_attn.o_proj.weight,
                    layer.self_attn.q_norm.weight,
                    layer.self_attn.k_norm.weight,
                    layer.mlp.gate_proj.weight,
                    layer.mlp.up_proj.weight,
                    layer.mlp.down_proj.weight,
                )
            )
        return result

    @staticmethod
    def _weight_signature(tensors: list[torch.Tensor]) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                int(tensor.data_ptr()),
                str(tensor.device),
                str(tensor.dtype),
                tuple(int(value) for value in tensor.shape),
                tuple(int(value) for value in tensor.stride()),
                int(tensor._version),
            )
            for tensor in tensors
        )

    def _invalidate_q1_graph(self, reason: str) -> None:
        if self._native_abi_version >= 4 and self._ctx:
            self._lib.qwen_talker_q1_graph_invalidate(self._ctx)
        self._q1_graph_failure = str(reason)

    def _assert_q1_graph_contract(self) -> None:
        if self._q1_cuda_graph is None:
            return
        if self.max_cache_len != self._q1_graph_capacity:
            self._invalidate_q1_graph("native cache capacity changed after Q1 graph capture")
            raise RuntimeError(self._q1_graph_failure)
        current_weights = self._current_model_weight_tensors()
        if len(current_weights) != len(self._weight_refs) or any(
            current is not bound for current, bound in zip(current_weights, self._weight_refs)
        ):
            self._invalidate_q1_graph("bound Qwen weight tensors changed after Q1 graph capture")
            raise RuntimeError(self._q1_graph_failure)
        signature = self._weight_signature(current_weights)
        if signature != self._q1_graph_weight_signature:
            self._invalidate_q1_graph("Qwen weight storage or version changed after Q1 graph capture")
            raise RuntimeError(self._q1_graph_failure)

    def import_cache_async(
        self,
        cache: _SlotKVCache,
        *,
        stable_length: int,
        prefix_start: int,
        audio_start: int,
    ) -> None:
        self._ensure_open()
        if not isinstance(cache, _SlotKVCache) or len(cache._keys) != QWEN_NATIVE_NUM_LAYERS or len(cache._values) != QWEN_NATIVE_NUM_LAYERS:
            raise ValueError("cache Qwen import deve contenere tutti i layer")
        imported = QwenImportedCache()
        cache_meta_refs: list[QwenTensorMeta] = []
        self._cache_refs = []
        for layer_index in range(QWEN_NATIVE_NUM_LAYERS):
            key = cache._keys[layer_index]
            value = cache._values[layer_index]
            if tuple(key.shape[:2]) != (1, QWEN_NATIVE_NUM_KV_HEADS) or tuple(key.shape[3:]) != (QWEN_NATIVE_HEAD_DIM,):
                raise ValueError(f"shape K cache inattesa layer {layer_index}: {tuple(key.shape)}")
            if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16 or not key.is_contiguous() or not value.is_contiguous():
                raise ValueError("cache Qwen import deve essere CUDA BF16 contiguo")
            from duplex_kernel.talker_state import is_same_device
            if not is_same_device(key.device, self.device) or not is_same_device(value.device, self.device):
                raise ValueError("cache Qwen su device inatteso")
            key_meta = _meta(key)
            value_meta = _meta(value)
            cache_meta_refs.extend((key_meta, value_meta))
            self._cache_refs.extend((key, value))
            imported.layers[layer_index] = QwenImportedKV(
                k_ptr=int(key.data_ptr()), k_meta=ctypes.pointer(cache_meta_refs[-2]),
                v_ptr=int(value.data_ptr()), v_meta=ctypes.pointer(cache_meta_refs[-1]),
            )
        status = self._lib.qwen_talker_import_cache_async(
            self._ctx, ctypes.byref(imported), int(stable_length), int(prefix_start),
            int(audio_start), _stream_handle(self.device)
        )
        if status != 0:
            raise RuntimeError(f"qwen_talker_import_cache_async failed: {status}")
        self._cache_meta_refs = cache_meta_refs
        self._audio_q1_imported = False
        self._audio_q1_next_position = None
        self._audio_q1_limit_position = None
        self._joint_prefix_start = int(prefix_start)
        self._joint_audio_start = int(audio_start)
        self._joint_next_prefix = int(prefix_start)
        self._joint_next_audio = int(audio_start)

    def import_dynamic_cache_async(self, cache: Any, *, max_audio_frames: int) -> None:
        """Start a recurrent audio-Q1 response from an exact HF prefill cache.

        The imported keys already include Qwen's prefill RoPE. The native cache
        is reset per response, and subsequent positions are assigned internally
        from ``sequence_length`` without modifying or recasting these tensors.
        """
        self._ensure_open()
        if self._trace_enabled:
            raise RuntimeError(
                "la trace ABI corrente descrive i passi joint/terminal, non la sequenza audio-Q1"
            )
        sequence_length, layers = _dynamic_cache_kv_layers(cache)
        max_audio_frames = int(max_audio_frames)
        if max_audio_frames <= 0:
            raise ValueError("max_audio_frames deve essere positivo")
        if sequence_length + max_audio_frames > self.max_cache_len:
            raise ValueError(
                "cache nativa insufficiente per prefill e frame CB richiesti: "
                f"prefill={sequence_length}, audio={max_audio_frames}, capacity={self.max_cache_len}"
            )
        if (
            self._q1_cuda_graph is not None
            and self._q1_graph_max_audio_frames is not None
            and max_audio_frames > self._q1_graph_max_audio_frames
        ):
            raise RuntimeError(
                "budget frame Q1 oltre la graph capacity catturata: "
                f"requested={max_audio_frames}, captured={self._q1_graph_max_audio_frames}"
            )
        self._assert_q1_graph_contract()

        caller_stream = torch.cuda.current_stream(device=self.device)
        graph_stream = self._q1_graph_stream
        if graph_stream is not None and int(graph_stream.cuda_stream) != int(caller_stream.cuda_stream):
            graph_stream.wait_stream(caller_stream)
        native_stream = (
            int(graph_stream.cuda_stream) if graph_stream is not None else _stream_handle(self.device)
        )

        imported = QwenImportedCache()
        cache_meta_refs: list[QwenTensorMeta] = []
        cache_refs: list[torch.Tensor] = []
        for layer_index, (key, value) in enumerate(layers):
            from duplex_kernel.talker_state import is_same_device

            if key.device.type != "cuda" or value.device.type != "cuda":
                raise ValueError("DynamicCache Qwen import deve usare tensor CUDA")
            if not is_same_device(key.device, self.device) or not is_same_device(value.device, self.device):
                raise ValueError(f"DynamicCache layer {layer_index} su device inatteso")
            if graph_stream is not None:
                key.record_stream(graph_stream)
                value.record_stream(graph_stream)
            key_meta = _meta(key)
            value_meta = _meta(value)
            cache_meta_refs.extend((key_meta, value_meta))
            cache_refs.extend((key, value))
            imported.layers[layer_index] = QwenImportedKV(
                k_ptr=int(key.data_ptr()), k_meta=ctypes.pointer(cache_meta_refs[-2]),
                v_ptr=int(value.data_ptr()), v_meta=ctypes.pointer(cache_meta_refs[-1]),
            )

        self._audio_q1_imported = False
        self._audio_q1_next_position = None
        self._audio_q1_limit_position = None
        self._joint_prefix_start = None
        self._joint_audio_start = None
        self._joint_next_prefix = None
        self._joint_next_audio = None
        status = self._lib.qwen_talker_import_audio_cache_async(
            self._ctx,
            ctypes.byref(imported),
            int(sequence_length),
            int(max_audio_frames),
            native_stream,
        )
        if status != 0:
            last_error = int(self._lib.qwen_talker_runtime_last_error(self._ctx))
            raise RuntimeError(
                "qwen_talker_import_audio_cache_async failed: "
                f"{status} (native_last_error={last_error})"
            )
        self._cache_refs = cache_refs
        self._cache_meta_refs = cache_meta_refs
        self._audio_q1_imported = True
        self._audio_q1_next_position = sequence_length
        self._audio_q1_limit_position = sequence_length + max_audio_frames
        if graph_stream is not None:
            self._q1_graph_state_dirty = False
            if self._q1_graph_event is None:
                self._q1_graph_event = torch.cuda.Event()
            self._q1_graph_event.record(graph_stream)
            if int(graph_stream.cuda_stream) != int(caller_stream.cuda_stream):
                caller_stream.wait_event(self._q1_graph_event)

    def prepare_q1_graph(
        self,
        cache: Any,
        *,
        max_audio_frames: int,
        warmup_audio_embed: torch.Tensor,
        response_stream: torch.cuda.Stream | None = None,
    ) -> dict[str, Any]:
        """Warm, capture, validate and reset one slot-local plain-Q1 graph.

        This is a startup-only operation. The same native KV addresses and
        frozen weight pointers are reused for every imported response cache.
        The graph's input/output pointers and logical position are dynamic
        device state; only the response frame budget is bounded at import.
        """
        self._ensure_open()
        if self._trace_enabled:
            raise RuntimeError("la trace joint Qwen è incompatibile con il graph Q1")
        if self._native_abi_version < 4:
            raise RuntimeError("graph Q1 richiede ABI nativa 4; mantenere l'ABI 3 solo per eager")
        if self._q1_cuda_graph is not None or self._q1_graph_stream is not None:
            raise RuntimeError("graph Q1 già preparato per questo slot")
        max_audio_frames = int(max_audio_frames)
        if max_audio_frames < 1:
            raise ValueError("il graph Q1 startup richiede almeno 1 frame nel budget")
        self._embedding_ptr(warmup_audio_embed, "warmup_audio_embed")
        sequence_length, _ = _dynamic_cache_kv_layers(cache)
        if sequence_length + max_audio_frames > self.max_cache_len:
            raise ValueError(
                "cache nativa insufficiente per catturare Q1: "
                f"prefill={sequence_length}, audio={max_audio_frames}, capacity={self.max_cache_len}"
            )
        self._q1_graph_weight_signature = self._weight_signature(
            self._current_model_weight_tensors()
        )
        if self._q1_graph_weight_signature != self._weight_signature(self._weight_refs):
            raise RuntimeError("i pesi Qwen sono cambiati dopo il binding nativo")

        # Warm the exact Qwen path on the caller stream, then capture on the
        # leased response slot's stream when one is supplied.
        warmup_stream = torch.cuda.current_stream(device=self.device)
        with torch.inference_mode():
            self.import_dynamic_cache_async(cache, max_audio_frames=max_audio_frames)
            self.step_audio_async(warmup_audio_embed)
            warmup_finite = bool(torch.isfinite(self.last_hidden).all().item())
        if not warmup_finite:
            raise RuntimeError("warmup Q1 eager ha prodotto hidden non finito")
        if self._weight_signature(self._current_model_weight_tensors()) != self._q1_graph_weight_signature:
            raise RuntimeError("i pesi Qwen sono cambiati durante il warmup del graph Q1")

        graph_stream = response_stream or torch.cuda.Stream(device=self.device)
        graph_device = torch.device(graph_stream.device)
        if graph_device.type != self.device.type or (
            self.device.index is not None and graph_device.index not in (None, self.device.index)
        ):
            raise ValueError("Q1 graph response stream usa un device diverso dai pesi nativi")
        if int(graph_stream.cuda_stream) == 0:
            raise RuntimeError("Q1 graph response stream deve essere non-default")
        graph_stream.wait_stream(warmup_stream)
        status = self._lib.qwen_talker_q1_graph_bind_stream(
            self._ctx, int(graph_stream.cuda_stream)
        )
        if status != 0:
            raise RuntimeError(f"qwen_talker_q1_graph_bind_stream failed: {status}")
        self._q1_graph_stream = graph_stream
        self._q1_graph_capacity = self.max_cache_len
        self._q1_graph_max_audio_frames = max_audio_frames
        self._q1_graph_input = torch.empty(
            (1, 1, QWEN_NATIVE_HIDDEN_SIZE), device=self.device, dtype=torch.bfloat16
        )
        warmup_audio_embed.record_stream(graph_stream)
        self._q1_graph_event = torch.cuda.Event(enable_timing=False)
        graph = torch.cuda.CUDAGraph()
        capture_started = False
        try:
            with torch.inference_mode(), torch.cuda.stream(graph_stream):
                self.import_dynamic_cache_async(cache, max_audio_frames=max_audio_frames)
                assert self._q1_graph_input is not None
                self._q1_graph_input.copy_(warmup_audio_embed)
                # Warm the exact bound CUBLAS stream/workspace and ATen inline
                # bridges on the nonblocking side stream, then restore the
                # untouched response cache before capture.
                self.step_audio_async(self._q1_graph_input)
                side_stream_finite = bool(torch.isfinite(self.last_hidden).all().item())
                if not side_stream_finite:
                    raise RuntimeError("warmup Q1 side-stream ha prodotto hidden non finito")
                self.import_dynamic_cache_async(cache, max_audio_frames=max_audio_frames)
                self._q1_graph_input.copy_(warmup_audio_embed)
                status = self._lib.qwen_talker_q1_graph_capture_begin(self._ctx)
                if status != 0:
                    raise RuntimeError(f"qwen_talker_q1_graph_capture_begin failed: {status}")
                capture_started = True
                try:
                    with torch.cuda.graph(graph, stream=graph_stream):
                        self.step_audio_async(self._q1_graph_input)
                except BaseException:
                    self._lib.qwen_talker_q1_graph_capture_end(self._ctx, 0)
                    capture_started = False
                    raise
                status = self._lib.qwen_talker_q1_graph_capture_end(self._ctx, 1)
                capture_started = False
                if status != 0:
                    raise RuntimeError(f"qwen_talker_q1_graph_capture_end failed: {status}")
                native_captured = int(self._lib.qwen_talker_runtime_q1_graph_captured(self._ctx))
                cursor_supported = int(
                    self._lib.qwen_talker_runtime_q1_device_cursor_supported(self._ctx)
                )
                if native_captured != 1 or cursor_supported != 1:
                    raise RuntimeError(
                        "attestazione graph Q1 nativa mancante: "
                        f"captured={native_captured}, device_cursor={cursor_supported}"
                    )

                # Capture itself records the kernels. Re-import the real cache
                # before a finite startup replay so cursor and KV begin at the
                # exact prefill state.
                self.import_dynamic_cache_async(cache, max_audio_frames=max_audio_frames)
                self._q1_graph_input.copy_(warmup_audio_embed)
                status = self._lib.qwen_talker_q1_graph_can_replay(self._ctx)
                if status != 0:
                    raise RuntimeError(f"qwen_talker_q1_graph_can_replay failed: {status}")
                graph.replay()
                status = self._lib.qwen_talker_q1_graph_replay_commit(self._ctx)
                if status != 0:
                    raise RuntimeError(f"qwen_talker_q1_graph_replay_commit failed: {status}")
                replay_finite = bool(torch.isfinite(self.last_hidden).all().item())
                if not replay_finite:
                    raise RuntimeError("replay startup Q1 graph ha prodotto hidden non finito")
                self._snapshot_q1_graph_device_state(expected_cursor=sequence_length + 1)

                # Leave each slot ready for the first real response, not one
                # frame advanced by validation.
                self.import_dynamic_cache_async(cache, max_audio_frames=max_audio_frames)
            self._q1_cuda_graph = graph
            self._q1_graph_capture_count = 1
            warmup_stream.wait_stream(graph_stream)
            return self.q1_graph_info()
        except BaseException as exc:
            if capture_started:
                self._lib.qwen_talker_q1_graph_capture_end(self._ctx, 0)
            self._lib.qwen_talker_q1_graph_invalidate(self._ctx)
            self._q1_graph_failure = f"{type(exc).__name__}: {exc}"
            self._q1_cuda_graph = None
            self._q1_graph_input = None
            self._q1_graph_event = None
            raise

    def q1_graph_info(self) -> dict[str, Any]:
        """Return native-confirmed whole-step graph status for this slot."""
        if self._closed or not self._ctx:
            return {"enabled": False, "captured": False, "replays": 0}
        if self._native_abi_version < 4:
            return {
                "enabled": False,
                "captured": False,
                "replays": 0,
                "abi_version": self._native_abi_version,
                "reason": "graph Q1 requires native ABI 4",
            }
        captured = bool(self._lib.qwen_talker_runtime_q1_graph_captured(self._ctx))
        graph_object_ready = self._q1_cuda_graph is not None
        return {
            "enabled": captured and graph_object_ready,
            "captured": captured,
            "capture_count": self._q1_graph_capture_count,
            "replays": int(self._lib.qwen_talker_runtime_q1_graph_replays(self._ctx)),
            "startup_validation_replays": 1 if graph_object_ready else 0,
            "serving_replays": max(
                0,
                int(self._lib.qwen_talker_runtime_q1_graph_replays(self._ctx))
                - (1 if graph_object_ready else 0),
            ),
            "abi_version": int(self._lib.qwen_talker_abi_version()),
            "failure": self._q1_graph_failure,
            "device_cursor": bool(
                self._lib.qwen_talker_runtime_q1_device_cursor_supported(self._ctx)
            ),
            "device_cursor_actual": self._q1_graph_last_device_cursor,
            "device_cursor_error": int(self._q1_graph_last_device_error),
            "stream_handle_nonnull": bool(
                self._q1_graph_stream is not None and int(self._q1_graph_stream.cuda_stream) != 0
            ),
            "cache_capacity": self._q1_graph_capacity,
            "max_audio_frames": self._q1_graph_max_audio_frames,
            "input_shape": [1, 1, QWEN_NATIVE_HIDDEN_SIZE]
            if self._q1_graph_input is not None else None,
            "input_dtype": str(self._q1_graph_input.dtype)
            if self._q1_graph_input is not None else None,
            "library_sha256": self._library_sha256,
        }

    def _snapshot_q1_graph_device_state(self, *, expected_cursor: int | None) -> tuple[int, int]:
        # Read device cursor/fault once at a startup or response fence.
        graph_stream = self._q1_graph_stream
        if graph_stream is None or self._q1_graph_event is None:
            raise RuntimeError("snapshot Q1 device state richiede graph stream/event")
        caller_stream = torch.cuda.current_stream(device=self.device)
        if int(graph_stream.cuda_stream) != int(caller_stream.cuda_stream):
            graph_stream.wait_stream(caller_stream)
        status = self._lib.qwen_talker_q1_graph_snapshot_device_state(
            self._ctx,
            int(self._q1_graph_device_state.data_ptr()),
            int(graph_stream.cuda_stream),
        )
        if status != 0:
            raise RuntimeError(f"qwen_talker_q1_graph_snapshot_device_state failed: {status}")
        self._q1_graph_event.record(graph_stream)
        if int(graph_stream.cuda_stream) != int(caller_stream.cuda_stream):
            caller_stream.wait_event(self._q1_graph_event)
        with torch.cuda.stream(caller_stream):
            state = self._q1_graph_device_state.detach().to(device="cpu", non_blocking=False).tolist()
        actual_cursor, device_error = int(state[0]), int(state[1])
        self._q1_graph_last_device_cursor = actual_cursor
        self._q1_graph_last_device_error = device_error
        self._q1_graph_state_dirty = False
        if device_error:
            self._q1_graph_failure = f"device KV cursor out of bounds (actual={actual_cursor})"
            self._lib.qwen_talker_q1_graph_invalidate(self._ctx)
            raise RuntimeError(self._q1_graph_failure)
        if expected_cursor is not None and actual_cursor != int(expected_cursor):
            self._q1_graph_failure = (
                f"device KV cursor mismatch: expected={int(expected_cursor)}, actual={actual_cursor}"
            )
            self._lib.qwen_talker_q1_graph_invalidate(self._ctx)
            raise RuntimeError(self._q1_graph_failure)
        return actual_cursor, device_error

    def step_audio_graph_async(self, audio_embed: torch.Tensor) -> torch.Tensor:
        """Replay the captured Q1 step and return an owned hidden clone."""
        self._ensure_open()
        if self._q1_cuda_graph is None or self._q1_graph_stream is None or self._q1_graph_input is None:
            raise RuntimeError("graph Q1 non catturato per questo slot")
        if self._native_abi_version < 4:
            raise RuntimeError("graph Q1 richiede ABI nativa 4")
        if not bool(self._lib.qwen_talker_runtime_q1_graph_captured(self._ctx)):
            raise RuntimeError("il backend nativo non attesta il graph Q1 catturato")
        self._embedding_ptr(audio_embed, "audio_embed")
        from duplex_kernel.talker_state import is_same_device
        if not is_same_device(audio_embed.device, self.device):
            raise ValueError("audio_embed su device diverso dal graph Q1")
        if self._audio_q1_next_position is None or self._audio_q1_limit_position is None:
            raise RuntimeError("limite frame audio della risposta non inizializzato")
        next_position = _advance_audio_q1_position(
            self._audio_q1_next_position, self._audio_q1_limit_position
        )

        caller_stream = torch.cuda.current_stream(device=self.device)
        graph_stream = self._q1_graph_stream
        if int(graph_stream.cuda_stream) != int(caller_stream.cuda_stream):
            graph_stream.wait_stream(caller_stream)
        with torch.inference_mode(), torch.cuda.stream(graph_stream):
            audio_embed.record_stream(graph_stream)
            self._q1_graph_input.copy_(audio_embed)
            status = self._lib.qwen_talker_q1_graph_can_replay(self._ctx)
            if status != 0:
                raise RuntimeError(f"qwen_talker_q1_graph_can_replay failed: {status}")
            try:
                self._q1_cuda_graph.replay()
            except BaseException as exc:
                self._lib.qwen_talker_q1_graph_invalidate(self._ctx)
                self._q1_graph_failure = f"{type(exc).__name__}: {exc}"
                raise
            status = self._lib.qwen_talker_q1_graph_replay_commit(self._ctx)
            if status != 0:
                self._lib.qwen_talker_q1_graph_invalidate(self._ctx)
                self._q1_graph_failure = f"native replay commit failed: {status}"
                raise RuntimeError(f"qwen_talker_q1_graph_replay_commit failed: {status}")
            output = self.last_hidden.detach().clone()
            assert self._q1_graph_event is not None
            self._q1_graph_event.record(graph_stream)
        self._audio_q1_next_position = next_position
        self._q1_graph_state_dirty = True
        if int(graph_stream.cuda_stream) != int(caller_stream.cuda_stream):
            output.record_stream(caller_stream)
            caller_stream.wait_event(self._q1_graph_event)
        return output

    def fence_q1_graph(self) -> None:
        """Order slot reuse and check device cursor once after response work."""
        if self._q1_cuda_graph is None or self._q1_graph_stream is None:
            return
        if self._q1_graph_state_dirty:
            self._snapshot_q1_graph_device_state(
                expected_cursor=self._audio_q1_next_position
            )
            return
        caller_stream = torch.cuda.current_stream(device=self.device)
        graph_stream = self._q1_graph_stream
        if int(graph_stream.cuda_stream) != int(caller_stream.cuda_stream):
            graph_stream.wait_stream(caller_stream)
            assert self._q1_graph_event is not None
            self._q1_graph_event.record(graph_stream)
            caller_stream.wait_event(self._q1_graph_event)

    def set_trace_expected_steps(self, expected_steps: int) -> None:
        self._ensure_open()
        if not self._trace_enabled:
            raise RuntimeError("trace native non abilitato")
        expected_steps = int(expected_steps)
        if expected_steps <= 0:
            raise ValueError("expected_steps deve essere positivo")
        status = self._lib.qwen_talker_trace_set_expected_steps(
            self._ctx, ctypes.c_uint64(expected_steps)
        )
        if status != 0:
            raise RuntimeError(f"qwen_talker_trace_set_expected_steps failed: {status}")

    @staticmethod
    def _embedding_ptr(tensor: torch.Tensor, label: str) -> int:
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda" or tensor.dtype != torch.bfloat16:
            raise ValueError(f"{label} deve essere CUDA BF16")
        if (
            tuple(tensor.shape) != (1, 1, QWEN_NATIVE_HIDDEN_SIZE)
            or not tensor.is_contiguous()
            or tuple(tensor.stride()) != (QWEN_NATIVE_HIDDEN_SIZE, QWEN_NATIVE_HIDDEN_SIZE, 1)
        ):
            raise ValueError(f"{label} deve avere shape/stride compatte [1,1,2048], got {tuple(tensor.shape)}/{tuple(tensor.stride())}")
        return int(tensor.data_ptr())

    def step_joint_async(
        self,
        prefix_embed: torch.Tensor,
        audio_embed: torch.Tensor,
        *,
        prefix_position: int,
        audio_position: int,
        prefix_rope_position: int | None = None,
        audio_rope_position: int | None = None,
        prefix_attention_mode: int = 0,
        visible_prefix_count: int = 0,
        prefix_log_gate: float = -8.0,
    ) -> torch.Tensor:
        self._ensure_open()
        prefix_rope_position = int(prefix_position if prefix_rope_position is None else prefix_rope_position)
        audio_rope_position = int(audio_position if audio_rope_position is None else audio_rope_position)
        if prefix_attention_mode not in (0, 1, 2) or not math.isfinite(float(prefix_log_gate)):
            raise ValueError("contratto prefix gate nativo non valido")
        if prefix_attention_mode == 2 and int(visible_prefix_count) <= 0:
            raise ValueError("il prefix gate richiede prefix visibili")
        prefix_ptr = self._embedding_ptr(prefix_embed, "prefix_embed")
        audio_ptr = self._embedding_ptr(audio_embed, "audio_embed")
        prefix_meta = _meta(prefix_embed)
        audio_meta = _meta(audio_embed)
        output_meta = _meta(self.last_hidden)
        status = self._lib.qwen_talker_step_joint_async(
            self._ctx, prefix_ptr, ctypes.byref(prefix_meta), audio_ptr, ctypes.byref(audio_meta),
            int(prefix_position), int(audio_position), int(prefix_rope_position),
            int(audio_rope_position), int(prefix_attention_mode), int(visible_prefix_count),
            ctypes.c_float(float(prefix_log_gate)), int(self.last_hidden.data_ptr()),
            ctypes.byref(output_meta), _stream_handle(self.device),
        )
        if status != 0:
            last_error = int(self._lib.qwen_talker_runtime_last_error(self._ctx))
            raise RuntimeError(
                f"qwen_talker_step_joint_async failed: {status} (native_last_error={last_error})"
            )
        self._joint_next_prefix = int(prefix_position) + 1
        self._joint_next_audio = int(audio_position) + 1
        return self.last_hidden

    def step_terminal_async(
        self,
        audio_embed: torch.Tensor,
        *,
        prefix_visible_position: int,
        audio_position: int,
        audio_rope_position: int | None = None,
        prefix_attention_mode: int = 0,
        visible_prefix_count: int = 0,
        prefix_log_gate: float = -8.0,
    ) -> torch.Tensor:
        """Run the one audio-only EOS query after the preceding JOINT-Q=2 path."""
        self._ensure_open()
        audio_rope_position = int(audio_position if audio_rope_position is None else audio_rope_position)
        if prefix_attention_mode not in (0, 1, 2) or not math.isfinite(float(prefix_log_gate)):
            raise ValueError("contratto prefix gate nativo non valido")
        if prefix_attention_mode == 2 and int(visible_prefix_count) <= 0:
            raise ValueError("il prefix gate richiede prefix visibili")
        audio_ptr = self._embedding_ptr(audio_embed, "audio_embed")
        audio_meta = _meta(audio_embed)
        output_meta = _meta(self.last_hidden)
        status = self._lib.qwen_talker_step_terminal_async(
            self._ctx,
            audio_ptr,
            ctypes.byref(audio_meta),
            int(prefix_visible_position),
            int(audio_position),
            int(audio_rope_position),
            int(prefix_attention_mode),
            int(visible_prefix_count),
            ctypes.c_float(float(prefix_log_gate)),
            int(self.last_hidden.data_ptr()),
            ctypes.byref(output_meta),
            _stream_handle(self.device),
        )
        if status != 0:
            last_error = int(self._lib.qwen_talker_runtime_last_error(self._ctx))
            raise RuntimeError(
                "qwen_talker_step_terminal_async failed: "
                f"{status} (native_last_error={last_error})"
            )
        return self.last_hidden

    def step_audio_async(
        self,
        audio_embed: torch.Tensor,
        *,
        max_audio_frames: int | None = None,
        rope_position: int | None = None,
        prefix_attention_mode: int = 0,
        visible_prefix_count: int = 0,
        prefix_log_gate: float = -8.0,
    ) -> torch.Tensor:
        """Advance one recurrent audio-only Q1 step on the imported native KV."""
        self._ensure_open()
        if getattr(self, "_q1_cuda_graph", None) is not None:
            raise RuntimeError("graph Q1 abilitato: il percorso eager non è un fallback valido")
        if not self._audio_q1_imported:
            if (
                max_audio_frames is None
                or self._joint_prefix_start is None
                or self._joint_audio_start is None
                or self._joint_next_prefix != self._joint_audio_start
                or self._joint_next_audio is None
            ):
                raise RuntimeError(
                    "Qwen audio-Q1 suffix richiede tutti i prefix reali contigui"
                )
            max_audio_frames = int(max_audio_frames)
            if max_audio_frames <= 0 or self._joint_audio_start + max_audio_frames > self.max_cache_len:
                raise ValueError("budget Q1 non valido per la cache Qwen importata")
            self._audio_q1_imported = True
            self._audio_q1_next_position = self._joint_next_audio
            self._audio_q1_limit_position = self._joint_audio_start + max_audio_frames
        if self._audio_q1_next_position is None or self._audio_q1_limit_position is None:
            raise RuntimeError("limite frame audio della risposta non inizializzato")
        physical_position = int(self._audio_q1_next_position)
        next_position = _advance_audio_q1_position(
            self._audio_q1_next_position, self._audio_q1_limit_position
        )
        rope_position = int(physical_position if rope_position is None else rope_position)
        if prefix_attention_mode not in (0, 1, 2) or not math.isfinite(float(prefix_log_gate)):
            raise ValueError("contratto prefix gate nativo non valido")
        if prefix_attention_mode == 2 and int(visible_prefix_count) <= 0:
            raise ValueError("il prefix gate richiede prefix visibili")
        audio_ptr = self._embedding_ptr(audio_embed, "audio_embed")
        audio_meta = _meta(audio_embed)
        output_meta = _meta(self.last_hidden)
        status = self._lib.qwen_talker_step_audio_async(
            self._ctx, audio_ptr, ctypes.byref(audio_meta),
            int(rope_position), int(prefix_attention_mode), int(visible_prefix_count),
            ctypes.c_float(float(prefix_log_gate)),
            int(self.last_hidden.data_ptr()), ctypes.byref(output_meta),
            _stream_handle(self.device),
        )
        if status != 0:
            last_error = int(self._lib.qwen_talker_runtime_last_error(self._ctx))
            raise RuntimeError(
                f"qwen_talker_step_audio_async failed: {status} "
                f"(native_last_error={last_error})"
            )
        self._audio_q1_next_position = next_position
        return self.last_hidden

    def set_trace_enabled(self, enabled: bool = True) -> None:
        self._ensure_open()
        status = self._lib.qwen_talker_trace_set_enabled(self._ctx, 1 if enabled else 0)
        if status != 0:
            raise RuntimeError(f"qwen_talker_trace_set_enabled failed: {status}")
        self._trace_enabled = bool(enabled)

    def trace_info(self) -> dict[str, Any]:
        return {
            "schema_version": NATIVE_TRACE_SCHEMA_VERSION,
            "backend": "native",
            "attestation_source": NATIVE_TRACE_ATTESTATION_SOURCE,
            "abi_version": int(self._lib.qwen_talker_trace_abi_version()),
            "library_sha256": self._library_sha256,
            "topology_sha256": str(self._topology["topology_sha256"]),
            "hash_algorithm": NATIVE_TRACE_HASH_ALGORITHM,
            "native_trace_chain_algorithm": NATIVE_TRACE_NATIVE_CHAIN_ALGORITHM,
            "mask_capacity": self.max_cache_len,
        }

    def trace_completion(self) -> dict[str, Any]:
        self._ensure_open()
        native_step = QwenTraceStep()
        status = self._lib.qwen_talker_trace_last_step(self._ctx, ctypes.byref(native_step))
        if status != 0:
            raise RuntimeError(f"qwen_talker_trace_last_step failed: {status}")
        result = {
            "source": "native_abi",
            "expected_step_count": int(native_step.expected_step_count),
            "completed_step_count": int(native_step.completed_step_count),
            "last_step_index": int(native_step.step_index),
            "trace_generation": int(native_step.trace_generation),
            "complete": bool(native_step.native_complete),
            "cache_imports": int(native_step.cache_imports),
            "cuda_error_state": int(native_step.cuda_error_state),
            "native_chain_sha256": bytes(native_step.native_chain).hex(),
        }
        terminal_count = int(self._lib.qwen_talker_runtime_terminal_query_count(self._ctx))
        if terminal_count:
            result["terminal_query"] = {
                "source": "native_abi",
                "count": terminal_count,
                "query_length": 1,
                "visible_prefix_position": int(
                    self._lib.qwen_talker_runtime_terminal_visible_prefix_position(self._ctx)
                ),
                "audio_position": int(
                    self._lib.qwen_talker_runtime_terminal_query_position(self._ctx)
                ),
                "prefix_cache_appends": int(
                    self._lib.qwen_talker_runtime_terminal_prefix_cache_appends(self._ctx)
                ),
                "audio_cache_appends": int(
                    self._lib.qwen_talker_runtime_terminal_audio_cache_appends(self._ctx)
                ),
            }
        return result

    @staticmethod
    def _trace_digest_record(tensor: torch.Tensor) -> dict[str, Any]:
        contiguous = tensor.detach().to(device="cpu").contiguous()
        return {
            "shape": [int(value) for value in contiguous.shape],
            "dtype": str(contiguous.dtype),
            "numel": int(contiguous.numel()),
            "finite": bool(torch.isfinite(contiguous).all().item()),
            "nonzero": int(torch.count_nonzero(contiguous).item()),
            "sha256": tensor_digest(contiguous),
        }

    def trace_snapshot(self, *, output_hidden: torch.Tensor) -> dict[str, Any]:
        """Copy native-owned evidence after a completed JOINT-Q=2 step.

        Every payload below is copied by a native ABI function from native
        CUDA-owned storage.  ``output_hidden`` is intentionally ignored: a
        Python observer must not become the source of native attestation.
        """
        del output_hidden
        self._ensure_open()
        if not self._trace_enabled:
            raise RuntimeError("trace native non abilitato")
        mask = torch.empty(
            (2, self.max_cache_len), device=self.device, dtype=torch.bfloat16
        )
        prefix = torch.empty(
            (1, 1, QWEN_NATIVE_HIDDEN_SIZE), device=self.device, dtype=torch.bfloat16
        )
        audio = torch.empty_like(prefix)
        position_ids = torch.empty((2,), device=self.device, dtype=torch.int64)
        output = torch.empty_like(prefix)
        cache_key = torch.empty(
            (QWEN_NATIVE_NUM_LAYERS, 2, QWEN_NATIVE_NUM_KV_HEADS, QWEN_NATIVE_HEAD_DIM),
            device=self.device,
            dtype=torch.bfloat16,
        )
        cache_value = torch.empty_like(cache_key)
        mask_meta = _meta(mask)
        prefix_meta = _meta(prefix)
        audio_meta = _meta(audio)
        position_meta = _meta(position_ids)
        output_meta = _meta(output)
        cache_key_meta = _meta(cache_key)
        cache_value_meta = _meta(cache_value)
        stream = _stream_handle(self.device)
        status = self._lib.qwen_talker_trace_copy_effective_mask_async(
            self._ctx, int(mask.data_ptr()), ctypes.byref(mask_meta), stream
        )
        if status != 0:
            raise RuntimeError(f"qwen_talker_trace_copy_effective_mask_async failed: {status}")
        status = self._lib.qwen_talker_trace_copy_inputs_async(
            self._ctx,
            int(prefix.data_ptr()), ctypes.byref(prefix_meta),
            int(audio.data_ptr()), ctypes.byref(audio_meta), stream,
        )
        if status != 0:
            raise RuntimeError(f"qwen_talker_trace_copy_inputs_async failed: {status}")
        status = self._lib.qwen_talker_trace_copy_position_ids_async(
            self._ctx, int(position_ids.data_ptr()), ctypes.byref(position_meta), stream
        )
        if status != 0:
            raise RuntimeError(f"qwen_talker_trace_copy_position_ids_async failed: {status}")
        status = self._lib.qwen_talker_trace_copy_output_async(
            self._ctx, int(output.data_ptr()), ctypes.byref(output_meta), stream
        )
        if status != 0:
            raise RuntimeError(f"qwen_talker_trace_copy_output_async failed: {status}")
        status = self._lib.qwen_talker_trace_copy_cache_rows_async(
            self._ctx,
            int(cache_key.data_ptr()), ctypes.byref(cache_key_meta),
            int(cache_value.data_ptr()), ctypes.byref(cache_value_meta), stream,
        )
        if status != 0:
            raise RuntimeError(f"qwen_talker_trace_copy_cache_rows_async failed: {status}")
        native_step = QwenTraceStep()
        status = self._lib.qwen_talker_trace_last_step(self._ctx, ctypes.byref(native_step))
        if status != 0:
            raise RuntimeError(f"qwen_talker_trace_last_step failed: {status}")
        torch.cuda.current_stream(device=self.device).synchronize()
        return {
            "step_index": int(native_step.step_index),
            "trace_generation": int(native_step.trace_generation),
            "expected_step_count": int(native_step.expected_step_count),
            "completed_step_count": int(native_step.completed_step_count),
            "joint_steps": int(native_step.joint_steps),
            "cache_imports": int(native_step.cache_imports),
            "native_previous_chain_sha256": bytes(native_step.native_previous_chain).hex(),
            "native_chain_sha256": bytes(native_step.native_chain).hex(),
            "stable_length": int(native_step.stable_length),
            "prefix_start": int(native_step.prefix_start),
            "audio_start": int(native_step.audio_start),
            "prefix_position": int(native_step.prefix_position),
            "audio_position": int(native_step.audio_position),
            "prefix_embed": self._trace_digest_record(prefix),
            "audio_embed": self._trace_digest_record(audio),
            "effective_mask": self._trace_digest_record(mask),
            "position_ids": self._trace_digest_record(position_ids),
            "position_id_values": [int(value) for value in position_ids.tolist()],
            "position_ids_source": NATIVE_TRACE_ATTESTATION_SOURCE,
            "output_hidden": self._trace_digest_record(output),
            "output_hidden_source": NATIVE_TRACE_ATTESTATION_SOURCE,
            "cache_write": {
                "layers": int(native_step.cache_layers_written),
                "prefix_position": int(native_step.prefix_position),
                "audio_position": int(native_step.audio_position),
                "source": NATIVE_TRACE_ATTESTATION_SOURCE,
                "key": self._trace_digest_record(cache_key),
                "value": self._trace_digest_record(cache_value),
            },
            "cuda_error_state": int(native_step.cuda_error_state),
        }

    def debug_execute_layer_async(
        self,
        layer_idx: int,
        prefix_embed: torch.Tensor,
        audio_embed: torch.Tensor,
        *,
        prefix_position: int,
        audio_position: int,
    ) -> torch.Tensor:
        self._ensure_open()
        prefix_ptr = self._embedding_ptr(prefix_embed, "prefix_embed")
        audio_ptr = self._embedding_ptr(audio_embed, "audio_embed")
        prefix_meta = _meta(prefix_embed)
        audio_meta = _meta(audio_embed)
        output_meta = _meta(self.debug_hidden)
        status = self._lib.qwen_talker_debug_execute_layer_async(
            self._ctx, int(layer_idx), prefix_ptr, ctypes.byref(prefix_meta), audio_ptr,
            ctypes.byref(audio_meta), int(prefix_position), int(audio_position),
            int(self.debug_hidden.data_ptr()), ctypes.byref(output_meta), _stream_handle(self.device),
        )
        if status != 0:
            last_error = int(self._lib.qwen_talker_runtime_last_error(self._ctx))
            raise RuntimeError(
                f"qwen_talker_debug_execute_layer_async failed: {status} (native_last_error={last_error})"
            )
        return self.debug_hidden

    def debug_copy_layer_cache_positions_async(
        self,
        layer_idx: int,
        *,
        prefix_position: int,
        audio_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (2, QWEN_NATIVE_NUM_KV_HEADS, QWEN_NATIVE_HEAD_DIM)
        key = torch.empty(shape, device=self.device, dtype=torch.bfloat16)
        value = torch.empty_like(key)
        key_meta = _meta(key)
        value_meta = _meta(value)
        status = self._lib.qwen_talker_debug_copy_layer_cache_positions_async(
            self._ctx, int(layer_idx), int(prefix_position), int(audio_position),
            int(key.data_ptr()), ctypes.byref(key_meta), int(value.data_ptr()),
            ctypes.byref(value_meta), int(key.numel()), int(value.numel()), _stream_handle(self.device),
        )
        if status != 0:
            raise RuntimeError(
                "qwen_talker_debug_copy_layer_cache_positions_async failed: "
                f"{status}"
            )
        return key, value

    def debug_copy_tap_async(self, tap_id: int, shape: tuple[int, ...]) -> torch.Tensor:
        self._ensure_open()
        elements = 1
        for value in shape:
            elements *= int(value)
        destination = torch.empty(shape, device=self.device, dtype=torch.bfloat16)
        destination_meta = _meta(destination)
        status = self._lib.qwen_talker_debug_copy_tap_async(
            self._ctx, int(tap_id), int(destination.data_ptr()), ctypes.byref(destination_meta),
            int(elements), _stream_handle(self.device),
        )
        if status != 0:
            raise RuntimeError(f"qwen_talker_debug_copy_tap_async failed: {status}")
        return destination

    def debug_copy_input_rms_mean_async(self) -> torch.Tensor:
        self._ensure_open()
        destination = torch.empty((2,), device=self.device, dtype=torch.float32)
        destination_meta = _meta(destination)
        status = self._lib.qwen_talker_debug_copy_input_rms_mean_async(
            self._ctx, int(destination.data_ptr()), ctypes.byref(destination_meta), _stream_handle(self.device)
        )
        if status != 0:
            raise RuntimeError(f"qwen_talker_debug_copy_input_rms_mean_async failed: {status}")
        return destination

    def debug_copy_layer_cache_async(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        key = torch.empty(
            (1, QWEN_NATIVE_NUM_KV_HEADS, self.max_cache_len, QWEN_NATIVE_HEAD_DIM),
            device=self.device,
            dtype=torch.bfloat16,
        )
        value = torch.empty_like(key)
        key_meta = _meta(key)
        value_meta = _meta(value)
        status = self._lib.qwen_talker_debug_copy_layer_cache_async(
            self._ctx, int(layer_idx), int(key.data_ptr()), ctypes.byref(key_meta),
            int(value.data_ptr()), ctypes.byref(value_meta), _stream_handle(self.device),
        )
        if status != 0:
            raise RuntimeError(f"qwen_talker_debug_copy_layer_cache_async failed: {status}")
        return key, value

    def runtime_info(self) -> dict[str, Any]:
        if self._closed:
            return {
                "enabled": False,
                "implementation": "QWEN_NATIVE_INCREMENTAL_Q2_V1",
                "num_layers": QWEN_NATIVE_NUM_LAYERS,
                "execute_layers": self.execute_layers,
                "hidden_size": QWEN_NATIVE_HIDDEN_SIZE,
                "max_cache_len": self.max_cache_len,
                "imported_cache": False,
                "cache_imports": 0,
                "joint_steps": 0,
                "terminal_query_count": 0,
                "terminal_prefix_cache_appends": 0,
                "terminal_audio_cache_appends": 0,
                "terminal_query_position": -1,
                "terminal_visible_prefix_position": -1,
                "native_workspace_bytes": 0,
                "attention_backend": "QWEN_NATIVE_EXACT_NO_ALLOC_V1",
                "rmsnorm_backend": "QWEN_HIDDEN_ATEN_GRAPH_QK_NATIVE_V1",
                "cuda_error_state": -1,
                "attention_graph_captured": False,
                "rmsnorm_graph_captured": False,
                "q1_graph": {"enabled": False, "captured": False, "replays": 0},
            }
        return {
            "enabled": True,
            "implementation": "QWEN_NATIVE_INCREMENTAL_Q2_V1",
            "num_layers": QWEN_NATIVE_NUM_LAYERS,
            "execute_layers": int(self._lib.qwen_talker_runtime_execute_layers(self._ctx)),
            "hidden_size": QWEN_NATIVE_HIDDEN_SIZE,
            "max_cache_len": int(self._lib.qwen_talker_runtime_max_cache_len(self._ctx)),
            "current_prefix_position": int(self._lib.qwen_talker_runtime_current_prefix_position(self._ctx)),
            "current_audio_position": int(self._lib.qwen_talker_runtime_current_audio_position(self._ctx)),
            "imported_cache": int(self._lib.qwen_talker_runtime_cache_imports(self._ctx)) > 0,
            "cache_imports": int(self._lib.qwen_talker_runtime_cache_imports(self._ctx)),
            "joint_steps": int(self._lib.qwen_talker_runtime_joint_steps(self._ctx)),
            "terminal_query_count": int(self._lib.qwen_talker_runtime_terminal_query_count(self._ctx)),
            "terminal_prefix_cache_appends": int(
                self._lib.qwen_talker_runtime_terminal_prefix_cache_appends(self._ctx)
            ),
            "terminal_audio_cache_appends": int(
                self._lib.qwen_talker_runtime_terminal_audio_cache_appends(self._ctx)
            ),
            "terminal_query_position": (
                int(self._lib.qwen_talker_runtime_terminal_query_position(self._ctx))
                if int(self._lib.qwen_talker_runtime_terminal_query_count(self._ctx))
                else -1
            ),
            "terminal_visible_prefix_position": (
                int(self._lib.qwen_talker_runtime_terminal_visible_prefix_position(self._ctx))
                if int(self._lib.qwen_talker_runtime_terminal_query_count(self._ctx))
                else -1
            ),
            "native_workspace_bytes": int(self._lib.qwen_talker_runtime_workspace_bytes(self._ctx)),
            "attention_backend": "QWEN_NATIVE_EXACT_NO_ALLOC_V1",
            "rmsnorm_backend": "QWEN_HIDDEN_ATEN_GRAPH_QK_NATIVE_V1",
            "cuda_error_state": int(self._lib.qwen_talker_runtime_last_error(self._ctx)),
            "attention_graph_captured": bool(self._lib.qwen_talker_runtime_attention_graph_captured(self._ctx)),
            "rmsnorm_graph_captured": bool(self._lib.qwen_talker_runtime_rmsnorm_graph_captured(self._ctx)),
            "last_hidden_lifetime": "borrowed_until_next_step",
            "q1_graph": self.q1_graph_info(),
        }

    def close(self) -> None:
        if not self._closed:
            if self._q1_graph_stream is not None:
                self._q1_graph_stream.synchronize()
            self._q1_cuda_graph = None
            self._q1_graph_input = None
            self._lib.qwen_talker_destroy(self._ctx)
            self._ctx = ctypes.c_void_p()
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["NativeQwenTalker"]
