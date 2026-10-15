"""Python binding for the versioned native listener microbatch ABI.

This is a low-level adapter only.  It accepts an already-attested
``NativeGemmaListenerLM`` as a borrowed-weight provider, closes that provider's
single-session native arena before creating the batch owner, and retains the
model tensors for the lifetime of one shared 35-layer batch context. It never
falls back to repeated ABI3 frame calls.
"""
from __future__ import annotations

import ctypes
import hashlib
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import get_ident
from typing import Any, Mapping

import torch

from duplex_kernel.microbatch_contract import (
    BatchRowDescriptor,
    ListenerBatchManifest,
    NativeBatchCapability,
    RuntimeSignature,
    MIN_BATCH_SIZE,
    SUPPORTED_LISTENER_STREAM_PRIORITIES,
)
from duplex_kernel.native_listener import (
    GEMMA4_ATTN_NATIVE_SPLITK_GQA,
    GEMMA4_HIDDEN_SIZE,
    GEMMA4_RMS_NATIVE_EXACT,
    Gemma4ImportedCache,
    Gemma4ListenerConfig,
    Gemma4ListenerWeights,
    Gemma4TensorMeta,
    NativeGemmaListenerLM,
)


BATCH_API_VERSION = 2
BATCH_CAPABILITIES_V1 = 0x7F
BATCH_DYNAMIC_CAPABILITY = 0x200
BATCH_OUTPUT_HEALTH_CAPABILITY = 0x400
BATCH_PROBE_CAPABILITY = 0x80
BATCH_PROBE_API_VERSION = 1
BATCH_TRACE_CAPABILITY = 0x100
BATCH_TRACE_API_VERSION = 1
BATCH_GEMM_GRAPH_CAPABILITY = 0x800
BATCH_CACHE_COPY_CAPABILITY = 0x1000
BATCH_SUPPORTED_MASK_ANY_POSITIVE_N = 0x1
BATCH_PREFIX_MIN_LEN = 1
BATCH_PREFIX_MAX_LEN = 291
BATCH_PREFIX_LENGTHS = range(BATCH_PREFIX_MIN_LEN, BATCH_PREFIX_MAX_LEN + 1)
TRACE_KIND_GEMM = 1
TRACE_KIND_RMS = 2
TRACE_KIND_ELEMENTWISE = 3
TRACE_KIND_ATTENTION = 4
TRACE_GEMM_WRAPPER_NONE = 0
TRACE_GEMM_WRAPPER_LEGACY_CUBLAS = 1
TRACE_GEMM_WRAPPER_BATCH_ROW_GEMM = 2
TRACE_FLAG_INPUTS_CAPTURED = 1
TRACE_FLAG_ACTUAL_CAPTURED = 2
TRACE_FLAG_REFERENCE_CAPTURED = 4
TRACE_FLAG_REFERENCE_DEFERRED = 8
TRACE_FLAG_ATTENTION_OUTPUT_ROW0 = 1 << 8
TRACE_FLAG_ATTENTION_OUTPUT_ROW1 = 1 << 9
TRACE_STAGE_NAMES = (
    "AUDIO_PLE_GEMM", "AUDIO_PLE_SCALE", "AUDIO_PLE_RMS",
    "AUDIO_PLE_FRAME_PAD", "AUDIO_PLE_PAD_RESIDUAL", "AUDIO_PLE_INPUT_SCALE",
    "L0_INPUT_NORM", "L0_Q_PROJ", "L0_Q_NORM", "L0_K_PROJ", "L0_V_PROJ",
    "L0_K_NORM", "L0_V_NORM", "L0_ATTENTION", "L0_O_PROJ",
    "L0_POST_ATTN_NORM", "L0_POST_ATTN_RESIDUAL", "L0_PRE_FF_NORM",
    "L0_GATE_PROJ", "L0_UP_PROJ", "L0_GATE_GELU", "L0_GATE_UP_MUL",
    "L0_DOWN_PROJ", "L0_POST_FF_NORM", "L0_POST_FF_RESIDUAL",
    "L0_PLE_GATE_PROJ", "L0_PLE_GELU", "L0_PLE_LAYER_MULT", "L0_PLE_PROJ",
    "L0_PLE_NORM", "L0_PLE_RESIDUAL", "L0_LAYER_SCALE",
    "FINAL_LAYER_OUTPUT",
)
TRACE_STAGE_KINDS = (
    TRACE_KIND_GEMM, TRACE_KIND_ELEMENTWISE, TRACE_KIND_RMS,
    TRACE_KIND_ELEMENTWISE, TRACE_KIND_ELEMENTWISE, TRACE_KIND_ELEMENTWISE,
    TRACE_KIND_RMS, TRACE_KIND_GEMM, TRACE_KIND_RMS, TRACE_KIND_GEMM,
    TRACE_KIND_GEMM, TRACE_KIND_RMS, TRACE_KIND_RMS, TRACE_KIND_ATTENTION,
    TRACE_KIND_GEMM, TRACE_KIND_RMS, TRACE_KIND_ELEMENTWISE, TRACE_KIND_RMS,
    TRACE_KIND_GEMM, TRACE_KIND_GEMM, TRACE_KIND_ELEMENTWISE,
    TRACE_KIND_ELEMENTWISE, TRACE_KIND_GEMM, TRACE_KIND_RMS,
    TRACE_KIND_ELEMENTWISE, TRACE_KIND_GEMM, TRACE_KIND_ELEMENTWISE,
    TRACE_KIND_ELEMENTWISE, TRACE_KIND_GEMM, TRACE_KIND_RMS,
    TRACE_KIND_ELEMENTWISE, TRACE_KIND_ELEMENTWISE, TRACE_KIND_ELEMENTWISE,
)
TRACE_MAX_RAW_BYTES = 100 * 1024 * 1024


def trace_reference_layout(stage_name: str, *, n: int) -> dict[str, int]:
    """Return the fixed layer-0 N=2 row split for an RMS trace stage."""
    if type(n) is not int or n != 2:
        raise ValueError("operator trace row geometry is restricted to N=2")
    q_rows = n * 4
    if stage_name == "AUDIO_PLE_RMS":
        return {"actual_rows": q_rows * 35, "reference_rows": 4 * 35, "dim": 256}
    if stage_name == "L0_Q_NORM":
        return {"actual_rows": q_rows * 8, "reference_rows": 4 * 8, "dim": 256}
    if stage_name in {
        "L0_INPUT_NORM", "L0_POST_ATTN_NORM", "L0_PRE_FF_NORM",
        "L0_POST_FF_NORM", "L0_PLE_NORM",
    }:
        return {"actual_rows": q_rows, "reference_rows": 4, "dim": 1536}
    raise ValueError(f"trace stage {stage_name!r} has no RMS reference layout")
NATIVE_ABI_VERSION = 3
LISTENER_STREAM_PRIORITY = -1
LISTENER_STREAM_Q_LEN = 4
LISTENER_MAX_SEQ_LEN = 16_384
LISTENER_MAX_ROLLOVER_AUDIO_TOKENS = 64 * LISTENER_STREAM_Q_LEN
_UINT64_MAX = (1 << 64) - 1


class CBatchRow(ctypes.Structure):
    """Must match ``gemma4_listener_batch_row`` in the versioned C header."""

    _fields_ = [
        ("native_context_id", ctypes.c_uint64),
        ("session_generation", ctypes.c_uint64),
        ("state_epoch_id", ctypes.c_uint64),
        ("frame_id", ctypes.c_uint64),
        ("listener_state_version", ctypes.c_uint64),
        ("sequence_position", ctypes.c_uint64),
        ("cache_position", ctypes.c_uint32),
        ("valid_history_tokens", ctypes.c_uint32),
    ]


class CBatchConfig(ctypes.Structure):
    """Must match ``gemma4_listener_batch_config`` in the versioned C header."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("max_batch_size", ctypes.c_uint32),
        ("attention_backend", ctypes.c_uint32),
        ("rmsnorm_backend", ctypes.c_uint32),
        ("listener_stream_priority", ctypes.c_int32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class CBatchRuntimeInfo(ctypes.Structure):
    """Runtime facts are queried after the actual CUDA stream is bound."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("native_abi_version", ctypes.c_uint32),
        ("batch_api_version", ctypes.c_uint32),
        ("max_batch_size", ctypes.c_uint32),
        ("supported_batch_mask", ctypes.c_uint32),
        ("attention_backend_id", ctypes.c_uint32),
        ("rmsnorm_backend_id", ctypes.c_uint32),
        ("listener_stream_priority", ctypes.c_int32),
        ("max_seq_len", ctypes.c_uint32),
        ("stream_q_len", ctypes.c_uint32),
        ("stream_priority_verified", ctypes.c_uint32),
    ]


class CBatchCounters(ctypes.Structure):
    _fields_ = [
        ("forward_calls", ctypes.c_uint64),
        ("layers_executed", ctypes.c_uint64),
        ("projection_gemm_calls", ctypes.c_uint64),
        ("projection_rows", ctypes.c_uint64),
        ("session_rows", ctypes.c_uint64),
        ("output_generation", ctypes.c_uint64),
        ("batch_api_version", ctypes.c_uint32),
        ("max_batch_size", ctypes.c_uint32),
    ]


class CBatchSessionState(ctypes.Structure):
    """Versioned read-only native slot cursor returned by the parity snapshot API."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("slot", ctypes.c_uint32),
        ("native_context_id", ctypes.c_uint64),
        ("session_generation", ctypes.c_uint64),
        ("state_epoch_id", ctypes.c_uint64),
        ("expected_frame_id", ctypes.c_uint64),
        ("expected_state_version", ctypes.c_uint64),
        ("expected_sequence_position", ctypes.c_uint64),
        ("prefix_len", ctypes.c_uint32),
        ("current_position", ctypes.c_uint32),
        ("valid_kv_len", ctypes.c_uint32),
        ("max_seq_len", ctypes.c_uint32),
        ("primed", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class CBatchProbeInfo(ctypes.Structure):
    """Proof-only numerical boundary metadata from the candidate DSO."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("state", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("rms_dim", ctypes.c_uint32),
        ("rms_backend_id", ctypes.c_uint32),
        ("rms_batched_block_x", ctypes.c_uint32),
        ("rms_batched_block_y", ctypes.c_uint32),
        ("rms_reference_block_x", ctypes.c_uint32),
        ("rms_reference_block_y", ctypes.c_uint32),
        ("q_dim", ctypes.c_uint32),
        ("q_batched_m", ctypes.c_uint32),
        ("q_split_m", ctypes.c_uint32),
        ("q_k", ctypes.c_uint32),
        ("q_lda", ctypes.c_uint32),
        ("q_ldb", ctypes.c_uint32),
        ("q_ldc", ctypes.c_uint32),
        ("q4_gemv4_opt_in", ctypes.c_uint32),
        ("q4_gemv4_split0_eligible", ctypes.c_uint32),
        ("q4_gemv4_split1_eligible", ctypes.c_uint32),
        ("cublas_math_mode", ctypes.c_uint32),
        ("cublas_atomics_mode", ctypes.c_uint32),
        ("cublas_pointer_mode", ctypes.c_uint32),
        ("cublas_stream_matches_listener", ctypes.c_uint32),
        ("forward_generation", ctypes.c_uint64),
        ("rms_weight_device_ptr", ctypes.c_uint64),
        ("q_weight_device_ptr", ctypes.c_uint64),
        ("listener_stream_handle", ctypes.c_uint64),
        ("rms_eps", ctypes.c_float),
    ]


class CBatchTraceInfo(ctypes.Structure):
    """Versioned row-major operator trace metadata from the candidate DSO."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("stage_id", ctypes.c_uint32),
        ("kind", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("layer_index", ctypes.c_uint32),
        ("q_len", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("reference_rows_per_sid", ctypes.c_uint32),
        ("input0_rows", ctypes.c_uint32),
        ("input0_cols", ctypes.c_uint32),
        ("input1_rows", ctypes.c_uint32),
        ("input1_cols", ctypes.c_uint32),
        ("input2_rows", ctypes.c_uint32),
        ("input2_cols", ctypes.c_uint32),
        ("actual_rows", ctypes.c_uint32),
        ("actual_cols", ctypes.c_uint32),
        ("reference_rows", ctypes.c_uint32),
        ("reference_cols", ctypes.c_uint32),
        ("rms_backend_id", ctypes.c_uint32),
        ("attention_backend_id", ctypes.c_uint32),
        ("attention_is_sliding", ctypes.c_uint32),
        ("sliding_window", ctypes.c_uint32),
        ("attention_mask_causal", ctypes.c_uint32),
        ("gemm_m", ctypes.c_uint32),
        ("gemm_n", ctypes.c_uint32),
        ("gemm_k", ctypes.c_uint32),
        ("gemm_lda", ctypes.c_uint32),
        ("gemm_ldb", ctypes.c_uint32),
        ("gemm_ldc", ctypes.c_uint32),
        ("cublas_math_mode", ctypes.c_uint32),
        ("cublas_atomics_mode", ctypes.c_uint32),
        ("cublas_pointer_mode", ctypes.c_uint32),
        ("cublas_stream_matches_listener", ctypes.c_uint32),
        ("launch_block_x", ctypes.c_uint32),
        ("launch_block_y", ctypes.c_uint32),
        ("launch_block_z", ctypes.c_uint32),
        ("launch_grid_x", ctypes.c_uint32),
        ("gemm_actual_wrapper", ctypes.c_uint32),
        ("gemm_reference_wrapper", ctypes.c_uint32),
        ("q4_gemv4_opt_in", ctypes.c_uint32),
        ("q4_gemv4_split0_eligible", ctypes.c_uint32),
        ("q4_gemv4_split1_eligible", ctypes.c_uint32),
        ("q4_gemv4_split0_selected", ctypes.c_uint32),
        ("q4_gemv4_split1_selected", ctypes.c_uint32),
        ("cache_start", ctypes.c_uint32 * 2),
        ("visible_len", ctypes.c_uint32 * 2),
        ("prefix_len", ctypes.c_uint32 * 2),
        ("base_position", ctypes.c_uint32 * 2),
        ("input0_elements", ctypes.c_uint64),
        ("input1_elements", ctypes.c_uint64),
        ("input2_elements", ctypes.c_uint64),
        ("actual_elements", ctypes.c_uint64),
        ("reference_elements", ctypes.c_uint64),
        ("weight_device_ptr", ctypes.c_uint64),
        ("forward_generation", ctypes.c_uint64),
        ("epsilon", ctypes.c_float),
        ("scalar", ctypes.c_float),
    ]


@dataclass(frozen=True, slots=True)
class NativeBatchApiFeatures:
    probe: bool
    trace: bool
    gemm_graphs: bool = False
    cache_copy: bool = False

    def __bool__(self) -> bool:
        """Preserve existing bool use: truth means the RMS/q-proj probe exists."""
        return self.probe


@dataclass(frozen=True, slots=True)
class NativeBatchOutput:
    """A caller-owned clone and its asynchronous Listener-stream completion."""

    manifest_id: str
    generation: int
    rows: tuple[BatchRowDescriptor, ...]
    tensor: torch.Tensor
    completion_event: torch.cuda.Event


@dataclass(frozen=True, slots=True)
class NativeBatchOutputHealth:
    """Immutable native finite-guard result pinned to one output row."""

    generation: int
    row: BatchRowDescriptor
    checked: bool
    nonfinite: bool
    status: str


@dataclass(frozen=True, slots=True)
class NativeBatchSessionState:
    slot: int
    native_context_id: int
    session_generation: int
    state_epoch_id: int
    expected_frame_id: int
    expected_state_version: int
    expected_sequence_position: int
    prefix_len: int
    current_position: int
    valid_kv_len: int
    max_seq_len: int
    primed: bool


@dataclass(frozen=True, slots=True)
class NativeBatchSessionSnapshot:
    """CPU-owned K/V evidence copied from unique source layers on Listener stream."""

    state: NativeBatchSessionState
    kv_by_layer: Mapping[int, tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True, slots=True)
class NativeCacheCopySource:
    """Owner-produced fence and state for cross-thread cache promotion."""

    row: BatchRowDescriptor
    state: NativeBatchSessionState
    prefix_len: int
    ready_event: torch.cuda.Event
    _owner: Any = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class NativeCacheCopyResult:
    """Promotion boundary plus the event fencing its asynchronous D2D copy.

    The source owner and source-ready event stay referenced until the caller
    observes ``completion_event``; the shadow stream must wait on that event
    before mutating or retiring the source cache.
    """

    destination_row: BatchRowDescriptor
    prefix_len: int
    completion_event: torch.cuda.Event
    _source: NativeCacheCopySource = field(repr=False, compare=False)


@dataclass(slots=True)
class _SessionBinding:
    slot: int
    session_id: str
    native_context_id: int
    generation: int
    epoch_text: str
    epoch_numeric: int
    expected_frame_id: int
    expected_state_version: int
    expected_sequence_position: int
    cache_position: int
    valid_history_tokens: int
    prefix_len: int


def _unsigned(name: str, value: Any, bits: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << bits):
        raise ValueError(f"{name} must be an unsigned {bits}-bit integer")
    return value


def _epoch_id_u64(epoch_text: str) -> int:
    """Map the Python epoch string to a stable nonzero C-ABI identity."""
    if not isinstance(epoch_text, str) or not epoch_text.strip():
        raise ValueError("state_epoch_id must be non-empty")
    value = int.from_bytes(hashlib.sha256(epoch_text.encode("utf-8")).digest()[:8], "little")
    return value or 1


def _row_to_c(row: BatchRowDescriptor) -> CBatchRow:
    if not isinstance(row, BatchRowDescriptor):
        raise TypeError("native batch rows must be BatchRowDescriptor instances")
    return CBatchRow(
        _unsigned("native_context_id", row.native_context_id, 64),
        _unsigned("session_generation", row.session_generation, 64),
        _epoch_id_u64(row.state_epoch_id),
        _unsigned("frame_id", row.frame_id, 64),
        _unsigned("listener_state_version", row.listener_state_version, 64),
        _unsigned("sequence_position", row.sequence_position, 64),
        _unsigned("cache_position", row.cache_position, 32),
        _unsigned("valid_history_tokens", row.valid_history_tokens, 32),
    )


def _c_row_values(row: CBatchRow) -> tuple[int, ...]:
    return tuple(int(getattr(row, name)) for name, _ctype in CBatchRow._fields_)


def _configure_native_signatures(lib: Any) -> None:
    """Bind required batch-v2 exports; proof-only exports are configured separately."""
    signatures = {
        "gemma4_listener_abi_version": (ctypes.c_int, []),
        "gemma4_listener_manifest_sha256": (ctypes.c_char_p, []),
        "gemma4_listener_batch_api_version": (ctypes.c_uint32, []),
        "gemma4_listener_batch_capabilities": (ctypes.c_uint64, []),
        "gemma4_listener_batch_create": (
            ctypes.c_int,
            [
                ctypes.POINTER(Gemma4ListenerConfig),
                ctypes.POINTER(Gemma4ListenerWeights),
                ctypes.POINTER(CBatchConfig),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        ),
        "gemma4_listener_batch_ensure_capacity": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)],
        ),
        "gemma4_listener_batch_destroy": (ctypes.c_int, [ctypes.c_void_p]),
        "gemma4_listener_batch_bind_stream": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
        "gemma4_listener_batch_import_cache_async": (
            ctypes.c_int,
            [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(CBatchRow),
                ctypes.POINTER(Gemma4ImportedCache),
                ctypes.POINTER(Gemma4TensorMeta),
                ctypes.c_uint32,
                ctypes.c_void_p,
            ],
        ),
        "gemma4_listener_batch_step_async": (
            ctypes.c_int,
            [
                ctypes.c_void_p,
                ctypes.POINTER(CBatchRow),
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.POINTER(Gemma4TensorMeta),
                ctypes.c_void_p,
            ],
        ),
        "gemma4_listener_batch_output_acquire": (
            ctypes.c_int,
            [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.POINTER(CBatchRow)),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        ),
        "gemma4_listener_batch_output_clone_async": (
            ctypes.c_int,
            [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.POINTER(Gemma4TensorMeta),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ],
        ),
        "gemma4_listener_batch_output_release": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        ),
        "gemma4_listener_batch_output_health": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(CBatchRow), ctypes.c_uint32,
             ctypes.POINTER(ctypes.c_int)],
        ),
        "gemma4_listener_batch_output_abandon_async": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p],
        ),
        "gemma4_listener_batch_rollover_async": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(CBatchRow), ctypes.c_uint32,
             ctypes.c_void_p],
        ),
        "gemma4_listener_batch_release_session_async": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(CBatchRow), ctypes.c_void_p],
        ),
        "gemma4_listener_batch_get_counters": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.POINTER(CBatchCounters)],
        ),
        "gemma4_listener_batch_get_runtime_info": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.POINTER(CBatchRuntimeInfo)],
        ),
        "gemma4_listener_batch_get_session_state": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(CBatchRow),
             ctypes.POINTER(CBatchSessionState)],
        ),
        "gemma4_listener_batch_copy_session_kv_async": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(CBatchRow), ctypes.c_uint32,
             ctypes.c_void_p, ctypes.POINTER(Gemma4TensorMeta), ctypes.c_void_p,
             ctypes.POINTER(Gemma4TensorMeta), ctypes.c_void_p],
        ),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)  # AttributeError is intentionally fail-closed.
        function.restype = restype
        function.argtypes = argtypes


def _configure_probe_signatures(lib: Any) -> None:
    """Bind diagnostic symbols only when capability bit 7 is advertised."""
    signatures = {
        "gemma4_listener_batch_probe_api_version": (ctypes.c_uint32, []),
        "gemma4_listener_batch_probe_arm": (ctypes.c_int, [ctypes.c_void_p]),
        "gemma4_listener_batch_probe_read_rms": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
             ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
             ctypes.POINTER(CBatchProbeInfo)],
        ),
        "gemma4_listener_batch_probe_mark_rms_exact": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32],
        ),
        "gemma4_listener_batch_probe_qproj_async": (
            ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p],
        ),
        "gemma4_listener_batch_probe_read_qproj": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
             ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
             ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(CBatchProbeInfo)],
        ),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)  # Advertised-but-missing exports fail closed.
        function.restype = restype
        function.argtypes = argtypes


def _configure_trace_signatures(lib: Any) -> None:
    """Bind trace exports only when optional capability bit 8 is advertised."""
    signatures = {
        "gemma4_listener_batch_trace_api_version": (ctypes.c_uint32, []),
        "gemma4_listener_batch_trace_stage_count": (ctypes.c_uint32, []),
        "gemma4_listener_batch_trace_arm": (ctypes.c_int, [ctypes.c_void_p]),
        "gemma4_listener_batch_trace_get_stage_info": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(CBatchTraceInfo)],
        ),
        "gemma4_listener_batch_trace_read_stage": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint32,
             ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
             ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
             ctypes.c_void_p, ctypes.c_size_t],
        ),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)  # Advertised-but-missing exports fail closed.
        function.restype = restype
        function.argtypes = argtypes


def _configure_native_api(lib: Any) -> NativeBatchApiFeatures:
    """Validate dynamic-capacity batch v2 and configure optional APIs."""
    _configure_native_signatures(lib)
    if int(lib.gemma4_listener_abi_version()) != NATIVE_ABI_VERSION:
        raise RuntimeError("native batch DSO does not expose the required ABI3 weight layout")
    if int(lib.gemma4_listener_batch_api_version()) != BATCH_API_VERSION:
        raise RuntimeError("native batch API version mismatch; no ABI3 fallback is allowed")
    capabilities = int(lib.gemma4_listener_batch_capabilities())
    if capabilities & BATCH_CAPABILITIES_V1 != BATCH_CAPABILITIES_V1:
        raise RuntimeError("native batch capability mask is missing required batch features")
    if capabilities & BATCH_DYNAMIC_CAPABILITY != BATCH_DYNAMIC_CAPABILITY:
        raise RuntimeError("native batch DSO does not attest arbitrary positive row counts")
    if capabilities & BATCH_OUTPUT_HEALTH_CAPABILITY != BATCH_OUTPUT_HEALTH_CAPABILITY:
        raise RuntimeError("native batch DSO does not attest row-bound output health")
    gemm_graphs_available = bool(capabilities & BATCH_GEMM_GRAPH_CAPABILITY)
    if gemm_graphs_available:
        for name, restype, argtypes in (
            ("gemma4_listener_batch_prepare_gemm_graphs", ctypes.c_int,
             [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]),
            ("gemma4_listener_batch_gemm_graph_count", ctypes.c_uint64,
             [ctypes.c_void_p]),
        ):
            function = getattr(lib, name)
            function.restype = restype
            function.argtypes = argtypes
    cache_copy_available = bool(capabilities & BATCH_CACHE_COPY_CAPABILITY)
    if cache_copy_available:
        function = getattr(lib, "gemma4_listener_batch_copy_session_cache_async")
        function.restype = ctypes.c_int
        function.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(CBatchRow),
            ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(CBatchRow), ctypes.c_void_p,
        ]
    probe_available = bool(capabilities & BATCH_PROBE_CAPABILITY)
    if probe_available:
        _configure_probe_signatures(lib)
        if int(lib.gemma4_listener_batch_probe_api_version()) != BATCH_PROBE_API_VERSION:
            raise RuntimeError("native numerical-probe API version mismatch")
    trace_available = bool(capabilities & BATCH_TRACE_CAPABILITY)
    if trace_available:
        _configure_trace_signatures(lib)
        if int(lib.gemma4_listener_batch_trace_api_version()) != BATCH_TRACE_API_VERSION:
            raise RuntimeError("native operator-trace API version mismatch")
        stage_count = int(lib.gemma4_listener_batch_trace_stage_count())
        if stage_count != len(TRACE_STAGE_NAMES):
            raise RuntimeError(
                f"native operator-trace stage inventory mismatch: {stage_count} "
                f"!= {len(TRACE_STAGE_NAMES)}"
            )
    return NativeBatchApiFeatures(
        probe=probe_available, trace=trace_available, gemm_graphs=gemm_graphs_available,
        cache_copy=cache_copy_available,
    )


class NativeGemmaListenerMicrobatch:
    """Own one shared native model and a growable set of independent KV slots.

    ``provider`` must already have verified the live model topology and ABI3
    tensor layouts. Its native single-session allocation
    is closed before this object allocates its one batch context.
    """

    def __init__(
        self,
        provider: NativeGemmaListenerLM,
        *,
        library_path: str | Path,
        runtime_signature: RuntimeSignature,
        listener_stream: torch.cuda.Stream,
        max_batch_size: int = 4,
        enable_gemm_graphs: bool = False,
        stream_priority: int = LISTENER_STREAM_PRIORITY,
    ) -> None:
        self._owner_thread_id = get_ident()
        self._ctx = ctypes.c_void_p()
        self._poisoned = False
        self._closed = False
        self._active_output_generation: int | None = None
        self._last_health_generation = 0
        self._last_probe_info: dict[str, int | float] | None = None
        self._probe_api_available = False
        self._trace_api_available = False
        self._gemm_graphs_api_available = False
        self._cache_copy_api_available = False
        self.gemm_graphs_enabled = False
        self.gemm_graph_count = 0
        self.gemm_graph_prepare_ms = 0.0
        self._session_by_id: dict[str, _SessionBinding] = {}
        self._session_by_context: dict[int, _SessionBinding] = {}
        self._epoch_by_numeric: dict[int, str] = {}
        self._import_refs: dict[int, tuple[torch.Tensor, ...]] = {}
        self._provider = provider
        self.runtime_signature = runtime_signature
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
            raise TypeError("max_batch_size must be a positive integer capacity hint")
        self.max_batch_size = _unsigned("max_batch_size", max_batch_size, 32)
        if self.max_batch_size < MIN_BATCH_SIZE:
            raise ValueError("max_batch_size must be a positive initial capacity hint")
        if type(enable_gemm_graphs) is not bool:
            raise TypeError("enable_gemm_graphs must be a bool")
        if isinstance(stream_priority, bool) or not isinstance(stream_priority, int):
            raise TypeError("stream_priority must be an integer")
        if stream_priority not in SUPPORTED_LISTENER_STREAM_PRIORITIES:
            raise ValueError("stream_priority must be -1 or 0")
        self._enable_gemm_graphs = enable_gemm_graphs
        self._stream_priority = stream_priority
        self.listener_stream = listener_stream
        self._model_device = getattr(provider, "_model_device", None)
        self._validate_signature(runtime_signature)
        self._validate_provider(provider, runtime_signature)
        if self._model_device is None or self._model_device.type != "cuda":
            raise ValueError("weight provider must hold one CUDA model device")
        if not isinstance(listener_stream, torch.cuda.Stream):
            raise TypeError("listener_stream must be a torch.cuda.Stream")
        stream_device = torch.device(listener_stream.device)
        if stream_device != self._model_device or int(listener_stream.cuda_stream) == 0:
            raise ValueError("Listener stream must be non-default and on the model CUDA device")
        self._native_stream_identity = self._stream_identity(listener_stream)

        path = Path(library_path).expanduser().resolve(strict=True)
        self.library_path = path
        self._lib = ctypes.CDLL(str(path))
        native_features = _configure_native_api(self._lib)
        self._probe_api_available = native_features.probe
        self._trace_api_available = native_features.trace
        self._gemm_graphs_api_available = native_features.gemm_graphs
        self._cache_copy_api_available = native_features.cache_copy
        if self._enable_gemm_graphs and not self._gemm_graphs_api_available:
            raise RuntimeError("native batch DSO does not advertise optional GEMM graph support")
        topology_raw = self._lib.gemma4_listener_manifest_sha256()
        topology_sha = topology_raw.decode("ascii") if topology_raw else ""
        provider_topology = str(getattr(provider, "_manifest_sha256", ""))
        if not topology_sha or topology_sha != provider_topology:
            raise RuntimeError("native batch topology differs from the loaded provider topology")
        if topology_sha != runtime_signature.topology_sha256:
            raise RuntimeError("native batch topology differs from the runtime signature")

        # Keep every borrowed model/metadata object alive before the provider
        # drops its ABI3 arena and its own tensor-reference list.
        self._language_model = provider.language_model
        self._weight_tensors = tuple(provider._pinned_tensors)
        self._weight_tensors += tuple(provider.language_model.parameters())
        self._weight_meta = provider._weight_meta
        self._weights_struct = provider._weights_struct
        self._frame_pad_ple = provider.frame_pad_ple
        self._frame_pad_ple_meta = provider._frame_pad_ple_meta
        self._cache_provider = object.__new__(NativeGemmaListenerLM)
        self._cache_provider.language_model = self._language_model
        self._cache_provider._canonicalize_cuda_tensor = NativeGemmaListenerLM._canonicalize_cuda_tensor
        self._cache_provider._tensor_meta = NativeGemmaListenerLM._tensor_meta
        if tuple(self._frame_pad_ple.shape) != (1, LISTENER_STREAM_Q_LEN, 35, 256):
            raise ValueError("provider frame_pad_ple must have exact [1,4,35,256] layout")
        if self._frame_pad_ple.dtype != torch.bfloat16 or not self._frame_pad_ple.is_contiguous():
            raise ValueError("provider frame_pad_ple must be contiguous CUDA BF16")
        self._frame_pad_ple.record_stream(listener_stream)
        # Weight tensors may have been materialized on the caller's current
        # stream. Queue one device-side dependency before native use; this is
        # startup ordering, not a per-frame host synchronization.
        listener_stream.wait_stream(torch.cuda.current_stream(device=self._model_device))

        # Copy the attested ABI3 descriptors while clearing single-session output.
        config = provider._config_struct
        self._config = Gemma4ListenerConfig(
            max_seq_len=LISTENER_MAX_SEQ_LEN,
            stream_q_len=LISTENER_STREAM_Q_LEN,
            enable_cuda_graph=False,
            output_tensor_ptr=None,
            frame_pad_ple_ptr=int(self._frame_pad_ple.data_ptr()),
            weight_meta=ctypes.cast(self._weight_meta, ctypes.POINTER(Gemma4TensorMeta)),
            weight_meta_count=len(self._weight_meta),
            output_meta=None,
            frame_pad_ple_meta=ctypes.pointer(self._frame_pad_ple_meta),
        )
        del config

        # Relinquish the old per-session cache/scratch allocation before the
        # batch context allocates its single model arena plus private KV slots.
        provider.close()
        self._provider = None
        batch_config = CBatchConfig(
            struct_size=ctypes.sizeof(CBatchConfig),
            api_version=BATCH_API_VERSION,
            max_batch_size=self.max_batch_size,
            attention_backend=GEMMA4_ATTN_NATIVE_SPLITK_GQA,
            rmsnorm_backend=GEMMA4_RMS_NATIVE_EXACT,
            listener_stream_priority=self._stream_priority,
            reserved=(ctypes.c_uint32 * 2)(0, 0),
        )
        create_status = int(self._lib.gemma4_listener_batch_create(
            ctypes.byref(self._config), ctypes.byref(self._weights_struct),
            ctypes.byref(batch_config), ctypes.byref(self._ctx),
        ))
        if create_status != 0 or not self._ctx:
            raise RuntimeError(f"gemma4_listener_batch_create failed with code {create_status}")

        stream_ptr = ctypes.c_void_p(int(listener_stream.cuda_stream))
        bind_status = int(self._lib.gemma4_listener_batch_bind_stream(self._ctx, stream_ptr))
        if bind_status != 0:
            self._destroy_after_init_failure()
            raise RuntimeError(f"native Listener stream bind/priority check failed with code {bind_status}")
        info = CBatchRuntimeInfo(struct_size=ctypes.sizeof(CBatchRuntimeInfo))
        info_status = int(self._lib.gemma4_listener_batch_get_runtime_info(
            self._ctx, ctypes.byref(info),
        ))
        if info_status != 0:
            self._destroy_after_init_failure()
            raise RuntimeError(f"native batch runtime-info query failed with code {info_status}")
        self._validate_runtime_info(info)
        self.capability = NativeBatchCapability(
            api_version=int(info.batch_api_version),
            native_abi_version=int(info.native_abi_version),
            attention_backend_id=int(info.attention_backend_id),
            rmsnorm_backend_id=int(info.rmsnorm_backend_id),
            max_seq_len=int(info.max_seq_len),
            stream_q_len=int(info.stream_q_len),
            supports_arbitrary_batch_size=True,
            stream_priority=int(info.listener_stream_priority),
            stream_identity=self._native_stream_identity,
        )
        self.capability.validate_for(runtime_signature)
        if self._enable_gemm_graphs:
            try:
                self.prepare_gemm_graphs()
            except Exception:
                self._destroy_after_init_failure()
                raise

    @staticmethod
    def _validate_signature(signature: RuntimeSignature) -> None:
        if not isinstance(signature, RuntimeSignature):
            raise TypeError("an immutable RuntimeSignature is required")
        expected = {
            "native_abi_version": NATIVE_ABI_VERSION,
            "batch_api_version": BATCH_API_VERSION,
            "attention_backend_id": GEMMA4_ATTN_NATIVE_SPLITK_GQA,
            "rmsnorm_backend_id": GEMMA4_RMS_NATIVE_EXACT,
            "max_seq_len": LISTENER_MAX_SEQ_LEN,
            "stream_q_len": LISTENER_STREAM_Q_LEN,
        }
        mismatches = {
            name: (getattr(signature, name), value)
            for name, value in expected.items()
            if getattr(signature, name) != value
        }
        if mismatches:
            raise RuntimeError(f"runtime signature is incompatible with the native batch ABI: {mismatches}")

    @staticmethod
    def _validate_provider(provider: NativeGemmaListenerLM, signature: RuntimeSignature) -> None:
        if not isinstance(provider, NativeGemmaListenerLM):
            raise TypeError("provider must be an attested NativeGemmaListenerLM")
        if not provider._ctx:
            raise RuntimeError("ABI3 weight provider is already closed")
        if provider.enable_cuda_graph:
            raise RuntimeError("listener microbatch does not support per-session CUDA graphs")
        if provider.max_seq_len != LISTENER_MAX_SEQ_LEN or provider.stream_q_len != LISTENER_STREAM_Q_LEN:
            raise RuntimeError("provider must preserve exact 16384-token capacity and Q=4")
        info = provider.native_runtime_info()
        expected = {
            "attention_backend_id": GEMMA4_ATTN_NATIVE_SPLITK_GQA,
            "rmsnorm_backend_id": GEMMA4_RMS_NATIVE_EXACT,
            "max_seq_len": LISTENER_MAX_SEQ_LEN,
            "cuda_graph_enabled": False,
            "manifest_sha256": signature.topology_sha256,
        }
        mismatch = {
            name: (info.get(name), value)
            for name, value in expected.items()
            if info.get(name) != value
        }
        if mismatch:
            raise RuntimeError(f"ABI3 weight provider is incompatible with batch runtime signature: {mismatch}")

    def _validate_runtime_info(self, info: CBatchRuntimeInfo) -> None:
        expected = {
            "native_abi_version": NATIVE_ABI_VERSION,
            "batch_api_version": BATCH_API_VERSION,
            "max_batch_size": self.max_batch_size,
            "supported_batch_mask": BATCH_SUPPORTED_MASK_ANY_POSITIVE_N,
            "attention_backend_id": GEMMA4_ATTN_NATIVE_SPLITK_GQA,
            "rmsnorm_backend_id": GEMMA4_RMS_NATIVE_EXACT,
            "listener_stream_priority": self._stream_priority,
            "max_seq_len": LISTENER_MAX_SEQ_LEN,
            "stream_q_len": LISTENER_STREAM_Q_LEN,
            "stream_priority_verified": 1,
        }
        mismatch = {
            name: (int(getattr(info, name)), value)
            for name, value in expected.items()
            if int(getattr(info, name)) != value
        }
        if mismatch:
            self._destroy_after_init_failure()
            raise RuntimeError(f"native batch runtime attestation mismatch: {mismatch}")

    def _destroy_after_init_failure(self) -> None:
        if self._ctx:
            self._lib.gemma4_listener_batch_destroy(self._ctx)
            self._ctx = ctypes.c_void_p()

    def _assert_owner_thread(self) -> None:
        if get_ident() != self._owner_thread_id:
            raise RuntimeError("native listener microbatch is confined to its creating host thread")

    def _assert_open(self) -> None:
        self._assert_owner_thread()
        if self._closed or not self._ctx:
            raise RuntimeError("native listener microbatch is closed")
        if self._poisoned:
            raise RuntimeError("native listener microbatch is poisoned; no retry/fallback is allowed")

    @staticmethod
    def _stream_identity(stream: torch.cuda.Stream) -> str:
        device = torch.device(stream.device)
        if device.type != "cuda" or device.index is None:
            raise ValueError("Listener stream identity requires an indexed CUDA device")
        handle = _unsigned("listener_stream.cuda_stream", int(stream.cuda_stream), 64)
        if handle == 0:
            raise ValueError("Listener stream identity cannot use the default stream")
        return f"cuda:{device.index}:stream:0x{handle:x}"

    def _stream_ptr(self) -> ctypes.c_void_p:
        return ctypes.c_void_p(int(self.listener_stream.cuda_stream))

    def _check_producer_stream(self, producer_stream: torch.cuda.Stream | None) -> None:
        if producer_stream is None:
            current = torch.cuda.current_stream(device=self._model_device)
            if int(current.cuda_stream) != int(self.listener_stream.cuda_stream):
                raise ValueError("producer_stream is required when inputs were not produced on Listener")
            return
        if torch.device(producer_stream.device) != self._model_device:
            raise ValueError("producer stream and Listener stream must use the same CUDA device")
        if int(producer_stream.cuda_stream) != int(self.listener_stream.cuda_stream):
            # PyTorch implements this as a device-side event dependency.
            self.listener_stream.wait_stream(producer_stream)

    @staticmethod
    def _require_cuda_tensor(
        tensor: torch.Tensor,
        *,
        name: str,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> None:
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.device != device or tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have exact {dtype}/{shape}/{device} layout; got "
                f"{tensor.dtype}/{tuple(tensor.shape)}/{tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    @staticmethod
    def _cache_source_tensors(past_key_values: Any) -> tuple[torch.Tensor, ...]:
        """Return strong references to the original cache inputs used by import."""
        tensors: list[torch.Tensor] = []
        seen: set[int] = set()

        def keep(value: Any) -> None:
            if value is None or not isinstance(value, torch.Tensor):
                return
            if id(value) not in seen:
                seen.add(id(value))
                tensors.append(value)

        for layer in getattr(past_key_values, "layers", ()) or ():
            keep(getattr(layer, "keys", None))
            keep(getattr(layer, "values", None))

        shared_layers = getattr(past_key_values, "shared_layers", None)
        shared_values = getattr(shared_layers, "values", None)
        if callable(shared_values):
            for pair in shared_values():
                if isinstance(pair, (tuple, list)):
                    for tensor in pair:
                        keep(tensor)
                else:
                    keep(getattr(pair, "keys", None))
                    keep(getattr(pair, "values", None))
        return tuple(tensors)

    def ensure_capacity(self, requested_capacity: int) -> int:
        """Grow native slot/scratch capacity at the owner-thread tick boundary."""
        self._assert_open()
        if isinstance(requested_capacity, bool) or not isinstance(requested_capacity, int):
            raise TypeError("requested_capacity must be a positive integer")
        requested_capacity = _unsigned("requested_capacity", requested_capacity, 32)
        if requested_capacity < MIN_BATCH_SIZE:
            raise ValueError("requested_capacity must be positive")
        if requested_capacity <= self.max_batch_size:
            return self.max_batch_size
        if self._active_output_generation is not None:
            raise RuntimeError("native capacity cannot grow while an output lease is active")
        out_capacity = ctypes.c_uint32(self.max_batch_size)
        status = int(self._lib.gemma4_listener_batch_ensure_capacity(
            self._ctx, ctypes.c_uint32(requested_capacity), ctypes.byref(out_capacity),
        ))
        if status != 0:
            if status in {-2, -5}:
                self._poisoned = True
            raise RuntimeError(
                f"native capacity growth to {requested_capacity} failed with code {status}; "
                f"previous capacity {self.max_batch_size} remains the reported limit"
            )
        actual_capacity = int(out_capacity.value)
        if actual_capacity < requested_capacity:
            self._poisoned = True
            raise RuntimeError(
                "native capacity growth returned less than requested; context quarantined"
            )
        self.max_batch_size = actual_capacity
        if self.gemm_graphs_enabled:
            graph_count = int(self._lib.gemma4_listener_batch_gemm_graph_count(self._ctx))
            if graph_count <= 0:
                self._poisoned = True
                raise RuntimeError("native capacity grew without rebuilding the enabled GEMM graph cache")
            self.gemm_graph_count = graph_count
        return actual_capacity

    def prepare_gemm_graphs(self) -> int:
        """Pre-capture stable M=4 GEMM signatures before session admission."""
        self._assert_open()
        if not self._gemm_graphs_api_available:
            raise RuntimeError("native batch DSO does not advertise optional GEMM graph support")
        if self.gemm_graphs_enabled:
            return self.gemm_graph_count
        if self._active_output_generation is not None:
            raise RuntimeError("GEMM graph preparation cannot run while an output lease is active")
        out_count = ctypes.c_uint64()
        started_ns = time.perf_counter_ns()
        status = int(self._lib.gemma4_listener_batch_prepare_gemm_graphs(
            self._ctx, ctypes.byref(out_count),
        ))
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        if status != 0:
            raise RuntimeError(f"native GEMM graph preparation failed with code {status}")
        graph_count = int(out_count.value)
        if graph_count <= 0:
            self._poisoned = True
            raise RuntimeError("native GEMM graph preparation returned an empty cache")
        self.gemm_graphs_enabled = True
        self.gemm_graph_count = graph_count
        self.gemm_graph_prepare_ms = elapsed_ms
        return graph_count

    def import_session(
        self,
        row: BatchRowDescriptor,
        past_key_values: Any,
        prefix_attention: torch.Tensor,
        *,
        prefix_len: int,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> int:
        """Import one private DynamicCache and bind it to a native slot."""
        self._assert_open()
        c_row = _row_to_c(row)
        prefix_len = int(prefix_len)
        if prefix_len not in BATCH_PREFIX_LENGTHS:
            raise ValueError(
                f"prefix_len must be in [{BATCH_PREFIX_MIN_LEN}, {BATCH_PREFIX_MAX_LEN}]"
            )
        if row.cache_position < prefix_len or row.valid_history_tokens != row.cache_position:
            raise ValueError("descriptor must report total valid KV length in cache_position")
        if row.cache_position > LISTENER_MAX_SEQ_LEN:
            raise ValueError("descriptor cache_position exceeds the configured native capacity")
        if row.session_id in self._session_by_id or row.native_context_id in self._session_by_context:
            raise ValueError("session ID or native context handle is already bound")
        if len(self._session_by_id) >= self.max_batch_size:
            self.ensure_capacity(len(self._session_by_id) + 1)
        free_slots = [index for index in range(self.max_batch_size)
                      if index not in {item.slot for item in self._session_by_id.values()}]
        if not free_slots:
            raise RuntimeError("native batch has no free private KV slot")
        if prefix_attention.dtype != torch.int64 or tuple(prefix_attention.shape) != (1, prefix_len):
            raise ValueError("prefix_attention must be CUDA int64 [1,prefix_len]")
        self._require_cuda_tensor(
            prefix_attention, name="prefix_attention", dtype=torch.int64,
            shape=(1, prefix_len), device=self._model_device,
        )
        epoch_numeric = int(c_row.state_epoch_id)
        other_epoch = self._epoch_by_numeric.get(epoch_numeric)
        if other_epoch is not None and other_epoch != row.state_epoch_id:
            raise RuntimeError("state epoch hash collision; native import was not submitted")

        # Check and enqueue the producer dependency before any canonicalizing
        # operation can allocate or copy cache tensors on the wrong stream.
        self._check_producer_stream(producer_stream)
        source_tensors = self._cache_source_tensors(past_key_values)
        with torch.cuda.stream(self.listener_stream):
            imported, descriptors, imported_tensors = NativeGemmaListenerLM._prepare_imported_cache(
                self._cache_provider,
                past_key_values, row.cache_position,
            )
            canonical_prefix_attention = NativeGemmaListenerLM._canonicalize_cuda_tensor(
                prefix_attention, name="prefix_attention", dtype=torch.int64,
            )
            prefix_meta = NativeGemmaListenerLM._tensor_meta(
                canonical_prefix_attention, dtype_code=2,
            )
            retained: list[Any] = []
            retained_ids: set[int] = set()
            for value in (
                past_key_values, *source_tensors, *imported_tensors,
                prefix_attention, canonical_prefix_attention,
            ):
                if value is not None and id(value) not in retained_ids:
                    retained_ids.add(id(value))
                    retained.append(value)
            for tensor in (
                *source_tensors, *imported_tensors,
                prefix_attention, canonical_prefix_attention,
            ):
                tensor.record_stream(self.listener_stream)
            # Keep both the source cache and contiguous imports alive while the
            # native D2D import is in flight (and until the slot is retired).
            self._import_refs[free_slots[0]] = tuple(retained)
            status = int(self._lib.gemma4_listener_batch_import_cache_async(
                self._ctx, ctypes.c_uint32(free_slots[0]), ctypes.byref(c_row),
                ctypes.byref(imported), ctypes.byref(prefix_meta), ctypes.c_uint32(prefix_len),
                self._stream_ptr(),
            ))
            del descriptors
        if status != 0:
            self._poisoned = True
            raise RuntimeError(f"native batch cache import failed with code {status}; batch quarantined")
        self._epoch_by_numeric[epoch_numeric] = row.state_epoch_id
        binding = _SessionBinding(
            slot=free_slots[0], session_id=row.session_id,
            native_context_id=row.native_context_id, generation=row.session_generation,
            epoch_text=row.state_epoch_id, epoch_numeric=epoch_numeric,
            expected_frame_id=row.frame_id,
            expected_state_version=row.listener_state_version,
            expected_sequence_position=row.sequence_position,
            cache_position=row.cache_position,
            valid_history_tokens=row.valid_history_tokens,
            prefix_len=prefix_len,
        )
        self._session_by_id[row.session_id] = binding
        self._session_by_context[row.native_context_id] = binding
        return binding.slot

    def _preflight_manifest(self, manifest: ListenerBatchManifest) -> tuple[CBatchRow, ...]:
        self._assert_open()
        if not isinstance(manifest, ListenerBatchManifest):
            raise TypeError("a ListenerBatchManifest is required")
        if manifest.runtime_signature != self.runtime_signature:
            raise RuntimeError("batch manifest runtime signature differs from the shared native model")
        if (
            manifest.stream_identity != self.capability.stream_identity
            or manifest.stream_priority != self.capability.stream_priority
        ):
            raise RuntimeError("batch manifest names a different Listener stream contract")
        if manifest.batch_size != len(manifest.rows) or manifest.batch_size < MIN_BATCH_SIZE:
            raise ValueError("manifest must contain at least one real row")
        if manifest.batch_size > self.max_batch_size:
            self.ensure_capacity(manifest.batch_size)
        c_rows: list[CBatchRow] = []
        context_ids: set[int] = set()
        session_ids: set[str] = set()
        for row in manifest.rows:
            binding = self._session_by_id.get(row.session_id)
            if binding is None or binding.native_context_id != row.native_context_id:
                raise RuntimeError(f"manifest row has no bound native session: {row.session_id}")
            if row.native_context_id in context_ids:
                raise ValueError("manifest contains duplicate native contexts")
            if row.session_id in session_ids:
                raise ValueError("manifest contains duplicate session IDs")
            session_ids.add(row.session_id)
            context_ids.add(row.native_context_id)
            if (
                row.session_generation != binding.generation
                or row.state_epoch_id != binding.epoch_text
                or row.frame_id != binding.expected_frame_id
                or row.listener_state_version != binding.expected_state_version
                or row.sequence_position != binding.expected_sequence_position
                or row.cache_position != binding.cache_position
                or row.valid_history_tokens != binding.valid_history_tokens
            ):
                raise RuntimeError(f"manifest row is stale for native session {row.session_id}")
            c_rows.append(_row_to_c(row))
        return tuple(c_rows)

    def step_async(
        self,
        manifest: ListenerBatchManifest,
        audio_soft: torch.Tensor,
        *,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> NativeBatchOutput:
        """Run one true N-row/35-layer forward and return a cloned output lease."""
        c_rows = self._preflight_manifest(manifest)
        row_count = len(c_rows)
        self._require_cuda_tensor(
            audio_soft, name="audio_soft", dtype=torch.bfloat16,
            shape=(row_count, LISTENER_STREAM_Q_LEN, GEMMA4_HIDDEN_SIZE),
            device=self._model_device,
        )
        if self._active_output_generation is not None:
            raise RuntimeError("the previous native batch output lease is still active")
        self._check_producer_stream(producer_stream)
        audio_soft.record_stream(self.listener_stream)
        c_rows_array = (CBatchRow * row_count)(*c_rows)
        audio_meta = NativeGemmaListenerLM._tensor_meta(audio_soft, dtype_code=1)
        status = int(self._lib.gemma4_listener_batch_step_async(
            self._ctx, c_rows_array, ctypes.c_uint32(row_count),
            ctypes.c_void_p(int(audio_soft.data_ptr())), ctypes.byref(audio_meta), self._stream_ptr(),
        ))
        if status != 0:
            self._poisoned = True
            raise RuntimeError(f"native batched forward failed with code {status}; all rows quarantined")

        generation = ctypes.c_uint64()
        borrowed = ctypes.c_void_p()
        native_rows = ctypes.POINTER(CBatchRow)()
        native_row_count = ctypes.c_uint32()
        producer_event = ctypes.c_void_p()
        status = int(self._lib.gemma4_listener_batch_output_acquire(
            self._ctx, ctypes.byref(generation), ctypes.byref(borrowed),
            ctypes.byref(native_rows), ctypes.byref(native_row_count),
            ctypes.byref(producer_event),
        ))
        provenance_matches = (
            status == 0
            and int(native_row_count.value) == row_count
            and bool(native_rows)
            and all(
                _c_row_values(native_rows[index]) == _c_row_values(c_rows[index])
                for index in range(row_count)
            )
        )
        if status != 0 or not borrowed.value or not producer_event.value or not provenance_matches:
            self._poisoned = True
            # A successful forward owns an output slot even if the acquire
            # response is malformed. Recover its generation only to retire
            # the private lease safely; the borrowed pointer never escapes.
            if not generation.value:
                counters = CBatchCounters()
                counter_status = int(self._lib.gemma4_listener_batch_get_counters(
                    self._ctx, ctypes.byref(counters),
                ))
                if counter_status == 0:
                    generation.value = counters.output_generation
            if generation.value:
                self._active_output_generation = int(generation.value)
                cleanup_status = int(self._lib.gemma4_listener_batch_output_abandon_async(
                    self._ctx, generation, self._stream_ptr(),
                ))
                if cleanup_status == 0:
                    self._active_output_generation = None
            raise RuntimeError(
                "native batch output acquire/provenance failed "
                f"(status={status}, row_count={int(native_row_count.value)}, "
                f"expected_rows={row_count}, provenance_matches={provenance_matches}); batch quarantined"
            )
        self._active_output_generation = int(generation.value)
        native_clone_event = ctypes.c_void_p()
        status = -100
        try:
            clone = torch.empty(
                (row_count, LISTENER_STREAM_Q_LEN, GEMMA4_HIDDEN_SIZE),
                dtype=torch.bfloat16, device=self._model_device,
            )
            clone_meta = NativeGemmaListenerLM._tensor_meta(clone, dtype_code=1)
            status = int(self._lib.gemma4_listener_batch_output_clone_async(
                self._ctx, generation, ctypes.c_uint32(row_count),
                ctypes.c_void_p(int(clone.data_ptr())), ctypes.byref(clone_meta),
                self._stream_ptr(), ctypes.byref(native_clone_event),
            ))
            if status != 0:
                raise RuntimeError(f"native output clone failed with code {status}")
            clone.record_stream(self.listener_stream)
            completion = torch.cuda.Event(enable_timing=False, blocking=False)
            completion.record(self.listener_stream)
            status = int(self._lib.gemma4_listener_batch_output_release(
                self._ctx, generation, native_clone_event, self._stream_ptr(),
            ))
            if status != 0:
                raise RuntimeError(f"native output lease release failed with code {status}")
        except BaseException as exc:
            self._poisoned = True
            abandon_status = int(self._lib.gemma4_listener_batch_output_abandon_async(
                self._ctx, generation, self._stream_ptr(),
            ))
            if abandon_status != 0:
                raise RuntimeError(
                    "native output clone failed and the batch lease could not be safely retired; "
                    f"clone_error={exc}, native_status={status}, abandon_error={abandon_status}"
                )
            self._active_output_generation = None
            raise
        self._active_output_generation = None
        for row in manifest.rows:
            binding = self._session_by_id[row.session_id]
            binding.expected_frame_id += 1
            binding.expected_state_version += 1
            binding.expected_sequence_position += 1
            binding.cache_position += LISTENER_STREAM_Q_LEN
            binding.valid_history_tokens += LISTENER_STREAM_Q_LEN
        return NativeBatchOutput(
            manifest_id=manifest.manifest_id,
            generation=int(generation.value),
            rows=tuple(manifest.rows),
            tensor=clone,
            completion_event=completion,
        )

    def verify_output_health(
        self,
        output: NativeBatchOutput,
    ) -> tuple[NativeBatchOutputHealth, ...]:
        """Read the native batch-wide finite guard once and bind it to exact rows.

        Call on the native owner thread after output completion and before the
        rows are delivered to serving. A batch-wide nonfinite flag is applied
        to every row because the native guard does not attribute bad elements.
        """
        self._assert_open()
        if self._active_output_generation is not None:
            raise RuntimeError("cannot verify output health while a native output lease is active")
        if not isinstance(output, NativeBatchOutput) or not output.rows:
            raise TypeError("output health requires a non-empty NativeBatchOutput")
        generation = _unsigned("output generation", output.generation, 64)
        if generation == 0 or generation != self._last_health_generation + 1:
            raise RuntimeError("native output health generation is stale or out of order")
        c_rows: list[CBatchRow] = []
        for row in output.rows:
            binding = self._session_by_id.get(row.session_id)
            if binding is None or binding.native_context_id != row.native_context_id:
                self._poisoned = True
                raise RuntimeError("output health row has no matching native session binding")
            if (
                row.session_generation != binding.generation
                or row.state_epoch_id != binding.epoch_text
                or row.frame_id + 1 != binding.expected_frame_id
                or row.listener_state_version + 1 != binding.expected_state_version
                or row.sequence_position + 1 != binding.expected_sequence_position
                or row.cache_position + LISTENER_STREAM_Q_LEN != binding.cache_position
                or row.valid_history_tokens + LISTENER_STREAM_Q_LEN != binding.valid_history_tokens
            ):
                self._poisoned = True
                raise RuntimeError("output health row provenance is stale for its native session")
            c_rows.append(_row_to_c(row))
        c_rows_array = (CBatchRow * len(c_rows))(*c_rows)
        nonfinite = ctypes.c_int(-1)
        status = int(self._lib.gemma4_listener_batch_output_health(
            self._ctx,
            ctypes.c_uint64(generation),
            c_rows_array,
            ctypes.c_uint32(len(c_rows)),
            ctypes.byref(nonfinite),
        ))
        if status not in (0, 1):
            self._poisoned = True
            raise RuntimeError(
                "native batch output health/provenance check failed "
                f"(generation={generation}, rows={len(c_rows)}, status={status})"
            )
        if (status == 0 and nonfinite.value != 0) or (status == 1 and nonfinite.value == 0):
            self._poisoned = True
            raise RuntimeError("native output health status disagrees with its finite-guard value")
        has_nonfinite = status == 1
        self._last_health_generation = generation
        if has_nonfinite:
            self._poisoned = True
        row_status = "nonfinite_guard" if has_nonfinite else "healthy"
        return tuple(
            NativeBatchOutputHealth(
                generation=generation,
                row=row,
                checked=True,
                nonfinite=has_nonfinite,
                status=row_status,
            )
            for row in output.rows
        )

    def rollover_session_async(
        self,
        row: BatchRowDescriptor,
        *,
        retained_audio_tokens: int,
    ) -> None:
        self._assert_open()
        if self._active_output_generation is not None:
            raise RuntimeError("cannot roll over while a native output lease is active")
        binding = self._session_by_id.get(row.session_id)
        if binding is None or binding.native_context_id != row.native_context_id:
            raise RuntimeError("rollover row does not identify a bound native context")
        if (
            row.session_generation != binding.generation
            or row.state_epoch_id != binding.epoch_text
            or row.frame_id != binding.expected_frame_id
            or row.listener_state_version != binding.expected_state_version
            or row.sequence_position != binding.expected_sequence_position
            or row.cache_position != binding.cache_position
            or row.valid_history_tokens != binding.valid_history_tokens
        ):
            raise RuntimeError("rollover row is stale for its native session")
        if isinstance(retained_audio_tokens, bool) or not isinstance(retained_audio_tokens, int):
            raise ValueError("retained_audio_tokens must be an integer")
        if (
            retained_audio_tokens <= 0
            or retained_audio_tokens > LISTENER_MAX_ROLLOVER_AUDIO_TOKENS
            or retained_audio_tokens % LISTENER_STREAM_Q_LEN
        ):
            raise ValueError("retained_audio_tokens must be a positive Q=4 multiple up to 256")
        c_row = _row_to_c(row)
        status = int(self._lib.gemma4_listener_batch_rollover_async(
            self._ctx, ctypes.c_uint32(binding.slot), ctypes.byref(c_row),
            ctypes.c_uint32(retained_audio_tokens), self._stream_ptr(),
        ))
        if status != 0:
            self._poisoned = True
            raise RuntimeError(f"native session rollover failed with code {status}; batch quarantined")
        binding.cache_position = binding.prefix_len + retained_audio_tokens
        binding.valid_history_tokens = binding.cache_position

    def session_state(self, row: BatchRowDescriptor) -> NativeBatchSessionState:
        """Read a session's native cursor; the row must describe its current state."""
        self._assert_open()
        if self._active_output_generation is not None:
            raise RuntimeError("cannot snapshot session state while an output lease is active")
        binding = self._session_by_id.get(row.session_id)
        if binding is None or binding.native_context_id != row.native_context_id:
            raise RuntimeError("snapshot row does not identify a bound native context")
        c_row = _row_to_c(row)
        c_state = CBatchSessionState(struct_size=ctypes.sizeof(CBatchSessionState))
        status = int(self._lib.gemma4_listener_batch_get_session_state(
            self._ctx, ctypes.c_uint32(binding.slot), ctypes.byref(c_row),
            ctypes.byref(c_state),
        ))
        if status != 0:
            raise RuntimeError(f"native session snapshot preflight failed with code {status}")
        state = NativeBatchSessionState(
            slot=int(c_state.slot),
            native_context_id=int(c_state.native_context_id),
            session_generation=int(c_state.session_generation),
            state_epoch_id=int(c_state.state_epoch_id),
            expected_frame_id=int(c_state.expected_frame_id),
            expected_state_version=int(c_state.expected_state_version),
            expected_sequence_position=int(c_state.expected_sequence_position),
            prefix_len=int(c_state.prefix_len),
            current_position=int(c_state.current_position),
            valid_kv_len=int(c_state.valid_kv_len),
            max_seq_len=int(c_state.max_seq_len),
            primed=bool(c_state.primed),
        )
        expected = {
            "native_context_id": row.native_context_id,
            "session_generation": row.session_generation,
            "state_epoch_id": _epoch_id_u64(row.state_epoch_id),
            "expected_frame_id": row.frame_id,
            "expected_state_version": row.listener_state_version,
            "expected_sequence_position": row.sequence_position,
            "prefix_len": binding.prefix_len,
            "current_position": row.cache_position,
            "valid_kv_len": row.valid_history_tokens,
            "max_seq_len": LISTENER_MAX_SEQ_LEN,
        }
        mismatch = {
            name: (getattr(c_state, name), value)
            for name, value in expected.items()
            if int(getattr(c_state, name)) != value
        }
        if mismatch or not state.primed:
            raise RuntimeError(f"native session snapshot state mismatch: {mismatch}")
        if state.prefix_len not in BATCH_PREFIX_LENGTHS or not (
            state.prefix_len <= state.current_position <= state.max_seq_len
        ):
            raise RuntimeError("native snapshot reports invalid prefix or cache bounds")
        return state

    @staticmethod
    def _validate_cache_copy_rows(
        source_row: BatchRowDescriptor,
        destination_row: BatchRowDescriptor,
    ) -> None:
        if not isinstance(source_row, BatchRowDescriptor) or not isinstance(
            destination_row, BatchRowDescriptor
        ):
            raise TypeError("cache promotion requires BatchRowDescriptor rows")
        if source_row.session_id != destination_row.session_id:
            raise ValueError("cache promotion rows must describe the same listener session")
        if source_row.native_context_id == destination_row.native_context_id:
            raise ValueError("cache promotion rows must identify distinct native contexts")
        shared_boundary = (
            "session_generation", "state_epoch_id", "frame_id",
            "listener_state_version", "sequence_position",
        )
        mismatch = {
            name: (getattr(source_row, name), getattr(destination_row, name))
            for name in shared_boundary
            if getattr(source_row, name) != getattr(destination_row, name)
        }
        if mismatch:
            raise ValueError(f"cache promotion rows do not share a logical boundary: {mismatch}")
        if (
            source_row.valid_history_tokens != source_row.cache_position or
            destination_row.valid_history_tokens != destination_row.cache_position
        ):
            raise ValueError("cache promotion rows must describe complete valid KV cursors")

    def prepare_cache_copy_source(
        self,
        row: BatchRowDescriptor,
    ) -> NativeCacheCopySource:
        """Snapshot a shadow row and fence its stream on the shadow owner thread.

        The returned immutable token is safe to hand to the primary owner. The
        shadow owner must not step, retire, or otherwise mutate this row until
        the primary returns a ``NativeCacheCopyResult`` and the shadow stream
        waits on its completion event.
        """
        self._assert_open()
        state = self.session_state(row)
        if self._active_output_generation is not None:
            raise RuntimeError("cannot export a cache-copy source with an output lease")
        ready = torch.cuda.Event(enable_timing=False, blocking=False)
        ready.record(self.listener_stream)
        return NativeCacheCopySource(
            row=row,
            state=state,
            prefix_len=state.prefix_len,
            ready_event=ready,
            _owner=self,
        )

    def copy_session_cache_from(
        self,
        source: NativeCacheCopySource,
        destination_row: BatchRowDescriptor,
    ) -> NativeCacheCopyResult:
        """Queue an owner-ordered semantic-prefix/KV promotion from a shadow.

        The owner-produced source token crosses the shadow/primary thread
        boundary without calling methods on the shadow from the primary
        thread. The two rows identify the same logical listener boundary but
        distinct native contexts. The destination row describes its current
        CAS state; the returned row preserves that identity and logical
        boundary while taking the source cache cursor and prefix length. This
        method performs no host synchronization. The shadow owner must retain
        the token and wait on the returned completion event before reusing its
        source context.
        """
        self._assert_open()
        if not isinstance(source, NativeCacheCopySource):
            raise TypeError("source must be a NativeCacheCopySource from its owner thread")
        source_native = source._owner
        if not isinstance(source_native, NativeGemmaListenerMicrobatch):
            raise TypeError("cache-copy source token has no native listener owner")
        if source_native._closed or not source_native._ctx or source_native._poisoned:
            raise RuntimeError("cache-copy source owner is closed or poisoned")
        if not self._cache_copy_api_available or not source_native._cache_copy_api_available:
            raise RuntimeError("native DSO does not advertise cross-context cache promotion")
        if not isinstance(source.ready_event, torch.cuda.Event):
            raise TypeError("source token must contain a CUDA ready event")
        if source_native is self or self._ctx.value == source_native._ctx.value:
            raise ValueError("cache promotion requires distinct native owners")
        if self.library_path != source_native.library_path:
            raise ValueError("cache promotion requires the same native DSO")
        if self.runtime_signature != source_native.runtime_signature:
            raise ValueError("cache promotion requires identical runtime signatures")
        if self._model_device != source_native._model_device:
            raise ValueError("cache promotion requires source and destination on the same CUDA device")
        if self._stream_priority != -1 or source_native._stream_priority != 0:
            raise ValueError("cache promotion requires primary priority -1 and shadow priority 0")
        self._validate_cache_copy_rows(source.row, destination_row)

        source_state = source.state
        destination_state = self.session_state(destination_row)
        binding = self._session_by_id.get(destination_row.session_id)
        if binding is None or binding.native_context_id != destination_row.native_context_id:
            raise RuntimeError("destination session binding changed before cache promotion")
        if source.prefix_len != source_state.prefix_len or source.prefix_len not in BATCH_PREFIX_LENGTHS:
            raise RuntimeError("source snapshot reports an invalid semantic prefix length")
        if source_state.current_position < source_state.prefix_len or (
            source_state.current_position - source_state.prefix_len
        ) % LISTENER_STREAM_Q_LEN:
            raise RuntimeError("source snapshot cursor is not aligned to complete Q4 frames")
        if destination_state.current_position != destination_row.cache_position:
            raise RuntimeError("destination cache cursor changed before promotion")

        c_source_row = _row_to_c(source.row)
        c_destination_row = _row_to_c(destination_row)
        with torch.cuda.stream(self.listener_stream):
            # Event wait orders source writes before the primary stream's D2D
            # copies without making the host or shared model lock wait.
            self.listener_stream.wait_event(source.ready_event)
            status = int(self._lib.gemma4_listener_batch_copy_session_cache_async(
                self._ctx,
                ctypes.c_uint32(destination_state.slot),
                ctypes.byref(c_destination_row),
                source_native._ctx,
                ctypes.c_uint32(source_state.slot),
                ctypes.byref(c_source_row),
                self._stream_ptr(),
            ))
            if status != 0:
                if status == -3:
                    self._poisoned = True
                    # The native call may have queued a partial copy before
                    # detecting an asynchronous enqueue error; fence it before
                    # allowing the source owner to be released.
                    try:
                        self.listener_stream.synchronize()
                    except Exception:
                        pass
                raise RuntimeError(f"native session cache promotion failed with code {status}")
            completion = torch.cuda.Event(enable_timing=False, blocking=False)
            try:
                completion.record(self.listener_stream)
            except Exception:
                self._poisoned = True
                try:
                    self.listener_stream.synchronize()
                except Exception:
                    pass
                raise

        promoted_row = replace(
            destination_row,
            cache_position=source_state.current_position,
            valid_history_tokens=source_state.current_position,
        )
        binding.cache_position = source_state.current_position
        binding.valid_history_tokens = source_state.current_position
        binding.prefix_len = source_state.prefix_len
        return NativeCacheCopyResult(
            destination_row=promoted_row,
            prefix_len=source_state.prefix_len,
            completion_event=completion,
            _source=source,
        )

    def copy_session_kv_async(
        self,
        row: BatchRowDescriptor,
        layer_idx: int,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        *,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> torch.cuda.Event:
        """Enqueue one unique source-layer K/V snapshot and return its stream fence.

        This diagnostic API is not part of the serving hot path. The host may
        wait on the returned event after copying into caller-owned buffers.
        CUDA's event captures all earlier work submitted to the same stream;
        see the Runtime API event contract at
        https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html.
        """
        state = self.session_state(row)
        if self._active_output_generation is not None:
            raise RuntimeError("cannot copy session KV while an output lease is active")
        specs = self._source_kv_layer_specs()
        head_dim = specs.get(layer_idx)
        if head_dim is None:
            raise ValueError("layer_idx must name a unique non-shared KV source layer")
        expected_shape = (1, 1, state.current_position, head_dim)
        self._require_cuda_tensor(
            key_tensor, name="snapshot_key", dtype=torch.bfloat16,
            shape=expected_shape, device=self._model_device,
        )
        self._require_cuda_tensor(
            value_tensor, name="snapshot_value", dtype=torch.bfloat16,
            shape=expected_shape, device=self._model_device,
        )
        self._check_producer_stream(producer_stream)
        key_tensor.record_stream(self.listener_stream)
        value_tensor.record_stream(self.listener_stream)
        c_row = _row_to_c(row)
        key_meta = NativeGemmaListenerLM._tensor_meta(key_tensor, dtype_code=1)
        value_meta = NativeGemmaListenerLM._tensor_meta(value_tensor, dtype_code=1)
        with torch.cuda.stream(self.listener_stream):
            status = int(self._lib.gemma4_listener_batch_copy_session_kv_async(
                self._ctx, ctypes.c_uint32(state.slot), ctypes.byref(c_row),
                ctypes.c_uint32(layer_idx), ctypes.c_void_p(int(key_tensor.data_ptr())),
                ctypes.byref(key_meta), ctypes.c_void_p(int(value_tensor.data_ptr())),
                ctypes.byref(value_meta), self._stream_ptr(),
            ))
            if status != 0:
                self._poisoned = True
                raise RuntimeError(f"native session K/V snapshot enqueue failed with code {status}")
            completion = torch.cuda.Event(enable_timing=False, blocking=False)
            completion.record(self.listener_stream)
        return completion

    def snapshot_session(self, row: BatchRowDescriptor) -> NativeBatchSessionSnapshot:
        """Copy all unique source K/V layers to CPU evidence, one layer at a time."""
        state = self.session_state(row)
        specs = self._source_kv_layer_specs()
        copied: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_idx, head_dim in specs.items():
            with torch.cuda.stream(self.listener_stream):
                key = torch.empty(
                    (1, 1, state.current_position, head_dim),
                    dtype=torch.bfloat16, device=self._model_device,
                )
                value = torch.empty_like(key)
            event = self.copy_session_kv_async(
                row, layer_idx, key, value, producer_stream=self.listener_stream,
            )
            # Diagnostic gate only: retire one layer's temporary GPU buffers
            # before allocating the next. This is not used in serving.
            event.synchronize()
            copied[layer_idx] = (key.cpu().contiguous(), value.cpu().contiguous())
            del key, value
        final_state = self.session_state(row)
        if final_state != state:
            raise RuntimeError("native session state changed while its snapshot was copied")
        return NativeBatchSessionSnapshot(state=state, kv_by_layer=copied)

    def _source_kv_layer_specs(self) -> dict[int, int]:
        layers = getattr(self._language_model, "layers", None)
        if layers is None or len(layers) != 35:
            raise RuntimeError("loaded Gemma language model does not expose the attested 35-layer topology")
        specs: dict[int, int] = {}
        for index, layer in enumerate(layers):
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                raise RuntimeError(f"Gemma layer {index} has no self_attn topology")
            if bool(getattr(attention, "is_kv_shared_layer", False)):
                continue
            key_projection = getattr(attention, "k_proj", None)
            weight = getattr(key_projection, "weight", None)
            if weight is None or len(weight.shape) != 2:
                raise RuntimeError(f"Gemma source layer {index} has no attested K projection shape")
            specs[index] = int(weight.shape[0])
        if not specs:
            raise RuntimeError("Gemma topology has no unique KV source layers")
        return specs

    def release_session_async(self, row: BatchRowDescriptor) -> None:
        """Retire one SID's slot; its next session receives a fresh cache import."""
        self._assert_open()
        if self._active_output_generation is not None:
            raise RuntimeError("cannot retire a session while a native output lease is active")
        binding = self._session_by_id.get(row.session_id)
        if binding is None or binding.native_context_id != row.native_context_id:
            raise RuntimeError("retirement row does not identify a bound native context")
        if (
            row.session_generation != binding.generation
            or row.state_epoch_id != binding.epoch_text
            or row.frame_id != binding.expected_frame_id
            or row.listener_state_version != binding.expected_state_version
            or row.sequence_position != binding.expected_sequence_position
            or row.cache_position != binding.cache_position
            or row.valid_history_tokens != binding.valid_history_tokens
        ):
            raise RuntimeError("retirement row is stale for its native session")
        c_row = _row_to_c(row)
        status = int(self._lib.gemma4_listener_batch_release_session_async(
            self._ctx, ctypes.c_uint32(binding.slot), ctypes.byref(c_row), self._stream_ptr(),
        ))
        if status != 0:
            self._poisoned = True
            raise RuntimeError(f"native session slot retirement failed with code {status}; batch quarantined")
        del self._session_by_id[row.session_id]
        del self._session_by_context[row.native_context_id]
        self._import_refs.pop(binding.slot, None)
        if self._epoch_by_numeric.get(binding.epoch_numeric) == binding.epoch_text:
            del self._epoch_by_numeric[binding.epoch_numeric]

    def counters(self) -> dict[str, int]:
        self._assert_open()
        value = CBatchCounters()
        status = int(self._lib.gemma4_listener_batch_get_counters(self._ctx, ctypes.byref(value)))
        if status != 0:
            raise RuntimeError(f"native batch counter query failed with code {status}")
        return {
            name: int(getattr(value, name))
            for name, _ctype in CBatchCounters._fields_
        }

    @staticmethod
    def _probe_info_values(value: CBatchProbeInfo) -> dict[str, int | float]:
        return {
            name: (float(getattr(value, name)) if name == "rms_eps"
                   else int(getattr(value, name)))
            for name, _ctype in CBatchProbeInfo._fields_
            if name != "struct_size"
        }

    @staticmethod
    def _probe_host_buffer(size_bytes: int) -> ctypes.Array[Any]:
        if size_bytes <= 0:
            raise ValueError("diagnostic host buffer size must be positive")
        return ctypes.create_string_buffer(size_bytes)

    def probe_arm(self) -> None:
        """Arm the candidate-only N=2 RMS diagnostic for exactly one forward."""
        self._assert_open()
        self._require_probe_api()
        if self._active_output_generation is not None:
            raise RuntimeError("cannot arm numerical probe with an outstanding output lease")
        status = int(self._lib.gemma4_listener_batch_probe_arm(self._ctx))
        if status != 0:
            raise RuntimeError(f"native N=2 numerical probe arm failed with code {status}")
        self._last_probe_info = None

    def probe_read_rms(self) -> tuple[dict[str, int | float], dict[str, bytes]]:
        """Copy all eight RMS rows: X/B8/S4+S4, each contiguous [8,1536] BF16."""
        self._assert_open()
        self._require_probe_api()
        size_bytes = 8 * GEMMA4_HIDDEN_SIZE * 2
        buffers = [self._probe_host_buffer(size_bytes) for _ in range(3)]
        info = CBatchProbeInfo(struct_size=ctypes.sizeof(CBatchProbeInfo))
        status = int(self._lib.gemma4_listener_batch_probe_read_rms(
            self._ctx,
            ctypes.cast(buffers[0], ctypes.c_void_p), size_bytes,
            ctypes.cast(buffers[1], ctypes.c_void_p), size_bytes,
            ctypes.cast(buffers[2], ctypes.c_void_p), size_bytes,
            ctypes.byref(info),
        ))
        if status != 0:
            raise RuntimeError(f"native RMS diagnostic read failed with code {status}")
        self._last_probe_info = self._probe_info_values(info)
        return self._last_probe_info.copy(), {
            "input_bits": buffers[0].raw,
            "batched_bits": buffers[1].raw,
            "split_bits": buffers[2].raw,
        }

    def probe_mark_rms_exact(self, exact_match: bool) -> None:
        """Gate the follow-up projection diagnostic on a strict CPU bitwise pass."""
        self._assert_open()
        self._require_probe_api()
        if type(exact_match) is not bool:
            raise TypeError("exact_match must be a bool computed from raw BF16 bits")
        status = int(self._lib.gemma4_listener_batch_probe_mark_rms_exact(
            self._ctx, ctypes.c_uint32(1 if exact_match else 0),
        ))
        if status != 0:
            raise RuntimeError(f"native RMS diagnostic gate update failed with code {status}")

    def probe_qproj_async(self) -> None:
        """Compare replayed M=8 and two M=4 q_proj calls on the same eight X rows."""
        self._assert_open()
        self._require_probe_api()
        if self._last_probe_info is None or self._last_probe_info.get("state") != 2:
            raise RuntimeError("q_proj probe requires the captured RMS result first")
        status = int(self._lib.gemma4_listener_batch_probe_qproj_async(
            self._ctx, self._stream_ptr(),
        ))
        if status != 0:
            raise RuntimeError(f"native q_proj diagnostic launch failed with code {status}")

    def probe_read_qproj(self) -> tuple[dict[str, int | float], dict[str, bytes]]:
        """Copy all eight rows; q outputs are contiguous [8,q_dim] BF16."""
        self._assert_open()
        self._require_probe_api()
        if self._last_probe_info is None:
            raise RuntimeError("q_proj probe metadata is unavailable before RMS capture")
        q_dim = int(self._last_probe_info.get("q_dim", 0))
        if q_dim <= 0:
            raise RuntimeError("q_proj probe metadata has no valid output dimension")
        input_bytes = 8 * GEMMA4_HIDDEN_SIZE * 2
        q_bytes = 8 * q_dim * 2
        buffers = [self._probe_host_buffer(input_bytes)] + [
            self._probe_host_buffer(q_bytes) for _ in range(3)
        ]
        info = CBatchProbeInfo(struct_size=ctypes.sizeof(CBatchProbeInfo))
        status = int(self._lib.gemma4_listener_batch_probe_read_qproj(
            self._ctx,
            ctypes.cast(buffers[0], ctypes.c_void_p), input_bytes,
            ctypes.cast(buffers[1], ctypes.c_void_p), q_bytes,
            ctypes.cast(buffers[2], ctypes.c_void_p), q_bytes,
            ctypes.cast(buffers[3], ctypes.c_void_p), q_bytes,
            ctypes.byref(info),
        ))
        if status != 0:
            raise RuntimeError(f"native q_proj diagnostic read failed with code {status}")
        self._last_probe_info = self._probe_info_values(info)
        return self._last_probe_info.copy(), {
            "input_bits": buffers[0].raw,
            "actual_bits": buffers[1].raw,
            "batched_bits": buffers[2].raw,
            "split_bits": buffers[3].raw,
        }

    def _require_probe_api(self) -> None:
        if not self._probe_api_available:
            raise RuntimeError("native batch DSO does not advertise the optional numerical probe")

    def probe_trace_arm(self) -> None:
        """Arm one N=2 audio-processor/layer-0 operator trace."""
        self._assert_owner_thread()
        self._assert_open()
        self._require_trace_api()
        if self._active_output_generation is not None:
            raise RuntimeError("cannot arm operator trace with an outstanding output lease")
        status = int(self._lib.gemma4_listener_batch_trace_arm(self._ctx))
        if status != 0:
            raise RuntimeError(f"native operator trace arm failed with code {status}")

    @staticmethod
    def _trace_info_values(value: CBatchTraceInfo) -> dict[str, int | float | list[int]]:
        output: dict[str, int | float | list[int]] = {}
        for name, _ctype in CBatchTraceInfo._fields_:
            if name == "struct_size":
                continue
            field_value = getattr(value, name)
            if name in {"cache_start", "visible_len", "prefix_len", "base_position"}:
                output[name] = [int(item) for item in field_value]
            elif name in {"epsilon", "scalar"}:
                output[name] = float(field_value)
            else:
                output[name] = int(field_value)
        return output

    def probe_trace_read_all(self) -> list[dict[str, Any]]:
        """Copy every trace stage's captured inputs and batch actuals.

        The complete metadata pass enforces the native 100 MiB bound before
        allocating any host arrays. Reference arrays are intentionally empty:
        this diagnostic isolates native batch boundaries from split-reference
        execution so callers can compare against an independently run standard
        listener step.
        """
        self._assert_owner_thread()
        self._assert_open()
        self._require_trace_api()
        count = int(self._lib.gemma4_listener_batch_trace_stage_count())
        if count != len(TRACE_STAGE_NAMES):
            raise RuntimeError("native trace stage count changed after capability preflight")
        stage_infos: list[dict[str, int | float | list[int]]] = []
        total_bytes = 0
        for stage_id, stage_name in enumerate(TRACE_STAGE_NAMES):
            info = CBatchTraceInfo(struct_size=ctypes.sizeof(CBatchTraceInfo))
            status = int(self._lib.gemma4_listener_batch_trace_get_stage_info(
                self._ctx, ctypes.c_uint32(stage_id), ctypes.byref(info),
            ))
            if status != 0:
                raise RuntimeError(f"native trace metadata read failed for {stage_name}: {status}")
            values = self._trace_info_values(info)
            if values["stage_id"] != stage_id or values["row_count"] != 2 or values["q_len"] != 4:
                raise RuntimeError(f"native trace metadata identity mismatch for {stage_name}")
            if values["kind"] != TRACE_STAGE_KINDS[stage_id]:
                raise RuntimeError(f"native trace kind mismatch for {stage_name}")
            flags = int(values["flags"])
            required_flags = TRACE_FLAG_INPUTS_CAPTURED | TRACE_FLAG_ACTUAL_CAPTURED
            if flags & required_flags != required_flags:
                raise RuntimeError(f"native trace omitted captured inputs/actual for {stage_name}")
            for source, rows_key, cols_key in (
                ("input0_elements", "input0_rows", "input0_cols"),
                ("input1_elements", "input1_rows", "input1_cols"),
                ("input2_elements", "input2_rows", "input2_cols"),
                ("actual_elements", "actual_rows", "actual_cols"),
                ("reference_elements", "reference_rows", "reference_cols"),
            ):
                elements = int(values[source])
                rows = int(values[rows_key])
                cols = int(values[cols_key])
                if elements < 0 or (elements and (rows == 0 or cols == 0 or rows * cols != elements)):
                    raise RuntimeError(f"native trace shape/count mismatch for {stage_name}:{source}")
                if not elements and (rows or cols):
                    raise RuntimeError(f"native trace empty-array shape mismatch for {stage_name}:{source}")
                total_bytes += elements * 2
                if total_bytes > TRACE_MAX_RAW_BYTES:
                    raise RuntimeError("native operator trace exceeds the 100 MiB host-array bound")
            if not int(values["actual_elements"]) or not int(values["input0_elements"]):
                raise RuntimeError(f"native trace omitted required arrays for {stage_name}")
            reference_deferred = bool(flags & TRACE_FLAG_REFERENCE_DEFERRED)
            if reference_deferred and int(values["reference_elements"]):
                raise RuntimeError(f"deferred reference unexpectedly has storage for {stage_name}")
            if stage_name == "AUDIO_PLE_FRAME_PAD":
                expected_elements = LISTENER_STREAM_Q_LEN * int(values["actual_cols"])
                if (int(values["input0_rows"]) != LISTENER_STREAM_Q_LEN or
                        int(values["input0_cols"]) != int(values["actual_cols"]) or
                        int(values["input0_elements"]) != expected_elements):
                    raise RuntimeError("frame-pad trace must capture all four immutable source rows")
            if int(values["kind"]) == TRACE_KIND_GEMM:
                if int(values["gemm_actual_wrapper"]) != TRACE_GEMM_WRAPPER_BATCH_ROW_GEMM:
                    raise RuntimeError(f"N-row batch GEMM path is not attested for {stage_name}")
                if (not reference_deferred and
                        int(values["gemm_reference_wrapper"]) != TRACE_GEMM_WRAPPER_LEGACY_CUBLAS):
                    raise RuntimeError(f"legacy split GEMM reference path is not attested for {stage_name}")
                q4_opt_in = int(values["q4_gemv4_opt_in"])
                q4_eligible = [int(values[f"q4_gemv4_split{sid}_eligible"]) for sid in range(2)]
                q4_selected = [int(values[f"q4_gemv4_split{sid}_selected"]) for sid in range(2)]
                if q4_opt_in not in (0, 1) or any(value not in (0, 1) for value in q4_eligible + q4_selected):
                    raise RuntimeError(f"invalid Q4/GEMV dispatch metadata for {stage_name}")
                if (not reference_deferred and
                        (q4_selected != q4_eligible or (not q4_opt_in and any(q4_eligible)))):
                    raise RuntimeError(f"legacy Q4/GEMV selection does not match its recorded eligibility for {stage_name}")
            if int(values["kind"]) == TRACE_KIND_ATTENTION:
                attention_flags = TRACE_FLAG_REFERENCE_DEFERRED | TRACE_FLAG_ATTENTION_OUTPUT_ROW0 | TRACE_FLAG_ATTENTION_OUTPUT_ROW1
                if flags & attention_flags != attention_flags or not int(values["attention_mask_causal"]):
                    raise RuntimeError("attention trace lacks both row outputs, causal-mask metadata, or deferred-reference flag")
            else:
                if not reference_deferred and (
                        not int(values["reference_elements"]) or
                        flags & TRACE_FLAG_REFERENCE_CAPTURED == 0):
                    raise RuntimeError(f"native split reference is absent for {stage_name}")
            values["stage_name"] = stage_name
            stage_infos.append(values)

        def allocate(elements: int) -> ctypes.Array[Any] | None:
            byte_count = elements * 2
            return self._probe_host_buffer(byte_count) if byte_count else None

        def pointer(buffer: ctypes.Array[Any] | None) -> ctypes.c_void_p:
            return ctypes.cast(buffer, ctypes.c_void_p) if buffer is not None else ctypes.c_void_p()

        records: list[dict[str, Any]] = []
        for stage_id, metadata in enumerate(stage_infos):
            sizes = [int(metadata[key]) * 2 for key in (
                "input0_elements", "input1_elements", "input2_elements",
                "actual_elements", "reference_elements",
            )]
            buffers = [allocate(size // 2) for size in sizes]
            status = int(self._lib.gemma4_listener_batch_trace_read_stage(
                self._ctx, ctypes.c_uint32(stage_id),
                pointer(buffers[0]), ctypes.c_size_t(sizes[0]),
                pointer(buffers[1]), ctypes.c_size_t(sizes[1]),
                pointer(buffers[2]), ctypes.c_size_t(sizes[2]),
                pointer(buffers[3]), ctypes.c_size_t(sizes[3]),
                pointer(buffers[4]), ctypes.c_size_t(sizes[4]),
            ))
            if status != 0:
                raise RuntimeError(
                    f"native trace raw-array read failed for {metadata['stage_name']}: {status}"
                )
            raw = {
                "input0_bits": buffers[0].raw if buffers[0] is not None else b"",
                "input1_bits": buffers[1].raw if buffers[1] is not None else b"",
                "input2_bits": buffers[2].raw if buffers[2] is not None else b"",
                "actual_bits": buffers[3].raw if buffers[3] is not None else b"",
                "reference_bits": buffers[4].raw if buffers[4] is not None else b"",
            }
            records.append({"metadata": metadata, "raw": raw})
        return records

    def _require_trace_api(self) -> None:
        if not self._trace_api_available:
            raise RuntimeError("native batch DSO does not advertise the optional operator trace")

    def close(self) -> None:
        self._assert_owner_thread()
        if self._closed:
            return
        if self._ctx and self._active_output_generation is not None:
            status = int(self._lib.gemma4_listener_batch_output_abandon_async(
                self._ctx, ctypes.c_uint64(self._active_output_generation), self._stream_ptr(),
            ))
            if status != 0:
                raise RuntimeError(f"failed to abandon pending native batch output: {status}")
            self._active_output_generation = None
        if self._ctx:
            status = int(self._lib.gemma4_listener_batch_destroy(self._ctx))
            if status != 0:
                raise RuntimeError(f"gemma4_listener_batch_destroy failed with code {status}")
            self._ctx = ctypes.c_void_p()
        self._closed = True
        self._import_refs.clear()
        self._session_by_id.clear()
        self._session_by_context.clear()
        self._weight_tensors = ()
        self._language_model = None
        self._frame_pad_ple = None

    def __enter__(self) -> NativeGemmaListenerMicrobatch:
        self._assert_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "NativeGemmaListenerMicrobatch",
    "NativeBatchOutput",
    "NativeBatchOutputHealth",
    "NativeCacheCopySource",
    "NativeCacheCopyResult",
    "CBatchRow",
    "CBatchConfig",
    "CBatchRuntimeInfo",
    "CBatchCounters",
    "CBatchSessionState",
    "CBatchProbeInfo",
    "CBatchTraceInfo",
    "NativeBatchApiFeatures",
    "NativeBatchSessionState",
    "NativeBatchSessionSnapshot",
    "BATCH_API_VERSION",
    "BATCH_CAPABILITIES_V1",
    "BATCH_DYNAMIC_CAPABILITY",
    "BATCH_GEMM_GRAPH_CAPABILITY",
    "BATCH_CACHE_COPY_CAPABILITY",
    "BATCH_OUTPUT_HEALTH_CAPABILITY",
    "BATCH_PROBE_CAPABILITY",
    "BATCH_PROBE_API_VERSION",
    "BATCH_TRACE_CAPABILITY",
    "BATCH_TRACE_API_VERSION",
    "BATCH_SUPPORTED_MASK_ANY_POSITIVE_N",
    "TRACE_KIND_GEMM",
    "TRACE_KIND_RMS",
    "TRACE_KIND_ELEMENTWISE",
    "TRACE_KIND_ATTENTION",
    "TRACE_GEMM_WRAPPER_NONE",
    "TRACE_GEMM_WRAPPER_LEGACY_CUBLAS",
    "TRACE_GEMM_WRAPPER_BATCH_ROW_GEMM",
    "TRACE_STAGE_NAMES",
    "trace_reference_layout",
]
