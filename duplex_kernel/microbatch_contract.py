"""Immutable, fail-closed contracts for the opt-in listener microbatch path.

This module is deliberately independent of CUDA and the serving lifecycle. It
defines the batch-v2 contract for the single-owner GPU worker, including real
positive row counts and session metadata that must survive native capacity
growth.
"""
from __future__ import annotations

import hashlib
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np


SAMPLE_RATE_HZ = 16_000
FRAME_DURATION_MS = 80
PCM_SAMPLES_PER_FRAME = SAMPLE_RATE_HZ * FRAME_DURATION_MS // 1_000
PCM_TAIL_SAMPLES = 160
CONTROL_HIDDEN_WIDTH = 1_536
LISTENER_Q_PER_FRAME = 4
LISTENER_MAX_SEQ_LEN = 16_384
LISTENER_ATTENTION_BACKEND_ID = 3
LISTENER_RMSNORM_BACKEND_ID = 2
MEL_BINS = 128
MIN_BATCH_SIZE = 1
MAX_COALESCE_WAIT_NS = 2_000_000
CANONICAL_PREFIX_TOKENS = 34
LISTENER_STREAM_PRIORITY = -1
SUPPORTED_LISTENER_STREAM_PRIORITIES = frozenset({-1, 0})


def _strict_nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _sha256(name: str, value: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return normalized


@dataclass(frozen=True, slots=True)
class RuntimeSignature:
    """Identity of the one shared model/native runtime used by a batch."""

    model_id: str
    model_config_sha256: str
    checkpoint_sha256: str
    topology_sha256: str
    native_library_sha256: str
    native_abi_version: int
    batch_api_version: int
    attention_backend_id: int
    rmsnorm_backend_id: int
    max_seq_len: int
    stream_q_len: int

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        for name in (
            "model_config_sha256",
            "checkpoint_sha256",
            "topology_sha256",
            "native_library_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        for name in (
            "native_abi_version",
            "batch_api_version",
            "attention_backend_id",
            "rmsnorm_backend_id",
            "max_seq_len",
            "stream_q_len",
        ):
            value = _strict_nonnegative_int(name, getattr(self, name))
            if value == 0 and name in {
                "native_abi_version",
                "batch_api_version",
                "max_seq_len",
                "stream_q_len",
            }:
                raise ValueError(f"{name} must be positive")
        if self.max_seq_len != LISTENER_MAX_SEQ_LEN:
            raise ValueError(
                f"listener microbatch requires max_seq_len={LISTENER_MAX_SEQ_LEN}"
            )
        if self.stream_q_len != LISTENER_Q_PER_FRAME:
            raise ValueError(f"listener microbatch requires stream_q_len={LISTENER_Q_PER_FRAME}")
        if self.attention_backend_id != LISTENER_ATTENTION_BACKEND_ID:
            raise ValueError(
                "listener microbatch requires attention backend "
                f"{LISTENER_ATTENTION_BACKEND_ID}"
            )
        if self.rmsnorm_backend_id != LISTENER_RMSNORM_BACKEND_ID:
            raise ValueError(
                "listener microbatch requires RMSNorm backend "
                f"{LISTENER_RMSNORM_BACKEND_ID}"
            )


@dataclass(frozen=True, slots=True)
class NativeBatchCapability:
    """Capability attested by a separate batch-native ABI entry point."""

    api_version: int
    native_abi_version: int
    attention_backend_id: int
    rmsnorm_backend_id: int
    max_seq_len: int
    stream_q_len: int
    supports_arbitrary_batch_size: bool
    stream_priority: int
    stream_identity: str

    def __post_init__(self) -> None:
        for name in (
            "api_version",
            "native_abi_version",
            "attention_backend_id",
            "rmsnorm_backend_id",
            "max_seq_len",
            "stream_q_len",
        ):
            _strict_nonnegative_int(name, getattr(self, name))
        if self.api_version == 0:
            raise ValueError("batch API capability is missing")
        if self.native_abi_version == 0:
            raise ValueError("native ABI capability is missing")
        if self.max_seq_len != LISTENER_MAX_SEQ_LEN:
            raise ValueError(
                f"native batch runtime must preserve max_seq_len={LISTENER_MAX_SEQ_LEN}"
            )
        if self.stream_q_len != LISTENER_Q_PER_FRAME:
            raise ValueError(
                f"native batch runtime must preserve stream_q_len={LISTENER_Q_PER_FRAME}"
            )
        if self.attention_backend_id != LISTENER_ATTENTION_BACKEND_ID:
            raise ValueError(
                "native batch runtime must preserve attention backend "
                f"{LISTENER_ATTENTION_BACKEND_ID}"
            )
        if self.rmsnorm_backend_id != LISTENER_RMSNORM_BACKEND_ID:
            raise ValueError(
                "native batch runtime must preserve RMSNorm backend "
                f"{LISTENER_RMSNORM_BACKEND_ID}"
            )
        if not isinstance(self.supports_arbitrary_batch_size, bool) or not self.supports_arbitrary_batch_size:
            raise ValueError("native batch capability must support every positive batch size")
        if isinstance(self.stream_priority, bool) or not isinstance(self.stream_priority, int):
            raise ValueError("stream_priority must be an integer")
        if (
            isinstance(self.stream_priority, bool)
            or not isinstance(self.stream_priority, int)
            or self.stream_priority not in SUPPORTED_LISTENER_STREAM_PRIORITIES
        ):
            raise ValueError("listener batch stream priority must be -1 or 0")
        if not self.stream_identity.strip():
            raise ValueError("stream_identity must be non-empty")

    def validate_for(self, signature: RuntimeSignature) -> None:
        if self.api_version != signature.batch_api_version:
            raise ValueError("batch capability API version does not match runtime signature")
        if self.native_abi_version != signature.native_abi_version:
            raise ValueError("batch capability native ABI does not match runtime signature")
        if (
            self.attention_backend_id != signature.attention_backend_id
            or self.rmsnorm_backend_id != signature.rmsnorm_backend_id
        ):
            raise ValueError("batch capability backend IDs do not match runtime signature")
        if not self.supports_arbitrary_batch_size:
            raise ValueError("batch capability does not support arbitrary positive row counts")
        if self.max_seq_len != signature.max_seq_len:
            raise ValueError("batch capability max_seq_len does not match runtime signature")
        if self.stream_q_len != signature.stream_q_len:
            raise ValueError("batch capability stream_q_len does not match runtime signature")


@dataclass(frozen=True, slots=True, init=False)
class ListenerFrame:
    """One complete real input frame; PCM is stored as immutable little-endian bytes."""

    session_id: str
    generation_id: int
    state_epoch_id: str
    frame_id: int
    ready_at_ns: int
    reference_frame_id: int | None
    _pcm_f32le: bytes = field(repr=False)
    _reference_pcm_f32le: bytes | None = field(repr=False)
    pcm_sha256: str
    reference_pcm_sha256: str | None

    def __init__(
        self,
        *,
        session_id: str,
        generation_id: int,
        state_epoch_id: str,
        frame_id: int,
        ready_at_ns: int,
        pcm: np.ndarray,
        reference_pcm: np.ndarray | None = None,
        reference_frame_id: int | None = None,
    ) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        _strict_nonnegative_int("generation_id", generation_id)
        _strict_nonnegative_int("frame_id", frame_id)
        if not isinstance(state_epoch_id, str) or not state_epoch_id.strip():
            raise ValueError("state_epoch_id must be non-empty")
        _strict_nonnegative_int("ready_at_ns", ready_at_ns)
        samples = self._encode_pcm(pcm, name="pcm")
        reference: bytes | None = None
        if reference_pcm is None:
            if reference_frame_id is not None:
                raise ValueError("reference_frame_id requires reference_pcm")
        else:
            if reference_frame_id is None:
                raise ValueError("reference_pcm requires its source frame id")
            _strict_nonnegative_int("reference_frame_id", reference_frame_id)
            if reference_frame_id >= frame_id:
                raise ValueError("assistant reference must precede the current frame")
            reference = self._encode_pcm(reference_pcm, name="reference_pcm")

        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "generation_id", generation_id)
        object.__setattr__(self, "state_epoch_id", state_epoch_id)
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "ready_at_ns", ready_at_ns)
        object.__setattr__(self, "reference_frame_id", reference_frame_id)
        object.__setattr__(self, "_pcm_f32le", samples)
        object.__setattr__(self, "_reference_pcm_f32le", reference)
        object.__setattr__(self, "pcm_sha256", hashlib.sha256(samples).hexdigest())
        object.__setattr__(
            self,
            "reference_pcm_sha256",
            hashlib.sha256(reference).hexdigest() if reference is not None else None,
        )

    @staticmethod
    def _encode_pcm(pcm: np.ndarray, *, name: str) -> bytes:
        samples = np.asarray(pcm, dtype="<f4").reshape(-1)
        if samples.size != PCM_SAMPLES_PER_FRAME:
            raise ValueError(
                f"{name} must contain exactly {PCM_SAMPLES_PER_FRAME} samples"
            )
        if not np.isfinite(samples).all():
            raise ValueError(f"{name} must contain only finite samples")
        return np.ascontiguousarray(samples, dtype="<f4").tobytes()

    @property
    def pcm(self) -> np.ndarray:
        """Read-only view over the frame's immutable PCM snapshot."""
        return np.frombuffer(self._pcm_f32le, dtype="<f4")

    @property
    def reference_pcm(self) -> np.ndarray | None:
        if self._reference_pcm_f32le is None:
            return None
        return np.frombuffer(self._reference_pcm_f32le, dtype="<f4")


@dataclass(slots=True)
class ListenerSessionState:
    """Private per-session cursor and mutable frontend/native state.

    Registering a state transfers exclusive mutation ownership to the future
    GPU worker; callers must stop mutating it after registration. The worker
    updates fields only for the row it owns. Opaque cache/context fields are
    never shared between sessions, and observers use ListenerSessionSnapshot.
    """

    session_id: str
    generation_id: int
    state_epoch_id: str
    runtime_signature: RuntimeSignature
    next_frame_id: int = 0
    state_version: int = 0
    sequence_position: int = 0
    cache_position: int = CANONICAL_PREFIX_TOKENS
    valid_history_tokens: int = CANONICAL_PREFIX_TOKENS
    native_context_id: int | None = None
    user_pcm_tail: np.ndarray | None = field(default=None, repr=False)
    assistant_pcm_tail: np.ndarray | None = field(default=None, repr=False)
    user_mel_history: Any = field(default_factory=list, repr=False)
    assistant_mel_history: Any = field(default_factory=list, repr=False)
    attention_mask: Any | None = field(default=None, repr=False)
    native_kv_cache: Any | None = field(default=None, repr=False)
    native_context: Any | None = field(default=None, repr=False)
    closed: bool = False
    poisoned_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be non-empty")
        if not isinstance(self.state_epoch_id, str) or not self.state_epoch_id.strip():
            raise ValueError("state_epoch_id must be non-empty")
        if not isinstance(self.runtime_signature, RuntimeSignature):
            raise TypeError("runtime_signature is required for every listener session")
        for name in (
            "generation_id",
            "next_frame_id",
            "state_version",
            "sequence_position",
            "cache_position",
            "valid_history_tokens",
        ):
            _strict_nonnegative_int(name, getattr(self, name))
        if self.native_context_id is not None:
            if (
                isinstance(self.native_context_id, bool)
                or not isinstance(self.native_context_id, int)
                or self.native_context_id <= 0
            ):
                raise ValueError("native_context_id must be a positive opaque native handle")
        self.user_pcm_tail = self._copy_tail(self.user_pcm_tail, "user_pcm_tail")
        self.assistant_pcm_tail = self._copy_tail(self.assistant_pcm_tail, "assistant_pcm_tail")

    @staticmethod
    def _copy_tail(value: np.ndarray | None, name: str) -> np.ndarray | None:
        if value is None:
            return None
        tail = np.asarray(value, dtype="<f4").reshape(-1)
        if tail.size != PCM_TAIL_SAMPLES or not np.isfinite(tail).all():
            raise ValueError(f"{name} must be a finite {PCM_TAIL_SAMPLES}-sample tail")
        return np.array(tail, dtype="<f4", copy=True)


@dataclass(frozen=True, slots=True)
class StateReferenceSnapshot:
    """Safe metadata for opaque mutable state that cannot be copied on CPU."""

    type_name: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    device: str | None = None


@dataclass(frozen=True, slots=True)
class ListenerSessionSnapshot:
    """Read-only observer view; opaque cache/context contents are not exposed."""

    session_id: str
    generation_id: int
    state_epoch_id: str
    next_frame_id: int
    state_version: int
    sequence_position: int
    cache_position: int
    valid_history_tokens: int
    native_context_id: int | None
    user_pcm_tail: np.ndarray | None = field(repr=False)
    assistant_pcm_tail: np.ndarray | None = field(repr=False)
    user_mel_history: tuple[Any, ...] = field(repr=False)
    assistant_mel_history: tuple[Any, ...] = field(repr=False)
    attention_mask: Any = field(repr=False)
    native_kv_cache: Any = field(repr=False)
    native_context: Any = field(default=None, repr=False)
    closed: bool = False
    poisoned_reason: str | None = None


def _immutable_array_snapshot(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    # A bytes-backed view cannot be made writable again with setflags().
    immutable = array.tobytes(order="C")
    return np.frombuffer(immutable, dtype=array.dtype).reshape(array.shape)


def _reference_snapshot(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return _immutable_array_snapshot(value)
    if isinstance(value, tuple):
        return tuple(_reference_snapshot(item) for item in value)
    if isinstance(value, (list, deque)):
        return tuple(_reference_snapshot(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            (_reference_snapshot(key), _reference_snapshot(item))
            for key, item in value.items()
        )
    if isinstance(value, (set, frozenset)):
        return frozenset(_reference_snapshot(item) for item in value)
    # Do not call .cpu(), .item(), or inspect cache contents on a serving
    # thread.  Tensor/object values are represented only by host metadata.
    shape = getattr(value, "shape", None)
    normalized_shape = None
    if shape is not None:
        try:
            normalized_shape = tuple(int(dim) for dim in shape)
        except (TypeError, ValueError):
            normalized_shape = None
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    return StateReferenceSnapshot(
        type_name=f"{type(value).__module__}.{type(value).__qualname__}",
        shape=normalized_shape,
        dtype=str(dtype) if dtype is not None else None,
        device=str(device) if device is not None else None,
    )


def snapshot_listener_state(state: ListenerSessionState) -> ListenerSessionSnapshot:
    """Copy observable host metadata without exposing GPU-owned references."""
    user_history = _reference_snapshot(state.user_mel_history)
    assistant_history = _reference_snapshot(state.assistant_mel_history)
    return ListenerSessionSnapshot(
        session_id=state.session_id,
        generation_id=state.generation_id,
        state_epoch_id=state.state_epoch_id,
        next_frame_id=state.next_frame_id,
        state_version=state.state_version,
        sequence_position=state.sequence_position,
        cache_position=state.cache_position,
        valid_history_tokens=state.valid_history_tokens,
        native_context_id=state.native_context_id,
        user_pcm_tail=_immutable_array_snapshot(state.user_pcm_tail),
        assistant_pcm_tail=_immutable_array_snapshot(state.assistant_pcm_tail),
        user_mel_history=(
            user_history if isinstance(user_history, tuple) else (() if user_history is None else (user_history,))
        ),
        assistant_mel_history=(
            assistant_history
            if isinstance(assistant_history, tuple)
            else (() if assistant_history is None else (assistant_history,))
        ),
        attention_mask=_reference_snapshot(state.attention_mask),
        native_kv_cache=_reference_snapshot(state.native_kv_cache),
        native_context=(
            None if state.native_context is None else _reference_snapshot(state.native_context)
        ),
        closed=state.closed,
        poisoned_reason=state.poisoned_reason,
    )


@dataclass(frozen=True, slots=True)
class BatchRowDescriptor:
    session_id: str
    session_generation: int
    state_epoch_id: str
    frame_id: int
    listener_state_version: int
    sequence_position: int
    cache_position: int
    valid_history_tokens: int
    native_context_id: int
    input_frame_sha256: str
    reference_pcm_sha256: str | None
    frame_ready_at_ns: int
    reference_frame_id: int | None

    def __post_init__(self) -> None:
        if not self.session_id.strip() or not self.state_epoch_id.strip():
            raise ValueError("batch row requires session and epoch identities")
        for name in (
            "session_generation",
            "frame_id",
            "listener_state_version",
            "sequence_position",
            "cache_position",
            "valid_history_tokens",
            "native_context_id",
            "frame_ready_at_ns",
        ):
            _strict_nonnegative_int(name, getattr(self, name))
        if self.native_context_id == 0:
            raise ValueError("batch row requires a non-null native context handle")
        object.__setattr__(self, "input_frame_sha256", _sha256("input_frame_sha256", self.input_frame_sha256))
        if self.reference_pcm_sha256 is not None:
            object.__setattr__(
                self,
                "reference_pcm_sha256",
                _sha256("reference_pcm_sha256", self.reference_pcm_sha256),
            )
        if self.reference_frame_id is not None:
            _strict_nonnegative_int("reference_frame_id", self.reference_frame_id)
            if self.reference_frame_id >= self.frame_id:
                raise ValueError("batch row reference must precede its frame")


@dataclass(frozen=True, slots=True)
class ListenerBatchManifest:
    manifest_id: str
    runtime_signature: RuntimeSignature
    stream_identity: str
    stream_priority: int
    dispatch_at_ns: int
    max_coalesce_wait_ns: int
    dispatch_reason: str
    rows: tuple[BatchRowDescriptor, ...]
    padded_rows: int = 0

    def __post_init__(self) -> None:
        if not self.manifest_id:
            raise ValueError("manifest_id must be non-empty")
        if not isinstance(self.runtime_signature, RuntimeSignature):
            raise TypeError("manifest requires an immutable runtime signature")
        if (
            isinstance(self.stream_priority, bool)
            or not isinstance(self.stream_priority, int)
            or self.stream_priority not in SUPPORTED_LISTENER_STREAM_PRIORITIES
        ):
            raise ValueError("listener batch manifest priority must be -1 or 0")
        if not self.stream_identity:
            raise ValueError("stream_identity must be non-empty")
        _strict_nonnegative_int("dispatch_at_ns", self.dispatch_at_ns)
        _strict_nonnegative_int("max_coalesce_wait_ns", self.max_coalesce_wait_ns)
        if self.max_coalesce_wait_ns > MAX_COALESCE_WAIT_NS:
            raise ValueError("coalesce window may not exceed 2 ms")
        if self.dispatch_reason not in {
            "batch_full",
            "multiple_ready",
            "coalesce_deadline",
            "carryover_ready",
        }:
            raise ValueError(f"unsupported dispatch reason: {self.dispatch_reason!r}")
        rows = tuple(self.rows)
        if len(rows) < MIN_BATCH_SIZE:
            raise ValueError("batch must contain at least one real row")
        if len({row.session_id for row in rows}) != len(rows):
            raise ValueError("a manifest may contain at most one row per session")
        if self.padded_rows != 0:
            raise ValueError("listener microbatches may not contain padded or synthetic rows")
        object.__setattr__(self, "rows", rows)

    @property
    def batch_size(self) -> int:
        return len(self.rows)


@dataclass(frozen=True, slots=True)
class ListenerBatch:
    manifest: ListenerBatchManifest
    frames: tuple[ListenerFrame, ...]

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        if len(frames) != self.manifest.batch_size:
            raise ValueError("manifest/frame row count mismatch")
        for row, frame in zip(self.manifest.rows, frames, strict=True):
            if (
                row.session_id != frame.session_id
                or row.session_generation != frame.generation_id
                or row.state_epoch_id != frame.state_epoch_id
                or row.frame_id != frame.frame_id
                or row.input_frame_sha256 != frame.pcm_sha256
                or row.reference_pcm_sha256 != frame.reference_pcm_sha256
                or row.frame_ready_at_ns != frame.ready_at_ns
                or row.reference_frame_id != frame.reference_frame_id
            ):
                raise ValueError("manifest row does not describe its real input frame")
        object.__setattr__(self, "frames", frames)

    def pcm_batch(self) -> np.ndarray:
        """Return exact ``[N, 1280]`` real PCM rows; this method never pads."""
        result = np.stack([frame.pcm for frame in self.frames], axis=0)
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class RowStateUpdate:
    """Per-row cursor values returned only after that row's GPU work is complete."""

    session_id: str
    frame_id: int
    cache_position: int
    valid_history_tokens: int

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        for name in ("frame_id", "cache_position", "valid_history_tokens"):
            _strict_nonnegative_int(name, getattr(self, name))


@runtime_checkable
class CompletionFence(Protocol):
    """Small host-queryable view of a producer/consumer completion event."""

    def query(self) -> bool:
        """Return immediately; never block waiting for device completion."""


class OutputSlotBusy(RuntimeError):
    pass


class BatchOutputLease:
    """Lease for a native output slot whose storage is borrowed until all uses finish."""

    def __init__(
        self,
        *,
        generation: int,
        borrowed_output: Any,
        producer_fence: CompletionFence,
        state_lock: threading.RLock,
    ) -> None:
        self.generation = generation
        self.borrowed_output = borrowed_output
        self._producer_fence = producer_fence
        self._consumer_fences: list[CompletionFence] = []
        self._retired = False
        self._consumer_registration_in_progress = 0
        self._state_lock = state_lock

    def retain_consumer(
        self,
        clone_or_consume: Callable[[Any, CompletionFence], Any],
        *,
        completion_fence: CompletionFence,
    ) -> Any:
        """Schedule a consumer and retain its completion fence before slot reuse.

        The callback receives the producer fence so a cross-stream clone can
        enqueue its documented event dependency without synchronizing the host.
        """
        if not callable(getattr(completion_fence, "query", None)):
            raise TypeError("completion_fence must expose a nonblocking query()")
        # Registration and enqueue are one host-side critical section with slot
        # retirement. The callback must enqueue only; it must not wait for GPU
        # completion. A recorded-but-not-yet-enqueued fence must never be missed.
        with self._state_lock:
            if self._retired:
                raise RuntimeError("output lease has already been retired")
            self._consumer_fences.append(completion_fence)
            self._consumer_registration_in_progress += 1
            try:
                return clone_or_consume(self.borrowed_output, self._producer_fence)
            finally:
                self._consumer_registration_in_progress -= 1

    def is_reusable(self) -> bool:
        with self._state_lock:
            if self._retired:
                return True
            if self._consumer_registration_in_progress:
                return False
            try:
                return bool(self._producer_fence.query()) and all(
                    bool(fence.query()) for fence in self._consumer_fences
                )
            except BaseException:
                return False

    def _retire(self) -> None:
        with self._state_lock:
            if not self.is_reusable():
                raise OutputSlotBusy("borrowed output still has pending producer/consumer work")
            self._retired = True


class BatchOutputSlot:
    """CPU-testable guard for the native batch API's reusable borrowed output slot."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._generation = 0
        self._current: BatchOutputLease | None = None

    def publish(
        self,
        borrowed_output: Any,
        *,
        producer_fence: CompletionFence,
    ) -> BatchOutputLease:
        if not callable(getattr(producer_fence, "query", None)):
            raise TypeError("producer_fence must expose a nonblocking query()")
        with self._lock:
            if self._current is not None:
                self._current._retire()
            self._generation += 1
            lease = BatchOutputLease(
                generation=self._generation,
                borrowed_output=borrowed_output,
                producer_fence=producer_fence,
                state_lock=self._lock,
            )
            self._current = lease
            return lease
