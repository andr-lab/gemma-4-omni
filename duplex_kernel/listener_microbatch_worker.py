"""Single-owner execution worker for the opt-in listener microbatch path.

The serving process supplies already-loaded/frozen model objects and factories;
this module never loads checkpoints.  A dedicated host thread creates the
stream-bound frontend/native/policy resources, owns their lifecycle, and runs
one real N-row transaction at a time.  It deliberately has no per-row fallback.

Ingress/control messages use Python's synchronized FIFO ``queue.Queue``
contract (https://docs.python.org/3.13/library/queue.html). CUDA completion uses
the pinned Event record/synchronize API, with blocking restricted to the owner
thread (https://docs.pytorch.org/docs/2.12/generated/torch.cuda.Event.html).
``concurrent.futures.Future`` callbacks run in a process thread and can run
inline when the owner completes a future (https://docs.python.org/3.13/library/concurrent.futures.html).
Such callbacks must schedule work back to the serving loop and must not block
on another worker command.
"""
from __future__ import annotations

from concurrent.futures import Future
import copy
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import logging
import math
import os
import queue
import threading
import time
import uuid
from collections import OrderedDict, deque
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from duplex_kernel.backchannel_guard import BACKCHANNEL_MAX_TEXT_TOKENS
from duplex_kernel.gemma_native_control6_batch_policy import (
    DecisionRowSource,
    GemmaNativeControl6BatchPolicy,
    PolicyPreHeadContext,
)
from duplex_kernel.gemma_native_control6_policy import GemmaNativeControl6Policy
from duplex_kernel.listener_microbatch import ListenerMicrobatchCoordinator, RetiredSession
from duplex_kernel.listener_microbatch_frontend import ListenerMicrobatchAudioFrontend
from duplex_kernel.microbatch_contract import (
    BatchRowDescriptor,
    CONTROL_HIDDEN_WIDTH,
    ListenerBatch,
    ListenerBatchManifest,
    ListenerFrame,
    ListenerSessionSnapshot,
    ListenerSessionState,
    LISTENER_MAX_SEQ_LEN,
    LISTENER_Q_PER_FRAME,
    NativeBatchCapability,
    QueuedAudioDrainMetrics,
    RowStateUpdate,
    snapshot_listener_state,
)
from duplex_kernel.native_listener_microbatch import (
    MAX_STREAM_Q_LEN,
    LISTENER_STREAM_Q_LEN,
    NativeBatchOutput,
    NativeBatchOutputHealth,
)
from duplex_kernel.native_control6_prompt import NATIVE_CONTROL6_PREFIX_MAX_TOKENS
from duplex_kernel.turn_policy import ControlAction, PolicyDecision, PolicyInput

MAX_QUEUED_Q2_BATCH_FRAMES = 32
OPERATIONAL_CACHE_BUDGET_TOKENS = 4096
MEMORY_PREPARATION_TRIGGER_TOKENS = 2048
DESIRED_SWAP_THRESHOLD_TOKENS = 3072
NEW_CACHE_TARGET_TOKENS = 2048
INITIAL_PREFIX_MAX_TOKENS = 512
GENERATED_MEMORY_MAX_TOKENS = 256
RECENT_AUDIO_REFERENCE_FRAMES = 200
ASSISTANT_RESPONSE_MAX_TOKENS = 768
MAX_RECENT_LISTENER_Q2_FRAMES = RECENT_AUDIO_REFERENCE_FRAMES
LISTENER_COMPACTION_TRIGGER_TOKENS = MEMORY_PREPARATION_TRIGGER_TOKENS
LISTENER_COMPACTION_TARGET_TOKENS = NEW_CACHE_TARGET_TOKENS


class KvOwnerRole(str, Enum):
    CANONICAL = "canonical"
    ACTIVE = "active"
    WORK = "work"
    RETIRING = "retiring"


class KvWorkKind(str, Enum):
    REPLAN = "replan"
    SUMMARY = "summary"
    REBUILD = "rebuild"


class MemoryMaintenanceState(str, Enum):
    IDLE = "idle"
    SUMMARIZING = "summarizing"
    BUILDING = "building"
    CATCH_UP = "catch_up"
    READY = "ready"
    SWAP = "swap"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class MemoryMaintenanceHandle:
    handle_id: str
    session_id: str
    session_generation_id: int
    state_epoch_id: str
    work_kind: KvWorkKind
    created_at_ns: int


@dataclass(frozen=True, slots=True)
class MemoryPrefillChunk:
    """One bounded text quantum or one archived Q2 frame, never live PCM."""

    token_ids: tuple[int, ...] = ()
    audio_soft: Any = field(default=None, repr=False, compare=False)
    first_frame_id: int | None = None
    last_frame_id: int | None = None
    turn_id: int | None = None
    chunk_id: int | None = None
    revision: int | None = None
    turn_open: bool = False

    def __post_init__(self) -> None:
        from duplex_kernel.session import _archive_cpu_reference

        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        if self.audio_soft is not None:
            object.__setattr__(self, "audio_soft", _archive_cpu_reference(self.audio_soft))


@dataclass(frozen=True, slots=True)
class KvCacheBudgets:
    operational_cache_budget_tokens: int = OPERATIONAL_CACHE_BUDGET_TOKENS
    memory_preparation_trigger_tokens: int = MEMORY_PREPARATION_TRIGGER_TOKENS
    desired_swap_threshold_tokens: int = DESIRED_SWAP_THRESHOLD_TOKENS
    new_cache_target_tokens: int = NEW_CACHE_TARGET_TOKENS
    initial_prefix_max_tokens: int = INITIAL_PREFIX_MAX_TOKENS
    generated_memory_max_tokens: int = GENERATED_MEMORY_MAX_TOKENS
    recent_audio_reference_frames: int = RECENT_AUDIO_REFERENCE_FRAMES
    assistant_response_max_tokens: int = ASSISTANT_RESPONSE_MAX_TOKENS

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not (
            self.initial_prefix_max_tokens <= self.new_cache_target_tokens
            <= self.memory_preparation_trigger_tokens
            <= self.desired_swap_threshold_tokens
            <= self.operational_cache_budget_tokens <= LISTENER_MAX_SEQ_LEN
        ):
            raise ValueError("KV cache budgets must fit the ordered service thresholds and native ABI")
        if self.generated_memory_max_tokens > self.new_cache_target_tokens:
            raise ValueError("generated memory budget must fit the new cache target")
        if self.assistant_response_max_tokens > ASSISTANT_RESPONSE_MAX_TOKENS:
            raise ValueError("assistant response budget may not exceed 768 tokens")


@dataclass(frozen=True, slots=True)
class BranchTokenBudget:
    prefix_tokens: int
    transition_tokens: int
    frozen_assistant_prefix_tokens: int
    remaining_generation_tokens: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def total_tokens(self) -> int:
        return (self.prefix_tokens + self.transition_tokens
                + self.frozen_assistant_prefix_tokens + self.remaining_generation_tokens)


@dataclass(frozen=True, slots=True)
class KvWorkOwner:
    session_id: str
    session_generation_id: int
    session_epoch_id: str
    kv_owner_epoch: str
    kind: KvWorkKind
    memory_revision_id: str | int | None
    source_history_revision: str | int | None
    shadow_job: bool = False

    @property
    def kv_owner_role(self) -> KvOwnerRole:
        return KvOwnerRole.WORK


@dataclass(slots=True)
class _ResponseTokenBudget:
    limit: int
    consumed: int = 0

    @property
    def remaining(self) -> int:
        return self.limit - self.consumed


def gc_pause_overlap(
    samples: Sequence[Any], policy_start_ns: int, policy_end_ns: int
) -> tuple[float, bool]:
    """Return the GC pause duration intersecting one host policy interval."""
    start_ns = int(policy_start_ns)
    end_ns = int(policy_end_ns)
    if end_ns <= start_ns:
        return 0.0, False
    overlap_ns = 0
    for sample in samples:
        sample_start = getattr(sample, "start_ns", None)
        sample_end = getattr(sample, "end_ns", None)
        if (
            isinstance(sample_start, bool)
            or not isinstance(sample_start, int)
            or isinstance(sample_end, bool)
            or not isinstance(sample_end, int)
            or sample_end <= sample_start
        ):
            continue
        overlap_ns += max(0, min(end_ns, sample_end) - max(start_ns, sample_start))
    return overlap_ns / 1_000_000.0, overlap_ns > 0


def _perf_counter_ns() -> int:
    """Isolated clock hook for deterministic timing-contract tests."""
    return time.monotonic_ns()


def _unified_kv_integrity_trace_enabled() -> bool:
    """Read cursor diagnostics independently, retaining the legacy opt-in."""
    return (
        os.environ.get("DUPLEX_UNIFIED_KV_INTEGRITY_TRACE", "0").strip() == "1"
        or os.environ.get("DUPLEX_NATIVE_CONTROL6_TRACE", "0").strip() == "1"
    )


def _chunk_queued_q2_replay(
    replay: list[tuple[int, torch.Tensor]],
) -> tuple[tuple[tuple[int, torch.Tensor], ...], ...]:
    return tuple(
        tuple(replay[offset:offset + MAX_QUEUED_Q2_BATCH_FRAMES])
        for offset in range(0, len(replay), MAX_QUEUED_Q2_BATCH_FRAMES)
    )


def _normalize_stale_assistant_queue_rows(
    rows: Sequence[BatchRowDescriptor],
    *,
    queue_open_sessions: set[str],
    valid_sessions: set[str],
) -> tuple[tuple[BatchRowDescriptor, ...], set[str]]:
    """Drop queue ownership from valid Q0 rows that crossed a closed drain."""
    stale_sessions: set[str] = set()
    normalized_rows: list[BatchRowDescriptor] = []
    logger = logging.getLogger(__name__)
    for row in rows:
        if (
            row.queue_audio_for_next_turn
            and row.session_id in valid_sessions
            and row.session_id not in queue_open_sessions
        ):
            stale_sessions.add(row.session_id)
            logger.warning(
                "ignoring stale assistant-phase Q0 after queue closure: session=%s frame=%s",
                row.session_id,
                row.frame_id,
            )
            row = replace(
                row,
                queue_audio_for_next_turn=False,
                queue_audio_reason=None,
                queue_audio_onset_frame_id=None,
            )
        normalized_rows.append(row)
    return tuple(normalized_rows), stale_sessions


PrefixImportFactory = Callable[
    [ListenerSessionState, BatchRowDescriptor], "PrefixCacheImport"
]


@dataclass(frozen=True, slots=True)
class PrefixCacheImport:
    """One session's already-computed prefix cache for native slot import."""

    past_key_values: Any
    prefix_attention: torch.Tensor
    prefix_len: int
    cache_position: int | None = None
    semantic_context_revision: int | None = None
    semantic_replay_fallback_reason: str | None = None
    prefix_build_ms: float | None = None
    replay_ms: float | None = None
    ready_event: Any | None = None
    replay_frame_ids: tuple[int, ...] = ()
    fair_lock_wait_ms: float | None = None
    teacher_prefix_fenced_wall_ms: float | None = None
    applied_history_revision: str | int | None = None


@dataclass(frozen=True, slots=True)
class MemorySourceSnapshot:
    """Explicit private prefix and frozen source coverage, not the live KV."""

    prefix: PrefixCacheImport
    through_frame_id: int
    source_ranges: tuple[tuple[int, int], ...] = ()
    source_revisions: tuple[int, ...] = ()
    parent_memory_id: str | None = None
    eos_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_ranges", tuple(tuple(item) for item in self.source_ranges))
        object.__setattr__(self, "source_revisions", tuple(self.source_revisions))
        object.__setattr__(self, "eos_token_ids", tuple(self.eos_token_ids))
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (*self.source_revisions, *self.eos_token_ids)):
            raise ValueError("source revisions and EOS IDs must be nonnegative integers")
        if not isinstance(self.prefix, PrefixCacheImport):
            raise TypeError("memory sources require an explicit PrefixCacheImport")
        if self.through_frame_id < -1:
            raise ValueError("memory source cutoff must be at least -1")
        if len(self.source_ranges) != len(self.source_revisions):
            raise ValueError("memory source ranges and revisions must have equal lengths")
        previous = -1
        for first, last in self.source_ranges:
            if first <= previous or last < first or last > self.through_frame_id:
                raise ValueError("memory source ranges must be ordered within the cutoff")
            previous = last


@dataclass(slots=True)
class _MemoryMaintenanceJob:
    handle: MemoryMaintenanceHandle
    owner: KvWorkOwner
    sources: MemorySourceSnapshot
    row: BatchRowDescriptor
    covered_through_frame_id: int
    parent_memory_revision: Any
    pending_memory_tokens: deque[int] = field(default_factory=deque)
    memory_revision: Any = None
    logits: Any = None
    generated_token_ids: list[int] = field(default_factory=list)
    summary_done: bool = False
    last_ready_event: Any = None


@dataclass(slots=True)
class _RetiringMemoryOwner:
    session_id: str
    row: BatchRowDescriptor
    ready_event: Any = None


@dataclass(frozen=True, slots=True)
class TextAppendResult:
    """Native text append outputs fenced after the LM head on the owner stream."""

    logits: torch.Tensor
    final_norm_hidden: torch.Tensor
    ready_event: Any
    timings_ns: Mapping[str, int]
    cache_position: int
    prepared_transition_cache_hit: bool | None = None


@dataclass(frozen=True, slots=True)
class EphemeralDecodeHandle:
    """Opaque owner-issued identity for an isolated, bounded decode branch."""

    handle_id: str
    parent_session_id: str
    parent_generation_id: int
    parent_state_epoch_id: str
    source_cache_position: int
    fork_session_id: str
    fork_native_context_id: int
    max_tokens: int
    max_prompt_tokens: int = 0
    purpose: str = "backchannel"
    source_frame_id: int = 0
    source_state_version: int = 0
    source_sequence_position: int = 0
    response_id: str | None = None
    response_generation_id: int | None = None
    revision_id: str | None = None
    source_cache_epoch_id: int = 0
    source_history_revision: str | int | None = None
    kv_owner_epoch: str = ""
    memory_revision_id: str | int | None = None

    @property
    def session_id(self) -> str:
        return self.parent_session_id

    @property
    def session_generation_id(self) -> int:
        return self.parent_generation_id

    @property
    def kv_owner_role(self) -> KvOwnerRole:
        if self.purpose == "candidate_response":
            return KvOwnerRole.WORK
        return KvOwnerRole.ACTIVE

    @property
    def cache_position(self) -> int:
        """The immutable source cursor captured when this fork was created."""
        return self.source_cache_position


@dataclass(frozen=True, slots=True)
class PrivateTextAppendResult:
    """One private-branch append, with the branch cursor kept out of main state."""

    handle: EphemeralDecodeHandle
    cache_position: int
    token_count: int
    logits: torch.Tensor
    final_norm_hidden: torch.Tensor
    ready_event: Any
    timings_ns: Mapping[str, int]


@dataclass(slots=True)
class _EphemeralDecodeBranch:
    handle: EphemeralDecodeHandle
    row: BatchRowDescriptor
    token_count: int = 0
    transition_token_count: int = 0
    prompt_token_count: int = 0
    last_ready_event: Any | None = None


class ResponseDecodePurpose(str, Enum):
    """Private response branches that can coexist with the legacy backchannel."""

    ACTIVE = "active_response"
    CANDIDATE = "candidate_response"

    @classmethod
    def for_action(cls, action: ControlAction) -> ResponseDecodePurpose:
        """Keep response creation separate from the ephemeral BACKCHANNEL path."""
        if action is ControlAction.SPEAK:
            return cls.ACTIVE
        if action is ControlAction.REPLAN:
            return cls.CANDIDATE
        raise ValueError("only SPEAK or REPLAN can open a response decode branch")


@dataclass(frozen=True, slots=True)
class ListenerWatermark:
    """Exact completed listener identity captured before a private fork."""

    session_id: str
    generation_id: int
    state_epoch_id: str
    frame_id: int
    state_version: int
    sequence_position: int
    cache_position: int
    cache_epoch_id: int | None = None
    applied_history_revision: str | int | None = None


ResponseDecodeHandle = EphemeralDecodeHandle


@dataclass(frozen=True, slots=True)
class CompactionRequestContext:
    """Server-built confirmed history and exact token prefix for one rebase."""

    prefix_token_ids: tuple[int, ...]
    history_context: Any = None
    history_revision: str | int | None = None
    replay_after_frame_id: int | None = None

    def __post_init__(self) -> None:
        ids = tuple(self.prefix_token_ids)
        if not 1 <= len(ids) <= NATIVE_CONTROL6_PREFIX_MAX_TOKENS:
            raise ValueError(
                "compaction prefix token IDs must fit the native 1.."
                f"{NATIVE_CONTROL6_PREFIX_MAX_TOKENS} token import capacity"
            )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids):
            raise ValueError("compaction prefix token IDs must be non-negative integers")
        if self.history_revision is not None and (
            isinstance(self.history_revision, bool)
            or not isinstance(self.history_revision, (str, int))
            or (isinstance(self.history_revision, str) and not self.history_revision.strip())
            or (isinstance(self.history_revision, int) and self.history_revision < 0)
        ):
            raise ValueError("history_revision must be a non-empty string or non-negative integer")
        if self.replay_after_frame_id is not None and (
            isinstance(self.replay_after_frame_id, bool)
            or not isinstance(self.replay_after_frame_id, int)
            or self.replay_after_frame_id < 0
        ):
            raise ValueError("replay_after_frame_id must be a non-negative integer or None")
        object.__setattr__(self, "prefix_token_ids", ids)


@dataclass(frozen=True, slots=True)
class ListenerCompactionRequest:
    """Frozen shadow-worker input; recent Q2 replay is captured by the owner."""

    session_id: str
    state_epoch_id: str
    current_frame_id: int
    listener_state_version: int
    sequence_position: int
    semantic_context_revision: int
    prefix_token_ids: tuple[int, ...]
    prefix_len: int
    replay_q2: tuple[torch.Tensor, ...]
    replay_frame_ids: tuple[int, ...]
    cache_position: int
    history_context: Any = None
    history_revision: str | int | None = None
    replay_after_frame_id: int | None = None
    session_generation_id: int | None = None


@dataclass(frozen=True, slots=True)
class CanonicalListenerQ2Frame:
    """One committed, non-ephemeral Q2 row plus immutable PCM provenance."""

    frame: ListenerFrame
    row: BatchRowDescriptor
    audio_soft_tokens: torch.Tensor


CompactionRequestBuilder = Callable[
    [ListenerSessionSnapshot, tuple[CanonicalListenerQ2Frame, ...], str | None],
    CompactionRequestContext | None,
]


class CompactionBuilderCapacityError(RuntimeError):
    """The per-session history/Q2 proof cannot produce a safe compaction."""

    def __init__(self, reason: str):
        value = str(reason).strip()
        if not value:
            raise ValueError("compaction capacity reason must be non-empty")
        self.reason = value
        super().__init__(value)


class StaleListenerWatermarkError(RuntimeError):
    """Recoverable CAS rejection for a response fork against an old watermark."""


class StaleResponseRevisionError(RuntimeError):
    """Recoverable rejection when a newer response revision superseded a request."""


class RecoverableSessionCapacityError(RuntimeError):
    """One session cannot accept more listener KV yet; its frame was not queued."""

    def __init__(
        self,
        session_id: str,
        *,
        generation_id: int,
        frame_id: int,
        cache_position: int,
        requested_tokens: int,
        max_seq_len: int,
        reason: str | None = None,
    ) -> None:
        self.session_id = str(session_id)
        self.generation_id = int(generation_id)
        self.frame_id = int(frame_id)
        self.cache_position = int(cache_position)
        self.requested_tokens = int(requested_tokens)
        self.max_seq_len = int(max_seq_len)
        self.reason = None if reason is None else str(reason)
        reason_suffix = "" if self.reason is None else f"; compaction={self.reason}"
        super().__init__(
            "listener KV capacity reached for session "
            f"{self.session_id}: frame={self.frame_id}, cursor={self.cache_position}, "
            f"requested={self.requested_tokens}, max={self.max_seq_len}; "
            "input was not accepted and must be retried after compaction"
            f"{reason_suffix}"
        )


class _Q2DecisionViewUnavailable(RuntimeError):
    """Raised when queued onset audio cannot form a truthful scratch view."""


def _new_shadow_timing_stats() -> dict[str, float | int]:
    return {"sum_ms": 0.0, "count": 0, "max_ms": 0.0}


def _update_shadow_timing_stats(
    stats: dict[str, float | int], duration_ms: float
) -> None:
    elapsed_ms = max(0.0, float(duration_ms))
    stats["sum_ms"] = round(float(stats["sum_ms"]) + elapsed_ms, 3)
    stats["count"] = int(stats["count"]) + 1
    stats["max_ms"] = round(max(float(stats["max_ms"]), elapsed_ms), 3)


def _empty_semantic_shadow_diagnostics() -> Mapping[str, Any]:
    return {
        "current_job": {
            "session_id": None,
            "listener_generation_id": None,
            "state_epoch_id": None,
            "semantic_context_revision": None,
            "request_base_frame_id": None,
            "phase": "idle",
            "stage_requested_at_ns": None,
            "stage_queued_at_ns": None,
            "stage_execution_started_at_ns": None,
            "fair_lock_wait_ms": None,
            "teacher_prefix_build_wall_ms": None,
            "teacher_prefix_replay_wall_ms": None,
            "teacher_prefix_fenced_wall_ms": None,
            "shadow_native_prefix_import_ms": None,
        },
        "q2": {
            "submitted_count": 0,
            "completed_count": 0,
            "last_submitted_session_id": None,
            "last_submitted_frame_id": None,
            "last_completed_session_id": None,
            "last_completed_frame_id": None,
            "pending_depth": 0,
            "max_pending_depth": 0,
            "timings": {
                name: _new_shadow_timing_stats()
                for name in ("wait", "fence", "output_health_validation")
            },
        },
        "candidate_cursor_rejections": 0,
        "candidate_last_rejection_reason": None,
        "promotion_preflight_rejections": 0,
        "promotion_preflight_last_rejection_reason": None,
        "promotion_copy_rejections": 0,
        "promotion_copy_last_rejection_reason": None,
        "cache_candidate_copy_enqueue": _new_shadow_timing_stats(),
    }


@dataclass(frozen=True, slots=True)
class PreparedInitialPrefix:
    """Off-owner semantic KV build awaiting an owner-thread private-slot import.

    The request is prepared from an immutable session snapshot on a separate
    CUDA stream.  This envelope carries the event dependency and the stable
    listener-state CAS; it never represents an input audio frame.
    """

    prefix: PrefixCacheImport
    ready_event: Any
    session_id: str
    listener_generation_id: int
    state_epoch_id: str
    semantic_context_revision: int
    expected_cache_position: int
    frame_id: int = 0
    state_version: int = 0
    sequence_position: int = 0
    applied_history_revision: str | int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prefix, PrefixCacheImport):
            raise TypeError("prefix must be PrefixCacheImport")
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be non-empty")
        if not isinstance(self.state_epoch_id, str) or not self.state_epoch_id.strip():
            raise ValueError("state_epoch_id must be non-empty")
        for name in (
            "listener_generation_id", "semantic_context_revision",
            "expected_cache_position", "frame_id", "state_version",
            "sequence_position",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


CacheIdentityCommitter = Callable[
    [str, int, int | None, str | int | None, Callable[[], None]], None
]


@dataclass(frozen=True, slots=True)
class InitialPrefixAdmission:
    """Owner result for initial semantic-prefix admission."""

    status: str
    reason: str | None = None
    prefix_len: int | None = None
    semantic_context_revision: int | None = None
    import_ms: float | None = None


@dataclass(slots=True)
class ListenerMicrobatchRuntime:
    """Dependencies created by ``runtime_factory`` on the owner thread.

    ``policy_inputs`` supplies only the ordinary per-row policy metadata.  The
    hidden tensor is bound by ``step_native_output_batch`` from the native
    output and cannot be supplied by this callback.  ``completion_barrier`` is
    a CPU-test hook; CUDA execution always records and synchronizes a final
    event on the dedicated owner thread itself.
    """

    listener_stream: Any
    frontend: ListenerMicrobatchAudioFrontend
    native: Any
    policy: GemmaNativeControl6BatchPolicy | None
    policy_inputs: Callable[
        [ListenerBatch, tuple[ListenerSessionState, ...]],
        Sequence[PolicyInput | PolicyPreHeadContext],
    ] | None
    listener_model_lock: Any | None = None
    completion_barrier: Callable[[Any], None] | None = None
    close_resources: Callable[[], None] | None = None
    prefix_importer: PrefixImportFactory | None = None
    control6_scorer: GemmaNativeControl6BatchPolicy | None = None
    shadow_factory: Callable[[], "SemanticPrefixShadowRuntime"] | None = None
    shadow_prefix_builder: Callable[[Any, Any], PrefixCacheImport] | None = None
    token_adapter: Any | None = None
    compaction_request_builder: CompactionRequestBuilder | None = None
    compaction_trigger_tokens: int | None = None
    compaction_target_tokens: int | None = None
    kv_budgets: KvCacheBudgets = field(default_factory=KvCacheBudgets)
    memory_token_encoder: Callable[[str], Sequence[int]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kv_budgets, KvCacheBudgets):
            raise TypeError("kv_budgets must be KvCacheBudgets")
        if self.compaction_trigger_tokens is None:
            self.compaction_trigger_tokens = self.kv_budgets.memory_preparation_trigger_tokens
        if self.compaction_target_tokens is None:
            self.compaction_target_tokens = self.kv_budgets.new_cache_target_tokens


@dataclass(slots=True)
class SemanticPrefixShadowRuntime:
    """Priority-0 native owner used only for semantic-prefix staging."""

    native: Any
    listener_stream: Any


@dataclass(frozen=True, slots=True)
class ShadowPrefixStage:
    """Frozen semantic request paired with the stable listener cursor."""

    request: Any
    listener_generation_id: int
    sequence_position: int
    requested_at_ns: int
    queued_at_ns: int | None = None
    execution_started_at_ns: int | None = None


@dataclass(frozen=True, slots=True)
class _PrimaryAppliedSemanticPrefix:
    session_id: str
    listener_generation_id: int
    state_epoch_id: str
    semantic_context_revision: int
    request_current_frame_id: int | None
    request_current_frame_source: str
    current_frame_id: int
    listener_state_version: int
    prefix_len: int
    replay_frame_ids: tuple[int, ...]

    @property
    def identity_key(self) -> tuple[str, int, str]:
        return (self.session_id, self.listener_generation_id, self.state_epoch_id)

    def status(self) -> Mapping[str, Any]:
        return {
            "session_id": self.session_id,
            "listener_generation_id": self.listener_generation_id,
            "state_epoch_id": self.state_epoch_id,
            "semantic_context_revision": self.semantic_context_revision,
            "request_current_frame_id": self.request_current_frame_id,
            "request_current_frame_source": self.request_current_frame_source,
            "current_frame_id": self.current_frame_id,
            "listener_state_version": self.listener_state_version,
            "prefix_len": self.prefix_len,
            "replay_count": len(self.replay_frame_ids),
            "replay_first_q2_frame_id": (
                self.replay_frame_ids[0] if self.replay_frame_ids else None
            ),
            "replay_last_q2_frame_id": (
                self.replay_frame_ids[-1] if self.replay_frame_ids else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ShadowQ2Frame:
    """One owned real frontend Q2 and its pre-step listener provenance."""

    frame: ListenerFrame
    row: BatchRowDescriptor
    audio_soft: torch.Tensor
    ready_event: Any | None


@dataclass(frozen=True, slots=True)
class ShadowPrefixCandidate:
    """A shadow slot caught up to an exact active listener cursor."""

    candidate_id: int
    stage: ShadowPrefixStage
    prefix: PrefixCacheImport
    native: Any
    source_token: Any
    row: BatchRowDescriptor
    ready_event: Any | None
    catchup_frames: int
    caught_up_at_ns: int


@dataclass(frozen=True, slots=True)
class _PendingShadowPromotion:
    candidate: ShadowPrefixCandidate
    copy_result: Any
    copy_enqueue_ms: float


class SemanticPrefixShadowWorker:
    """Build and replay semantic KV on a reusable, lower-priority owner.

    The shadow slot is single-session and private.  It consumes only Q2 values
    emitted by the real frontend, never re-encodes PCM, and publishes a cache
    only when its row cursor can be compared with the active owner.  Promotion
    is deliberately performed by the primary owner through the native cache
    copy API.
    """

    def __init__(
        self,
        runtime_factory: Callable[[], SemanticPrefixShadowRuntime],
        prefix_builder: Callable[[Any, Any], PrefixCacheImport],
        *,
        runtime_signature: Any | None = None,
        max_pending_q2: int = 64,
        on_promotion_fenced: Callable[[ShadowPrefixCandidate], None] | None = None,
    ) -> None:
        if not callable(runtime_factory) or not callable(prefix_builder):
            raise TypeError("shadow runtime and prefix builder must be callable")
        if isinstance(max_pending_q2, bool) or not isinstance(max_pending_q2, int) or max_pending_q2 < 1:
            raise ValueError("max_pending_q2 must be a positive integer")
        if on_promotion_fenced is not None and not callable(on_promotion_fenced):
            raise TypeError("on_promotion_fenced must be callable or None")
        self._runtime_factory = runtime_factory
        self._prefix_builder = prefix_builder
        self._runtime_signature = runtime_signature
        self._max_pending_q2 = max_pending_q2
        self._on_promotion_fenced = on_promotion_fenced
        self._condition = threading.Condition()
        self._desired: dict[str, ShadowPrefixStage] = {}
        self._desired_order: deque[str] = deque()
        self._queued_stage_ids: set[str] = set()
        self._primary_applied: dict[
            tuple[str, int, str], _PrimaryAppliedSemanticPrefix
        ] = {}
        self._shadow_promoted_identities: set[tuple[Any, ...]] = set()
        # A single shadow slot may be busy with another session while real
        # primary ticks continue. Keep each staged session's contiguous Q2
        # suffix independently so it can start from its frozen watermark.
        self._q2: dict[str, deque[ShadowQ2Frame]] = {}
        self._controls: deque[tuple[str, Any, Any]] = deque()
        self._thread: threading.Thread | None = None
        self._runtime: SemanticPrefixShadowRuntime | None = None
        self._startup: Future[None] = Future()
        self._stopped: Future[None] = Future()
        self._stop_requested = False
        self._active_stage: ShadowPrefixStage | None = None
        self._active_row: BatchRowDescriptor | None = None
        self._active_prefix: PrefixCacheImport | None = None
        self._active_ready_event: Any | None = None
        self._active_catchup_frames = 0
        self._active_session_id: str | None = None
        self._candidate: ShadowPrefixCandidate | None = None
        self._candidate_claimed = False
        self._q2_processing = False
        self._claimed_promotion: tuple[ShadowPrefixCandidate, Any] | None = None
        self._next_candidate_id = 1
        self._next_context_id = 1
        self._copy_leases: list[tuple[Any, Any]] = []
        self._latest_error: str | None = None
        self._promotions_completed = 0
        self._diagnostic_stage: ShadowPrefixStage | None = None
        self._diagnostic_phase = "idle"
        self._diagnostic_fair_lock_wait_ms: float | None = None
        self._diagnostic_prefix_build_wall_ms: float | None = None
        self._diagnostic_prefix_replay_wall_ms: float | None = None
        self._diagnostic_teacher_prefix_fenced_wall_ms: float | None = None
        self._diagnostic_native_prefix_import_ms: float | None = None
        self._q2_submitted_count = 0
        self._q2_completed_count = 0
        self._q2_last_submitted_session_id: str | None = None
        self._q2_last_submitted_frame_id: int | None = None
        self._q2_last_completed_session_id: str | None = None
        self._q2_last_completed_frame_id: int | None = None
        self._q2_max_pending_depth = 0
        self._q2_timing_stats = {
            name: _new_shadow_timing_stats()
            for name in ("wait", "fence", "output_health_validation")
        }
        self._candidate_cursor_rejections = 0
        self._candidate_last_rejection_reason: str | None = None
        self._promotion_preflight_rejections = 0
        self._promotion_preflight_last_rejection_reason: str | None = None
        self._promotion_copy_rejections = 0
        self._promotion_copy_last_rejection_reason: str | None = None
        self._cache_candidate_copy_enqueue_stats = _new_shadow_timing_stats()

    def start(self) -> Future[None]:
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("semantic shadow worker was already started")
            self._thread = threading.Thread(
                target=self._run,
                name="listener-semantic-shadow-owner",
                daemon=False,
            )
            self._thread.start()
        return self._startup

    def stage(
        self,
        request: Any,
        *,
        listener_generation_id: int,
        sequence_position: int,
    ) -> bool:
        """Replace the pending snapshot for a session without blocking a tick."""
        stage = ShadowPrefixStage(
            request=request,
            listener_generation_id=_strict_nonnegative_int_value(
                "listener_generation_id", listener_generation_id
            ),
            sequence_position=_strict_nonnegative_int_value(
                "sequence_position", sequence_position
            ),
            requested_at_ns=time.monotonic_ns(),
        )
        _validate_shadow_stage(stage)
        with self._condition:
            if self._stop_requested or self._thread is None:
                return False
            session_id = str(request.session_id)
            previous = self._desired.get(session_id)
            if session_id in self._queued_stage_ids and previous is not None:
                stage = replace(stage, queued_at_ns=previous.queued_at_ns)
            identity_changed = (
                previous is None
                or _shadow_stage_identity(previous) != _shadow_stage_identity(stage)
            )
            if self._stage_suppressed_locked(stage):
                if (
                    previous is not None
                    and _shadow_stage_identity(previous) == _shadow_stage_identity(stage)
                ):
                    self._remove_desired_stage_locked(session_id)
                    self._q2.pop(session_id, None)
                self._condition.notify_all()
                return True
            self._desired[session_id] = stage
            if (
                not self._stage_suppressed_locked(stage)
                and not self._stage_already_promoted_locked(stage)
                and session_id not in self._queued_stage_ids
                and (
                    identity_changed
                    and (
                        session_id == self._active_session_id
                        or (self._candidate is not None and self._candidate.row.session_id == session_id)
                    )
                    or session_id != self._active_session_id
                    and (self._candidate is None or self._candidate.row.session_id != session_id)
                )
            ):
                self._queue_stage_locked(session_id)
            elif self._active_stage is None and self._candidate is None:
                self._select_queued_diagnostic_job_locked()
            self._condition.notify_all()
        return True

    def mark_primary_applied(
        self,
        request: Any,
        *,
        listener_generation_id: int,
        current_frame_id: int | None = None,
        listener_state_version: int | None = None,
    ) -> None:
        """Record a prefix only after its primary cache/state commit succeeds.

        This is a CPU-only notification path. It may prune queued work under
        the condition lock, but native retirement remains on the shadow owner.
        """
        session_id = str(getattr(request, "session_id", ""))
        state_epoch_id = str(getattr(request, "state_epoch_id", ""))
        if not session_id.strip() or not state_epoch_id.strip():
            raise ValueError("primary-applied prefix needs session and state epoch IDs")
        generation = _strict_nonnegative_int_value(
            "listener_generation_id", listener_generation_id
        )
        revision = _strict_nonnegative_int_value(
            "semantic_context_revision", getattr(request, "semantic_context_revision", None)
        )
        request_frame = getattr(request, "current_frame_id", None)
        request_frame_source = "semantic_prefix_request" if request_frame is not None else "unknown"
        if request_frame is None:
            request_frame = getattr(request, "semantic_prefix_request_current_frame_id", None)
            if request_frame is not None:
                request_frame_source = "committed_row_prefix_metadata"
        if request_frame is not None:
            request_frame = _strict_nonnegative_int_value(
                "request_current_frame_id", request_frame
            )
        committed_frame = _strict_nonnegative_int_value(
            "current_frame_id",
            request_frame if current_frame_id is None else current_frame_id,
        )
        state_version = _strict_nonnegative_int_value(
            "listener_state_version",
            getattr(request, "listener_state_version", 0)
            if listener_state_version is None else listener_state_version,
        )
        prefix_len = _strict_nonnegative_int_value(
            "prefix_len",
            getattr(request, "prefix_len", getattr(request, "semantic_prefix_len", None)),
        )
        replay_frame_ids = tuple(
            _strict_nonnegative_int_value("replay_frame_id", value)
            for value in tuple(
                getattr(
                    request,
                    "replay_frame_ids",
                    getattr(request, "semantic_prefix_replay_frame_ids", ()),
                ) or ()
            )
        )
        applied = _PrimaryAppliedSemanticPrefix(
            session_id=session_id,
            listener_generation_id=generation,
            state_epoch_id=state_epoch_id,
            semantic_context_revision=revision,
            request_current_frame_id=request_frame,
            request_current_frame_source=request_frame_source,
            current_frame_id=committed_frame,
            listener_state_version=state_version,
            prefix_len=prefix_len,
            replay_frame_ids=replay_frame_ids,
        )
        with self._condition:
            previous = self._primary_applied.get(applied.identity_key)
            if (
                previous is None
                or applied.semantic_context_revision >= previous.semantic_context_revision
            ):
                self._primary_applied[applied.identity_key] = applied
            current = self._primary_applied.get(applied.identity_key, applied)
            self._prune_primary_obsolete_stage_locked(
                current.session_id,
                current.listener_generation_id,
                current.state_epoch_id,
                current.semantic_context_revision,
            )
            self._condition.notify_all()

    def publish_q2(self, value: ShadowQ2Frame) -> bool:
        """Offer an owned real Q2 to any staged session, including queued ones."""
        if not isinstance(value, ShadowQ2Frame):
            raise TypeError("value must be ShadowQ2Frame")
        with self._condition:
            stage = self._desired.get(value.frame.session_id)
            session_id = value.frame.session_id
            active_row = self._active_row if self._active_session_id == session_id else None
            candidate_row = (
                self._candidate.row
                if self._candidate is not None and self._candidate.row.session_id == session_id
                else None
            )
            expected_frame_id = (
                active_row.frame_id
                if active_row is not None
                else candidate_row.frame_id
                if candidate_row is not None
                else int(stage.request.current_frame_id) if stage is not None else -1
            )
            if (
                stage is None
                or self._stage_suppressed_locked(stage)
                or self._stage_already_promoted_locked(stage)
                or int(value.frame.frame_id) < expected_frame_id
            ):
                return False
            pending = self._q2.setdefault(session_id, deque())
            if len(pending) >= self._max_pending_q2:
                self._latest_error = (
                    f"shadow Q2 backlog exceeded {self._max_pending_q2} rows for "
                    f"{session_id}"
                )
                self._desired.pop(session_id, None)
                self._desired_order = deque(
                    sid for sid in self._desired_order if sid != session_id
                )
                self._queued_stage_ids.discard(session_id)
                self._q2.pop(session_id, None)
                self._controls.append(("abort", session_id, self._latest_error))
                self._condition.notify_all()
                return False
            if pending and value.frame.frame_id <= pending[-1].frame.frame_id:
                # Duplicate publication is harmless only when it is the same
                # committed row; replacement would make Q2 provenance ambiguous.
                if value.frame.frame_id == pending[-1].frame.frame_id and value.row == pending[-1].row:
                    return True
                self._latest_error = f"shadow Q2 rows are not strictly ordered for {session_id}"
                self._controls.append(("abort", session_id, self._latest_error))
                self._condition.notify_all()
                return False
            pending.append(value)
            self._q2_submitted_count += 1
            self._q2_last_submitted_session_id = str(value.frame.session_id)
            self._q2_last_submitted_frame_id = int(value.frame.frame_id)
            pending_depth = sum(len(rows) for rows in self._q2.values())
            self._q2_max_pending_depth = max(self._q2_max_pending_depth, pending_depth)
            if (
                self._active_session_id == session_id
                and not self._q2_processing
                and not self._candidate_claimed
            ):
                self._set_diagnostic_phase_locked("q2_pending", self._active_stage)
            self._condition.notify_all()
        return True

    def needs_q2(self, session_id: str, frame_id: int) -> bool:
        """Return whether a real frontend row can advance this staged request."""
        with self._condition:
            stage = self._desired.get(session_id)
            return bool(
                stage is not None
                and not self._stage_suppressed_locked(stage)
                and not self._stage_already_promoted_locked(stage)
                and int(frame_id)
                >= (
                    self._active_row.frame_id
                    if self._active_session_id == session_id and self._active_row is not None
                    else self._candidate.row.frame_id
                    if self._candidate is not None and self._candidate.row.session_id == session_id
                    else int(stage.request.current_frame_id)
                )
            )

    def candidate_for(
        self,
        destination_row: BatchRowDescriptor,
        *,
        semantic_context_revision: int | None = None,
    ) -> ShadowPrefixCandidate | None:
        """Lease a candidate only when all non-cache cursors still match."""
        with self._condition:
            candidate = self._candidate
            if candidate is None:
                return None
            rejection_reason = None
            if self._candidate_claimed:
                rejection_reason = "candidate_already_claimed"
            elif self._q2_processing:
                rejection_reason = "q2_step_inflight"
            stage = self._desired.get(candidate.row.session_id)
            if rejection_reason is None and stage is None:
                rejection_reason = "desired_stage_missing"
            elif rejection_reason is None and _shadow_stage_identity(stage) != _shadow_stage_identity(candidate.stage):
                rejection_reason = "semantic_identity_changed"
            elif rejection_reason is None and stage.listener_generation_id != destination_row.session_generation:
                rejection_reason = "generation_mismatch"
            elif rejection_reason is None and stage.request.state_epoch_id != destination_row.state_epoch_id:
                rejection_reason = "epoch_mismatch"
            elif (
                rejection_reason is None
                and semantic_context_revision is not None
                and stage.request.semantic_context_revision != semantic_context_revision
            ):
                rejection_reason = "semantic_revision_mismatch"
            elif rejection_reason is None and not _same_listener_cursor(candidate.row, destination_row):
                rejection_reason = "listener_cursor_mismatch"
            if rejection_reason is not None:
                self._record_candidate_cursor_rejection_locked(rejection_reason)
                return None
            self._candidate_claimed = True
            self._set_diagnostic_phase_locked("promotion_preflight", candidate.stage)
            return candidate

    def record_candidate_cursor_rejection(self, session_id: str, reason: str) -> None:
        with self._condition:
            candidate = self._candidate
            if candidate is not None and candidate.row.session_id == str(session_id):
                self._record_candidate_cursor_rejection_locked(reason)

    def record_promotion_rejection(self, kind: str, reason: str) -> None:
        if kind not in {"preflight", "copy"}:
            raise ValueError("promotion rejection kind must be 'preflight' or 'copy'")
        concise_reason = str(reason)[:120]
        with self._condition:
            if kind == "preflight":
                self._promotion_preflight_rejections += 1
                self._promotion_preflight_last_rejection_reason = concise_reason
            else:
                self._promotion_copy_rejections += 1
                self._promotion_copy_last_rejection_reason = concise_reason

    def record_cache_candidate_copy_enqueue(self, duration_ms: float) -> None:
        with self._condition:
            _update_shadow_timing_stats(self._cache_candidate_copy_enqueue_stats, duration_ms)

    def finish_promotion(self, candidate: ShadowPrefixCandidate, copy_result: Any) -> None:
        if not isinstance(candidate, ShadowPrefixCandidate):
            raise TypeError("candidate must be ShadowPrefixCandidate")
        completion_event = getattr(copy_result, "completion_event", None)
        destination_row = getattr(copy_result, "destination_row", None)
        if completion_event is None or not isinstance(destination_row, BatchRowDescriptor):
            raise TypeError("cache copy must return destination_row and completion_event")
        with self._condition:
            if (
                not self._candidate_claimed
                or self._candidate is None
                or self._candidate.candidate_id != candidate.candidate_id
            ):
                raise RuntimeError("cache copy result does not match the claimed shadow candidate")
            if self._claimed_promotion is not None:
                raise RuntimeError("a shadow promotion result is already pending")
            # This direct slot outranks queued invalidation/discard controls.
            # The shadow owner must first wait on this event before it can
            # release or reuse the source row.
            self._claimed_promotion = (candidate, copy_result)
            self._set_diagnostic_phase_locked("promotion_copy_pending_fence", candidate.stage)
            self._condition.notify_all()

    def discard_candidate(
        self,
        candidate: ShadowPrefixCandidate,
        reason: str,
        *,
        copy_not_enqueued: bool = False,
    ) -> None:
        with self._condition:
            if self._candidate_claimed and copy_not_enqueued:
                if self._candidate is None or self._candidate.candidate_id != candidate.candidate_id:
                    raise RuntimeError("cannot cancel a different shadow promotion claim")
                self._candidate_claimed = False
            self._controls.append(("discard", candidate, str(reason)))
            self._condition.notify_all()

    def invalidate_session(self, session_id: str) -> None:
        with self._condition:
            self._remove_desired_stage_locked(session_id)
            for key in tuple(self._primary_applied):
                if key[0] == session_id:
                    self._primary_applied.pop(key, None)
            self._shadow_promoted_identities = {
                identity for identity in self._shadow_promoted_identities
                if identity[0] != session_id
            }
            self._q2.pop(session_id, None)
            self._controls.append(("invalidate", session_id, None))
            self._condition.notify_all()

    def owns_session_work(self, session_id: str) -> bool:
        with self._condition:
            return bool(
                session_id in self._queued_stage_ids
                or self._active_session_id == session_id
                or (self._candidate is not None and self._candidate.row.session_id == session_id)
                or (self._claimed_promotion is not None
                    and self._claimed_promotion[0].row.session_id == session_id)
            )

    @property
    def status(self) -> Mapping[str, Any]:
        with self._condition:
            candidate = self._candidate
            active = self._active_stage
            active_row = self._active_row
            candidate_row = None if candidate is None else candidate.row
            applied = tuple(
                record.status()
                for _key, record in sorted(self._primary_applied.items())
            )
            return {
                "initialized": self._startup.done() and self._startup.exception() is None,
                "thread_alive": bool(self._thread is not None and self._thread.is_alive()),
                "active_session_id": self._active_session_id,
                "active_revision": (
                    None if active is None else int(active.request.semantic_context_revision)
                ),
                "active_stage": self._stage_status(active, active_row),
                "staged_sessions": len(self._desired),
                "queued_sessions": len(self._queued_stage_ids),
                "candidate_session_id": None if candidate is None else candidate.row.session_id,
                "candidate_revision": (
                    None if candidate is None
                    else int(candidate.stage.request.semantic_context_revision)
                ),
                "candidate_frame_id": None if candidate is None else candidate.row.frame_id,
                "candidate_stage": (
                    None if candidate is None
                    else self._stage_status(candidate.stage, candidate_row)
                ),
                "candidate_catchup_frames": None if candidate is None else candidate.catchup_frames,
                "candidate_claimed": self._candidate_claimed,
                "pending_q2": sum(len(rows) for rows in self._q2.values()),
                "promotions_completed": self._promotions_completed,
                "primary_applied": applied,
                "latest_error": self._latest_error,
                "diagnostics": self._diagnostics_locked(),
            }

    def _diagnostics_locked(self) -> Mapping[str, Any]:
        stage = self._diagnostic_stage
        request = None if stage is None else stage.request
        pending_depth = sum(len(rows) for rows in self._q2.values())
        current_job = {
            "session_id": None if request is None else str(request.session_id),
            "listener_generation_id": (
                None if stage is None else int(stage.listener_generation_id)
            ),
            "state_epoch_id": None if request is None else str(request.state_epoch_id),
            "semantic_context_revision": (
                None if request is None else int(request.semantic_context_revision)
            ),
            "request_base_frame_id": (
                None if request is None else int(request.current_frame_id)
            ),
            "phase": self._diagnostic_phase,
            "stage_requested_at_ns": (
                None if stage is None else int(stage.requested_at_ns)
            ),
            "stage_queued_at_ns": (
                None if stage is None or stage.queued_at_ns is None
                else int(stage.queued_at_ns)
            ),
            "stage_execution_started_at_ns": (
                None if stage is None or stage.execution_started_at_ns is None
                else int(stage.execution_started_at_ns)
            ),
            "fair_lock_wait_ms": self._diagnostic_fair_lock_wait_ms,
            "teacher_prefix_build_wall_ms": self._diagnostic_prefix_build_wall_ms,
            "teacher_prefix_replay_wall_ms": self._diagnostic_prefix_replay_wall_ms,
            "teacher_prefix_fenced_wall_ms": (
                self._diagnostic_teacher_prefix_fenced_wall_ms
            ),
            "shadow_native_prefix_import_ms": self._diagnostic_native_prefix_import_ms,
        }
        return {
            "current_job": current_job,
            "q2": {
                "submitted_count": self._q2_submitted_count,
                "completed_count": self._q2_completed_count,
                "last_submitted_session_id": self._q2_last_submitted_session_id,
                "last_submitted_frame_id": self._q2_last_submitted_frame_id,
                "last_completed_session_id": self._q2_last_completed_session_id,
                "last_completed_frame_id": self._q2_last_completed_frame_id,
                "pending_depth": pending_depth,
                "max_pending_depth": self._q2_max_pending_depth,
                "timings": {
                    name: dict(stats) for name, stats in self._q2_timing_stats.items()
                },
            },
            "candidate_cursor_rejections": self._candidate_cursor_rejections,
            "candidate_last_rejection_reason": self._candidate_last_rejection_reason,
            "promotion_preflight_rejections": self._promotion_preflight_rejections,
            "promotion_preflight_last_rejection_reason": (
                self._promotion_preflight_last_rejection_reason
            ),
            "promotion_copy_rejections": self._promotion_copy_rejections,
            "promotion_copy_last_rejection_reason": (
                self._promotion_copy_last_rejection_reason
            ),
            "cache_candidate_copy_enqueue": dict(
                self._cache_candidate_copy_enqueue_stats
            ),
        }

    def _begin_diagnostic_job_locked(
        self, stage: ShadowPrefixStage, phase: str
    ) -> None:
        self._diagnostic_stage = stage
        self._diagnostic_phase = phase
        self._diagnostic_fair_lock_wait_ms = None
        self._diagnostic_prefix_build_wall_ms = None
        self._diagnostic_prefix_replay_wall_ms = None
        self._diagnostic_teacher_prefix_fenced_wall_ms = None
        self._diagnostic_native_prefix_import_ms = None

    def _set_diagnostic_phase_locked(
        self, phase: str, stage: ShadowPrefixStage | None = None
    ) -> None:
        if stage is not None:
            self._diagnostic_stage = stage
        self._diagnostic_phase = phase

    def _select_queued_diagnostic_job_locked(self) -> None:
        if self._active_stage is not None or self._candidate is not None:
            return
        for session_id in self._desired_order:
            if session_id not in self._queued_stage_ids:
                continue
            stage = self._desired.get(session_id)
            if (
                stage is not None
                and not self._stage_suppressed_locked(stage)
                and not self._stage_already_promoted_locked(stage)
            ):
                if (
                    self._diagnostic_stage is None
                    or _shadow_stage_identity(self._diagnostic_stage)
                    != _shadow_stage_identity(stage)
                ):
                    self._begin_diagnostic_job_locked(stage, "queued")
                else:
                    self._set_diagnostic_phase_locked("queued", stage)
                return
        self._diagnostic_stage = None
        self._diagnostic_phase = "idle"
        self._diagnostic_fair_lock_wait_ms = None
        self._diagnostic_prefix_build_wall_ms = None
        self._diagnostic_prefix_replay_wall_ms = None
        self._diagnostic_teacher_prefix_fenced_wall_ms = None
        self._diagnostic_native_prefix_import_ms = None

    def _record_candidate_cursor_rejection_locked(self, reason: str) -> None:
        self._candidate_cursor_rejections += 1
        self._candidate_last_rejection_reason = str(reason)[:120]

    def _record_q2_timing(self, name: str, duration_ms: float) -> None:
        with self._condition:
            _update_shadow_timing_stats(self._q2_timing_stats[name], duration_ms)

    def stop(self) -> Future[None]:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
        return self._stopped

    def _run(self) -> None:
        try:
            runtime = self._runtime_factory()
            self._validate_runtime(runtime)
            self._runtime = runtime
            with _listener_stream_context(runtime.listener_stream):
                capacity = runtime.native.ensure_capacity(1)
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
                raise RuntimeError("shadow native context failed capacity-1 prewarm")
            self._startup.set_result(None)
            while True:
                work: tuple[str, Any] | None = None
                stage_to_build: ShadowPrefixStage | None = None
                with self._condition:
                    self._collect_copy_leases_locked()
                    # The primary owner drains its copy stream before it asks
                    # the shadow owner to stop. At that point a claim without
                    # a result can only be a poisoned/unrecordable copy, so it
                    # is safe to abandon the source slot during shutdown.
                    if (
                        self._stop_requested
                        and self._candidate_claimed
                        and self._claimed_promotion is None
                    ):
                        self._candidate_claimed = False
                    if (
                        self._stop_requested
                        and not self._controls
                        and self._claimed_promotion is None
                    ):
                        break
                    if self._candidate_claimed and self._claimed_promotion is not None:
                        candidate, copy_result = self._claimed_promotion
                        self._claimed_promotion = None
                        work = ("promoted", (candidate, copy_result))
                    elif self._controls and not self._control_waits_for_claimed_candidate_locked(
                        self._controls[0]
                    ):
                        operation, value, extra = self._controls.popleft()
                        work = (operation, (value, extra))
                    elif (
                        not self._candidate_claimed
                        and self._active_stage is not None
                        and self._stage_obsolete_locked(self._active_stage)
                    ):
                        work = ("abort_active", "desired semantic request changed")
                    elif self._candidate is not None and not self._candidate_claimed and self._stage_obsolete_locked(
                        self._candidate.stage
                    ):
                        work = ("abort_candidate", "desired semantic request changed")
                    elif self._active_stage is None and self._candidate is None:
                        stage_to_build = self._pop_pending_stage_locked()
                    elif not self._candidate_claimed and self._active_session_id is not None:
                        pending = self._q2.get(self._active_session_id)
                        if pending:
                            q2 = pending.popleft()
                            if not pending:
                                self._q2.pop(self._active_session_id, None)
                            self._q2_processing = True
                            work = ("q2", q2)
                    if work is None and stage_to_build is None:
                        self._condition.wait(timeout=0.02)
                        continue
                if stage_to_build is not None:
                    self._build_stage(stage_to_build)
                elif work is not None:
                    try:
                        self._execute_work(*work)
                    finally:
                        if work[0] == "q2":
                            with self._condition:
                                self._q2_processing = False
                                self._condition.notify_all()
        except BaseException as exc:
            self._latest_error = f"{type(exc).__name__}: {exc}"
            if not self._startup.done():
                self._startup.set_exception(exc)
        finally:
            self._cleanup_shadow()
            if not self._startup.done():
                self._startup.set_exception(RuntimeError("shadow worker stopped before startup"))
            if not self._stopped.done():
                self._stopped.set_result(None)

    def _validate_runtime(self, runtime: SemanticPrefixShadowRuntime) -> None:
        if not isinstance(runtime, SemanticPrefixShadowRuntime):
            raise TypeError("shadow factory must return SemanticPrefixShadowRuntime")
        native = runtime.native
        if (
            self._runtime_signature is not None
            and getattr(native, "runtime_signature", None) != self._runtime_signature
        ):
            raise RuntimeError("shadow native runtime signature differs from primary listener")
        capability = getattr(native, "capability", None)
        if not isinstance(capability, NativeBatchCapability):
            raise TypeError("shadow native must expose an attested NativeBatchCapability")
        if capability.stream_priority != 0:
            raise RuntimeError("semantic-history shadow native stream must have priority 0")
        if not bool(getattr(capability, "supports_arbitrary_batch_size", False)):
            raise RuntimeError("shadow native must attest arbitrary positive batch sizes")
        if getattr(native, "listener_stream", None) is not runtime.listener_stream:
            raise RuntimeError("shadow native and shadow runtime must share their exact stream")
        if not callable(getattr(native, "import_session", None)):
            raise TypeError("shadow native lacks import_session")
        if not callable(getattr(native, "step_async", None)):
            raise TypeError("shadow native lacks step_async")
        if not callable(getattr(native, "release_session_async", None)):
            raise TypeError("shadow native lacks release_session_async")
        if not callable(getattr(native, "verify_output_health", None)):
            raise TypeError("shadow native lacks output health checks")
        if not callable(getattr(native, "prepare_cache_copy_source", None)):
            raise TypeError("shadow native lacks owner-bound cache-copy source snapshots")

    def _pop_pending_stage_locked(self) -> ShadowPrefixStage | None:
        while self._desired_order:
            session_id = self._desired_order.popleft()
            self._queued_stage_ids.discard(session_id)
            stage = self._desired.get(session_id)
            if (
                stage is not None
                and not self._stage_suppressed_locked(stage)
                and not self._stage_already_promoted_locked(stage)
            ):
                rows = self._q2.get(session_id)
                if rows:
                    watermark = int(stage.request.current_frame_id)
                    self._q2[session_id] = deque(
                        item for item in rows if int(item.frame.frame_id) >= watermark
                    )
                    rows = self._q2.get(session_id)
                    if not rows:
                        self._q2.pop(session_id, None)
                    else:
                        first_frame_id = int(rows[0].frame.frame_id)
                        if first_frame_id > watermark:
                            # A queued stage snapshot cannot bridge a missing real
                            # Q2. Drop this stale job; the next server snapshot can
                            # replay a fresh contiguous suffix from listener state.
                            self._q2.pop(session_id, None)
                            self._latest_error = (
                                f"shadow stage watermark stale for {session_id}: "
                                f"request={watermark}, first_q2={first_frame_id}"
                            )
                            self._select_queued_diagnostic_job_locked()
                            continue
                stage = replace(stage, execution_started_at_ns=time.monotonic_ns())
                self._desired[session_id] = stage
                self._active_stage = stage
                self._active_session_id = session_id
                self._active_catchup_frames = 0
                self._begin_diagnostic_job_locked(stage, "prefix_building")
                return stage
        return None

    def _control_waits_for_claimed_candidate_locked(
        self, control: tuple[str, Any, Any]
    ) -> bool:
        if not self._candidate_claimed or self._candidate is None:
            return False
        operation, value, _extra = control
        candidate = self._candidate
        if operation in {"invalidate", "abort"}:
            session_id = value[0] if operation == "abort" else value
            return str(session_id) == candidate.row.session_id
        if operation == "discard":
            claimed = value[0]
            return (
                isinstance(claimed, ShadowPrefixCandidate)
                and claimed.candidate_id == candidate.candidate_id
            )
        return False

    def _queue_stage_locked(self, session_id: str) -> None:
        if session_id not in self._queued_stage_ids:
            stage = self._desired.get(session_id)
            if stage is not None and stage.queued_at_ns is None:
                stage = replace(stage, queued_at_ns=time.monotonic_ns())
                self._desired[session_id] = stage
            self._desired_order.append(session_id)
            self._queued_stage_ids.add(session_id)
        if self._active_stage is None and self._candidate is None:
            self._select_queued_diagnostic_job_locked()

    def _stage_suppressed_locked(self, stage: ShadowPrefixStage) -> bool:
        request = stage.request
        identity_key = (
            str(request.session_id),
            int(stage.listener_generation_id),
            str(request.state_epoch_id),
        )
        revision = int(request.semantic_context_revision)
        primary = self._primary_applied.get(identity_key)
        return bool(primary is not None and revision <= primary.semantic_context_revision)

    def _stage_already_promoted_locked(self, stage: ShadowPrefixStage) -> bool:
        return _shadow_stage_identity(stage) in self._shadow_promoted_identities

    def _remove_desired_stage_locked(self, session_id: str) -> None:
        self._desired.pop(session_id, None)
        self._desired_order = deque(
            sid for sid in self._desired_order if sid != session_id
        )
        self._queued_stage_ids.discard(session_id)
        if (
            self._active_stage is None
            and self._candidate is None
            and self._diagnostic_stage is not None
            and str(self._diagnostic_stage.request.session_id) == session_id
        ):
            self._diagnostic_stage = None
            self._diagnostic_phase = "idle"
            self._diagnostic_fair_lock_wait_ms = None
            self._diagnostic_prefix_build_wall_ms = None
            self._diagnostic_prefix_replay_wall_ms = None
            self._diagnostic_teacher_prefix_fenced_wall_ms = None
            self._diagnostic_native_prefix_import_ms = None
            self._select_queued_diagnostic_job_locked()

    def _prune_primary_obsolete_stage_locked(
        self,
        session_id: str,
        listener_generation_id: int,
        state_epoch_id: str,
        semantic_context_revision: int,
    ) -> None:
        desired = self._desired.get(session_id)
        matching_identity = (
            desired is not None
            and desired.listener_generation_id == listener_generation_id
            and str(desired.request.state_epoch_id) == state_epoch_id
        )
        if (
            matching_identity
            and int(desired.request.semantic_context_revision) <= semantic_context_revision
        ):
            self._remove_desired_stage_locked(session_id)
            # Queued Q2 belongs to the obsolete desired identity. Active Q2
            # work is already owned by the shadow thread and will retire there.
            self._q2.pop(session_id, None)
            self._shadow_promoted_identities = {
                identity for identity in self._shadow_promoted_identities
                if not (
                    identity[0] == session_id
                    and identity[1] == listener_generation_id
                    and identity[2] == state_epoch_id
                    and int(identity[3]) <= semantic_context_revision
                )
            }

    @staticmethod
    def _stage_status(
        stage: ShadowPrefixStage | None,
        row: BatchRowDescriptor | None,
    ) -> Mapping[str, Any] | None:
        if stage is None:
            return None
        request = stage.request
        replay_ids = tuple(int(value) for value in request.replay_frame_ids)
        return {
            "session_id": str(request.session_id),
            "listener_generation_id": int(stage.listener_generation_id),
            "state_epoch_id": str(request.state_epoch_id),
            "semantic_context_revision": int(request.semantic_context_revision),
            "request_current_frame_id": int(request.current_frame_id),
            "current_frame_id": (
                int(row.frame_id) if row is not None else int(request.current_frame_id)
            ),
            "listener_state_version": (
                int(row.listener_state_version)
                if row is not None else int(request.listener_state_version)
            ),
            "prefix_len": int(request.prefix_len),
            "replay_count": len(replay_ids),
            "replay_first_q2_frame_id": replay_ids[0] if replay_ids else None,
            "replay_last_q2_frame_id": replay_ids[-1] if replay_ids else None,
        }

    def _stage_obsolete_locked(self, stage: ShadowPrefixStage) -> bool:
        latest = self._desired.get(str(stage.request.session_id))
        return latest is None or _shadow_stage_identity(latest) != _shadow_stage_identity(stage)

    def _build_stage(self, stage: ShadowPrefixStage) -> None:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("shadow runtime is unavailable")
        request = stage.request
        row: BatchRowDescriptor | None = None
        import_succeeded = False
        try:
            base_frame_id, base_state_version, base_sequence_position = (
                _shadow_replay_base_cursor(stage)
            )
            prefix = self._prefix_builder(request, runtime.listener_stream)
            if not isinstance(prefix, PrefixCacheImport):
                raise TypeError("shadow prefix builder must return PrefixCacheImport")
            if (
                prefix.prefix_len != request.prefix_len
                or prefix.cache_position != request.prefix_len
                or (
                    prefix.semantic_context_revision is not None
                    and prefix.semantic_context_revision != request.semantic_context_revision
                )
                or tuple(prefix.replay_frame_ids) != tuple(request.replay_frame_ids)
            ):
                raise RuntimeError("shadow prefix builder changed the frozen prefix or replay watermark")
            with self._condition:
                self._diagnostic_fair_lock_wait_ms = prefix.fair_lock_wait_ms
                self._diagnostic_prefix_build_wall_ms = prefix.prefix_build_ms
                self._diagnostic_prefix_replay_wall_ms = prefix.replay_ms
                self._diagnostic_teacher_prefix_fenced_wall_ms = (
                    prefix.teacher_prefix_fenced_wall_ms
                )
                self._set_diagnostic_phase_locked("native_prefix_import", stage)
            context_id = self._next_context_id
            self._next_context_id += 1
            row = BatchRowDescriptor(
                session_id=request.session_id,
                session_generation=stage.listener_generation_id,
                state_epoch_id=request.state_epoch_id,
                frame_id=base_frame_id,
                listener_state_version=base_state_version,
                sequence_position=base_sequence_position,
                cache_position=request.prefix_len,
                valid_history_tokens=request.prefix_len,
                native_context_id=context_id,
                input_frame_sha256=hashlib.sha256(
                    f"semantic-shadow:{request.session_id}:{request.semantic_context_revision}:"
                    f"{request.current_frame_id}".encode("utf-8")
                ).hexdigest(),
                reference_pcm_sha256=None,
                frame_ready_at_ns=0,
                reference_frame_id=None,
            )
            with _listener_stream_context(runtime.listener_stream):
                if prefix.ready_event is not None:
                    wait_event = getattr(runtime.listener_stream, "wait_event", None)
                    if not callable(wait_event):
                        raise RuntimeError("shadow stream cannot wait for semantic prefill")
                    wait_event(prefix.ready_event)
                import_started_ns = time.perf_counter_ns()
                try:
                    runtime.native.import_session(
                        row,
                        prefix.past_key_values,
                        prefix.prefix_attention,
                        prefix_len=prefix.prefix_len,
                        producer_stream=runtime.listener_stream,
                    )
                    import_succeeded = True
                finally:
                    import_ms = round(
                        max(0, time.perf_counter_ns() - import_started_ns)
                        / 1_000_000.0,
                        3,
                    )
                    with self._condition:
                        self._diagnostic_native_prefix_import_ms = import_ms
                ready_event = _record_shadow_event(runtime.listener_stream)
            with self._condition:
                self._active_row = row
                self._active_prefix = prefix
                self._active_ready_event = ready_event

            if self._shadow_stage_cancelled(stage):
                self._release_active_row(row)
                self._clear_active(stage)
                return

            replay_q2 = tuple(request.replay_q2)
            replay_frame_ids = tuple(request.replay_frame_ids)
            current_row = row
            ready_event = self._active_ready_event
            if replay_q2:
                with self._condition:
                    self._set_diagnostic_phase_locked("native_history_replay", stage)
            for frame_id, audio_soft in zip(replay_frame_ids, replay_q2, strict=True):
                if self._shadow_stage_cancelled(stage):
                    self._release_active_row(current_row)
                    self._clear_active(stage)
                    return
                if frame_id != current_row.frame_id:
                    raise RuntimeError("frozen replay frame ID differs from the derived native cursor")
                replay_row = replace(
                    current_row,
                    input_frame_sha256=hashlib.sha256(
                        f"semantic-shadow-q2:{request.session_id}:{request.semantic_context_revision}:"
                        f"{frame_id}".encode("utf-8")
                    ).hexdigest(),
                )
                manifest = ListenerBatchManifest(
                    manifest_id=uuid.uuid4().hex,
                    runtime_signature=runtime.native.runtime_signature,
                    stream_identity=runtime.native.capability.stream_identity,
                    stream_priority=runtime.native.capability.stream_priority,
                    dispatch_at_ns=time.monotonic_ns(),
                    max_coalesce_wait_ns=0,
                    dispatch_reason="coalesce_deadline",
                    rows=(replay_row,),
                )
                with _listener_stream_context(runtime.listener_stream):
                    if ready_event is not None:
                        wait_event = getattr(runtime.listener_stream, "wait_event", None)
                        if not callable(wait_event):
                            raise RuntimeError("shadow stream cannot wait for frozen replay input")
                        wait_event(ready_event)
                    output = runtime.native.step_async(
                        manifest,
                        audio_soft.unsqueeze(0).contiguous(),
                        producer_stream=runtime.listener_stream,
                    )
                    _wait_shadow_event(output.completion_event, runtime.listener_stream)
                    next_row = replace(
                        replay_row,
                        frame_id=replay_row.frame_id + 1,
                        listener_state_version=replay_row.listener_state_version + 1,
                        sequence_position=replay_row.sequence_position + 1,
                        cache_position=replay_row.cache_position + LISTENER_Q_PER_FRAME,
                        valid_history_tokens=replay_row.valid_history_tokens + LISTENER_Q_PER_FRAME,
                    )
                    with self._condition:
                        self._active_row = next_row
                        self._active_ready_event = output.completion_event
                    health_rows = tuple(runtime.native.verify_output_health(output))
                    if len(health_rows) != 1:
                        raise RuntimeError("shadow history replay returned an incomplete health row set")
                    health = health_rows[0]
                    if (
                        health.row != replay_row
                        or health.generation != output.generation
                        or health.checked is not True
                        or health.nonfinite is not False
                        or health.status != "healthy"
                    ):
                        raise RuntimeError(
                            "shadow native history output failed row-bound health/provenance"
                        )
                current_row = next_row
                ready_event = output.completion_event
                with self._condition:
                    self._active_row = current_row
                    self._active_ready_event = ready_event

            target_cursor = (
                request.current_frame_id,
                request.listener_state_version,
                stage.sequence_position,
                request.cache_position,
            )
            actual_cursor = (
                current_row.frame_id,
                current_row.listener_state_version,
                current_row.sequence_position,
                current_row.cache_position,
            )
            if actual_cursor != target_cursor:
                raise RuntimeError(
                    "shadow native replay did not reach the frozen request cursor: "
                    f"actual={actual_cursor}, target={target_cursor}"
                )
            if self._shadow_stage_cancelled(stage):
                self._release_active_row(current_row)
                self._clear_active(stage)
                return

            with _listener_stream_context(runtime.listener_stream):
                source_token = runtime.native.prepare_cache_copy_source(current_row)
            with self._condition:
                if self._stage_obsolete_locked(stage) or self._stop_requested:
                    stale = True
                else:
                    stale = False
                    self._active_row = current_row
                    self._active_prefix = prefix
                    self._active_ready_event = ready_event
                    self._publish_candidate_locked(
                        stage, prefix, current_row, ready_event, source_token
                    )
            if stale:
                self._release_active_row(current_row)
                self._clear_active(stage)
        except BaseException as exc:
            with self._condition:
                self._latest_error = f"shadow stage {request.session_id}: {type(exc).__name__}: {exc}"
            release_row = self._active_row if self._active_row is not None else row
            if import_succeeded:
                self._release_active_row(release_row)
            self._clear_active(stage)

    def _execute_work(self, operation: str, value: Any) -> None:
        if operation == "q2":
            self._process_q2(value)
        elif operation == "abort_active":
            self._abort_active(str(value))
        elif operation == "abort_candidate":
            self._abort_candidate(str(value))
        elif operation == "invalidate":
            session_id, _ = value
            self._invalidate_on_owner(str(session_id))
        elif operation == "discard":
            candidate, reason = value
            self._discard_on_owner(candidate, str(reason))
        elif operation == "promoted":
            candidate, copy_result = value
            self._promoted_on_owner(candidate, copy_result)
        elif operation == "abort":
            session_id, reason = value
            if self._active_session_id == session_id:
                self._abort_active(str(reason))
            elif self._candidate is not None and self._candidate.row.session_id == session_id:
                self._abort_candidate(str(reason))
        else:
            raise RuntimeError(f"unknown semantic shadow work: {operation}")

    def _process_q2(self, item: ShadowQ2Frame) -> None:
        runtime = self._runtime
        stage = self._active_stage
        current = self._active_row
        prefix = self._active_prefix
        if runtime is None or stage is None or current is None or prefix is None:
            return
        frame = item.frame
        expected = current.frame_id
        if frame.session_id != current.session_id or frame.frame_id < expected:
            return
        if frame.frame_id != expected:
            self._abort_active(
                f"shadow replay gap: expected frame {expected}, got {frame.frame_id}"
            )
            return
        if (
            item.row.session_generation != current.session_generation
            or item.row.state_epoch_id != current.state_epoch_id
            or item.row.listener_state_version != current.listener_state_version
            or item.row.sequence_position != current.sequence_position
            or item.row.frame_id != current.frame_id
        ):
            self._abort_active("shadow Q2 provenance no longer matches its private cursor")
            return
        if self._is_obsolete(stage):
            self._abort_active("desired semantic context changed during shadow catch-up")
            return
        if tuple(item.audio_soft.shape) != (LISTENER_Q_PER_FRAME, CONTROL_HIDDEN_WIDTH):
            self._abort_active("semantic replay must have exact Q=2 [2,1536] layout")
            return
        shadow_row = replace(
            item.row,
            cache_position=current.cache_position,
            valid_history_tokens=current.valid_history_tokens,
            native_context_id=current.native_context_id,
        )
        manifest = ListenerBatchManifest(
            manifest_id=uuid.uuid4().hex,
            runtime_signature=runtime.native.runtime_signature,
            stream_identity=runtime.native.capability.stream_identity,
            stream_priority=runtime.native.capability.stream_priority,
            dispatch_at_ns=time.monotonic_ns(),
            max_coalesce_wait_ns=0,
            dispatch_reason="coalesce_deadline",
            rows=(shadow_row,),
        )
        try:
            with _listener_stream_context(runtime.listener_stream):
                if item.ready_event is not None:
                    with self._condition:
                        self._set_diagnostic_phase_locked("q2_ready_wait", stage)
                    wait_started_ns = time.perf_counter_ns()
                    try:
                        runtime.listener_stream.wait_event(item.ready_event)
                    finally:
                        self._record_q2_timing(
                            "wait",
                            max(0, time.perf_counter_ns() - wait_started_ns)
                            / 1_000_000.0,
                        )
                with self._condition:
                    self._set_diagnostic_phase_locked("q2_native_step", stage)
                output = runtime.native.step_async(
                    manifest,
                    item.audio_soft.unsqueeze(0).contiguous(),
                    producer_stream=runtime.listener_stream,
                )
                with self._condition:
                    self._set_diagnostic_phase_locked("q2_completion_fence", stage)
                fence_started_ns = time.perf_counter_ns()
                try:
                    _wait_shadow_event(output.completion_event, runtime.listener_stream)
                finally:
                    self._record_q2_timing(
                        "fence",
                        max(0, time.perf_counter_ns() - fence_started_ns)
                        / 1_000_000.0,
                    )
                with self._condition:
                    self._set_diagnostic_phase_locked("output_health_validation", stage)
                health_started_ns = time.perf_counter_ns()
                try:
                    health_rows = tuple(runtime.native.verify_output_health(output))
                    if len(health_rows) != 1:
                        raise RuntimeError("shadow finite guard returned an incomplete row set")
                    health = health_rows[0]
                    if (
                        health.row != shadow_row
                        or health.generation != output.generation
                        or health.checked is not True
                        or health.nonfinite is not False
                        or health.status != "healthy"
                    ):
                        raise RuntimeError("shadow native output failed row-bound health/provenance")
                finally:
                    self._record_q2_timing(
                        "output_health_validation",
                        max(0, time.perf_counter_ns() - health_started_ns)
                        / 1_000_000.0,
                    )
            with self._condition:
                self._q2_completed_count += 1
                self._q2_last_completed_session_id = str(frame.session_id)
                self._q2_last_completed_frame_id = int(frame.frame_id)
            next_row = replace(
                shadow_row,
                frame_id=shadow_row.frame_id + 1,
                listener_state_version=shadow_row.listener_state_version + 1,
                sequence_position=shadow_row.sequence_position + 1,
                cache_position=shadow_row.cache_position + LISTENER_Q_PER_FRAME,
                valid_history_tokens=shadow_row.valid_history_tokens + LISTENER_Q_PER_FRAME,
            )
            source_token = runtime.native.prepare_cache_copy_source(next_row)
            with self._condition:
                if self._stage_obsolete_locked(stage):
                    stale = True
                else:
                    stale = False
                    self._active_row = next_row
                    self._active_catchup_frames += 1
                    self._active_ready_event = output.completion_event
                    self._publish_candidate_locked(
                        stage, prefix, next_row, output.completion_event, source_token
                    )
            if stale:
                self._abort_active("desired semantic context changed after shadow step")
        except BaseException as exc:
            with self._condition:
                self._latest_error = f"shadow catch-up {frame.session_id}/{frame.frame_id}: {type(exc).__name__}: {exc}"
            self._abort_active("shadow catch-up failed")

    def _publish_candidate_locked(
        self,
        stage: ShadowPrefixStage,
        prefix: PrefixCacheImport,
        row: BatchRowDescriptor,
        ready_event: Any | None,
        source_token: Any,
    ) -> None:
        self._candidate = ShadowPrefixCandidate(
            candidate_id=self._next_candidate_id,
            stage=stage,
            prefix=prefix,
            native=self._runtime.native if self._runtime is not None else None,
            source_token=source_token,
            row=row,
            ready_event=ready_event,
            catchup_frames=self._active_catchup_frames,
            caught_up_at_ns=time.monotonic_ns(),
        )
        self._next_candidate_id += 1
        self._candidate_claimed = False
        self._set_diagnostic_phase_locked("candidate_ready", stage)

    def _is_obsolete(self, stage: ShadowPrefixStage) -> bool:
        with self._condition:
            return self._stage_obsolete_locked(stage)

    def _shadow_stage_cancelled(self, stage: ShadowPrefixStage) -> bool:
        with self._condition:
            return self._stop_requested or self._stage_obsolete_locked(stage)

    def _abort_active(self, reason: str) -> None:
        row = self._active_row
        stage = self._active_stage
        self._release_active_row(row)
        if stage is not None:
            self._clear_active(stage)
        with self._condition:
            self._candidate = None
            self._candidate_claimed = False
            self._latest_error = reason
            self._queue_latest_stage_locked(stage)

    def _abort_candidate(self, reason: str) -> None:
        candidate = self._candidate
        if candidate is not None:
            self._release_active_row(candidate.row)
        with self._condition:
            self._candidate = None
            self._candidate_claimed = False
            self._claimed_promotion = None
            self._clear_active_locked()
            self._latest_error = reason
            self._queue_latest_stage_locked(candidate.stage if candidate is not None else None)

    def _invalidate_on_owner(self, session_id: str) -> None:
        if self._active_session_id == session_id:
            self._abort_active("session invalidated")
            return
        candidate = self._candidate
        if candidate is not None and candidate.row.session_id == session_id:
            self._abort_candidate("session invalidated")

    def _discard_on_owner(self, candidate: ShadowPrefixCandidate, reason: str) -> None:
        current = self._candidate
        if current is None or current.candidate_id != candidate.candidate_id:
            return
        self._release_active_row(current.row)
        with self._condition:
            self._candidate = None
            self._candidate_claimed = False
            self._clear_active_locked()
            self._latest_error = reason
            self._queue_latest_stage_locked(current.stage)

    def _promoted_on_owner(self, candidate: ShadowPrefixCandidate, copy_result: Any) -> None:
        runtime = self._runtime
        current = self._candidate
        if runtime is None or current is None or current.candidate_id != candidate.candidate_id:
            raise RuntimeError("promoted shadow candidate is no longer the owned candidate")
        completion_event = copy_result.completion_event
        with self._condition:
            self._set_diagnostic_phase_locked("promotion_copy_fence", current.stage)
        with _listener_stream_context(runtime.listener_stream):
            runtime.listener_stream.wait_event(completion_event)
            # The host-side callback that publishes cache_epoch_id must not
            # run merely because the stream wait was enqueued. Fence the
            # primary copy on this lower-priority shadow owner first; no
            # session/listener lock is held during this wait.
            synchronize_copy = getattr(completion_event, "synchronize", None)
            if callable(synchronize_copy):
                synchronize_copy()
            # Keep the native row alive through the queued D2D transfer and
            # order retirement before any later import into this one-slot ctx.
            self._release_active_row(current.row, copy_completion_fenced=True)
            retirement_event = _record_shadow_event(runtime.listener_stream)
        with self._condition:
            self._copy_leases.append((retirement_event, copy_result))
            self._promotions_completed += 1
            self._shadow_promoted_identities.add(_shadow_stage_identity(current.stage))
            rows = self._q2.get(current.row.session_id)
            if rows:
                self._q2[current.row.session_id] = deque(
                    item for item in rows
                    if int(item.frame.frame_id) > int(current.row.frame_id)
                )
                if not self._q2[current.row.session_id]:
                    self._q2.pop(current.row.session_id, None)
            self._candidate = None
            self._candidate_claimed = False
            self._clear_active_locked()
            self._latest_error = None
            self._queue_latest_stage_locked(current.stage, retry_same_identity=False)
        if self._on_promotion_fenced is not None:
            self._on_promotion_fenced(candidate)

    def _clear_active(self, stage: ShadowPrefixStage) -> None:
        with self._condition:
            if self._active_stage is stage:
                self._clear_active_locked()

    def _clear_active_locked(self) -> None:
        self._active_stage = None
        self._active_row = None
        self._active_prefix = None
        self._active_ready_event = None
        self._active_catchup_frames = 0
        self._active_session_id = None
        self._diagnostic_stage = None
        self._diagnostic_phase = "idle"
        self._diagnostic_fair_lock_wait_ms = None
        self._diagnostic_prefix_build_wall_ms = None
        self._diagnostic_prefix_replay_wall_ms = None
        self._diagnostic_native_prefix_import_ms = None
        self._select_queued_diagnostic_job_locked()
        # Other sessions can be staged behind this one; their captured Q2 is
        # independent and must survive use of the single shadow native slot.

    def _queue_latest_stage_locked(
        self,
        completed: ShadowPrefixStage | None,
        *,
        retry_same_identity: bool = True,
    ) -> None:
        if completed is None:
            return
        session_id = str(completed.request.session_id)
        latest = self._desired.get(session_id)
        if (
            latest is not None
            and (
                not self._stage_suppressed_locked(latest)
                and not self._stage_already_promoted_locked(latest)
                and (
                    retry_same_identity
                    or _shadow_stage_identity(latest) != _shadow_stage_identity(completed)
                )
            )
            and session_id != self._active_session_id
            and (self._candidate is None or self._candidate.row.session_id != session_id)
        ):
            self._queue_stage_locked(session_id)

    def _release_active_row(
        self,
        row: BatchRowDescriptor | None,
        *,
        copy_completion_fenced: bool = False,
    ) -> None:
        runtime = self._runtime
        if runtime is None or row is None:
            return
        with self._condition:
            if (
                self._candidate_claimed
                and self._candidate is not None
                and self._candidate.row == row
                and not copy_completion_fenced
            ):
                raise RuntimeError(
                    "cannot release a claimed shadow source before cache-copy completion"
                )
        try:
            with _listener_stream_context(runtime.listener_stream):
                runtime.native.release_session_async(row)
        except BaseException as exc:
            with self._condition:
                self._latest_error = f"shadow slot release failed: {type(exc).__name__}: {exc}"

    def _collect_copy_leases_locked(self) -> None:
        retained: list[tuple[Any, Any]] = []
        for event, lease in self._copy_leases:
            query = getattr(event, "query", None)
            if callable(query):
                try:
                    if query():
                        continue
                except BaseException:
                    pass
            else:
                # Unknown/fake fences are conservatively retained until stop.
                pass
            retained.append((event, lease))
        self._copy_leases = retained

    def _cleanup_shadow(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        row = self._active_row
        if row is None and self._candidate is not None:
            row = self._candidate.row
        with self._condition:
            claimed_without_fence = self._candidate_claimed
        try:
            # The primary owner has drained its copy stream before requesting
            # shadow shutdown. If a post-enqueue failure left a source claimed,
            # close the context after draining this stream instead of issuing
            # an unfenced slot release.
            if row is not None and not claimed_without_fence:
                self._release_active_row(row)
        finally:
            try:
                _drain_cuda_stream_for_cleanup(runtime.listener_stream)
            finally:
                try:
                    runtime.native.close()
                finally:
                    self._runtime = None


@dataclass(frozen=True, slots=True)
class WorkerStopReport:
    """Truthful owner shutdown status; pending rows are never called drained."""

    status: str
    stopped_cleanly: bool
    drained: bool
    processed_batches: int
    pending_frames: int
    owner_thread_id: int | None
    fatal_error: str | None = None
    cleanup_error: str | None = None
    notification_error: str | None = None


@dataclass(frozen=True, slots=True)
class TemporaryQ2DecisionView:
    """True provenance for a short-lived Q2 replay on a private native slot."""

    source_row: BatchRowDescriptor
    fork_row: BatchRowDescriptor
    source_q2_frame_ids: tuple[int, ...]
    manifest: ListenerBatchManifest
    native_output: NativeBatchOutput
    native_output_health: NativeBatchOutputHealth
    gemma_hidden_t: torch.Tensor
    gemma_audio_queries_t: torch.Tensor
    replay_count: int
    fork_count: int = 1
    replayed_q2_count: int = 0
    cache_hit: bool = False
    decision_view_ms: float = 0.0
    context_mode: str = "legacy"

    @property
    def output_generation(self) -> int:
        return self.native_output.generation


@dataclass(frozen=True, slots=True)
class ListenerMicrobatchRowOutput:
    """Committed current-frame output for one listener row."""

    session_id: str
    generation_id: int
    state_epoch_id: str
    frame_id: int
    listener_state_version: int
    sequence_position: int
    cache_position: int
    valid_history_tokens: int
    logical_q_len: int
    gemma_hidden_t: torch.Tensor
    gemma_audio_queries_t: torch.Tensor
    audio_tower_hidden: torch.Tensor
    state_snapshot: ListenerSessionSnapshot
    native_output_health: NativeBatchOutputHealth
    ephemeral_reason: str | None = None
    decision_view_unavailable_reason: str | None = None
    queue_audio_for_next_turn: bool = False
    kv_rollback_tokens: int = 0
    kv_rollback_enqueue_ms: float = 0.0
    native_control6_raw_scores: Any | None = None
    semantic_context_revision: int | None = None
    semantic_replay_fallback_reason: str | None = None
    semantic_prefix_len: int | None = None
    semantic_prefix_replay_frame_ids: tuple[int, ...] = ()
    semantic_prefix_refresh_event: bool = False
    semantic_prefix_refresh_event_id: str | None = None
    semantic_prefix_refresh_revision: int | None = None
    native_state: Any | None = None
    decision_view: TemporaryQ2DecisionView | None = None
    policy_decision: PolicyDecision | None = None
    listener_model_lock_wait_ms: float | None = None
    listener_model_lock_hold_ms: float | None = None
    capacity_growth_ms: float | None = None
    frontend_encode_ms: float | None = None
    mel_prepare_ms: float | None = None
    frontend_copy_ms: float | None = None
    frontend_copy_cuda_ms: float | None = None
    audio_tower_lock_wait_ms: float | None = None
    audio_tower_lock_hold_ms: float | None = None
    audio_tower_workspace_lock_wait_ms: float | None = None
    audio_tower_workspace_lock_hold_ms: float | None = None
    audio_tower_row_copy_cuda_ms: float | None = None
    audio_tower_graph_submit_ms: float | None = None
    cache_epoch_id: int = 0
    applied_history_revision: str | int | None = None
    audio_tower_cuda_ms: float | None = None
    audio_tower_batch_cuda_span_ms: float | None = None
    audio_tower_execution_mode: str | None = None
    native_submit_ms: float | None = None
    native_cuda_ms: float | None = None
    native_output_health_ms: float | None = None
    native_control6_raw_score_ms: float | None = None
    semantic_prefix_build_ms: float | None = None
    semantic_prefix_replay_ms: float | None = None
    semantic_prefix_import_ms: float | None = None
    control6_policy_ms: float | None = None
    cuda_completion_wait_ms: float | None = None
    worker_queue_wait_ms: float | None = None
    worker_wall_ms: float | None = None


@dataclass(frozen=True, slots=True)
class QueuedAudioSoftFrame:
    """A tower-produced Q2 frame held outside the assistant KV branch."""

    session_id: str
    session_generation: int
    state_epoch_id: str
    source_frame_id: int
    source_state_version: int
    source_sequence_position: int
    audio_soft_tokens: torch.Tensor
    source_queue_epoch: int | None = None


@dataclass(slots=True)
class _Q2DecisionCache:
    queue_epoch: str
    anchor: tuple[Any, ...]
    fork_row: BatchRowDescriptor
    base_cache_position: int
    last_applied_frame_id: int
    q2_applied_count: int
    stable_turn_context: bool = False


@dataclass(slots=True)
class _AssistantQueueFrameSpan:
    """Compact source provenance for adjacent assistant-phase Q0 frames."""

    first_frame_id: int
    last_frame_id: int
    retained: bool
    first_state_version: int
    last_state_version: int
    first_sequence_position: int
    last_sequence_position: int
    queue_epoch: int | None = None
    reason: str | None = None
    onset_frame_id: int | None = None

    @property
    def source_queue_epoch(self) -> int | None:
        """Compatibility name used by the existing drain partitioning code."""
        return self.queue_epoch


@dataclass(slots=True)
class _QueuedAudioDrainState:
    """Owner-owned, resumable replay of a sealed queue and later handoffs."""

    session_id: str
    generation_id: int
    state_epoch_id: str
    expected_cache_position: int
    sealed_spans: tuple[_AssistantQueueFrameSpan, ...]
    queued: list[QueuedAudioSoftFrame]
    handoff_spans: tuple[_AssistantQueueFrameSpan, ...]
    handoff_queue: list[QueuedAudioSoftFrame]
    future: Future[ListenerSessionSnapshot]
    submitted_at_ns: int
    started_at_ns: int
    user_turn_already_open: bool
    queue_response_id: str | None
    queue_epoch: str | None
    queue_first_frame_id: int | None
    max_seq_len: int
    trace_enabled: bool
    native_cursor_before: int | None = None
    native_cursor_snapshot_error: str | None = None
    index: int = 0
    replay_forward_ms: float = 0.0
    last_replay_completed_ns: int | None = None


_PENDING_OWNER_RESULT = object()


@dataclass(slots=True)
class _Command:
    operation: str
    args: tuple[Any, ...]
    future: Future[Any] | None = None
    submitted_ns: int | None = None


class _OwnerAwareFuture(Future[Any]):
    """A worker Future that rejects pending waits from its sole owner thread."""

    def __init__(self, owner_thread_id: Callable[[], int | None]) -> None:
        super().__init__()
        self._owner_thread_id_getter = owner_thread_id

    def result(self, timeout: float | None = None) -> Any:
        self._reject_owner_wait("result")
        return super().result(timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        self._reject_owner_wait("exception")
        return super().exception(timeout)

    def _reject_owner_wait(self, operation: str) -> None:
        owner_thread_id = self._owner_thread_id_getter()
        if (
            owner_thread_id is not None
            and threading.get_ident() == owner_thread_id
            and not self.done()
        ):
            raise RuntimeError(
                "listener worker owner thread cannot wait on its own pending "
                f"Future.{operation}(); schedule follow-up work with "
                "loop.call_soon_threadsafe and wait outside the owner thread"
            )


class ListenerMicrobatchWorker:
    """Serialize session commands and real N-row inference on one GPU owner.

    Public methods enqueue commands and return ``Future`` objects.  They do not
    call the coordinator, model, CUDA stream, native binding, or frontend from
    the submitting thread.  A serving adapter should use its event loop's
    ``call_soon_threadsafe`` inside ``on_result``/``on_error`` rather than
    touching WebSocket state directly from this worker thread.
    """

    def __init__(
        self,
        coordinator: ListenerMicrobatchCoordinator | None,
        *,
        runtime_factory: Callable[[], ListenerMicrobatchRuntime],
        on_result: Callable[[ListenerBatch, tuple[PolicyDecision, ...]], None] | None = None,
        on_batch_result: Callable[
            [ListenerBatch, Mapping[str, ListenerMicrobatchRowOutput]], None
        ] | None = None,
        on_error: Callable[[ListenerBatch, BaseException], None] | None = None,
        on_cache_identity_commit: CacheIdentityCommitter | None = None,
        stable_turn_policy_q2_context: bool = False,
        gc_pause_snapshot_provider: Callable[[], Sequence[Any]] | None = None,
    ) -> None:
        if coordinator is not None and not isinstance(coordinator, ListenerMicrobatchCoordinator):
            raise TypeError("coordinator must be ListenerMicrobatchCoordinator or None")
        if not callable(runtime_factory):
            raise TypeError("runtime_factory must create owner-thread resources")
        if type(stable_turn_policy_q2_context) is not bool:
            raise TypeError("stable_turn_policy_q2_context must be a bool")
        if gc_pause_snapshot_provider is not None and not callable(gc_pause_snapshot_provider):
            raise TypeError("gc_pause_snapshot_provider must be callable or None")
        if on_cache_identity_commit is not None and not callable(on_cache_identity_commit):
            raise TypeError("on_cache_identity_commit must be callable or None")
        self._coordinator = coordinator
        self._runtime_factory = runtime_factory
        self._stable_turn_policy_q2_context = stable_turn_policy_q2_context
        self._gc_pause_snapshot_provider = gc_pause_snapshot_provider
        self._on_result = on_result
        self._on_batch_result = on_batch_result
        self._on_error = on_error
        self._on_cache_identity_commit = on_cache_identity_commit
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._deferred_owner_commands: dict[str, deque[_Command]] = {}
        self._wake = threading.Condition()
        self._thread: threading.Thread | None = None
        self._owner_thread_id: int | None = None
        self._runtime: ListenerMicrobatchRuntime | None = None
        self._shadow_worker: SemanticPrefixShadowWorker | None = None
        self._session_states: dict[str, ListenerSessionState] = {}
        self._prefix_importers: dict[str, PrefixImportFactory] = {}
        self._prefix_revisions: dict[str, int | None] = {}
        self._unified_prefix_revisions: dict[str, int] = {}
        self._prefix_fallback_reasons: dict[str, str | None] = {}
        self._prefix_lengths: dict[str, int] = {}
        self._prefix_replay_frame_ids: dict[str, tuple[int, ...]] = {}
        self._imported_sessions: set[str] = set()
        self._last_native_rows: dict[str, BatchRowDescriptor] = {}
        self._text_append_committed_scopes: dict[str, set[str]] = {}
        self._queued_audio_soft: dict[str, deque[QueuedAudioSoftFrame]] = {}
        self._pre_roll_ring: dict[str, deque[QueuedAudioSoftFrame]] = {}
        self._assistant_audio_active_onset: dict[str, int | None] = {}
        self._policy_q2_soft_pending: dict[str, deque[QueuedAudioSoftFrame]] = {}
        self._ephemeral_decode_branches: dict[str, _EphemeralDecodeBranch] = {}
        self._kv_slot_roles: dict[str, KvOwnerRole] = {}
        self._kv_work_lock = threading.RLock()
        self._work_owners: dict[str, KvWorkOwner] = {}
        self._maintenance_states: dict[str, MemoryMaintenanceState] = {}
        self._maintenance_jobs: dict[str, _MemoryMaintenanceJob] = {}
        self._retiring_owners: dict[str, Any] = {}
        self._closing_sessions: set[str] = set()
        self._response_token_budgets: dict[tuple[str, int, str, int], _ResponseTokenBudget] = {}
        self._detached_response_decode_branches: deque[_EphemeralDecodeBranch] = deque()
        self._assistant_audio_queue_open: set[str] = set()
        self._assistant_audio_queue_first_frame_id: dict[str, int] = {}
        self._assistant_audio_queue_frame_spans: dict[str, list[_AssistantQueueFrameSpan]] = {}
        self._assistant_audio_queue_epoch: dict[str, str] = {}
        self._assistant_audio_queue_close_epoch: dict[str, int] = {}
        self._assistant_audio_queue_response_id: dict[str, str] = {}
        self._active_queued_audio_drains: dict[str, _QueuedAudioDrainState] = {}
        self._q2_decision_caches: dict[str, _Q2DecisionCache] = {}
        self._prepared_transition_ready_events: OrderedDict[int, Any] = OrderedDict()
        self._stable_q2_unusable_epochs: dict[str, str] = {}
        self._pending_q2_prefills: dict[str, tuple[int, str]] = {}
        self._q2_cache_disabled_for_tick: set[str] = set()
        self._control6_queue_trace_counts: dict[str, int] = {}
        self._reserved_tail_tokens: dict[str, int] = {}
        self._pending_promotions: dict[str, _PendingShadowPromotion] = {}
        self._outstanding_frames: dict[str, int] = {}
        self._outstanding_logical_tokens: dict[str, int] = {}
        self._recent_listener_q2: dict[str, deque[CanonicalListenerQ2Frame]] = {}
        self._compaction_requested: dict[str, int] = {}
        self._compaction_status: dict[str, dict[str, Any]] = {}
        self._compaction_revision: dict[str, int] = {}
        self._compaction_reason: dict[str, str] = {}
        self._requested_history_revision: dict[str, str] = {}
        self._compaction_attempt_history_revision: dict[str, str | None] = {}
        self._response_revision_lock = threading.Lock()
        self._latest_response_revisions: dict[tuple[str, str, int, str], str] = {}
        self._tick_future: Future[Mapping[str, ListenerMicrobatchRowOutput]] | None = None
        self._tick_expected: dict[str, int] | None = None
        self._tick_submitted_ns: int | None = None
        self._tick_cuda_dependencies: tuple[Any, ...] = ()
        self._processed_batches = 0
        self._torch_profile_trace_path = os.environ.get(
            "DUPLEX_LISTENER_TORCH_PROFILE_TRACE_PATH", ""
        ).strip()
        self._torch_profile_frame_ids: frozenset[int] = frozenset()
        self._torch_profiled_frame_ids: set[int] = set()
        self._torch_profiler: Any | None = None
        self._torch_profile_failed = False
        if self._torch_profile_trace_path:
            raw_profile_frames = os.environ.get(
                "DUPLEX_LISTENER_TORCH_PROFILE_FRAME_IDS", ""
            )
            try:
                parsed_profile_frames = frozenset(
                    int(part.strip())
                    for part in raw_profile_frames.split(",")
                    if part.strip()
                )
                if not parsed_profile_frames or any(frame_id < 0 for frame_id in parsed_profile_frames):
                    raise ValueError("frame IDs must be a non-empty set of non-negative integers")
                self._torch_profile_frame_ids = parsed_profile_frames
            except (TypeError, ValueError) as exc:
                logging.getLogger(__name__).warning(
                    "listener torch profiler disabled: invalid frame ID selection: %s", exc
                )
                self._torch_profile_trace_path = ""
        self._fatal_error: BaseException | None = None
        self._cleanup_error: BaseException | None = None
        self._notification_error: BaseException | None = None
        self._accepting = False
        self._stop_enqueued = False
        self._stop_requested = False
        self._drain_on_stop = True
        self._stop_deadline_ns: int | None = None
        self._drain_timeout_reached = False
        owner_thread_id = lambda: self._owner_thread_id
        self._startup: Future[None] = _OwnerAwareFuture(owner_thread_id)
        self._stopped: Future[WorkerStopReport] = _OwnerAwareFuture(owner_thread_id)

    def _trace_control6_queue_handoff(
        self,
        session_id: str,
        event: str,
        **fields: Any,
    ) -> None:
        if os.environ.get("DUPLEX_NATIVE_CONTROL6_TRACE") != "1":
            return
        emitted = self._control6_queue_trace_counts.get(session_id, 0)
        if emitted >= 30:
            return
        self._control6_queue_trace_counts[session_id] = emitted + 1
        logging.getLogger(__name__).info(
            "[control6_queue_handoff] seq=%d session=%s event=%s fields=%s",
            emitted + 1,
            session_id,
            event,
            fields,
        )

    @staticmethod
    def _trace_unified_kv_integrity(
        session_id: str,
        event: str,
        **fields: Any,
    ) -> None:
        """Emit cursor/provenance evidence only in the explicit diagnostic run."""
        if not _unified_kv_integrity_trace_enabled():
            return
        payload = {"session_id": session_id, "event": event, **fields}
        logging.getLogger(__name__).info(
            "[unified_kv_integrity] %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str),
        )

    def start(self) -> Future[None]:
        """Start the dedicated owner and return a future for resource readiness."""
        with self._wake:
            if self._thread is not None:
                raise RuntimeError("listener microbatch worker was already started")
            self._accepting = True
            self._thread = threading.Thread(
                target=self._run,
                name="listener-microbatch-gpu-owner",
                daemon=False,
            )
            self._thread.start()
        return self._startup

    def register_session(
        self,
        state: ListenerSessionState,
        *,
        prefix_import: PrefixImportFactory | None = None,
    ) -> Future[None]:
        """Register private state on the owner; cache import is lazy to first real row."""
        return self._submit_command("register", state, prefix_import)

    def submit_frame(self, frame: ListenerFrame) -> Future[None]:
        """Queue an immutable complete frame for owner-thread admission."""
        return self._submit_command("frame", frame)

    def submit_tick(
        self,
        frames: Sequence[ListenerFrame],
        *,
        cuda_dependencies: Sequence[Any] = (),
    ) -> Future[Mapping[str, ListenerMicrobatchRowOutput]]:
        """Submit a real tick and its preceding compute-stream dependencies."""
        rows = tuple(frames)
        dependencies = tuple(cuda_dependencies)
        future: Future[Mapping[str, ListenerMicrobatchRowOutput]] = _OwnerAwareFuture(
            lambda: self._owner_thread_id
        )
        if not rows:
            future.set_exception(ValueError("a shared listener tick must contain at least one real row"))
            return future
        if any(not isinstance(frame, ListenerFrame) for frame in rows):
            future.set_exception(TypeError("tick frames must be ListenerFrame values"))
            return future
        if len({frame.session_id for frame in rows}) != len(rows):
            future.set_exception(ValueError("a shared tick may contain at most one row per session"))
            return future
        if any(event is None for event in dependencies):
            future.set_exception(ValueError("CUDA dependencies cannot contain None"))
            return future
        return self._submit_command("submit_tick", rows, dependencies)

    def reset_session(
        self,
        session_id: str,
        new_state: ListenerSessionState,
        *,
        prefix_import: PrefixImportFactory,
    ) -> Future[None]:
        """Retire an idle native slot and replace its coordinator generation."""
        return self._submit_command("reset", session_id, new_state, prefix_import)

    def refresh_session_prefix(
        self,
        session_id: str,
        *,
        expected_frame_id: int,
        expected_state_version: int,
        cache_position: int,
        prefix_import: PrefixImportFactory,
    ) -> Future[None]:
        """Retire and rebase one idle row before its current tick is submitted."""
        return self._submit_command(
            "refresh_prefix",
            session_id,
            expected_frame_id,
            expected_state_version,
            cache_position,
            prefix_import,
        )

    def refresh_unified_prefix_before_audio(
        self,
        session_id: str,
        prefix_token_ids: Sequence[int],
        *,
        new_revision: int,
        expected_generation_id: int,
        expected_cache_position: int,
        prefix_import: PrefixImportFactory,
    ) -> Future[None]:
        """Replace the token-only unified prefix before the first audio frame.

        This is an owner-thread compare-and-swap for a pristine frame-zero
        session. It cannot replay, rebuild, or revise a prefix after user audio
        has entered the listener timeline.
        """
        token_ids = tuple(prefix_token_ids)
        if (
            not token_ids
            or len(token_ids) > NATIVE_CONTROL6_PREFIX_MAX_TOKENS
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in token_ids)
        ):
            future: Future[None] = Future()
            future.set_exception(ValueError(
                "unified prefix IDs must be 1.."
                f"{NATIVE_CONTROL6_PREFIX_MAX_TOKENS} non-negative integers"
            ))
            return future
        if (
            isinstance(new_revision, bool)
            or not isinstance(new_revision, int)
            or new_revision < 1
        ):
            future = Future()
            future.set_exception(ValueError("new unified prefix revision must be a positive integer"))
            return future
        if (
            isinstance(expected_generation_id, bool)
            or not isinstance(expected_generation_id, int)
            or expected_generation_id < 0
        ):
            future = Future()
            future.set_exception(ValueError("expected_generation_id must be a non-negative integer"))
            return future
        if (
            isinstance(expected_cache_position, bool)
            or not isinstance(expected_cache_position, int)
            or expected_cache_position < 0
        ):
            future = Future()
            future.set_exception(ValueError("expected_cache_position must be a non-negative integer"))
            return future
        if not callable(prefix_import):
            future = Future()
            future.set_exception(TypeError("prefix_import must be a callable owner-thread cache provider"))
            return future
        return self._submit_command(
            "refresh_unified_prefix_before_audio",
            session_id,
            token_ids,
            int(new_revision),
            int(expected_generation_id),
            int(expected_cache_position),
            prefix_import,
        )

    def prime_initial_prefix(
        self,
        session_id: str,
        prepared: PreparedInitialPrefix,
    ) -> Future[InitialPrefixAdmission]:
        """Import a prepared semantic prefix before frame zero without advancing it."""
        if not isinstance(prepared, PreparedInitialPrefix):
            future: Future[InitialPrefixAdmission] = Future()
            future.set_exception(TypeError("prepared must be PreparedInitialPrefix"))
            return future
        return self._submit_command("prime_initial_prefix", session_id, prepared)

    def append_text_tokens(
        self,
        session_id: str,
        token_ids: Sequence[int],
        *,
        expected_generation_id: int,
        expected_cache_position: int,
        consume_reserved_tokens: int = 0,
        allow_initial_cursor_rebase: bool = False,
        append_scope_id: str | None = None,
        response_cancel_event: Any | None = None,
        prepared_protocol: str | None = None,
    ) -> Future[TextAppendResult]:
        """Append one to eight text token IDs to the session's native KV owner.

        The future resolves to logits and final-norm hidden for the last
        committed token, plus a CUDA event recorded after the LM head on the
        listener owner stream. Consumers on another stream must wait for that
        event before reading either tensor. No audio frontend or prefix
        importer is invoked.
        """
        ids = tuple(token_ids)
        if not 1 <= len(ids) <= 8 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in ids
        ):
            future: Future[TextAppendResult] = Future()
            future.set_exception(ValueError("text append requires one to eight non-negative token IDs"))
            return future
        if (
            isinstance(consume_reserved_tokens, bool)
            or not isinstance(consume_reserved_tokens, int)
            or consume_reserved_tokens < 0
        ):
            future: Future[TextAppendResult] = Future()
            future.set_exception(ValueError("consume_reserved_tokens must be a non-negative integer"))
            return future
        if not isinstance(allow_initial_cursor_rebase, bool):
            future = Future()
            future.set_exception(ValueError("allow_initial_cursor_rebase must be a bool"))
            return future
        if append_scope_id is not None and (
            not isinstance(append_scope_id, str) or not append_scope_id
        ):
            future = Future()
            future.set_exception(ValueError("append_scope_id must be a non-empty string when provided"))
            return future
        if allow_initial_cursor_rebase and append_scope_id is None:
            future = Future()
            future.set_exception(ValueError("initial cursor rebase requires an append_scope_id"))
            return future
        if response_cancel_event is not None and not callable(
            getattr(response_cancel_event, "is_set", None)
        ):
            future = Future()
            future.set_exception(ValueError("response_cancel_event must expose is_set()"))
            return future
        if prepared_protocol is not None and (
            not isinstance(prepared_protocol, str) or not prepared_protocol.strip()
        ):
            future = Future()
            future.set_exception(ValueError("prepared_protocol must be a non-empty string or None"))
            return future
        request_submitted_ns = time.monotonic_ns()
        queue_depth_before_submit = self._commands.qsize()
        return self._submit_command(
            "append_text_tokens", session_id, ids,
            int(expected_generation_id), int(expected_cache_position),
            int(consume_reserved_tokens), allow_initial_cursor_rebase, append_scope_id,
            response_cancel_event, prepared_protocol, request_submitted_ns,
            queue_depth_before_submit,
        )

    def prepare_text_token_inputs(
        self,
        token_ids: Sequence[int],
        *,
        protocol: str,
    ) -> Future[Mapping[str, int]]:
        """Precompute one stable protocol transition on the GPU owner stream."""
        ids = tuple(token_ids)
        if not 1 <= len(ids) <= 8 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in ids
        ):
            future: Future[Mapping[str, int]] = Future()
            future.set_exception(ValueError(
                "prepared transition requires one to eight non-negative token IDs"
            ))
            return future
        if not isinstance(protocol, str) or not protocol.strip():
            future = Future()
            future.set_exception(ValueError("prepared transition requires a non-empty protocol"))
            return future
        return self._submit_command("prepare_text_token_inputs", ids, protocol.strip())

    def begin_memory_job(
        self, session_id: str, source_snapshot: Any,
    ) -> Future[MemoryMaintenanceHandle]:
        """Reserve SUMMARY on the owner thread using only declared sources."""
        return self._submit_command("begin_memory_job", session_id, source_snapshot)

    def append_memory_input(
        self, handle: MemoryMaintenanceHandle, prepared_chunk: Any,
    ) -> Future[Any]:
        return self._submit_command("append_memory_input", handle, prepared_chunk)

    def advance_memory_decode(self, handle: MemoryMaintenanceHandle) -> Future[Any]:
        return self._submit_command("advance_memory_decode", handle)

    def begin_canonical_rebuild(
        self, session_id: str, memory_revision: Any, retained_sources: Any,
    ) -> Future[MemoryMaintenanceHandle]:
        return self._submit_command("begin_canonical_rebuild", session_id, memory_revision, retained_sources)

    def append_rebuild_chunk(
        self, handle: MemoryMaintenanceHandle, prepared_chunk: Any,
    ) -> Future[Any]:
        """Append a private quantum; None closes the build or catch-up phase."""
        return self._submit_command("append_rebuild_chunk", handle, prepared_chunk)

    def promote_canonical_rebuild(
        self, handle: MemoryMaintenanceHandle, expected_source_revisions: Any = None,
    ) -> Future[Any]:
        return self._submit_command("promote_canonical_rebuild", handle, expected_source_revisions)

    def cancel_maintenance(self, handle: MemoryMaintenanceHandle) -> Future[None]:
        return self._submit_command("cancel_maintenance", handle)

    def begin_ephemeral_decode(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
    ) -> Future[EphemeralDecodeHandle]:
        """Fork an isolated native decode slot with a hard six-token budget.

        The fork does not register a listener session, open an assistant audio
        queue, or move any canonical cursor. Owner commands serialize its
        native work with listener Q2 while leaving the main row untouched.
        """
        return self._submit_command(
            "begin_ephemeral_decode", str(session_id),
            int(expected_generation_id), int(expected_cache_position),
        )

    def append_private_tokens(
        self,
        handle: EphemeralDecodeHandle,
        token_ids: Sequence[int],
        *,
        expected_cache_position: int,
    ) -> Future[PrivateTextAppendResult]:
        """Append one or two token IDs to a private decode branch."""
        ids = tuple(token_ids)
        if not isinstance(handle, EphemeralDecodeHandle):
            future: Future[PrivateTextAppendResult] = Future()
            future.set_exception(TypeError("private append requires an EphemeralDecodeHandle"))
            return future
        if len(ids) not in (1, 2) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in ids
        ):
            future = Future()
            future.set_exception(ValueError("private append requires one or two non-negative token IDs"))
            return future
        if (
            isinstance(expected_cache_position, bool)
            or not isinstance(expected_cache_position, int)
            or expected_cache_position < 0
        ):
            future = Future()
            future.set_exception(ValueError("expected_cache_position must be a non-negative integer"))
            return future
        return self._submit_command(
            "append_private_tokens", handle, ids, int(expected_cache_position),
        )

    def append_private_transition_tokens(
        self,
        handle: EphemeralDecodeHandle,
        token_ids: Sequence[int],
        *,
        expected_cache_position: int,
    ) -> Future[PrivateTextAppendResult]:
        """Append branch-local transition/prompt IDs without using sample budget.

        One logical transition of up to eight IDs is committed through one
        variable-Q native step. Physical even padding is excluded from the
        private cursor. The transition is accepted once before sampled tokens
        and never mutates the canonical session.
        """
        ids = tuple(token_ids)
        if not isinstance(handle, EphemeralDecodeHandle):
            future: Future[PrivateTextAppendResult] = Future()
            future.set_exception(TypeError("private transition requires an EphemeralDecodeHandle"))
            return future
        if len(ids) > MAX_STREAM_Q_LEN:
            future = Future()
            future.set_exception(ValueError("private transition accepts one to eight token IDs per native step"))
            return future
        if not ids or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in ids
        ):
            future = Future()
            future.set_exception(ValueError("private transition requires non-empty non-negative token IDs"))
            return future
        if (
            isinstance(expected_cache_position, bool)
            or not isinstance(expected_cache_position, int)
            or expected_cache_position < 0
        ):
            future = Future()
            future.set_exception(ValueError("expected_cache_position must be a non-negative integer"))
            return future
        return self._submit_command(
            "append_private_transition_tokens", handle, ids,
            int(expected_cache_position),
        )

    def end_ephemeral_decode(self, handle: EphemeralDecodeHandle) -> Future[None]:
        """Release a private decode slot; repeated end calls are harmless."""
        if not isinstance(handle, EphemeralDecodeHandle):
            future: Future[None] = Future()
            future.set_exception(TypeError("private decode end requires an EphemeralDecodeHandle"))
            return future
        return self._submit_command("end_ephemeral_decode", handle)

    def begin_response_decode(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        response_id: str,
        response_generation_id: int,
        revision_id: str | int,
        purpose: ResponseDecodePurpose | str = ResponseDecodePurpose.ACTIVE,
        source_watermark: ListenerWatermark | None = None,
        max_tokens: int = 768,
        prompt_token_budget: int = 774,
        transition_tokens: int | None = None,
        frozen_assistant_prefix_tokens: int = 0,
    ) -> Future[ResponseDecodeHandle]:
        """Atomically capture the latest completed listener row and fork it.

        When no source watermark is supplied, the returned handle carries the
        exact completed watermark observed by this owner command. Supplying a
        watermark adds a full listener-state CAS. ACTIVE and CANDIDATE forks
        have separate per-session slots and never append to canonical KV.
        """
        try:
            normalized_purpose = ResponseDecodePurpose(purpose)
        except (TypeError, ValueError):
            future: Future[ResponseDecodeHandle] = Future()
            future.set_exception(ValueError("purpose must be active_response or candidate_response"))
            return future
        if not isinstance(session_id, str) or not session_id.strip():
            future = Future()
            future.set_exception(ValueError("session_id must be non-empty"))
            return future
        if not isinstance(response_id, str) or not response_id.strip():
            future = Future()
            future.set_exception(ValueError("response_id must be non-empty"))
            return future
        for name, value in (
            ("expected_generation_id", expected_generation_id),
            ("response_generation_id", response_generation_id),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                future = Future()
                future.set_exception(ValueError(f"{name} must be a non-negative integer"))
                return future
        revision = str(revision_id)
        if not revision:
            future = Future()
            future.set_exception(ValueError("revision_id must be non-empty"))
            return future
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            future = Future()
            future.set_exception(ValueError("max_tokens must be a positive integer"))
            return future
        if max_tokens > ASSISTANT_RESPONSE_MAX_TOKENS:
            future = Future()
            future.set_exception(ValueError("response branch token budget may not exceed 768"))
            return future
        if (
            isinstance(prompt_token_budget, bool)
            or not isinstance(prompt_token_budget, int)
            or prompt_token_budget < 0
            or prompt_token_budget > LISTENER_MAX_SEQ_LEN
        ):
            future = Future()
            future.set_exception(ValueError(
                "prompt_token_budget must be in ["
                f"0,{LISTENER_MAX_SEQ_LEN}]"
            ))
            return future
        for name, value in (("transition_tokens", transition_tokens),
                            ("frozen_assistant_prefix_tokens", frozen_assistant_prefix_tokens)):
            if value is None and name == "transition_tokens":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                future = Future()
                future.set_exception(ValueError(f"{name} must be a non-negative integer"))
                return future
        if source_watermark is not None and (
            not isinstance(source_watermark, ListenerWatermark)
            or source_watermark.session_id != session_id
        ):
            future = Future()
            future.set_exception(ValueError("source_watermark must identify this session"))
            return future
        if normalized_purpose is ResponseDecodePurpose.CANDIDATE and (
            source_watermark is None
            or source_watermark.cache_epoch_id is None
            or source_watermark.applied_history_revision is None
        ):
            future = Future()
            future.set_exception(ValueError(
                "candidate response forks require a cache epoch and applied history watermark"
            ))
            return future
        revision_key = (
            session_id,
            str(response_id),
            int(response_generation_id),
            normalized_purpose.value,
        )
        with self._response_revision_lock:
            previous_revision = self._latest_response_revisions.get(revision_key)
            self._latest_response_revisions[revision_key] = revision
        return self._submit_command(
            "begin_response_decode",
            session_id,
            int(expected_generation_id),
            str(response_id),
            int(response_generation_id),
            revision,
            normalized_purpose,
            source_watermark,
            int(max_tokens),
            int(prompt_token_budget),
            revision_key,
            previous_revision,
            transition_tokens,
            frozen_assistant_prefix_tokens,
        )

    def append_response_decode(
        self,
        handle: ResponseDecodeHandle,
        token_ids: Sequence[int],
        *,
        expected_cache_position: int,
        generated: bool = True,
    ) -> Future[PrivateTextAppendResult]:
        """Append at most eight prompt or generated IDs to a private response.

        Prompt/template/frozen-assistant IDs use generated=False and must
        precede sampled response IDs. Each call is one owner quantum, allowing
        listener-ready rows to run before the next chunk.
        """
        ids = tuple(token_ids)
        if not isinstance(handle, EphemeralDecodeHandle) or handle.purpose not in {
            ResponseDecodePurpose.ACTIVE.value,
            ResponseDecodePurpose.CANDIDATE.value,
        }:
            future: Future[PrivateTextAppendResult] = Future()
            future.set_exception(TypeError("response append requires a response decode handle"))
            return future
        if not 1 <= len(ids) <= MAX_STREAM_Q_LEN or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in ids
        ):
            future = Future()
            future.set_exception(ValueError("response append requires one to eight non-negative token IDs"))
            return future
        if isinstance(expected_cache_position, bool) or not isinstance(
            expected_cache_position, int
        ) or expected_cache_position < 0:
            future = Future()
            future.set_exception(ValueError("expected_cache_position must be a non-negative integer"))
            return future
        if type(generated) is not bool:
            future = Future()
            future.set_exception(TypeError("generated must be a bool"))
            return future
        return self._submit_command(
            "append_response_decode",
            handle,
            ids,
            int(expected_cache_position),
            generated,
        )

    def promote_response_decode(
        self,
        candidate_handle: ResponseDecodeHandle,
        expected_active_handle: ResponseDecodeHandle | None,
        expected_active_revision_id: str | int | None,
    ) -> Future[ResponseDecodeHandle]:
        """Commit a prepared candidate branch into the active response slot.

        Promotion reuses the same private native slot and handle identity. The
        returned handle has purpose ``active_response`` and must replace the
        caller's candidate handle; decoder continuation state remains with the
        caller and the branch's KV cursor/logits are not rebuilt.
        """
        if (
            not isinstance(candidate_handle, EphemeralDecodeHandle)
            or candidate_handle.purpose != ResponseDecodePurpose.CANDIDATE.value
        ):
            future: Future[ResponseDecodeHandle] = Future()
            future.set_exception(TypeError("response promotion requires a candidate response handle"))
            return future
        if expected_active_handle is not None and (
            not isinstance(expected_active_handle, EphemeralDecodeHandle)
            or expected_active_handle.purpose != ResponseDecodePurpose.ACTIVE.value
        ):
            future = Future()
            future.set_exception(TypeError("expected_active_handle must be an active response handle or None"))
            return future
        if expected_active_revision_id is None:
            expected_revision = None
        elif isinstance(expected_active_revision_id, bool) or not isinstance(
            expected_active_revision_id, (str, int)
        ):
            future = Future()
            future.set_exception(TypeError("expected_active_revision_id must be a string, integer, or None"))
            return future
        else:
            expected_revision = str(expected_active_revision_id)
            if not expected_revision:
                future = Future()
                future.set_exception(ValueError("expected_active_revision_id must be non-empty"))
                return future
        if (expected_active_handle is None) != (expected_revision is None):
            future = Future()
            future.set_exception(ValueError(
                "expected active handle and revision must both be present or both be None"
            ))
            return future
        if expected_active_handle is not None and expected_active_handle.revision_id != expected_revision:
            future = Future()
            future.set_exception(ValueError(
                "expected_active_revision_id must match expected_active_handle.revision_id"
            ))
            return future
        return self._submit_command(
            "promote_response_decode",
            candidate_handle,
            expected_active_handle,
            expected_revision,
        )

    def end_response_decode(self, handle: ResponseDecodeHandle) -> Future[None]:
        """Release one private response branch after prior stream work."""
        if not isinstance(handle, EphemeralDecodeHandle) or handle.purpose not in {
            ResponseDecodePurpose.ACTIVE.value,
            ResponseDecodePurpose.CANDIDATE.value,
        }:
            future: Future[None] = Future()
            future.set_exception(TypeError("response end requires a response decode handle"))
            return future
        return self._submit_command("end_response_decode", handle)

    def begin_assistant_audio_queue(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
        reserve_tail_tokens: int = 0,
        response_id: str | None = None,
    ) -> Future[ListenerSessionSnapshot]:
        """Atomically reserve the close tail and open bounded soft-audio capture."""
        if (
            isinstance(reserve_tail_tokens, bool)
            or not isinstance(reserve_tail_tokens, int)
            or reserve_tail_tokens < 0
        ):
            future: Future[ListenerSessionSnapshot] = Future()
            future.set_exception(ValueError("reserve_tail_tokens must be a non-negative integer"))
            return future
        if response_id is not None and (not isinstance(response_id, str) or not response_id):
            future = Future()
            future.set_exception(ValueError("response_id must be a non-empty string when provided"))
            return future
        return self._submit_command(
            "begin_assistant_audio_queue", session_id,
            int(expected_generation_id), int(expected_cache_position),
            reserve_tail_tokens, response_id,
        )

    def mark_assistant_audio_queue_closing(
        self,
        session_id: str,
        *,
        queue_close_epoch: int,
    ) -> Future[None]:
        """Publish the server queue-close epoch before its drain command is queued."""
        if isinstance(queue_close_epoch, bool) or not isinstance(queue_close_epoch, int) or queue_close_epoch < 0:
            future: Future[None] = Future()
            future.set_exception(ValueError("queue_close_epoch must be a non-negative integer"))
            return future
        return self._submit_command(
            "mark_assistant_audio_queue_closing", str(session_id), int(queue_close_epoch),
        )

    @property
    def q2_decision_prefill_supported(self) -> bool:
        runtime = self._runtime
        native = None if runtime is None else runtime.native
        return bool(
            native is not None
            and getattr(native, "_metadata_rebind_api_available", False) is True
            and callable(getattr(native, "rebind_ephemeral_session_metadata", None))
            and callable(getattr(native, "fork_session_slot", None))
            and callable(getattr(native, "step_queued_audio_async", None))
        )

    def request_q2_decision_prefill(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        response_id: str,
    ) -> Future[bool]:
        """Request owner-idle prefill without waiting on native replay."""
        if not isinstance(response_id, str) or not response_id:
            future: Future[bool] = Future()
            future.set_exception(ValueError("response_id must be a non-empty string"))
            return future
        return self._submit_command(
            "request_q2_decision_prefill",
            str(session_id),
            int(expected_generation_id),
            response_id,
        )

    def reserve_tail_tokens(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
        token_count: int,
    ) -> Future[int]:
        """CAS-update reserved KV for a model-turn close boundary and next Q2."""
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            future: Future[int] = Future()
            future.set_exception(ValueError("token_count must be a non-negative integer"))
            return future
        return self._submit_command(
            "reserve_tail_tokens", session_id,
            int(expected_generation_id), int(expected_cache_position), token_count,
        )

    def reserve_response_decode_budget(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
        requested_tokens: int,
        transition_tokens: int,
        sentence_completion_tokens: int = 0,
    ) -> Future[dict[str, int | str | None]]:
        """Preflight Thinker capacity and reserve its text append before decode."""
        values = (requested_tokens, transition_tokens, sentence_completion_tokens)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            future: Future[dict[str, int | str | None]] = Future()
            future.set_exception(ValueError("response decode budgets must be non-negative integers"))
            return future
        if sentence_completion_tokens > 32:
            future = Future()
            future.set_exception(ValueError("sentence completion extension cannot exceed 32 tokens"))
            return future
        return self._submit_command(
            "reserve_response_decode_budget",
            str(session_id),
            int(expected_generation_id),
            int(expected_cache_position),
            int(requested_tokens),
            int(transition_tokens),
            int(sentence_completion_tokens),
        )

    def drain_queued_audio(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
        user_turn_already_open: bool = False,
    ) -> Future[ListenerSessionSnapshot]:
        """Commit assistant-phase Q2 soft tokens after the caller's user boundary."""
        if type(user_turn_already_open) is not bool:
            future: Future[ListenerSessionSnapshot] = Future()
            future.set_exception(TypeError("user_turn_already_open must be a bool"))
            return future
        drain_submitted_at_ns = _perf_counter_ns()
        return self._submit_command(
            "drain_queued_audio", session_id,
            int(expected_generation_id), int(expected_cache_position),
            user_turn_already_open, int(drain_submitted_at_ns),
        )

    def rollback_assistant_branch(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
        target_cache_position: int,
    ) -> Future[ListenerSessionSnapshot]:
        """CAS-rewind generated assistant KV while retaining queued user audio."""
        return self._submit_command(
            "rollback_assistant_branch", session_id,
            int(expected_generation_id), int(expected_cache_position),
            int(target_cache_position),
        )

    def stage_semantic_prefix(
        self,
        request: Any,
        *,
        listener_generation_id: int,
        sequence_position: int,
    ) -> bool:
        """Stage a frozen context refresh without delaying the active tick."""
        shadow = self._shadow_worker
        if shadow is None:
            raise RuntimeError("semantic-history shadow runtime is not enabled")
        session_id = str(request.session_id)
        with self._kv_work_lock:
            state = self._session_states.get(session_id)
            if (state is None or state.closed or session_id in self._closing_sessions
                    or self._fatal_error is not None
                    or state.generation_id != listener_generation_id):
                return False
            self._reap_shadow_work_owner(session_id)
            budget = BranchTokenBudget(int(request.cache_position), 0, 0, 0)
            self._check_branch_budget(session_id, budget)
            if (budget.total_tokens > self._kv_budgets.new_cache_target_tokens
                    or int(request.prefix_len) > self._kv_budgets.initial_prefix_max_tokens):
                raise ValueError("rebuilt cache exceeds its service token target or prefix budget")
            self._reap_retiring_owners(session_id=session_id)
            if any(item.session_id == session_id for item in self._retiring_owners.values()):
                return False
            owner = self._work_owners.get(session_id)
            if owner is not None and not owner.shadow_job:
                return False
            if owner is None:
                owner = self._reserve_work_owner(
                    session_id, KvWorkKind.REBUILD,
                    BranchTokenBudget(int(request.cache_position), 0, 0, 0), shadow_job=True,
                )
            try:
                accepted = shadow.stage(
                    request, listener_generation_id=listener_generation_id,
                    sequence_position=sequence_position,
                )
            except BaseException:
                self._reap_shadow_work_owner(session_id)
                raise
            self._reap_shadow_work_owner(session_id)
            return accepted

    def mark_semantic_prefix_applied(
        self,
        request: Any,
        *,
        listener_generation_id: int,
        current_frame_id: int | None = None,
        listener_state_version: int | None = None,
    ) -> bool:
        """Notify the shadow after primary cache import and state commit.

        The call only updates shadow bookkeeping under its condition lock; all
        native stage retirement stays on the shadow owner thread.
        """
        shadow = self._shadow_worker
        if shadow is None:
            return False
        shadow.mark_primary_applied(
            request,
            listener_generation_id=listener_generation_id,
            current_frame_id=current_frame_id,
            listener_state_version=listener_state_version,
        )
        return True

    def invalidate_semantic_prefix(self, session_id: str) -> bool:
        """Invalidate a staged semantic cache when its request is no longer safe."""
        shadow = self._shadow_worker
        if shadow is None:
            return False
        shadow.invalidate_session(str(session_id))
        return True

    def close_session(self, session_id: str) -> Future[RetiredSession]:
        """Close an idle session after ordered native-context release."""
        return self._submit_command("close", session_id)

    def fork_native_session(self, source_session_id: str, fork_session_id: str) -> Future[str]:
        """Create an independent native slot at one idle primary boundary."""
        return self._submit_command(
            "fork_native_session", str(source_session_id), str(fork_session_id)
        )

    def session_state(self, session_id: str) -> Future[ListenerSessionSnapshot]:
        """Request an owner-thread snapshot between batches, never during one."""
        return self._submit_command("snapshot", session_id)

    def request_listener_compaction(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        reason: str = "confirmed_history_commit",
        history_revision: str | int,
    ) -> Future[Mapping[str, Any]]:
        """Schedule an asynchronous context rebase after confirmed history changes.

        This only requests a background shadow stage; it never mutates the
        canonical cache or waits for prefix construction/replay.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            future: Future[Mapping[str, Any]] = Future()
            future.set_exception(ValueError("session_id must be non-empty"))
            return future
        if (
            isinstance(expected_generation_id, bool)
            or not isinstance(expected_generation_id, int)
            or expected_generation_id < 0
        ):
            future = Future()
            future.set_exception(ValueError("expected_generation_id must be non-negative"))
            return future
        if not isinstance(reason, str) or not reason.strip():
            future = Future()
            future.set_exception(ValueError("reason must be non-empty"))
            return future
        if (
            isinstance(history_revision, bool)
            or not isinstance(history_revision, (str, int))
            or (isinstance(history_revision, str) and not history_revision.strip())
            or (isinstance(history_revision, int) and history_revision < 0)
        ):
            future = Future()
            future.set_exception(ValueError("history_revision must be a non-empty string or non-negative integer"))
            return future
        return self._submit_command(
            "request_listener_compaction",
            session_id,
            int(expected_generation_id),
            reason.strip(),
            str(history_revision),
        )

    def listener_compaction_status(
        self, session_id: str,
    ) -> Future[Mapping[str, Any]]:
        """Read one session's bounded-history staging and admission status."""
        return self._submit_command("compaction_status", str(session_id))

    def stop(
        self,
        *,
        drain: bool = True,
        drain_timeout_s: float = 5.0,
    ) -> Future[WorkerStopReport]:
        """Stop after draining accepted frames, or report exact pending depth."""
        self._reject_owner_reentry("stop")
        if not isinstance(drain, bool):
            raise TypeError("drain must be a bool")
        if (
            isinstance(drain_timeout_s, bool)
            or not isinstance(drain_timeout_s, (int, float))
            or not math.isfinite(float(drain_timeout_s))
            or drain_timeout_s <= 0
        ):
            raise ValueError("drain_timeout_s must be a finite positive number")
        with self._wake:
            if self._thread is None:
                if not self._stopped.done():
                    self._stopped.set_result(self._stop_report("not_started", drained=True))
                return self._stopped
            if not self._stop_enqueued:
                self._accepting = False
                self._stop_enqueued = True
                deadline_ns = time.monotonic_ns() + int(float(drain_timeout_s) * 1_000_000_000)
                self._commands.put(_Command("stop", (drain, deadline_ns)))
                self._wake.notify_all()
        return self._stopped

    @property
    def owner_thread_id(self) -> int | None:
        return self._owner_thread_id

    @property
    def runtime_signature(self):
        """Shared runtime signature, available after startup completes."""
        if self._coordinator is None:
            raise RuntimeError("listener worker runtime has not completed startup")
        return self._coordinator.runtime_signature

    @property
    def fatal_error(self) -> BaseException | None:
        return self._fatal_error

    @property
    def semantic_shadow_status(self) -> Mapping[str, Any]:
        """Return a lock-safe CPU snapshot of the optional history shadow."""
        shadow = self._shadow_worker
        if shadow is None:
            return {
                "enabled": False,
                "initialized": False,
                "thread_alive": False,
                "latest_error": None,
                "staged_sessions": 0,
                "queued_sessions": 0,
                "candidate_session_id": None,
                "active_stage": None,
                "candidate_stage": None,
                "primary_applied": (),
                "promotions_completed": 0,
                "diagnostics": _empty_semantic_shadow_diagnostics(),
            }
        return {"enabled": True, **dict(shadow.status)}

    def _submit_command(self, operation: str, *args: Any) -> Future[Any]:
        self._reject_owner_reentry(operation)
        future: Future[Any] = _OwnerAwareFuture(lambda: self._owner_thread_id)
        with self._wake:
            if self._thread is None:
                future.set_exception(RuntimeError("worker must be started before commands are submitted"))
                return future
            if not self._accepting:
                future.set_exception(RuntimeError("worker is stopping; new commands are rejected"))
                return future
            self._commands.put(_Command(
                operation,
                args,
                future,
                submitted_ns=time.perf_counter_ns(),
            ))
            self._wake.notify_all()
        return future

    def _notify_shadow_promotion_fenced(
        self, candidate: ShadowPrefixCandidate,
    ) -> None:
        """Queue CPU metadata publication after the shadow owner fenced the copy."""
        with self._wake:
            if self._thread is None or not self._accepting:
                return
            self._commands.put(_Command(
                "semantic_promotion_fenced",
                (str(candidate.row.session_id), int(candidate.candidate_id)),
            ))
            self._wake.notify_all()

    def _reject_owner_reentry(self, operation: str) -> None:
        if (
            self._owner_thread_id is not None
            and threading.get_ident() == self._owner_thread_id
        ):
            raise RuntimeError(
                f"listener worker owner thread cannot submit {operation!r}; "
                "callbacks must schedule follow-up work with "
                "loop.call_soon_threadsafe and return"
            )

    def _run(self) -> None:
        self._owner_thread_id = threading.get_ident()
        try:
            runtime = self._runtime_factory()
            if isinstance(runtime, ListenerMicrobatchRuntime):
                self._runtime = runtime
            if self._coordinator is None:
                native = getattr(runtime, "native", None)
                signature = getattr(native, "runtime_signature", None)
                capability = getattr(native, "capability", None)
                if signature is None or capability is None:
                    raise TypeError("runtime native binding must expose signature and capability")
                self._coordinator = ListenerMicrobatchCoordinator(
                    signature, capability, coalesce_window_ns=0
                )
            self._validate_runtime(runtime)
            self._runtime = runtime
            if runtime.shadow_factory is not None or runtime.shadow_prefix_builder is not None:
                if not callable(runtime.shadow_factory) or not callable(runtime.shadow_prefix_builder):
                    raise TypeError("shadow runtime factory and prefix builder must be configured together")
                self._shadow_worker = SemanticPrefixShadowWorker(
                    runtime.shadow_factory,
                    runtime.shadow_prefix_builder,
                    runtime_signature=runtime.native.runtime_signature,
                    max_pending_q2=self._kv_budgets.recent_audio_reference_frames,
                    on_promotion_fenced=self._notify_shadow_promotion_fenced,
                )
                self._shadow_worker.start().result()
            self._startup.set_result(None)
            while True:
                self._drain_commands()
                if self._should_stop():
                    break
                if self._fatal_error is not None:
                    self._wait_for_work(None)
                    continue
                coordinator = self._coordinator
                if coordinator is None:
                    raise RuntimeError("listener coordinator is unavailable")
                batch = coordinator.take_ready(
                    time.monotonic_ns(),
                    before_manifest=self._promote_semantic_before_manifest,
                )
                if batch is not None:
                    self._process_batch(batch)
                    continue
                if coordinator.ready_depth == 0:
                    self._promote_idle_semantic_candidates()
                if self._run_one_queued_audio_drain_quantum():
                    continue
                if self._run_pending_q2_prefill_idle_step():
                    continue
                timeout = self._next_timeout_seconds()
                shadow_compaction_in_progress = any(
                    status.get("status") in {"building", "catching_up", "promoting"}
                    for status in self._compaction_status.values()
                )
                if self._shadow_worker is not None and shadow_compaction_in_progress:
                    timeout = 0.02 if timeout is None else min(timeout, 0.02)
                self._wait_for_work(timeout)
        except BaseException as exc:
            if not self._startup.done():
                self._startup.set_exception(exc)
            if self._fatal_error is None:
                self._fatal_error = exc
        finally:
            with self._wake:
                self._accepting = False
                self._stop_enqueued = True
                self._wake.notify_all()
            runtime = self._runtime
            if runtime is not None:
                try:
                    self._end_all_ephemeral_decodes()
                except BaseException as exc:
                    if self._cleanup_error is None:
                        self._cleanup_error = exc
                for session_id in tuple(self._q2_decision_caches):
                    try:
                        self._release_q2_decision_cache(session_id)
                    except BaseException as exc:
                        if self._cleanup_error is None:
                            self._cleanup_error = exc
                try:
                    _drain_cuda_stream_for_cleanup(runtime.listener_stream)
                    self._reap_retiring_owners()
                    self._maintenance_states.clear()
                except BaseException as exc:
                    self._cleanup_error = exc
                shadow = self._shadow_worker
                self._shadow_worker = None
                if shadow is not None:
                    try:
                        shadow.stop().result()
                    except BaseException as exc:
                        if self._cleanup_error is None:
                            self._cleanup_error = exc
                try:
                    runtime.native.close()
                except BaseException as exc:
                    if self._cleanup_error is None:
                        self._cleanup_error = exc
                if runtime.close_resources is not None:
                    try:
                        runtime.close_resources()
                    except BaseException as exc:
                        if self._cleanup_error is None:
                            self._cleanup_error = exc
            self._fail_pending_commands()
            if not self._startup.done():
                self._startup.set_exception(RuntimeError("worker stopped before startup completed"))
            if not self._stopped.done():
                coordinator = self._coordinator
                self._stopped.set_result(
                    self._stop_report(
                        "stopped",
                        drained=coordinator is None or coordinator.ready_depth == 0,
                    )
                )
            tick_future = self._tick_future
            self._tick_future = None
            self._tick_expected = None
            self._tick_submitted_ns = None
            if tick_future is not None and not tick_future.done():
                tick_future.set_exception(RuntimeError("listener worker stopped before shared tick completed"))
            for drain in tuple(self._active_queued_audio_drains.values()):
                if not drain.future.done():
                    drain.future.set_exception(
                        RuntimeError("listener worker stopped before queued-audio drain cleanup")
                    )
                drain.queued.clear()
                drain.handoff_queue.clear()
            self._active_queued_audio_drains.clear()
            for ring in self._pre_roll_ring.values():
                ring.clear()
            for backlog in self._queued_audio_soft.values():
                backlog.clear()
            for pending in self._policy_q2_soft_pending.values():
                pending.clear()
            self._assistant_audio_active_onset.clear()

    def _validate_runtime(self, runtime: ListenerMicrobatchRuntime) -> None:
        if not isinstance(runtime, ListenerMicrobatchRuntime):
            raise TypeError("runtime_factory must return ListenerMicrobatchRuntime")
        if not isinstance(runtime.kv_budgets, KvCacheBudgets):
            raise TypeError("kv_budgets must be KvCacheBudgets")
        if (
            isinstance(runtime.compaction_trigger_tokens, bool)
            or not isinstance(runtime.compaction_trigger_tokens, int)
            or runtime.compaction_trigger_tokens < 1
        ):
            raise ValueError("compaction_trigger_tokens must be a positive integer")
        if (
            isinstance(runtime.compaction_target_tokens, bool)
            or not isinstance(runtime.compaction_target_tokens, int)
            or not 1 <= runtime.compaction_target_tokens <= runtime.compaction_trigger_tokens
        ):
            raise ValueError("compaction_target_tokens must be positive and no greater than the trigger")
        if runtime.compaction_request_builder is not None and (
            not callable(runtime.shadow_factory)
            or not callable(runtime.shadow_prefix_builder)
        ):
            raise ValueError(
                "automatic listener compaction requires the independent shadow runtime and prefix builder"
            )
        if not isinstance(runtime.frontend, ListenerMicrobatchAudioFrontend):
            raise TypeError("runtime must provide the staged batched audio frontend")
        if runtime.policy is not None and not isinstance(runtime.policy, GemmaNativeControl6BatchPolicy):
            raise TypeError("runtime policy must be batched native Control6 or None")
        if runtime.control6_scorer is not None and not isinstance(
            runtime.control6_scorer, GemmaNativeControl6BatchPolicy
        ):
            raise TypeError("runtime raw scorer must be batched native Control6 or None")
        if not callable(getattr(runtime.native, "step_async", None)):
            raise TypeError("runtime native binding lacks batch step_async")
        if not callable(getattr(runtime.native, "ensure_capacity", None)):
            raise TypeError("runtime native binding lacks dynamic ensure_capacity")
        if not callable(getattr(runtime.native, "import_session", None)):
            raise TypeError("runtime native binding lacks import_session")
        if not callable(getattr(runtime.native, "release_session_async", None)):
            raise TypeError("runtime native binding lacks release_session_async")
        if not callable(getattr(runtime.native, "close", None)):
            raise TypeError("runtime native binding lacks owner-thread close")
        if runtime.shadow_factory is not None and not callable(
            getattr(runtime.native, "copy_session_cache_from", None)
        ):
            raise TypeError("primary native binding lacks semantic-cache promotion")
        if runtime.shadow_factory is not None and not bool(
            getattr(runtime.native, "_cache_copy_api_available", False)
        ):
            raise RuntimeError(
                "native batch DSO lacks the cache-promotion API required by semantic staging"
            )
        if runtime.policy is not None and not callable(runtime.policy_inputs):
            raise TypeError("runtime policy_inputs factory is required with batched Control6")
        if runtime.native.runtime_signature != self._coordinator.runtime_signature:
            raise RuntimeError("native runtime signature differs from coordinator signature")
        capability = runtime.native.capability
        if not isinstance(capability, NativeBatchCapability):
            raise TypeError("native binding must expose its attested batch capability")
        capability.validate_for(self._coordinator.runtime_signature)
        if capability != self._coordinator.capability:
            raise RuntimeError("native stream/capability differs from coordinator capability")
        if capability.stream_priority != -1:
            raise RuntimeError("production listener dispatcher must retain stream priority -1")
        if not bool(getattr(capability, "supports_arbitrary_batch_size", False)):
            raise RuntimeError("native listener must attest arbitrary positive batch sizes")
        if runtime.native.listener_stream is not runtime.listener_stream:
            raise RuntimeError("frontend/native runtime must use the exact same Listener stream object")
        stream_device = torch.device(runtime.listener_stream.device)
        if runtime.frontend.device != stream_device:
            raise RuntimeError("frontend and Listener stream device differ")
        if runtime.policy is not None and runtime.policy.adapter.lora_A.device != stream_device:
            raise RuntimeError("Control6 adapter and Listener stream device differ")
        if runtime.control6_scorer is not None and runtime.control6_scorer.adapter.lora_A.device != stream_device:
            raise RuntimeError("Control6 adapter and Listener stream device differ")
        native_device = getattr(runtime.native, "_model_device", stream_device)
        if torch.device(native_device) != stream_device:
            raise RuntimeError("native listener and Listener stream device differ")
        if stream_device.type == "cuda" and runtime.completion_barrier is not None:
            raise ValueError("custom completion barriers are test-only; CUDA uses a recorded event")

    def _drain_commands(self) -> None:
        command = self._take_ready_deferred_command()
        if command is None:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
        try:
            if command.future is not None and command.future.cancelled():
                return
            if command.operation in {"reset", "close"}:
                session_id = str(command.args[0])
                if session_id in self._active_queued_audio_drains:
                    self._deferred_owner_commands.setdefault(
                        session_id, deque()
                    ).append(command)
                    return
            if self._fatal_error is not None and command.operation not in {
                "snapshot", "stop", "semantic_promotion_fenced",
            }:
                raise RuntimeError(f"worker is fail-closed after batch error: {self._fatal_error}")
            self._reap_retiring_owners()
            result = self._execute_command(command)
            if (isinstance(result, MemoryMaintenanceHandle) and command.future is not None
                    and command.future.cancelled()):
                self._cancel_maintenance(result)
                return
            if (
                command.operation in {"begin_ephemeral_decode", "begin_response_decode"}
                and command.future is not None
                and command.future.cancelled()
                and isinstance(result, EphemeralDecodeHandle)
            ):
                self._end_ephemeral_decode(result)
                return
            if (
                command.operation == "fork_native_session"
                and command.future is not None
                and command.future.cancelled()
            ):
                fork_session_id = str(command.args[1])
                if fork_session_id in self._session_states:
                    self._close_session(fork_session_id)
                return
            if (
                command.operation == "drain_queued_audio"
                and result is _PENDING_OWNER_RESULT
            ):
                return
            if (
                command.operation != "submit_tick"
                and command.future is not None
                and not command.future.done()
            ):
                command.future.set_result(result)
            if command.operation == "promote_response_decode":
                # The owner CAS has already removed the displaced handle from
                # the live registry. Resolve the promotion Future before
                # fencing/releasing its native row so the application CAS
                # never waits on cleanup while holding response locks.
                self._drain_detached_response_decode_cleanup()
            # _run() checks coordinator readiness before entering this method
            # again, so continuous control traffic cannot starve submit_tick.
        except BaseException as exc:
            if (
                command.operation == "append_text_tokens"
                and _unified_kv_integrity_trace_enabled()
            ):
                (
                    session_id,
                    token_ids,
                    generation_id,
                    expected_cache_position,
                    _consume_reserved_tokens,
                    _allow_initial_cursor_rebase,
                    append_scope_id,
                    _response_cancel_event,
                    _prepared_protocol,
                    request_submitted_ns,
                    queue_depth_before_submit,
                ) = command.args
                state = self._session_states.get(str(session_id))
                failure_fields: dict[str, Any] = {
                    "status": "failed",
                    "response_id": append_scope_id,
                    "generation_id": int(generation_id),
                    "request_q_len": len(token_ids),
                    "expected_cache_position": int(expected_cache_position),
                    "request_submitted_ns": int(request_submitted_ns),
                    "queue_depth_before_submit": int(queue_depth_before_submit),
                    "failure_observed_ns": time.monotonic_ns(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                if state is not None:
                    failure_fields.update(
                        state_epoch_id=str(state.state_epoch_id),
                        current_generation_id=int(state.generation_id),
                    )
                self._trace_unified_kv_integrity(
                    str(session_id), "canonical_text_append", **failure_fields
                )
            if command.future is not None and not command.future.done():
                if command.operation == "begin_response_decode":
                    revision_id = str(command.args[4])
                    revision_key = command.args[9]
                    previous_revision = command.args[10]
                    if not isinstance(exc, StaleResponseRevisionError):
                        with self._response_revision_lock:
                            if self._latest_response_revisions.get(revision_key) == revision_id:
                                if previous_revision is None:
                                    self._latest_response_revisions.pop(revision_key, None)
                                else:
                                    self._latest_response_revisions[revision_key] = previous_revision
                command.future.set_exception(exc)

    def _take_ready_deferred_command(self) -> _Command | None:
        for session_id, pending in tuple(self._deferred_owner_commands.items()):
            if session_id in self._active_queued_audio_drains or not pending:
                continue
            command = pending.popleft()
            if not pending:
                self._deferred_owner_commands.pop(session_id, None)
            return command
        return None

    def _execute_command(self, command: _Command) -> Any:
        maintenance_commands = {
            "begin_memory_job": self._begin_memory_job,
            "append_memory_input": self._append_memory_input,
            "advance_memory_decode": self._advance_memory_decode,
            "begin_canonical_rebuild": self._begin_canonical_rebuild,
            "append_rebuild_chunk": self._append_rebuild_chunk,
            "promote_canonical_rebuild": self._promote_canonical_rebuild,
            "cancel_maintenance": self._cancel_maintenance,
        }
        if command.operation in maintenance_commands:
            return maintenance_commands[command.operation](*command.args)
        if command.operation == "register":
            state, prefix_import = command.args
            return self._register_session(state, prefix_import)
        if command.operation == "fork_native_session":
            source_session_id, fork_session_id = command.args
            return self._fork_native_session(source_session_id, fork_session_id)
        if command.operation == "frame":
            (frame,) = command.args
            if not isinstance(frame, ListenerFrame):
                raise TypeError("frame must be a complete ListenerFrame")
            if self._coordinator.ready_depth == 0:
                self._promote_idle_semantic_candidates(only_session_id=frame.session_id)
            self._require_listener_frame_capacity(frame)
            self._coordinator.enqueue(frame)
            self._outstanding_frames[frame.session_id] = (
                self._outstanding_frames.get(frame.session_id, 0) + 1
            )
            self._outstanding_logical_tokens[frame.session_id] = (
                self._outstanding_logical_tokens.get(frame.session_id, 0)
                + int(frame.logical_q_len)
            )
            return None
        if command.operation == "submit_tick":
            if len(command.args) == 2:
                frames, cuda_dependencies = command.args
            else:
                (frames,) = command.args
                cuda_dependencies = ()
            if command.future is None:
                raise RuntimeError("submit_tick requires a result Future")
            if self._tick_future is not None and not self._tick_future.done():
                raise RuntimeError("only one shared tick may be in flight")
            normalized_frames: list[ListenerFrame] = []
            for frame in frames:
                state = self._session_states.get(frame.session_id)
                if (
                    frame.queue_audio_for_next_turn
                    and frame.session_id not in self._assistant_audio_queue_open
                    and state is not None
                ):
                    logging.getLogger(__name__).warning(
                        "reclassifying assistant-phase voice frame after queue closure: "
                        "session=%s frame=%s",
                        frame.session_id,
                        frame.frame_id,
                    )
                    frame = ListenerFrame(
                        session_id=frame.session_id,
                        generation_id=frame.generation_id,
                        state_epoch_id=frame.state_epoch_id,
                        frame_id=frame.frame_id,
                        ready_at_ns=frame.ready_at_ns,
                        pcm=frame.pcm,
                        logical_q_len=LISTENER_Q_PER_FRAME,
                        ephemeral_reason=None,
                        queue_audio_for_next_turn=False,
                        queue_epoch=frame.queue_epoch,
                        queue_audio_reason=None,
                        queue_audio_onset_frame_id=None,
                        reference_pcm=frame.reference_pcm,
                        reference_frame_id=frame.reference_frame_id,
                        policy_context=frame.policy_context,
                    )
                if frame.logical_q_len == 0 and frame.session_id in self._assistant_audio_queue_open:
                    close_epoch = self._assistant_audio_queue_close_epoch.get(frame.session_id)
                    crossed_queue_close = (
                        close_epoch is not None
                        and frame.queue_epoch is not None
                        and int(frame.queue_epoch) >= close_epoch
                    )
                    retain_after_close = bool(
                        crossed_queue_close
                        and frame.logical_q_len > 0
                        and not frame.queue_audio_for_next_turn
                    )
                    self._trace_control6_queue_handoff(
                        frame.session_id,
                        "owner_normalize_shared_queue",
                        frame_id=int(frame.frame_id),
                        source_logical_q_len=int(frame.logical_q_len),
                        source_ephemeral_reason=frame.ephemeral_reason,
                        source_queue_audio_for_next_turn=bool(frame.queue_audio_for_next_turn),
                        queue_close_epoch=close_epoch,
                        source_queue_epoch=frame.queue_epoch,
                        retain_post_close_frame=retain_after_close,
                        queue_open=True,
                    )
                    frame = ListenerFrame(
                        session_id=frame.session_id,
                        generation_id=frame.generation_id,
                        state_epoch_id=frame.state_epoch_id,
                        frame_id=frame.frame_id,
                        ready_at_ns=frame.ready_at_ns,
                        pcm=frame.pcm,
                        logical_q_len=0,
                        ephemeral_reason="assistant_turn_open",
                        queue_audio_for_next_turn=bool(
                            frame.queue_audio_for_next_turn or retain_after_close
                        ),
                        queue_epoch=frame.queue_epoch,
                        queue_audio_reason=(
                            frame.queue_audio_reason
                            if frame.queue_audio_for_next_turn
                            else "utterance_body" if retain_after_close else None
                        ),
                        queue_audio_onset_frame_id=frame.queue_audio_onset_frame_id,
                        reference_pcm=frame.reference_pcm,
                        reference_frame_id=frame.reference_frame_id,
                        policy_context=frame.policy_context,
                    )
                normalized_frames.append(frame)
            frames = tuple(normalized_frames)
            expected = {frame.session_id: int(frame.frame_id) for frame in frames}
            if len(expected) != len(frames):
                raise ValueError("shared tick session IDs must be unique")
            for frame in frames:
                state = self._session_states.get(frame.session_id)
                if state is None:
                    raise KeyError(f"unregistered listener session: {frame.session_id}")
                if (
                    frame.generation_id != state.generation_id
                    or frame.state_epoch_id != state.state_epoch_id
                    or frame.frame_id != state.next_frame_id
                ):
                    raise ValueError(
                        f"shared tick row identity is stale for session {frame.session_id}"
                    )
            if self._coordinator.ready_depth == 0:
                for frame in frames:
                    self._promote_idle_semantic_candidates(only_session_id=frame.session_id)
            for frame in frames:
                self._require_listener_frame_capacity(frame)
            # The shared dispatcher has already closed this tick's row set.
            # Enqueue all rows in this one owner command so coordinator wakeup
            # cannot start a singleton before sibling rows are admitted.
            self._coordinator.coalesce_window_ns = 0
            for frame in frames:
                self._trace_control6_queue_handoff(
                    frame.session_id,
                    "frame_enqueue",
                    frame_id=int(frame.frame_id),
                    logical_q_len=int(frame.logical_q_len),
                    queue_audio_for_next_turn=bool(frame.queue_audio_for_next_turn),
                    queue_open_at_enqueue=frame.session_id in self._assistant_audio_queue_open,
                    cache_position=int(self._session_states[frame.session_id].cache_position),
                )
                self._coordinator.enqueue(frame)
                self._outstanding_frames[frame.session_id] = (
                    self._outstanding_frames.get(frame.session_id, 0) + 1
                )
                self._outstanding_logical_tokens[frame.session_id] = (
                    self._outstanding_logical_tokens.get(frame.session_id, 0)
                    + int(frame.logical_q_len)
                )
            self._tick_expected = expected
            self._tick_future = command.future
            self._tick_submitted_ns = (
                time.perf_counter_ns()
                if command.submitted_ns is None
                else int(command.submitted_ns)
            )
            self._tick_cuda_dependencies = tuple(cuda_dependencies)
            return None
        if command.operation == "reset":
            session_id, state, prefix_import = command.args
            return self._reset_session(session_id, state, prefix_import)
        if command.operation == "refresh_prefix":
            session_id, frame_id, state_version, cache_position, prefix_import = command.args
            return self._refresh_session_prefix(
                session_id,
                frame_id,
                state_version,
                cache_position,
                prefix_import,
            )
        if command.operation == "refresh_unified_prefix_before_audio":
            session_id, token_ids, revision, generation_id, cache_position, prefix_import = command.args
            return self._refresh_unified_prefix_before_audio(
                session_id,
                token_ids,
                new_revision=revision,
                expected_generation_id=generation_id,
                expected_cache_position=cache_position,
                prefix_import=prefix_import,
            )
        if command.operation == "prime_initial_prefix":
            session_id, prepared = command.args
            return self._prime_initial_prefix(session_id, prepared)
        if command.operation == "append_text_tokens":
            (session_id, token_ids, generation_id, cache_position,
             consume_reserved_tokens, allow_initial_cursor_rebase, append_scope_id,
             response_cancel_event, prepared_protocol, request_submitted_ns,
             queue_depth_before_submit) = command.args
            return self._append_text_tokens(
                session_id, token_ids,
                expected_generation_id=generation_id,
                expected_cache_position=cache_position,
                consume_reserved_tokens=consume_reserved_tokens,
                allow_initial_cursor_rebase=allow_initial_cursor_rebase,
                append_scope_id=append_scope_id,
                response_cancel_event=response_cancel_event,
                prepared_protocol=prepared_protocol,
                request_submitted_ns=request_submitted_ns,
                queue_depth_before_submit=queue_depth_before_submit,
            )
        if command.operation == "prepare_text_token_inputs":
            token_ids, protocol = command.args
            return self._prepare_text_token_inputs(token_ids, protocol=protocol)
        if command.operation == "begin_ephemeral_decode":
            session_id, generation_id, cache_position = command.args
            return self._begin_ephemeral_decode(
                session_id,
                expected_generation_id=generation_id,
                expected_cache_position=cache_position,
            )
        if command.operation == "begin_response_decode":
            (
                session_id,
                generation_id,
                response_id,
                response_generation_id,
                revision_id,
                purpose,
                source_watermark,
                max_tokens,
                prompt_token_budget,
                revision_key,
                _previous_revision,
                transition_tokens,
                frozen_assistant_prefix_tokens,
            ) = command.args
            return self._begin_response_decode(
                session_id,
                expected_generation_id=generation_id,
                response_id=response_id,
                response_generation_id=response_generation_id,
                revision_id=revision_id,
                purpose=purpose,
                source_watermark=source_watermark,
                max_tokens=max_tokens,
                prompt_token_budget=prompt_token_budget,
                revision_key=revision_key,
                transition_tokens=transition_tokens,
                frozen_assistant_prefix_tokens=frozen_assistant_prefix_tokens,
            )
        if command.operation == "promote_response_decode":
            candidate_handle, expected_active_handle, expected_revision = command.args
            return self._promote_response_decode(
                candidate_handle,
                expected_active_handle=expected_active_handle,
                expected_active_revision_id=expected_revision,
            )
        if command.operation == "append_private_tokens":
            handle, token_ids, cache_position = command.args
            return self._append_private_tokens(
                handle, token_ids, expected_cache_position=cache_position,
            )
        if command.operation == "append_response_decode":
            handle, token_ids, cache_position, generated = command.args
            return self._append_response_decode_ids(
                handle,
                token_ids,
                expected_cache_position=cache_position,
                generated=generated,
            )
        if command.operation == "append_private_transition_tokens":
            handle, token_ids, cache_position = command.args
            return self._append_private_transition_tokens(
                handle, token_ids, expected_cache_position=cache_position,
            )
        if command.operation == "end_ephemeral_decode":
            (handle,) = command.args
            return self._end_ephemeral_decode(handle)
        if command.operation == "end_response_decode":
            (handle,) = command.args
            return self._end_response_decode(handle)
        if command.operation == "begin_assistant_audio_queue":
            session_id, generation_id, cache_position, reserve_tail_tokens, response_id = command.args
            return self._begin_assistant_audio_queue(
                session_id, expected_generation_id=generation_id,
                expected_cache_position=cache_position,
                reserve_tail_tokens=reserve_tail_tokens,
                response_id=response_id,
            )
        if command.operation == "mark_assistant_audio_queue_closing":
            session_id, queue_close_epoch = command.args
            return self._mark_assistant_audio_queue_closing(
                session_id, queue_close_epoch=queue_close_epoch,
            )
        if command.operation == "request_q2_decision_prefill":
            session_id, generation_id, response_id = command.args
            return self._request_q2_decision_prefill(
                session_id,
                expected_generation_id=generation_id,
                response_id=response_id,
            )
        if command.operation == "reserve_tail_tokens":
            session_id, generation_id, cache_position, token_count = command.args
            return self._reserve_tail_tokens(
                session_id,
                expected_generation_id=generation_id,
                expected_cache_position=cache_position,
                token_count=token_count,
            )
        if command.operation == "reserve_response_decode_budget":
            session_id, generation_id, cache_position, requested, transition, extension = command.args
            return self._reserve_response_decode_budget(
                session_id,
                expected_generation_id=generation_id,
                expected_cache_position=cache_position,
                requested_tokens=requested,
                transition_tokens=transition,
                sentence_completion_tokens=extension,
            )
        if command.operation == "drain_queued_audio":
            (
                session_id, generation_id, cache_position, user_turn_already_open,
                submitted_at_ns,
            ) = command.args
            return self._drain_queued_audio(
                session_id, expected_generation_id=generation_id,
                expected_cache_position=cache_position,
                user_turn_already_open=user_turn_already_open,
                drain_submitted_at_ns=submitted_at_ns,
                drain_future=command.future,
            )
        if command.operation == "rollback_assistant_branch":
            session_id, generation_id, cache_position, target_position = command.args
            return self._rollback_assistant_branch(
                session_id, expected_generation_id=generation_id,
                expected_cache_position=cache_position,
                target_cache_position=target_position,
            )
        if command.operation == "close":
            (session_id,) = command.args
            return self._close_session(session_id)
        if command.operation == "semantic_promotion_fenced":
            session_id, candidate_id = command.args
            return self._finalize_shadow_promotion(
                str(session_id), int(candidate_id)
            )
        if command.operation == "snapshot":
            (session_id,) = command.args
            return self._coordinator.session_state(session_id)
        if command.operation == "compaction_status":
            (session_id,) = command.args
            status = self._compaction_status.get(str(session_id))
            if status is None:
                raise KeyError(f"unknown listener session: {session_id}")
            return {
                **status,
                "recent_q2_frames": len(self._recent_listener_q2.get(str(session_id), ())),
                "operational_cache_budget_tokens": self._kv_budgets.operational_cache_budget_tokens,
                "memory_preparation_trigger_tokens": self._kv_budgets.memory_preparation_trigger_tokens,
                "desired_swap_threshold_tokens": self._kv_budgets.desired_swap_threshold_tokens,
                "desired_swap_due": self._session_states[str(session_id)].cache_position
                    >= self._kv_budgets.desired_swap_threshold_tokens,
                "shadow": dict(self.semantic_shadow_status),
            }
        if command.operation == "request_listener_compaction":
            session_id, generation_id, reason, history_revision = command.args
            state = self._session_states.get(str(session_id))
            if state is None:
                raise KeyError(f"unknown listener session: {session_id}")
            if state.generation_id != int(generation_id):
                raise StaleListenerWatermarkError(
                    "confirmed-history compaction request belongs to a stale generation"
                )
            self._compaction_reason[str(session_id)] = str(reason)
            self._requested_history_revision[str(session_id)] = str(history_revision)
            self._maybe_request_listener_compaction(
                str(session_id), force_retry=True,
            )
            return {
                **self._compaction_status.get(str(session_id), {}),
                "requested_history_revision": str(history_revision),
            }
        if command.operation == "stop":
            drain, deadline_ns = command.args
            self._stop_requested = True
            self._drain_on_stop = bool(drain)
            self._stop_deadline_ns = int(deadline_ns)
            return None
        raise RuntimeError(f"unknown owner command: {command.operation}")

    def _append_text_tokens(
        self,
        session_id: str,
        token_ids: tuple[int, ...],
        *,
        expected_generation_id: int,
        expected_cache_position: int,
        consume_reserved_tokens: int = 0,
        allow_initial_cursor_rebase: bool = False,
        append_scope_id: str | None = None,
        response_cancel_event: Any | None = None,
        prepared_protocol: str | None = None,
        request_submitted_ns: int,
        queue_depth_before_submit: int,
    ) -> TextAppendResult:
        if session_id in self._active_queued_audio_drains:
            raise RuntimeError("canonical text append rejected while queued-audio drain is active")
        owner_started_ns = time.monotonic_ns()
        timings_ns: dict[str, int] = {
            "request_submitted_ns": int(request_submitted_ns),
            "owner_started_ns": owner_started_ns,
            "owner_queue_wait_ns": max(0, owner_started_ns - int(request_submitted_ns)),
        }
        runtime = self._runtime
        coordinator = self._coordinator
        if runtime is None or coordinator is None:
            raise RuntimeError("unified native text runtime is unavailable")
        adapter = runtime.token_adapter
        if adapter is None or not callable(getattr(adapter, "prepare", None)) or not callable(
            getattr(adapter, "logits_from_final_norm_hidden", None)
        ):
            raise RuntimeError("runtime has no exact Gemma token/PLE adapter")
        if session_id not in self._imported_sessions:
            raise RuntimeError(
                "unified text append requires the session's true chat prefix imported into native KV"
            )
        if response_cancel_event is not None and response_cancel_event.is_set():
            raise RuntimeError("text append response canceled before native KV mutation")
        snapshot = coordinator.session_state(session_id)
        if snapshot.generation_id != expected_generation_id:
            self._trace_control6_queue_handoff(
                session_id,
                "append_reject_generation",
                scope=append_scope_id,
                expected_generation=int(expected_generation_id),
                actual_generation=int(snapshot.generation_id),
                expected_cursor=int(expected_cache_position),
                actual_cursor=int(snapshot.cache_position),
                valid_history=int(snapshot.valid_history_tokens),
            )
            raise RuntimeError(
                "unified text append lost its generation/cache-position compare-and-swap: "
                f"expected_generation={expected_generation_id}, "
                f"actual_generation={snapshot.generation_id}, "
                f"expected_cache_position={expected_cache_position}, "
                f"actual_cache_position={snapshot.cache_position}, "
                f"actual_valid_history_tokens={snapshot.valid_history_tokens}"
            )
        if (
            allow_initial_cursor_rebase
            and append_scope_id in self._text_append_committed_scopes.get(session_id, set())
        ):
            self._trace_control6_queue_handoff(
                session_id,
                "append_reject_rebase_reused",
                scope=append_scope_id,
                expected_cursor=int(expected_cache_position),
                actual_cursor=int(snapshot.cache_position),
            )
            raise RuntimeError("initial text cursor rebase is only valid before this response's first append")
        self._trace_control6_queue_handoff(
            session_id,
            "append_check",
            scope=append_scope_id,
            token_count=len(token_ids),
            first_append=(append_scope_id not in self._text_append_committed_scopes.get(session_id, set())),
            allow_initial_cursor_rebase=bool(allow_initial_cursor_rebase),
            expected_generation=int(expected_generation_id),
            actual_generation=int(snapshot.generation_id),
            expected_cursor=int(expected_cache_position),
            actual_cursor=int(snapshot.cache_position),
            valid_history=int(snapshot.valid_history_tokens),
            queue_open=session_id in self._assistant_audio_queue_open,
            outstanding_frames=int(self._outstanding_frames.get(session_id, 0)),
        )
        if snapshot.cache_position != expected_cache_position:
            can_rebase_initial_append = (
                allow_initial_cursor_rebase
                and snapshot.cache_position >= expected_cache_position
                and snapshot.valid_history_tokens == snapshot.cache_position
            )
            if not can_rebase_initial_append:
                self._trace_control6_queue_handoff(
                    session_id,
                    "append_reject_cursor",
                    scope=append_scope_id,
                    expected_cursor=int(expected_cache_position),
                    actual_cursor=int(snapshot.cache_position),
                    valid_history=int(snapshot.valid_history_tokens),
                )
                raise RuntimeError(
                    "unified text append lost its generation/cache-position compare-and-swap: "
                    f"expected_generation={expected_generation_id}, "
                    f"actual_generation={snapshot.generation_id}, "
                    f"expected_cache_position={expected_cache_position}, "
                    f"actual_cache_position={snapshot.cache_position}, "
                    f"actual_valid_history_tokens={snapshot.valid_history_tokens}"
                )
            # This command runs on the same owner thread as listener KV advancement.
            # Use the current, contiguous cursor for the very first uncommitted append.
            self._trace_control6_queue_handoff(
                session_id,
                "append_rebase_first",
                scope=append_scope_id,
                expected_cursor=int(expected_cache_position),
                actual_cursor=int(snapshot.cache_position),
                valid_history=int(snapshot.valid_history_tokens),
            )
            expected_cache_position = int(snapshot.cache_position)
        elif snapshot.valid_history_tokens != expected_cache_position:
            self._trace_control6_queue_handoff(
                session_id,
                "append_reject_valid_history",
                scope=append_scope_id,
                expected_cursor=int(expected_cache_position),
                actual_cursor=int(snapshot.cache_position),
                valid_history=int(snapshot.valid_history_tokens),
            )
            raise RuntimeError(
                "unified text append lost its generation/cache-position compare-and-swap: "
                f"expected_generation={expected_generation_id}, "
                f"actual_generation={snapshot.generation_id}, "
                f"expected_cache_position={expected_cache_position}, "
                f"actual_cache_position={snapshot.cache_position}, "
                f"actual_valid_history_tokens={snapshot.valid_history_tokens}"
            )
        if self._outstanding_frames.get(session_id, 0) != 0 or (
            self._tick_expected is not None and session_id in self._tick_expected
        ):
            raise RuntimeError("text append requires completed listener frame watermark")
        reserved_tail = self._reserved_tail_tokens.get(session_id, 0)
        if (
            isinstance(consume_reserved_tokens, bool)
            or not isinstance(consume_reserved_tokens, int)
            or consume_reserved_tokens < 0
            or consume_reserved_tokens > len(token_ids)
            or consume_reserved_tokens > reserved_tail
        ):
            raise ValueError("text append cannot consume more reserved tail tokens than it commits")
        previous_row = self._last_native_rows.get(session_id)
        if previous_row is None or snapshot.native_context_id is None:
            raise RuntimeError("unified text append has no imported native session row")
        queued_tokens = 2 * len(self._queued_audio_soft.get(session_id, ()))
        remaining_reserved = reserved_tail - consume_reserved_tokens
        if (
            snapshot.cache_position + len(token_ids) + queued_tokens + remaining_reserved
            > self._operational_max_seq_len()
        ):
            raise RuntimeError(
                "native listener capacity cannot preserve queued next-turn audio before text append; "
                f"session={session_id}, cache_position={snapshot.cache_position}, "
                f"append_tokens={len(token_ids)}, queued_audio_tokens={queued_tokens}, "
                f"reserved_tail_tokens={remaining_reserved}, "
                f"max_seq_len={self._operational_max_seq_len()}"
            )
        physical_q = max(LISTENER_Q_PER_FRAME, len(token_ids) + (len(token_ids) & 1))
        if snapshot.cache_position > self._operational_max_seq_len() - physical_q:
            raise RuntimeError(
                "native listener physical text-Q capacity exhausted before text append; "
                f"session={session_id}, cache_position={snapshot.cache_position}, "
                f"physical_q={physical_q}, max_seq_len={self._operational_max_seq_len()}"
            )

        ready_ns = time.monotonic_ns()
        token_fingerprint = hashlib.sha256(
            b"unified-text-token-ids:" + b",".join(str(value).encode("ascii") for value in token_ids)
        ).hexdigest()
        row = replace(
            previous_row,
            session_generation=snapshot.generation_id,
            state_epoch_id=snapshot.state_epoch_id,
            frame_id=snapshot.next_frame_id,
            listener_state_version=snapshot.state_version,
            sequence_position=snapshot.sequence_position,
            cache_position=snapshot.cache_position,
            valid_history_tokens=snapshot.valid_history_tokens,
            native_context_id=int(snapshot.native_context_id),
            input_frame_sha256=token_fingerprint,
            reference_pcm_sha256=None,
            frame_ready_at_ns=ready_ns,
            reference_frame_id=None,
            logical_q_len=len(token_ids),
            ephemeral_reason=None,
            queue_audio_for_next_turn=False,
            queue_audio_reason=None,
            queue_audio_onset_frame_id=None,
        )
        capability = coordinator.capability
        manifest = ListenerBatchManifest(
            manifest_id=uuid.uuid4().hex,
            runtime_signature=coordinator.runtime_signature,
            stream_identity=capability.stream_identity,
            stream_priority=capability.stream_priority,
            dispatch_at_ns=ready_ns,
            max_coalesce_wait_ns=0,
            dispatch_reason="carryover_ready",
            rows=(row,),
        )
        with _listener_stream_context(runtime.listener_stream), torch.inference_mode():
            timings_ns["token_prepare_started_ns"] = time.monotonic_ns()
            prepare_kwargs = {"logical_q_lens": len(token_ids)}
            if prepared_protocol is not None:
                prepare_kwargs["protocol"] = prepared_protocol
            cache_before = getattr(adapter, "prepared_transition_cache_info", {})
            hits_before = int(cache_before.get("hits", 0)) if isinstance(cache_before, Mapping) else 0
            prepared = adapter.prepare(token_ids, **prepare_kwargs)
            cache_after = getattr(adapter, "prepared_transition_cache_info", {})
            hits_after = int(cache_after.get("hits", 0)) if isinstance(cache_after, Mapping) else hits_before
            transition_cache_hit = (
                hits_after > hits_before if prepared_protocol is not None else None
            )
            for tensor in (prepared.inputs_embeds, prepared.per_layer_inputs):
                if tensor.is_cuda:
                    tensor.record_stream(runtime.listener_stream)
            readiness_event = self._prepared_transition_ready_events.get(id(prepared))
            if readiness_event is not None:
                runtime.listener_stream.wait_event(readiness_event)
                self._prepared_transition_ready_events.move_to_end(id(prepared))
            timings_ns["token_prepare_returned_ns"] = time.monotonic_ns()
            if tuple(prepared.logical_q_lens) != (len(token_ids),):
                raise RuntimeError("token adapter logical-Q metadata differs from text append")
            if tuple(prepared.inputs_embeds.shape) != (1, physical_q, CONTROL_HIDDEN_WIDTH):
                raise RuntimeError("token adapter returned an incompatible physical text-Q embedding shape")
            if tuple(prepared.per_layer_inputs.shape) != (1, physical_q, 35, 256):
                raise RuntimeError("token adapter returned an incompatible projected PLE shape")
            mutation_started = False
            try:
                mutation_started = True
                if response_cancel_event is not None and response_cancel_event.is_set():
                    mutation_started = False
                    raise RuntimeError("text append response canceled before native KV mutation")
                timings_ns["native_step_started_ns"] = time.monotonic_ns()
                output = runtime.native.step_async(
                    manifest,
                    prepared.inputs_embeds,
                    per_layer_inputs=prepared.per_layer_inputs,
                    producer_stream=runtime.listener_stream,
                )
                timings_ns["native_step_returned_ns"] = time.monotonic_ns()
                timings_ns["native_health_started_ns"] = time.monotonic_ns()
                health_rows = tuple(runtime.native.verify_output_health(output))
                timings_ns["native_health_returned_ns"] = time.monotonic_ns()
                if (
                    len(health_rows) != 1
                    or health_rows[0].row != row
                    or health_rows[0].generation != output.generation
                    or not health_rows[0].checked
                    or health_rows[0].nonfinite
                ):
                    raise RuntimeError("native text output health/provenance check failed")
                updated = coordinator.advance_cache_position(
                    session_id,
                    token_count=len(token_ids),
                    expected_generation_id=expected_generation_id,
                    expected_cache_position=expected_cache_position,
                )
                if append_scope_id is not None:
                    self._text_append_committed_scopes.setdefault(session_id, set()).add(
                        append_scope_id
                    )
                self._trace_control6_queue_handoff(
                    session_id,
                    "append_commit",
                    scope=append_scope_id,
                    token_count=len(token_ids),
                    previous_cursor=int(expected_cache_position),
                    actual_cursor=int(updated.cache_position),
                    queue_open=session_id in self._assistant_audio_queue_open,
                )
                self._last_native_rows[session_id] = replace(
                    row,
                    frame_id=updated.next_frame_id,
                    listener_state_version=updated.state_version,
                    sequence_position=updated.sequence_position,
                    cache_position=updated.cache_position,
                    valid_history_tokens=updated.valid_history_tokens,
                )
                self._reserved_tail_tokens[session_id] = remaining_reserved
                output_row = 0 if len(token_ids) > LISTENER_Q_PER_FRAME else len(token_ids) - 1
                hidden = output.tensor[0, output_row, :].reshape(1, 1, -1)
                timings_ns["lm_head_started_ns"] = time.monotonic_ns()
                logits = adapter.logits_from_final_norm_hidden(hidden)
                timings_ns["lm_head_returned_ns"] = time.monotonic_ns()
                if logits.ndim == 3:
                    logits = logits[:, -1, :]
                if logits.ndim != 2 or logits.shape[0] != 1:
                    raise RuntimeError("native text LM head returned an incompatible logits shape")
                ready_event = torch.cuda.Event(
                    enable_timing=False,
                    blocking=False,
                    interprocess=False,
                )
                ready_event.record(runtime.listener_stream)
                timings_ns["ready_event_recorded_ns"] = time.monotonic_ns()
                timings_ns["owner_completed_ns"] = time.monotonic_ns()
                self._trace_unified_kv_integrity(
                    session_id,
                    "canonical_text_append",
                    status="committed",
                    response_id=append_scope_id,
                    generation_id=int(expected_generation_id),
                    state_epoch_id=str(snapshot.state_epoch_id),
                    native_context_id=int(snapshot.native_context_id),
                    request_q_len=len(token_ids),
                    queue_depth_before_submit=int(queue_depth_before_submit),
                    cache_position_before=int(snapshot.cache_position),
                    cache_position_after=int(updated.cache_position),
                    timings_ns=dict(timings_ns),
                )
                return TextAppendResult(
                    logits=logits,
                    final_norm_hidden=hidden,
                    ready_event=ready_event,
                    timings_ns=dict(timings_ns),
                    cache_position=int(updated.cache_position),
                    prepared_transition_cache_hit=transition_cache_hit,
                )
            except BaseException as exc:
                self._trace_control6_queue_handoff(
                    session_id,
                    "append_exception",
                    scope=append_scope_id,
                    expected_cursor=int(expected_cache_position),
                    queue_open=session_id in self._assistant_audio_queue_open,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                if mutation_started:
                    self._fatal_error = RuntimeError(
                        f"unified native text append failed after cache mutation: {exc}"
                    )
                    raise self._fatal_error from exc
                raise

    def _prepare_text_token_inputs(
        self,
        token_ids: tuple[int, ...],
        *,
        protocol: str,
    ) -> Mapping[str, int]:
        runtime = self._runtime
        adapter = None if runtime is None else runtime.token_adapter
        if runtime is None or adapter is None or not callable(getattr(adapter, "prepare", None)):
            raise RuntimeError("unified native text runtime is unavailable for transition preparation")
        with _listener_stream_context(runtime.listener_stream), torch.inference_mode():
            prepared = adapter.prepare(
                token_ids,
                logical_q_lens=len(token_ids),
                protocol=protocol,
            )
            for tensor in (prepared.inputs_embeds, prepared.per_layer_inputs):
                if tensor.is_cuda:
                    tensor.record_stream(runtime.listener_stream)
            if prepared.inputs_embeds.is_cuda:
                readiness_event = torch.cuda.Event(
                    enable_timing=False,
                    blocking=False,
                    interprocess=False,
                )
                readiness_event.record(runtime.listener_stream)
                prepared_id = id(prepared)
                self._prepared_transition_ready_events[prepared_id] = readiness_event
                self._prepared_transition_ready_events.move_to_end(prepared_id)
                while len(self._prepared_transition_ready_events) > 8:
                    self._prepared_transition_ready_events.popitem(last=False)
        info = getattr(adapter, "prepared_transition_cache_info", {})
        return {
            "logical_q_len": len(token_ids),
            "physical_q_len": int(prepared.inputs_embeds.shape[1]),
            "cache_size": int(info.get("size", 0)),
            "cache_hits": int(info.get("hits", 0)),
            "cache_misses": int(info.get("misses", 0)),
        }

    def _check_unified_session_cas(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
    ) -> ListenerSessionSnapshot:
        runtime = self._runtime
        coordinator = self._coordinator
        if runtime is None or coordinator is None or runtime.token_adapter is None:
            raise RuntimeError("unified native session owner is unavailable")
        if session_id not in self._imported_sessions:
            raise RuntimeError("unified native operation requires an imported true chat prefix")
        snapshot = coordinator.session_state(session_id)
        if (
            snapshot.generation_id != expected_generation_id
            or snapshot.cache_position != expected_cache_position
            or snapshot.valid_history_tokens != expected_cache_position
        ):
            raise RuntimeError("unified native operation lost its generation/cache-position compare-and-swap")
        return snapshot

    @property
    def _kv_budgets(self) -> KvCacheBudgets:
        runtime = self._runtime
        return KvCacheBudgets() if runtime is None else runtime.kv_budgets

    def _operational_max_seq_len(self) -> int:
        if self._coordinator is None:
            raise RuntimeError("listener coordinator is unavailable")
        return min(self._kv_budgets.operational_cache_budget_tokens,
                   int(self._coordinator.capability.max_seq_len))

    def _check_branch_budget(self, session_id: str, budget: BranchTokenBudget) -> None:
        limit = self._operational_max_seq_len()
        if budget.total_tokens > limit:
            state = self._session_states[session_id]
            raise RecoverableSessionCapacityError(
                session_id, generation_id=state.generation_id, frame_id=state.next_frame_id,
                cache_position=budget.prefix_tokens,
                requested_tokens=budget.total_tokens - budget.prefix_tokens,
                max_seq_len=limit, reason="context_limit",
            )

    def _set_maintenance_state(self, session_id: str, state: MemoryMaintenanceState) -> None:
        self._maintenance_states[session_id] = state

    def _maintenance_job(self, handle: MemoryMaintenanceHandle) -> _MemoryMaintenanceJob:
        if not isinstance(handle, MemoryMaintenanceHandle):
            raise TypeError("maintenance requires a MemoryMaintenanceHandle")
        job = self._maintenance_jobs.get(handle.session_id)
        if job is None or job.handle != handle:
            raise RuntimeError("unknown or stale maintenance handle")
        state = self._session_states.get(handle.session_id)
        if (state is None or state.closed or state.generation_id != handle.session_generation_id
                or state.state_epoch_id != handle.state_epoch_id):
            self._fail_maintenance(job)
            raise RuntimeError("maintenance lost its session generation")
        if self._work_owners.get(handle.session_id) != job.owner:
            raise RuntimeError("maintenance lost its WORK owner")
        return job

    def _reap_retiring_owners(self, *, session_id: str | None = None) -> None:
        runtime = self._runtime
        if runtime is None or self._fatal_error is not None:
            return
        for key, retired in tuple(self._retiring_owners.items()):
            if session_id is not None and retired.session_id != session_id:
                continue
            if retired.ready_event is not None and not retired.ready_event.query():
                continue
            try:
                with _listener_stream_context(runtime.listener_stream):
                    runtime.native.release_session_async(retired.row)
            except BaseException as exc:
                self._cleanup_error = exc
                self._fatal_error = exc
                raise
            self._retiring_owners.pop(key)
            self._kv_slot_roles.pop(key, None)

    def _retire_memory_row(self, session_id: str, row: BatchRowDescriptor, event: Any) -> None:
        self._kv_slot_roles[row.session_id] = KvOwnerRole.RETIRING
        self._retiring_owners[row.session_id] = _RetiringMemoryOwner(session_id, row, event)

    def _fail_maintenance(self, job: _MemoryMaintenanceJob) -> None:
        try:
            with _listener_stream_context(self._runtime.listener_stream):
                job.last_ready_event = _record_shadow_event(self._runtime.listener_stream)
        except BaseException as cleanup_error:
            self._cleanup_error = cleanup_error
            self._fatal_error = cleanup_error
        self._retire_memory_row(job.handle.session_id, job.row, job.last_ready_event)
        self._maintenance_jobs.pop(job.handle.session_id, None)
        self._release_work_owner(job.owner)
        self._set_maintenance_state(job.handle.session_id, MemoryMaintenanceState.FAILED)
        if bool(getattr(self._runtime.native, "_poisoned", False)):
            self._fatal_error = RuntimeError("native memory maintenance failed after KV mutation")

    def _open_maintenance(
        self, session_id: str, sources: MemorySourceSnapshot, kind: KvWorkKind,
        *, covered_through_frame_id: int, memory_revision: Any = None,
    ) -> MemoryMaintenanceHandle:
        if not isinstance(sources, MemorySourceSnapshot):
            raise TypeError("maintenance needs a frozen MemorySourceSnapshot")
        runtime = self._runtime
        state = self._session_states.get(session_id)
        if runtime is None or state is None or session_id not in self._imported_sessions:
            raise RuntimeError("maintenance requires an imported live session")
        native = runtime.native
        if not isinstance(getattr(native, "_session_by_id", None), dict) or not isinstance(
            getattr(native, "_session_by_context", None), dict
        ):
            raise RuntimeError("maintenance requires the native session binding registry")
        if self._outstanding_frames.get(session_id, 0) or (
            self._tick_expected is not None and session_id in self._tick_expected
        ):
            raise RuntimeError("maintenance must begin at a completed listener tick")
        prefix = sources.prefix
        position = prefix.prefix_len if prefix.cache_position is None else prefix.cache_position
        if position < prefix.prefix_len or prefix.prefix_len > self._kv_budgets.initial_prefix_max_tokens:
            raise ValueError("maintenance prefix exceeds its stable prefix budget")
        if sources.through_frame_id >= state.next_frame_id:
            raise ValueError("memory source cutoff exceeds completed listener frames")
        if kind is KvWorkKind.SUMMARY and not sources.eos_token_ids:
            raise ValueError("SUMMARY requires the model's explicit semantic EOS token IDs")
        if kind is KvWorkKind.REBUILD:
            canonical_binding = native._session_by_id[session_id]
            if canonical_binding.prefix_len != prefix.prefix_len:
                raise ValueError("REBUILD must retain the native stable prefix length")
        with self._kv_work_lock:
            self._reap_retiring_owners(session_id=session_id)
            self._reap_shadow_work_owner(session_id)
            if session_id in self._work_owners or any(
                item.session_id == session_id for item in self._retiring_owners.values()
            ):
                raise RuntimeError("WORK role already occupied")
            handle = MemoryMaintenanceHandle(
                uuid.uuid4().hex, session_id, state.generation_id, state.state_epoch_id,
                kind, time.monotonic_ns(),
            )
            owner = self._reserve_work_owner(
                session_id, kind, BranchTokenBudget(position, 0, 0,
                    self._kv_budgets.generated_memory_max_tokens if kind is KvWorkKind.SUMMARY else 0),
                kv_owner_epoch=handle.handle_id,
            )
            row = replace(
                self._last_native_rows[session_id],
                session_id=f"__memory__{session_id}__{handle.handle_id}",
                native_context_id=self._private_context_id(int(state.native_context_id)),
                cache_position=position, valid_history_tokens=position,
                logical_q_len=LISTENER_Q_PER_FRAME, ephemeral_reason=None,
                queue_audio_for_next_turn=False, queue_audio_reason=None,
                queue_audio_onset_frame_id=None,
            )
            try:
                self._ensure_capacity_for_registration(self._q2_slot_capacity(additional_slots=1))
                with _listener_stream_context(runtime.listener_stream):
                    if prefix.ready_event is not None:
                        runtime.listener_stream.wait_event(prefix.ready_event)
                    native.import_session(row, prefix.past_key_values, prefix.prefix_attention,
                                          prefix_len=prefix.prefix_len,
                                          producer_stream=runtime.listener_stream)
                    # The C ABI identifies contexts, not Python session names. An
                    # imported private slot uses the same metadata CAS as a fork.
                    native._session_by_id[row.session_id].ephemeral_fork = True
                    event = _record_shadow_event(runtime.listener_stream)
                self._maintenance_jobs[session_id] = _MemoryMaintenanceJob(
                    handle, owner, sources, row, covered_through_frame_id,
                    state.semantic_context_revision, memory_revision=memory_revision,
                    last_ready_event=event,
                )
                self._kv_slot_roles[row.session_id] = KvOwnerRole.WORK
                self._set_maintenance_state(session_id, MemoryMaintenanceState.SUMMARIZING
                                            if kind is KvWorkKind.SUMMARY else MemoryMaintenanceState.BUILDING)
                return handle
            except BaseException as exc:
                if bool(getattr(native, "_poisoned", False)):
                    self._fatal_error = exc
                if row.session_id in native._session_by_id:
                    event = None
                    try:
                        with _listener_stream_context(runtime.listener_stream):
                            event = _record_shadow_event(runtime.listener_stream)
                    except BaseException as cleanup_error:
                        self._cleanup_error = cleanup_error
                        self._fatal_error = cleanup_error
                    self._retire_memory_row(session_id, row, event)
                self._release_work_owner(owner)
                self._set_maintenance_state(session_id, MemoryMaintenanceState.FAILED)
                raise

    def _begin_memory_job(self, session_id: str, source_snapshot: Any) -> MemoryMaintenanceHandle:
        return self._open_maintenance(session_id, source_snapshot, KvWorkKind.SUMMARY,
                                      covered_through_frame_id=-1)

    def _memory_prefill(self, job: _MemoryMaintenanceJob, chunk: MemoryPrefillChunk) -> Any:
        if not isinstance(chunk, MemoryPrefillChunk):
            raise TypeError("memory prefill requires a MemoryPrefillChunk")
        if bool(chunk.token_ids) == (chunk.audio_soft is not None):
            raise ValueError("memory quantum must contain either token IDs or archived soft-tokens")
        runtime = self._runtime
        adapter = runtime.token_adapter
        if adapter is None:
            raise RuntimeError("private memory decoding requires the frozen Gemma token adapter")
        audio = chunk.audio_soft is not None
        if audio:
            soft = chunk.audio_soft
            if isinstance(soft, torch.Tensor) and soft.device.type != "cpu":
                raise ValueError("historical soft-tokens must come from the CPU archive")
            array = np.asarray(soft)
            if array.shape != (LISTENER_Q_PER_FRAME, CONTROL_HIDDEN_WIDTH) or not np.isfinite(array).all():
                raise ValueError("memory audio quantum must be finite CPU soft-tokens [2,1536]")
            count = LISTENER_Q_PER_FRAME
        else:
            if not 1 <= len(chunk.token_ids) <= MAX_STREAM_Q_LEN or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in chunk.token_ids
            ):
                raise ValueError("memory text quantum must contain one to eight token IDs")
            count = len(chunk.token_ids)

        reserve = (self._kv_budgets.generated_memory_max_tokens - len(job.generated_token_ids)
                   if job.handle.work_kind is KvWorkKind.SUMMARY else 0)
        self._check_branch_budget(job.handle.session_id,
                                 BranchTokenBudget(job.row.cache_position + count, 0, 0, reserve))
        if job.handle.work_kind is KvWorkKind.REBUILD and (
            job.row.cache_position + count > self._kv_budgets.new_cache_target_tokens
        ):
            raise ValueError("rebuilt cache exceeds its token target")
        with _listener_stream_context(runtime.listener_stream), torch.inference_mode():
            if audio:
                soft = torch.tensor(array, device=runtime.listener_stream.device,
                                    dtype=torch.bfloat16).unsqueeze(0)
                prepared = None
            else:
                prepared = adapter.prepare(chunk.token_ids, logical_q_lens=count)
                if tuple(prepared.logical_q_lens) != (count,):
                    raise RuntimeError("memory token adapter logical-Q metadata differs")
                soft = prepared.inputs_embeds
            row = replace(job.row, logical_q_len=count)
            manifest = self._single_row_private_manifest(row)
            if audio:
                output = runtime.native.step_queued_audio_async(manifest, soft,
                                                              producer_stream=runtime.listener_stream)
            else:
                output = runtime.native.step_async(manifest, soft, per_layer_inputs=prepared.per_layer_inputs,
                                                   producer_stream=runtime.listener_stream)
            # Preserve the current descriptor even if health validation fails
            # after a successfully enqueued native append.
            job.row = replace(row, cache_position=row.cache_position + count,
                              valid_history_tokens=row.valid_history_tokens + count)
            job.last_ready_event = _record_shadow_event(runtime.listener_stream)
            health = tuple(runtime.native.verify_output_health(output))
            if (len(health) != 1 or not health[0].checked or health[0].nonfinite
                    or health[0].row != row or health[0].generation != output.generation
                    or output.rows != (row,) or output.manifest_id != manifest.manifest_id):
                raise RuntimeError("private memory output health/provenance check failed")
            output_index = 0 if count > LISTENER_Q_PER_FRAME else count - 1
            hidden = output.tensor[0, output_index, :].reshape(1, 1, -1)
            job.logits = adapter.logits_from_final_norm_hidden(hidden)
            return {"cache_position": job.row.cache_position, "ready_event": job.last_ready_event}

    def _append_memory_input(self, handle: MemoryMaintenanceHandle, prepared_chunk: Any) -> Any:
        job = self._maintenance_job(handle)
        try:
            if handle.work_kind is not KvWorkKind.SUMMARY or job.summary_done or job.generated_token_ids:
                raise RuntimeError("SUMMARY input is closed")
            if isinstance(prepared_chunk, MemoryPrefillChunk) and prepared_chunk.audio_soft is not None:
                if (prepared_chunk.first_frame_id is None
                        or prepared_chunk.first_frame_id < 0
                        or prepared_chunk.last_frame_id != prepared_chunk.first_frame_id
                        or prepared_chunk.last_frame_id > job.sources.through_frame_id):
                    raise ValueError("SUMMARY audio is outside its frozen source cutoff")
                if job.sources.source_ranges and not any(
                    first <= prepared_chunk.first_frame_id <= last
                    and prepared_chunk.revision == revision
                    for (first, last), revision in zip(job.sources.source_ranges, job.sources.source_revisions)
                ):
                    raise ValueError("SUMMARY audio does not match its frozen source revision")
            return self._memory_prefill(job, prepared_chunk)
        except BaseException:
            self._fail_maintenance(job)
            raise

    def _advance_memory_decode(self, handle: MemoryMaintenanceHandle) -> Any:
        job = self._maintenance_job(handle)
        try:
            if handle.work_kind is not KvWorkKind.SUMMARY or job.logits is None:
                raise RuntimeError("SUMMARY decode requires private prefill logits")
            if job.summary_done:
                return {"token_ids": tuple(job.generated_token_ids), "done": True}
            if not bool(torch.isfinite(job.logits).all().item()):
                raise RuntimeError("SUMMARY produced nonfinite semantic logits")
            if len(job.generated_token_ids) >= self._kv_budgets.generated_memory_max_tokens:
                raise RuntimeError("SUMMARY exhausted its memory token budget before EOS")
            token = int(torch.argmax(job.logits.reshape(-1)).item())
            if token in job.sources.eos_token_ids:
                job.summary_done = True
            else:
                job.generated_token_ids.append(token)
                self._memory_prefill(job, MemoryPrefillChunk(token_ids=(token,)))
            return {"token_ids": tuple(job.generated_token_ids), "done": job.summary_done}
        except BaseException:
            self._fail_maintenance(job)
            raise

    def _begin_canonical_rebuild(
        self, session_id: str, memory_revision: Any, retained_sources: Any,
    ) -> MemoryMaintenanceHandle:
        from duplex_kernel.session import MemoryRevision, serialize_memory_revision

        previous = self._maintenance_jobs.get(session_id)
        if previous is not None:
            self._maintenance_job(previous.handle)
            if previous.handle.work_kind is not KvWorkKind.SUMMARY:
                raise RuntimeError("WORK role already occupied")
            if not previous.summary_done:
                raise RuntimeError("SUMMARY must complete before REBUILD")
            if previous.last_ready_event is not None and not previous.last_ready_event.query():
                raise RuntimeError("SUMMARY completion fence is pending")
        elif session_id in self._work_owners:
            raise RuntimeError("WORK role already occupied")
        try:
            if not isinstance(memory_revision, MemoryRevision) or not memory_revision.semantic_notes.strip():
                raise ValueError("REBUILD requires a nonempty MemoryRevision")
            if not isinstance(retained_sources, MemorySourceSnapshot):
                raise TypeError("REBUILD requires frozen retained sources")
            memory_revision = replace(
                memory_revision,
                covered_source_ranges=tuple(tuple(item) for item in memory_revision.covered_source_ranges),
                source_revisions=tuple(memory_revision.source_revisions),
                open_turn_state=copy.deepcopy(memory_revision.open_turn_state),
            )
            MemorySourceSnapshot(
                prefix=retained_sources.prefix, through_frame_id=retained_sources.through_frame_id,
                source_ranges=memory_revision.covered_source_ranges,
                source_revisions=memory_revision.source_revisions,
            )
            covered = max((last for first, last in memory_revision.covered_source_ranges), default=-1)
            if previous is not None:
                if previous.parent_memory_revision != self._session_states[session_id].semantic_context_revision:
                    raise RuntimeError("parent memory revision changed during SUMMARY")
                if (memory_revision.covered_source_ranges != previous.sources.source_ranges
                        or memory_revision.source_revisions != previous.sources.source_revisions
                        or memory_revision.parent_memory_id != previous.sources.parent_memory_id):
                    raise RuntimeError("memory revision differs from frozen SUMMARY sources")
                covered = previous.sources.through_frame_id
            if covered > retained_sources.through_frame_id:
                raise ValueError("REBUILD cutoff precedes its memory coverage")
            encoder = self._runtime.memory_token_encoder
            if not callable(encoder):
                raise RuntimeError("REBUILD requires the model's memory_token_encoder")
            tokens = tuple(encoder(serialize_memory_revision(memory_revision)))
            if not tokens or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
                raise ValueError("memory_token_encoder must return nonempty token IDs")
            position = retained_sources.prefix.cache_position or retained_sources.prefix.prefix_len
            if position + len(tokens) > self._kv_budgets.new_cache_target_tokens:
                raise ValueError("memory prefill exceeds the new cache target")
            with self._kv_work_lock:
                if previous is not None:
                    with _listener_stream_context(self._runtime.listener_stream):
                        self._runtime.native.release_session_async(previous.row)
                    self._kv_slot_roles.pop(previous.row.session_id, None)
                    self._maintenance_jobs.pop(session_id)
                    self._release_work_owner(previous.owner)
                handle = self._open_maintenance(session_id, retained_sources, KvWorkKind.REBUILD,
                                                covered_through_frame_id=covered, memory_revision=memory_revision)
                self._maintenance_jobs[session_id].pending_memory_tokens.extend(tokens)
                return handle
        except BaseException:
            if previous is not None:
                if self._maintenance_jobs.get(session_id) is previous:
                    self._fail_maintenance(previous)
                else:
                    self._set_maintenance_state(session_id, MemoryMaintenanceState.FAILED)
            raise

    def _append_rebuild_chunk(self, handle: MemoryMaintenanceHandle, prepared_chunk: Any) -> Any:
        job = self._maintenance_job(handle)
        try:
            if handle.work_kind is not KvWorkKind.REBUILD:
                raise RuntimeError("rebuild append requires a REBUILD handle")
            phase = self._maintenance_states[handle.session_id]
            if phase not in {MemoryMaintenanceState.BUILDING, MemoryMaintenanceState.CATCH_UP}:
                raise RuntimeError("REBUILD input is closed")
            if job.pending_memory_tokens:
                if prepared_chunk is not None:
                    raise RuntimeError("flush private memory prefill with None before retained sources")
                tokens = tuple(list(job.pending_memory_tokens)[:MAX_STREAM_Q_LEN])
                self._memory_prefill(job, MemoryPrefillChunk(token_ids=tokens))
                for _ in tokens:
                    job.pending_memory_tokens.popleft()
            elif prepared_chunk is None:
                target = (job.sources.through_frame_id if phase is MemoryMaintenanceState.BUILDING
                          else self._session_states[handle.session_id].next_frame_id - 1)
                if job.covered_through_frame_id < target:
                    raise RuntimeError("REBUILD has incomplete source coverage")
                self._set_maintenance_state(handle.session_id, MemoryMaintenanceState.CATCH_UP
                                            if phase is MemoryMaintenanceState.BUILDING else MemoryMaintenanceState.READY)
            else:
                chunk = prepared_chunk
                if not isinstance(chunk, MemoryPrefillChunk):
                    raise TypeError("rebuild prefill requires a MemoryPrefillChunk")
                if chunk.audio_soft is not None:
                    if (chunk.first_frame_id != job.covered_through_frame_id + 1
                            or chunk.last_frame_id != chunk.first_frame_id
                            or chunk.last_frame_id >= self._session_states[handle.session_id].next_frame_id):
                        raise ValueError("REBUILD audio chunks must cover consecutive single Q2 frames")
                    if job.sources.source_ranges and chunk.first_frame_id <= job.sources.through_frame_id:
                        if not any(first <= chunk.first_frame_id <= last and chunk.revision == revision
                                   for (first, last), revision in zip(job.sources.source_ranges, job.sources.source_revisions)):
                            raise ValueError("REBUILD audio does not match its frozen source revision")
                self._memory_prefill(job, chunk)
                if chunk.audio_soft is not None:
                    job.covered_through_frame_id = chunk.last_frame_id
            return {"state": self._maintenance_states[handle.session_id],
                    "covered_through_frame_id": job.covered_through_frame_id,
                    "memory_tokens_remaining": len(job.pending_memory_tokens),
                    "cache_position": job.row.cache_position}
        except BaseException:
            self._fail_maintenance(job)
            raise

    def _promote_canonical_rebuild(
        self, handle: MemoryMaintenanceHandle, expected_source_revisions: Any = None,
    ) -> Any:
        job = self._maintenance_job(handle)
        state = self._session_states[handle.session_id]
        if handle.work_kind is not KvWorkKind.REBUILD or self._maintenance_states[handle.session_id] is not MemoryMaintenanceState.READY:
            raise RuntimeError("canonical promotion requires a READY rebuild")
        if (job.covered_through_frame_id < state.next_frame_id - 1
                or self._outstanding_frames.get(handle.session_id, 0)
                or (self._tick_expected is not None and handle.session_id in self._tick_expected)):
            self._set_maintenance_state(handle.session_id, MemoryMaintenanceState.CATCH_UP)
            raise RuntimeError("canonical promotion needs catch-up at a completed tick")
        if job.last_ready_event is not None and not job.last_ready_event.query():
            raise RuntimeError("REBUILD completion fence is pending")
        swapped = False
        try:
            revisions = tuple(job.memory_revision.source_revisions) + tuple(job.sources.source_revisions)
            if expected_source_revisions is not None and tuple(expected_source_revisions) != revisions:
                raise RuntimeError("source revisions changed before canonical promotion")
            if state.semantic_context_revision != job.parent_memory_revision:
                raise RuntimeError("parent memory revision changed before canonical promotion")
            native = self._runtime.native
            canonical = self._last_native_rows[handle.session_id]
            with _listener_stream_context(self._runtime.listener_stream):
                rebuilt = native.rebind_ephemeral_session_metadata(job.row, canonical)
                expected = replace(job.row, state_epoch_id=canonical.state_epoch_id,
                                   frame_id=canonical.frame_id,
                                   listener_state_version=canonical.listener_state_version,
                                   sequence_position=canonical.sequence_position)
                if rebuilt != expected:
                    raise RuntimeError("REBUILD metadata CAS changed private KV ownership or cursor")
                job.row = rebuilt
                native.session_state(rebuilt)
                native.session_state(canonical)
                old_binding = native._session_by_id[handle.session_id]
                new_binding = native._session_by_id[rebuilt.session_id]
                if (old_binding.native_context_id != canonical.native_context_id
                        or new_binding.native_context_id != rebuilt.native_context_id):
                    raise RuntimeError("canonical native binding identity changed")
                self._release_q2_decision_cache(handle.session_id)
                self._set_maintenance_state(handle.session_id, MemoryMaintenanceState.SWAP)
                self._coordinator.rebase_cache_position(handle.session_id, rebuilt.cache_position)
                # Session names are Python binding keys only. Context IDs and
                # slots remain unchanged in C, so no KV copy or fourth slot is needed.
                retiring_row = replace(canonical, session_id=rebuilt.session_id)
                promoted_row = replace(rebuilt, session_id=handle.session_id)
                old_binding.session_id = retiring_row.session_id
                new_binding.session_id = promoted_row.session_id
                new_binding.ephemeral_fork = False
                native._session_by_id[retiring_row.session_id] = old_binding
                native._session_by_id[promoted_row.session_id] = new_binding
                swapped = True
                cache_epoch = state.cache_epoch_id + 1
                memory_revision = (state.semantic_context_revision or 0) + 1
                history_revision = job.sources.prefix.applied_history_revision
                if history_revision is None:
                    history_revision = state.applied_history_revision

                def commit_identity() -> None:
                    state.native_context_id = promoted_row.native_context_id
                    state.cache_epoch_id = cache_epoch
                    state.semantic_context_revision = memory_revision
                    state.applied_history_revision = history_revision

                if self._on_cache_identity_commit is None:
                    commit_identity()
                else:
                    self._on_cache_identity_commit(handle.session_id, cache_epoch, memory_revision,
                                                   history_revision, commit_identity)
                self._last_native_rows[handle.session_id] = promoted_row
                self._prefix_lengths[handle.session_id] = new_binding.prefix_len
                self._prefix_revisions[handle.session_id] = state.semantic_context_revision
                self._prefix_replay_frame_ids[handle.session_id] = ()
                self._retire_memory_row(handle.session_id, retiring_row,
                                        _record_shadow_event(self._runtime.listener_stream))
                self._kv_slot_roles[handle.session_id] = KvOwnerRole.CANONICAL
            self._maintenance_jobs.pop(handle.session_id)
            self._release_work_owner(job.owner)
            self._set_maintenance_state(handle.session_id, MemoryMaintenanceState.IDLE)
            return snapshot_listener_state(state)
        except BaseException:
            if swapped:
                self._maintenance_jobs.pop(handle.session_id, None)
                self._release_work_owner(job.owner)
                self._set_maintenance_state(handle.session_id, MemoryMaintenanceState.FAILED)
                self._fatal_error = RuntimeError("canonical memory swap failed after ownership transfer")
            else:
                self._fail_maintenance(job)
            raise

    def _cancel_maintenance(self, handle: MemoryMaintenanceHandle) -> None:
        if not isinstance(handle, MemoryMaintenanceHandle):
            raise TypeError("maintenance cancellation requires a MemoryMaintenanceHandle")
        job = self._maintenance_jobs.get(handle.session_id)
        if job is None or job.handle != handle:
            return
        self._set_maintenance_state(handle.session_id, MemoryMaintenanceState.CANCELLED)
        self._retire_memory_row(handle.session_id, job.row, job.last_ready_event)
        self._maintenance_jobs.pop(handle.session_id)
        self._release_work_owner(job.owner)
        self._set_maintenance_state(handle.session_id, MemoryMaintenanceState.IDLE)

    def _clear_session_maintenance(self, session_id: str) -> None:
        job = self._maintenance_jobs.get(session_id)
        if job is not None:
            self._cancel_maintenance(job.handle)
        self._maintenance_states.pop(session_id, None)
        self._reap_retiring_owners(session_id=session_id)

    def _reap_shadow_work_owner(self, session_id: str) -> None:
        owner = self._work_owners.get(session_id)
        shadow = self._shadow_worker
        if owner is not None and owner.shadow_job and session_id not in self._pending_promotions:
            if shadow is None or not shadow.owns_session_work(session_id):
                self._work_owners.pop(session_id, None)

    def _reserve_work_owner(
        self, session_id: str, kind: KvWorkKind, budget: BranchTokenBudget,
        *, kv_owner_epoch: str | None = None, shadow_job: bool = False,
    ) -> KvWorkOwner:
        kind = KvWorkKind(kind)
        state = self._session_states.get(session_id)
        if (state is None or state.closed or session_id in self._closing_sessions
                or self._fatal_error is not None):
            raise RuntimeError("WORK requires a live session and a healthy owner")
        self._check_branch_budget(session_id, budget)
        if kind is KvWorkKind.SUMMARY and (
            budget.remaining_generation_tokens > self._kv_budgets.generated_memory_max_tokens
        ):
            raise ValueError("generated memory exceeds its token budget")
        if kind is KvWorkKind.REBUILD and budget.total_tokens > self._kv_budgets.new_cache_target_tokens:
            raise ValueError("rebuilt cache exceeds its token target")
        with self._kv_work_lock:
            self._reap_retiring_owners(session_id=session_id)
            self._reap_shadow_work_owner(session_id)
            if any(item.session_id == session_id for item in self._retiring_owners.values()):
                raise RuntimeError("WORK role already occupied")
            if session_id in self._work_owners or (
                self._shadow_worker is not None and self._shadow_worker.owns_session_work(session_id)
            ):
                raise RuntimeError("session already has a WORK owner")
            owner = KvWorkOwner(
                session_id=session_id, session_generation_id=state.generation_id,
                session_epoch_id=state.state_epoch_id,
                kv_owner_epoch=kv_owner_epoch or uuid.uuid4().hex, kind=kind,
                memory_revision_id=state.semantic_context_revision,
                source_history_revision=state.applied_history_revision, shadow_job=shadow_job,
            )
            self._work_owners[session_id] = owner
            return owner

    def _release_work_owner(self, owner: KvWorkOwner) -> None:
        with self._kv_work_lock:
            if self._work_owners.get(owner.session_id) == owner:
                self._work_owners.pop(owner.session_id, None)

    @staticmethod
    def _response_budget_key(handle: EphemeralDecodeHandle) -> tuple[str, int, str, int]:
        return (handle.session_id, handle.session_generation_id,
                str(handle.response_id), int(handle.response_generation_id or 0))

    def _private_context_id(self, source_context_id: int) -> int:
        context_id = int(uuid.uuid4().int & ((1 << 63) - 1)) or 1
        while context_id == int(source_context_id) or any(
            context_id == branch.handle.fork_native_context_id
            for branch in self._ephemeral_decode_branches.values()
        ) or any(
            context_id == item.row.native_context_id
            for item in (*self._maintenance_jobs.values(), *self._retiring_owners.values())
        ) or any(
            context_id == int(snapshot.native_context_id)
            for snapshot in self._session_states.values()
            if snapshot.native_context_id is not None
        ):
            context_id = int(uuid.uuid4().int & ((1 << 63) - 1)) or 1
        return context_id

    def _begin_ephemeral_decode(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
    ) -> EphemeralDecodeHandle:
        runtime = self._runtime
        coordinator = self._coordinator
        if runtime is None or coordinator is None:
            raise RuntimeError("private decode requires the active native owner")
        if not callable(getattr(runtime.native, "fork_session_slot", None)):
            raise RuntimeError("native owner lacks same-owner session-slot fork")
        adapter = runtime.token_adapter
        if adapter is None or not callable(getattr(adapter, "prepare", None)) or not callable(
            getattr(adapter, "logits_from_final_norm_hidden", None)
        ):
            raise RuntimeError("private decode requires the exact Gemma token/PLE adapter")
        snapshot = self._check_unified_session_cas(
            session_id,
            expected_generation_id=expected_generation_id,
            expected_cache_position=expected_cache_position,
        )
        if self._outstanding_frames.get(session_id, 0) != 0 or (
            self._tick_expected is not None and session_id in self._tick_expected
        ):
            raise RuntimeError("private decode fork requires a completed listener frame watermark")
        if any(
            branch.handle.parent_session_id == session_id
            and branch.handle.purpose == "backchannel"
            for branch in self._ephemeral_decode_branches.values()
        ):
            raise RuntimeError("session already has a live private decode fork")
        source_row = self._last_native_rows.get(session_id)
        if source_row is None or snapshot.native_context_id is None:
            raise RuntimeError("private decode source has no imported native cache row")
        if (
            source_row.session_generation != snapshot.generation_id
            or source_row.state_epoch_id != snapshot.state_epoch_id
            or source_row.cache_position != snapshot.cache_position
            or source_row.valid_history_tokens != snapshot.valid_history_tokens
            or source_row.native_context_id != int(snapshot.native_context_id)
        ):
            raise RuntimeError("private decode source row no longer matches its CAS snapshot")

        self._check_branch_budget(
            session_id, BranchTokenBudget(snapshot.cache_position, MAX_STREAM_Q_LEN, 0,
                                           BACKCHANNEL_MAX_TEXT_TOKENS),
        )
        handle_id = uuid.uuid4().hex
        fork_session_id = f"__decode__{session_id}__{handle_id}"
        fork_context_id = self._private_context_id(int(snapshot.native_context_id))
        handle = EphemeralDecodeHandle(
            handle_id=handle_id,
            parent_session_id=session_id,
            parent_generation_id=snapshot.generation_id,
            parent_state_epoch_id=snapshot.state_epoch_id,
            source_cache_position=snapshot.cache_position,
            fork_session_id=fork_session_id,
            fork_native_context_id=fork_context_id,
            max_tokens=BACKCHANNEL_MAX_TEXT_TOKENS,
            purpose="backchannel",
            source_frame_id=snapshot.next_frame_id,
            source_state_version=snapshot.state_version,
            source_sequence_position=snapshot.sequence_position,
            source_cache_epoch_id=int(snapshot.cache_epoch_id),
            source_history_revision=snapshot.applied_history_revision,
            kv_owner_epoch=handle_id,
            memory_revision_id=snapshot.semantic_context_revision,
        )
        expected_fork_row = replace(
            source_row,
            session_id=fork_session_id,
            native_context_id=fork_context_id,
        )
        self._ensure_capacity_for_registration(
            self._q2_slot_capacity(additional_slots=1)
        )
        forked = False
        try:
            with _listener_stream_context(runtime.listener_stream):
                fork_row = runtime.native.fork_session_slot(
                    source_row,
                    fork_session_id=fork_session_id,
                    fork_native_context_id=fork_context_id,
                )
            forked = True
            if fork_row != expected_fork_row:
                raise RuntimeError("native private decode fork changed its source boundary or cursor")
            self._ephemeral_decode_branches[handle_id] = _EphemeralDecodeBranch(
                handle=handle,
                row=fork_row,
            )
            self._kv_slot_roles[fork_session_id] = handle.kv_owner_role
            return handle
        except BaseException:
            # A failed native clone can have reserved a destination slot before
            # reporting its failure. Retire it best-effort without touching the
            # canonical source row.
            if forked:
                try:
                    runtime.native.release_session_async(fork_row)
                except BaseException as cleanup_error:
                    if self._cleanup_error is None:
                        self._cleanup_error = cleanup_error
            raise

    def _begin_response_decode(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        response_id: str,
        response_generation_id: int,
        revision_id: str,
        purpose: ResponseDecodePurpose,
        source_watermark: ListenerWatermark | None,
        max_tokens: int,
        prompt_token_budget: int,
        revision_key: tuple[str, str, int, str],
        transition_tokens: int | None = None,
        frozen_assistant_prefix_tokens: int = 0,
    ) -> ResponseDecodeHandle:
        with self._response_revision_lock:
            latest_revision = self._latest_response_revisions.get(revision_key)
        if latest_revision != revision_id:
            raise StaleResponseRevisionError(
                f"response revision {revision_id} was superseded by {latest_revision}"
            )
        runtime = self._runtime
        coordinator = self._coordinator
        if runtime is None or coordinator is None:
            raise RuntimeError("response decode requires the active native owner")
        if not callable(getattr(runtime.native, "fork_session_slot", None)):
            raise RuntimeError("native owner lacks same-owner session-slot fork")
        adapter = runtime.token_adapter
        if adapter is None or not callable(getattr(adapter, "prepare", None)) or not callable(
            getattr(adapter, "logits_from_final_norm_hidden", None)
        ):
            raise RuntimeError("response decode requires the exact Gemma token/PLE adapter")
        snapshot = coordinator.session_state(session_id)
        if snapshot.generation_id != expected_generation_id:
            raise StaleListenerWatermarkError(
                "response fork listener generation is stale: "
                f"expected={expected_generation_id}, actual={snapshot.generation_id}"
            )
        if source_watermark is not None:
            supplied = (
                source_watermark.generation_id,
                source_watermark.state_epoch_id,
                source_watermark.frame_id,
                source_watermark.state_version,
                source_watermark.sequence_position,
                source_watermark.cache_position,
                source_watermark.cache_epoch_id,
                source_watermark.applied_history_revision,
            )
            actual = (
                snapshot.generation_id,
                snapshot.state_epoch_id,
                snapshot.next_frame_id,
                snapshot.state_version,
                snapshot.sequence_position,
                snapshot.cache_position,
                snapshot.cache_epoch_id,
                snapshot.applied_history_revision,
            )
            if supplied != actual:
                raise StaleListenerWatermarkError(
                    f"response fork listener watermark is stale: expected={supplied}, actual={actual}"
                )
        if session_id in self._pending_promotions:
            raise StaleListenerWatermarkError(
                "response fork is blocked until the listener cache promotion is fenced"
            )
        if self._outstanding_frames.get(session_id, 0) != 0 or (
            self._tick_expected is not None and session_id in self._tick_expected
        ):
            raise StaleListenerWatermarkError(
                "response fork requires a completed listener frame watermark"
            )
        source_row = self._last_native_rows.get(session_id)
        if source_row is None or snapshot.native_context_id is None:
            raise RuntimeError("response fork source has no imported native cache row")
        if (
            source_row.session_generation != snapshot.generation_id
            or source_row.state_epoch_id != snapshot.state_epoch_id
            or source_row.frame_id != snapshot.next_frame_id
            or source_row.listener_state_version != snapshot.state_version
            or source_row.sequence_position != snapshot.sequence_position
            or source_row.cache_position != snapshot.cache_position
            or source_row.valid_history_tokens != snapshot.valid_history_tokens
            or source_row.native_context_id != int(snapshot.native_context_id)
        ):
            raise StaleListenerWatermarkError(
                "response fork source native row no longer matches its completed listener watermark"
            )
        budget_key = (session_id, snapshot.generation_id, response_id, response_generation_id)
        response_budget = self._response_token_budgets.get(budget_key)
        if response_budget is None:
            response_budget = _ResponseTokenBudget(min(max_tokens, self._kv_budgets.assistant_response_max_tokens))
        max_tokens = min(max_tokens, response_budget.remaining, self._kv_budgets.assistant_response_max_tokens)
        if max_tokens < 1:
            raise RuntimeError("response token budget is exhausted across revisions")
        if transition_tokens is None:
            # Legacy callers reserve a combined prompt allowance. Exact callers
            # supply both components, including the frozen assistant prefix.
            transition_tokens = prompt_token_budget
        prompt_token_budget = transition_tokens + frozen_assistant_prefix_tokens
        admission = BranchTokenBudget(
            snapshot.cache_position, transition_tokens, frozen_assistant_prefix_tokens, max_tokens,
        )
        self._check_branch_budget(session_id, admission)

        with self._kv_work_lock:
            self._reap_shadow_work_owner(session_id)
            if purpose is ResponseDecodePurpose.CANDIDATE and session_id in self._work_owners:
                raise RuntimeError("session already has a WORK owner")

        for old in tuple(self._ephemeral_decode_branches.values()):
            old_handle = old.handle
            if old_handle.parent_session_id != session_id or old_handle.purpose == "backchannel":
                continue
            same_response = (
                old_handle.response_id == response_id
                and old_handle.response_generation_id == response_generation_id
            )
            if old_handle.purpose == purpose.value or not same_response:
                self._end_ephemeral_decode(old_handle)

        required = (
            len(self._session_states)
            + len(self._ephemeral_decode_branches)
            + len(self._q2_decision_caches)
            + 1
        )
        self._ensure_capacity_for_registration(required)
        handle_id = uuid.uuid4().hex
        fork_session_id = f"__response__{session_id}__{handle_id}"
        fork_context_id = self._private_context_id(int(snapshot.native_context_id))
        handle = EphemeralDecodeHandle(
            handle_id=handle_id,
            parent_session_id=session_id,
            parent_generation_id=snapshot.generation_id,
            parent_state_epoch_id=snapshot.state_epoch_id,
            source_cache_position=snapshot.cache_position,
            fork_session_id=fork_session_id,
            fork_native_context_id=fork_context_id,
            max_tokens=max_tokens,
            max_prompt_tokens=prompt_token_budget,
            purpose=purpose.value,
            source_frame_id=snapshot.next_frame_id,
            source_state_version=snapshot.state_version,
            source_sequence_position=snapshot.sequence_position,
            response_id=response_id,
            response_generation_id=response_generation_id,
            revision_id=revision_id,
            source_cache_epoch_id=int(snapshot.cache_epoch_id),
            source_history_revision=snapshot.applied_history_revision,
            kv_owner_epoch=handle_id,
            memory_revision_id=snapshot.semantic_context_revision,
        )
        expected_fork_row = replace(
            source_row,
            session_id=fork_session_id,
            native_context_id=fork_context_id,
        )
        work_owner = None
        if purpose is ResponseDecodePurpose.CANDIDATE:
            work_owner = self._reserve_work_owner(
                session_id, KvWorkKind.REPLAN, admission, kv_owner_epoch=handle_id,
            )
        try:
            with _listener_stream_context(runtime.listener_stream):
                fork_row = runtime.native.fork_session_slot(
                    source_row, fork_session_id=fork_session_id,
                    fork_native_context_id=fork_context_id,
                )
            if fork_row != expected_fork_row:
                raise RuntimeError("native response fork changed its listener watermark or KV cursor")
            self._ephemeral_decode_branches[handle_id] = _EphemeralDecodeBranch(
                handle=handle, row=fork_row,
            )
            self._kv_slot_roles[fork_session_id] = handle.kv_owner_role
            self._response_token_budgets[budget_key] = response_budget
        except BaseException:
            # The destination may already exist when the native clone fails.
            try:
                runtime.native.release_session_async(expected_fork_row)
            except BaseException as cleanup_error:
                self._cleanup_error = cleanup_error
                self._fatal_error = cleanup_error
            else:
                if work_owner is not None:
                    self._release_work_owner(work_owner)
            raise
        return handle

    def _promote_response_decode(
        self,
        candidate_handle: ResponseDecodeHandle,
        *,
        expected_active_handle: ResponseDecodeHandle | None,
        expected_active_revision_id: str | None,
    ) -> ResponseDecodeHandle:
        """Transfer a live candidate branch to ACTIVE without changing its KV."""
        if candidate_handle.purpose != ResponseDecodePurpose.CANDIDATE.value:
            raise StaleResponseRevisionError("response promotion candidate is no longer a candidate")
        candidate_revision_key = (
            candidate_handle.parent_session_id,
            str(candidate_handle.response_id),
            int(candidate_handle.response_generation_id or 0),
            ResponseDecodePurpose.CANDIDATE.value,
        )
        active_revision_key = (
            candidate_handle.parent_session_id,
            str(candidate_handle.response_id),
            int(candidate_handle.response_generation_id or 0),
            ResponseDecodePurpose.ACTIVE.value,
        )
        old_active: _EphemeralDecodeBranch | None = None
        with self._response_revision_lock:
            latest_candidate_revision = self._latest_response_revisions.get(
                candidate_revision_key
            )
            if latest_candidate_revision != candidate_handle.revision_id:
                raise StaleResponseRevisionError(
                    "candidate response revision was superseded before promotion: "
                    f"expected={candidate_handle.revision_id}, actual={latest_candidate_revision}"
                )

            candidate_branch = self._ephemeral_decode_branches.get(
                candidate_handle.handle_id
            )
            if candidate_branch is None or candidate_branch.handle != candidate_handle:
                raise StaleResponseRevisionError(
                    "candidate response branch is no longer live"
                )

            active_branches = tuple(
                branch
                for branch in self._ephemeral_decode_branches.values()
                if branch.handle.parent_session_id == candidate_handle.parent_session_id
                and branch.handle.purpose == ResponseDecodePurpose.ACTIVE.value
            )
            latest_active_revision = self._latest_response_revisions.get(
                active_revision_key
            )
            if expected_active_handle is None:
                if active_branches or latest_active_revision is not None:
                    raise StaleResponseRevisionError(
                        "active response slot changed before candidate promotion"
                    )
            else:
                if (
                    len(active_branches) != 1
                    or active_branches[0].handle != expected_active_handle
                    or expected_active_handle.revision_id != expected_active_revision_id
                    or latest_active_revision != expected_active_revision_id
                    or expected_active_handle.response_id != candidate_handle.response_id
                    or expected_active_handle.response_generation_id
                    != candidate_handle.response_generation_id
                    or expected_active_handle.parent_generation_id
                    != candidate_handle.parent_generation_id
                    or expected_active_handle.parent_state_epoch_id
                    != candidate_handle.parent_state_epoch_id
                ):
                    raise StaleResponseRevisionError(
                        "active response identity or revision changed before candidate promotion"
                    )
                old_active = active_branches[0]

            parent = self._session_states.get(candidate_handle.parent_session_id)
            candidate_row = candidate_branch.row
            if (
                parent is None
                or parent.closed
                or parent.generation_id != candidate_handle.parent_generation_id
                or parent.state_epoch_id != candidate_handle.parent_state_epoch_id
                or candidate_handle.response_id is None
                or candidate_handle.response_generation_id is None
                or candidate_handle.revision_id is None
                or candidate_row.session_generation
                != candidate_handle.parent_generation_id
                or candidate_row.state_epoch_id != candidate_handle.parent_state_epoch_id
                or candidate_row.frame_id != candidate_handle.source_frame_id
                or candidate_row.listener_state_version
                != candidate_handle.source_state_version
                or candidate_row.sequence_position
                != candidate_handle.source_sequence_position
            ):
                raise StaleResponseRevisionError(
                    "candidate response no longer belongs to the current session generation"
                )

            # This is the in-memory commit point. No native call or GPU copy is
            # made while the revision lock protects the compare-and-swap.
            promoted_handle = replace(
                candidate_handle,
                purpose=ResponseDecodePurpose.ACTIVE.value,
            )
            candidate_branch.handle = promoted_handle
            self._kv_slot_roles[promoted_handle.fork_session_id] = KvOwnerRole.ACTIVE
            work_owner = self._work_owners.get(candidate_handle.session_id)
            if work_owner is not None and work_owner.kv_owner_epoch == candidate_handle.kv_owner_epoch:
                self._release_work_owner(work_owner)
            if old_active is not None:
                # Detach at the CAS point: stale appends can no longer find
                # the old ACTIVE branch, and a later promotion sees exactly
                # one ACTIVE handle. The queued branch owns its row and last
                # CUDA event until the owner loop releases it after replying.
                self._ephemeral_decode_branches.pop(old_active.handle.handle_id, None)
                self._detached_response_decode_branches.append(old_active)
            self._latest_response_revisions.pop(candidate_revision_key, None)
            self._latest_response_revisions[active_revision_key] = str(
                candidate_handle.revision_id
            )
        return promoted_handle

    def _drain_detached_response_decode_cleanup(self) -> None:
        """Release owner-detached response rows after their CAS reply is sent."""
        runtime = self._runtime
        if runtime is None and self._detached_response_decode_branches:
            raise RuntimeError("private decode owner is unavailable during detached cleanup")
        while self._detached_response_decode_branches:
            detached = self._detached_response_decode_branches.popleft()
            try:
                assert runtime is not None
                with _listener_stream_context(runtime.listener_stream):
                    if detached.last_ready_event is not None:
                        wait_event = getattr(runtime.listener_stream, "wait_event", None)
                        if callable(wait_event):
                            wait_event(detached.last_ready_event)
                    runtime.native.release_session_async(detached.row)
                    self._kv_slot_roles.pop(detached.handle.fork_session_id, None)
            except BaseException as exc:
                if self._cleanup_error is None:
                    self._cleanup_error = exc

    def _append_private_tokens(
        self,
        handle: EphemeralDecodeHandle,
        token_ids: tuple[int, ...],
        *,
        expected_cache_position: int,
    ) -> PrivateTextAppendResult:
        return self._append_private_ids(
            handle,
            token_ids,
            expected_cache_position=expected_cache_position,
            is_transition=False,
        )

    def _append_private_transition_tokens(
        self,
        handle: EphemeralDecodeHandle,
        token_ids: tuple[int, ...],
        *,
        expected_cache_position: int,
    ) -> PrivateTextAppendResult:
        return self._append_private_ids(
            handle,
            token_ids,
            expected_cache_position=expected_cache_position,
            is_transition=True,
        )

    def _append_response_decode_ids(
        self,
        handle: ResponseDecodeHandle,
        token_ids: tuple[int, ...],
        *,
        expected_cache_position: int,
        generated: bool,
    ) -> PrivateTextAppendResult:
        if handle.purpose not in {
            ResponseDecodePurpose.ACTIVE.value,
            ResponseDecodePurpose.CANDIDATE.value,
        }:
            raise TypeError("response append requires an active or candidate response fork")
        return self._append_private_ids(
            handle,
            token_ids,
            expected_cache_position=expected_cache_position,
            is_transition=False,
            response_prompt=not generated,
        )

    def _append_private_ids(
        self,
        handle: EphemeralDecodeHandle,
        token_ids: tuple[int, ...],
        *,
        expected_cache_position: int,
        is_transition: bool,
        response_prompt: bool = False,
    ) -> PrivateTextAppendResult:
        runtime = self._runtime
        coordinator = self._coordinator
        if runtime is None or coordinator is None:
            raise RuntimeError("private decode owner is unavailable")
        if handle.purpose in {
            ResponseDecodePurpose.ACTIVE.value,
            ResponseDecodePurpose.CANDIDATE.value,
        }:
            revision_key = (
                handle.parent_session_id,
                str(handle.response_id),
                int(handle.response_generation_id or 0),
                handle.purpose,
            )
            with self._response_revision_lock:
                latest_revision = self._latest_response_revisions.get(revision_key)
            if latest_revision != handle.revision_id:
                raise StaleResponseRevisionError(
                    f"response revision {handle.revision_id} was superseded by {latest_revision}"
                )
        branch = self._ephemeral_decode_branches.get(handle.handle_id)
        if branch is None or branch.handle != handle:
            raise RuntimeError("unknown private decode fork")
        parent = self._session_states.get(handle.parent_session_id)
        if (
            parent is None
            or parent.closed
            or parent.generation_id != handle.parent_generation_id
            or parent.state_epoch_id != handle.parent_state_epoch_id
        ):
            self._end_ephemeral_decode(handle)
            raise RuntimeError("private decode fork lost its parent generation fence")
        if (
            branch.row.session_generation != handle.parent_generation_id
            or branch.row.state_epoch_id != handle.parent_state_epoch_id
            or branch.row.frame_id != handle.source_frame_id
            or branch.row.listener_state_version != handle.source_state_version
            or branch.row.sequence_position != handle.source_sequence_position
        ):
            raise StaleListenerWatermarkError(
                "private decode branch lost its source listener watermark"
            )
        if expected_cache_position != branch.row.cache_position:
            raise RuntimeError("private decode append lost its branch cache-position compare-and-swap")
        response_budget = None
        if handle.response_id is not None:
            response_budget = self._response_token_budgets.get(self._response_budget_key(handle))
            if response_budget is None:
                raise RuntimeError("response token budget no longer belongs to this session")
            if not is_transition and not response_prompt and len(token_ids) > response_budget.remaining:
                raise RuntimeError("response token budget is exhausted across revisions")
        if is_transition and branch.token_count:
            raise RuntimeError("private transition is allowed once before sampled tokens")
        if response_prompt and (
            handle.purpose not in {
                ResponseDecodePurpose.ACTIVE.value,
                ResponseDecodePurpose.CANDIDATE.value,
            }
            or branch.token_count
        ):
            raise RuntimeError("response prompt chunks are only allowed before generated tokens")
        counts_as_prompt = response_prompt or (is_transition and response_budget is not None)
        if counts_as_prompt and (
            branch.prompt_token_count + branch.transition_token_count + len(token_ids)
            > handle.max_prompt_tokens
        ):
            raise RecoverableSessionCapacityError(
                handle.parent_session_id,
                generation_id=handle.parent_generation_id,
                frame_id=handle.source_frame_id,
                cache_position=branch.row.cache_position,
                requested_tokens=len(token_ids),
                max_seq_len=self._operational_max_seq_len(),
            )
        if is_transition and len(token_ids) > (
            self._operational_max_seq_len() - branch.row.cache_position
        ):
            raise RuntimeError("private transition exceeds remaining native KV context")
        if (
            not is_transition
            and not response_prompt
            and branch.token_count + len(token_ids) > handle.max_tokens
        ):
            raise RuntimeError(
                f"private decode budget is limited to {handle.max_tokens} generated tokens"
            )
        adapter = runtime.token_adapter
        if adapter is None:
            raise RuntimeError("private decode requires the exact Gemma token/PLE adapter")

        if not token_ids or len(token_ids) > MAX_STREAM_Q_LEN:
            raise ValueError("private append accepts one to eight token IDs per native step")
        physical_q = max(LISTENER_Q_PER_FRAME, len(token_ids) + (len(token_ids) & 1))
        self._check_branch_budget(
            handle.session_id, BranchTokenBudget(
                branch.row.cache_position, len(token_ids), 0,
                min(handle.max_tokens - branch.token_count,
                    response_budget.remaining if response_budget is not None else handle.max_tokens)
                if counts_as_prompt or is_transition else 0,
            ),
        )
        if branch.row.cache_position > coordinator.capability.max_seq_len - physical_q:
            if handle.purpose in {
                ResponseDecodePurpose.ACTIVE.value,
                ResponseDecodePurpose.CANDIDATE.value,
            }:
                raise RecoverableSessionCapacityError(
                    handle.parent_session_id,
                    generation_id=handle.parent_generation_id,
                    frame_id=handle.source_frame_id,
                    cache_position=branch.row.cache_position,
                    requested_tokens=physical_q,
                    max_seq_len=self._operational_max_seq_len(),
                )
            raise RuntimeError("private decode branch exhausted physical text-Q cache capacity")

        mutation_started = False
        try:
            with _listener_stream_context(runtime.listener_stream), torch.inference_mode():
                ready_ns = time.monotonic_ns()
                digest = hashlib.sha256(
                    (b"private-decode-transition:" if is_transition else b"private-decode-token-ids:")
                    + handle.handle_id.encode("ascii") + b":"
                    + b",".join(str(value).encode("ascii") for value in token_ids)
                ).hexdigest()
                row = replace(
                    branch.row,
                    input_frame_sha256=digest,
                    reference_pcm_sha256=None,
                    frame_ready_at_ns=ready_ns,
                    reference_frame_id=None,
                    logical_q_len=len(token_ids),
                    ephemeral_reason=None,
                    queue_audio_for_next_turn=False,
                    queue_audio_reason=None,
                    queue_audio_onset_frame_id=None,
                )
                manifest = self._single_row_private_manifest(row)
                prepared = adapter.prepare(token_ids, logical_q_lens=len(token_ids))
                if tuple(prepared.logical_q_lens) != (len(token_ids),):
                    raise RuntimeError("private token adapter logical-Q metadata differs")
                if tuple(prepared.inputs_embeds.shape) != (1, physical_q, CONTROL_HIDDEN_WIDTH) or tuple(
                    prepared.per_layer_inputs.shape
                ) != (1, physical_q, 35, 256):
                    raise RuntimeError("private token adapter returned incompatible physical text-Q inputs")
                mutation_started = True
                output = runtime.native.step_async(
                    manifest,
                    prepared.inputs_embeds,
                    per_layer_inputs=prepared.per_layer_inputs,
                    producer_stream=runtime.listener_stream,
                )
                health = tuple(runtime.native.verify_output_health(output))
                if (
                    len(health) != 1
                    or health[0].row != row
                    or health[0].generation != output.generation
                    or not health[0].checked
                    or health[0].nonfinite
                    or output.manifest_id != manifest.manifest_id
                    or output.rows != (row,)
                ):
                    raise RuntimeError("private token output health/provenance check failed")
                output_row = 0 if len(token_ids) > LISTENER_Q_PER_FRAME else len(token_ids) - 1
                final_hidden = output.tensor[0, output_row, :].reshape(1, 1, -1)
                final_logits = adapter.logits_from_final_norm_hidden(final_hidden)
                if final_logits.ndim == 3:
                    final_logits = final_logits[:, -1, :]
                if final_logits.ndim != 2 or final_logits.shape[0] != 1:
                    raise RuntimeError("private token LM head returned incompatible logits")
                new_position = row.cache_position + len(token_ids)
                branch.row = replace(
                    row,
                    cache_position=new_position,
                    valid_history_tokens=new_position,
                )
                ready_event = _record_shadow_event(runtime.listener_stream)
                if is_transition:
                    branch.transition_token_count += len(token_ids)
                elif response_prompt:
                    branch.prompt_token_count += len(token_ids)
                else:
                    branch.token_count += len(token_ids)
                    if response_budget is not None:
                        response_budget.consumed += len(token_ids)
                branch.last_ready_event = ready_event
                return PrivateTextAppendResult(
                    handle=handle,
                    cache_position=branch.row.cache_position,
                    token_count=branch.token_count,
                    logits=final_logits,
                    final_norm_hidden=final_hidden,
                    ready_event=ready_event,
                    timings_ns={"owner_completed_ns": time.monotonic_ns()},
                )
        except BaseException as exc:
            if mutation_started:
                self._end_ephemeral_decode(handle)
                if bool(getattr(runtime.native, "_poisoned", False)):
                    self._fatal_error = RuntimeError(
                        f"native private decode failed after KV mutation: {exc}"
                    )
                    raise self._fatal_error from exc
            raise

    def _end_ephemeral_decode(self, handle: EphemeralDecodeHandle) -> None:
        branch = self._ephemeral_decode_branches.get(handle.handle_id)
        if branch is None:
            return
        if branch.handle != handle:
            raise RuntimeError("private decode handle identity does not match its owner record")
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("private decode owner is unavailable")
        try:
            with _listener_stream_context(runtime.listener_stream):
                if branch.last_ready_event is not None:
                    wait_event = getattr(runtime.listener_stream, "wait_event", None)
                    if callable(wait_event):
                        wait_event(branch.last_ready_event)
                runtime.native.release_session_async(branch.row)
        except BaseException as exc:
            if self._cleanup_error is None:
                self._cleanup_error = exc
            self._fatal_error = exc
            raise
        finally:
            self._ephemeral_decode_branches.pop(handle.handle_id, None)
            self._kv_slot_roles.pop(handle.fork_session_id, None)
            work_owner = self._work_owners.get(handle.session_id)
            if work_owner is not None and work_owner.kv_owner_epoch == handle.kv_owner_epoch:
                self._release_work_owner(work_owner)
            if handle.purpose in {
                ResponseDecodePurpose.ACTIVE.value,
                ResponseDecodePurpose.CANDIDATE.value,
            }:
                revision_key = (
                    handle.parent_session_id,
                    str(handle.response_id),
                    int(handle.response_generation_id or 0),
                    handle.purpose,
                )
                with self._response_revision_lock:
                    if self._latest_response_revisions.get(revision_key) == handle.revision_id:
                        self._latest_response_revisions.pop(revision_key, None)

    def _end_response_decode(self, handle: ResponseDecodeHandle) -> None:
        branch = self._ephemeral_decode_branches.get(handle.handle_id)
        if branch is None:
            return
        if branch.handle != handle:
            # A queued cancellation/cleanup can retain the pre-promotion
            # candidate value. That alias must never release the transferred
            # ACTIVE branch that now has the same stable handle_id.
            if (
                handle.purpose == ResponseDecodePurpose.CANDIDATE.value
                and branch.handle.purpose == ResponseDecodePurpose.ACTIVE.value
                and replace(branch.handle, purpose=ResponseDecodePurpose.CANDIDATE.value)
                == handle
            ):
                return
            raise RuntimeError("response decode handle identity does not match its owner record")
        self._end_ephemeral_decode(handle)

    def _end_parent_ephemeral_decodes(self, session_id: str) -> None:
        handles = tuple(
            branch.handle
            for branch in self._ephemeral_decode_branches.values()
            if branch.handle.parent_session_id == session_id
        )
        first_error: BaseException | None = None
        for handle in handles:
            try:
                self._end_ephemeral_decode(handle)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        with self._response_revision_lock:
            self._latest_response_revisions = {
                key: revision
                for key, revision in self._latest_response_revisions.items()
                if key[0] != session_id
            }
        with self._kv_work_lock:
            self._work_owners.pop(session_id, None)
        self._response_token_budgets = {
            key: budget for key, budget in self._response_token_budgets.items() if key[0] != session_id
        }
        self._drain_detached_response_decode_cleanup()
        if first_error is not None:
            raise first_error

    def _end_all_ephemeral_decodes(self) -> None:
        for session_id in tuple(self._maintenance_jobs):
            self._clear_session_maintenance(session_id)
        self._drain_detached_response_decode_cleanup()
        handles = tuple(branch.handle for branch in self._ephemeral_decode_branches.values())
        for handle in handles:
            self._end_ephemeral_decode(handle)
        with self._response_revision_lock:
            self._latest_response_revisions.clear()
        with self._kv_work_lock:
            self._work_owners.clear()
        self._response_token_budgets.clear()

    def _single_row_private_manifest(self, row: BatchRowDescriptor) -> ListenerBatchManifest:
        coordinator = self._coordinator
        if coordinator is None:
            raise RuntimeError("private manifest requires the live listener coordinator")
        capability = coordinator.capability
        return ListenerBatchManifest(
            manifest_id=uuid.uuid4().hex,
            runtime_signature=coordinator.runtime_signature,
            stream_identity=capability.stream_identity,
            stream_priority=capability.stream_priority,
            dispatch_at_ns=time.monotonic_ns(),
            max_coalesce_wait_ns=0,
            dispatch_reason="carryover_ready",
            rows=(row,),
        )

    def _release_q2_decision_cache(self, session_id: str) -> None:
        self._mark_stable_q2_epoch_unusable(session_id)
        cache = self._q2_decision_caches.get(session_id)
        if cache is None:
            return
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("cannot release a Q2 decision fork without its native owner")
        with _listener_stream_context(runtime.listener_stream):
            runtime.native.release_session_async(cache.fork_row)
        self._q2_decision_caches.pop(session_id, None)

    def _mark_stable_q2_epoch_unusable(
        self, session_id: str, *, queue_epoch: str | None = None,
    ) -> None:
        if not self._stable_turn_policy_q2_context:
            return
        if session_id not in self._assistant_audio_queue_open:
            return
        active_epoch = self._assistant_audio_queue_epoch.get(session_id)
        if active_epoch is not None and (queue_epoch is None or queue_epoch == active_epoch):
            self._stable_q2_unusable_epochs[session_id] = active_epoch

    def _stable_q2_epoch_usable(self, session_id: str, queue_epoch: str) -> bool:
        return bool(
            self._stable_turn_policy_q2_context
            and session_id in self._assistant_audio_queue_open
            and self._stable_q2_unusable_epochs.get(session_id) != queue_epoch
        )

    def _q2_cache_anchor(
        self, source_row: BatchRowDescriptor, queue_epoch: str,
    ) -> tuple[Any, ...]:
        prefix_anchor = (
            self._prefix_lengths.get(source_row.session_id),
            self._prefix_revisions.get(source_row.session_id),
            self._unified_prefix_revisions.get(source_row.session_id),
            self._prefix_replay_frame_ids.get(source_row.session_id, ()),
            id(self._prefix_importers.get(source_row.session_id)),
        )
        return (
            queue_epoch,
            int(self._assistant_audio_queue_first_frame_id.get(source_row.session_id, -1)),
            source_row.session_id,
            source_row.session_generation,
            source_row.state_epoch_id,
            source_row.native_context_id,
            self._assistant_audio_queue_response_id.get(source_row.session_id),
            *prefix_anchor,
        )

    def _stable_q2_cache_anchor(
        self,
        source_row: BatchRowDescriptor,
        queue_epoch: str,
        first_frame_id: int,
    ) -> tuple[Any, ...]:
        prefix_anchor = (
            self._prefix_lengths.get(source_row.session_id),
            self._prefix_revisions.get(source_row.session_id),
            self._unified_prefix_revisions.get(source_row.session_id),
            self._prefix_replay_frame_ids.get(source_row.session_id, ()),
            id(self._prefix_importers.get(source_row.session_id)),
        )
        return (
            queue_epoch,
            int(first_frame_id),
            source_row.session_generation,
            source_row.native_context_id,
            *prefix_anchor,
        )

    def _capture_stable_q2_context(
        self,
        source_row: BatchRowDescriptor,
        *,
        queue_epoch: str,
        first_frame_id: int,
    ) -> _Q2DecisionCache | None:
        """Fork one frozen policy context at queue-open, before text can advance KV."""
        runtime = self._runtime
        coordinator = self._coordinator
        if (
            runtime is None
            or coordinator is None
            or getattr(runtime.native, "_metadata_rebind_api_available", False) is not True
            or not callable(getattr(runtime.native, "rebind_ephemeral_session_metadata", None))
            or not callable(getattr(runtime.native, "fork_session_slot", None))
            or not callable(getattr(runtime.native, "step_queued_audio_async", None))
        ):
            return None
        required_capacity = self._q2_slot_capacity(additional_slots=1)
        native_capacity = getattr(runtime.native, "max_batch_size", None)
        if (
            isinstance(native_capacity, int)
            and not isinstance(native_capacity, bool)
            and required_capacity > native_capacity
        ):
            return None
        self._ensure_capacity_for_registration(required_capacity)
        fork_session_id = f"__q2view__{source_row.session_id}__{uuid.uuid4().hex}"
        fork_context_id = self._private_context_id(source_row.native_context_id)
        expected_fork_row = replace(
            source_row,
            session_id=fork_session_id,
            native_context_id=fork_context_id,
        )
        fork_row = runtime.native.fork_session_slot(
            source_row,
            fork_session_id=fork_session_id,
            fork_native_context_id=fork_context_id,
        )
        if fork_row != expected_fork_row:
            try:
                runtime.native.release_session_async(fork_row)
            finally:
                raise RuntimeError("stable queue-open Q2 fork changed its canonical source boundary")
        return _Q2DecisionCache(
            queue_epoch=queue_epoch,
            anchor=self._stable_q2_cache_anchor(source_row, queue_epoch, first_frame_id),
            fork_row=fork_row,
            base_cache_position=source_row.cache_position,
            last_applied_frame_id=first_frame_id - 1,
            q2_applied_count=0,
            stable_turn_context=True,
        )

    def _q2_slot_capacity(self, *, additional_slots: int = 0) -> int:
        if isinstance(additional_slots, bool) or not isinstance(additional_slots, int) or additional_slots < 0:
            raise ValueError("additional Q2 slot count must be a non-negative integer")
        return (
            len(self._session_states)
            + len(self._ephemeral_decode_branches)
            + len(self._q2_decision_caches)
            + len(self._maintenance_jobs)
            + len(self._retiring_owners)
            + additional_slots
        )

    @staticmethod
    def _q2_cache_expected_cursor(cache: _Q2DecisionCache) -> int:
        return cache.base_cache_position + LISTENER_Q_PER_FRAME * cache.q2_applied_count

    def _record_assistant_queue_frame(
        self,
        row: BatchRowDescriptor,
        *,
        reason: str | None,
        onset_frame_id: int | None,
        queue_epoch: int | None,
    ) -> None:
        """Record one assistant-phase Q0 classification with capture provenance."""
        session_id = row.session_id
        if session_id not in self._assistant_audio_queue_open:
            return
        if row.logical_q_len != 0 or row.ephemeral_reason != "assistant_turn_open":
            raise RuntimeError("assistant audio queue only accepts assistant-phase Q0 rows")
        first_frame_id = self._assistant_audio_queue_first_frame_id.get(session_id)
        spans = self._assistant_audio_queue_frame_spans.get(session_id)
        if first_frame_id is None or spans is None:
            raise RuntimeError("assistant audio queue source watermark is unavailable")
        expected_frame_id = int(first_frame_id) + (
            0 if not spans else spans[-1].last_frame_id - int(first_frame_id) + 1
        )
        if int(row.frame_id) != expected_frame_id:
            raise RuntimeError("assistant queue Q0 source frame IDs are not contiguous")
        retained = bool(row.queue_audio_for_next_turn)
        if retained:
            if reason not in {"pre_roll", "utterance_body", "endpoint_tail"}:
                raise RuntimeError("retained assistant queue frame has no valid retention reason")
            if onset_frame_id is None or int(onset_frame_id) < 0:
                raise RuntimeError("retained assistant queue frame has no onset owner")
            if reason == "pre_roll" and int(row.frame_id) >= int(onset_frame_id):
                raise RuntimeError("pre-roll frame must precede its owning onset")
            if reason != "pre_roll" and int(row.frame_id) < int(onset_frame_id):
                raise RuntimeError("utterance frame precedes its owning onset")
        elif reason is not None or onset_frame_id is not None:
            raise RuntimeError("unretained assistant queue frame carries retention metadata")
        if spans:
            previous = spans[-1]
            if (
                previous.retained == retained
                and previous.queue_epoch == queue_epoch
                and previous.reason == reason
                and previous.onset_frame_id == onset_frame_id
                and previous.last_frame_id + 1 == int(row.frame_id)
                and previous.last_state_version + 1 == int(row.listener_state_version)
                and previous.last_sequence_position + 1 == int(row.sequence_position)
            ):
                previous.last_frame_id = int(row.frame_id)
                previous.last_state_version = int(row.listener_state_version)
                previous.last_sequence_position = int(row.sequence_position)
                return
        spans.append(_AssistantQueueFrameSpan(
            first_frame_id=int(row.frame_id),
            last_frame_id=int(row.frame_id),
            retained=retained,
            first_state_version=int(row.listener_state_version),
            last_state_version=int(row.listener_state_version),
            first_sequence_position=int(row.sequence_position),
            last_sequence_position=int(row.sequence_position),
            queue_epoch=queue_epoch,
            reason=reason,
            onset_frame_id=onset_frame_id,
        ))

    @staticmethod
    def _assistant_queue_span_contains_retained_frame(
        spans: Sequence[_AssistantQueueFrameSpan],
        frame_id: int,
    ) -> bool:
        return any(
            span.retained and span.first_frame_id <= frame_id <= span.last_frame_id
            for span in spans
        )

    def _promote_pre_roll_span(
        self,
        session_id: str,
        item: QueuedAudioSoftFrame,
        *,
        reason: str,
        onset_frame_id: int,
        queue_epoch: int | None,
    ) -> bool:
        """Promote one already-consumed ring frame without changing its source ID."""
        spans = self._assistant_audio_queue_frame_spans.get(session_id)
        if spans is None or item.source_queue_epoch != queue_epoch:
            return False
        frame_id = int(item.source_frame_id)
        for index, span in enumerate(spans):
            if not span.first_frame_id <= frame_id <= span.last_frame_id:
                continue
            if span.retained or span.queue_epoch != queue_epoch:
                return False
            offset = frame_id - span.first_frame_id
            promoted: list[_AssistantQueueFrameSpan] = []
            if offset:
                promoted.append(replace(
                    span,
                    last_frame_id=frame_id - 1,
                    last_state_version=span.first_state_version + offset - 1,
                    last_sequence_position=span.first_sequence_position + offset - 1,
                ))
            promoted.append(_AssistantQueueFrameSpan(
                first_frame_id=frame_id,
                last_frame_id=frame_id,
                retained=True,
                first_state_version=span.first_state_version + offset,
                last_state_version=span.first_state_version + offset,
                first_sequence_position=span.first_sequence_position + offset,
                last_sequence_position=span.first_sequence_position + offset,
                queue_epoch=queue_epoch,
                reason=reason,
                onset_frame_id=int(onset_frame_id),
            ))
            suffix_width = span.last_frame_id - frame_id
            if suffix_width:
                promoted.append(replace(
                    span,
                    first_frame_id=frame_id + 1,
                    first_state_version=span.first_state_version + offset + 1,
                    first_sequence_position=span.first_sequence_position + offset + 1,
                ))
            spans[index:index + 1] = promoted
            return True
        return False

    def _transfer_pre_roll_at_onset(
        self,
        row: BatchRowDescriptor,
        *,
        onset_frame_id: int,
        queue_epoch: int | None,
    ) -> None:
        """Atomically promote eligible prior Q0 frames and clear the speculative ring."""
        session_id = row.session_id
        ring = self._pre_roll_ring.setdefault(session_id, deque(maxlen=3))
        queued = self._queued_audio_soft.setdefault(session_id, deque())
        queued_ids = {int(item.source_frame_id) for item in queued}
        spans = self._assistant_audio_queue_frame_spans.get(session_id, ())
        candidates = sorted(tuple(ring), key=lambda item: int(item.source_frame_id))
        for item in candidates:
            frame_id = int(item.source_frame_id)
            if (
                item.session_id != session_id
                or item.session_generation != row.session_generation
                or item.state_epoch_id != row.state_epoch_id
                or item.source_queue_epoch != queue_epoch
                or frame_id < int(self._assistant_audio_queue_first_frame_id.get(session_id, row.frame_id))
                or frame_id >= row.frame_id
                or frame_id in queued_ids
                or self._assistant_queue_span_contains_retained_frame(spans, frame_id)
            ):
                continue
            reason = "pre_roll" if frame_id < onset_frame_id else "utterance_body"
            if not self._promote_pre_roll_span(
                session_id,
                item,
                reason=reason,
                onset_frame_id=int(onset_frame_id),
                queue_epoch=queue_epoch,
            ):
                continue
            if queued and frame_id <= int(queued[-1].source_frame_id):
                raise RuntimeError("pre-roll transfer would reorder retained audio frames")
            queued.append(item)
            queued_ids.add(frame_id)
        ring.clear()
        self._compact_assistant_queue_spans(
            self._assistant_audio_queue_frame_spans.get(session_id, []),
        )

    def _remember_pre_roll_frame(
        self,
        session_id: str,
        item: QueuedAudioSoftFrame,
        *,
        source_pcm: np.ndarray,
    ) -> None:
        """Keep recent non-silent, unretained Q2 frames for this exact identity."""
        if not np.any(source_pcm):
            return
        ring = self._pre_roll_ring.setdefault(session_id, deque(maxlen=3))
        if ring:
            latest = ring[-1]
            same_identity = (
                latest.session_generation == item.session_generation
                and latest.state_epoch_id == item.state_epoch_id
                and latest.source_queue_epoch == item.source_queue_epoch
            )
            if not same_identity:
                ring.clear()
            elif item.source_frame_id <= latest.source_frame_id:
                raise RuntimeError("pre-roll ring frame IDs must increase within one queue epoch")
        ring.append(item)

    @staticmethod
    def _compact_assistant_queue_spans(
        spans: list[_AssistantQueueFrameSpan],
    ) -> None:
        compacted: list[_AssistantQueueFrameSpan] = []
        for span in spans:
            if compacted:
                previous = compacted[-1]
                if (
                    previous.retained == span.retained
                    and previous.queue_epoch == span.queue_epoch
                    and previous.reason == span.reason
                    and previous.onset_frame_id == span.onset_frame_id
                    and previous.last_frame_id + 1 == span.first_frame_id
                    and previous.last_state_version + 1 == span.first_state_version
                    and previous.last_sequence_position + 1 == span.first_sequence_position
                ):
                    previous.last_frame_id = span.last_frame_id
                    previous.last_state_version = span.last_state_version
                    previous.last_sequence_position = span.last_sequence_position
                    continue
            compacted.append(span)
        spans[:] = compacted

    def _validate_assistant_queue_retention(
        self,
        session_id: str,
        snapshot: ListenerSessionSnapshot,
        queued: Sequence[QueuedAudioSoftFrame],
    ) -> None:
        """Validate sparse retained frames against submitted Q0 retention spans."""
        first_frame_id = self._assistant_audio_queue_first_frame_id.get(session_id)
        spans = self._assistant_audio_queue_frame_spans.get(session_id)
        if first_frame_id is None or spans is None:
            raise RuntimeError("queued Q2 onset/policy watermark is missing")
        cursor = int(first_frame_id)
        retained_sources: list[tuple[int, int, int, int | None]] = []
        for span in spans:
            width = span.last_frame_id - span.first_frame_id + 1
            if (
                width <= 0
                or span.first_frame_id != cursor
                or span.last_state_version - span.first_state_version != width - 1
                or span.last_sequence_position - span.first_sequence_position != width - 1
            ):
                raise RuntimeError("assistant queue retention spans have invalid source provenance")
            if span.retained:
                if (
                    span.reason not in {"pre_roll", "utterance_body", "endpoint_tail"}
                    or span.onset_frame_id is None
                    or (
                        span.reason == "pre_roll"
                        and span.last_frame_id >= span.onset_frame_id
                    )
                    or (
                        span.reason != "pre_roll"
                        and span.first_frame_id < span.onset_frame_id
                    )
                ):
                    raise RuntimeError("assistant queue retained span has invalid onset classification")
                retained_sources.extend(
                    (
                        frame_id,
                        span.first_state_version + frame_id - span.first_frame_id,
                        span.first_sequence_position + frame_id - span.first_frame_id,
                        span.source_queue_epoch,
                    )
                    for frame_id in range(span.first_frame_id, span.last_frame_id + 1)
                )
            elif span.reason is not None or span.onset_frame_id is not None:
                raise RuntimeError("assistant queue discarded span carries retention metadata")
            cursor = span.last_frame_id + 1
        if cursor != int(snapshot.next_frame_id):
            raise RuntimeError("assistant queue retention spans do not cover the submitted Q0 watermark")
        actual_ids = tuple(int(item.source_frame_id) for item in queued)
        expected_ids = tuple(source[0] for source in retained_sources)
        if actual_ids != expected_ids:
            raise RuntimeError(
                "queued Q2 retained frame IDs differ from submitted retention provenance"
            )
        for item, (frame_id, state_version, sequence_position, source_queue_epoch) in zip(
            queued, retained_sources, strict=True
        ):
            if (
                item.session_id != session_id
                or item.session_generation != snapshot.generation_id
                or item.state_epoch_id != snapshot.state_epoch_id
                or item.source_frame_id != frame_id
                or item.source_state_version != state_version
                or item.source_sequence_position != sequence_position
                or item.source_queue_epoch != source_queue_epoch
                or item.source_frame_id >= snapshot.next_frame_id
                or tuple(item.audio_soft_tokens.shape)
                != (LISTENER_Q_PER_FRAME, CONTROL_HIDDEN_WIDTH)
                or item.audio_soft_tokens.dtype != torch.bfloat16
                or item.audio_soft_tokens.device.type != "cuda"
            ):
                raise RuntimeError("queued audio provenance/tensor does not match submitted retention")

    def _run_pending_q2_prefill_idle_step(self) -> bool:
        """Advance at most one already-queued Q2 while the owner has no ready work."""
        runtime = self._runtime
        coordinator = self._coordinator
        if (
            not self._pending_q2_prefills
            or self._active_queued_audio_drains
            or runtime is None
            or coordinator is None
            or not self.q2_decision_prefill_supported
            or not self._commands.empty()
            or coordinator.ready_depth != 0
            or self._tick_expected is not None
            or any(self._outstanding_frames.values())
        ):
            return False
        with self._wake:
            if not self._commands.empty():
                return False
        session_id, (generation_id, response_id) = next(iter(self._pending_q2_prefills.items()))
        if (
            session_id not in self._assistant_audio_queue_open
            or self._assistant_audio_queue_response_id.get(session_id) != response_id
        ):
            self._pending_q2_prefills.pop(session_id, None)
            return False
        snapshot = coordinator.session_state(session_id)
        if snapshot.generation_id != generation_id or snapshot.native_context_id is None:
            self._pending_q2_prefills.pop(session_id, None)
            self._release_q2_decision_cache(session_id)
            return False
        pending = tuple(self._policy_q2_soft_pending.get(session_id, ()))
        if not pending:
            return False
        first_frame_id = self._assistant_audio_queue_first_frame_id.get(session_id)
        queue_epoch = self._assistant_audio_queue_epoch.get(session_id)
        if first_frame_id is None or queue_epoch is None:
            self._pending_q2_prefills.pop(session_id, None)
            self._release_q2_decision_cache(session_id)
            return False
        cache = self._q2_decision_caches.get(session_id)
        applied_count = 0 if cache is None else cache.q2_applied_count
        last_applied_frame_id = (
            int(first_frame_id) - 1
            if cache is None
            else cache.last_applied_frame_id
        )
        expected_frame_id = int(last_applied_frame_id) + 1
        if cache is not None and (
            isinstance(applied_count, bool)
            or not isinstance(applied_count, int)
            or applied_count < 0
            or isinstance(last_applied_frame_id, bool)
            or not isinstance(last_applied_frame_id, int)
            or last_applied_frame_id != int(first_frame_id) + applied_count - 1
            or cache.fork_row.cache_position != self._q2_cache_expected_cursor(cache)
        ):
            self._pending_q2_prefills.pop(session_id, None)
            self._release_q2_decision_cache(session_id)
            return False
        for item in pending:
            if (
                item.session_id != session_id
                or item.session_generation != generation_id
                or item.state_epoch_id != snapshot.state_epoch_id
                or item.source_frame_id != expected_frame_id
                or item.source_frame_id >= snapshot.next_frame_id
                or item.source_state_version >= snapshot.state_version
                or item.source_sequence_position >= snapshot.sequence_position
                or tuple(item.audio_soft_tokens.shape) != (LISTENER_Q_PER_FRAME, CONTROL_HIDDEN_WIDTH)
                or item.audio_soft_tokens.dtype != torch.bfloat16
            ):
                self._pending_q2_prefills.pop(session_id, None)
                self._release_q2_decision_cache(session_id)
                return False
            expected_frame_id += 1

        previous_row = self._last_native_rows.get(session_id)
        if previous_row is None:
            self._pending_q2_prefills.pop(session_id, None)
            return False
        source_row = replace(
            previous_row,
            session_generation=snapshot.generation_id,
            state_epoch_id=snapshot.state_epoch_id,
            frame_id=snapshot.next_frame_id,
            listener_state_version=snapshot.state_version,
            sequence_position=snapshot.sequence_position,
            cache_position=snapshot.cache_position,
            valid_history_tokens=snapshot.valid_history_tokens,
            native_context_id=int(snapshot.native_context_id),
        )
        stable_context = self._stable_q2_epoch_usable(session_id, queue_epoch)
        if self._stable_turn_policy_q2_context and (
            not stable_context or cache is None or not cache.stable_turn_context
        ):
            # A stable fork is created exactly once at queue-open. A missing
            # or replaced slot must never silently re-anchor from text-advanced KV.
            self._mark_stable_q2_epoch_unusable(session_id, queue_epoch=queue_epoch)
            self._pending_q2_prefills.pop(session_id, None)
            self._release_q2_decision_cache(session_id)
            return False
        anchor = (
            self._stable_q2_cache_anchor(source_row, queue_epoch, int(first_frame_id))
            if stable_context
            else self._q2_cache_anchor(source_row, queue_epoch)
        )
        if cache is not None and (
            cache.queue_epoch != queue_epoch
            or cache.anchor != anchor
            or cache.fork_row.cache_position != self._q2_cache_expected_cursor(cache)
        ):
            self._release_q2_decision_cache(session_id)
            self._mark_stable_q2_epoch_unusable(session_id, queue_epoch=queue_epoch)
            self._pending_q2_prefills.pop(session_id, None)
            return False
        forked = cache is None
        replay_offset = 0
        branch_row: BatchRowDescriptor
        if not forked:
            try:
                rebind = runtime.native.rebind_ephemeral_session_metadata
                branch_row = rebind(cache.fork_row, source_row)
                if (
                    branch_row.session_id != cache.fork_row.session_id
                    or branch_row.native_context_id != cache.fork_row.native_context_id
                    or branch_row.session_generation != cache.fork_row.session_generation
                    or branch_row.cache_position != cache.fork_row.cache_position
                    or branch_row.valid_history_tokens != cache.fork_row.valid_history_tokens
                    or branch_row.frame_id != source_row.frame_id
                    or branch_row.state_epoch_id != source_row.state_epoch_id
                    or branch_row.listener_state_version != source_row.listener_state_version
                    or branch_row.sequence_position != source_row.sequence_position
                ):
                    raise RuntimeError("idle Q2 prefill metadata CAS changed fork KV provenance")
            except Exception:
                self._release_q2_decision_cache(session_id)
                if self._stable_turn_policy_q2_context:
                    self._mark_stable_q2_epoch_unusable(session_id, queue_epoch=queue_epoch)
                    self._pending_q2_prefills.pop(session_id, None)
                    return False
                cache = None
                forked = True
                stable_context = False
                anchor = self._q2_cache_anchor(source_row, queue_epoch)
            else:
                self._q2_decision_caches[session_id] = replace(
                    cache, fork_row=branch_row,
                )
                replay_offset = 0
        if forked:
            if self._stable_turn_policy_q2_context:
                self._mark_stable_q2_epoch_unusable(session_id, queue_epoch=queue_epoch)
                self._pending_q2_prefills.pop(session_id, None)
                return False
            required_capacity = self._q2_slot_capacity(additional_slots=1)
            native_capacity = getattr(runtime.native, "max_batch_size", None)
            if (
                isinstance(native_capacity, int)
                and not isinstance(native_capacity, bool)
                and required_capacity > native_capacity
            ):
                return False
            self._ensure_capacity_for_registration(required_capacity)
            handle = uuid.uuid4().hex
            fork_session_id = f"__q2view__{session_id}__{handle}"
            fork_context_id = self._private_context_id(source_row.native_context_id)
            expected_fork_row = replace(
                source_row,
                session_id=fork_session_id,
                native_context_id=fork_context_id,
            )
            branch_row = runtime.native.fork_session_slot(
                source_row,
                fork_session_id=fork_session_id,
                fork_native_context_id=fork_context_id,
            )
            if branch_row != expected_fork_row:
                runtime.native.release_session_async(branch_row)
                raise RuntimeError("idle Q2 prefill fork changed the canonical source boundary")
            replay_offset = 0

        if replay_offset >= len(pending):
            if forked:
                runtime.native.release_session_async(branch_row)
            self._pending_q2_prefills.pop(session_id, None)
            return False
        item = pending[replay_offset]
        source_frame_id = item.source_frame_id
        cursor = branch_row.cache_position
        if cursor > self._operational_max_seq_len() - LISTENER_Q_PER_FRAME:
            if forked:
                runtime.native.release_session_async(branch_row)
            self._pending_q2_prefills.pop(session_id, None)
            return False
        required_capacity = self._q2_slot_capacity(additional_slots=int(forked))
        self._ensure_capacity_for_registration(required_capacity)
        step_row = replace(
            branch_row,
            cache_position=cursor,
            valid_history_tokens=cursor,
            logical_q_len=LISTENER_Q_PER_FRAME,
            ephemeral_reason=None,
            queue_audio_for_next_turn=False,
            queue_audio_reason=None,
            queue_audio_onset_frame_id=None,
            input_frame_sha256=hashlib.sha256(
                f"q2-view:{session_id}:{generation_id}:{source_row.state_epoch_id}:"
                f"{source_frame_id}".encode("utf-8")
            ).hexdigest(),
        )
        manifest = self._single_row_private_manifest(step_row)
        try:
            with _listener_stream_context(runtime.listener_stream), torch.inference_mode():
                if bool(getattr(
                    item.audio_soft_tokens,
                    "is_cuda",
                    item.audio_soft_tokens.device.type == "cuda",
                )):
                    item.audio_soft_tokens.record_stream(runtime.listener_stream)
                output = runtime.native.step_queued_audio_async(
                    manifest,
                    item.audio_soft_tokens.unsqueeze(0).contiguous(),
                    producer_stream=runtime.listener_stream,
                )
                health_rows = tuple(runtime.native.verify_output_health(output))
                if (
                    not output.preserves_listener_cursors
                    or output.manifest_id != manifest.manifest_id
                    or output.rows != (step_row,)
                    or len(health_rows) != 1
                    or health_rows[0].row != step_row
                    or health_rows[0].generation != output.generation
                    or not health_rows[0].checked
                    or health_rows[0].nonfinite
                ):
                    raise RuntimeError("idle Q2 prefill output health/provenance failed")
            next_cursor = cursor + LISTENER_Q_PER_FRAME
            fork_row = replace(
                step_row,
                cache_position=next_cursor,
                valid_history_tokens=next_cursor,
            )
            self._q2_decision_caches[session_id] = _Q2DecisionCache(
                queue_epoch=queue_epoch,
                anchor=anchor,
                fork_row=fork_row,
                base_cache_position=(
                    cache.base_cache_position if cache is not None else source_row.cache_position
                ),
                last_applied_frame_id=source_frame_id,
                q2_applied_count=applied_count + 1,
                stable_turn_context=bool(cache is not None and cache.stable_turn_context),
            )
            if replay_offset + 1 == len(pending):
                self._pending_q2_prefills.pop(session_id, None)
            if self._policy_q2_soft_pending.get(session_id):
                first_pending = self._policy_q2_soft_pending[session_id][0]
                if first_pending.source_frame_id == item.source_frame_id:
                    self._policy_q2_soft_pending[session_id].popleft()
            return True
        except BaseException:
            self._pending_q2_prefills.pop(session_id, None)
            self._release_q2_decision_cache(session_id)
            if forked:
                try:
                    runtime.native.release_session_async(branch_row)
                except BaseException:
                    pass
            raise

    def _build_speaking_q2_decision_view(
        self,
        source_row: BatchRowDescriptor,
        current_audio_soft: torch.Tensor,
    ) -> TemporaryQ2DecisionView:
        """Build an exact Q2 view, extending its private fork within one queue."""
        decision_started_ns = time.perf_counter_ns()
        runtime = self._runtime
        coordinator = self._coordinator
        if runtime is None or coordinator is None:
            raise RuntimeError("temporary Q2 view requires the active native owner")
        if not callable(getattr(runtime.native, "fork_session_slot", None)) or not callable(
            getattr(runtime.native, "step_queued_audio_async", None)
        ):
            raise RuntimeError("native owner lacks the isolated Q2 replay APIs")
        if source_row.session_id not in self._assistant_audio_queue_open:
            raise RuntimeError("temporary Q2 view requires an open assistant-phase queue")
        if (
            source_row.logical_q_len != 0
            or source_row.ephemeral_reason != "assistant_turn_open"
        ):
            raise RuntimeError("temporary Q2 view requires an assistant-phase Q0 listener row")

        pending = tuple(self._policy_q2_soft_pending.get(source_row.session_id, ()))
        first_frame_id = self._assistant_audio_queue_first_frame_id.get(source_row.session_id)
        queue_epoch = self._assistant_audio_queue_epoch.get(source_row.session_id)
        if first_frame_id is None or queue_epoch is None:
            self._release_q2_decision_cache(source_row.session_id)
            if self._stable_turn_policy_q2_context:
                raise _Q2DecisionViewUnavailable(
                    "stable queue-open Q2 epoch is unavailable; refusing canonical Q0 policy fallback"
                )
            raise _Q2DecisionViewUnavailable(
                "assistant-turn queue epoch/watermark is unavailable; canonical Q0 is the explicit fallback"
            )
        if self._stable_turn_policy_q2_context and not self._stable_q2_epoch_usable(
            source_row.session_id, queue_epoch
        ):
            self._release_q2_decision_cache(source_row.session_id)
            raise _Q2DecisionViewUnavailable(
                "stable queue-open Q2 fork is unusable; refusing canonical Q0 policy fallback"
            )
        cache = self._q2_decision_caches.get(source_row.session_id)
        q2_applied_count = 0 if cache is None else cache.q2_applied_count
        last_applied_frame_id = (
            int(first_frame_id) - 1 if cache is None else cache.last_applied_frame_id
        )
        expected_frame_id = int(last_applied_frame_id) + 1
        if cache is not None and (
            isinstance(q2_applied_count, bool)
            or not isinstance(q2_applied_count, int)
            or q2_applied_count < 0
            or isinstance(last_applied_frame_id, bool)
            or not isinstance(last_applied_frame_id, int)
            or last_applied_frame_id != int(first_frame_id) + q2_applied_count - 1
            or cache.fork_row.cache_position != self._q2_cache_expected_cursor(cache)
        ):
            self._release_q2_decision_cache(source_row.session_id)
            raise _Q2DecisionViewUnavailable(
                "temporary Q2 cache watermark and physical fork cursor disagree"
            )
        replay: list[tuple[int, torch.Tensor]] = []
        for item in pending:
            # Policy Q2 is separate from canonical drain retention and includes
            # every submitted assistant-phase Q0 frame.
            if (
                item.session_id != source_row.session_id
                or item.session_generation != source_row.session_generation
                or item.state_epoch_id != source_row.state_epoch_id
                or item.source_frame_id != expected_frame_id
                or item.source_frame_id >= source_row.frame_id
                or item.source_state_version >= source_row.listener_state_version
                or item.source_sequence_position >= source_row.sequence_position
                or tuple(item.audio_soft_tokens.shape) != (LISTENER_Q_PER_FRAME, CONTROL_HIDDEN_WIDTH)
                or item.audio_soft_tokens.dtype != torch.bfloat16
            ):
                self._release_q2_decision_cache(source_row.session_id)
                raise RuntimeError("temporary Q2 view pending policy provenance is stale or non-contiguous")
            replay.append((item.source_frame_id, item.audio_soft_tokens))
            expected_frame_id += 1
        if expected_frame_id != source_row.frame_id:
            self._release_q2_decision_cache(source_row.session_id)
            raise _Q2DecisionViewUnavailable(
                "temporary Q2 view is missing one or more policy frames; canonical Q0 is the explicit fallback"
            )
        if (
            tuple(current_audio_soft.shape) != (LISTENER_Q_PER_FRAME, CONTROL_HIDDEN_WIDTH)
            or current_audio_soft.dtype != torch.bfloat16
        ):
            raise RuntimeError("temporary Q2 view current frontend output is not exact BF16 Q2")
        replay.append((source_row.frame_id, current_audio_soft))
        source_q2_frame_ids = tuple(
            range(int(first_frame_id), int(source_row.frame_id) + 1)
        )
        rebind = getattr(runtime.native, "rebind_ephemeral_session_metadata", None)
        persistent_cache_available = (
            getattr(runtime.native, "_metadata_rebind_api_available", False) is True
            and callable(rebind)
            and source_row.session_id not in self._q2_cache_disabled_for_tick
        )
        stable_context = self._stable_q2_epoch_usable(source_row.session_id, queue_epoch)
        if self._stable_turn_policy_q2_context and stable_context and (
            not persistent_cache_available
            or cache is None
            or not cache.stable_turn_context
        ):
            # Rebinding is required to preserve current frame identity while
            # retaining the private physical cursor. Never recreate this fork.
            self._release_q2_decision_cache(source_row.session_id)
            self._mark_stable_q2_epoch_unusable(source_row.session_id, queue_epoch=queue_epoch)
            raise _Q2DecisionViewUnavailable(
                "stable queue-open Q2 fork lost its metadata/cache contract; "
                "refusing canonical Q0 policy fallback"
            )
        anchor = (
            self._stable_q2_cache_anchor(source_row, queue_epoch, int(first_frame_id))
            if stable_context
            else self._q2_cache_anchor(source_row, queue_epoch)
        )
        cache_hit = False
        replay_offset = 0
        branch_row: BatchRowDescriptor | None = None
        if cache is not None and (
            not persistent_cache_available
            or cache.stable_turn_context != stable_context
            or cache.queue_epoch != queue_epoch
            or cache.anchor != anchor
            or cache.last_applied_frame_id
            != int(first_frame_id) + cache.q2_applied_count - 1
            or cache.last_applied_frame_id >= source_row.frame_id
            or cache.fork_row.cache_position != self._q2_cache_expected_cursor(cache)
        ):
            self._release_q2_decision_cache(source_row.session_id)
            raise _Q2DecisionViewUnavailable(
                "temporary Q2 fork cache no longer matches the consumed policy-frame prefix"
            )
        if cache is not None and persistent_cache_available:
            try:
                rebound = rebind(cache.fork_row, source_row)
                if (
                    rebound.session_id != cache.fork_row.session_id
                    or rebound.native_context_id != cache.fork_row.native_context_id
                    or rebound.session_generation != cache.fork_row.session_generation
                    or rebound.cache_position != cache.fork_row.cache_position
                    or rebound.valid_history_tokens != cache.fork_row.valid_history_tokens
                    or rebound.frame_id != source_row.frame_id
                    or rebound.state_epoch_id != source_row.state_epoch_id
                    or rebound.listener_state_version != source_row.listener_state_version
                    or rebound.sequence_position != source_row.sequence_position
                ):
                    raise RuntimeError("metadata CAS changed Q2 fork provenance or KV cursor")
                branch_row = rebound
                replay_offset = 0
                cache_hit = True
            except Exception:
                self._release_q2_decision_cache(source_row.session_id)
                raise _Q2DecisionViewUnavailable(
                    "temporary Q2 fork metadata CAS failed after policy Q2 consumption"
                )

        if self._stable_turn_policy_q2_context and not cache_hit:
            raise _Q2DecisionViewUnavailable(
                "stable queue-open Q2 fork cannot be recreated after suffix append; "
                "refusing canonical Q0 policy fallback"
            )

        forked_this_tick = False
        if not cache_hit:
            handle = uuid.uuid4().hex
            fork_session_id = f"__q2view__{source_row.session_id}__{handle}"
            fork_context_id = self._private_context_id(source_row.native_context_id)
            expected_fork_row = replace(
                source_row,
                session_id=fork_session_id,
                native_context_id=fork_context_id,
            )
            requested = self._q2_slot_capacity(additional_slots=1)
            native_capacity = getattr(runtime.native, "max_batch_size", None)
            if (
                isinstance(native_capacity, int)
                and not isinstance(native_capacity, bool)
                and requested > native_capacity
            ):
                raise _Q2DecisionViewUnavailable(
                    "native listener has no free slot for a temporary Q2 decision fork"
                )
            self._ensure_capacity_for_registration(requested)
            branch_row = runtime.native.fork_session_slot(
                source_row,
                fork_session_id=fork_session_id,
                fork_native_context_id=fork_context_id,
            )
            forked_this_tick = True
            if branch_row != expected_fork_row:
                try:
                    runtime.native.release_session_async(branch_row)
                finally:
                    raise RuntimeError("temporary Q2 fork changed source boundary or cursor")
            replay_offset = 0

        final_output: NativeBatchOutput | None = None
        final_health: NativeBatchOutputHealth | None = None
        final_manifest: ListenerBatchManifest | None = None
        cursor = branch_row.cache_position
        base_cache_position = (
            cache.base_cache_position if cache is not None else source_row.cache_position
        )
        replay_to_run = replay[replay_offset:]
        if not replay_to_run:
            if forked_this_tick:
                runtime.native.release_session_async(branch_row)
            raise RuntimeError("temporary Q2 cache did not receive a new current frame")
        try:
            with _listener_stream_context(runtime.listener_stream), torch.inference_mode():
                batch_append = getattr(runtime.native, "step_queued_audio_batch_async", None)
                can_batch_append = (
                    bool(getattr(runtime.native, "supports_queued_audio_batch", False))
                    and callable(batch_append)
                )

                def replay_row(source_frame_id: int, cache_cursor: int) -> BatchRowDescriptor:
                    row = replace(
                        branch_row,
                        cache_position=cache_cursor,
                        valid_history_tokens=cache_cursor,
                        logical_q_len=LISTENER_Q_PER_FRAME,
                        ephemeral_reason=None,
                        queue_audio_for_next_turn=False,
                        queue_audio_reason=None,
                        queue_audio_onset_frame_id=None,
                    )
                    # The row identifies the decision boundary; its digest
                    # carries the last real Q2 source included by this call.
                    if source_frame_id != source_row.frame_id:
                        return replace(
                            row,
                            input_frame_sha256=hashlib.sha256(
                                f"q2-view:{source_row.session_id}:{source_row.session_generation}:"
                                f"{source_row.state_epoch_id}:{source_frame_id}".encode("utf-8")
                            ).hexdigest(),
                        )
                    return replace(row, input_frame_sha256=source_row.input_frame_sha256)

                def accept_replay_output(
                    output: NativeBatchOutput, manifest: ListenerBatchManifest,
                    step_row: BatchRowDescriptor,
                ) -> None:
                    nonlocal final_output, final_health, final_manifest
                    health_rows = tuple(runtime.native.verify_output_health(output))
                    if (
                        not output.preserves_listener_cursors
                        or output.manifest_id != manifest.manifest_id
                        or output.rows != (step_row,)
                        or len(health_rows) != 1
                        or health_rows[0].row != step_row
                        or health_rows[0].generation != output.generation
                        or not health_rows[0].checked
                        or health_rows[0].nonfinite
                    ):
                        raise RuntimeError("temporary Q2 view output health/provenance failed")
                    final_output = output
                    final_health = health_rows[0]
                    final_manifest = manifest

                if can_batch_append:
                    for chunk in _chunk_queued_q2_replay(replay_to_run):
                        query_count = len(chunk) * LISTENER_Q_PER_FRAME
                        if cursor > self._operational_max_seq_len() - query_count:
                            raise RuntimeError("temporary Q2 view exceeds native physical Q capacity")
                        step_row = replay_row(chunk[-1][0], cursor)
                        manifest = self._single_row_private_manifest(step_row)
                        for _source_frame_id, audio_soft in chunk:
                            if bool(getattr(audio_soft, "is_cuda", audio_soft.device.type == "cuda")):
                                audio_soft.record_stream(runtime.listener_stream)
                        packed_audio = torch.cat(
                            [audio_soft for _source_frame_id, audio_soft in chunk], dim=0
                        ).contiguous()
                        if bool(getattr(
                            packed_audio, "is_cuda", packed_audio.device.type == "cuda"
                        )):
                            packed_audio.record_stream(runtime.listener_stream)
                        self._ensure_capacity_for_registration(
                            self._q2_slot_capacity(additional_slots=int(forked_this_tick))
                        )
                        try:
                            output = batch_append(
                                manifest, packed_audio, producer_stream=runtime.listener_stream,
                            )
                        except ValueError as exc:
                            # The binding raises ValueError only for its
                            # read-only variable-Q preflight (or a C ABI
                            # precondition rejection). The append did not
                            # mutate this private fork, so fail this optional
                            # decision view closed without quarantining the
                            # canonical listener owner.
                            raise _Q2DecisionViewUnavailable(
                                "temporary Q2 variable-Q append was rejected before KV mutation: "
                                f"{exc}"
                            ) from exc
                        accept_replay_output(output, manifest, step_row)
                        cursor += query_count
                        branch_row = replace(
                            step_row,
                            cache_position=cursor,
                            valid_history_tokens=cursor,
                        )
                else:
                    for source_frame_id, audio_soft in replay_to_run:
                        if cursor > self._operational_max_seq_len() - LISTENER_Q_PER_FRAME:
                            raise RuntimeError("temporary Q2 view exceeds native physical Q2 capacity")
                        step_row = replay_row(source_frame_id, cursor)
                        manifest = self._single_row_private_manifest(step_row)
                        if bool(getattr(audio_soft, "is_cuda", audio_soft.device.type == "cuda")):
                            audio_soft.record_stream(runtime.listener_stream)
                        self._ensure_capacity_for_registration(
                            self._q2_slot_capacity(additional_slots=int(forked_this_tick))
                        )
                        output = runtime.native.step_queued_audio_async(
                            manifest,
                            audio_soft.unsqueeze(0).contiguous(),
                            producer_stream=runtime.listener_stream,
                        )
                        accept_replay_output(output, manifest, step_row)
                        cursor += LISTENER_Q_PER_FRAME
                        branch_row = replace(
                            step_row,
                            cache_position=cursor,
                            valid_history_tokens=cursor,
                        )
            if final_output is None or final_health is None or final_manifest is None:
                raise RuntimeError("temporary Q2 view replay produced no current-frame output")
            current_step_row = final_manifest.rows[0]
            next_q2_applied_count = q2_applied_count + len(replay_to_run)
            if cursor != base_cache_position + LISTENER_Q_PER_FRAME * next_q2_applied_count:
                raise RuntimeError("temporary Q2 append cursor differs from its applied-frame watermark")
            self._q2_decision_caches.pop(source_row.session_id, None)
            if persistent_cache_available:
                self._q2_decision_caches[source_row.session_id] = _Q2DecisionCache(
                    queue_epoch=queue_epoch,
                    anchor=anchor,
                    fork_row=branch_row,
                    base_cache_position=base_cache_position,
                    last_applied_frame_id=source_row.frame_id,
                    q2_applied_count=next_q2_applied_count,
                    stable_turn_context=stable_context,
                )
            decision_view_ms = round(
                max(0, time.perf_counter_ns() - decision_started_ns) / 1_000_000.0,
                3,
            )
            return TemporaryQ2DecisionView(
                source_row=source_row,
                fork_row=current_step_row,
                source_q2_frame_ids=source_q2_frame_ids,
                manifest=final_manifest,
                native_output=final_output,
                native_output_health=final_health,
                gemma_hidden_t=final_output.tensor[0, 0, :].reshape(1, 1, -1),
                gemma_audio_queries_t=final_output.tensor[0],
                replay_count=next_q2_applied_count,
                fork_count=int(forked_this_tick),
                replayed_q2_count=len(replay_to_run),
                cache_hit=cache_hit,
                decision_view_ms=decision_view_ms,
                context_mode=(
                    "stable_turn" if stable_context else
                    "legacy_fallback" if self._stable_turn_policy_q2_context else "legacy"
                ),
            )
        except BaseException:
            if persistent_cache_available:
                self._q2_decision_caches.pop(source_row.session_id, None)
                if forked_this_tick or cache_hit:
                    try:
                        with _listener_stream_context(runtime.listener_stream):
                            runtime.native.release_session_async(branch_row)
                    except BaseException as cleanup_error:
                        if self._cleanup_error is None:
                            self._cleanup_error = cleanup_error
                        raise
            raise
        finally:
            if not persistent_cache_available and forked_this_tick:
                try:
                    with _listener_stream_context(runtime.listener_stream):
                        runtime.native.release_session_async(branch_row)
                except BaseException as cleanup_error:
                    if self._cleanup_error is None:
                        self._cleanup_error = cleanup_error
                    raise

    def _begin_assistant_audio_queue(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
        reserve_tail_tokens: int,
        response_id: str | None = None,
    ) -> ListenerSessionSnapshot:
        if session_id in self._active_queued_audio_drains:
            raise RuntimeError("cannot open an assistant queue while queued-audio drain is active")
        snapshot = self._check_unified_session_cas(
            session_id,
            expected_generation_id=expected_generation_id,
            expected_cache_position=expected_cache_position,
        )
        if self._outstanding_frames.get(session_id, 0) != 0:
            raise RuntimeError("assistant audio queue must open at an idle frame boundary")
        if session_id in self._assistant_audio_queue_open:
            raise RuntimeError("assistant audio queue is already open for this model turn")
        if (
            self._queued_audio_soft.get(session_id)
            or self._policy_q2_soft_pending.get(session_id)
        ):
            raise RuntimeError("cannot open a new assistant turn with undrained queued audio")
        self._validate_tail_reservation(reserve_tail_tokens)
        coordinator = self._coordinator
        assert coordinator is not None
        if snapshot.cache_position + reserve_tail_tokens > self._operational_max_seq_len():
            raise RuntimeError(
                "native listener capacity cannot reserve the model-turn close boundary; "
                f"session={session_id}, cursor={snapshot.cache_position}, "
                f"reserved_tail_tokens={reserve_tail_tokens}, "
                f"max_seq_len={self._operational_max_seq_len()}"
            )
        self._release_q2_decision_cache(session_id)
        runtime = self._runtime
        if runtime is None or not callable(getattr(runtime.native, "step_queued_audio_async", None)):
            raise RuntimeError("native runtime lacks queued-audio replay; no PCM re-tower fallback is allowed")
        queue_epoch = uuid.uuid4().hex
        first_frame_id = int(snapshot.next_frame_id)
        stable_cache: _Q2DecisionCache | None = None
        stable_epoch_unusable = False
        if self._stable_turn_policy_q2_context:
            previous_row = self._last_native_rows.get(session_id)
            if (
                previous_row is None
                or previous_row.session_generation != snapshot.generation_id
                or previous_row.state_epoch_id != snapshot.state_epoch_id
                or previous_row.frame_id not in (
                    snapshot.next_frame_id - 1,
                    snapshot.next_frame_id,
                )
                or previous_row.listener_state_version != snapshot.state_version
                or previous_row.sequence_position != snapshot.sequence_position
                or previous_row.cache_position != snapshot.cache_position
                or previous_row.valid_history_tokens != snapshot.valid_history_tokens
                or previous_row.native_context_id != snapshot.native_context_id
            ):
                stable_epoch_unusable = True
            else:
                source_row = replace(
                    previous_row,
                    session_generation=snapshot.generation_id,
                    state_epoch_id=snapshot.state_epoch_id,
                    frame_id=snapshot.next_frame_id,
                    listener_state_version=snapshot.state_version,
                    sequence_position=snapshot.sequence_position,
                    cache_position=snapshot.cache_position,
                    valid_history_tokens=snapshot.valid_history_tokens,
                    native_context_id=int(snapshot.native_context_id),
                )
                stable_cache = self._capture_stable_q2_context(
                    source_row,
                    queue_epoch=queue_epoch,
                    first_frame_id=first_frame_id,
                )
                stable_epoch_unusable = stable_cache is None
            if stable_cache is None or stable_epoch_unusable:
                raise RuntimeError(
                    "stable queue-open Q2 context is unavailable; refusing to open a SPEAK queue"
                )
        self._reserved_tail_tokens[session_id] = reserve_tail_tokens
        self._assistant_audio_queue_open.add(session_id)
        self._assistant_audio_queue_first_frame_id[session_id] = first_frame_id
        self._assistant_audio_queue_frame_spans[session_id] = []
        self._pre_roll_ring.setdefault(session_id, deque(maxlen=3)).clear()
        self._assistant_audio_active_onset[session_id] = None
        self._policy_q2_soft_pending[session_id] = deque()
        self._assistant_audio_queue_epoch[session_id] = queue_epoch
        self._stable_q2_unusable_epochs.pop(session_id, None)
        if stable_cache is not None:
            self._q2_decision_caches[session_id] = stable_cache
        elif stable_epoch_unusable:
            self._stable_q2_unusable_epochs[session_id] = queue_epoch
        if response_id is None:
            self._assistant_audio_queue_response_id.pop(session_id, None)
        else:
            self._assistant_audio_queue_response_id[session_id] = response_id
        self._pending_q2_prefills.pop(session_id, None)
        self._trace_control6_queue_handoff(
            session_id,
            "queue_open",
            generation=int(snapshot.generation_id),
            cursor=int(snapshot.cache_position),
            next_frame_id=int(snapshot.next_frame_id),
            outstanding_frames=int(self._outstanding_frames.get(session_id, 0)),
            reserve_tail_tokens=int(reserve_tail_tokens),
        )
        return snapshot

    def _mark_assistant_audio_queue_closing(
        self,
        session_id: str,
        *,
        queue_close_epoch: int,
    ) -> None:
        if session_id not in self._assistant_audio_queue_open:
            return
        previous_epoch = self._assistant_audio_queue_close_epoch.get(session_id)
        if previous_epoch is not None and queue_close_epoch < previous_epoch:
            raise RuntimeError("assistant queue-close epochs must be monotone")
        self._assistant_audio_queue_close_epoch[session_id] = int(queue_close_epoch)
        self._trace_control6_queue_handoff(
            session_id,
            "queue_close_published",
            queue_close_epoch=int(queue_close_epoch),
        )

    def _request_q2_decision_prefill(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        response_id: str,
    ) -> bool:
        if (
            not self.q2_decision_prefill_supported
            or session_id not in self._assistant_audio_queue_open
            or self._assistant_audio_queue_response_id.get(session_id) != response_id
        ):
            return False
        try:
            snapshot = self._coordinator.session_state(session_id)
        except KeyError:
            return False
        if snapshot.generation_id != expected_generation_id:
            return False
        self._pending_q2_prefills[session_id] = (int(expected_generation_id), response_id)
        return True

    @staticmethod
    def _validate_tail_reservation(token_count: int) -> None:
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            raise ValueError("reserved tail token count must be a non-negative integer")

    def _reserve_response_decode_budget(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
        requested_tokens: int,
        transition_tokens: int,
        sentence_completion_tokens: int,
    ) -> dict[str, int | str | None]:
        """Atomically cap and reserve text output against queued audio and close KV."""
        if session_id in self._active_queued_audio_drains:
            raise RuntimeError("cannot reserve response text while queued-audio drain is active")
        snapshot = self._check_unified_session_cas(
            session_id,
            expected_generation_id=expected_generation_id,
            expected_cache_position=expected_cache_position,
        )
        if self._outstanding_frames.get(session_id, 0) != 0:
            raise RuntimeError("response decode preflight requires a completed listener frame watermark")
        if self._tick_expected is not None and session_id in self._tick_expected:
            raise RuntimeError("response decode preflight cannot overlap a shared listener tick")
        self._validate_tail_reservation(requested_tokens)
        self._validate_tail_reservation(transition_tokens)
        self._validate_tail_reservation(sentence_completion_tokens)
        if sentence_completion_tokens > 32:
            raise ValueError("sentence completion extension cannot exceed 32 tokens")

        coordinator = self._coordinator
        if coordinator is None:
            raise RuntimeError("unified native runtime is unavailable")
        max_seq_len = int(self._operational_max_seq_len())
        queued_frames = len(self._queued_audio_soft.get(session_id, ()))
        policy_pending_frames = len(self._policy_q2_soft_pending.get(session_id, ()))
        queued_audio_tokens = 2 * queued_frames
        policy_pending_tokens = 2 * policy_pending_frames
        close_reservation = int(self._reserved_tail_tokens.get(session_id, 0))
        transition_physical = (
            max(LISTENER_Q_PER_FRAME, transition_tokens + (transition_tokens & 1))
            if transition_tokens else 0
        )
        transition_padding = max(0, transition_physical - transition_tokens)
        physical_guard = max(0, LISTENER_Q_PER_FRAME - close_reservation)
        available = max(
            0,
            max_seq_len
            - int(snapshot.cache_position)
            - queued_audio_tokens
            - policy_pending_tokens
            - close_reservation
            - transition_tokens
            - transition_padding
            - physical_guard,
        )
        effective_tokens = min(int(requested_tokens), available, self._kv_budgets.assistant_response_max_tokens)
        actual_extension = min(
            int(sentence_completion_tokens),
            max(0, available - effective_tokens),
            self._kv_budgets.assistant_response_max_tokens - effective_tokens,
        )
        reserved_for_decode = (
            transition_tokens + transition_padding + effective_tokens + actual_extension
        )
        total_reservation = close_reservation + reserved_for_decode + physical_guard
        if (
            int(snapshot.cache_position)
            + queued_audio_tokens
            + total_reservation
            > max_seq_len
        ):
            raise RuntimeError("response decode reservation exceeds native KV capacity")
        self._reserved_tail_tokens[session_id] = total_reservation
        return {
            "requested_tokens": int(requested_tokens),
            "effective_tokens": int(effective_tokens),
            "sentence_completion_tokens": int(actual_extension),
            "cap_reason": (
                "assistant_response_budget"
                if effective_tokens < requested_tokens
                and effective_tokens == self._kv_budgets.assistant_response_max_tokens
                else "kv_capacity" if effective_tokens < requested_tokens else None
            ),
            "max_seq_len": max_seq_len,
            "cache_position": int(snapshot.cache_position),
            "queued_audio_frames": queued_frames,
            "queued_audio_tokens": queued_audio_tokens,
            "policy_pending_audio_frames": policy_pending_frames,
            "policy_pending_audio_tokens": policy_pending_tokens,
            "close_reservation_tokens": close_reservation,
            "reservation_tokens": total_reservation,
            "transition_tokens": int(transition_tokens),
            "transition_padding_tokens": transition_padding,
            "physical_guard_tokens": physical_guard,
        }

    def _reserve_tail_tokens(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
        token_count: int,
    ) -> int:
        if session_id in self._active_queued_audio_drains:
            raise RuntimeError("cannot reserve text-tail tokens while queued-audio drain is active")
        snapshot = self._check_unified_session_cas(
            session_id,
            expected_generation_id=expected_generation_id,
            expected_cache_position=expected_cache_position,
        )
        self._validate_tail_reservation(token_count)
        current_reservation = self._reserved_tail_tokens.get(session_id, 0)
        if self._outstanding_frames.get(session_id, 0) != 0:
            raise RuntimeError("tail reservation requires a completed listener frame watermark")
        if self._tick_expected is not None and session_id in self._tick_expected:
            raise RuntimeError("tail reservation cannot overlap a shared listener tick")
        coordinator = self._coordinator
        assert coordinator is not None
        queued_tokens = 2 * len(self._queued_audio_soft.get(session_id, ()))
        if snapshot.cache_position + queued_tokens + token_count > self._operational_max_seq_len():
            raise RuntimeError(
                "native listener capacity cannot preserve queued audio and reserved tail; "
                f"session={session_id}, cursor={snapshot.cache_position}, "
                f"queued_audio_tokens={queued_tokens}, reserved_tail_tokens={token_count}, "
                f"max_seq_len={self._operational_max_seq_len()}"
            )
        # A matching reservation remains a no-op only after the completed
        # listener watermark is observed; finalizers cannot close over Q0 rows
        # that have been accepted but not yet consumed.
        if current_reservation == token_count:
            return current_reservation
        self._reserved_tail_tokens[session_id] = token_count
        return token_count

    def _rollback_assistant_branch(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
        target_cache_position: int,
    ) -> ListenerSessionSnapshot:
        if session_id in self._active_queued_audio_drains:
            raise RuntimeError("cannot roll back text while queued-audio drain is active")
        runtime = self._runtime
        coordinator = self._coordinator
        if runtime is None or coordinator is None:
            raise RuntimeError("unified native runtime is unavailable")
        snapshot = self._check_unified_session_cas(
            session_id,
            expected_generation_id=expected_generation_id,
            expected_cache_position=expected_cache_position,
        )
        if self._outstanding_frames.get(session_id, 0) != 0:
            raise RuntimeError("assistant rollback requires a completed listener frame watermark")
        if self._tick_expected is not None and session_id in self._tick_expected:
            raise RuntimeError("assistant rollback cannot cross a shared listener tick")
        prefix_len = self._prefix_lengths.get(session_id)
        if prefix_len is None or not prefix_len <= target_cache_position <= snapshot.cache_position:
            raise ValueError(
                "assistant rollback target must remain between the true chat prefix and current cursor"
            )
        row = self._last_native_rows.get(session_id)
        if row is None:
            raise RuntimeError("assistant rollback has no native owner row")
        self._pre_roll_ring.setdefault(session_id, deque(maxlen=3)).clear()
        queue_response_id = self._assistant_audio_queue_response_id.get(session_id)
        queue_epoch = self._assistant_audio_queue_epoch.get(session_id)
        queue_first_frame_id = self._assistant_audio_queue_first_frame_id.get(session_id)
        queued_frame_ids = [
            int(item.source_frame_id)
            for item in self._queued_audio_soft.get(session_id, ())
        ]
        self._release_q2_decision_cache(session_id)
        self._pending_q2_prefills.pop(session_id, None)
        self._assistant_audio_queue_response_id.pop(session_id, None)
        current_row = replace(
            row,
            session_generation=snapshot.generation_id,
            state_epoch_id=snapshot.state_epoch_id,
            frame_id=snapshot.next_frame_id,
            listener_state_version=snapshot.state_version,
            sequence_position=snapshot.sequence_position,
            cache_position=snapshot.cache_position,
            valid_history_tokens=snapshot.valid_history_tokens,
            native_context_id=int(snapshot.native_context_id or 0),
        )
        trace_enabled = _unified_kv_integrity_trace_enabled()
        native_cursor_before: int | None = None
        native_cursor_snapshot_error: str | None = None
        if trace_enabled:
            snapshot_native_state = getattr(runtime.native, "session_state", None)
            if callable(snapshot_native_state):
                try:
                    native_state_before = snapshot_native_state(current_row)
                    native_cursor_before = int(native_state_before.current_position)
                except Exception as trace_error:
                    native_cursor_snapshot_error = type(trace_error).__name__
        try:
            with _listener_stream_context(runtime.listener_stream):
                rewound = runtime.native.rewind_session_async(
                    current_row, target_position=target_cache_position,
                )
                updated = coordinator.rewind_cache_position(
                    session_id,
                    target_cache_position=target_cache_position,
                    expected_generation_id=expected_generation_id,
                    expected_cache_position=expected_cache_position,
                )
            if (
                rewound.cache_position != updated.cache_position
                or rewound.valid_history_tokens != updated.valid_history_tokens
            ):
                raise RuntimeError("native and coordinator cursors diverged after assistant rollback")
            # Cancel/rollback returns to the open user turn. The assistant close
            # suffix is no longer required, so queued soft tokens can consume
            # the released tail immediately.
            self._reserved_tail_tokens[session_id] = 0
            self._last_native_rows[session_id] = replace(
                current_row,
                frame_id=updated.next_frame_id,
                listener_state_version=updated.state_version,
                sequence_position=updated.sequence_position,
                cache_position=updated.cache_position,
                valid_history_tokens=updated.valid_history_tokens,
            )
            if trace_enabled:
                trace_fields: dict[str, Any] = {
                    "generation_id": int(snapshot.generation_id),
                    "state_epoch_id": snapshot.state_epoch_id,
                    "native_context_id": int(snapshot.native_context_id or 0),
                    "response_id": queue_response_id,
                    "queue_epoch": queue_epoch,
                    "queue_first_frame_id": queue_first_frame_id,
                    "queued_frame_ids": queued_frame_ids,
                    "requested_exclusive_target": int(target_cache_position),
                    "coordinator_cursor_before": int(snapshot.cache_position),
                    "coordinator_cursor_after": int(updated.cache_position),
                    "native_cursor_before": native_cursor_before,
                    "native_cursor_after": int(rewound.cache_position),
                    "native_cursor_snapshot_error": native_cursor_snapshot_error,
                }
                if (
                    native_cursor_before is not None
                    and int(rewound.cache_position) <= native_cursor_before
                ):
                    trace_fields["rolled_back_tokens_count"] = (
                        native_cursor_before - int(rewound.cache_position)
                    )
                self._trace_unified_kv_integrity(
                    session_id,
                    "assistant_branch_rollback_committed",
                    **trace_fields,
                )
            return updated
        except BaseException as exc:
            self._pre_roll_ring.get(session_id, deque()).clear()
            self._queued_audio_soft.get(session_id, deque()).clear()
            self._policy_q2_soft_pending.get(session_id, deque()).clear()
            self._assistant_audio_active_onset[session_id] = None
            if trace_enabled:
                self._trace_unified_kv_integrity(
                    session_id,
                    "assistant_branch_rollback_failed",
                    generation_id=int(snapshot.generation_id),
                    state_epoch_id=snapshot.state_epoch_id,
                    native_context_id=int(snapshot.native_context_id or 0),
                    response_id=queue_response_id,
                    queue_epoch=queue_epoch,
                    requested_exclusive_target=int(target_cache_position),
                    coordinator_cursor_before=int(snapshot.cache_position),
                    native_cursor_before=native_cursor_before,
                    native_cursor_snapshot_error=native_cursor_snapshot_error,
                    error_type=type(exc).__name__,
                )
            self._fatal_error = RuntimeError(
                f"assistant branch rollback failed after native CAS: {exc}"
            )
            raise self._fatal_error from exc

    def _drain_queued_audio(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
        user_turn_already_open: bool,
        drain_submitted_at_ns: int | None = None,
        drain_future: Future[ListenerSessionSnapshot] | None = None,
    ) -> ListenerSessionSnapshot | object:
        drain_started_ns = _perf_counter_ns()
        submitted_at_ns = (
            drain_started_ns if drain_submitted_at_ns is None else int(drain_submitted_at_ns)
        )
        drain_queue_wait_ms = max(0, drain_started_ns - submitted_at_ns) / 1_000_000.0
        if session_id in self._active_queued_audio_drains:
            raise RuntimeError("queued-audio drain is already active for this session")
        runtime = self._runtime
        coordinator = self._coordinator
        if runtime is None or coordinator is None:
            raise RuntimeError("unified native runtime is unavailable")
        snapshot = self._check_unified_session_cas(
            session_id,
            expected_generation_id=expected_generation_id,
            expected_cache_position=expected_cache_position,
        )
        if self._outstanding_frames.get(session_id, 0) != 0:
            raise RuntimeError(
                "queued-audio drain needs the assistant Q0 watermark to complete; retry after the prior tick"
            )
        if self._tick_expected is not None and session_id in self._tick_expected:
            raise RuntimeError("queued-audio drain cannot overlap a shared listener tick")
        if session_id not in self._assistant_audio_queue_open:
            raise RuntimeError("queued-audio drain has no open assistant-turn queue")
        queued = tuple(self._queued_audio_soft.get(session_id, ()))
        trace_enabled = _unified_kv_integrity_trace_enabled()
        queue_epoch = self._assistant_audio_queue_epoch.get(session_id)
        queue_response_id = self._assistant_audio_queue_response_id.get(session_id)
        queue_first_frame_id = self._assistant_audio_queue_first_frame_id.get(session_id)
        native_cursor_before: int | None = None
        native_cursor_snapshot_error: str | None = None
        trace_base_row = self._last_native_rows.get(session_id)
        snapshot_native_state = getattr(runtime.native, "session_state", None)
        if queued and trace_enabled and trace_base_row is not None and callable(snapshot_native_state):
            try:
                native_cursor_before = int(
                    snapshot_native_state(trace_base_row).current_position
                )
            except Exception as trace_error:
                native_cursor_snapshot_error = type(trace_error).__name__
        max_seq_len = int(self._operational_max_seq_len())
        required = 2 * len(queued)
        reserved_tail = self._reserved_tail_tokens.get(session_id, 0)
        if user_turn_already_open and reserved_tail != 0:
            raise RuntimeError("cancel drain requires assistant tail reservation to be released by rollback")
        if snapshot.cache_position + required + reserved_tail > max_seq_len:
            raise RuntimeError(
                "queued audio cannot fit the current native KV capacity; no queued frame was dropped "
                f"(session={session_id}, cursor={snapshot.cache_position}, "
                f"queued_tokens={required}, reserved_tail_tokens={reserved_tail}, "
                f"max_seq_len={max_seq_len})"
            )
        self._validate_assistant_queue_retention(session_id, snapshot, queued)
        spans = tuple(self._assistant_audio_queue_frame_spans.get(session_id, ()))
        close_epoch = self._assistant_audio_queue_close_epoch.get(session_id)
        if close_epoch is None:
            sealed_spans, post_close_spans = spans, ()
            sealed_queue, post_close_queue = queued, ()
        else:
            sealed_spans = tuple(
                span for span in spans
                if span.source_queue_epoch is None or span.source_queue_epoch < close_epoch
            )
            post_close_spans = tuple(
                span for span in spans
                if span.source_queue_epoch is not None
                and span.source_queue_epoch >= close_epoch
            )
            sealed_queue = tuple(
                item for item in queued
                if item.source_queue_epoch is None or item.source_queue_epoch < close_epoch
            )
            post_close_queue = tuple(
                item for item in queued
                if item.source_queue_epoch is not None
                and item.source_queue_epoch >= close_epoch
            )

        self._release_q2_decision_cache(session_id)
        if not queued:
            self._reserved_tail_tokens[session_id] = 0
            self._retire_assistant_audio_queue_after_drain(session_id)
            drain_completed_ns = _perf_counter_ns()
            drain_total_elapsed_ms = max(0, drain_completed_ns - drain_started_ns) / 1_000_000.0
            if trace_enabled:
                self._trace_unified_kv_integrity(
                    session_id,
                    "queued_audio_drain_committed",
                    generation_id=int(snapshot.generation_id),
                    state_epoch_id=snapshot.state_epoch_id,
                    native_context_id=int(snapshot.native_context_id or 0),
                    response_id=queue_response_id,
                    queue_epoch=queue_epoch,
                    queue_first_frame_id=queue_first_frame_id,
                    user_turn_already_open=bool(user_turn_already_open),
                    audio_boa_reused_existing_user_turn=bool(user_turn_already_open),
                    source_frame_ids=[],
                    reinjected_frame_ids=[],
                    coordinator_cursor_before=int(snapshot.cache_position),
                    coordinator_cursor_after=int(snapshot.cache_position),
                    q2_tokens_reinjected=0,
                    native_cursor_before=native_cursor_before,
                    native_cursor_after=native_cursor_before,
                    native_cursor_snapshot_error=native_cursor_snapshot_error,
                )
            drain_metrics = QueuedAudioDrainMetrics(
                elapsed_ms=drain_total_elapsed_ms,
                frame_count=0,
                token_count=0,
                retained_context_source="assistant_phase_q2_queue_empty",
                source_frame_ids=(),
                queue_response_id=queue_response_id,
                queue_epoch=queue_epoch,
                queue_first_frame_id=queue_first_frame_id,
                drain_queue_wait_ms=drain_queue_wait_ms,
                owner_wait_ms=drain_queue_wait_ms,
                drain_replay_forward_ms=0.0,
                retained_frames_count=0,
                drain_total_elapsed_ms=drain_total_elapsed_ms,
            )
            return replace(snapshot, queued_audio_drain_metrics=drain_metrics)
        if not callable(getattr(runtime.native, "step_queued_audio_async", None)):
            raise RuntimeError("native runtime lacks queued-audio replay; no PCM re-tower fallback is allowed")
        if drain_future is None:
            raise RuntimeError("cooperative queued-audio drain requires its completion Future")
        if trace_base_row is None:
            raise RuntimeError("queued-audio drain has no native session row")
        state = _QueuedAudioDrainState(
            session_id=session_id,
            generation_id=int(snapshot.generation_id),
            state_epoch_id=str(snapshot.state_epoch_id),
            expected_cache_position=int(snapshot.cache_position),
            sealed_spans=sealed_spans,
            queued=list(sealed_queue),
            handoff_spans=post_close_spans,
            handoff_queue=list(post_close_queue),
            future=drain_future,
            submitted_at_ns=submitted_at_ns,
            started_at_ns=drain_started_ns,
            user_turn_already_open=bool(user_turn_already_open),
            queue_response_id=queue_response_id,
            queue_epoch=queue_epoch,
            queue_first_frame_id=queue_first_frame_id,
            max_seq_len=max_seq_len,
            trace_enabled=trace_enabled,
            native_cursor_before=native_cursor_before,
            native_cursor_snapshot_error=native_cursor_snapshot_error,
        )
        self._active_queued_audio_drains[session_id] = state
        try:
            self._rotate_assistant_audio_queue_for_drain(session_id, snapshot)
        except BaseException as exc:
            self._fail_queued_audio_drain(state, exc)
        return _PENDING_OWNER_RESULT

    def _rotate_assistant_audio_queue_for_drain(
        self,
        session_id: str,
        snapshot: ListenerSessionSnapshot,
    ) -> None:
        """Start a distinct post-close Q0 queue while the drain owns sealed spans."""
        self._queued_audio_soft[session_id] = deque()
        self._pre_roll_ring.setdefault(session_id, deque(maxlen=3)).clear()
        self._assistant_audio_active_onset[session_id] = None
        self._policy_q2_soft_pending[session_id] = deque()
        first_frame_id = int(snapshot.next_frame_id)
        self._assistant_audio_queue_first_frame_id[session_id] = first_frame_id
        self._assistant_audio_queue_frame_spans[session_id] = []
        queue_epoch = uuid.uuid4().hex
        self._assistant_audio_queue_epoch[session_id] = queue_epoch
        self._assistant_audio_queue_response_id.pop(session_id, None)
        self._pending_q2_prefills.pop(session_id, None)
        self._stable_q2_unusable_epochs.pop(session_id, None)
        if self._stable_turn_policy_q2_context:
            base_row = self._last_native_rows.get(session_id)
            if base_row is None:
                raise RuntimeError("post-close Q0 queue has no canonical native row")
            source_row = replace(
                base_row,
                session_generation=snapshot.generation_id,
                state_epoch_id=snapshot.state_epoch_id,
                frame_id=snapshot.next_frame_id,
                listener_state_version=snapshot.state_version,
                sequence_position=snapshot.sequence_position,
                cache_position=snapshot.cache_position,
                valid_history_tokens=snapshot.valid_history_tokens,
                native_context_id=int(snapshot.native_context_id or 0),
            )
            stable_cache = self._capture_stable_q2_context(
                source_row,
                queue_epoch=queue_epoch,
                first_frame_id=first_frame_id,
            )
            if stable_cache is None:
                self._stable_q2_unusable_epochs[session_id] = queue_epoch
            else:
                self._q2_decision_caches[session_id] = stable_cache

    def _retire_assistant_audio_queue_after_drain(self, session_id: str) -> None:
        self._release_q2_decision_cache(session_id)
        self._assistant_audio_queue_open.discard(session_id)
        self._queued_audio_soft.get(session_id, deque()).clear()
        self._pre_roll_ring.setdefault(session_id, deque(maxlen=3)).clear()
        self._assistant_audio_active_onset.pop(session_id, None)
        self._policy_q2_soft_pending.get(session_id, deque()).clear()
        self._assistant_audio_queue_first_frame_id.pop(session_id, None)
        self._assistant_audio_queue_frame_spans.pop(session_id, None)
        self._assistant_audio_queue_close_epoch.pop(session_id, None)
        self._assistant_audio_queue_epoch.pop(session_id, None)
        self._stable_q2_unusable_epochs.pop(session_id, None)
        self._assistant_audio_queue_response_id.pop(session_id, None)
        self._pending_q2_prefills.pop(session_id, None)

    def _run_one_queued_audio_drain_quantum(self) -> bool:
        """Run one Q2 replay, handoff a later Q0 queue, or publish completion."""
        if not self._active_queued_audio_drains:
            return False
        coordinator = self._coordinator
        runtime = self._runtime
        if self._commands.qsize() or (coordinator is not None and coordinator.ready_depth):
            return False
        if coordinator is None or runtime is None:
            session_id, state = next(iter(self._active_queued_audio_drains.items()))
            self._fail_queued_audio_drain(state, RuntimeError(
                "unified native runtime is unavailable during queued-audio drain"
            ))
            return True
        session_id = None
        state = None
        for candidate_id, candidate in tuple(self._active_queued_audio_drains.items()):
            if (
                self._outstanding_frames.get(candidate_id, 0) == 0
                and (self._tick_expected is None or candidate_id not in self._tick_expected)
            ):
                session_id, state = candidate_id, candidate
                break
        if session_id is None or state is None:
            return False
        if state.index >= len(state.queued) and state.handoff_queue:
            state.sealed_spans = state.sealed_spans + state.handoff_spans
            state.queued.extend(state.handoff_queue)
            state.handoff_spans = ()
            state.handoff_queue.clear()
            self._yield_queued_audio_drain_turn(state)
            return True
        if state.index < len(state.queued):
            try:
                current = coordinator.session_state(session_id)
                self._validate_active_drain_snapshot(state, current)
                if current.cache_position != state.expected_cache_position:
                    raise RuntimeError(
                        "queued-audio drain cursor changed outside its owner quantum "
                        f"(expected={state.expected_cache_position}, actual={current.cache_position})"
                    )
                if current.cache_position > state.max_seq_len - LISTENER_Q_PER_FRAME:
                    raise RuntimeError(
                        "native physical Q2 capacity exhausted before queued-audio replay"
                    )
                base_row = self._last_native_rows.get(session_id)
                if base_row is None:
                    raise RuntimeError("queued-audio drain has no native session row")
                item = state.queued[state.index]
                input_digest = hashlib.sha256(
                    f"queued-soft-audio:{item.session_id}:{item.session_generation}:"
                    f"{item.state_epoch_id}:{item.source_frame_id}:"
                    f"{item.source_state_version}:{item.source_sequence_position}".encode("utf-8")
                ).hexdigest()
                row = replace(
                    base_row,
                    session_generation=current.generation_id,
                    state_epoch_id=current.state_epoch_id,
                    frame_id=current.next_frame_id,
                    listener_state_version=current.state_version,
                    sequence_position=current.sequence_position,
                    cache_position=current.cache_position,
                    valid_history_tokens=current.valid_history_tokens,
                    native_context_id=int(current.native_context_id or 0),
                    input_frame_sha256=input_digest,
                    reference_pcm_sha256=None,
                    frame_ready_at_ns=time.monotonic_ns(),
                    reference_frame_id=None,
                    logical_q_len=LISTENER_Q_PER_FRAME,
                    ephemeral_reason=None,
                    queue_audio_for_next_turn=False,
                    queue_audio_reason=None,
                    queue_audio_onset_frame_id=None,
                )
                capability = coordinator.capability
                manifest = ListenerBatchManifest(
                    manifest_id=uuid.uuid4().hex,
                    runtime_signature=coordinator.runtime_signature,
                    stream_identity=capability.stream_identity,
                    stream_priority=capability.stream_priority,
                    dispatch_at_ns=time.monotonic_ns(),
                    max_coalesce_wait_ns=0,
                    dispatch_reason="carryover_ready",
                    rows=(row,),
                )
                replay_started_ns = _perf_counter_ns()
                with _listener_stream_context(runtime.listener_stream), torch.inference_mode():
                    item.audio_soft_tokens.record_stream(runtime.listener_stream)
                    output = runtime.native.step_queued_audio_async(
                        manifest,
                        item.audio_soft_tokens.unsqueeze(0).contiguous(),
                        producer_stream=runtime.listener_stream,
                    )
                    health = tuple(runtime.native.verify_output_health(output))
                    if (
                        len(health) != 1
                        or health[0].row != row
                        or health[0].generation != output.generation
                        or not health[0].checked
                        or health[0].nonfinite
                        or not output.preserves_listener_cursors
                    ):
                        raise RuntimeError("queued-audio native output health/provenance failed")
                    updated = coordinator.advance_cache_position(
                        session_id,
                        token_count=LISTENER_Q_PER_FRAME,
                        expected_generation_id=state.generation_id,
                        expected_cache_position=current.cache_position,
                    )
                replay_completed_ns = _perf_counter_ns()
                state.replay_forward_ms += max(
                    0, replay_completed_ns - replay_started_ns
                ) / 1_000_000.0
                state.last_replay_completed_ns = replay_completed_ns
                state.index += 1
                state.expected_cache_position = int(updated.cache_position)
                self._last_native_rows[session_id] = replace(
                    row,
                    frame_id=updated.next_frame_id,
                    listener_state_version=updated.state_version,
                    sequence_position=updated.sequence_position,
                    cache_position=updated.cache_position,
                    valid_history_tokens=updated.valid_history_tokens,
                )
                self._yield_queued_audio_drain_turn(state)
                return True
            except BaseException as exc:
                self._fail_queued_audio_drain(state, exc)
                return True

        if not self._commands.empty():
            return False
        try:
            current = coordinator.session_state(session_id)
            self._validate_active_drain_snapshot(state, current)
            if current.cache_position != state.expected_cache_position:
                raise RuntimeError(
                    "queued-audio drain cursor changed before publication "
                    f"(expected={state.expected_cache_position}, actual={current.cache_position})"
                )
            queued = tuple(self._queued_audio_soft.get(session_id, ()))
            spans = tuple(self._assistant_audio_queue_frame_spans.get(session_id, ()))
            self._validate_assistant_queue_retention(session_id, current, queued)
            if queued:
                reserved_tail = self._reserved_tail_tokens.get(session_id, 0)
                remaining_tokens = 2 * (len(state.queued) - state.index + len(queued))
                if current.cache_position + remaining_tokens + reserved_tail > state.max_seq_len:
                    raise RuntimeError(
                        "post-close queued audio cannot fit native KV capacity; no frame was dropped"
                    )
                state.sealed_spans = state.sealed_spans + spans
                state.queued.extend(queued)
                self._release_q2_decision_cache(session_id)
                self._rotate_assistant_audio_queue_for_drain(session_id, current)
                state.expected_cache_position = int(current.cache_position)
                self._yield_queued_audio_drain_turn(state)
                return True
            state.sealed_spans = state.sealed_spans + spans
            self._complete_queued_audio_drain(state, current)
        except BaseException as exc:
            self._fail_queued_audio_drain(state, exc)
        return True

    def _yield_queued_audio_drain_turn(self, state: _QueuedAudioDrainState) -> None:
        """Rotate resumable drains so one session cannot monopolize the owner."""
        current = self._active_queued_audio_drains.get(state.session_id)
        if current is state:
            self._active_queued_audio_drains.pop(state.session_id)
            self._active_queued_audio_drains[state.session_id] = state

    @staticmethod
    def _validate_active_drain_snapshot(
        state: _QueuedAudioDrainState,
        snapshot: ListenerSessionSnapshot,
    ) -> None:
        if snapshot.generation_id != state.generation_id:
            raise RuntimeError("queued-audio drain lost its listener generation")
        if snapshot.state_epoch_id != state.state_epoch_id:
            raise RuntimeError("queued-audio drain lost its listener epoch")

    def _complete_queued_audio_drain(
        self,
        state: _QueuedAudioDrainState,
        current: ListenerSessionSnapshot,
    ) -> None:
        session_id = state.session_id
        initial_position = state.expected_cache_position - state.index * LISTENER_Q_PER_FRAME
        reinjected_frame_ids = [int(item.source_frame_id) for item in state.queued]
        if state.trace_enabled:
            native_cursor_after = None
            native_cursor_snapshot_error = state.native_cursor_snapshot_error
            snapshot_native_state = getattr(self._runtime.native, "session_state", None)
            if callable(snapshot_native_state):
                try:
                    native_cursor_after = int(
                        snapshot_native_state(self._last_native_rows[session_id]).current_position
                    )
                except Exception as trace_error:
                    native_cursor_snapshot_error = type(trace_error).__name__
            self._trace_unified_kv_integrity(
                session_id,
                "queued_audio_drain_committed",
                generation_id=state.generation_id,
                state_epoch_id=state.state_epoch_id,
                native_context_id=int(current.native_context_id or 0),
                response_id=state.queue_response_id,
                queue_epoch=state.queue_epoch,
                queue_first_frame_id=state.queue_first_frame_id,
                user_turn_already_open=state.user_turn_already_open,
                audio_boa_reused_existing_user_turn=state.user_turn_already_open,
                source_frame_ids=reinjected_frame_ids,
                reinjected_frame_ids=reinjected_frame_ids,
                coordinator_cursor_before=initial_position,
                coordinator_cursor_after=int(current.cache_position),
                q2_tokens_reinjected=(int(current.cache_position) - initial_position),
                native_cursor_before=state.native_cursor_before,
                native_cursor_after=native_cursor_after,
                native_cursor_snapshot_error=native_cursor_snapshot_error,
                sealed_span_count=len(state.sealed_spans),
            )
        self._reserved_tail_tokens[session_id] = 0
        self._retire_assistant_audio_queue_after_drain(session_id)
        completed_ns = _perf_counter_ns()
        total_elapsed_ms = max(0, completed_ns - state.started_at_ns) / 1_000_000.0
        completion_wait_ms = (
            0.0
            if state.last_replay_completed_ns is None
            else max(0, completed_ns - state.last_replay_completed_ns) / 1_000_000.0
        )
        source_frame_ids = tuple(int(item.source_frame_id) for item in state.queued)
        metrics = QueuedAudioDrainMetrics(
            elapsed_ms=total_elapsed_ms,
            frame_count=len(source_frame_ids),
            token_count=LISTENER_Q_PER_FRAME * len(source_frame_ids),
            retained_context_source="assistant_phase_q2_queue",
            source_frame_ids=source_frame_ids,
            queue_response_id=state.queue_response_id,
            queue_epoch=state.queue_epoch,
            queue_first_frame_id=state.queue_first_frame_id,
            drain_queue_wait_ms=max(
                0, state.started_at_ns - state.submitted_at_ns
            ) / 1_000_000.0,
            owner_wait_ms=max(
                0, state.started_at_ns - state.submitted_at_ns
            ) / 1_000_000.0,
            drain_replay_forward_ms=state.replay_forward_ms,
            retained_frames_count=len(source_frame_ids),
            drain_total_elapsed_ms=total_elapsed_ms,
            drain_completion_wait_ms=completion_wait_ms,
            source_queue_epochs=tuple(
                item.source_queue_epoch for item in state.queued
            ),
        )
        self._active_queued_audio_drains.pop(session_id, None)
        if not state.future.done():
            state.future.set_result(replace(current, queued_audio_drain_metrics=metrics))

    def _fail_queued_audio_drain(
        self,
        state: _QueuedAudioDrainState,
        error: BaseException,
    ) -> None:
        session_id = state.session_id
        if state.trace_enabled:
            try:
                snapshot = self._coordinator.session_state(session_id)
                self._trace_unified_kv_integrity(
                    session_id,
                    "queued_audio_drain_failed",
                    generation_id=state.generation_id,
                    state_epoch_id=state.state_epoch_id,
                    native_context_id=int(snapshot.native_context_id or 0),
                    response_id=state.queue_response_id,
                    queue_epoch=state.queue_epoch,
                    queue_first_frame_id=state.queue_first_frame_id,
                    user_turn_already_open=state.user_turn_already_open,
                    source_frame_ids=[int(item.source_frame_id) for item in state.queued],
                    coordinator_cursor_before=(
                        state.expected_cache_position
                        - state.index * LISTENER_Q_PER_FRAME
                    ),
                    coordinator_cursor_after=int(snapshot.cache_position),
                    native_cursor_before=state.native_cursor_before,
                    native_cursor_snapshot_error=state.native_cursor_snapshot_error,
                    error_type=type(error).__name__,
                )
            except Exception:
                pass
        fatal = RuntimeError(
            "queued-audio replay failed after native cache mutation; session is quarantined: "
            f"{type(error).__name__}: {error}"
        )
        self._fatal_error = fatal
        self._active_queued_audio_drains.pop(session_id, None)
        state.queued.clear()
        state.handoff_queue.clear()
        self._pre_roll_ring.get(session_id, deque()).clear()
        self._queued_audio_soft.get(session_id, deque()).clear()
        self._policy_q2_soft_pending.get(session_id, deque()).clear()
        self._assistant_audio_active_onset[session_id] = None
        self._assistant_audio_queue_open.discard(session_id)
        self._assistant_audio_queue_frame_spans.pop(session_id, None)
        self._assistant_audio_queue_first_frame_id.pop(session_id, None)
        self._assistant_audio_queue_epoch.pop(session_id, None)
        self._assistant_audio_queue_close_epoch.pop(session_id, None)
        self._assistant_audio_queue_response_id.pop(session_id, None)
        self._reserved_tail_tokens[session_id] = 0
        if not state.future.done():
            state.future.set_exception(fatal)

    def _register_session(
        self,
        state: ListenerSessionState,
        prefix_import: PrefixImportFactory | None,
    ) -> None:
        if not isinstance(state, ListenerSessionState):
            raise TypeError("state must be ListenerSessionState")
        if prefix_import is None and self._runtime is not None:
            prefix_import = self._runtime.prefix_importer
        if not callable(prefix_import):
            raise TypeError("prefix_import must be a callable owner-thread cache provider")
        self._ensure_capacity_for_registration(len(self._session_states) + 1)
        self._coordinator.register_session(state)
        self._session_states[state.session_id] = state
        self._kv_slot_roles[state.session_id] = KvOwnerRole.CANONICAL
        self._maintenance_states[state.session_id] = MemoryMaintenanceState.IDLE
        self._closing_sessions.discard(state.session_id)
        self._prefix_importers[state.session_id] = prefix_import
        self._prefix_revisions[state.session_id] = None
        self._unified_prefix_revisions[state.session_id] = 0
        self._prefix_fallback_reasons[state.session_id] = None
        self._prefix_lengths[state.session_id] = int(state.cache_position)
        self._prefix_replay_frame_ids[state.session_id] = ()
        self._outstanding_frames[state.session_id] = 0
        self._outstanding_logical_tokens[state.session_id] = 0
        self._recent_listener_q2[state.session_id] = deque(
            maxlen=self._kv_budgets.recent_audio_reference_frames
        )
        self._compaction_requested.pop(state.session_id, None)
        self._compaction_status[state.session_id] = {
            "status": "idle",
            "reason": None,
            "cache_position": int(state.cache_position),
        }
        self._compaction_reason.pop(state.session_id, None)
        self._requested_history_revision.pop(state.session_id, None)
        self._compaction_attempt_history_revision.pop(state.session_id, None)
        self._compaction_revision.setdefault(state.session_id, 0)
        self._queued_audio_soft[state.session_id] = deque()
        self._pre_roll_ring[state.session_id] = deque(maxlen=3)
        self._assistant_audio_active_onset[state.session_id] = None
        self._policy_q2_soft_pending[state.session_id] = deque()
        self._reserved_tail_tokens[state.session_id] = 0

    def _fork_native_session(self, source_session_id: str, fork_session_id: str) -> str:
        """Copy one idle native cache into a separate slot and private row state."""
        source_id = str(source_session_id)
        fork_id = str(fork_session_id)
        if not fork_id or fork_id == source_id:
            raise ValueError("secondary native fork needs a distinct non-empty session ID")
        if fork_id in self._session_states:
            raise ValueError("secondary native fork session ID is already registered")
        if self._runtime is None or self._coordinator is None:
            raise RuntimeError("secondary native fork requires the live primary owner")
        if source_id not in self._imported_sessions:
            raise RuntimeError("secondary native fork requires an imported primary cache")
        if (self._outstanding_frames.get(source_id, 0) != 0
                or (self._tick_expected is not None and source_id in self._tick_expected)
                or source_id in self._assistant_audio_queue_open
                or self._queued_audio_soft.get(source_id)):
            raise RuntimeError("secondary native fork requires an idle primary frame/queue boundary")
        fork_api = getattr(self._runtime.native, "fork_session_cache_within_owner", None)
        if not callable(fork_api):
            raise RuntimeError("native owner lacks independent secondary-slot cache fork")
        source_snapshot = self._coordinator.session_state(source_id)
        source_row = self._last_native_rows.get(source_id)
        source_state = self._session_states[source_id]
        if source_row is None or source_state.native_context_id is None:
            raise RuntimeError("secondary native fork has no primary native row/context")
        context_id = int(uuid.uuid4().int & ((1 << 63) - 1)) or 1
        while any(
            int(state.native_context_id or 0) == context_id
            for state in self._session_states.values()
        ):
            context_id = int(uuid.uuid4().int & ((1 << 63) - 1)) or 1

        def clone_state(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().clone()
            if isinstance(value, np.ndarray):
                return np.array(value, copy=True)
            return copy.deepcopy(value)

        fork_state = ListenerSessionState(
            session_id=fork_id,
            generation_id=int(source_snapshot.generation_id),
            state_epoch_id=str(source_snapshot.state_epoch_id),
            runtime_signature=source_state.runtime_signature,
            next_frame_id=int(source_snapshot.next_frame_id),
            state_version=int(source_snapshot.state_version),
            sequence_position=int(source_snapshot.sequence_position),
            cache_position=int(source_snapshot.cache_position),
            valid_history_tokens=int(source_snapshot.valid_history_tokens),
            native_context_id=context_id,
            user_pcm_tail=clone_state(source_state.user_pcm_tail),
            assistant_pcm_tail=clone_state(source_state.assistant_pcm_tail),
            user_mel_history=clone_state(source_state.user_mel_history),
            assistant_mel_history=clone_state(source_state.assistant_mel_history),
            attention_mask=clone_state(source_state.attention_mask),
            # Native KV/context live in the copied slot, never by Python alias.
            native_kv_cache=None,
            native_context=None,
            cache_epoch_id=int(source_snapshot.cache_epoch_id),
            semantic_context_revision=source_snapshot.semantic_context_revision,
            applied_history_revision=source_snapshot.applied_history_revision,
        )
        digest = hashlib.sha256(
            f"secondary-native-fork:{source_id}:{fork_id}:{source_snapshot.cache_position}".encode()
        ).hexdigest()
        destination_row = replace(
            source_row,
            session_id=fork_id,
            native_context_id=context_id,
            input_frame_sha256=digest,
            frame_ready_at_ns=time.monotonic_ns(),
        )
        self._ensure_capacity_for_registration(len(self._session_states) + 1)
        self._coordinator.register_session(fork_state)
        self._session_states[fork_id] = fork_state
        self._kv_slot_roles[fork_id] = KvOwnerRole.CANONICAL
        self._maintenance_states[fork_id] = MemoryMaintenanceState.IDLE
        self._prefix_importers[fork_id] = self._prefix_importers[source_id]
        self._prefix_revisions[fork_id] = self._prefix_revisions.get(source_id)
        self._unified_prefix_revisions[fork_id] = self._unified_prefix_revisions.get(source_id, 0)
        self._prefix_fallback_reasons[fork_id] = self._prefix_fallback_reasons.get(source_id)
        self._prefix_lengths[fork_id] = self._prefix_lengths[source_id]
        self._prefix_replay_frame_ids[fork_id] = self._prefix_replay_frame_ids.get(source_id, ())
        self._outstanding_frames[fork_id] = 0
        self._outstanding_logical_tokens[fork_id] = 0
        source_ring = self._recent_listener_q2.get(source_id, ())
        self._recent_listener_q2[fork_id] = deque(
            (
                CanonicalListenerQ2Frame(
                    frame=replace(item.frame, session_id=fork_id),
                    row=replace(item.row, session_id=fork_id, native_context_id=context_id),
                    audio_soft_tokens=item.audio_soft_tokens,
                )
                for item in source_ring
            ),
            maxlen=self._kv_budgets.recent_audio_reference_frames,
        )
        self._compaction_status[fork_id] = {
            "status": "idle",
            "reason": None,
            "cache_position": int(source_snapshot.cache_position),
        }
        self._compaction_revision[fork_id] = self._compaction_revision.get(source_id, 0)
        self._queued_audio_soft[fork_id] = deque()
        self._pre_roll_ring[fork_id] = deque(maxlen=3)
        self._assistant_audio_active_onset[fork_id] = None
        self._policy_q2_soft_pending[fork_id] = deque()
        self._reserved_tail_tokens[fork_id] = 0
        try:
            copied = fork_api(source_row, destination_row)
            copied_row = getattr(copied, "destination_row", copied)
            if copied_row != destination_row:
                raise RuntimeError("secondary native fork returned a different destination boundary")
            self._last_native_rows[fork_id] = destination_row
            self._imported_sessions.add(fork_id)
            return fork_id
        except BaseException:
            self._coordinator.close_session(fork_id)
            self._session_states.pop(fork_id, None)
            self._prefix_importers.pop(fork_id, None)
            self._prefix_revisions.pop(fork_id, None)
            self._unified_prefix_revisions.pop(fork_id, None)
            self._prefix_fallback_reasons.pop(fork_id, None)
            self._prefix_lengths.pop(fork_id, None)
            self._prefix_replay_frame_ids.pop(fork_id, None)
            self._outstanding_frames.pop(fork_id, None)
            self._outstanding_logical_tokens.pop(fork_id, None)
            self._recent_listener_q2.pop(fork_id, None)
            self._kv_slot_roles.pop(fork_id, None)
            self._compaction_status.pop(fork_id, None)
            self._compaction_revision.pop(fork_id, None)
            self._queued_audio_soft.pop(fork_id, None)
            self._pre_roll_ring.pop(fork_id, None)
            self._assistant_audio_active_onset.pop(fork_id, None)
            self._policy_q2_soft_pending.pop(fork_id, None)
            self._reserved_tail_tokens.pop(fork_id, None)
            raise

    def _refresh_unified_prefix_before_audio(
        self,
        session_id: str,
        prefix_token_ids: tuple[int, ...],
        *,
        new_revision: int,
        expected_generation_id: int,
        expected_cache_position: int,
        prefix_import: PrefixImportFactory,
    ) -> None:
        """Rebase only a pristine unified listener before any audio admission."""
        if self._coordinator is None:
            raise RuntimeError("unified prefix refresh requires a listener coordinator")
        if not callable(prefix_import):
            raise TypeError("prefix_import must be a callable owner-thread cache provider")
        if not 1 <= len(prefix_token_ids) <= NATIVE_CONTROL6_PREFIX_MAX_TOKENS:
            raise ValueError(
                "unified prefix token count must be in [1,"
                f"{NATIVE_CONTROL6_PREFIX_MAX_TOKENS}]"
            )
        state = self._session_states.get(session_id)
        if state is None:
            raise KeyError(f"unknown listener session: {session_id}")
        if state.closed or state.poisoned_reason is not None:
            raise RuntimeError("unified prefix refresh requires a live, unpoisoned session")
        if (
            state.generation_id != expected_generation_id
            or state.next_frame_id != 0
            or state.state_version != 0
            or state.sequence_position != 0
            or state.cache_position != expected_cache_position
            or state.valid_history_tokens != expected_cache_position
        ):
            raise RuntimeError(
                "unified prefix refresh lost its pristine frame-zero generation/cursor compare-and-swap"
            )
        current_revision = self._unified_prefix_revisions.get(session_id)
        if current_revision is None or new_revision <= current_revision:
            raise RuntimeError(
                "unified prefix refresh revision must advance monotonically "
                f"(current={current_revision}, requested={new_revision})"
            )
        if self._outstanding_frames.get(session_id, 0) != 0:
            raise RuntimeError("unified prefix refresh cannot cross an accepted listener frame")
        if self._tick_expected is not None and session_id in self._tick_expected:
            raise RuntimeError("unified prefix refresh cannot overlap a shared listener tick")
        if (
            session_id in self._assistant_audio_queue_open
            or self._queued_audio_soft.get(session_id)
            or self._reserved_tail_tokens.get(session_id, 0) != 0
        ):
            raise RuntimeError("unified prefix refresh cannot cross an assistant turn or queued audio")
        if len(prefix_token_ids) > self._kv_budgets.initial_prefix_max_tokens:
            raise ValueError("initial imported prefix exceeds its service token budget")
        imported = session_id in self._imported_sessions
        last_row = self._last_native_rows.get(session_id)
        if imported:
            if last_row is None or (
                last_row.session_generation != state.generation_id
                or last_row.state_epoch_id != state.state_epoch_id
                or last_row.frame_id != state.next_frame_id
                or last_row.listener_state_version != state.state_version
                or last_row.sequence_position != state.sequence_position
                or last_row.cache_position != expected_cache_position
                or last_row.valid_history_tokens != expected_cache_position
            ):
                raise RuntimeError("unified prefix refresh found a stale native cache binding")
        elif last_row is not None:
            raise RuntimeError("unified prefix refresh found an unowned native cache row")

        try:
            self._release_q2_decision_cache(session_id)
            if imported:
                self._release_native_session(session_id)
            self._coordinator.rebase_cache_position(
                session_id, int(len(prefix_token_ids))
            )
            self._prefix_importers[session_id] = prefix_import
            self._unified_prefix_revisions[session_id] = int(new_revision)
            self._prefix_revisions[session_id] = int(new_revision)
            self._prefix_fallback_reasons[session_id] = None
            self._prefix_lengths[session_id] = int(len(prefix_token_ids))
            self._prefix_replay_frame_ids[session_id] = ()
        except BaseException as exc:
            # A native slot may already have been released. Do not let the
            # previous prefix be used after a partially applied rebase.
            self._fatal_error = RuntimeError(
                f"unified prefix refresh failed after native retirement: {type(exc).__name__}: {exc}"
            )
            raise self._fatal_error from exc

    def _refresh_session_prefix(
        self,
        session_id: str,
        expected_frame_id: int,
        expected_state_version: int,
        cache_position: int,
        prefix_import: PrefixImportFactory,
    ) -> None:
        if not callable(prefix_import):
            raise TypeError("semantic prefix importer must be callable")
        if self._outstanding_frames.get(session_id, 0) != 0:
            raise RuntimeError("semantic prefix refresh requires an idle session")
        state = self._session_states.get(session_id)
        if state is None:
            raise KeyError(f"unknown listener session: {session_id}")
        if (
            state.next_frame_id != expected_frame_id
            or state.state_version != expected_state_version
        ):
            raise RuntimeError("semantic prefix refresh lost its frame/state compare-and-swap")
        if self._tick_future is not None and not self._tick_future.done():
            raise RuntimeError("semantic prefix refresh cannot overlap a shared tick")
        if not 1 <= cache_position <= self._kv_budgets.new_cache_target_tokens:
            raise ValueError("canonical replacement exceeds its new cache target")
        try:
            self._release_q2_decision_cache(session_id)
            self._release_native_session(session_id)
            self._coordinator.rebase_cache_position(session_id, int(cache_position))
            self._prefix_importers[session_id] = prefix_import
            self._prefix_revisions[session_id] = None
            self._prefix_fallback_reasons[session_id] = None
            self._prefix_lengths[session_id] = 0
            self._prefix_replay_frame_ids[session_id] = ()
        except BaseException as exc:
            self._fatal_error = exc
            raise

    def _prime_initial_prefix(
        self,
        session_id: str,
        prepared: PreparedInitialPrefix,
    ) -> InitialPrefixAdmission:
        """CAS, rebase, and import a prebuilt prefix on the GPU owner thread."""
        state = self._session_states.get(session_id)
        if state is None:
            raise KeyError(f"unknown listener session: {session_id}")
        if prepared.session_id != session_id:
            raise ValueError("prepared semantic prefix belongs to a different session")
        if prepared.prefix.cache_position not in (None, prepared.prefix.prefix_len):
            raise ValueError("initial semantic admission cannot include prior-frame replay")
        prefix_position = (
            prepared.prefix.prefix_len
            if prepared.prefix.cache_position is None
            else prepared.prefix.cache_position
        )
        if not 1 <= prefix_position <= min(NATIVE_CONTROL6_PREFIX_MAX_TOKENS, self._kv_budgets.initial_prefix_max_tokens):
            raise ValueError(
                "initial semantic prefix position must be in [1,"
                f"{NATIVE_CONTROL6_PREFIX_MAX_TOKENS}]"
            )

        # The owner deliberately accepts only a pristine listener cursor.  A
        # frame arriving during preparation is left for the existing lazy
        # refresh path; no synthetic row or silence is ever admitted here.
        if (
            state.closed
            or state.poisoned_reason is not None
            or state.generation_id != prepared.listener_generation_id
            or state.state_epoch_id != prepared.state_epoch_id
            or state.next_frame_id != prepared.frame_id
            or state.state_version != prepared.state_version
            or state.sequence_position != prepared.sequence_position
            or state.cache_position != prepared.expected_cache_position
            or (state.next_frame_id, state.state_version, state.sequence_position) != (0, 0, 0)
            or self._outstanding_frames.get(session_id, 0) != 0
            or (self._tick_future is not None and not self._tick_future.done())
        ):
            return InitialPrefixAdmission(
                status="ineligible",
                reason="listener_state_changed_before_initial_prefix_import",
            )

        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("listener worker runtime is unavailable during initial prefix admission")
        importer = self._prefix_importers.get(session_id)
        if not callable(importer):
            raise RuntimeError("initial semantic admission has no registered prefix importer")

        try:
            if session_id in self._imported_sessions:
                # Repeated session.update before frame zero is a safe prefix
                # replacement: retire the old private row, then import the
                # newest frozen semantic revision without advancing cursors.
                self._release_q2_decision_cache(session_id)
                self._release_native_session(session_id)
            self._coordinator.rebase_cache_position(session_id, int(prefix_position))
            row = BatchRowDescriptor(
                session_id=state.session_id,
                session_generation=state.generation_id,
                state_epoch_id=state.state_epoch_id,
                frame_id=state.next_frame_id,
                listener_state_version=state.state_version,
                sequence_position=state.sequence_position,
                cache_position=state.cache_position,
                valid_history_tokens=state.valid_history_tokens,
                native_context_id=int(state.native_context_id or 0),
                # This descriptor identifies the cache-import transaction, not
                # an input frame. No ListenerFrame is fabricated or committed.
                input_frame_sha256=hashlib.sha256(
                    b"shared-listener-initial-semantic-prefix-import"
                ).hexdigest(),
                reference_pcm_sha256=None,
                frame_ready_at_ns=0,
                reference_frame_id=None,
            )
            with _listener_stream_context(runtime.listener_stream):
                if prepared.ready_event is not None:
                    wait_event = getattr(runtime.listener_stream, "wait_event", None)
                    if not callable(wait_event):
                        raise RuntimeError("listener stream cannot wait on prepared semantic-prefix event")
                    wait_event(prepared.ready_event)
                import_started_ns = time.perf_counter_ns()
                runtime.native.import_session(
                    row,
                    prepared.prefix.past_key_values,
                    prepared.prefix.prefix_attention,
                    prefix_len=prepared.prefix.prefix_len,
                    producer_stream=runtime.listener_stream,
                )
                import_ms = round(
                    max(0, time.perf_counter_ns() - import_started_ns) / 1_000_000.0,
                    3,
                )
            self._imported_sessions.add(session_id)
            self._last_native_rows[session_id] = row
            self._prefix_importers[session_id] = importer
            self._prefix_revisions[session_id] = prepared.semantic_context_revision
            state.semantic_context_revision = int(prepared.semantic_context_revision)
            prepared_history_revision = (
                prepared.applied_history_revision
                if prepared.applied_history_revision is not None
                else prepared.prefix.applied_history_revision
            )
            if prepared_history_revision is not None:
                state.applied_history_revision = prepared_history_revision
            self._prefix_fallback_reasons[session_id] = (
                prepared.prefix.semantic_replay_fallback_reason
            )
            self._prefix_lengths[session_id] = int(prepared.prefix.prefix_len)
            self._prefix_replay_frame_ids[session_id] = tuple(
                int(frame_id) for frame_id in prepared.prefix.replay_frame_ids
            )
            return InitialPrefixAdmission(
                status="primed",
                prefix_len=int(prepared.prefix.prefix_len),
                semantic_context_revision=prepared.semantic_context_revision,
                import_ms=import_ms,
            )
        except BaseException as exc:
            self._fatal_error = exc
            raise

    def _ensure_capacity_for_registration(self, requested_capacity: int) -> None:
        """Grow native session slots before exposing the new session state."""
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("listener worker runtime is unavailable during session registration")
        with _listener_stream_context(runtime.listener_stream):
            actual_capacity = runtime.native.ensure_capacity(requested_capacity)
        if (
            isinstance(actual_capacity, bool)
            or not isinstance(actual_capacity, int)
            or actual_capacity < requested_capacity
        ):
            raise RuntimeError(
                "native listener returned insufficient capacity: "
                f"requested={requested_capacity}, actual={actual_capacity!r}"
            )

    def _reset_session(
        self,
        session_id: str,
        state: ListenerSessionState,
        prefix_import: PrefixImportFactory,
    ) -> None:
        if self._outstanding_frames.get(session_id, 0) != 0:
            raise RuntimeError("reset requires an idle session with no accepted frames")
        if self._queued_audio_soft.get(session_id):
            raise RuntimeError("reset cannot discard queued user audio; drain or close the old session")
        old_state = self._session_states.get(session_id)
        if old_state is None:
            raise KeyError(f"unknown listener session: {session_id}")
        if state.session_id != session_id:
            raise ValueError("reset must preserve the session ID")
        if state.runtime_signature != self._coordinator.runtime_signature:
            raise ValueError("reset state runtime signature differs from the worker")
        if state.generation_id <= old_state.generation_id or state.state_epoch_id == old_state.state_epoch_id:
            raise ValueError("reset must advance generation and replace the state epoch")
        if session_id in self._pending_promotions:
            raise RuntimeError("reset cannot overtake an unfenced listener cache promotion")
        if state.native_context_id is None or not callable(prefix_import):
            raise ValueError("reset requires a native context and owner-thread prefix importer")
        self._closing_sessions.add(session_id)
        state.cache_epoch_id = 0
        state.semantic_context_revision = None
        state.applied_history_revision = None
        try:
            self._clear_session_maintenance(session_id)
            self._end_parent_ephemeral_decodes(session_id)
            self._release_q2_decision_cache(session_id)
            if self._shadow_worker is not None:
                self._shadow_worker.invalidate_session(session_id)
            self._release_native_session(session_id)
            with _listener_stream_context(self._runtime.listener_stream):
                self._runtime.frontend.close_session(session_id)
            retired_old_state = self._coordinator.close_session(session_id)
            retired_old_state.state.cache_epoch_id = 0
            retired_old_state.state.semantic_context_revision = None
            retired_old_state.state.applied_history_revision = None
            del self._session_states[session_id]
            del self._prefix_importers[session_id]
            self._text_append_committed_scopes.pop(session_id, None)
            self._prefix_revisions.pop(session_id, None)
            self._unified_prefix_revisions.pop(session_id, None)
            self._prefix_fallback_reasons.pop(session_id, None)
            self._prefix_lengths.pop(session_id, None)
            self._prefix_replay_frame_ids.pop(session_id, None)
            self._queued_audio_soft.pop(session_id, None)
            self._pre_roll_ring.pop(session_id, None)
            self._assistant_audio_active_onset.pop(session_id, None)
            self._policy_q2_soft_pending.pop(session_id, None)
            self._reserved_tail_tokens.pop(session_id, None)
            self._assistant_audio_queue_open.discard(session_id)
            self._assistant_audio_queue_first_frame_id.pop(session_id, None)
            self._assistant_audio_queue_frame_spans.pop(session_id, None)
            self._assistant_audio_queue_epoch.pop(session_id, None)
            self._assistant_audio_queue_close_epoch.pop(session_id, None)
            self._stable_q2_unusable_epochs.pop(session_id, None)
            self._assistant_audio_queue_response_id.pop(session_id, None)
            self._pending_q2_prefills.pop(session_id, None)
            del self._outstanding_frames[session_id]
            self._outstanding_logical_tokens.pop(session_id, None)
            self._recent_listener_q2.pop(session_id, None)
            self._compaction_requested.pop(session_id, None)
            self._compaction_status.pop(session_id, None)
            self._compaction_reason.pop(session_id, None)
            self._requested_history_revision.pop(session_id, None)
            self._compaction_attempt_history_revision.pop(session_id, None)
            self._register_session(state, prefix_import)
        except BaseException as exc:
            # A reset that fails after native retirement cannot safely resume
            # either generation on this runtime.
            self._fatal_error = exc
            raise

    def _close_session(self, session_id: str) -> RetiredSession:
        if self._outstanding_frames.get(session_id, 0) != 0:
            raise RuntimeError("close requires an idle session with no accepted frames")
        if session_id not in self._session_states:
            raise KeyError(f"unknown listener session: {session_id}")
        if session_id in self._pending_promotions:
            raise RuntimeError("close cannot overtake an unfenced listener cache promotion")
        self._closing_sessions.add(session_id)
        try:
            self._clear_session_maintenance(session_id)
            self._end_parent_ephemeral_decodes(session_id)
            self._release_q2_decision_cache(session_id)
            if self._shadow_worker is not None:
                self._shadow_worker.invalidate_session(session_id)
            self._release_native_session(session_id)
            with _listener_stream_context(self._runtime.listener_stream):
                self._runtime.frontend.close_session(session_id)
            retired = self._coordinator.close_session(session_id)
            retired.state.cache_epoch_id = 0
            retired.state.semantic_context_revision = None
            retired.state.applied_history_revision = None
            self._session_states.pop(session_id)
            self._kv_slot_roles.pop(session_id, None)
            self._closing_sessions.discard(session_id)
            self._prefix_importers.pop(session_id)
            self._text_append_committed_scopes.pop(session_id, None)
            self._prefix_revisions.pop(session_id, None)
            self._unified_prefix_revisions.pop(session_id, None)
            self._prefix_fallback_reasons.pop(session_id, None)
            self._prefix_lengths.pop(session_id, None)
            self._prefix_replay_frame_ids.pop(session_id, None)
            self._outstanding_frames.pop(session_id)
            self._queued_audio_soft.pop(session_id, None)
            self._pre_roll_ring.pop(session_id, None)
            self._assistant_audio_active_onset.pop(session_id, None)
            self._policy_q2_soft_pending.pop(session_id, None)
            self._reserved_tail_tokens.pop(session_id, None)
            self._assistant_audio_queue_open.discard(session_id)
            self._assistant_audio_queue_first_frame_id.pop(session_id, None)
            self._assistant_audio_queue_frame_spans.pop(session_id, None)
            self._assistant_audio_queue_epoch.pop(session_id, None)
            self._assistant_audio_queue_close_epoch.pop(session_id, None)
            self._stable_q2_unusable_epochs.pop(session_id, None)
            self._assistant_audio_queue_response_id.pop(session_id, None)
            self._pending_q2_prefills.pop(session_id, None)
            self._outstanding_logical_tokens.pop(session_id, None)
            self._recent_listener_q2.pop(session_id, None)
            self._compaction_requested.pop(session_id, None)
            self._compaction_status.pop(session_id, None)
            self._compaction_revision.pop(session_id, None)
            self._compaction_reason.pop(session_id, None)
            self._requested_history_revision.pop(session_id, None)
            self._compaction_attempt_history_revision.pop(session_id, None)
            self._last_native_rows.pop(session_id, None)
            self._pending_promotions.pop(session_id, None)
            return retired
        except BaseException as exc:
            self._fatal_error = exc
            raise

    def _release_native_session(self, session_id: str) -> None:
        if session_id not in self._imported_sessions:
            return
        assert self._runtime is not None
        state = self._session_states[session_id]
        previous = self._last_native_rows[session_id]
        # The native release ABI consumes cursor metadata. The hashes and ready
        # timestamp remain historical provenance from the last real frame and
        # are not treated as a new frame or passed through the coordinator.
        current = replace(
            previous,
            session_generation=state.generation_id,
            state_epoch_id=state.state_epoch_id,
            frame_id=state.next_frame_id,
            listener_state_version=state.state_version,
            sequence_position=state.sequence_position,
            cache_position=state.cache_position,
            valid_history_tokens=state.valid_history_tokens,
            native_context_id=state.native_context_id,
        )
        try:
            self._runtime.native.release_session_async(current)
        except BaseException as exc:
            self._fatal_error = exc
            raise
        self._imported_sessions.remove(session_id)
        self._last_native_rows.pop(session_id, None)

    @staticmethod
    def _invoke_prefix_importer(
        importer: PrefixImportFactory,
        state: ListenerSessionState,
        row: BatchRowDescriptor,
        model_lock: Any | None,
    ) -> tuple[PrefixCacheImport, float | None, float | None]:
        """Build a prefix under the shared LM lock only when the importer requires it.

        Importers that run ShapeStable against shared Gemma weights opt in with
        ``requires_shared_lm_lock = True``. Cached-prefix importers leave it
        false, so their private native cache copy does not contend with thinker
        work. Native import itself always happens after this method releases the
        shared model lock.
        """
        requires_lock = getattr(importer, "requires_shared_lm_lock", False)
        if type(requires_lock) is not bool:
            raise TypeError("prefix importer requires_shared_lm_lock must be a bool")
        if not requires_lock:
            return importer(state, row), None, None
        if model_lock is None:
            raise RuntimeError(
                "prefix importer requires the shared Gemma LM lock, but none is configured"
            )

        wait_started_ns = time.perf_counter_ns()
        acquire_fair = getattr(model_lock, "acquire_fair", None)
        if callable(acquire_fair):
            acquired = bool(acquire_fair("listener"))
            if not acquired:
                raise RuntimeError("semantic prefix model-lock admission was cancelled")
        else:
            model_lock.acquire()
        acquired_ns = time.perf_counter_ns()
        wait_ms = round(max(0, acquired_ns - wait_started_ns) / 1_000_000.0, 3)
        try:
            prefix = importer(state, row)
        finally:
            hold_ms = round(
                max(0, time.perf_counter_ns() - acquired_ns) / 1_000_000.0,
                3,
            )
            model_lock.release()
        return prefix, wait_ms, hold_ms

    def _process_batch(self, batch: ListenerBatch) -> None:
        worker_started_ns = time.perf_counter_ns()
        submitted_ns = self._tick_submitted_ns
        worker_queue_wait_ms = (
            None
            if submitted_ns is None
            else round(max(0, worker_started_ns - submitted_ns) / 1_000_000.0, 3)
        )
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("worker runtime is unavailable")
        wait_event = getattr(runtime.listener_stream, "wait_event", None)
        if self._tick_cuda_dependencies and not callable(wait_event):
            raise RuntimeError("listener stream cannot order admitted compute completion events")
        for event in self._tick_cuda_dependencies:
            wait_event(event)
        normalized_rows, _stale_queue_sessions = _normalize_stale_assistant_queue_rows(
            batch.manifest.rows,
            queue_open_sessions=self._assistant_audio_queue_open,
            valid_sessions=set(self._session_states),
        )
        if normalized_rows != batch.manifest.rows:
            batch = replace(
                batch,
                manifest=replace(batch.manifest, rows=normalized_rows),
            )
        profile_frames = (
            {
                int(row.frame_id) for row in batch.manifest.rows
                if int(row.frame_id) in self._torch_profile_frame_ids
            }
            if self._torch_profile_trace_path and not self._torch_profile_failed
            else set()
        )
        if profile_frames and self._torch_profile_trace_path and not self._torch_profile_failed:
            self._torch_profiled_frame_ids.update(profile_frames)
            if self._torch_profiler is None:
                try:
                    profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=False,
                    )
                    profiler.start()
                    self._torch_profiler = profiler
                    logging.getLogger(__name__).info(
                        "listener torch profiler started: frames=%s trace=%s",
                        sorted(self._torch_profile_frame_ids),
                        self._torch_profile_trace_path,
                    )
                except Exception as exc:
                    self._torch_profile_failed = True
                    logging.getLogger(__name__).warning(
                        "listener torch profiler unavailable: %s: %s",
                        type(exc).__name__, exc,
                    )
        self._q2_cache_disabled_for_tick.clear()
        states = tuple(self._session_states[row.session_id] for row in batch.manifest.rows)
        trace_enabled = _unified_kv_integrity_trace_enabled()
        speaking_native_cursor_before: dict[str, int] = {}
        speaking_native_snapshot_errors: dict[str, str] = {}
        if trace_enabled:
            snapshot_native_state = getattr(runtime.native, "session_state", None)
            if callable(snapshot_native_state):
                for row in batch.manifest.rows:
                    if not row.queue_audio_for_next_turn:
                        continue
                    try:
                        native_state = snapshot_native_state(row)
                        speaking_native_cursor_before[row.session_id] = int(
                            native_state.current_position
                        )
                    except Exception as trace_error:
                        speaking_native_snapshot_errors[row.session_id] = type(
                            trace_error
                        ).__name__
        for row, state in zip(batch.manifest.rows, states, strict=True):
            self._trace_control6_queue_handoff(
                row.session_id,
                "frame_process",
                frame_id=int(row.frame_id),
                logical_q_len=int(row.logical_q_len),
                queue_audio_for_next_turn=bool(row.queue_audio_for_next_turn),
                queue_open_at_process=row.session_id in self._assistant_audio_queue_open,
                cache_position=int(row.cache_position),
                valid_history=int(row.valid_history_tokens),
                next_frame_id=int(state.next_frame_id),
            )
        prepared = None
        frontend_commit_attempted = False
        listener_model_lock_wait_ms: float | None = None
        listener_model_lock_hold_ms: float | None = None
        capacity_growth_ms: float | None = None
        frontend_encode_ms: float | None = None
        frontend_copy_ms: float | None = None
        frontend_copy_cuda_ms: float | None = None
        audio_tower_lock_wait_ms: float | None = None
        audio_tower_lock_hold_ms: float | None = None
        audio_tower_workspace_lock_wait_ms: float | None = None
        audio_tower_workspace_lock_hold_ms: float | None = None
        audio_tower_row_copy_cuda_ms: float | None = None
        audio_tower_graph_submit_ms: float | None = None
        audio_tower_cuda_ms: float | None = None
        audio_tower_batch_cuda_span_ms: float | None = None
        audio_tower_execution_mode: str | None = None
        native_submit_ms: float | None = None
        native_cuda_ms: float | None = None
        native_control6_raw_score_ms: float | None = None
        control6_policy_ms: float | None = None
        cuda_completion_wait_ms: float | None = None
        native_control6_raw_scores: tuple[Any, ...] | None = None
        # Prefix work is recorded on the batch that actually performs an
        # import.  Session metadata such as revision and prefix length persists,
        # but build/replay/import durations are deliberately batch-local.
        prefix_build_ms_by_session: dict[str, float | None] = {}
        prefix_replay_ms_by_session: dict[str, float | None] = {}
        prefix_import_ms_by_session: dict[str, float] = {}
        prefix_refresh_event_ids_by_session: dict[str, str] = {}
        prefix_refresh_revisions_by_session: dict[str, int | None] = {}
        batch_promotions: dict[str, _PendingShadowPromotion] = {}
        policy_q2_to_commit: list[
            tuple[BatchRowDescriptor, QueuedAudioSoftFrame, np.ndarray]
        ] = []
        canonical_q2_to_commit: list[CanonicalListenerQ2Frame] = []
        decision_views: dict[str, TemporaryQ2DecisionView] = {}
        decision_view_unavailable_reasons: dict[str, str] = {}
        policy_inputs: tuple[PolicyInput | PolicyPreHeadContext, ...] | None = None
        for row in batch.manifest.rows:
            pending_promotion = self._pending_promotions.get(row.session_id)
            if pending_promotion is None:
                continue
            promoted_row = pending_promotion.copy_result.destination_row
            if (
                promoted_row.session_id != row.session_id
                or promoted_row.session_generation != row.session_generation
                or promoted_row.state_epoch_id != row.state_epoch_id
                or promoted_row.frame_id != row.frame_id
                or promoted_row.listener_state_version != row.listener_state_version
                or promoted_row.sequence_position != row.sequence_position
                or promoted_row.cache_position != row.cache_position
            ):
                raise RuntimeError("promoted semantic cache does not match its next manifest row")
            batch_promotions[row.session_id] = pending_promotion
            prefix = pending_promotion.candidate.prefix
            prefix_build_ms_by_session[row.session_id] = prefix.prefix_build_ms
            prefix_replay_ms_by_session[row.session_id] = prefix.replay_ms
            prefix_import_ms_by_session[row.session_id] = pending_promotion.copy_enqueue_ms
            prefix_refresh_event_ids_by_session[row.session_id] = (
                f"shadow:{pending_promotion.candidate.candidate_id}:{row.session_id}:"
                f"{row.state_epoch_id}:{row.frame_id}:"
                f"{pending_promotion.candidate.stage.request.semantic_context_revision}"
            )
            prefix_refresh_revisions_by_session[row.session_id] = int(
                pending_promotion.candidate.stage.request.semantic_context_revision
            )
        model_lock = runtime.listener_model_lock
        try:
            with _listener_stream_context(runtime.listener_stream):
                capacity_started_ns = time.perf_counter_ns()
                q2_session_ids = {
                    row.session_id for row in batch.manifest.rows
                    if (
                        row.logical_q_len == 0
                        and row.ephemeral_reason == "assistant_turn_open"
                        and row.session_id in self._assistant_audio_queue_open
                    )
                }
                metadata_rebind_available = (
                    getattr(runtime.native, "_metadata_rebind_api_available", False) is True
                    and callable(getattr(
                        runtime.native, "rebind_ephemeral_session_metadata", None,
                    ))
                )
                new_q2_slots = (
                    sum(session_id not in self._q2_decision_caches for session_id in q2_session_ids)
                    if metadata_rebind_available
                    else int(bool(q2_session_ids))
                )
                requested_capacity = (
                    len(self._session_states)
                    + len(self._ephemeral_decode_branches)
                    + len(self._q2_decision_caches)
                    + new_q2_slots
                )
                actual_capacity = runtime.native.ensure_capacity(
                    requested_capacity
                )
                capacity_growth_ms = round(
                    max(0, time.perf_counter_ns() - capacity_started_ns) / 1_000_000.0,
                    3,
                )
                if (
                    isinstance(actual_capacity, bool)
                    or not isinstance(actual_capacity, int)
                    or actual_capacity < requested_capacity
                ):
                    raise RuntimeError(
                        "native listener returned insufficient capacity at tick boundary: "
                        f"requested={requested_capacity}, actual={actual_capacity!r}"
                    )
                for row, state in zip(batch.manifest.rows, states, strict=True):
                    if row.session_id in self._imported_sessions:
                        continue
                    is_semantic_refresh = self._prefix_lengths.get(row.session_id) == 0
                    prefix, prefix_lock_wait_ms, prefix_lock_hold_ms = (
                        self._invoke_prefix_importer(
                            self._prefix_importers[row.session_id],
                            state,
                            row,
                            model_lock,
                        )
                    )
                    if prefix_lock_wait_ms is not None:
                        listener_model_lock_wait_ms = round(
                            (listener_model_lock_wait_ms or 0.0) + prefix_lock_wait_ms,
                            3,
                        )
                    if prefix_lock_hold_ms is not None:
                        listener_model_lock_hold_ms = round(
                            (listener_model_lock_hold_ms or 0.0) + prefix_lock_hold_ms,
                            3,
                        )
                    if not isinstance(prefix, PrefixCacheImport):
                        raise TypeError("prefix importer must return PrefixCacheImport")
                    if prefix.prefix_len > self._kv_budgets.initial_prefix_max_tokens:
                        raise ValueError("initial imported prefix exceeds its service token budget")
                    if row.cache_position > self._operational_max_seq_len():
                        raise ValueError("imported cache exceeds its operational token budget")
                    expected_position = row.cache_position
                    if prefix.cache_position is not None and prefix.cache_position != expected_position:
                        raise RuntimeError("prefix cache cursor differs from its current manifest row")
                    cache_length = getattr(prefix.past_key_values, "get_seq_length", None)
                    if callable(cache_length) and int(cache_length(0)) != expected_position:
                        raise RuntimeError("prefix cache DynamicCache length differs from the row cursor")
                    import_started_ns = time.perf_counter_ns()
                    runtime.native.import_session(
                        row,
                        prefix.past_key_values,
                        prefix.prefix_attention,
                        prefix_len=prefix.prefix_len,
                        producer_stream=runtime.listener_stream,
                    )
                    prefix_import_ms = round(
                        max(0, time.perf_counter_ns() - import_started_ns) / 1_000_000.0,
                        3,
                    )
                    self._imported_sessions.add(row.session_id)
                    self._last_native_rows[row.session_id] = row
                    self._prefix_revisions[row.session_id] = (
                        prefix.semantic_context_revision
                        if prefix.semantic_context_revision is not None
                        else self._unified_prefix_revisions.get(row.session_id)
                    )
                    state.semantic_context_revision = self._prefix_revisions[row.session_id]
                    if prefix.applied_history_revision is not None:
                        state.applied_history_revision = prefix.applied_history_revision
                    self._prefix_fallback_reasons[row.session_id] = prefix.semantic_replay_fallback_reason
                    self._prefix_lengths[row.session_id] = int(prefix.prefix_len)
                    self._prefix_replay_frame_ids[row.session_id] = tuple(
                        int(frame_id) for frame_id in prefix.replay_frame_ids
                    )
                    prefix_build_ms_by_session[row.session_id] = prefix.prefix_build_ms
                    prefix_replay_ms_by_session[row.session_id] = prefix.replay_ms
                    prefix_import_ms_by_session[row.session_id] = prefix_import_ms
                    if is_semantic_refresh:
                        prefix_refresh_event_ids_by_session[row.session_id] = (
                            f"{batch.manifest.manifest_id}:{row.session_id}:"
                            f"{row.state_epoch_id}:{row.frame_id}:"
                            f"{prefix.semantic_context_revision}"
                        )
                        prefix_refresh_revisions_by_session[row.session_id] = (
                            prefix.semantic_context_revision
                        )

            with _listener_stream_context(runtime.listener_stream):
                frontend_started_ns = time.perf_counter_ns()
                if self._torch_profiler is not None:
                    with torch.profiler.record_function("listener.frontend_encode_batch"):
                        prepared = runtime.frontend.encode_batch(batch, states)
                else:
                    prepared = runtime.frontend.encode_batch(batch, states)
                frontend_encode_ms = round(
                    max(0, time.perf_counter_ns() - frontend_started_ns) / 1_000_000.0,
                    3,
                )
                frontend_copy_ms = prepared.frontend_copy_ms
                audio_tower_lock_wait_ms = prepared.audio_tower_lock_wait_ms
                audio_tower_lock_hold_ms = prepared.audio_tower_lock_hold_ms
                audio_tower_workspace_lock_wait_ms = prepared.audio_tower_workspace_lock_wait_ms
                audio_tower_workspace_lock_hold_ms = prepared.audio_tower_workspace_lock_hold_ms
                audio_tower_graph_submit_ms = prepared.audio_tower_graph_submit_ms
                audio_tower_execution_mode = prepared.audio_tower_execution_mode
                if prepared.manifest_id != batch.manifest.manifest_id:
                    raise RuntimeError("frontend output belongs to a different manifest")
                audio_soft = torch.stack(prepared.audio_soft_tokens, dim=0).contiguous()
                expected_shape = (batch.manifest.batch_size, LISTENER_Q_PER_FRAME, CONTROL_HIDDEN_WIDTH)
                if tuple(audio_soft.shape) != expected_shape or audio_soft.dtype != torch.bfloat16:
                    raise RuntimeError(
                        "frontend must return exact BF16 [N,2,1536] real listener tokens"
                    )
                if prepared.audio_stream_cursors:
                    validate_frontend = getattr(
                        runtime.native, "validate_audio_frontend_metadata", None
                    )
                    if not callable(validate_frontend):
                        raise RuntimeError(
                            "native listener adapter lacks incremental audio cursor validation"
                        )
                    validate_frontend(batch.manifest, prepared)
                for row in batch.manifest.rows:
                    if row.logical_q_len != 0 or row.session_id not in self._assistant_audio_queue_open:
                        continue
                    if row.logical_q_len != 0 or row.ephemeral_reason != "assistant_turn_open":
                        raise RuntimeError("only assistant-phase Q0 rows may belong to an open queue")
                    pending = self._queued_audio_soft.get(row.session_id)
                    if pending is None:
                        raise RuntimeError("queued audio has no registered per-session owner buffer")
                    if row.queue_audio_for_next_turn:
                        reserved_tail = self._reserved_tail_tokens.get(row.session_id, 0)
                        active_drain = self._active_queued_audio_drains.get(row.session_id)
                        active_onset = self._assistant_audio_active_onset.get(row.session_id)
                        requested_onset = row.queue_audio_onset_frame_id
                        starts_onset = active_onset is None or (
                            requested_onset is not None
                            and int(requested_onset) != int(active_onset)
                        )
                        pre_roll_additions = 0
                        if starts_onset:
                            queued_ids = {
                                int(item.source_frame_id) for item in pending
                            }
                            spans = self._assistant_audio_queue_frame_spans.get(
                                row.session_id, (),
                            )
                            queue_epoch = row.queue_epoch
                            first_frame_id = self._assistant_audio_queue_first_frame_id.get(
                                row.session_id, row.frame_id,
                            )
                            pre_roll_additions = sum(
                                1
                                for item in self._pre_roll_ring.get(row.session_id, ())
                                if (
                                    item.session_id == row.session_id
                                    and item.session_generation == row.session_generation
                                    and item.state_epoch_id == row.state_epoch_id
                                    and item.source_queue_epoch == queue_epoch
                                    and first_frame_id <= item.source_frame_id < row.frame_id
                                    and item.source_frame_id not in queued_ids
                                    and not self._assistant_queue_span_contains_retained_frame(
                                        spans, int(item.source_frame_id),
                                    )
                                )
                            )
                        draining_tokens = (
                            0
                            if active_drain is None
                            else 2 * (len(active_drain.queued) - active_drain.index)
                        )
                        required_end = (
                            row.cache_position
                            + draining_tokens
                            + 2 * (len(pending) + pre_roll_additions + 1)
                            + reserved_tail
                        )
                        if required_end > int(self._operational_max_seq_len()):
                            raise RuntimeError(
                                "queued user audio and reserved model-turn tail exceed native KV capacity; "
                                "no frame was dropped "
                                f"(session={row.session_id}, cursor={row.cache_position}, "
                                f"queued_frames={len(pending) + pre_roll_additions + 1}, "
                                f"sealed_drain_frames={draining_tokens // 2}, "
                                f"reserved_tail_tokens={reserved_tail}, "
                                f"max_seq_len={self._operational_max_seq_len()})"
                            )
                        if pending and row.frame_id <= pending[-1].source_frame_id:
                            raise RuntimeError("queued assistant-phase audio frames are not strictly ordered")
                if (
                    runtime.policy is not None
                    and runtime.policy_inputs is not None
                    and any(row.session_id in self._assistant_audio_queue_open for row in batch.manifest.rows)
                ):
                    policy_inputs = tuple(runtime.policy_inputs(batch, states))
                    if len(policy_inputs) != batch.manifest.batch_size:
                        raise RuntimeError("batched Control6 policy inputs returned an incomplete row set")
                for index, row in enumerate(batch.manifest.rows):
                    if row.logical_q_len != 0 or row.session_id not in self._assistant_audio_queue_open:
                        continue
                    try:
                        if self._torch_profiler is not None:
                            with torch.profiler.record_function("listener.q2_decision_view"):
                                decision_views[row.session_id] = self._build_speaking_q2_decision_view(
                                    row,
                                    prepared.audio_soft_tokens[index],
                                )
                        else:
                            decision_views[row.session_id] = self._build_speaking_q2_decision_view(
                                row,
                                prepared.audio_soft_tokens[index],
                            )
                    except _Q2DecisionViewUnavailable as exc:
                        decision_view_unavailable_reasons[row.session_id] = str(exc)
                    except Exception as exc:
                        self._release_q2_decision_cache(row.session_id)
                        self._mark_stable_q2_epoch_unusable(
                            row.session_id,
                            queue_epoch=self._assistant_audio_queue_epoch.get(row.session_id),
                        )
                        decision_view_unavailable_reasons[row.session_id] = (
                            "temporary Q2 decision view failed closed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                self._q2_cache_disabled_for_tick.clear()
                native_cuda_events = _timed_cuda_event_pair(runtime.listener_stream)
                if native_cuda_events is not None:
                    native_cuda_events[0].record(runtime.listener_stream)
                native_started_ns = time.perf_counter_ns()
                if self._torch_profiler is not None:
                    with torch.profiler.record_function("listener.native_step_async"):
                        native_output = runtime.native.step_async(
                            batch.manifest,
                            audio_soft,
                            producer_stream=runtime.listener_stream,
                        )
                else:
                    native_output = runtime.native.step_async(
                        batch.manifest,
                        audio_soft,
                        producer_stream=runtime.listener_stream,
                    )
                if native_cuda_events is not None:
                    native_cuda_events[1].record(runtime.listener_stream)
                native_submit_ms = round(
                    max(0, time.perf_counter_ns() - native_started_ns) / 1_000_000.0,
                    3,
                )
                if not isinstance(native_output, NativeBatchOutput):
                    raise TypeError("native batch step must return NativeBatchOutput")
                verify_output_health = getattr(runtime.native, "verify_output_health", None)
                if not callable(verify_output_health):
                    raise RuntimeError("native batch adapter lacks row-bound output health")
                health_started_ns = time.perf_counter_ns()
                native_output_health = tuple(verify_output_health(native_output))
                native_output_health_ms = round(
                    max(0, time.perf_counter_ns() - health_started_ns) / 1_000_000.0,
                    3,
                )
                native_cuda_ms = _elapsed_timed_cuda_events(native_cuda_events)
                if len(native_output_health) != batch.manifest.batch_size:
                    raise RuntimeError("native batch output health returned an incomplete row set")
                for row, health in zip(batch.manifest.rows, native_output_health, strict=True):
                    if (
                        not isinstance(health, NativeBatchOutputHealth)
                        or health.row != row
                        or health.generation != native_output.generation
                        or health.checked is not True
                        or type(health.nonfinite) is not bool
                        or health.status not in {"healthy", "nonfinite_guard"}
                        or health.nonfinite != (health.status == "nonfinite_guard")
                    ):
                        raise RuntimeError("native output health provenance does not match its row")
                if any(health.nonfinite for health in native_output_health):
                    raise RuntimeError(
                        "native batch finite guard rejected the output; all rows quarantined "
                        f"(generation={native_output.generation})"
                    )
                if runtime.control6_scorer is not None:
                    score_started_ns = time.perf_counter_ns()
                    native_control6_raw_scores = tuple(
                        runtime.control6_scorer.score_raw_batch(
                            native_output,
                            batch.manifest,
                        )
                    )
                    native_control6_raw_score_ms = round(
                        max(0, time.perf_counter_ns() - score_started_ns) / 1_000_000.0,
                        3,
                    )
                    if len(native_control6_raw_scores) != batch.manifest.batch_size:
                        raise RuntimeError("raw Control6 scoring returned an incomplete row set")
                    for row, snapshot in zip(
                        batch.manifest.rows, native_control6_raw_scores, strict=True
                    ):
                        if (
                            snapshot.manifest_id != batch.manifest.manifest_id
                            or snapshot.output_generation != native_output.generation
                            or snapshot.session_id != row.session_id
                            or snapshot.session_generation != row.session_generation
                            or snapshot.state_epoch_id != row.state_epoch_id
                            or snapshot.listener_state_version != row.listener_state_version
                            or snapshot.frame_id != row.frame_id
                            or snapshot.sequence_position != row.sequence_position
                            or snapshot.cache_position != row.cache_position
                        ):
                            raise RuntimeError(
                                "raw Control6 score snapshot provenance differs from its manifest row"
                            )
                decisions: tuple[PolicyDecision, ...] | None = None
                if runtime.policy is not None:
                    if runtime.policy_inputs is None:
                        raise RuntimeError("batched Control6 requires policy_inputs")
                    mixed_source_policy = getattr(runtime.policy, "step_native_output_views", None)
                    if not callable(mixed_source_policy):
                        raise RuntimeError("batched Control6 lacks single-pass mixed-source policy evaluation")
                    if policy_inputs is None:
                        policy_inputs = tuple(runtime.policy_inputs(batch, states))
                    policy_started_ns = time.perf_counter_ns()
                    unavailable_indexes = {
                        index
                        for index, row in enumerate(batch.manifest.rows)
                        if self._stable_turn_policy_q2_context
                        and row.session_id in decision_view_unavailable_reasons
                    }
                    valid_indexes = tuple(
                        index
                        for index in range(batch.manifest.batch_size)
                        if index not in unavailable_indexes
                    )
                    decisions_by_index: list[PolicyDecision | None] = [
                        None
                    ] * batch.manifest.batch_size
                    if valid_indexes:
                        output_views = tuple(
                            DecisionRowSource(
                                native_output=(view.native_output if view is not None else native_output),
                                manifest=(view.manifest if view is not None else batch.manifest),
                                row_index=(0 if view is not None else index),
                                decision_row=row,
                            )
                            for index in valid_indexes
                            for row in (batch.manifest.rows[index],)
                            for view in (decision_views.get(row.session_id),)
                        )
                        valid_policy_inputs = tuple(policy_inputs[index] for index in valid_indexes)
                        policy_call_wall_start_ns = time.perf_counter_ns()
                        policy_call_thread_cpu_start_ns = time.thread_time_ns()
                        valid_decisions = tuple(
                            mixed_source_policy(output_views, valid_policy_inputs)
                        )
                        policy_call_wall_end_ns = time.perf_counter_ns()
                        policy_call_thread_cpu_end_ns = time.thread_time_ns()
                        if len(valid_decisions) != len(valid_indexes):
                            raise RuntimeError("Control6 returned an incomplete row set")
                        policy_diagnostics = {
                            "listener_batch_control6_policy_wall_start_ns": policy_call_wall_start_ns,
                            "listener_batch_control6_policy_wall_end_ns": policy_call_wall_end_ns,
                            "listener_batch_control6_policy_thread_cpu_start_ns": policy_call_thread_cpu_start_ns,
                            "listener_batch_control6_policy_thread_cpu_end_ns": policy_call_thread_cpu_end_ns,
                        }
                        if self._gc_pause_snapshot_provider is not None:
                            try:
                                gc_pause_ms, gc_pause_overlapped = gc_pause_overlap(
                                    self._gc_pause_snapshot_provider(),
                                    policy_call_wall_start_ns,
                                    policy_call_wall_end_ns,
                                )
                            except Exception:
                                gc_pause_ms, gc_pause_overlapped = None, None
                            policy_diagnostics.update({
                                "gc_pause_ms": gc_pause_ms,
                                "gc_pause_overlapped": gc_pause_overlapped,
                            })
                        valid_decisions = tuple(
                            replace(
                                decision,
                                diagnostics={
                                    **dict(decision.diagnostics or {}),
                                    **policy_diagnostics,
                                },
                            )
                            for decision in valid_decisions
                        )
                        for index, decision in zip(valid_indexes, valid_decisions, strict=True):
                            decisions_by_index[index] = decision
                    for index, row in enumerate(batch.manifest.rows):
                        reason = decision_view_unavailable_reasons.get(row.session_id)
                        if reason is not None:
                            if self._stable_turn_policy_q2_context:
                                context = policy_inputs[index]
                                policy_state = (
                                    context.policy_input
                                    if isinstance(context, PolicyPreHeadContext)
                                    else context
                                )
                                if not isinstance(policy_state, PolicyInput):
                                    raise TypeError("stable Q2 fallback lacks its canonical policy state")
                                expected_turn_generation = (
                                    context.tick_id[1]
                                    if isinstance(context, PolicyPreHeadContext)
                                    else row.session_generation
                                )
                                decisions_by_index[index] = PolicyDecision(
                                    action=ControlAction.HARD_STOP,
                                    frame_id=int(row.frame_id),
                                    reason="stable_q2_context_unavailable_fail_closed",
                                    generation_id=int(policy_state.generation_id),
                                    listener_frame_id=int(row.frame_id),
                                    session=row.session_id,
                                    source="control6_context_fail_closed",
                                    diagnostics={
                                        "decision_view_unavailable_reason": reason,
                                        "policy_head_skipped": True,
                                        "system_fail_closed_action": True,
                                        "native_output_provenance": {
                                            "manifest_id": batch.manifest.manifest_id,
                                            "output_generation": native_output.generation,
                                            "native_session_id": row.session_id,
                                            "canonical_session_id": row.session_id,
                                            "listener_generation_id": row.session_generation,
                                            "turn_generation_id": expected_turn_generation,
                                            "state_epoch_id": row.state_epoch_id,
                                            "listener_state_version": row.listener_state_version,
                                            "frame_id": row.frame_id,
                                            "sequence_position": row.sequence_position,
                                            "cache_position": row.cache_position,
                                            "used_for_policy": False,
                                        },
                                    },
                                )
                                continue
                            decision = decisions_by_index[index]
                            if decision is None:
                                raise RuntimeError("Control6 omitted a decision for a canonical row")
                            diagnostics = dict(decision.diagnostics or {})
                            diagnostics["decision_view_fallback_reason"] = reason
                            decisions_by_index[index] = replace(
                                decision, diagnostics=diagnostics,
                            )
                    if any(decision is None for decision in decisions_by_index):
                        raise RuntimeError("Control6 returned an incomplete row set")
                    decisions = tuple(
                        decision for decision in decisions_by_index if decision is not None
                    )
                    control6_policy_ms = round(
                        max(0, time.perf_counter_ns() - policy_started_ns) / 1_000_000.0,
                        3,
                    )
                    for row, context, decision in zip(
                        batch.manifest.rows, policy_inputs, decisions, strict=True
                    ):
                        expected_turn_generation = (
                            context.tick_id[1]
                            if isinstance(context, PolicyPreHeadContext)
                            else row.session_generation
                        )
                        if (
                            decision.session != row.session_id
                            or decision.generation_id != expected_turn_generation
                            or decision.frame_id != row.frame_id
                            or decision.listener_frame_id != row.frame_id
                        ):
                            raise RuntimeError(
                                "Control6 decision identity does not match its native row: "
                                f"session={decision.session!r}/{row.session_id!r}, "
                                f"generation={decision.generation_id}/{expected_turn_generation}, "
                                f"frame={decision.frame_id}/{row.frame_id}, "
                                f"listener_frame={decision.listener_frame_id}/{row.frame_id}"
                            )
                completion_started_ns = time.perf_counter_ns()
                if self._torch_profiler is not None:
                    with torch.profiler.record_function("listener.cuda_completion_wait"):
                        self._wait_batch_completion(runtime)
                else:
                    self._wait_batch_completion(runtime)
                self._tick_cuda_dependencies = ()
                for index, row in enumerate(batch.manifest.rows):
                    if row.logical_q_len != 0 or row.session_id not in self._assistant_audio_queue_open:
                        continue
                    soft = prepared.audio_soft_tokens[index]
                    if soft.is_cuda:
                        soft.record_stream(runtime.listener_stream)
                    owned_soft = soft.detach().clone().contiguous()
                    if owned_soft.is_cuda:
                        owned_soft.record_stream(runtime.listener_stream)
                    policy_q2_to_commit.append((
                        row,
                        QueuedAudioSoftFrame(
                            session_id=row.session_id,
                            session_generation=row.session_generation,
                            state_epoch_id=row.state_epoch_id,
                            source_frame_id=row.frame_id,
                            source_state_version=row.listener_state_version,
                            source_sequence_position=row.sequence_position,
                            audio_soft_tokens=owned_soft,
                            source_queue_epoch=batch.frames[index].queue_epoch,
                        ),
                        batch.frames[index].pcm,
                    ))
                cuda_completion_wait_ms = round(
                    max(0, time.perf_counter_ns() - completion_started_ns) / 1_000_000.0,
                    3,
                )
                if prepared is not None:
                    (
                        frontend_copy_cuda_ms,
                        audio_tower_row_copy_cuda_ms,
                        audio_tower_cuda_ms,
                    ) = (
                        prepared.cuda_phase_timings_after_completion()
                    )
                    audio_tower_batch_cuda_span_ms = (
                        prepared.audio_tower_batch_cuda_span_ms_after_completion()
                    )

                shadow_q2_rows: list[ShadowQ2Frame] = []
                shadow = self._shadow_worker
                for frame, row, audio_soft_row in zip(
                    batch.frames,
                    batch.manifest.rows,
                    prepared.audio_soft_tokens,
                    strict=True,
                ):
                    if row.logical_q_len != LISTENER_Q_PER_FRAME or row.ephemeral_reason is not None:
                        continue
                    if audio_soft_row.is_cuda:
                        audio_soft_row.record_stream(runtime.listener_stream)
                    canonical_q2_to_commit.append(CanonicalListenerQ2Frame(
                        frame=frame,
                        row=row,
                        audio_soft_tokens=audio_soft_row.detach().clone().contiguous(),
                    ))
                if shadow is not None:
                    for frame, row, audio_soft_row in zip(
                        batch.frames,
                        batch.manifest.rows,
                        prepared.audio_soft_tokens,
                        strict=True,
                    ):
                        if not shadow.needs_q2(row.session_id, row.frame_id):
                            continue
                        if audio_soft_row.is_cuda:
                            audio_soft_row.record_stream(runtime.listener_stream)
                        shadow_q2_rows.append(ShadowQ2Frame(
                            frame=frame,
                            row=row,
                            audio_soft=audio_soft_row.detach().clone().contiguous(),
                            ready_event=None,
                        ))
                    if shadow_q2_rows:
                        q2_ready_event = _record_shadow_event(runtime.listener_stream)
                        shadow_q2_rows = [
                            replace(item, ready_event=q2_ready_event)
                            for item in shadow_q2_rows
                        ]

            # Frontend buffers and coordinator cursors move only after every
            # downstream GPU consumer is complete. The single owner makes the
            # frontend commit and validated all-row cursor commit one serialized
            # transaction; no other thread may inspect or mutate live state.
            updates = tuple(
                RowStateUpdate(
                    session_id=row.session_id,
                    frame_id=row.frame_id,
                    cache_position=row.cache_position + row.logical_q_len,
                    valid_history_tokens=row.valid_history_tokens + row.logical_q_len,
                )
                for row in batch.manifest.rows
            )

            def commit_frontend_state() -> None:
                nonlocal frontend_commit_attempted
                frontend_commit_attempted = True
                prepared.commit()

            self._coordinator.complete(
                batch,
                updates,
                before_cursor_commit=commit_frontend_state,
            )
            for row, state in zip(batch.manifest.rows, states, strict=True):
                updated_row = replace(
                    row,
                    frame_id=state.next_frame_id,
                    listener_state_version=state.state_version,
                    sequence_position=state.sequence_position,
                    cache_position=state.cache_position,
                    valid_history_tokens=state.valid_history_tokens,
                )
                self._last_native_rows[row.session_id] = updated_row
                self._outstanding_frames[row.session_id] = max(
                    0, self._outstanding_frames.get(row.session_id, 0) - 1
                )
                self._outstanding_logical_tokens[row.session_id] = max(
                    0,
                    self._outstanding_logical_tokens.get(row.session_id, 0)
                    - int(row.logical_q_len),
                )
            # The primary stream's completion fence above covers any cache
            # copy ordered before this frame. Publish the new physical cache
            # identity before observers receive snapshots for the committed row.
            for session_id, pending in tuple(batch_promotions.items()):
                self._finalize_shadow_promotion(
                    session_id, int(pending.candidate.candidate_id)
                )
            for item in canonical_q2_to_commit:
                ring = self._recent_listener_q2.get(item.frame.session_id)
                if ring is None:
                    continue
                if ring and item.frame.frame_id <= ring[-1].frame.frame_id:
                    if item.frame.frame_id == ring[-1].frame.frame_id:
                        continue
                    raise RuntimeError("canonical listener Q2 history is not strictly ordered")
                ring.append(item)
                self._maybe_request_listener_compaction(item.frame.session_id)
            for row, policy_q2, source_pcm in policy_q2_to_commit:
                session_id = row.session_id
                queue_epoch = policy_q2.source_queue_epoch
                if row.queue_audio_for_next_turn:
                    retention_reason = row.queue_audio_reason or "utterance_body"
                    active_onset = self._assistant_audio_active_onset.get(session_id)
                    requested_onset = row.queue_audio_onset_frame_id
                    new_onset = active_onset is None or (
                        requested_onset is not None
                        and int(requested_onset) != int(active_onset)
                    )
                    if new_onset:
                        onset_frame_id = (
                            int(requested_onset)
                            if requested_onset is not None
                            else int(row.frame_id)
                        )
                        self._transfer_pre_roll_at_onset(
                            row,
                            onset_frame_id=onset_frame_id,
                            queue_epoch=queue_epoch,
                        )
                        active_onset = onset_frame_id
                    elif requested_onset is not None:
                        active_onset = int(requested_onset)
                    assert active_onset is not None
                    self._record_assistant_queue_frame(
                        row,
                        reason=retention_reason,
                        onset_frame_id=int(active_onset),
                        queue_epoch=queue_epoch,
                    )
                    queued = self._queued_audio_soft.setdefault(session_id, deque())
                    if any(item.source_frame_id == row.frame_id for item in queued):
                        raise RuntimeError("assistant queue attempted to retain a duplicate source frame")
                    if queued and row.frame_id <= queued[-1].source_frame_id:
                        raise RuntimeError("assistant queue retained source frame IDs out of order")
                    queued.append(policy_q2)
                    self._assistant_audio_active_onset[session_id] = (
                        None if retention_reason == "endpoint_tail" else int(active_onset)
                    )
                else:
                    self._record_assistant_queue_frame(
                        row,
                        reason=None,
                        onset_frame_id=None,
                        queue_epoch=queue_epoch,
                    )
                    self._assistant_audio_active_onset[session_id] = None
                    self._remember_pre_roll_frame(
                        session_id,
                        policy_q2,
                        source_pcm=source_pcm,
                    )
                self._policy_q2_soft_pending[row.session_id].append(policy_q2)
                view = decision_views.get(row.session_id)
                cache = self._q2_decision_caches.get(row.session_id)
                if (
                    view is not None
                    and cache is not None
                    and cache.q2_applied_count > 0
                    and cache.last_applied_frame_id == row.frame_id
                ):
                    pending_policy_q2 = self._policy_q2_soft_pending[row.session_id]
                    while (
                        pending_policy_q2
                        and pending_policy_q2[0].source_frame_id <= row.frame_id
                    ):
                        pending_policy_q2.popleft()
            if trace_enabled:
                snapshot_native_state = getattr(runtime.native, "session_state", None)
                for index, (row, state) in enumerate(zip(
                    batch.manifest.rows, states, strict=True
                )):
                    if not row.queue_audio_for_next_turn:
                        continue
                    native_cursor_after: int | None = None
                    snapshot_error = speaking_native_snapshot_errors.get(row.session_id)
                    updated_row = self._last_native_rows[row.session_id]
                    if callable(snapshot_native_state):
                        try:
                            native_state_after = snapshot_native_state(updated_row)
                            native_cursor_after = int(native_state_after.current_position)
                        except Exception as trace_error:
                            snapshot_error = type(trace_error).__name__
                    native_cursor_before = speaking_native_cursor_before.get(row.session_id)
                    native_cursor_delta = None
                    if native_cursor_before is not None and native_cursor_after is not None:
                        native_cursor_delta = native_cursor_after - native_cursor_before
                    view = decision_views.get(row.session_id)
                    queued_ids = [
                        int(item.source_frame_id)
                        for item in self._queued_audio_soft.get(row.session_id, ())
                    ]
                    rollback_tokens = (
                        int(native_output.kv_rollback_tokens_by_row[index])
                        if native_output.kv_rollback_tokens_by_row
                        else None
                    )
                    self._trace_unified_kv_integrity(
                        row.session_id,
                        "speaking_q2_frame_committed",
                        generation_id=int(row.session_generation),
                        state_epoch_id=row.state_epoch_id,
                        response_id=self._assistant_audio_queue_response_id.get(row.session_id),
                        queue_epoch=self._assistant_audio_queue_epoch.get(row.session_id),
                        queue_open=(row.session_id in self._assistant_audio_queue_open),
                        source_frame_id=int(row.frame_id),
                        queued_soft_frame_ids=queued_ids,
                        canonical_native_context_id=int(row.native_context_id),
                        canonical_logical_q_len=int(row.logical_q_len),
                        canonical_cursor_before=int(row.cache_position),
                        canonical_cursor_after=int(state.cache_position),
                        canonical_cursor_delta_tokens=(
                            int(state.cache_position) - int(row.cache_position)
                        ),
                        canonical_audio_tokens_committed=(
                            int(state.cache_position) - int(row.cache_position)
                        ),
                        native_cursor_before=native_cursor_before,
                        native_cursor_after=native_cursor_after,
                        native_cursor_delta_tokens=native_cursor_delta,
                        native_cursor_snapshot_error=snapshot_error,
                        canonical_q0_physical_rollback_tokens=rollback_tokens,
                        private_native_context_id=(
                            None if view is None else int(view.fork_row.native_context_id)
                        ),
                        private_context_distinct=(
                            None if view is None else
                            int(view.fork_row.native_context_id) != int(row.native_context_id)
                        ),
                        private_q2_source_frame_ids=(
                            [] if view is None else [int(x) for x in view.source_q2_frame_ids]
                        ),
                        private_q2_replayed_frame_count=(
                            0 if view is None else int(view.replayed_q2_count)
                        ),
                        private_q2_cursor_after=(
                            None if view is None else int(view.fork_row.cache_position)
                        ),
                        decision_view_unavailable_reason=(
                            decision_view_unavailable_reasons.get(row.session_id)
                        ),
                    )
            shadow = self._shadow_worker
            if shadow is not None:
                for shadow_q2 in shadow_q2_rows:
                    shadow.publish_q2(shadow_q2)
            self._processed_batches += 1
            if decisions is not None and self._on_result is not None:
                try:
                    self._on_result(batch, decisions)
                except BaseException as dispatch_error:
                    # State is committed and cannot be rolled back, so stop
                    # later work rather than silently losing a decision.
                    self._notification_error = dispatch_error
                    self._fatal_error = RuntimeError(
                        f"result dispatch failed after state commit: {dispatch_error}"
                    )
                    raise self._fatal_error from dispatch_error

            if self._tick_future is not None:
                expected = self._tick_expected or {}
                actual = {row.session_id: row.frame_id for row in batch.manifest.rows}
                if actual != expected:
                    raise RuntimeError(
                        "worker batch does not exactly match the submitted shared tick "
                        f"(expected={expected}, actual={actual})"
                    )
                assert prepared is not None
                outputs: dict[str, ListenerMicrobatchRowOutput] = {}
                for index, (row, state) in enumerate(zip(batch.manifest.rows, states, strict=True)):
                    updated_row = self._last_native_rows[row.session_id]
                    native_state = None
                    snapshot_native_state = getattr(runtime.native, "session_state", None)
                    if callable(snapshot_native_state):
                        native_state = snapshot_native_state(updated_row)
                    outputs[row.session_id] = ListenerMicrobatchRowOutput(
                        session_id=row.session_id,
                        generation_id=row.session_generation,
                        state_epoch_id=row.state_epoch_id,
                        frame_id=row.frame_id,
                        listener_state_version=state.state_version,
                        sequence_position=state.sequence_position,
                        cache_position=state.cache_position,
                        valid_history_tokens=state.valid_history_tokens,
                        logical_q_len=row.logical_q_len,
                        ephemeral_reason=row.ephemeral_reason,
                        queue_audio_for_next_turn=row.queue_audio_for_next_turn,
                        kv_rollback_tokens=(
                            native_output.kv_rollback_tokens_by_row[index]
                            if native_output.kv_rollback_tokens_by_row
                            else LISTENER_Q_PER_FRAME - row.logical_q_len
                        ),
                        kv_rollback_enqueue_ms=float(native_output.kv_rollback_enqueue_ms),
                        gemma_hidden_t=native_output.tensor[index, 0, :].unsqueeze(0),
                        gemma_audio_queries_t=native_output.tensor[index],
                        audio_tower_hidden=prepared.audio_soft_tokens[index],
                        state_snapshot=snapshot_listener_state(state),
                        native_state=native_state,
                        decision_view=decision_views.get(row.session_id),
                        decision_view_unavailable_reason=(
                            decision_view_unavailable_reasons.get(row.session_id)
                        ),
                        policy_decision=(None if decisions is None else decisions[index]),
                        native_output_health=native_output_health[index],
                        native_output_health_ms=native_output_health_ms,
                        native_control6_raw_scores=(
                            None
                            if native_control6_raw_scores is None
                            else native_control6_raw_scores[index]
                        ),
                        semantic_context_revision=self._prefix_revisions.get(row.session_id),
                        cache_epoch_id=int(state.cache_epoch_id),
                        applied_history_revision=state.applied_history_revision,
                        semantic_replay_fallback_reason=self._prefix_fallback_reasons.get(row.session_id),
                        semantic_prefix_len=self._prefix_lengths.get(row.session_id),
                        semantic_prefix_replay_frame_ids=(
                            self._prefix_replay_frame_ids.get(row.session_id, ())
                        ),
                        semantic_prefix_refresh_event=(
                            row.session_id in prefix_refresh_event_ids_by_session
                        ),
                        semantic_prefix_refresh_event_id=(
                            prefix_refresh_event_ids_by_session.get(row.session_id)
                        ),
                        semantic_prefix_refresh_revision=(
                            prefix_refresh_revisions_by_session.get(row.session_id)
                        ),
                        listener_model_lock_wait_ms=listener_model_lock_wait_ms,
                        listener_model_lock_hold_ms=listener_model_lock_hold_ms,
                        capacity_growth_ms=capacity_growth_ms,
                        frontend_encode_ms=frontend_encode_ms,
                        mel_prepare_ms=prepared.mel_prepare_ms_by_row[index],
                        frontend_copy_ms=frontend_copy_ms,
                        frontend_copy_cuda_ms=frontend_copy_cuda_ms,
                        audio_tower_lock_wait_ms=audio_tower_lock_wait_ms,
                        audio_tower_lock_hold_ms=audio_tower_lock_hold_ms,
                        audio_tower_workspace_lock_wait_ms=audio_tower_workspace_lock_wait_ms,
                        audio_tower_workspace_lock_hold_ms=audio_tower_workspace_lock_hold_ms,
                        audio_tower_row_copy_cuda_ms=audio_tower_row_copy_cuda_ms,
                        audio_tower_graph_submit_ms=audio_tower_graph_submit_ms,
                        audio_tower_cuda_ms=audio_tower_cuda_ms,
                        audio_tower_batch_cuda_span_ms=audio_tower_batch_cuda_span_ms,
                        audio_tower_execution_mode=audio_tower_execution_mode,
                        native_submit_ms=native_submit_ms,
                        native_cuda_ms=native_cuda_ms,
                        native_control6_raw_score_ms=native_control6_raw_score_ms,
                        semantic_prefix_build_ms=prefix_build_ms_by_session.get(row.session_id),
                        semantic_prefix_replay_ms=prefix_replay_ms_by_session.get(row.session_id),
                        semantic_prefix_import_ms=prefix_import_ms_by_session.get(row.session_id),
                        control6_policy_ms=control6_policy_ms,
                        cuda_completion_wait_ms=cuda_completion_wait_ms,
                    )
                if self._on_batch_result is not None:
                    self._on_batch_result(batch, outputs)
                worker_wall_ms = round(
                    max(0, time.perf_counter_ns() - worker_started_ns) / 1_000_000.0,
                    3,
                )
                outputs = {
                    session_id: replace(
                        output,
                        worker_queue_wait_ms=worker_queue_wait_ms,
                        worker_wall_ms=worker_wall_ms,
                    )
                    for session_id, output in outputs.items()
                }
                tick_future = self._tick_future
                self._tick_future = None
                self._tick_expected = None
                self._tick_submitted_ns = None
                if not tick_future.done():
                    tick_future.set_result(outputs)
        except BaseException as exc:
            if prepared is not None and not frontend_commit_attempted:
                try:
                    prepared.abort()
                except BaseException as abort_error:
                    if self._cleanup_error is None:
                        self._cleanup_error = abort_error
            if self._coordinator.active_manifest_id == batch.manifest.manifest_id:
                try:
                    self._coordinator.fail(batch, exc)
                except BaseException as fail_error:
                    if self._cleanup_error is None:
                        self._cleanup_error = fail_error
            for row in batch.manifest.rows:
                self._outstanding_frames[row.session_id] = 0
                self._pre_roll_ring.get(row.session_id, deque()).clear()
                self._queued_audio_soft.get(row.session_id, deque()).clear()
                self._policy_q2_soft_pending.get(row.session_id, deque()).clear()
                self._assistant_audio_active_onset[row.session_id] = None
                self._assistant_audio_queue_open.discard(row.session_id)
                self._assistant_audio_queue_frame_spans.pop(row.session_id, None)
                self._assistant_audio_queue_first_frame_id.pop(row.session_id, None)
                self._assistant_audio_queue_epoch.pop(row.session_id, None)
                self._assistant_audio_queue_close_epoch.pop(row.session_id, None)
                self._assistant_audio_queue_response_id.pop(row.session_id, None)
                self._reserved_tail_tokens[row.session_id] = 0
            self._fatal_error = exc
            tick_future = self._tick_future
            self._tick_future = None
            self._tick_expected = None
            self._tick_submitted_ns = None
            if tick_future is not None and not tick_future.done():
                tick_future.set_exception(exc)
            if self._on_error is not None:
                try:
                    self._on_error(batch, exc)
                except BaseException as notification_error:
                    self._notification_error = notification_error
        finally:
            if self._torch_profiler is not None and (
                profile_frames.intersection(self._torch_profile_frame_ids)
                and self._torch_profile_frame_ids.issubset(self._torch_profiled_frame_ids)
                and (
                    self._fatal_error is not None
                    or max(self._torch_profile_frame_ids) in profile_frames
                )
            ):
                profiler = self._torch_profiler
                self._torch_profiler = None
                try:
                    profiler.stop()
                    trace_path = os.path.abspath(self._torch_profile_trace_path)
                    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                    profiler.export_chrome_trace(trace_path)
                    logging.getLogger(__name__).info(
                        "listener torch profiler exported: frames=%s trace=%s",
                        sorted(self._torch_profiled_frame_ids), trace_path,
                    )
                except Exception as exc:
                    self._torch_profile_failed = True
                    logging.getLogger(__name__).warning(
                        "listener torch profiler export failed: %s: %s",
                        type(exc).__name__, exc,
                    )

    def _promote_semantic_before_manifest(
        self,
        frames: tuple[ListenerFrame, ...],
    ) -> Mapping[str, int] | None:
        shadow = self._shadow_worker
        runtime = self._runtime
        if shadow is None or runtime is None:
            return None
        rebases: dict[str, int] = {}
        if frames:
            candidate_rows: tuple[tuple[str, ListenerFrame | None], ...] = tuple(
                (frame.session_id, frame) for frame in frames
            )
        else:
            candidate_session_id = shadow.status.get("candidate_session_id")
            candidate_rows = (
                () if candidate_session_id is None
                else ((str(candidate_session_id), None),)
            )
        for session_id, frame in candidate_rows:
            if session_id not in self._imported_sessions:
                continue
            state = self._session_states.get(session_id)
            previous = self._last_native_rows.get(session_id)
            if state is None or previous is None:
                continue
            if frame is None and (
                self._coordinator is None
                or self._coordinator.ready_depth != 0
                or self._outstanding_frames.get(session_id, 0) != 0
                or (self._tick_expected is not None and session_id in self._tick_expected)
            ):
                continue
            destination_row = replace(
                previous,
                session_generation=state.generation_id,
                state_epoch_id=state.state_epoch_id,
                frame_id=state.next_frame_id,
                listener_state_version=state.state_version,
                sequence_position=state.sequence_position,
                cache_position=state.cache_position,
                valid_history_tokens=state.valid_history_tokens,
                native_context_id=int(state.native_context_id or 0),
            )
            if frame is not None and (
                destination_row.frame_id != frame.frame_id
                or destination_row.session_generation != frame.generation_id
                or destination_row.state_epoch_id != frame.state_epoch_id
            ):
                shadow.record_candidate_cursor_rejection(
                    session_id, "live_frame_cursor_mismatch"
                )
                continue
            candidate = shadow.candidate_for(destination_row)
            if candidate is None:
                continue
            candidate_request = candidate.stage.request
            if (candidate.prefix.prefix_len > self._kv_budgets.initial_prefix_max_tokens
                    or candidate.row.cache_position > self._kv_budgets.new_cache_target_tokens):
                shadow.discard_candidate(candidate, "rebuilt cache exceeds service token target")
                continue
            if isinstance(candidate_request, ListenerCompactionRequest):
                requested_history_revision = self._requested_history_revision.get(session_id)
                candidate_history_revision = (
                    None
                    if candidate_request.history_revision is None
                    else str(candidate_request.history_revision)
                )
                if (
                    requested_history_revision is not None
                    and candidate_history_revision != requested_history_revision
                ):
                    shadow.record_promotion_rejection(
                        "preflight", "stale_compaction_history_revision"
                    )
                    shadow.discard_candidate(
                        candidate,
                        "compaction candidate no longer matches the requested history revision",
                        copy_not_enqueued=True,
                    )
                    continue
            if session_id in self._pending_promotions:
                shadow.record_promotion_rejection(
                    "preflight", "prior_promotion_pending"
                )
                shadow.discard_candidate(
                    candidate,
                    "a prior semantic cache promotion has not reached its manifest",
                    copy_not_enqueued=True,
                )
                continue
            started_ns = time.perf_counter_ns()
            try:
                copy_result = runtime.native.copy_session_cache_from(
                    candidate.source_token,
                    destination_row,
                )
            except BaseException as exc:
                copy_enqueue_ms = round(
                    max(0, time.perf_counter_ns() - started_ns) / 1_000_000.0,
                    3,
                )
                shadow.record_cache_candidate_copy_enqueue(copy_enqueue_ms)
                if bool(getattr(runtime.native, "_poisoned", False)):
                    shadow.record_promotion_rejection(
                        "preflight", f"native_copy_preflight_poisoned:{type(exc).__name__}"
                    )
                    # The C adapter may have enqueued a partial copy. Keep the
                    # claimed shadow row pinned; primary shutdown drains its
                    # stream before stopping the shadow owner.
                    raise RuntimeError(
                        f"semantic cache promotion poisoned the primary owner: {exc}"
                    ) from exc
                shadow.record_promotion_rejection(
                    "preflight", f"native_copy_preflight_rejected:{type(exc).__name__}"
                )
                shadow.discard_candidate(
                    candidate,
                    f"cache promotion preflight rejected: {type(exc).__name__}: {exc}",
                    copy_not_enqueued=True,
                )
                continue
            copy_enqueue_ms = round(
                max(0, time.perf_counter_ns() - started_ns) / 1_000_000.0,
                3,
            )
            shadow.record_cache_candidate_copy_enqueue(copy_enqueue_ms)
            try:
                shadow.finish_promotion(candidate, copy_result)
            except BaseException as exc:
                shadow.record_promotion_rejection(
                    "copy", f"finish_promotion_rejected:{type(exc).__name__}"
                )
                raise
            promoted_row = getattr(copy_result, "destination_row", None)
            completion_event = getattr(copy_result, "completion_event", None)
            if not isinstance(promoted_row, BatchRowDescriptor) or completion_event is None:
                shadow.record_promotion_rejection("copy", "copy_result_incomplete")
                raise RuntimeError("native cache promotion returned an incomplete copy result")
            if (
                promoted_row.session_id != destination_row.session_id
                or promoted_row.session_generation != destination_row.session_generation
                or promoted_row.state_epoch_id != destination_row.state_epoch_id
                or promoted_row.frame_id != destination_row.frame_id
                or promoted_row.listener_state_version != destination_row.listener_state_version
                or promoted_row.sequence_position != destination_row.sequence_position
                or promoted_row.native_context_id != destination_row.native_context_id
                or promoted_row.cache_position != candidate.row.cache_position
                or promoted_row.valid_history_tokens != candidate.row.valid_history_tokens
                or int(getattr(copy_result, "prefix_len", -1)) != candidate.prefix.prefix_len
            ):
                shadow.record_promotion_rejection(
                    "copy", "copy_result_destination_cursor_mismatch"
                )
                raise RuntimeError("native cache promotion changed the destination boundary unexpectedly")
            self._pending_promotions[session_id] = _PendingShadowPromotion(
                candidate=candidate,
                copy_result=copy_result,
                copy_enqueue_ms=copy_enqueue_ms,
            )
            self._last_native_rows[session_id] = promoted_row
            self._prefix_revisions[session_id] = int(
                candidate.stage.request.semantic_context_revision
            )
            self._prefix_fallback_reasons[session_id] = (
                candidate.prefix.semantic_replay_fallback_reason
            )
            self._prefix_lengths[session_id] = candidate.prefix.prefix_len
            self._prefix_replay_frame_ids[session_id] = tuple(
                int(frame_id) for frame_id in candidate.prefix.replay_frame_ids
            )
            if isinstance(candidate.stage.request, ListenerCompactionRequest):
                self._compaction_status[session_id] = {
                    "status": "promoting",
                    "reason": None,
                    "cache_position": int(promoted_row.cache_position),
                    "frame_id": int(promoted_row.frame_id),
                    "revision": int(candidate.stage.request.semantic_context_revision),
                    "history_revision": candidate.stage.request.history_revision,
                }
            if frame is None:
                self._coordinator.rebase_cache_position(
                    session_id, int(promoted_row.cache_position)
                )
            rebases[session_id] = promoted_row.cache_position
        return rebases

    def _promote_idle_semantic_candidates(
        self, *, only_session_id: str | None = None,
    ) -> None:
        coordinator = self._coordinator
        shadow = self._shadow_worker
        if coordinator is None or shadow is None or coordinator.ready_depth != 0:
            return
        candidate_session_id = shadow.status.get("candidate_session_id")
        if candidate_session_id is None:
            return
        if only_session_id is not None and str(candidate_session_id) != str(only_session_id):
            return
        self._promote_semantic_before_manifest(())

    def _finalize_shadow_promotion(self, session_id: str, candidate_id: int) -> bool:
        """Commit cache identity only after the primary copy has been fenced."""
        pending = self._pending_promotions.get(session_id)
        if pending is None or int(pending.candidate.candidate_id) != int(candidate_id):
            return False
        candidate = pending.candidate
        request = candidate.stage.request
        state = self._session_states.get(session_id)
        if state is None or state.closed:
            self._pending_promotions.pop(session_id, None)
            return False

        is_compaction = isinstance(request, ListenerCompactionRequest)
        cache_epoch_id = int(state.cache_epoch_id) + int(is_compaction)
        technical_revision = int(request.semantic_context_revision)
        history_revision = (
            request.history_revision
            if is_compaction
            else getattr(candidate.prefix, "applied_history_revision", None)
        )
        if history_revision is None:
            history_revision = getattr(
                request, "history_revision", state.applied_history_revision
            )
        if history_revision is None:
            history_revision = state.applied_history_revision

        def commit_owner_metadata() -> None:
            state.cache_epoch_id = cache_epoch_id
            state.semantic_context_revision = technical_revision
            state.applied_history_revision = history_revision

        committer = self._on_cache_identity_commit
        if committer is None:
            commit_owner_metadata()
        else:
            # The server callback takes session._generation_lock and
            # listener_state._lock, then invokes this closure in that same
            # critical section. It must not wait on GPU work or a Future.
            committer(
                session_id,
                cache_epoch_id,
                technical_revision,
                history_revision,
                commit_owner_metadata,
            )
        if is_compaction:
            self._mark_compaction_promoted(session_id, candidate)
        self._pending_promotions.pop(session_id, None)
        with self._kv_work_lock:
            self._reap_shadow_work_owner(session_id)
        return True

    def _mark_compaction_promoted(
        self, session_id: str, candidate: ShadowPrefixCandidate,
    ) -> None:
        request = candidate.stage.request
        if not isinstance(request, ListenerCompactionRequest):
            return
        self._compaction_status[session_id] = {
            "status": "promoted",
            "reason": None,
            "cache_position": int(candidate.row.cache_position),
            "frame_id": int(candidate.row.frame_id),
            "revision": int(request.semantic_context_revision),
            "history_revision": request.history_revision,
            "target_cache_position": int(request.cache_position),
            "replay_frames": len(request.replay_frame_ids),
        }
        latest_history_revision = self._requested_history_revision.get(session_id)
        request_history_revision = (
            None if request.history_revision is None else str(request.history_revision)
        )
        if (
            latest_history_revision is None
            or latest_history_revision == request_history_revision
        ):
            self._compaction_requested.pop(session_id, None)
            self._compaction_reason.pop(session_id, None)
            self._requested_history_revision.pop(session_id, None)
            self._compaction_attempt_history_revision.pop(session_id, None)

    def _require_listener_frame_capacity(self, frame: ListenerFrame) -> None:
        coordinator = self._coordinator
        if coordinator is None:
            raise RuntimeError("listener coordinator is unavailable")
        state = self._session_states.get(frame.session_id)
        if state is None:
            raise KeyError(f"unregistered listener session: {frame.session_id}")
        if state.closed:
            raise RuntimeError("listener session is closed")
        if state.poisoned_reason is not None:
            raise RuntimeError(f"listener session is poisoned: {state.poisoned_reason}")
        expected_frame_id = int(state.next_frame_id) + int(
            self._outstanding_frames.get(frame.session_id, 0)
        )
        if (
            frame.generation_id != state.generation_id
            or frame.state_epoch_id != state.state_epoch_id
        ):
            raise ValueError("frame generation or epoch does not match listener state")
        if frame.frame_id != expected_frame_id:
            raise ValueError(
                f"frame id must equal next expected frame {expected_frame_id}, got {frame.frame_id}"
            )
        projected_cursor = (
            int(state.cache_position)
            + int(self._outstanding_logical_tokens.get(frame.session_id, 0))
        )
        max_seq_len = int(self._operational_max_seq_len())
        if projected_cursor + LISTENER_Q_PER_FRAME <= max_seq_len:
            return
        self._maybe_request_listener_compaction(frame.session_id, force_retry=True)
        status = self._compaction_status.get(frame.session_id, {})
        reason = str(status.get("status") or status.get("reason") or "compaction_not_ready")
        raise RecoverableSessionCapacityError(
            frame.session_id,
            generation_id=frame.generation_id,
            frame_id=frame.frame_id,
            cache_position=projected_cursor,
            requested_tokens=LISTENER_Q_PER_FRAME,
            max_seq_len=max_seq_len,
            reason="context_limit: " + reason,
        )

    def _maybe_request_listener_compaction(
        self, session_id: str, *, force_retry: bool = False,
    ) -> None:
        runtime = self._runtime
        coordinator = self._coordinator
        if runtime is None or coordinator is None:
            return
        try:
            snapshot = coordinator.session_state(session_id)
        except (KeyError, RuntimeError):
            return
        trigger = min(int(runtime.compaction_trigger_tokens), self._kv_budgets.memory_preparation_trigger_tokens)
        target = min(int(runtime.compaction_target_tokens), self._kv_budgets.new_cache_target_tokens)
        requested_history_revision = self._requested_history_revision.get(session_id)
        forced_history_refresh = (
            self._compaction_reason.get(session_id) == "confirmed_history_commit"
            and requested_history_revision is not None
        )
        if snapshot.cache_position < trigger and not forced_history_refresh:
            return

        builder = runtime.compaction_request_builder
        shadow = self._shadow_worker
        if builder is None or shadow is None:
            self._compaction_status[session_id] = {
                "status": "unavailable",
                "reason": "compaction_builder_or_shadow_not_configured",
                "cache_position": int(snapshot.cache_position),
            }
            self._compaction_requested[session_id] = int(snapshot.cache_position)
            return

        shadow_status = shadow.status
        active = shadow_status.get("active_stage") or {}
        candidate = shadow_status.get("candidate_stage") or {}
        current_stage = active if active.get("session_id") == session_id else candidate
        history_revision_changed = (
            requested_history_revision is not None
            and requested_history_revision
            != self._compaction_attempt_history_revision.get(session_id)
        )
        if current_stage.get("session_id") == session_id and not history_revision_changed:
            self._compaction_requested.setdefault(session_id, int(snapshot.cache_position))
            self._compaction_status[session_id] = {
                "status": "catching_up" if candidate.get("session_id") == session_id else "building",
                "reason": None,
                "cache_position": int(snapshot.cache_position),
                "revision": int(current_stage.get("semantic_context_revision", 0)),
                "frame_id": int(snapshot.next_frame_id),
            }
            return

        last_attempt = self._compaction_requested.get(session_id)
        retry_tokens = 16
        if (
            not force_retry
            and not history_revision_changed
            and last_attempt is not None
            and snapshot.cache_position - int(last_attempt) < retry_tokens
        ):
            return

        source_row = self._last_native_rows.get(session_id)
        if (
            session_id not in self._imported_sessions
            or source_row is None
            or source_row.session_generation != snapshot.generation_id
            or source_row.state_epoch_id != snapshot.state_epoch_id
            or source_row.frame_id != snapshot.next_frame_id
            or source_row.listener_state_version != snapshot.state_version
            or source_row.sequence_position != snapshot.sequence_position
            or source_row.cache_position != snapshot.cache_position
        ):
            self._compaction_status[session_id] = {
                "status": "waiting_for_listener_watermark",
                "reason": "completed_native_row_does_not_match_listener_snapshot",
                "cache_position": int(snapshot.cache_position),
            }
            return

        ring = tuple(self._recent_listener_q2.get(session_id, ()))
        ring = tuple(
            item for item in ring
            if item.frame.generation_id == snapshot.generation_id
            and item.frame.state_epoch_id == snapshot.state_epoch_id
            and item.frame.frame_id < snapshot.next_frame_id
        )
        if snapshot.next_frame_id and (
            not ring or ring[-1].frame.frame_id != snapshot.next_frame_id - 1
        ):
            self._compaction_status[session_id] = {
                "status": "waiting_for_recent_q2",
                "reason": "latest_committed_frame_has_no_canonical_q2_replay",
                "cache_position": int(snapshot.cache_position),
            }
            self._compaction_requested[session_id] = int(snapshot.cache_position)
            return

        try:
            context = builder(snapshot, ring, requested_history_revision)
        except CompactionBuilderCapacityError as exc:
            self._compaction_requested[session_id] = int(snapshot.cache_position)
            self._compaction_attempt_history_revision[session_id] = requested_history_revision
            self._compaction_status[session_id] = {
                "status": "capacity_blocked",
                "reason": exc.reason,
                "cache_position": int(snapshot.cache_position),
                "history_revision": requested_history_revision,
            }
            return
        except BaseException as exc:
            self._compaction_requested[session_id] = int(snapshot.cache_position)
            self._compaction_attempt_history_revision[session_id] = requested_history_revision
            self._compaction_status[session_id] = {
                "status": "builder_error",
                "reason": f"{type(exc).__name__}: {exc}",
                "cache_position": int(snapshot.cache_position),
            }
            return
        if context is None:
            self._compaction_requested[session_id] = int(snapshot.cache_position)
            self._compaction_attempt_history_revision[session_id] = requested_history_revision
            self._compaction_status[session_id] = {
                "status": "waiting_for_confirmed_history",
                "reason": "history_builder_returned_not_ready",
                "cache_position": int(snapshot.cache_position),
            }
            return
        if not isinstance(context, CompactionRequestContext):
            self._compaction_requested[session_id] = int(snapshot.cache_position)
            self._compaction_attempt_history_revision[session_id] = requested_history_revision
            self._compaction_status[session_id] = {
                "status": "builder_error",
                "reason": "history_builder_must_return_CompactionRequestContext_or_None",
                "cache_position": int(snapshot.cache_position),
            }
            return

        context_history_revision = (
            None if context.history_revision is None else str(context.history_revision)
        )
        if (
            requested_history_revision is not None
            and context_history_revision is not None
            and context_history_revision != requested_history_revision
        ):
            self._compaction_requested[session_id] = int(snapshot.cache_position)
            self._compaction_attempt_history_revision[session_id] = requested_history_revision
            self._compaction_status[session_id] = {
                "status": "waiting_for_requested_history_revision",
                "reason": "history_builder_returned_a_different_revision",
                "cache_position": int(snapshot.cache_position),
                "requested_history_revision": requested_history_revision,
                "built_history_revision": context_history_revision,
            }
            return

        prefix_ids = context.prefix_token_ids
        native_prefix_capacity = int(getattr(
            runtime.native,
            "prefix_capacity_tokens",
            NATIVE_CONTROL6_PREFIX_MAX_TOKENS,
        ))
        prefix_capacity = min(NATIVE_CONTROL6_PREFIX_MAX_TOKENS, native_prefix_capacity,
                              self._kv_budgets.initial_prefix_max_tokens)
        if len(prefix_ids) > prefix_capacity:
            self._compaction_requested[session_id] = int(snapshot.cache_position)
            self._compaction_status[session_id] = {
                "status": "capacity_blocked",
                "reason": f"semantic_prefix_exceeds_native_capacity_{prefix_capacity}",
                "cache_position": int(snapshot.cache_position),
            }
            return
        replay_capacity = (target - len(prefix_ids)) // LISTENER_Q_PER_FRAME
        if replay_capacity < 0:
            self._compaction_requested[session_id] = int(snapshot.cache_position)
            self._compaction_status[session_id] = {
                "status": "capacity_blocked",
                "reason": "semantic_prefix_exceeds_compaction_target",
                "cache_position": int(snapshot.cache_position),
            }
            return
        if context.replay_after_frame_id is None:
            expected_replay_start = 0
        else:
            if context.replay_after_frame_id >= snapshot.next_frame_id:
                self._compaction_requested[session_id] = int(snapshot.cache_position)
                self._compaction_status[session_id] = {
                    "status": "builder_error",
                    "reason": "replay_after_frame_id_must_precede_listener_watermark",
                    "cache_position": int(snapshot.cache_position),
                }
                return
            expected_replay_start = int(context.replay_after_frame_id) + 1
        if expected_replay_start < snapshot.next_frame_id:
            expected_ids = tuple(range(expected_replay_start, snapshot.next_frame_id))
            available_ids = tuple(int(item.frame.frame_id) for item in ring)
            if available_ids[-len(expected_ids):] != expected_ids:
                self._compaction_requested[session_id] = int(snapshot.cache_position)
                self._compaction_attempt_history_revision[session_id] = requested_history_revision
                self._compaction_status[session_id] = {
                    "status": "capacity_blocked",
                    "reason": "confirmed_history_cutoff_does_not_cover_missing_q2_frames",
                    "cache_position": int(snapshot.cache_position),
                    "replay_after_frame_id": context.replay_after_frame_id,
                    "available_q2_first_frame_id": (
                        None if not ring else int(ring[0].frame.frame_id)
                    ),
                }
                return
            selected = ring[-len(expected_ids):]
        else:
            selected = ()
        if len(selected) > replay_capacity:
            self._compaction_requested[session_id] = int(snapshot.cache_position)
            self._compaction_attempt_history_revision[session_id] = requested_history_revision
            self._compaction_status[session_id] = {
                "status": "capacity_blocked",
                "reason": "confirmed_recent_q2_exceeds_compaction_target_without_truncation",
                "cache_position": int(snapshot.cache_position),
                "replay_frames_required": len(selected),
                "replay_frames_capacity": replay_capacity,
            }
            return
        replay_ids = tuple(int(item.frame.frame_id) for item in selected)
        if replay_ids and replay_ids != tuple(
            range(snapshot.next_frame_id - len(replay_ids), snapshot.next_frame_id)
        ):
            self._compaction_requested[session_id] = int(snapshot.cache_position)
            self._compaction_status[session_id] = {
                "status": "waiting_for_contiguous_q2",
                "reason": "recent_q2_replay_is_not_contiguous_at_listener_watermark",
                "cache_position": int(snapshot.cache_position),
            }
            return

        revision = max(
            int(self._compaction_revision.get(session_id, 0)) + 1,
            int(self._prefix_revisions.get(session_id) or 0) + 1,
        )
        request = ListenerCompactionRequest(
            session_id=session_id,
            state_epoch_id=snapshot.state_epoch_id,
            current_frame_id=snapshot.next_frame_id,
            listener_state_version=snapshot.state_version,
            sequence_position=snapshot.sequence_position,
            semantic_context_revision=revision,
            prefix_token_ids=prefix_ids,
            prefix_len=len(prefix_ids),
            replay_q2=tuple(item.audio_soft_tokens for item in selected),
            replay_frame_ids=replay_ids,
            cache_position=len(prefix_ids) + LISTENER_Q_PER_FRAME * len(selected),
            history_context=context.history_context,
            history_revision=(
                context.history_revision
                if context.history_revision is not None
                else requested_history_revision
            ),
            replay_after_frame_id=context.replay_after_frame_id,
            session_generation_id=snapshot.generation_id,
        )
        try:
            accepted = self.stage_semantic_prefix(
                request,
                listener_generation_id=snapshot.generation_id,
                sequence_position=snapshot.sequence_position,
            )
        except BaseException as exc:
            self._compaction_requested[session_id] = int(snapshot.cache_position)
            self._compaction_attempt_history_revision[session_id] = requested_history_revision
            self._compaction_status[session_id] = {
                "status": "stage_error",
                "reason": f"{type(exc).__name__}: {exc}",
                "cache_position": int(snapshot.cache_position),
            }
            return
        if not accepted:
            self._compaction_requested[session_id] = int(snapshot.cache_position)
            self._compaction_attempt_history_revision[session_id] = requested_history_revision
            self._compaction_status[session_id] = {
                "status": "stage_coalesced_or_current",
                "reason": "shadow_worker_already_has_this_semantic_revision",
                "cache_position": int(snapshot.cache_position),
            }
            return
        self._compaction_revision[session_id] = revision
        self._compaction_requested[session_id] = int(snapshot.cache_position)
        self._compaction_attempt_history_revision[session_id] = requested_history_revision
        self._compaction_status[session_id] = {
            "status": "building",
            "reason": None,
            "cache_position": int(snapshot.cache_position),
            "target_cache_position": int(request.cache_position),
            "replay_frames": len(selected),
            "revision": revision,
            "frame_id": snapshot.next_frame_id,
        }

    @staticmethod
    def _wait_batch_completion(runtime: ListenerMicrobatchRuntime) -> None:
        stream = runtime.listener_stream
        if runtime.completion_barrier is not None:
            runtime.completion_barrier(stream)
            return
        if torch.device(stream.device).type != "cuda":
            return
        # CUDA waits run only on the dedicated owner thread. Recording after
        # frontend, native, and policy work fences every consumer before host
        # state can be committed; the event loop never blocks on this wait.
        event = torch.cuda.Event(enable_timing=False, blocking=False)
        event.record(stream)
        event.synchronize()

    def _should_stop(self) -> bool:
        if not self._stop_requested:
            return False
        if self._active_queued_audio_drains:
            return False
        if self._fatal_error is not None or not self._drain_on_stop:
            return True
        if self._coordinator.ready_depth == 0 and self._commands.empty():
            return True
        if self._stop_deadline_ns is not None and time.monotonic_ns() >= self._stop_deadline_ns:
            self._drain_timeout_reached = True
            return True
        return False

    def _next_timeout_seconds(self) -> float | None:
        deadline_ns = self._coordinator.next_wakeup_ns(time.monotonic_ns())
        if self._stop_requested and self._drain_on_stop and self._stop_deadline_ns is not None:
            deadline_ns = (
                self._stop_deadline_ns
                if deadline_ns is None
                else min(deadline_ns, self._stop_deadline_ns)
            )
        if deadline_ns is None:
            return None
        return max(0.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000)

    def _wait_for_work(self, timeout: float | None) -> None:
        with self._wake:
            if not self._commands.empty():
                return
            self._wake.wait(timeout=timeout)

    def _fail_pending_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            if command.future is not None and not command.future.done():
                command.future.set_exception(RuntimeError("listener worker stopped before command execution"))
        for pending in self._deferred_owner_commands.values():
            while pending:
                command = pending.popleft()
                if command.future is not None and not command.future.done():
                    command.future.set_exception(
                        RuntimeError("listener worker stopped before deferred owner command execution")
                    )
        self._deferred_owner_commands.clear()

    def _stop_report(self, status: str, *, drained: bool) -> WorkerStopReport:
        fatal = None if self._fatal_error is None else f"{type(self._fatal_error).__name__}: {self._fatal_error}"
        cleanup = None if self._cleanup_error is None else f"{type(self._cleanup_error).__name__}: {self._cleanup_error}"
        notification = (
            None
            if self._notification_error is None
            else f"{type(self._notification_error).__name__}: {self._notification_error}"
        )
        clean = status == "stopped" and drained and fatal is None and cleanup is None
        if fatal is not None:
            status = "failed"
        elif cleanup is not None:
            status = "cleanup_failed"
        elif not drained and status == "stopped":
            status = "drain_timeout" if self._drain_timeout_reached else "incomplete"
        return WorkerStopReport(
            status=status,
            stopped_cleanly=clean,
            drained=drained,
            processed_batches=self._processed_batches,
            pending_frames=(
                0 if self._coordinator is None else self._coordinator.ready_depth
            ),
            owner_thread_id=self._owner_thread_id,
            fatal_error=fatal,
            cleanup_error=cleanup,
            notification_error=notification,
        )


def _listener_stream_context(stream: Any):
    device = torch.device(stream.device)
    if device.type == "cuda":
        return torch.cuda.stream(stream)
    return _NullContext()


class _CompletedShadowEvent:
    """CPU-test fence with the same minimal query/wait surface as CUDA events."""

    def record(self, _stream: Any = None) -> None:
        return None

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None

    def wait(self, _stream: Any) -> None:
        return None


def _strict_nonnegative_int_value(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _shadow_replay_base_cursor(
    stage: ShadowPrefixStage,
) -> tuple[int, int, int]:
    request = stage.request
    replay_q2 = tuple(request.replay_q2)
    replay_ids = tuple(request.replay_frame_ids)
    replay_count = len(replay_q2)
    target_frame_id = _strict_nonnegative_int_value(
        "current_frame_id", request.current_frame_id
    )
    target_state_version = _strict_nonnegative_int_value(
        "listener_state_version", request.listener_state_version
    )
    target_sequence_position = _strict_nonnegative_int_value(
        "sequence_position", stage.sequence_position
    )
    if target_frame_id < replay_count:
        raise ValueError("current_frame_id underflows the frozen replay count")
    if target_state_version < replay_count:
        raise ValueError("listener_state_version underflows the frozen replay count")
    if target_sequence_position < replay_count:
        raise ValueError("sequence_position underflows the frozen replay count")
    base_frame_id = target_frame_id - replay_count
    if replay_ids != tuple(range(base_frame_id, target_frame_id)):
        raise ValueError(
            "semantic replay frame IDs must be contiguous and match the frozen target watermark"
        )
    expected_cache_position = (
        _strict_nonnegative_int_value("prefix_len", request.prefix_len)
        + LISTENER_Q_PER_FRAME * replay_count
    )
    cache_position = _strict_nonnegative_int_value(
        "cache_position", request.cache_position
    )
    if cache_position != expected_cache_position:
        raise ValueError(
            "semantic request cache cursor must equal prefix_len plus two tokens per replay row"
        )
    return (
        base_frame_id,
        target_state_version - replay_count,
        target_sequence_position - replay_count,
    )


def _validate_shadow_stage(stage: ShadowPrefixStage) -> None:
    request = stage.request
    for name in (
        "session_id", "state_epoch_id", "current_frame_id", "listener_state_version",
        "semantic_context_revision", "prefix_token_ids", "prefix_len", "replay_q2",
        "replay_frame_ids", "cache_position",
    ):
        if not hasattr(request, name):
            raise TypeError(f"semantic request lacks {name}")
    if not isinstance(request.session_id, str) or not request.session_id.strip():
        raise ValueError("semantic request session_id must be non-empty")
    if not isinstance(request.state_epoch_id, str) or not request.state_epoch_id.strip():
        raise ValueError("semantic request state_epoch_id must be non-empty")
    for name in (
        "current_frame_id", "listener_state_version", "semantic_context_revision",
        "prefix_len", "cache_position",
    ):
        _strict_nonnegative_int_value(name, getattr(request, name))
    request_generation = getattr(request, "session_generation_id", None)
    if request_generation is not None:
        # This is the websocket/session turn generation. The listener-state
        # generation used for cancellation and promotion is carried separately
        # by ShadowPrefixStage and must remain authoritative there.
        _strict_nonnegative_int_value("session_generation_id", request_generation)
    if not 1 <= int(request.prefix_len) <= NATIVE_CONTROL6_PREFIX_MAX_TOKENS:
        raise ValueError(
            "semantic request prefix_len must be in [1,"
            f"{NATIVE_CONTROL6_PREFIX_MAX_TOKENS}]"
        )
    replay_q2 = tuple(request.replay_q2)
    replay_ids = tuple(request.replay_frame_ids)
    if len(replay_q2) != len(replay_ids) or len(replay_q2) > MAX_RECENT_LISTENER_Q2_FRAMES:
        raise ValueError(
            "semantic replay must contain at most "
            f"{MAX_RECENT_LISTENER_Q2_FRAMES} aligned Q2/frame rows"
        )
    for frame_id in replay_ids:
        _strict_nonnegative_int_value("replay_frame_id", frame_id)
    _shadow_replay_base_cursor(stage)
    for value in replay_q2:
        if not isinstance(value, torch.Tensor):
            raise ValueError("semantic replay rows must be torch tensors")
        if tuple(getattr(value, "shape", ())) != (LISTENER_Q_PER_FRAME, CONTROL_HIDDEN_WIDTH):
            raise ValueError("semantic replay rows must have exact Q=2 [2,1536] shape")
        if value.dtype != torch.bfloat16:
            raise ValueError("semantic replay rows must use the native BF16 Q2 dtype")


def _shadow_stage_identity(stage: ShadowPrefixStage) -> tuple[Any, ...]:
    request = stage.request
    return (
        str(request.session_id),
        int(stage.listener_generation_id),
        str(request.state_epoch_id),
        int(request.semantic_context_revision),
    )


def _same_listener_cursor(left: BatchRowDescriptor, right: BatchRowDescriptor) -> bool:
    return (
        left.session_id == right.session_id
        and left.session_generation == right.session_generation
        and left.state_epoch_id == right.state_epoch_id
        and left.frame_id == right.frame_id
        and left.listener_state_version == right.listener_state_version
        and left.sequence_position == right.sequence_position
    )


def _record_shadow_event(stream: Any) -> Any:
    if torch.device(stream.device).type != "cuda":
        return _CompletedShadowEvent()
    event = torch.cuda.Event(enable_timing=False, blocking=False)
    event.record(stream)
    return event


def _timed_cuda_event_pair(stream: Any) -> tuple[Any, Any] | None:
    if torch.device(stream.device).type != "cuda":
        return None
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def _elapsed_timed_cuda_events(events: tuple[Any, Any] | None) -> float | None:
    if events is None:
        return None
    start, end = events
    return round(max(0.0, float(start.elapsed_time(end))), 3)


def _wait_shadow_event(event: Any, stream: Any) -> None:
    synchronize = getattr(event, "synchronize", None)
    if callable(synchronize):
        synchronize()
        return
    wait = getattr(event, "wait", None)
    if callable(wait):
        wait(stream)
        return
    raise TypeError("shadow native completion event is not waitable")


def _drain_cuda_stream_for_cleanup(stream: Any) -> None:
    """Fence teardown on the dedicated owner thread, never on an event loop."""
    if torch.device(stream.device).type != "cuda":
        return
    # Event.record(stream) orders after earlier stream work; synchronize() is
    # intentionally confined to this worker's cleanup path.
    event = torch.cuda.Event(enable_timing=False, blocking=False)
    event.record(stream)
    event.synchronize()


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False


__all__ = [
    "ListenerMicrobatchRuntime",
    "ListenerMicrobatchWorker",
    "PrefixCacheImport",
    "WorkerStopReport",
]
