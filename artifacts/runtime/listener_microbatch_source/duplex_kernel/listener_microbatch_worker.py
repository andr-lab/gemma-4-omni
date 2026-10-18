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
from dataclasses import dataclass, replace
import hashlib
import math
import queue
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Mapping, Sequence

import torch

from duplex_kernel.gemma_native_control6_batch_policy import GemmaNativeControl6BatchPolicy
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
    LISTENER_Q_PER_FRAME,
    NativeBatchCapability,
    RowStateUpdate,
    snapshot_listener_state,
)
from duplex_kernel.native_listener_microbatch import NativeBatchOutput, NativeBatchOutputHealth
from duplex_kernel.turn_policy import PolicyDecision, PolicyInput


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


@dataclass(frozen=True, slots=True)
class TextAppendResult:
    """Native text append outputs fenced after the LM head on the owner stream."""

    logits: torch.Tensor
    final_norm_hidden: torch.Tensor
    ready_event: Any
    timings_ns: Mapping[str, int]


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
        [ListenerBatch, tuple[ListenerSessionState, ...]], Sequence[PolicyInput]
    ] | None
    listener_model_lock: Any | None = None
    completion_barrier: Callable[[Any], None] | None = None
    close_resources: Callable[[], None] | None = None
    prefix_importer: PrefixImportFactory | None = None
    control6_scorer: GemmaNativeControl6BatchPolicy | None = None
    shadow_factory: Callable[[], "SemanticPrefixShadowRuntime"] | None = None
    shadow_prefix_builder: Callable[[Any, Any], PrefixCacheImport] | None = None
    token_adapter: Any | None = None


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
    ) -> None:
        if not callable(runtime_factory) or not callable(prefix_builder):
            raise TypeError("shadow runtime and prefix builder must be callable")
        if isinstance(max_pending_q2, bool) or not isinstance(max_pending_q2, int) or max_pending_q2 < 1:
            raise ValueError("max_pending_q2 must be a positive integer")
        self._runtime_factory = runtime_factory
        self._prefix_builder = prefix_builder
        self._runtime_signature = runtime_signature
        self._max_pending_q2 = max_pending_q2
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


@dataclass(slots=True)
class _Command:
    operation: str
    args: tuple[Any, ...]
    future: Future[Any] | None = None


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
    ) -> None:
        if coordinator is not None and not isinstance(coordinator, ListenerMicrobatchCoordinator):
            raise TypeError("coordinator must be ListenerMicrobatchCoordinator or None")
        if not callable(runtime_factory):
            raise TypeError("runtime_factory must create owner-thread resources")
        self._coordinator = coordinator
        self._runtime_factory = runtime_factory
        self._on_result = on_result
        self._on_batch_result = on_batch_result
        self._on_error = on_error
        self._commands: queue.Queue[_Command] = queue.Queue()
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
        self._queued_audio_soft: dict[str, deque[QueuedAudioSoftFrame]] = {}
        self._assistant_audio_queue_open: set[str] = set()
        self._reserved_tail_tokens: dict[str, int] = {}
        self._pending_promotions: dict[str, _PendingShadowPromotion] = {}
        self._outstanding_frames: dict[str, int] = {}
        self._tick_future: Future[Mapping[str, ListenerMicrobatchRowOutput]] | None = None
        self._tick_expected: dict[str, int] | None = None
        self._processed_batches = 0
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
        self, frames: Sequence[ListenerFrame]
    ) -> Future[Mapping[str, ListenerMicrobatchRowOutput]]:
        """Submit every real row in one fixed-grid tick atomically."""
        rows = tuple(frames)
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
        return self._submit_command("submit_tick", rows)

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
            or len(token_ids) > 291
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in token_ids)
        ):
            future: Future[None] = Future()
            future.set_exception(ValueError("unified prefix IDs must be 1..291 non-negative integers"))
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
    ) -> Future[TextAppendResult]:
        """Append one or two text token IDs to the session's native KV owner.

        The future resolves to logits and final-norm hidden for the last
        committed token, plus a CUDA event recorded after the LM head on the
        listener owner stream. Consumers on another stream must wait for that
        event before reading either tensor. No audio frontend or prefix
        importer is invoked.
        """
        ids = tuple(token_ids)
        if len(ids) not in (1, 2) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in ids
        ):
            future: Future[TextAppendResult] = Future()
            future.set_exception(ValueError("text append requires one or two non-negative token IDs"))
            return future
        if (
            isinstance(consume_reserved_tokens, bool)
            or not isinstance(consume_reserved_tokens, int)
            or consume_reserved_tokens < 0
        ):
            future: Future[TextAppendResult] = Future()
            future.set_exception(ValueError("consume_reserved_tokens must be a non-negative integer"))
            return future
        return self._submit_command(
            "append_text_tokens", session_id, ids,
            int(expected_generation_id), int(expected_cache_position),
            int(consume_reserved_tokens),
        )

    def begin_assistant_audio_queue(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
        reserve_tail_tokens: int = 0,
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
        return self._submit_command(
            "begin_assistant_audio_queue", session_id,
            int(expected_generation_id), int(expected_cache_position),
            reserve_tail_tokens,
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
        return self._submit_command(
            "drain_queued_audio", session_id,
            int(expected_generation_id), int(expected_cache_position),
            user_turn_already_open,
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
        return shadow.stage(
            request,
            listener_generation_id=listener_generation_id,
            sequence_position=sequence_position,
        )

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

    def session_state(self, session_id: str) -> Future[ListenerSessionSnapshot]:
        """Request an owner-thread snapshot between batches, never during one."""
        return self._submit_command("snapshot", session_id)

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
            self._commands.put(_Command(operation, args, future))
            self._wake.notify_all()
        return future

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
                self._wait_for_work(self._next_timeout_seconds())
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
                    _drain_cuda_stream_for_cleanup(runtime.listener_stream)
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
                self._stopped.set_result(self._stop_report("stopped", drained=self._coordinator.ready_depth == 0))
            tick_future = self._tick_future
            self._tick_future = None
            self._tick_expected = None
            if tick_future is not None and not tick_future.done():
                tick_future.set_exception(RuntimeError("listener worker stopped before shared tick completed"))

    def _validate_runtime(self, runtime: ListenerMicrobatchRuntime) -> None:
        if not isinstance(runtime, ListenerMicrobatchRuntime):
            raise TypeError("runtime_factory must return ListenerMicrobatchRuntime")
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
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                if self._fatal_error is not None and command.operation not in {"snapshot", "stop"}:
                    raise RuntimeError(f"worker is fail-closed after batch error: {self._fatal_error}")
                result = self._execute_command(command)
                if (
                    command.operation != "submit_tick"
                    and command.future is not None
                    and not command.future.done()
                ):
                    command.future.set_result(result)
            except BaseException as exc:
                if command.future is not None and not command.future.done():
                    command.future.set_exception(exc)

    def _execute_command(self, command: _Command) -> Any:
        if command.operation == "register":
            state, prefix_import = command.args
            return self._register_session(state, prefix_import)
        if command.operation == "frame":
            (frame,) = command.args
            if not isinstance(frame, ListenerFrame):
                raise TypeError("frame must be a complete ListenerFrame")
            self._coordinator.enqueue(frame)
            self._outstanding_frames[frame.session_id] = (
                self._outstanding_frames.get(frame.session_id, 0) + 1
            )
            return None
        if command.operation == "submit_tick":
            (frames,) = command.args
            if command.future is None:
                raise RuntimeError("submit_tick requires a result Future")
            if self._tick_future is not None and not self._tick_future.done():
                raise RuntimeError("only one shared tick may be in flight")
            expected = {frame.session_id: int(frame.frame_id) for frame in frames}
            if len(expected) != len(frames):
                raise ValueError("shared tick session IDs must be unique")
            for frame in frames:
                state = self._session_states.get(frame.session_id)
                if state is None:
                    raise KeyError(f"unregistered listener session: {frame.session_id}")
                if frame.queue_audio_for_next_turn and frame.session_id not in self._assistant_audio_queue_open:
                    raise RuntimeError(
                        "assistant-phase audio queue is closed for session "
                        f"{frame.session_id}; stale Q0 cannot cross the next-turn drain boundary"
                    )
                if (
                    frame.generation_id != state.generation_id
                    or frame.state_epoch_id != state.state_epoch_id
                    or frame.frame_id != state.next_frame_id
                ):
                    raise ValueError(
                        f"shared tick row identity is stale for session {frame.session_id}"
                    )
            # The shared dispatcher has already closed this tick's row set.
            # Enqueue all rows in this one owner command so coordinator wakeup
            # cannot start a singleton before sibling rows are admitted.
            self._coordinator.coalesce_window_ns = 0
            for frame in frames:
                self._coordinator.enqueue(frame)
                self._outstanding_frames[frame.session_id] = (
                    self._outstanding_frames.get(frame.session_id, 0) + 1
                )
            self._tick_expected = expected
            self._tick_future = command.future
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
            session_id, token_ids, generation_id, cache_position, consume_reserved_tokens = command.args
            return self._append_text_tokens(
                session_id, token_ids,
                expected_generation_id=generation_id,
                expected_cache_position=cache_position,
                consume_reserved_tokens=consume_reserved_tokens,
            )
        if command.operation == "begin_assistant_audio_queue":
            session_id, generation_id, cache_position, reserve_tail_tokens = command.args
            return self._begin_assistant_audio_queue(
                session_id, expected_generation_id=generation_id,
                expected_cache_position=cache_position,
                reserve_tail_tokens=reserve_tail_tokens,
            )
        if command.operation == "reserve_tail_tokens":
            session_id, generation_id, cache_position, token_count = command.args
            return self._reserve_tail_tokens(
                session_id,
                expected_generation_id=generation_id,
                expected_cache_position=cache_position,
                token_count=token_count,
            )
        if command.operation == "drain_queued_audio":
            session_id, generation_id, cache_position, user_turn_already_open = command.args
            return self._drain_queued_audio(
                session_id, expected_generation_id=generation_id,
                expected_cache_position=cache_position,
                user_turn_already_open=user_turn_already_open,
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
        if command.operation == "snapshot":
            (session_id,) = command.args
            return self._coordinator.session_state(session_id)
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
    ) -> TextAppendResult:
        timings_ns: dict[str, int] = {"owner_started_ns": time.monotonic_ns()}
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
        snapshot = coordinator.session_state(session_id)
        if (
            snapshot.generation_id != expected_generation_id
            or snapshot.cache_position != expected_cache_position
            or snapshot.valid_history_tokens != expected_cache_position
        ):
            raise RuntimeError("unified text append lost its generation/cache-position compare-and-swap")
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
            > coordinator.capability.max_seq_len
        ):
            raise RuntimeError(
                "native listener capacity cannot preserve queued next-turn audio before text append; "
                f"session={session_id}, cache_position={snapshot.cache_position}, "
                f"append_tokens={len(token_ids)}, queued_audio_tokens={queued_tokens}, "
                f"reserved_tail_tokens={remaining_reserved}, "
                f"max_seq_len={coordinator.capability.max_seq_len}"
            )
        if snapshot.cache_position > coordinator.capability.max_seq_len - LISTENER_Q_PER_FRAME:
            raise RuntimeError(
                "native listener physical Q2 capacity exhausted before text append; "
                f"session={session_id}, cache_position={snapshot.cache_position}, "
                f"max_seq_len={coordinator.capability.max_seq_len}"
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
            prepared = adapter.prepare(token_ids, logical_q_lens=len(token_ids))
            timings_ns["token_prepare_returned_ns"] = time.monotonic_ns()
            if tuple(prepared.logical_q_lens) != (len(token_ids),):
                raise RuntimeError("token adapter logical-Q metadata differs from text append")
            if tuple(prepared.inputs_embeds.shape) != (1, LISTENER_Q_PER_FRAME, CONTROL_HIDDEN_WIDTH):
                raise RuntimeError("token adapter returned an incompatible physical Q2 embedding shape")
            if tuple(prepared.per_layer_inputs.shape) != (1, LISTENER_Q_PER_FRAME, 35, 256):
                raise RuntimeError("token adapter returned an incompatible projected PLE shape")
            mutation_started = False
            try:
                mutation_started = True
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
                self._last_native_rows[session_id] = replace(
                    row,
                    frame_id=updated.next_frame_id,
                    listener_state_version=updated.state_version,
                    sequence_position=updated.sequence_position,
                    cache_position=updated.cache_position,
                    valid_history_tokens=updated.valid_history_tokens,
                )
                self._reserved_tail_tokens[session_id] = remaining_reserved
                hidden = output.tensor[0, len(token_ids) - 1, :].reshape(1, 1, -1)
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
                return TextAppendResult(
                    logits=logits,
                    final_norm_hidden=hidden,
                    ready_event=ready_event,
                    timings_ns=dict(timings_ns),
                )
            except BaseException as exc:
                if mutation_started:
                    self._fatal_error = RuntimeError(
                        f"unified native text append failed after cache mutation: {exc}"
                    )
                    raise self._fatal_error from exc
                raise

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

    def _begin_assistant_audio_queue(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
        reserve_tail_tokens: int,
    ) -> ListenerSessionSnapshot:
        snapshot = self._check_unified_session_cas(
            session_id,
            expected_generation_id=expected_generation_id,
            expected_cache_position=expected_cache_position,
        )
        if self._outstanding_frames.get(session_id, 0) != 0:
            raise RuntimeError("assistant audio queue must open at an idle frame boundary")
        if session_id in self._assistant_audio_queue_open:
            raise RuntimeError("assistant audio queue is already open for this model turn")
        if self._queued_audio_soft.get(session_id):
            raise RuntimeError("cannot open a new assistant turn with undrained queued audio")
        self._validate_tail_reservation(reserve_tail_tokens)
        coordinator = self._coordinator
        assert coordinator is not None
        if snapshot.cache_position + reserve_tail_tokens > coordinator.capability.max_seq_len:
            raise RuntimeError(
                "native listener capacity cannot reserve the model-turn close boundary; "
                f"session={session_id}, cursor={snapshot.cache_position}, "
                f"reserved_tail_tokens={reserve_tail_tokens}, "
                f"max_seq_len={coordinator.capability.max_seq_len}"
            )
        runtime = self._runtime
        if runtime is None or not callable(getattr(runtime.native, "step_queued_audio_async", None)):
            raise RuntimeError("native runtime lacks queued-audio replay; no PCM re-tower fallback is allowed")
        self._reserved_tail_tokens[session_id] = reserve_tail_tokens
        self._assistant_audio_queue_open.add(session_id)
        return snapshot

    @staticmethod
    def _validate_tail_reservation(token_count: int) -> None:
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            raise ValueError("reserved tail token count must be a non-negative integer")

    def _reserve_tail_tokens(
        self,
        session_id: str,
        *,
        expected_generation_id: int,
        expected_cache_position: int,
        token_count: int,
    ) -> int:
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
        if snapshot.cache_position + queued_tokens + token_count > coordinator.capability.max_seq_len:
            raise RuntimeError(
                "native listener capacity cannot preserve queued audio and reserved tail; "
                f"session={session_id}, cursor={snapshot.cache_position}, "
                f"queued_audio_tokens={queued_tokens}, reserved_tail_tokens={token_count}, "
                f"max_seq_len={coordinator.capability.max_seq_len}"
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
            return updated
        except BaseException as exc:
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
    ) -> ListenerSessionSnapshot:
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
        max_seq_len = int(coordinator.capability.max_seq_len)
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
        previous_frame_id = -1
        for item in queued:
            if (
                item.session_id != session_id
                or item.session_generation != snapshot.generation_id
                or item.state_epoch_id != snapshot.state_epoch_id
                or item.source_frame_id <= previous_frame_id
                or item.source_frame_id >= snapshot.next_frame_id
                or item.source_state_version >= snapshot.state_version
                or item.source_sequence_position >= snapshot.sequence_position
                or tuple(item.audio_soft_tokens.shape) != (LISTENER_Q_PER_FRAME, CONTROL_HIDDEN_WIDTH)
                or item.audio_soft_tokens.dtype != torch.bfloat16
                or item.audio_soft_tokens.device.type != "cuda"
            ):
                raise RuntimeError("queued audio provenance/tensor does not match the current listener turn")
            previous_frame_id = item.source_frame_id

        # Closing this phase before native replay is the FIFO watermark: every
        # accepted pre-boundary frame is already committed, and later Q0 queue
        # submissions are rejected until the next model turn explicitly opens.
        self._assistant_audio_queue_open.remove(session_id)
        if not queued:
            self._reserved_tail_tokens[session_id] = 0
            return snapshot
        if not callable(getattr(runtime.native, "step_queued_audio_async", None)):
            raise RuntimeError("native runtime lacks queued-audio replay; no PCM re-tower fallback is allowed")
        current = snapshot
        base_row = self._last_native_rows.get(session_id)
        if base_row is None:
            raise RuntimeError("queued-audio drain has no native session row")
        try:
            with _listener_stream_context(runtime.listener_stream), torch.inference_mode():
                for item in queued:
                    if current.cache_position > max_seq_len - LISTENER_Q_PER_FRAME:
                        raise RuntimeError(
                            "native physical Q2 capacity exhausted before queued-audio replay"
                        )
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
                    current = coordinator.advance_cache_position(
                        session_id,
                        token_count=LISTENER_Q_PER_FRAME,
                        expected_generation_id=expected_generation_id,
                        expected_cache_position=current.cache_position,
                    )
                    base_row = replace(
                        row,
                        frame_id=current.next_frame_id,
                        listener_state_version=current.state_version,
                        sequence_position=current.sequence_position,
                        cache_position=current.cache_position,
                        valid_history_tokens=current.valid_history_tokens,
                    )
                    self._last_native_rows[session_id] = base_row
            self._queued_audio_soft[session_id].clear()
            if current.cache_position != snapshot.cache_position + required:
                raise RuntimeError("queued-audio drain committed an unexpected number of KV tokens")
            # The caller has crossed the user boundary on success, or rollback
            # already cleared the assistant close reservation on cancellation.
            self._reserved_tail_tokens[session_id] = 0
            return current
        except BaseException as exc:
            self._fatal_error = RuntimeError(
                "queued-audio replay failed after native cache mutation; session is quarantined: "
                f"{type(exc).__name__}: {exc}"
            )
            raise self._fatal_error from exc

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
        self._prefix_importers[state.session_id] = prefix_import
        self._prefix_revisions[state.session_id] = None
        self._unified_prefix_revisions[state.session_id] = 0
        self._prefix_fallback_reasons[state.session_id] = None
        self._prefix_lengths[state.session_id] = int(state.cache_position)
        self._prefix_replay_frame_ids[state.session_id] = ()
        self._outstanding_frames[state.session_id] = 0
        self._queued_audio_soft[state.session_id] = deque()
        self._reserved_tail_tokens[state.session_id] = 0

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
        if not 1 <= len(prefix_token_ids) <= 291:
            raise ValueError("unified prefix token count must be in [1,291]")
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
        try:
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
        if not 1 <= prefix_position <= 291:
            raise ValueError("initial semantic prefix position must be in [1,291]")

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
        if state.native_context_id is None or not callable(prefix_import):
            raise ValueError("reset requires a native context and owner-thread prefix importer")
        try:
            if self._shadow_worker is not None:
                self._shadow_worker.invalidate_session(session_id)
            self._release_native_session(session_id)
            self._coordinator.close_session(session_id)
            del self._session_states[session_id]
            del self._prefix_importers[session_id]
            self._prefix_revisions.pop(session_id, None)
            self._unified_prefix_revisions.pop(session_id, None)
            self._prefix_fallback_reasons.pop(session_id, None)
            self._prefix_lengths.pop(session_id, None)
            self._prefix_replay_frame_ids.pop(session_id, None)
            self._queued_audio_soft.pop(session_id, None)
            self._reserved_tail_tokens.pop(session_id, None)
            self._assistant_audio_queue_open.discard(session_id)
            del self._outstanding_frames[session_id]
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
        try:
            if self._shadow_worker is not None:
                self._shadow_worker.invalidate_session(session_id)
            self._release_native_session(session_id)
            retired = self._coordinator.close_session(session_id)
            self._session_states.pop(session_id)
            self._prefix_importers.pop(session_id)
            self._prefix_revisions.pop(session_id, None)
            self._unified_prefix_revisions.pop(session_id, None)
            self._prefix_fallback_reasons.pop(session_id, None)
            self._prefix_lengths.pop(session_id, None)
            self._prefix_replay_frame_ids.pop(session_id, None)
            self._outstanding_frames.pop(session_id)
            self._queued_audio_soft.pop(session_id, None)
            self._reserved_tail_tokens.pop(session_id, None)
            self._assistant_audio_queue_open.discard(session_id)
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
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("worker runtime is unavailable")
        states = tuple(self._session_states[row.session_id] for row in batch.manifest.rows)
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
        queued_soft_to_commit: list[QueuedAudioSoftFrame] = []
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
                actual_capacity = runtime.native.ensure_capacity(
                    len(self._session_states)
                )
                capacity_growth_ms = round(
                    max(0, time.perf_counter_ns() - capacity_started_ns) / 1_000_000.0,
                    3,
                )
                if (
                    isinstance(actual_capacity, bool)
                    or not isinstance(actual_capacity, int)
                    or actual_capacity < len(self._session_states)
                ):
                    raise RuntimeError(
                        "native listener returned insufficient capacity at tick boundary: "
                        f"requested={len(self._session_states)}, actual={actual_capacity!r}"
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
                for index, row in enumerate(batch.manifest.rows):
                    if not row.queue_audio_for_next_turn:
                        continue
                    if row.logical_q_len != 0 or row.ephemeral_reason != "assistant_turn_open":
                        raise RuntimeError("only assistant-phase Q0 rows may queue next-turn audio")
                    if row.session_id not in self._assistant_audio_queue_open:
                        raise RuntimeError(
                            "assistant-phase Q0 arrived after the queue boundary closed; "
                            "stale audio cannot be silently dropped"
                        )
                    pending = self._queued_audio_soft.get(row.session_id)
                    if pending is None:
                        raise RuntimeError("queued audio has no registered per-session owner buffer")
                    reserved_tail = self._reserved_tail_tokens.get(row.session_id, 0)
                    required_end = (
                        row.cache_position + 2 * (len(pending) + 1) + reserved_tail
                    )
                    if required_end > int(self._coordinator.capability.max_seq_len):
                        raise RuntimeError(
                            "queued user audio and reserved model-turn tail exceed native KV capacity; "
                            "no frame was dropped "
                            f"(session={row.session_id}, cursor={row.cache_position}, "
                            f"queued_frames={len(pending) + 1}, "
                            f"reserved_tail_tokens={reserved_tail}, "
                            f"max_seq_len={self._coordinator.capability.max_seq_len})"
                        )
                    if pending and row.frame_id <= pending[-1].source_frame_id:
                        raise RuntimeError("queued assistant-phase audio frames are not strictly ordered")
                native_cuda_events = _timed_cuda_event_pair(runtime.listener_stream)
                if native_cuda_events is not None:
                    native_cuda_events[0].record(runtime.listener_stream)
                native_started_ns = time.perf_counter_ns()
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
                    policy_inputs = tuple(runtime.policy_inputs(batch, states))
                    policy_started_ns = time.perf_counter_ns()
                    decisions = tuple(runtime.policy.step_native_output_batch(
                        native_output,
                        batch.manifest,
                        policy_inputs,
                    ))
                    control6_policy_ms = round(
                        max(0, time.perf_counter_ns() - policy_started_ns) / 1_000_000.0,
                        3,
                    )
                    if len(decisions) != batch.manifest.batch_size:
                        raise RuntimeError("Control6 returned an incomplete row set")
                    for row, decision in zip(batch.manifest.rows, decisions, strict=True):
                        if (
                            decision.session != row.session_id
                            or decision.generation_id != row.session_generation
                            or decision.frame_id != row.frame_id
                            or decision.listener_frame_id != row.frame_id
                        ):
                            raise RuntimeError("Control6 decision identity does not match its native row")
                completion_started_ns = time.perf_counter_ns()
                self._wait_batch_completion(runtime)
                for index, row in enumerate(batch.manifest.rows):
                    if not row.queue_audio_for_next_turn:
                        continue
                    soft = prepared.audio_soft_tokens[index]
                    if soft.is_cuda:
                        soft.record_stream(runtime.listener_stream)
                    owned_soft = soft.detach().clone().contiguous()
                    if owned_soft.is_cuda:
                        owned_soft.record_stream(runtime.listener_stream)
                    queued_soft_to_commit.append(QueuedAudioSoftFrame(
                        session_id=row.session_id,
                        session_generation=row.session_generation,
                        state_epoch_id=row.state_epoch_id,
                        source_frame_id=row.frame_id,
                        source_state_version=row.listener_state_version,
                        source_sequence_position=row.sequence_position,
                        audio_soft_tokens=owned_soft,
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
            for queued_soft in queued_soft_to_commit:
                self._queued_audio_soft[queued_soft.session_id].append(queued_soft)
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
                        native_output_health=native_output_health[index],
                        native_output_health_ms=native_output_health_ms,
                        native_control6_raw_scores=(
                            None
                            if native_control6_raw_scores is None
                            else native_control6_raw_scores[index]
                        ),
                        semantic_context_revision=self._prefix_revisions.get(row.session_id),
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
                for session_id in batch_promotions:
                    self._pending_promotions.pop(session_id, None)
                tick_future = self._tick_future
                self._tick_future = None
                self._tick_expected = None
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
            self._fatal_error = exc
            tick_future = self._tick_future
            self._tick_future = None
            self._tick_expected = None
            if tick_future is not None and not tick_future.done():
                tick_future.set_exception(exc)
            if self._on_error is not None:
                try:
                    self._on_error(batch, exc)
                except BaseException as notification_error:
                    self._notification_error = notification_error

    def _promote_semantic_before_manifest(
        self,
        frames: tuple[ListenerFrame, ...],
    ) -> Mapping[str, int] | None:
        shadow = self._shadow_worker
        runtime = self._runtime
        if shadow is None or runtime is None:
            return None
        rebases: dict[str, int] = {}
        for frame in frames:
            session_id = frame.session_id
            if session_id not in self._imported_sessions:
                continue
            state = self._session_states.get(session_id)
            previous = self._last_native_rows.get(session_id)
            if state is None or previous is None:
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
            if (
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
            rebases[session_id] = promoted_row.cache_position
        return rebases

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
            pending_frames=self._coordinator.ready_depth,
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
    if not 1 <= int(request.prefix_len) <= 291:
        raise ValueError("semantic request prefix_len must be in [1,291]")
    replay_q2 = tuple(request.replay_q2)
    replay_ids = tuple(request.replay_frame_ids)
    if len(replay_q2) != len(replay_ids) or len(replay_q2) > 64:
        raise ValueError("semantic replay must contain at most 64 aligned Q2/frame rows")
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
