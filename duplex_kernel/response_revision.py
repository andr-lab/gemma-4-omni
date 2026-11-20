"""Bounded response revision jobs and the atomic stream/ledger install seam."""
from __future__ import annotations

from dataclasses import dataclass, replace
import threading
import time
from typing import Any, Callable, Mapping


class StaleResponseRevision(RuntimeError):
    """A candidate no longer matches the response or speech frontier."""


@dataclass(frozen=True, slots=True)
class ResponseRevisionRequest:
    session_id: str
    response_id: str
    generation_id: int
    frame_id: int
    requested_at_ns: int
    attempt: int = 0


@dataclass(frozen=True, slots=True)
class ResponseRevisionCandidate:
    """Immutable output and CAS snapshot produced on a private native fork."""

    session_id: str
    response_id: str
    generation_id: int
    base_revision_id: str
    revision_id: str
    history_revision: int
    cache_epoch_id: str | int | None
    listener_watermark: Any
    latest_listener_snapshot: Any
    requested_frame_id: int
    stream_frontier: Any
    ledger_snapshot: Mapping[str, Any]
    frozen_prefix_token_ids: tuple[int, ...]
    source_rows: tuple[Mapping[str, Any], ...]
    packets: tuple[Any, ...]
    packet_records: tuple[Mapping[str, Any], ...]
    cumulative_text: str
    think_result: Mapping[str, Any]
    cache_position: int
    handle: Any = None
    cancel_event: Any = None
    cleanup_callback: Callable[[], Any] | None = None

    def cleanup(self) -> None:
        """Release a private candidate only when ownership was not transferred."""
        if callable(self.cleanup_callback):
            self.cleanup_callback()


class ResponseRevisionCoordinator:
    """Run at most one cancellable candidate plus one coalesced latest request."""

    def __init__(
        self,
        *,
        session_id: str,
        response_id: str,
        generation_id: int,
        run_candidate: Callable[[ResponseRevisionRequest, threading.Event], Any],
        commit_candidate: Callable[[Any], bool],
        refresh_request: Callable[[ResponseRevisionRequest], ResponseRevisionRequest | None]
        | None = None,
        max_stale_retries: int = 2,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("response_id must be non-empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (generation_id, max_stale_retries)
        ):
            raise ValueError("generation_id and max_stale_retries must be non-negative integers")
        if not callable(run_candidate) or not callable(commit_candidate):
            raise TypeError("revision candidate and commit callbacks are required")
        self.session_id = session_id
        self.response_id = response_id
        self.generation_id = int(generation_id)
        self._run_candidate = run_candidate
        self._commit_candidate = commit_candidate
        self._refresh_request = refresh_request
        self._max_stale_retries = int(max_stale_retries)
        self._monotonic_ns = monotonic_ns
        self._condition = threading.Condition(threading.RLock())
        self._pending: ResponseRevisionRequest | None = None
        self._current_cancel: threading.Event | None = None
        self._worker: threading.Thread | None = None
        self._closed = False
        self._eos_requested = False
        self._eos_consumed = False
        self._finalize_claimed = False
        self._latest_adopted: Any = None
        self._last_error: str | None = None
        self._metrics: dict[str, int | float | str | None] = {
            "requests": 0,
            "coalesced_requests": 0,
            "candidate_started": 0,
            "candidate_cancelled": 0,
            "candidate_stale": 0,
            "candidate_committed": 0,
            "latest_requested_frame_id": None,
            "last_candidate_frame_id": None,
            "last_adopted_frame_id": None,
            "last_adaptation_lag_frames": None,
            "last_candidate_duration_ms": None,
            "last_error": None,
        }

    def request_revision(self, frame_id: int) -> bool:
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError("frame_id must be a non-negative integer")
        with self._condition:
            if self._closed or self._eos_consumed or self._finalize_claimed:
                return False
            request = ResponseRevisionRequest(
                session_id=self.session_id,
                response_id=self.response_id,
                generation_id=self.generation_id,
                frame_id=int(frame_id),
                requested_at_ns=int(self._monotonic_ns()),
            )
            self._metrics["requests"] = int(self._metrics["requests"]) + 1
            self._metrics["latest_requested_frame_id"] = int(frame_id)
            if self._pending is not None or self._current_cancel is not None:
                self._metrics["coalesced_requests"] = int(
                    self._metrics["coalesced_requests"]
                ) + 1
            self._pending = request
            # Keep the in-flight candidate alive. Cancelling it for every new
            # 80 ms frame prevents any candidate from finishing while the user
            # is speaking. Its captured watermark is immutable; the atomic
            # commit checks that watermark, then the single pending slot runs
            # the newest request if newer audio arrived in the meantime.
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._run_pending,
                    name=f"response-revision-{self.response_id[:12]}",
                    daemon=True,
                )
                self._worker.start()
            self._condition.notify_all()
            return True

    def _run_pending(self) -> None:
        while True:
            with self._condition:
                if self._closed or self._pending is None:
                    self._current_cancel = None
                    self._condition.notify_all()
                    return
                request = self._pending
                self._pending = None
                cancel_event = threading.Event()
                self._current_cancel = cancel_event
                self._metrics["candidate_started"] = int(
                    self._metrics["candidate_started"]
                ) + 1
                self._metrics["last_candidate_frame_id"] = request.frame_id
            started_ns = int(self._monotonic_ns())
            candidate = None
            stale = False
            try:
                candidate = self._run_candidate(request, cancel_event)
                if candidate is not None and not cancel_event.is_set():
                    stale = not bool(self._commit_candidate(candidate))
                elif cancel_event.is_set():
                    with self._condition:
                        self._metrics["candidate_cancelled"] = int(
                            self._metrics["candidate_cancelled"]
                        ) + 1
                if candidate is not None and not stale and not cancel_event.is_set():
                    with self._condition:
                        self._latest_adopted = candidate
                        self._metrics["candidate_committed"] = int(
                            self._metrics["candidate_committed"]
                        ) + 1
                        self._metrics["last_adopted_frame_id"] = request.frame_id
                        watermark = getattr(candidate, "listener_watermark", None)
                        candidate_frame = getattr(watermark, "frame_id", None)
                        if isinstance(candidate_frame, int):
                            self._metrics["last_adaptation_lag_frames"] = max(
                                0, request.frame_id - (int(candidate_frame) - 1)
                            )
                elif stale:
                    with self._condition:
                        self._metrics["candidate_stale"] = int(
                            self._metrics["candidate_stale"]
                        ) + 1
            except StaleResponseRevision:
                stale = True
                with self._condition:
                    self._metrics["candidate_stale"] = int(
                        self._metrics["candidate_stale"]
                    ) + 1
            except BaseException as exc:
                if cancel_event.is_set():
                    with self._condition:
                        self._metrics["candidate_cancelled"] = int(
                            self._metrics["candidate_cancelled"]
                        ) + 1
                else:
                    with self._condition:
                        self._last_error = f"{type(exc).__name__}: {exc}"[:240]
                        self._metrics["last_error"] = self._last_error
            finally:
                cleanup = getattr(candidate, "cleanup", None)
                if callable(cleanup):
                    try:
                        cleanup()
                    except BaseException as exc:
                        with self._condition:
                            self._last_error = f"cleanup {type(exc).__name__}: {exc}"[:240]
                            self._metrics["last_error"] = self._last_error
                with self._condition:
                    self._metrics["last_candidate_duration_ms"] = max(
                        0.0,
                        (int(self._monotonic_ns()) - started_ns) / 1_000_000.0,
                    )
                    if self._current_cancel is cancel_event:
                        self._current_cancel = None
                    self._condition.notify_all()

            if stale and request.attempt < self._max_stale_retries and self._refresh_request is not None:
                try:
                    refreshed = self._refresh_request(request)
                except BaseException as exc:
                    refreshed = None
                    with self._condition:
                        self._last_error = f"retry {type(exc).__name__}: {exc}"[:240]
                        self._metrics["last_error"] = self._last_error
                if refreshed is not None:
                    with self._condition:
                        if not self._closed and not self._eos_consumed:
                            if self._pending is None or self._pending.frame_id <= refreshed.frame_id:
                                self._pending = replace(
                                    refreshed,
                                    attempt=request.attempt + 1,
                                )

    def cancel_all(self, reason: str = "cancelled") -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._last_error = str(reason)[:240]
            self._metrics["last_error"] = self._last_error
            if self._current_cancel is not None:
                self._current_cancel.set()
            self._condition.notify_all()

    def request_eos(self) -> None:
        with self._condition:
            self._eos_requested = True
            self._condition.notify_all()

    def note_eos_consumed(self) -> None:
        with self._condition:
            self._eos_consumed = True
            self._eos_requested = True
            self._pending = None
            if self._current_cancel is not None:
                self._current_cancel.set()
            self._condition.notify_all()

    def wait_until_idle(self, timeout_s: float | None = None) -> bool:
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative or None")
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        with self._condition:
            while self._pending is not None or self._current_cancel is not None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def join(self, timeout_s: float | None = None) -> bool:
        """Join the producer after cancellation or after its work drained."""
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative or None")
        with self._condition:
            worker = self._worker
        if worker is None:
            return True
        if worker is threading.current_thread():
            return False
        worker.join(timeout_s)
        return not worker.is_alive()

    def claim_finalize_if_idle(self) -> bool:
        """Atomically close the revision window after EOS and all jobs drain."""
        with self._condition:
            if (
                not self._eos_requested
                or self._closed
                or self._pending is not None
                or self._current_cancel is not None
                or self._finalize_claimed
            ):
                return False
            self._finalize_claimed = True
            return True

    def latest_adopted_after(self, revision_id: str) -> Any | None:
        with self._condition:
            candidate = self._latest_adopted
            if candidate is None or str(getattr(candidate, "revision_id", "")) == str(revision_id):
                return None
            return candidate

    def metrics_snapshot(self) -> dict[str, int | float | str | None]:
        with self._condition:
            return dict(self._metrics)

    @property
    def pending(self) -> bool:
        with self._condition:
            return self._pending is not None or self._current_cancel is not None

    @property
    def closed(self) -> bool:
        with self._condition:
            return bool(self._closed)


def install_response_revision_transaction(
    *,
    session: Any,
    branch: dict[str, Any],
    stream: Any,
    ledger: Any,
    candidate: ResponseRevisionCandidate,
    promote_candidate: Callable[[ResponseRevisionCandidate], Any] | None = None,
) -> bool:
    """Install stream, ledger, and ACK records under the agreed lock order.

    All allocations and validations happen before the first assignment. The
    lock order is generation -> played-prefix -> stream -> ledger. No callback
    or application callback is invoked from this transaction. An optional
    owner CAS may detach/promote the already prepared native handle after every
    Python plan is validated and before any Python state is assigned.
    """
    generation_lock = getattr(session, "_generation_lock", None)
    branch_lock = branch.get("spoken_prefix_lock")
    stream_lock = getattr(stream, "_condition", None)
    ledger_lock = getattr(ledger, "_lock", None)
    if any(lock is None for lock in (generation_lock, branch_lock, stream_lock, ledger_lock)):
        raise RuntimeError("response revision transaction lacks one of its CAS locks")
    if not isinstance(candidate, ResponseRevisionCandidate):
        raise TypeError("candidate must be a ResponseRevisionCandidate")

    with generation_lock:
        with branch_lock:
            with stream_lock:
                with ledger_lock:
                    if candidate.cancel_event is not None and candidate.cancel_event.is_set():
                        return False
                    if (
                        int(getattr(session, "session_generation_id", -1))
                        != candidate.generation_id
                        or int(getattr(stream, "generation_id", -1))
                        != candidate.generation_id
                        or int(getattr(ledger, "generation_id", -1))
                        != candidate.generation_id
                        or str(getattr(session, "response_id", "")) != candidate.response_id
                        or not bool(getattr(session, "active_response", False))
                    ):
                        return False
                    if (
                        str(branch.get("response_id")) != candidate.response_id
                        or int(branch.get("session_generation_id", -1)) != candidate.generation_id
                        or str(branch.get("revision_id", "r0")) != candidate.base_revision_id
                        or int(getattr(session, "semantic_context_revision", -1))
                        != candidate.history_revision
                    ):
                        return False
                    watermark = candidate.listener_watermark
                    latest = candidate.latest_listener_snapshot
                    if (
                        getattr(latest, "generation_id", None)
                        != getattr(watermark, "generation_id", None)
                        or getattr(latest, "state_epoch_id", None)
                        != getattr(watermark, "state_epoch_id", None)
                    ):
                        return False
                    if hasattr(watermark, "cache_epoch_id"):
                        source_cache_epoch = getattr(watermark, "cache_epoch_id")
                        if (
                            getattr(latest, "cache_epoch_id", None) != source_cache_epoch
                            or candidate.cache_epoch_id != source_cache_epoch
                        ):
                            return False
                        # Native compaction publishes this mirror under the
                        # same generation lock. Reading it here makes cache
                        # epoch and applied-history CAS linearize with adoption.
                        listener_state = getattr(session, "listener_state", None)
                        if (
                            listener_state is None
                            or getattr(listener_state, "cache_epoch_id", None)
                            != source_cache_epoch
                            or getattr(listener_state, "native_semantic_context_revision", None)
                            != getattr(latest, "semantic_context_revision", None)
                            or getattr(listener_state, "applied_history_revision", None)
                            != getattr(watermark, "applied_history_revision", None)
                            or getattr(listener_state, "applied_history_revision", None)
                            != candidate.history_revision
                        ):
                            return False
                    if hasattr(watermark, "applied_history_revision"):
                        if (
                            getattr(watermark, "applied_history_revision")
                            != candidate.history_revision
                            or getattr(latest, "applied_history_revision", None)
                            != candidate.history_revision
                        ):
                            return False
                    if (
                        not stream._same_frozen_source_frontier(
                            stream._source_frontier_locked(), candidate.stream_frontier
                        )
                        or stream.cache_epoch != candidate.stream_frontier.cache_epoch
                    ):
                        return False
                    current_ledger_snapshot = ledger.source_frontier_snapshot()
                    expected_ledger = dict(candidate.ledger_snapshot)
                    if any(
                        current_ledger_snapshot.get(field) != expected_ledger.get(field)
                        for field in (
                            "frozen_source_sequence",
                            "revision_id",
                            "cache_epoch_id",
                            "tts_published_source_seq",
                            "tts_reserved_source_seq",
                            "tts_consumed_source_seq",
                        )
                    ):
                        return False
                    if (
                        int(current_ledger_snapshot["frozen_source_sequence"])
                        != int(candidate.stream_frontier.frozen_source_sequence)
                        or current_ledger_snapshot["revision_id"] != candidate.base_revision_id
                        or current_ledger_snapshot["cache_epoch_id"]
                        != (
                            None if candidate.stream_frontier.cache_epoch is None
                            else str(candidate.stream_frontier.cache_epoch)
                        )
                    ):
                        return False

                    prefix_sequences = tuple(
                        sequence
                        for sequence in sorted(ledger._source_tokens)
                        if sequence <= int(candidate.stream_frontier.frozen_source_sequence)
                    )
                    prefix_ids = tuple(
                        int(ledger._source_tokens[sequence]["token_id"])
                        for sequence in prefix_sequences
                    )
                    if prefix_ids != candidate.frozen_prefix_token_ids:
                        return False

                    old_packets = branch.get("say_token_packets", ())
                    old_records = branch.get("segment_token_packets", ())
                    if not isinstance(old_packets, (tuple, list)) or not isinstance(
                        old_records, (tuple, list)
                    ) or len(old_packets) != len(old_records):
                        raise RuntimeError("ACK packet rows are not aligned before revision")
                    frozen_pairs = [
                        (record, packet)
                        for record, packet in zip(old_records, old_packets, strict=True)
                        if isinstance(record, Mapping)
                        and isinstance(record.get("source_token_seq"), int)
                        and not isinstance(record.get("source_token_seq"), bool)
                        and int(record["source_token_seq"])
                        <= int(candidate.stream_frontier.frozen_source_sequence)
                    ]
                    branch_prefix_ids = tuple(int(record["token_id"]) for record, _ in frozen_pairs)
                    if branch_prefix_ids != candidate.frozen_prefix_token_ids:
                        return False
                    if len(candidate.source_rows) != len(candidate.packets) or len(
                        candidate.source_rows
                    ) != len(candidate.packet_records):
                        raise ValueError("candidate source, ACK, and packet rows have different lengths")
                    replacement_tokens = tuple(dict(row) for row in candidate.source_rows)
                    prefix_count = len(candidate.frozen_prefix_token_ids)
                    replacement_records = tuple(
                        {
                            **dict(row),
                            "token_index": prefix_count + index,
                        }
                        for index, row in enumerate(candidate.packet_records)
                    )
                    replacement_packets = tuple(candidate.packets)
                    previous_sequence = int(candidate.stream_frontier.frozen_source_sequence)
                    for source, record, packet in zip(
                        replacement_tokens, replacement_records, replacement_packets, strict=True
                    ):
                        sequence = source.get("source_token_seq")
                        if (
                            isinstance(sequence, bool)
                            or not isinstance(sequence, int)
                            or sequence <= previous_sequence
                            or record.get("source_token_seq") != sequence
                            or int(record.get("token_id", -1)) != int(source.get("token_id", -2))
                            or int(packet.token_id) != int(source.get("token_id", -2))
                            or str(record.get("revision_id")) != candidate.revision_id
                        ):
                            raise ValueError("candidate logical source rows are not aligned")
                        previous_sequence = int(sequence)

                    stream_plan = stream.prepare_unconsumed_suffix_locked(
                        replacement_tokens,
                        expected_frontier=candidate.stream_frontier,
                        frozen_prefix_token_ids=candidate.frozen_prefix_token_ids,
                        revision_id=candidate.revision_id,
                        cache_epoch=candidate.cache_epoch_id,
                    )
                    ledger_plan = ledger.prepare_unconsumed_suffix_locked(
                        int(candidate.stream_frontier.frozen_source_sequence) + 1,
                        replacement_tokens,
                        expected_source_frontier=int(
                            expected_ledger["frozen_source_sequence"]
                        ),
                        expected_revision_id=str(expected_ledger["revision_id"]),
                        expected_cache_epoch_id=expected_ledger["cache_epoch_id"],
                        expected_tts_published_source_seq=int(
                            expected_ledger["tts_published_source_seq"]
                        ),
                        expected_tts_reserved_source_seq=int(
                            expected_ledger["tts_reserved_source_seq"]
                        ),
                        expected_tts_consumed_source_seq=int(
                            expected_ledger["tts_consumed_source_seq"]
                        ),
                        revision_id=candidate.revision_id,
                        cache_epoch_id=(
                            None if candidate.cache_epoch_id is None
                            else str(candidate.cache_epoch_id)
                        ),
                        intended_text=candidate.cumulative_text,
                    )
                    if tuple(ledger_plan.sequences) != tuple(
                        int(row["source_token_seq"]) for row in replacement_tokens
                    ):
                        raise ValueError("ledger changed candidate logical source identities")

                    frozen_packets = tuple(packet for _record, packet in frozen_pairs)
                    frozen_records = tuple(
                        {**dict(record), "token_index": index}
                        for index, (record, _packet) in enumerate(frozen_pairs)
                    )
                    new_packets = frozen_packets + replacement_packets
                    new_records = frozen_records + replacement_records
                    new_adopted_budget = len(candidate.frozen_prefix_token_ids) + len(
                        candidate.source_rows
                    )
                    adopted_think = dict(candidate.think_result)
                    revision_metrics = dict(branch.get("revision_metrics", {}))
                    revision_metrics["adopted_revisions"] = int(
                        revision_metrics.get("adopted_revisions", 0)
                    ) + 1
                    revision_metrics["last_adaptation_lag_frames"] = max(
                        0,
                        int(candidate.requested_frame_id)
                        - (
                            int(getattr(
                                candidate.listener_watermark,
                                "frame_id",
                                candidate.requested_frame_id + 1,
                            )) - 1
                        ),
                    )
                    revision_metrics["last_adopted_watermark_frame_id"] = int(
                        getattr(candidate.listener_watermark, "frame_id", -1)
                    )
                    revision_metrics["last_adopted_cache_epoch_id"] = candidate.cache_epoch_id
                    revision_metrics["last_adopted_history_revision"] = candidate.history_revision

                    candidate_handle = candidate.handle
                    new_packets_list = list(new_packets)
                    new_records_list = list(new_records)
                    listener_state_epoch_id = str(
                        getattr(candidate.listener_watermark, "state_epoch_id", "")
                    )
                    listener_generation_id = int(
                        getattr(candidate.listener_watermark, "generation_id", -1)
                    )
                    source_frame_id = int(getattr(candidate.listener_watermark, "frame_id", -1))
                    source_state_version = int(
                        getattr(candidate.listener_watermark, "state_version", -1)
                    )
                    source_sequence_position = int(
                        getattr(candidate.listener_watermark, "sequence_position", -1)
                    )
                    listener_source_cache_position = int(
                        getattr(candidate.listener_watermark, "cache_position", -1)
                    )
                    candidate_source_cursor = int(getattr(
                        candidate.handle, "source_cache_position", candidate.cache_position
                    ))
                    candidate_cache_position = int(candidate.cache_position)
                    candidate_spoken_base_cursor = int(
                        candidate.think_result.get(
                            "spoken_prefix_base_cursor", candidate.cache_position
                        )
                    ) - prefix_count
                    candidate_transition_ids = tuple(
                        int(value) for value in candidate.think_result.get(
                            "transition_token_ids", branch.get("transition_token_ids", ())
                        )
                    )
                    candidate_native_append_count = int(
                        candidate.think_result.get(
                            "native_token_append_count",
                            branch.get("native_token_append_count", 0),
                        )
                    )
                    adopted_source_watermark = int(
                        getattr(candidate.listener_watermark, "frame_id", -1)
                    )
                    adaptation_lag_frames = max(
                        0,
                        int(candidate.requested_frame_id)
                        - int(getattr(
                            candidate.listener_watermark,
                            "frame_id",
                            candidate.requested_frame_id,
                        )),
                    )
                    adopted_terminal_reason = candidate.think_result.get("terminal_reason")
                    controller = candidate.think_result.get("_native_session_controller")
                    session_owner = None
                    if isinstance(controller, dict):
                        session_owner = controller.get("session")
                        current_native_handle = getattr(session_owner, "native_handle", None)
                        if (
                            getattr(current_native_handle, "handle_id", None)
                            != getattr(candidate_handle, "handle_id", None)
                        ):
                            raise RuntimeError("candidate decoder session and handle identity differ")
                        validate_adoption = getattr(
                            session_owner, "validate_native_handle_adoption", None
                        )
                        if not callable(validate_adoption):
                            raise RuntimeError("candidate session cannot adopt its promoted handle")
                        expected_promoted_handle = replace(
                            candidate_handle, purpose="active_response"
                        )
                        validate_adoption(expected_promoted_handle)
                    ownership_state = candidate.think_result.get("_native_ownership_state")
                    handoff_state = candidate.think_result.get("_revision_handoff_state")
                    if promote_candidate is not None:
                        if candidate_handle is None:
                            raise ValueError("candidate native handle is required for promotion")
                        # The worker command performs its own owner-side CAS and
                        # contains no forward, replay, callback, or displaced-row
                        # release. Failure leaves every prepared Python view
                        # untouched; successful promotion is followed only by
                        # prevalidated pointer/list assignments below.
                        candidate_handle = promote_candidate(candidate)

                    stream.install_unconsumed_suffix_locked(stream_plan)
                    ledger.install_unconsumed_suffix_locked(ledger_plan)
                    branch["say_token_packets"] = new_packets_list
                    branch["segment_token_packets"] = new_records_list
                    if promote_candidate is not None:
                        branch["response_decode_handle"] = candidate_handle
                        branch["response_decode_purpose"] = "active_response"
                    branch["revision_id"] = candidate.revision_id
                    branch["cache_epoch_id"] = candidate.cache_epoch_id
                    branch["history_revision"] = candidate.history_revision
                    branch["listener_cache_epoch_id"] = candidate.cache_epoch_id
                    branch["listener_state_epoch_id"] = listener_state_epoch_id
                    branch["listener_generation_id"] = listener_generation_id
                    branch["source_frame_id"] = source_frame_id
                    branch["source_state_version"] = source_state_version
                    branch["source_sequence_position"] = source_sequence_position
                    branch["listener_source_cache_position"] = listener_source_cache_position
                    branch["turn_start_cache_position"] = candidate_source_cursor
                    branch["cache_position"] = candidate_cache_position
                    branch["spoken_prefix_base_cursor"] = candidate_spoken_base_cursor
                    branch["spoken_prefix_transition_cursor"] = candidate_spoken_base_cursor
                    branch["transition_token_ids"] = candidate_transition_ids
                    branch["generated_token_count"] = new_adopted_budget
                    branch["native_token_append_count"] = candidate_native_append_count
                    branch["_generated_budget_used"] = new_adopted_budget
                    branch["_adopted_source_watermark"] = adopted_source_watermark
                    branch["_adaptation_lag_frames"] = adaptation_lag_frames
                    branch["terminal_reason"] = adopted_terminal_reason
                    terminal_unappended_count = int(
                        candidate.think_result.get("unappended_terminal_token_count", 0)
                    )
                    branch["unappended_terminal_token_count"] = terminal_unappended_count
                    branch["sampled_generated_token_count"] = (
                        new_adopted_budget + terminal_unappended_count
                    )
                    branch["_last_adopted_think"] = adopted_think
                    if isinstance(ownership_state, dict) and promote_candidate is not None:
                        ownership_state["promoted"] = True
                    if isinstance(handoff_state, dict):
                        handoff_state["promoted"] = promote_candidate is not None
                        handoff_state["target_branch"] = branch
                        handoff_state["token_index_offset"] = prefix_count
                    if isinstance(controller, dict):
                        handle_ref = controller.get("handle_ref")
                        if isinstance(handle_ref, dict):
                            handle_ref["value"] = candidate_handle
                        session_owner.adopt_native_handle(candidate_handle)
                        controller["promoted"] = True
                        controller["source_token_index_offset"] = prefix_count
                        controller_handoff = controller.get("handoff_state")
                        if isinstance(controller_handoff, dict):
                            controller_handoff["promoted"] = True
                            controller_handoff["target_branch"] = branch
                            controller_handoff["token_index_offset"] = prefix_count
                    branch["_adopted_native_session_controller"] = controller
                    branch["_streaming_revision_id"] = candidate.revision_id
                    branch["_last_candidate_handle_id"] = getattr(candidate.handle, "handle_id", None)
                    branch["revision_metrics"] = revision_metrics
                    return True


__all__ = [
    "ResponseRevisionCandidate",
    "ResponseRevisionCoordinator",
    "ResponseRevisionRequest",
    "StaleResponseRevision",
    "install_response_revision_transaction",
]
