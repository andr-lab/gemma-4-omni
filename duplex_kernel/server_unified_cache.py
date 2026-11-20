"""Unified native cache branch finalization."""
from __future__ import annotations

import asyncio
from contextlib import nullcontext
from dataclasses import replace as dataclass_replace
import time
from typing import Any, Callable, Mapping, Sequence

from duplex_kernel.server_response_control import _response_branch_has_pending_pcm


def _deferred_acoustic_history_consumption_complete(branch: Mapping[str, Any]) -> bool:
    ledger = branch.get("response_event_ledger")
    project = getattr(ledger, "project", None)
    if callable(project):
        return bool(project().all_pcm_consumed)
    return bool(
        not _response_branch_has_pending_pcm(branch)
        and branch.get("playback_cancelled") is not True
        and int(branch.get("client_consumed_pcm_sequence", -1)) >= 0
    )


def _deferred_acoustic_history_archive_status(
    branch: Mapping[str, Any], acknowledged_text: str
) -> str:
    ledger = branch.get("response_event_ledger")
    project = getattr(ledger, "project", None)
    if callable(project):
        status = str(project().text_status)
        return status if status in {"completed", "partial", "absent"} else "absent"
    if _deferred_acoustic_history_consumption_complete(branch) and acknowledged_text:
        return "completed"
    return "partial" if acknowledged_text else "absent"


def _commit_deferred_acoustic_history_archive(
    branch: dict[str, Any],
    session: Any,
    *,
    response_token: Any,
    response_id: str,
    user_onset_frame_id: int | None,
) -> tuple[bool, str, Any, Any]:
    metadata = branch["deferred_acoustic_history_commit"]
    ledger = branch.get("response_event_ledger")
    project = getattr(ledger, "project", None)
    if callable(project):
        projection = project()
        acknowledged_text = str(projection.confirmed_text)
        acknowledged_cursor = projection.confirmed_exclusive_kv_cursor
        archive_id = metadata.get("archive_id")
        done_at_ns = metadata.get("response_done_at_monotonic_ns")
        committed = bool(
            callable(getattr(session, "commit_response_ledger_projection", None))
            and session.commit_response_ledger_projection(
                ledger,
                archive_id=(int(archive_id) if isinstance(archive_id, int) else None),
                response_done_at_monotonic_ns=(
                    int(done_at_ns) if isinstance(done_at_ns, int) else None
                ),
            )
        )
        branch["deferred_acoustic_history_committed"] = committed
        return committed, acknowledged_text, acknowledged_cursor, done_at_ns
    with branch.get("spoken_prefix_lock", nullcontext()):
        acknowledged_text = str(branch.get("client_consumed_prefix_text", "")).strip()
        acknowledged_cursor = branch.get("client_consumed_prefix_exclusive_cursor")
    archive_id = metadata.get("archive_id")
    done_at_ns = metadata.get("response_done_at_monotonic_ns")
    if isinstance(archive_id, int):
        committed = session.update_audio_history_archive_assistant(
            int(archive_id),
            acknowledged_text,
            status=_deferred_acoustic_history_archive_status(branch, acknowledged_text),
            response_done_at_monotonic_ns=(
                int(done_at_ns) if isinstance(done_at_ns, int) else None
            ),
        )
    elif acknowledged_text:
        committed = session.commit_completed_assistant_response(
            response_token,
            response_id,
            acknowledged_text,
            user_onset_frame_id=user_onset_frame_id,
            response_done_at_monotonic_ns=(
                int(done_at_ns) if isinstance(done_at_ns, int) else None
            ),
        )
    else:
        committed = True
    branch["deferred_acoustic_history_committed"] = bool(committed)
    return bool(committed), acknowledged_text, acknowledged_cursor, done_at_ns


def _prepare_deferred_acoustic_history_finalization(
    branch: dict[str, Any],
    *,
    history_committed: bool,
    commit_deferred_acoustic_history: Callable[[], Any],
) -> bool:
    deferred_commit = branch.get("deferred_acoustic_history_commit")
    has_deferred_commit = isinstance(deferred_commit, Mapping)
    ledger = branch.get("response_event_ledger")
    project = getattr(ledger, "project", None)
    if callable(project):
        projection = project()
        source_rows = tuple(projection.confirmed_source_tokens)
        source_ids = tuple(
            int(row["token_id"])
            for row in source_rows
            if isinstance(row.get("token_id"), int)
            and not isinstance(row.get("token_id"), bool)
        )
        source_text_parts = tuple(str(row.get("text_piece", "")) for row in source_rows)
        base_cursor = int(branch.get(
            "spoken_prefix_base_cursor", branch.get("turn_start_cache_position", 0)
        ))
        cursor = projection.confirmed_exclusive_kv_cursor
        if not source_ids:
            cursor = base_cursor
        physical_cursor = (
            int(cursor)
            if isinstance(cursor, int) and not isinstance(cursor, bool)
            else None
        )
        with branch.get("spoken_prefix_lock", nullcontext()):
            branch["spoken_prefix_token_ids"] = source_ids
            branch["spoken_prefix_text_parts"] = source_text_parts
            branch["spoken_prefix_text"] = str(projection.confirmed_text)
            branch["spoken_prefix_exclusive_cursor"] = physical_cursor
            branch["client_consumed_prefix_token_ids"] = source_ids
            branch["client_consumed_prefix_text_parts"] = source_text_parts
            branch["client_consumed_prefix_text"] = str(projection.confirmed_text)
            branch["client_consumed_prefix_exclusive_cursor"] = physical_cursor
            branch["ledger_revision_id"] = str(projection.revision_id)
            branch["ledger_cache_epoch_id"] = projection.cache_epoch_id
            branch["spoken_prefix_physical_cursor_contiguous"] = bool(
                projection.physical_cursor_contiguous
            )
            branch["spoken_prefix_source_token_seq"] = tuple(
                int(row["source_token_seq"])
                for row in source_rows
                if isinstance(row.get("source_token_seq"), int)
                and not isinstance(row.get("source_token_seq"), bool)
            )
        if not has_deferred_commit:
            return history_committed
        if projection.text_status == "completed":
            commit_deferred_acoustic_history()
            history_committed = True
        else:
            # The history callback persists only this exact ledger projection;
            # canonical KV finalization rebuilds the same complete-unit prefix.
            commit_deferred_acoustic_history()
            history_committed = False
        branch["history_committed"] = history_committed
        return history_committed
    if "client_consumed_prefix_token_ids" in branch or has_deferred_commit:
        with branch.get("spoken_prefix_lock", nullcontext()):
            branch["spoken_prefix_token_ids"] = tuple(
                int(value) for value in branch.get("client_consumed_prefix_token_ids", ())
            )
            branch["spoken_prefix_exclusive_cursor"] = int(branch.get(
                "client_consumed_prefix_exclusive_cursor",
                branch.get("spoken_prefix_base_cursor", branch.get("turn_start_cache_position", 0)),
            ))
            branch["spoken_prefix_text_parts"] = tuple(
                branch.get("client_consumed_prefix_text_parts", ())
            )
            branch["spoken_prefix_text"] = str(
                branch.get("client_consumed_prefix_text", "")
            )
    if not has_deferred_commit:
        return history_committed
    if _deferred_acoustic_history_consumption_complete(branch):
        commit_deferred_acoustic_history()
        history_committed = True
    else:
        history_committed = False
    branch["history_committed"] = history_committed
    return history_committed


def _close_shared_audio_queue_boundary(branch: dict[str, Any]) -> int | None:
    """Publish one monotone queue-close epoch while the transition lock is held."""
    if not branch.get("queue_open"):
        existing = branch.get("queue_close_epoch")
        return int(existing) if isinstance(existing, int) else None
    state = branch.get("queue_epoch_state")
    if isinstance(state, dict):
        epoch = int(state.get("value", 0)) + 1
        state["value"] = epoch
    else:
        epoch = int(branch.get("queue_boundary_epoch", branch.get("queue_open_epoch", 0))) + 1
    branch["queue_boundary_epoch"] = epoch
    branch["queue_close_epoch"] = epoch
    branch["queue_closed_at_ns"] = time.monotonic_ns()
    branch["queue_open"] = False
    return epoch


def _record_drain_server_timings(
    result: Any,
    *,
    completion_wait_ms: float,
    transition_lock_wait_ms: float,
) -> Any:
    """Add event-loop completion wait to the immutable worker-owned drain metrics."""
    metrics = getattr(result, "queued_audio_drain_metrics", None)
    if metrics is None:
        return result
    # Owner wait and replay forward have independent clocks. Preserve only the
    # residual listener/tick completion and publication wait here.
    completion_residual_ms = max(
        0.0,
        float(completion_wait_ms)
        - float(getattr(metrics, "drain_queue_wait_ms", 0.0))
        - float(getattr(metrics, "drain_replay_forward_ms", 0.0)),
    )
    try:
        from duplex_kernel.microbatch_contract import QueuedAudioDrainMetrics
        if isinstance(metrics, QueuedAudioDrainMetrics):
            metrics = dataclass_replace(
                metrics,
                drain_completion_wait_ms=completion_residual_ms,
                drain_transition_lock_wait_ms=max(0.0, float(transition_lock_wait_ms)),
            )
            return dataclass_replace(result, queued_audio_drain_metrics=metrics)
    except (TypeError, ValueError):
        pass
    try:
        setattr(metrics, "drain_completion_wait_ms", completion_residual_ms)
        setattr(
            metrics,
            "drain_transition_lock_wait_ms",
            max(0.0, float(transition_lock_wait_ms)),
        )
    except (AttributeError, TypeError):
        return result
    return result


async def _await_owner_completion_ordered(owner_future: Any) -> tuple[Any, bool]:
    """Wait for owner cleanup even when the requesting asyncio task is cancelled."""
    wrapped = asyncio.wrap_future(owner_future)
    cancellation_requested = False
    while True:
        try:
            result = await asyncio.shield(wrapped)
            return result, cancellation_requested
        except asyncio.CancelledError:
            if wrapped.cancelled():
                raise
            cancellation_requested = True
            if wrapped.done():
                return wrapped.result(), cancellation_requested


def _flatten_drain_metrics(result: Any, branch: Mapping[str, Any]) -> dict[str, Any]:
    from duplex_kernel.server_diagnostics import _queued_audio_drain_metric_fields
    metrics = _queued_audio_drain_metric_fields(result)
    open_epoch = branch.get("queue_open_epoch")
    close_epoch = branch.get("queue_close_epoch")
    metrics["queued_audio_queue_open_epoch"] = (
        int(open_epoch) if isinstance(open_epoch, int) else None
    )
    metrics["queued_audio_queue_close_epoch"] = (
        int(close_epoch) if isinstance(close_epoch, int) else None
    )
    return metrics


def _queue_worker_close_epoch(worker: Any, session_id: str, branch: Mapping[str, Any]) -> None:
    epoch = branch.get("queue_close_epoch")
    marker = getattr(worker, "mark_assistant_audio_queue_closing", None)
    if isinstance(epoch, int) and callable(marker):
        marker(session_id, queue_close_epoch=epoch)

async def _finalize_unified_native_cache_branch(
    worker: Any,
    session_id: str,
    branch: dict[str, Any],
    *,
    history_committed: bool,
    close_ids: tuple[int, ...],
    retry_timeout_s: float = 2.0,
) -> Any:
    """Close or roll back one native assistant branch, then drain queued audio.
    The transition lock protects each native mutation submission only; it is
    released before awaiting owner-thread completion or listener watermarks. A
    successful close consumes the branch's reserved suffix budget; any failed
    close attempts a rollback to the user-turn boundary before reporting the
    error, so a partially appended close suffix is never left as a live branch.
    """
    from duplex_kernel.server_speech_ack import _retire_response_speech_ack_streams
    if branch.get("response_decode_handle") is not None or branch.get("response_decode_purpose"):
        handle = branch.get("response_decode_handle")
        if handle is not None:
            await asyncio.shield(asyncio.wrap_future(worker.end_response_decode(handle)))
            branch["response_decode_handle"] = None
        branch["queue_open"] = False
        branch["finalized"] = True
        branch["private_slot_released"] = True
        _retire_response_speech_ack_streams(branch)
        return None
    if branch.get("ephemeral_private"):
        handle = branch.get("ephemeral_decode_handle")
        release = getattr(worker, "end_ephemeral_decode", None)
        if handle is None or not callable(release):
            raise RuntimeError("private native branch lacks its ephemeral release contract")
        release_future = asyncio.wrap_future(release(handle))
        try:
            await asyncio.shield(release_future)
        except asyncio.CancelledError:
            # Keep the native row alive until the owner has completed release;
            # cancellation must not turn this slot into a use-after-free.
            await asyncio.shield(release_future)
            raise
        branch["cache_position"] = int(getattr(handle, "cache_position", branch["cache_position"]))
        branch["finalized"] = True
        branch["private_slot_released"] = True
        branch["history_committed"] = False
        _retire_response_speech_ack_streams(branch)
        return None

    listener_generation_id = int(branch["listener_generation_id"])
    turn_start = int(branch["turn_start_cache_position"])
    cache_cursor = int(branch["cache_position"])
    cache_position_before_rewind: int | None = None
    physical_rewind_tokens = 0

    async def submit_transition_mutation(
        operation: Any,
        *args: Any,
        close_audio_queue: bool = False,
        **kwargs: Any,
    ) -> Any:
        transition_lock = branch.get("transition_lock")
        lock_wait_started_ns = time.monotonic_ns()
        if transition_lock is None:
            lock_wait_ms = 0.0
            if close_audio_queue:
                _close_shared_audio_queue_boundary(branch)
                branch["queued_audio_drain_lock_wait_ms"] = lock_wait_ms
                _queue_worker_close_epoch(worker, session_id, branch)
            return operation(*args, **kwargs)
        deadline = asyncio.get_running_loop().time() + retry_timeout_s
        while not transition_lock.acquire(blocking=False):
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("unified transition lock wait exceeded retry timeout")
            await asyncio.sleep(0.002)
        lock_wait_ms = max(0, time.monotonic_ns() - lock_wait_started_ns) / 1_000_000.0
        try:
            if close_audio_queue:
                _close_shared_audio_queue_boundary(branch)
                branch["queued_audio_drain_lock_wait_ms"] = lock_wait_ms
                _queue_worker_close_epoch(worker, session_id, branch)
            return operation(*args, **kwargs)
        finally:
            transition_lock.release()

    async def current_snapshot() -> Any:
        snapshot = await asyncio.wrap_future(worker.session_state(session_id))
        if int(snapshot.generation_id) != listener_generation_id:
            raise RuntimeError("unified finalizer lost its listener generation")
        if int(snapshot.cache_position) < turn_start:
            raise RuntimeError("unified finalizer observed a cursor before the user-turn boundary")
        return snapshot

    async def rollback_to_turn_start() -> None:
        nonlocal cache_cursor, cache_position_before_rewind, physical_rewind_tokens
        rollback_deadline = asyncio.get_running_loop().time() + retry_timeout_s
        while True:
            latest = await current_snapshot()
            cache_cursor = int(latest.cache_position)
            cursor_before_rewind = cache_cursor
            if cache_position_before_rewind is None:
                cache_position_before_rewind = cursor_before_rewind
            try:
                rollback_future = await submit_transition_mutation(
                    worker.rollback_assistant_branch,
                    session_id,
                    expected_generation_id=listener_generation_id,
                    expected_cache_position=cache_cursor,
                    target_cache_position=turn_start,
                )
                rolled_back = await asyncio.wrap_future(rollback_future)
                cache_cursor = int(rolled_back.cache_position)
                actual_rewind = cursor_before_rewind - cache_cursor
                if actual_rewind < 0:
                    raise RuntimeError("unified rollback returned a cursor after its source cursor")
                physical_rewind_tokens += actual_rewind
                branch["cache_position"] = cache_cursor
                break
            except RuntimeError as exc:
                if (
                    str(exc) not in {
                        "assistant rollback requires a completed listener frame watermark",
                        "assistant rollback cannot cross a shared listener tick",
                    }
                    or asyncio.get_running_loop().time() >= rollback_deadline
                ):
                    raise
                await asyncio.sleep(0.002)

    async def drain_after_rollback(*, user_turn_already_open: bool) -> Any:
        nonlocal cache_cursor
        drain_deadline = asyncio.get_running_loop().time() + retry_timeout_s
        while True:
            try:
                drain_future = await submit_transition_mutation(
                    worker.drain_queued_audio,
                    session_id,
                    expected_generation_id=listener_generation_id,
                    expected_cache_position=cache_cursor,
                    user_turn_already_open=user_turn_already_open,
                    close_audio_queue=True,
                )
                completion_wait_started_ns = time.monotonic_ns()
                drained, cancellation_requested = await _await_owner_completion_ordered(
                    drain_future
                )
                completion_wait_ms = max(
                    0, time.monotonic_ns() - completion_wait_started_ns
                ) / 1_000_000.0
                drained = _record_drain_server_timings(
                    drained,
                    completion_wait_ms=completion_wait_ms,
                    transition_lock_wait_ms=float(
                        branch.get("queued_audio_drain_lock_wait_ms", 0.0)
                    ),
                )
                branch["queued_audio_drain_metrics"] = _flatten_drain_metrics(drained, branch)
                branch["cache_position"] = int(drained.cache_position)
                branch["queue_open"] = False
                branch["finalized"] = True
                branch["rollback_recovered"] = True
                _retire_response_speech_ack_streams(branch)
                if cancellation_requested:
                    raise asyncio.CancelledError
                return drained
            except RuntimeError as exc:
                if (
                    str(exc) not in {
                        "queued-audio drain needs the assistant Q0 watermark to complete; retry after the prior tick",
                        "queued-audio drain cannot overlap a shared listener tick",
                    }
                    or asyncio.get_running_loop().time() >= drain_deadline
                ):
                    raise
                await asyncio.sleep(0.002)

    async def append_spoken_prefix_after_rewind() -> None:
        nonlocal cache_cursor
        transition_ids = tuple(
            int(value) for value in branch.get("transition_token_ids", ())
        )
        speech_ids = tuple(
            int(value) for value in branch.get("spoken_prefix_token_ids", ())
        )
        if not transition_ids:
            raise RuntimeError(
                "spoken-prefix KV rebuild has no original assistant transition tokens"
            )
        if not speech_ids:
            raise RuntimeError(
                "spoken-prefix KV rebuild has no exact Gemma token IDs from a consumed segment"
            )
        rebuild_and_close_ids = (*transition_ids, *speech_ids, *tuple(close_ids))
        reserve_deadline = asyncio.get_running_loop().time() + retry_timeout_s
        while True:
            try:
                reserve_future = await submit_transition_mutation(
                    worker.reserve_tail_tokens,
                    session_id,
                    expected_generation_id=listener_generation_id,
                    expected_cache_position=cache_cursor,
                    token_count=len(rebuild_and_close_ids) + 2,
                )
                await asyncio.wrap_future(reserve_future)
                break
            except RuntimeError as exc:
                if (
                    str(exc) not in {
                        "tail reservation requires a completed listener frame watermark",
                        "tail reservation cannot overlap a shared listener tick",
                    }
                    or asyncio.get_running_loop().time() >= reserve_deadline
                ):
                    raise
                await asyncio.sleep(0.002)

        for start in range(0, len(rebuild_and_close_ids), 2):
            token_group = tuple(
                int(value) for value in rebuild_and_close_ids[start : start + 2]
            )
            append_started_ns = time.monotonic_ns()
            while True:
                owner_future = await submit_transition_mutation(
                    worker.append_text_tokens,
                    session_id,
                    token_group,
                    expected_generation_id=listener_generation_id,
                    expected_cache_position=cache_cursor,
                    consume_reserved_tokens=len(token_group),
                )
                try:
                    await asyncio.shield(asyncio.wrap_future(owner_future))
                    break
                except asyncio.CancelledError:
                    await asyncio.shield(asyncio.wrap_future(owner_future))
                    cache_cursor += len(token_group)
                    branch["cache_position"] = cache_cursor
                    raise
                except RuntimeError as exc:
                    if (
                        str(exc) != "text append requires completed listener frame watermark"
                        or time.monotonic_ns() - append_started_ns
                        >= int(retry_timeout_s * 1e9)
                    ):
                        raise
                    latest = await current_snapshot()
                    if int(latest.cache_position) != cache_cursor:
                        raise RuntimeError(
                            "spoken-prefix KV rebuild observed an untracked cache append"
                        ) from exc
                    await asyncio.sleep(0.002)
            cache_cursor += len(token_group)
            branch["cache_position"] = cache_cursor
        expected_exclusive_cursor = branch.get("spoken_prefix_exclusive_cursor")
        rebuilt_prefix_cursor = cache_cursor - len(tuple(close_ids))
        if (branch.get("spoken_prefix_physical_cursor_contiguous", True) is True
                and expected_exclusive_cursor is not None
                and rebuilt_prefix_cursor != int(expected_exclusive_cursor)):
            raise RuntimeError(
                "rebuilt spoken prefix does not end at its acknowledged exclusive KV cursor"
            )

    def record_rollback_metrics(cache_after_rebuild: int) -> None:
        transition_ids = tuple(int(value) for value in branch.get("transition_token_ids", ()))
        source_ids = tuple(int(value) for value in branch.get("spoken_prefix_token_ids", ()))
        rebuilt = bool(branch.get("spoken_prefix_rebuilt"))
        committed_generated = branch.get("generated_token_count")
        sampled_generated = branch.get("sampled_generated_token_count")
        unappended_terminal = branch.get("unappended_terminal_token_count")
        if unappended_terminal is None and isinstance(committed_generated, int) and isinstance(
            sampled_generated, int
        ):
            unappended_terminal = max(0, int(sampled_generated) - int(committed_generated))
        client_ack_sequence = int(branch.get("client_consumed_pcm_sequence", -1))
        ack_provenance = (
            "client_consumed_pcm_logical_device_buffer"
            if client_ack_sequence >= 0
            else branch.get("spoken_prefix_ack_provenance")
        )
        before = cache_position_before_rewind
        if before is None:
            before = int(cache_after_rebuild)
        # Signed delta: positive means the rebuilt cursor is shorter than the
        # cursor that existed immediately before the first physical rewind.
        branch["rollback_metrics"] = {
            "physical_rewind_tokens": int(physical_rewind_tokens),
            "committed_generated_tokens": (
                int(committed_generated) if isinstance(committed_generated, int) else None
            ),
            "sampled_generated_tokens": (
                int(sampled_generated) if isinstance(sampled_generated, int) else None
            ),
            "unappended_terminal_tokens": (
                int(unappended_terminal) if isinstance(unappended_terminal, int) else None
            ),
            "transition_tokens_replayed": len(transition_ids) if rebuilt else 0,
            "acknowledged_source_tokens_replayed": len(source_ids) if rebuilt else 0,
            "acknowledged_source_token_ids_replayed": list(source_ids) if rebuilt else [],
            "acknowledged_source_exclusive_cursor": (
                int(branch["spoken_prefix_exclusive_cursor"])
                if rebuilt and isinstance(branch.get("spoken_prefix_exclusive_cursor"), int)
                else None
            ),
            "acknowledged_source_ack_provenance": ack_provenance,
            "client_consumed_pcm_ack_sequence": (
                client_ack_sequence if client_ack_sequence >= 0 else None
            ),
            "close_tokens_replayed": len(tuple(close_ids)) if rebuilt else 0,
            "cache_position_before_rewind": int(before),
            "cache_position_after_rebuild": int(cache_after_rebuild),
            "net_cache_cursor_tokens_removed_after_rebuild": int(before) - int(cache_after_rebuild),
        }

    async def rollback_and_drain() -> Any:
        await rollback_to_turn_start()
        acknowledged_text = str(branch.get("spoken_prefix_text", ""))
        if acknowledged_text:
            try:
                if not tuple(branch.get("spoken_prefix_token_ids", ())):
                    raise RuntimeError(
                        "client-consumed spoken prefix has text but no original Gemma token IDs"
                    )
                await append_spoken_prefix_after_rewind()
            except BaseException as rebuild_error:
                # Preserve the user turn boundary when canonical reconstruction
                # cannot fit. The caller sees an error and can quarantine the
                # branch; acknowledged text is never silently represented by raw JSON.
                await rollback_to_turn_start()
                branch["history_committed"] = False
                branch["spoken_prefix_rebuilt"] = False
                record_rollback_metrics(cache_cursor)
                drained = await drain_after_rollback(user_turn_already_open=True)
                raise RuntimeError(
                    "spoken-prefix KV rebuild failed; branch was rolled back to the open user turn"
                ) from rebuild_error
            branch["spoken_prefix_rebuilt"] = True
            branch["history_committed"] = False
            record_rollback_metrics(cache_cursor)
            drained = await drain_after_rollback(user_turn_already_open=False)
            return drained
        record_rollback_metrics(cache_cursor)
        drained = await drain_after_rollback(user_turn_already_open=True)
        branch["history_committed"] = False
        branch["spoken_prefix_rebuilt"] = False
        return drained

    if not history_committed:
        drained = await rollback_and_drain()
        return drained

    close_error: BaseException | None = None
    try:
        latest = await current_snapshot()
        latest_cursor = int(latest.cache_position)
        if latest_cursor < cache_cursor:
            raise RuntimeError("unified finalizer observed a backward cache cursor")
        if latest_cursor > cache_cursor:
            raise RuntimeError(
                "unified finalizer observed an untracked cache append after Thinker completion"
            )
        cache_cursor = latest_cursor
        branch["cache_position"] = cache_cursor
        # Reassert capacity against the post-decode cursor and all queued Q2
        # frames before the first close token can mutate native cache.
        reserve_deadline = asyncio.get_running_loop().time() + retry_timeout_s
        while True:
            try:
                reserve_future = await submit_transition_mutation(
                    worker.reserve_tail_tokens,
                    session_id,
                    expected_generation_id=listener_generation_id,
                    expected_cache_position=cache_cursor,
                    token_count=len(close_ids) + 2,
                )
                await asyncio.wrap_future(reserve_future)
                break
            except RuntimeError as exc:
                if (
                    str(exc) not in {
                        "tail reservation requires a completed listener frame watermark",
                        "tail reservation cannot overlap a shared listener tick",
                    }
                    or asyncio.get_running_loop().time() >= reserve_deadline
                ):
                    raise
                await asyncio.sleep(0.002)
        for start in range(0, len(close_ids), 2):
            token_group = tuple(int(value) for value in close_ids[start : start + 2])
            owner_future = await submit_transition_mutation(
                worker.append_text_tokens,
                session_id,
                token_group,
                expected_generation_id=listener_generation_id,
                expected_cache_position=cache_cursor,
                consume_reserved_tokens=len(token_group),
            )
            try:
                await asyncio.shield(asyncio.wrap_future(owner_future))
            except asyncio.CancelledError:
                await asyncio.shield(asyncio.wrap_future(owner_future))
                cache_cursor += len(token_group)
                branch["cache_position"] = cache_cursor
                raise
            cache_cursor += len(token_group)
            branch["cache_position"] = cache_cursor
    except BaseException as exc:
        close_error = exc

    if close_error is not None:
        try:
            await rollback_and_drain()
        except BaseException as recovery_error:
            raise RuntimeError(
                "unified native close failed and assistant-branch rollback/audio drain also failed; "
                "the session must remain quarantined"
            ) from recovery_error
        raise RuntimeError(
            "unified native close failed; assistant branch was rolled back and queued audio was "
            "drained before the session was quarantined"
        ) from close_error

    drain_deadline = asyncio.get_running_loop().time() + retry_timeout_s
    while True:
        try:
            drain_future = await submit_transition_mutation(
                worker.drain_queued_audio,
                session_id,
                expected_generation_id=listener_generation_id,
                expected_cache_position=cache_cursor,
                user_turn_already_open=False,
                close_audio_queue=True,
            )
            completion_wait_started_ns = time.monotonic_ns()
            drained, cancellation_requested = await _await_owner_completion_ordered(
                drain_future
            )
            completion_wait_ms = max(
                0, time.monotonic_ns() - completion_wait_started_ns
            ) / 1_000_000.0
            drained = _record_drain_server_timings(
                drained,
                completion_wait_ms=completion_wait_ms,
                transition_lock_wait_ms=float(
                    branch.get("queued_audio_drain_lock_wait_ms", 0.0)
                ),
            )
            branch["queued_audio_drain_metrics"] = _flatten_drain_metrics(drained, branch)
            branch["cache_position"] = int(drained.cache_position)
            branch["queue_open"] = False
            branch["finalized"] = True
            branch["history_committed"] = True
            _retire_response_speech_ack_streams(branch)
            if cancellation_requested:
                raise asyncio.CancelledError
            return drained
        except RuntimeError as exc:
            if (
                str(exc) not in {
                    "queued-audio drain needs the assistant Q0 watermark to complete; retry after the prior tick",
                    "queued-audio drain cannot overlap a shared listener tick",
                }
                or asyncio.get_running_loop().time() >= drain_deadline
            ):
                try:
                    await rollback_and_drain()
                except BaseException as recovery_error:
                    raise RuntimeError(
                        "unified native audio drain failed after close; rollback/audio recovery also "
                        "failed and the session must remain quarantined"
                    ) from recovery_error
                raise RuntimeError(
                    "unified native audio drain failed after close; branch was rolled back and "
                    "queued audio was drained before the session was quarantined"
                ) from exc
            await asyncio.sleep(0.002)
